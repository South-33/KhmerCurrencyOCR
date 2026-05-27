from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from local_runtime import configure_project_cache

ROOT = Path(__file__).resolve().parents[1]

configure_project_cache()

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny banknote-fragment denomination classifier.")
    parser.add_argument("--data", required=True, help="ImageFolder dataset with train/val folders.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--pretrained", action="store_true", help="Initialize MobileNetV3 from torchvision ImageNet weights.")
    parser.add_argument("--balanced-loss", action="store_true", help="Use inverse-frequency class weights.")
    parser.add_argument("--balanced-sampler", action="store_true", help="Sample classes with inverse-frequency weights.")
    parser.add_argument("--project", default=str(ROOT / "runs" / "fragment_classifier"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.20),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18, hue=0.03),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tf, val_tf


def build_model(class_count: int, pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, class_count)
    return model


def class_counts(targets: list[int], class_count: int) -> list[int]:
    counts = [0] * class_count
    for target in targets:
        counts[int(target)] += 1
    return counts


def inverse_frequency_weights(counts: list[int]) -> torch.Tensor:
    total = sum(counts)
    class_count = len(counts)
    weights = [total / max(1, class_count * count) for count in counts]
    return torch.tensor(weights, dtype=torch.float32)


def accuracy_by_class(logits: torch.Tensor, targets: torch.Tensor, class_count: int) -> tuple[int, int, list[int], list[int]]:
    preds = logits.argmax(dim=1)
    correct = (preds == targets)
    class_correct = [0] * class_count
    class_total = [0] * class_count
    for target, is_correct in zip(targets.detach().cpu().tolist(), correct.detach().cpu().tolist(), strict=False):
        class_total[int(target)] += 1
        class_correct[int(target)] += int(is_correct)
    return int(correct.sum().item()), int(targets.numel()), class_correct, class_total


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    class_count: int,
) -> dict[str, float | list[float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    class_correct = [0] * class_count
    class_total = [0] * class_count
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        batch_correct, batch_total, batch_class_correct, batch_class_total = accuracy_by_class(logits, targets, class_count)
        total_loss += float(loss.item()) * batch_total
        total_correct += batch_correct
        total_seen += batch_total
        class_correct = [a + b for a, b in zip(class_correct, batch_class_correct, strict=False)]
        class_total = [a + b for a, b in zip(class_total, batch_class_total, strict=False)]

    per_class = [
        (class_correct[index] / class_total[index]) if class_total[index] else 0.0
        for index in range(class_count)
    ]
    return {
        "loss": total_loss / max(1, total_seen),
        "accuracy": total_correct / max(1, total_seen),
        "per_class_accuracy": per_class,
    }


def write_results(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_onnx(model: nn.Module, path: Path, image_size: int, device: torch.device) -> None:
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=12,
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )


def main() -> None:
    args = parse_args()
    data_dir = resolve(args.data)
    run_dir = resolve(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_tf, val_tf = build_transforms(args.image_size)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=val_tf)
    class_names = train_ds.classes
    if class_names != val_ds.classes:
        raise SystemExit(f"train/val classes differ: {class_names} != {val_ds.classes}")

    device = choose_device(args.device)
    model = build_model(len(class_names), args.pretrained).to(device)
    counts = class_counts(train_ds.targets, len(class_names))
    loss_weight = inverse_frequency_weights(counts).to(device) if args.balanced_loss else None
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sampler = None
    shuffle = True
    if args.balanced_sampler:
        class_weight = inverse_frequency_weights(counts)
        sample_weights = [float(class_weight[target]) for target in train_ds.targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    print(f"classes: {class_names}", flush=True)
    print(f"train class counts: {dict(zip(class_names, counts, strict=False))}", flush=True)
    print(f"device: {device}", flush=True)
    rows: list[dict[str, str]] = []
    best_accuracy = -1.0
    best_path = run_dir / "best.pt"
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, len(class_names))
        val_metrics = run_epoch(model, val_loader, criterion, device, None, len(class_names))
        row = {
            "epoch": str(epoch),
            "time": f"{time.time() - start:.1f}",
            "train_loss": f"{train_metrics['loss']:.5f}",
            "train_accuracy": f"{train_metrics['accuracy']:.5f}",
            "val_loss": f"{val_metrics['loss']:.5f}",
            "val_accuracy": f"{val_metrics['accuracy']:.5f}",
            "val_per_class_accuracy": json.dumps(val_metrics["per_class_accuracy"]),
        }
        rows.append(row)
        write_results(run_dir / "results.csv", rows)
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_acc={train_metrics['accuracy']:.3f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_loss={val_metrics['loss']:.3f}",
            flush=True,
        )
        if float(val_metrics["accuracy"]) > best_accuracy:
            best_accuracy = float(val_metrics["accuracy"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": class_names,
                    "image_size": args.image_size,
                    "architecture": "mobilenet_v3_small",
                    "pretrained": args.pretrained,
                },
                best_path,
            )

    (run_dir / "classes.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    if args.export_onnx:
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        export_onnx(model, run_dir / "best.onnx", args.image_size, device)
    print(f"best val accuracy: {best_accuracy:.3f}", flush=True)
    print(f"saved: {best_path}", flush=True)


if __name__ == "__main__":
    main()
