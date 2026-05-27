from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from local_runtime import configure_project_cache

configure_project_cache()

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained fragment classifier on an ImageFolder split.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--predictions-out", default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(class_count: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, class_count)
    return model


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_names: list[str] = checkpoint["classes"]
    image_size = int(checkpoint.get("image_size", 224))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(resolve(args.data) / args.split, transform=transform, allow_empty=True)
    if dataset.classes != class_names:
        raise SystemExit(f"dataset classes differ from checkpoint: {dataset.classes} != {class_names}")
    device = choose_device(args.device)
    model = build_model(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64)
    prediction_fieldnames = ["split", "image_path", "target", "prediction", "confidence", "correct"]
    prediction_rows: list[dict[str, str]] = []
    sample_offset = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images).cpu()
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            for batch_index, (target, pred) in enumerate(zip(targets, preds, strict=False)):
                confusion[int(target), int(pred)] += 1
                sample_path, _ = dataset.samples[sample_offset + batch_index]
                confidence = float(probs[batch_index, int(pred)].item())
                prediction_rows.append(
                    {
                        "split": args.split,
                        "image_path": str(Path(sample_path).relative_to(ROOT)),
                        "target": class_names[int(target)],
                        "prediction": class_names[int(pred)],
                        "confidence": f"{confidence:.6f}",
                        "correct": str(int(target) == int(pred)).lower(),
                    }
                )
            sample_offset += len(targets)

    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    rows: list[dict[str, str]] = []
    for index, class_name in enumerate(class_names):
        class_total = int(confusion[index].sum().item())
        class_correct = int(confusion[index, index].item())
        worst_pred = ""
        if class_total:
            row = confusion[index].clone()
            row[index] = 0
            worst_index = int(row.argmax().item())
            if int(row[worst_index].item()):
                worst_pred = f"{class_names[worst_index]}:{int(row[worst_index].item())}"
        rows.append(
            {
                "class_name": class_name,
                "correct": str(class_correct),
                "total": str(class_total),
                "accuracy": f"{class_correct / class_total:.4f}" if class_total else "",
                "worst_confusion": worst_pred,
            }
        )

    print(f"split={args.split} accuracy={correct / max(1, total):.4f} correct={correct} total={total}")
    for row in rows:
        print(
            f"{row['class_name']}: {row['accuracy']} "
            f"({row['correct']}/{row['total']}) worst={row['worst_confusion']}"
        )
    if args.out:
        out = resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (out.with_suffix(".confusion.json")).write_text(json.dumps(confusion.tolist()), encoding="utf-8")
    if args.predictions_out:
        predictions_out = resolve(args.predictions_out)
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        with predictions_out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=prediction_fieldnames)
            writer.writeheader()
            writer.writerows(prediction_rows)


if __name__ == "__main__":
    main()
