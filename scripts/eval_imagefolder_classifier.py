#!/usr/bin/env python
"""Evaluate a MobileNetV3 ImageFolder classifier and export confusion evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from local_runtime import configure_project_cache


configure_project_cache()


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--focus-pair",
        action="append",
        default=[],
        help="True:Pred class pair to prioritize in the visual sheet. Repeatable or comma-separate.",
    )
    parser.add_argument("--max-examples-per-pair", type=int, default=24)
    parser.add_argument("--sheet-columns", type=int, default=6)
    parser.add_argument("--thumb-size", type=int, default=160)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def choose_device(value: str) -> torch.device:
    if value != "auto":
        if value.isdigit():
            return torch.device(f"cuda:{value}")
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(class_count: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, class_count)
    return model


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, list[str], int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    classes = [str(name) for name in checkpoint["classes"]]
    image_size = int(checkpoint.get("image_size", 224))
    model = build_model(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, classes, image_size


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def image_folder(path: Path, transform: transforms.Compose) -> datasets.ImageFolder:
    try:
        return datasets.ImageFolder(path, transform=transform, allow_empty=True)
    except TypeError:
        return datasets.ImageFolder(path, transform=transform)


def parse_focus_pairs(values: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        for token in value.replace(",", " ").split():
            if ":" not in token:
                raise SystemExit(f"--focus-pair must be TRUE:PRED, got {token!r}")
            true_name, pred_name = token.split(":", 1)
            pairs.append((true_name, pred_name))
    return pairs


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_sheet(
    *,
    rows: list[dict[str, Any]],
    out_path: Path,
    columns: int,
    thumb_size: int,
) -> None:
    if not rows:
        return
    font = load_font(13)
    label_h = 44
    cell_w = thumb_size
    cell_h = thumb_size + label_h
    columns = max(1, columns)
    sheet_rows = (len(rows) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, sheet_rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        image_path = resolve(str(row["image_path"]))
        with Image.open(image_path).convert("RGB") as image:
            image.thumbnail((thumb_size, thumb_size))
            px = x + (thumb_size - image.width) // 2
            py = y + (thumb_size - image.height) // 2
            sheet.paste(image, (px, py))
        text = f"{row['true_class']} -> {row['pred_class']}\nconf {float(row['pred_confidence']):.2f}"
        draw.text((x + 4, y + thumb_size + 3), text, fill=(20, 20, 20), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint)
    data_dir = resolve(args.data)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model, classes, image_size = load_checkpoint(checkpoint_path, device)
    dataset = image_folder(data_dir / args.split, transform=eval_transform(image_size))
    if dataset.classes != classes:
        raise SystemExit(f"checkpoint/data classes differ: {classes} != {dataset.classes}")

    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch),
        shuffle=False,
        num_workers=max(0, args.workers),
        pin_memory=device.type == "cuda",
    )
    rows: list[dict[str, Any]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    sample_paths = [Path(path) for path, _target in dataset.samples]
    offset = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            probs = torch.softmax(model(images), dim=1).detach().cpu()
            pred_conf, preds = probs.max(dim=1)
            for batch_index, (target, pred, confidence) in enumerate(
                zip(targets.cpu().tolist(), preds.tolist(), pred_conf.tolist(), strict=False)
            ):
                image_path = sample_paths[offset + batch_index]
                true_class = classes[int(target)]
                pred_class = classes[int(pred)]
                is_correct = int(target) == int(pred)
                totals[true_class] += 1
                if is_correct:
                    correct[true_class] += 1
                else:
                    confusion[(true_class, pred_class)] += 1
                rows.append(
                    {
                        "image_path": repo_rel(image_path),
                        "true_class": true_class,
                        "pred_class": pred_class,
                        "pred_confidence": round(float(confidence), 6),
                        "correct": is_correct,
                    }
                )
            offset += len(targets)

    per_class = []
    for class_name in classes:
        total = int(totals[class_name])
        ok = int(correct[class_name])
        per_class.append(
            {
                "class_name": class_name,
                "total": total,
                "correct": ok,
                "accuracy": ok / total if total else None,
            }
        )
    confusion_rows = [
        {"true_class": true_name, "pred_class": pred_name, "count": count}
        for (true_name, pred_name), count in confusion.most_common()
    ]
    summary = {
        "schema": "cashsnap_imagefolder_classifier_eval_v1",
        "checkpoint": repo_rel(checkpoint_path),
        "data": repo_rel(data_dir),
        "split": args.split,
        "images": len(dataset),
        "accuracy": sum(correct.values()) / max(1, sum(totals.values())),
        "per_class": per_class,
        "top_confusions": confusion_rows[:40],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(
        out_dir / "predictions.csv",
        rows,
        ["image_path", "true_class", "pred_class", "pred_confidence", "correct"],
    )
    write_csv(out_dir / "confusion.csv", confusion_rows, ["true_class", "pred_class", "count"])

    focus_pairs = parse_focus_pairs(args.focus_pair)
    if not focus_pairs:
        focus_pairs = [(row["true_class"], row["pred_class"]) for row in confusion_rows[:12]]
    examples: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["correct"]:
            continue
        key = (str(row["true_class"]), str(row["pred_class"]))
        if key in focus_pairs:
            grouped[key].append(row)
    for key in focus_pairs:
        candidates = sorted(grouped.get(key, []), key=lambda row: float(row["pred_confidence"]), reverse=True)
        examples.extend(candidates[: max(0, args.max_examples_per_pair)])
    write_csv(
        out_dir / "review_examples.csv",
        examples,
        ["image_path", "true_class", "pred_class", "pred_confidence", "correct"],
    )
    make_sheet(
        rows=examples,
        out_path=out_dir / "review_sheet.jpg",
        columns=args.sheet_columns,
        thumb_size=args.thumb_size,
    )
    print(
        f"eval_imagefolder={repo_rel(out_dir)} accuracy={summary['accuracy']:.4f} "
        f"confusions={len(confusion_rows)} examples={len(examples)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
