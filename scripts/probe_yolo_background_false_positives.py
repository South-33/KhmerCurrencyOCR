#!/usr/bin/env python
"""Probe YOLO false positives on known zero-label/background image roots."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from local_runtime import configure_project_cache


configure_project_cache()

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="Model path or label=path. Repeatable.")
    parser.add_argument("--image-root", action="append", required=True, type=Path, help="Directory of zero-label images. Repeatable.")
    parser.add_argument("--conf", default="0.05,0.18,0.25", help="Comma-separated confidence thresholds.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", default="0")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_model(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        label = label.strip()
        path = Path(raw_path.strip())
    else:
        path = Path(value.strip())
        label = path.parent.parent.name if path.name == "best.pt" else path.stem
    if not label:
        raise SystemExit(f"empty model label: {value!r}")
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing model: {resolved}")
    return label, resolved


def parse_confs(value: str) -> list[float]:
    confs = [float(item.strip()) for item in re.split(r"[,\s]+", value) if item.strip()]
    if not confs:
        raise SystemExit("--conf must include at least one threshold")
    return confs


def image_rows(root: Path) -> list[Path]:
    resolved = resolve(root)
    if not resolved.exists():
        raise SystemExit(f"missing image root: {resolved}")
    rows = [path for path in sorted(resolved.glob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    if not rows:
        raise SystemExit(f"image root has no images: {resolved}")
    return rows


def class_names(model: YOLO) -> dict[int, str]:
    names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def probe_root(
    *,
    model: YOLO,
    names: dict[int, str],
    label: str,
    model_path: Path,
    image_root: Path,
    images: list[Path],
    conf: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    detections = 0
    images_with_fp = 0
    by_class: Counter[str] = Counter()
    top: list[dict[str, Any]] = []
    results = model.predict(
        source=[str(path) for path in images],
        imgsz=args.imgsz,
        conf=conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
        save=False,
    )
    for image_path, result in zip(images, results, strict=True):
        boxes = result.boxes
        count = int(len(boxes))
        if count:
            images_with_fp += 1
            detections += count
        if not count:
            continue
        classes = boxes.cls.cpu().numpy().astype(int).tolist()
        scores = boxes.conf.cpu().numpy().tolist()
        for class_id, score in zip(classes, scores, strict=True):
            class_name = names.get(int(class_id), f"class_{class_id}")
            by_class[class_name] += 1
            top.append(
                {
                    "image": repo_rel(image_path),
                    "class": class_name,
                    "confidence": round(float(score), 6),
                }
            )
    top.sort(key=lambda row: float(row["confidence"]), reverse=True)
    return {
        "model_label": label,
        "model": repo_rel(model_path),
        "image_root": repo_rel(resolve(image_root)),
        "conf": conf,
        "images": len(images),
        "images_with_fp": images_with_fp,
        "detections": detections,
        "fp_per_image": detections / len(images),
        "by_class": dict(sorted(by_class.items())),
        "top": top[:10],
    }


def main() -> int:
    args = parse_args()
    models = [parse_model(value) for value in args.model]
    confs = parse_confs(args.conf)
    roots = [(root, image_rows(root)) for root in args.image_root]
    rows: list[dict[str, Any]] = []
    for model_label, model_path in models:
        model = YOLO(str(model_path))
        names = class_names(model)
        for root, images in roots:
            for conf in confs:
                row = probe_root(
                    model=model,
                    names=names,
                    label=model_label,
                    model_path=model_path,
                    image_root=root,
                    images=images,
                    conf=conf,
                    args=args,
                )
                rows.append(row)
                print(
                    f"{row['model_label']} conf={conf:g} root={row['image_root']} "
                    f"images_with_fp={row['images_with_fp']}/{row['images']} "
                    f"detections={row['detections']} fp_per_image={row['fp_per_image']:.3f} "
                    f"classes={row['by_class']}"
                )
    payload = {
        "schema": "cashsnap_yolo_background_false_positive_probe_v1",
        "imgsz": args.imgsz,
        "iou": args.iou,
        "max_det": args.max_det,
        "device": args.device,
        "rows": rows,
    }
    if args.json_out is not None:
        json_out = resolve(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote_json={repo_rel(json_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
