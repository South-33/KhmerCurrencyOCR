#!/usr/bin/env python
"""Materialize teacher pseudo-labels for semantic-bridge target-positive rows."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageOps

from local_runtime import configure_project_cache


configure_project_cache()

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--max-det", type=int, default=20)
    parser.add_argument("--preview-items", type=int, default=64)
    parser.add_argument("--clean", action="store_true")
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


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(path)} must be a YAML mapping")
    return data


def read_image_list(path: Path) -> list[Path]:
    rows: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image = Path(line)
        rows.append(image if image.is_absolute() else ROOT / image)
    if not rows:
        raise SystemExit(f"empty image list: {repo_rel(path)}")
    return rows


def batched(items: list[Path], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def class_names(model: YOLO) -> dict[int, str]:
    names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def yolo_line(box: list[float], class_id: int, width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    cx = ((left + right) / 2.0) / width
    cy = ((top + bottom) / 2.0) / height
    bw = max(0.0, right - left) / width
    bh = max(0.0, bottom - top) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def safe_stem(index: int, path: Path) -> str:
    return f"pseudo_{index:06d}_{path.stem}"


def draw_preview(rows: list[dict[str, Any]], out_path: Path, count: int) -> None:
    selected = rows[:count]
    if not selected:
        return
    thumb_w, thumb_h = 300, 240
    caption_h = 42
    cols = min(4, len(selected))
    sheet_rows = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, sheet_rows * (thumb_h + caption_h)), (238, 238, 238))
    draw_sheet = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        image_path = resolve(row["image"])
        with Image.open(image_path).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for det in row["detections"]:
                draw.rectangle(det["xyxy"], outline=(40, 180, 70), width=3)
            thumb = ImageOps.contain(image, (thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (index % cols) * thumb_w
        y = (index // cols) * (thumb_h + caption_h)
        sheet.paste(thumb, (x + (thumb_w - thumb.width) // 2, y))
        caption = f"{row['classes']} n={row['detections_count']}"
        draw_sheet.text((x + 5, y + thumb_h + 4), caption[:48], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> int:
    args = parse_args()
    out_root = resolve(args.out_root)
    if args.clean and out_root.exists():
        if ROOT not in out_root.resolve().parents:
            raise SystemExit(f"refusing to clean outside repo: {out_root}")
        shutil.rmtree(out_root)
    (out_root / "images" / args.split).mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / args.split).mkdir(parents=True, exist_ok=True)
    (out_root / "qa").mkdir(parents=True, exist_ok=True)

    data_path = resolve(args.data)
    data_config = load_yaml(data_path)
    names = data_config.get("names", {})
    images = read_image_list(resolve(args.image_list))
    model = YOLO(str(resolve(args.teacher_model)))
    teacher_names = class_names(model)
    if teacher_names:
        names = {int(key): value for key, value in teacher_names.items()}

    manifest_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    copied = 0
    for batch in batched(images, max(1, args.batch)):
        results = model.predict(
            source=[str(path) for path in batch],
            imgsz=args.imgsz,
            conf=args.conf,
            batch=len(batch),
            device=args.device,
            max_det=args.max_det,
            verbose=False,
        )
        for source_path, result in zip(batch, results, strict=True):
            with Image.open(source_path) as opened:
                width, height = opened.size
            detections: list[dict[str, Any]] = []
            if result.boxes is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                cls = result.boxes.cls.cpu().numpy()
                conf = result.boxes.conf.cpu().numpy()
                for box, class_id, score in zip(xyxy, cls, conf, strict=True):
                    class_id_int = int(class_id)
                    class_name = str(names.get(class_id_int, f"class_{class_id_int}"))
                    detections.append(
                        {
                            "class_id": class_id_int,
                            "class_name": class_name,
                            "confidence": float(score),
                            "xyxy": [float(value) for value in box.tolist()],
                        }
                    )
            if not detections:
                continue
            copied += 1
            stem = safe_stem(copied, source_path)
            image_out = out_root / "images" / args.split / f"{stem}{source_path.suffix.lower()}"
            label_out = out_root / "labels" / args.split / f"{stem}.txt"
            shutil.copy2(source_path, image_out)
            label_lines = [
                yolo_line(det["xyxy"], int(det["class_id"]), width, height)
                for det in detections
            ]
            label_out.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            for det in detections:
                class_counts[str(det["class_name"])] += 1
            manifest_rows.append(
                {
                    "source_image": repo_rel(source_path),
                    "image": repo_rel(image_out),
                    "label": repo_rel(label_out),
                    "detections_count": len(detections),
                    "classes": ",".join(sorted({str(det["class_name"]) for det in detections})),
                    "detections": detections,
                }
            )

    data_out = out_root / "data.yaml"
    data_out.write_text(
        yaml.safe_dump(
            {
                "path": str(out_root.resolve()),
                "train": f"images/{args.split}",
                "val": str((data_path.parent / str(data_config.get("val", "images/val"))).resolve()),
                "test": str((data_path.parent / str(data_config.get("test", "images/test"))).resolve()),
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_image", "image", "label", "detections_count", "classes"],
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames or []})
    summary = {
        "schema": "cashsnap_teacher_pseudo_target_bridge_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data": repo_rel(data_path),
        "image_list": repo_rel(resolve(args.image_list)),
        "teacher_model": repo_rel(resolve(args.teacher_model)),
        "out_root": repo_rel(out_root),
        "split": args.split,
        "input_images": len(images),
        "materialized_images": len(manifest_rows),
        "conf": args.conf,
        "class_counts": dict(sorted(class_counts.items())),
        "data_yaml": repo_rel(data_out),
        "manifest": repo_rel(manifest_path),
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    draw_preview(manifest_rows, out_root / "qa" / "preview.jpg", args.preview_items)
    print(
        f"pseudo_bridge={repo_rel(out_root)} images={len(manifest_rows)}/{len(images)} "
        f"classes={dict(sorted(class_counts.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
