from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from local_runtime import configure_project_cache

configure_project_cache()

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render YOLO predictions over an image for visual diagnostics.")
    parser.add_argument("--image", type=Path, required=True, help="Image to run inference on.")
    parser.add_argument("--model", type=Path, required=True, help="YOLO checkpoint path.")
    parser.add_argument("--out", type=Path, required=True, help="Output preview image path.")
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional CSV with prediction boxes.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--max-side", type=int, default=2000, help="Resize preview so the longest side is at most this.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float, float, float],
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x1, y1, x2, y2 = xy
    draw.rectangle(xy, outline=color, width=4)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_height = bottom - top
    pad = 5
    label_y1 = max(0, y1 - text_height - pad * 2)
    label_y2 = label_y1 + text_height + pad * 2
    label_x2 = min(draw.im.size[0], x1 + text_width + pad * 2)
    draw.rectangle((x1, label_y1, label_x2, label_y2), fill=color)
    draw.text((x1 + pad, label_y1 + pad), text, fill=(0, 0, 0), font=font)


def main() -> None:
    args = parse_args()
    image_path = resolve_path(args.image)
    model_path = resolve_path(args.model)
    out_path = resolve_path(args.out)
    out_csv = resolve_path(args.out_csv) if args.out_csv is not None else None

    model = YOLO(str(model_path))
    names = {int(key): value for key, value in model.names.items()}
    result = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        verbose=False,
    )[0]
    boxes = result.boxes
    rows: list[dict[str, str]] = []
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy().astype(float)
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy().astype(float)
        for index, box in enumerate(xyxy):
            class_id = int(cls[index])
            rows.append(
                {
                    "index": str(index),
                    "class_id": str(class_id),
                    "class_name": names.get(class_id, str(class_id)),
                    "conf": f"{float(conf[index]):.6f}",
                    "x1": f"{float(box[0]):.2f}",
                    "y1": f"{float(box[1]):.2f}",
                    "x2": f"{float(box[2]):.2f}",
                    "y2": f"{float(box[3]):.2f}",
                }
            )

    with Image.open(image_path) as image:
        preview = image.convert("RGB")
    original_width, original_height = preview.size
    scale = min(1.0, args.max_side / max(original_width, original_height))
    if scale < 1.0:
        preview = preview.resize((round(original_width * scale), round(original_height * scale)), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(preview)
    font = load_font(max(18, round(max(preview.size) * 0.018)))
    for row in rows:
        class_id = int(row["class_id"])
        color = COLORS[class_id % len(COLORS)]
        xy = tuple(float(row[key]) * scale for key in ["x1", "y1", "x2", "y2"])
        label = f"{row['class_name']} {float(row['conf']):.2f}"
        draw_label(draw, xy, label, color, font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out_path, quality=92)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "class_id", "class_name", "conf", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {out_path.relative_to(ROOT)} with {len(rows)} predictions")


if __name__ == "__main__":
    main()
