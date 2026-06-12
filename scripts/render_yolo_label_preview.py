from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAMES = {
    0: "USD_1",
    1: "USD_5",
    2: "USD_10",
    3: "USD_20",
    4: "USD_50",
    5: "USD_100",
    6: "KHR_500",
    7: "KHR_1000",
    8: "KHR_2000",
    9: "KHR_5000",
    10: "KHR_10000",
    11: "KHR_20000",
    12: "KHR_50000",
}
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
    parser = argparse.ArgumentParser(description="Render a YOLO detect or OBB label file over an image for review.")
    parser.add_argument("--image", type=Path, required=True, help="Image to annotate.")
    parser.add_argument("--labels", type=Path, required=True, help="YOLO detect or OBB .txt label file.")
    parser.add_argument("--out", type=Path, required=True, help="Output preview image path.")
    parser.add_argument("--data-yaml", type=Path, default=None, help="Optional YOLO data.yaml to read class names.")
    parser.add_argument("--format", choices=["auto", "detect", "obb"], default="auto", help="Label format.")
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


def read_labels(path: Path, label_format: str) -> list[tuple[int, list[float]]]:
    rows: list[tuple[int, list[float]]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        expected_lengths = {"detect": {5}, "obb": {9}, "auto": {5, 9}}[label_format]
        if len(parts) not in expected_lengths:
            expected = " or ".join(str(value) for value in sorted(expected_lengths))
            raise ValueError(f"{path}: line {line_number} has {len(parts)} fields; expected {expected}")
        class_id = int(parts[0])
        rows.append((class_id, [float(value) for value in parts[1:]]))
    return rows


def load_names(data_yaml: Path | None) -> dict[int, str]:
    if data_yaml is None:
        return DEFAULT_NAMES
    path = resolve_path(data_yaml)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    raw_names = data.get("names")
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(index): str(name) for index, name in raw_names.items()}
    raise ValueError(f"{path}: expected names list or mapping")


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[float, float, float, float], text: str, color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
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


def draw_polygon_label(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], text: str, color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    draw.line(points + [points[0]], fill=color, width=4)
    x1 = min(x for x, _y in points)
    y1 = min(y for _x, y in points)
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
    label_path = resolve_path(args.labels)
    out_path = resolve_path(args.out)

    with Image.open(image_path) as image:
        preview = image.convert("RGB")
    names = load_names(args.data_yaml)
    original_width, original_height = preview.size
    scale = min(1.0, args.max_side / max(original_width, original_height))
    if scale < 1.0:
        preview = preview.resize((round(original_width * scale), round(original_height * scale)), Image.Resampling.LANCZOS)

    width, height = preview.size
    draw = ImageDraw.Draw(preview)
    font = load_font(max(18, round(max(width, height) * 0.018)))
    for class_id, values in read_labels(label_path, args.format):
        color = COLORS[class_id % len(COLORS)]
        name = names.get(class_id, f"class_{class_id}")
        if len(values) == 4:
            cx, cy, box_width, box_height = values
            x1 = (cx - box_width / 2) * width
            y1 = (cy - box_height / 2) * height
            x2 = (cx + box_width / 2) * width
            y2 = (cy + box_height / 2) * height
            draw_label(draw, (x1, y1, x2, y2), name, color, font)
        else:
            points = [(values[index] * width, values[index + 1] * height) for index in range(0, 8, 2)]
            draw_polygon_label(draw, points, name, color, font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out_path, quality=92)
    print(f"wrote {out_path.relative_to(ROOT)} ({width}x{height})")


if __name__ == "__main__":
    main()
