from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create contact sheets for YOLO dataset visual QA.")
    parser.add_argument("--data", default="configs/cashsnap_v1.yaml")
    parser.add_argument("--out", default="data/audit")
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--edge-only", action="store_true", help="Only sample labels touching the image edge.")
    parser.add_argument("--edge-margin", type=float, default=0.01)
    parser.add_argument("--min-bbox-area", type=float, default=None)
    parser.add_argument("--max-bbox-area", type=float, default=None)
    parser.add_argument("--select", choices=["random", "smallest", "largest"], default="random")
    return parser.parse_args()


def load_config(path: Path) -> tuple[Path, dict[int, str]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = (path.parent / config["path"]).resolve() if "path" in config else path.parent.resolve()
    raw_names = config["names"]
    if isinstance(raw_names, dict):
        names = {int(key): value for key, value in raw_names.items()}
    else:
        names = {index: value for index, value in enumerate(raw_names)}
    return root, names


def find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in [".jpg", ".jpeg", ".png", ".webp"]:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def split_dirs(root: Path, split: str) -> tuple[Path, Path] | None:
    candidates = [
        (root / "labels" / split, root / "images" / split),
        (root / split / "labels", root / split / "images"),
    ]
    for label_dir, image_dir in candidates:
        if label_dir.exists() and image_dir.exists():
            return label_dir, image_dir
    return None


def parse_yolo_shape(parts: list[str]) -> tuple[list[float], list[tuple[float, float]] | None] | None:
    if len(parts) == 5:
        return [float(value) for value in parts[1:]], None
    if len(parts) >= 7 and len(parts[1:]) % 2 == 0:
        coords = [float(value) for value in parts[1:]]
        xs = coords[0::2]
        ys = coords[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        polygon = list(zip(xs, ys, strict=False))
        return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], polygon
    return None


def sample_matches(box: list[float], args: argparse.Namespace) -> bool:
    x_center, y_center, box_w, box_h = box
    bbox_area = box_w * box_h
    if args.min_bbox_area is not None and bbox_area < args.min_bbox_area:
        return False
    if args.max_bbox_area is not None and bbox_area > args.max_bbox_area:
        return False
    if args.edge_only:
        x1 = x_center - box_w / 2
        y1 = y_center - box_h / 2
        x2 = x_center + box_w / 2
        y2 = y_center + box_h / 2
        return (
            x1 <= args.edge_margin
            or y1 <= args.edge_margin
            or x2 >= 1.0 - args.edge_margin
            or y2 >= 1.0 - args.edge_margin
        )
    return True


def collect_samples(
    root: Path,
    names: dict[int, str],
    args: argparse.Namespace,
) -> dict[int, list[tuple[Path, list[float], list[tuple[float, float]] | None]]]:
    samples: dict[int, list[tuple[Path, list[float], list[tuple[float, float]] | None]]] = defaultdict(list)
    for split in ["train", "valid", "val", "test"]:
        dirs = split_dirs(root, split)
        if dirs is None:
            continue
        label_dir, image_dir = dirs
        for label_path in label_dir.glob("*.txt"):
            image_path = find_image(image_dir, label_path.stem)
            if image_path is None:
                continue
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                class_id = int(parts[0])
                if class_id not in names:
                    continue
                shape = parse_yolo_shape(parts)
                if shape is None:
                    continue
                box, polygon = shape
                if sample_matches(box, args):
                    samples[class_id].append((image_path, box, polygon))
    return samples


def draw_crop(
    image_path: Path,
    box: list[float],
    polygon: list[tuple[float, float]] | None,
    label: str,
) -> Image.Image:
    with Image.open(image_path).convert("RGB") as image:
        width, height = image.size
        x_center, y_center, box_w, box_h = box
        x1 = int((x_center - box_w / 2) * width)
        y1 = int((y_center - box_h / 2) * height)
        x2 = int((x_center + box_w / 2) * width)
        y2 = int((y_center + box_h / 2) * height)
        draw = ImageDraw.Draw(image)
        line_width = max(2, width // 250)
        if polygon:
            pixel_points = [(int(x * width), int(y * height)) for x, y in polygon]
            draw.line(pixel_points + [pixel_points[0]], fill="lime", width=line_width)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=max(2, width // 250))
        draw.rectangle([0, 0, min(width, 340), 28], fill="white")
        draw.text((4, 6), f"{label} area={box_w * box_h:.4f} | {image_path.name[:22]}", fill="black")
        image.thumbnail((260, 180))
        return image.copy()


def make_sheet(
    class_id: int,
    label: str,
    chosen: list[tuple[Path, list[float], list[tuple[float, float]] | None]],
    out_dir: Path,
) -> None:
    thumbs = [draw_crop(image_path, box, polygon, label) for image_path, box, polygon in chosen]
    if not thumbs:
        return
    cols = 4
    cell_w, cell_h = 260, 180
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for index, thumb in enumerate(thumbs):
        x = (index % cols) * cell_w
        y = (index // cols) * cell_h
        sheet.paste(thumb, (x, y))
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(out_dir / f"{class_id:02d}_{label}.jpg", quality=92)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    root, names = load_config(Path(args.data))
    samples = collect_samples(root, names, args)
    out_dir = Path(args.out)
    for class_id, label in names.items():
        available = samples.get(class_id, [])
        if args.select == "smallest":
            chosen = sorted(available, key=lambda item: item[1][2] * item[1][3])[: args.per_class]
        elif args.select == "largest":
            chosen = sorted(available, key=lambda item: item[1][2] * item[1][3], reverse=True)[: args.per_class]
        else:
            chosen = random.sample(available, min(args.per_class, len(available)))
        make_sheet(class_id, label, chosen, out_dir)
        print(f"{label}: {len(available)} available, wrote {len(chosen)}")


if __name__ == "__main__":
    main()
