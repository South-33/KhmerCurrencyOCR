from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class CropItem:
    split: str
    image_path: Path
    class_id: int
    class_name: str
    label_index: int
    xyxy: tuple[int, int, int, int]
    image_size: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build contact-sheet review packs from YOLO-labeled crops.")
    parser.add_argument("--data", default="configs/cashsnap_v1.yaml", help="YOLO dataset YAML.")
    parser.add_argument("--out", required=True, help="Output review-pack directory under data/.")
    parser.add_argument("--classes", required=True, help="Comma-separated classes to include.")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated splits to include.")
    parser.add_argument("--max-per-class-split", type=int, default=120)
    parser.add_argument("--min-side", type=int, default=36)
    parser.add_argument("--pad", type=float, default=0.04)
    parser.add_argument("--thumb", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--edge-only", action="store_true", help="Only include boxes touching the image edge.")
    parser.add_argument("--edge-margin", type=float, default=0.01)
    parser.add_argument("--min-box-area", type=float, default=None, help="Minimum normalized bbox area.")
    parser.add_argument("--max-box-area", type=float, default=None, help="Maximum normalized bbox area.")
    parser.add_argument("--select", choices=["random", "smallest", "largest"], default="random")
    parser.add_argument(
        "--canonicalize-roboflow-currency",
        action="store_true",
        help="Map raw names like 20000-riel-b to review_class=KHR_20000 while preserving side metadata.",
    )
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (ROOT / "data").resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise SystemExit(f"Refusing to clean outside {allowed_root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def resolve(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else base / path


def load_config(path: Path) -> tuple[Path, dict[int, str], dict[str, object]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    dataset_root = resolve(Path(config["path"]), path.parent).resolve() if "path" in config else path.parent.resolve()
    raw_names = config["names"]
    if isinstance(raw_names, dict):
        names = {int(key): str(value) for key, value in raw_names.items()}
    else:
        names = {index: str(value) for index, value in enumerate(raw_names)}
    return dataset_root, names, config


def split_images(dataset_root: Path, split_value: object) -> list[Path]:
    values = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for raw_value in values:
        value = str(raw_value)
        path = resolve(Path(value), dataset_root)
        if not path.exists():
            parts = {part.lower() for part in Path(value).parts}
            fallback_split = next((split for split in ["train", "valid", "val", "test"] if split in parts), "")
            if fallback_split:
                normalized = "valid" if fallback_split == "val" else fallback_split
                fallback = dataset_root / normalized / "images"
                if fallback.exists():
                    path = fallback
        if path.suffix.lower() == ".txt":
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                image = Path(line)
                images.append(image if image.is_absolute() else dataset_root / image)
        else:
            images.extend(sorted(p for p in path.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES))
    return images


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def parse_yolo_box(parts: list[str]) -> tuple[float, float, float, float] | None:
    if len(parts) == 5:
        return tuple(float(value) for value in parts[1:])  # type: ignore[return-value]
    if len(parts) >= 7 and len(parts[1:]) % 2 == 0:
        coords = [float(value) for value in parts[1:]]
        xs = coords[0::2]
        ys = coords[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return None


def normalized_box_matches(box: tuple[float, float, float, float], args: argparse.Namespace) -> bool:
    cx, cy, box_w, box_h = box
    box_area = box_w * box_h
    if args.min_box_area is not None and box_area < args.min_box_area:
        return False
    if args.max_box_area is not None and box_area > args.max_box_area:
        return False
    if args.edge_only:
        x1 = cx - box_w / 2
        y1 = cy - box_h / 2
        x2 = cx + box_w / 2
        y2 = cy + box_h / 2
        return (
            x1 <= args.edge_margin
            or y1 <= args.edge_margin
            or x2 >= 1.0 - args.edge_margin
            or y2 >= 1.0 - args.edge_margin
        )
    return True


def item_area_frac(item: CropItem) -> float:
    x1, y1, x2, y2 = item.xyxy
    image_w, image_h = item.image_size
    return ((x2 - x1) * (y2 - y1)) / max(1, image_w * image_h)


def roboflow_currency_metadata(class_name: str, canonicalize: bool) -> dict[str, str]:
    if not canonicalize:
        return {
            "currency": "",
            "denomination": "",
            "side": "",
            "canonical_class": class_name,
            "review_class": class_name,
        }
    parts = class_name.lower().split("-")
    side = {"f": "front", "b": "back"}.get(parts[-1] if parts else "", "unknown")
    amount = next((part for part in parts if part.isdigit()), "")
    if "riel" in parts:
        currency = "KHR"
    elif "us" in parts:
        currency = "USD"
    else:
        currency = ""
    canonical_class = f"{currency}_{amount}" if currency and amount else class_name
    return {
        "currency": currency,
        "denomination": amount,
        "side": side,
        "canonical_class": canonical_class,
        "review_class": canonical_class,
    }


def read_items(
    split: str,
    image_path: Path,
    names: dict[int, str],
    wanted: set[str],
    min_side: int,
    args: argparse.Namespace,
) -> list[CropItem]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []
    with Image.open(image_path) as image:
        width, height = image.size
    items: list[CropItem] = []
    for label_index, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        parts = raw_line.split()
        class_id = int(float(parts[0]))
        class_name = names.get(class_id, "")
        if class_name not in wanted:
            continue
        box = parse_yolo_box(parts)
        if box is None:
            continue
        if not normalized_box_matches(box, args):
            continue
        cx, cy, box_w, box_h = box
        x1 = max(0, int((cx - box_w / 2) * width))
        y1 = max(0, int((cy - box_h / 2) * height))
        x2 = min(width, int((cx + box_w / 2) * width))
        y2 = min(height, int((cy + box_h / 2) * height))
        if x2 - x1 < min_side or y2 - y1 < min_side:
            continue
        items.append(CropItem(split, image_path, class_id, class_name, label_index, (x1, y1, x2, y2), (width, height)))
    return items


def padded_box(item: CropItem, pad: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = item.xyxy
    width, height = item.image_size
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(round(box_w * pad))
    pad_y = int(round(box_h * pad))
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def write_contact_sheet(rows: list[dict[str, str]], out_path: Path, thumb: int, limit: int = 96) -> None:
    selected = rows[:limit]
    if not selected:
        return
    cols = 6
    label_h = 42
    sheet = Image.new("RGB", (cols * thumb, ((len(selected) + cols - 1) // cols) * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        image_path = ROOT / row["crop_path"]
        with Image.open(image_path).convert("RGB") as image:
            image = ImageOps.contain(image, (thumb, thumb))
            tile = Image.new("RGB", (thumb, thumb), (244, 244, 244))
            tile.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
        x = (index % cols) * thumb
        y = (index // cols) * (thumb + label_h)
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + thumb + 4), f"{row['split']} {row['class_name']}", fill="black")
        draw.text((x + 4, y + thumb + 22), row["crop_id"][:28], fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    config_path = resolve(Path(args.data), ROOT)
    dataset_root, names, config = load_config(config_path)
    wanted = {value.strip() for value in args.classes.split(",") if value.strip()}
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    out_dir = resolve(Path(args.out), ROOT)
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for split in splits:
        if split not in config:
            continue
        items = [
            item
            for image in split_images(dataset_root, config[split])
            for item in read_items(split, image, names, wanted, args.min_side, args)
        ]
        by_class: dict[str, list[CropItem]] = defaultdict(list)
        for item in items:
            by_class[item.class_name].append(item)
        for class_name in sorted(wanted):
            selected = by_class[class_name][:]
            if args.select == "smallest":
                selected = sorted(selected, key=item_area_frac)[: args.max_per_class_split]
            elif args.select == "largest":
                selected = sorted(selected, key=item_area_frac, reverse=True)[: args.max_per_class_split]
            else:
                rng.shuffle(selected)
                selected = selected[: args.max_per_class_split]
            for index, item in enumerate(selected):
                box = padded_box(item, args.pad)
                crop_id = f"{item.split}_{class_name}_{index:04d}_{item.image_path.stem}_{item.label_index}"
                crop_path = out_dir / "crops" / class_name / f"{crop_id}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(item.image_path).convert("RGB") as image:
                    image.crop(box).save(crop_path, quality=92)
                x1, y1, x2, y2 = item.xyxy
                image_w, image_h = item.image_size
                rows.append(
                    {
                        "crop_id": crop_id,
                        "split": item.split,
                        "class_id": str(item.class_id),
                        "class_name": class_name,
                        **roboflow_currency_metadata(class_name, args.canonicalize_roboflow_currency),
                        "source_image": str(item.image_path.relative_to(ROOT)),
                        "crop_path": str(crop_path.relative_to(ROOT)),
                        "label_index": str(item.label_index),
                        "xyxy": " ".join(str(value) for value in item.xyxy),
                        "padded_xyxy": " ".join(str(value) for value in box),
                        "box_area_frac": f"{((x2 - x1) * (y2 - y1)) / max(1, image_w * image_h):.6f}",
                        "review_include": "",
                        "review_notes": "",
                    }
                )
            print(f"{split} {class_name}: source_boxes={len(by_class[class_name])} review_crops={len(selected)}")

    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "crop_id",
            "split",
            "class_id",
            "class_name",
            "currency",
            "denomination",
            "side",
            "canonical_class",
            "source_image",
            "crop_path",
            "label_index",
            "xyxy",
            "padded_xyxy",
            "box_area_frac",
            "review_include",
            "review_class",
            "review_notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for class_name in sorted(wanted):
        class_rows = [row for row in rows if row["class_name"] == class_name]
        write_contact_sheet(class_rows, out_dir / f"contact_sheet_{class_name}.jpg", args.thumb)
    write_contact_sheet(rows, out_dir / "contact_sheet_mixed.jpg", args.thumb)
    print(f"wrote {len(rows)} review crops to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
