from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class BoxItem:
    image_path: Path
    class_id: int
    class_name: str
    xyxy: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a real-photo fragment classifier dataset from YOLO labels.")
    parser.add_argument("--data", default="configs/cashsnap_v1.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--classes", default="", help="Optional comma-separated class names. Defaults to all classes.")
    parser.add_argument("--fragments-per-box", type=int, default=2)
    parser.add_argument("--max-train-per-class", type=int, default=800)
    parser.add_argument("--max-val-per-class", type=int, default=180)
    parser.add_argument("--max-test-per-class", type=int, default=140)
    parser.add_argument("--background-train", type=int, default=800)
    parser.add_argument("--background-val", type=int, default=180)
    parser.add_argument("--background-test", type=int, default=140)
    parser.add_argument("--seed", type=int, default=20260527)
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


def load_config(path: Path) -> tuple[Path, dict[int, str], dict[str, str | list[str]]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    dataset_root = resolve(Path(config["path"]), path.parent).resolve()
    names = {int(key): value for key, value in config["names"].items()}
    return dataset_root, names, config


def split_images(dataset_root: Path, split_value: str | list[str]) -> list[Path]:
    values = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for value in values:
        path = resolve(Path(value), dataset_root)
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


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_boxes(image_path: Path, names: dict[int, str]) -> list[BoxItem]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []
    with Image.open(image_path) as image:
        width, height = image.size
    boxes: list[BoxItem] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        cx, cy, box_w, box_h = [float(value) for value in parts[1:]]
        x1 = max(0, int((cx - box_w / 2) * width))
        y1 = max(0, int((cy - box_h / 2) * height))
        x2 = min(width, int((cx + box_w / 2) * width))
        y2 = min(height, int((cy + box_h / 2) * height))
        if x2 - x1 < 12 or y2 - y1 < 12 or class_id not in names:
            continue
        boxes.append(BoxItem(image_path, class_id, names[class_id], (x1, y1, x2, y2)))
    return boxes


def fragment_box(xyxy: tuple[int, int, int, int], rng: random.Random) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    mode = rng.choice(["vertical_strip", "wide_strip", "corner", "center_patch", "fullish"])
    if mode == "vertical_strip":
        crop_w = max(10, round(width * rng.uniform(0.12, 0.42)))
        crop_h = max(10, round(height * rng.uniform(0.62, 1.0)))
        left = rng.randint(x1, max(x1, x2 - crop_w))
        top = rng.randint(y1, max(y1, y2 - crop_h))
    elif mode == "wide_strip":
        crop_w = max(10, round(width * rng.uniform(0.35, 0.85)))
        crop_h = max(10, round(height * rng.uniform(0.28, 0.75)))
        left = rng.randint(x1, max(x1, x2 - crop_w))
        top = rng.randint(y1, max(y1, y2 - crop_h))
    elif mode == "corner":
        crop_w = max(10, round(width * rng.uniform(0.22, 0.58)))
        crop_h = max(10, round(height * rng.uniform(0.35, 0.82)))
        left = rng.choice([x1, max(x1, x2 - crop_w)])
        top = rng.choice([y1, max(y1, y2 - crop_h)])
    elif mode == "center_patch":
        crop_w = max(10, round(width * rng.uniform(0.28, 0.62)))
        crop_h = max(10, round(height * rng.uniform(0.35, 0.82)))
        center_x = rng.randint(x1 + width // 4, max(x1 + width // 4, x1 + width * 3 // 4))
        center_y = rng.randint(y1 + height // 4, max(y1 + height // 4, y1 + height * 3 // 4))
        left = min(max(x1, center_x - crop_w // 2), max(x1, x2 - crop_w))
        top = min(max(y1, center_y - crop_h // 2), max(y1, y2 - crop_h))
    else:
        crop_w = max(10, round(width * rng.uniform(0.70, 1.0)))
        crop_h = max(10, round(height * rng.uniform(0.70, 1.0)))
        left = rng.randint(x1, max(x1, x2 - crop_w))
        top = rng.randint(y1, max(y1, y2 - crop_h))
    return left, top, min(x2, left + crop_w), min(y2, top + crop_h)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def random_background_box(image_size: tuple[int, int], boxes: list[BoxItem], rng: random.Random) -> tuple[int, int, int, int] | None:
    width, height = image_size
    for _ in range(80):
        crop_w = rng.randint(max(24, width // 12), max(25, width // 3))
        crop_h = rng.randint(max(24, height // 12), max(25, height // 3))
        left = rng.randint(0, max(0, width - crop_w))
        top = rng.randint(0, max(0, height - crop_h))
        candidate = (left, top, left + crop_w, top + crop_h)
        if all(iou(candidate, box.xyxy) < 0.03 for box in boxes):
            return candidate
    return None


def write_crop(
    image: Image.Image,
    xyxy: tuple[int, int, int, int],
    out_dir: Path,
    split: str,
    class_name: str,
    stem: str,
    index: int,
    rows: list[dict[str, str]],
    source_path: Path,
) -> None:
    target = out_dir / split / class_name / f"{stem}_{index:06d}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    image.crop(xyxy).convert("RGB").save(target, quality=91)
    rows.append(
        {
            "split": split,
            "class_name": class_name,
            "source_path": str(source_path.relative_to(ROOT)),
            "image_path": str(target.relative_to(ROOT)),
            "xyxy": " ".join(str(value) for value in xyxy),
        }
    )


def cap_items(items: list[BoxItem], cap: int, rng: random.Random) -> list[BoxItem]:
    shuffled = items[:]
    rng.shuffle(shuffled)
    return shuffled[:cap]


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    config_path = resolve(Path(args.data), ROOT)
    dataset_root, names, config = load_config(config_path)
    wanted = {item.strip() for item in args.classes.split(",") if item.strip()} or set(names.values())
    out_dir = resolve(Path(args.out), ROOT)
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caps = {
        "train": args.max_train_per_class,
        "val": args.max_val_per_class,
        "test": args.max_test_per_class,
    }
    background_caps = {
        "train": args.background_train,
        "val": args.background_val,
        "test": args.background_test,
    }
    rows: list[dict[str, str]] = []
    split_boxes: dict[str, list[BoxItem]] = {}
    split_images_by_name: dict[str, list[Path]] = {}
    for split in ["train", "val", "test"]:
        if split not in config:
            continue
        images = split_images(dataset_root, config[split])
        split_images_by_name[split] = images
        boxes = [box for image in images for box in read_boxes(image, names) if box.class_name in wanted]
        split_boxes[split] = boxes
        by_class: dict[str, list[BoxItem]] = defaultdict(list)
        for box in boxes:
            by_class[box.class_name].append(box)
        for class_name in sorted(wanted):
            selected = cap_items(by_class[class_name], caps[split], rng)
            written = 0
            for item in selected:
                with Image.open(item.image_path).convert("RGB") as image:
                    for _ in range(args.fragments_per_box):
                        crop_box = fragment_box(item.xyxy, rng)
                        write_crop(image, crop_box, out_dir, split, class_name, item.image_path.stem, written, rows, item.image_path)
                        written += 1
            print(f"{split} {class_name}: boxes={len(by_class[class_name])} fragments={written}")

        background_written = 0
        images_shuffled = images[:]
        rng.shuffle(images_shuffled)
        boxes_by_image: dict[Path, list[BoxItem]] = defaultdict(list)
        for box in split_boxes[split]:
            boxes_by_image[box.image_path].append(box)
        while background_written < background_caps[split] and images_shuffled:
            image_path = images_shuffled[background_written % len(images_shuffled)]
            with Image.open(image_path).convert("RGB") as image:
                crop_box = random_background_box(image.size, boxes_by_image[image_path], rng)
                if crop_box is None:
                    background_written += 1
                    continue
                write_crop(image, crop_box, out_dir, split, "background", image_path.stem, background_written, rows, image_path)
            background_written += 1
        print(f"{split} background: fragments={background_written}")

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} crops to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
