from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {11: "KHR_20000", 12: "KHR_50000"}
IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package KHR_20000/KHR_50000 crops for background-removal cleanup."
    )
    parser.add_argument("--dataset", default="data/cashsnap_v1", help="YOLO dataset root.")
    parser.add_argument("--out", default="data/asset_candidates/picwish_input", help="Output folder.")
    parser.add_argument("--padding", type=float, default=0.08, help="Box padding ratio around each crop.")
    parser.add_argument("--include-nbc", action="store_true", help="Also copy curated NBC reference notes.")
    parser.add_argument("--include-numista", action="store_true", help="Also copy downloaded Numista rare-note references.")
    parser.add_argument("--max-per-class", type=int, default=0, help="Optional cap per class, 0 means no cap.")
    return parser.parse_args()


def find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def crop_box(size: tuple[int, int], yolo_box: list[float], padding: float) -> tuple[int, int, int, int]:
    width, height = size
    x, y, w, h = yolo_box
    pad_x = w * width * padding
    pad_y = h * height * padding
    left = max(0, int((x - w / 2) * width - pad_x))
    top = max(0, int((y - h / 2) * height - pad_y))
    right = min(width, int((x + w / 2) * width + pad_x))
    bottom = min(height, int((y + h / 2) * height + pad_y))
    return left, top, right, bottom


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(paths: list[Path], output: Path, title: str) -> None:
    if not paths:
        return
    thumb_w, thumb_h = 220, 160
    cols = 5
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 34) + 34), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for index, path in enumerate(paths):
        with Image.open(path).convert("RGB") as image:
            image.thumbnail((thumb_w, thumb_h))
            x = (index % cols) * thumb_w
            y = 34 + (index // cols) * (thumb_h + 34)
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + thumb_h + 4), path.name[:31], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def package_dataset_crops(dataset: Path, out_dir: Path, padding: float, max_per_class: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counts = {label: 0 for label in TARGETS.values()}
    for split in ["train", "val", "test"]:
        label_dir = dataset / "labels" / split
        image_dir = dataset / "images" / split
        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = find_image(image_dir, label_path.stem)
            if image_path is None:
                continue
            rare_boxes: list[tuple[int, list[str], list[float]]] = []
            for box_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(float(parts[0]))
                if class_id not in TARGETS:
                    continue
                label = TARGETS[class_id]
                if max_per_class and counts[label] >= max_per_class:
                    continue
                rare_boxes.append((box_index, parts, [float(value) for value in parts[1:]]))
            if not rare_boxes:
                continue
            with Image.open(image_path).convert("RGB") as image:
                for box_index, parts, box in rare_boxes:
                    class_id = int(float(parts[0]))
                    label = TARGETS[class_id]
                    if max_per_class and counts[label] >= max_per_class:
                        continue
                    left, top, right, bottom = crop_box(image.size, box, padding)
                    if right <= left or bottom <= top:
                        continue
                    crop = image.crop((left, top, right, bottom))
                    counts[label] += 1
                    target = out_dir / label / "scene_crops" / f"{split}_{label}_{counts[label]:04d}_{label_path.stem}_{box_index}.jpg"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(target, quality=94)
                    rows.append(
                        {
                            "label": label,
                            "kind": "scene_crop",
                            "split": split,
                            "source_image": str(image_path.relative_to(ROOT)),
                            "source_label": str(label_path.relative_to(ROOT)),
                            "crop_path": str(target.relative_to(ROOT)),
                            "source_box_yolo": " ".join(parts[1:]),
                            "crop_box_xyxy": f"{left} {top} {right} {bottom}",
                        }
                    )
    return rows


def copy_nbc_references(out_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    nbc_root = ROOT / "data" / "curated" / "reference" / "khr_nbc" / "target"
    for label in TARGETS.values():
        source_dir = nbc_root / label
        if not source_dir.exists():
            continue
        for source in sorted(source_dir.glob("*")):
            if source.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            target = out_dir / label / "nbc_reference" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            rows.append(
                {
                    "label": label,
                    "kind": "nbc_reference",
                    "split": "",
                    "source_image": str(source.relative_to(ROOT)),
                    "source_label": "",
                    "crop_path": str(target.relative_to(ROOT)),
                    "source_box_yolo": "",
                    "crop_box_xyxy": "full_image",
                }
            )
    return rows


def copy_numista_references(out_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    source_root = ROOT / "data" / "reference" / "numista_rare_khr"
    if not source_root.exists():
        return rows
    for source in sorted(source_root.rglob("*")):
        if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = next((part for part in source.parts if part in TARGETS.values()), "")
        if not label:
            continue
        target = out_dir / label / "numista_reference" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append(
            {
                "label": label,
                "kind": "numista_reference",
                "split": "",
                "source_image": str(source.relative_to(ROOT)),
                "source_label": "",
                "crop_path": str(target.relative_to(ROOT)),
                "source_box_yolo": "",
                "crop_box_xyxy": "full_image",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    dataset = (ROOT / args.dataset).resolve()
    out_dir = (ROOT / args.out).resolve()
    rows = package_dataset_crops(dataset, out_dir, args.padding, args.max_per_class)
    if args.include_nbc:
        rows.extend(copy_nbc_references(out_dir))
    if args.include_numista:
        rows.extend(copy_numista_references(out_dir))

    write_csv(out_dir / "manifest.csv", rows)
    for label in TARGETS.values():
        paths = [ROOT / row["crop_path"] for row in rows if row["label"] == label]
        make_contact_sheet(paths, out_dir / f"{label}_contact.jpg", f"{label} PicWish input candidates")

    print(f"Wrote {len(rows)} candidate assets to {out_dir}")
    for label in TARGETS.values():
        print(f"{label}: {sum(1 for row in rows if row['label'] == label)}")
    print(f"Manifest: {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
