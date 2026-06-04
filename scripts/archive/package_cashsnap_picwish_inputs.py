from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
CSV_FIELDS = [
    "label",
    "class_id",
    "kind",
    "split",
    "source_image",
    "source_label",
    "crop_path",
    "source_box_yolo",
    "crop_box_xyxy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package CashSnap YOLO crops for background-removal cutout generation."
    )
    parser.add_argument("--dataset", default="data/cashsnap_v1", help="YOLO dataset root.")
    parser.add_argument("--data-yaml", default="", help="Optional data.yaml path; defaults to <dataset>/data.yaml.")
    parser.add_argument(
        "--out",
        default="data/asset_candidates/cashsnap_picwish_input",
        help="Output folder for candidate crops.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        help="Class names to package. Accepts repeated names or comma-separated lists. Defaults to all KHR_* classes.",
    )
    parser.add_argument(
        "--class-ids",
        nargs="*",
        help="Class ids to package. Accepts repeated ids or comma-separated lists.",
    )
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated YOLO splits to scan.")
    parser.add_argument("--padding", type=float, default=0.08, help="Box padding ratio around each crop.")
    parser.add_argument("--max-per-class", type=int, default=0, help="Optional cap per class, 0 means no cap.")
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=0.0,
        help="Skip YOLO boxes smaller than this normalized area.",
    )
    parser.add_argument(
        "--include-nbc",
        action="store_true",
        help="Also copy curated NBC references for selected KHR classes.",
    )
    parser.add_argument(
        "--nbc-buckets",
        default="target_modern_common,legacy_or_low_priority",
        help="Comma-separated curated NBC buckets to copy when --include-nbc is set.",
    )
    parser.add_argument("--force", action="store_true", help="Clear the output folder before writing.")
    return parser.parse_args()


def split_tokens(values: list[str] | str | None) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    tokens: list[str] = []
    for value in values:
        tokens.extend(token.strip() for token in value.split(",") if token.strip())
    return tokens


def safe_clear_dir(path: Path) -> None:
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise SystemExit(f"Refusing to clear path outside repo: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def read_class_names(data_yaml: Path) -> dict[int, str]:
    if not data_yaml.exists():
        raise SystemExit(f"Missing data yaml: {data_yaml}")

    names: dict[int, str] = {}
    in_names = False
    pattern = re.compile(r"^\s*(\d+)\s*:\s*['\"]?([^'\"]+)['\"]?\s*$")
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "names:":
            in_names = True
            continue
        if in_names and not line.startswith((" ", "\t")):
            break
        if not in_names:
            continue
        match = pattern.match(line)
        if match:
            names[int(match.group(1))] = match.group(2).strip()

    if not names:
        raise SystemExit(f"No class names found in {data_yaml}")
    return names


def selected_targets(args: argparse.Namespace, names: dict[int, str]) -> dict[int, str]:
    selected_ids: set[int] = set()

    class_tokens = split_tokens(args.classes)
    if class_tokens:
        by_name = {name: class_id for class_id, name in names.items()}
        unknown = [name for name in class_tokens if name not in by_name]
        if unknown:
            raise SystemExit(f"Unknown classes: {', '.join(unknown)}")
        selected_ids.update(by_name[name] for name in class_tokens)

    id_tokens = split_tokens(args.class_ids)
    if id_tokens:
        try:
            ids = {int(token) for token in id_tokens}
        except ValueError as exc:
            raise SystemExit(f"Class ids must be integers: {', '.join(id_tokens)}") from exc
        unknown_ids = sorted(class_id for class_id in ids if class_id not in names)
        if unknown_ids:
            raise SystemExit(f"Unknown class ids: {', '.join(str(class_id) for class_id in unknown_ids)}")
        selected_ids.update(ids)

    if not selected_ids:
        selected_ids = {class_id for class_id, name in names.items() if name.startswith("KHR_")}

    return {class_id: names[class_id] for class_id in sorted(selected_ids)}


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
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


def package_dataset_crops(
    dataset: Path,
    out_dir: Path,
    targets: dict[int, str],
    splits: list[str],
    padding: float,
    max_per_class: int,
    min_box_area: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    counts = {label: 0 for label in targets.values()}
    for split in splits:
        label_dir = dataset / "labels" / split
        image_dir = dataset / "images" / split
        if not label_dir.exists() or not image_dir.exists():
            raise SystemExit(f"Missing YOLO split folders for {split}: {label_dir} / {image_dir}")

        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = find_image(image_dir, label_path.stem)
            if image_path is None:
                continue
            target_boxes: list[tuple[int, list[str], list[float]]] = []
            for box_index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(float(parts[0]))
                if class_id not in targets:
                    continue
                label = targets[class_id]
                if max_per_class and counts[label] >= max_per_class:
                    continue
                yolo_box = [float(value) for value in parts[1:]]
                if yolo_box[2] * yolo_box[3] < min_box_area:
                    continue
                target_boxes.append((box_index, parts, yolo_box))

            if not target_boxes:
                continue

            with Image.open(image_path).convert("RGB") as image:
                for box_index, parts, box in target_boxes:
                    class_id = int(float(parts[0]))
                    label = targets[class_id]
                    if max_per_class and counts[label] >= max_per_class:
                        continue
                    left, top, right, bottom = crop_box(image.size, box, padding)
                    if right <= left or bottom <= top:
                        continue
                    crop = image.crop((left, top, right, bottom))
                    counts[label] += 1
                    target = (
                        out_dir
                        / label
                        / "scene_crops"
                        / f"{split}_{label}_{counts[label]:04d}_{label_path.stem}_{box_index}.jpg"
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    crop.save(target, quality=94)
                    rows.append(
                        {
                            "label": label,
                            "class_id": str(class_id),
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


def copy_nbc_references(out_dir: Path, targets: dict[int, str], buckets: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    nbc_root = ROOT / "data" / "curated" / "reference" / "khr_nbc"
    for class_id, label in targets.items():
        for bucket in buckets:
            source_dir = nbc_root / bucket / label
            if not source_dir.exists():
                continue
            for source in sorted(source_dir.glob("*")):
                if source.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                target = out_dir / label / f"nbc_{bucket}" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                rows.append(
                    {
                        "label": label,
                        "class_id": str(class_id),
                        "kind": f"nbc_{bucket}",
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
    data_yaml = (ROOT / args.data_yaml).resolve() if args.data_yaml else dataset / "data.yaml"
    out_dir = (ROOT / args.out).resolve()
    if args.force:
        safe_clear_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    names = read_class_names(data_yaml)
    targets = selected_targets(args, names)
    splits = split_tokens(args.splits)
    nbc_buckets = split_tokens(args.nbc_buckets)

    rows = package_dataset_crops(
        dataset=dataset,
        out_dir=out_dir,
        targets=targets,
        splits=splits,
        padding=args.padding,
        max_per_class=args.max_per_class,
        min_box_area=args.min_box_area,
    )
    if args.include_nbc:
        rows.extend(copy_nbc_references(out_dir, targets, nbc_buckets))

    write_csv(out_dir / "manifest.csv", rows)
    for label in targets.values():
        paths = [ROOT / row["crop_path"] for row in rows if row["label"] == label]
        make_contact_sheet(paths[:100], out_dir / f"{label}_contact.jpg", f"{label} background-removal candidates")

    print(f"Wrote {len(rows)} candidate assets to {out_dir}")
    for label in targets.values():
        print(f"{label}: {sum(1 for row in rows if row['label'] == label)}")
    print(f"Manifest: {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
