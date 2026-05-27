from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
CSV_CACHE: dict[Path, list[dict[str, str]]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build review manifests/contact sheets from classifier prediction CSVs.")
    parser.add_argument("--predictions", nargs="+", required=True, help="One or more evaluate_fragment_classifier prediction CSVs.")
    parser.add_argument("--out", required=True, help="Output review-pack directory, normally under data/review/.")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-per-pair", type=int, default=60)
    parser.add_argument("--pairs", default="", help="Optional comma list like KHR_5000->KHR_10000,KHR_20000->KHR_5000.")
    parser.add_argument("--include-correct", action="store_true", help="Include correct predictions too.")
    parser.add_argument("--thumb", type=int, default=180)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (ROOT / "data").resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise SystemExit(f"Refusing to clean outside {allowed_root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def wanted_pairs(raw_pairs: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for raw_pair in raw_pairs.split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if "->" not in pair:
            raise SystemExit(f"Pair must use target->prediction form: {pair}")
        target, prediction = [part.strip() for part in pair.split("->", 1)]
        pairs.add((target, prediction))
    return pairs


def relative_to_root(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def normalized_path(path_text: str) -> str:
    return relative_to_root(resolve(path_text)).replace("\\", "/")


def read_csv(path: Path) -> list[dict[str, str]]:
    resolved = path.resolve()
    if resolved not in CSV_CACHE:
        with resolved.open("r", newline="", encoding="utf-8") as handle:
            CSV_CACHE[resolved] = list(csv.DictReader(handle))
    return CSV_CACHE[resolved]


def source_row_for(dataset_row: dict[str, str]) -> dict[str, str]:
    source_manifest_text = dataset_row.get("source_manifest", "")
    source_crop_text = dataset_row.get("source_crop", "")
    if not source_manifest_text or not source_crop_text:
        return {}
    source_manifest = resolve(source_manifest_text)
    if not source_manifest.exists():
        return {}
    source_crop = normalized_path(source_crop_text)
    for source_row in read_csv(source_manifest):
        if normalized_path(source_row.get("crop_path", "")) == source_crop:
            return source_row
    return {}


def metadata_for_image(image_path: Path) -> dict[str, str]:
    dataset_root = image_path.parents[2]
    dataset_manifest = dataset_root / "manifest.csv"
    if not dataset_manifest.exists():
        return {}
    image_key = relative_to_root(image_path).replace("\\", "/")
    for dataset_row in read_csv(dataset_manifest):
        if normalized_path(dataset_row.get("image_path", "")) != image_key:
            continue
        source_row = source_row_for(dataset_row)
        return {
            "side": source_row.get("side", ""),
            "source_image": source_row.get("source_image", ""),
            "source_crop": dataset_row.get("source_crop", ""),
            "source_review_manifest": dataset_row.get("source_manifest", ""),
            "label_index": source_row.get("label_index", ""),
            "box_area_frac": source_row.get("box_area_frac", ""),
        }
    return {}


def read_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    pairs = wanted_pairs(args.pairs)
    rows: list[dict[str, str]] = []
    for prediction_csv_text in args.predictions:
        prediction_csv = resolve(prediction_csv_text)
        with prediction_csv.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                target = row.get("target", "")
                prediction = row.get("prediction", "")
                confidence = float(row.get("confidence", "0") or 0)
                correct = row.get("correct", "").strip().lower() == "true"
                if not args.include_correct and correct:
                    continue
                if confidence < args.min_confidence:
                    continue
                if pairs and (target, prediction) not in pairs:
                    continue
                image_path = resolve(row["image_path"])
                if not image_path.exists():
                    raise SystemExit(f"Missing prediction image: {image_path}")
                pair = f"{target}->{prediction}"
                metadata = metadata_for_image(image_path)
                rows.append(
                    {
                        "split": row.get("split", ""),
                        "target": target,
                        "prediction": prediction,
                        "confidence": f"{confidence:.6f}",
                        "correct": str(correct).lower(),
                        "failure_pair": pair,
                        "class_name": target,
                        "crop_path": relative_to_root(image_path),
                        "source_prediction_csv": relative_to_root(prediction_csv),
                        **metadata,
                    }
                )
    rows.sort(key=lambda row: (row["failure_pair"], -float(row["confidence"]), row["crop_path"]))
    selected: list[dict[str, str]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        if counts[row["failure_pair"]] >= args.max_per_pair:
            continue
        counts[row["failure_pair"]] += 1
        selected.append(row)
    return selected


def write_contact_sheet(rows: list[dict[str, str]], out_path: Path, thumb: int, limit: int = 96) -> None:
    selected = rows[:limit]
    if not selected:
        return
    cols = 6
    label_h = 48
    sheet = Image.new("RGB", (cols * thumb, ((len(selected) + cols - 1) // cols) * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        image_path = resolve(row["crop_path"])
        with Image.open(image_path).convert("RGB") as image:
            image = ImageOps.contain(image, (thumb, thumb))
            tile = Image.new("RGB", (thumb, thumb), (244, 244, 244))
            tile.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
        x = (index % cols) * thumb
        y = (index // cols) * (thumb + label_h)
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + thumb + 4), row["failure_pair"][:28], fill="black")
        draw.text((x + 4, y + thumb + 22), f"{row['split']} conf {row['confidence']}", fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out)
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args)
    for index, row in enumerate(rows):
        row["crop_id"] = f"{row['split']}_{row['failure_pair'].replace('->', '_to_')}_{index:04d}"
        row["review_include"] = ""
        row["review_class"] = row["target"]
        row["review_notes"] = ""

    fieldnames = [
        "crop_id",
        "split",
        "target",
        "prediction",
        "confidence",
        "correct",
        "failure_pair",
        "class_name",
        "crop_path",
        "source_prediction_csv",
        "source_review_manifest",
        "source_crop",
        "source_image",
        "side",
        "label_index",
        "box_area_frac",
        "review_include",
        "review_class",
        "review_notes",
    ]
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_pair: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_pair[row["failure_pair"]].append(row)
    for pair, pair_rows in sorted(by_pair.items()):
        safe_pair = pair.replace("->", "_to_")
        write_contact_sheet(pair_rows, out_dir / f"contact_sheet_{safe_pair}.jpg", args.thumb)
        print(f"{pair}: {len(pair_rows)}")
    write_contact_sheet(rows, out_dir / "contact_sheet_mixed.jpg", args.thumb)
    print(f"wrote {len(rows)} rows to {relative_to_root(out_dir / 'manifest.csv')}")


if __name__ == "__main__":
    main()
