from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = [
    ROOT / "data" / "review" / "roboflow_cuurecy_detection_is_oldcommon_highconf_failure_review_v1" / "manifest.csv",
    ROOT / "data" / "review" / "roboflow_cuurecy_detection_is_khr_5k_10k_partial_review_v1" / "manifest.csv",
    ROOT / "data" / "review" / "roboflow_cuurecy_detection_is_khr_20k_50k_partial_review_v1" / "manifest.csv",
]
DEFAULT_OUT = ROOT / "data" / "review" / "cashsnap_p1_oldcommon_partial_focus_review_v1"
TARGET_CLASSES = {"KHR_5000", "KHR_10000", "KHR_20000"}
FIELDNAMES = [
    "crop_id",
    "priority_reason",
    "source_manifest",
    "source_crop_id",
    "split",
    "failure_pair",
    "target",
    "prediction",
    "confidence",
    "class_name",
    "side",
    "box_area_frac",
    "crop_path",
    "source_crop",
    "source_image",
    "review_include",
    "review_class",
    "review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a focused P1 old/common partial-note review queue.")
    parser.add_argument("--manifest", nargs="*", type=Path, default=DEFAULT_MANIFESTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-per-class-side", type=int, default=12)
    parser.add_argument("--max-failure-rows", type=int, default=60)
    parser.add_argument("--thumb", type=int, default=180)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_value(row: dict[str, str], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def area_value(row: dict[str, str]) -> float:
    try:
        return float(row.get("box_area_frac", "") or "999")
    except ValueError:
        return 999.0


def confidence_value(row: dict[str, str]) -> float:
    try:
        return float(row.get("confidence", "") or "0")
    except ValueError:
        return 0.0


def normalize_row(row: dict[str, str], manifest: Path, reason: str) -> dict[str, str]:
    review_class = first_value(row, ["review_class", "target", "canonical_class", "class_name"])
    return {
        "crop_id": "",
        "priority_reason": reason,
        "source_manifest": repo_path(manifest),
        "source_crop_id": row.get("crop_id", ""),
        "split": row.get("split", ""),
        "failure_pair": row.get("failure_pair", ""),
        "target": row.get("target", review_class),
        "prediction": row.get("prediction", ""),
        "confidence": row.get("confidence", ""),
        "class_name": review_class,
        "side": row.get("side", ""),
        "box_area_frac": row.get("box_area_frac", ""),
        "crop_path": row.get("crop_path", ""),
        "source_crop": row.get("source_crop", ""),
        "source_image": row.get("source_image", ""),
        "review_include": "",
        "review_class": review_class,
        "review_notes": "",
    }


def select_rows(manifests: list[Path], max_per_class_side: int, max_failure_rows: int) -> list[dict[str, str]]:
    failure_rows: list[dict[str, str]] = []
    partial_rows: list[dict[str, str]] = []
    for manifest in manifests:
        for row in read_rows(manifest):
            class_name = first_value(row, ["review_class", "target", "canonical_class", "class_name"])
            if class_name not in TARGET_CLASSES:
                continue
            if row.get("failure_pair", "").strip():
                failure_rows.append(normalize_row(row, manifest, "high_conf_failure"))
            else:
                partial_rows.append(normalize_row(row, manifest, "small_partial"))

    selected = sorted(failure_rows, key=lambda row: (-confidence_value(row), area_value(row)))[:max_failure_rows]
    grouped: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in partial_rows:
        grouped[(row["class_name"], row["side"] or "unknown")].append(row)
    for key in sorted(grouped):
        selected.extend(sorted(grouped[key], key=area_value)[:max_per_class_side])

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in selected:
        key = row["crop_path"].replace("\\", "/")
        if not key or key in seen:
            continue
        seen.add(key)
        row["crop_id"] = f"p1_focus_{len(deduped):04d}"
        deduped.append(row)
    return deduped


def existing_crop_path(row: dict[str, str]) -> tuple[str, bool]:
    crop_path = row.get("crop_path", "").strip()
    if crop_path and resolve(Path(crop_path)).exists():
        return crop_path, False
    source_crop = row.get("source_crop", "").strip()
    if source_crop and resolve(Path(source_crop)).exists():
        return source_crop, True
    return crop_path or source_crop, False


def keep_existing_crop_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], int, int]:
    kept: list[dict[str, str]] = []
    missing = 0
    recovered = 0
    for row in rows:
        crop_path, used_source_crop = existing_crop_path(row)
        if not crop_path:
            missing += 1
            continue
        if used_source_crop:
            row["crop_path"] = crop_path
            recovered += 1
        if not resolve(Path(crop_path)).exists():
            missing += 1
            continue
        kept.append(row)
    return kept, missing, recovered


def write_contact_sheet(rows: list[dict[str, str]], out_path: Path, thumb: int, limit: int = 96) -> None:
    selected = rows[:limit]
    if not selected:
        return
    cols = 6
    label_h = 58
    sheet = Image.new("RGB", (cols * thumb, ((len(selected) + cols - 1) // cols) * (thumb + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        image_path = resolve(Path(row["crop_path"]))
        if not image_path.exists():
            continue
        with Image.open(image_path).convert("RGB") as image:
            image = ImageOps.contain(image, (thumb, thumb))
            tile = Image.new("RGB", (thumb, thumb), (244, 244, 244))
            tile.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
        x = (index % cols) * thumb
        y = (index // cols) * (thumb + label_h)
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + thumb + 4), row["crop_id"], fill="black")
        draw.text((x + 4, y + thumb + 22), f"{row['class_name']} {row['side']}".strip()[:28], fill="black")
        draw.text((x + 4, y + thumb + 40), row["priority_reason"][:28], fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out)
    if args.clean and out_dir.exists():
        allowed_root = (ROOT / "data" / "review").resolve()
        resolved = out_dir.resolve()
        if allowed_root not in resolved.parents and resolved != allowed_root:
            raise SystemExit(f"Refusing to clean outside {allowed_root}: {resolved}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = [resolve(path) for path in args.manifest]
    missing = [path for path in manifests if not path.exists()]
    if missing:
        raise SystemExit(f"Missing manifest: {repo_path(missing[0])}")

    rows = select_rows(manifests, args.max_per_class_side, args.max_failure_rows)
    rows, missing_crops, recovered_crops = keep_existing_crop_rows(rows)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    write_contact_sheet(rows, out_dir / "contact_sheet.jpg", args.thumb)
    print(f"wrote {len(rows)} focused rows to {repo_path(out_dir / 'manifest.csv')}")
    if missing_crops:
        print(f"skipped_missing_crops={missing_crops}")
    if recovered_crops:
        print(f"recovered_source_crops={recovered_crops}")


if __name__ == "__main__":
    main()
