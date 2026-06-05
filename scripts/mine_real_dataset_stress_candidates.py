#!/usr/bin/env python
"""Mine a real YOLO dataset for partial-note stress review candidates.

This does not promote candidates into benchmark/capture inventory. It creates a
review queue from existing labeled real data so stress slices can be audited
instead of rediscovered by scrolling through thousands of images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "cashsnap_v1" / "data.yaml"
DEFAULT_OUT_CSV = ROOT / "runs" / "cashsnap" / "real_dataset_stress_candidates_latest.csv"
DEFAULT_REVIEW_CSV = ROOT / "runs" / "cashsnap" / "real_dataset_stress_candidate_review_latest.csv"
DEFAULT_JSON_OUT = ROOT / "runs" / "cashsnap" / "real_dataset_stress_candidates_latest.json"
DEFAULT_CONTACT_DIR = ROOT / "runs" / "cashsnap" / "real_dataset_stress_candidate_sheets_latest"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SCENE_TYPES = [
    "single_khr",
    "simple_overlap",
    "hand_fan",
    "same_denomination_fan",
    "partial_off_frame",
    "thin_slice_khr_5000",
    "khr_5000_face_number_overlap",
    "thin_slice_khr_20000",
    "weak_khr_20000",
    "weak_khr_50000",
    "mixed_usd_khr",
    "mixed_usd_khr_rare_common",
    "blank_label_or_unlabeled",
]

COMMON_KHR_CLASSES = {"KHR_500", "KHR_1000", "KHR_2000", "KHR_5000", "KHR_10000"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="YOLO data.yaml to mine.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Dataset splits to scan.")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--contact-dir", type=Path, default=DEFAULT_CONTACT_DIR)
    parser.add_argument("--max-per-scene", type=int, default=32, help="Max unique-origin examples per contact sheet.")
    parser.add_argument("--max-images", type=int, default=0, help="Optional scan cap for smoke/debug runs.")
    parser.add_argument("--skip-contact-sheets", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return resolve(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_path(resolved)}: expected YAML object")
    return data


def dataset_root(data_yaml: dict[str, Any], data_path: Path) -> Path:
    raw = data_yaml.get("path")
    if raw:
        path = Path(str(raw))
        return path if path.is_absolute() else resolve(data_path).parent / path
    return resolve(data_path).parent


def class_names(data_yaml: dict[str, Any]) -> list[str]:
    raw = data_yaml.get("names")
    if isinstance(raw, dict):
        return [str(raw[index]) for index in sorted(raw, key=lambda value: int(value))]
    if isinstance(raw, list):
        return [str(value) for value in raw]
    raise SystemExit("data.yaml names must be a list or mapping")


def image_paths_from_value(root: Path, raw_value: Any) -> list[Path]:
    if isinstance(raw_value, list):
        rows: list[Path] = []
        for item in raw_value:
            rows.extend(image_paths_from_value(root, item))
        return rows
    if raw_value is None:
        return []
    value = Path(str(raw_value))
    path = value if value.is_absolute() else root / value
    if path.is_dir():
        return sorted(
            image
            for image in path.rglob("*")
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
        )
    if path.is_file() and path.suffix.lower() == ".txt":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            candidate = Path(line)
            rows.append(candidate if candidate.is_absolute() else ROOT / candidate)
        return rows
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    return []


def label_path_for_image(root: Path, image_path: Path) -> Path:
    try:
        rel = image_path.relative_to(root)
        parts = list(rel.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            return root / Path(*parts).with_suffix(".txt")
    except ValueError:
        pass
    return image_path.with_suffix(".txt").parent.parent / "labels" / image_path.with_suffix(".txt").name


def read_labels(label_path: Path, names: list[str]) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_path(label_path)}:{line_no}: expected YOLO detect label with 5 columns")
        class_id = int(float(parts[0]))
        x, y, w, h = [float(value) for value in parts[1:]]
        if class_id < 0 or class_id >= len(names):
            raise SystemExit(f"{repo_path(label_path)}:{line_no}: class id {class_id} outside names")
        x1 = max(0.0, x - w / 2)
        y1 = max(0.0, y - h / 2)
        x2 = min(1.0, x + w / 2)
        y2 = min(1.0, y + h / 2)
        rows.append(
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                "edge_touch": x1 <= 0.025 or y1 <= 0.025 or x2 >= 0.975 or y2 >= 0.975,
                "thin_or_small": min(w, h) <= 0.12 or (w * h) <= 0.045,
            }
        )
    return rows


def box_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    ix1 = max(float(a["x1"]), float(b["x1"]))
    iy1 = max(float(a["y1"]), float(b["y1"]))
    ix2 = min(float(a["x2"]), float(b["x2"]))
    iy2 = min(float(a["y2"]), float(b["y2"]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = float(a["area"])
    area_b = float(b["area"])
    return inter / max(1e-9, area_a + area_b - inter)


def overlap_pairs(labels: list[dict[str, Any]]) -> int:
    count = 0
    for index, first in enumerate(labels):
        for second in labels[index + 1 :]:
            if box_iou(first, second) >= 0.02:
                count += 1
    return count


def origin_key(path: Path) -> str:
    stem = path.stem
    if ".rf." in stem:
        return stem.split(".rf.", 1)[0].lower()
    return stem.lower()


def metrics_for_image(labels: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = Counter(str(row["class_name"]) for row in labels)
    classes = sorted(class_counts)
    box_count = len(labels)
    edge_count = sum(1 for row in labels if row["edge_touch"])
    thin_count = sum(1 for row in labels if row["thin_or_small"])
    overlap_count = overlap_pairs(labels)
    areas = [float(row["area"]) for row in labels]
    same_class_max = max(class_counts.values(), default=0)
    khr_count = sum(1 for name in class_counts for _ in range(class_counts[name]) if name.startswith("KHR_"))
    usd_count = sum(1 for name in class_counts for _ in range(class_counts[name]) if name.startswith("USD_"))
    return {
        "box_count": box_count,
        "classes": classes,
        "denominations": classes,
        "class_counts": dict(sorted(class_counts.items())),
        "same_class_max": same_class_max,
        "edge_touch_count": edge_count,
        "thin_or_small_count": thin_count,
        "overlap_pair_count": overlap_count,
        "min_area": min(areas) if areas else 0.0,
        "max_area": max(areas) if areas else 0.0,
        "mean_area": sum(areas) / len(areas) if areas else 0.0,
        "khr_count": khr_count,
        "usd_count": usd_count,
    }


def infer_scene_scores(metrics: dict[str, Any]) -> dict[str, float]:
    box_count = int(metrics["box_count"])
    same_class_max = int(metrics["same_class_max"])
    edge_count = int(metrics["edge_touch_count"])
    thin_count = int(metrics["thin_or_small_count"])
    overlaps = int(metrics["overlap_pair_count"])
    classes = set(metrics["classes"])
    khr_count = int(metrics["khr_count"])
    usd_count = int(metrics["usd_count"])
    scores: dict[str, float] = {}

    if box_count == 0:
        scores["blank_label_or_unlabeled"] = 1.0
        return scores
    if box_count == 1 and khr_count == 1:
        scores["single_khr"] = 2.0 + float(metrics["max_area"])
    if 2 <= box_count <= 5 and (overlaps > 0 or edge_count > 0):
        scores["simple_overlap"] = 2.0 + overlaps * 2.0 + edge_count * 0.5 + box_count * 0.2
    if box_count >= 6:
        scores["hand_fan"] = 2.0 + box_count * 0.35 + thin_count * 0.4 + edge_count * 0.25 + same_class_max * 0.2
    if box_count >= 4 and same_class_max >= 3:
        scores["same_denomination_fan"] = 2.0 + same_class_max * 1.2 + box_count * 0.2
    if edge_count > 0:
        scores["partial_off_frame"] = 1.0 + edge_count * 0.8 + thin_count * 0.2
    if "KHR_5000" in classes and (thin_count > 0 or edge_count > 0):
        scores["thin_slice_khr_5000"] = 1.5 + thin_count * 0.7 + edge_count * 0.5
    if "KHR_5000" in classes and box_count >= 2 and (overlaps > 0 or edge_count > 0 or thin_count > 0):
        scores["khr_5000_face_number_overlap"] = 1.5 + overlaps * 1.0 + thin_count * 0.4 + edge_count * 0.3
    if "KHR_20000" in classes and (thin_count > 0 or edge_count > 0):
        scores["thin_slice_khr_20000"] = 1.5 + thin_count * 0.7 + edge_count * 0.5
    if "KHR_20000" in classes:
        scores["weak_khr_20000"] = 1.0 + box_count * 0.1 + edge_count * 0.2
    if "KHR_50000" in classes:
        scores["weak_khr_50000"] = 1.0 + box_count * 0.1 + edge_count * 0.2
    if khr_count > 0 and usd_count > 0:
        scores["mixed_usd_khr"] = 1.0 + min(khr_count, usd_count) * 0.5 + box_count * 0.1
    if usd_count > 0 and "KHR_50000" in classes and classes.intersection(COMMON_KHR_CLASSES):
        common_count = sum(1 for class_name in COMMON_KHR_CLASSES if class_name in classes)
        scores["mixed_usd_khr_rare_common"] = (
            2.0
            + min(usd_count, khr_count) * 0.5
            + common_count * 0.6
            + overlaps * 0.4
            + box_count * 0.1
        )
    return scores


def draw_overlay(image_path: Path, labels: list[dict[str, Any]], names: list[str], size: tuple[int, int]) -> Image.Image:
    del names
    with Image.open(image_path) as image:
        base = image.convert("RGB")
    base.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (28, 28, 28))
    offset_x = (size[0] - base.width) // 2
    offset_y = (size[1] - base.height) // 2
    canvas.paste(base, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    for row in labels:
        class_id = int(row["class_id"])
        color = (
            80 + (class_id * 37) % 176,
            80 + (class_id * 73) % 176,
            80 + (class_id * 109) % 176,
        )
        x1 = offset_x + float(row["x1"]) * base.width
        y1 = offset_y + float(row["y1"]) * base.height
        x2 = offset_x + float(row["x2"]) * base.width
        y2 = offset_y + float(row["y2"]) * base.height
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        label = str(row["class_name"])
        text_box = draw.textbbox((x1 + 2, y1 + 2), label)
        draw.rectangle(text_box, fill=color)
        draw.text((x1 + 2, y1 + 2), label, fill=(0, 0, 0))
    return canvas


def write_contact_sheet(
    scene_type: str,
    rows: list[dict[str, Any]],
    labels_by_image: dict[str, list[dict[str, Any]]],
    names: list[str],
    out_dir: Path,
    max_rows: int,
) -> Path | None:
    if not rows:
        return None
    selected_by_origin: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (-float(item["score"]), str(item["image"]))):
        key = str(row["origin_key"])
        if key not in selected_by_origin:
            selected_by_origin[key] = row
        if len(selected_by_origin) >= max_rows:
            break
    selected = list(selected_by_origin.values())
    cell_w, image_h, caption_h = 360, 270, 70
    cols = 4
    row_h = image_h + caption_h
    sheet_rows = math.ceil(len(selected) / cols)
    sheet = Image.new("RGB", (cols * cell_w, sheet_rows * row_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for index, row in enumerate(selected):
        x = (index % cols) * cell_w
        y = (index // cols) * row_h
        image_path = resolve(Path(str(row["image"])))
        thumb = draw_overlay(image_path, labels_by_image[str(row["image"])], names, (cell_w, image_h))
        sheet.paste(thumb, (x, y))
        caption = (
            f"{row['split']} {Path(str(row['image'])).stem[:30]}\n"
            f"boxes={row['box_count']} same={row['same_class_max']} "
            f"edge={row['edge_touch_count']} thin={row['thin_or_small_count']} "
            f"overlap={row['overlap_pair_count']}\n"
            f"score={float(row['score']):.2f}"
        )
        draw.multiline_text((x + 6, y + image_h + 5), caption, fill=(235, 235, 235), font=font, spacing=2)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scene_type}.jpg"
    sheet.save(path, quality=92)
    return path


def iter_split_images(data_yaml: dict[str, Any], root: Path, splits: Iterable[str]) -> Iterable[tuple[str, Path]]:
    for split in splits:
        for image_path in image_paths_from_value(root, data_yaml.get(split)):
            yield split, image_path


def split_fingerprints(image_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        by_split[str(row["split"])].append(row)
    fingerprints: dict[str, dict[str, Any]] = {}
    for split, rows in sorted(by_split.items()):
        image_digest = hashlib.sha256()
        label_digest = hashlib.sha256()
        missing_labels = 0
        for row in sorted(rows, key=lambda item: str(item["image"])):
            image_path = str(row["image"])
            label_path = str(row["label"])
            image_digest.update(image_path.encode("utf-8"))
            image_digest.update(b"\n")
            label_digest.update(label_path.encode("utf-8"))
            label_digest.update(b"\0")
            resolved_label = resolve(Path(label_path))
            if resolved_label.exists() and resolved_label.is_file():
                label_digest.update(file_sha256(resolved_label).encode("utf-8"))
            else:
                label_digest.update(b"MISSING")
                missing_labels += 1
            label_digest.update(b"\n")
        fingerprints[split] = {
            "image_count": len(rows),
            "image_listing_sha256": image_digest.hexdigest(),
            "label_content_sha256": label_digest.hexdigest(),
            "missing_label_count": missing_labels,
        }
    return fingerprints


def main() -> int:
    args = parse_args()
    data_path = resolve(args.data)
    data_yaml = read_yaml(data_path)
    root = dataset_root(data_yaml, data_path)
    names = class_names(data_yaml)

    image_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    labels_by_image: dict[str, list[dict[str, Any]]] = {}
    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned = 0

    for split, image_path in iter_split_images(data_yaml, root, args.splits):
        if args.max_images and scanned >= args.max_images:
            break
        scanned += 1
        label_path = label_path_for_image(root, image_path)
        labels = read_labels(label_path, names)
        metrics = metrics_for_image(labels)
        scores = infer_scene_scores(metrics)
        candidate_scene_types = sorted(scores, key=lambda key: (-scores[key], key))
        row = {
            "split": split,
            "image": repo_path(image_path),
            "label": repo_path(label_path),
            "origin_key": origin_key(image_path),
            "candidate_scene_types": ";".join(candidate_scene_types),
            "candidate_count": len(candidate_scene_types),
            "score_max": round(max(scores.values(), default=0.0), 4),
            "denominations": ";".join(metrics["denominations"]),
            "class_counts_json": json.dumps(metrics["class_counts"], sort_keys=True),
            **{
                key: round(value, 6) if isinstance(value, float) else value
                for key, value in metrics.items()
                if key not in {"classes", "denominations", "class_counts"}
            },
        }
        image_rows.append(row)
        labels_by_image[row["image"]] = labels
        for scene_type, score in scores.items():
            review_row = {
                **row,
                "scene_type": scene_type,
                "score": round(score, 4),
                "promotion_status": "candidate_needs_visual_review",
            }
            review_rows.append(review_row)
            scene_rows[scene_type].append(review_row)

    for path in (resolve(args.out_csv), resolve(args.review_csv), resolve(args.json_out)):
        path.parent.mkdir(parents=True, exist_ok=True)

    image_fieldnames = [
        "split",
        "image",
        "label",
        "origin_key",
        "candidate_scene_types",
        "candidate_count",
        "score_max",
        "denominations",
        "class_counts_json",
        "box_count",
        "same_class_max",
        "edge_touch_count",
        "thin_or_small_count",
        "overlap_pair_count",
        "min_area",
        "max_area",
        "mean_area",
        "khr_count",
        "usd_count",
    ]
    with resolve(args.out_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=image_fieldnames)
        writer.writeheader()
        writer.writerows(image_rows)

    review_fieldnames = [
        "scene_type",
        "promotion_status",
        "score",
        *image_fieldnames,
    ]
    with resolve(args.review_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=review_fieldnames)
        writer.writeheader()
        writer.writerows(sorted(review_rows, key=lambda row: (row["scene_type"], -float(row["score"]), row["image"])))

    contact_sheets: dict[str, str] = {}
    if not args.skip_contact_sheets:
        contact_dir = resolve(args.contact_dir)
        for scene_type in SCENE_TYPES:
            sheet = write_contact_sheet(
                scene_type,
                scene_rows.get(scene_type, []),
                labels_by_image,
                names,
                contact_dir,
                args.max_per_scene,
            )
            if sheet:
                contact_sheets[scene_type] = repo_path(sheet)

    summary = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "data": repo_path(data_path),
        "data_config_sha256": file_sha256(data_path),
        "dataset_root": repo_path(root),
        "splits": args.splits,
        "split_fingerprints": split_fingerprints(image_rows),
        "images_scanned": scanned,
        "candidate_images": sum(1 for row in image_rows if int(row["candidate_count"]) > 0),
        "scene_candidate_counts": {
            scene_type: len(scene_rows.get(scene_type, []))
            for scene_type in SCENE_TYPES
            if scene_rows.get(scene_type)
        },
        "scene_unique_origin_counts": {
            scene_type: len({str(row["origin_key"]) for row in scene_rows.get(scene_type, [])})
            for scene_type in SCENE_TYPES
            if scene_rows.get(scene_type)
        },
        "outputs": {
            "image_csv": repo_path(resolve(args.out_csv)),
            "review_csv": repo_path(resolve(args.review_csv)),
            "contact_sheets": contact_sheets,
        },
        "policy": {
            "promotion_status": "candidate_needs_visual_review",
            "not_counted_as_capture_inventory": True,
            "reason": "geometry/name heuristics can queue likely stress scenes, but visible denomination evidence and rights/use status still need review before promotion",
        },
    }
    resolve(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "ok: mined real dataset stress candidates "
        f"(images={scanned}, candidate_images={summary['candidate_images']}, "
        f"scene_types={len(summary['scene_candidate_counts'])})"
    )
    for scene_type, count in summary["scene_candidate_counts"].items():
        unique = summary["scene_unique_origin_counts"][scene_type]
        print(f"- {scene_type}: candidates={count} unique_origins={unique}")
    print(f"wrote_csv={repo_path(resolve(args.out_csv))}")
    print(f"wrote_review_csv={repo_path(resolve(args.review_csv))}")
    print(f"wrote_json={repo_path(resolve(args.json_out))}")
    if contact_sheets:
        print(f"wrote_contact_dir={repo_path(resolve(args.contact_dir))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
