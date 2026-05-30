#!/usr/bin/env python
"""Render and check a small deterministic WebGL variant batch."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "USD_1",
    "USD_5",
    "USD_10",
    "USD_20",
    "USD_50",
    "USD_100",
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_20000",
    "KHR_50000",
]
OBB_MIN_LARGEST_COMPONENT_FRAC = 0.85
OBB_MIN_RECT_FILL_FRAC = 0.35
FRAGMENT_MIN_PIXELS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("data/synthetic/cashsnap_webgl_variant_batch_smoke"))
    parser.add_argument("--start-variant", type=int, default=0)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--scene-mode", choices=["auto", "stack", "fan", "qa3"], default="auto")
    parser.add_argument("--background-dir", type=Path, help="Optional reviewed-clean background image directory.")
    parser.add_argument("--skip-render", action="store_true", help="Only recheck/contact-sheet existing outputs.")
    parser.add_argument("--skip-yolo-check", action="store_true", help="Do not run check_yolo_dataset.py on the packaged dataset.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def render_variant(variant: int, out_dir: Path, scene_mode: str, background_dir: Path | None) -> None:
    cmd = [
        sys.executable,
        "scripts/run_with_headroom.py",
        "--max-percent",
        "90",
        "--resume-percent",
        "82",
        "--max-ram-percent",
        "90",
        "--max-gpu-mem-percent",
        "90",
        "--min-free-ram-gb",
        "3",
        "--preflight-timeout",
        "120",
        "--",
        "node",
        "renderers/webgl/src/render-smoke.mjs",
        "--variant",
        str(variant),
        "--scene-mode",
        scene_mode,
        "--out-dir",
        str(out_dir),
    ]
    if background_dir is not None:
        cmd.extend(["--background-dir", str(background_dir)])
    run(cmd)


def check_variant(out_dir: Path, allow_no_occluder: bool = False) -> None:
    cmd = [sys.executable, "scripts/check_webgl_smoke_output.py", "--out-dir", str(out_dir)]
    if allow_no_occluder:
        cmd.append("--allow-no-occluder")
    run(cmd)


def write_contact_sheet(variant_dirs: list[tuple[int, Path]], out_path: Path) -> None:
    cell_w, cell_h = 320, 240
    header_h = 30
    row_h = cell_h + header_h
    sheet = Image.new("RGB", (cell_w * 2, row_h * len(variant_dirs)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)

    for row, (variant, out_dir) in enumerate(variant_dirs):
        y = row * row_h
        visual = Image.open(out_dir / "visual.png").convert("RGB").resize((cell_w, cell_h))
        mask = Image.open(out_dir / "id.png").convert("RGB").resize((cell_w, cell_h))
        sheet.paste(visual, (0, y + header_h))
        sheet.paste(mask, (cell_w, y + header_h))
        draw.text((8, y + 8), f"variant {variant} visual", fill=(255, 255, 255))
        draw.text((cell_w + 8, y + 8), f"variant {variant} id", fill=(255, 255, 255))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def obb_audit_for_mask(mask: np.ndarray) -> dict[str, float | int | str]:
    component_count, _component_ids, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA] if component_count > 1 else np.array([], dtype=np.int32)
    total_pixels = int(mask.sum())
    largest_component_pixels = int(component_areas.max()) if len(component_areas) else 0
    largest_component_frac = largest_component_pixels / total_pixels if total_pixels else 0.0

    ys, xs = np.where(mask)
    points = np.column_stack([xs, ys]).astype(np.float32)
    rect = cv2.minAreaRect(points)
    rect_width, rect_height = rect[1]
    rect_area = float(max(rect_width * rect_height, 1.0))
    rect_fill_frac = float(total_pixels / rect_area)

    status = "exported"
    if len(xs) < 4:
        status = "too_few_pixels"
    elif component_count > 2 and largest_component_frac < OBB_MIN_LARGEST_COMPONENT_FRAC:
        status = "fragmented_visible_mask"
    elif rect_fill_frac < OBB_MIN_RECT_FILL_FRAC:
        status = "loose_min_area_rect"

    return {
        "status": status,
        "visible_pixels": total_pixels,
        "component_count": int(max(0, component_count - 1)),
        "largest_component_pixels": largest_component_pixels,
        "largest_component_frac": round(float(largest_component_frac), 4),
        "rect_fill_frac": round(float(rect_fill_frac), 4),
        "rect_width_px": round(float(rect_width), 2),
        "rect_height_px": round(float(rect_height), 2),
    }


def build_obb_label(id_path: Path, boxes_path: Path) -> tuple[list[str], list[dict[str, object]]]:
    id_image = np.array(Image.open(id_path).convert("RGB"))
    height, width = id_image.shape[:2]
    boxes_doc = json.loads(boxes_path.read_text(encoding="utf-8"))
    rows: list[str] = []
    metadata_rows: list[dict[str, object]] = []

    for box in boxes_doc.get("boxes", []):
        color = np.array(box["color"], dtype=np.uint8)
        mask = np.all(id_image == color, axis=2)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        audit = obb_audit_for_mask(mask)
        metadata_row: dict[str, object] = {
            "classIndex": int(box["classIndex"]),
            "className": box.get("className"),
            "color": box["color"],
            **audit,
        }
        metadata_rows.append(metadata_row)
        if audit["status"] != "exported":
            continue
        points = np.column_stack([xs, ys]).astype(np.float32)
        rect = cv2.minAreaRect(points)
        corners = cv2.boxPoints(rect)
        normalized = []
        for x, y in corners:
            normalized.extend([x / width, y / height])
        rows.append(
            f"{int(box['classIndex'])} "
            + " ".join(f"{max(0.0, min(1.0, value)):.6f}" for value in normalized)
        )

    return rows, metadata_rows


def build_fragment_labels(id_path: Path, boxes_path: Path) -> tuple[list[str], list[dict[str, object]]]:
    id_image = np.array(Image.open(id_path).convert("RGB"))
    height, width = id_image.shape[:2]
    boxes_doc = json.loads(boxes_path.read_text(encoding="utf-8"))
    rows: list[str] = []
    metadata_rows: list[dict[str, object]] = []

    for parent_index, box in enumerate(boxes_doc.get("boxes", [])):
        color = np.array(box["color"], dtype=np.uint8)
        mask = np.all(id_image == color, axis=2)
        component_count, _component_ids, stats, _centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        kept_component_index = 0
        for component_id in range(1, component_count):
            pixels = int(stats[component_id, cv2.CC_STAT_AREA])
            if pixels < FRAGMENT_MIN_PIXELS:
                continue
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            component_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            cx = (x + component_width / 2) / width
            cy = (y + component_height / 2) / height
            normalized_width = component_width / width
            normalized_height = component_height / height
            rows.append(
                f"{int(box['classIndex'])} "
                f"{cx:.6f} {cy:.6f} {normalized_width:.6f} {normalized_height:.6f}"
            )
            metadata_rows.append(
                {
                    "classIndex": int(box["classIndex"]),
                    "className": box.get("className"),
                    "parentVisibleIndex": parent_index,
                    "parentColor": box["color"],
                    "componentIndex": kept_component_index,
                    "componentId": component_id,
                    "visible_pixels": pixels,
                    "bbox_xywh_px": [x, y, component_width, component_height],
                    "component_fraction_of_parent": round(float(pixels / max(1, int(mask.sum()))), 4),
                }
            )
            kept_component_index += 1

    return rows, metadata_rows


def write_lines(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prepare_empty_dir(directory: Path, out_root: Path) -> None:
    resolved_directory = directory.resolve()
    resolved_root = out_root.resolve()
    if resolved_directory != resolved_root and resolved_root not in resolved_directory.parents:
        raise RuntimeError(f"refusing to clear directory outside output root: {directory}")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def write_yolo_dataset(variant_dirs: list[tuple[int, Path]], out_root: Path) -> tuple[Path, Path, Path]:
    images_dir = out_root / "images" / "train"
    labels_dir = out_root / "labels" / "train"
    ids_dir = out_root / "ids" / "train"
    metadata_dir = out_root / "metadata"
    obb_images_dir = out_root / "obb" / "images" / "train"
    obb_labels_dir = out_root / "obb" / "labels" / "train"
    obb_metadata_dir = out_root / "obb" / "metadata" / "train"
    obb_rejected_labels_dir = out_root / "obb" / "rejected_labels" / "train"
    obb_rejected_metadata_dir = out_root / "obb" / "rejected_metadata" / "train"
    fragment_images_dir = out_root / "fragments" / "images" / "train"
    fragment_labels_dir = out_root / "fragments" / "labels" / "train"
    fragment_metadata_dir = out_root / "fragments" / "metadata" / "train"
    for directory in (
        images_dir,
        labels_dir,
        ids_dir,
        metadata_dir,
        obb_images_dir,
        obb_labels_dir,
        obb_metadata_dir,
        obb_rejected_labels_dir,
        obb_rejected_metadata_dir,
        fragment_images_dir,
        fragment_labels_dir,
        fragment_metadata_dir,
    ):
        prepare_empty_dir(directory, out_root)

    manifest = []
    obb_image_status_counts: Counter[str] = Counter()
    obb_instance_status_counts: Counter[str] = Counter()
    fragment_counts: Counter[str] = Counter()
    for variant, out_dir in variant_dirs:
        stem = f"variant_{variant:04d}"
        image_path = images_dir / f"{stem}.png"
        label_path = labels_dir / f"{stem}.txt"
        id_path = ids_dir / f"{stem}.png"
        boxes_path = metadata_dir / f"{stem}_visible_boxes.json"
        audit_path = metadata_dir / f"{stem}_layer_audit.json"
        source_metadata_path = metadata_dir / f"{stem}_metadata.json"
        obb_image_path = obb_images_dir / f"{stem}.png"
        obb_label_path = obb_labels_dir / f"{stem}.txt"
        obb_metadata_path = obb_metadata_dir / f"{stem}.json"
        obb_rejected_label_path = obb_rejected_labels_dir / f"{stem}.txt"
        obb_rejected_metadata_path = obb_rejected_metadata_dir / f"{stem}.json"
        fragment_image_path = fragment_images_dir / f"{stem}.png"
        fragment_label_path = fragment_labels_dir / f"{stem}.txt"
        fragment_metadata_path = fragment_metadata_dir / f"{stem}.json"
        shutil.copyfile(out_dir / "visual.png", image_path)
        shutil.copyfile(out_dir / "labels_visible.txt", label_path)
        shutil.copyfile(out_dir / "id.png", id_path)
        shutil.copyfile(out_dir / "visible_boxes.json", boxes_path)
        shutil.copyfile(out_dir / "layer_audit.json", audit_path)
        shutil.copyfile(out_dir / "metadata.json", source_metadata_path)
        fragment_rows, fragment_metadata = build_fragment_labels(id_path, boxes_path)
        shutil.copyfile(out_dir / "visual.png", fragment_image_path)
        write_lines(fragment_label_path, fragment_rows)
        write_json(fragment_metadata_path, fragment_metadata)
        fragment_counts["images"] += 1
        fragment_counts["fragments"] += len(fragment_rows)
        obb_rows, obb_metadata = build_obb_label(id_path, boxes_path)
        obb_reject_reasons = sorted({str(row["status"]) for row in obb_metadata if row["status"] != "exported"})
        obb_instance_status_counts.update(str(row["status"]) for row in obb_metadata)
        manifest_row = {
            "variant": variant,
            "image": str(image_path.relative_to(out_root)),
            "label": str(label_path.relative_to(out_root)),
            "id": str(id_path.relative_to(out_root)),
            "visible_boxes": str(boxes_path.relative_to(out_root)),
            "layer_audit": str(audit_path.relative_to(out_root)),
            "fragment_image": str(fragment_image_path.relative_to(out_root)),
            "fragment_label": str(fragment_label_path.relative_to(out_root)),
            "fragment_metadata": str(fragment_metadata_path.relative_to(out_root)),
            "obb_status": "accepted" if not obb_reject_reasons else "rejected",
            "obb_reject_reasons": obb_reject_reasons,
        }
        if obb_reject_reasons:
            obb_image_status_counts["rejected"] += 1
            write_lines(obb_rejected_label_path, obb_rows)
            write_json(obb_rejected_metadata_path, obb_metadata)
            manifest_row.update(
                {
                    "obb_diagnostic_label": str(obb_rejected_label_path.relative_to(out_root)),
                    "obb_diagnostic_metadata": str(obb_rejected_metadata_path.relative_to(out_root)),
                }
            )
        else:
            obb_image_status_counts["accepted"] += 1
            shutil.copyfile(out_dir / "visual.png", obb_image_path)
            write_lines(obb_label_path, obb_rows)
            write_json(obb_metadata_path, obb_metadata)
            manifest_row.update(
                {
                    "obb_image": str(obb_image_path.relative_to(out_root)),
                    "obb_label": str(obb_label_path.relative_to(out_root)),
                    "obb_metadata": str(obb_metadata_path.relative_to(out_root)),
                }
            )
        manifest.append(manifest_row)

    write_json(out_root / "manifest.json", manifest)
    write_json(
        out_root / "obb" / "summary.json",
        {
            "trainable_obb_images": obb_image_status_counts["accepted"],
            "rejected_obb_images": obb_image_status_counts["rejected"],
            "compact_instance_obbs": obb_instance_status_counts["exported"],
            "instance_status_counts": dict(sorted(obb_instance_status_counts.items())),
            "policy": {
                "trainable_obb_dataset": "all visible instances in an image must have exported OBB labels",
                "rejected_labels": "diagnostic only; do not train from rejected_labels because skipped visible bills would become background",
                "min_largest_component_frac": OBB_MIN_LARGEST_COMPONENT_FRAC,
                "min_rect_fill_frac": OBB_MIN_RECT_FILL_FRAC,
            },
        },
    )
    write_json(
        out_root / "fragments" / "summary.json",
        {
            "images": fragment_counts["images"],
            "fragments": fragment_counts["fragments"],
            "min_fragment_pixels": FRAGMENT_MIN_PIXELS,
            "policy": {
                "label_meaning": "visible connected evidence components, not physical bill counts",
                "counting": "use parent metadata or downstream fusion to merge components from one physical bill",
            },
        },
    )
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/train",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    data_obb_yaml = out_root / "data_obb.yaml"
    data_obb_yaml.write_text(
        "\n".join(
            [
                f"path: {(out_root / 'obb').as_posix()}",
                "train: images/train",
                "val: images/train",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    data_fragments_yaml = out_root / "data_fragments.yaml"
    data_fragments_yaml.write_text(
        "\n".join(
            [
                f"path: {(out_root / 'fragments').as_posix()}",
                "train: images/train",
                "val: images/train",
                "names:",
                *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml, data_obb_yaml, data_fragments_yaml


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")

    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    variant_dirs: list[tuple[int, Path]] = []

    for variant in range(args.start_variant, args.start_variant + args.count):
        out_dir = out_root / f"variant_{variant:04d}"
        if not args.skip_render:
            render_variant(variant, out_dir, args.scene_mode, args.background_dir)
        check_variant(out_dir, allow_no_occluder=args.scene_mode == "qa3")
        variant_dirs.append((variant, out_dir))

    contact_sheet = out_root / "contact_sheet.png"
    write_contact_sheet(variant_dirs, contact_sheet)
    data_yaml, data_obb_yaml, data_fragments_yaml = write_yolo_dataset(variant_dirs, out_root)
    if not args.skip_yolo_check:
        run([sys.executable, "scripts/check_yolo_dataset.py", "--data", str(data_yaml)])
    print(f"wrote {contact_sheet.relative_to(ROOT)}")
    print(f"wrote {data_yaml.relative_to(ROOT)}")
    print(f"wrote {data_obb_yaml.relative_to(ROOT)}")
    print(f"wrote {data_fragments_yaml.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
