#!/usr/bin/env python
"""Render and check a small deterministic WebGL variant batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
    parser.add_argument("--scene-mode", choices=["auto", "clean", "stack", "fan", "qa3"], default="auto")
    parser.add_argument("--background-dir", type=Path, help="Optional reviewed-clean background image directory.")
    parser.add_argument("--recipe-name", default="", help="Human-readable recipe name to write into recipe.json.")
    parser.add_argument(
        "--artifact-status",
        choices=["smoke", "diagnostic", "trainable-candidate"],
        default="smoke",
        help="Declare whether the batch is a smoke, diagnostic, or trainable-candidate artifact.",
    )
    parser.add_argument("--intended-use", default="", help="Short intended-use note to write into recipe.json.")
    parser.add_argument("--notes", default="", help="Optional short recipe notes to write into recipe.json.")
    parser.add_argument("--headroom-max-percent", default="90", help="CPU/GPU percent cap passed to run_with_headroom.py.")
    parser.add_argument("--headroom-resume-percent", default="82", help="Resume threshold passed to run_with_headroom.py.")
    parser.add_argument("--headroom-max-ram-percent", default="90", help="RAM percent cap passed to run_with_headroom.py.")
    parser.add_argument("--headroom-max-gpu-mem-percent", default="90", help="GPU memory percent cap passed to run_with_headroom.py.")
    parser.add_argument("--min-free-ram-gb", default="3", help="Free-RAM preflight floor passed to run_with_headroom.py.")
    parser.add_argument("--preflight-timeout", default="120", help="Initial headroom wait timeout in seconds.")
    parser.add_argument("--skip-render", action="store_true", help="Only recheck/contact-sheet existing outputs.")
    parser.add_argument("--skip-yolo-check", action="store_true", help="Do not run check_yolo_dataset.py on the packaged dataset.")
    parser.add_argument("--skip-label-view-check", action="store_true", help="Do not run check_webgl_label_views.py on packaged label views.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def render_variant(variant: int, out_dir: Path, scene_mode: str, background_dir: Path | None, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/run_with_headroom.py",
        "--max-percent",
        args.headroom_max_percent,
        "--resume-percent",
        args.headroom_resume_percent,
        "--max-ram-percent",
        args.headroom_max_ram_percent,
        "--max-gpu-mem-percent",
        args.headroom_max_gpu_mem_percent,
        "--min-free-ram-gb",
        args.min_free_ram_gb,
        "--preflight-timeout",
        args.preflight_timeout,
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


def check_variant(out_dir: Path, allow_no_occluder: bool = False, allow_no_overlap: bool = False) -> None:
    cmd = [sys.executable, "scripts/check_webgl_smoke_output.py", "--out-dir", str(out_dir)]
    if allow_no_occluder:
        cmd.append("--allow-no-occluder")
    if allow_no_overlap:
        cmd.append("--allow-no-overlap")
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


def draw_label_previews(
    image_path: Path,
    visible_boxes: list[dict[str, object]],
    fragment_metadata: list[dict[str, object]],
    detect_out: Path,
    fragment_out: Path,
) -> None:
    with Image.open(image_path).convert("RGB") as base:
        detect = base.copy()
        fragment = base.copy()
    detect_draw = ImageDraw.Draw(detect)
    fragment_draw = ImageDraw.Draw(fragment)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for box in visible_boxes:
        x1 = int(box["minX"])
        y1 = int(box["minY"])
        x2 = int(box["maxX"])
        y2 = int(box["maxY"])
        label = str(box.get("className", ""))
        detect_draw.rectangle((x1, y1, x2, y2), outline=(255, 230, 0), width=4)
        detect_draw.text((x1 + 4, y1 + 4), label, fill=(255, 230, 0), font=font)

    for fragment_row in fragment_metadata:
        x, y, width, height = [int(value) for value in fragment_row["bbox_xywh_px"]]
        label = f"{fragment_row.get('className', '')}#{fragment_row.get('componentIndex', '')}"
        fragment_draw.rectangle((x, y, x + width, y + height), outline=(0, 255, 255), width=4)
        fragment_draw.text((x + 4, y + 4), label, fill=(0, 255, 255), font=font)

    detect_out.parent.mkdir(parents=True, exist_ok=True)
    fragment_out.parent.mkdir(parents=True, exist_ok=True)
    detect.save(detect_out, quality=92)
    fragment.save(fragment_out, quality=92)


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


def build_fragment_labels(id_path: Path, boxes_path: Path) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
    id_image = np.array(Image.open(id_path).convert("RGB"))
    height, width = id_image.shape[:2]
    boxes_doc = json.loads(boxes_path.read_text(encoding="utf-8"))
    rows: list[str] = []
    metadata_rows: list[dict[str, object]] = []
    ignored_rows: list[dict[str, object]] = []

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
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            component_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            component_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            component_metadata = {
                "classIndex": int(box["classIndex"]),
                "className": box.get("className"),
                "parentVisibleIndex": parent_index,
                "parentColor": box["color"],
                "componentId": component_id,
                "visible_pixels": pixels,
                "bbox_xywh_px": [x, y, component_width, component_height],
                "component_fraction_of_parent": round(float(pixels / max(1, int(mask.sum()))), 4),
            }
            if pixels < FRAGMENT_MIN_PIXELS:
                ignored_rows.append(
                    {
                        **component_metadata,
                        "ignore_reason": "below_min_fragment_pixels",
                        "min_fragment_pixels": FRAGMENT_MIN_PIXELS,
                    }
                )
                continue
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
                    **component_metadata,
                    "componentIndex": kept_component_index,
                    "ignore_reason": "",
                }
            )
            kept_component_index += 1

    return rows, metadata_rows, ignored_rows


def write_lines(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return round(float(ordered[index]), 4)


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": round(float(min(values)), 4),
        "p50": quantile(values, 0.5),
        "p90": quantile(values, 0.9),
        "max": round(float(max(values)), 4),
        "mean": round(float(sum(values) / len(values)), 4),
    }


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
    fragment_ignored_metadata_dir = out_root / "fragments" / "ignored_metadata" / "train"
    qa_dir = out_root / "qa"
    preview_dir = qa_dir / "previews"
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
        fragment_ignored_metadata_dir,
        qa_dir,
        preview_dir,
    ):
        prepare_empty_dir(directory, out_root)

    manifest = []
    obb_image_status_counts: Counter[str] = Counter()
    obb_instance_status_counts: Counter[str] = Counter()
    fragment_counts: Counter[str] = Counter()
    ignored_fragment_counts: Counter[str] = Counter()
    ignored_fragment_reason_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    layer_audit_totals: Counter[str] = Counter()
    scene_mode_counts: Counter[str] = Counter()
    surface_counts: Counter[str] = Counter()
    background_counts: Counter[str] = Counter()
    visible_pixels_per_instance: list[float] = []
    visible_instances_per_image: list[float] = []
    fragments_per_image: list[float] = []
    fragments_per_parent_values: list[float] = []
    obb_reject_reason_counts: Counter[str] = Counter()
    image_summary_rows: list[dict[str, object]] = []
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
        fragment_ignored_metadata_path = fragment_ignored_metadata_dir / f"{stem}.json"
        detect_preview_path = preview_dir / f"{stem}_detect.jpg"
        fragment_preview_path = preview_dir / f"{stem}_fragments.jpg"
        shutil.copyfile(out_dir / "visual.png", image_path)
        shutil.copyfile(out_dir / "labels_visible.txt", label_path)
        shutil.copyfile(out_dir / "id.png", id_path)
        shutil.copyfile(out_dir / "visible_boxes.json", boxes_path)
        shutil.copyfile(out_dir / "layer_audit.json", audit_path)
        shutil.copyfile(out_dir / "metadata.json", source_metadata_path)
        boxes_doc = json.loads(boxes_path.read_text(encoding="utf-8"))
        visible_boxes = boxes_doc.get("boxes", [])
        layer_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        scene_mode_counts[str(source_metadata.get("sceneMode", "unknown"))] += 1
        scene_config = source_metadata.get("sceneConfig", {})
        if isinstance(scene_config, dict):
            surface = scene_config.get("surface", {})
            if isinstance(surface, dict):
                surface_counts[str(surface.get("name", "unknown"))] += 1
                background = surface.get("background")
                background_counts["file" if background else "procedural"] += 1
        fragment_rows, fragment_metadata, ignored_fragment_metadata = build_fragment_labels(id_path, boxes_path)
        shutil.copyfile(out_dir / "visual.png", fragment_image_path)
        write_lines(fragment_label_path, fragment_rows)
        write_json(fragment_metadata_path, fragment_metadata)
        write_json(fragment_ignored_metadata_path, ignored_fragment_metadata)
        draw_label_previews(image_path, visible_boxes, fragment_metadata, detect_preview_path, fragment_preview_path)
        fragment_counts["images"] += 1
        fragment_counts["fragments"] += len(fragment_rows)
        ignored_fragment_counts["ignored_fragments"] += len(ignored_fragment_metadata)
        ignored_fragment_reason_counts.update(str(row["ignore_reason"]) for row in ignored_fragment_metadata)
        fragments_per_image.append(float(len(fragment_rows)))
        parent_fragment_counts: Counter[tuple[int, str]] = Counter(
            (int(fragment["parentVisibleIndex"]), str(fragment["className"]))
            for fragment in fragment_metadata
        )
        fragments_per_parent_values.extend(float(count) for count in parent_fragment_counts.values())
        obb_rows, obb_metadata = build_obb_label(id_path, boxes_path)
        obb_reject_reasons = sorted({str(row["status"]) for row in obb_metadata if row["status"] != "exported"})
        obb_reject_reason_counts.update(obb_reject_reasons)
        obb_instance_status_counts.update(str(row["status"]) for row in obb_metadata)
        class_counts.update(str(box.get("className", "unknown")) for box in visible_boxes)
        visible_instances_per_image.append(float(len(visible_boxes)))
        visible_pixels = [float(box.get("pixels", 0)) for box in visible_boxes]
        visible_pixels_per_instance.extend(visible_pixels)
        for key in ("visiblePixels", "overlapPixels", "occluderPixels", "violations"):
            layer_audit_totals[key] += int(layer_audit.get(key, 0))
        manifest_row = {
            "variant": variant,
            "image": str(image_path.relative_to(out_root)),
            "label": str(label_path.relative_to(out_root)),
            "id": str(id_path.relative_to(out_root)),
            "visible_boxes": str(boxes_path.relative_to(out_root)),
            "layer_audit": str(audit_path.relative_to(out_root)),
            "source_metadata": str(source_metadata_path.relative_to(out_root)),
            "fragment_image": str(fragment_image_path.relative_to(out_root)),
            "fragment_label": str(fragment_label_path.relative_to(out_root)),
            "fragment_metadata": str(fragment_metadata_path.relative_to(out_root)),
            "fragment_ignored_metadata": str(fragment_ignored_metadata_path.relative_to(out_root)),
            "detect_preview": str(detect_preview_path.relative_to(out_root)),
            "fragment_preview": str(fragment_preview_path.relative_to(out_root)),
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
        image_summary_rows.append(
            {
                "variant": variant,
                "image": str(image_path.relative_to(out_root)),
                "scene_mode": str(source_metadata.get("sceneMode", "unknown")),
                "visible_instances": len(visible_boxes),
                "visible_pixels": int(sum(visible_pixels)),
                "fragments": len(fragment_rows),
                "ignored_fragments": len(ignored_fragment_metadata),
                "split_parents": sum(1 for count in parent_fragment_counts.values() if count > 1),
                "obb_status": manifest_row["obb_status"],
                "obb_reject_reasons": obb_reject_reasons,
                "layer_audit": {
                    "visiblePixels": int(layer_audit.get("visiblePixels", 0)),
                    "overlapPixels": int(layer_audit.get("overlapPixels", 0)),
                    "occluderPixels": int(layer_audit.get("occluderPixels", 0)),
                    "violations": int(layer_audit.get("violations", 0)),
                },
                "sha256": {
                    "visual": sha256_file(image_path),
                    "id": sha256_file(id_path),
                    "detect_label": sha256_file(label_path),
                    "fragment_label": sha256_file(fragment_label_path),
                    "detect_preview": sha256_file(detect_preview_path),
                    "fragment_preview": sha256_file(fragment_preview_path),
                },
                "previews": {
                    "detect": str(detect_preview_path.relative_to(out_root)),
                    "fragments": str(fragment_preview_path.relative_to(out_root)),
                },
            }
        )

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
            "ignored_fragments": ignored_fragment_counts["ignored_fragments"],
            "ignored_reason_counts": dict(sorted(ignored_fragment_reason_counts.items())),
            "min_fragment_pixels": FRAGMENT_MIN_PIXELS,
            "policy": {
                "label_meaning": "visible connected evidence components, not physical bill counts",
                "ignored_fragments": "components below min_fragment_pixels are recorded as ignored metadata instead of forced training labels",
                "counting": "use parent metadata or downstream fusion to merge components from one physical bill",
            },
        },
    )
    write_json(
        qa_dir / "summary.json",
        {
            "images": len(manifest),
            "variants": {
                "start": min((variant for variant, _out_dir in variant_dirs), default=0),
                "end": max((variant for variant, _out_dir in variant_dirs), default=0),
            },
            "scene_modes": dict(sorted(scene_mode_counts.items())),
            "surfaces": dict(sorted(surface_counts.items())),
            "backgrounds": dict(sorted(background_counts.items())),
            "class_counts": dict(sorted(class_counts.items())),
            "visible_instances": {
                "total": int(sum(visible_instances_per_image)),
                "per_image": summarize_values(visible_instances_per_image),
                "visible_pixels_per_instance": summarize_values(visible_pixels_per_instance),
            },
            "fragments": {
                "total": fragment_counts["fragments"],
                "ignored_total": ignored_fragment_counts["ignored_fragments"],
                "ignored_reason_counts": dict(sorted(ignored_fragment_reason_counts.items())),
                "per_image": summarize_values(fragments_per_image),
                "per_parent": summarize_values(fragments_per_parent_values),
                "split_parent_count": sum(1 for value in fragments_per_parent_values if value > 1),
                "min_fragment_pixels": FRAGMENT_MIN_PIXELS,
            },
            "obb": {
                "image_status_counts": dict(sorted(obb_image_status_counts.items())),
                "instance_status_counts": dict(sorted(obb_instance_status_counts.items())),
                "reject_reason_counts": dict(sorted(obb_reject_reason_counts.items())),
                "min_largest_component_frac": OBB_MIN_LARGEST_COMPONENT_FRAC,
                "min_rect_fill_frac": OBB_MIN_RECT_FILL_FRAC,
            },
            "layer_audit_totals": dict(sorted(layer_audit_totals.items())),
            "policy": {
                "detect_labels": "visible-only AABB labels for detect-compatible probes",
                "fragments": "visible connected evidence components, not direct count labels",
                "obb": "trainable only when every visible instance in the image has an honest OBB",
                "counting": "physical parent counts live in visible_boxes/source metadata and require fusion for fragment outputs",
            },
            "images_detail": image_summary_rows,
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


def rel(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_recipe_metadata(
    args: argparse.Namespace,
    out_root: Path,
    contact_sheet: Path,
    data_yaml: Path,
    data_obb_yaml: Path,
    data_fragments_yaml: Path,
) -> None:
    recipe_name = args.recipe_name or f"webgl_{args.scene_mode}_variants_{args.start_variant}_{args.start_variant + args.count - 1}"
    payload = {
        "recipe_name": recipe_name,
        "artifact_status": args.artifact_status,
        "intended_use": args.intended_use,
        "notes": args.notes,
        "renderer": "renderers/webgl/src/render-smoke.mjs",
        "packager": "scripts/render_webgl_variant_batch.py",
        "output_root": rel(out_root),
        "variant_seed_range": {
            "start": args.start_variant,
            "count": args.count,
            "end": args.start_variant + args.count - 1,
        },
        "scene_mode": args.scene_mode,
        "background_dir": rel(args.background_dir) if args.background_dir else "",
        "headroom": {
            "max_percent": args.headroom_max_percent,
            "resume_percent": args.headroom_resume_percent,
            "max_ram_percent": args.headroom_max_ram_percent,
            "max_gpu_mem_percent": args.headroom_max_gpu_mem_percent,
            "min_free_ram_gb": args.min_free_ram_gb,
            "preflight_timeout": args.preflight_timeout,
        },
        "checks": {
            "render_smoke_check": True,
            "detect_yolo_check": not args.skip_yolo_check,
            "label_view_check": not args.skip_label_view_check,
        },
        "outputs": {
            "detect_data_yaml": rel(data_yaml),
            "obb_data_yaml": rel(data_obb_yaml),
            "fragment_data_yaml": rel(data_fragments_yaml),
            "manifest": rel(out_root / "manifest.json"),
            "qa_summary": rel(out_root / "qa" / "summary.json"),
                "contact_sheet": rel(contact_sheet),
                "preview_dir": rel(out_root / "qa" / "previews"),
            },
        "policy": {
            "smoke": "pipeline functionality proof; do not train final claims from this artifact",
            "diagnostic": "use for visual/model diagnosis unless separately promoted",
            "trainable-candidate": "may enter a bounded training comparison only after QA summary and real guardrails pass",
        },
        "command": [Path(sys.executable).name, *sys.argv],
    }
    write_json(out_root / "recipe.json", payload)


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
            render_variant(variant, out_dir, args.scene_mode, args.background_dir, args)
        check_variant(
            out_dir,
            allow_no_occluder=args.scene_mode in {"clean", "qa3"},
            allow_no_overlap=args.scene_mode == "clean",
        )
        variant_dirs.append((variant, out_dir))

    contact_sheet = out_root / "contact_sheet.png"
    write_contact_sheet(variant_dirs, contact_sheet)
    data_yaml, data_obb_yaml, data_fragments_yaml = write_yolo_dataset(variant_dirs, out_root)
    write_recipe_metadata(args, out_root, contact_sheet, data_yaml, data_obb_yaml, data_fragments_yaml)
    if not args.skip_yolo_check:
        run([sys.executable, "scripts/check_yolo_dataset.py", "--data", str(data_yaml)])
    if not args.skip_label_view_check:
        run([sys.executable, "scripts/check_webgl_label_views.py", "--root", str(out_root)])
    print(f"wrote {contact_sheet.relative_to(ROOT)}")
    print(f"wrote {data_yaml.relative_to(ROOT)}")
    print(f"wrote {data_obb_yaml.relative_to(ROOT)}")
    print(f"wrote {data_fragments_yaml.relative_to(ROOT)}")
    print(f"wrote {(out_root / 'recipe.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
