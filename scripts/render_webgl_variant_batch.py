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
FRAGMENT_REVIEW_MIN_PIXELS = 1000
FRAGMENT_REVIEW_MIN_PARENT_FRACTION = 0.01
VISUAL_MIN_MEAN_LUMA = 20.0
VISUAL_MAX_MEAN_LUMA = 235.0
VISUAL_MIN_LUMA_STD = 1.0
VISUAL_MAX_DARK_FRACTION = 0.98
VISUAL_MAX_LIGHT_FRACTION = 0.98


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=Path("data/synthetic/cashsnap_webgl_variant_batch_smoke"))
    parser.add_argument("--start-variant", type=int, default=0)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--scene-mode", choices=["auto", "clean", "negative", "stack", "fan", "thin_edge", "hand_occlusion", "qa3"], default="auto")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--visual-scale", default="2", help="Visual WebGL supersampling scale passed to render-smoke.mjs.")
    parser.add_argument("--background-dir", type=Path, help="Optional reviewed-clean background image directory.")
    parser.add_argument(
        "--background-bank-config",
        type=Path,
        default=Path("configs/synthetic_recipes/cashsnap_webgl_background_banks_v1.json"),
        help="Review registry used to gate --background-dir usage.",
    )
    parser.add_argument(
        "--asset-side-policy",
        choices=["any", "front_only", "back_only", "front_back_mix"],
        default="any",
        help="Constrain banknote scan side sampling for front/back confusion recipes.",
    )
    parser.add_argument(
        "--camera-profile",
        choices=[
            "generic_phone_jitter",
            "phone_auto",
            "iphone_8_like",
            "iphone_12_wide_like",
            "budget_android_wide_like",
            "browser_upload_resized",
        ],
        default="generic_phone_jitter",
        help="Select WebGL camera/FOV/framing profile.",
    )
    parser.add_argument("--recipe-name", default="", help="Human-readable recipe name to write into recipe.json.")
    parser.add_argument(
        "--artifact-status",
        choices=["smoke", "diagnostic", "trainable-candidate"],
        default="smoke",
        help="Declare whether the batch is a smoke, diagnostic, or trainable-candidate artifact.",
    )
    parser.add_argument("--intended-use", default="", help="Short intended-use note to write into recipe.json.")
    parser.add_argument("--notes", default="", help="Optional short recipe notes to write into recipe.json.")
    parser.add_argument(
        "--fragment-review-policy",
        choices=["diagnostic", "ignore"],
        default="diagnostic",
        help="diagnostic keeps review-required fragment labels; ignore moves them to ignored metadata for trainable fragment views.",
    )
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


def check_background_bank(background_dir: Path | None, artifact_status: str, config: Path) -> None:
    if background_dir is None:
        return
    cmd = [
        sys.executable,
        "scripts/check_webgl_background_banks.py",
        "--config",
        str(config),
        "--require-path",
        str(background_dir),
        "--artifact-status",
        artifact_status,
    ]
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


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
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--visual-scale",
        str(args.visual_scale),
        "--asset-side-policy",
        args.asset_side_policy,
        "--camera-profile",
        args.camera_profile,
        "--out-dir",
        str(out_dir),
    ]
    if background_dir is not None:
        cmd.extend(["--background-dir", str(background_dir)])
    run(cmd)


def check_variant(
    out_dir: Path,
    allow_no_occluder: bool = False,
    allow_no_overlap: bool = False,
    allow_no_boxes: bool = False,
) -> None:
    cmd = [sys.executable, "scripts/check_webgl_smoke_output.py", "--out-dir", str(out_dir)]
    if allow_no_occluder:
        cmd.append("--allow-no-occluder")
    if allow_no_overlap:
        cmd.append("--allow-no-overlap")
    if allow_no_boxes:
        cmd.append("--allow-no-boxes")
    run(cmd)


def write_contact_sheet(variant_dirs: list[tuple[int, Path]], out_path: Path) -> list[dict[str, object]]:
    cell_w, cell_h = 320, 240
    header_h = 30
    row_h = cell_h + header_h
    sheet = Image.new("RGB", (cell_w * 2, row_h * len(variant_dirs)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    index_rows: list[dict[str, object]] = []

    for row, (variant, out_dir) in enumerate(variant_dirs):
        y = row * row_h
        visual = Image.open(out_dir / "visual.png").convert("RGB").resize((cell_w, cell_h))
        mask = Image.open(out_dir / "id.png").convert("RGB").resize((cell_w, cell_h))
        sheet.paste(visual, (0, y + header_h))
        sheet.paste(mask, (cell_w, y + header_h))
        draw.text((8, y + 8), f"variant {variant} visual", fill=(255, 255, 255))
        draw.text((cell_w + 8, y + 8), f"variant {variant} id", fill=(255, 255, 255))
        index_rows.append(
            {
                "variant": variant,
                "row": row,
                "visual_cell_xywh": [0, y + header_h, cell_w, cell_h],
                "id_cell_xywh": [cell_w, y + header_h, cell_w, cell_h],
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return index_rows


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


def write_id_overlay(visual_path: Path, id_path: Path, out_path: Path) -> None:
    visual = Image.open(visual_path).convert("RGB")
    id_image = Image.open(id_path).convert("RGB")
    overlay = Image.new("RGB", visual.size, (0, 0, 0))
    visual_pixels = visual.load()
    id_pixels = id_image.load()
    overlay_pixels = overlay.load()
    for y in range(visual.height):
        for x in range(visual.width):
            mask_color = id_pixels[x, y]
            if mask_color == (0, 0, 0):
                overlay_pixels[x, y] = visual_pixels[x, y]
            else:
                vr, vg, vb = visual_pixels[x, y]
                mr, mg, mb = mask_color
                overlay_pixels[x, y] = (
                    int(vr * 0.58 + mr * 0.42),
                    int(vg * 0.58 + mg * 0.42),
                    int(vb * 0.58 + mb * 0.42),
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path, quality=92)


def visual_quality_for_image(image_path: Path) -> dict[str, object]:
    with Image.open(image_path).convert("RGB") as image:
        pixels = np.asarray(image, dtype=np.float32)
    height, width = pixels.shape[:2]
    luma = pixels[:, :, 0] * 0.2126 + pixels[:, :, 1] * 0.7152 + pixels[:, :, 2] * 0.0722
    mean_luma = float(luma.mean())
    luma_std = float(luma.std())
    dark_fraction = float((luma < 8).mean())
    light_fraction = float((luma > 247).mean())
    failures: list[str] = []
    if mean_luma < VISUAL_MIN_MEAN_LUMA:
        failures.append("mean_luma_too_low")
    if mean_luma > VISUAL_MAX_MEAN_LUMA:
        failures.append("mean_luma_too_high")
    if luma_std < VISUAL_MIN_LUMA_STD:
        failures.append("luma_std_too_low")
    if dark_fraction > VISUAL_MAX_DARK_FRACTION:
        failures.append("dark_fraction_too_high")
    if light_fraction > VISUAL_MAX_LIGHT_FRACTION:
        failures.append("light_fraction_too_high")
    return {
        "status": "accepted" if not failures else "rejected",
        "failures": failures,
        "width": int(width),
        "height": int(height),
        "mean_luma": round(mean_luma, 4),
        "luma_std": round(luma_std, 4),
        "dark_fraction": round(dark_fraction, 6),
        "light_fraction": round(light_fraction, 6),
    }


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


def build_fragment_labels(
    id_path: Path,
    boxes_path: Path,
    fragment_review_policy: str,
) -> tuple[list[str], list[dict[str, object]], list[dict[str, object]]]:
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
                        "evidence_status": "ignored",
                        "evidence_warnings": ["below_min_fragment_pixels"],
                        "ignore_reason": "below_min_fragment_pixels",
                        "min_fragment_pixels": FRAGMENT_MIN_PIXELS,
                    }
                )
                continue
            evidence_warnings = []
            if pixels < FRAGMENT_REVIEW_MIN_PIXELS:
                evidence_warnings.append("below_review_fragment_pixels")
            if component_metadata["component_fraction_of_parent"] < FRAGMENT_REVIEW_MIN_PARENT_FRACTION:
                evidence_warnings.append("low_parent_visible_fraction")
            if evidence_warnings and fragment_review_policy == "ignore":
                ignored_rows.append(
                    {
                        **component_metadata,
                        "evidence_status": "ignored",
                        "evidence_warnings": evidence_warnings,
                        "ignore_reason": "requires_human_review",
                        "min_fragment_pixels": FRAGMENT_MIN_PIXELS,
                        "review_min_fragment_pixels": FRAGMENT_REVIEW_MIN_PIXELS,
                        "review_min_parent_fraction": FRAGMENT_REVIEW_MIN_PARENT_FRACTION,
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
                    "evidence_status": "review_required" if evidence_warnings else "trainable",
                    "evidence_warnings": evidence_warnings,
                    "review_min_fragment_pixels": FRAGMENT_REVIEW_MIN_PIXELS,
                    "review_min_parent_fraction": FRAGMENT_REVIEW_MIN_PARENT_FRACTION,
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


def write_yolo_dataset(
    variant_dirs: list[tuple[int, Path]],
    out_root: Path,
    fallback_scene_mode: str,
    fragment_review_policy: str,
) -> tuple[Path, Path, Path]:
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
    fragment_evidence_status_counts: Counter[str] = Counter()
    fragment_evidence_warning_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    layer_audit_totals: Counter[str] = Counter()
    scene_mode_counts: Counter[str] = Counter()
    surface_counts: Counter[str] = Counter()
    background_counts: Counter[str] = Counter()
    asset_side_policy_counts: Counter[str] = Counter()
    asset_side_counts: Counter[str] = Counter()
    front_back_mix_counts: Counter[str] = Counter()
    camera_profile_request_counts: Counter[str] = Counter()
    camera_profile_counts: Counter[str] = Counter()
    visible_pixels_per_instance: list[float] = []
    visible_instances_per_image: list[float] = []
    fragments_per_image: list[float] = []
    fragments_per_parent_values: list[float] = []
    obb_reject_reason_counts: Counter[str] = Counter()
    image_summary_rows: list[dict[str, object]] = []
    quarantine_rows: list[dict[str, object]] = []
    visual_quality_rows: list[dict[str, object]] = []
    visual_quality_status_counts: Counter[str] = Counter()
    visual_quality_failure_counts: Counter[str] = Counter()
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
        id_overlay_path = preview_dir / f"{stem}_id_overlay.jpg"
        shutil.copyfile(out_dir / "visual.png", image_path)
        visual_quality = visual_quality_for_image(image_path)
        visual_quality_status_counts[str(visual_quality["status"])] += 1
        visual_quality_failure_counts.update(str(reason) for reason in visual_quality["failures"])
        visual_quality_row = {
            "variant": variant,
            "image": str(image_path.relative_to(out_root)),
            **visual_quality,
        }
        visual_quality_rows.append(visual_quality_row)
        if visual_quality["status"] != "accepted":
            quarantine_rows.append(
                {
                    "variant": variant,
                    "image": str(image_path.relative_to(out_root)),
                    "view": "visual",
                    "action": "visual_quality_rejected",
                    "reasons": visual_quality["failures"],
                }
            )
        shutil.copyfile(out_dir / "labels_visible.txt", label_path)
        shutil.copyfile(out_dir / "id.png", id_path)
        shutil.copyfile(out_dir / "visible_boxes.json", boxes_path)
        shutil.copyfile(out_dir / "layer_audit.json", audit_path)
        shutil.copyfile(out_dir / "metadata.json", source_metadata_path)
        boxes_doc = json.loads(boxes_path.read_text(encoding="utf-8"))
        visible_boxes = boxes_doc.get("boxes", [])
        layer_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        source_scene_mode = str(source_metadata.get("sceneMode") or "")
        if not source_scene_mode or source_scene_mode == "unknown":
            source_scene_mode = fallback_scene_mode if fallback_scene_mode != "auto" else "unknown"
            source_metadata["sceneMode"] = source_scene_mode
            write_json(source_metadata_path, source_metadata)
        scene_mode_counts[source_scene_mode] += 1
        scene_config = source_metadata.get("sceneConfig", {})
        if isinstance(scene_config, dict):
            surface = scene_config.get("surface", {})
            if isinstance(surface, dict):
                surface_counts[str(surface.get("name", "unknown"))] += 1
                background = surface.get("background")
                background_counts["file" if background else "procedural"] += 1
            camera = scene_config.get("camera", {})
            if isinstance(camera, dict):
                camera_profile_request_counts[str(camera.get("profileRequested", "unknown"))] += 1
                camera_profile_counts[str(camera.get("profile", "unknown"))] += 1
        asset_selection = source_metadata.get("assetSelection", {})
        if isinstance(asset_selection, dict):
            side_policy = str(asset_selection.get("sidePolicy", "unknown"))
            asset_side_policy_counts[side_policy] += 1
            side_counts = asset_selection.get("sideCounts", {})
            if isinstance(side_counts, dict):
                for side, count in side_counts.items():
                    asset_side_counts[str(side)] += int(count)
            if side_policy == "front_back_mix":
                front_back_mix_counts[
                    "satisfied" if bool(asset_selection.get("frontBackMixSatisfied")) else "unsatisfied"
                ] += 1
        fragment_rows, fragment_metadata, ignored_fragment_metadata = build_fragment_labels(
            id_path,
            boxes_path,
            fragment_review_policy,
        )
        shutil.copyfile(out_dir / "visual.png", fragment_image_path)
        write_lines(fragment_label_path, fragment_rows)
        write_json(fragment_metadata_path, fragment_metadata)
        write_json(fragment_ignored_metadata_path, ignored_fragment_metadata)
        draw_label_previews(image_path, visible_boxes, fragment_metadata, detect_preview_path, fragment_preview_path)
        write_id_overlay(image_path, id_path, id_overlay_path)
        fragment_counts["images"] += 1
        fragment_counts["fragments"] += len(fragment_rows)
        ignored_fragment_counts["ignored_fragments"] += len(ignored_fragment_metadata)
        ignored_fragment_reason_counts.update(str(row["ignore_reason"]) for row in ignored_fragment_metadata)
        fragment_evidence_status_counts.update(str(row["evidence_status"]) for row in fragment_metadata)
        for row in fragment_metadata:
            fragment_evidence_warning_counts.update(str(reason) for reason in row.get("evidence_warnings", []))
        review_required_fragments = [
            row for row in fragment_metadata if row.get("evidence_status") == "review_required"
        ]
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
            "id_overlay": str(id_overlay_path.relative_to(out_root)),
            "obb_status": "accepted" if not obb_reject_reasons else "rejected",
            "obb_reject_reasons": obb_reject_reasons,
            }
        if obb_reject_reasons:
            obb_image_status_counts["rejected"] += 1
            write_lines(obb_rejected_label_path, obb_rows)
            write_json(obb_rejected_metadata_path, obb_metadata)
            quarantine_rows.append(
                {
                    "variant": variant,
                    "image": str(image_path.relative_to(out_root)),
                    "view": "obb",
                    "action": "excluded_from_trainable_obb",
                    "reasons": obb_reject_reasons,
                }
            )
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
        if ignored_fragment_metadata:
            ignored_reasons = sorted({str(row["ignore_reason"]) for row in ignored_fragment_metadata})
            quarantine_rows.append(
                {
                    "variant": variant,
                    "image": str(image_path.relative_to(out_root)),
                    "view": "fragment",
                    "action": (
                        "ignored_ambiguous_fragment_components"
                        if ignored_reasons == ["requires_human_review"]
                        else "ignored_fragment_components"
                    ),
                    "reasons": ignored_reasons,
                    "ignored_fragments": len(ignored_fragment_metadata),
                }
            )
        if review_required_fragments:
            quarantine_rows.append(
                {
                    "variant": variant,
                    "image": str(image_path.relative_to(out_root)),
                    "view": "fragment",
                    "action": "fragment_evidence_review_required",
                    "reasons": sorted(
                        {
                            str(reason)
                            for row in review_required_fragments
                            for reason in row.get("evidence_warnings", [])
                        }
                    ),
                    "review_required_fragments": len(review_required_fragments),
                }
            )
        image_summary_rows.append(
            {
                "variant": variant,
                "image": str(image_path.relative_to(out_root)),
                "scene_mode": source_scene_mode,
                "asset_selection": asset_selection if isinstance(asset_selection, dict) else {},
                "camera": scene_config.get("camera", {}) if isinstance(scene_config, dict) else {},
                "visible_instances": len(visible_boxes),
                "visible_pixels": int(sum(visible_pixels)),
                "fragments": len(fragment_rows),
                "ignored_fragments": len(ignored_fragment_metadata),
                "review_required_fragments": len(review_required_fragments),
                "split_parents": sum(1 for count in parent_fragment_counts.values() if count > 1),
                "obb_status": manifest_row["obb_status"],
                "obb_reject_reasons": obb_reject_reasons,
                "visual_quality": visual_quality,
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
                    "id_overlay": sha256_file(id_overlay_path),
                },
                "previews": {
                    "detect": str(detect_preview_path.relative_to(out_root)),
                    "fragments": str(fragment_preview_path.relative_to(out_root)),
                    "id_overlay": str(id_overlay_path.relative_to(out_root)),
                },
            }
        )

    write_json(out_root / "manifest.json", manifest)
    write_json(
        qa_dir / "quarantine.json",
        {
            "rows": quarantine_rows,
            "counts": dict(sorted(Counter(str(row["action"]) for row in quarantine_rows).items())),
            "policy": {
                "visual_quality_rejected": "visual image failed deterministic brightness/blankness gates and must not be promoted without review",
                "excluded_from_trainable_obb": "image remains valid for detect/fragment views, but is excluded from the trainable OBB split",
                "ignored_below_threshold_components": "tiny connected components are recorded for review and not forced into fragment training labels",
                "ignored_fragment_components": "below-threshold or ambiguous connected components are recorded as ignored metadata instead of forced fragment labels",
                "ignored_ambiguous_fragment_components": "ambiguous review-required components are excluded from fragment labels and kept as ignored metadata",
                "fragment_evidence_review_required": "fragment is labeled for diagnostics but should not enter trainable-candidate mixes until accepted by policy or review",
            },
        },
    )
    write_json(
        qa_dir / "visual_quality.json",
        {
            "rows": visual_quality_rows,
            "counts": dict(sorted(visual_quality_status_counts.items())),
            "failure_counts": dict(sorted(visual_quality_failure_counts.items())),
            "thresholds": {
                "min_mean_luma": VISUAL_MIN_MEAN_LUMA,
                "max_mean_luma": VISUAL_MAX_MEAN_LUMA,
                "min_luma_std": VISUAL_MIN_LUMA_STD,
                "max_dark_fraction": VISUAL_MAX_DARK_FRACTION,
                "max_light_fraction": VISUAL_MAX_LIGHT_FRACTION,
            },
            "policy": "Deterministic blankness/exposure gate; accepted status is necessary but not sufficient for human visual realism review.",
        },
    )
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
            "evidence_status_counts": dict(sorted(fragment_evidence_status_counts.items())),
            "evidence_warning_counts": dict(sorted(fragment_evidence_warning_counts.items())),
            "min_fragment_pixels": FRAGMENT_MIN_PIXELS,
            "review_min_fragment_pixels": FRAGMENT_REVIEW_MIN_PIXELS,
            "review_min_parent_fraction": FRAGMENT_REVIEW_MIN_PARENT_FRACTION,
            "fragment_review_policy": fragment_review_policy,
            "label_transform_policy": {
                "geometric_postprocess": "disallowed_until_rgb_id_and_labels_share_the_same_exact_transform",
                "current_postprocess": "non_geometric_rgb_only",
            },
            "policy": {
                "label_meaning": "visible connected evidence components, not physical bill counts",
                "ignored_fragments": "components below min_fragment_pixels are recorded as ignored metadata instead of forced training labels",
                "review_required": "small/low-fraction components keep labels for diagnostics but require review before trainable-candidate promotion",
                "fragment_review_policy": "diagnostic keeps review-required labels; ignore excludes review-required labels and records them as ignored metadata",
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
            "asset_selection": {
                "side_policy_counts": dict(sorted(asset_side_policy_counts.items())),
                "side_counts": dict(sorted(asset_side_counts.items())),
                "front_back_mix_counts": dict(sorted(front_back_mix_counts.items())),
            },
            "camera_profiles": {
                "requested_counts": dict(sorted(camera_profile_request_counts.items())),
                "selected_counts": dict(sorted(camera_profile_counts.items())),
            },
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
                "evidence_status_counts": dict(sorted(fragment_evidence_status_counts.items())),
                "evidence_warning_counts": dict(sorted(fragment_evidence_warning_counts.items())),
                "per_image": summarize_values(fragments_per_image),
                "per_parent": summarize_values(fragments_per_parent_values),
                "split_parent_count": sum(1 for value in fragments_per_parent_values if value > 1),
                "min_fragment_pixels": FRAGMENT_MIN_PIXELS,
                "review_min_fragment_pixels": FRAGMENT_REVIEW_MIN_PIXELS,
                "review_min_parent_fraction": FRAGMENT_REVIEW_MIN_PARENT_FRACTION,
                "fragment_review_policy": fragment_review_policy,
            },
            "obb": {
                "image_status_counts": dict(sorted(obb_image_status_counts.items())),
                "instance_status_counts": dict(sorted(obb_instance_status_counts.items())),
                "reject_reason_counts": dict(sorted(obb_reject_reason_counts.items())),
                "min_largest_component_frac": OBB_MIN_LARGEST_COMPONENT_FRAC,
                "min_rect_fill_frac": OBB_MIN_RECT_FILL_FRAC,
            },
            "layer_audit_totals": dict(sorted(layer_audit_totals.items())),
            "visual_quality": {
                "status_counts": dict(sorted(visual_quality_status_counts.items())),
                "failure_counts": dict(sorted(visual_quality_failure_counts.items())),
                "thresholds": {
                    "min_mean_luma": VISUAL_MIN_MEAN_LUMA,
                    "max_mean_luma": VISUAL_MAX_MEAN_LUMA,
                    "min_luma_std": VISUAL_MIN_LUMA_STD,
                    "max_dark_fraction": VISUAL_MAX_DARK_FRACTION,
                    "max_light_fraction": VISUAL_MAX_LIGHT_FRACTION,
                },
            },
            "label_transform_policy": {
                "geometric_postprocess": "disallowed_until_rgb_id_and_labels_share_the_same_exact_transform",
                "current_postprocess": "non_geometric_rgb_only",
            },
            "policy": {
                "detect_labels": "visible-only AABB labels for detect-compatible probes",
                "fragments": "visible connected evidence components, not direct count labels",
                "obb": "trainable only when every visible instance in the image has an honest OBB",
                "counting": "physical parent counts live in visible_boxes/source metadata and require fusion for fragment outputs",
                "asset_side_policy": "front/back policies constrain scan-side sampling before render; front_back_mix should show both sides in multi-note images",
                "camera_profiles": "profile requests choose auditable phone-like FOV/framing ranges before RGB/ID extraction; no geometric postprocess is applied after labels",
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
        "render": {
            "width": args.width,
            "height": args.height,
            "visual_scale": args.visual_scale,
        },
        "asset_side_policy": args.asset_side_policy,
        "camera_profile": args.camera_profile,
        "background_dir": rel(args.background_dir) if args.background_dir else "",
        "fragment_review_policy": args.fragment_review_policy,
        "label_transform_policy": {
            "geometric_postprocess": "disallowed_until_rgb_id_and_labels_share_the_same_exact_transform",
            "current_postprocess": "non_geometric_rgb_only",
        },
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
            "contact_index": rel(out_root / "qa" / "contact_index.json"),
            "preview_dir": rel(out_root / "qa" / "previews"),
            "quarantine": rel(out_root / "qa" / "quarantine.json"),
            "visual_quality": rel(out_root / "qa" / "visual_quality.json"),
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
    check_background_bank(args.background_dir, args.artifact_status, args.background_bank_config)
    variant_dirs: list[tuple[int, Path]] = []

    for variant in range(args.start_variant, args.start_variant + args.count):
        out_dir = out_root / f"variant_{variant:04d}"
        if not args.skip_render:
            render_variant(variant, out_dir, args.scene_mode, args.background_dir, args)
        check_variant(
            out_dir,
            allow_no_occluder=args.scene_mode in {"clean", "negative", "qa3"},
            allow_no_overlap=args.scene_mode in {"clean", "negative"},
            allow_no_boxes=args.scene_mode == "negative",
        )
        variant_dirs.append((variant, out_dir))

    contact_sheet = out_root / "contact_sheet.png"
    contact_index = write_contact_sheet(variant_dirs, contact_sheet)
    data_yaml, data_obb_yaml, data_fragments_yaml = write_yolo_dataset(
        variant_dirs,
        out_root,
        args.scene_mode,
        args.fragment_review_policy,
    )
    write_json(
        out_root / "qa" / "contact_index.json",
        {
            "contact_sheet": str(contact_sheet.relative_to(out_root)),
            "rows": contact_index,
            "cell_size": {"width": 320, "height": 240, "header_height": 30},
        },
    )
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
