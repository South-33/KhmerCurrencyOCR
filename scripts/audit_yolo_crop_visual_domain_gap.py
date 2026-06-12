#!/usr/bin/env python
"""Compare real and synthetic visual statistics on YOLO box crops."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

import audit_yolo_domain_gap as domain_gap


ROOT = Path(__file__).resolve().parents[1]
CROP_SHARPNESS_SIZE = (128, 128)
CROP_STAT_KEYS = [
    "crop_width_px",
    "crop_height_px",
    "crop_area_px",
    "luma_mean",
    "luma_std",
    "luma_p05",
    "luma_p95",
    "saturation_mean",
    "saturation_std",
    "sharpness_grad_var",
    "red_mean",
    "green_mean",
    "blue_mean",
]
CROP_VISUAL_GAP_PRESETS = {
    "clean_dynamic_range_v1": {
        "crop": {
            "luma_std": 0.07,
            "luma_p05": 0.10,
            "luma_p95": 0.08,
            "saturation_std": 0.06,
            "sharpness_grad_var": 0.015,
        },
        "class_crop": {},
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="YOLO data YAML.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--pad-frac", type=float, default=0.0, help="Optional crop padding as a fraction of box width/height.")
    parser.add_argument("--min-crop-pixels", type=int, default=16, help="Skip degenerate tiny crops below this area.")
    parser.add_argument("--top-class-deltas", type=int, default=8)
    parser.add_argument(
        "--max-images-per-family",
        type=int,
        default=0,
        help="Deterministically sample at most this many images per source family before crop extraction.",
    )
    parser.add_argument(
        "--max-crops-per-family",
        type=int,
        default=0,
        help="Stop extracting crops after this many accepted crops per source family.",
    )
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for deterministic image sampling.")
    parser.add_argument(
        "--gate-preset",
        default="",
        choices=["", *sorted(CROP_VISUAL_GAP_PRESETS)],
        help="Named crop visual gap limit set. Explicit threshold flags override preset values.",
    )
    parser.add_argument(
        "--class-name",
        action="append",
        default=[],
        metavar="CLASS[,CLASS...]",
        help="Restrict rows and gate checks to specific classes. Repeatable or comma-separated.",
    )
    parser.add_argument(
        "--max-abs-crop-delta",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Limit abs(synthetic-real) aggregate crop-stat mean delta. Repeatable.",
    )
    parser.add_argument(
        "--max-abs-class-crop-delta",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Limit abs(synthetic-real) per-class crop-stat mean delta. Repeatable.",
    )
    parser.add_argument("--min-real-crops", type=int, default=1)
    parser.add_argument("--min-synthetic-crops", type=int, default=1)
    parser.add_argument("--fail-on-gap", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_class_names(values: list[str]) -> list[str]:
    class_names: list[str] = []
    seen: set[str] = set()
    for value in values:
        for raw_class in value.replace(",", " ").split():
            class_name = raw_class.strip()
            if class_name and class_name not in seen:
                class_names.append(class_name)
                seen.add(class_name)
    return class_names


def parse_metric_limits(specs: list[str]) -> dict[str, float]:
    limits: dict[str, float] = {}
    valid = set(CROP_STAT_KEYS)
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"crop delta limit must be METRIC=VALUE, got {spec!r}")
        metric, raw_value = (part.strip() for part in spec.split("=", 1))
        if metric not in valid:
            raise SystemExit(f"unknown crop metric {metric!r}; expected one of {sorted(valid)}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise SystemExit(f"crop delta limit for {metric!r} must be numeric, got {raw_value!r}") from exc
        if value < 0:
            raise SystemExit(f"crop delta limit for {metric!r} must be non-negative")
        limits[metric] = value
    return limits


def read_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"expected YAML mapping: {repo_rel(path)}")
    return document


def crop_box(row: dict[str, Any], image_size: tuple[int, int], pad_frac: float) -> tuple[int, int, int, int] | None:
    image_w, image_h = image_size
    box_w = float(row["box_width"]) * image_w
    box_h = float(row["box_height"]) * image_h
    cx = float(row["x_center"]) * image_w
    cy = float(row["y_center"]) * image_h
    pad_x = box_w * pad_frac
    pad_y = box_h * pad_frac
    left = max(0, int(round(cx - box_w / 2 - pad_x)))
    top = max(0, int(round(cy - box_h / 2 - pad_y)))
    right = min(image_w, int(round(cx + box_w / 2 + pad_x)))
    bottom = min(image_h, int(round(cy + box_h / 2 + pad_y)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def crop_stats(crop: Image.Image) -> dict[str, Any]:
    rgb = crop.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    sharpness_rgb = rgb.resize(CROP_SHARPNESS_SIZE, Image.Resampling.BILINEAR)
    sharpness_array = np.asarray(sharpness_rgb, dtype=np.float32) / 255.0
    height, width = array.shape[:2]
    luma = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    sharpness_luma = (
        0.2126 * sharpness_array[..., 0]
        + 0.7152 * sharpness_array[..., 1]
        + 0.0722 * sharpness_array[..., 2]
    )
    max_rgb = array.max(axis=2)
    min_rgb = array.min(axis=2)
    saturation = np.divide(max_rgb - min_rgb, np.maximum(max_rgb, 1e-6))
    sharp_h, sharp_w = sharpness_luma.shape[:2]
    dx = np.diff(sharpness_luma, axis=1) if sharp_w > 1 else np.asarray([0.0], dtype=np.float32)
    dy = np.diff(sharpness_luma, axis=0) if sharp_h > 1 else np.asarray([0.0], dtype=np.float32)
    return {
        "crop_width_px": int(width),
        "crop_height_px": int(height),
        "crop_area_px": int(width * height),
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "luma_p05": float(np.quantile(luma, 0.05)),
        "luma_p95": float(np.quantile(luma, 0.95)),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "sharpness_grad_var": float(dx.var() + dy.var()),
        "red_mean": float(array[..., 0].mean()),
        "green_mean": float(array[..., 1].mean()),
        "blue_mean": float(array[..., 2].mean()),
    }


def stable_sample_key(path: Path, seed: int) -> str:
    key = f"{seed}:{repo_rel(path)}".encode("utf-8")
    return hashlib.sha256(key).hexdigest()


def image_records(image_paths: list[Path], max_images_per_family: int, sample_seed: int) -> list[dict[str, Any]]:
    records = [
        {
            "path": image_path,
            "source_group": domain_gap.source_group(image_path),
            "source_family": domain_gap.source_family(domain_gap.source_group(image_path)),
        }
        for image_path in image_paths
    ]
    if max_images_per_family <= 0:
        return records
    sampled: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_family.setdefault(str(record["source_family"]), []).append(record)
    for _family, family_records in sorted(by_family.items()):
        sampled.extend(
            sorted(family_records, key=lambda record: stable_sample_key(Path(record["path"]), sample_seed))[
                :max_images_per_family
            ]
        )
    return sampled


def crop_rows(
    config_path: Path,
    split: str,
    class_filter: set[str],
    pad_frac: float,
    min_crop_pixels: int,
    max_images_per_family: int,
    max_crops_per_family: int,
    sample_seed: int,
) -> list[dict[str, Any]]:
    config = read_yaml(config_path)
    dataset_root = domain_gap.data_root(config_path, config)
    split_value = config.get(split)
    if not isinstance(split_value, (str, list)):
        raise SystemExit(f"{repo_rel(config_path)} split {split!r} must be a string or list")
    names = config.get("names", {})
    rows: list[dict[str, Any]] = []
    crop_counts_by_family: Counter[str] = Counter()
    images = image_records(domain_gap.iter_split_images(dataset_root, split_value), max_images_per_family, sample_seed)
    for record in images:
        image_path = Path(record["path"])
        source_group = str(record["source_group"])
        source_family = str(record["source_family"])
        if max_crops_per_family > 0 and crop_counts_by_family[source_family] >= max_crops_per_family:
            continue
        label_rows = domain_gap.label_rows(image_path, names)
        if not label_rows:
            continue
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            for label_index, label_row in enumerate(label_rows):
                if max_crops_per_family > 0 and crop_counts_by_family[source_family] >= max_crops_per_family:
                    break
                class_name = str(label_row["class_name"])
                if class_filter and class_name not in class_filter:
                    continue
                box = crop_box(label_row, image.size, pad_frac)
                if box is None:
                    continue
                left, top, right, bottom = box
                if (right - left) * (bottom - top) < min_crop_pixels:
                    continue
                stats = crop_stats(image.crop(box))
                rows.append(
                    {
                        "image": repo_rel(image_path),
                        "label_index": label_index,
                        "class_name": class_name,
                        "source_group": source_group,
                        "source_family": source_family,
                        "crop_left": left,
                        "crop_top": top,
                        "crop_right": right,
                        "crop_bottom": bottom,
                        **stats,
                    }
                )
                crop_counts_by_family[source_family] += 1
    return rows


def summarize_class_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_names = sorted({str(row["class_name"]) for row in rows})
    return {
        class_name: domain_gap.summarize_numeric(
            [row for row in rows if str(row["class_name"]) == class_name],
            CROP_STAT_KEYS,
        )
        for class_name in class_names
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({str(row["source_group"]) for row in rows})
    families = sorted({str(row["source_family"]) for row in rows})

    def group_summary(key: str, value: str) -> dict[str, Any]:
        group_rows = [row for row in rows if str(row[key]) == value]
        class_counts = Counter(str(row["class_name"]) for row in group_rows)
        return {
            "crops": len(group_rows),
            "class_counts": dict(sorted(class_counts.items())),
            "crop_stats": domain_gap.summarize_numeric(group_rows, CROP_STAT_KEYS),
            "class_crop_stats": summarize_class_stats(group_rows),
        }

    by_group = {group: group_summary("source_group", group) for group in groups}
    by_family = {family: group_summary("source_family", family) for family in families}
    deltas: dict[str, Any] = {}
    if "real" in by_family and "synthetic" in by_family:
        deltas["synthetic_minus_real"] = {"crop_stats": {}, "class_crop_stats": {}}
        real_crop_stats = by_family["real"].get("crop_stats", {})
        synthetic_crop_stats = by_family["synthetic"].get("crop_stats", {})
        for metric, real_metric_stats in real_crop_stats.items():
            synth_mean = synthetic_crop_stats.get(metric, {}).get("mean")
            real_mean = real_metric_stats.get("mean")
            deltas["synthetic_minus_real"]["crop_stats"][metric] = (
                None if real_mean is None or synth_mean is None else synth_mean - real_mean
            )
        real_stats = by_family["real"].get("class_crop_stats", {})
        synthetic_stats = by_family["synthetic"].get("class_crop_stats", {})
        for class_name in sorted(set(real_stats) | set(synthetic_stats)):
            deltas["synthetic_minus_real"]["class_crop_stats"][class_name] = {}
            for metric, real_metric_stats in real_stats.get(class_name, {}).items():
                synth_mean = synthetic_stats.get(class_name, {}).get(metric, {}).get("mean")
                real_mean = real_metric_stats.get("mean")
                deltas["synthetic_minus_real"]["class_crop_stats"][class_name][metric] = (
                    None if real_mean is None or synth_mean is None else synth_mean - real_mean
                )
    return {"by_family": by_family, "by_group": by_group, "deltas": deltas}


def gate(
    payload: dict[str, Any],
    crop_limits: dict[str, float],
    class_limits: dict[str, float],
    classes: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    requested = bool(args.fail_on_gap or crop_limits or class_limits)
    failures: list[str] = []
    by_family = payload.get("by_family", {})
    real = by_family.get("real")
    synthetic = by_family.get("synthetic")
    if requested:
        if not isinstance(real, dict):
            failures.append("missing real crop family")
        if not isinstance(synthetic, dict):
            failures.append("missing synthetic crop family")
    if not isinstance(real, dict) or not isinstance(synthetic, dict):
        return {
            "requested": requested,
            "passed": not failures,
            "failures": failures,
            "limits": {"crop": crop_limits, "class_crop": class_limits},
        }

    real_crops = int(real.get("crops", 0) or 0)
    synthetic_crops = int(synthetic.get("crops", 0) or 0)
    if requested and real_crops < args.min_real_crops:
        failures.append(f"real crop count {real_crops} below minimum {args.min_real_crops}")
    if requested and synthetic_crops < args.min_synthetic_crops:
        failures.append(f"synthetic crop count {synthetic_crops} below minimum {args.min_synthetic_crops}")

    deltas = payload.get("deltas", {}).get("synthetic_minus_real", {})
    crop_deltas = deltas.get("crop_stats", {})
    if not isinstance(crop_deltas, dict):
        crop_deltas = {}
    for metric, limit in crop_limits.items():
        delta = crop_deltas.get(metric)
        if delta is None:
            if requested:
                failures.append(f"crop_stats.{metric} is unavailable")
            continue
        if abs(float(delta)) > limit:
            failures.append(f"crop_stats.{metric} delta {float(delta):.6f} exceeds abs limit {limit:.6f}")

    class_deltas_by_name = deltas.get("class_crop_stats", {})
    if not isinstance(class_deltas_by_name, dict):
        class_deltas_by_name = {}
    target_classes = classes or sorted(class_deltas_by_name)
    for class_name in target_classes:
        class_deltas = class_deltas_by_name.get(class_name)
        if not isinstance(class_deltas, dict):
            if requested:
                failures.append(f"missing crop deltas for class {class_name}")
            continue
        for metric, limit in class_limits.items():
            delta = class_deltas.get(metric)
            if delta is None:
                if requested:
                    failures.append(f"class_crop_stats.{class_name}.{metric} is unavailable")
                continue
            if abs(float(delta)) > limit:
                failures.append(
                    f"class_crop_stats.{class_name}.{metric} delta {float(delta):.6f} exceeds abs limit {limit:.6f}"
                )
    return {
        "requested": requested,
        "passed": not failures,
        "failures": failures,
        "limits": {"crop": crop_limits, "class_crop": class_limits},
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_top_class_deltas(payload: dict[str, Any], top_count: int, metrics: list[str]) -> None:
    deltas = payload.get("deltas", {}).get("synthetic_minus_real", {}).get("class_crop_stats", {})
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for class_name, class_deltas in deltas.items():
        if not isinstance(class_deltas, dict):
            continue
        score = sum(abs(float(class_deltas[metric])) for metric in metrics if class_deltas.get(metric) is not None)
        scored.append((score, str(class_name), class_deltas))
    for _score, class_name, class_deltas in sorted(scored, reverse=True)[:top_count]:
        metric_text = " ".join(
            f"{metric} {float(class_deltas[metric]):+.3f}"
            for metric in metrics
            if class_deltas.get(metric) is not None
        )
        print(f"- {class_name}: {metric_text}")


def main() -> int:
    args = parse_args()
    if args.pad_frac < 0:
        raise SystemExit("--pad-frac must be non-negative")
    if args.min_crop_pixels < 1:
        raise SystemExit("--min-crop-pixels must be positive")
    if args.max_images_per_family < 0:
        raise SystemExit("--max-images-per-family must be non-negative")
    if args.max_crops_per_family < 0:
        raise SystemExit("--max-crops-per-family must be non-negative")
    data_path = resolve(args.data)
    class_names = parse_class_names(args.class_name)
    rows = crop_rows(
        data_path,
        args.split,
        set(class_names),
        args.pad_frac,
        args.min_crop_pixels,
        args.max_images_per_family,
        args.max_crops_per_family,
        args.sample_seed,
    )
    payload = summarize(rows)
    preset = CROP_VISUAL_GAP_PRESETS.get(args.gate_preset, {})
    crop_limits = dict(preset.get("crop", {})) if isinstance(preset.get("crop", {}), dict) else {}
    class_limits = dict(preset.get("class_crop", {})) if isinstance(preset.get("class_crop", {}), dict) else {}
    crop_limits.update(parse_metric_limits(args.max_abs_crop_delta))
    class_limits.update(parse_metric_limits(args.max_abs_class_crop_delta))
    payload["crop_visual_gap_gate"] = gate(payload, crop_limits, class_limits, class_names, args)
    payload["crop_visual_gap_gate"]["preset"] = args.gate_preset
    payload["data"] = repo_rel(data_path)
    payload["split"] = args.split
    payload["classes"] = class_names
    payload["sampling"] = {
        "max_images_per_family": args.max_images_per_family,
        "max_crops_per_family": args.max_crops_per_family,
        "sample_seed": args.sample_seed,
    }
    if args.json_out:
        json_out = resolve(args.json_out)
        write_json(json_out, payload)
        print(f"wrote_json={repo_rel(json_out)}")
    if args.csv_out:
        csv_out = resolve(args.csv_out)
        write_csv(csv_out, rows)
        print(f"wrote_csv={repo_rel(csv_out)}")

    by_family = payload.get("by_family", {})
    real_crops = by_family.get("real", {}).get("crops", 0) if isinstance(by_family.get("real"), dict) else 0
    synthetic_crops = (
        by_family.get("synthetic", {}).get("crops", 0) if isinstance(by_family.get("synthetic"), dict) else 0
    )
    print(f"crops=real:{real_crops} synthetic:{synthetic_crops}")
    crop_deltas = payload.get("deltas", {}).get("synthetic_minus_real", {}).get("crop_stats", {})
    if isinstance(crop_deltas, dict) and crop_deltas:
        aggregate_metrics = list(crop_limits) if crop_limits else [
            "luma_mean",
            "luma_std",
            "luma_p05",
            "luma_p95",
            "saturation_std",
            "sharpness_grad_var",
        ]
        metric_text = " ".join(
            f"{metric} {float(crop_deltas[metric]):+.3f}"
            for metric in aggregate_metrics
            if crop_deltas.get(metric) is not None
        )
        print(f"aggregate_crop_deltas: {metric_text}")
    metrics = list(class_limits) if class_limits else ["luma_mean", "saturation_mean", "sharpness_grad_var", "red_mean"]
    print("top_class_crop_deltas:")
    print_top_class_deltas(payload, args.top_class_deltas, metrics)
    gate_result = payload["crop_visual_gap_gate"]
    if gate_result.get("requested"):
        print("crop_visual_gap_gate=" + ("passed" if gate_result.get("passed") else "failed"))
        for failure in gate_result.get("failures", []):
            print(f"- {failure}")
    if args.fail_on_gap and not gate_result.get("passed"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
