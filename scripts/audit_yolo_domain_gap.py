#!/usr/bin/env python
"""Compare real and synthetic image/label statistics inside a YOLO dataset split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_STAT_KEYS = [
    "width",
    "height",
    "aspect",
    "luma_mean",
    "luma_std",
    "luma_p05",
    "luma_p95",
    "saturation_mean",
    "saturation_std",
    "sharpness_grad_var",
]
BOX_STAT_KEYS = ["box_width", "box_height", "box_area", "box_aspect"]
DOMAIN_GAP_PRESETS = {
    "accepted_blend_v1": {
        "image": {
            "luma_mean": 0.18,
            "luma_std": 0.16,
            "luma_p05": 0.25,
            "luma_p95": 0.25,
            "saturation_mean": 0.08,
            "saturation_std": 0.13,
            "sharpness_grad_var": 0.04,
        },
        "box": {
            "box_area": 0.45,
            "box_width": 0.50,
            "box_height": 0.45,
            "box_aspect": 0.15,
        },
        "max_synthetic_image_ratio": 0.60,
        "max_synthetic_box_ratio": 1.25,
        "max_synthetic_class_box_ratio": 3.00,
    },
    "accepted_blend_geometry_v1": {
        "image": {
            "luma_mean": 0.18,
            "luma_std": 0.16,
            "luma_p05": 0.25,
            "luma_p95": 0.25,
            "saturation_mean": 0.08,
            "saturation_std": 0.13,
            "sharpness_grad_var": 0.04,
        },
        "box": {
            "box_area": 0.25,
            "box_width": 0.35,
            "box_height": 0.35,
            "box_aspect": 0.20,
        },
        "class_box": {
            "box_area": 0.30,
            "box_width": 0.40,
            "box_height": 0.40,
            "box_aspect": 0.25,
        },
        "max_synthetic_image_ratio": 0.60,
        "max_synthetic_box_ratio": 1.25,
        "max_synthetic_class_box_ratio": 3.00,
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="YOLO data YAML.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--image-csv-out", type=Path, default=None)
    parser.add_argument("--box-csv-out", type=Path, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--fail-on-gap",
        action="store_true",
        help="Exit non-zero if requested real/synthetic domain-gap limits fail.",
    )
    parser.add_argument(
        "--gate-preset",
        default="",
        choices=["", *sorted(DOMAIN_GAP_PRESETS)],
        help="Named domain-gap limit set. Explicit threshold flags override preset values.",
    )
    parser.add_argument(
        "--max-abs-image-delta",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Limit abs(synthetic-real) image-stat mean delta, e.g. luma_mean=0.18. Repeatable.",
    )
    parser.add_argument(
        "--max-abs-box-delta",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help="Limit abs(synthetic-real) box-stat mean delta, e.g. box_area=0.45. Repeatable.",
    )
    parser.add_argument(
        "--max-abs-class-box-delta",
        action="append",
        default=[],
        metavar="METRIC=VALUE",
        help=(
            "Limit abs(synthetic-real) per-class box-stat mean delta for classes present in both families, "
            "e.g. box_area=0.30. Repeatable."
        ),
    )
    parser.add_argument(
        "--class-box-delta-class",
        action="append",
        default=[],
        metavar="CLASS[,CLASS...]",
        help="Restrict --max-abs-class-box-delta checks to specific class names. Repeatable or comma-separated.",
    )
    parser.add_argument("--max-synthetic-image-ratio", type=float, default=None)
    parser.add_argument("--max-synthetic-box-ratio", type=float, default=None)
    parser.add_argument("--max-synthetic-class-box-ratio", type=float, default=None)
    parser.add_argument("--min-real-images", type=int, default=1)
    parser.add_argument("--min-synthetic-images", type=int, default=1)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return document


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    root_value = Path(str(config.get("path", "."))).expanduser()
    return root_value if root_value.is_absolute() else (config_path.parent / root_value).resolve()


def split_root(dataset_root: Path, split_path: str) -> Path:
    path = Path(split_path)
    return path if path.is_absolute() else dataset_root / path


def read_split_list(dataset_root: Path, split_path: str) -> list[Path]:
    list_path = split_root(dataset_root, split_path)
    images: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        images.append(path if path.is_absolute() else dataset_root / path)
    return images


def iter_split_images(dataset_root: Path, split_value: str | list[str]) -> list[Path]:
    split_paths = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for split_path in split_paths:
        resolved = split_root(dataset_root, str(split_path))
        if resolved.suffix.lower() == ".txt":
            images.extend(read_split_list(dataset_root, str(split_path)))
        else:
            images.extend(
                sorted(path for path in resolved.glob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
            )
    return images


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def source_group(image: Path) -> str:
    rel = repo_rel(image)
    if rel.startswith("data/synthetic/"):
        parts = rel.split("/")
        return "synthetic:" + parts[2] if len(parts) > 2 else "synthetic"
    if rel.startswith("data/cashsnap_v1/"):
        return "real"
    return "other"


def source_family(group: str) -> str:
    return "synthetic" if group.startswith("synthetic:") else group


def class_name(names: dict[Any, Any] | list[Any], class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, list) and class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def image_stats(image: Path) -> dict[str, Any]:
    with Image.open(image) as handle:
        rgb = handle.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0

    height, width = array.shape[:2]
    luma = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    max_rgb = array.max(axis=2)
    min_rgb = array.min(axis=2)
    saturation = np.divide(max_rgb - min_rgb, np.maximum(max_rgb, 1e-6))
    dx = np.diff(luma, axis=1)
    dy = np.diff(luma, axis=0)
    sharpness = float(dx.var() + dy.var())
    return {
        "width": width,
        "height": height,
        "aspect": width / height,
        "luma_mean": float(luma.mean()),
        "luma_std": float(luma.std()),
        "luma_p05": float(np.quantile(luma, 0.05)),
        "luma_p95": float(np.quantile(luma, 0.95)),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "sharpness_grad_var": sharpness,
    }


def label_rows(image: Path, names: dict[Any, Any] | list[Any]) -> list[dict[str, Any]]:
    label = label_path_for_image(image)
    if not label.exists():
        raise FileNotFoundError(f"missing label for {image}: {label}")
    rows = []
    for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label}:{line_no} expected 5 YOLO fields, found {len(parts)}")
        class_id = int(parts[0])
        width = float(parts[3])
        height = float(parts[4])
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name(names, class_id),
                "x_center": float(parts[1]),
                "y_center": float(parts[2]),
                "box_width": width,
                "box_height": height,
                "box_area": width * height,
                "box_aspect": width / height if height else None,
            }
        )
    return rows


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    average = sum(values) / len(values)
    return float(math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1)))


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def summarize_numeric(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        summary[key] = {
            "mean": mean(values),
            "stdev": stdev(values),
            "p05": quantile(values, 0.05),
            "p50": quantile(values, 0.50),
            "p95": quantile(values, 0.95),
        }
    return summary


def summarize_class_box_stats(box_rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_names = sorted({str(row["class_name"]) for row in box_rows})
    return {
        class_name: summarize_numeric([row for row in box_rows if str(row["class_name"]) == class_name], BOX_STAT_KEYS)
        for class_name in class_names
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


def summarize(image_rows: list[dict[str, Any]], box_rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({row["source_group"] for row in image_rows})
    families = sorted({row["source_family"] for row in image_rows})

    def group_summary(group_key: str, group_value: str) -> dict[str, Any]:
        group_images = [row for row in image_rows if row[group_key] == group_value]
        group_boxes = [row for row in box_rows if row[group_key] == group_value]
        class_counts = Counter(row["class_name"] for row in group_boxes)
        return {
            "images": len(group_images),
            "backgrounds": sum(1 for row in group_images if int(row["box_count"]) == 0),
            "boxes": len(group_boxes),
            "class_counts": dict(sorted(class_counts.items())),
            "image_stats": summarize_numeric(group_images, IMAGE_STAT_KEYS),
            "box_stats": summarize_numeric(group_boxes, BOX_STAT_KEYS),
            "class_box_stats": summarize_class_box_stats(group_boxes),
        }

    by_group = {group: group_summary("source_group", group) for group in groups}
    by_family = {family: group_summary("source_family", family) for family in families}
    deltas: dict[str, Any] = {}
    if "real" in by_family and "synthetic" in by_family:
        deltas["synthetic_minus_real"] = {}
        for section in ["image_stats", "box_stats"]:
            deltas["synthetic_minus_real"][section] = {}
            for metric, real_stats in by_family["real"][section].items():
                synth_stats = by_family["synthetic"][section].get(metric, {})
                real_mean = real_stats.get("mean")
                synth_mean = synth_stats.get("mean")
                deltas["synthetic_minus_real"][section][metric] = (
                    None if real_mean is None or synth_mean is None else synth_mean - real_mean
                )
        deltas["synthetic_minus_real"]["class_box_stats"] = {}
        real_class_stats = by_family["real"].get("class_box_stats", {})
        synthetic_class_stats = by_family["synthetic"].get("class_box_stats", {})
        for class_name in sorted(set(real_class_stats) | set(synthetic_class_stats)):
            deltas["synthetic_minus_real"]["class_box_stats"][class_name] = {}
            for metric, real_stats in real_class_stats.get(class_name, {}).items():
                synth_stats = synthetic_class_stats.get(class_name, {}).get(metric, {})
                real_mean = real_stats.get("mean")
                synth_mean = synth_stats.get("mean")
                deltas["synthetic_minus_real"]["class_box_stats"][class_name][metric] = (
                    None if real_mean is None or synth_mean is None else synth_mean - real_mean
                )
    return {
        "by_family": by_family,
        "by_group": by_group,
        "deltas": deltas,
    }


def parse_metric_limits(specs: list[str], valid_keys: list[str], label: str) -> dict[str, float]:
    limits: dict[str, float] = {}
    valid = set(valid_keys)
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"{label} limit must be METRIC=VALUE, got {spec!r}")
        metric, raw_value = (part.strip() for part in spec.split("=", 1))
        if metric not in valid:
            raise SystemExit(f"unknown {label} metric {metric!r}; expected one of {sorted(valid)}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise SystemExit(f"{label} limit for {metric!r} must be numeric, got {raw_value!r}") from exc
        if value < 0:
            raise SystemExit(f"{label} limit for {metric!r} must be non-negative")
        limits[metric] = value
    return limits


def parse_class_names(values: list[str]) -> list[str]:
    class_names: list[str] = []
    seen: set[str] = set()
    for value in values:
        for raw_class in value.split(","):
            class_name = raw_class.strip()
            if class_name and class_name not in seen:
                class_names.append(class_name)
                seen.add(class_name)
    return class_names


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def gate_domain_gap(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    preset = DOMAIN_GAP_PRESETS.get(args.gate_preset, {})
    image_limits = dict(preset.get("image", {}))
    image_limits.update(parse_metric_limits(args.max_abs_image_delta, IMAGE_STAT_KEYS, "image"))
    box_limits = dict(preset.get("box", {}))
    box_limits.update(parse_metric_limits(args.max_abs_box_delta, BOX_STAT_KEYS, "box"))
    class_box_limits = dict(preset.get("class_box", {}))
    class_box_limits.update(parse_metric_limits(args.max_abs_class_box_delta, BOX_STAT_KEYS, "class box"))
    preset_class_box_classes = [str(item) for item in preset.get("class_box_classes", [])]
    class_box_classes = parse_class_names(preset_class_box_classes + args.class_box_delta_class)
    max_synthetic_image_ratio = (
        args.max_synthetic_image_ratio
        if args.max_synthetic_image_ratio is not None
        else preset.get("max_synthetic_image_ratio")
    )
    max_synthetic_box_ratio = (
        args.max_synthetic_box_ratio
        if args.max_synthetic_box_ratio is not None
        else preset.get("max_synthetic_box_ratio")
    )
    max_synthetic_class_box_ratio = (
        args.max_synthetic_class_box_ratio
        if args.max_synthetic_class_box_ratio is not None
        else preset.get("max_synthetic_class_box_ratio")
    )
    gate_requested = bool(
        args.fail_on_gap
        or args.gate_preset
        or image_limits
        or box_limits
        or class_box_limits
        or max_synthetic_image_ratio is not None
        or max_synthetic_box_ratio is not None
        or max_synthetic_class_box_ratio is not None
    )
    failures: list[str] = []
    by_family = payload.get("by_family", {})
    real = by_family.get("real")
    synthetic = by_family.get("synthetic")

    if gate_requested:
        if not isinstance(real, dict):
            failures.append("missing real image family")
        if not isinstance(synthetic, dict):
            failures.append("missing synthetic image family")
    if not isinstance(real, dict) or not isinstance(synthetic, dict):
        return {
            "requested": gate_requested,
            "passed": not failures,
            "failures": failures,
            "limits": {"image": image_limits, "box": box_limits, "class_box": class_box_limits},
        }

    real_images = int(real.get("images", 0) or 0)
    synthetic_images = int(synthetic.get("images", 0) or 0)
    real_boxes = int(real.get("boxes", 0) or 0)
    synthetic_boxes = int(synthetic.get("boxes", 0) or 0)
    if gate_requested and real_images < args.min_real_images:
        failures.append(f"real image count {real_images} below minimum {args.min_real_images}")
    if gate_requested and synthetic_images < args.min_synthetic_images:
        failures.append(f"synthetic image count {synthetic_images} below minimum {args.min_synthetic_images}")

    image_ratio = ratio(synthetic_images, real_images)
    box_ratio = ratio(synthetic_boxes, real_boxes)
    if max_synthetic_image_ratio is not None:
        if image_ratio is None:
            failures.append("cannot compute synthetic/real image ratio with zero real images")
        elif image_ratio > max_synthetic_image_ratio:
            failures.append(f"synthetic/real image ratio {image_ratio:.4f} exceeds {max_synthetic_image_ratio:.4f}")
    if max_synthetic_box_ratio is not None:
        if box_ratio is None:
            failures.append("cannot compute synthetic/real box ratio with zero real boxes")
        elif box_ratio > max_synthetic_box_ratio:
            failures.append(f"synthetic/real box ratio {box_ratio:.4f} exceeds {max_synthetic_box_ratio:.4f}")
    class_box_ratios: dict[str, float | None] = {}
    if max_synthetic_class_box_ratio is not None:
        real_classes = real.get("class_counts", {})
        synthetic_classes = synthetic.get("class_counts", {})
        if not isinstance(real_classes, dict) or not isinstance(synthetic_classes, dict):
            failures.append("cannot compute class box ratios from malformed class_counts")
        else:
            for class_name in sorted(set(real_classes) | set(synthetic_classes)):
                real_count = int(real_classes.get(class_name, 0) or 0)
                synthetic_count = int(synthetic_classes.get(class_name, 0) or 0)
                class_ratio = ratio(synthetic_count, real_count)
                class_box_ratios[str(class_name)] = class_ratio
                if class_ratio is None:
                    if synthetic_count > 0:
                        failures.append(f"{class_name} has {synthetic_count} synthetic boxes and zero real boxes")
                elif class_ratio > max_synthetic_class_box_ratio:
                    failures.append(
                        f"{class_name} synthetic/real box ratio {class_ratio:.4f} exceeds "
                        f"{max_synthetic_class_box_ratio:.4f}"
                    )

    deltas = payload.get("deltas", {}).get("synthetic_minus_real", {})
    for section_name, limits in (("image_stats", image_limits), ("box_stats", box_limits)):
        section = deltas.get(section_name, {})
        for metric, limit in limits.items():
            delta = section.get(metric)
            if delta is None:
                failures.append(f"missing synthetic-real delta for {section_name}.{metric}")
                continue
            if abs(float(delta)) > limit:
                failures.append(f"{section_name}.{metric} delta {float(delta):.6f} exceeds abs limit {limit:.6f}")

    checked_class_box_deltas: dict[str, dict[str, float | None]] = {}
    if class_box_limits:
        class_deltas = deltas.get("class_box_stats", {})
        real_classes = real.get("class_counts", {})
        synthetic_classes = synthetic.get("class_counts", {})
        if not isinstance(real_classes, dict) or not isinstance(synthetic_classes, dict):
            failures.append("cannot compute per-class box deltas from malformed class_counts")
        elif not isinstance(class_deltas, dict):
            failures.append("cannot compute per-class box deltas from malformed class_box_stats")
        else:
            checked_classes = class_box_classes or sorted(set(real_classes) & set(synthetic_classes))
            for class_name in checked_classes:
                class_delta = class_deltas.get(class_name)
                checked_class_box_deltas[class_name] = {}
                if not isinstance(class_delta, dict):
                    failures.append(f"missing synthetic-real class box delta for {class_name}")
                    continue
                for metric, limit in class_box_limits.items():
                    delta = class_delta.get(metric)
                    checked_class_box_deltas[class_name][metric] = delta
                    if delta is None:
                        failures.append(f"missing synthetic-real delta for class_box_stats.{class_name}.{metric}")
                        continue
                    if abs(float(delta)) > limit:
                        failures.append(
                            f"class_box_stats.{class_name}.{metric} delta {float(delta):.6f} exceeds "
                            f"abs limit {limit:.6f}"
                        )

    return {
        "requested": gate_requested,
        "passed": not failures,
        "failures": failures,
        "limits": {
            "image": image_limits,
            "box": box_limits,
            "class_box": class_box_limits,
            "class_box_classes": class_box_classes,
            "preset": args.gate_preset,
            "max_synthetic_image_ratio": max_synthetic_image_ratio,
            "max_synthetic_box_ratio": max_synthetic_box_ratio,
            "max_synthetic_class_box_ratio": max_synthetic_class_box_ratio,
            "min_real_images": args.min_real_images,
            "min_synthetic_images": args.min_synthetic_images,
        },
        "observed": {
            "real_images": real_images,
            "synthetic_images": synthetic_images,
            "real_boxes": real_boxes,
            "synthetic_boxes": synthetic_boxes,
            "synthetic_image_ratio": image_ratio,
            "synthetic_box_ratio": box_ratio,
            "synthetic_class_box_ratios": class_box_ratios,
            "checked_class_box_deltas": checked_class_box_deltas,
        },
    }


def main() -> int:
    args = parse_args()
    data_path = resolve(args.data)
    config = read_yaml(data_path)
    root = data_root(data_path, config)
    if args.split not in config:
        raise SystemExit(f"split {args.split!r} missing from {data_path}")
    names = config.get("names", {})
    images = iter_split_images(root, config[args.split])
    if args.max_images is not None:
        images = images[: args.max_images]

    image_rows: list[dict[str, Any]] = []
    box_rows: list[dict[str, Any]] = []
    for image in images:
        group = source_group(image)
        family = source_family(group)
        boxes = label_rows(image, names)
        row = {
            "image": repo_rel(image),
            "source_group": group,
            "source_family": family,
            "box_count": len(boxes),
            **image_stats(image),
        }
        image_rows.append(row)
        for box in boxes:
            box_rows.append(
                {
                    "image": repo_rel(image),
                    "source_group": group,
                    "source_family": family,
                    **box,
                }
            )

    payload = {
        "data": repo_rel(data_path),
        "split": args.split,
        "images": len(image_rows),
        "boxes": len(box_rows),
        **summarize(image_rows, box_rows),
    }
    payload["domain_gap_gate"] = gate_domain_gap(payload, args)
    if args.json_out:
        write_json(resolve(args.json_out), payload)
    if args.image_csv_out:
        write_csv(resolve(args.image_csv_out), image_rows)
    if args.box_csv_out:
        write_csv(resolve(args.box_csv_out), box_rows)

    print(
        f"images={payload['images']} boxes={payload['boxes']} "
        f"families={','.join(sorted(payload['by_family']))}",
        flush=True,
    )
    if args.json_out:
        print(f"wrote_json={repo_rel(resolve(args.json_out))}")
    gate = payload["domain_gap_gate"]
    if gate["requested"]:
        print("domain_gap_gate=" + ("passed" if gate["passed"] else "failed"))
        for failure in gate["failures"]:
            print(f"- {failure}")
    if args.fail_on_gap and not gate["passed"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
