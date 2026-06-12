from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "cashsnap_two_stage_balanced_real_p24_trueempty_proposal_gate_browser_stack.json"
DEFAULT_ALLOWED_STATUSES = [
    "diagnostic_not_production",
    "diagnostic_current_product_clue_not_production",
    "diagnostic_current_product_clue_tuned_operating_point_not_production",
    "diagnostic_historical_conservative_product_clue_not_production",
    "diagnostic_mechanism_candidate_not_promoted",
]
VALID_NMS_SCORES = {"detector_conf", "fragment_conf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CashSnap browser/mobile stack artifacts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-total-mb", type=float, default=20.0)
    parser.add_argument(
        "--allow-status",
        action="append",
        default=None,
        help="Allowed stack status. Defaults to known diagnostic statuses; repeat to allow more statuses.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def check_artifact(label: str, path_text: str) -> tuple[float, list[str]]:
    path = resolve(Path(path_text))
    if not path.exists():
        return 0.0, [f"{label}: missing {path_text}"]
    if not path.is_file():
        return 0.0, [f"{label}: not a file {path_text}"]
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"{label}: {display_path(path)} {size_mb:.2f} MB")
    return size_mb, []


def number_between(
    section: dict[str, Any],
    key: str,
    low: float,
    high: float,
    errors: list[str],
    label: str,
) -> None:
    if key not in section:
        return
    try:
        value = float(section[key])
    except (TypeError, ValueError):
        errors.append(f"{label}.{key}: expected number")
        return
    if not low <= value <= high:
        errors.append(f"{label}.{key}: expected {low} <= value <= {high}, got {value}")


def positive_int(section: dict[str, Any], key: str, errors: list[str], label: str) -> None:
    try:
        value = int(section.get(key, 0))
    except (TypeError, ValueError):
        errors.append(f"{label}.{key}: expected positive integer")
        return
    if value <= 0:
        errors.append(f"{label}.{key}: expected positive integer, got {section.get(key)!r}")


def class_list(section: dict[str, Any], label: str, errors: list[str]) -> list[str]:
    classes = section.get("classes", [])
    if not isinstance(classes, list) or not classes:
        errors.append(f"{label}.classes: expected non-empty list")
        return []
    names = [str(item).strip() for item in classes]
    empty_count = sum(1 for name in names if not name)
    if empty_count:
        errors.append(f"{label}.classes: contains {empty_count} empty class name(s)")
    duplicates = sorted({name for name in names if name and names.count(name) > 1})
    if duplicates:
        errors.append(f"{label}.classes: duplicate class name(s) {duplicates}")
    return [name for name in names if name]


def normalization_errors(section: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    normalization = section.get("normalization", {})
    if not isinstance(normalization, dict):
        return [f"{label}.normalization: expected object"]
    for key, require_positive in [("mean", False), ("std", True)]:
        values = normalization.get(key, [])
        if not isinstance(values, list) or len(values) != 3:
            errors.append(f"{label}.normalization.{key}: expected 3-number list")
            continue
        for index, raw_value in enumerate(values):
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                errors.append(f"{label}.normalization.{key}[{index}]: expected number")
                continue
            if require_positive and value <= 0:
                errors.append(f"{label}.normalization.{key}[{index}]: expected positive std")
    return errors


def class_threshold_errors(
    section: dict[str, Any],
    *,
    key: str,
    classes: list[str],
    label: str,
) -> list[str]:
    raw_thresholds = section.get(key, {})
    if raw_thresholds in ({}, None):
        return []
    if not isinstance(raw_thresholds, dict):
        return [f"{label}.{key}: expected object mapping class name to threshold"]
    errors: list[str] = []
    class_set = set(classes)
    for raw_name, raw_value in raw_thresholds.items():
        class_name = str(raw_name).strip()
        if class_name not in class_set:
            errors.append(f"{label}.{key}: unknown class {class_name!r}")
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"{label}.{key}.{class_name}: expected number")
            continue
        if not 0.0 <= value <= 1.0:
            errors.append(f"{label}.{key}.{class_name}: expected 0 <= value <= 1, got {value}")
    return errors


def validate_stack_config(config: dict[str, Any], allowed_statuses: set[str]) -> list[str]:
    errors: list[str] = []
    status = str(config.get("status", "")).strip()
    if status not in allowed_statuses:
        errors.append(f"status {status!r} not in allowed statuses {sorted(allowed_statuses)}")
    if status.startswith("diagnostic"):
        known_limits = config.get("known_limits", [])
        if not isinstance(known_limits, list) or not known_limits:
            errors.append("diagnostic stack must document known_limits")

    detector = config.get("detector", {})
    fragment = config.get("fragment_classifier", {})
    fusion = config.get("fusion", {})
    for label, section in [("detector", detector), ("fragment_classifier", fragment), ("fusion", fusion)]:
        if not isinstance(section, dict):
            errors.append(f"{label}: expected object")
    if not isinstance(detector, dict) or not isinstance(fragment, dict) or not isinstance(fusion, dict):
        return errors

    detector_classes = class_list(detector, "detector", errors)
    fragment_classes = class_list(fragment, "fragment_classifier", errors)
    positive_int(detector, "input_size", errors, "detector")
    positive_int(fragment, "input_size", errors, "fragment_classifier")
    number_between(detector, "proposal_confidence", 0.0, 1.0, errors, "detector")
    number_between(detector, "proposal_iou", 0.0, 1.0, errors, "detector")
    errors.extend(
        class_threshold_errors(
            detector,
            key="class_min_conf",
            classes=detector_classes,
            label="detector",
        )
    )
    number_between(fragment, "crop_padding", 0.0, 0.5, errors, "fragment_classifier")
    errors.extend(normalization_errors(fragment, "fragment_classifier"))

    number_between(fusion, "detector_override_confidence", 0.0, 1.0, errors, "fusion")
    number_between(fusion, "nms_iou", 0.0, 1.0, errors, "fusion")
    number_between(fusion, "reject_fragment_disagreement_min_conf", 0.0, 1.0, errors, "fusion")
    number_between(fusion, "reject_fragment_class_min_conf", 0.0, 1.0, errors, "fusion")
    nms_score = str(fusion.get("nms_score", "detector_conf")).strip()
    if nms_score not in VALID_NMS_SCORES:
        errors.append(f"fusion.nms_score: expected one of {sorted(VALID_NMS_SCORES)}, got {nms_score!r}")
    for key in ["fragment_override_enabled", "reject_after_nms", "reject_fragment_disagreement"]:
        if key in fusion and not isinstance(fusion[key], bool):
            errors.append(f"fusion.{key}: expected boolean")
    reject_classes = fusion.get("reject_fragment_classes", [])
    if reject_classes:
        if not isinstance(reject_classes, list):
            errors.append("fusion.reject_fragment_classes: expected list")
        else:
            unknown = sorted({str(item).strip() for item in reject_classes} - set(fragment_classes))
            if unknown:
                errors.append(f"fusion.reject_fragment_classes unknown fragment class(es): {unknown}")
    if detector_classes and fragment_classes:
        print(f"detector_class_sample={detector_classes[:3]}")
        print(f"fragment_class_sample={fragment_classes[:3]}")
    return errors


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    allowed_statuses = set(args.allow_status or DEFAULT_ALLOWED_STATUSES)
    errors: list[str] = []
    if not isinstance(config, dict):
        raise SystemExit(f"{args.config}: expected JSON object")
    print(f"config: {display_path(config_path)}")
    errors.extend(validate_stack_config(config, allowed_statuses))
    total_mb = 0.0
    for label, section in [
        ("detector", config.get("detector", {})),
        ("fragment_classifier", config.get("fragment_classifier", {})),
    ]:
        if isinstance(section, dict):
            size_mb, artifact_errors = check_artifact(label, str(section.get("path", "")))
        else:
            size_mb, artifact_errors = 0.0, [f"{label}: expected object"]
        total_mb += size_mb
        errors.extend(artifact_errors)
    print(f"detector_classes={len(config.get('detector', {}).get('classes', []))}")
    print(f"fragment_classifier_classes={len(config.get('fragment_classifier', {}).get('classes', []))}")
    print(f"total_artifacts={total_mb:.2f} MB")
    if total_mb > args.max_total_mb:
        errors.append(f"total artifact size {total_mb:.2f} MB exceeds {args.max_total_mb:.2f} MB")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("browser stack artifact check passed")


if __name__ == "__main__":
    main()
