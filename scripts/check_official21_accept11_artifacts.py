from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[1]

SUMMARY_JSON = REPO / "runs/cashsnap/official21_missing_schema_seed_accept11_v1/summary.json"
MATERIALIZED_ROOT = REPO / "runs/cashsnap/official21_missing_schema_seed_accept11_v1/materialized"
REGISTRY_JSON = REPO / "configs/synthetic_recipes/cashsnap_data_lifecycle_registry_v1.json"

CANDIDATE_CONFIG = (
    REPO
    / "configs/official21/"
    "cashsnap_official21_roboflow_plus_current_accept6_cap180_empty360_plus_accept11_usd2_khr200_v1.yaml"
)
ROWCOUNT_CONTROL_CONFIG = (
    REPO
    / "configs/official21/"
    "cashsnap_official21_roboflow_plus_current_accept6_cap180_empty360_plus_accept11_usd2_khr200_rowcountctrl_v1.yaml"
)

EXPECTED_SUMMARY_COUNTS = {"USD_2": 4, "KHR_100": 6, "KHR_200": 1}
EXPECTED_MATERIALIZED_CLASS_COUNTS = {1: 4, 8: 6, 9: 1}
EXPECTED_CANDIDATE_VS_CONTROL_DELTA = {1: 4, 9: 1}
EXPECTED_TRAIN_ROWS = 4370
REGISTRY_ID = "cashsnap_official21_missing_schema_seed_accept11_review_bridge_v1"


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def load_yaml(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        fail(errors, f"missing YAML: {path.relative_to(REPO)}")
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail(errors, f"YAML is not a mapping: {path.relative_to(REPO)}")
        return {}
    return data


def resolve_train_list(config_path: Path, config: dict[str, Any], errors: list[str]) -> Path:
    train = config.get("train")
    if not isinstance(train, str) or not train:
        fail(errors, f"{config_path.relative_to(REPO)} has no string train path")
        return REPO / "__missing_train_list__"
    direct = REPO / train
    if direct.exists():
        return direct
    local = config_path.parent / train
    if local.exists():
        return local
    fail(errors, f"missing train list for {config_path.relative_to(REPO)}: {train}")
    return direct


def label_path_for_image(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    if "images" not in parts:
        return None
    index = parts.index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def train_list_stats(config_path: Path, errors: list[str]) -> tuple[int, Counter[int], int]:
    config = load_yaml(config_path, errors)
    train_list = resolve_train_list(config_path, config, errors)
    if not train_list.exists():
        return 0, Counter(), 0

    rows = [line.strip() for line in train_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    class_counts: Counter[int] = Counter()
    empty_labels = 0
    missing_images: list[str] = []
    missing_labels: list[str] = []
    bad_rows: list[str] = []

    for row in rows:
        image_path = REPO / row
        if not image_path.exists():
            missing_images.append(row)
            continue
        label_path = label_path_for_image(image_path)
        if label_path is None:
            bad_rows.append(row)
            continue
        if not label_path.exists():
            missing_labels.append(str(label_path.relative_to(REPO)))
            continue
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            empty_labels += 1
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            parts = line.split()
            if len(parts) != 5:
                fail(errors, f"bad YOLO row {label_path.relative_to(REPO)}:{line_no}: {line}")
                continue
            try:
                class_counts[int(parts[0])] += 1
            except ValueError:
                fail(errors, f"non-integer class id {label_path.relative_to(REPO)}:{line_no}: {parts[0]}")

    if missing_images:
        fail(errors, f"{config_path.relative_to(REPO)} missing images: {missing_images[:3]}")
    if missing_labels:
        fail(errors, f"{config_path.relative_to(REPO)} missing labels: {missing_labels[:3]}")
    if bad_rows:
        fail(errors, f"{config_path.relative_to(REPO)} rows without images segment: {bad_rows[:3]}")

    return len(rows), class_counts, empty_labels


def check_summary(errors: list[str]) -> None:
    if not SUMMARY_JSON.exists():
        fail(errors, f"missing summary: {SUMMARY_JSON.relative_to(REPO)}")
        return
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    counts = summary.get("class_counts")
    if counts != EXPECTED_SUMMARY_COUNTS:
        fail(errors, f"unexpected summary class_counts: {counts}")
    if summary.get("accepted_rows") != 11:
        fail(errors, f"unexpected accepted_rows: {summary.get('accepted_rows')}")
    if summary.get("unique_images") != 11:
        fail(errors, f"unexpected unique_images: {summary.get('unique_images')}")


def check_materialized(errors: list[str]) -> Counter[int]:
    label_dir = MATERIALIZED_ROOT / "labels/train"
    if not label_dir.exists():
        fail(errors, f"missing materialized labels: {label_dir.relative_to(REPO)}")
        return Counter()

    counts: Counter[int] = Counter()
    label_files = sorted(label_dir.glob("*.txt"))
    for label_file in label_files:
        text = label_file.read_text(encoding="utf-8").strip()
        if not text:
            fail(errors, f"empty materialized label: {label_file.relative_to(REPO)}")
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            parts = line.split()
            if len(parts) != 5:
                fail(errors, f"bad materialized YOLO row {label_file.relative_to(REPO)}:{line_no}: {line}")
                continue
            try:
                counts[int(parts[0])] += 1
            except ValueError:
                fail(errors, f"bad materialized class id {label_file.relative_to(REPO)}:{line_no}: {parts[0]}")

    if counts != EXPECTED_MATERIALIZED_CLASS_COUNTS:
        fail(errors, f"unexpected materialized class counts: {dict(counts)}")
    if len(label_files) != 11:
        fail(errors, f"unexpected materialized label file count: {len(label_files)}")
    return counts


def check_registry(errors: list[str]) -> None:
    if not REGISTRY_JSON.exists():
        fail(errors, f"missing registry: {REGISTRY_JSON.relative_to(REPO)}")
        return
    registry = json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    entries = registry.get("entries", [])
    match = next((entry for entry in entries if entry.get("id") == REGISTRY_ID), None)
    if not match:
        fail(errors, f"registry entry not found: {REGISTRY_ID}")
        return
    expected_path = "runs/cashsnap/official21_missing_schema_seed_accept11_v1/materialized"
    if match.get("path") != expected_path:
        fail(errors, f"registry path mismatch for {REGISTRY_ID}: {match.get('path')}")
    if match.get("state") != "diagnostic":
        fail(errors, f"registry state mismatch for {REGISTRY_ID}: {match.get('state')}")


def check_candidate_vs_control(errors: list[str]) -> None:
    candidate_rows, candidate_counts, candidate_empty = train_list_stats(CANDIDATE_CONFIG, errors)
    control_rows, control_counts, control_empty = train_list_stats(ROWCOUNT_CONTROL_CONFIG, errors)

    if candidate_rows != EXPECTED_TRAIN_ROWS:
        fail(errors, f"candidate row count mismatch: {candidate_rows}")
    if control_rows != EXPECTED_TRAIN_ROWS:
        fail(errors, f"row-count control row count mismatch: {control_rows}")

    delta = Counter(candidate_counts)
    delta.subtract(control_counts)
    nonzero_delta = {class_id: count for class_id, count in sorted(delta.items()) if count}
    if nonzero_delta != EXPECTED_CANDIDATE_VS_CONTROL_DELTA:
        fail(errors, f"unexpected candidate/control class delta: {nonzero_delta}")

    print(
        "candidate/control:",
        f"rows={candidate_rows}/{control_rows}",
        f"empty_labels={candidate_empty}/{control_empty}",
        f"delta={nonzero_delta}",
    )


def main() -> int:
    errors: list[str] = []
    check_summary(errors)
    materialized_counts = check_materialized(errors)
    check_registry(errors)
    check_candidate_vs_control(errors)

    if errors:
        print("official21 accept11 artifact check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "official21 accept11 artifact check passed:",
        f"materialized_counts={dict(sorted(materialized_counts.items()))}",
        f"summary_counts={EXPECTED_SUMMARY_COUNTS}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
