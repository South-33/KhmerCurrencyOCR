#!/usr/bin/env python
"""Summarize CashSnap synthetic-data quality against the project rubric.

This is a thin scorecard over existing gates. It does not replace readiness,
domain-gap, label-view, or model comparison checks; it makes their state visible
through the dataset-quality axes from the local research PDF.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from check_synthetic_governance import DEFAULT_MANIFEST as DEFAULT_GOVERNANCE
from check_synthetic_governance import check_manifest as check_governance_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READINESS = ROOT / "runs" / "cashsnap" / "synthetic_pipeline_readiness_latest.json"
DEFAULT_DOMAIN_GAP = ROOT / "runs" / "cashsnap" / "domain_gap_accepted_nowarmup_train.json"
DEFAULT_GEOMETRY_DOMAIN_GAP = ROOT / "runs" / "cashsnap" / "domain_gap_accepted_nowarmup_train_geometry.json"
DEFAULT_MINED_REVIEW = ROOT / "runs" / "cashsnap" / "mined_real_benchmark_review_latest.json"
DEFAULT_MINED_REVIEW_QUALITY = ROOT / "runs" / "cashsnap" / "mined_real_benchmark_review_quality_summary_latest.json"
DEFAULT_SPLIT_COVERAGE = ROOT / "runs" / "cashsnap" / "cashsnap_v1_split_coverage_latest.json"
DEFAULT_MINED_REAL_UTILITY_COMPARISONS = [
    ROOT / "runs" / "cashsnap" / "mined_real_holdout_scoreboard_accepted_vs_p24_seed0_i416_present_classes.json",
    ROOT / "runs" / "cashsnap" / "mined_real_holdout_scoreboard_accepted_vs_p24_seed1_i416_present_classes.json",
]
DEFAULT_BROWSER_SYNTHETIC_STRESS = ROOT / "runs" / "cashsnap" / "browser_synthetic_stress_cases_v1.json"
DEFAULT_BROWSER_SYNTHETIC_MANIFEST = ROOT / "manifests" / "browser_synthetic_stress_cases.csv"
DEFAULT_BROWSER_MINED_REAL_STRESS = ROOT / "runs" / "cashsnap" / "browser_mined_real_scoreable_default_latest.json"
DEFAULT_BROWSER_MINED_REAL_MANIFEST = ROOT / "runs" / "cashsnap" / "mined_real_browser_cases_latest.csv"
DEFAULT_TAXONOMY_DATA = ROOT / "data" / "cashsnap_v1" / "data.yaml"
DEFAULT_JSON_OUT = ROOT / "runs" / "cashsnap" / "synthetic_dataset_scorecard_latest.json"

STATUS_ORDER = {"pass": 0, "review": 1, "missing": 2, "blocked": 3}
RUBRIC_SOURCE = "docs/research/What Makes a Dataset Perfect for Synthetic Data Pipelines.pdf"
RUBRIC_SOURCES = [
    RUBRIC_SOURCE,
    "docs/research/CashSnap_Fact_Checked_Dataset_Strategy.pdf",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
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
CLASS_VALUES = {
    "USD_1": ("usd", 1),
    "USD_5": ("usd", 5),
    "USD_10": ("usd", 10),
    "USD_20": ("usd", 20),
    "USD_50": ("usd", 50),
    "USD_100": ("usd", 100),
    "KHR_500": ("khr", 500),
    "KHR_1000": ("khr", 1000),
    "KHR_2000": ("khr", 2000),
    "KHR_5000": ("khr", 5000),
    "KHR_10000": ("khr", 10000),
    "KHR_20000": ("khr", 20000),
    "KHR_50000": ("khr", 50000),
}
OFFICIAL_CURRENT_CLASS_NAMES = [
    "USD_1",
    "USD_2",
    "USD_5",
    "USD_10",
    "USD_20",
    "USD_50",
    "USD_100",
    "KHR_50",
    "KHR_100",
    "KHR_200",
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_15000",
    "KHR_20000",
    "KHR_30000",
    "KHR_50000",
    "KHR_100000",
    "KHR_200000",
]
OFFICIAL_TAXONOMY_SOURCES = {
    "USD": "https://www.uscurrency.gov/denominations",
    "KHR": "https://www.nbc.gov.kh/english/about_the_bank/banknotes_in_circulation.php",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--domain-gap", type=Path, default=DEFAULT_DOMAIN_GAP)
    parser.add_argument("--geometry-domain-gap", type=Path, default=DEFAULT_GEOMETRY_DOMAIN_GAP)
    parser.add_argument("--mined-review", type=Path, default=DEFAULT_MINED_REVIEW)
    parser.add_argument("--mined-review-quality", type=Path, default=DEFAULT_MINED_REVIEW_QUALITY)
    parser.add_argument("--split-coverage", type=Path, default=DEFAULT_SPLIT_COVERAGE)
    parser.add_argument("--governance", type=Path, default=DEFAULT_GOVERNANCE)
    parser.add_argument(
        "--min-real-train-class-images",
        type=int,
        default=48,
        help="Minimum unique clean-real train images per class for the split-coverage scorecard axis.",
    )
    parser.add_argument(
        "--mined-real-utility-comparison",
        type=Path,
        action="append",
        default=[],
        help="Optional compare_yolo_metrics.py JSON for the mined held-out diagnostic utility axis.",
    )
    parser.add_argument(
        "--no-default-mined-real-utility",
        action="store_true",
        help="Do not load the default accepted-WebGL-vs-p24 mined held-out comparisons.",
    )
    parser.add_argument("--browser-synthetic-manifest", type=Path, default=DEFAULT_BROWSER_SYNTHETIC_MANIFEST)
    parser.add_argument("--browser-synthetic-stress", type=Path, default=DEFAULT_BROWSER_SYNTHETIC_STRESS)
    parser.add_argument("--browser-mined-real-manifest", type=Path, default=DEFAULT_BROWSER_MINED_REAL_MANIFEST)
    parser.add_argument("--browser-mined-real-stress", type=Path, default=DEFAULT_BROWSER_MINED_REAL_STRESS)
    parser.add_argument(
        "--taxonomy-data",
        type=Path,
        default=DEFAULT_TAXONOMY_DATA,
        help="YOLO data YAML whose class names define the current model taxonomy scope.",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any axis is blocked or missing.")
    parser.add_argument("--require-pass", action="store_true", help="Exit non-zero unless every scorecard axis passes.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return resolve(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    resolved = resolve(path)
    if not resolved.exists():
        if required:
            raise SystemExit(f"missing JSON file: {repo_path(resolved)}")
        return {}
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_path(resolved)}: expected JSON object")
    return data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_listing_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(child for child in path.rglob("*") if child.is_file()):
        rel = repo_path(item)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\n")
        digest.update(file_sha256(item).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_comparison_jsons(paths: list[Path]) -> list[dict[str, Any]]:
    comparisons = []
    for path in paths:
        resolved = resolve(path)
        if not resolved.exists():
            comparisons.append({"_source": repo_path(resolved), "_missing": True})
            continue
        data = read_json(resolved)
        data["_source"] = repo_path(resolved)
        comparisons.append(data)
    return comparisons


def read_browser_stress_report(path: Path, *, required: bool = True) -> dict[str, Any]:
    resolved = resolve(path)
    if not resolved.exists():
        if required:
            raise SystemExit(f"missing JSON file: {repo_path(resolved)}")
        return {"cases": [], "_source": repo_path(resolved)}
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {
            "schema": "legacy_browser_synthetic_stress_array",
            "legacy_array": True,
            "cases": data,
            "_source": repo_path(resolved),
        }
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_path(resolved)}: expected JSON object or legacy JSON array")
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise SystemExit(f"{repo_path(resolved)}: expected 'cases' JSON array")
    data["_source"] = repo_path(resolved)
    return data


def read_csv_rows(path: Path, *, required: bool = True) -> list[dict[str, str]]:
    resolved = resolve(path)
    if not resolved.exists():
        if required:
            raise SystemExit(f"missing CSV file: {repo_path(resolved)}")
        return []
    with resolved.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def yolo_label_count_values(path: Path) -> dict[str, int]:
    resolved = resolve(path)
    count = 0
    khr_value = 0
    usd_value = 0
    for line_number, raw_line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_path(resolved)}:{line_number}: expected YOLO detect label with 5 fields")
        raw_class = float(parts[0])
        if not raw_class.is_integer():
            raise SystemExit(f"{repo_path(resolved)}:{line_number}: class id {parts[0]} is not an integer")
        class_id = int(raw_class)
        if not 0 <= class_id < len(CLASS_NAMES):
            raise SystemExit(f"{repo_path(resolved)}:{line_number}: class id {class_id} outside class map")
        class_name = CLASS_NAMES[class_id]
        currency, value = CLASS_VALUES[class_name]
        count += 1
        if currency == "khr":
            khr_value += value
        else:
            usd_value += value
    return {"count": count, "khr_value": khr_value, "usd_value": usd_value}


def verify_fingerprint_row(row: dict[str, Any], *, label: str, required: bool = True) -> list[str]:
    failures: list[str] = []
    if not isinstance(row, dict):
        return [f"{label} fingerprint row is malformed"]
    path_text = str(row.get("path", "")).strip()
    if not path_text:
        return [f"{label} fingerprint row is missing path"]
    exists = bool(row.get("exists"))
    path = resolve(Path(path_text))
    if required and not exists:
        failures.append(f"{label} was missing when the report was generated: {path_text}")
    if not path.exists():
        if exists or required:
            failures.append(f"{label} path is missing: {path_text}")
        return failures
    if path.is_dir():
        expected_listing = str(row.get("listing_sha256", "")).strip()
        if not expected_listing:
            failures.append(f"{label} is missing listing_sha256; regenerate the report with the current script")
            return failures
        actual_listing = directory_listing_sha256(path)
        if actual_listing != expected_listing:
            failures.append(f"{label} directory fingerprint is stale: {path_text}")
        return failures
    if not path.is_file():
        failures.append(f"{label} path is not a file: {path_text}")
        return failures
    expected = str(row.get("sha256", "")).strip()
    if not expected:
        failures.append(f"{label} is missing sha256; regenerate the report with the current script")
        return failures
    actual = file_sha256(path)
    if actual != expected:
        failures.append(f"{label} fingerprint is stale: {path_text}")
    return failures


def read_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_path(resolved)}: expected YAML object")
    return data


def dataset_root(data_yaml: dict[str, Any], data_path: Path) -> Path:
    raw = data_yaml.get("path")
    if raw:
        path = Path(str(raw))
        return path if path.is_absolute() else resolve(data_path).parent / path
    return resolve(data_path).parent


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


def current_real_candidate_split_fingerprints(data_path: Path, splits: list[str]) -> dict[str, dict[str, Any]]:
    data_yaml = read_yaml(data_path)
    root = dataset_root(data_yaml, data_path)
    fingerprints: dict[str, dict[str, Any]] = {}
    for split in splits:
        image_paths = image_paths_from_value(root, data_yaml.get(split))
        image_digest = hashlib.sha256()
        label_digest = hashlib.sha256()
        missing_labels = 0
        for image_path in sorted(image_paths, key=repo_path):
            image_text = repo_path(image_path)
            label_path = label_path_for_image(root, image_path)
            label_text = repo_path(label_path)
            image_digest.update(image_text.encode("utf-8"))
            image_digest.update(b"\n")
            label_digest.update(label_text.encode("utf-8"))
            label_digest.update(b"\0")
            if label_path.exists() and label_path.is_file():
                label_digest.update(file_sha256(label_path).encode("utf-8"))
            else:
                label_digest.update(b"MISSING")
                missing_labels += 1
            label_digest.update(b"\n")
        fingerprints[str(split)] = {
            "image_count": len(image_paths),
            "image_listing_sha256": image_digest.hexdigest(),
            "label_content_sha256": label_digest.hexdigest(),
            "missing_label_count": missing_labels,
        }
    return fingerprints


def real_dataset_candidate_summary_freshness_failures(summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not str(summary.get("generated_at_utc", "")).strip():
        failures.append("real dataset candidate summary is missing generated_at_utc; regenerate it with the current miner")
    data = str(summary.get("data", "")).strip()
    expected_data_sha = str(summary.get("data_config_sha256", "")).strip()
    if not data:
        failures.append("real dataset candidate summary is missing data config path")
        return failures
    data_path = resolve(Path(data))
    if not data_path.exists():
        failures.append(f"real dataset candidate data config is missing: {data}")
        return failures
    if not expected_data_sha:
        failures.append("real dataset candidate summary is missing data_config_sha256; regenerate it with the current miner")
    elif file_sha256(data_path) != expected_data_sha:
        failures.append(f"real dataset candidate data config fingerprint is stale: {data}")

    splits = summary.get("splits", [])
    if not isinstance(splits, list) or not splits:
        failures.append("real dataset candidate summary is missing scanned splits")
        return failures
    expected_fingerprints = summary.get("split_fingerprints", {})
    if not isinstance(expected_fingerprints, dict) or not expected_fingerprints:
        failures.append("real dataset candidate summary is missing split_fingerprints; regenerate it with the current miner")
        return failures
    current_fingerprints = current_real_candidate_split_fingerprints(data_path, [str(split) for split in splits])
    for split in [str(item) for item in splits]:
        expected = expected_fingerprints.get(split)
        current = current_fingerprints.get(split)
        if not isinstance(expected, dict) or not isinstance(current, dict):
            failures.append(f"real dataset candidate split fingerprint is malformed: {split}")
            continue
        for key in ["image_count", "image_listing_sha256", "label_content_sha256", "missing_label_count"]:
            if expected.get(key) != current.get(key):
                failures.append(f"real dataset candidate split {split} {key} fingerprint is stale")
    return failures


def readiness_freshness_failures(readiness: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not str(readiness.get("generated_at_utc", "")).strip():
        failures.append("readiness report is missing generated_at_utc; regenerate it with the current script")
    input_fingerprints = readiness.get("input_fingerprints", {})
    if not isinstance(input_fingerprints, dict) or not input_fingerprints:
        return failures + ["readiness report is missing input_fingerprints; regenerate it with the current script"]

    required_inputs = [
        "targets",
        "catalog",
        "suite",
        "sources",
        "quality",
        "capture_inventory",
        "capture_requirements",
    ]
    for key in required_inputs:
        failures.extend(verify_fingerprint_row(input_fingerprints.get(key, {}), label=f"readiness input {key}"))
    real_dataset_candidates = readiness.get("real_dataset_candidates", {})
    real_candidates_loaded = isinstance(real_dataset_candidates, dict) and bool(real_dataset_candidates.get("loaded"))
    failures.extend(
        verify_fingerprint_row(
            input_fingerprints.get("real_dataset_candidates", {}),
            label="readiness input real_dataset_candidates",
            required=real_candidates_loaded,
        )
    )
    if real_candidates_loaded and isinstance(real_dataset_candidates, dict):
        failures.extend(real_dataset_candidate_summary_freshness_failures(real_dataset_candidates))

    if bool(readiness.get("check_existing")):
        reports = readiness.get("suite_package_reports", {})
        if not isinstance(reports, dict):
            failures.append("readiness suite_package_reports is malformed")
        else:
            for recipe_id, report in sorted(reports.items()):
                if not isinstance(report, dict):
                    failures.append(f"readiness package report is malformed: {recipe_id}")
                    continue
                if not report.get("exists"):
                    continue
                fingerprints = report.get("fingerprints", {})
                if not isinstance(fingerprints, dict) or not fingerprints:
                    failures.append(f"{recipe_id}: package report is missing fingerprints; regenerate readiness")
                    continue
                for key in ["recipe_json", "qa_summary", "data_yaml", "manifest"]:
                    failures.extend(
                        verify_fingerprint_row(
                            fingerprints.get(key, {}),
                            label=f"readiness package {recipe_id} {key}",
                        )
                    )
    return failures


def readiness_freshness_axis(readiness: dict[str, Any]) -> dict[str, Any]:
    failures = readiness_freshness_failures(readiness)
    status = "pass" if not failures else "blocked"
    input_fingerprints = readiness.get("input_fingerprints", {})
    if not isinstance(input_fingerprints, dict):
        input_fingerprints = {}
    package_reports = readiness.get("suite_package_reports", {})
    if not isinstance(package_reports, dict):
        package_reports = {}
    return axis(
        "readiness_freshness",
        status,
        "Readiness inputs and checked package metadata are fresh."
        if status == "pass"
        else "Readiness input/package fingerprints are stale or missing.",
        evidence={
            "generated_at_utc": readiness.get("generated_at_utc", ""),
            "input_fingerprints": input_fingerprints,
            "check_existing": bool(readiness.get("check_existing")),
            "package_report_count": len(package_reports),
        },
        blockers=failures,
        next_action="" if status == "pass" else "Regenerate synthetic pipeline readiness with the current script and --check-existing.",
    )


def mined_review_freshness_failures(mined_review: dict[str, Any]) -> list[str]:
    if not mined_review:
        return []
    failures: list[str] = []
    if not str(mined_review.get("generated_at_utc", "")).strip():
        failures.append("mined review summary is missing generated_at_utc; regenerate it with the current script")
    input_fingerprints = mined_review.get("input_fingerprints", {})
    if not isinstance(input_fingerprints, dict) or not input_fingerprints:
        failures.append("mined review summary is missing input_fingerprints; regenerate it with the current script")
    else:
        failures.extend(verify_fingerprint_row(input_fingerprints.get("review_csv", {}), label="mined review input review_csv"))
    output_fingerprints = mined_review.get("output_fingerprints", {})
    if not isinstance(output_fingerprints, dict) or not output_fingerprints:
        failures.append("mined review summary is missing output_fingerprints; regenerate it with the current script")
    else:
        for key in ["sources_out", "tasks_out", "quality_template_out", "review_index", "draft_label_dir"]:
            failures.extend(verify_fingerprint_row(output_fingerprints.get(key, {}), label=f"mined review output {key}"))
    return failures


def mined_review_quality_freshness_failures(mined_review_quality: dict[str, Any]) -> list[str]:
    if not mined_review_quality:
        return []
    failures: list[str] = []
    if not str(mined_review_quality.get("generated_at_utc", "")).strip():
        failures.append("mined review quality summary is missing generated_at_utc; regenerate it with the current script")
    input_fingerprints = mined_review_quality.get("input_fingerprints", {})
    if not isinstance(input_fingerprints, dict) or not input_fingerprints:
        failures.append("mined review quality summary is missing input_fingerprints; regenerate it with the current script")
    else:
        for key in ["sources", "quality", "draft_label_dir"]:
            failures.extend(
                verify_fingerprint_row(
                    input_fingerprints.get(key, {}),
                    label=f"mined review quality input {key}",
                )
            )
    return failures


def axis(
    name: str,
    status: str,
    summary: str,
    *,
    evidence: dict[str, Any] | None = None,
    blockers: list[str] | None = None,
    next_action: str = "",
) -> dict[str, Any]:
    if status not in STATUS_ORDER:
        raise ValueError(f"unknown scorecard status {status!r}")
    return {
        "name": name,
        "status": status,
        "summary": summary,
        "evidence": evidence or {},
        "blockers": blockers or [],
        "next_action": next_action,
    }


def conditions_by_id(readiness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = readiness.get("conditions", [])
    if not isinstance(rows, list):
        return {}
    return {str(row.get("condition_id", "")): row for row in rows if isinstance(row, dict)}


def condition_blockers(readiness: dict[str, Any], condition_id: str) -> list[str]:
    condition = conditions_by_id(readiness).get(condition_id, {})
    blockers = condition.get("blockers", [])
    if not isinstance(blockers, list):
        return []
    return [str(item) for item in blockers]


def condition_axis(
    readiness: dict[str, Any],
    *,
    name: str,
    condition_id: str,
    label: str,
    next_action: str,
) -> dict[str, Any]:
    condition = conditions_by_id(readiness).get(condition_id)
    if not condition:
        return axis(
            name,
            "missing",
            f"{label} condition is missing from readiness.",
            blockers=[f"missing readiness condition {condition_id}"],
            next_action=next_action,
        )
    blockers = condition.get("blockers", [])
    if not isinstance(blockers, list):
        blockers = ["condition blockers are malformed"]
    status = "pass" if not blockers else "blocked"
    state = str(condition.get("state", ""))
    summary = f"{label} is ready." if status == "pass" else f"{label} is blocked ({state or 'unknown state'})."
    return axis(
        name,
        status,
        summary,
        evidence={
            "condition_id": condition_id,
            "state": state,
            "priority": condition.get("priority", ""),
            "target_status": condition.get("target_status", ""),
            "catalog_recipe_ids": condition.get("catalog_recipe_ids", []),
            "active_suite_recipe_ids": condition.get("active_suite_recipe_ids", []),
            "capture_requirements": condition.get("capture_requirements", []),
            "real_dataset_review_candidates": condition.get("real_dataset_review_candidates", []),
        },
        blockers=[str(item) for item in blockers],
        next_action=next_action,
    )


def package_blockers(readiness: dict[str, Any]) -> list[str]:
    reports = readiness.get("suite_package_reports", {})
    if not isinstance(reports, dict):
        return ["suite_package_reports missing or malformed"]
    blockers: list[str] = []
    for recipe_id, report in sorted(reports.items()):
        if not isinstance(report, dict):
            blockers.append(f"{recipe_id}: malformed package report")
            continue
        for item in report.get("blockers", []):
            blockers.append(f"{recipe_id}: {item}")
    return blockers


def candidate_totals(readiness: dict[str, Any]) -> dict[str, int]:
    candidates = readiness.get("real_dataset_candidates", {})
    if not isinstance(candidates, dict):
        return {}
    counts = candidates.get("scene_unique_origin_counts", {})
    if not isinstance(counts, dict):
        return {}
    return {str(key): int(value or 0) for key, value in counts.items()}


def candidate_report_has_hits(row: dict[str, Any]) -> bool:
    return int(row.get("candidate_count", 0) or 0) > 0 or int(row.get("unique_origin_count", 0) or 0) > 0


def required_candidate_inventory(readiness: dict[str, Any]) -> dict[str, Any]:
    mapped_condition_ids: list[str] = []
    hit_condition_ids: list[str] = []
    missing_scene_hints: list[dict[str, Any]] = []
    for condition_id, condition in sorted(conditions_by_id(readiness).items()):
        if not bool(condition.get("required_for_v1")):
            continue
        rows = condition.get("real_dataset_review_candidates", [])
        if not isinstance(rows, list) or not rows:
            continue
        mapped_condition_ids.append(condition_id)
        has_any_hit = False
        for row in rows:
            if not isinstance(row, dict):
                continue
            if candidate_report_has_hits(row):
                has_any_hit = True
            else:
                missing_scene_hints.append(
                    {
                        "condition_id": condition_id,
                        "scene_type": row.get("scene_type", ""),
                        "candidate_count": int(row.get("candidate_count", 0) or 0),
                        "unique_origin_count": int(row.get("unique_origin_count", 0) or 0),
                    }
                )
        if has_any_hit:
            hit_condition_ids.append(condition_id)
    return {
        "mapped_condition_ids": mapped_condition_ids,
        "hit_condition_ids": hit_condition_ids,
        "missing_scene_hints": missing_scene_hints,
    }


def domain_gap_freshness_failures(domain_gap: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    data = str(domain_gap.get("data", "")).strip()
    expected_data_sha = str(domain_gap.get("data_config_sha256", "")).strip()
    if not data:
        failures.append("domain-gap report is missing data config path")
    elif not expected_data_sha:
        failures.append("domain-gap report is missing data_config_sha256; regenerate it with the current audit script")
    else:
        data_path = resolve(Path(data))
        if not data_path.exists():
            failures.append(f"domain-gap data config is missing: {data}")
        else:
            actual_data_sha = file_sha256(data_path)
            if actual_data_sha != expected_data_sha:
                failures.append(f"domain-gap data config fingerprint is stale: {data}")

    split_sources = domain_gap.get("split_sources", [])
    if not isinstance(split_sources, list) or not split_sources:
        failures.append("domain-gap report is missing split source fingerprints; regenerate it with the current audit script")
        return failures

    for source in split_sources:
        if not isinstance(source, dict):
            failures.append("domain-gap split source fingerprint row is malformed")
            continue
        path_text = str(source.get("path", "")).strip()
        if not path_text:
            failures.append("domain-gap split source fingerprint row is missing path")
            continue
        path = resolve(Path(path_text))
        if not path.exists():
            failures.append(f"domain-gap split source is missing: {path_text}")
            continue
        kind = str(source.get("kind", "")).strip()
        if kind == "list":
            expected = str(source.get("sha256", "")).strip()
            if not expected:
                failures.append(f"domain-gap split list is missing sha256: {path_text}")
                continue
            actual = file_sha256(path)
            if actual != expected:
                failures.append(f"domain-gap split list fingerprint is stale: {path_text}")
        elif kind == "directory":
            expected = str(source.get("listing_sha256", "")).strip()
            if not expected:
                failures.append(f"domain-gap split directory is missing listing_sha256: {path_text}")
                continue
            image_rows = sorted(
                repo_path(item)
                for item in path.glob("*")
                if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )
            digest = hashlib.sha256()
            for row in image_rows:
                digest.update(row.encode("utf-8"))
                digest.update(b"\n")
            if digest.hexdigest() != expected:
                failures.append(f"domain-gap split directory listing fingerprint is stale: {path_text}")
        else:
            failures.append(f"domain-gap split source has unknown kind {kind!r}: {path_text}")
    return failures


def domain_gap_axis(
    domain_gap: dict[str, Any],
    *,
    name: str = "fidelity_domain_gap",
    label: str = "Accepted-blend domain-gap",
    expected_preset: str = "accepted_blend_v1",
    blocked_next_action: str = "Repair synthetic/real dose or distribution drift before model spend.",
) -> dict[str, Any]:
    if not domain_gap:
        return axis(
            name,
            "missing",
            f"No {label.lower()} report was supplied.",
            next_action=f"Run audit_yolo_domain_gap.py with the {expected_preset} preset.",
        )
    gate = domain_gap.get("domain_gap_gate", {})
    if not isinstance(gate, dict) or not gate.get("requested"):
        return axis(
            name,
            "review",
            f"{label} report exists but no gate was requested.",
            evidence={"data": domain_gap.get("data", ""), "split": domain_gap.get("split", "")},
            next_action=f"Regenerate the report with --gate-preset {expected_preset} --fail-on-gap.",
        )
    failures = [str(item) for item in gate.get("failures", [])]
    freshness_failures = domain_gap_freshness_failures(domain_gap)
    blockers = [*failures, *freshness_failures]
    status = "pass" if gate.get("passed") and not freshness_failures else "blocked"
    if status == "pass":
        summary = f"{label} gate passed."
    elif freshness_failures and gate.get("passed"):
        summary = f"{label} report is stale or missing provenance."
    else:
        summary = f"{label} gate failed."
    return axis(
        name,
        status,
        summary,
        evidence={
            "data": domain_gap.get("data", ""),
            "split": domain_gap.get("split", ""),
            "data_config_sha256": domain_gap.get("data_config_sha256", ""),
            "split_sources": domain_gap.get("split_sources", []),
            "observed": gate.get("observed", {}),
            "limits": gate.get("limits", {}),
        },
        blockers=blockers,
        next_action="" if status == "pass" else blocked_next_action,
    )


def comparison_freshness_failures(comparison: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for key, label in (("baseline", "baseline"), ("candidate", "candidate")):
        path_text = str(comparison.get(f"{key}_path", "")).strip()
        expected = str(comparison.get(f"{key}_sha256", "")).strip()
        if not path_text:
            failures.append(f"comparison report is missing {label}_path")
            continue
        if not expected:
            failures.append(f"comparison report is missing {label}_sha256; regenerate it with the current compare_yolo_metrics.py")
            continue
        path = resolve(Path(path_text))
        if not path.exists():
            failures.append(f"comparison {label} metrics file is missing: {path_text}")
            continue
        actual = file_sha256(path)
        if actual != expected:
            failures.append(f"comparison {label} metrics fingerprint is stale: {path_text}")

    summary_path = str(comparison.get("classes_from_summary_path", "")).strip()
    expected_summary = str(comparison.get("classes_from_summary_sha256", "")).strip()
    if summary_path or expected_summary:
        if not summary_path:
            failures.append("comparison report has classes_from_summary_sha256 but no path")
        elif not expected_summary:
            failures.append("comparison report is missing classes_from_summary_sha256; regenerate it with the current compare_yolo_metrics.py")
        else:
            path = resolve(Path(summary_path))
            if not path.exists():
                failures.append(f"comparison classes summary is missing: {summary_path}")
            else:
                actual = file_sha256(path)
                if actual != expected_summary:
                    failures.append(f"comparison classes summary fingerprint is stale: {summary_path}")
    return failures


def mined_real_utility_axis(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    if not comparisons:
        return axis(
            "diagnostic_real_utility",
            "missing",
            "No mined held-out diagnostic model comparisons were supplied.",
            next_action="Run val_yolo.py on the mined held-out scoreable dataset and compare candidates with compare_yolo_metrics.py --classes-from-summary.",
        )

    evidence_rows = []
    blockers = []
    for comparison in comparisons:
        source = str(comparison.get("_source", ""))
        if comparison.get("_missing"):
            evidence_rows.append({"source": source, "passed": False, "missing": True})
            blockers.append(f"{source}: comparison JSON missing")
            continue
        freshness_failures = comparison_freshness_failures(comparison)
        passed = bool(comparison.get("passed")) and not freshness_failures
        delta = float(comparison.get("delta", 0.0) or 0.0)
        per_class_failures = comparison.get("per_class_failures", [])
        if not isinstance(per_class_failures, list):
            per_class_failures = []
        failed_classes = [
            str(row.get("class_name"))
            for row in per_class_failures
            if isinstance(row, dict) and row.get("class_name") is not None
        ]
        checks = comparison.get("checks", [])
        if not isinstance(checks, list):
            checks = []
        failed_checks = [
            str(check.get("name"))
            for check in checks
            if isinstance(check, dict) and not check.get("passed", False)
        ]
        evidence_rows.append(
            {
                "source": source,
                "passed": passed,
                "baseline_path": comparison.get("baseline_path", ""),
                "baseline_sha256": comparison.get("baseline_sha256", ""),
                "candidate_path": comparison.get("candidate_path", ""),
                "candidate_sha256": comparison.get("candidate_sha256", ""),
                "classes_from_summary_path": comparison.get("classes_from_summary_path", ""),
                "baseline": comparison.get("baseline"),
                "candidate": comparison.get("candidate"),
                "delta": comparison.get("delta"),
                "failed_checks": failed_checks,
                "failed_classes": failed_classes,
                "freshness_failures": freshness_failures,
            }
        )
        if freshness_failures:
            blockers.extend(f"{source or 'comparison'}: {failure}" for failure in freshness_failures)
        if not passed and not freshness_failures:
            reason = f"{source or comparison.get('candidate_path', 'candidate')}: delta {delta:+.6f}"
            if failed_checks:
                reason += f"; failed checks {', '.join(failed_checks)}"
            if failed_classes:
                reason += f"; failed classes {', '.join(failed_classes)}"
            blockers.append(reason)

    pass_count = sum(1 for row in evidence_rows if row["passed"])
    status = "review" if pass_count == len(evidence_rows) else "blocked"
    summary = f"{pass_count}/{len(evidence_rows)} mined held-out diagnostic comparison(s) pass."
    return axis(
        "diagnostic_real_utility",
        status,
        summary,
        evidence={"comparisons": evidence_rows},
        blockers=blockers,
        next_action=(
            "Treat failed mined held-out comparisons as a stop sign for synthetic scale; even passing diagnostic slices still need protected real fan/overlap proof."
        ),
    )


def browser_report_freshness_failures(report: dict[str, Any], manifest_path: Path, *, label: str) -> list[str]:
    failures: list[str] = []
    if report.get("legacy_array"):
        return [f"{label} report is a legacy JSON array; regenerate it with run_browser_smoke_cases.py"]
    if not str(report.get("generated_at_utc", "")).strip():
        failures.append(f"{label} report is missing generated_at_utc; regenerate it with run_browser_smoke_cases.py")
    input_fingerprints = report.get("input_fingerprints", {})
    if not isinstance(input_fingerprints, dict) or not input_fingerprints:
        failures.append(f"{label} report is missing input_fingerprints; regenerate it with run_browser_smoke_cases.py")
        return failures

    expected_manifest = repo_path(resolve(manifest_path))
    manifest_row = input_fingerprints.get("case_manifest", {})
    if isinstance(manifest_row, dict):
        report_manifest = str(manifest_row.get("path", "")).strip()
        if report_manifest and report_manifest != expected_manifest:
            failures.append(
                f"{label} manifest path drifted: report has {report_manifest}, scorecard has {expected_manifest}"
            )
    required_keys = [
        "case_manifest",
        "smoke_runner",
        "smoke_cdp",
        "browser_app",
        "browser_index",
        "browser_stack_config",
        "detector_model",
        "fragment_classifier_model",
    ]
    for key in required_keys:
        failures.extend(verify_fingerprint_row(input_fingerprints.get(key, {}), label=f"{label} input {key}"))
    failures.extend(str(item) for item in report.get("failures", []) if str(item).strip())
    return failures


def browser_synthetic_stress_axis(
    report: dict[str, Any],
    manifest_rows: list[dict[str, str]],
    manifest_path: Path,
    *,
    name: str = "browser_synthetic_stress",
    label: str = "browser synthetic stress",
    missing_next_action: str = "Run the browser synthetic stress manifest through smoke_browser_demo_cdp.cjs before any deploy/count claim.",
    blocked_next_action: str = "Use browser stress failures as deploy/curriculum probes; count-contract pass alone is not enough while count, value, or class recall fails.",
) -> dict[str, Any]:
    rows = report.get("cases", [])
    if not isinstance(rows, list):
        rows = []
    if not rows:
        return axis(
            name,
            "missing",
            f"No {label} JSON was supplied.",
            next_action=missing_next_action,
        )

    required_contract_flags = [
        "uiTotalMatchesFinal",
        "predClassTotalMatchesFinal",
        "debugFinalMatchesFinal",
        "finalNotMoreThanClassified",
        "fragmentClassifiedNotMoreThanClassified",
    ]
    evidence_rows: list[dict[str, Any]] = []
    blockers: list[str] = browser_report_freshness_failures(report, manifest_path, label=label)
    pass_count = 0
    manifest_by_case = {str(row.get("case_id", "")): row for row in manifest_rows}
    run_case_ids = {str(item.get("caseId", "")) for item in rows if isinstance(item, dict)}
    if manifest_by_case:
        missing_runs = sorted(set(manifest_by_case) - run_case_ids)
        extra_runs = sorted(run_case_ids - set(manifest_by_case))
        blockers.extend(f"missing {label} run for manifest case {case_id}" for case_id in missing_runs)
        blockers.extend(f"{label} run has extra case not in manifest: {case_id}" for case_id in extra_runs)
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            blockers.append(f"browser stress row {index} is malformed")
            continue
        case_id = str(item.get("caseId") or f"row_{index}")
        manifest_row = manifest_by_case.get(case_id, {})
        contract = item.get("countContract", {})
        if not isinstance(contract, dict):
            contract = {}
        evaluation = item.get("evaluation", {})
        if not isinstance(evaluation, dict):
            evaluation = {}

        contract_failures = [flag for flag in required_contract_flags if not bool(contract.get(flag))]
        if contract.get("mode") != contract.get("expectedMode"):
            contract_failures.append("mode")
        if contract.get("countSource") != contract.get("expectedCountSource"):
            contract_failures.append("countSource")

        gt_count = int(evaluation.get("gtCount", 0) or 0)
        count_error = int(evaluation.get("countError", 0) or 0)
        khr_error = int(evaluation.get("khrValueError", 0) or 0)
        usd_error = int(evaluation.get("usdValueError", 0) or 0)
        recall_same = float(evaluation.get("recallSameClass", 0.0) or 0.0)
        labels = str(evaluation.get("labels", "")).strip()
        manifest_labels = str(manifest_row.get("labels", "")).strip()
        gt_failures: list[str] = []
        if manifest_labels and labels and Path(labels).as_posix() != Path(manifest_labels).as_posix():
            gt_failures.append("labels_mismatch")
        if not labels:
            gt_failures.append("missing_labels")
        else:
            label_path = resolve(Path(labels))
            if not label_path.exists():
                gt_failures.append("labels_missing")
            else:
                truth = yolo_label_count_values(label_path)
                if truth["count"] != gt_count:
                    gt_failures.append(f"gt_count {gt_count}!={truth['count']}")
                if int(evaluation.get("gtKhrValue", 0) or 0) != truth["khr_value"]:
                    gt_failures.append(f"gt_khr {evaluation.get('gtKhrValue')}!={truth['khr_value']}")
                if int(evaluation.get("gtUsdValue", 0) or 0) != truth["usd_value"]:
                    gt_failures.append(f"gt_usd {evaluation.get('gtUsdValue')}!={truth['usd_value']}")
        class_pass = recall_same >= 0.999 if gt_count > 0 else True
        passed = not contract_failures and not gt_failures and count_error == 0 and khr_error == 0 and usd_error == 0 and class_pass
        if passed:
            pass_count += 1
        else:
            failure_bits: list[str] = []
            if contract_failures:
                failure_bits.append(f"contract {','.join(contract_failures)}")
            if gt_failures:
                failure_bits.append(f"ground_truth {','.join(gt_failures)}")
            if count_error:
                failure_bits.append(f"count_error {count_error:+d}")
            if khr_error:
                failure_bits.append(f"khr_error {khr_error:+d}")
            if usd_error:
                failure_bits.append(f"usd_error {usd_error:+d}")
            if not class_pass:
                failure_bits.append(f"recall_same_class {recall_same:.3f}")
            blockers.append(f"{case_id}: {'; '.join(failure_bits)}")
        evidence_rows.append(
            {
                "case_id": case_id,
                "passed": passed,
                "gt_count": gt_count,
                "pred_count": evaluation.get("predCount"),
                "count_error": count_error,
                "khr_value_error": khr_error,
                "usd_value_error": usd_error,
                "recall_same_class": recall_same,
                "contract_failures": contract_failures,
                "ground_truth_failures": gt_failures,
                "labels": labels,
            }
        )

    status = "pass" if pass_count == len(rows) and not blockers else "blocked"
    return axis(
        name,
        status,
        f"{pass_count}/{len(rows)} {label} case(s) pass strict count/value/class deploy guard.",
        evidence={
            "generated_at_utc": report.get("generated_at_utc", ""),
            "schema": report.get("schema", ""),
            "source": report.get("_source", ""),
            "case_manifest": report.get("case_manifest", ""),
            "input_fingerprints": report.get("input_fingerprints", {}),
            "cases": evidence_rows,
        },
        blockers=blockers,
        next_action=blocked_next_action,
    )


def split_coverage_freshness_failures(split_coverage: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    data = str(split_coverage.get("data", "")).strip()
    expected_data_sha = str(split_coverage.get("data_config_sha256", "")).strip()
    if not data:
        failures.append("split coverage report is missing data config path")
    elif not expected_data_sha:
        failures.append("split coverage report is missing data_config_sha256; regenerate it with the current check_yolo_dataset.py")
    else:
        data_path = resolve(Path(data))
        if not data_path.exists():
            failures.append(f"split coverage data config is missing: {data}")
        else:
            actual_data_sha = file_sha256(data_path)
            if actual_data_sha != expected_data_sha:
                failures.append(f"split coverage data config fingerprint is stale: {data}")

    split_sources = split_coverage.get("split_sources", {})
    if not isinstance(split_sources, dict) or not split_sources:
        failures.append("split coverage report is missing split source fingerprints; regenerate it with the current check_yolo_dataset.py")
        return failures
    for split_name, rows in sorted(split_sources.items()):
        if not isinstance(rows, list) or not rows:
            failures.append(f"split coverage {split_name} split has no source fingerprints")
            continue
        for source in rows:
            if not isinstance(source, dict):
                failures.append(f"split coverage {split_name} source fingerprint row is malformed")
                continue
            path_text = str(source.get("path", "")).strip()
            if not path_text:
                failures.append(f"split coverage {split_name} source fingerprint row is missing path")
                continue
            path = resolve(Path(path_text))
            if not path.exists():
                failures.append(f"split coverage {split_name} source is missing: {path_text}")
                continue
            kind = str(source.get("kind", "")).strip()
            if kind == "list":
                expected = str(source.get("sha256", "")).strip()
                if not expected:
                    failures.append(f"split coverage {split_name} list is missing sha256: {path_text}")
                    continue
                actual = file_sha256(path)
                if actual != expected:
                    failures.append(f"split coverage {split_name} list fingerprint is stale: {path_text}")
            elif kind == "directory":
                expected = str(source.get("listing_sha256", "")).strip()
                if not expected:
                    failures.append(f"split coverage {split_name} directory is missing listing_sha256: {path_text}")
                    continue
                image_rows = sorted(
                    repo_path(item)
                    for item in path.glob("*")
                    if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                )
                digest = hashlib.sha256()
                for row in image_rows:
                    digest.update(row.encode("utf-8"))
                    digest.update(b"\n")
                if digest.hexdigest() != expected:
                    failures.append(f"split coverage {split_name} directory listing fingerprint is stale: {path_text}")
            else:
                failures.append(f"split coverage {split_name} source has unknown kind {kind!r}: {path_text}")
    return failures


def real_train_class_coverage_axis(split_coverage: dict[str, Any], min_unique_images: int) -> dict[str, Any]:
    if not split_coverage:
        return axis(
            "real_train_class_coverage",
            "missing",
            "No clean-real split coverage report was supplied.",
            next_action="Run check_yolo_dataset.py with --json-out for configs/cashsnap_v1.yaml.",
        )
    train = split_coverage.get("splits", {}).get("train", {})
    classes = train.get("classes", {}) if isinstance(train, dict) else {}
    if not isinstance(classes, dict) or not classes:
        return axis(
            "real_train_class_coverage",
            "missing",
            "Clean-real split coverage report has no train class summary.",
            evidence={"data": split_coverage.get("data", "")},
            next_action="Regenerate split coverage with the current check_yolo_dataset.py.",
        )

    class_counts: dict[str, int] = {}
    failing: dict[str, int] = {}
    for class_name, row in sorted(classes.items()):
        if not isinstance(row, dict):
            continue
        unique_images = int(row.get("unique_images", 0) or 0)
        class_counts[str(class_name)] = unique_images
        if unique_images < min_unique_images:
            failing[str(class_name)] = unique_images

    freshness_failures = split_coverage_freshness_failures(split_coverage)
    status = "pass" if not failing and not freshness_failures else "blocked"
    summary = (
        f"Clean-real train split has at least {min_unique_images} unique image(s) for every class."
        if status == "pass"
        else f"Clean-real train split has {len(failing)} class(es) below {min_unique_images} unique train image(s)."
    )
    blockers = [f"{name}: {count}/{min_unique_images} unique train images" for name, count in failing.items()]
    return axis(
        "real_train_class_coverage",
        status,
        summary,
        evidence={
            "data": split_coverage.get("data", ""),
            "data_config_sha256": split_coverage.get("data_config_sha256", ""),
            "split_sources": split_coverage.get("split_sources", {}),
            "train_images": train.get("images"),
            "train_background_images": train.get("background_images"),
            "min_unique_images": min_unique_images,
            "class_unique_images": class_counts,
        },
        blockers=[*blockers, *freshness_failures],
        next_action=(
            "Add or promote genuinely unique rare-class real examples before treating synthetic rare support as scale-ready."
        ),
    )


def read_yolo_class_names(data_config: Path) -> tuple[list[str], list[str]]:
    resolved = resolve(data_config)
    if not resolved.exists():
        return [], [f"taxonomy data config is missing: {repo_path(resolved)}"]
    try:
        data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        return [], [f"taxonomy data config could not be read: {exc}"]
    if not isinstance(data, dict):
        return [], [f"taxonomy data config is malformed: {repo_path(resolved)}"]
    raw_names = data.get("names", {})
    if isinstance(raw_names, dict):
        rows: list[tuple[int, str]] = []
        for key, value in raw_names.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                return [], [f"taxonomy data config has non-integer class id: {key!r}"]
            rows.append((class_id, str(value)))
        return [name for _, name in sorted(rows)], []
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names], []
    return [], [f"taxonomy data config has no usable names map: {repo_path(resolved)}"]


def currency_taxonomy_scope_axis(data_config: Path) -> dict[str, Any]:
    class_names, failures = read_yolo_class_names(data_config)
    if failures:
        return axis(
            "currency_taxonomy_scope",
            "missing",
            "Current model taxonomy scope could not be checked.",
            evidence={"data": repo_path(resolve(data_config)), "official_sources": OFFICIAL_TAXONOMY_SOURCES},
            blockers=failures,
            next_action="Point --taxonomy-data at the current YOLO data.yaml before any perfect/done claim.",
        )

    present = set(class_names)
    missing_official = [name for name in OFFICIAL_CURRENT_CLASS_NAMES if name not in present]
    outside_official = [name for name in class_names if name not in set(OFFICIAL_CURRENT_CLASS_NAMES)]
    status = "pass" if not missing_official else "blocked"
    summary = (
        f"Current model taxonomy covers all {len(OFFICIAL_CURRENT_CLASS_NAMES)} official current USD/KHR denomination class(es)."
        if status == "pass"
        else (
            f"Current model taxonomy covers {len(class_names)}/{len(OFFICIAL_CURRENT_CLASS_NAMES)} official current "
            "USD/KHR denomination class(es)."
        )
    )
    blockers = [f"missing official current denomination class: {name}" for name in missing_official]
    return axis(
        "currency_taxonomy_scope",
        status,
        summary,
        evidence={
            "data": repo_path(resolve(data_config)),
            "data_config_sha256": file_sha256(resolve(data_config)),
            "current_class_names": class_names,
            "official_current_class_names": OFFICIAL_CURRENT_CLASS_NAMES,
            "missing_official_class_names": missing_official,
            "outside_official_class_names": outside_official,
            "official_sources": OFFICIAL_TAXONOMY_SOURCES,
        },
        blockers=blockers,
        next_action=(
            ""
            if status == "pass"
            else "Make an explicit product-scope decision, then add real/synthetic assets and labels for missing denominations or document them as out of scope before any perfect/done claim."
        ),
    )


def governance_axis(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return axis(
            "governance_and_provenance",
            "missing",
            "No synthetic governance manifest/check report was supplied.",
            next_action="Add and run the synthetic governance manifest before any perfect/done or release claim.",
        )

    status = str(report.get("status", "blocked"))
    if status not in STATUS_ORDER:
        status = "blocked"
    source_artifacts = report.get("source_artifacts", [])
    if not isinstance(source_artifacts, list):
        source_artifacts = []
    warnings = report.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    blockers = report.get("blockers", [])
    if not isinstance(blockers, list):
        blockers = ["governance blockers are malformed"]
    release_status = str(report.get("release_status", "")).strip()
    summary = (
        f"Synthetic governance manifest passes for {release_status or 'unknown'} scope "
        f"with {len(source_artifacts)} source artifact(s)."
        if status == "pass"
        else f"Synthetic governance manifest is {status}."
    )
    return axis(
        "governance_and_provenance",
        status,
        summary,
        evidence={
            "manifest": report.get("manifest", ""),
            "manifest_sha256": report.get("manifest_sha256", ""),
            "release_status": release_status,
            "public_release_allowed": bool(report.get("public_release_allowed")),
            "model_release_allowed": bool(report.get("model_release_allowed")),
            "public_release_blocker": report.get("public_release_blocker", ""),
            "model_release_blocker": report.get("model_release_blocker", ""),
            "rights_status_counts": report.get("rights_status_counts", {}),
            "release_limited_source_artifacts": report.get("release_limited_source_artifacts", []),
            "check_counts": report.get("check_counts", {}),
            "warnings": warnings,
        },
        blockers=[str(item) for item in blockers],
        next_action=""
        if status == "pass"
        else "Repair the governance manifest, source-artifact paths, release blockers, or rights/privacy gating.",
    )


def build_scorecard(
    readiness: dict[str, Any],
    domain_gap: dict[str, Any],
    geometry_domain_gap: dict[str, Any],
    mined_review: dict[str, Any],
    mined_review_quality: dict[str, Any],
    split_coverage: dict[str, Any],
    min_real_train_class_images: int,
    mined_real_utility_comparisons: list[dict[str, Any]],
    governance_report: dict[str, Any],
    browser_synthetic_stress: dict[str, Any],
    browser_synthetic_manifest: list[dict[str, str]],
    browser_synthetic_manifest_path: Path,
    browser_mined_real_stress: dict[str, Any],
    browser_mined_real_manifest: list[dict[str, str]],
    browser_mined_real_manifest_path: Path,
    taxonomy_data: Path,
) -> dict[str, Any]:
    required = int(readiness.get("required_conditions", 0) or 0)
    trainable = int(readiness.get("required_with_trainable_candidate", 0) or 0)
    real_ready = int(readiness.get("required_with_real_role_labels", 0) or 0)
    real_total = int(readiness.get("required_real_role_conditions", 0) or 0)
    usable_captures = int(readiness.get("usable_capture_images", 0) or 0)
    capture_inventory_issues = [str(item) for item in readiness.get("capture_inventory_issues", [])]
    blocked_conditions = [str(item) for item in readiness.get("blocked_required_conditions", [])]
    missing_trainable_conditions = [
        condition_id
        for condition_id, condition in sorted(conditions_by_id(readiness).items())
        if bool(condition.get("required_for_v1")) and not condition.get("active_suite_recipe_ids")
    ]

    axes: list[dict[str, Any]] = []
    axes.append(readiness_freshness_axis(readiness))
    axes.append(currency_taxonomy_scope_axis(taxonomy_data))
    axes.append(
        axis(
            "target_condition_coverage",
            "pass" if required and trainable == required else "blocked",
            f"{trainable}/{required} required conditions have active trainable-candidate synthetic coverage.",
            evidence={
                "required_conditions": required,
                "required_with_trainable_candidate": trainable,
                "missing_trainable_conditions": missing_trainable_conditions,
            },
            blockers=[] if trainable == required else missing_trainable_conditions or ["not every required condition has trainable-candidate coverage"],
            next_action=""
            if trainable == required
            else "Add trainable coverage only after each missing condition's real bridge and controls are ready; keep diagnostic recipes diagnostic until then.",
        )
    )

    blockers = package_blockers(readiness)
    axes.append(
        axis(
            "label_and_package_trust",
            "pass" if readiness.get("check_existing") and not blockers else "blocked",
            "Rendered suite packages were checked and have no package blockers."
            if readiness.get("check_existing") and not blockers
            else "Rendered suite packages are not fully verified.",
            evidence={"check_existing": bool(readiness.get("check_existing")), "package_report_count": len(readiness.get("suite_package_reports", {}))},
            blockers=blockers if blockers else ([] if readiness.get("check_existing") else ["readiness was not run with --check-existing"]),
            next_action="" if readiness.get("check_existing") and not blockers else "Run readiness with --check-existing and repair package blockers.",
        )
    )

    axes.append(
        axis(
            "real_anchor_and_holdout",
            "pass" if real_total and real_ready == real_total and usable_captures > 0 else "blocked",
            f"{real_ready}/{real_total} role-gated conditions have promoted real labels; usable capture inventory has {usable_captures} images.",
            evidence={
                "promoted_real_role_counts": readiness.get("promoted_real_role_counts", {}),
                "scoreable_real_images": readiness.get("scoreable_real_images", []),
                "usable_capture_images": usable_captures,
                "capture_inventory_issues": capture_inventory_issues,
            },
            blockers=(
                []
                if real_total and real_ready == real_total and usable_captures > 0 and not capture_inventory_issues
                else [*blocked_conditions, *capture_inventory_issues]
            ),
            next_action="Promote reviewed real stress labels or register usable captures before claiming transfer proof.",
        )
    )

    candidates = candidate_totals(readiness)
    candidate_inventory = required_candidate_inventory(readiness)
    candidate_condition_count = len(candidate_inventory["hit_condition_ids"])
    candidate_mapped_condition_count = len(candidate_inventory["mapped_condition_ids"])
    missing_candidate_hints = candidate_inventory["missing_scene_hints"]
    mined_review_total = int(mined_review.get("selected_total", 0) or 0)
    mined_review_scenes = mined_review.get("selected_by_scene", {})
    if not isinstance(mined_review_scenes, dict):
        mined_review_scenes = {}
    edge_summary = (
        f"Mined real-dataset review candidates exist for {candidate_condition_count}/{candidate_mapped_condition_count} "
        "candidate-mapped required condition(s)."
    )
    if missing_candidate_hints:
        edge_summary += f" Missing candidate hints for {len(missing_candidate_hints)} required scene slice(s)."
    if mined_review_total:
        edge_summary += f" A draft-only review package has {mined_review_total} selected candidate(s)."
    mined_quality_summary: dict[str, Any] = {}
    mined_review_freshness = mined_review_freshness_failures(mined_review)
    mined_review_quality_freshness = mined_review_quality_freshness_failures(mined_review_quality)
    if mined_review_quality:
        ready_scoreable = int(mined_review_quality.get("ready_scoreable_images", 0) or 0)
        ready_stress = int(mined_review_quality.get("ready_stress_images", 0) or 0)
        scoreable_boxes = int(mined_review_quality.get("scoreable_boxes", 0) or 0)
        mined_quality_summary = {
            "images": mined_review_quality.get("images", 0),
            "draft_boxes": mined_review_quality.get("draft_boxes", 0),
            "quality_rows": mined_review_quality.get("quality_rows", 0),
            "ready_scoreable_images": ready_scoreable,
            "ready_stress_images": ready_stress,
            "scoreable_boxes": scoreable_boxes,
            "status_counts": mined_review_quality.get("status_counts", {}),
            "quality_counts": mined_review_quality.get("quality_counts", {}),
            "count_for_score_states": mined_review_quality.get("count_for_score_states", {}),
            "by_role": mined_review_quality.get("by_role", {}),
            "freshness_failures": mined_review_quality_freshness,
        }
        edge_summary += f" Quality review has {ready_scoreable} ready scoreable image(s), {scoreable_boxes} scoreable box(es)."
    edge_freshness_failures = [
        *(f"mined review: {failure}" for failure in mined_review_freshness),
        *(f"mined review quality: {failure}" for failure in mined_review_quality_freshness),
    ]
    edge_blockers = [
        f"{row['condition_id']}: {row['scene_type']} {row['candidate_count']} candidates/{row['unique_origin_count']} origins"
        for row in missing_candidate_hints
    ]
    edge_blockers.extend(edge_freshness_failures)
    if not edge_blockers and not candidate_condition_count:
        edge_blockers.append("no mined real-dataset candidate hints were loaded")
    axes.append(
        axis(
            "edge_case_inventory",
            "blocked" if missing_candidate_hints or edge_freshness_failures or not candidate_condition_count else "review",
            edge_summary,
            evidence={
                "unique_origin_counts": candidates,
                "candidate_mapped_required_conditions": candidate_inventory["mapped_condition_ids"],
                "candidate_hit_required_conditions": candidate_inventory["hit_condition_ids"],
                "missing_candidate_hints": missing_candidate_hints,
                "mined_review_package": {
                    "selected_total": mined_review_total,
                    "selected_by_scene": mined_review_scenes,
                    "review_index": mined_review.get("review_index", ""),
                    "quality_template_out": mined_review.get("quality_template_out", ""),
                    "quality_template_rows": mined_review.get("quality_template_rows", 0),
                    "policy": mined_review.get("policy", {}),
                    "freshness_failures": mined_review_freshness,
                },
                "mined_review_quality": mined_quality_summary,
            },
            blockers=edge_blockers,
            next_action="Visually audit the mined review package, add per-box quality rows only for protected/use-safe labels, and keep true fan/hand/hard-negative gaps separate.",
        )
    )
    axes.append(real_train_class_coverage_axis(split_coverage, min_real_train_class_images))

    hard_negative_blockers = condition_blockers(readiness, "hard_negatives_and_non_banknote_paper")
    axes.append(
        axis(
            "hard_negatives",
            "pass" if not hard_negative_blockers else "blocked",
            "Hard-negative/no-note validation is covered." if not hard_negative_blockers else "Hard-negative/no-note validation is still blocked.",
            evidence={"condition_id": "hard_negatives_and_non_banknote_paper"},
            blockers=hard_negative_blockers,
            next_action="Use reviewed no-note and non-banknote prop captures; blank-label banknote images do not count.",
        )
    )
    axes.append(
        condition_axis(
            readiness,
            name="mixed_cross_currency_bridge",
            condition_id="mixed_rare_common_cross_currency_stack",
            label="Mixed rare/common USD+KHR validation bridge",
            next_action="Capture or promote rights-clear mixed USD+KHR scenes containing KHR_50000 plus common KHR, then rerun matched row-count/class-mix probes.",
        )
    )

    axes.append(domain_gap_axis(domain_gap))
    axes.append(
        domain_gap_axis(
            geometry_domain_gap,
            name="visible_note_geometry_gap",
            label="Accepted-blend visible-note geometry domain-gap",
            expected_preset="accepted_blend_geometry_v1",
            blocked_next_action="Repair synthetic visible-note scale and per-class geometry before more training spend.",
        )
    )
    axes.append(mined_real_utility_axis(mined_real_utility_comparisons))
    axes.append(
        browser_synthetic_stress_axis(browser_synthetic_stress, browser_synthetic_manifest, browser_synthetic_manifest_path)
    )
    axes.append(
        browser_synthetic_stress_axis(
            browser_mined_real_stress,
            browser_mined_real_manifest,
            browser_mined_real_manifest_path,
            name="browser_mined_real_stress",
            label="browser mined-real stress",
            missing_next_action="Build mined-real browser cases and run them through the browser stack before treating deploy behavior as real-stress checked.",
            blocked_next_action=(
                "Use mined-real browser failures to steer deploy fusion and curriculum probes; this diagnostic set is not protected transfer proof."
            ),
        )
    )

    axes.append(
        axis(
            "real_utility_gate",
            "pass" if readiness.get("ready_for_synthetic_scale") else "blocked",
            "All required conditions are ready for synthetic scale."
            if readiness.get("ready_for_synthetic_scale")
            else "Synthetic scale is blocked until real-transfer and capture/role gaps close.",
            evidence={"blocked_required_conditions": blocked_conditions},
            blockers=blocked_conditions,
            next_action="Run bounded model probes only when the relevant real scoreboard/controls are available.",
        )
    )

    axes.append(governance_axis(governance_report))

    status_counts = Counter(str(row["status"]) for row in axes)
    overall = "pass"
    if status_counts.get("blocked", 0):
        overall = "blocked"
    elif status_counts.get("missing", 0):
        overall = "missing"
    elif status_counts.get("review", 0):
        overall = "review"

    return {
        "rubric_source": RUBRIC_SOURCE,
        "rubric_sources": RUBRIC_SOURCES,
        "readiness": readiness.get("ready_for_synthetic_scale", False),
        "overall_status": overall,
        "status_counts": dict(sorted(status_counts.items())),
        "axes": axes,
    }


def main() -> int:
    args = parse_args()
    readiness = read_json(args.readiness)
    domain_gap = read_json(args.domain_gap, required=False)
    geometry_domain_gap = read_json(args.geometry_domain_gap, required=False)
    mined_review = read_json(args.mined_review, required=False)
    mined_review_quality = read_json(args.mined_review_quality, required=False)
    split_coverage = read_json(args.split_coverage, required=False)
    governance_report = check_governance_manifest(args.governance)
    comparison_paths = [] if args.no_default_mined_real_utility else list(DEFAULT_MINED_REAL_UTILITY_COMPARISONS)
    comparison_paths.extend(args.mined_real_utility_comparison)
    mined_real_utility_comparisons = read_comparison_jsons(comparison_paths)
    browser_synthetic_stress = read_browser_stress_report(args.browser_synthetic_stress, required=False)
    browser_synthetic_manifest = read_csv_rows(args.browser_synthetic_manifest, required=False)
    browser_mined_real_stress = read_browser_stress_report(args.browser_mined_real_stress, required=False)
    browser_mined_real_manifest = read_csv_rows(args.browser_mined_real_manifest, required=False)
    scorecard = build_scorecard(
        readiness,
        domain_gap,
        geometry_domain_gap,
        mined_review,
        mined_review_quality,
        split_coverage,
        args.min_real_train_class_images,
        mined_real_utility_comparisons,
        governance_report,
        browser_synthetic_stress,
        browser_synthetic_manifest,
        args.browser_synthetic_manifest,
        browser_mined_real_stress,
        browser_mined_real_manifest,
        args.browser_mined_real_manifest,
        args.taxonomy_data,
    )
    out = resolve(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "synthetic_dataset_scorecard="
        f"{scorecard['overall_status']} "
        + " ".join(f"{key}={value}" for key, value in sorted(scorecard["status_counts"].items()))
    )
    for row in scorecard["axes"]:
        print(f"{row['status']}: {row['name']} - {row['summary']}")
        for blocker in row["blockers"][:3]:
            print(f"  - {blocker}")
        if len(row["blockers"]) > 3:
            print(f"  - ... {len(row['blockers']) - 3} more")
        if row["next_action"] and row["status"] != "pass":
            print(f"  next: {row['next_action']}")
    print(f"wrote_json={repo_path(out)}")
    if args.require_pass and scorecard["overall_status"] != "pass":
        raise SystemExit(1)
    if args.strict and any(row["status"] in {"blocked", "missing"} for row in scorecard["axes"]):
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
