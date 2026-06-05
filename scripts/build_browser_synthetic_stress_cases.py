#!/usr/bin/env python
"""Build a browser-smoke manifest from representative WebGL synthetic stress cases."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "manifests" / "browser_synthetic_stress_cases.csv"
DEFAULT_SYNTHETIC_ROOT = ROOT / "data" / "synthetic"
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
FIELDNAMES = [
    "case_id",
    "image",
    "labels",
    "min_same_class",
    "min_any_class",
    "max_count_error",
    "max_khr_error",
    "max_usd_error",
    "notes",
]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    root_name: str
    selector: str
    description: str
    guard_zero: bool = False


CASE_SPECS = [
    CaseSpec(
        case_id="synthetic_clean_base",
        root_name="cashsnap_webgl_clean_base_candidate_v1",
        selector="max_boxes",
        description="Diagnostic clean WebGL case; report browser transfer metrics before using as a guard.",
    ),
    CaseSpec(
        case_id="synthetic_overlap_stack",
        root_name="cashsnap_webgl_overlap_stack_candidate_v1",
        selector="max_boxes",
        description="Diagnostic overlap-stack WebGL case; report transfer metrics without gating yet.",
    ),
    CaseSpec(
        case_id="synthetic_fan_fullschema",
        root_name="cashsnap_webgl_fan_fullschema_candidate_v1",
        selector="max_boxes",
        description="Diagnostic fan WebGL case; stresses fan geometry and fragments.",
    ),
    CaseSpec(
        case_id="synthetic_same_class_fan",
        root_name="cashsnap_webgl_same_class_repeat_fan_balanced_audit_v1",
        selector="max_same_then_boxes",
        description="Diagnostic same-class fan WebGL case; count/fusion stress.",
    ),
    CaseSpec(
        case_id="synthetic_hand_occlusion",
        root_name="cashsnap_webgl_hand_occlusion_candidate_v1",
        selector="max_boxes",
        description="Diagnostic hand-occlusion WebGL case; stresses finger/hand occlusion pressure.",
    ),
    CaseSpec(
        case_id="synthetic_thin_edge",
        root_name="cashsnap_webgl_thin_edge_partial_candidate_v1",
        selector="max_boxes",
        description="Diagnostic thin-edge WebGL case; stresses sliver/partial evidence.",
    ),
    CaseSpec(
        case_id="synthetic_hard_negative",
        root_name="cashsnap_webgl_hard_negative_candidate_v1",
        selector="first_zero",
        description="Guard hard-negative WebGL case; must stay at 0 final detections and 0 value.",
        guard_zero=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, default=DEFAULT_SYNTHETIC_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="Print rows without writing the manifest.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return resolve(path).relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolve(path))


def read_manifest(root: Path) -> list[dict]:
    manifest = root / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"missing manifest: {repo_path(manifest)}")
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{repo_path(manifest)} must contain a non-empty list")
    return rows


def class_counts(label_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not label_path.exists():
        raise SystemExit(f"missing labels: {repo_path(label_path)}")
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_path(label_path)}:{line_number}: expected YOLO detect format")
        class_id = int(parts[0])
        if not 0 <= class_id < len(CLASS_NAMES):
            raise SystemExit(f"{repo_path(label_path)}:{line_number}: class {class_id} outside class map")
        counts[CLASS_NAMES[class_id]] += 1
    return counts


def score_for(selector: str, variant: int, counts: Counter[str]) -> tuple:
    box_count = sum(counts.values())
    max_same = max(counts.values(), default=0)
    if selector == "max_boxes":
        return (box_count, max_same, -variant)
    if selector == "max_same_then_boxes":
        return (max_same, box_count, -variant)
    if selector == "first_zero":
        return (1 if box_count == 0 else 0, -variant)
    raise SystemExit(f"unknown selector: {selector}")


def selected_row(spec: CaseSpec, synthetic_root: Path) -> tuple[dict, Counter[str]]:
    root = synthetic_root / spec.root_name
    best_row: dict | None = None
    best_counts: Counter[str] = Counter()
    best_score: tuple | None = None
    for row in read_manifest(root):
        if not isinstance(row, dict):
            raise SystemExit(f"{repo_path(root / 'manifest.json')}: row must be an object")
        variant = int(row.get("variant"))
        label_path = root / str(row.get("label", ""))
        counts = class_counts(label_path)
        score = score_for(spec.selector, variant, counts)
        if best_score is None or score > best_score:
            best_row = row
            best_counts = counts
            best_score = score
    if best_row is None:
        raise SystemExit(f"{spec.root_name}: no selectable rows")
    return best_row, best_counts


def counts_text(counts: Counter[str]) -> str:
    if not counts:
        return "0 targets"
    return ", ".join(f"{name}={counts[name]}" for name in sorted(counts))


def build_rows(synthetic_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in CASE_SPECS:
        root = synthetic_root / spec.root_name
        row, counts = selected_row(spec, synthetic_root)
        variant = int(row["variant"])
        case_id = f"{spec.case_id}_v{variant}"
        thresholds = {"max_count_error": "", "max_khr_error": "", "max_usd_error": ""}
        if spec.guard_zero:
            thresholds = {"max_count_error": "0", "max_khr_error": "0", "max_usd_error": "0"}
        rows.append(
            {
                "case_id": case_id,
                "image": f"/{repo_path(root / row['image'])}",
                "labels": repo_path(root / row["label"]),
                "min_same_class": "",
                "min_any_class": "",
                **thresholds,
                "notes": f"{spec.description} Selected by {spec.selector}; labels: {counts_text(counts)}.",
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = build_rows(resolve(args.synthetic_root))
    for row in rows:
        print(",".join(row[field] for field in FIELDNAMES))
    if not args.dry_run:
        out_path = resolve(args.out)
        write_rows(out_path, rows)
        print(f"wrote {repo_path(out_path)}")


if __name__ == "__main__":
    main()
