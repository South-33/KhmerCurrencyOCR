#!/usr/bin/env python
"""Summarize CashSnap synthetic-data quality against the project rubric.

This is a thin scorecard over existing gates. It does not replace readiness,
domain-gap, label-view, or model comparison checks; it makes their state visible
through the dataset-quality axes from the local research PDF.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_READINESS = ROOT / "runs" / "cashsnap" / "synthetic_pipeline_readiness_latest.json"
DEFAULT_DOMAIN_GAP = ROOT / "runs" / "cashsnap" / "domain_gap_accepted_nowarmup_train.json"
DEFAULT_MINED_REVIEW = ROOT / "runs" / "cashsnap" / "mined_real_benchmark_review_latest.json"
DEFAULT_MINED_REVIEW_QUALITY = ROOT / "runs" / "cashsnap" / "mined_real_benchmark_review_quality_summary_latest.json"
DEFAULT_JSON_OUT = ROOT / "runs" / "cashsnap" / "synthetic_dataset_scorecard_latest.json"

STATUS_ORDER = {"pass": 0, "review": 1, "missing": 2, "blocked": 3}
RUBRIC_SOURCE = "docs/research/What Makes a Dataset Perfect for Synthetic Data Pipelines.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--domain-gap", type=Path, default=DEFAULT_DOMAIN_GAP)
    parser.add_argument("--mined-review", type=Path, default=DEFAULT_MINED_REVIEW)
    parser.add_argument("--mined-review-quality", type=Path, default=DEFAULT_MINED_REVIEW_QUALITY)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any axis is blocked or missing.")
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


def review_candidate_condition_count(readiness: dict[str, Any]) -> int:
    total = 0
    for condition in conditions_by_id(readiness).values():
        rows = condition.get("real_dataset_review_candidates", [])
        if isinstance(rows, list) and rows:
            total += 1
    return total


def domain_gap_axis(domain_gap: dict[str, Any]) -> dict[str, Any]:
    if not domain_gap:
        return axis(
            "fidelity_domain_gap",
            "missing",
            "No accepted-blend domain-gap report was supplied.",
            next_action="Run audit_yolo_domain_gap.py with the accepted_blend_v1 preset.",
        )
    gate = domain_gap.get("domain_gap_gate", {})
    if not isinstance(gate, dict) or not gate.get("requested"):
        return axis(
            "fidelity_domain_gap",
            "review",
            "Domain-gap report exists but no gate was requested.",
            evidence={"data": domain_gap.get("data", ""), "split": domain_gap.get("split", "")},
            next_action="Regenerate the report with --gate-preset accepted_blend_v1 --fail-on-gap.",
        )
    failures = [str(item) for item in gate.get("failures", [])]
    status = "pass" if gate.get("passed") else "blocked"
    summary = "Accepted-blend domain-gap gate passed." if status == "pass" else "Accepted-blend domain-gap gate failed."
    return axis(
        "fidelity_domain_gap",
        status,
        summary,
        evidence={
            "data": domain_gap.get("data", ""),
            "split": domain_gap.get("split", ""),
            "observed": gate.get("observed", {}),
            "limits": gate.get("limits", {}),
        },
        blockers=failures,
        next_action="" if status == "pass" else "Repair synthetic/real dose or distribution drift before model spend.",
    )


def build_scorecard(
    readiness: dict[str, Any],
    domain_gap: dict[str, Any],
    mined_review: dict[str, Any],
    mined_review_quality: dict[str, Any],
) -> dict[str, Any]:
    required = int(readiness.get("required_conditions", 0) or 0)
    trainable = int(readiness.get("required_with_trainable_candidate", 0) or 0)
    real_ready = int(readiness.get("required_with_real_role_labels", 0) or 0)
    real_total = int(readiness.get("required_real_role_conditions", 0) or 0)
    usable_captures = int(readiness.get("usable_capture_images", 0) or 0)
    blocked_conditions = [str(item) for item in readiness.get("blocked_required_conditions", [])]

    axes: list[dict[str, Any]] = []
    axes.append(
        axis(
            "target_condition_coverage",
            "pass" if required and trainable == required else "blocked",
            f"{trainable}/{required} required conditions have active trainable-candidate synthetic coverage.",
            evidence={"required_conditions": required, "required_with_trainable_candidate": trainable},
            blockers=[] if trainable == required else ["not every required condition has trainable-candidate coverage"],
            next_action="" if trainable == required else "Add or promote missing synthetic recipe coverage.",
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
            },
            blockers=[] if real_total and real_ready == real_total and usable_captures > 0 else blocked_conditions,
            next_action="Promote reviewed real stress labels or register usable captures before claiming transfer proof.",
        )
    )

    candidates = candidate_totals(readiness)
    candidate_condition_count = review_candidate_condition_count(readiness)
    mined_review_total = int(mined_review.get("selected_total", 0) or 0)
    mined_review_scenes = mined_review.get("selected_by_scene", {})
    if not isinstance(mined_review_scenes, dict):
        mined_review_scenes = {}
    edge_summary = f"Mined real-dataset review candidates exist for {candidate_condition_count} required condition(s)."
    if mined_review_total:
        edge_summary += f" A draft-only review package has {mined_review_total} selected candidate(s)."
    mined_quality_summary: dict[str, Any] = {}
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
        }
        edge_summary += f" Quality review has {ready_scoreable} ready scoreable image(s), {scoreable_boxes} scoreable box(es)."
    axes.append(
        axis(
            "edge_case_inventory",
            "review" if candidate_condition_count else "blocked",
            edge_summary,
            evidence={
                "unique_origin_counts": candidates,
                "mined_review_package": {
                    "selected_total": mined_review_total,
                    "selected_by_scene": mined_review_scenes,
                    "review_index": mined_review.get("review_index", ""),
                    "quality_template_out": mined_review.get("quality_template_out", ""),
                    "quality_template_rows": mined_review.get("quality_template_rows", 0),
                    "policy": mined_review.get("policy", {}),
                },
                "mined_review_quality": mined_quality_summary,
            },
            blockers=[] if candidate_condition_count else ["no mined real-dataset candidate hints were loaded"],
            next_action="Visually audit the mined review package, add per-box quality rows only for protected/use-safe labels, and keep true fan/hand/hard-negative gaps separate.",
        )
    )

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

    axes.append(domain_gap_axis(domain_gap))

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

    axes.append(
        axis(
            "governance_and_provenance",
            "review",
            "Core provenance paths exist, but a complete release-grade datasheet/data-card/privacy package is not claimed.",
            evidence={
                "rubric_source": RUBRIC_SOURCE,
                "targets": readiness.get("targets", ""),
                "catalog": readiness.get("catalog", ""),
                "suite": readiness.get("suite", ""),
            },
            next_action="Before any public/release dataset, add datasheet/data card, license/provenance audit, intended use, limitations, and privacy/memorization checks.",
        )
    )

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
        "readiness": readiness.get("ready_for_synthetic_scale", False),
        "overall_status": overall,
        "status_counts": dict(sorted(status_counts.items())),
        "axes": axes,
    }


def main() -> int:
    args = parse_args()
    readiness = read_json(args.readiness)
    domain_gap = read_json(args.domain_gap, required=False)
    mined_review = read_json(args.mined_review, required=False)
    mined_review_quality = read_json(args.mined_review_quality, required=False)
    scorecard = build_scorecard(readiness, domain_gap, mined_review, mined_review_quality)
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
    if args.strict and any(row["status"] in {"blocked", "missing"} for row in scorecard["axes"]):
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
