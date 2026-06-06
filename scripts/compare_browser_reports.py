from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = [
    "failures",
    "perfect_cases",
    "matched_same_class",
    "matched_any_class",
    "abs_count_error",
    "abs_khr_error",
    "abs_usd_error",
    "hard_negative_pred_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two CashSnap browser stress reports.")
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline run_browser_smoke_cases.py JSON report.")
    parser.add_argument("--candidate", required=True, type=Path, help="Candidate run_browser_smoke_cases.py JSON report.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional machine-readable comparison output.")
    parser.add_argument(
        "--metric",
        action="append",
        choices=DEFAULT_METRICS,
        default=[],
        help="Metric to gate. Repeatable. Defaults to all conservative non-regression metrics.",
    )
    parser.add_argument("--min-perfect-delta", type=int, default=0)
    parser.add_argument("--min-same-delta", type=int, default=0)
    parser.add_argument("--min-any-delta", type=int, default=0)
    parser.add_argument("--max-failure-delta", type=int, default=0)
    parser.add_argument("--max-abs-count-delta", type=int, default=0)
    parser.add_argument("--max-abs-khr-delta", type=int, default=0)
    parser.add_argument("--max-abs-usd-delta", type=int, default=0)
    parser.add_argument("--max-hard-negative-pred-delta", type=int, default=0)
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 after writing/printing the comparison.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    resolved = resolve(path)
    try:
        return resolved.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved.resolve())


def read_report(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"schema": "legacy_browser_stress_array", "cases": data}
    if not isinstance(data, dict):
        raise ValueError(f"{repo_path(resolved)}: expected JSON object or legacy array")
    if "reports" in data and "cases" not in data:
        raise ValueError(
            f"{repo_path(resolved)} contains report-summary rows, not raw browser cases; "
            "compare the source reports listed in its 'reports' array."
        )
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"{repo_path(resolved)}: expected 'cases' array")
    return data


def int_value(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def case_id(case: dict[str, Any], index: int) -> str:
    return str(case.get("caseId") or case.get("case_id") or f"row_{index}")


def summarize_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    cases = [case for case in report.get("cases", []) if isinstance(case, dict)]
    failures = len(report.get("failures", []) or [])
    totals = {
        "source": repo_path(path),
        "case_count": len(cases),
        "failures": failures,
        "gt_count": 0,
        "pred_count": 0,
        "matched_same_class": 0,
        "matched_any_class": 0,
        "abs_count_error": 0,
        "abs_khr_error": 0,
        "abs_usd_error": 0,
        "perfect_cases": 0,
        "rejected_proposals": 0,
        "hard_negative_case_count": 0,
        "hard_negative_pred_count": 0,
        "positive_case_count": 0,
        "positive_under_count": 0,
        "positive_over_count": 0,
    }
    per_case = []
    for index, case in enumerate(cases, start=1):
        evaluation = case.get("evaluation") if isinstance(case.get("evaluation"), dict) else {}
        debug = case.get("debug") if isinstance(case.get("debug"), dict) else {}
        gt_count = int_value(evaluation.get("gtCount"))
        pred_count = int_value(evaluation.get("predCount", case.get("totalCount")))
        same = int_value(evaluation.get("matchedSameClass"))
        any_match = int_value(evaluation.get("matchedAnyClass"))
        count_error = int_value(evaluation.get("countError"), pred_count - gt_count)
        khr_error = int_value(evaluation.get("khrValueError"))
        usd_error = int_value(evaluation.get("usdValueError"))
        rejected = int_value(debug.get("rejectedProposals"))
        perfect = same == gt_count and count_error == 0 and khr_error == 0 and usd_error == 0

        totals["gt_count"] += gt_count
        totals["pred_count"] += pred_count
        totals["matched_same_class"] += same
        totals["matched_any_class"] += any_match
        totals["abs_count_error"] += abs(count_error)
        totals["abs_khr_error"] += abs(khr_error)
        totals["abs_usd_error"] += abs(usd_error)
        totals["rejected_proposals"] += rejected
        if perfect:
            totals["perfect_cases"] += 1
        if gt_count == 0:
            totals["hard_negative_case_count"] += 1
            totals["hard_negative_pred_count"] += pred_count
        else:
            totals["positive_case_count"] += 1
            if count_error < 0:
                totals["positive_under_count"] += abs(count_error)
            elif count_error > 0:
                totals["positive_over_count"] += count_error

        per_case.append(
            {
                "case_id": case_id(case, index),
                "gt_count": gt_count,
                "pred_count": pred_count,
                "matched_same_class": same,
                "matched_any_class": any_match,
                "count_error": count_error,
                "khr_value_error": khr_error,
                "usd_value_error": usd_error,
                "rejected_proposals": rejected,
                "perfect": perfect,
            }
        )

    totals["per_case"] = per_case
    return totals


def compare_cases(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_cases = {str(row["case_id"]): row for row in baseline["per_case"]}
    candidate_cases = {str(row["case_id"]): row for row in candidate["per_case"]}
    rows: list[dict[str, Any]] = []
    for case_key in sorted(set(baseline_cases) | set(candidate_cases)):
        base_row = baseline_cases.get(case_key)
        cand_row = candidate_cases.get(case_key)
        if base_row is None or cand_row is None:
            rows.append(
                {
                    "case_id": case_key,
                    "present_in_baseline": base_row is not None,
                    "present_in_candidate": cand_row is not None,
                }
            )
            continue
        rows.append(
            {
                "case_id": case_key,
                "present_in_baseline": True,
                "present_in_candidate": True,
                "count_error_delta": int(cand_row["count_error"]) - int(base_row["count_error"]),
                "abs_count_error_delta": abs(int(cand_row["count_error"])) - abs(int(base_row["count_error"])),
                "abs_khr_error_delta": abs(int(cand_row["khr_value_error"])) - abs(int(base_row["khr_value_error"])),
                "abs_usd_error_delta": abs(int(cand_row["usd_value_error"])) - abs(int(base_row["usd_value_error"])),
                "matched_same_class_delta": int(cand_row["matched_same_class"]) - int(base_row["matched_same_class"]),
                "matched_any_class_delta": int(cand_row["matched_any_class"]) - int(base_row["matched_any_class"]),
                "pred_count_delta": int(cand_row["pred_count"]) - int(base_row["pred_count"]),
                "rejected_proposals_delta": int(cand_row["rejected_proposals"]) - int(base_row["rejected_proposals"]),
                "baseline_perfect": bool(base_row["perfect"]),
                "candidate_perfect": bool(cand_row["perfect"]),
            }
        )
    return rows


def check_threshold(name: str, delta: int, relation: str, threshold: int) -> dict[str, Any]:
    if relation == ">=":
        passed = delta >= threshold
    elif relation == "<=":
        passed = delta <= threshold
    else:
        raise ValueError(f"unknown relation {relation}")
    return {
        "name": name,
        "passed": passed,
        "delta": delta,
        "relation": relation,
        "threshold": threshold,
    }


def build_checks(args: argparse.Namespace, baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = args.metric or DEFAULT_METRICS
    checks = []
    if "failures" in metrics:
        checks.append(
            check_threshold(
                "failures",
                int(candidate["failures"]) - int(baseline["failures"]),
                "<=",
                args.max_failure_delta,
            )
        )
    if "perfect_cases" in metrics:
        checks.append(
            check_threshold(
                "perfect_cases",
                int(candidate["perfect_cases"]) - int(baseline["perfect_cases"]),
                ">=",
                args.min_perfect_delta,
            )
        )
    if "matched_same_class" in metrics:
        checks.append(
            check_threshold(
                "matched_same_class",
                int(candidate["matched_same_class"]) - int(baseline["matched_same_class"]),
                ">=",
                args.min_same_delta,
            )
        )
    if "matched_any_class" in metrics:
        checks.append(
            check_threshold(
                "matched_any_class",
                int(candidate["matched_any_class"]) - int(baseline["matched_any_class"]),
                ">=",
                args.min_any_delta,
            )
        )
    if "abs_count_error" in metrics:
        checks.append(
            check_threshold(
                "abs_count_error",
                int(candidate["abs_count_error"]) - int(baseline["abs_count_error"]),
                "<=",
                args.max_abs_count_delta,
            )
        )
    if "abs_khr_error" in metrics:
        checks.append(
            check_threshold(
                "abs_khr_error",
                int(candidate["abs_khr_error"]) - int(baseline["abs_khr_error"]),
                "<=",
                args.max_abs_khr_delta,
            )
        )
    if "abs_usd_error" in metrics:
        checks.append(
            check_threshold(
                "abs_usd_error",
                int(candidate["abs_usd_error"]) - int(baseline["abs_usd_error"]),
                "<=",
                args.max_abs_usd_delta,
            )
        )
    if "hard_negative_pred_count" in metrics:
        checks.append(
            check_threshold(
                "hard_negative_pred_count",
                int(candidate["hard_negative_pred_count"]) - int(baseline["hard_negative_pred_count"]),
                "<=",
                args.max_hard_negative_pred_delta,
            )
        )
    return checks


def main() -> int:
    args = parse_args()
    baseline_path = resolve(args.baseline)
    candidate_path = resolve(args.candidate)
    baseline_report = read_report(baseline_path)
    candidate_report = read_report(candidate_path)
    baseline = summarize_report(baseline_path, baseline_report)
    candidate = summarize_report(candidate_path, candidate_report)
    case_deltas = compare_cases(baseline, candidate)
    checks = build_checks(args, baseline, candidate)
    passed = all(row["passed"] for row in checks)
    payload = {
        "schema": "cashsnap_browser_report_comparison_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "baseline": {key: value for key, value in baseline.items() if key != "per_case"},
        "candidate": {key: value for key, value in candidate.items() if key != "per_case"},
        "deltas": {
            key: int(candidate[key]) - int(baseline[key])
            for key in baseline
            if key != "per_case" and isinstance(baseline.get(key), int) and isinstance(candidate.get(key), int)
        },
        "checks": checks,
        "passed": passed,
        "case_deltas": case_deltas,
    }
    if args.json_out is not None:
        out_path = resolve(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_comparison={repo_path(out_path)}")

    delta = payload["deltas"]
    print(
        "browser_report_comparison="
        f"{'pass' if passed else 'blocked'} "
        f"perfect_delta={delta.get('perfect_cases', 0)} "
        f"same_delta={delta.get('matched_same_class', 0)} "
        f"any_delta={delta.get('matched_any_class', 0)} "
        f"abs_count_delta={delta.get('abs_count_error', 0)} "
        f"abs_khr_delta={delta.get('abs_khr_error', 0)} "
        f"abs_usd_delta={delta.get('abs_usd_error', 0)} "
        f"hardneg_pred_delta={delta.get('hard_negative_pred_count', 0)} "
        f"failure_delta={delta.get('failures', 0)}"
    )
    if passed or args.no_fail:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
