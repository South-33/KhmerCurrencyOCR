from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from compare_yolo_metrics import DEFAULT_METRIC
from compare_yolo_metrics import per_class_by_name
from compare_yolo_metrics import read_json
from compare_yolo_metrics import repo_rel
from compare_yolo_metrics import resolve
from compare_yolo_metrics import result_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a replicated set of val_yolo.py metrics JSON files against baseline metrics."
    )
    parser.add_argument(
        "--baseline",
        action="append",
        required=True,
        help="Baseline metrics JSON. Repeat to summarize multiple baseline runs.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate metrics JSON. Repeat once per seed/run.",
    )
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Metric key under the JSON 'results' object.")
    parser.add_argument(
        "--per-class-metric",
        default="map50_95",
        help="Per-class metric key to compare when --max-per-class-drop is set.",
    )
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=2,
        help="Require at least this many candidate metric files.",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Compare each candidate to the baseline at the same argument position.",
    )
    parser.add_argument(
        "--max-worst-drop",
        type=float,
        default=0.0,
        help="Allow at most this much candidate-min drop relative to baseline mean.",
    )
    parser.add_argument(
        "--min-mean-delta",
        type=float,
        default=None,
        help="Require candidate mean - baseline mean >= this value.",
    )
    parser.add_argument(
        "--max-std",
        type=float,
        default=None,
        help="Require candidate population standard deviation <= this value.",
    )
    parser.add_argument(
        "--max-per-class-drop",
        type=float,
        default=None,
        help="Allow at most this much per-class candidate-min drop relative to baseline mean.",
    )
    parser.add_argument("--json-out", default=None, help="Optional machine-readable comparison output.")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0, even when the comparison fails.")
    return parser.parse_args()


def read_metric_documents(path_values: list[str]) -> list[tuple[Path, dict[str, Any]]]:
    documents = []
    for path_value in path_values:
        path = resolve(path_value)
        documents.append((path, read_json(path)))
    return documents


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def result_values(documents: list[tuple[Path, dict[str, Any]]], metric: str) -> list[float]:
    return [result_metric(document, metric) for _, document in documents]


def class_metric_values(documents: list[tuple[Path, dict[str, Any]]], metric_name: str) -> dict[str, list[float | None]]:
    by_class: dict[str, list[float | None]] = {}
    all_names: set[str] = set()
    per_document = []
    for _, document in documents:
        rows = per_class_by_name(document)
        per_document.append(rows)
        all_names.update(rows)

    for class_name in sorted(all_names):
        values: list[float | None] = []
        for rows in per_document:
            value = rows.get(class_name, {}).get(metric_name)
            values.append(None if value is None else float(value))
        by_class[class_name] = values
    return by_class


def compare_per_class_replicates(
    baseline_documents: list[tuple[Path, dict[str, Any]]],
    candidate_documents: list[tuple[Path, dict[str, Any]]],
    metric_name: str,
) -> list[dict[str, Any]]:
    baseline_by_class = class_metric_values(baseline_documents, metric_name)
    candidate_by_class = class_metric_values(candidate_documents, metric_name)
    rows: list[dict[str, Any]] = []
    for class_name in sorted(set(baseline_by_class) | set(candidate_by_class)):
        baseline_values = baseline_by_class.get(class_name, [])
        candidate_values = candidate_by_class.get(class_name, [])
        baseline_numeric = [value for value in baseline_values if value is not None]
        candidate_numeric = [value for value in candidate_values if value is not None]
        row = {
            "class_name": class_name,
            "baseline_values": baseline_values,
            "candidate_values": candidate_values,
            "baseline_mean": None,
            "candidate_mean": None,
            "candidate_min": None,
            "mean_delta": None,
            "min_delta": None,
            "missing_baseline_values": len(baseline_values) - len(baseline_numeric),
            "missing_candidate_values": len(candidate_values) - len(candidate_numeric),
        }
        if baseline_numeric and candidate_numeric:
            baseline_mean = statistics.fmean(baseline_numeric)
            candidate_mean = statistics.fmean(candidate_numeric)
            candidate_min = min(candidate_numeric)
            row.update(
                {
                    "baseline_mean": baseline_mean,
                    "candidate_mean": candidate_mean,
                    "candidate_min": candidate_min,
                    "mean_delta": candidate_mean - baseline_mean,
                    "min_delta": candidate_min - baseline_mean,
                }
            )
        rows.append(row)
    return rows


def compare_per_class_paired(
    baseline_documents: list[tuple[Path, dict[str, Any]]],
    candidate_documents: list[tuple[Path, dict[str, Any]]],
    metric_name: str,
) -> list[dict[str, Any]]:
    baseline_by_class = class_metric_values(baseline_documents, metric_name)
    candidate_by_class = class_metric_values(candidate_documents, metric_name)
    rows: list[dict[str, Any]] = []
    for class_name in sorted(set(baseline_by_class) | set(candidate_by_class)):
        baseline_values = baseline_by_class.get(class_name, [])
        candidate_values = candidate_by_class.get(class_name, [])
        pair_deltas: list[float | None] = []
        for baseline_value, candidate_value in zip(baseline_values, candidate_values):
            if baseline_value is None or candidate_value is None:
                pair_deltas.append(None)
            else:
                pair_deltas.append(float(candidate_value) - float(baseline_value))
        numeric_deltas = [delta for delta in pair_deltas if delta is not None]
        baseline_numeric = [value for value in baseline_values if value is not None]
        candidate_numeric = [value for value in candidate_values if value is not None]
        row = {
            "class_name": class_name,
            "baseline_values": baseline_values,
            "candidate_values": candidate_values,
            "pair_deltas": pair_deltas,
            "baseline_mean": statistics.fmean(baseline_numeric) if baseline_numeric else None,
            "candidate_mean": statistics.fmean(candidate_numeric) if candidate_numeric else None,
            "candidate_min": min(candidate_numeric) if candidate_numeric else None,
            "mean_delta": statistics.fmean(numeric_deltas) if numeric_deltas else None,
            "min_delta": min(numeric_deltas) if numeric_deltas else None,
            "missing_baseline_values": len(baseline_values) - len(baseline_numeric),
            "missing_candidate_values": len(candidate_values) - len(candidate_numeric),
        }
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.min_candidates < 1:
        raise ValueError("--min-candidates must be at least 1")
    if args.max_worst_drop < 0:
        raise ValueError("--max-worst-drop must be non-negative")
    if args.max_std is not None and args.max_std < 0:
        raise ValueError("--max-std must be non-negative")
    if args.max_per_class_drop is not None and args.max_per_class_drop < 0:
        raise ValueError("--max-per-class-drop must be non-negative")

    baseline_documents = read_metric_documents(args.baseline)
    candidate_documents = read_metric_documents(args.candidate)
    if args.paired and len(baseline_documents) != len(candidate_documents):
        raise ValueError("--paired requires the same number of --baseline and --candidate files")
    baseline_summary = summarize(result_values(baseline_documents, args.metric))
    candidate_summary = summarize(result_values(candidate_documents, args.metric))

    baseline_mean = float(baseline_summary["mean"])
    candidate_mean = float(candidate_summary["mean"])
    candidate_min = float(candidate_summary["min"])
    if args.paired:
        baseline_values = result_values(baseline_documents, args.metric)
        candidate_values = result_values(candidate_documents, args.metric)
        pair_deltas = [candidate - baseline for baseline, candidate in zip(baseline_values, candidate_values)]
        mean_delta = statistics.fmean(pair_deltas)
        worst_delta = min(pair_deltas)
    else:
        pair_deltas = []
        mean_delta = candidate_mean - baseline_mean
        worst_delta = candidate_min - baseline_mean

    checks: list[dict[str, Any]] = [
        {
            "name": "min_candidates",
            "passed": len(candidate_documents) >= args.min_candidates,
            "threshold": args.min_candidates,
            "actual": len(candidate_documents),
        },
        {
            "name": "max_worst_drop",
            "passed": worst_delta >= -args.max_worst_drop,
            "threshold": -args.max_worst_drop,
            "actual_delta": worst_delta,
        },
    ]
    if args.min_mean_delta is not None:
        checks.append(
            {
                "name": "min_mean_delta",
                "passed": mean_delta >= args.min_mean_delta,
                "threshold": args.min_mean_delta,
                "actual_delta": mean_delta,
            }
        )
    if args.max_std is not None:
        candidate_std = float(candidate_summary["std"])
        checks.append(
            {
                "name": "max_std",
                "passed": candidate_std <= args.max_std,
                "threshold": args.max_std,
                "actual": candidate_std,
            }
        )

    per_class_rows: list[dict[str, Any]] = []
    per_class_failures: list[dict[str, Any]] = []
    if args.max_per_class_drop is not None:
        if args.paired:
            per_class_rows = compare_per_class_paired(
                baseline_documents,
                candidate_documents,
                args.per_class_metric,
            )
        else:
            per_class_rows = compare_per_class_replicates(
                baseline_documents,
                candidate_documents,
                args.per_class_metric,
            )
        threshold = -args.max_per_class_drop
        for row in per_class_rows:
            if (
                row["missing_baseline_values"]
                or row["missing_candidate_values"]
                or row["min_delta"] is None
                or float(row["min_delta"]) < threshold
            ):
                per_class_failures.append(row)
        worst_row = min(
            (row for row in per_class_rows if row["min_delta"] is not None),
            key=lambda row: float(row["min_delta"]),
            default=None,
        )
        checks.append(
            {
                "name": "max_per_class_drop",
                "passed": not per_class_failures,
                "threshold": threshold,
                "actual_delta": None if worst_row is None else float(worst_row["min_delta"]),
                "worst_class": None if worst_row is None else worst_row["class_name"],
                "failed_classes": [row["class_name"] for row in per_class_failures],
            }
        )

    passed = all(check["passed"] for check in checks)
    payload = {
        "passed": passed,
        "metric": args.metric,
        "baseline_paths": [repo_rel(path) for path, _ in baseline_documents],
        "candidate_paths": [repo_rel(path) for path, _ in candidate_documents],
        "comparison_mode": "paired" if args.paired else "baseline_mean",
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "pair_deltas": pair_deltas,
        "mean_delta": mean_delta,
        "worst_delta": worst_delta,
        "checks": checks,
        "per_class_metric": args.per_class_metric,
        "max_per_class_drop": args.max_per_class_drop,
        "per_class_failures": per_class_failures,
        "per_class": per_class_rows,
    }
    if args.json_out:
        write_json(resolve(args.json_out), payload)

    verdict = "PASS" if passed else "FAIL"
    print(
        f"{verdict}: {args.metric} baseline_mean={baseline_mean:.6f} "
        f"candidate_mean={candidate_mean:.6f} candidate_min={candidate_min:.6f} "
        f"mean_delta={mean_delta:+.6f} worst_delta={worst_delta:+.6f} "
        f"runs={len(candidate_documents)} mode={payload['comparison_mode']}"
    )
    if args.max_per_class_drop is not None:
        per_class_check = next(check for check in checks if check["name"] == "max_per_class_drop")
        actual_delta = per_class_check["actual_delta"]
        actual_delta_text = "none" if actual_delta is None else f"{actual_delta:+.6f}"
        print(
            "per_class "
            f"worst={per_class_check['worst_class']} "
            f"min_delta={actual_delta_text} "
            f"threshold={per_class_check['threshold']:+.6f} "
            f"failures={len(per_class_failures)}"
        )
    if args.json_out:
        print(f"wrote_json={repo_rel(resolve(args.json_out))}")
    return 0 if passed or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
