from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRIC = "metrics/mAP50-95(B)"


def resolve(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two val_yolo.py metrics JSON files.")
    parser.add_argument("--baseline", required=True, help="Baseline metrics JSON.")
    parser.add_argument("--candidate", required=True, help="Candidate metrics JSON.")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Metric key under the JSON 'results' object.")
    parser.add_argument("--min-delta", type=float, default=None, help="Require candidate - baseline >= this value.")
    parser.add_argument("--max-drop", type=float, default=0.0, help="Allow at most this much negative delta.")
    parser.add_argument(
        "--per-class-metric",
        default="map50_95",
        help="Per-class metric key to compare when --max-per-class-drop is set.",
    )
    parser.add_argument(
        "--max-per-class-drop",
        type=float,
        default=None,
        help="Allow at most this much negative delta for each comparable class.",
    )
    parser.add_argument(
        "--only-classes",
        default="",
        help="Optional comma/semicolon separated class-name filter for per-class comparison.",
    )
    parser.add_argument(
        "--classes-from-summary",
        default=None,
        help="Optional dataset summary JSON whose 'classes' keys define the per-class comparison filter.",
    )
    parser.add_argument(
        "--ignore-missing-classes",
        action="store_true",
        help="Do not fail the per-class guard for filtered classes missing from one metrics file.",
    )
    parser.add_argument("--json-out", default=None, help="Optional machine-readable comparison output.")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0, even when the comparison fails.")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected JSON object: {path}")
    return document


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_metric(document: dict, metric: str) -> float:
    results = document.get("results", {})
    if not isinstance(results, dict) or metric not in results:
        raise KeyError(f"missing results metric {metric!r}")
    return float(results[metric])


def per_class_by_name(document: dict) -> dict[str, dict]:
    rows = document.get("per_class", [])
    if not isinstance(rows, list):
        return {}
    by_name = {}
    for row in rows:
        if isinstance(row, dict) and row.get("class_name") is not None:
            by_name[str(row["class_name"])] = row
    return by_name


def parse_class_filter(raw_value: str) -> set[str]:
    return {value.strip() for value in raw_value.replace(";", ",").split(",") if value.strip()}


def summary_classes(path: Path) -> set[str]:
    document = read_json(path)
    classes = document.get("classes", {})
    if not isinstance(classes, dict):
        raise ValueError(f"{path}: expected object at 'classes'")
    return {str(name) for name in classes}


def compare_per_class(
    baseline: dict,
    candidate: dict,
    metric_name: str = "map50_95",
    only_classes: set[str] | None = None,
) -> list[dict]:
    baseline_rows = per_class_by_name(baseline)
    candidate_rows = per_class_by_name(candidate)
    rows = []
    class_names = only_classes if only_classes is not None else set(baseline_rows) | set(candidate_rows)
    for class_name in sorted(class_names):
        baseline_value = baseline_rows.get(class_name, {}).get(metric_name)
        candidate_value = candidate_rows.get(class_name, {}).get(metric_name)
        row = {
            "class_name": class_name,
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": None,
        }
        if baseline_value is not None and candidate_value is not None:
            row["delta"] = float(candidate_value) - float(baseline_value)
        rows.append(row)
    return rows


def per_class_guard(rows: list[dict], max_drop: float, *, ignore_missing_classes: bool = False) -> tuple[dict, list[dict]]:
    if max_drop < 0:
        raise ValueError("--max-per-class-drop must be non-negative")

    comparable = [row for row in rows if row.get("delta") is not None]
    missing = [row for row in rows if row.get("delta") is None]
    failures = [row for row in comparable if float(row["delta"]) < -max_drop]
    worst = min(comparable, key=lambda row: float(row["delta"]), default=None)
    blocking_missing = [] if ignore_missing_classes else missing
    check = {
        "name": "max_per_class_drop",
        "passed": not failures and not blocking_missing,
        "threshold": -max_drop,
        "actual_delta": None if worst is None else float(worst["delta"]),
        "worst_class": None if worst is None else worst.get("class_name"),
        "failed_classes": [row.get("class_name") for row in failures],
        "missing_classes": [row.get("class_name") for row in missing],
        "ignore_missing_classes": ignore_missing_classes,
    }
    return check, failures + blocking_missing


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    baseline_path = resolve(args.baseline)
    candidate_path = resolve(args.candidate)
    classes_from_summary_path = resolve(args.classes_from_summary) if args.classes_from_summary else None
    baseline = read_json(baseline_path)
    candidate = read_json(candidate_path)
    baseline_value = result_metric(baseline, args.metric)
    candidate_value = result_metric(candidate, args.metric)
    delta = candidate_value - baseline_value
    only_classes = parse_class_filter(args.only_classes)
    if classes_from_summary_path:
        only_classes.update(summary_classes(classes_from_summary_path))
    per_class_rows = compare_per_class(
        baseline,
        candidate,
        args.per_class_metric,
        only_classes=only_classes or None,
    )

    checks = [
        {
            "name": "max_drop",
            "passed": delta >= -args.max_drop,
            "threshold": -args.max_drop,
            "actual_delta": delta,
        }
    ]
    if args.min_delta is not None:
        checks.append(
            {
                "name": "min_delta",
                "passed": delta >= args.min_delta,
                "threshold": args.min_delta,
                "actual_delta": delta,
            }
        )
    per_class_failures = []
    if args.max_per_class_drop is not None:
        check, per_class_failures = per_class_guard(
            per_class_rows,
            args.max_per_class_drop,
            ignore_missing_classes=args.ignore_missing_classes,
        )
        checks.append(check)

    passed = all(check["passed"] for check in checks)
    payload = {
        "passed": passed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "metric": args.metric,
        "baseline_path": repo_rel(baseline_path),
        "baseline_sha256": file_sha256(baseline_path),
        "candidate_path": repo_rel(candidate_path),
        "candidate_sha256": file_sha256(candidate_path),
        "classes_from_summary_path": repo_rel(classes_from_summary_path) if classes_from_summary_path else "",
        "classes_from_summary_sha256": file_sha256(classes_from_summary_path) if classes_from_summary_path else "",
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": delta,
        "checks": checks,
        "per_class_metric": args.per_class_metric,
        "max_per_class_drop": args.max_per_class_drop,
        "only_classes": sorted(only_classes),
        "per_class_failures": per_class_failures,
        "per_class_map50_95": per_class_rows,
    }
    if args.json_out:
        write_json(resolve(args.json_out), payload)

    verdict = "PASS" if passed else "FAIL"
    print(
        f"{verdict}: {args.metric} baseline={baseline_value:.6f} "
        f"candidate={candidate_value:.6f} delta={delta:+.6f}"
    )
    if args.max_per_class_drop is not None:
        per_class_check = next(check for check in checks if check["name"] == "max_per_class_drop")
        actual_delta = per_class_check["actual_delta"]
        actual_delta_text = "none" if actual_delta is None else f"{actual_delta:+.6f}"
        print(
            "per_class "
            f"worst={per_class_check['worst_class']} "
            f"delta={actual_delta_text} "
            f"threshold={per_class_check['threshold']:+.6f} "
            f"failures={len(per_class_failures)}"
        )
    if args.json_out:
        print(f"wrote_json={repo_rel(resolve(args.json_out))}")
    return 0 if passed or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
