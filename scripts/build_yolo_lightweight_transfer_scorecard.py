#!/usr/bin/env python
"""Build a bounded transfer scorecard from lightweight YOLO real evals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-pair",
        action="append",
        default=[],
        metavar="NAME=BASELINE_JSON,CANDIDATE_JSON",
        help="Lightweight eval pair to compare. Repeat for conf005, conf01, etc.",
    )
    parser.add_argument("--self-summary", type=Path, default=None)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--min-recall-delta", type=float, default=0.0)
    parser.add_argument("--max-total-fp-increase", type=int, default=0)
    parser.add_argument("--max-prediction-increase", type=int, default=0)
    parser.add_argument("--max-images-with-fp-increase", type=int, default=0)
    parser.add_argument("--max-background-fp-image-increase", type=int, default=0)
    parser.add_argument("--max-per-class-recall-drop", type=float, default=0.05)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--allow-self-fail", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"{repo_rel(resolved)}: expected JSON object")
    return document


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_pair(raw: str) -> tuple[str, Path, Path]:
    if "=" not in raw:
        raise SystemExit(f"--eval-pair must be NAME=BASELINE_JSON,CANDIDATE_JSON, got {raw!r}")
    name, rest = raw.split("=", 1)
    separator = "," if "," in rest else None
    parts = [part.strip() for part in rest.split(separator, 1)]
    if not name.strip() or len(parts) != 2 or not parts[0] or not parts[1]:
        raise SystemExit(f"--eval-pair must be NAME=BASELINE_JSON,CANDIDATE_JSON, got {raw!r}")
    return name.strip(), Path(parts[0]), Path(parts[1])


def number(document: dict[str, Any], key: str, default: int = 0) -> int:
    value = document.get(key, default)
    return int(value) if value is not None else default


def optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def delta(candidate: float | int | None, baseline: float | int | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate) - float(baseline)


def signed_delta_text(value: float | int | None) -> str:
    return "none" if value is None else f"{float(value):+.6f}"


def check_row(name: str, passed: bool, actual: Any, threshold: Any, detail: str = "") -> dict[str, Any]:
    row = {"name": name, "passed": bool(passed), "actual": actual, "threshold": threshold}
    if detail:
        row["detail"] = detail
    return row


def metadata_checks(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "data",
        "split",
        "imgsz",
        "conf",
        "iou",
        "nms_iou",
        "agnostic_nms",
        "images",
        "gt",
        "background_images",
    )
    rows = []
    for field in fields:
        if field == "nms_iou":
            base = baseline.get(field, 0.7)
            cand = candidate.get(field, 0.7)
        elif field == "agnostic_nms":
            base = baseline.get(field, False)
            cand = candidate.get(field, False)
        else:
            base = baseline.get(field)
            cand = candidate.get(field)
        rows.append(
            check_row(
                f"metadata_{field}_match",
                base == cand,
                {"baseline": base, "candidate": cand},
                "equal",
            )
        )
    return rows


def class_deltas(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_per_class_recall_drop: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_rows = baseline.get("per_class", {})
    candidate_rows = candidate.get("per_class", {})
    if not isinstance(baseline_rows, dict):
        baseline_rows = {}
    if not isinstance(candidate_rows, dict):
        candidate_rows = {}

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for class_name in sorted(set(baseline_rows) | set(candidate_rows)):
        base = baseline_rows.get(class_name, {})
        cand = candidate_rows.get(class_name, {})
        if not isinstance(base, dict):
            base = {}
        if not isinstance(cand, dict):
            cand = {}
        base_recall = optional_float(base.get("recall"))
        cand_recall = optional_float(cand.get("recall"))
        recall_delta = delta(cand_recall, base_recall)
        row = {
            "class_name": class_name,
            "baseline_gt": number(base, "gt"),
            "candidate_gt": number(cand, "gt"),
            "baseline_tp": number(base, "tp"),
            "candidate_tp": number(cand, "tp"),
            "baseline_fp": number(base, "fp"),
            "candidate_fp": number(cand, "fp"),
            "baseline_recall": base_recall,
            "candidate_recall": cand_recall,
            "recall_delta": recall_delta,
            "fp_delta": number(cand, "fp") - number(base, "fp"),
        }
        if recall_delta is not None and recall_delta < -max_per_class_recall_drop:
            failures.append(row)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            float("inf") if row["recall_delta"] is None else float(row["recall_delta"]),
            -abs(int(row["fp_delta"])),
            row["class_name"],
        )
    )
    failures.sort(key=lambda row: float(row["recall_delta"] or 0.0))
    return rows, failures


def source_deltas(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_rows = baseline.get("per_source", {})
    candidate_rows = candidate.get("per_source", {})
    if not isinstance(baseline_rows, dict):
        baseline_rows = {}
    if not isinstance(candidate_rows, dict):
        candidate_rows = {}

    rows: list[dict[str, Any]] = []
    for source_name in sorted(set(baseline_rows) | set(candidate_rows)):
        base = baseline_rows.get(source_name, {})
        cand = candidate_rows.get(source_name, {})
        if not isinstance(base, dict):
            base = {}
        if not isinstance(cand, dict):
            cand = {}
        rows.append(
            {
                "source_group": source_name,
                "baseline_images": number(base, "images"),
                "candidate_images": number(cand, "images"),
                "baseline_tp": number(base, "tp"),
                "candidate_tp": number(cand, "tp"),
                "baseline_fp": number(base, "fp"),
                "candidate_fp": number(cand, "fp"),
                "baseline_background_images_with_fp": number(base, "background_images_with_fp"),
                "candidate_background_images_with_fp": number(cand, "background_images_with_fp"),
                "recall_delta": delta(optional_float(cand.get("recall")), optional_float(base.get("recall"))),
                "precision_delta": delta(
                    optional_float(cand.get("precision")),
                    optional_float(base.get("precision")),
                ),
            }
        )
    return sorted(rows, key=lambda row: (-(int(row["candidate_fp"]) - int(row["baseline_fp"])), row["source_group"]))


def metric_summary(document: dict[str, Any]) -> dict[str, Any]:
    return {
        "recall": optional_float(document.get("recall")),
        "precision": optional_float(document.get("precision")),
        "tp": number(document, "tp"),
        "fp": number(document, "fp"),
        "fn": number(document, "fn"),
        "gt": number(document, "gt"),
        "total_predictions": number(document, "total_predictions"),
        "images_with_fp": number(document, "images_with_fp"),
        "background_images": number(document, "background_images"),
        "background_images_with_fp": number(document, "background_images_with_fp"),
    }


def compare_eval_pair(
    *,
    name: str,
    baseline_path: Path,
    candidate_path: Path,
    min_recall_delta: float,
    max_total_fp_increase: int,
    max_prediction_increase: int,
    max_images_with_fp_increase: int,
    max_background_fp_image_increase: int,
    max_per_class_recall_drop: float,
) -> dict[str, Any]:
    baseline = read_json(baseline_path)
    candidate = read_json(candidate_path)
    class_rows, class_failures = class_deltas(
        baseline,
        candidate,
        max_per_class_recall_drop=max_per_class_recall_drop,
    )

    recall_delta = delta(optional_float(candidate.get("recall")), optional_float(baseline.get("recall")))
    precision_delta = delta(optional_float(candidate.get("precision")), optional_float(baseline.get("precision")))
    fp_delta = number(candidate, "fp") - number(baseline, "fp")
    prediction_delta = number(candidate, "total_predictions") - number(baseline, "total_predictions")
    images_with_fp_delta = number(candidate, "images_with_fp") - number(baseline, "images_with_fp")
    background_fp_image_delta = number(candidate, "background_images_with_fp") - number(
        baseline,
        "background_images_with_fp",
    )

    checks = metadata_checks(baseline, candidate)
    checks.extend(
        [
            check_row("min_recall_delta", (recall_delta or 0.0) >= min_recall_delta, recall_delta, min_recall_delta),
            check_row(
                "max_total_fp_increase",
                fp_delta <= max_total_fp_increase,
                fp_delta,
                max_total_fp_increase,
            ),
            check_row(
                "max_prediction_increase",
                prediction_delta <= max_prediction_increase,
                prediction_delta,
                max_prediction_increase,
            ),
            check_row(
                "max_images_with_fp_increase",
                images_with_fp_delta <= max_images_with_fp_increase,
                images_with_fp_delta,
                max_images_with_fp_increase,
            ),
            check_row(
                "max_background_fp_image_increase",
                background_fp_image_delta <= max_background_fp_image_increase,
                background_fp_image_delta,
                max_background_fp_image_increase,
            ),
            check_row(
                "max_per_class_recall_drop",
                not class_failures,
                len(class_failures),
                f"no class below -{max_per_class_recall_drop}",
            ),
        ]
    )

    return {
        "name": name,
        "passed": all(row["passed"] for row in checks),
        "baseline_path": repo_rel(resolve(baseline_path)),
        "baseline_sha256": file_sha256(baseline_path),
        "candidate_path": repo_rel(resolve(candidate_path)),
        "candidate_sha256": file_sha256(candidate_path),
        "data": baseline.get("data"),
        "split": baseline.get("split"),
        "imgsz": baseline.get("imgsz"),
        "conf": baseline.get("conf"),
        "iou": baseline.get("iou"),
        "baseline": metric_summary(baseline),
        "candidate": metric_summary(candidate),
        "deltas": {
            "recall": recall_delta,
            "precision": precision_delta,
            "tp": number(candidate, "tp") - number(baseline, "tp"),
            "fp": fp_delta,
            "fn": number(candidate, "fn") - number(baseline, "fn"),
            "total_predictions": prediction_delta,
            "images_with_fp": images_with_fp_delta,
            "background_images_with_fp": background_fp_image_delta,
        },
        "checks": checks,
        "per_class_failures": class_failures,
        "per_class_deltas": class_rows,
        "top_source_deltas": source_deltas(baseline, candidate)[:8],
    }


def self_summary(path: Path, allow_self_fail: bool) -> dict[str, Any]:
    document = read_json(path)
    passed = bool(document.get("passed", False))
    return {
        "passed": passed or allow_self_fail,
        "raw_passed": passed,
        "path": repo_rel(resolve(path)),
        "sha256": file_sha256(path),
        "delta": document.get("delta"),
        "comparison": document.get("comparison"),
        "baseline_metrics": document.get("baseline_metrics"),
        "candidate_metrics": document.get("candidate_metrics"),
        "baseline_train_run": document.get("baseline_train_run"),
        "candidate_train_run": document.get("candidate_train_run"),
        "candidate_train_overrides": document.get("candidate_train_overrides", {}),
    }


def blocker_summary(scorecard: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    self_eval = scorecard.get("self_eval")
    if isinstance(self_eval, dict) and not self_eval.get("passed", False):
        blockers.append("synthetic self-eval failed")
    for row in scorecard.get("evals", []):
        if not isinstance(row, dict) or row.get("passed", False):
            continue
        failed_checks = [
            str(check.get("name"))
            for check in row.get("checks", [])
            if isinstance(check, dict) and not check.get("passed", False)
        ]
        blockers.append(f"{row.get('name')}: {', '.join(failed_checks)}")
    return blockers


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.eval_pair:
        raise SystemExit("At least one --eval-pair is required")

    evals = [
        compare_eval_pair(
            name=name,
            baseline_path=baseline_path,
            candidate_path=candidate_path,
            min_recall_delta=args.min_recall_delta,
            max_total_fp_increase=args.max_total_fp_increase,
            max_prediction_increase=args.max_prediction_increase,
            max_images_with_fp_increase=args.max_images_with_fp_increase,
            max_background_fp_image_increase=args.max_background_fp_image_increase,
            max_per_class_recall_drop=args.max_per_class_recall_drop,
        )
        for name, baseline_path, candidate_path in map(parse_pair, args.eval_pair)
    ]
    payload: dict[str, Any] = {
        "schema": "cashsnap_yolo_lightweight_transfer_scorecard_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "thresholds": {
            "min_recall_delta": args.min_recall_delta,
            "max_total_fp_increase": args.max_total_fp_increase,
            "max_prediction_increase": args.max_prediction_increase,
            "max_images_with_fp_increase": args.max_images_with_fp_increase,
            "max_background_fp_image_increase": args.max_background_fp_image_increase,
            "max_per_class_recall_drop": args.max_per_class_recall_drop,
        },
        "evals": evals,
    }
    if args.self_summary:
        payload["self_eval"] = self_summary(args.self_summary, args.allow_self_fail)
    payload["passed"] = all(row["passed"] for row in evals) and bool(payload.get("self_eval", {"passed": True})["passed"])
    payload["blockers"] = blocker_summary(payload)
    write_json(args.json_out, payload)

    print(f"wrote_json={repo_rel(resolve(args.json_out))}")
    print(f"{'PASS' if payload['passed'] else 'FAIL'}: eval_pairs={len(evals)}")
    if "self_eval" in payload:
        self_eval = payload["self_eval"]
        print(
            "self_eval: "
            f"{'PASS' if self_eval['passed'] else 'FAIL'} "
            f"raw={self_eval['raw_passed']} delta={self_eval.get('delta')}"
        )
    for row in evals:
        deltas = row["deltas"]
        print(
            f"{row['name']}: {'PASS' if row['passed'] else 'FAIL'} "
            f"conf={row.get('conf')} recall_delta={signed_delta_text(deltas['recall'])} "
            f"fp_delta={deltas['fp']:+d} bg_fp_img_delta={deltas['background_images_with_fp']:+d}"
        )
    if payload["blockers"]:
        print("blockers:")
        for blocker in payload["blockers"]:
            print(f"- {blocker}")
    return 0 if payload["passed"] or args.no_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
