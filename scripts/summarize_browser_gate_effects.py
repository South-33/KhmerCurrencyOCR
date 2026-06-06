from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize how browser proposal gating changed detector-source behavior case by case."
    )
    parser.add_argument("--report", required=True, type=Path, help="run_browser_smoke_cases.py JSON report.")
    parser.add_argument("--artifacts-dir", type=Path, default=None, help="Optional per-case browser artifact directory.")
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
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
    if not isinstance(data, dict):
        raise ValueError(f"{repo_path(resolved)}: expected JSON object")
    cases = data.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError(f"{repo_path(resolved)}: expected 'cases' array")
    return data


def int_value(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def source_metrics(evaluation: dict[str, Any], source: str) -> dict[str, int]:
    sources = evaluation.get("sources", {})
    row = sources.get(source, {}) if isinstance(sources, dict) else {}
    row = row if isinstance(row, dict) else {}
    return {
        "pred": int_value(row.get("predCount")),
        "same": int_value(row.get("matchedSameClass")),
        "any": int_value(row.get("matchedAnyClass")),
        "count_error": int_value(row.get("countError")),
    }


def source_exists(evaluation: dict[str, Any], source: str) -> bool:
    sources = evaluation.get("sources", {})
    return isinstance(sources, dict) and isinstance(sources.get(source), dict)


def classify_effect(
    gt_count: int,
    final: dict[str, int],
    detector: dict[str, int],
    proposals: dict[str, int],
    clustered: dict[str, int],
    debug: dict[str, int],
) -> str:
    final_abs_count = abs(final["count_error"])
    detector_abs_count = abs(detector["count_error"])
    rejected = debug["rejected_proposals"] or debug["rejected_after_nms"]
    detector_proposals = debug["detector_proposals"]
    if gt_count == 0:
        if rejected and final["pred"] == 0:
            return "hard_negative_help"
        if final["pred"] < detector["pred"] or final["pred"] < proposals["pred"]:
            return "hard_negative_help"
        if final["pred"] > detector["pred"]:
            return "hard_negative_harm"
        return "hard_negative_same"
    if rejected and final_abs_count > 0:
        return "positive_reject_harm"
    if rejected and final_abs_count == 0:
        return "positive_reject_safe"
    if (
        final_abs_count > 0
        and detector_proposals > final["pred"]
        and (proposals["same"] > final["same"] or proposals["any"] > final["any"] or clustered["pred"] > final["pred"])
    ):
        return "positive_nms_harm"
    if final["same"] < detector["same"] or final["any"] < detector["any"] or final_abs_count > detector_abs_count:
        return "positive_harm"
    if final["same"] > detector["same"] or final["any"] > detector["any"] or final_abs_count < detector_abs_count:
        return "positive_help"
    return "positive_same"


def artifact_paths(artifacts_dir: Path | None, case_id: str) -> dict[str, str]:
    if artifacts_dir is None:
        return {"artifact_json": "", "artifact_csv": "", "artifact_png": ""}
    resolved = resolve(artifacts_dir)
    return {
        "artifact_json": repo_path(resolved / f"{case_id}.json") if (resolved / f"{case_id}.json").exists() else "",
        "artifact_csv": repo_path(resolved / f"{case_id}.csv") if (resolved / f"{case_id}.csv").exists() else "",
        "artifact_png": repo_path(resolved / f"{case_id}.png") if (resolved / f"{case_id}.png").exists() else "",
    }


def summarize_case(case: dict[str, Any], artifacts_dir: Path | None) -> dict[str, Any]:
    case_id = str(case.get("caseId", ""))
    evaluation = case.get("evaluation", {}) if isinstance(case.get("evaluation"), dict) else {}
    debug = case.get("debug", {}) if isinstance(case.get("debug"), dict) else {}
    gt_count = int_value(evaluation.get("gtCount"))
    final = source_metrics(evaluation, "final")
    detector = source_metrics(evaluation, "clustered_detector" if source_exists(evaluation, "clustered_detector") else "detector")
    proposals = source_metrics(
        evaluation,
        "proposals_detector" if source_exists(evaluation, "proposals_detector") else "detector",
    )
    clustered = source_metrics(
        evaluation,
        "clustered_detector" if source_exists(evaluation, "clustered_detector") else "detector",
    )
    if final["pred"] == 0 and int_value(evaluation.get("predCount")):
        final = {
            "pred": int_value(evaluation.get("predCount")),
            "same": int_value(evaluation.get("matchedSameClass")),
            "any": int_value(evaluation.get("matchedAnyClass")),
            "count_error": int_value(evaluation.get("countError")),
        }
    debug_counts = {
        "detector_proposals": int_value(debug.get("detectorProposals", debug.get("proposals"))),
        "classified_proposals": int_value(debug.get("classifiedProposals", debug.get("classified"))),
        "clustered_proposals": int_value(debug.get("clusteredProposals")),
        "rejected_proposals": int_value(debug.get("rejectedProposals", debug.get("rejected"))),
        "rejected_after_nms": int_value(debug.get("rejectedAfterNms")),
    }
    effect = classify_effect(gt_count, final, detector, proposals, clustered, debug_counts)
    row: dict[str, Any] = {
        "case_id": case_id,
        "effect": effect,
        "gt_count": gt_count,
        "final_pred": final["pred"],
        "detector_pred": detector["pred"],
        "final_same": final["same"],
        "detector_same": detector["same"],
        "final_any": final["any"],
        "detector_any": detector["any"],
        "proposal_pred": proposals["pred"],
        "proposal_same": proposals["same"],
        "proposal_any": proposals["any"],
        "clustered_pred": clustered["pred"],
        "clustered_same": clustered["same"],
        "clustered_any": clustered["any"],
        "final_count_error": final["count_error"],
        "detector_count_error": detector["count_error"],
        "final_minus_detector_pred": final["pred"] - detector["pred"],
        "final_minus_detector_same": final["same"] - detector["same"],
        "final_minus_detector_any": final["any"] - detector["any"],
        "final_minus_detector_abs_count_error": abs(final["count_error"]) - abs(detector["count_error"]),
        **debug_counts,
        "notes": str(case.get("notes", "")),
    }
    row.update(artifact_paths(artifacts_dir, case_id))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "case_id",
        "effect",
        "gt_count",
        "final_pred",
        "detector_pred",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    report = read_report(args.report)
    rows = [
        summarize_case(case, args.artifacts_dir)
        for case in report.get("cases", [])
        if isinstance(case, dict)
    ]
    effect_counts = Counter(str(row["effect"]) for row in rows)
    payload = {
        "schema": "cashsnap_browser_gate_effects_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "report": repo_path(resolve(args.report)),
        "artifacts_dir": repo_path(resolve(args.artifacts_dir)) if args.artifacts_dir else "",
        "case_count": len(rows),
        "effect_counts": dict(sorted(effect_counts.items())),
        "rows": rows,
    }
    if args.csv_out:
        out = resolve(args.csv_out)
        write_csv(out, rows)
        print(f"wrote_csv={repo_path(out)}")
    if args.json_out:
        out = resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_json={repo_path(out)}")
    print(
        "browser_gate_effects "
        + " ".join(f"{key}={value}" for key, value in sorted(effect_counts.items()))
    )


if __name__ == "__main__":
    main()
