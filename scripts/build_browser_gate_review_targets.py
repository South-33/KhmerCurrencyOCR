from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = ROOT / "manifests" / "real_partial_capture_requirements.csv"
DEFAULT_INBOX = ROOT / "data" / "inbox" / "real_partial_photos"

NEUTRAL_EFFECTS = {"positive_same", "hard_negative_same"}
EFFECT_PRIORITY = {
    "positive_reject_harm": 1,
    "positive_nms_harm": 1,
    "hard_negative_harm": 1,
    "positive_harm": 2,
    "positive_reject_safe": 3,
    "hard_negative_help": 3,
    "positive_help": 4,
    "positive_same": 5,
    "hard_negative_same": 5,
}
EFFECT_ACTION = {
    "positive_reject_harm": "Review rejected proposal crops; collect protected real hard positives before loosening the gate.",
    "positive_nms_harm": "Review proposal clustering and duplicate suppression; keep this as a fusion/calibration target.",
    "hard_negative_harm": "Collect matching real no-note or prop negatives before trusting the detector/gate stack.",
    "positive_harm": "Review detector-vs-final deltas; protect this case before changing proposal calibration.",
    "positive_reject_safe": "Keep as calibration-safe evidence; it shows rejection can remove duplicates without losing count.",
    "hard_negative_help": "Keep as hard-negative safety evidence; match it with real no-note/prop captures before promotion.",
    "positive_help": "Keep as positive calibration evidence.",
    "positive_same": "No immediate action; retain as regression coverage.",
    "hard_negative_same": "No immediate action; retain as hard-negative regression coverage.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert browser gate-effect rows into ranked review and real-capture targets."
    )
    parser.add_argument(
        "--gate-effects",
        action="append",
        type=Path,
        default=[],
        help="CSV written by summarize_browser_gate_effects.py. Repeatable.",
    )
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--include-neutral", action="store_true", help="Also include positive_same/hard_negative_same rows.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum rows to write after ranking; 0 means all.")
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    resolved = resolve(path)
    try:
        return resolved.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved.resolve())


def read_csv(path: Path) -> list[dict[str, str]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def int_value(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def infer_source(path: Path) -> str:
    stem = resolve(path).stem.lower()
    if "mined_real" in stem:
        return "mined_real"
    if "synthetic" in stem:
        return "synthetic"
    return stem.replace("browser_gate_effects_", "")


def load_requirements(path: Path) -> dict[str, dict[str, str]]:
    requirements: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        requirement_id = row.get("requirement_id", "").strip()
        if requirement_id:
            requirements[requirement_id] = row
    return requirements


def drop_folder(requirement: dict[str, str] | None, inbox: Path) -> str:
    if not requirement:
        return ""
    if requirement.get("match_column", "").strip() != "scene_type":
        return ""
    match_value = requirement.get("match_value", "").strip()
    if not match_value:
        return ""
    return repo_path(resolve(inbox) / match_value)


def combined_text(row: dict[str, str]) -> str:
    return f"{row.get('case_id', '')} {row.get('notes', '')}".lower()


def suggest_requirement(row: dict[str, str], source: str) -> str:
    text = combined_text(row)
    effect = row.get("effect", "").strip()

    if effect.startswith("hard_negative"):
        if "no_note" in text or "empty" in text or "background" in text:
            return "no_note_background"
        return "non_banknote_paper_props"

    if "same_class_fan" in text or "same_denomination_fan" in text:
        return "same_denomination_fan"
    if "hand_fan" in text or "_fan_" in text or " fan " in text:
        return "hand_fan"
    if "khr_5000_face_number_overlap" in text:
        return "khr_5000_face_number_overlap"
    if "thin_slice_khr_20000" in text:
        return "thin_slice_khr_20000"
    if "thin_slice_khr_5000" in text:
        return "thin_slice_khr_5000"
    if "partial_off_frame" in text or "off-frame" in text:
        return "partial_off_frame"
    if "weak_khr_50000" in text:
        return "khr_50000_hard_positive_partials"
    if "weak_khr_20000" in text:
        return "weak_khr_20000"
    if "mixed_usd_khr" in text:
        return "mixed_usd_khr_rare_common_stack"
    if "usd_" in text and ("partial" in text or "edge" in text or "reject" in effect):
        return "usd_hard_positive_partials"
    if "khr_50000" in text and ("partial" in text or "edge" in text or "weak" in text or "reject" in effect):
        return "khr_50000_hard_positive_partials"
    if "khr_20000" in text and source == "mined_real":
        return "weak_khr_20000"
    return ""


def target_priority(row: dict[str, str], source: str) -> int:
    effect = row.get("effect", "").strip()
    priority = EFFECT_PRIORITY.get(effect, 4)
    if source == "mined_real" and priority > 1:
        priority -= 1
    if abs(int_value(row.get("final_count_error"))) >= 3 and priority > 1:
        priority -= 1
    return max(1, min(priority, 5))


def should_include(row: dict[str, str], include_neutral: bool) -> bool:
    if include_neutral:
        return True
    return row.get("effect", "").strip() not in NEUTRAL_EFFECTS


def build_target(
    row: dict[str, str],
    source: str,
    requirements: dict[str, dict[str, str]],
    inbox: Path,
) -> dict[str, Any]:
    requirement_id = suggest_requirement(row, source)
    requirement = requirements.get(requirement_id)
    effect = row.get("effect", "").strip()
    rejected = int_value(row.get("rejected_proposals")) + int_value(row.get("rejected_after_nms"))
    abs_count_error = abs(int_value(row.get("final_count_error")))
    target = {
        "priority": target_priority(row, source),
        "source": source,
        "case_id": row.get("case_id", ""),
        "effect": effect,
        "review_action": EFFECT_ACTION.get(effect, "Review this case before changing browser proposal calibration."),
        "capture_requirement": requirement_id,
        "capture_description": requirement.get("description", "") if requirement else "",
        "drop_folder": drop_folder(requirement, inbox),
        "requirement_priority": requirement.get("priority", "") if requirement else "",
        "gt_count": int_value(row.get("gt_count")),
        "final_pred": int_value(row.get("final_pred")),
        "proposal_pred": int_value(row.get("proposal_pred")),
        "clustered_pred": int_value(row.get("clustered_pred")),
        "detector_pred": int_value(row.get("detector_pred")),
        "final_count_error": int_value(row.get("final_count_error")),
        "abs_count_error": abs_count_error,
        "rejected_total": rejected,
        "final_minus_detector_same": int_value(row.get("final_minus_detector_same")),
        "final_minus_detector_any": int_value(row.get("final_minus_detector_any")),
        "final_minus_detector_abs_count_error": int_value(row.get("final_minus_detector_abs_count_error")),
        "artifact_png": row.get("artifact_png", ""),
        "artifact_csv": row.get("artifact_csv", ""),
        "artifact_json": row.get("artifact_json", ""),
        "notes": row.get("notes", ""),
    }
    target["score"] = (
        (6 - int(target["priority"])) * 100
        + abs_count_error * 10
        + rejected
        + (10 if source == "mined_real" else 0)
    )
    return target


def sort_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        targets,
        key=lambda row: (
            int(row["priority"]),
            str(row["source"]) != "mined_real",
            -int(row["abs_count_error"]),
            -int(row["rejected_total"]),
            str(row["case_id"]),
        ),
    )


def clean_table(value: Any) -> str:
    text = str(value).replace("\n", " ").replace("|", "/").strip()
    return text


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "priority",
        "source",
        "case_id",
        "effect",
        "review_action",
        "capture_requirement",
        "capture_description",
        "drop_folder",
        "requirement_priority",
        "gt_count",
        "final_pred",
        "proposal_pred",
        "clustered_pred",
        "detector_pred",
        "final_count_error",
        "abs_count_error",
        "rejected_total",
        "final_minus_detector_same",
        "final_minus_detector_any",
        "final_minus_detector_abs_count_error",
        "artifact_png",
        "artifact_csv",
        "artifact_json",
        "notes",
        "score",
    ]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote_csv={repo_path(out)}")


def write_markdown(path: Path, rows: list[dict[str, Any]], source_paths: list[Path]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    effect_counts = Counter(str(row["effect"]) for row in rows)
    priority_counts = Counter(f"P{row['priority']}" for row in rows)
    lines = [
        "# CashSnap Browser Gate Review Targets",
        "",
        "These rows convert browser gate-effect diagnostics into review and capture targets.",
        "",
        "Source gate-effect CSVs:",
        *[f"- `{repo_path(path)}`" for path in source_paths],
        "",
        "Effect counts: "
        + (", ".join(f"{key}={value}" for key, value in sorted(effect_counts.items())) or "none"),
        "Priority counts: "
        + (", ".join(f"{key}={value}" for key, value in sorted(priority_counts.items())) or "none"),
        "",
        "| Priority | Source | Effect | Case | Capture Target | Error | Rejected | Artifact | Action |",
        "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        artifact = row.get("artifact_png") or row.get("artifact_json") or row.get("artifact_csv") or ""
        artifact_text = f"`{artifact}`" if artifact else ""
        capture = row.get("capture_requirement") or "review_only"
        if row.get("drop_folder"):
            capture = f"`{capture}` -> `{row['drop_folder']}`"
        else:
            capture = f"`{capture}`"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"P{row['priority']}",
                    clean_table(row["source"]),
                    clean_table(row["effect"]),
                    f"`{clean_table(row['case_id'])}`",
                    capture,
                    str(row["final_count_error"]),
                    str(row["rejected_total"]),
                    artifact_text,
                    clean_table(row["review_action"]),
                ]
            )
            + " |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote_md={repo_path(out)}")


def main() -> None:
    args = parse_args()
    if not args.gate_effects:
        raise SystemExit("--gate-effects is required")

    requirements = load_requirements(args.requirements)
    targets: list[dict[str, Any]] = []
    for path in args.gate_effects:
        source = infer_source(path)
        for row in read_csv(path):
            if not should_include(row, args.include_neutral):
                continue
            targets.append(build_target(row, source, requirements, args.inbox))
    targets = sort_targets(targets)
    if args.limit > 0:
        targets = targets[: args.limit]

    payload = {
        "schema": "cashsnap_browser_gate_review_targets_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_gate_effects": [repo_path(path) for path in args.gate_effects],
        "target_count": len(targets),
        "effect_counts": dict(sorted(Counter(str(row["effect"]) for row in targets).items())),
        "priority_counts": dict(sorted(Counter(f"P{row['priority']}" for row in targets).items())),
        "targets": targets,
    }
    if args.csv_out:
        write_csv(args.csv_out, targets)
    if args.md_out:
        write_markdown(args.md_out, targets, [resolve(path) for path in args.gate_effects])
    if args.json_out:
        out = resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_json={repo_path(out)}")
    print(
        "browser_gate_review_targets="
        + str(len(targets))
        + " "
        + " ".join(f"{key}={value}" for key, value in sorted(payload["effect_counts"].items()))
    )


if __name__ == "__main__":
    main()
