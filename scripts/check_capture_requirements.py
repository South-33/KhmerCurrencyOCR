from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image

try:
    from capture_artifact_guard import derived_capture_reason
except ModuleNotFoundError:  # Allows importing this script as scripts.check_capture_requirements.
    from scripts.capture_artifact_guard import derived_capture_reason


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "manifests" / "real_partial_capture_inventory.csv"
DEFAULT_REQUIREMENTS = ROOT / "manifests" / "real_partial_capture_requirements.csv"
DEFAULT_INBOX = ROOT / "data" / "inbox" / "real_partial_photos"
DEFAULT_READINESS = ROOT / "runs" / "cashsnap" / "synthetic_pipeline_readiness_latest.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check real partial-note capture inventory against CashSnap needs.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX, help="Inbox folder to scan for unregistered images.")
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS, help="Optional readiness JSON used to annotate mined-candidate gaps.")
    parser.add_argument("--gate-review-targets", type=Path, default=None, help="Optional browser gate review target CSV used to annotate shot-list rows.")
    parser.add_argument("--json-out", type=Path, help="Optional machine-readable gap report path.")
    parser.add_argument("--shot-list-out", type=Path, help="Optional Markdown shot list for the missing capture set.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when requirements are not met.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_optional_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    resolved = resolve(path)
    if not resolved.exists():
        return []
    with resolved.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_optional_json(path: Path) -> dict[str, object]:
    resolved = resolve(path)
    if not resolved.exists():
        return {}
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_path(resolved)}: expected JSON object")
    return data


def split_values(value: str) -> set[str]:
    return {part.strip() for part in value.replace(";", ",").split(",") if part.strip()}


def row_matches(row: dict[str, str], column: str, value: str) -> bool:
    if column == "denominations":
        return value in split_values(row.get(column, ""))
    return row.get(column, "").strip() == value


def drop_folder_for_requirement(req: dict[str, str], inbox: Path) -> str:
    if req["match_column"] == "scene_type":
        return repo_path(resolve(inbox) / req["match_value"])
    return ""


def hint_for_requirement(req: dict[str, str], inbox: Path) -> str:
    drop_folder = drop_folder_for_requirement(req, inbox)
    return f"drop_folder={drop_folder}" if drop_folder else ""


def priority_value(req: dict[str, str]) -> int:
    try:
        return int(req.get("priority", "5") or "5")
    except ValueError:
        return 5


def mined_candidate_gap_scenes(readiness: dict[str, object]) -> dict[str, list[str]]:
    gaps = readiness.get("real_dataset_candidate_hint_gaps", [])
    if not isinstance(gaps, list):
        return {}
    by_scene: dict[str, list[str]] = {}
    for row in gaps:
        if not isinstance(row, dict):
            continue
        scene_type = str(row.get("scene_type", "")).strip()
        condition_id = str(row.get("condition_id", "")).strip()
        if not scene_type:
            continue
        by_scene.setdefault(scene_type, [])
        if condition_id and condition_id not in by_scene[scene_type]:
            by_scene[scene_type].append(condition_id)
    return {scene_type: sorted(condition_ids) for scene_type, condition_ids in sorted(by_scene.items())}


def append_note(notes: str, extra: str) -> str:
    notes = notes.strip()
    return f"{notes} {extra}" if notes else extra


def priority_int(value: str) -> int:
    try:
        return int(value or "9")
    except ValueError:
        return 9


def short_text(value: str, limit: int = 72) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."


def format_gate_target(row: dict[str, str]) -> str:
    priority = priority_int(row.get("priority", ""))
    source = row.get("source", "").strip()
    effect = row.get("effect", "").strip()
    case_id = short_text(row.get("case_id", ""))
    return f"P{priority} {source} {effect} `{case_id}`".strip()


def gate_targets_by_requirement(path: Path | None) -> dict[str, list[dict[str, str]]]:
    targets: dict[str, list[dict[str, str]]] = {}
    for row in read_optional_csv(path):
        requirement_id = row.get("capture_requirement", "").strip()
        if not requirement_id:
            continue
        targets.setdefault(requirement_id, []).append(row)
    return {
        requirement_id: sorted(rows, key=lambda row: (priority_int(row.get("priority", "")), row.get("case_id", "")))
        for requirement_id, rows in targets.items()
    }


def image_ok(row: dict[str, str]) -> tuple[bool, str]:
    local_path = row.get("local_path", "").strip()
    if not local_path:
        return False, "missing local_path"
    derived_reason = derived_capture_reason(local_path)
    if derived_reason:
        return False, derived_reason
    path = resolve(Path(local_path))
    if not path.exists():
        return False, f"missing file {local_path}"
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        return False, f"unreadable image {local_path}: {exc}"
    if row.get("rights_status", "").strip().lower() not in {"own_photo", "rights_clear", "public_domain", "cc0"}:
        return False, "rights_status not marked usable"
    return True, ""


def inbox_images(inbox: Path) -> list[str]:
    inbox = resolve(inbox)
    if not inbox.exists():
        return []
    return sorted(
        repo_path(path)
        for path in inbox.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_shot_list(path: Path, reports: list[dict[str, object]], inbox: Path) -> None:
    rows = sorted(
        reports,
        key=lambda row: (
            int(row["priority"]),
            not bool(row["required"]),
            int(row["missing"]) == 0,
            str(row["requirement_id"]),
        ),
    )
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    inbox_path = repo_path(resolve(inbox))
    lines = [
        "# CashSnap Real Capture Shot List",
        "",
        "Create or refresh the capture folders and guide files with:",
        "",
        "`rl python scripts/init_capture_inbox.py --write-guides`",
        "",
        "Drop raw, unannotated phone photos into the listed folders; do not use screenshots, prediction overlays, contact sheets, synthetic renders, or edited review crops.",
        "",
        "Then register them with:",
        "",
        f"`rl python scripts/register_capture_photos.py --images-dir {inbox_path} --recursive --scene-type-from-parent --dry-run`",
        f"`rl python scripts/register_capture_photos.py --images-dir {inbox_path} --recursive --scene-type-from-parent`",
        "",
        "| Priority | Requirement | Need | Drop Folder | Notes |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        missing = int(row["missing"])
        status = "done" if missing == 0 else f"{missing} more"
        drop_folder = str(row.get("drop_folder", ""))
        drop_text = f"`{drop_folder}`" if drop_folder else "any registered scene; set `denominations`"
        notes = str(row.get("notes", "")).replace("|", "/")
        lines.append(
            f"| P{row['priority']} | `{row['requirement_id']}` | {status} | {drop_text} | {notes} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote_shot_list={repo_path(out)}")


def main() -> None:
    args = parse_args()
    rows = read_csv(args.inventory)
    requirements = read_csv(args.requirements)
    readiness = read_optional_json(args.readiness)
    gate_review_targets = gate_targets_by_requirement(args.gate_review_targets)
    mined_gap_scenes = mined_candidate_gap_scenes(readiness)
    registered_paths = {row.get("local_path", "").strip() for row in rows}
    unregistered_inbox = [path for path in inbox_images(args.inbox) if path not in registered_paths]
    usable_rows: list[dict[str, str]] = []
    errors: list[str] = []
    for row in rows:
        ok, error = image_ok(row)
        if ok:
            usable_rows.append(row)
        else:
            errors.append(f"{row.get('image_id', '<missing image_id>')}: {error}")

    unmet = False
    print(f"inventory_rows={len(rows)} usable_rows={len(usable_rows)} unregistered_inbox_images={len(unregistered_inbox)}")
    gaps: list[tuple[int, int, dict[str, str], int]] = []
    requirement_reports: list[dict[str, object]] = []
    for req in requirements:
        count = sum(1 for row in usable_rows if row_matches(row, req["match_column"], req["match_value"]))
        minimum = int(req["min_images"])
        required = req.get("required", "yes").strip().lower() not in {"0", "false", "no", "optional"}
        status = "ok" if count >= minimum else ("missing" if required else "optional_missing")
        unmet = unmet or (required and count < minimum)
        if count < minimum:
            gaps.append((priority_value(req), minimum - count, req, count))
        hint = hint_for_requirement(req, args.inbox)
        drop_folder = drop_folder_for_requirement(req, args.inbox)
        mined_gap_conditions = (
            mined_gap_scenes.get(req["match_value"].strip(), []) if req.get("match_column", "").strip() == "scene_type" else []
        )
        notes = req.get("notes", "")
        if mined_gap_conditions and count < minimum:
            notes = append_note(
                notes,
                "Current mined real scan has zero candidate hints for this scene; fresh phone capture is mandatory for the bridge.",
            )
        review_targets = gate_review_targets.get(req["requirement_id"], [])
        if review_targets and count < minimum:
            examples = "; ".join(format_gate_target(row) for row in review_targets[:3])
            if len(review_targets) > 3:
                examples += f"; +{len(review_targets) - 3} more"
            notes = append_note(notes, f"Browser gate review targets: {examples}.")
        suffix = f" ({hint})" if hint and count < minimum else ""
        priority = req.get("priority", "").strip()
        priority_text = f" priority={priority}" if priority else ""
        print(f"{status}: {req['requirement_id']} {count}/{minimum}{priority_text} - {req['description']}{suffix}")
        requirement_reports.append(
            {
                "requirement_id": req["requirement_id"],
                "description": req["description"],
                "match_column": req["match_column"],
                "match_value": req["match_value"],
                "required": required,
                "priority": priority_value(req),
                "minimum": minimum,
                "count": count,
                "missing": max(0, minimum - count),
                "status": status,
                "hint": hint,
                "drop_folder": drop_folder,
                "notes": notes,
                "mined_candidate_hint_gap": bool(mined_gap_conditions),
                "mined_candidate_gap_conditions": mined_gap_conditions,
                "gate_review_targets": review_targets,
            }
        )
    if gaps:
        print("Next capture priorities:")
        for priority, missing, req, count in sorted(gaps, key=lambda item: (item[0], item[2]["required"] != "yes", -item[1], item[2]["requirement_id"]))[:6]:
            hint = hint_for_requirement(req, args.inbox)
            suffix = f" ({hint})" if hint else ""
            print(f"- P{priority} {req['requirement_id']}: need {missing} more, have {count}{suffix}")
    if errors:
        print("Inventory issues:")
        for error in errors:
            print(f"- {error}")
    if unregistered_inbox:
        print("Unregistered inbox images:")
        for path in unregistered_inbox[:20]:
            print(f"- {path}")
        if len(unregistered_inbox) > 20:
            print(f"- ... {len(unregistered_inbox) - 20} more")
        inbox_path = repo_path(resolve(args.inbox))
        print("Register dropped photos before trusting requirement counts:")
        print(f"rl python scripts/register_capture_photos.py --images-dir {inbox_path} --recursive --scene-type-from-parent --dry-run")
        print(f"rl python scripts/register_capture_photos.py --images-dir {inbox_path} --recursive --scene-type-from-parent")
    if args.json_out:
        report = {
            "inventory_rows": len(rows),
            "usable_rows": len(usable_rows),
            "unregistered_inbox_images": unregistered_inbox,
            "requirements": requirement_reports,
            "next_capture_priorities": [
                {
                    "priority": priority,
                    "requirement_id": req["requirement_id"],
                    "description": req["description"],
                    "missing": missing,
                    "count": count,
                    "hint": hint_for_requirement(req, args.inbox),
                    "drop_folder": drop_folder_for_requirement(req, args.inbox),
                }
                for priority, missing, req, count in sorted(gaps, key=lambda item: (item[0], item[2]["required"] != "yes", -item[1], item[2]["requirement_id"]))[:6]
            ],
            "inventory_issues": errors,
        }
        out_path = resolve(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote_json={repo_path(out_path)}")
    if args.shot_list_out:
        write_shot_list(args.shot_list_out, requirement_reports, args.inbox)
    if args.strict and (unmet or errors or unregistered_inbox):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
