from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "manifests" / "real_partial_capture_inventory.csv"
DEFAULT_REQUIREMENTS = ROOT / "manifests" / "real_partial_capture_requirements.csv"
DEFAULT_INBOX = ROOT / "data" / "inbox" / "real_partial_photos"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check real partial-note capture inventory against CashSnap needs.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX, help="Inbox folder to scan for unregistered images.")
    parser.add_argument("--json-out", type=Path, help="Optional machine-readable gap report path.")
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


def split_values(value: str) -> set[str]:
    return {part.strip() for part in value.replace(";", ",").split(",") if part.strip()}


def row_matches(row: dict[str, str], column: str, value: str) -> bool:
    if column == "denominations":
        return value in split_values(row.get(column, ""))
    return row.get(column, "").strip() == value


def hint_for_requirement(req: dict[str, str]) -> str:
    if req["match_column"] == "scene_type":
        return f"drop_folder={(DEFAULT_INBOX / req['match_value']).relative_to(ROOT).as_posix()}"
    return ""


def priority_value(req: dict[str, str]) -> int:
    try:
        return int(req.get("priority", "5") or "5")
    except ValueError:
        return 5


def image_ok(row: dict[str, str]) -> tuple[bool, str]:
    local_path = row.get("local_path", "").strip()
    if not local_path:
        return False, "missing local_path"
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


def main() -> None:
    args = parse_args()
    rows = read_csv(args.inventory)
    requirements = read_csv(args.requirements)
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
        hint = hint_for_requirement(req)
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
            }
        )
    if gaps:
        print("Next capture priorities:")
        for priority, missing, req, count in sorted(gaps, key=lambda item: (item[0], item[2]["required"] != "yes", -item[1], item[2]["requirement_id"]))[:6]:
            hint = hint_for_requirement(req)
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
        print(f"lr python scripts/register_capture_photos.py --images-dir {inbox_path} --recursive --scene-type-from-parent --dry-run")
        print(f"lr python scripts/register_capture_photos.py --images-dir {inbox_path} --recursive --scene-type-from-parent")
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
                    "hint": hint_for_requirement(req),
                }
                for priority, missing, req, count in sorted(gaps, key=lambda item: (item[0], item[2]["required"] != "yes", -item[1], item[2]["requirement_id"]))[:6]
            ],
            "inventory_issues": errors,
        }
        out_path = resolve(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote_json={repo_path(out_path)}")
    if args.strict and (unmet or errors or unregistered_inbox):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
