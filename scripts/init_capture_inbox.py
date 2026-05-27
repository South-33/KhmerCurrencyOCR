from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = ROOT / "manifests" / "real_partial_capture_requirements.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "inbox" / "real_partial_photos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CashSnap real-capture inbox folders from requirements.")
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--include-optional", action="store_true", help="Also create optional scene folders.")
    parser.add_argument("--write-guides", action="store_true", help="Write a short .capture_guide.txt in each folder.")
    parser.add_argument("--dry-run", action="store_true", help="Print folders without creating them.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def is_required(row: dict[str, str]) -> bool:
    return row.get("required", "yes").strip().lower() not in {"0", "false", "no", "optional"}


def priority_value(row: dict[str, str]) -> int:
    try:
        return int(row.get("priority", "5") or "5")
    except ValueError:
        return 5


def scene_requirements(requirements: Path, include_optional: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with requirements.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("match_column") != "scene_type":
                continue
            if not include_optional and not is_required(row):
                continue
            folder = row.get("match_value", "").strip()
            if folder and folder not in seen:
                rows.append(row)
                seen.add(folder)
    return sorted(rows, key=lambda row: (priority_value(row), row.get("required", "yes").strip().lower() not in {"yes", "1", "true"}, row["match_value"]))


def denomination_hint(scene_type: str) -> str:
    lower = scene_type.lower()
    if lower.startswith("thin_slice_khr_"):
        return f" --denominations \"KHR_{lower.rsplit('_', 1)[-1]}\""
    if lower == "single_khr":
        return " --denominations \"KHR_...\""
    if lower == "mixed_usd_khr":
        return " --denominations \"KHR_...;USD_...\""
    return ""


def guide_text(row: dict[str, str], folder_path: Path) -> str:
    scene_type = row["match_value"].strip()
    register_hint = (
        "lr python scripts/register_capture_photos.py "
        f"--images-dir {repo_path(folder_path)} --scene-type {scene_type}"
        f"{denomination_hint(scene_type)} --dry-run"
    )
    return "\n".join(
        [
            f"CashSnap capture inbox: {scene_type}",
            f"Requirement: {row['description']}",
            f"Minimum usable images: {row['min_images']}",
            f"Priority: P{priority_value(row)}",
            f"Notes: {row.get('notes', '').strip()}",
            "",
            "Drop rights-clear phone photos for this scene in this folder.",
            "Keep faces, IDs, cards, receipts, screens, and location details out of frame.",
            "",
            "Registration dry run:",
            register_hint,
            "",
        ]
    )


def root_guide_text(rows: list[dict[str, str]], out_dir: Path) -> str:
    folder_lines = [f"- P{priority_value(row)} {row['match_value']}: {row['description']} ({row['min_images']} usable)" for row in rows]
    return "\n".join(
        [
            "CashSnap real partial-photo inbox",
            "",
            "Drop rights-clear phone photos into the scene folders below.",
            "Keep faces, IDs, cards, receipts, screens, and location details out of frame.",
            "",
            "Folders:",
            *folder_lines,
            "",
            "After adding photos, register them with:",
            f"lr python scripts/register_capture_photos.py --images-dir {repo_path(out_dir)} --recursive --scene-type-from-parent --dry-run",
            f"lr python scripts/register_capture_photos.py --images-dir {repo_path(out_dir)} --recursive --scene-type-from-parent",
            "Thin-slice folder names auto-fill their KHR denomination during registration.",
            "",
            "Then check remaining gaps with:",
            "lr python scripts/check_capture_requirements.py",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    requirements = resolve(args.requirements)
    out_dir = resolve(args.out_dir)
    rows = scene_requirements(requirements, args.include_optional)
    print(f"out_dir={repo_path(out_dir)} folders={len(rows)}")
    if not args.dry_run and args.write_guides:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ".capture_guide.txt").write_text(root_guide_text(rows, out_dir), encoding="utf-8")
    for row in rows:
        folder = row["match_value"].strip()
        path = out_dir / folder
        print(repo_path(path))
        if not args.dry_run:
            path.mkdir(parents=True, exist_ok=True)
            if args.write_guides:
                (path / ".capture_guide.txt").write_text(guide_text(row, path), encoding="utf-8")


if __name__ == "__main__":
    main()
