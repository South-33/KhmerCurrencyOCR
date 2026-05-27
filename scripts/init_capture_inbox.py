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


def scene_folders(requirements: Path, include_optional: bool) -> list[str]:
    folders: list[str] = []
    with requirements.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("match_column") != "scene_type":
                continue
            if not include_optional and not is_required(row):
                continue
            folder = row.get("match_value", "").strip()
            if folder and folder not in folders:
                folders.append(folder)
    return folders


def main() -> None:
    args = parse_args()
    requirements = resolve(args.requirements)
    out_dir = resolve(args.out_dir)
    folders = scene_folders(requirements, args.include_optional)
    print(f"out_dir={repo_path(out_dir)} folders={len(folders)}")
    for folder in folders:
        path = out_dir / folder
        print(repo_path(path))
        if not args.dry_run:
            path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
