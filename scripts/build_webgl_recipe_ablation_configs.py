#!/usr/bin/env python
"""Build balanced real+single-WebGL-recipe YOLO ablation configs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "configs" / "cashsnap_v1_plus_webgl_trainable_candidates.yaml"
DEFAULT_SUITE = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_trainable_candidates_v1.json"
DEFAULT_OUT_DIR = ROOT / "configs" / "webgl_ablation"
DEFAULT_LIST_DIR = ROOT / "configs" / "generated_lists" / "webgl_ablation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--list-dir", type=Path, default=DEFAULT_LIST_DIR)
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--backgrounds", type=int, default=24)
    parser.add_argument("--include-real-only", action="store_true", default=True)
    parser.add_argument("--no-real-only", action="store_false", dest="include_real_only")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def read_suite(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("recipes", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"suite has no recipes: {path}")
    return rows


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def build_command(
    data_path: Path,
    out_path: Path,
    train_list: Path,
    per_class: int,
    backgrounds: int,
    prefix: str | None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/build_yolo_balanced_subset.py",
        "--data",
        rel(data_path),
        "--out",
        rel(out_path),
        "--train-list",
        rel(train_list),
        "--per-class",
        str(per_class),
        "--backgrounds",
        str(backgrounds),
    ]
    if prefix is None:
        command.append("--no-always-include")
    else:
        command.extend(["--always-include-prefix", prefix])
    return command


def main() -> int:
    args = parse_args()
    data_path = resolve(args.data)
    suite_path = resolve(args.suite)
    out_dir = resolve(args.out_dir)
    list_dir = resolve(args.list_dir)
    rows = read_suite(suite_path)

    if args.include_real_only:
        stem = "cashsnap_v1_balanced_real_only_probe"
        run(
            build_command(
                data_path=data_path,
                out_path=out_dir / f"{stem}.yaml",
                train_list=list_dir / f"{stem}_train.txt",
                per_class=args.per_class,
                backgrounds=args.backgrounds,
                prefix=None,
            ),
            args.dry_run,
        )

    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("suite recipe rows must be objects")
        recipe_id = str(row["recipe_id"])
        out_root = str(row["out_root"]).replace("\\", "/").strip("/")
        stem = f"cashsnap_v1_plus_{slug(recipe_id)}_probe"
        run(
            build_command(
                data_path=data_path,
                out_path=out_dir / f"{stem}.yaml",
                train_list=list_dir / f"{stem}_train.txt",
                per_class=args.per_class,
                backgrounds=args.backgrounds,
                prefix=f"{out_root}/",
            ),
            args.dry_run,
        )

    print(f"ok: generated {len(rows) + int(args.include_real_only)} ablation config command(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
