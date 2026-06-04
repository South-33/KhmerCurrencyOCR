#!/usr/bin/env python
"""Run the configured WebGL trainable-candidate recipe suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_trainable_candidates_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--only", action="append", default=[], help="Recipe id to run; can be repeated.")
    parser.add_argument("--skip-render", action="store_true", help="Repackage/check existing rendered variants.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--visual-scale", default=None, help="Override every recipe visual_scale. Defaults to each suite row, then 2.")
    parser.add_argument("--browser-executable", type=Path, default=None, help="Optional Chromium/Edge executable override.")
    parser.add_argument("--headroom-max-percent", default="90")
    parser.add_argument("--headroom-resume-percent", default="82")
    parser.add_argument("--headroom-max-ram-percent", default="90")
    parser.add_argument("--headroom-max-gpu-mem-percent", default="90")
    parser.add_argument("--min-free-ram-gb", default="3")
    parser.add_argument("--preflight-timeout", default="120")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict:
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing suite config: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def train_views_arg(row: dict) -> str:
    views = row.get("train_views", ["detect"])
    if isinstance(views, list):
        return ",".join(str(item) for item in views)
    return str(views)


def build_command(row: dict, args: argparse.Namespace) -> list[str]:
    visual_scale = args.visual_scale if args.visual_scale is not None else row.get("visual_scale", "2")
    cmd = [
        sys.executable,
        "scripts/run_webgl_recipe.py",
        "--recipe-id",
        str(row["recipe_id"]),
        "--out-root",
        str(row["out_root"]),
        "--start-variant",
        str(row["start_variant"]),
        "--count",
        str(row["count"]),
        "--scene-mode",
        str(row["scene_mode"]),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--visual-scale",
        str(visual_scale),
        "--asset-side-policy",
        str(row.get("asset_side_policy", "any")),
        "--camera-profile",
        str(row.get("camera_profile", "generic_phone_jitter")),
        "--artifact-status",
        "trainable-candidate",
        "--train-views",
        train_views_arg(row),
        "--headroom-max-percent",
        args.headroom_max_percent,
        "--headroom-resume-percent",
        args.headroom_resume_percent,
        "--headroom-max-ram-percent",
        args.headroom_max_ram_percent,
        "--headroom-max-gpu-mem-percent",
        args.headroom_max_gpu_mem_percent,
        "--min-free-ram-gb",
        args.min_free_ram_gb,
        "--preflight-timeout",
        args.preflight_timeout,
    ]
    if bool(row.get("allow_zero_visible")):
        cmd.append("--allow-zero-visible-trainable")
    if str(row.get("class_sequence", "")).strip():
        cmd.extend(["--class-sequence", str(row["class_sequence"])])
    if str(row.get("background_dir", "")).strip():
        cmd.extend(["--background-dir", str(row["background_dir"])])
    if args.browser_executable:
        cmd.extend(["--browser-executable", str(args.browser_executable)])
    if args.skip_render:
        cmd.append("--skip-render")
    return cmd


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    suite_path = resolve(args.suite)
    subprocess.run([sys.executable, "scripts/check_webgl_trainable_candidate_suite.py", "--suite", str(suite_path)], cwd=ROOT, check=True)
    suite = read_json(suite_path)
    rows = suite.get("recipes", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("suite recipes must be a non-empty list")

    selected = set(args.only)
    ran = 0
    known_ids = {str(row.get("recipe_id", "")) for row in rows if isinstance(row, dict)}
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("suite recipe rows must be objects")
        recipe_id = str(row.get("recipe_id", ""))
        if selected and recipe_id not in selected:
            continue
        cmd = build_command(row, args)
        run(cmd, args.dry_run)
        ran += 1

    if selected and ran != len(selected):
        missing = sorted(selected - known_ids)
        raise SystemExit(f"selected recipe(s) not found in suite: {missing}")
    print(f"ok: {'planned' if args.dry_run else 'ran'} {ran} WebGL trainable-candidate recipe(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
