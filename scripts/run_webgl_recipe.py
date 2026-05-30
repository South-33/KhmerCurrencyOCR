#!/usr/bin/env python
"""Run a named WebGL synthetic recipe from the recipe catalog."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_recipe_catalog_v1.json"
RUNNABLE_SCENE_MODES = {"auto", "clean", "negative", "stack", "fan", "thin_edge", "hand_occlusion", "qa3"}
STATUS_TO_BATCH_STATUS = {
    "planned": "diagnostic",
    "smoke_ready": "smoke",
    "label_policy_ready": "diagnostic",
    "diagnostic": "diagnostic",
    "trainable-candidate": "trainable-candidate",
    "promoted": "trainable-candidate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--recipe-id", required=True)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--start-variant", type=int, default=0)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--scene-mode", default="", help="Override the first runnable scene mode from the catalog.")
    parser.add_argument("--artifact-status", choices=["smoke", "diagnostic", "trainable-candidate"], default="")
    parser.add_argument("--background-dir", type=Path, default=None)
    parser.add_argument("--min-free-ram-gb", default="3")
    parser.add_argument("--preflight-timeout", default="120")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-yolo-check", action="store_true")
    parser.add_argument("--skip-label-view-check", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing catalog: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def find_recipe(catalog: dict, recipe_id: str) -> dict:
    for row in catalog.get("recipes", []):
        if row.get("id") == recipe_id:
            return row
    raise SystemExit(f"recipe not found: {recipe_id}")


def choose_scene_mode(recipe: dict, override: str) -> str:
    if override:
        if override not in RUNNABLE_SCENE_MODES:
            raise SystemExit(f"--scene-mode must be one of {sorted(RUNNABLE_SCENE_MODES)}")
        return override
    for scene_mode in recipe.get("scene_modes", []):
        if scene_mode in RUNNABLE_SCENE_MODES:
            return str(scene_mode)
    raise SystemExit(f"{recipe['id']}: no runnable scene_modes; current modes={recipe.get('scene_modes', [])}")


def main() -> int:
    args = parse_args()
    catalog = read_json(resolve(args.catalog))
    recipe = find_recipe(catalog, args.recipe_id)
    scene_mode = choose_scene_mode(recipe, args.scene_mode)
    artifact_status = args.artifact_status or STATUS_TO_BATCH_STATUS.get(str(recipe.get("artifact_status")), "diagnostic")
    out_root = args.out_root or Path("data") / "synthetic" / f"{slug(args.recipe_id)}_v{args.start_variant}_{args.start_variant + args.count - 1}"

    cmd = [
        sys.executable,
        "scripts/render_webgl_variant_batch.py",
        "--out-root",
        str(out_root),
        "--start-variant",
        str(args.start_variant),
        "--count",
        str(args.count),
        "--scene-mode",
        scene_mode,
        "--recipe-name",
        args.recipe_id,
        "--artifact-status",
        artifact_status,
        "--intended-use",
        str(recipe.get("intended_use", "")),
        "--notes",
        f"promotion_gate={recipe.get('promotion_gate', '')}; current_blocker={recipe.get('current_blocker', '')}",
        "--min-free-ram-gb",
        args.min_free_ram_gb,
        "--preflight-timeout",
        args.preflight_timeout,
    ]
    if args.background_dir:
        cmd.extend(["--background-dir", str(args.background_dir)])
    if args.skip_render:
        cmd.append("--skip-render")
    if args.skip_yolo_check:
        cmd.append("--skip-yolo-check")
    if args.skip_label_view_check:
        cmd.append("--skip-label-view-check")

    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
