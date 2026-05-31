#!/usr/bin/env python
"""Run a named WebGL synthetic recipe from the recipe catalog."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from webgl_constants import WEBGL_ASSET_SIDE_POLICIES, WEBGL_CAMERA_PROFILES


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
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--visual-scale", default="2", help="Visual WebGL supersampling scale.")
    parser.add_argument("--browser-executable", type=Path, default=None, help="Optional Chromium/Edge executable override.")
    parser.add_argument("--asset-side-policy", default="", help="Override catalog asset-side sampling policy.")
    parser.add_argument("--camera-profile", default="", help="Override catalog WebGL camera/FOV/framing profile.")
    parser.add_argument("--artifact-status", choices=["smoke", "diagnostic", "trainable-candidate"], default="")
    parser.add_argument("--background-dir", type=Path, default=None)
    parser.add_argument("--headroom-max-percent", default="90")
    parser.add_argument("--headroom-resume-percent", default="82")
    parser.add_argument("--headroom-max-ram-percent", default="90")
    parser.add_argument("--headroom-max-gpu-mem-percent", default="90")
    parser.add_argument("--min-free-ram-gb", default="3")
    parser.add_argument("--preflight-timeout", default="120")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-yolo-check", action="store_true")
    parser.add_argument("--skip-label-view-check", action="store_true")
    parser.add_argument("--skip-smoke-gate", action="store_true")
    parser.add_argument("--skip-trainable-gate", action="store_true")
    parser.add_argument("--train-views", default="detect", help="Comma-separated train views for trainable-candidate gates.")
    parser.add_argument(
        "--fragment-review-policy",
        choices=["auto", "diagnostic", "ignore"],
        default="auto",
        help="auto ignores review-required fragments only when fragment is a selected trainable-candidate view.",
    )
    parser.add_argument("--allow-zero-visible-trainable", action="store_true", help="Allow zero visible banknotes in a trainable-candidate gate.")
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


def parse_train_views(value: str) -> set[str]:
    return {item.strip() for item in re.split(r"[,;\s]+", value) if item.strip()}


def choose_asset_side_policy(recipe: dict, override: str) -> str:
    policy = override or str(recipe.get("asset_side_policy", "any"))
    if policy not in WEBGL_ASSET_SIDE_POLICIES:
        raise SystemExit(f"--asset-side-policy must be one of {sorted(WEBGL_ASSET_SIDE_POLICIES)}")
    return policy


def choose_camera_profile(recipe: dict, override: str) -> str:
    profile = override or str(recipe.get("camera_profile", "generic_phone_jitter"))
    if profile not in WEBGL_CAMERA_PROFILES:
        raise SystemExit(f"--camera-profile must be one of {sorted(WEBGL_CAMERA_PROFILES)}")
    return profile


def main() -> int:
    args = parse_args()
    catalog = read_json(resolve(args.catalog))
    recipe = find_recipe(catalog, args.recipe_id)
    scene_mode = choose_scene_mode(recipe, args.scene_mode)
    asset_side_policy = choose_asset_side_policy(recipe, args.asset_side_policy)
    camera_profile = choose_camera_profile(recipe, args.camera_profile)
    artifact_status = args.artifact_status or STATUS_TO_BATCH_STATUS.get(str(recipe.get("artifact_status")), "diagnostic")
    fragment_review_policy = args.fragment_review_policy
    if fragment_review_policy == "auto":
        train_views = parse_train_views(args.train_views)
        fragment_review_policy = (
            "ignore"
            if artifact_status == "trainable-candidate" and "fragment" in train_views
            else "diagnostic"
        )
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
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--visual-scale",
        str(args.visual_scale),
        "--asset-side-policy",
        asset_side_policy,
        "--camera-profile",
        camera_profile,
        "--recipe-name",
        args.recipe_id,
        "--artifact-status",
        artifact_status,
        "--fragment-review-policy",
        fragment_review_policy,
        "--intended-use",
        str(recipe.get("intended_use", "")),
        "--notes",
        f"promotion_gate={recipe.get('promotion_gate', '')}; current_blocker={recipe.get('current_blocker', '')}",
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
    if args.background_dir:
        cmd.extend(["--background-dir", str(args.background_dir)])
    if args.browser_executable:
        cmd.extend(["--browser-executable", str(args.browser_executable)])
    if args.skip_render:
        cmd.append("--skip-render")
    if args.skip_yolo_check:
        cmd.append("--skip-yolo-check")
    if args.skip_label_view_check:
        cmd.append("--skip-label-view-check")

    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    if artifact_status == "smoke" and not args.skip_smoke_gate:
        gate_cmd = [
            sys.executable,
            "scripts/check_webgl_smoke_gate.py",
            "--root",
            str(out_root),
            "--require-recipe",
            args.recipe_id,
        ]
        if scene_mode != "auto":
            gate_cmd.extend(["--require-scene-mode", scene_mode])
        gate_cmd.extend(["--require-asset-side-policy", asset_side_policy])
        gate_cmd.extend(["--require-camera-profile", camera_profile])
        print(" ".join(gate_cmd), flush=True)
        subprocess.run(gate_cmd, cwd=ROOT, check=True)
    if artifact_status == "trainable-candidate" and not args.skip_trainable_gate:
        gate_cmd = [
            sys.executable,
            "scripts/check_webgl_trainable_candidate_gate.py",
            "--root",
            str(out_root),
            "--require-recipe",
            args.recipe_id,
            "--train-views",
            args.train_views,
        ]
        if scene_mode != "auto":
            gate_cmd.extend(["--require-scene-mode", scene_mode])
        gate_cmd.extend(["--require-asset-side-policy", asset_side_policy])
        gate_cmd.extend(["--require-camera-profile", camera_profile])
        if args.allow_zero_visible_trainable:
            gate_cmd.append("--allow-zero-visible")
        print(" ".join(gate_cmd), flush=True)
        subprocess.run(gate_cmd, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
