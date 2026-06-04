#!/usr/bin/env python
"""Gate a packaged WebGL artifact before it enters a trainable-candidate mix."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from webgl_constants import WEBGL_ASSET_SIDE_POLICIES, WEBGL_CAMERA_PROFILES


ROOT = Path(__file__).resolve().parents[1]
VALID_TRAIN_VIEWS = {"detect", "fragment", "obb"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL dataset root.")
    parser.add_argument("--min-images", type=int, default=1)
    parser.add_argument("--require-recipe", default="")
    parser.add_argument("--require-scene-mode", default="")
    parser.add_argument("--require-asset-side-policy", default="")
    parser.add_argument("--require-camera-profile", default="")
    parser.add_argument("--train-views", default="detect", help="Comma-separated views intended for training: detect,fragment,obb.")
    parser.add_argument(
        "--allow-artifact-status",
        action="append",
        default=None,
        help="Allowed recipe artifact_status. Defaults to trainable-candidate; repeat for tests.",
    )
    parser.add_argument("--allow-zero-visible", action="store_true", help="Allow zero visible banknotes, e.g. reviewed hard negatives.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> object:
    if not path.exists():
        raise SystemExit(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def int_at(document: dict, *keys: str, default: int = 0) -> int:
    value: object = document
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return int(value or 0)


def parse_train_views(value: str) -> set[str]:
    views = {item.strip() for item in re.split(r"[,;\s]+", value) if item.strip()}
    unknown = views - VALID_TRAIN_VIEWS
    if unknown:
        raise SystemExit(f"unknown train view(s): {sorted(unknown)}")
    if not views:
        raise SystemExit("--train-views must include at least one view")
    return views


def main() -> int:
    args = parse_args()
    dataset_root = resolve(args.root)
    subprocess.run([sys.executable, "scripts/check_webgl_label_views.py", "--root", str(dataset_root)], cwd=ROOT, check=True)
    appearance_cmd = [
        sys.executable,
        "scripts/check_webgl_appearance_diversity.py",
        "--root",
        str(dataset_root),
        "--min-images",
        str(args.min_images),
    ]
    if args.require_camera_profile and args.require_camera_profile != "phone_auto":
        appearance_cmd.extend(["--min-camera-profiles", "1"])
    subprocess.run(appearance_cmd, cwd=ROOT, check=True)

    recipe = read_json(dataset_root / "recipe.json")
    summary = read_json(dataset_root / "qa" / "summary.json")
    visual_quality = read_json(dataset_root / "qa" / "visual_quality.json")
    require(isinstance(recipe, dict), "recipe.json must be an object")
    require(isinstance(summary, dict), "qa/summary.json must be an object")
    require(isinstance(visual_quality, dict), "qa/visual_quality.json must be an object")

    allowed_statuses = set(args.allow_artifact_status or ["trainable-candidate"])
    artifact_status = str(recipe.get("artifact_status", ""))
    require(artifact_status in allowed_statuses, f"artifact_status {artifact_status!r} not in allowed statuses {sorted(allowed_statuses)}")
    recipe_name = str(recipe.get("recipe_name", ""))
    if args.require_recipe:
        require(recipe_name == args.require_recipe, f"expected recipe {args.require_recipe!r}, got {recipe_name!r}")

    images = int(summary.get("images", 0))
    require(images >= args.min_images, f"expected at least {args.min_images} images, got {images}")
    scene_modes = summary.get("scene_modes", {})
    require(isinstance(scene_modes, dict) and scene_modes, "qa summary must name scene_modes")
    if args.require_scene_mode:
        require(scene_modes == {args.require_scene_mode: images}, f"unexpected scene_modes: {scene_modes}")
    if args.require_asset_side_policy:
        require(
            args.require_asset_side_policy in WEBGL_ASSET_SIDE_POLICIES,
            f"--require-asset-side-policy must be one of {sorted(WEBGL_ASSET_SIDE_POLICIES)}",
        )
        require(
            recipe.get("asset_side_policy", "any") == args.require_asset_side_policy,
            f"expected recipe asset_side_policy={args.require_asset_side_policy!r}, got {recipe.get('asset_side_policy', 'any')!r}",
        )
        asset_selection = summary.get("asset_selection", {})
        require(isinstance(asset_selection, dict), "qa summary must include asset_selection")
        side_policy_counts = asset_selection.get("side_policy_counts", {})
        require(isinstance(side_policy_counts, dict), "asset_selection.side_policy_counts must be an object")
        require(
            side_policy_counts == {args.require_asset_side_policy: images},
            f"unexpected asset side policy counts: {side_policy_counts}",
        )
        if args.require_asset_side_policy == "front_back_mix":
            mix_counts = asset_selection.get("front_back_mix_counts", {})
            require(isinstance(mix_counts, dict), "asset_selection.front_back_mix_counts must be an object")
            require(int(mix_counts.get("unsatisfied", 0)) == 0, "front_back_mix has unsatisfied images")
            side_counts = asset_selection.get("side_counts", {})
            require(isinstance(side_counts, dict), "asset_selection.side_counts must be an object")
            require(int(side_counts.get("front", 0)) > 0, "front_back_mix rendered no fronts")
            require(int(side_counts.get("back", 0)) > 0, "front_back_mix rendered no backs")
    if args.require_camera_profile:
        require(
            args.require_camera_profile in WEBGL_CAMERA_PROFILES,
            f"--require-camera-profile must be one of {sorted(WEBGL_CAMERA_PROFILES)}",
        )
        require(
            recipe.get("camera_profile", "generic_phone_jitter") == args.require_camera_profile,
            f"expected recipe camera_profile={args.require_camera_profile!r}, got {recipe.get('camera_profile', 'generic_phone_jitter')!r}",
        )
        camera_profiles = summary.get("camera_profiles", {})
        require(isinstance(camera_profiles, dict), "qa summary must include camera_profiles")
        requested_counts = camera_profiles.get("requested_counts", {})
        require(isinstance(requested_counts, dict), "camera_profiles.requested_counts must be an object")
        require(
            requested_counts == {args.require_camera_profile: images},
            f"unexpected camera profile request counts: {requested_counts}",
        )
        selected_counts = camera_profiles.get("selected_counts", {})
        require(isinstance(selected_counts, dict) and selected_counts, "camera_profiles.selected_counts must be non-empty")
    require(summary.get("layer_audit_totals", {}).get("violations") == 0, "layer-order violations must be zero")

    visual_quality_counts = visual_quality.get("counts", {})
    require(isinstance(visual_quality_counts, dict), "visual_quality counts must be an object")
    require(int(visual_quality_counts.get("rejected", 0)) == 0, "visual_quality rejected images must be zero")

    visible = int_at(summary, "visible_instances", "total")
    if not args.allow_zero_visible:
        require(visible > 0, "trainable candidates must expose at least one visible banknote unless --allow-zero-visible is set")

    train_views = parse_train_views(args.train_views)
    fragment_status_counts = summary.get("fragments", {}).get("evidence_status_counts", {})
    require(isinstance(fragment_status_counts, dict), "fragment evidence_status_counts must be an object")
    review_required = int(fragment_status_counts.get("review_required", 0))
    if "fragment" in train_views:
        require(review_required == 0, "fragment train view cannot include review_required fragments")

    obb_status_counts = summary.get("obb", {}).get("image_status_counts", {})
    require(isinstance(obb_status_counts, dict), "obb image_status_counts must be an object")
    rejected_obb = int(obb_status_counts.get("rejected", 0))
    accepted_obb = int(obb_status_counts.get("accepted", 0))
    if "obb" in train_views:
        require(rejected_obb == 0, "OBB train view cannot include rejected OBB images")
        require(accepted_obb == images, "OBB train view requires every image to be accepted")

    print(
        f"ok: {recipe_name} trainable-candidate gate passed "
        f"({images} images, views={','.join(sorted(train_views))}, review_required={review_required}, rejected_obb={rejected_obb})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
