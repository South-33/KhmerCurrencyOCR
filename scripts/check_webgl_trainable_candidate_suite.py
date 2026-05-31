#!/usr/bin/env python
"""Validate the WebGL trainable-candidate suite manifest."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from webgl_constants import WEBGL_ASSET_SIDE_POLICIES, WEBGL_CAMERA_PROFILES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_trainable_candidates_v1.json"
DEFAULT_CATALOG = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_recipe_catalog_v1.json"
VALID_TRAIN_VIEWS = {"detect", "fragment", "obb"}
RUNNABLE_STATUSES = {"smoke_ready", "label_policy_ready", "diagnostic", "trainable-candidate", "promoted"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--check-existing", action="store_true", help="Also gate rendered package outputs and exact seed ranges.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    return resolve(path).resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> object:
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing JSON file: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def parse_train_views(value: object, recipe_id: str) -> list[str]:
    if isinstance(value, list):
        views = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str):
        views = [item for item in re.split(r"[,;\s]+", value) if item]
    else:
        raise SystemExit(f"{recipe_id}: train_views must be a list or string")
    unknown = sorted(set(views) - VALID_TRAIN_VIEWS)
    require(not unknown, f"{recipe_id}: unknown train view(s): {unknown}")
    require(bool(views), f"{recipe_id}: train_views must be non-empty")
    return sorted(set(views))


def parse_visual_scale(value: object, recipe_id: str) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError as exc:
        raise SystemExit(f"{recipe_id}: visual_scale must be a number") from exc
    require(1.0 <= number <= 4.0, f"{recipe_id}: visual_scale must be between 1 and 4")
    return text


def require_declared_path(row: dict, out_root: Path, field: str, suffix: str) -> None:
    declared = str(row.get(field, "")).replace("\\", "/")
    expected = repo_rel(out_root / suffix)
    require(declared == expected, f"{row['recipe_id']}: {field} must be {expected!r}, got {declared!r}")


def check_existing(row: dict, out_root: Path, train_views: list[str]) -> None:
    cmd = [
        sys.executable,
        "scripts/check_webgl_trainable_candidate_gate.py",
        "--root",
        str(out_root),
        "--require-recipe",
        str(row["recipe_id"]),
        "--require-scene-mode",
        str(row["scene_mode"]),
        "--require-asset-side-policy",
        str(row["asset_side_policy"]),
        "--require-camera-profile",
        str(row["camera_profile"]),
        "--min-images",
        str(row["count"]),
        "--train-views",
        ",".join(train_views),
    ]
    if bool(row.get("allow_zero_visible")):
        cmd.append("--allow-zero-visible")
    subprocess.run(cmd, cwd=ROOT, check=True)

    recipe = read_json(out_root / "recipe.json")
    summary = read_json(out_root / "qa" / "summary.json")
    require(isinstance(recipe, dict), f"{row['recipe_id']}: recipe.json must be an object")
    require(isinstance(summary, dict), f"{row['recipe_id']}: qa/summary.json must be an object")
    seed_range = recipe.get("variant_seed_range", {})
    require(isinstance(seed_range, dict), f"{row['recipe_id']}: recipe variant_seed_range must be an object")
    expected_end = int(row["start_variant"]) + int(row["count"]) - 1
    require(seed_range.get("start") == row["start_variant"], f"{row['recipe_id']}: rendered start_variant mismatch")
    require(seed_range.get("count") == row["count"], f"{row['recipe_id']}: rendered count mismatch")
    require(seed_range.get("end") == expected_end, f"{row['recipe_id']}: rendered end_variant mismatch")
    require(
        recipe.get("asset_side_policy", "any") == row["asset_side_policy"],
        f"{row['recipe_id']}: rendered asset_side_policy mismatch",
    )
    require(
        recipe.get("camera_profile", "generic_phone_jitter") == row["camera_profile"],
        f"{row['recipe_id']}: rendered camera_profile mismatch",
    )
    expected_background_dir = str(row.get("background_dir", "")).replace("\\", "/")
    if expected_background_dir:
        rendered_background_dir = str(recipe.get("background_dir", "")).replace("\\", "/")
        require(
            rendered_background_dir == expected_background_dir,
            f"{row['recipe_id']}: rendered background_dir mismatch",
        )
    render = recipe.get("render", {})
    if isinstance(render, dict) and "visual_scale" in render:
        require(
            str(render.get("visual_scale")) == str(row["visual_scale"]),
            f"{row['recipe_id']}: rendered visual_scale mismatch",
        )
    else:
        command = recipe.get("command", [])
        require(
            str(row["visual_scale"]) == "2" and isinstance(command, list) and "--visual-scale" not in command,
            f"{row['recipe_id']}: rendered recipe is missing visual_scale metadata",
        )
    require(summary.get("images") == row["count"], f"{row['recipe_id']}: QA image count mismatch")
    asset_selection = summary.get("asset_selection", {})
    require(isinstance(asset_selection, dict), f"{row['recipe_id']}: QA summary must include asset_selection")
    side_policy_counts = asset_selection.get("side_policy_counts", {})
    require(isinstance(side_policy_counts, dict), f"{row['recipe_id']}: asset_selection.side_policy_counts must be an object")
    require(
        set(side_policy_counts) == {row["asset_side_policy"]},
        f"{row['recipe_id']}: QA side policy counts must only contain {row['asset_side_policy']!r}",
    )
    if row["asset_side_policy"] == "front_back_mix":
        front_back_mix_counts = asset_selection.get("front_back_mix_counts", {})
        require(isinstance(front_back_mix_counts, dict), f"{row['recipe_id']}: front_back_mix_counts must be an object")
        require(
            int(front_back_mix_counts.get("unsatisfied", 0)) == 0,
            f"{row['recipe_id']}: front_back_mix has unsatisfied rendered images",
        )
        side_counts = asset_selection.get("side_counts", {})
        require(isinstance(side_counts, dict), f"{row['recipe_id']}: side_counts must be an object")
        require(int(side_counts.get("front", 0)) > 0, f"{row['recipe_id']}: front_back_mix rendered no fronts")
        require(int(side_counts.get("back", 0)) > 0, f"{row['recipe_id']}: front_back_mix rendered no backs")
    camera_profiles = summary.get("camera_profiles", {})
    require(isinstance(camera_profiles, dict), f"{row['recipe_id']}: QA summary must include camera_profiles")
    requested_counts = camera_profiles.get("requested_counts", {})
    require(isinstance(requested_counts, dict), f"{row['recipe_id']}: camera_profiles.requested_counts must be an object")
    require(
        requested_counts == {row["camera_profile"]: row["count"]},
        f"{row['recipe_id']}: unexpected camera profile request counts {requested_counts}",
    )
    selected_counts = camera_profiles.get("selected_counts", {})
    require(isinstance(selected_counts, dict) and selected_counts, f"{row['recipe_id']}: selected camera profiles must be non-empty")
    for suffix in ("manifest.json", "qa/summary.json", "data.yaml", "recipe.json"):
        require((out_root / suffix).exists(), f"{row['recipe_id']}: missing rendered output {out_root / suffix}")


def main() -> int:
    args = parse_args()
    suite = read_json(args.suite)
    catalog = read_json(args.catalog)
    require(isinstance(suite, dict), "suite config must be an object")
    require(isinstance(catalog, dict), "recipe catalog must be an object")
    require(suite.get("artifact_status") == "trainable-candidate", "suite artifact_status must be trainable-candidate")
    require(str(suite.get("mix_output", "")).strip(), "suite must declare mix_output")

    catalog_rows = catalog.get("recipes", [])
    require(isinstance(catalog_rows, list) and catalog_rows, "catalog recipes must be a non-empty list")
    catalog_by_id = {str(row.get("id", "")): row for row in catalog_rows if isinstance(row, dict)}

    rows = suite.get("recipes", [])
    require(isinstance(rows, list) and rows, "suite recipes must be a non-empty list")
    seen_roots: set[str] = set()
    seen_ranges: set[tuple[str, int]] = set()
    view_counts: dict[str, int] = {view: 0 for view in sorted(VALID_TRAIN_VIEWS)}
    total_images = 0

    for row in rows:
        require(isinstance(row, dict), "suite recipe rows must be objects")
        for key in (
            "recipe_id",
            "scene_mode",
            "asset_side_policy",
            "camera_profile",
            "visual_scale",
            "out_root",
            "start_variant",
            "count",
            "train_views",
            "allow_zero_visible",
            "asset_manifest",
            "qa_summary",
            "data_yaml",
            "intended_use",
            "promotion_gate",
            "promotion_blocker",
        ):
            require(key in row, f"{row.get('recipe_id', '<unknown>')}: missing {key}")

        recipe_id = str(row["recipe_id"])
        catalog_row = catalog_by_id.get(recipe_id)
        require(catalog_row is not None, f"{recipe_id}: not found in recipe catalog")
        catalog_status = str(catalog_row.get("artifact_status", ""))
        require(catalog_status in RUNNABLE_STATUSES, f"{recipe_id}: catalog status {catalog_status!r} is not runnable")
        catalog_modes = {str(item) for item in catalog_row.get("scene_modes", [])}
        require(str(row["scene_mode"]) in catalog_modes, f"{recipe_id}: scene_mode {row['scene_mode']!r} is not in catalog modes {sorted(catalog_modes)}")
        asset_side_policy = str(row["asset_side_policy"])
        require(
            asset_side_policy in WEBGL_ASSET_SIDE_POLICIES,
            f"{recipe_id}: asset_side_policy must be one of {sorted(WEBGL_ASSET_SIDE_POLICIES)}",
        )
        catalog_asset_side_policy = str(catalog_row.get("asset_side_policy", "any"))
        require(
            asset_side_policy == catalog_asset_side_policy,
            f"{recipe_id}: suite asset_side_policy {asset_side_policy!r} does not match catalog {catalog_asset_side_policy!r}",
        )
        camera_profile = str(row["camera_profile"])
        require(
            camera_profile in WEBGL_CAMERA_PROFILES,
            f"{recipe_id}: camera_profile must be one of {sorted(WEBGL_CAMERA_PROFILES)}",
        )
        catalog_camera_profile = str(catalog_row.get("camera_profile", "generic_phone_jitter"))
        require(
            camera_profile == catalog_camera_profile,
            f"{recipe_id}: suite camera_profile {camera_profile!r} does not match catalog {catalog_camera_profile!r}",
        )
        row["visual_scale"] = parse_visual_scale(row["visual_scale"], recipe_id)

        try:
            start_variant = int(row["start_variant"])
            count = int(row["count"])
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{recipe_id}: start_variant and count must be integers") from exc
        require(start_variant >= 0, f"{recipe_id}: start_variant must be non-negative")
        require(count > 0, f"{recipe_id}: count must be positive")
        row["start_variant"] = start_variant
        row["count"] = count
        total_images += count

        out_root = resolve(Path(str(row["out_root"])))
        try:
            out_root.resolve().relative_to((ROOT / "data" / "synthetic").resolve())
        except ValueError as exc:
            raise SystemExit(f"{recipe_id}: out_root must stay under data/synthetic") from exc
        root_key = repo_rel(out_root)
        require(root_key not in seen_roots, f"{recipe_id}: duplicate out_root {root_key}")
        seen_roots.add(root_key)
        require_declared_path(row, out_root, "asset_manifest", "manifest.json")
        require_declared_path(row, out_root, "qa_summary", "qa/summary.json")
        require_declared_path(row, out_root, "data_yaml", "data.yaml")

        train_views = parse_train_views(row["train_views"], recipe_id)
        for train_view in train_views:
            view_counts[train_view] += 1
        allow_zero_visible = bool(row.get("allow_zero_visible"))
        if str(row["scene_mode"]) == "negative":
            require(allow_zero_visible, f"{recipe_id}: negative candidate must set allow_zero_visible=true")
            require(asset_side_policy == "any", f"{recipe_id}: negative candidate must use asset_side_policy=any")
        else:
            require(not allow_zero_visible, f"{recipe_id}: only negative candidates may set allow_zero_visible=true")
        if recipe_id == "webgl_back_side_confusion_v1":
            require(asset_side_policy == "front_back_mix", f"{recipe_id}: must require front_back_mix sampling")

        for variant in range(start_variant, start_variant + count):
            range_key = (str(row["scene_mode"]), variant)
            require(range_key not in seen_ranges, f"{recipe_id}: duplicate scene/variant seed {range_key}")
            seen_ranges.add(range_key)

        if args.check_existing:
            check_existing(row, out_root, train_views)

    print(
        f"ok: {suite.get('name')} declares {len(rows)} trainable-candidate recipe(s), "
        f"{total_images} images, views={view_counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
