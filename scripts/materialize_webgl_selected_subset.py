#!/usr/bin/env python
"""Materialize a selected WebGL geometry subset into a packaged diagnostic root."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import render_webgl_variant_batch as batch


ROOT = Path(__file__).resolve().parents[1]
MIXED_SELECTED = "mixed_selected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", type=Path, required=True, help="Selection JSON from select_webgl_geometry_subset.py.")
    parser.add_argument("--out-root", type=Path, required=True, help="Output packaged WebGL root.")
    parser.add_argument("--recipe-name", default="webgl_selected_geometry_subset_v1")
    parser.add_argument("--artifact-status", choices=["smoke", "diagnostic", "trainable-candidate"], default="diagnostic")
    parser.add_argument("--scene-mode", default="stack")
    parser.add_argument("--intended-use", default="Geometry-selected WebGL diagnostic subset.")
    parser.add_argument("--notes", default="")
    parser.add_argument("--fragment-review-policy", choices=["diagnostic", "ignore"], default="diagnostic")
    parser.add_argument("--skip-yolo-check", action="store_true")
    parser.add_argument("--skip-label-view-check", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"missing JSON file: {repo_rel(path)}")
    return json.loads(path.read_text(encoding="utf-8"))


def selected_variant_dirs(selection: dict[str, Any]) -> list[tuple[int, Path]]:
    rows = selection.get("selected_variants", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("selection JSON must contain a non-empty selected_variants list")
    variant_dirs: list[tuple[int, Path]] = []
    used_variants: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("selected_variants rows must be objects")
        root = resolve(Path(str(row.get("root", ""))))
        variant_name = str(row.get("variant", "")).strip()
        if not variant_name.startswith("variant_"):
            raise SystemExit(f"invalid variant name in selection JSON: {variant_name!r}")
        try:
            variant = int(variant_name.removeprefix("variant_"))
        except ValueError as exc:
            raise SystemExit(f"invalid variant id in selection JSON: {variant_name!r}") from exc
        if variant in used_variants:
            raise SystemExit(f"duplicate variant id {variant}; duplicate remapping is not implemented")
        out_dir = root / variant_name
        if not out_dir.exists():
            raise SystemExit(f"missing selected variant dir: {repo_rel(out_dir)}")
        for required in ("visual.png", "id.png", "labels_visible.txt", "visible_boxes.json", "layer_audit.json", "metadata.json"):
            if not (out_dir / required).exists():
                raise SystemExit(f"selected variant missing {required}: {repo_rel(out_dir)}")
        used_variants.add(variant)
        variant_dirs.append((variant, out_dir))
    return sorted(variant_dirs, key=lambda item: item[0])


def source_render_defaults(variant_dirs: list[tuple[int, Path]]) -> dict[str, Any]:
    metadata_rows = [read_json(path / "metadata.json") for _variant, path in variant_dirs]
    first = metadata_rows[0] if isinstance(metadata_rows[0], dict) else {}
    scene_config = first.get("sceneConfig", {}) if isinstance(first, dict) else {}
    asset_selection = first.get("assetSelection", {}) if isinstance(first, dict) else {}
    camera = scene_config.get("camera", {}) if isinstance(scene_config, dict) else {}
    asset_side_policy_counts: Counter[str] = Counter()
    camera_profile_counts: Counter[str] = Counter()
    stack_pose_policy_counts: Counter[str] = Counter()
    for row in metadata_rows:
        if not isinstance(row, dict):
            continue
        row_scene = row.get("sceneConfig", {})
        row_asset = row.get("assetSelection", {})
        row_camera = row_scene.get("camera", {}) if isinstance(row_scene, dict) else {}
        if isinstance(row_asset, dict):
            asset_side_policy_counts[str(row_asset.get("sidePolicy", "any"))] += 1
            stack_pose_policy_counts[str(row_asset.get("stackPosePolicy", "default"))] += 1
        if isinstance(row_camera, dict):
            camera_profile_counts[str(row_camera.get("profileRequested", "phone_closeup_clean_like"))] += 1

    def selected_policy(counts: Counter[str], fallback: str) -> str:
        if not counts:
            return fallback
        return next(iter(counts)) if len(counts) == 1 else MIXED_SELECTED

    render = {
        "width": int(first.get("width", 1440) or 1440),
        "height": int(first.get("height", 1080) or 1080),
        "visual_scale": str(first.get("visualScale", 1) or 1),
        "asset_side_policy": selected_policy(
            asset_side_policy_counts,
            str(asset_selection.get("sidePolicy", "any")) if isinstance(asset_selection, dict) else "any",
        ),
        "camera_profile": selected_policy(
            camera_profile_counts,
            str(camera.get("profileRequested", "phone_closeup_clean_like")) if isinstance(camera, dict) else "phone_closeup_clean_like",
        ),
        "stack_pose_policy": selected_policy(stack_pose_policy_counts, "default"),
        "source_policy_counts": {
            "asset_side_policy": dict(sorted(asset_side_policy_counts.items())),
            "camera_profile": dict(sorted(camera_profile_counts.items())),
            "stack_pose_policy": dict(sorted(stack_pose_policy_counts.items())),
        },
    }
    return render


def recipe_args(args: argparse.Namespace, selection: dict[str, Any], variant_dirs: list[tuple[int, Path]]) -> SimpleNamespace:
    render = source_render_defaults(variant_dirs)
    notes = args.notes.strip()
    if not notes:
        notes = (
            "materialized_from="
            f"{repo_rel(resolve(args.selection_json))}; "
            f"selection_gate={selection.get('gate_preset', '')}; "
            f"selection_score={selection.get('metrics', {}).get('score', '')}"
        )
    variants = [variant for variant, _path in variant_dirs]
    class_counts = selection.get("metrics", {}).get("class_counts", {})
    min_per_class = min((int(value) for value in class_counts.values()), default=0) if isinstance(class_counts, dict) else 0
    return SimpleNamespace(
        recipe_name=args.recipe_name,
        artifact_status=args.artifact_status,
        intended_use=args.intended_use,
        notes=notes,
        scene_mode=args.scene_mode,
        start_variant=min(variants),
        count=len(variants),
        width=render["width"],
        height=render["height"],
        visual_scale=render["visual_scale"],
        browser_executable=None,
        note_condition_policy="mixed",
        lens_distortion_policy="off",
        stack_pose_policy=render["stack_pose_policy"],
        asset_side_policy=render["asset_side_policy"],
        camera_profile=render["camera_profile"],
        class_sequence="",
        balanced_subset_count=len(variants),
        balanced_subset_classes="",
        balanced_subset_min_per_class=min_per_class,
        balanced_subset_max_class_spread=-1,
        balanced_subset_max_class_ratio=0.0,
        background_dir=None,
        environment_dir=None,
        fragment_review_policy=args.fragment_review_policy,
        headroom_max_percent="",
        headroom_resume_percent="",
        headroom_max_ram_percent="",
        headroom_max_gpu_mem_percent="",
        min_free_ram_gb="",
        preflight_timeout="",
        skip_yolo_check=args.skip_yolo_check,
        skip_label_view_check=args.skip_label_view_check,
    )


def run_check(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def annotate_recipe_metadata(out_root: Path, selection_path: Path, selection: dict[str, Any], variant_dirs: list[tuple[int, Path]]) -> None:
    recipe_path = out_root / "recipe.json"
    recipe = read_json(recipe_path)
    if not isinstance(recipe, dict):
        raise SystemExit(f"{repo_rel(recipe_path)} must contain a JSON object")
    variants = [variant for variant, _path in variant_dirs]
    render = source_render_defaults(variant_dirs)
    recipe["variant_seed_range"] = {
        "source": "selected_noncontiguous_variants",
        "start": min(variants),
        "end": max(variants),
        "count": len(variants),
        "selected": variants,
    }
    recipe["selector"] = {
        "selection_json": repo_rel(selection_path),
        "gate_preset": selection.get("gate_preset", ""),
        "metrics": selection.get("metrics", {}),
        "roots": selection.get("roots", []),
        "source_policy_counts": render.get("source_policy_counts", {}),
    }
    batch.write_json(recipe_path, recipe)


def main() -> int:
    args = parse_args()
    selection_path = resolve(args.selection_json)
    out_root = resolve(args.out_root)
    selection = read_json(selection_path)
    if not isinstance(selection, dict):
        raise SystemExit("selection JSON must be an object")
    variant_dirs = selected_variant_dirs(selection)
    out_root.mkdir(parents=True, exist_ok=True)

    contact_sheet = out_root / "contact_sheet.png"
    contact_index = batch.write_contact_sheet(variant_dirs, contact_sheet)
    balanced_report = {
        "enabled": True,
        "strategy": "geometry_subset",
        "selection_json": repo_rel(selection_path),
        "selection_metrics": selection.get("metrics", {}),
        "selected_variants": selection.get("selected_variants", []),
    }
    data_yaml, data_obb_yaml, data_fragments_yaml, count_targets_path, count_summary_path = batch.write_yolo_dataset(
        variant_dirs,
        out_root,
        args.scene_mode,
        args.fragment_review_policy,
        balanced_report,
    )
    readable_pages = batch.write_readable_contact_sheets(
        variant_dirs,
        out_root / "qa" / "contact_sheets",
        include_id=args.scene_mode != "negative",
    )
    batch.write_json(
        out_root / "qa" / "contact_index.json",
        {
            "contact_sheet": str(contact_sheet.relative_to(out_root)),
            "readable_contact_sheets": readable_pages,
            "rows": contact_index,
            "cell_size": {"width": 320, "height": 240, "header_height": 30},
        },
    )
    batch.write_recipe_metadata(
        recipe_args(args, selection, variant_dirs),
        out_root,
        contact_sheet,
        data_yaml,
        data_obb_yaml,
        data_fragments_yaml,
        count_targets_path,
        count_summary_path,
    )
    annotate_recipe_metadata(out_root, selection_path, selection, variant_dirs)
    if not args.skip_yolo_check:
        run_check([sys.executable, "scripts/check_yolo_dataset.py", "--data", str(data_yaml)])
    if not args.skip_label_view_check:
        run_check([sys.executable, "scripts/check_webgl_label_views.py", "--root", str(out_root)])
    print(f"materialized={repo_rel(out_root)} images={len(variant_dirs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
