#!/usr/bin/env python
"""Run diagnostic gates declared on a WebGL recipe catalog entry."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_recipe_catalog_v1.json"
NOTE_CONDITION_POLICIES = {"mixed", "pristine_only", "heavy_wear", "wet_stress"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL dataset root.")
    parser.add_argument("--recipe-id", required=True, help="Recipe id in the catalog.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing JSON file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return data


def find_recipe(catalog: dict[str, Any], recipe_id: str) -> dict[str, Any]:
    recipes = catalog.get("recipes", [])
    if not isinstance(recipes, list):
        raise SystemExit("catalog recipes must be a list")
    for row in recipes:
        if isinstance(row, dict) and row.get("id") == recipe_id:
            return row
    raise SystemExit(f"recipe not found: {recipe_id}")


def add_int_option(cmd: list[str], gate: dict[str, Any], key: str, option: str) -> None:
    if key in gate:
        cmd.extend([option, str(int(gate[key]))])


def add_float_option(cmd: list[str], gate: dict[str, Any], key: str, option: str) -> None:
    if key in gate:
        cmd.extend([option, str(float(gate[key]))])


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"diagnostic gate command failed with exit code {completed.returncode}")


def run_class_distribution_gate(root: Path, gate: dict[str, Any]) -> None:
    expected_classes = gate.get("expected_classes", [])
    if not isinstance(expected_classes, list) or not expected_classes:
        raise SystemExit("class_distribution.expected_classes must be a non-empty list")
    cmd = [
        sys.executable,
        "scripts/check_webgl_class_distribution.py",
        "--root",
        str(root),
        "--expected-classes",
        ",".join(str(item).strip() for item in expected_classes if str(item).strip()),
    ]
    add_int_option(cmd, gate, "min_images", "--min-images")
    add_int_option(cmd, gate, "min_total", "--min-total")
    add_int_option(cmd, gate, "min_per_class", "--min-per-class")
    add_int_option(cmd, gate, "max_class_spread", "--max-class-spread")
    add_float_option(cmd, gate, "max_class_ratio", "--max-class-ratio")
    if gate.get("allow_extra_classes"):
        cmd.append("--allow-extra-classes")
    run(cmd)


def run_count_stress_gate(root: Path, gate: dict[str, Any]) -> None:
    cmd = [
        sys.executable,
        "scripts/check_webgl_count_stress.py",
        "--root",
        str(root),
    ]
    add_int_option(cmd, gate, "min_images", "--min-images")
    add_int_option(cmd, gate, "min_repeat_images", "--min-repeat-images")
    add_int_option(cmd, gate, "min_max_same_class", "--min-max-same-class")
    add_int_option(cmd, gate, "min_kept_split_parent_count", "--min-kept-split-parent-count")
    add_int_option(cmd, gate, "min_all_split_parent_count", "--min-all-split-parent-count")
    add_int_option(cmd, gate, "min_naive_kept_fragment_overcount", "--min-naive-kept-fragment-overcount")
    add_int_option(cmd, gate, "min_naive_all_fragment_overcount", "--min-naive-all-fragment-overcount")
    run(cmd)


def run_note_condition_diversity_gate(root: Path, gate: dict[str, Any], recipe: dict[str, Any]) -> None:
    cmd = [
        sys.executable,
        "scripts/check_webgl_note_condition_diversity.py",
        "--root",
        str(root),
    ]
    expected_policy = str(gate.get("expected_policy", recipe.get("note_condition_policy", ""))).strip()
    if expected_policy:
        if expected_policy not in NOTE_CONDITION_POLICIES:
            raise SystemExit(f"note_condition_diversity.expected_policy must be one of {sorted(NOTE_CONDITION_POLICIES)}")
        cmd.extend(["--expected-policy", expected_policy])
    add_int_option(cmd, gate, "min_notes", "--min-notes")
    add_int_option(cmd, gate, "min_profiles", "--min-profiles")
    add_float_option(cmd, gate, "min_dirtiness_range", "--min-dirtiness-range")
    add_float_option(cmd, gate, "min_crinkle_range", "--min-crinkle-range")
    add_float_option(cmd, gate, "min_wetness_range", "--min-wetness-range")
    add_int_option(cmd, gate, "min_dirty_notes", "--min-dirty-notes")
    add_int_option(cmd, gate, "min_pristine_notes", "--min-pristine-notes")
    add_int_option(cmd, gate, "min_wet_notes", "--min-wet-notes")
    run(cmd)


def run_hard_negative_diversity_gate(root: Path, gate: dict[str, Any]) -> None:
    cmd = [
        sys.executable,
        "scripts/check_webgl_hard_negative_diversity.py",
        "--root",
        str(root),
    ]
    add_int_option(cmd, gate, "min_images", "--min-images")
    add_int_option(cmd, gate, "min_total_props", "--min-total-props")
    add_int_option(cmd, gate, "min_prop_kinds", "--min-prop-kinds")
    add_int_option(cmd, gate, "min_textured_props", "--min-textured-props")
    if gate.get("require_zero_assets"):
        cmd.append("--require-zero-assets")
    run(cmd)


def main() -> int:
    args = parse_args()
    root = resolve(args.root)
    catalog = read_json(resolve(args.catalog))
    recipe = find_recipe(catalog, args.recipe_id)
    gates = recipe.get("diagnostic_gates", {})
    if not gates:
        print(f"ok: {args.recipe_id} declares no diagnostic gates")
        return 0
    if not isinstance(gates, dict):
        raise SystemExit(f"{args.recipe_id}: diagnostic_gates must be an object")

    class_distribution = gates.get("class_distribution")
    if class_distribution is not None:
        if not isinstance(class_distribution, dict):
            raise SystemExit(f"{args.recipe_id}: class_distribution gate must be an object")
        run_class_distribution_gate(root, class_distribution)

    count_stress = gates.get("count_stress")
    if count_stress is not None:
        if not isinstance(count_stress, dict):
            raise SystemExit(f"{args.recipe_id}: count_stress gate must be an object")
        run_count_stress_gate(root, count_stress)

    note_condition_diversity = gates.get("note_condition_diversity")
    if note_condition_diversity is not None:
        if not isinstance(note_condition_diversity, dict):
            raise SystemExit(f"{args.recipe_id}: note_condition_diversity gate must be an object")
        run_note_condition_diversity_gate(root, note_condition_diversity, recipe)

    hard_negative_diversity = gates.get("hard_negative_diversity")
    if hard_negative_diversity is not None:
        if not isinstance(hard_negative_diversity, dict):
            raise SystemExit(f"{args.recipe_id}: hard_negative_diversity gate must be an object")
        run_hard_negative_diversity_gate(root, hard_negative_diversity)

    print(f"ok: {args.recipe_id} diagnostic gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
