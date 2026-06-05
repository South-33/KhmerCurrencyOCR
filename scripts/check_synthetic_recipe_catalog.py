#!/usr/bin/env python
"""Validate CashSnap synthetic target and recipe coverage configs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from webgl_constants import WEBGL_ASSET_SIDE_POLICIES, WEBGL_CAMERA_PROFILES


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {
    "USD_1",
    "USD_5",
    "USD_10",
    "USD_20",
    "USD_50",
    "USD_100",
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_20000",
    "KHR_50000",
}
DEFAULT_TARGETS = ROOT / "configs" / "synthetic_targets" / "cashsnap_real_target_matrix_v1.json"
DEFAULT_RECIPES = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_recipe_catalog_v1.json"
VALID_RECIPE_STATUSES = {
    "planned",
    "smoke_ready",
    "label_policy_ready",
    "diagnostic",
    "trainable-candidate",
    "promoted",
    "rejected_probe",
}
NOTE_CONDITION_POLICIES = {"mixed", "pristine_only", "heavy_wear", "wet_stress"}
LENS_DISTORTION_POLICIES = {"off", "phone_mild"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def require_nonnegative_int(value: object, message: str) -> None:
    require(type(value) is int and value >= 0, message)


def validate_diagnostic_gates(recipe_id: str, gates: object) -> None:
    if gates in (None, ""):
        return
    require(isinstance(gates, dict), f"{recipe_id}: diagnostic_gates must be an object")
    allowed_gate_names = {"class_distribution", "count_stress", "note_condition_diversity", "hard_negative_diversity"}
    unknown_gate_names = sorted(str(key) for key in gates if str(key) not in allowed_gate_names)
    require(not unknown_gate_names, f"{recipe_id}: unknown diagnostic_gates {unknown_gate_names}")

    class_distribution = gates.get("class_distribution")
    if class_distribution is not None:
        require(isinstance(class_distribution, dict), f"{recipe_id}: class_distribution gate must be an object")
        expected_classes = class_distribution.get("expected_classes")
        require(
            isinstance(expected_classes, list) and expected_classes,
            f"{recipe_id}: class_distribution.expected_classes must be a non-empty list",
        )
        expected = [str(item).strip() for item in expected_classes]
        require(all(expected), f"{recipe_id}: class_distribution expected classes must be non-empty")
        duplicate_expected = [item for item, count in Counter(expected).items() if count > 1]
        require(not duplicate_expected, f"{recipe_id}: duplicate class_distribution expected classes {duplicate_expected}")
        unknown_classes = sorted(class_name for class_name in expected if class_name not in CLASS_NAMES)
        require(not unknown_classes, f"{recipe_id}: unknown class_distribution expected classes {unknown_classes}")
        for field in ("min_images", "min_total", "min_per_class", "max_class_spread"):
            if field in class_distribution:
                require_nonnegative_int(
                    class_distribution[field],
                    f"{recipe_id}: class_distribution.{field} must be a non-negative integer",
                )
        if "max_class_ratio" in class_distribution:
            ratio = class_distribution["max_class_ratio"]
            require(
                isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and ratio > 0,
                f"{recipe_id}: class_distribution.max_class_ratio must be positive",
            )
        if "allow_extra_classes" in class_distribution:
            require(
                type(class_distribution["allow_extra_classes"]) is bool,
                f"{recipe_id}: class_distribution.allow_extra_classes must be boolean",
            )

    count_stress = gates.get("count_stress")
    if count_stress is not None:
        require(isinstance(count_stress, dict), f"{recipe_id}: count_stress gate must be an object")
        for field in (
            "min_images",
            "min_repeat_images",
            "min_max_same_class",
            "min_kept_split_parent_count",
            "min_all_split_parent_count",
            "min_naive_kept_fragment_overcount",
            "min_naive_all_fragment_overcount",
        ):
            if field in count_stress:
                require_nonnegative_int(
                    count_stress[field],
                    f"{recipe_id}: count_stress.{field} must be a non-negative integer",
                )

    note_condition_diversity = gates.get("note_condition_diversity")
    if note_condition_diversity is not None:
        require(isinstance(note_condition_diversity, dict), f"{recipe_id}: note_condition_diversity gate must be an object")
        for field in ("min_notes", "min_profiles", "min_dirty_notes", "min_pristine_notes", "min_wet_notes"):
            if field in note_condition_diversity:
                require_nonnegative_int(
                    note_condition_diversity[field],
                    f"{recipe_id}: note_condition_diversity.{field} must be a non-negative integer",
                )
        for field in ("min_dirtiness_range", "min_crinkle_range", "min_wetness_range"):
            if field in note_condition_diversity:
                value = note_condition_diversity[field]
                require(
                    isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0,
                    f"{recipe_id}: note_condition_diversity.{field} must be a non-negative number",
                )
        expected_policy = str(note_condition_diversity.get("expected_policy", "")).strip()
        require(
            not expected_policy or expected_policy in NOTE_CONDITION_POLICIES,
            f"{recipe_id}: note_condition_diversity.expected_policy must be one of {sorted(NOTE_CONDITION_POLICIES)}",
        )

    hard_negative_diversity = gates.get("hard_negative_diversity")
    if hard_negative_diversity is not None:
        require(isinstance(hard_negative_diversity, dict), f"{recipe_id}: hard_negative_diversity gate must be an object")
        for field in ("min_images", "min_total_props", "min_prop_kinds", "min_textured_props"):
            if field in hard_negative_diversity:
                require_nonnegative_int(
                    hard_negative_diversity[field],
                    f"{recipe_id}: hard_negative_diversity.{field} must be a non-negative integer",
                )
        if "require_zero_assets" in hard_negative_diversity:
            require(
                type(hard_negative_diversity["require_zero_assets"]) is bool,
                f"{recipe_id}: hard_negative_diversity.require_zero_assets must be boolean",
            )


def main() -> int:
    args = parse_args()
    targets_path = resolve(args.targets)
    recipes_path = resolve(args.recipes)
    targets = read_json(targets_path)
    recipes = read_json(recipes_path)

    conditions = targets.get("conditions")
    require(isinstance(conditions, list) and conditions, "target matrix must contain conditions")
    condition_ids = [str(row.get("id", "")).strip() for row in conditions]
    duplicate_conditions = [item for item, count in Counter(condition_ids).items() if count > 1]
    require(all(condition_ids), "target condition ids must be non-empty")
    require(not duplicate_conditions, f"duplicate target condition ids: {duplicate_conditions}")
    condition_set = set(condition_ids)
    required_condition_ids = {
        str(row["id"])
        for row in conditions
        if bool(row.get("required_for_v1")) and str(row.get("priority")) != "out_of_scope"
    }

    recipe_rows = recipes.get("recipes")
    require(isinstance(recipe_rows, list) and recipe_rows, "recipe catalog must contain recipes")
    recipe_ids = [str(row.get("id", "")).strip() for row in recipe_rows]
    duplicate_recipes = [item for item, count in Counter(recipe_ids).items() if count > 1]
    require(all(recipe_ids), "recipe ids must be non-empty")
    require(not duplicate_recipes, f"duplicate recipe ids: {duplicate_recipes}")

    coverage: dict[str, list[str]] = defaultdict(list)
    statuses = Counter()
    for row in recipe_rows:
        recipe_id = str(row["id"])
        status = str(row.get("artifact_status", ""))
        statuses[status] += 1
        require(status in VALID_RECIPE_STATUSES, f"{recipe_id}: invalid artifact_status {status}")
        target_ids = row.get("target_condition_ids")
        require(isinstance(target_ids, list) and target_ids, f"{recipe_id}: target_condition_ids must be non-empty")
        unknown_targets = sorted(str(item) for item in target_ids if str(item) not in condition_set)
        require(not unknown_targets, f"{recipe_id}: unknown target_condition_ids {unknown_targets}")
        require(str(row.get("intended_use", "")).strip(), f"{recipe_id}: missing intended_use")
        require(str(row.get("promotion_gate", "")).strip(), f"{recipe_id}: missing promotion_gate")
        require(str(row.get("current_blocker", "")).strip(), f"{recipe_id}: missing current_blocker")
        asset_side_policy = str(row.get("asset_side_policy", ""))
        require(asset_side_policy in WEBGL_ASSET_SIDE_POLICIES, f"{recipe_id}: invalid asset_side_policy {asset_side_policy!r}")
        camera_profile = str(row.get("camera_profile", ""))
        require(camera_profile in WEBGL_CAMERA_PROFILES, f"{recipe_id}: invalid camera_profile {camera_profile!r}")
        note_condition_policy = str(row.get("note_condition_policy", "mixed"))
        require(
            note_condition_policy in NOTE_CONDITION_POLICIES,
            f"{recipe_id}: invalid note_condition_policy {note_condition_policy!r}",
        )
        lens_distortion_policy = str(row.get("lens_distortion_policy", "off"))
        require(
            lens_distortion_policy in LENS_DISTORTION_POLICIES,
            f"{recipe_id}: invalid lens_distortion_policy {lens_distortion_policy!r}",
        )
        validate_diagnostic_gates(recipe_id, row.get("diagnostic_gates"))
        for target_id in target_ids:
            coverage[str(target_id)].append(recipe_id)

    uncovered_required = sorted(target_id for target_id in required_condition_ids if not coverage.get(target_id))
    require(not uncovered_required, f"required target conditions without recipes: {uncovered_required}")
    print(
        f"ok: {len(condition_ids)} target conditions, {len(recipe_ids)} recipes, "
        f"{len(required_condition_ids)} required conditions covered, statuses={dict(sorted(statuses.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
