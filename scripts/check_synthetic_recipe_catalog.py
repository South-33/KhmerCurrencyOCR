#!/usr/bin/env python
"""Validate CashSnap synthetic target and recipe coverage configs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "configs" / "synthetic_targets" / "cashsnap_real_target_matrix_v1.json"
DEFAULT_RECIPES = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_recipe_catalog_v1.json"
VALID_RECIPE_STATUSES = {"planned", "smoke_ready", "label_policy_ready", "diagnostic", "trainable-candidate", "promoted"}


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
