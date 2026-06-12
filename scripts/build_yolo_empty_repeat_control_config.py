#!/usr/bin/env python
"""Build a YOLO config by appending repeated empty-label control rows."""

from __future__ import annotations

import argparse
import copy
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_yolo_class_repeat_config as repeat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument(
        "--control-pool-config",
        type=Path,
        help="Config to draw empty rows from. Defaults to --base-config.",
    )
    parser.add_argument("--empty-count", required=True, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-config", required=True, type=Path)
    parser.add_argument("--out-list", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--intended-use", default="")
    parser.add_argument("--promotion-rule", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def class_names_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return repeat.names_by_id(left) == repeat.names_by_id(right)


def main() -> int:
    args = parse_args()
    if args.empty_count < 1:
        raise SystemExit("--empty-count must be at least 1")

    base_path = repeat.resolve(args.base_config)
    pool_path = repeat.resolve(args.control_pool_config or args.base_config)
    base_config = repeat.read_yaml(base_path)
    pool_config = repeat.read_yaml(pool_path)
    if not class_names_match(base_config, pool_config):
        raise SystemExit("base and control-pool configs must use identical class names")

    names = repeat.names_by_id(base_config)
    base_rows, base_sources = repeat.train_rows(base_path, base_config)
    pool_rows, pool_sources = repeat.train_rows(pool_path, pool_config)
    empty_pool = [row for row in pool_rows if not repeat.label_class_ids(row)]
    if args.empty_count > len(empty_pool):
        raise SystemExit(
            f"--empty-count {args.empty_count} exceeds available empty control rows {len(empty_pool)}"
        )

    rng = random.Random(args.seed)
    selected = list(empty_pool)
    rng.shuffle(selected)
    selected = selected[: args.empty_count]
    combined_rows = base_rows + selected

    base_counts, base_empty = repeat.exposure_counts(base_rows)
    selected_counts, selected_empty = repeat.exposure_counts(selected)
    combined_counts, combined_empty = repeat.exposure_counts(combined_rows)
    summary = {
        "schema": "cashsnap_yolo_empty_repeat_control_config_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "base_config": repeat.repo_rel(base_path),
        "control_pool_config": repeat.repo_rel(pool_path),
        "base_train_sources": base_sources,
        "control_pool_sources": pool_sources,
        "out_config": repeat.repo_rel(repeat.resolve(args.out_config)),
        "out_list": repeat.repo_rel(repeat.resolve(args.out_list)),
        "seed": args.seed,
        "requested_empty_rows": args.empty_count,
        "available_empty_rows": len(empty_pool),
        "selected_empty_rows": len(selected),
        "selected_unique_rows": len(set(selected)),
        "selected_sample": sorted(selected)[:20],
        "base_rows": len(base_rows),
        "base_unique_rows": len(set(base_rows)),
        "combined_rows": len(combined_rows),
        "combined_unique_rows": len(set(combined_rows)),
        "combined_duplicate_rows": len(combined_rows) - len(set(combined_rows)),
        "base_empty_rows": base_empty,
        "selected_empty_label_rows": selected_empty,
        "combined_empty_rows": combined_empty,
        "base_class_counts": repeat.named_counts(base_counts, names),
        "selected_class_counts": repeat.named_counts(Counter(selected_counts), names),
        "combined_class_counts": repeat.named_counts(combined_counts, names),
        "intended_use": args.intended_use,
        "promotion_rule": args.promotion_rule,
    }

    output_config = copy.deepcopy(base_config)
    output_config["path"] = repeat.rel_between(repeat.resolve(args.out_config).parent, repeat.ROOT)
    output_config["train"] = repeat.repo_rel(repeat.resolve(args.out_list))
    output_config["cashsnap_empty_repeat_control"] = summary
    if args.intended_use:
        output_config.setdefault("cashsnap_policy", {})["intended_use"] = args.intended_use
    if args.promotion_rule:
        output_config.setdefault("cashsnap_policy", {})["promotion_rule"] = args.promotion_rule

    if not args.dry_run:
        repeat.write_image_list(repeat.resolve(args.out_list), combined_rows)
        repeat.write_yaml(repeat.resolve(args.out_config), output_config)
        repeat.write_json(repeat.resolve(args.summary_json), summary)

    print(
        "empty_repeat_control "
        f"base_rows={len(base_rows)} selected_empty={len(selected)} "
        f"combined_rows={len(combined_rows)} duplicate_rows={len(combined_rows) - len(set(combined_rows))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
