#!/usr/bin/env python
"""Build a YOLO config from WebGL recipes that passed ablation gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "configs" / "cashsnap_v1_plus_webgl_trainable_candidates.yaml"
DEFAULT_SUITE = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_trainable_candidates_v1.json"
DEFAULT_SUMMARY = ROOT / "runs" / "cashsnap" / "webgl_ablation_nowarmup_i416_summary.json"
DEFAULT_OUT = ROOT / "configs" / "cashsnap_v1_plus_webgl_accepted_nowarmup_probe.yaml"
DEFAULT_TRAIN_LIST = ROOT / "configs" / "generated_lists" / "cashsnap_v1_plus_webgl_accepted_nowarmup_probe_train.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--per-class-summary", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-list", type=Path, default=DEFAULT_TRAIN_LIST)
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--backgrounds", type=int, default=24)
    parser.add_argument("--max-per-class-drop-vs-real-only", type=float, default=0.05)
    parser.add_argument("--allow-real-only-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected JSON object: {path}")
    return document


def default_per_class_summary(summary: Path) -> Path:
    name = summary.name
    if name.endswith("_summary.json"):
        return summary.with_name(name[: -len("_summary.json")] + "_per_class.json")
    return summary.with_name(summary.stem + "_per_class.json")


def suite_out_roots(suite_path: Path) -> dict[str, str]:
    suite = read_json(suite_path)
    recipes = suite.get("recipes", [])
    if not isinstance(recipes, list):
        raise ValueError(f"suite recipes must be a list: {suite_path}")
    roots: dict[str, str] = {}
    for recipe in recipes:
        if not isinstance(recipe, dict):
            continue
        recipe_id = recipe.get("recipe_id")
        out_root = recipe.get("out_root")
        if recipe_id is None or out_root is None:
            continue
        roots[str(recipe_id)] = str(out_root).replace("\\", "/").strip("/")
    return roots


def worst_per_class_deltas(per_class_path: Path) -> dict[str, float]:
    if not per_class_path.exists():
        return {}
    document = read_json(per_class_path)
    rows = document.get("rows", [])
    if not isinstance(rows, list):
        return {}
    worst: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        recipe_id = str(row.get("recipe_id", ""))
        if not recipe_id:
            continue
        delta = row.get("delta_vs_real_only")
        if delta is None:
            continue
        value = float(delta)
        worst[recipe_id] = min(worst.get(recipe_id, value), value)
    return worst


def select_recipes(
    summary_path: Path,
    per_class_path: Path,
    max_per_class_drop: float,
    allow_real_only_failures: bool,
) -> tuple[list[str], dict[str, str]]:
    summary = read_json(summary_path)
    rows = summary.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"summary rows must be a list: {summary_path}")
    worst_deltas = worst_per_class_deltas(per_class_path)

    selected: list[str] = []
    rejected: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        recipe_id = str(row.get("recipe_id", ""))
        if not recipe_id:
            continue
        if not allow_real_only_failures and row.get("pass_vs_real_only") is not True:
            rejected[recipe_id] = "failed real-only mAP50-95 gate"
            continue
        worst_delta = worst_deltas.get(recipe_id)
        if worst_delta is not None and worst_delta < -max_per_class_drop:
            rejected[recipe_id] = (
                f"failed per-class drop gate ({worst_delta:+.6f} < {-max_per_class_drop:+.6f})"
            )
            continue
        selected.append(recipe_id)

    if not selected:
        raise SystemExit("no recipes passed the requested acceptance gates")
    return selected, rejected


def build_command(args: argparse.Namespace, prefixes: list[str]) -> list[str]:
    command = [
        sys.executable,
        "scripts/build_yolo_balanced_subset.py",
        "--data",
        repo_rel(args.data),
        "--out",
        repo_rel(args.out),
        "--train-list",
        repo_rel(args.train_list),
        "--per-class",
        str(args.per_class),
        "--backgrounds",
        str(args.backgrounds),
    ]
    for prefix in prefixes:
        command.extend(["--always-include-prefix", f"{prefix}/"])
    return command


def annotate_output(args: argparse.Namespace, selected: list[str], rejected: dict[str, str], prefixes: list[str]) -> None:
    config = yaml.safe_load(args.out.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"expected YAML mapping: {args.out}")
    config["cashsnap_webgl_acceptance"] = {
        "summary": repo_rel(args.summary),
        "per_class_summary": repo_rel(args.per_class_summary),
        "max_per_class_drop_vs_real_only": args.max_per_class_drop_vs_real_only,
        "selected_recipe_ids": selected,
        "selected_prefixes": [f"{prefix}/" for prefix in prefixes],
        "rejected_recipe_ids": rejected,
    }
    args.out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.data = resolve(args.data)
    args.suite = resolve(args.suite)
    args.summary = resolve(args.summary)
    args.per_class_summary = resolve(args.per_class_summary) if args.per_class_summary else default_per_class_summary(args.summary)
    args.out = resolve(args.out)
    args.train_list = resolve(args.train_list)

    selected, rejected = select_recipes(
        summary_path=args.summary,
        per_class_path=args.per_class_summary,
        max_per_class_drop=args.max_per_class_drop_vs_real_only,
        allow_real_only_failures=args.allow_real_only_failures,
    )
    roots = suite_out_roots(args.suite)
    missing = sorted(set(selected) - set(roots))
    if missing:
        raise SystemExit(f"selected recipe(s) missing from suite: {missing}")
    prefixes = [roots[recipe_id] for recipe_id in selected]

    print("selected_recipes=" + ",".join(selected), flush=True)
    if rejected:
        print("rejected_recipes=" + json.dumps(rejected, sort_keys=True), flush=True)

    command = build_command(args, prefixes)
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0

    subprocess.run(command, cwd=ROOT, check=True)
    annotate_output(args, selected, rejected, prefixes)
    print(f"wrote {repo_rel(args.out)}")
    print(f"wrote {repo_rel(args.train_list)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
