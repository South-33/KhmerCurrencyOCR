#!/usr/bin/env python
"""Build accepted-WebGL blend variant configs for bounded interaction probes."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "configs" / "cashsnap_v1_plus_webgl_accepted_nowarmup_probe.yaml"
DEFAULT_OUT_DIR = ROOT / "configs" / "webgl_blend_variants"
DEFAULT_LIST_DIR = ROOT / "configs" / "generated_lists" / "webgl_blend_variants"
DEFAULT_DOMAIN_GAP_DIR = ROOT / "runs" / "cashsnap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Accepted blend YAML to variant from.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--list-dir", type=Path, default=DEFAULT_LIST_DIR)
    parser.add_argument("--domain-gap-dir", type=Path, default=DEFAULT_DOMAIN_GAP_DIR)
    parser.add_argument("--per-class", type=int, default=None, help="Defaults to the base config subset policy.")
    parser.add_argument("--backgrounds", type=int, default=None, help="Defaults to the base config subset policy.")
    parser.add_argument(
        "--variant",
        choices=["leave-one-out"],
        default="leave-one-out",
        help="Variant family to generate.",
    )
    parser.add_argument(
        "--only-recipe",
        action="append",
        default=[],
        help="Limit leave-one-out generation to omitted recipe id(s). Repeatable.",
    )
    parser.add_argument("--domain-gap-preset", default="accepted_blend_v1")
    parser.add_argument("--skip-domain-gap-gate", action="store_true")
    parser.add_argument(
        "--fail-on-domain-gap",
        action="store_true",
        help="Abort when a variant fails the domain-gap gate. Default records pass/fail and continues.",
    )
    parser.add_argument("--skip-yolo-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def short_recipe_id(recipe_id: str) -> str:
    value = slug(recipe_id)
    if value.startswith("webgl_"):
        value = value[len("webgl_") :]
    if value.endswith("_v1"):
        value = value[: -len("_v1")]
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"expected YAML mapping: {repo_rel(path)}")
    return document


def read_json(path: Path) -> dict[str, Any]:
    import json

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"expected JSON mapping: {repo_rel(path)}")
    return document


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def acceptance_metadata(config: dict[str, Any], base_path: Path) -> tuple[list[str], list[str]]:
    metadata = config.get("cashsnap_webgl_acceptance", {})
    if not isinstance(metadata, dict):
        raise SystemExit(f"{repo_rel(base_path)} is missing cashsnap_webgl_acceptance")
    recipe_ids = [str(item) for item in metadata.get("selected_recipe_ids", [])]
    prefixes = [str(item).replace("\\", "/").strip() for item in metadata.get("selected_prefixes", [])]
    if not recipe_ids or len(recipe_ids) != len(prefixes):
        raise SystemExit(f"{repo_rel(base_path)} selected recipes/prefixes are missing or mismatched")
    return recipe_ids, prefixes


def source_data(config: dict[str, Any], base_path: Path) -> str:
    sources = config.get("cashsnap_sources", {})
    if not isinstance(sources, dict) or not str(sources.get("source_data", "")).strip():
        raise SystemExit(f"{repo_rel(base_path)} is missing cashsnap_sources.source_data")
    return str(sources["source_data"])


def subset_policy_value(config: dict[str, Any], key: str, fallback: int) -> int:
    policy = config.get("cashsnap_subset_policy", {})
    if not isinstance(policy, dict):
        return fallback
    value = policy.get(key)
    return fallback if value is None else int(value)


def build_subset_command(
    source_data_path: str,
    out_path: Path,
    train_list: Path,
    prefixes: list[str],
    per_class: int,
    backgrounds: int,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/build_yolo_balanced_subset.py",
        "--data",
        source_data_path,
        "--out",
        repo_rel(out_path),
        "--train-list",
        repo_rel(train_list),
        "--per-class",
        str(per_class),
        "--backgrounds",
        str(backgrounds),
    ]
    for prefix in prefixes:
        command.extend(["--always-include-prefix", prefix])
    return command


def domain_gap_command(out_path: Path, json_out: Path, preset: str, skip: bool, fail_on_gap: bool) -> list[str]:
    if skip or not preset.strip():
        return []
    command = [
        sys.executable,
        "scripts/audit_yolo_domain_gap.py",
        "--data",
        repo_rel(out_path),
        "--split",
        "train",
        "--json-out",
        repo_rel(json_out),
        "--gate-preset",
        preset,
    ]
    if fail_on_gap:
        command.append("--fail-on-gap")
    return command


def yolo_check_command(out_path: Path, skip: bool) -> list[str]:
    if skip:
        return []
    return [sys.executable, "scripts/check_yolo_dataset.py", "--data", repo_rel(out_path)]


def annotate_variant(
    out_path: Path,
    base_path: Path,
    variant_kind: str,
    omitted_recipe_ids: list[str],
    selected_recipe_ids: list[str],
    selected_prefixes: list[str],
    domain_gap_json: Path | None,
    domain_gap_preset: str,
) -> None:
    config = read_yaml(out_path)
    config["cashsnap_webgl_blend_variant"] = {
        "base_config": repo_rel(base_path),
        "variant_kind": variant_kind,
        "omitted_recipe_ids": omitted_recipe_ids,
        "selected_recipe_ids": selected_recipe_ids,
        "selected_prefixes": selected_prefixes,
    }
    if domain_gap_json is not None:
        domain_gap = read_json(domain_gap_json).get("domain_gap_gate", {})
        passed = bool(domain_gap.get("passed")) if isinstance(domain_gap, dict) else False
        config["cashsnap_domain_gap_gate"] = {
            "status": "pass" if passed else "fail",
            "preset": domain_gap_preset,
            "summary": repo_rel(domain_gap_json),
        }
    write_yaml(out_path, config)


def variant_rows(recipe_ids: list[str], prefixes: list[str], only_recipe: set[str]) -> list[tuple[str, list[str], list[str]]]:
    rows: list[tuple[str, list[str], list[str]]] = []
    for omitted in recipe_ids:
        if only_recipe and omitted not in only_recipe:
            continue
        selected_ids = [recipe_id for recipe_id in recipe_ids if recipe_id != omitted]
        selected_prefixes = [prefix for recipe_id, prefix in zip(recipe_ids, prefixes) if recipe_id != omitted]
        rows.append((omitted, selected_ids, selected_prefixes))
    if only_recipe and not rows:
        raise SystemExit(f"none of --only-recipe values are selected in the base config: {sorted(only_recipe)}")
    return rows


def main() -> int:
    args = parse_args()
    base_path = resolve(args.base)
    out_dir = resolve(args.out_dir)
    list_dir = resolve(args.list_dir)
    domain_gap_dir = resolve(args.domain_gap_dir)
    base_config = read_yaml(base_path)
    recipe_ids, prefixes = acceptance_metadata(base_config, base_path)
    source = source_data(base_config, base_path)
    per_class = args.per_class or subset_policy_value(base_config, "per_class_real_target", 24)
    backgrounds = args.backgrounds or subset_policy_value(base_config, "background_target", 24)

    rows = variant_rows(recipe_ids, prefixes, set(args.only_recipe))
    print(f"base={repo_rel(base_path)} variants={len(rows)}")
    for omitted, selected_ids, selected_prefixes in rows:
        stem = f"cashsnap_v1_plus_webgl_accepted_minus_{short_recipe_id(omitted)}"
        out_path = out_dir / f"{stem}.yaml"
        train_list = list_dir / f"{stem}_train.txt"
        domain_gap_json = domain_gap_dir / f"domain_gap_{stem}_train.json"
        gap_command = domain_gap_command(
            out_path,
            domain_gap_json,
            args.domain_gap_preset,
            args.skip_domain_gap_gate,
            args.fail_on_domain_gap,
        )
        check_command = yolo_check_command(out_path, args.skip_yolo_check)

        run(
            build_subset_command(
                source_data_path=source,
                out_path=out_path,
                train_list=train_list,
                prefixes=selected_prefixes,
                per_class=per_class,
                backgrounds=backgrounds,
            ),
            args.dry_run,
        )
        if args.dry_run:
            if gap_command:
                print(" ".join(gap_command), flush=True)
            if check_command:
                print(" ".join(check_command), flush=True)
            continue

        if gap_command:
            run(gap_command, args.dry_run)
        if check_command:
            run(check_command, args.dry_run)
        annotate_variant(
            out_path=out_path,
            base_path=base_path,
            variant_kind=args.variant,
            omitted_recipe_ids=[omitted],
            selected_recipe_ids=selected_ids,
            selected_prefixes=selected_prefixes,
            domain_gap_json=domain_gap_json if gap_command else None,
            domain_gap_preset=args.domain_gap_preset,
        )
        print(f"wrote {repo_rel(out_path)}")
        print(f"wrote {repo_rel(train_list)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
