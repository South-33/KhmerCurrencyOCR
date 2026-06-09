#!/usr/bin/env python
"""Build a list-backed YOLO config with a capped external-negative dose."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    ROOT
    / "configs"
    / "webgl_ablation"
    / "cashsnap_target_anchor_transplant_poisson_contact_geoscale205_minshort190_bal96_scaled_foreignhardneg6_probe_puresynth_realval_v1.yaml"
)
DEFAULT_BANK_CONFIG = ROOT / "configs" / "synthetic_recipes" / "cashsnap_external_negative_banks_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--bank-config", type=Path, default=DEFAULT_BANK_CONFIG)
    parser.add_argument("--bank-id", default="", help="Accepted external-negative bank id to draw from.")
    parser.add_argument("--bank-path", type=Path, default=None, help="Accepted external-negative bank path to draw from.")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--selection", choices=("first", "spread", "random"), default="spread")
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument(
        "--replace-existing-root",
        type=Path,
        default=None,
        help="Remove existing train rows under this image root before inserting the selected dose. Defaults to the selected bank path.",
    )
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(resolve(path))}: expected JSON mapping")
    return data


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(resolve(path))}: expected YAML mapping")
    return data


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw = Path(str(config.get("path", "."))).expanduser()
    return raw if raw.is_absolute() else (config_path.parent / raw).resolve()


def split_root(dataset_root: Path, split_path: str) -> Path:
    path = Path(split_path)
    return path if path.is_absolute() else dataset_root / path


def read_image_list(path: Path) -> list[str]:
    rows: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            rows.append(line.replace("\\", "/"))
    return rows


def train_rows(config_path: Path, config: dict[str, Any]) -> tuple[list[str], list[str]]:
    root = data_root(config_path, config)
    train = config.get("train")
    if isinstance(train, str):
        train_items = [train]
    elif isinstance(train, list) and all(isinstance(item, str) for item in train):
        train_items = [str(item) for item in train]
    else:
        raise SystemExit(f"{repo_rel(config_path)} train split must be a string or list of strings")

    rows: list[str] = []
    sources: list[str] = []
    for item in train_items:
        path = split_root(root, item)
        if path.suffix.lower() != ".txt":
            raise SystemExit(f"{repo_rel(config_path)} train item must point to a .txt list: {item}")
        rows.extend(read_image_list(path))
        sources.append(repo_rel(path))
    return rows, sources


def write_image_list(path: Path, rows: list[str]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def label_path_for_image(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def is_empty_label(image: str) -> bool:
    label = resolve(label_path_for_image(image))
    if not label.exists():
        return True
    return not any(line.strip() for line in label.read_text(encoding="utf-8").splitlines())


def select_rows(rows: list[str], count: int, selection: str, rng: random.Random) -> list[str]:
    if count < 1:
        raise SystemExit("--count must be at least 1")
    rows = sorted(dict.fromkeys(rows))
    if len(rows) < count:
        raise SystemExit(f"not enough external-negative rows: {len(rows)} < {count}")
    if selection == "first":
        return rows[:count]
    if selection == "random":
        return sorted(rng.sample(rows, count))
    if count == 1:
        return [rows[len(rows) // 2]]
    indexes = sorted({round(index * (len(rows) - 1) / (count - 1)) for index in range(count)})
    selected = [rows[index] for index in indexes]
    cursor = 0
    while len(selected) < count and cursor < len(rows):
        row = rows[cursor]
        if row not in selected:
            selected.append(row)
        cursor += 1
    return sorted(selected)


def remove_existing(rows: list[str], prefix: str) -> tuple[list[str], int, int]:
    normalized = prefix.replace("\\", "/").rstrip("/") + "/"
    indices = [index for index, row in enumerate(rows) if row.startswith(normalized)]
    if not indices:
        return rows, 0, len(rows)
    insert_at = indices[0]
    kept: list[str] = []
    adjusted_insert_at = 0
    for index, row in enumerate(rows):
        if row.startswith(normalized):
            continue
        if index < insert_at:
            adjusted_insert_at += 1
        kept.append(row)
    return kept, len(indices), adjusted_insert_at


def find_bank(config: dict[str, Any], bank_id: str, bank_path: Path | None) -> dict[str, Any]:
    banks = config.get("banks", [])
    if not isinstance(banks, list):
        raise SystemExit("external negative bank config must contain banks")
    matches: list[dict[str, Any]] = []
    for row in banks:
        if not isinstance(row, dict):
            continue
        if bank_id and str(row.get("id", "")).strip() == bank_id:
            matches.append(row)
        elif bank_path is not None and repo_rel(resolve(Path(str(row.get("path", ""))))) == repo_rel(resolve(bank_path)):
            matches.append(row)
    if not bank_id and bank_path is None:
        raise SystemExit("provide --bank-id or --bank-path")
    if not matches:
        selector = bank_id or repo_rel(resolve(bank_path or Path(".")))
        raise SystemExit(f"external negative bank not found: {selector}")
    if len(matches) > 1:
        raise SystemExit(f"external negative bank selector is ambiguous: {bank_id or bank_path}")
    return matches[0]


def verify_bank_allowed(bank_config: Path, bank_path: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "check_external_negative_banks.py"),
        "--config",
        str(resolve(bank_config)),
        "--require-path",
        str(bank_path),
        "--artifact-status",
        "trainable-candidate",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if result.returncode:
        message = (result.stderr or result.stdout).strip()
        raise SystemExit(message or "external negative bank is not allowed for trainable-candidate use")


def bank_image_rows(bank: dict[str, Any]) -> tuple[Path, list[str]]:
    bank_path = resolve(Path(str(bank.get("path", "")))).resolve()
    assets = bank.get("assets", [])
    if not isinstance(assets, list) or not assets:
        raise SystemExit(f"{bank.get('id', '<missing id>')}: accepted bank must list assets")
    rows: list[str] = []
    for asset in assets:
        if not isinstance(asset, dict):
            raise SystemExit(f"{bank.get('id', '<missing id>')}: asset rows must be objects")
        local_path = (bank_path / str(asset.get("file", ""))).resolve()
        rows.append(repo_rel(local_path))
    non_empty = [row for row in rows if not is_empty_label(row)]
    if non_empty:
        raise SystemExit(f"external-negative pool has non-empty labels: {non_empty[:5]}")
    return bank_path, rows


def main() -> int:
    args = parse_args()
    bank_config = read_json(args.bank_config)
    bank = find_bank(bank_config, args.bank_id, args.bank_path)
    bank_path = resolve(Path(str(bank.get("path", "")))).resolve()
    verify_bank_allowed(args.bank_config, bank_path)
    bank_path, pool = bank_image_rows(bank)

    base_path = resolve(args.base)
    base_config = read_yaml(base_path)
    base_rows, base_sources = train_rows(base_path, base_config)

    replace_root = resolve(args.replace_existing_root) if args.replace_existing_root else bank_path
    selected = select_rows(pool, args.count, args.selection, random.Random(args.seed))
    kept_rows, removed_rows, insert_at = remove_existing(base_rows, repo_rel(replace_root))
    combined_rows = kept_rows[:insert_at] + selected + kept_rows[insert_at:]
    if len(set(combined_rows)) != len(combined_rows):
        raise SystemExit("external-negative dose introduced duplicate train rows")

    report = {
        "schema": "cashsnap_external_negative_dose_v1",
        "base_config": repo_rel(base_path),
        "base_train_sources": base_sources,
        "bank_config": repo_rel(resolve(args.bank_config)),
        "bank_id": str(bank.get("id", "")),
        "bank_path": repo_rel(bank_path),
        "replace_existing_root": repo_rel(replace_root),
        "selection": str(args.selection),
        "seed": int(args.seed),
        "requested_count": int(args.count),
        "removed_existing_rows": int(removed_rows),
        "insert_at": int(insert_at),
        "base_images": len(base_rows),
        "selected_external_negative_images": len(selected),
        "combined_images": len(combined_rows),
        "selected_image_rows": selected,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    write_image_list(args.out_list, combined_rows)
    out_config = resolve(args.out_config)
    config = copy.deepcopy(base_config)
    config["path"] = rel_between(out_config.parent, ROOT)
    config["train"] = repo_rel(resolve(args.out_list))
    sources = copy.deepcopy(config.get("cashsnap_sources", {}))
    if not isinstance(sources, dict):
        sources = {}
    sources["external_negative_dose_base_config"] = repo_rel(base_path)
    sources["external_negative_bank_config"] = repo_rel(resolve(args.bank_config))
    sources["external_negative_bank_id"] = str(bank.get("id", ""))
    sources["external_negative_bank_path"] = repo_rel(bank_path)
    config["cashsnap_sources"] = sources
    config["cashsnap_external_negative_dose"] = report
    policy = copy.deepcopy(config.get("cashsnap_policy", {}))
    if not isinstance(policy, dict):
        policy = {}
    policy["intended_use"] = (
        "pure-synth plus reviewed external-negative TSTR probe for foreign/unknown currency "
        "and clutter false-positive repair"
    )
    policy["promotion_rule"] = (
        "reject unless external negatives reduce real-empty FPs while preserving full, clean-visible, "
        "labeled, stress, and protected positive guardrails"
    )
    config["cashsnap_policy"] = policy
    write_yaml(out_config, config)
    print(json.dumps(report, indent=2))
    print(f"wrote_list={repo_rel(resolve(args.out_list))}")
    print(f"wrote_config={repo_rel(out_config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
