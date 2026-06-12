#!/usr/bin/env python
"""Build a row-count matched YOLO control with nearest class-mix exposure."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Base YOLO config to append controls to.")
    parser.add_argument("--target", type=Path, required=True, help="Target YOLO config whose row delta should be matched.")
    parser.add_argument(
        "--control-pool",
        type=Path,
        default=None,
        help="Optional YOLO config to draw control rows from. Defaults to --base.",
    )
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument("--tag", default="row_count_class_mix_control")
    parser.add_argument("--intended-use", default="")
    parser.add_argument("--promotion-rule", default="")
    parser.add_argument(
        "--allow-repeat",
        action="store_true",
        help="Allow the same pool row to be duplicated more times than it appears in the pool.",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Only draw non-synthetic control rows from --control-pool.",
    )
    parser.add_argument(
        "--include-empty-fillers",
        action="store_true",
        help="Allow empty-label rows to fill row count when class-mix rows cannot.",
    )
    parser.add_argument(
        "--allow-target-added-rows",
        action="store_true",
        help="Allow rows newly added by the target config to appear in the control pool.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def read_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(resolved)}: expected YAML mapping")
    return payload


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
            rows.append(repo_rel(resolve(line)))
    return rows


def image_rows(root: Path) -> list[str]:
    image_dir = resolve(root)
    if not image_dir.exists():
        raise SystemExit(f"missing image dir: {repo_rel(image_dir)}")
    rows = [
        repo_rel(path)
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    if not rows:
        raise SystemExit(f"no images found under: {repo_rel(image_dir)}")
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
        if path.suffix.lower() == ".txt":
            rows.extend(read_image_list(path))
        elif path.is_dir():
            rows.extend(image_rows(path))
        else:
            raise SystemExit(f"{repo_rel(config_path)} train item must point to a .txt list or image directory: {item}")
        sources.append(repo_rel(path))
    return rows, sources


def class_names(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names", {})
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def label_path_for_image(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def label_class_ids(image: str) -> list[int]:
    label_path = resolve(label_path_for_image(image))
    if not label_path.exists():
        return []
    class_ids: list[int] = []
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected 5 YOLO fields")
        try:
            class_ids.append(int(float(parts[0])))
        except ValueError as exc:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} has malformed class id: {parts[0]}") from exc
    return class_ids


def exposure_counts(rows: list[str]) -> tuple[Counter[int], int]:
    counts: Counter[int] = Counter()
    empty = 0
    for row in rows:
        class_ids = label_class_ids(row)
        if not class_ids:
            empty += 1
        for class_id in class_ids:
            counts[class_id] += 1
    return counts, empty


def is_synthetic_row(row: str) -> bool:
    normalized = row.replace("\\", "/").lower()
    return normalized.startswith("data/synthetic/") or "/data/synthetic/" in normalized


def named_counts(counts: Counter[int], names: dict[int, str]) -> dict[str, int]:
    return {names.get(class_id, f"class_{class_id}"): int(counts[class_id]) for class_id in sorted(counts)}


def write_image_list(path: Path, rows: list[str]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def solve_pattern_counts(
    patterns: list[tuple[int, ...]],
    upper_bounds: list[int | None],
    target: tuple[int, ...],
    row_delta: int,
) -> tuple[list[int], dict[str, Any]]:
    class_count = len(target)
    pattern_count = len(patterns)
    slack_count = class_count
    variable_count = pattern_count + slack_count
    integrality = np.ones(variable_count)

    lower = np.zeros(variable_count)
    upper = np.full(variable_count, np.inf)
    for index, bound in enumerate(upper_bounds):
        if bound is not None:
            upper[index] = bound
    for offset, value in enumerate(target):
        upper[pattern_count + offset] = value
    bounds = Bounds(lower, upper)

    row = np.zeros(variable_count)
    row[:pattern_count] = 1.0
    constraints = [LinearConstraint(row, row_delta, row_delta)]

    for class_index, target_value in enumerate(target):
        equation = np.zeros(variable_count)
        for pattern_index, pattern in enumerate(patterns):
            equation[pattern_index] = pattern[class_index]
        equation[pattern_count + class_index] = 1.0
        constraints.append(LinearConstraint(equation, target_value, target_value))

    objective = np.zeros(variable_count)
    objective[pattern_count:] = 1.0
    phase1 = milp(c=objective, integrality=integrality, bounds=bounds, constraints=constraints)
    if not phase1.success:
        raise SystemExit(f"row-count class-mix MILP failed: {phase1.message}")
    best_missing = int(round(float(sum(phase1.x[pattern_count:]))))

    missing_row = np.zeros(variable_count)
    missing_row[pattern_count:] = 1.0
    phase2_constraints = constraints + [LinearConstraint(missing_row, 0.0, float(best_missing))]
    phase2_objective = np.zeros(variable_count)
    for offset, target_value in enumerate(target):
        phase2_objective[pattern_count + offset] = 1.0 / max(1.0, float(target_value))
    phase2 = milp(c=phase2_objective, integrality=integrality, bounds=bounds, constraints=phase2_constraints)
    if not phase2.success:
        raise SystemExit(f"row-count class-mix phase2 MILP failed: {phase2.message}")

    pattern_counts = [int(round(value)) for value in phase2.x[:pattern_count]]
    slack = [int(round(value)) for value in phase2.x[pattern_count:]]
    return pattern_counts, {
        "missing_total": int(sum(slack)),
        "missing_by_target_index": slack,
        "phase1_objective": float(phase1.fun),
        "phase2_objective": float(phase2.fun),
    }


def main() -> int:
    args = parse_args()
    base_path = resolve(args.base)
    target_path = resolve(args.target)
    pool_path = resolve(args.control_pool or args.base)
    base_config = read_yaml(base_path)
    target_config = read_yaml(target_path)
    pool_config = read_yaml(pool_path)
    names = class_names(base_config)
    if names != class_names(target_config) or names != class_names(pool_config):
        raise SystemExit("base, target, and control-pool configs must use identical class names")

    base_rows, base_sources = train_rows(base_path, base_config)
    target_rows, target_sources = train_rows(target_path, target_config)
    pool_rows, pool_sources = train_rows(pool_path, pool_config)
    row_delta = len(target_rows) - len(base_rows)
    if row_delta <= 0:
        raise SystemExit(f"target row count must exceed base row count: delta={row_delta}")

    base_counts, base_empty = exposure_counts(base_rows)
    target_counts, target_empty = exposure_counts(target_rows)
    delta_counts = Counter(
        {class_id: target_counts[class_id] - base_counts[class_id] for class_id in set(base_counts) | set(target_counts)}
    )
    negative = {class_id: value for class_id, value in delta_counts.items() if value < 0}
    if negative:
        raise SystemExit(f"target has lower class exposure than base: {named_counts(Counter(negative), names)}")
    delta_counts = Counter({class_id: value for class_id, value in delta_counts.items() if value > 0})
    delta_empty = target_empty - base_empty
    if delta_empty < 0:
        raise SystemExit(f"target has lower empty exposure than base: {target_empty} < {base_empty}")

    base_row_set = set(base_rows)
    target_added_rows = [row for row in target_rows if row not in base_row_set]
    target_added_set = set(target_added_rows)
    target_class_ids = sorted(delta_counts)

    grouped_rows: dict[tuple[tuple[int, ...], str], list[str]] = defaultdict(list)
    empty_rows: list[str] = []
    rejected_rows = 0
    for row in pool_rows:
        if args.real_only and is_synthetic_row(row):
            continue
        if not args.allow_target_added_rows and row in target_added_set:
            continue
        class_ids = label_class_ids(row)
        counts = Counter(class_ids)
        if not counts:
            empty_rows.append(row)
            continue
        if any(class_id not in delta_counts for class_id in counts):
            rejected_rows += 1
            continue
        if any(counts[class_id] > delta_counts[class_id] for class_id in counts):
            rejected_rows += 1
            continue
        vector = tuple(int(counts.get(class_id, 0)) for class_id in target_class_ids)
        source_group = "synthetic" if is_synthetic_row(row) else "real"
        grouped_rows[(vector, source_group)].append(row)

    if args.include_empty_fillers:
        if not empty_rows:
            raise SystemExit("control pool has no empty rows for --include-empty-fillers")
        grouped_rows[(tuple(0 for _ in target_class_ids), "empty")].extend(empty_rows)
    if not grouped_rows:
        raise SystemExit("control pool produced no eligible class-mix rows")

    pattern_keys = sorted(
        grouped_rows,
        key=lambda key: (
            sum(key[0]),
            key[1] != "real",
            key[1],
            key[0],
        ),
        reverse=True,
    )
    patterns = [key[0] for key in pattern_keys]
    upper_bounds = [None if args.allow_repeat else len(grouped_rows[key]) for key in pattern_keys]
    pattern_counts, solution = solve_pattern_counts(
        patterns=patterns,
        upper_bounds=upper_bounds,
        target=tuple(delta_counts[class_id] for class_id in target_class_ids),
        row_delta=row_delta,
    )

    control_rows: list[str] = []
    selected_by_group: Counter[str] = Counter()
    selected_pattern_report: list[dict[str, Any]] = []
    for key, count in zip(pattern_keys, pattern_counts, strict=True):
        if count <= 0:
            continue
        rows = grouped_rows[key]
        if not args.allow_repeat and count > len(rows):
            raise SystemExit("internal error: selected more rows than allowed")
        for index in range(count):
            control_rows.append(rows[index % len(rows)])
        vector, source_group = key
        selected_by_group[source_group] += count
        selected_pattern_report.append(
            {
                "rows": int(count),
                "source_group": source_group,
                "class_counts": {
                    names[class_id]: int(vector[offset])
                    for offset, class_id in enumerate(target_class_ids)
                    if vector[offset]
                },
            }
        )

    if len(control_rows) != row_delta:
        raise SystemExit(f"internal error: selected {len(control_rows)} rows for delta {row_delta}")

    combined_rows = base_rows + control_rows
    combined_counts, combined_empty = exposure_counts(combined_rows)
    missing_counts = Counter({class_id: target_counts[class_id] - combined_counts[class_id] for class_id in target_class_ids})
    extra_counts = Counter(
        {
            class_id: combined_counts[class_id] - target_counts[class_id]
            for class_id in set(combined_counts) | set(target_counts)
            if combined_counts[class_id] - target_counts[class_id] > 0
        }
    )

    report = {
        "schema": "cashsnap_yolo_row_count_class_mix_control_v1",
        "tag": args.tag,
        "base_config": repo_rel(base_path),
        "target_config": repo_rel(target_path),
        "control_pool_config": repo_rel(pool_path),
        "base_train_sources": base_sources,
        "target_train_sources": target_sources,
        "control_pool_sources": pool_sources,
        "allow_repeat": bool(args.allow_repeat),
        "real_only": bool(args.real_only),
        "include_empty_fillers": bool(args.include_empty_fillers),
        "base_images": len(base_rows),
        "target_images": len(target_rows),
        "row_delta": int(row_delta),
        "target_added_images": len(target_added_rows),
        "combined_images": len(combined_rows),
        "control_images": len(control_rows),
        "control_unique_images": len(set(control_rows)),
        "control_duplicate_images": len(control_rows) - len(set(control_rows)),
        "pool_rejected_rows": int(rejected_rows),
        "eligible_patterns": len(pattern_keys),
        "selected_by_group": dict(sorted(selected_by_group.items())),
        "base_empty": int(base_empty),
        "target_empty": int(target_empty),
        "combined_empty": int(combined_empty),
        "delta_empty": int(delta_empty),
        "base_class_counts": named_counts(base_counts, names),
        "target_class_counts": named_counts(target_counts, names),
        "target_delta_class_counts": named_counts(delta_counts, names),
        "combined_class_counts": named_counts(combined_counts, names),
        "missing_class_counts": named_counts(Counter({k: v for k, v in missing_counts.items() if v > 0}), names),
        "extra_class_counts": named_counts(extra_counts, names),
        "class_mix_exact": not any(value > 0 for value in missing_counts.values()) and not extra_counts,
        "row_count_exact": len(combined_rows) == len(target_rows),
        "solver": solution,
        "selected_patterns": selected_pattern_report,
    }

    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    write_image_list(args.out_list, combined_rows)
    out_config = resolve(args.out_config)
    config = copy.deepcopy(base_config)
    config["path"] = rel_between(out_config.parent, ROOT)
    config["train"] = repo_rel(resolve(args.out_list))
    config["names"] = {index: names[index] for index in sorted(names)}

    sources = copy.deepcopy(config.get("cashsnap_sources", {}))
    if not isinstance(sources, dict):
        sources = {}
    sources["row_count_class_mix_control_base_config"] = repo_rel(base_path)
    sources["row_count_class_mix_control_target_config"] = repo_rel(target_path)
    sources["row_count_class_mix_control_pool_config"] = repo_rel(pool_path)
    config["cashsnap_sources"] = sources
    config["cashsnap_row_count_control"] = {
        "source": "real_class_mix:target_added_rows" if args.real_only else "class_mix:target_added_rows",
        "matched_dose_images": int(row_delta),
        "class_mix_all_exact": bool(report["class_mix_exact"]),
        "class_mix_exact_matches": None,
        "row_count_exact": bool(report["row_count_exact"]),
        "missing_class_counts": report["missing_class_counts"],
        "extra_class_counts": report["extra_class_counts"],
    }
    config["cashsnap_row_count_class_mix_control"] = report

    policy = copy.deepcopy(config.get("cashsnap_policy", {}))
    if not isinstance(policy, dict):
        policy = {}
    if args.intended_use:
        policy["intended_use"] = args.intended_use
    if args.promotion_rule:
        policy["promotion_rule"] = args.promotion_rule
    config["cashsnap_policy"] = policy

    write_yaml(out_config, config)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote_list={repo_rel(resolve(args.out_list))}")
    print(f"wrote_config={repo_rel(out_config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
