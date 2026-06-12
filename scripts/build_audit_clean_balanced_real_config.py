from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATA = ROOT / "configs" / "cashsnap_v1_plus_webgl_trainable_candidates.yaml"
DEFAULT_POSITIVE_LIST = ROOT / "runs" / "cashsnap" / "real_data_label_audit_v1" / "candidate_positive_train_audit_clean_v1.txt"
DEFAULT_EMPTY_LIST = (
    ROOT / "runs" / "cashsnap" / "real_data_label_audit_v1" / "candidate_empty_train_lowrisk_no_teacher_unmatched_v1.txt"
)
SOURCE_PREFIXES = {
    "asian_currency_": "asian_currency",
    "billsbank_": "billsbank",
    "cambodia_currency_project_": "cambodia_currency_project",
    "cashcountingxl_": "cashcountingxl",
    "khmer_us_currency_": "khmer_us_currency",
    "usd_total_": "usd_total",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an audit-clean balanced real-only YOLO config.")
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--positive-list", type=Path, default=DEFAULT_POSITIVE_LIST)
    parser.add_argument("--empty-list", type=Path, default=DEFAULT_EMPTY_LIST)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--backgrounds", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--exclude-source", action="append", default=[])
    parser.add_argument(
        "--max-positive-per-source",
        type=int,
        default=None,
        help="Soft cap for selected positive images per source; rare-class fallback may exceed it.",
    )
    parser.add_argument("--tag", default="auditclean_real_p24_bg24_v1")
    parser.add_argument("--intended-use", default="")
    parser.add_argument("--promotion-rule", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(path)}: expected YAML mapping")
    return payload


def read_list(path: Path) -> list[str]:
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            rows.append(repo_rel(resolve(line)))
    return rows


def label_path_for_image(row: str) -> Path:
    path = Path(row)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return resolve(path.with_suffix(".txt"))
    parts[index] = "labels"
    return resolve(Path(*parts).with_suffix(".txt"))


def class_ids(row: str) -> list[int]:
    label_path = label_path_for_image(row)
    if not label_path.exists():
        raise SystemExit(f"missing label for {row}: {repo_rel(label_path)}")
    ids: list[int] = []
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected 5 YOLO fields")
        try:
            ids.append(int(float(parts[0])))
        except ValueError as exc:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} malformed class id {parts[0]!r}") from exc
    return ids


def source_group(row: str) -> str:
    name = Path(row).name.lower()
    for prefix, group in SOURCE_PREFIXES.items():
        if name.startswith(prefix):
            return group
    return name.split("_", 1)[0] if "_" in name else "unknown"


def split_to_repo_relative(config_path: Path, config: dict[str, Any], split_value: str | list[str]) -> str | list[str]:
    raw_root = Path(str(config.get("path", "."))).expanduser()
    root = raw_root if raw_root.is_absolute() else (config_path.parent / raw_root).resolve()

    def convert(value: str) -> str:
        path = Path(value)
        return repo_rel(path if path.is_absolute() else root / path)

    if isinstance(split_value, list):
        return [convert(str(value)) for value in split_value]
    return convert(str(split_value))


def named_counter(counter: Counter[int], names: dict[int, str]) -> dict[str, int]:
    return {names.get(index, f"class_{index}"): counter[index] for index in sorted(names)}


def sample_positive_rows(
    rows: list[str],
    *,
    class_count: int,
    per_class: int,
    max_positive_per_source: int | None,
    rng: random.Random,
) -> tuple[list[str], Counter[int], dict[int, int], Counter[str], Counter[str]]:
    by_class: dict[int, list[str]] = defaultdict(list)
    row_classes: dict[str, set[int]] = {}
    source_counts = Counter()
    for row in rows:
        unique = {class_id for class_id in class_ids(row) if 0 <= class_id < class_count}
        if not unique:
            continue
        row_classes[row] = unique
        for class_id in unique:
            by_class[class_id].append(row)
    for pool in by_class.values():
        rng.shuffle(pool)

    counts: Counter[int] = Counter()
    selected: list[str] = []
    selected_set: set[str] = set()
    source_counts: Counter[str] = Counter()
    source_cap_overrides: Counter[str] = Counter()
    pool_sizes = {class_id: len(by_class.get(class_id, [])) for class_id in range(class_count)}
    ordered_class_ids = sorted(range(class_count), key=lambda item: (pool_sizes[item], item))
    for class_id in ordered_class_ids:
        for row in by_class.get(class_id, []):
            if counts[class_id] >= per_class:
                break
            if row in selected_set:
                continue
            source = source_group(row)
            if max_positive_per_source is not None and source_counts[source] >= max_positive_per_source:
                continue
            selected.append(row)
            selected_set.add(row)
            source_counts[source] += 1
            for represented in row_classes[row]:
                counts[represented] += 1

    # Rare classes can have source-concentrated pools. Preserve class coverage if
    # the diversity cap blocks the requested target.
    for class_id in ordered_class_ids:
        for row in by_class.get(class_id, []):
            if counts[class_id] >= per_class:
                break
            if row in selected_set:
                continue
            source = source_group(row)
            selected.append(row)
            selected_set.add(row)
            if max_positive_per_source is not None and source_counts[source] >= max_positive_per_source:
                source_cap_overrides[source] += 1
            source_counts[source] += 1
            for represented in row_classes[row]:
                counts[represented] += 1

    return selected, counts, pool_sizes, source_counts, source_cap_overrides


def sample_empty_rows(rows: list[str], *, count: int, rng: random.Random) -> tuple[list[str], Counter[str]]:
    candidates = [row for row in rows if not class_ids(row)]
    rng.shuffle(candidates)
    selected = candidates[:count]
    return selected, Counter(source_group(row) for row in selected)


def main() -> None:
    args = parse_args()
    if args.per_class < 1:
        raise SystemExit("--per-class must be positive")
    if args.backgrounds < 0:
        raise SystemExit("--backgrounds must be non-negative")
    if args.max_positive_per_source is not None and args.max_positive_per_source < 1:
        raise SystemExit("--max-positive-per-source must be positive when set")

    source_data = resolve(args.source_data)
    positive_list = resolve(args.positive_list)
    empty_list = resolve(args.empty_list)
    out_config = resolve(args.out_config)
    out_list = resolve(args.out_list)
    source_config = load_yaml(source_data)
    raw_names = source_config.get("names", {})
    if isinstance(raw_names, dict):
        names = {int(key): str(value) for key, value in raw_names.items()}
    elif isinstance(raw_names, list):
        names = {index: str(value) for index, value in enumerate(raw_names)}
    else:
        raise SystemExit(f"{repo_rel(source_data)}: names must be a list or mapping")
    class_count = len(names)
    excluded_sources = {str(item).strip() for item in args.exclude_source if str(item).strip()}

    positive_rows_all = read_list(positive_list)
    empty_rows_all = read_list(empty_list)
    positive_rows = [row for row in positive_rows_all if source_group(row) not in excluded_sources]
    empty_rows = [row for row in empty_rows_all if source_group(row) not in excluded_sources]

    rng = random.Random(args.seed)
    (
        selected_positive,
        positive_counts,
        positive_pool_sizes,
        positive_source_counts,
        source_cap_overrides,
    ) = sample_positive_rows(
        positive_rows,
        class_count=class_count,
        per_class=args.per_class,
        max_positive_per_source=args.max_positive_per_source,
        rng=rng,
    )
    selected_empty, empty_source_counts = sample_empty_rows(empty_rows, count=args.backgrounds, rng=rng)
    selected = selected_positive + selected_empty
    missing = [names[index] for index in range(class_count) if positive_counts[index] == 0]
    if missing:
        raise SystemExit(f"selected subset has no labels for: {', '.join(missing)}")

    policy = {
        "schema": "cashsnap_audit_clean_balanced_real_config_v1",
        "tag": args.tag,
        "source_data": repo_rel(source_data),
        "positive_list": repo_rel(positive_list),
        "empty_list": repo_rel(empty_list),
        "seed": args.seed,
        "per_class_real_target": args.per_class,
        "background_target": args.backgrounds,
        "excluded_sources": sorted(excluded_sources),
        "max_positive_per_source": args.max_positive_per_source,
        "source_cap_overrides": dict(sorted(source_cap_overrides.items())),
        "positive_pool_images_before_source_exclusion": len(positive_rows_all),
        "positive_pool_images": len(positive_rows),
        "empty_pool_images_before_source_exclusion": len(empty_rows_all),
        "empty_pool_images": len(empty_rows),
        "selected_positive_images": len(selected_positive),
        "selected_backgrounds": len(selected_empty),
        "selected_images": len(selected),
        "selected_class_images": named_counter(positive_counts, names),
        "positive_pool_class_images": {names[index]: positive_pool_sizes.get(index, 0) for index in range(class_count)},
        "selected_positive_sources": dict(sorted(positive_source_counts.items())),
        "selected_background_sources": dict(sorted(empty_source_counts.items())),
        "intended_use": args.intended_use,
        "promotion_rule": args.promotion_rule,
    }
    out_payload = {
        "path": rel_between(out_config.parent, ROOT),
        "train": repo_rel(out_list),
        "val": split_to_repo_relative(source_data, source_config, source_config["val"]),
        "test": split_to_repo_relative(source_data, source_config, source_config["test"]),
        "names": raw_names,
        "cashsnap_sources": {
            "source_data": repo_rel(source_data),
            "positive_list": repo_rel(positive_list),
            "empty_list": repo_rel(empty_list),
            "excluded_sources": sorted(excluded_sources),
        },
        "cashsnap_subset_policy": policy,
    }

    print(
        f"selected={len(selected)} positives={len(selected_positive)} backgrounds={len(selected_empty)} "
        f"excluded_sources={sorted(excluded_sources)}",
        flush=True,
    )
    for index in range(class_count):
        print(f"  {names[index]}: selected={positive_counts[index]} pool={positive_pool_sizes.get(index, 0)}")
    print(f"selected_positive_sources={dict(sorted(positive_source_counts.items()))}")
    print(f"selected_background_sources={dict(sorted(empty_source_counts.items()))}")
    if source_cap_overrides:
        print(f"source_cap_overrides={dict(sorted(source_cap_overrides.items()))}")

    if args.dry_run:
        return
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_list.parent.mkdir(parents=True, exist_ok=True)
    out_list.write_text("\n".join(selected) + "\n", encoding="utf-8")
    out_config.write_text(yaml.safe_dump(out_payload, sort_keys=False), encoding="utf-8")
    print(f"wrote {repo_rel(out_config)}")
    print(f"wrote {repo_rel(out_list)}")


if __name__ == "__main__":
    main()
