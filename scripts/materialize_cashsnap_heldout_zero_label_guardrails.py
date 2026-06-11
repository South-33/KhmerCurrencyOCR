#!/usr/bin/env python
"""Materialize held-out zero-label money guardrails for CashSnap detector eval."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("configs/cashsnap_v1.yaml")
DEFAULT_DATA_ROOT = Path("data/cashsnap_v1")
DEFAULT_OUT_DIR = Path("configs/generated_lists/audit")
DEFAULT_CONFIG_DIR = Path("configs/audit")
DEFAULT_TAG = "cashsnap_heldout_zero_label_money_guardrails_v1"
DEFAULT_TRAIN_HARDNEG_CAPS = {
    "foreign_asian_currency": 150,
    "unknown_usd2": 60,
    "unknown_khr100": 30,
}
BUCKETS = {
    "foreign_asian_currency": "asian_currency",
    "unknown_usd2": "usd_total_2Dollar",
    "unknown_khr100": "khmer_us_currency_100-riel",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--train-hardneg-out", type=Path)
    parser.add_argument("--train-hardneg-seed", type=int, default=20260611)
    parser.add_argument(
        "--train-hardneg-cap",
        action="append",
        default=[],
        metavar="BUCKET=N",
        help="Override sampled train zero-label hard-negative cap for a bucket.",
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
    return Path(target.resolve()).relative_to(ROOT).as_posix() if from_dir == ROOT else (
        Path(*Path(target.resolve()).relative_to(from_dir.resolve()).parts).as_posix()
    )


def label_path_for_image(data_root: Path, split: str, image: Path) -> Path:
    return data_root / "labels" / split / f"{image.stem}.txt"


def is_zero_label(label_path: Path) -> bool:
    return (not label_path.exists()) or (not label_path.read_text(encoding="utf-8").strip())


def read_names(path: Path) -> dict[int, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(path)}: expected YAML mapping")
    raw_names = payload.get("names")
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    raise SystemExit(f"{repo_rel(path)}: missing names mapping")


def write_list(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def parse_train_caps(values: list[str]) -> dict[str, int]:
    caps = dict(DEFAULT_TRAIN_HARDNEG_CAPS)
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--train-hardneg-cap expected BUCKET=N, got {value!r}")
        bucket, raw_count = value.split("=", 1)
        bucket = bucket.strip()
        if bucket not in BUCKETS:
            raise SystemExit(f"unknown train hardneg bucket {bucket!r}; expected one of {sorted(BUCKETS)}")
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise SystemExit(f"invalid cap for {bucket}: {raw_count!r}") from exc
        if count < 0:
            raise SystemExit(f"cap for {bucket} must be non-negative")
        caps[bucket] = count
    return caps


def main() -> int:
    args = parse_args()
    data_config = resolve(args.data)
    data_root = resolve(args.data_root)
    out_dir = resolve(args.out_dir)
    config_dir = resolve(args.config_dir)
    names = read_names(data_config)
    rng = random.Random(args.train_hardneg_seed)
    train_caps = parse_train_caps(args.train_hardneg_cap)

    summary: dict[str, Any] = {
        "schema": "cashsnap_heldout_zero_label_money_guardrails_v1",
        "tag": args.tag,
        "source_data": repo_rel(data_config),
        "data_root": repo_rel(data_root),
        "buckets": {},
        "combined": {},
        "train_hardneg": {},
    }
    lists_by_bucket: dict[str, dict[str, Path]] = {}
    combined_by_split: dict[str, list[str]] = {split: [] for split in args.splits}

    for bucket, pattern in BUCKETS.items():
        bucket_lists: dict[str, Path] = {}
        bucket_summary: dict[str, Any] = {"pattern": pattern, "splits": {}}
        for split in args.splits:
            image_dir = data_root / "images" / split
            if not image_dir.exists():
                raise SystemExit(f"missing image dir: {repo_rel(image_dir)}")
            rows: list[str] = []
            skipped_labeled = 0
            for image in sorted(image_dir.glob(f"*{pattern}*")):
                label = label_path_for_image(data_root, split, image)
                if is_zero_label(label):
                    rows.append(repo_rel(image))
                else:
                    skipped_labeled += 1
            if not rows:
                raise SystemExit(f"no zero-label rows for bucket={bucket} split={split}")
            list_path = out_dir / f"{args.tag}_{bucket}_{split}.txt"
            bucket_lists[split] = list_path
            combined_by_split[split].extend(rows)
            bucket_summary["splits"][split] = {
                "zero_label_rows": len(rows),
                "skipped_labeled_rows": skipped_labeled,
                "list": repo_rel(list_path),
            }
            if not args.dry_run:
                write_list(list_path, rows)
        lists_by_bucket[bucket] = bucket_lists
        summary["buckets"][bucket] = bucket_summary

    if args.train_hardneg_out:
        train_hardneg_rows: list[str] = []
        for bucket, pattern in BUCKETS.items():
            image_dir = data_root / "images" / "train"
            rows: list[str] = []
            skipped_labeled = 0
            for image in sorted(image_dir.glob(f"*{pattern}*")):
                label = label_path_for_image(data_root, "train", image)
                if is_zero_label(label):
                    rows.append(repo_rel(image))
                else:
                    skipped_labeled += 1
            cap = train_caps[bucket]
            selected = rows if cap >= len(rows) else rng.sample(rows, cap)
            selected = sorted(selected)
            train_hardneg_rows.extend(selected)
            summary["train_hardneg"][bucket] = {
                "available_zero_label_rows": len(rows),
                "skipped_labeled_rows": skipped_labeled,
                "selected_rows": len(selected),
                "cap": cap,
            }
        train_hardneg_rows = list(dict.fromkeys(train_hardneg_rows))
        summary["train_hardneg"]["combined"] = {
            "selected_rows": len(train_hardneg_rows),
            "seed": args.train_hardneg_seed,
            "list": repo_rel(resolve(args.train_hardneg_out)),
            "label_policy": "Train-split zero-label money hard negatives; safe for training, not held-out proof.",
        }
        if not args.dry_run:
            write_list(resolve(args.train_hardneg_out), train_hardneg_rows)

    combined_lists: dict[str, Path] = {}
    for split, rows in combined_by_split.items():
        unique_rows = list(dict.fromkeys(rows))
        list_path = out_dir / f"{args.tag}_combined_{split}.txt"
        combined_lists[split] = list_path
        summary["combined"][split] = {
            "zero_label_rows": len(unique_rows),
            "list": repo_rel(list_path),
        }
        if not args.dry_run:
            write_list(list_path, unique_rows)

    configs: dict[str, str] = {}
    for bucket, split_lists in {**lists_by_bucket, "combined": combined_lists}.items():
        config_path = config_dir / f"{args.tag}_{bucket}_eval.yaml"
        config_payload = {
            "path": "../..",
            "train": "images/train",
            "val": repo_rel(split_lists["val"]),
            "test": repo_rel(split_lists["test"]),
            "names": names,
            "cashsnap_heldout_zero_label_guardrail": {
                "schema": "cashsnap_heldout_zero_label_money_guardrail_eval_v1",
                "tag": args.tag,
                "bucket": bucket,
                "source_data": repo_rel(data_config),
                "label_policy": "Eval-only held-out zero-label rows; do not include in detector training.",
                "promotion_use": (
                    "Measure target-class hallucinations on foreign, out-of-schema, or unknown-money rows "
                    "from val/test splits only."
                ),
            },
        }
        configs[bucket] = repo_rel(config_path)
        if not args.dry_run:
            write_yaml(config_path, config_payload)

    summary["configs"] = configs
    summary_path = out_dir / f"{args.tag}_summary.json"
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(
        "heldout_zero_label_guardrails "
        f"tag={args.tag} combined={summary['combined']} configs={configs}",
        flush=True,
    )
    print(f"summary={repo_rel(summary_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
