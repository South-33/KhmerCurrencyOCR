#!/usr/bin/env python
"""Build the CashSnap Production Pilot v1 list-backed YOLO config."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAG = "cashsnap_production_pilot_v1"

BASE_REAL_LIST = Path("configs/generated_lists/webgl_ablation/cashsnap_v1_balanced_real_only_probe_train.txt")
BASE_MIX_CONFIG = Path("configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_v1.yaml")
BASE_MIX_LIST = Path(
    "configs/generated_lists/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_v1_train.txt"
)
VIS70_LIST = Path("configs/generated_lists/webgl_ablation/cashsnap_countsafe_vis70_p24_v1_extra.txt")
VIS70_CENTER50_LIST = Path("configs/generated_lists/webgl_ablation/cashsnap_countsafe_vis70_plus_center50_p24_v1_extra.txt")
REVIEWED_POSITIVE_LISTS = [
    Path("configs/generated_lists/webgl_ablation/cashsnap_reviewed_countable_khr_borderpartial24_v1_extra.txt"),
    Path("configs/generated_lists/webgl_ablation/cashsnap_reviewed_visibleevidence_obvioussafe_v1_extra.txt"),
    Path("runs/cashsnap/visible_evidence_qa_mined_partialstress_cap8_v1/strict_partial_khr_codex_reviewed_v1_images.txt"),
    Path("runs/cashsnap/real_overlap_focus_materialized_reviewed_v1/train_anchor_candidate_images.txt"),
]
HARDNEG_LISTS = [
    Path("runs/cashsnap/visible_evidence_qa_lowrisk_empty_candidates_v1/coin_hardneg12_codex_reviewed_v1_images.txt"),
    Path("configs/generated_lists/webgl_ablation/cashsnap_reviewed_foreignhardneg_koreanwon24_v1_extra.txt"),
]
TRAIN_EMPTY_FP_ANALOGS = Path(
    "runs/cashsnap/ve_v4_trainanchors_guard_v2/champion_train_empty_fp_analogs_allclasses_conf015_v1.json"
)
DEFAULT_OUT_CONFIG = Path("configs/webgl_ablation/cashsnap_production_pilot_v1.yaml")
DEFAULT_OUT_LIST = Path("configs/generated_lists/webgl_ablation/cashsnap_production_pilot_v1_train.txt")
DEFAULT_OUT_MANIFEST = Path("configs/generated_lists/webgl_ablation/cashsnap_production_pilot_v1_manifest.csv")
DEFAULT_OUT_SUMMARY = Path("configs/generated_lists/webgl_ablation/cashsnap_production_pilot_v1_summary.json")
PROTECTED_CLASS_NAMES = {"KHR_50000", "KHR_20000", "USD_50", "USD_100"}
BLOCKED_PARTIAL_RE = re.compile(r"corner_[xy]_vis0p5|corner_.*vis0p5", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--out-config", type=Path, default=DEFAULT_OUT_CONFIG)
    parser.add_argument("--out-list", type=Path, default=DEFAULT_OUT_LIST)
    parser.add_argument("--out-manifest", type=Path, default=DEFAULT_OUT_MANIFEST)
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_OUT_SUMMARY)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--clean-real-repeat", type=int, default=3)
    parser.add_argument("--strict-synth-repeat", type=int, default=2)
    parser.add_argument("--partial-generated-max", type=int, default=240)
    parser.add_argument("--partial-repeat", type=int, default=2)
    parser.add_argument("--hardneg-repeat", type=int, default=6)
    parser.add_argument("--protector-exposures", type=int, default=180)
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
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(resolved)}: expected YAML mapping")
    return data


def read_list(path: Path) -> list[str]:
    resolved = resolve(path)
    rows: list[str] = []
    for raw_line in resolved.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            rows.append(repo_rel(resolve(line)))
    if not rows:
        raise SystemExit(f"empty image list: {repo_rel(resolved)}")
    return rows


def ordered_unique(rows: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        if row not in seen:
            seen.add(row)
            out.append(row)
    return out


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
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} malformed class id: {parts[0]}") from exc
    return class_ids


def names_by_id(config: dict[str, Any]) -> dict[int, str]:
    raw = config.get("names", {})
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    raise SystemExit("base config has no YOLO names mapping")


def sample_without_replacement(rows: list[str], count: int, rng: random.Random) -> list[str]:
    unique = ordered_unique(rows)
    if count >= len(unique):
        selected = list(unique)
    else:
        selected = rng.sample(unique, count)
    return selected


def sample_with_replacement(rows: list[str], count: int, rng: random.Random) -> list[str]:
    unique = ordered_unique(rows)
    if count < 0:
        raise SystemExit("sample count must be non-negative")
    if count == 0:
        return []
    if not unique:
        raise SystemExit("cannot sample from an empty pool")
    selected: list[str] = []
    pool = list(unique)
    while len(selected) < count:
        rng.shuffle(pool)
        selected.extend(pool[: count - len(selected)])
    return selected


def repeat_rows(rows: list[str], repeat: int) -> list[str]:
    if repeat < 0:
        raise SystemExit("repeat counts must be non-negative")
    return list(rows) * repeat


def class_counts(rows: list[str]) -> tuple[Counter[int], int]:
    counts: Counter[int] = Counter()
    empty = 0
    for row in rows:
        ids = label_class_ids(row)
        if not ids:
            empty += 1
        counts.update(ids)
    return counts, empty


def named_counts(counts: Counter[int], names: dict[int, str]) -> dict[str, int]:
    return {names.get(class_id, f"class_{class_id}"): counts[class_id] for class_id in sorted(counts)}


def train_empty_fp_rows(path: Path) -> list[str]:
    resolved = resolve(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    out: list[str] = []
    for row in payload.get("rows", []):
        for item in row.get("top", []):
            image = item.get("image")
            if isinstance(image, str):
                out.append(repo_rel(resolve(image)))
    return ordered_unique(out)


def validate_hardneg(rows: list[str]) -> None:
    offenders = [row for row in rows if label_class_ids(row)]
    if offenders:
        raise SystemExit(f"hard-negative rows must be zero-label; offenders: {offenders[:5]}")
    non_train = [row for row in rows if "/images/train/" not in row.replace("\\", "/")]
    if non_train:
        raise SystemExit(f"hard-negative rows must come from train split; offenders: {non_train[:5]}")


def validate_partial(rows: list[str]) -> None:
    blocked = [row for row in rows if BLOCKED_PARTIAL_RE.search(row)]
    if blocked:
        raise SystemExit(f"blocked corner-50 partial rows slipped into pilot: {blocked[:5]}")
    empty = [row for row in rows if not label_class_ids(row)]
    if empty:
        raise SystemExit(f"partial rows must be labeled positives; offenders: {empty[:5]}")


def write_list(path: Path, rows: list[str]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_manifest(path: Path, component_rows: list[tuple[str, str]], names: dict[int, str]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "component", "image", "label_ids", "label_names", "is_empty"],
        )
        writer.writeheader()
        for index, (component, image) in enumerate(component_rows):
            ids = label_class_ids(image)
            writer.writerow(
                {
                    "index": index,
                    "component": component,
                    "image": image,
                    "label_ids": " ".join(str(class_id) for class_id in ids),
                    "label_names": " ".join(names.get(class_id, f"class_{class_id}") for class_id in ids),
                    "is_empty": int(not ids),
                }
            )


def component_pairs(component: str, rows: list[str]) -> list[tuple[str, str]]:
    return [(component, row) for row in rows]


def build_config(
    args: argparse.Namespace,
    names: dict[int, str],
    component_sources: dict[str, list[str]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    base_config = read_yaml(BASE_MIX_CONFIG)
    out_config = resolve(args.out_config)
    return {
        "path": rel_between(out_config.parent, ROOT),
        "train": repo_rel(resolve(args.out_list)),
        "val": base_config["val"],
        "test": base_config["test"],
        "names": names,
        "cashsnap_production_pilot": {
            "schema": "cashsnap_production_pilot_config_v1",
            "tag": args.tag,
            "base_reference_config": repo_rel(resolve(BASE_MIX_CONFIG)),
            "recommended_init_checkpoint": (
                "runs/cashsnap/"
                "fixed_step_countsafe_vis70_p24_from_last_e50_i416_b2_w0_adamw_lr5e6_"
                "nowarmup_noamp_cachefalse_freeze22_steps318_seed0/weights/last.pt"
            ),
            "component_sources": component_sources,
            "summary": summary,
            "label_policy": [
                "Train from existing train-split/list-backed rows only; no new data root is introduced.",
                "Human-countable partial positives are allowed; unreviewed corner_*.vis0p5 rows are blocked.",
                "Hard-negative rows must be zero-label train-split rows.",
                "Eval/failure-slice-mined rows are kept out of hard-negative training unless they are train-split analogs.",
            ],
        },
        "cashsnap_policy": {
            "intended_use": (
                "Production Pilot v1 one-detector training recipe: clean replay plus strictbest synth, "
                "countable partial/visible-evidence positives, train-safe hard negatives, and protected "
                "high-risk class replay."
            ),
            "promotion_rule": (
                "Compare against the p24 synth+real champion and p24 vis70 clue on full real, strict clean, "
                "source-excluded clean, filtered countable partial, source-FP review queue, hard-slice "
                "count/value, and per-class guards; kill if gains come from duplicate or wrong-denomination boxes."
            ),
        },
    }


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    base_config = read_yaml(BASE_MIX_CONFIG)
    names = names_by_id(base_config)
    name_to_id = {value: key for key, value in names.items()}
    protected_ids = {name_to_id[name] for name in PROTECTED_CLASS_NAMES}

    clean_real = read_list(BASE_REAL_LIST)
    base_mix = read_list(BASE_MIX_LIST)
    clean_real_set = set(clean_real)
    strict_synth = [row for row in base_mix if row not in clean_real_set]
    if len(clean_real) + len(strict_synth) != len(base_mix):
        raise SystemExit("base mix is expected to be clean-real rows plus strictbest synth rows")

    partial_generated = ordered_unique(read_list(VIS70_LIST) + read_list(VIS70_CENTER50_LIST))
    partial_generated = [row for row in partial_generated if not BLOCKED_PARTIAL_RE.search(row)]
    reviewed_positive: list[str] = []
    for source in REVIEWED_POSITIVE_LISTS:
        reviewed_positive.extend(read_list(source))
    reviewed_positive = ordered_unique(reviewed_positive)
    partial_generated_selected = sample_without_replacement(partial_generated, args.partial_generated_max, rng)
    partial_rows = ordered_unique(partial_generated_selected + reviewed_positive)
    validate_partial(partial_rows)

    hardneg_rows: list[str] = []
    for source in HARDNEG_LISTS:
        hardneg_rows.extend(read_list(source))
    hardneg_rows.extend(train_empty_fp_rows(TRAIN_EMPTY_FP_ANALOGS))
    hardneg_rows = ordered_unique(hardneg_rows)
    validate_hardneg(hardneg_rows)

    protector_pool = [
        row
        for row in ordered_unique(clean_real + strict_synth)
        if protected_ids.intersection(label_class_ids(row))
    ]
    protector_rows = sample_with_replacement(protector_pool, args.protector_exposures, rng)

    components = {
        "clean_real_replay": repeat_rows(clean_real, args.clean_real_repeat),
        "strictbest_synth_replay": repeat_rows(strict_synth, args.strict_synth_repeat),
        "countable_partial_replay": repeat_rows(partial_rows, args.partial_repeat),
        "train_safe_hard_negative_replay": repeat_rows(hardneg_rows, args.hardneg_repeat),
        "high_risk_class_protectors": protector_rows,
    }
    component_rows: list[tuple[str, str]] = []
    for component, rows in components.items():
        component_rows.extend(component_pairs(component, rows))
    rng.shuffle(component_rows)
    final_rows = [row for _, row in component_rows]
    final_counts, final_empty = class_counts(final_rows)
    summary = {
        "seed": args.seed,
        "rows": len(final_rows),
        "unique_rows": len(set(final_rows)),
        "duplicate_exposures": len(final_rows) - len(set(final_rows)),
        "empty_rows": final_empty,
        "component_rows": {component: len(rows) for component, rows in components.items()},
        "component_unique_rows": {component: len(set(rows)) for component, rows in components.items()},
        "component_percent": {
            component: round(len(rows) / len(final_rows), 4) if final_rows else 0.0
            for component, rows in components.items()
        },
        "class_counts": named_counts(final_counts, names),
        "clean_real_unique": len(clean_real),
        "strict_synth_unique": len(strict_synth),
        "partial_generated_available": len(partial_generated),
        "partial_generated_selected": len(partial_generated_selected),
        "reviewed_positive_unique": len(reviewed_positive),
        "partial_total_unique": len(partial_rows),
        "hardneg_unique": len(hardneg_rows),
        "protector_pool_unique": len(protector_pool),
    }
    component_sources = {
        "clean_real_replay": [repo_rel(resolve(BASE_REAL_LIST))],
        "strictbest_synth_replay": [repo_rel(resolve(BASE_MIX_LIST)), "derived: rows not in clean_real_replay"],
        "countable_partial_replay": [
            repo_rel(resolve(VIS70_LIST)),
            repo_rel(resolve(VIS70_CENTER50_LIST)),
            *[repo_rel(resolve(source)) for source in REVIEWED_POSITIVE_LISTS],
        ],
        "train_safe_hard_negative_replay": [
            *[repo_rel(resolve(source)) for source in HARDNEG_LISTS],
            repo_rel(resolve(TRAIN_EMPTY_FP_ANALOGS)),
        ],
        "high_risk_class_protectors": [repo_rel(resolve(BASE_REAL_LIST)), repo_rel(resolve(BASE_MIX_LIST))],
    }
    output_config = build_config(args, names, component_sources, summary)

    if not args.dry_run:
        write_list(args.out_list, final_rows)
        write_manifest(args.out_manifest, component_rows, names)
        write_json(args.out_summary, summary)
        out_config = resolve(args.out_config)
        out_config.parent.mkdir(parents=True, exist_ok=True)
        out_config.write_text(yaml.safe_dump(output_config, sort_keys=False), encoding="utf-8")

    print(
        "production_pilot "
        f"tag={args.tag} rows={summary['rows']} unique={summary['unique_rows']} "
        f"components={summary['component_rows']} empty={summary['empty_rows']}",
        flush=True,
    )
    print(f"out_config={repo_rel(resolve(args.out_config))}", flush=True)
    print(f"out_list={repo_rel(resolve(args.out_list))}", flush=True)
    print(f"out_summary={repo_rel(resolve(args.out_summary))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
