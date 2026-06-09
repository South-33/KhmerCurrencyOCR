#!/usr/bin/env python
"""Build a YOLO config with negative rows mined from a background-FP probe."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_BASE = (
    ROOT
    / "configs"
    / "webgl_ablation"
    / "cashsnap_target_anchor_transplant_poisson_contact_geoscale205_minshort190_bal96_scaled_foreignhardneg6_probe_puresynth_realval_v1.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--probe-json", type=Path, required=True)
    parser.add_argument("--model-label", default="", help="Probe row model_label. Defaults to the first row.")
    parser.add_argument("--image-root", default="", help="Probe row image_root. Defaults to any root for --model-label.")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--selection", choices=("top", "spread", "random"), default="top")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument(
        "--replace-existing-root",
        type=Path,
        default=None,
        help="Remove existing train rows under this image root before inserting the mined dose. Defaults to the probe image_root.",
    )
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


def image_rows(root: Path) -> list[str]:
    image_dir = resolve(root)
    if not image_dir.exists():
        raise SystemExit(f"missing image dir: {repo_rel(image_dir)}")
    return [
        repo_rel(path)
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


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


def normalize_repo_path(value: str) -> str:
    return repo_rel(resolve(Path(value)))


def find_probe_row(probe: dict[str, Any], model_label: str, image_root: str) -> dict[str, Any]:
    rows = probe.get("rows", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("probe JSON must contain rows")
    wanted_root = normalize_repo_path(image_root) if image_root else ""
    matches: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if model_label and str(row.get("model_label", "")) != model_label:
            continue
        if wanted_root and normalize_repo_path(str(row.get("image_root", ""))) != wanted_root:
            continue
        matches.append(row)
    if not model_label and not image_root:
        return rows[0]
    if not matches:
        raise SystemExit(f"probe row not found for model_label={model_label!r} image_root={image_root!r}")
    if len(matches) > 1:
        raise SystemExit(f"probe row selector is ambiguous: model_label={model_label!r} image_root={image_root!r}")
    return matches[0]


def select_ranked_rows(
    ranked: list[dict[str, Any]],
    *,
    count: int,
    selection: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if len(ranked) < count:
        raise SystemExit(
            f"not enough unique FP-mined negative images: {len(ranked)} < {count}; "
            "rerun the probe with a lower conf or larger --json-top-k"
        )
    if selection == "top":
        return ranked[:count]
    if selection == "random":
        indexes = sorted(rng.sample(range(len(ranked)), count))
        return [ranked[index] for index in indexes]
    if selection == "spread":
        if count == 1:
            return [ranked[0]]
        indexes = sorted({round(index * (len(ranked) - 1) / (count - 1)) for index in range(count)})
        cursor = 0
        while len(indexes) < count and cursor < len(ranked):
            if cursor not in indexes:
                indexes.append(cursor)
            cursor += 1
        indexes = sorted(indexes[:count])
        return [ranked[index] for index in indexes]
    raise SystemExit(f"unsupported selection: {selection}")


def mined_rows(
    row: dict[str, Any],
    count: int,
    *,
    selection: str,
    rng: random.Random,
) -> tuple[list[str], dict[str, Any]]:
    if count < 1:
        raise SystemExit("--count must be at least 1")
    top = row.get("top", [])
    if not isinstance(top, list) or not top:
        raise SystemExit("probe row has no top false-positive detections to mine")

    per_image: dict[str, dict[str, Any]] = {}
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in top:
        if not isinstance(item, dict):
            continue
        image = str(item.get("image", "")).strip()
        if not image:
            continue
        confidence = float(item.get("confidence", 0.0) or 0.0)
        class_name = str(item.get("class", "")).strip()
        record = per_image.setdefault(
            image,
            {
                "image": image,
                "detections_in_top": 0,
                "max_confidence": 0.0,
                "classes": Counter(),
            },
        )
        record["detections_in_top"] += 1
        record["max_confidence"] = max(float(record["max_confidence"]), confidence)
        if class_name:
            record["classes"][class_name] += 1
            class_counts[image][class_name] += 1

    ranked = sorted(
        per_image.values(),
        key=lambda item: (
            float(item["max_confidence"]),
            int(item["detections_in_top"]),
            str(item["image"]),
        ),
        reverse=True,
    )
    selected = select_ranked_rows(ranked, count=count, selection=selection, rng=rng)
    selected_rows = [str(item["image"]) for item in selected]
    non_empty = [image for image in selected_rows if not is_empty_label(image)]
    if non_empty:
        raise SystemExit(f"FP-mined negative pool has non-empty labels: {non_empty[:5]}")
    report = {
        "source_top_detections": len(top),
        "unique_fp_images_in_top": len(ranked),
        "selection": selection,
        "selected_images": [
            {
                "image": str(item["image"]),
                "detections_in_top": int(item["detections_in_top"]),
                "max_confidence": round(float(item["max_confidence"]), 6),
                "classes": dict(sorted(item["classes"].items())),
            }
            for item in selected
        ],
    }
    return selected_rows, report


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


def main() -> int:
    args = parse_args()
    probe_path = resolve(args.probe_json)
    probe = read_json(probe_path)
    row = find_probe_row(probe, args.model_label, args.image_root)
    selected_rows, mining_report = mined_rows(
        row,
        args.count,
        selection=args.selection,
        rng=random.Random(args.seed),
    )

    base_path = resolve(args.base)
    base_config = read_yaml(base_path)
    base_rows, base_sources = train_rows(base_path, base_config)
    image_root = normalize_repo_path(str(row.get("image_root", "")))
    replace_root = repo_rel(resolve(args.replace_existing_root)) if args.replace_existing_root else image_root
    kept_rows, removed_rows, insert_at = remove_existing(base_rows, replace_root)
    combined_rows = kept_rows[:insert_at] + selected_rows + kept_rows[insert_at:]
    if len(set(combined_rows)) != len(combined_rows):
        raise SystemExit("FP-mined negative dose introduced duplicate train rows")

    report = {
        "schema": "cashsnap_fp_mined_negative_dose_v1",
        "base_config": repo_rel(base_path),
        "base_train_sources": base_sources,
        "probe_json": repo_rel(probe_path),
        "probe_schema": probe.get("schema", ""),
        "model_label": row.get("model_label", ""),
        "model": row.get("model", ""),
        "image_root": image_root,
        "conf": row.get("conf", ""),
        "replace_existing_root": replace_root,
        "requested_count": int(args.count),
        "selection": args.selection,
        "seed": int(args.seed),
        "removed_existing_rows": int(removed_rows),
        "insert_at": int(insert_at),
        "base_images": len(base_rows),
        "selected_fp_mined_negative_images": len(selected_rows),
        "combined_images": len(combined_rows),
        **mining_report,
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
    sources["fp_mined_negative_dose_base_config"] = repo_rel(base_path)
    sources["fp_mined_negative_probe_json"] = repo_rel(probe_path)
    sources["fp_mined_negative_image_root"] = image_root
    config["cashsnap_sources"] = sources
    config["cashsnap_fp_mined_negative_dose"] = report
    policy = copy.deepcopy(config.get("cashsnap_policy", {}))
    if not isinstance(policy, dict):
        policy = {}
    policy["intended_use"] = (
        "pure-synth TSTR probe adding zero-label negatives selected from donor false positives"
    )
    policy["promotion_rule"] = (
        "reject unless mined negatives reduce real-empty FPs while preserving full, clean-visible, "
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
