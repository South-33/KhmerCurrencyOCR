#!/usr/bin/env python
"""Build train-only analog queues from the current strict-best failure ledger.

This does not create training data. It turns current real-test failure modes into
train-split positive/style/review seeds for the next paired synthetic generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET_SOURCE_GROUPS = {
    "billsbank",
    "cambodia_currency_project",
    "khmer_scan",
    "khmer_us_currency",
    "usd_total",
}
MIXED_CURRENCY_SOURCE_GROUPS = {"asian_currency", "cashcountingxl"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lightweight-eval", required=True, type=Path)
    parser.add_argument("--data", default="data/cashsnap_v1/data.yaml", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--semantic-manifest",
        default="runs/cashsnap/empty_label_semantic_bridge_train_v1/manifest.csv",
        type=Path,
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--weak-class",
        default="",
        help="Comma-separated classes. Empty means infer from lightweight FN/recall thresholds.",
    )
    parser.add_argument("--min-positive-fn", type=int, default=40)
    parser.add_argument("--max-positive-recall", type=float, default=0.50)
    parser.add_argument("--max-positive-per-class-source", type=int, default=120)
    parser.add_argument("--min-source-background-fp", type=int, default=100)
    parser.add_argument("--max-source-precision", type=float, default=0.10)
    parser.add_argument("--min-source-fp", type=int, default=100)
    parser.add_argument("--max-seeds-per-source-bucket", type=int, default=160)
    parser.add_argument("--safe-bucket", default="likely_true_empty")
    parser.add_argument("--review-bucket", default="currency_review,model_review,student_overfire_review")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path | str) -> str:
    value = resolve(path).resolve()
    try:
        return value.relative_to(ROOT).as_posix()
    except ValueError:
        return value.as_posix()


def parse_csv_tokens(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(path)} must be a YAML mapping")
    return data


def load_lightweight_eval(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "cashsnap_yolo_lightweight_recall_eval_v2":
        raise SystemExit(f"{repo_rel(path)} is not a lightweight eval v2 JSON")
    return payload


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw = Path(str(config.get("path", "."))).expanduser()
    return raw if raw.is_absolute() else (config_path.parent / raw).resolve()


def split_images(config_path: Path, config: dict[str, Any], split: str) -> list[Path]:
    root = data_root(config_path, config)
    split_value = config.get(split)
    if split_value is None:
        raise SystemExit(f"{repo_rel(config_path)} has no split {split!r}")
    values = split_value if isinstance(split_value, list) else [split_value]
    rows: list[Path] = []
    for raw in values:
        path = Path(str(raw))
        resolved = path if path.is_absolute() else root / path
        if resolved.suffix.lower() == ".txt":
            for raw_line in resolved.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    image = Path(line)
                    rows.append(image if image.is_absolute() else root / image)
        else:
            rows.extend(sorted(item for item in resolved.glob("*") if item.suffix.lower() in IMAGE_EXTS))
    return rows


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def source_group(image_path: Path) -> str:
    name = image_path.name.lower()
    for group in sorted(TARGET_SOURCE_GROUPS | MIXED_CURRENCY_SOURCE_GROUPS, key=len, reverse=True):
        if name.startswith(group):
            return group
    match = re.match(r"([a-z]+(?:_[a-z]+)?)_", name)
    return match.group(1) if match else image_path.stem.split("_")[0].lower()


def canonical_image_key(path: Path | str) -> str:
    name = Path(str(path)).name.lower()
    if ".rf." in name:
        name = name.split(".rf.", 1)[0]
    return re.sub(r"\.(jpg|jpeg|png|webp)$", "", name)


def names_from_config(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names") or {}
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    raise SystemExit("data YAML must contain names")


def read_labels(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(path)}:{line_no} expected 5 YOLO fields")
        class_id = int(parts[0])
        cx, cy, bw, bh = [float(value) for value in parts[1:]]
        rows.append(
            {
                "class_id": class_id,
                "cx": cx,
                "cy": cy,
                "w": bw,
                "h": bh,
                "area": bw * bh,
            }
        )
    return rows


def infer_positive_obligations(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, set[str]]:
    explicit = set(parse_csv_tokens(args.weak_class))
    per_class = payload.get("per_class") or {}
    selected: dict[str, set[str]] = {}
    for class_name, row in sorted(per_class.items()):
        if explicit and class_name not in explicit:
            continue
        if not explicit:
            fn = int(row.get("fn", 0) or 0)
            recall = row.get("recall")
            recall_value = float(recall) if isinstance(recall, (float, int)) else 1.0
            if fn < args.min_positive_fn and recall_value > args.max_positive_recall:
                continue
        examples = (payload.get("fn_examples_by_class") or {}).get(class_name) or []
        sources = {str(example.get("source_group")) for example in examples if example.get("source_group")}
        selected[class_name] = sources
    return selected


def infer_negative_sources(payload: dict[str, Any], args: argparse.Namespace) -> set[str]:
    sources: set[str] = set()
    for source, row in sorted((payload.get("per_source") or {}).items()):
        if not isinstance(row, dict):
            continue
        background_fp = int(row.get("background_images_with_fp", 0) or 0)
        fp = int(row.get("fp", 0) or 0)
        precision = row.get("precision")
        precision_value = float(precision) if isinstance(precision, (float, int)) else 1.0
        if background_fp >= args.min_source_background_fp:
            sources.add(str(source))
        elif fp >= args.min_source_fp and precision_value <= args.max_source_precision:
            sources.add(str(source))
    return sources


def select_positive_rows(
    *,
    images: list[Path],
    names: dict[int, str],
    obligations: dict[str, set[str]],
    max_per_class_source: int,
) -> tuple[list[dict[str, Any]], Counter[tuple[str, str]]]:
    rows_by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    available: Counter[tuple[str, str]] = Counter()
    seen_canonical: dict[tuple[str, str], set[str]] = defaultdict(set)
    for image in images:
        label_path = label_path_for_image(image)
        labels = read_labels(label_path)
        if not labels:
            continue
        group = source_group(image)
        box_count = len(labels)
        seen_image_classes: set[str] = set()
        for label in labels:
            class_name = names.get(int(label["class_id"]), str(label["class_id"]))
            wanted_sources = obligations.get(class_name)
            if wanted_sources is None:
                continue
            if wanted_sources and group not in wanted_sources:
                continue
            key = (class_name, group)
            available[key] += 1
            if class_name in seen_image_classes:
                continue
            canonical = canonical_image_key(image)
            if canonical in seen_canonical[key]:
                continue
            seen_canonical[key].add(canonical)
            seen_image_classes.add(class_name)
            rows_by_group[key].append(
                {
                    "image": repo_rel(image),
                    "label": repo_rel(label_path),
                    "source_group": group,
                    "class_id": int(label["class_id"]),
                    "class_name": class_name,
                    "box_count": box_count,
                    "box_area_norm": round(float(label["area"]), 6),
                    "box_w_norm": round(float(label["w"]), 6),
                    "box_h_norm": round(float(label["h"]), 6),
                    "source_image_key": canonical,
                    "use_policy": "positive_style_or_asset_seed_only",
                }
            )
    selected: list[dict[str, Any]] = []
    for key, rows in sorted(rows_by_group.items()):
        selected.extend(spread_select(sorted(rows, key=lambda row: row["image"]), max_per_class_source))
    return selected, available


def spread_select(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    if limit == 1:
        return [rows[0]]
    last_index = len(rows) - 1
    indexes = [round(index * last_index / (limit - 1)) for index in range(limit)]
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for index in indexes:
        if index in seen:
            continue
        seen.add(index)
        selected.append(rows[index])
    cursor = 0
    while len(selected) < limit and cursor < len(rows):
        if cursor not in seen:
            selected.append(rows[cursor])
        cursor += 1
    return selected


def read_semantic_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def select_semantic_rows(
    *,
    rows: list[dict[str, str]],
    sources: set[str],
    buckets: set[str],
    max_per_source_bucket: int,
    use_policy: str,
) -> list[dict[str, Any]]:
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_canonical: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        source = str(row.get("source_group", ""))
        bucket = str(row.get("bucket", ""))
        if source not in sources or bucket not in buckets:
            continue
        key = (source, bucket)
        canonical = canonical_image_key(str(row.get("image", "")))
        if canonical in seen_canonical[key]:
            continue
        seen_canonical[key].add(canonical)
        by_group[key].append(
            {
                "image": row.get("image", ""),
                "label": row.get("label", ""),
                "source_group": source,
                "bucket": bucket,
                "source_image_key": canonical,
                "teacher_top_class": row.get("teacher_top_class", ""),
                "teacher_top_conf": row.get("teacher_top_conf", ""),
                "student_top_class": row.get("student_top_class", ""),
                "student_top_conf": row.get("student_top_conf", ""),
                "use_policy": use_policy,
            }
        )
    selected: list[dict[str, Any]] = []
    for key, values in sorted(by_group.items()):
        selected.extend(spread_select(sorted(values, key=lambda row: row["image"]), max_per_source_bucket))
    return selected


def build_pair_queue(positives: list[dict[str, Any]], safe_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not positives or not safe_rows:
        return []
    rows: list[dict[str, Any]] = []
    safe_sorted = sorted(safe_rows, key=lambda row: (row["source_group"], row["image"]))
    for index, positive in enumerate(sorted(positives, key=lambda row: (row["class_name"], row["image"]))):
        negative = safe_sorted[index % len(safe_sorted)]
        rows.append(
            {
                "pair_rank": index + 1,
                "positive_image": positive["image"],
                "positive_label": positive["label"],
                "positive_class": positive["class_name"],
                "positive_source_group": positive["source_group"],
                "negative_style_image": negative["image"],
                "negative_source_group": negative["source_group"],
                "negative_bucket": negative["bucket"],
                "use_policy": (
                    "paired_generator_seed: synthesize a target-positive and an unknown/empty "
                    "counterexample with matched style pressure; do not train directly from this CSV."
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_image_list(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [str(row.get("image", "")).strip() for row in rows if str(row.get("image", "")).strip()]
    path.write_text("\n".join(images) + ("\n" if images else ""), encoding="utf-8")


def write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Current-Best Obligation Train Queue",
        "",
        "Train-only queues for the next paired synthetic generator. These rows are seeds, not a YOLO train list.",
        "",
        "## Counts",
        "",
        f"- Positive analog rows: {summary['rows']['positive_train_analogs']}",
        f"- Safe empty/unknown style rows: {summary['rows']['safe_negative_style_seeds']}",
        f"- Review-required unknown/currency rows: {summary['rows']['review_unknown_style_seeds']}",
        f"- Paired generator rows: {summary['rows']['generator_pair_queue']}",
        "",
        "## Use Policy",
        "",
        "- Positive rows may seed source/geometry/style extraction for synthetic positives.",
        "- Safe rows may seed empty/unknown counterexamples after visual QA.",
        "- Review rows are label-unsafe; use them for manual review or style references only.",
        "- Pair rows express generator obligations and must not be treated as training labels.",
    ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    light_path = resolve(args.lightweight_eval)
    data_path = resolve(args.data)
    manifest_path = resolve(args.semantic_manifest)
    payload = load_lightweight_eval(light_path)
    config = load_yaml(data_path)
    names = names_from_config(config)
    images = split_images(data_path, config, args.split)
    positive_obligations = infer_positive_obligations(payload, args)
    negative_sources = infer_negative_sources(payload, args)

    positives, available_positive = select_positive_rows(
        images=images,
        names=names,
        obligations=positive_obligations,
        max_per_class_source=args.max_positive_per_class_source,
    )
    semantic_rows = read_semantic_manifest(manifest_path)
    safe_buckets = set(parse_csv_tokens(args.safe_bucket))
    review_buckets = set(parse_csv_tokens(args.review_bucket))
    safe_rows = select_semantic_rows(
        rows=semantic_rows,
        sources=negative_sources,
        buckets=safe_buckets,
        max_per_source_bucket=args.max_seeds_per_source_bucket,
        use_policy="safe_empty_or_unknown_negative_style_seed_after_visual_qa",
    )
    review_rows = select_semantic_rows(
        rows=semantic_rows,
        sources=negative_sources,
        buckets=review_buckets,
        max_per_source_bucket=args.max_seeds_per_source_bucket,
        use_policy="label_unsafe_review_or_style_seed_only",
    )
    pair_rows = build_pair_queue(positives, safe_rows)

    paths = {
        "positive_train_analogs": out_dir / "positive_train_analogs.csv",
        "positive_train_analogs_list": out_dir / "positive_train_analogs.txt",
        "safe_negative_style_seeds": out_dir / "safe_negative_style_seeds.csv",
        "safe_negative_style_seeds_list": out_dir / "safe_negative_style_seeds.txt",
        "review_unknown_style_seeds": out_dir / "review_unknown_style_seeds.csv",
        "review_unknown_style_seeds_list": out_dir / "review_unknown_style_seeds.txt",
        "generator_pair_queue": out_dir / "generator_pair_queue.csv",
    }
    write_csv(paths["positive_train_analogs"], positives)
    write_image_list(paths["positive_train_analogs_list"], positives)
    write_csv(paths["safe_negative_style_seeds"], safe_rows)
    write_image_list(paths["safe_negative_style_seeds_list"], safe_rows)
    write_csv(paths["review_unknown_style_seeds"], review_rows)
    write_image_list(paths["review_unknown_style_seeds_list"], review_rows)
    write_csv(paths["generator_pair_queue"], pair_rows)

    positive_counts = Counter((row["class_name"], row["source_group"]) for row in positives)
    safe_counts = Counter((row["source_group"], row["bucket"]) for row in safe_rows)
    review_counts = Counter((row["source_group"], row["bucket"]) for row in review_rows)
    summary = {
        "schema": "cashsnap_current_best_obligation_train_queue_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "lightweight_eval": repo_rel(light_path),
        "data": repo_rel(data_path),
        "split": args.split,
        "semantic_manifest": repo_rel(manifest_path),
        "positive_obligations": {
            class_name: sorted(sources) for class_name, sources in sorted(positive_obligations.items())
        },
        "negative_source_obligations": sorted(negative_sources),
        "available_positive_by_class_source": {
            f"{class_name}|{source}": count for (class_name, source), count in sorted(available_positive.items())
        },
        "selected_positive_by_class_source": {
            f"{class_name}|{source}": count for (class_name, source), count in sorted(positive_counts.items())
        },
        "selected_safe_by_source_bucket": {
            f"{source}|{bucket}": count for (source, bucket), count in sorted(safe_counts.items())
        },
        "selected_review_by_source_bucket": {
            f"{source}|{bucket}": count for (source, bucket), count in sorted(review_counts.items())
        },
        "rows": {
            "positive_train_analogs": len(positives),
            "safe_negative_style_seeds": len(safe_rows),
            "review_unknown_style_seeds": len(review_rows),
            "generator_pair_queue": len(pair_rows),
        },
        "outputs": {key: repo_rel(path) for key, path in paths.items()},
        "settings": {
            "min_positive_fn": args.min_positive_fn,
            "max_positive_recall": args.max_positive_recall,
            "max_positive_per_class_source": args.max_positive_per_class_source,
            "min_source_background_fp": args.min_source_background_fp,
            "max_source_precision": args.max_source_precision,
            "min_source_fp": args.min_source_fp,
            "max_seeds_per_source_bucket": args.max_seeds_per_source_bucket,
            "safe_buckets": sorted(safe_buckets),
            "review_buckets": sorted(review_buckets),
        },
    }
    summary_path = out_dir / "summary.json"
    readme_path = out_dir / "README.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_readme(readme_path, summary)
    print(
        f"obligation_train_queue={repo_rel(summary_path)} "
        f"positive={len(positives)} safe={len(safe_rows)} review={len(review_rows)} "
        f"pairs={len(pair_rows)}"
    )
    print(f"readme={repo_rel(readme_path)}")


if __name__ == "__main__":
    main()
