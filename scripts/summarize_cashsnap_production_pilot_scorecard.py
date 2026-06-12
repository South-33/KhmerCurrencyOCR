#!/usr/bin/env python
"""Summarize the blended CashSnap production-pilot detector scorecard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


METRIC_FILES = {
    "full_test": "eval_full_test_i416_metrics.json",
    "strict_clean_test": "eval_strict_clean_test_i416_metrics.json",
    "source_excluded_clean_test": "eval_strict_clean_no_khmer_test_i416_metrics.json",
    "partial_test_conf005": "light_eval_countablepartial_filtered_test_conf005.json",
    "partial_test_conf015": "light_eval_countablepartial_filtered_test_conf015.json",
    "partial_test_conf025": "light_eval_countablepartial_filtered_test_conf025.json",
    "partial_val_conf005": "light_eval_countablepartial_filtered_val_conf005.json",
    "partial_val_conf015": "light_eval_countablepartial_filtered_val_conf015.json",
    "partial_val_conf025": "light_eval_countablepartial_filtered_val_conf025.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[], metavar="LABEL=RUN_DIR")
    parser.add_argument(
        "--metric-override",
        action="append",
        default=[],
        metavar="LABEL:METRIC=PATH",
        help="Use a specific metric JSON for one model/metric slot.",
    )
    parser.add_argument("--background-json", action="append", default=[], type=Path)
    parser.add_argument("--baseline-label", default="pilotA")
    parser.add_argument("--out-json", required=True, type=Path)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def parse_model_specs(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--model expected LABEL=RUN_DIR, got {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise SystemExit(f"empty model label in {value!r}")
        out[label] = resolve(raw_path)
    if not out:
        raise SystemExit("at least one --model is required")
    return out


def parse_metric_overrides(values: list[str]) -> dict[tuple[str, str], Path]:
    out: dict[tuple[str, str], Path] = {}
    known_metrics = set(METRIC_FILES)
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[0]:
            raise SystemExit(f"--metric-override expected LABEL:METRIC=PATH, got {value!r}")
        raw_key, raw_path = value.split("=", 1)
        label, metric_name = raw_key.split(":", 1)
        label = label.strip()
        metric_name = metric_name.strip()
        if not label or not metric_name:
            raise SystemExit(f"empty label or metric in --metric-override {value!r}")
        if metric_name not in known_metrics:
            raise SystemExit(
                f"unknown metric {metric_name!r}; expected one of {sorted(known_metrics)}"
            )
        out[(label, metric_name)] = resolve(raw_path)
    return out


def metric_summary(path: Path) -> dict[str, Any]:
    data = read_json(path)
    box = data.get("box", {})
    return {
        "path": repo_rel(path),
        "map50_95": box.get("map50_95"),
        "map50": box.get("map50"),
        "precision": box.get("precision"),
        "recall": box.get("recall"),
    }


def lightweight_summary(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return {
        "path": repo_rel(path),
        "images": data.get("images"),
        "recall": data.get("recall"),
        "precision": data.get("precision"),
        "predictions": data.get("predictions"),
        "tp": data.get("tp"),
        "fp": data.get("fp"),
        "fn": data.get("fn"),
        "ignored_detections": data.get("ignored_detections"),
        "ignored_by_class": data.get("ignored_by_class", {}),
    }


def background_rows(paths: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for path in paths:
        payload = read_json(path)
        for row in payload.get("rows", []):
            label = str(row.get("model_label", ""))
            source = str(row.get("image_root", ""))
            conf = str(row.get("conf", ""))
            if not label or not source:
                continue
            key = f"{source}|conf={conf}"
            rows.setdefault(label, {})[key] = {
                "path": repo_rel(resolve(path)),
                "source": source,
                "conf": row.get("conf"),
                "images": row.get("images"),
                "images_with_fp": row.get("images_with_fp"),
                "detections": row.get("detections"),
                "fp_per_image": row.get("fp_per_image"),
                "by_class": row.get("by_class", {}),
                "ignored_detections": row.get("ignored_detections"),
                "ignored_by_class": row.get("ignored_by_class", {}),
            }
    return rows


def delta(value: Any, baseline: Any) -> float | None:
    if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
        return float(value) - float(baseline)
    return None


def add_deltas(models: dict[str, dict[str, Any]], baseline_label: str) -> None:
    baseline = models.get(baseline_label)
    if not baseline:
        return
    for label, model in models.items():
        if label == baseline_label:
            continue
        model["delta_vs_" + baseline_label] = {
            "full_map50_95": delta(
                model.get("full_test", {}).get("map50_95"),
                baseline.get("full_test", {}).get("map50_95"),
            ),
            "strict_clean_map50_95": delta(
                model.get("strict_clean_test", {}).get("map50_95"),
                baseline.get("strict_clean_test", {}).get("map50_95"),
            ),
            "source_excluded_map50_95": delta(
                model.get("source_excluded_clean_test", {}).get("map50_95"),
                baseline.get("source_excluded_clean_test", {}).get("map50_95"),
            ),
            "partial_test_conf005_recall": delta(
                model.get("partial_test_conf005", {}).get("recall"),
                baseline.get("partial_test_conf005", {}).get("recall"),
            ),
            "partial_test_conf005_precision": delta(
                model.get("partial_test_conf005", {}).get("precision"),
                baseline.get("partial_test_conf005", {}).get("precision"),
            ),
            "partial_val_conf005_recall": delta(
                model.get("partial_val_conf005", {}).get("recall"),
                baseline.get("partial_val_conf005", {}).get("recall"),
            ),
            "partial_val_conf005_precision": delta(
                model.get("partial_val_conf005", {}).get("precision"),
                baseline.get("partial_val_conf005", {}).get("precision"),
            ),
            "partial_val_conf015_recall": delta(
                model.get("partial_val_conf015", {}).get("recall"),
                baseline.get("partial_val_conf015", {}).get("recall"),
            ),
            "partial_val_conf015_precision": delta(
                model.get("partial_val_conf015", {}).get("precision"),
                baseline.get("partial_val_conf015", {}).get("precision"),
            ),
            "partial_val_conf025_recall": delta(
                model.get("partial_val_conf025", {}).get("recall"),
                baseline.get("partial_val_conf025", {}).get("recall"),
            ),
            "partial_val_conf025_precision": delta(
                model.get("partial_val_conf025", {}).get("precision"),
                baseline.get("partial_val_conf025", {}).get("precision"),
            ),
        }


def main() -> int:
    args = parse_args()
    model_dirs = parse_model_specs(args.model)
    metric_overrides = parse_metric_overrides(args.metric_override)
    bg_rows = background_rows([resolve(path) for path in args.background_json])
    models: dict[str, dict[str, Any]] = {}
    for label, run_dir in model_dirs.items():
        model: dict[str, Any] = {"run_dir": repo_rel(run_dir)}
        for metric_name, file_name in METRIC_FILES.items():
            path = metric_overrides.get((label, metric_name), run_dir / file_name)
            if not path.exists():
                continue
            if metric_name.startswith("partial_"):
                model[metric_name] = lightweight_summary(path)
            else:
                model[metric_name] = metric_summary(path)
        model["background_fp"] = bg_rows.get(label, {})
        models[label] = model

    add_deltas(models, args.baseline_label)
    summary = {
        "schema": "cashsnap_production_pilot_scorecard_v1",
        "baseline_label": args.baseline_label,
        "eval_blend": {
            "clean_non_overlap": [
                "full real test mAP50-95",
                "strict semantic+leakage-clean test mAP50-95",
                "source-excluded strict-clean test mAP50-95",
            ],
            "messy_countable_positive": [
                "filtered countable partial val/test lightweight recall and precision",
                "threshold sweep at conf 0.05/0.15/0.25 when available",
            ],
            "count_safety_negative": [
                "held-out val/test zero-label foreign/out-of-schema money rows",
                "likely true-empty test rows",
                "train-split hard-negative rows are diagnostic only if used in training",
            ],
        },
        "fairness_notes": [
            "Do not use train-split hard-negative rows as held-out promotion proof after adding them to a blend.",
            "Do not promote AP-only gains; partial recall/precision and unknown-money false positives are co-equal gates.",
            "A one-detector pilot may choose an operating confidence, but the scorecard must show the recall/safety trade.",
        ],
        "models": models,
    }
    out_path = resolve(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"scorecard={repo_rel(out_path)} models={','.join(models)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
