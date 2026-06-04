#!/usr/bin/env python
"""Train, test, compare, and summarize isolated WebGL recipe ablations."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "runs" / "cashsnap" / "yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8" / "weights" / "best.pt"
DEFAULT_SUITE = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_trainable_candidates_v1.json"
DEFAULT_CONFIG_DIR = ROOT / "configs" / "webgl_ablation"
DEFAULT_PROJECT = ROOT / "runs" / "cashsnap"
DEFAULT_CLEAN_METRICS = ROOT / "runs" / "cashsnap" / "clean_checkpoint_test_i416_metrics" / "metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--clean-metrics", type=Path, default=DEFAULT_CLEAN_METRICS)
    parser.add_argument("--recipe", action="append", default=[], help="Recipe id to run. Repeatable.")
    parser.add_argument("--recipes", nargs="+", default=[], help="Recipe ids to run.")
    parser.add_argument("--include-real-only", action="store_true", default=True)
    parser.add_argument("--no-real-only", action="store_false", dest="include_real_only")
    parser.add_argument("--reuse-existing", action="store_true", default=True)
    parser.add_argument("--rerun-existing", action="store_false", dest="reuse_existing")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.00005)
    parser.add_argument("--lrf", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--warmup-bias-lr", type=float, default=None)
    parser.add_argument("--warmup-momentum", type=float, default=0.937)
    parser.add_argument("--amp", action="store_true", help="Enable AMP. Default keeps AMP disabled.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-free-ram-gb", type=float, default=1.2)
    parser.add_argument("--max-ram-percent", type=float, default=94.0)
    parser.add_argument("--max-cpu-percent", type=float, default=90.0)
    parser.add_argument("--resume-cpu-percent", type=float, default=82.0)
    parser.add_argument("--max-gpu-mem-percent", type=float, default=90.0)
    parser.add_argument("--summary-stem", default=None)
    parser.add_argument("--run-label", default="", help="Optional suffix for train/test run names and default summary stem.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def short_recipe_id(recipe_id: str) -> str:
    value = slug(recipe_id)
    if value.startswith("webgl_"):
        value = value[len("webgl_") :]
    if value.endswith("_v1"):
        value = value[: -len("_v1")]
    return value


def lr_tag(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"lr{mantissa}e{abs(int(exponent))}"


def run_label_tag(args: argparse.Namespace) -> str:
    label = slug(str(args.run_label))
    return f"_{label}" if label else ""


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected JSON object: {path}")
    return document


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def train_run_name(kind: str, args: argparse.Namespace) -> str:
    amp_tag = "amp" if args.amp else "noamp"
    warmup_tag = "nowarmup" if float(args.warmup_epochs) == 0.0 else f"warmup{args.warmup_epochs:g}"
    return (
        f"webgl_ablation_{kind}_from_clean_e{args.epochs}_i{args.imgsz}_"
        f"{args.optimizer.lower()}_{lr_tag(args.lr0)}_{warmup_tag}_{amp_tag}{run_label_tag(args)}"
    )


def test_run_name(kind: str, args: argparse.Namespace) -> str:
    warmup_tag = "nowarmup" if float(args.warmup_epochs) == 0.0 else f"warmup{args.warmup_epochs:g}"
    return f"webgl_ablation_{kind}_{warmup_tag}_test_i{args.imgsz}{run_label_tag(args)}"


def config_path(kind: str, recipe_id: str | None, config_dir: Path) -> Path:
    if recipe_id is None:
        return config_dir / "cashsnap_v1_balanced_real_only_probe.yaml"
    return config_dir / f"cashsnap_v1_plus_{slug(recipe_id)}_probe.yaml"


def train_if_needed(kind: str, data_yaml: Path, args: argparse.Namespace) -> Path:
    run_name = train_run_name(kind, args)
    best = resolve(args.project) / run_name / "weights" / "best.pt"
    if args.reuse_existing and best.exists():
        print(f"reuse_train={repo_rel(best)}", flush=True)
        return best

    command = [
        sys.executable,
        "scripts/bench_train_with_headroom.py",
        "--model",
        repo_rel(resolve(args.model)),
        "--data",
        repo_rel(data_yaml),
        "--name",
        run_name,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--optimizer",
        args.optimizer,
        "--lr0",
        str(args.lr0),
        "--lrf",
        str(args.lrf),
        "--warmup-epochs",
        str(args.warmup_epochs),
        "--warmup-bias-lr",
        str(args.warmup_bias_lr if args.warmup_bias_lr is not None else args.lr0),
        "--warmup-momentum",
        str(args.warmup_momentum),
        "--quiet",
        "--exist-ok",
        "--min-free-ram-gb",
        str(args.min_free_ram_gb),
        "--max-ram-percent",
        str(args.max_ram_percent),
        "--floor-memory-action",
        "exit",
        "--adaptive-restarts",
        "0",
    ]
    if not args.amp:
        command.append("--no-amp")
    run(command, args.dry_run)
    return best


def eval_if_needed(kind: str, weights: Path, data_yaml: Path, args: argparse.Namespace) -> Path:
    test_name = test_run_name(kind, args)
    metrics = resolve(args.project) / test_name / "metrics.json"
    if args.reuse_existing and metrics.exists():
        print(f"reuse_test={repo_rel(metrics)}", flush=True)
        return metrics

    command = [
        sys.executable,
        "scripts/run_with_headroom.py",
        "--max-percent",
        str(args.max_cpu_percent),
        "--resume-percent",
        str(args.resume_cpu_percent),
        "--max-ram-percent",
        str(args.max_ram_percent),
        "--max-gpu-mem-percent",
        str(args.max_gpu_mem_percent),
        "--min-free-ram-gb",
        str(args.min_free_ram_gb),
        "--interval",
        "2",
        "--memory-action",
        "exit",
        "--",
        sys.executable,
        "scripts/val_yolo.py",
        "--model",
        repo_rel(weights),
        "--data",
        repo_rel(data_yaml),
        "--split",
        "test",
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--device",
        str(args.device),
        "--name",
        test_name,
        "--no-plots",
        "--quiet",
        "--exist-ok",
        "--metrics-json",
        repo_rel(metrics),
    ]
    run(command, args.dry_run)
    return metrics


def compare_metrics(baseline: Path, candidate: Path, json_out: Path, args: argparse.Namespace) -> Path:
    command = [
        sys.executable,
        "scripts/compare_yolo_metrics.py",
        "--baseline",
        repo_rel(baseline),
        "--candidate",
        repo_rel(candidate),
        "--max-drop",
        "0.0",
        "--json-out",
        repo_rel(json_out),
        "--no-fail",
    ]
    run(command, args.dry_run)
    return json_out


def collect_summary_row(
    kind: str,
    recipe_id: str | None,
    metrics_path: Path,
    compare_clean_path: Path | None,
    compare_real_path: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = read_json(metrics_path)
    results = metrics.get("results", {})
    compare_clean = read_json(compare_clean_path) if compare_clean_path and compare_clean_path.exists() else {}
    compare_real = read_json(compare_real_path) if compare_real_path and compare_real_path.exists() else {}
    return {
        "kind": kind,
        "recipe_id": recipe_id or "",
        "train_run": train_run_name(kind, args),
        "test_run": test_run_name(kind, args),
        "metrics": repo_rel(metrics_path),
        "precision": results.get("metrics/precision(B)"),
        "recall": results.get("metrics/recall(B)"),
        "map50": results.get("metrics/mAP50(B)"),
        "map50_95": results.get("metrics/mAP50-95(B)"),
        "delta_vs_clean": compare_clean.get("delta"),
        "delta_vs_real_only": compare_real.get("delta"),
        "pass_vs_clean": compare_clean.get("passed"),
        "pass_vs_real_only": compare_real.get("passed"),
    }


def per_class_delta_by_name(compare_path: Path | None) -> dict[str, Any]:
    if compare_path is None or not compare_path.exists():
        return {}
    compare = read_json(compare_path)
    rows = compare.get("per_class_map50_95", [])
    if not isinstance(rows, list):
        return {}
    deltas: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("class_name") is None:
            continue
        deltas[str(row["class_name"])] = row.get("delta")
    return deltas


def collect_per_class_rows(
    kind: str,
    recipe_id: str | None,
    metrics_path: Path,
    compare_clean_path: Path | None,
    compare_real_path: Path | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    metrics = read_json(metrics_path)
    rows = metrics.get("per_class", [])
    if not isinstance(rows, list):
        return []

    clean_deltas = per_class_delta_by_name(compare_clean_path)
    real_deltas = per_class_delta_by_name(compare_real_path)
    collected = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        class_name = str(row.get("class_name", row.get("class_id", "")))
        collected.append(
            {
                "kind": kind,
                "recipe_id": recipe_id or "",
                "train_run": train_run_name(kind, args),
                "test_run": test_run_name(kind, args),
                "metrics": repo_rel(metrics_path),
                "class_id": row.get("class_id"),
                "class_name": class_name,
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "map50": row.get("map50"),
                "map50_95": row.get("map50_95"),
                "delta_vs_clean": clean_deltas.get(class_name),
                "delta_vs_real_only": real_deltas.get(class_name),
            }
        )
    return collected


def selected_recipe_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    suite = read_json(resolve(args.suite))
    rows = suite.get("recipes", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"suite has no recipes: {args.suite}")
    selected = set(args.recipe + args.recipes)
    known = {str(row.get("recipe_id")) for row in rows if isinstance(row, dict)}
    missing = sorted(selected - known)
    if missing:
        raise SystemExit(f"selected recipe(s) not found in suite: {missing}")
    return [row for row in rows if isinstance(row, dict) and (not selected or str(row.get("recipe_id")) in selected)]


def main() -> int:
    args = parse_args()
    args.model = resolve(args.model)
    args.suite = resolve(args.suite)
    args.config_dir = resolve(args.config_dir)
    args.project = resolve(args.project)
    args.clean_metrics = resolve(args.clean_metrics)

    if args.warmup_bias_lr is None:
        args.warmup_bias_lr = args.lr0

    rows_to_run: list[tuple[str, str | None]] = []
    if args.include_real_only:
        rows_to_run.append(("real_only", None))
    rows_to_run.extend((short_recipe_id(str(row["recipe_id"])), str(row["recipe_id"])) for row in selected_recipe_rows(args))

    real_only_metrics: Path | None = None
    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []

    for kind, recipe_id in rows_to_run:
        data_yaml = config_path(kind, recipe_id, args.config_dir)
        if not data_yaml.exists():
            raise SystemExit(f"missing ablation config: {repo_rel(data_yaml)}")

        weights = train_if_needed(kind, data_yaml, args)
        metrics_path = eval_if_needed(kind, weights, data_yaml, args)

        test_dir = args.project / test_run_name(kind, args)
        compare_clean_path = compare_metrics(
            args.clean_metrics,
            metrics_path,
            test_dir / "compare_vs_clean_test.json",
            args,
        )

        compare_real_path = None
        if recipe_id is None:
            real_only_metrics = metrics_path
        elif real_only_metrics is not None:
            compare_real_path = compare_metrics(
                real_only_metrics,
                metrics_path,
                test_dir / "compare_vs_real_only_test.json",
                args,
            )

        if not args.dry_run:
            summary_rows.append(
                collect_summary_row(
                    kind=kind,
                    recipe_id=recipe_id,
                    metrics_path=metrics_path,
                    compare_clean_path=compare_clean_path,
                    compare_real_path=compare_real_path,
                    args=args,
                )
            )
            per_class_rows.extend(
                collect_per_class_rows(
                    kind=kind,
                    recipe_id=recipe_id,
                    metrics_path=metrics_path,
                    compare_clean_path=compare_clean_path,
                    compare_real_path=compare_real_path,
                    args=args,
                )
            )

    if not args.dry_run:
        label = run_label_tag(args)
        stem = args.summary_stem or f"webgl_ablation_{'nowarmup' if args.warmup_epochs == 0 else 'warmup'}_i{args.imgsz}{label}"
        summary = {
            "settings": {
                "model": repo_rel(args.model),
                "suite": repo_rel(args.suite),
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "workers": args.workers,
                "optimizer": args.optimizer,
                "lr0": args.lr0,
                "lrf": args.lrf,
                "warmup_epochs": args.warmup_epochs,
                "warmup_bias_lr": args.warmup_bias_lr,
                "warmup_momentum": args.warmup_momentum,
                "amp": args.amp,
                "run_label": args.run_label,
            },
            "rows": summary_rows,
        }
        write_json(args.project / f"{stem}_summary.json", summary)
        write_csv(args.project / f"{stem}_summary.csv", summary_rows)
        write_json(
            args.project / f"{stem}_per_class.json",
            {
                "settings": summary["settings"],
                "rows": per_class_rows,
            },
        )
        write_csv(args.project / f"{stem}_per_class.csv", per_class_rows)
        print(f"wrote_summary={repo_rel(args.project / f'{stem}_summary.json')}", flush=True)
        print(f"wrote_per_class={repo_rel(args.project / f'{stem}_per_class.json')}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
