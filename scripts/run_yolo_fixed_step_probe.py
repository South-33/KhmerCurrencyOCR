#!/usr/bin/env python
"""Run a fixed-step train/eval/compare probe for two YOLO configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "runs" / "cashsnap" / "yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8" / "weights" / "best.pt"
DEFAULT_PROJECT = ROOT / "runs" / "cashsnap"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-data", required=True, type=Path)
    parser.add_argument("--candidate-data", required=True, type=Path)
    parser.add_argument(
        "--step-reference-data",
        type=Path,
        default=None,
        help="Config whose train-list length defines --max-train-batches. Defaults to --baseline-data.",
    )
    parser.add_argument("--baseline-label", default=None)
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--preflight-json", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true", help="Write/print the fixed-step probe preflight without training.")
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="Enable AMP. Default keeps AMP disabled.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-free-ram-gb", type=float, default=1.2)
    parser.add_argument("--max-cpu-percent", type=float, default=90.0)
    parser.add_argument("--resume-cpu-percent", type=float, default=82.0)
    parser.add_argument("--max-ram-percent", type=float, default=94.0)
    parser.add_argument("--max-gpu-mem-percent", type=float, default=90.0)
    parser.add_argument("--max-per-class-drop", type=float, default=0.05)
    parser.add_argument("--fail-on-row-count-mismatch", action="store_true")
    parser.add_argument("--fail-on-step-reference-mismatch", action="store_true")
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


def lr_tag(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"lr{mantissa}e{abs(int(exponent))}"


def read_yaml(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return config


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    root = Path(str(config.get("path", "."))).expanduser()
    return root if root.is_absolute() else (config_path.parent / root).resolve()


def split_path(config_path: Path, config: dict[str, Any], value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return data_root(config_path, config) / path


def split_image_rows(config_path: Path, split_value: str | list[str]) -> list[str]:
    config = read_yaml(config_path)
    values = split_value if isinstance(split_value, list) else [split_value]
    rows: list[str] = []
    for value in values:
        path = split_path(config_path, config, str(value))
        if path.suffix.lower() == ".txt":
            rows.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        elif path.is_dir():
            rows.extend(str(item) for item in sorted(path.iterdir()) if item.is_file() and item.suffix.lower() in IMAGE_EXTS)
        else:
            raise FileNotFoundError(f"cannot count train split {value!r} resolved to {path}")
    return rows


def train_row_count(config_path: Path) -> int:
    return len(train_rows(config_path))


def train_rows(config_path: Path) -> list[str]:
    config = read_yaml(config_path)
    train = config.get("train")
    if not isinstance(train, (str, list)):
        raise ValueError(f"config train split must be a string or list: {config_path}")
    return split_image_rows(config_path, train)


def row_fingerprint(rows: list[str]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.replace("\\", "/").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def train_row_summary(config_path: Path) -> dict[str, Any]:
    rows = train_rows(config_path)
    unique_rows = set(rows)
    return {
        "data": repo_rel(config_path),
        "rows": len(rows),
        "unique_rows": len(unique_rows),
        "duplicate_rows": len(rows) - len(unique_rows),
        "fingerprint": row_fingerprint(rows),
        "first_rows": rows[:5],
        "last_rows": rows[-5:] if rows else [],
    }


def default_label(data_path: Path) -> str:
    name = data_path.stem
    for prefix in ("cashsnap_v1_plus_webgl_accepted_", "cashsnap_v1_plus_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    for suffix in ("_probe", "_seed"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return slug(name)


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def train_name(label: str, args: argparse.Namespace, steps: int) -> str:
    amp_tag = "amp" if args.amp else "noamp"
    warmup_tag = "nowarmup" if float(args.warmup_epochs) == 0.0 else f"warmup{args.warmup_epochs:g}"
    seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
    return (
        f"fixed_step_{label}_from_clean_e{args.epochs}_i{args.imgsz}_"
        f"{args.optimizer.lower()}_{lr_tag(args.lr0)}_{warmup_tag}_{amp_tag}"
        f"_b{steps}{seed_tag}"
    )


def test_name(label: str, args: argparse.Namespace, steps: int) -> str:
    seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
    return f"fixed_step_{label}_test_i{args.imgsz}_b{steps}{seed_tag}"


def train_if_needed(label: str, data_yaml: Path, args: argparse.Namespace, steps: int) -> Path:
    run_name = train_name(label, args, steps)
    best = args.project / run_name / "weights" / "best.pt"
    if args.reuse_existing and best.exists():
        print(f"reuse_train={repo_rel(best)}", flush=True)
        return best

    command = [
        sys.executable,
        "scripts/bench_train_with_headroom.py",
        "--model",
        repo_rel(args.model),
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
        "--seed",
        str(args.seed),
        "--max-train-batches",
        str(steps),
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


def eval_if_needed(label: str, weights: Path, data_yaml: Path, args: argparse.Namespace, steps: int) -> Path:
    run_name = test_name(label, args, steps)
    metrics = args.project / run_name / "metrics.json"
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
        run_name,
        "--no-plots",
        "--quiet",
        "--exist-ok",
        "--metrics-json",
        repo_rel(metrics),
    ]
    run(command, args.dry_run)
    return metrics


def compare_metrics(baseline: Path, candidate: Path, out_path: Path, args: argparse.Namespace) -> Path:
    command = [
        sys.executable,
        "scripts/compare_yolo_metrics.py",
        "--baseline",
        repo_rel(baseline),
        "--candidate",
        repo_rel(candidate),
        "--max-drop",
        "0.0",
        "--max-per-class-drop",
        str(args.max_per_class_drop),
        "--json-out",
        repo_rel(out_path),
        "--no-fail",
    ]
    run(command, args.dry_run)
    return out_path


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected JSON object: {path}")
    return document


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.baseline_data = resolve(args.baseline_data)
    args.candidate_data = resolve(args.candidate_data)
    args.step_reference_data = resolve(args.step_reference_data or args.baseline_data)
    args.model = resolve(args.model)
    args.project = resolve(args.project)
    if args.seed is None:
        raise SystemExit("--seed is required so fixed-step probes are reproducible")
    if args.batch < 1:
        raise SystemExit("--batch must be positive")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise SystemExit("--max-train-batches must be positive")

    baseline_label = slug(args.baseline_label or default_label(args.baseline_data))
    candidate_label = slug(args.candidate_label or default_label(args.candidate_data))
    baseline_rows = train_row_summary(args.baseline_data)
    candidate_rows = train_row_summary(args.candidate_data)
    reference_rows_summary = train_row_summary(args.step_reference_data)
    reference_rows = int(reference_rows_summary["rows"])
    steps = args.max_train_batches or math.ceil(reference_rows / args.batch)
    args.project.mkdir(parents=True, exist_ok=True)

    preflight = {
        "baseline_data": repo_rel(args.baseline_data),
        "candidate_data": repo_rel(args.candidate_data),
        "step_reference_data": repo_rel(args.step_reference_data),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "seed": args.seed,
        "batch": args.batch,
        "reference_train_rows": reference_rows,
        "max_train_batches": steps,
        "row_summaries": {
            "baseline": baseline_rows,
            "candidate": candidate_rows,
            "step_reference": reference_rows_summary,
        },
        "row_count_delta_candidate_minus_baseline": int(candidate_rows["rows"]) - int(baseline_rows["rows"]),
        "row_count_delta_reference_minus_baseline": reference_rows - int(baseline_rows["rows"]),
        "guards": {
            "fail_on_row_count_mismatch": args.fail_on_row_count_mismatch,
            "fail_on_step_reference_mismatch": args.fail_on_step_reference_mismatch,
        },
    }
    if args.preflight_json is not None:
        args.preflight_json = resolve(args.preflight_json)
        write_json(args.preflight_json, preflight)
        print(f"wrote_preflight={repo_rel(args.preflight_json)}", flush=True)
    if args.fail_on_row_count_mismatch and baseline_rows["rows"] != candidate_rows["rows"]:
        raise SystemExit(
            "baseline/candidate train row counts differ "
            f"({baseline_rows['rows']} vs {candidate_rows['rows']}); rerun without --fail-on-row-count-mismatch for deliberate unequal-dose probes"
        )
    if args.fail_on_step_reference_mismatch and reference_rows != baseline_rows["rows"]:
        raise SystemExit(
            "step-reference train row count differs from baseline "
            f"({reference_rows} vs {baseline_rows['rows']}); rerun without --fail-on-step-reference-mismatch if deliberate"
        )
    if args.preflight_only:
        print(
            "preflight_ok "
            f"baseline_rows={baseline_rows['rows']} "
            f"candidate_rows={candidate_rows['rows']} "
            f"reference_rows={reference_rows} "
            f"max_train_batches={steps}",
            flush=True,
        )
        return 0

    baseline_weights = train_if_needed(baseline_label, args.baseline_data, args, steps)
    candidate_weights = train_if_needed(candidate_label, args.candidate_data, args, steps)
    baseline_metrics = eval_if_needed(baseline_label, baseline_weights, args.baseline_data, args, steps)
    candidate_metrics = eval_if_needed(candidate_label, candidate_weights, args.candidate_data, args, steps)

    compare_dir = args.project / f"fixed_step_{baseline_label}_vs_{candidate_label}_b{steps}"
    compare_path = compare_metrics(baseline_metrics, candidate_metrics, compare_dir / "summary.json", args)
    if args.summary_json is None:
        args.summary_json = compare_dir / "probe.json"
    else:
        args.summary_json = resolve(args.summary_json)

    if not args.dry_run:
        comparison = read_json(compare_path)
        write_json(
            args.summary_json,
            {
                "baseline_data": repo_rel(args.baseline_data),
                "candidate_data": repo_rel(args.candidate_data),
                "step_reference_data": repo_rel(args.step_reference_data),
                "reference_train_rows": reference_rows,
                "batch": args.batch,
                "max_train_batches": steps,
                "preflight": preflight,
                "baseline_train_run": train_name(baseline_label, args, steps),
                "candidate_train_run": train_name(candidate_label, args, steps),
                "baseline_metrics": repo_rel(baseline_metrics),
                "candidate_metrics": repo_rel(candidate_metrics),
                "comparison": repo_rel(compare_path),
                "passed": comparison.get("passed"),
                "delta": comparison.get("delta"),
            },
        )
        print(f"wrote_probe={repo_rel(args.summary_json)}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
