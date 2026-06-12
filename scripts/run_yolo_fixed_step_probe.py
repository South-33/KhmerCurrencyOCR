#!/usr/bin/env python
"""Run a fixed-step train/eval/compare probe for two YOLO configs."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from local_runtime import configure_project_cache
from hardware_profile import (
    HEADROOM_MAX_GPU_MEM_PERCENT,
    HEADROOM_MAX_PERCENT,
    HEADROOM_MAX_RAM_PERCENT,
    HEADROOM_MIN_FREE_RAM_GB,
    HEADROOM_RESUME_PERCENT,
    recommended_val_batch,
    recommended_workers,
)


ROOT = Path(__file__).resolve().parents[1]
configure_project_cache()
DEFAULT_MODEL = ROOT / "runs" / "cashsnap" / "yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8" / "weights" / "best.pt"
DEFAULT_PROJECT = ROOT / "runs" / "cashsnap"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ULTRALYTICS_DEFAULT_CLOSE_MOSAIC_EPOCHS = 10
AUGMENT_OVERRIDE_KEYS = (
    "mosaic",
    "erasing",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "scale",
    "fliplr",
    "degrees",
    "shear",
    "perspective",
    "close_mosaic",
)


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
    parser.add_argument(
        "--allow-reuse-without-manifest",
        action="store_true",
        help="Allow legacy fixed-step outputs without a matching manifest. Unsafe for memory-aborted runs.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=recommended_workers("train"))
    parser.add_argument("--eval-batch", type=int, default=None)
    parser.add_argument("--eval-workers", type=int, default=None)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--box", type=float, default=None)
    parser.add_argument("--cls", type=float, default=None)
    parser.add_argument("--dfl", type=float, default=None)
    parser.add_argument("--lr0", type=float, default=0.00005)
    parser.add_argument("--lrf", type=float, default=0.2)
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--warmup-bias-lr", type=float, default=None)
    parser.add_argument("--warmup-momentum", type=float, default=0.937)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--cache",
        default="false",
        choices=["false", "ram", "disk", "true"],
        help="Ultralytics image cache mode for training.",
    )
    parser.add_argument(
        "--compile",
        nargs="?",
        const="default",
        default=None,
        help="Optional Ultralytics torch.compile mode for training.",
    )
    for key in AUGMENT_OVERRIDE_KEYS:
        parser.add_argument(
            f"--candidate-{key.replace('_', '-')}",
            type=int if key == "close_mosaic" else float,
            default=None,
            help=f"Candidate-only {key.replace('_', '-')} training augmentation override.",
        )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="Enable AMP. Default keeps AMP disabled.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-free-ram-gb", type=float, default=HEADROOM_MIN_FREE_RAM_GB)
    parser.add_argument("--max-cpu-percent", type=float, default=HEADROOM_MAX_PERCENT)
    parser.add_argument("--resume-cpu-percent", type=float, default=HEADROOM_RESUME_PERCENT)
    parser.add_argument("--max-ram-percent", type=float, default=HEADROOM_MAX_RAM_PERCENT)
    parser.add_argument("--max-gpu-mem-percent", type=float, default=HEADROOM_MAX_GPU_MEM_PERCENT)
    parser.add_argument(
        "--memory-clean-preset",
        choices=["winmemorycleaner", "winmemorycleaner-task", "memreduct"],
        default=None,
        help="Optional run_with_headroom RAM-cleaner preset for train/eval subprocesses.",
    )
    parser.add_argument("--memory-clean-task", default=None)
    parser.add_argument("--memory-clean-task-arg", default=None)
    parser.add_argument("--memory-clean-min-free-ram-gb", type=float, default=None)
    parser.add_argument("--memory-clean-cooldown-seconds", type=float, default=None)
    parser.add_argument("--memory-clean-timeout-seconds", type=float, default=None)
    parser.add_argument("--memory-clean-settle-seconds", type=float, default=None)
    parser.add_argument(
        "--train-adaptive-restarts",
        type=int,
        default=0,
        help="Relaunch training with smaller batch/workers after RAM pressure. Keep 0 for strict fixed-setting A/Bs.",
    )
    parser.add_argument("--max-per-class-drop", type=float, default=0.05)
    parser.add_argument("--fail-on-row-count-mismatch", action="store_true")
    parser.add_argument("--fail-on-step-reference-mismatch", action="store_true")
    parser.add_argument(
        "--fail-on-train-phase-mismatch",
        action="store_true",
        help=(
            "Fail preflight if fixed steps stop baseline/candidate at different "
            "scheduler or close-mosaic phases. Useful when comparing unequal train row counts."
        ),
    )
    parser.add_argument(
        "--fail-on-inexact-real-class-mix-control",
        action="store_true",
        help="Fail preflight if either config declares a real_class_mix row-count control that is not exact.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def append_optional(command: list[str], flag: str, value: Any | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def append_memory_clean_args(command: list[str], args: argparse.Namespace) -> None:
    append_optional(command, "--memory-clean-preset", args.memory_clean_preset)
    append_optional(command, "--memory-clean-task", args.memory_clean_task)
    append_optional(command, "--memory-clean-task-arg", args.memory_clean_task_arg)
    append_optional(command, "--memory-clean-min-free-ram-gb", args.memory_clean_min_free_ram_gb)
    append_optional(command, "--memory-clean-cooldown-seconds", args.memory_clean_cooldown_seconds)
    append_optional(command, "--memory-clean-timeout-seconds", args.memory_clean_timeout_seconds)
    append_optional(command, "--memory-clean-settle-seconds", args.memory_clean_settle_seconds)


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def lr_tag(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"lr{mantissa}e{abs(int(exponent))}"


def model_init_tag(model: Path) -> str:
    try:
        if model.resolve() == DEFAULT_MODEL.resolve():
            return "clean"
    except OSError:
        if model == DEFAULT_MODEL:
            return "clean"

    if model.name.lower() == "yolo26n.pt":
        return "yolo26n"
    if model.name.lower() == "best.pt" and model.parent.name.lower() == "weights":
        tag = slug(model.parent.parent.name)
    else:
        tag = slug(model.stem or model.name)
    return tag[:60].strip("_") or "model"


def read_yaml(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return config


def row_count_control(path: Path) -> dict[str, Any]:
    control = read_yaml(path).get("cashsnap_row_count_control", {})
    return control if isinstance(control, dict) else {}


def inexact_real_class_mix_control_reason(path: Path) -> str | None:
    control = row_count_control(path)
    source = str(control.get("source", ""))
    if not source.startswith("real_class_mix"):
        return None
    if control.get("class_mix_all_exact") is True:
        return None
    report = control.get("class_mix_match_report", [])
    total = len(report) if isinstance(report, list) else control.get("matched_dose_images", "?")
    matches = control.get("class_mix_exact_matches", "?")
    return f"{repo_rel(path)} real_class_mix control is inexact ({matches}/{total} exact)"


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


def effective_close_mosaic_epochs(train_overrides: dict[str, float | int] | None = None) -> tuple[int, str]:
    overrides = train_overrides or {}
    if "close_mosaic" in overrides:
        return int(overrides["close_mosaic"]), "candidate_override"
    return ULTRALYTICS_DEFAULT_CLOSE_MOSAIC_EPOCHS, "ultralytics_default_assumed"


def train_phase_summary(
    row_summary: dict[str, Any],
    args: argparse.Namespace,
    steps: int,
    train_overrides: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    rows = int(row_summary["rows"])
    batches_per_epoch = math.ceil(rows / args.batch)
    total_train_batches_available = batches_per_epoch * args.epochs
    steps_at_stop = min(steps, total_train_batches_available)
    close_mosaic_epochs, close_mosaic_source = effective_close_mosaic_epochs(train_overrides)
    close_mosaic_start_epoch = max(0, args.epochs - close_mosaic_epochs) if close_mosaic_epochs > 0 else args.epochs
    close_mosaic_start_step = close_mosaic_start_epoch * batches_per_epoch
    close_mosaic_steps_at_stop = max(0, steps_at_stop - close_mosaic_start_step)
    return {
        "rows": rows,
        "batches_per_epoch": batches_per_epoch,
        "epochs_requested": args.epochs,
        "total_train_batches_available": total_train_batches_available,
        "max_train_batches_requested": steps,
        "max_train_batches_reachable": steps <= total_train_batches_available,
        "steps_at_stop": steps_at_stop,
        "stop_epoch_one_based": math.ceil(steps_at_stop / batches_per_epoch) if steps_at_stop else 0,
        "epoch_progress_at_stop": steps_at_stop / batches_per_epoch if batches_per_epoch else 0.0,
        "lr_schedule_progress_at_stop": (
            steps_at_stop / total_train_batches_available if total_train_batches_available else 0.0
        ),
        "close_mosaic_epochs_effective": close_mosaic_epochs,
        "close_mosaic_source": close_mosaic_source,
        "close_mosaic_start_epoch": close_mosaic_start_epoch,
        "close_mosaic_start_step": close_mosaic_start_step,
        "close_mosaic_steps_at_stop": close_mosaic_steps_at_stop,
        "close_mosaic_fraction_of_stop_steps": (
            close_mosaic_steps_at_stop / steps_at_stop if steps_at_stop else 0.0
        ),
    }


def train_phase_mismatch_warnings(
    baseline_phase: dict[str, Any],
    candidate_phase: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    if not baseline_phase["max_train_batches_reachable"]:
        warnings.append("baseline cannot reach requested max_train_batches within requested epochs")
    if not candidate_phase["max_train_batches_reachable"]:
        warnings.append("candidate cannot reach requested max_train_batches within requested epochs")
    if baseline_phase["rows"] != candidate_phase["rows"]:
        warnings.append(
            "baseline/candidate row counts differ; fixed steps stop at different epoch/scheduler phases"
        )
    lr_delta = abs(
        float(candidate_phase["lr_schedule_progress_at_stop"])
        - float(baseline_phase["lr_schedule_progress_at_stop"])
    )
    if lr_delta > 0.05:
        warnings.append(
            "lr schedule progress differs at stop "
            f"(baseline={baseline_phase['lr_schedule_progress_at_stop']:.3f}, "
            f"candidate={candidate_phase['lr_schedule_progress_at_stop']:.3f})"
        )
    if baseline_phase["close_mosaic_steps_at_stop"] != candidate_phase["close_mosaic_steps_at_stop"]:
        warnings.append(
            "close-mosaic exposure differs at stop "
            f"(baseline_steps={baseline_phase['close_mosaic_steps_at_stop']}, "
            f"candidate_steps={candidate_phase['close_mosaic_steps_at_stop']})"
        )
    return warnings


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256_if_exists(path: Path) -> str | None:
    return file_sha256(path) if path.exists() else None


@contextlib.contextmanager
def output_lock(project: Path, name: str, dry_run: bool) -> Iterator[None]:
    if dry_run:
        yield
        return

    lock_dir = project / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{slug(name)}.lock"
    payload = {
        "name": name,
        "pid": os.getpid(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        try:
            holder = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            holder = ""
        detail = f" holder={holder}" if holder else ""
        raise RuntimeError(f"output lock already exists: {repo_rel(lock_path)}.{detail}") from exc

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def candidate_train_overrides(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        key: getattr(args, f"candidate_{key}")
        for key in AUGMENT_OVERRIDE_KEYS
        if getattr(args, f"candidate_{key}") is not None
    }


def augment_tag(overrides: dict[str, float | int]) -> str:
    if not overrides:
        return ""
    parts = []
    for key, value in sorted(overrides.items()):
        text = f"{value:g}" if isinstance(value, float) else str(value)
        parts.append(f"{key.replace('_', '')}{text}".replace(".", "p").replace("-", "m"))
    return "_aug" + "_".join(parts)


def loss_tag(args: argparse.Namespace) -> str:
    parts = []
    for key in ("box", "cls", "dfl"):
        value = getattr(args, key)
        if value is None:
            continue
        parts.append(f"{key}{value:g}".replace(".", "p").replace("-", "m"))
    return "_loss" + "_".join(parts) if parts else ""


def train_name(
    label: str,
    args: argparse.Namespace,
    steps: int,
    train_overrides: dict[str, float | int] | None = None,
) -> str:
    amp_tag = "amp" if args.amp else "noamp"
    warmup_tag = "nowarmup" if float(args.warmup_epochs) == 0.0 else f"warmup{args.warmup_epochs:g}"
    cache_tag = f"cache{args.cache}"
    compile_tag = f"_compile{slug(args.compile)}" if args.compile is not None else ""
    freeze_tag = f"_freeze{args.freeze}" if args.freeze is not None else ""
    seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
    override_tag = augment_tag(train_overrides or {})
    loss = loss_tag(args)
    init_tag = model_init_tag(args.model)
    return (
        f"fixed_step_{label}_from_{init_tag}_e{args.epochs}_i{args.imgsz}_"
        f"b{args.batch}_w{args.workers}_{args.optimizer.lower()}_{lr_tag(args.lr0)}_"
        f"{warmup_tag}_{amp_tag}_{cache_tag}{compile_tag}{freeze_tag}{loss}_steps{steps}{seed_tag}{override_tag}"
    )


def test_name(
    label: str,
    args: argparse.Namespace,
    steps: int,
    train_overrides: dict[str, float | int] | None = None,
) -> str:
    seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
    freeze_tag = f"_freeze{args.freeze}" if args.freeze is not None else ""
    override_tag = augment_tag(train_overrides or {})
    loss = loss_tag(args)
    return (
        f"fixed_step_{label}_test_i{args.imgsz}_"
        f"tb{args.eval_batch}_tw{args.eval_workers}{freeze_tag}{loss}_steps{steps}{seed_tag}{override_tag}"
    )


def run_settings(
    args: argparse.Namespace,
    steps: int,
    train_overrides: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "optimizer": args.optimizer,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "freeze": args.freeze,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "warmup_epochs": args.warmup_epochs,
        "warmup_bias_lr": args.warmup_bias_lr if args.warmup_bias_lr is not None else args.lr0,
        "warmup_momentum": args.warmup_momentum,
        "seed": args.seed,
        "cache": args.cache,
        "compile": args.compile,
        "max_train_batches_total": steps,
        "amp": args.amp,
    }
    if train_overrides:
        settings["train_overrides"] = train_overrides
    return settings


def train_manifest(
    label: str,
    data_yaml: Path,
    args: argparse.Namespace,
    steps: int,
    train_overrides: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "train",
        "label": label,
        "run_name": train_name(label, args, steps, train_overrides),
        "data": repo_rel(data_yaml),
        "train_rows": train_row_summary(data_yaml),
        "source_model": repo_rel(args.model),
        "source_model_sha256": file_sha256(args.model),
        "settings": run_settings(args, steps, train_overrides),
    }


def eval_manifest(
    label: str,
    weights: Path,
    data_yaml: Path,
    args: argparse.Namespace,
    steps: int,
    train_overrides: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "imgsz": args.imgsz,
        "batch": args.eval_batch,
        "workers": args.eval_workers,
        "device": str(args.device),
        "seed": args.seed,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "freeze": args.freeze,
        "max_train_batches_total": steps,
    }
    if train_overrides:
        settings["train_overrides"] = train_overrides
    return {
        "kind": "eval",
        "label": label,
        "run_name": test_name(label, args, steps, train_overrides),
        "data": repo_rel(data_yaml),
        "weights": repo_rel(weights),
        "weights_sha256": file_sha256_if_exists(weights),
        "settings": settings,
    }


def manifest_path(run_dir: Path, kind: str) -> Path:
    return run_dir / f"fixed_step_probe_{kind}.json"


def check_reuse_manifest(path: Path, expected: dict[str, Any], *, allow_missing: bool) -> None:
    if not path.exists():
        if allow_missing:
            print(f"reuse_without_manifest={repo_rel(path)}", flush=True)
            return
        raise RuntimeError(
            "refusing to reuse fixed-step output without manifest: "
            f"{repo_rel(path)}. Use --allow-reuse-without-manifest only for known-good legacy runs."
        )
        return
    actual = read_json(path)
    mismatches = []
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            mismatches.append(key)
    if mismatches:
        raise RuntimeError(f"reuse manifest mismatch for {repo_rel(path)}: {', '.join(mismatches)}")


def train_if_needed(
    label: str,
    data_yaml: Path,
    args: argparse.Namespace,
    steps: int,
    train_overrides: dict[str, float | int] | None = None,
) -> Path:
    overrides = train_overrides or {}
    run_name = train_name(label, args, steps, overrides)
    run_dir = args.project / run_name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    manifest = train_manifest(label, data_yaml, args, steps, overrides)
    manifest_out = manifest_path(run_dir, "train")
    existing_weights = best if best.exists() else last
    if args.reuse_existing and existing_weights.exists():
        check_reuse_manifest(manifest_out, manifest, allow_missing=args.allow_reuse_without_manifest)
        print(f"reuse_train={repo_rel(existing_weights)}", flush=True)
        return existing_weights

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
        "--cache",
        args.cache,
        "--max-train-batches",
        str(steps),
        "--quiet",
        "--no-val",
        "--exist-ok",
        "--min-free-ram-gb",
        str(args.min_free_ram_gb),
        "--max-ram-percent",
        str(args.max_ram_percent),
        "--floor-memory-action",
        "exit",
        "--adaptive-restarts",
        str(args.train_adaptive_restarts),
    ]
    append_memory_clean_args(command, args)
    for key in ("box", "cls", "dfl"):
        value = getattr(args, key)
        if value is not None:
            command.extend([f"--{key}", str(value)])
    if args.freeze is not None:
        command.extend(["--freeze", str(args.freeze)])
    if not args.amp:
        command.append("--no-amp")
    if args.compile is not None:
        command.extend(["--compile", args.compile])
    for key, value in sorted(overrides.items()):
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    with output_lock(args.project, run_name, args.dry_run):
        run(command, args.dry_run)
        if not args.dry_run:
            trained_weights = best if best.exists() else last
            if not trained_weights.exists():
                raise FileNotFoundError(
                    "expected trained weights were not written: "
                    f"{best} or {last}"
                )
            write_json(manifest_out, manifest)
    return best if best.exists() else last


def eval_if_needed(
    label: str,
    weights: Path,
    data_yaml: Path,
    args: argparse.Namespace,
    steps: int,
    train_overrides: dict[str, float | int] | None = None,
) -> Path:
    overrides = train_overrides or {}
    run_name = test_name(label, args, steps, overrides)
    run_dir = args.project / run_name
    metrics = run_dir / "metrics.json"
    manifest = eval_manifest(label, weights, data_yaml, args, steps, overrides)
    manifest_out = manifest_path(run_dir, "eval")
    if args.reuse_existing and metrics.exists():
        check_reuse_manifest(manifest_out, manifest, allow_missing=args.allow_reuse_without_manifest)
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
    ]
    append_memory_clean_args(command, args)
    command.extend([
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
        str(args.eval_batch),
        "--workers",
        str(args.eval_workers),
        "--device",
        str(args.device),
        "--name",
        run_name,
        "--no-plots",
        "--quiet",
        "--exist-ok",
        "--metrics-json",
        repo_rel(metrics),
    ])
    with output_lock(args.project, run_name, args.dry_run):
        run(command, args.dry_run)
        if not args.dry_run:
            if not metrics.exists():
                raise FileNotFoundError(f"expected eval metrics were not written: {metrics}")
            manifest["weights_sha256"] = file_sha256(weights)
            write_json(manifest_out, manifest)
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
    args.eval_batch = args.eval_batch or recommended_val_batch(args.imgsz)
    args.eval_workers = args.eval_workers if args.eval_workers is not None else recommended_workers("val")
    if args.eval_batch < 1:
        raise SystemExit("--eval-batch must be positive")
    if args.eval_workers < 0:
        raise SystemExit("--eval-workers must be >= 0")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise SystemExit("--max-train-batches must be positive")

    baseline_label = slug(args.baseline_label or default_label(args.baseline_data))
    candidate_label = slug(args.candidate_label or default_label(args.candidate_data))
    baseline_rows = train_row_summary(args.baseline_data)
    candidate_rows = train_row_summary(args.candidate_data)
    reference_rows_summary = train_row_summary(args.step_reference_data)
    reference_rows = int(reference_rows_summary["rows"])
    reference_batches_per_epoch = math.ceil(reference_rows / args.batch)
    steps = args.max_train_batches or (reference_batches_per_epoch * args.epochs)
    args.project.mkdir(parents=True, exist_ok=True)
    candidate_overrides = candidate_train_overrides(args)
    baseline_phase = train_phase_summary(baseline_rows, args, steps)
    candidate_phase = train_phase_summary(candidate_rows, args, steps, candidate_overrides)
    step_reference_phase = train_phase_summary(reference_rows_summary, args, steps)
    train_phase_warnings = train_phase_mismatch_warnings(baseline_phase, candidate_phase)
    real_class_mix_control_failures = [
        {"role": role, "reason": reason}
        for role, path in (("baseline", args.baseline_data), ("candidate", args.candidate_data))
        for reason in [inexact_real_class_mix_control_reason(path)]
        if reason is not None
    ]

    preflight = {
        "baseline_data": repo_rel(args.baseline_data),
        "candidate_data": repo_rel(args.candidate_data),
        "step_reference_data": repo_rel(args.step_reference_data),
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "seed": args.seed,
        "batch": args.batch,
        "cache": args.cache,
        "train_adaptive_restarts": args.train_adaptive_restarts,
        "compile": args.compile,
        "eval_batch": args.eval_batch,
        "eval_workers": args.eval_workers,
        "reference_train_rows": reference_rows,
        "reference_batches_per_epoch": reference_batches_per_epoch,
        "max_train_batches_total": steps,
        "candidate_train_overrides": candidate_overrides,
        "max_train_batches_source": "explicit" if args.max_train_batches is not None else "epochs_x_reference_batches",
        "train_phase_summaries": {
            "baseline": baseline_phase,
            "candidate": candidate_phase,
            "step_reference": step_reference_phase,
        },
        "train_phase_warnings": train_phase_warnings,
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
            "fail_on_train_phase_mismatch": args.fail_on_train_phase_mismatch,
            "fail_on_inexact_real_class_mix_control": args.fail_on_inexact_real_class_mix_control,
        },
        "real_class_mix_control_failures": real_class_mix_control_failures,
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
    if args.fail_on_train_phase_mismatch and train_phase_warnings:
        raise SystemExit("train phase mismatch warning(s): " + "; ".join(train_phase_warnings))
    if args.fail_on_inexact_real_class_mix_control and real_class_mix_control_failures:
        reasons = "; ".join(f"{row['role']}: {row['reason']}" for row in real_class_mix_control_failures)
        raise SystemExit(f"inexact real_class_mix row-count control(s): {reasons}")
    if args.preflight_only:
        print(
            "preflight_ok "
            f"baseline_rows={baseline_rows['rows']} "
            f"candidate_rows={candidate_rows['rows']} "
            f"reference_rows={reference_rows} "
            f"reference_batches_per_epoch={reference_batches_per_epoch} "
            f"max_train_batches_total={steps} "
            f"train_phase_warnings={len(train_phase_warnings)}",
            flush=True,
        )
        return 0

    baseline_weights = train_if_needed(baseline_label, args.baseline_data, args, steps)
    candidate_weights = train_if_needed(candidate_label, args.candidate_data, args, steps, candidate_overrides)
    baseline_metrics = eval_if_needed(baseline_label, baseline_weights, args.baseline_data, args, steps)
    candidate_metrics = eval_if_needed(
        candidate_label,
        candidate_weights,
        args.candidate_data,
        args,
        steps,
        candidate_overrides,
    )

    compare_seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
    compare_override_tag = augment_tag(candidate_overrides)
    compare_dir = args.project / f"fixed_step_{baseline_label}_vs_{candidate_label}_steps{steps}{compare_seed_tag}{compare_override_tag}"
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
                "eval_batch": args.eval_batch,
                "eval_workers": args.eval_workers,
                "reference_batches_per_epoch": reference_batches_per_epoch,
                "max_train_batches_total": steps,
                "candidate_train_overrides": candidate_overrides,
                "train_phase_summaries": preflight["train_phase_summaries"],
                "train_phase_warnings": train_phase_warnings,
                "preflight": preflight,
                "baseline_train_run": train_name(baseline_label, args, steps),
                "candidate_train_run": train_name(candidate_label, args, steps, candidate_overrides),
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
