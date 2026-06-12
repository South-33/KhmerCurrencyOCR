from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from local_runtime import configure_project_cache
from hardware_profile import (
    HEADROOM_MAX_GPU_MEM_PERCENT,
    HEADROOM_MAX_PERCENT,
    HEADROOM_MAX_RAM_PERCENT,
    HEADROOM_MIN_FREE_RAM_GB,
    HEADROOM_RESUME_PERCENT,
    recommended_device,
    recommended_train_batch,
    recommended_workers,
)


ROOT = Path(__file__).resolve().parents[1]
configure_project_cache()


@dataclass(frozen=True)
class GpuInfo:
    name: str
    util_percent: float
    mem_used_mb: float
    mem_total_mb: float

    @property
    def mem_percent(self) -> float:
        return self.mem_used_mb / self.mem_total_mb * 100.0 if self.mem_total_mb else 0.0

    @property
    def mem_free_mb(self) -> float:
        return max(0.0, self.mem_total_mb - self.mem_used_mb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe this machine and run YOLO training through the headroom governor."
    )
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", default=str(ROOT / "runs" / "cashsnap"))
    parser.add_argument("--batch", default="auto", help="Integer batch size or 'auto'.")
    parser.add_argument("--workers", default="auto", help="Integer worker count or 'auto'.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--box", type=float, default=None)
    parser.add_argument("--cls", type=float, default=None)
    parser.add_argument("--dfl", type=float, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--warmup-bias-lr", type=float, default=None)
    parser.add_argument("--warmup-momentum", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument(
        "--cache",
        default=None,
        choices=["false", "ram", "disk", "true"],
        help="Pass Ultralytics --cache mode to train_yolo.py.",
    )
    parser.add_argument(
        "--compile",
        nargs="?",
        const="default",
        default=None,
        help="Pass optional torch.compile mode to train_yolo.py.",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--freeze", type=int, default=None)
    parser.add_argument("--mosaic", type=float, default=None)
    parser.add_argument("--erasing", type=float, default=None)
    parser.add_argument("--hsv-h", type=float, default=None)
    parser.add_argument("--hsv-s", type=float, default=None)
    parser.add_argument("--hsv-v", type=float, default=None)
    parser.add_argument("--translate", type=float, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--fliplr", type=float, default=None)
    parser.add_argument("--degrees", type=float, default=None)
    parser.add_argument("--shear", type=float, default=None)
    parser.add_argument("--perspective", type=float, default=None)
    parser.add_argument("--close-mosaic", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Pass --no-amp to train_yolo.py.")
    parser.add_argument("--no-val", action="store_true", help="Pass train_yolo.py --no-val to skip epoch/final validation.")
    parser.add_argument("--quiet", action="store_true", help="Pass --quiet to train_yolo.py.")
    parser.add_argument("--plots", action="store_true", help="Allow Ultralytics plot generation.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    parser.add_argument(
        "--adaptive-restarts",
        type=int,
        default=3,
        help="How many times to relaunch with smaller settings after memory pressure.",
    )
    parser.add_argument(
        "--floor-memory-action",
        choices=["pause", "exit"],
        default="pause",
        help=(
            "What to do when memory pressure persists after batch/workers have reached "
            "their minimum safe values."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned command and exit.")
    parser.add_argument("--probe-seconds", type=float, default=4.0)
    parser.add_argument("--max-percent", type=float, default=HEADROOM_MAX_PERCENT)
    parser.add_argument("--resume-percent", type=float, default=HEADROOM_RESUME_PERCENT)
    parser.add_argument("--max-ram-percent", type=float, default=HEADROOM_MAX_RAM_PERCENT)
    parser.add_argument("--max-gpu-mem-percent", type=float, default=HEADROOM_MAX_GPU_MEM_PERCENT)
    parser.add_argument(
        "--min-free-ram-gb",
        type=float,
        default=HEADROOM_MIN_FREE_RAM_GB,
        help="Refuse to launch training below this available system RAM floor. Use 0 to disable.",
    )
    parser.add_argument(
        "--memory-clean-preset",
        choices=["winmemorycleaner", "winmemorycleaner-task", "memreduct"],
        default=None,
        help="Optional run_with_headroom RAM-cleaner preset to trigger under memory pressure.",
    )
    parser.add_argument("--memory-clean-task", default=None)
    parser.add_argument("--memory-clean-task-arg", default=None)
    parser.add_argument("--memory-clean-min-free-ram-gb", type=float, default=None)
    parser.add_argument("--memory-clean-cooldown-seconds", type=float, default=None)
    parser.add_argument("--memory-clean-timeout-seconds", type=float, default=None)
    parser.add_argument("--memory-clean-settle-seconds", type=float, default=None)
    parser.add_argument("--interval", type=float, default=2.0)
    return parser.parse_args()


def read_gpu() -> GpuInfo | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    first = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return None
    try:
        return GpuInfo(parts[0], float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError:
        return None


def probe(seconds: float) -> tuple[float, psutil._common.svmem, GpuInfo | None]:
    psutil.cpu_percent(interval=None)
    end = time.time() + max(0.1, seconds)
    samples: list[float] = []
    while time.time() < end:
        samples.append(psutil.cpu_percent(interval=min(1.0, max(0.1, end - time.time()))))
    cpu = max(samples) if samples else psutil.cpu_percent(interval=None)
    return cpu, psutil.virtual_memory(), read_gpu()


def parse_auto_int(value: str, chosen: int, name: str) -> int:
    if value == "auto":
        return chosen
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SystemExit(f"--{name} must be an integer or 'auto'") from exc
    if parsed < 0:
        raise SystemExit(f"--{name} must be >= 0")
    return parsed


def choose_batch(gpu: GpuInfo | None, ram: psutil._common.svmem, image_size: int) -> int:
    if ram.percent > 85 or ram.available < 2 * 1024**3:
        return 2
    if gpu is None:
        return 2
    return recommended_train_batch(image_size, ram_gb=ram.available / (1024**3))


def choose_workers(cpu_percent: float, ram: psutil._common.svmem) -> int:
    if cpu_percent > 75 or ram.percent > 85 or ram.available < 2 * 1024**3:
        return 0
    return recommended_workers("train", ram.available / (1024**3))


def append_optional(command: list[str], flag: str, value) -> None:
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


def build_command(
    args: argparse.Namespace,
    batch: int,
    workers: int,
    exist_ok: bool,
    memory_action: str,
) -> list[str]:
    train_command = [
        sys.executable,
        str(ROOT / "scripts" / "train_yolo.py"),
        "--model",
        args.model,
        "--data",
        args.data,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(batch),
        "--workers",
        str(workers),
        "--project",
        args.project,
        "--name",
        args.name,
    ]
    append_optional(train_command, "--device", args.device)
    append_optional(train_command, "--optimizer", args.optimizer)
    append_optional(train_command, "--box", args.box)
    append_optional(train_command, "--cls", args.cls)
    append_optional(train_command, "--dfl", args.dfl)
    append_optional(train_command, "--lr0", args.lr0)
    append_optional(train_command, "--lrf", args.lrf)
    append_optional(train_command, "--warmup-epochs", args.warmup_epochs)
    append_optional(train_command, "--warmup-bias-lr", args.warmup_bias_lr)
    append_optional(train_command, "--warmup-momentum", args.warmup_momentum)
    append_optional(train_command, "--seed", args.seed)
    append_optional(train_command, "--fraction", args.fraction)
    append_optional(train_command, "--cache", args.cache)
    append_optional(train_command, "--compile", args.compile)
    append_optional(train_command, "--max-train-batches", args.max_train_batches)
    append_optional(train_command, "--freeze", args.freeze)
    append_optional(train_command, "--mosaic", args.mosaic)
    append_optional(train_command, "--erasing", args.erasing)
    append_optional(train_command, "--hsv-h", args.hsv_h)
    append_optional(train_command, "--hsv-s", args.hsv_s)
    append_optional(train_command, "--hsv-v", args.hsv_v)
    append_optional(train_command, "--translate", args.translate)
    append_optional(train_command, "--scale", args.scale)
    append_optional(train_command, "--fliplr", args.fliplr)
    append_optional(train_command, "--degrees", args.degrees)
    append_optional(train_command, "--shear", args.shear)
    append_optional(train_command, "--perspective", args.perspective)
    append_optional(train_command, "--close-mosaic", args.close_mosaic)
    if args.no_amp:
        train_command.append("--no-amp")
    if args.no_val:
        train_command.append("--no-val")
    if args.quiet:
        train_command.append("--quiet")
    if not args.plots:
        train_command.append("--no-plots")
    if exist_ok:
        train_command.append("--exist-ok")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_with_headroom.py"),
        "--max-percent",
        str(args.max_percent),
        "--resume-percent",
        str(args.resume_percent),
        "--max-ram-percent",
        str(args.max_ram_percent),
        "--max-gpu-mem-percent",
        str(args.max_gpu_mem_percent),
        "--min-free-ram-gb",
        str(args.min_free_ram_gb),
        "--interval",
        str(args.interval),
        "--memory-action",
        memory_action,
    ]
    append_memory_clean_args(command, args)
    command.extend(["--", *train_command])
    return command


def smaller_settings(batch: int, workers: int) -> tuple[int, int] | None:
    if workers > 0:
        return batch, 0
    if batch > 1:
        return max(1, batch // 2), 0
    return None


def main() -> int:
    args = parse_args()
    if args.resume_percent >= args.max_percent:
        raise SystemExit("--resume-percent must be lower than --max-percent.")
    for name in ("max_percent", "max_ram_percent", "max_gpu_mem_percent"):
        if getattr(args, name) > 95.0:
            raise SystemExit(f"--{name.replace('_', '-')} must stay <= 95 so the PC remains usable.")

    cpu, ram, gpu = probe(args.probe_seconds)
    args.device = recommended_device(args.device)
    batch = parse_auto_int(args.batch, choose_batch(gpu, ram, args.imgsz), "batch")
    workers = parse_auto_int(args.workers, choose_workers(cpu, ram), "workers")

    gpu_text = "none"
    if gpu is not None:
        gpu_text = (
            f"{gpu.name}, util={gpu.util_percent:.0f}%, "
            f"vram={gpu.mem_used_mb:.0f}/{gpu.mem_total_mb:.0f}MB ({gpu.mem_percent:.0f}%)"
        )
    print(
        "[bench-headroom] probe "
        f"cpu_peak={cpu:.0f}% ram={ram.percent:.0f}% "
        f"ram_free={ram.available / (1024 ** 3):.1f}GB gpu={gpu_text}",
        flush=True,
    )
    print(f"[bench-headroom] selected batch={batch} workers={workers}", flush=True)
    if args.device is not None:
        print(f"[bench-headroom] selected device={args.device}", flush=True)

    command = build_command(args, batch, workers, args.exist_ok, "exit")
    print("[bench-headroom] command:", " ".join(command), flush=True)
    if args.dry_run:
        return 0
    free_ram_gb = ram.available / (1024**3)
    if args.min_free_ram_gb > 0 and free_ram_gb < args.min_free_ram_gb:
        print(
            "[bench-headroom] refusing to launch: "
            f"ram_free={free_ram_gb:.1f}GB is below --min-free-ram-gb={args.min_free_ram_gb:.1f}GB. "
            "Free RAM or pass --min-free-ram-gb 0 to override.",
            flush=True,
        )
        return 70

    attempts = 0
    exist_ok = args.exist_ok
    memory_action = "exit"
    while True:
        code = subprocess.call(build_command(args, batch, workers, exist_ok, memory_action))
        if code != 70:
            return code
        attempts += 1
        next_settings = smaller_settings(batch, workers)
        can_restart_smaller = attempts <= args.adaptive_restarts and next_settings is not None
        if can_restart_smaller:
            batch, workers = next_settings
            exist_ok = True
            print(
                f"[bench-headroom] memory pressure hit; relaunching with batch={batch} workers={workers}",
                flush=True,
            )
            continue

        if args.floor_memory_action == "pause" and memory_action != "pause":
            memory_action = "pause"
            exist_ok = True
            print(
                "[bench-headroom] memory pressure persisted at minimum restart settings; "
                "switching to pause/resume mode.",
                flush=True,
            )
            continue

        if next_settings is None:
            print("[bench-headroom] memory pressure persisted at minimum settings; leaving training stopped.", flush=True)
            return code
        print(
            "[bench-headroom] memory pressure persisted after adaptive restarts; leaving training stopped.",
            flush=True,
        )
        return code


if __name__ == "__main__":
    raise SystemExit(main())
