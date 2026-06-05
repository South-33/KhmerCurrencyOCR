from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--device", default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--lrf", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--warmup-bias-lr", type=float, default=None)
    parser.add_argument("--warmup-momentum", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Pass --no-amp to train_yolo.py.")
    parser.add_argument("--no-val", action="store_true", help="Pass --no-val to train_yolo.py.")
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
    parser.add_argument("--max-percent", type=float, default=90.0)
    parser.add_argument("--resume-percent", type=float, default=82.0)
    parser.add_argument("--max-ram-percent", type=float, default=90.0)
    parser.add_argument("--max-gpu-mem-percent", type=float, default=90.0)
    parser.add_argument(
        "--min-free-ram-gb",
        type=float,
        default=4.0,
        help="Refuse to launch training below this available system RAM floor. Use 0 to disable.",
    )
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
    if ram.percent > 65 or ram.available < 6 * 1024**3:
        return 2
    if gpu is None:
        return 2
    free = gpu.mem_free_mb
    if image_size >= 960:
        return 2
    if image_size >= 640:
        return 2 if free < 5000 else 4
    if free < 3500:
        return 2
    if free < 7000:
        return 4
    return 6 if ram.percent > 75 else 8


def choose_workers(cpu_percent: float, ram: psutil._common.svmem) -> int:
    cores = psutil.cpu_count(logical=True) or 1
    if cpu_percent > 50 or ram.percent > 65 or ram.available < 6 * 1024**3 or cores <= 8:
        return 0
    return 1


def append_optional(command: list[str], flag: str, value) -> None:
    if value is not None:
        command.extend([flag, str(value)])


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
    append_optional(train_command, "--lr0", args.lr0)
    append_optional(train_command, "--lrf", args.lrf)
    append_optional(train_command, "--warmup-epochs", args.warmup_epochs)
    append_optional(train_command, "--warmup-bias-lr", args.warmup_bias_lr)
    append_optional(train_command, "--warmup-momentum", args.warmup_momentum)
    append_optional(train_command, "--seed", args.seed)
    append_optional(train_command, "--fraction", args.fraction)
    append_optional(train_command, "--max-train-batches", args.max_train_batches)
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

    return [
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
        "--",
        *train_command,
    ]


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
    if args.device is None and gpu is not None:
        args.device = "0"
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
