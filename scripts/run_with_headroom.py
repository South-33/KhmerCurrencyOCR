from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

import psutil


@dataclass
class LoadSample:
    cpu_percent: float
    ram_percent: float
    gpu_percent: float | None
    gpu_mem_percent: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command while leaving system headroom for interactive use."
    )
    parser.add_argument(
        "--max-percent",
        type=float,
        default=90.0,
        help="Suspend at or above this active CPU/GPU load.",
    )
    parser.add_argument(
        "--resume-percent",
        type=float,
        default=82.0,
        help="Resume at or below this active CPU/GPU load.",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between load checks.")
    parser.add_argument("--no-priority", action="store_true", help="Do not lower child process priority.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --.")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("Provide a command after --, for example: -- python scripts/train_yolo.py ...")
    if args.resume_percent >= args.max_percent:
        raise SystemExit("--resume-percent must be lower than --max-percent.")
    return args


def read_gpu_load() -> tuple[float | None, float | None]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None

    first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 3:
        return None, None

    try:
        gpu_percent = float(parts[0])
        used_mb = float(parts[1])
        total_mb = float(parts[2])
    except ValueError:
        return None, None

    gpu_mem_percent = (used_mb / total_mb * 100.0) if total_mb else None
    return gpu_percent, gpu_mem_percent


def sample_load() -> LoadSample:
    gpu_percent, gpu_mem_percent = read_gpu_load()
    return LoadSample(
        cpu_percent=psutil.cpu_percent(interval=None),
        ram_percent=psutil.virtual_memory().percent,
        gpu_percent=gpu_percent,
        gpu_mem_percent=gpu_mem_percent,
    )


def throttle_load(sample: LoadSample) -> float:
    values = [sample.cpu_percent]
    if sample.gpu_percent is not None:
        values.append(sample.gpu_percent)
    # Pausing a process does not release RAM or CUDA VRAM, so memory is reported
    # for visibility but not used as a resume gate.
    return max(values)


def format_sample(sample: LoadSample) -> str:
    gpu = "n/a" if sample.gpu_percent is None else f"{sample.gpu_percent:.0f}%"
    gpu_mem = "n/a" if sample.gpu_mem_percent is None else f"{sample.gpu_mem_percent:.0f}%"
    return (
        f"cpu={sample.cpu_percent:.0f}% ram={sample.ram_percent:.0f}% "
        f"gpu={gpu} gpu_mem={gpu_mem}"
    )


def process_tree(root: psutil.Process) -> list[psutil.Process]:
    try:
        children = root.children(recursive=True)
    except psutil.Error:
        children = []
    return children + [root]


def set_low_priority(proc: psutil.Process) -> None:
    try:
        if sys.platform.startswith("win"):
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            proc.nice(10)
    except psutil.Error as exc:
        print(f"[headroom] Could not lower process priority: {exc}", flush=True)


def suspend_tree(proc: psutil.Process) -> None:
    for item in process_tree(proc):
        try:
            item.suspend()
        except psutil.Error:
            pass


def resume_tree(proc: psutil.Process) -> None:
    for item in reversed(process_tree(proc)):
        try:
            item.resume()
        except psutil.Error:
            pass


def main() -> int:
    args = parse_args()
    psutil.cpu_percent(interval=None)

    print(f"[headroom] Starting: {' '.join(args.command)}", flush=True)
    child = subprocess.Popen(args.command)
    child_proc = psutil.Process(child.pid)
    if not args.no_priority:
        set_low_priority(child_proc)

    suspended = False
    try:
        while child.poll() is None:
            time.sleep(args.interval)
            sample = sample_load()
            load = throttle_load(sample)

            if not suspended and load >= args.max_percent:
                print(f"[headroom] Pausing child at {format_sample(sample)}", flush=True)
                suspend_tree(child_proc)
                suspended = True
                continue

            if suspended and load <= args.resume_percent:
                print(f"[headroom] Resuming child at {format_sample(sample)}", flush=True)
                resume_tree(child_proc)
                suspended = False

        return child.returncode or 0
    finally:
        if suspended:
            resume_tree(child_proc)


if __name__ == "__main__":
    raise SystemExit(main())
