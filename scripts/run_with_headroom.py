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
    ram_available_gb: float
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
    parser.add_argument(
        "--max-ram-percent",
        type=float,
        default=90.0,
        help="Terminate the child if system RAM reaches this percent.",
    )
    parser.add_argument(
        "--max-gpu-mem-percent",
        type=float,
        default=90.0,
        help="Terminate the child if GPU memory reaches this percent.",
    )
    parser.add_argument(
        "--min-free-ram-gb",
        type=float,
        default=2.0,
        help=(
            "Wait before launch below this available-GB floor and warn during the run. "
            "Use 0 to disable. Runtime pausing still follows --max-ram-percent."
        ),
    )
    parser.add_argument(
        "--memory-action",
        choices=["pause", "exit"],
        default="pause",
        help="Pause on memory pressure, or exit with code 70 so a wrapper can relaunch smaller.",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between load checks.")
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for initial headroom before launching the child. Use 0 to disable.",
    )
    parser.add_argument("--no-priority", action="store_true", help="Do not lower child process priority.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --.")
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise SystemExit("Provide a command after --, for example: -- python scripts/train_yolo.py ...")
    if args.resume_percent >= args.max_percent:
        raise SystemExit("--resume-percent must be lower than --max-percent.")
    for name in ("max_percent", "max_ram_percent", "max_gpu_mem_percent"):
        if getattr(args, name) > 95.0:
            raise SystemExit(f"--{name.replace('_', '-')} must stay <= 95 so the PC remains usable.")
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
    ram = psutil.virtual_memory()
    return LoadSample(
        cpu_percent=psutil.cpu_percent(interval=None),
        ram_percent=ram.percent,
        ram_available_gb=ram.available / (1024**3),
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
        f"ram_free={sample.ram_available_gb:.1f}GB "
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


def terminate_tree(proc: psutil.Process) -> None:
    tree = process_tree(proc)
    for item in tree:
        try:
            item.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(tree, timeout=8)
    for item in alive:
        try:
            item.kill()
        except psutil.Error:
            pass


def memory_over_limit(sample: LoadSample, max_ram: float, max_gpu_mem: float, min_free_ram_gb: float) -> bool:
    if sample.ram_percent >= max_ram:
        return True
    if min_free_ram_gb > 0 and sample.ram_available_gb <= min_free_ram_gb:
        return True
    return sample.gpu_mem_percent is not None and sample.gpu_mem_percent >= max_gpu_mem


def hard_memory_over_limit(sample: LoadSample, max_ram: float, max_gpu_mem: float) -> bool:
    if sample.ram_percent >= max_ram:
        return True
    return sample.gpu_mem_percent is not None and sample.gpu_mem_percent >= max_gpu_mem


def soft_memory_pressure(sample: LoadSample, min_free_ram_gb: float) -> bool:
    return min_free_ram_gb > 0 and sample.ram_available_gb <= min_free_ram_gb


def wait_for_initial_headroom(args: argparse.Namespace) -> int | None:
    if args.preflight_timeout <= 0:
        return None
    deadline = time.time() + args.preflight_timeout
    last_print = 0.0
    while True:
        sample = sample_load()
        load = throttle_load(sample)
        memory_high = memory_over_limit(
            sample,
            args.max_ram_percent,
            args.max_gpu_mem_percent,
            args.min_free_ram_gb,
        )
        if load <= args.resume_percent and not memory_high:
            return None
        now = time.time()
        if now - last_print >= 20:
            print(f"[headroom] Waiting for initial headroom at {format_sample(sample)}", flush=True)
            last_print = now
        if now >= deadline:
            print(f"[headroom] Initial headroom timeout at {format_sample(sample)}", flush=True)
            return 75
        time.sleep(args.interval)


def main() -> int:
    args = parse_args()
    psutil.cpu_percent(interval=None)

    preflight_code = wait_for_initial_headroom(args)
    if preflight_code is not None:
        return preflight_code

    print(f"[headroom] Starting: {' '.join(args.command)}", flush=True)
    child = subprocess.Popen(args.command)
    child_proc = psutil.Process(child.pid)
    if not args.no_priority:
        set_low_priority(child_proc)

    suspended = False
    memory_suspended = False
    last_soft_memory_warning = 0.0
    try:
        while child.poll() is None:
            time.sleep(args.interval)
            sample = sample_load()
            load = throttle_load(sample)
            memory_high = hard_memory_over_limit(
                sample,
                args.max_ram_percent,
                args.max_gpu_mem_percent,
            )
            soft_memory_high = soft_memory_pressure(sample, args.min_free_ram_gb)

            if memory_high and args.memory_action == "exit":
                print(
                    f"[headroom] Memory limit exceeded at {format_sample(sample)}; "
                    "asking wrapper to relaunch smaller.",
                    flush=True,
                )
                terminate_tree(child_proc)
                return 70

            if memory_high and not suspended:
                print(f"[headroom] Pausing child for memory pressure at {format_sample(sample)}", flush=True)
                suspend_tree(child_proc)
                suspended = True
                memory_suspended = True
                continue

            if soft_memory_high and not memory_high:
                now = time.time()
                if now - last_soft_memory_warning >= 20:
                    print(
                        f"[headroom] Low free RAM warning at {format_sample(sample)}; "
                        "not pausing because soft free-RAM pressure does not release memory.",
                        flush=True,
                    )
                    last_soft_memory_warning = now

            if memory_suspended and not memory_high and load <= args.resume_percent:
                print(f"[headroom] Resuming child after memory pressure at {format_sample(sample)}", flush=True)
                resume_tree(child_proc)
                suspended = False
                memory_suspended = False
                continue

            if not suspended and load >= args.max_percent:
                print(f"[headroom] Pausing child at {format_sample(sample)}", flush=True)
                suspend_tree(child_proc)
                suspended = True
                continue

            if suspended and not memory_suspended and load <= args.resume_percent:
                print(f"[headroom] Resuming child at {format_sample(sample)}", flush=True)
                resume_tree(child_proc)
                suspended = False

        return child.returncode or 0
    finally:
        if suspended:
            resume_tree(child_proc)


if __name__ == "__main__":
    raise SystemExit(main())
