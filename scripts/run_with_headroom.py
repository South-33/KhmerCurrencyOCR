from __future__ import annotations

import argparse
import os
import shlex
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
)


configure_project_cache()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WINMEMORYCLEANER_ARGS = "/StandbyList /WorkingSet"


@dataclass
class LoadSample:
    cpu_percent: float
    ram_percent: float
    ram_available_gb: float
    gpu_percent: float | None
    gpu_mem_percent: float | None


@dataclass
class MemoryCleaner:
    exe: str | None
    args: list[str]
    task_name: str | None
    task_arg: str
    min_free_ram_gb: float
    cooldown_seconds: float
    timeout_seconds: float
    settle_seconds: float
    end_task_after_run: bool = False
    last_attempt: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.task_name or self.exe)

    @staticmethod
    def powershell_literal(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    def command(self) -> tuple[list[str], str]:
        if self.task_name:
            task_name = self.powershell_literal(self.task_name)
            task_run_arg = self.powershell_literal(self.task_arg) if self.task_arg else "$null"
            script = (
                "$ErrorActionPreference = 'Stop'; "
                "$service = New-Object -ComObject 'Schedule.Service'; "
                "$service.Connect(); "
                f"$task = $service.GetFolder('\\').GetTask({task_name}); "
                f"$null = $task.Run({task_run_arg})"
            )
            command = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ]
            return command, f"scheduled task {self.task_name} {self.task_arg}".strip()

        assert self.exe is not None
        command = [self.exe, *self.args]
        return command, " ".join(command)

    def maybe_run(self, reason: str) -> bool:
        if not self.enabled:
            return False

        now = time.time()
        if self.last_attempt and now - self.last_attempt < self.cooldown_seconds:
            return False

        self.last_attempt = now
        command, display = self.command()
        print(f"[headroom] Running memory cleaner for {reason}: {display}", flush=True)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout_seconds)
        except FileNotFoundError:
            print(f"[headroom] Memory cleaner command not found: {command[0]}", flush=True)
            return False
        except subprocess.TimeoutExpired:
            print(f"[headroom] Memory cleaner timed out after {self.timeout_seconds:.0f}s", flush=True)
            return False
        except OSError as exc:
            print(f"[headroom] Memory cleaner failed to start: {exc}", flush=True)
            return False

        if result.returncode:
            stderr = result.stderr.strip()
            detail = f": {stderr}" if stderr else ""
            print(f"[headroom] Memory cleaner exited {result.returncode}{detail}", flush=True)
        else:
            print("[headroom] Memory cleaner completed", flush=True)

        if self.settle_seconds > 0:
            time.sleep(self.settle_seconds)

        if self.task_name and self.end_task_after_run:
            end_result = subprocess.run(
                ["schtasks", "/End", "/TN", self.task_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if end_result.returncode == 0:
                print(f"[headroom] Ended memory cleaner task {self.task_name}", flush=True)
            else:
                detail = (end_result.stderr or end_result.stdout).strip()
                suffix = f": {detail}" if detail else ""
                print(f"[headroom] Memory cleaner task {self.task_name} was not ended{suffix}", flush=True)

        return True


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be numeric, got {raw!r}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command while leaving system headroom for interactive use."
    )
    parser.add_argument(
        "--max-percent",
        type=float,
        default=HEADROOM_MAX_PERCENT,
        help="Suspend at or above this active CPU load. GPU utilization is included only with --throttle-gpu-util.",
    )
    parser.add_argument(
        "--resume-percent",
        type=float,
        default=HEADROOM_RESUME_PERCENT,
        help="Resume at or below this active CPU load. GPU utilization is included only with --throttle-gpu-util.",
    )
    parser.add_argument(
        "--max-ram-percent",
        type=float,
        default=HEADROOM_MAX_RAM_PERCENT,
        help="Terminate the child if system RAM reaches this percent.",
    )
    parser.add_argument(
        "--max-gpu-mem-percent",
        type=float,
        default=HEADROOM_MAX_GPU_MEM_PERCENT,
        help="Terminate the child if GPU memory reaches this percent.",
    )
    parser.add_argument(
        "--min-free-ram-gb",
        type=float,
        default=HEADROOM_MIN_FREE_RAM_GB,
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
        "--throttle-gpu-util",
        action="store_true",
        help="Include GPU utilization in the active-load throttle. GPU memory is always guarded separately.",
    )
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for initial headroom before launching the child. Use 0 to disable.",
    )
    parser.add_argument(
        "--memory-clean-exe",
        default=os.environ.get("HEADROOM_MEMORY_CLEAN_EXE"),
        help=(
            "Optional RAM-cleaner executable to run on RAM pressure, for example WinMemoryCleaner or Mem Reduct. "
            "Can also be set with HEADROOM_MEMORY_CLEAN_EXE. For Mem Reduct's UAC helper task, "
            "prefer --memory-clean-task."
        ),
    )
    parser.add_argument(
        "--memory-clean-args",
        default=os.environ.get("HEADROOM_MEMORY_CLEAN_ARGS"),
        help=(
            "Shell-like argument string for --memory-clean-exe. Mem Reduct default cleanup is -clean; "
            "WinMemoryCleaner can use named areas such as /StandbyList /ModifiedFileCache /WorkingSet."
        ),
    )
    parser.add_argument(
        "--memory-clean-preset",
        choices=["winmemorycleaner", "winmemorycleaner-task", "memreduct"],
        default=os.environ.get("HEADROOM_MEMORY_CLEAN_PRESET") or None,
        help=(
            "Named RAM-cleaner preset. winmemorycleaner auto-finds the repo-local portable "
            ".cache_runtime tool and defaults to /StandbyList /WorkingSet; "
            "winmemorycleaner-task triggers the CashSnapWinMemoryCleaner scheduled task; "
            "memreduct triggers the installed memreductTask scheduled task with -clean, then ends it, "
            "unless an exe/task is supplied."
        ),
    )
    parser.add_argument(
        "--memory-clean-task",
        default=os.environ.get("HEADROOM_MEMORY_CLEAN_TASK"),
        help=(
            "Optional Windows Task Scheduler task to trigger on RAM pressure. "
            "Takes precedence over --memory-clean-exe and can also be set with HEADROOM_MEMORY_CLEAN_TASK."
        ),
    )
    parser.add_argument(
        "--memory-clean-task-arg",
        default=os.environ.get("HEADROOM_MEMORY_CLEAN_TASK_ARG"),
        help=(
            "Argument passed to --memory-clean-task. Defaults depend on preset: empty for "
            "winmemorycleaner-task and -clean for memreduct or a manual task. Use "
            "--memory-clean-task-arg=-clean when spelling it manually."
        ),
    )
    parser.add_argument(
        "--memory-clean-min-free-ram-gb",
        type=float,
        default=env_float("HEADROOM_MEMORY_CLEAN_MIN_FREE_RAM_GB", 1.5),
        help=(
            "Run the optional RAM cleaner only when available RAM is at or below this floor, "
            "or when --max-ram-percent is exceeded. This is separate from --min-free-ram-gb, "
            "which can be a gentler wait/warn threshold."
        ),
    )
    parser.add_argument(
        "--memory-clean-cooldown-seconds",
        type=float,
        default=env_float("HEADROOM_MEMORY_CLEAN_COOLDOWN_SECONDS", 600.0),
        help="Minimum seconds between RAM-cleaner attempts.",
    )
    parser.add_argument(
        "--memory-clean-timeout-seconds",
        type=float,
        default=env_float("HEADROOM_MEMORY_CLEAN_TIMEOUT_SECONDS", 30.0),
        help="Seconds to wait for the RAM cleaner before giving up.",
    )
    parser.add_argument(
        "--memory-clean-settle-seconds",
        type=float,
        default=env_float("HEADROOM_MEMORY_CLEAN_SETTLE_SECONDS", 5.0),
        help="Seconds to wait after a cleaner run before resampling memory.",
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
    if args.memory_clean_cooldown_seconds < 0:
        raise SystemExit("--memory-clean-cooldown-seconds must be >= 0.")
    if args.memory_clean_min_free_ram_gb < 0:
        raise SystemExit("--memory-clean-min-free-ram-gb must be >= 0.")
    if args.memory_clean_timeout_seconds <= 0:
        raise SystemExit("--memory-clean-timeout-seconds must be > 0.")
    if args.memory_clean_settle_seconds < 0:
        raise SystemExit("--memory-clean-settle-seconds must be >= 0.")
    return args


def find_cached_winmemorycleaner() -> str | None:
    tools_root = ROOT / ".cache_runtime" / "tools" / "winmemorycleaner"
    if not tools_root.exists():
        return None
    candidates = sorted(tools_root.glob("*/WinMemoryCleaner.exe"), reverse=True)
    return str(candidates[0]) if candidates else None


def build_memory_cleaner(args: argparse.Namespace) -> MemoryCleaner:
    exe = args.memory_clean_exe
    task_name = args.memory_clean_task
    task_arg = args.memory_clean_task_arg
    arg_text = args.memory_clean_args

    if args.memory_clean_preset == "winmemorycleaner":
        if not exe and not task_name:
            exe = find_cached_winmemorycleaner()
            if exe is None:
                raise SystemExit(
                    "WinMemoryCleaner preset requested but no cached executable was found under "
                    ".cache_runtime/tools/winmemorycleaner/*/WinMemoryCleaner.exe"
                )
        if arg_text is None:
            arg_text = DEFAULT_WINMEMORYCLEANER_ARGS
    elif args.memory_clean_preset == "winmemorycleaner-task":
        if not exe and not task_name:
            task_name = "CashSnapWinMemoryCleaner"
        if task_arg is None:
            task_arg = ""
        if arg_text is None:
            arg_text = ""
    elif args.memory_clean_preset == "memreduct":
        if not exe and not task_name:
            task_name = "memreductTask"
        if task_arg is None:
            task_arg = "-clean"
        if arg_text is None:
            arg_text = ""
    elif arg_text is None:
        exe_name = Path(exe).name.lower() if exe else ""
        arg_text = DEFAULT_WINMEMORYCLEANER_ARGS if exe_name == "winmemorycleaner.exe" else "-clean"
    if task_name and task_arg is None:
        task_arg = "-clean"
    if task_arg is None:
        task_arg = ""

    try:
        clean_args = shlex.split(arg_text) if arg_text else []
    except ValueError as exc:
        raise SystemExit(f"--memory-clean-args could not be parsed: {exc}") from exc
    return MemoryCleaner(
        exe=exe,
        args=clean_args,
        task_name=task_name,
        task_arg=task_arg,
        min_free_ram_gb=args.memory_clean_min_free_ram_gb,
        cooldown_seconds=args.memory_clean_cooldown_seconds,
        timeout_seconds=args.memory_clean_timeout_seconds,
        settle_seconds=args.memory_clean_settle_seconds,
        end_task_after_run=args.memory_clean_preset == "memreduct" and task_name == "memreductTask",
    )


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


def throttle_load(sample: LoadSample, *, include_gpu: bool) -> float:
    values = [sample.cpu_percent]
    if include_gpu and sample.gpu_percent is not None:
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


def ram_memory_pressure(sample: LoadSample, max_ram: float, min_free_ram_gb: float) -> bool:
    if sample.ram_percent >= max_ram:
        return True
    return soft_memory_pressure(sample, min_free_ram_gb)


def cleaner_ram_pressure(sample: LoadSample, max_ram: float, cleaner: MemoryCleaner) -> bool:
    if sample.ram_percent >= max_ram:
        return True
    return cleaner.min_free_ram_gb > 0 and sample.ram_available_gb <= cleaner.min_free_ram_gb


def wait_for_initial_headroom(args: argparse.Namespace, cleaner: MemoryCleaner) -> int | None:
    if args.preflight_timeout <= 0:
        return None
    deadline = time.time() + args.preflight_timeout
    last_print = 0.0
    while True:
        sample = sample_load()
        load = throttle_load(sample, include_gpu=args.throttle_gpu_util)
        memory_high = memory_over_limit(
            sample,
            args.max_ram_percent,
            args.max_gpu_mem_percent,
            args.min_free_ram_gb,
        )
        if load <= args.resume_percent and not memory_high:
            return None
        if cleaner_ram_pressure(sample, args.max_ram_percent, cleaner):
            if cleaner.maybe_run(f"preflight RAM pressure at {format_sample(sample)}"):
                continue
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
    cleaner = build_memory_cleaner(args)
    psutil.cpu_percent(interval=None)

    preflight_code = wait_for_initial_headroom(args, cleaner)
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
            load = throttle_load(sample, include_gpu=args.throttle_gpu_util)
            memory_high = hard_memory_over_limit(
                sample,
                args.max_ram_percent,
                args.max_gpu_mem_percent,
            )
            soft_memory_high = soft_memory_pressure(sample, args.min_free_ram_gb)
            ram_high = cleaner_ram_pressure(sample, args.max_ram_percent, cleaner)

            if ram_high and cleaner.maybe_run(f"runtime RAM pressure at {format_sample(sample)}"):
                sample = sample_load()
                load = throttle_load(sample, include_gpu=args.throttle_gpu_util)
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
