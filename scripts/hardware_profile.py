from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is present in the project env.
    psutil = None


HEADROOM_MAX_PERCENT = 95.0
HEADROOM_RESUME_PERCENT = 88.0
HEADROOM_MAX_RAM_PERCENT = 95.0
HEADROOM_MAX_GPU_MEM_PERCENT = 95.0
# YOLO imports/training routinely need several GB of system RAM before batch
# size matters; below this, the headroom guard tends to kill runs mid-epoch.
HEADROOM_MIN_FREE_RAM_GB = 4.0

WEBGL_RENDER_JOBS = 2
WEBGL_RENDERER_BATCH_SIZE = 32
WEBGL_CHECK_JOBS = 4


@dataclass(frozen=True)
class GpuSnapshot:
    name: str
    mem_total_mb: float
    mem_used_mb: float
    util_percent: float

    @property
    def mem_free_mb(self) -> float:
        return max(0.0, self.mem_total_mb - self.mem_used_mb)

    @property
    def mem_free_gb(self) -> float:
        return self.mem_free_mb / 1024.0


def _run_nvidia_smi() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def read_gpu() -> GpuSnapshot | None:
    text = _run_nvidia_smi()
    first = text.splitlines()[0] if text else ""
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return None
    try:
        return GpuSnapshot(
            name=parts[0],
            mem_total_mb=float(parts[1]),
            mem_used_mb=float(parts[2]),
            util_percent=float(parts[3]),
        )
    except ValueError:
        return None


def ram_available_gb() -> float | None:
    if psutil is None:
        return None
    return psutil.virtual_memory().available / (1024**3)


def logical_cpus() -> int:
    if psutil is not None:
        return psutil.cpu_count(logical=True) or os.cpu_count() or 1
    return os.cpu_count() or 1


def has_cuda_gpu() -> bool:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in {"-1", "none", "None"}:
        return False
    return read_gpu() is not None


def recommended_device(value: str | None = "auto") -> str:
    if value not in {None, "", "auto"}:
        return str(value)
    return "0" if has_cuda_gpu() else "cpu"


def parse_auto_int(value: str | int, recommended: int, name: str, *, allow_zero: bool = True) -> int:
    if isinstance(value, int):
        parsed = value
    elif str(value).lower() == "auto":
        parsed = recommended
    else:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise SystemExit(f"--{name} must be an integer or 'auto'") from exc
    floor = 0 if allow_zero else 1
    if parsed < floor:
        raise SystemExit(f"--{name} must be >= {floor}")
    return parsed


def recommended_train_batch(imgsz: int, gpu: GpuSnapshot | None = None, ram_gb: float | None = None) -> int:
    gpu = read_gpu() if gpu is None else gpu
    ram_gb = ram_available_gb() if ram_gb is None else ram_gb
    if gpu is None:
        return 2
    free = gpu.mem_free_gb
    ram = ram_gb if ram_gb is not None else 0.0
    if imgsz <= 416:
        if free >= 5.5 and ram >= 6.0:
            return 64
        if free >= 4.0 and ram >= 5.0:
            return 32
        if free >= 3.0 and ram >= 4.0:
            return 16
        return 8 if free >= 2.0 and ram >= 4.0 else 4
    if imgsz <= 640:
        return 8 if free >= 5.0 and ram >= 5.0 else 4
    if imgsz <= 960:
        return 4 if free >= 5.0 else 2
    return 2


def recommended_val_batch(imgsz: int, gpu: GpuSnapshot | None = None, ram_gb: float | None = None) -> int:
    gpu = read_gpu() if gpu is None else gpu
    ram_gb = ram_available_gb() if ram_gb is None else ram_gb
    if gpu is None:
        return 4
    free = gpu.mem_free_gb
    ram = ram_gb if ram_gb is not None else 0.0
    if imgsz <= 416:
        if free >= 5.0 and ram >= 4.0:
            return 64
        if free >= 3.0 and ram >= 3.0:
            return 32
        return 16 if free >= 2.0 else 8
    if imgsz <= 640:
        return 16 if free >= 5.0 and ram >= 4.0 else 8
    return 8 if free >= 4.0 else 4


def recommended_workers(kind: str = "train", ram_gb: float | None = None) -> int:
    ram_gb = ram_available_gb() if ram_gb is None else ram_gb
    ram = ram_gb if ram_gb is not None else 0.0
    cpus = logical_cpus()
    if cpus < 8 or ram < 3.0:
        return 0
    if kind in {"val", "eval", "validation"}:
        return 2
    if kind == "render":
        return WEBGL_RENDER_JOBS
    if kind == "browser":
        return 1
    if kind == "train":
        return 0
    return 2


def headroom_defaults() -> dict[str, float]:
    return {
        "max_percent": HEADROOM_MAX_PERCENT,
        "resume_percent": HEADROOM_RESUME_PERCENT,
        "max_ram_percent": HEADROOM_MAX_RAM_PERCENT,
        "max_gpu_mem_percent": HEADROOM_MAX_GPU_MEM_PERCENT,
        "min_free_ram_gb": HEADROOM_MIN_FREE_RAM_GB,
    }


def webgl_defaults() -> dict[str, int]:
    return {
        "render_jobs": WEBGL_RENDER_JOBS,
        "renderer_batch_size": WEBGL_RENDERER_BATCH_SIZE,
        "check_jobs": WEBGL_CHECK_JOBS,
    }
