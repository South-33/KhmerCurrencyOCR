from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[1]

DEFAULT_CANDIDATE_CONFIG = (
    REPO
    / "configs/official21/"
    "cashsnap_official21_roboflow_plus_current_accept6_cap180_empty360_plus_accept11_usd2_khr200_v1.yaml"
)
DEFAULT_CONTROL_CONFIG = (
    REPO
    / "configs/official21/"
    "cashsnap_official21_roboflow_plus_current_accept6_cap180_empty360_plus_accept11_usd2_khr200_rowcountctrl_v1.yaml"
)
DEFAULT_BROWSER_STACK_CONFIG = (
    REPO / "configs/cashsnap_two_stage_real_synth_p24_khr100_unknown_gate_browser_stack.json"
)
DEFAULT_ACCEPT11_README = REPO / "runs/cashsnap/official21_missing_schema_seed_accept11_v1/README.md"
DEFAULT_OUT_JSON = REPO / "runs/cashsnap/official21_missing_schema_seed_accept11_v1/launch_readiness_latest.json"
DEFAULT_MEMORY_CLEAN_TASKS = ("memreductTask=-clean", "CashSnapWinMemoryCleaner")


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def load_yaml(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        issues.append(f"missing YAML: {repo_rel(path)}")
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        issues.append(f"YAML is not a mapping: {repo_rel(path)}")
        return {}
    return data


def load_json(path: Path, issues: list[str]) -> dict[str, Any]:
    if not path.exists():
        issues.append(f"missing JSON: {repo_rel(path)}")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        issues.append(f"JSON is not a mapping: {repo_rel(path)}")
        return {}
    return data


def resolve_config_path(raw_path: str | None, config_path: Path, issues: list[str]) -> Path | None:
    if not raw_path:
        issues.append(f"missing path in {repo_rel(config_path)}")
        return None
    direct = REPO / raw_path
    if direct.exists():
        return direct
    local = config_path.parent / raw_path
    if local.exists():
        return local
    issues.append(f"path does not exist from {repo_rel(config_path)}: {raw_path}")
    return direct


def count_train_rows(config_path: Path, issues: list[str]) -> int | None:
    config = load_yaml(config_path, issues)
    train_path = resolve_config_path(config.get("train"), config_path, issues)
    if train_path is None or not train_path.exists():
        return None
    rows = [line for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(rows)


def get_memory() -> dict[str, float]:
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    if not ok:
        raise ctypes.WinError()
    gb = 1024**3
    free_gb = status.ullAvailPhys / gb
    total_gb = status.ullTotalPhys / gb
    return {
        "free_gb": round(free_gb, 3),
        "total_gb": round(total_gb, 3),
        "used_percent": round((1.0 - (free_gb / total_gb)) * 100.0, 1) if total_gb else 100.0,
    }


def query_scheduled_task(task_name: str) -> dict[str, Any]:
    result = subprocess.run(
        ["schtasks", "/Query", "/TN", task_name],
        cwd=REPO,
        text=True,
        capture_output=True,
    )
    return {
        "task": task_name,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-1000:],
        "stderr_tail": result.stderr[-1000:],
    }


def powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def parse_task_spec(task_spec: str) -> tuple[str, str]:
    task_name, separator, task_arg = task_spec.partition("=")
    task_name = task_name.strip()
    if not task_name:
        raise SystemExit(f"empty scheduled task name in memory-clean task spec: {task_spec!r}")
    return task_name, task_arg if separator else ""


def run_scheduled_task(task_name: str, task_arg: str) -> subprocess.CompletedProcess[str]:
    task_name_literal = powershell_literal(task_name)
    task_arg_literal = powershell_literal(task_arg) if task_arg else "$null"
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "$service = New-Object -ComObject 'Schedule.Service'; "
        "$service.Connect(); "
        f"$task = $service.GetFolder('\\').GetTask({task_name_literal}); "
        f"$null = $task.Run({task_arg_literal})"
    )
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        cwd=REPO,
        text=True,
        capture_output=True,
    )


def end_scheduled_task(task_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["schtasks", "/End", "/TN", task_name],
        cwd=REPO,
        text=True,
        capture_output=True,
    )


def run_memory_cleaners(task_specs: list[str], settle_seconds: float) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for task_spec in task_specs:
        task_name, task_arg = parse_task_spec(task_spec)
        before = get_memory()
        query_before = query_scheduled_task(task_name)
        run_result = run_scheduled_task(task_name, task_arg)
        if settle_seconds > 0:
            time.sleep(settle_seconds)
        end_result = None
        if task_name == "memreductTask" and task_arg == "-clean":
            end_result = end_scheduled_task(task_name)
        after = get_memory()
        query_after = query_scheduled_task(task_name)
        attempt = {
            "task": task_name,
            "task_arg": task_arg,
            "query_before": query_before,
            "run_returncode": run_result.returncode,
            "run_stdout_tail": run_result.stdout[-1000:],
            "run_stderr_tail": run_result.stderr[-1000:],
            "before_memory": before,
            "after_memory": after,
            "query_after": query_after,
        }
        if end_result is not None:
            attempt["end_returncode"] = end_result.returncode
            attempt["end_stdout_tail"] = end_result.stdout[-1000:]
            attempt["end_stderr_tail"] = end_result.stderr[-1000:]
        attempts.append(attempt)
    return attempts


def run_check(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
    }


def artifact_exists(path: str | None, base_path: Path, issues: list[str]) -> dict[str, Any]:
    if not path:
        issues.append(f"missing artifact path in {repo_rel(base_path)}")
        return {"path": None, "exists": False, "size_mb": 0.0}
    artifact_path = REPO / path
    exists = artifact_path.exists()
    if not exists:
        issues.append(f"missing artifact: {path}")
    size_mb = artifact_path.stat().st_size / (1024 * 1024) if exists else 0.0
    return {"path": path, "exists": exists, "size_mb": round(size_mb, 2)}


def check_browser_stack(config_path: Path, issues: list[str]) -> dict[str, Any]:
    config = load_json(config_path, issues)
    detector_config = config.get("detector", {})
    fragment_config = config.get("fragment_classifier", {})
    detector = artifact_exists(detector_config.get("onnx_path") or detector_config.get("path"), config_path, issues)
    fragment = artifact_exists(fragment_config.get("onnx_path") or fragment_config.get("path"), config_path, issues)
    total_mb = detector["size_mb"] + fragment["size_mb"]
    return {
        "config": repo_rel(config_path),
        "detector": detector,
        "fragment_classifier": fragment,
        "total_artifact_mb": round(total_mb, 2),
        "size_ok_for_browser": total_mb <= 20.0,
    }


def check_taxonomy() -> dict[str, Any]:
    latest = REPO / "runs/cashsnap/currency_taxonomy_coverage_latest.json"
    if not latest.exists():
        return {"latest_json": repo_rel(latest), "exists": False, "coverage": "missing"}
    data = json.loads(latest.read_text(encoding="utf-8"))
    return {
        "latest_json": repo_rel(latest),
        "exists": True,
        "coverage": data.get("coverage") or data.get("status"),
        "model_missing": data.get("missing", {}).get("model_schema")
        if isinstance(data.get("missing"), dict)
        else data.get("model_schema"),
    }


def build_recommended_command(candidate: Path, control: Path) -> list[str]:
    return [
        "rl",
        "python",
        "scripts\\run_yolo_fixed_step_probe.py",
        "--baseline-data",
        repo_rel(control).replace("/", "\\"),
        "--candidate-data",
        repo_rel(candidate).replace("/", "\\"),
        "--baseline-label",
        "accept11_rowcountctrl",
        "--candidate-label",
        "accept11_usd2_khr200",
        "--model",
        "runs\\cashsnap\\fixed_step_real_p24_plus_strictbest_synth_p24_from_clean_e1_i416_b2_w0_adamw_lr5e5_nowarmup_noamp_cachefalse_steps318_seed0\\weights\\last.pt",
        "--project",
        "runs\\cashsnap",
        "--epochs",
        "50",
        "--imgsz",
        "416",
        "--batch",
        "2",
        "--workers",
        "0",
        "--eval-batch",
        "2",
        "--eval-workers",
        "0",
        "--cache",
        "false",
        "--optimizer",
        "AdamW",
        "--lr0",
        "0.00005",
        "--warmup-epochs",
        "0",
        "--max-train-batches",
        "160",
        "--fail-on-row-count-mismatch",
        "--fail-on-step-reference-mismatch",
        "--fail-on-train-phase-mismatch",
        "--memory-clean-preset",
        "winmemorycleaner-task",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the next CashSnap detector launch is safe and well-scoped."
    )
    parser.add_argument("--candidate-data", type=Path, default=DEFAULT_CANDIDATE_CONFIG)
    parser.add_argument("--control-data", type=Path, default=DEFAULT_CONTROL_CONFIG)
    parser.add_argument("--browser-stack-config", type=Path, default=DEFAULT_BROWSER_STACK_CONFIG)
    parser.add_argument("--min-free-ram-gb", type=float, default=4.0)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--skip-subchecks", action="store_true")
    parser.add_argument(
        "--try-memory-cleaners",
        action="store_true",
        help="If RAM is below the floor, run scheduled cleaner tasks once and remeasure.",
    )
    parser.add_argument(
        "--memory-clean-task",
        action="append",
        default=None,
        help=(
            "Scheduled task to run with --try-memory-cleaners. Use TASK=ARG to pass a task argument, "
            "for example memreductTask=-clean. Defaults to installed Mem Reduct and WinMemoryCleaner tasks."
        ),
    )
    parser.add_argument("--memory-clean-settle-seconds", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    issues: list[str] = []
    warnings: list[str] = []

    candidate = (REPO / args.candidate_data).resolve() if not args.candidate_data.is_absolute() else args.candidate_data
    control = (REPO / args.control_data).resolve() if not args.control_data.is_absolute() else args.control_data
    browser_stack = (
        (REPO / args.browser_stack_config).resolve()
        if not args.browser_stack_config.is_absolute()
        else args.browser_stack_config
    )
    out_json = (REPO / args.out_json).resolve() if not args.out_json.is_absolute() else args.out_json

    candidate_rows = count_train_rows(candidate, issues)
    control_rows = count_train_rows(control, issues)
    if candidate_rows != control_rows:
        issues.append(f"candidate/control train rows differ: {candidate_rows}/{control_rows}")

    if not DEFAULT_ACCEPT11_README.exists():
        warnings.append(f"missing run-local README: {repo_rel(DEFAULT_ACCEPT11_README)}")

    memory_cleaner_attempts: list[dict[str, Any]] = []
    memory = get_memory()
    if args.try_memory_cleaners and memory["free_gb"] < args.min_free_ram_gb:
        tasks = args.memory_clean_task or list(DEFAULT_MEMORY_CLEAN_TASKS)
        memory_cleaner_attempts = run_memory_cleaners(tasks, args.memory_clean_settle_seconds)
        memory = get_memory()
    if memory["free_gb"] < args.min_free_ram_gb:
        issues.append(f"free RAM {memory['free_gb']} GB is below launch floor {args.min_free_ram_gb} GB")

    browser = check_browser_stack(browser_stack, issues)
    if not browser["size_ok_for_browser"]:
        warnings.append(f"browser stack is over 20 MB: {browser['total_artifact_mb']} MB")

    subchecks: dict[str, Any] = {}
    if not args.skip_subchecks:
        subchecks["accept11"] = run_check([sys.executable, "scripts/check_official21_accept11_artifacts.py"])
        subchecks["registry"] = run_check([sys.executable, "scripts/check_data_lifecycle_registry.py"])
        for name, result in subchecks.items():
            if result["returncode"] != 0:
                issues.append(f"subcheck failed: {name}")

    taxonomy = check_taxonomy()
    if taxonomy.get("coverage") == "blocked":
        missing = taxonomy.get("model_missing") or []
        suffix = f": {', '.join(missing)}" if missing else ""
        warnings.append(f"currency taxonomy coverage remains blocked for full official21 schema{suffix}")

    ready = not issues
    report = {
        "ready_for_detector_launch": ready,
        "issues": issues,
        "warnings": warnings,
        "memory": memory,
        "memory_cleaner_attempts": memory_cleaner_attempts,
        "min_free_ram_gb": args.min_free_ram_gb,
        "candidate": {"config": repo_rel(candidate), "train_rows": candidate_rows},
        "control": {"config": repo_rel(control), "train_rows": control_rows},
        "browser_stack": browser,
        "taxonomy": taxonomy,
        "subchecks": subchecks,
        "recommended_command": build_recommended_command(candidate, control),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"launch_readiness={str(ready).lower()} json={repo_rel(out_json)}")
    print(f"memory_free_gb={memory['free_gb']} train_rows={candidate_rows}/{control_rows}")
    if issues:
        print("issues:")
        for issue in issues:
            print(f"- {issue}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"- {warning}")
    missing_taxonomy = taxonomy.get("model_missing")
    if missing_taxonomy:
        print(f"taxonomy_model_missing={','.join(missing_taxonomy)}")
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
