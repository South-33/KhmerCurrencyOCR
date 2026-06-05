from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "manifests" / "browser_smoke_cases.csv"
DEFAULT_OUT_DIR = ROOT / ".agent" / "browser_smoke_cases"
DEFAULT_BROWSER_APP = ROOT / "demo" / "browser" / "app.js"
DEFAULT_BROWSER_INDEX = ROOT / "demo" / "browser" / "index.html"
DEFAULT_SMOKE_CDP = ROOT / "scripts" / "smoke_browser_demo_cdp.cjs"
NUMERIC_FIELDS = ["min_same_class", "min_any_class", "max_count_error", "max_khr_error", "max_usd_error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run labeled CashSnap browser smoke cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeout-ms", default="120000")
    parser.add_argument("--port-base", type=int, default=8877, help="First local HTTP port for smoke cases.")
    parser.add_argument("--debug-port-base", type=int, default=9323, help="First Edge DevTools port for smoke cases.")
    parser.add_argument("--edge", default="", help="Optional Edge executable path forwarded to the node smoke script.")
    parser.add_argument("--proposal-conf", default="", help="Optional browser detector proposal confidence override.")
    parser.add_argument("--detector-override", default="", help="Optional detector-vs-fragment fusion threshold override.")
    parser.add_argument("--nms-iou", default="", help="Optional fusion NMS IoU override.")
    parser.add_argument("--crop-padding", default="", help="Optional fragment crop padding override.")
    parser.add_argument("--summary-json", type=Path, help="Optional aggregate JSON summary output path.")
    parser.add_argument("--validate-only", action="store_true", help="Validate the case manifest without launching Edge.")
    parser.add_argument("--no-artifacts", action="store_true", help="Do not write per-case screenshots or detection CSVs.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: Path) -> dict[str, object]:
    resolved = resolve(path)
    row: dict[str, object] = {"path": repo_path(resolved), "exists": resolved.exists()}
    if resolved.is_file():
        row["sha256"] = file_sha256(resolved)
    return row


def browser_stack_config_path() -> Path:
    app_text = DEFAULT_BROWSER_APP.read_text(encoding="utf-8")
    match = re.search(r'STACK_CONFIG_URL\s*=\s*"([^"]+)"', app_text)
    if not match:
        raise ValueError(f"{repo_path(DEFAULT_BROWSER_APP)}: missing STACK_CONFIG_URL")
    raw = match.group(1)
    if raw.startswith("/"):
        return ROOT / raw.lstrip("/")
    return resolve(Path(raw))


def read_browser_stack_config(path: Path) -> dict:
    resolved = resolve(path)
    if not resolved.exists():
        return {}
    data = json.loads(resolved.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def browser_input_fingerprints(cases_path: Path) -> dict[str, dict[str, object]]:
    stack_path = browser_stack_config_path()
    stack_config = read_browser_stack_config(stack_path)
    fingerprints: dict[str, dict[str, object]] = {
        "case_manifest": fingerprint_file(cases_path),
        "smoke_runner": fingerprint_file(Path(__file__)),
        "smoke_cdp": fingerprint_file(DEFAULT_SMOKE_CDP),
        "browser_app": fingerprint_file(DEFAULT_BROWSER_APP),
        "browser_index": fingerprint_file(DEFAULT_BROWSER_INDEX),
        "browser_stack_config": fingerprint_file(stack_path),
    }
    detector_path = str((stack_config.get("detector") or {}).get("path", "")).strip()
    if detector_path:
        fingerprints["detector_model"] = fingerprint_file(Path(detector_path))
    fragment_path = str((stack_config.get("fragment_classifier") or {}).get("path", "")).strip()
    if fragment_path:
        fingerprints["fragment_classifier_model"] = fingerprint_file(Path(fragment_path))
    return fingerprints


def read_cases(path: Path) -> list[dict[str, str]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def repo_or_url_path(value: str) -> Path:
    if value.startswith("/"):
        return ROOT / value.lstrip("/")
    return resolve(Path(value))


def validate_case(case: dict[str, str], index: int) -> None:
    prefix = f"case row {index + 2}"
    for field in ["case_id", "image", "labels"]:
        if not case.get(field, "").strip():
            raise ValueError(f"{prefix}: missing {field}")
    image_path = repo_or_url_path(case["image"].strip())
    if not image_path.exists():
        raise ValueError(f"{case['case_id']}: image not found: {case['image']}")
    labels_path = resolve(Path(case["labels"].strip()))
    if not labels_path.exists():
        raise ValueError(f"{case['case_id']}: labels not found: {case['labels']}")
    for field in NUMERIC_FIELDS:
        value = case.get(field, "").strip()
        if not value:
            continue
        try:
            float(value)
        except ValueError as exc:
            raise ValueError(f"{case['case_id']}: {field} must be numeric, got {value!r}") from exc


def add_optional_number(command: list[str], flag: str, value: str) -> None:
    value = value.strip()
    if value:
        command.extend([flag, value])


def parse_summary(stdout: str) -> dict:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise ValueError("smoke output did not contain a JSON summary")
    return json.loads(stdout[start : end + 1])


def browser_stress_report(args: argparse.Namespace, summaries: list[dict], failures: list[str]) -> dict[str, object]:
    return {
        "schema": "cashsnap_browser_synthetic_stress_report_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "case_count": len(summaries),
        "case_manifest": repo_path(resolve(args.cases)),
        "input_fingerprints": browser_input_fingerprints(args.cases),
        "settings": {
            "timeout_ms": int(args.timeout_ms),
            "edge": args.edge,
            "proposal_conf": args.proposal_conf,
            "detector_override": args.detector_override,
            "nms_iou": args.nms_iou,
            "crop_padding": args.crop_padding,
        },
        "failures": failures,
        "cases": summaries,
    }


def command_for_case(case: dict[str, str], args: argparse.Namespace, index: int) -> list[str]:
    command = [
        "node",
        "scripts/smoke_browser_demo_cdp.cjs",
        "--image",
        case["image"].strip(),
        "--labels",
        case["labels"].strip(),
        "--port",
        str(args.port_base + index),
        "--debug-port",
        str(args.debug_port_base + index),
        "--timeout-ms",
        args.timeout_ms,
    ]
    add_optional_number(command, "--min-same-class", case.get("min_same_class", ""))
    add_optional_number(command, "--min-any-class", case.get("min_any_class", ""))
    add_optional_number(command, "--max-count-error", case.get("max_count_error", ""))
    add_optional_number(command, "--max-khr-error", case.get("max_khr_error", ""))
    add_optional_number(command, "--max-usd-error", case.get("max_usd_error", ""))
    if args.edge:
        command.extend(["--edge", args.edge])
    if args.proposal_conf:
        command.extend(["--proposal-conf", args.proposal_conf])
    if args.detector_override:
        command.extend(["--detector-override", args.detector_override])
    if args.nms_iou:
        command.extend(["--nms-iou", args.nms_iou])
    if args.crop_padding:
        command.extend(["--crop-padding", args.crop_padding])
    if not args.no_artifacts:
        out_dir = resolve(args.out_dir)
        case_id = case["case_id"]
        command.extend(
            [
                "--screenshot",
                str(out_dir / f"{case_id}.png"),
                "--out-csv",
                str(out_dir / f"{case_id}.csv"),
                "--out-json",
                str(out_dir / f"{case_id}.json"),
            ]
        )
    return command


def main() -> None:
    args = parse_args()
    cases = read_cases(args.cases)
    if not cases:
        raise SystemExit(f"no smoke cases found in {resolve(args.cases).relative_to(ROOT)}")
    for index, case in enumerate(cases):
        validate_case(case, index)
    if args.validate_only:
        print(f"validated {len(cases)} browser smoke case(s)")
        return
    if not args.no_artifacts:
        resolve(args.out_dir).mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    summaries: list[dict] = []
    for index, case in enumerate(cases):
        case_id = case["case_id"]
        command = command_for_case(case, args, index)
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        try:
            summary = parse_summary(result.stdout)
        except ValueError as exc:
            print(result.stdout, end="")
            print(result.stderr, end="")
            failures.append(f"{case_id}: exit {result.returncode}" if result.returncode else f"{case_id}: {exc}")
            continue
        summary["caseId"] = case_id
        summary["notes"] = case.get("notes", "")
        summary["smokeExitCode"] = result.returncode
        if result.stderr.strip():
            summary["smokeError"] = result.stderr.strip()
        summaries.append(summary)
        evaluation = summary.get("evaluation") or {}
        print(
            f"{case_id}: count={summary.get('totalCount')} "
            f"khr={summary.get('khrValue')} usd={summary.get('usdValue')} "
            f"same={evaluation.get('matchedSameClass')}/{evaluation.get('gtCount')} "
            f"any={evaluation.get('matchedAnyClass')}/{evaluation.get('gtCount')} "
            f"count_error={evaluation.get('countError')} "
            f"khr_error={evaluation.get('khrValueError')} usd_error={evaluation.get('usdValueError')}"
        )
        if result.returncode:
            if result.stderr:
                print(result.stderr, end="")
            failures.append(f"{case_id}: exit {result.returncode}")
            continue
    summary_json = args.summary_json
    if summary_json is None and not args.no_artifacts:
        summary_json = resolve(args.out_dir) / "summary.json"
    if summary_json is not None:
        summary_path = resolve(summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(browser_stress_report(args, summaries, failures), indent=2) + "\n", encoding="utf-8")
        print(f"wrote {repo_path(summary_path)}")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
