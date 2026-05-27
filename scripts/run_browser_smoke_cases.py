from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "manifests" / "browser_smoke_cases.csv"
DEFAULT_OUT_DIR = ROOT / ".agent" / "browser_smoke_cases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run labeled CashSnap browser smoke cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--timeout-ms", default="120000")
    parser.add_argument("--edge", default="", help="Optional Edge executable path forwarded to the node smoke script.")
    parser.add_argument("--no-artifacts", action="store_true", help="Do not write per-case screenshots or detection CSVs.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_cases(path: Path) -> list[dict[str, str]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def command_for_case(case: dict[str, str], args: argparse.Namespace) -> list[str]:
    command = [
        "node",
        "scripts/smoke_browser_demo_cdp.cjs",
        "--image",
        case["image"],
        "--labels",
        case["labels"],
        "--timeout-ms",
        args.timeout_ms,
    ]
    add_optional_number(command, "--min-same-class", case.get("min_same_class", ""))
    add_optional_number(command, "--min-any-class", case.get("min_any_class", ""))
    add_optional_number(command, "--max-khr-error", case.get("max_khr_error", ""))
    add_optional_number(command, "--max-usd-error", case.get("max_usd_error", ""))
    if args.edge:
        command.extend(["--edge", args.edge])
    if not args.no_artifacts:
        out_dir = resolve(args.out_dir)
        case_id = case["case_id"]
        command.extend(
            [
                "--screenshot",
                str(out_dir / f"{case_id}.png"),
                "--out-csv",
                str(out_dir / f"{case_id}.csv"),
            ]
        )
    return command


def main() -> None:
    args = parse_args()
    cases = read_cases(args.cases)
    if not cases:
        raise SystemExit(f"no smoke cases found in {resolve(args.cases).relative_to(ROOT)}")
    if not args.no_artifacts:
        resolve(args.out_dir).mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for case in cases:
        case_id = case["case_id"]
        command = command_for_case(case, args)
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if result.returncode:
            print(result.stdout, end="")
            print(result.stderr, end="")
            failures.append(f"{case_id}: exit {result.returncode}")
            continue
        summary = parse_summary(result.stdout)
        evaluation = summary.get("evaluation") or {}
        print(
            f"{case_id}: count={summary.get('totalCount')} "
            f"khr={summary.get('khrValue')} usd={summary.get('usdValue')} "
            f"same={evaluation.get('matchedSameClass')}/{evaluation.get('gtCount')} "
            f"any={evaluation.get('matchedAnyClass')}/{evaluation.get('gtCount')} "
            f"khr_error={evaluation.get('khrValueError')} usd_error={evaluation.get('usdValueError')}"
        )
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
