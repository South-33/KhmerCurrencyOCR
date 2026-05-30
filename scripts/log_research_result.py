"""Initialize and append CashSnap research-harness result rows."""

from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "results.tsv"
FIELDNAMES = [
    "timestamp",
    "commit",
    "phase",
    "run_dir",
    "status",
    "clean_map50",
    "clean_map5095",
    "real_same_class",
    "real_any_class",
    "browser_smoke",
    "peak_memory_gb",
    "description",
]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except subprocess.SubprocessError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def latest_results_metrics(run_dir: Path) -> dict[str, str]:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return {}
    with results_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    latest = rows[-1]
    return {
        "clean_map50": latest.get("metrics/mAP50(B)", "").strip(),
        "clean_map5095": latest.get("metrics/mAP50-95(B)", "").strip(),
    }


def ensure_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t", lineterminator="\n")
        writer.writeheader()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--init", action="store_true", help="Create the ledger header and exit.")
    parser.add_argument("--phase", default="", help="renderer, detector, classifier, browser, data, docs, etc.")
    parser.add_argument("--run-dir", default="", help="Optional Ultralytics run directory to read results.csv from.")
    parser.add_argument("--status", choices=["keep", "discard", "crash", "note"], default="note")
    parser.add_argument("--clean-map50", default="", help="Override or provide clean validation mAP50.")
    parser.add_argument("--clean-map5095", default="", help="Override or provide clean validation mAP50-95.")
    parser.add_argument("--real-same-class", default="", help="Scoreable real same-class count or metric.")
    parser.add_argument("--real-any-class", default="", help="Scoreable real any-class count or metric.")
    parser.add_argument("--browser-smoke", default="", help="pass/fail/na or brief browser smoke result.")
    parser.add_argument("--peak-memory-gb", default="", help="Peak memory in GB when known.")
    parser.add_argument("--description", default="", help="Short tab-free description.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ledger = resolve_path(args.ledger)
    ensure_header(ledger)
    if args.init:
        print(f"initialized {ledger}")
        return

    run_dir = resolve_path(args.run_dir) if args.run_dir else None
    metrics = latest_results_metrics(run_dir) if run_dir else {}
    clean_map50 = args.clean_map50 or metrics.get("clean_map50", "")
    clean_map5095 = args.clean_map5095 or metrics.get("clean_map5095", "")
    description = args.description.replace("\t", " ").strip()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "phase": args.phase,
        "run_dir": str(run_dir.relative_to(ROOT) if run_dir and run_dir.is_relative_to(ROOT) else run_dir or ""),
        "status": args.status,
        "clean_map50": clean_map50,
        "clean_map5095": clean_map5095,
        "real_same_class": args.real_same_class,
        "real_any_class": args.real_any_class,
        "browser_smoke": args.browser_smoke,
        "peak_memory_gb": args.peak_memory_gb,
        "description": description,
    }
    with ledger.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t", lineterminator="\n")
        writer.writerow(row)
    print(f"appended {args.status} row to {ledger}")


if __name__ == "__main__":
    main()
