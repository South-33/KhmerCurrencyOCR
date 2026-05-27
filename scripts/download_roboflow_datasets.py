"""
Download the Roboflow datasets/leads needed for CashSnap.

Usage:
    python scripts/download_roboflow_datasets.py

Reads ROBOFLOW_API_KEY from .env (or environment).
Each dataset is saved under data/raw_datasets/roboflow_<name>/.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download configured Roboflow CashSnap datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Download only a matching project_id or output_name. Repeat for multiple datasets.",
    )
    parser.add_argument("--list", action="store_true", help="List configured datasets and exit without an API key.")
    return parser.parse_args()


def load_env() -> None:
    """Load .env file into os.environ if present."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


# Dataset definitions: (workspace_id, project_id, version, output_name, export_format)
# Versions verified via API - use clean/raw versions, not heavily pre-augmented ones.
# Khmer-US-currency v3 = 1,782 raw images (best base; v5+ add augmentation we don't want pre-baked)
# Cambodia Currency Project v2 = only available version (~552-615 images, 7 KHR classes)
# KHMER SCAN v1 = 145 images with train/val/test splits
DATASETS = [
    (
        "robot-yfusg",
        "khmer-us-currency-jofw1",
        3,                              # v3 = 1,782 clean images (train:1254 val:348 test:180)
        "roboflow_khmer_us_currency",
        "yolov8",
    ),
    (
        "khmer-riel-classification-computer-vision",
        "cambodia-currency-project",
        2,                              # v2 = only public version (~615 images, 7 KHR classes)
        "roboflow_cambodia_currency_project",
        "yolov8",
    ),
    (
        "test-gl3sj",
        "khmer-scan",
        1,                              # v1 = 145 images (train:102 val:29 test:14)
        "roboflow_khmer_scan",
        "yolov8",
    ),
    (
        "ddd8889",
        "cuurecy-detection-is",
        2,                              # v2 = 2.3k instance-segmentation images, KHR/USD front/back classes
        "roboflow_cuurecy_detection_is",
        "yolov8",
    ),
    (
        "thesis-ivrzn",
        "currency-deteection",
        2,                              # v2 = 4.7k broad Asian-currency detector, limited KHR classes
        "roboflow_currency_deteection",
        "yolov8",
    ),
    (
        "mata-uang",
        "currency-detection-y4tb2",
        5,                              # v5 = ASEAN currency detector with one Cambodian Riel class
        "roboflow_asean_currency_detection",
        "yolov8",
    ),
]


def download_dataset(
    api_key: str,
    workspace: str,
    project_id: str,
    version: int,
    output_name: str,
    export_format: str,
) -> Path:
    from roboflow import Roboflow  # type: ignore

    out_dir = ROOT / "data" / "raw_datasets" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {workspace}/{project_id} v{version}")
    print(f"  -> {out_dir}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_id)
    dataset = project.version(version).download(
        model_format=export_format,
        location=str(out_dir),
        overwrite=True,
    )
    print(f"  OK: {dataset.location}")
    return out_dir


def main() -> None:
    args = parse_args()
    if args.list:
        for workspace, project_id, version, output_name, export_format in DATASETS:
            print(f"{output_name}: {workspace}/{project_id} v{version} format={export_format}")
        return

    load_env()
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        print("ERROR: ROBOFLOW_API_KEY not set. Add it to .env or export it.", file=sys.stderr)
        sys.exit(1)

    selected = set(args.dataset)
    datasets = [
        item
        for item in DATASETS
        if not selected or item[1] in selected or item[3] in selected
    ]
    if not datasets:
        print(f"ERROR: no configured dataset matched: {', '.join(args.dataset)}", file=sys.stderr)
        sys.exit(1)

    failed: list[str] = []
    for workspace, project_id, version, output_name, export_format in datasets:
        try:
            download_dataset(api_key, workspace, project_id, version, output_name, export_format)
        except Exception as exc:
            print(f"\n  FAIL {project_id}: {exc}", file=sys.stderr)
            failed.append(project_id)

    print(f"\n{'='*60}")
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"All {len(datasets)} datasets downloaded successfully.")
        print("\nNext step: run  python scripts/prepare_roboflow_datasets.py")


if __name__ == "__main__":
    main()
