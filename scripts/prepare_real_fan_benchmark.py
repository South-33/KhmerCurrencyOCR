from __future__ import annotations

import argparse
import csv
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "manifests" / "real_fan_benchmark_sources.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download/copy local real fan benchmark candidate images.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--force", action="store_true", help="Overwrite existing local images.")
    return parser.parse_args()


def download(url: str, target: Path, force: bool) -> str:
    if target.exists() and target.stat().st_size > 0 and not force:
        return "exists"
    response = requests.get(url, headers={"User-Agent": "CashSnap benchmark prep/0.1"}, timeout=45)
    response.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return f"downloaded {len(response.content)} bytes"


def main() -> None:
    args = parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        local_path = ROOT / row["local_path"]
        source_image = row.get("source_image", "").strip()
        if not source_image:
            print(f"{row['image_id']}: missing source_image")
            continue
        status = download(source_image, local_path, args.force)
        print(f"{row['image_id']}: {status} -> {local_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
