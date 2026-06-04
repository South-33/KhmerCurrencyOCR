from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KNOWN_BUCKETS = {
    "target_modern_common",
    "target_modern_rare",
    "legacy_or_low_priority",
    "junk_or_unusable",
    "needs_visual_review",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit transparent asset-bank rows against the current KHR scope.")
    parser.add_argument("--manifest", default="data/asset_candidates/rare_pristine_asset_bank_v1/manifest.csv")
    parser.add_argument("--nbc-manifest", default="manifests/khr_nbc_curated_manifest.csv")
    parser.add_argument("--out", default="manifests/rare_pristine_asset_bank_v1_scope_audit.csv")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def nbc_scope_by_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    scoped: dict[str, dict[str, str]] = {}
    for row in rows:
        source_name = Path(row["source"]).name
        match = re.match(r"(\d{3})_", source_name)
        if match:
            scoped[match.group(1)] = row
    return scoped


def classify_asset(asset_path: str, nbc_by_index: dict[str, dict[str, str]]) -> tuple[str, str]:
    name = Path(asset_path.replace("\\", "/")).name.lower()
    if "specimen" in name or "watermark" in name:
        return "junk_or_unusable", "filename indicates specimen or watermark"

    nbc_match = re.search(r"nbc_reference_(\d{3})_", name)
    if nbc_match:
        nbc_row = nbc_by_index.get(nbc_match.group(1))
        if nbc_row:
            return nbc_row["bucket"], f"NBC scoped source: {Path(nbc_row['source']).name}"
        return "needs_visual_review", "NBC reference index not found in curated manifest"

    if "numista_reference" in name:
        if "2017_issued_2018" in name:
            return "target_modern_common", "Numista filename matches current 20,000 riel issue"
        if any(year in name for year in ["1995", "1998", "2001", "2008"]):
            return "legacy_or_low_priority", "Numista filename indicates older issue"
        return "needs_visual_review", "Numista reference without recognized issue year"

    if "scene_crop" in name:
        return "needs_visual_review", "scene crop needs design/version visual audit"

    return "needs_visual_review", "unrecognized asset source"


def main() -> None:
    args = parse_args()
    manifest_path = (ROOT / args.manifest).resolve()
    nbc_manifest_path = (ROOT / args.nbc_manifest).resolve()
    out_path = (ROOT / args.out).resolve()

    nbc_rows = read_csv(nbc_manifest_path)
    nbc_by_index = nbc_scope_by_index(nbc_rows)
    rows = read_csv(manifest_path)

    audited: list[dict[str, str]] = []
    for row in rows:
        asset_path = row.get("asset_path") or row.get("path") or ""
        bucket, reason = classify_asset(asset_path, nbc_by_index)
        if bucket not in KNOWN_BUCKETS:
            raise ValueError(f"Unexpected scope bucket: {bucket}")
        audited.append({**row, "scope_bucket": bucket, "scope_reason": reason})

    write_csv(out_path, audited)
    print(f"Wrote {len(audited)} audited rows to {out_path}")
    for bucket in sorted(KNOWN_BUCKETS):
        print(f"{bucket}: {sum(1 for row in audited if row['scope_bucket'] == bucket)}")


if __name__ == "__main__":
    main()
