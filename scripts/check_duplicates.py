"""
Duplicate image checker for CashSnap datasets.

Two-pass approach:
  1. Exact duplicates  - SHA-256 hash of raw file bytes (catches identical files)
  2. Near-duplicates   - perceptual hash (pHash, 8x8 DCT) with Hamming distance <= threshold
                         (catches resized, recompressed, or slightly-cropped copies)

Usage:
    python scripts/check_duplicates.py                  # default threshold 8
    python scripts/check_duplicates.py --threshold 4    # stricter near-dup detection
    python scripts/check_duplicates.py --exact-only     # skip perceptual hashing (faster)
    python scripts/check_duplicates.py --across-only    # only report cross-dataset dups

Output:
    data/dedup/duplicate_report.json   - full machine-readable report
    data/dedup/exact_duplicates.csv    - exact dup pairs
    data/dedup/near_duplicates.csv     - near-dup pairs (cross-dataset only by default)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Datasets to check - add/remove as needed
DATASET_DIRS = {
    "hf_usd_side":               DATA / "raw_datasets" / "hf_usd_side_coco_annotations",
    "khmer_us_currency":         DATA / "raw_datasets" / "roboflow_khmer_us_currency",
    "cambodia_currency_project": DATA / "raw_datasets" / "roboflow_cambodia_currency_project",
    "khmer_scan":                DATA / "raw_datasets" / "roboflow_khmer_scan",
    "cashcountingxl":            DATA / "raw_datasets" / "roboflow_cashcountingxl",
    "usd_total":                 DATA / "raw_datasets" / "roboflow_usd_total",
    "billsbank":                 DATA / "raw_datasets" / "roboflow_billsbank",
    "asian_currency_detection":  DATA / "raw_datasets" / "roboflow_asian_currency_detection",
    "cuurecy_detection_is":      DATA / "raw_datasets" / "roboflow_cuurecy_detection_is",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def phash(path: Path) -> int | None:
    """Return 64-bit perceptual hash as int, or None on failure."""
    try:
        from PIL import Image
        import struct, math

        img = Image.open(path).convert("L").resize((32, 32), Image.LANCZOS)
        pixels = list(img.getdata())
        # DCT-based perceptual hash (8x8 from 32x32 DCT)
        size = 32
        dct_size = 8
        # compute 2D DCT manually (small size so ok)
        dct = [[0.0] * size for _ in range(size)]
        for u in range(size):
            for v in range(size):
                val = 0.0
                for x in range(size):
                    for y in range(size):
                        val += (pixels[x * size + y]
                                * math.cos((2 * x + 1) * u * math.pi / (2 * size))
                                * math.cos((2 * y + 1) * v * math.pi / (2 * size)))
                cu = (1 / math.sqrt(2)) if u == 0 else 1.0
                cv = (1 / math.sqrt(2)) if v == 0 else 1.0
                dct[u][v] = (2 / size) * cu * cv * val

        # Take top-left 8x8 (excluding [0,0] DC component)
        dct_low = [dct[u][v] for u in range(dct_size) for v in range(dct_size)]
        avg = (sum(dct_low) - dct_low[0]) / (dct_size * dct_size - 1)
        bits = [(1 if d > avg else 0) for d in dct_low]
        result = 0
        for b in bits:
            result = (result << 1) | b
        return result
    except Exception:
        return None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def collect_images(dataset_name: str, dataset_dir: Path) -> list[tuple[str, Path]]:
    """Return list of (dataset_name, image_path) for all images in dataset."""
    if not dataset_dir.exists():
        return []
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(dataset_dir.rglob(f"*{ext}"))
        imgs.extend(dataset_dir.rglob(f"*{ext.upper()}"))
    return [(dataset_name, p) for p in imgs]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for duplicate images across CashSnap datasets.")
    parser.add_argument("--threshold", type=int, default=8,
                        help="Hamming distance threshold for near-duplicates (default 8, max 64)")
    parser.add_argument("--exact-only", action="store_true", help="Skip perceptual hashing")
    parser.add_argument("--across-only", action="store_true",
                        help="Only report duplicates that span different datasets")
    args = parser.parse_args()

    out_dir = DATA / "dedup"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Collect all images -----------------------------------------------------
    print("Collecting images...")
    all_images: list[tuple[str, Path]] = []
    dataset_counts: dict[str, int] = {}
    for name, dirpath in DATASET_DIRS.items():
        imgs = collect_images(name, dirpath)
        dataset_counts[name] = len(imgs)
        all_images.extend(imgs)
        status = f"{len(imgs)} images" if len(imgs) > 0 else "NOT FOUND"
        print(f"  {name}: {status}")
    print(f"\nTotal: {len(all_images)} images across {len(DATASET_DIRS)} datasets\n")

    # -- Pass 1: Exact duplicates -----------------------------------------------
    print("Pass 1: Exact duplicates (SHA-256)...")
    hash_to_images: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for i, (ds, path) in enumerate(all_images):
        if i % 500 == 0:
            print(f"  Hashing {i}/{len(all_images)}...", end="\r")
        hash_to_images[sha256(path)].append((ds, path))

    exact_groups = {h: imgs for h, imgs in hash_to_images.items() if len(imgs) > 1}
    exact_rows = []
    exact_cross_count = 0
    for h, imgs in exact_groups.items():
        datasets_in_group = {ds for ds, _ in imgs}
        is_cross = len(datasets_in_group) > 1
        if args.across_only and not is_cross:
            continue
        if is_cross:
            exact_cross_count += 1
        for ds, path in imgs:
            exact_rows.append({
                "hash": h[:16] + "...",
                "dataset": ds,
                "file": str(path.relative_to(ROOT)),
                "group_size": len(imgs),
                "cross_dataset": is_cross,
                "datasets_in_group": ",".join(sorted(datasets_in_group)),
            })

    print(f"\n  Exact duplicate groups:        {len(exact_groups)}")
    print(f"  Cross-dataset exact dup groups: {exact_cross_count}")
    write_csv(out_dir / "exact_duplicates.csv", exact_rows,
              ["hash", "dataset", "file", "group_size", "cross_dataset", "datasets_in_group"])

    # -- Pass 2: Near-duplicates (perceptual hash) ------------------------------
    near_rows = []
    near_cross_count = 0

    if not args.exact_only:
        print(f"\nPass 2: Near-duplicates (pHash, threshold={args.threshold})...")
        print("  Computing perceptual hashes (this may take a few minutes)...")

        phashes: list[tuple[str, Path, int]] = []
        for i, (ds, path) in enumerate(all_images):
            if i % 200 == 0:
                print(f"  pHashing {i}/{len(all_images)}...", end="\r")
            ph = phash(path)
            if ph is not None:
                phashes.append((ds, path, ph))

        print(f"\n  Computed {len(phashes)} perceptual hashes")
        print(f"  Comparing pairs (n={len(phashes)}, this is O(n^2) - may be slow for large n)...")

        # For large n, only compare across datasets to keep it tractable
        near_pairs: list[tuple[str, Path, str, Path, int]] = []
        n = len(phashes)
        for i in range(n):
            if i % 100 == 0:
                print(f"  Comparing {i}/{n}...", end="\r")
            ds_a, path_a, ph_a = phashes[i]
            for j in range(i + 1, n):
                ds_b, path_b, ph_b = phashes[j]
                # Skip within-dataset pairs if across_only
                if args.across_only and ds_a == ds_b:
                    continue
                dist = hamming(ph_a, ph_b)
                if dist <= args.threshold:
                    # Skip exact duplicates already caught in pass 1
                    if sha256(path_a) == sha256(path_b):
                        continue
                    near_pairs.append((ds_a, path_a, ds_b, path_b, dist))

        for ds_a, path_a, ds_b, path_b, dist in near_pairs:
            is_cross = ds_a != ds_b
            if is_cross:
                near_cross_count += 1
            near_rows.append({
                "dataset_a": ds_a,
                "file_a": str(path_a.relative_to(ROOT)),
                "dataset_b": ds_b,
                "file_b": str(path_b.relative_to(ROOT)),
                "hamming_distance": dist,
                "cross_dataset": is_cross,
            })

        print(f"\n  Near-duplicate pairs:           {len(near_pairs)}")
        print(f"  Cross-dataset near-dup pairs:   {near_cross_count}")
        write_csv(out_dir / "near_duplicates.csv", near_rows,
                  ["dataset_a", "file_a", "dataset_b", "file_b", "hamming_distance", "cross_dataset"])
    else:
        print("\nSkipped perceptual hashing (--exact-only)")

    # -- Summary report ---------------------------------------------------------
    report = {
        "dataset_image_counts": dataset_counts,
        "total_images": len(all_images),
        "exact_duplicate_groups": len(exact_groups),
        "exact_cross_dataset_groups": exact_cross_count,
        "near_duplicate_threshold": args.threshold,
        "near_duplicate_pairs": len(near_rows),
        "near_cross_dataset_pairs": near_cross_count,
        "outputs": {
            "exact_duplicates_csv": str((out_dir / "exact_duplicates.csv").relative_to(ROOT)),
            "near_duplicates_csv": str((out_dir / "near_duplicates.csv").relative_to(ROOT)),
        },
    }
    (out_dir / "duplicate_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== DUPLICATE CHECK SUMMARY ===")
    print(f"Total images checked:           {len(all_images)}")
    print(f"Exact dup groups (cross-ds):    {exact_cross_count}")
    print(f"Near-dup pairs (cross-ds):      {near_cross_count}")
    print(f"\nReports saved to: {out_dir}")

    if exact_cross_count > 0 or near_cross_count > 0:
        print("\nACTION: Review cross-dataset duplicates and exclude them from the merge.")
        print("        Update REMAP in prepare_cashsnap_dataset.py to skip dup files,")
        print("        or deduplicate at the source level before merging.")
    else:
        print("\nNo cross-dataset duplicates found. Safe to merge all sources.")


if __name__ == "__main__":
    main()
