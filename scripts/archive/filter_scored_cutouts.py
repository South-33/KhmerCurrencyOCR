from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter scored transparent cutouts into a cleaner asset bank.")
    parser.add_argument("--scores", required=True, help="Path to cutout_scores.csv.")
    parser.add_argument("--out", required=True, help="Output asset-bank folder.")
    parser.add_argument("--verdict", default="gold", help="Required verdict.")
    parser.add_argument("--min-fill", type=float, default=0.70, help="Minimum bbox fill ratio.")
    parser.add_argument("--min-aspect-norm", type=float, default=1.55, help="Minimum long-side/short-side bbox aspect ratio.")
    parser.add_argument("--min-largest-component", type=float, default=0.94, help="Minimum largest component ratio.")
    parser.add_argument("--max-small-components", type=int, default=0, help="Maximum allowed small components.")
    return parser.parse_args()


def keep(row: dict[str, str], args: argparse.Namespace) -> bool:
    return (
        row.get("verdict") == args.verdict
        and float(row.get("bbox_fill_ratio", "0") or 0) >= args.min_fill
        and max(
            float(row.get("bbox_aspect", "0") or 0),
            1 / max(float(row.get("bbox_aspect", "0") or 0), 1e-6),
        )
        >= args.min_aspect_norm
        and float(row.get("largest_component_ratio", "0") or 0) >= args.min_largest_component
        and int(row.get("small_component_count", "99") or 99) <= args.max_small_components
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scores = (ROOT / args.scores).resolve()
    out_dir = (ROOT / args.out).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with scores.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if keep(row, args)]

    for row in selected:
        source = ROOT / row["path"]
        class_name = "KHR_20000" if "KHR_20000" in source.name else "KHR_50000" if "KHR_50000" in source.name else "unknown"
        target = out_dir / class_name / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        row["asset_path"] = str(target.relative_to(ROOT))

    write_csv(out_dir / "manifest.csv", selected)
    print(f"Selected {len(selected)} cutouts into {out_dir}")
    for class_name in ["KHR_20000", "KHR_50000", "unknown"]:
        print(f"{class_name}: {sum(1 for row in selected if class_name in row.get('asset_path', ''))}")


if __name__ == "__main__":
    main()
