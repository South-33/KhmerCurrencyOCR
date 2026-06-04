from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERDICT_RANK = {"gold": 0, "review": 1, "reject": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select the best transparent cutout per filename from scored runs.")
    parser.add_argument("--scores", nargs="+", required=True, help="cutout_scores.csv files to compare.")
    parser.add_argument("--out", default="data/asset_candidates/best_cutout_candidates", help="Selected cutout output folder.")
    return parser.parse_args()


def score_rank(row: dict[str, str], source_order: int) -> tuple[float, int]:
    verdict_rank = VERDICT_RANK.get(row.get("verdict", "reject"), 3)
    fill = float(row.get("bbox_fill_ratio", "0") or 0)
    largest = float(row.get("largest_component_ratio", "0") or 0)
    small_components = int(row.get("small_component_count", "99") or 99)
    # Lower is better. Favor verdict first, then rectangular connected masks.
    return (
        verdict_rank * 100
        - fill * 10
        - largest * 4
        + small_components * 2
        + source_order * 0.01,
        source_order,
    )


def read_rows(scores_path: Path, source_order: int) -> list[dict[str, str]]:
    with scores_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["source_scores"] = str(scores_path.relative_to(ROOT))
        row["source_order"] = str(source_order)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = (ROOT / args.out).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best: dict[str, dict[str, str]] = {}
    for source_order, score_arg in enumerate(args.scores):
        scores_path = (ROOT / score_arg).resolve()
        for row in read_rows(scores_path, source_order):
            name = Path(row["path"]).name
            if name not in best or score_rank(row, source_order) < score_rank(best[name], int(best[name]["source_order"])):
                best[name] = row

    selected_rows: list[dict[str, str]] = []
    for name, row in sorted(best.items()):
        source = ROOT / row["path"]
        verdict = row.get("verdict", "reject")
        target = out_dir / verdict / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        selected_rows.append({**row, "selected_path": str(target.relative_to(ROOT))})

    write_csv(out_dir / "selected_cutouts.csv", selected_rows)
    print(f"Selected {len(selected_rows)} cutouts into {out_dir}")
    for verdict in ["gold", "review", "reject"]:
        print(f"{verdict}: {sum(1 for row in selected_rows if row.get('verdict') == verdict)}")


if __name__ == "__main__":
    main()
