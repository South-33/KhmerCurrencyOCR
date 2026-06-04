from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else ROOT / path


def summarize(label_dir: Path) -> None:
    areas: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    boxes_per_image: list[int] = []
    for path in sorted(label_dir.glob("*.txt")):
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            width = float(parts[3])
            height = float(parts[4])
            widths.append(width)
            heights.append(height)
            areas.append(width * height * 100.0)
            count += 1
        boxes_per_image.append(count)

    print(label_dir)
    print(f"images: {len(boxes_per_image)}")
    print(f"boxes: {len(areas)}")
    if not areas:
        return
    area_arr = np.asarray(areas)
    width_arr = np.asarray(widths)
    height_arr = np.asarray(heights)
    count_arr = np.asarray(boxes_per_image)
    print(
        "boxes_per_image: "
        f"median={np.median(count_arr):.1f} p90={np.percentile(count_arr, 90):.1f}"
    )
    print(
        "box_area_pct: "
        f"p10={np.percentile(area_arr, 10):.3f} "
        f"median={np.median(area_arr):.3f} "
        f"p90={np.percentile(area_arr, 90):.3f} "
        f"small_under_1pct={(area_arr < 1.0).sum()}"
    )
    print(
        "box_shape: "
        f"width_median={np.median(width_arr):.3f} "
        f"height_median={np.median(height_arr):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize YOLO label geometry.")
    parser.add_argument("label_dir", nargs="+")
    args = parser.parse_args()

    for value in args.label_dir:
        summarize(resolve_path(value))


if __name__ == "__main__":
    main()
