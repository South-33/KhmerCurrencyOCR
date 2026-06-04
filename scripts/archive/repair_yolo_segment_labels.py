from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLO segmentation label rows to 5-field detection boxes.")
    parser.add_argument("--labels-root", default="data/cashsnap_v1/labels")
    return parser.parse_args()


def repair_file(path: Path) -> tuple[int, int]:
    changed = 0
    kept = 0
    repaired_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 5:
            repaired_lines.append(line)
            kept += 1
            continue
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            changed += 1
            continue
        cls = parts[0]
        try:
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            changed += 1
            continue
        xs = coords[0::2]
        ys = coords[1::2]
        x_min = max(0.0, min(xs))
        x_max = min(1.0, max(xs))
        y_min = max(0.0, min(ys))
        y_max = min(1.0, max(ys))
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            changed += 1
            continue
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        repaired_lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        changed += 1
    if changed:
        path.write_text("\n".join(repaired_lines) + ("\n" if repaired_lines else ""), encoding="utf-8")
    return changed, kept


def main() -> None:
    args = parse_args()
    labels_root = Path(args.labels_root)
    changed_files = 0
    changed_rows = 0
    kept_rows = 0
    for path in labels_root.rglob("*.txt"):
        changed, kept = repair_file(path)
        kept_rows += kept
        changed_rows += changed
        if changed:
            changed_files += 1
    print(f"changed_files={changed_files}")
    print(f"changed_or_dropped_rows={changed_rows}")
    print(f"kept_rows={kept_rows}")


if __name__ == "__main__":
    main()
