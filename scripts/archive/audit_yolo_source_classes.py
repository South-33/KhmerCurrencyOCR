from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count YOLO labels by source split and class.")
    parser.add_argument("--root", required=True, help="Dataset root or parent containing data.yaml files.")
    parser.add_argument("--recursive", action="store_true", help="Audit every nested data.yaml below root.")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def load_names(data_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    raw_names = data.get("names", {})
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    return {index: str(value) for index, value in enumerate(raw_names)}


def label_files_for_split(dataset_root: Path, split: str) -> list[Path]:
    for label_dir in [dataset_root / "labels" / split, dataset_root / split / "labels"]:
        if label_dir.exists():
            return sorted(label_dir.glob("*.txt"))
    return []


def audit_dataset(data_yaml: Path) -> None:
    dataset_root = data_yaml.parent
    names = load_names(data_yaml)
    print(f"\n{dataset_root.relative_to(ROOT)}")
    total = Counter()
    for split in ["train", "valid", "val", "test"]:
        label_files = label_files_for_split(dataset_root, split)
        if not label_files:
            continue
        counts = Counter()
        for label_file in label_files:
            for raw_line in label_file.read_text(encoding="utf-8").splitlines():
                parts = raw_line.split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                counts[names.get(class_id, str(class_id))] += 1
        total.update(counts)
        summary = "; ".join(f"{name}:{count}" for name, count in sorted(counts.items()))
        print(f"  {split}: files={len(label_files)} boxes={sum(counts.values())} {summary}")
    summary = "; ".join(f"{name}:{count}" for name, count in sorted(total.items()))
    print(f"  total: boxes={sum(total.values())} {summary}")


def main() -> None:
    args = parse_args()
    root = resolve(args.root)
    data_yamls = sorted(root.rglob("data.yaml")) if args.recursive else [root / "data.yaml"]
    for data_yaml in data_yamls:
        if data_yaml.exists():
            audit_dataset(data_yaml)


if __name__ == "__main__":
    main()
