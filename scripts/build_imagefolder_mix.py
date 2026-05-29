from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mixed ImageFolder dataset from base plus train-only extras.")
    parser.add_argument("--base", required=True, help="Base ImageFolder dataset with train/val/test splits.")
    parser.add_argument("--train-extra", nargs="*", default=[], help="Extra ImageFolder dataset(s); only train split is copied.")
    parser.add_argument("--out", required=True, help="Output ImageFolder dataset under data/.")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed_root = (ROOT / "data").resolve()
    if resolved == allowed_root or allowed_root not in resolved.parents:
        raise SystemExit(f"Refusing to clean outside {allowed_root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def class_dirs(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    if not split_dir.exists():
        return []
    return sorted(path for path in split_dir.iterdir() if path.is_dir())


def copy_split(source_root: Path, out_dir: Path, split: str, prefix: str, rows: list[dict[str, str]]) -> int:
    copied = 0
    for class_dir in class_dirs(source_root, split):
        class_name = class_dir.name
        target_dir = out_dir / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(class_dir.iterdir()):
            if source.suffix.lower() not in IMAGE_SUFFIXES or not source.is_file():
                continue
            target = target_dir / f"{prefix}_{source.name}"
            shutil.copy2(source, target)
            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "source_dataset": source_root.relative_to(ROOT).as_posix(),
                    "source_path": source.relative_to(ROOT).as_posix(),
                    "image_path": target.relative_to(ROOT).as_posix(),
                }
            )
            copied += 1
    return copied


def ensure_base_classes(base: Path, out_dir: Path) -> None:
    classes = sorted({path.name for split in ["train", "val", "test"] for path in class_dirs(base, split)})
    for split in ["train", "val", "test"]:
        for class_name in classes:
            (out_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    base = resolve(args.base)
    extras = [resolve(path) for path in args.train_extra]
    out_dir = resolve(args.out)
    if not base.exists():
        raise SystemExit(f"Base dataset does not exist: {base}")
    missing = [path for path in extras if not path.exists()]
    if missing:
        raise SystemExit(f"Extra dataset does not exist: {missing[0]}")
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_base_classes(base, out_dir)

    rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        counts[f"base_{split}"] = copy_split(base, out_dir, split, "base", rows)
    for index, extra in enumerate(extras):
        counts[f"extra{index}_train"] = copy_split(extra, out_dir, "train", f"extra{index}", rows)

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class_name", "source_dataset", "source_path", "image_path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} images to {out_dir.relative_to(ROOT)}")
    for key, count in counts.items():
        print(f"{key}: {count}")


if __name__ == "__main__":
    main()
