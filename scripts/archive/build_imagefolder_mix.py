from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mixed ImageFolder dataset from base plus train-only extras.")
    parser.add_argument("--base", required=True, help="Base ImageFolder dataset with train/val/test splits.")
    parser.add_argument(
        "--train-extra",
        nargs="*",
        default=[],
        help="Extra ImageFolder dataset(s) copied into output train.",
    )
    parser.add_argument(
        "--train-extra-splits",
        default="train",
        help="Comma-separated split names to read from each train-extra dataset and copy into output train.",
    )
    parser.add_argument("--out", required=True, help="Output ImageFolder dataset under data/.")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def split_list(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,\s]+", value) if item.strip()]


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


def copy_split(
    source_root: Path,
    out_dir: Path,
    split: str,
    prefix: str,
    rows: list[dict[str, str]],
    target_split: str | None = None,
) -> int:
    output_split = target_split or split
    copied = 0
    for class_dir in class_dirs(source_root, split):
        class_name = class_dir.name
        target_dir = out_dir / output_split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(class_dir.iterdir()):
            if source.suffix.lower() not in IMAGE_SUFFIXES or not source.is_file():
                continue
            target = target_dir / f"{prefix}_{source.name}"
            shutil.copy2(source, target)
            rows.append(
                {
                    "split": output_split,
                    "class_name": class_name,
                    "source_dataset": repo_rel(source_root),
                    "source_split": split,
                    "source_path": repo_rel(source),
                    "image_path": repo_rel(target),
                }
            )
            copied += 1
    return copied


def dataset_classes(roots: list[Path]) -> list[str]:
    return sorted({path.name for root in roots for split in ["train", "val", "test"] for path in class_dirs(root, split)})


def ensure_classes(roots: list[Path], out_dir: Path) -> None:
    classes = dataset_classes(roots)
    for split in ["train", "val", "test"]:
        for class_name in classes:
            (out_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    base = resolve(args.base)
    extras = [resolve(path) for path in args.train_extra]
    extra_splits = split_list(args.train_extra_splits)
    if not extra_splits:
        raise SystemExit("--train-extra-splits must include at least one split")
    out_dir = resolve(args.out)
    if not base.exists():
        raise SystemExit(f"Base dataset does not exist: {base}")
    missing = [path for path in extras if not path.exists()]
    if missing:
        raise SystemExit(f"Extra dataset does not exist: {missing[0]}")
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_classes([base, *extras], out_dir)

    rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        counts[f"base_{split}"] = copy_split(base, out_dir, split, "base", rows)
    for index, extra in enumerate(extras):
        for split in extra_splits:
            counts[f"extra{index}_{split}_to_train"] = copy_split(
                extra,
                out_dir,
                split,
                f"extra{index}_{split}",
                rows,
                "train",
            )

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "class_name", "source_dataset", "source_split", "source_path", "image_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    class_counts = Counter(f"{row['split']}/{row['class_name']}" for row in rows)
    summary = {
        "schema": "cashsnap_imagefolder_mix_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base": repo_rel(base),
        "train_extra": [repo_rel(path) for path in extras],
        "train_extra_splits": extra_splits,
        "out": repo_rel(out_dir),
        "rows": len(rows),
        "counts": counts,
        "class_counts": dict(sorted(class_counts.items())),
        "note": "Base ImageFolder splits are copied as-is; extra datasets are copied into the train split only.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {len(rows)} images to {repo_rel(out_dir)}")
    for key, count in counts.items():
        print(f"{key}: {count}")


if __name__ == "__main__":
    main()
