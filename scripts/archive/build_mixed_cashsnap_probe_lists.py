from __future__ import annotations

import argparse
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_OUT = ROOT / "data" / "sampled" / "cashsnap_current_thin_probe_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reproducible image lists for mixed CashSnap/synthetic probes.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--cashsnap-train-count", type=int, default=2400)
    parser.add_argument("--cashsnap-val-count", type=int, default=600)
    parser.add_argument("--cashsnap-background-count", type=int, default=160)
    parser.add_argument(
        "--synthetic-train-count",
        type=int,
        default=None,
        help="Optional cap for synthetic train images after roots are combined. Defaults to all.",
    )
    parser.add_argument(
        "--synthetic-val-count",
        type=int,
        default=None,
        help="Optional cap for synthetic val images after roots are combined. Defaults to all.",
    )
    parser.add_argument(
        "--extra-synthetic-root",
        action="append",
        type=Path,
        default=[],
        help="Extra YOLO dataset root to include; expects images/train and images/val.",
    )
    parser.add_argument("--no-default-synthetic", action="store_true", help="Only include roots passed with --extra-synthetic-root.")
    parser.add_argument("--clean", action="store_true", help="Replace existing list files.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def images_under(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(
        item.resolve()
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def sample(items: list[Path], count: int, rng: random.Random) -> list[Path]:
    if count >= len(items):
        return list(items)
    return sorted(rng.sample(items, count))


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        image_index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as exc:
        raise ValueError(f"image path does not contain an images directory: {image_path}") from exc
    parts[image_index] = "labels"
    return Path(*parts).with_suffix(".txt")


def class_ids_for_image(image_path: Path) -> set[int]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return set()
    class_ids: set[int] = set()
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        class_ids.add(int(line.split()[0]))
    return class_ids


def split_foreground_background(images: list[Path]) -> tuple[dict[int, list[Path]], list[Path]]:
    by_class: dict[int, list[Path]] = {class_id: [] for class_id in range(13)}
    backgrounds: list[Path] = []
    for image_path in images:
        class_ids = class_ids_for_image(image_path)
        if not class_ids:
            backgrounds.append(image_path)
            continue
        for class_id in class_ids:
            by_class.setdefault(class_id, []).append(image_path)
    return by_class, backgrounds


def balanced_foreground_sample(images: list[Path], count: int, rng: random.Random) -> list[Path]:
    by_class, _ = split_foreground_background(images)
    for class_images in by_class.values():
        rng.shuffle(class_images)
    selected: list[Path] = []
    selected_set: set[Path] = set()
    class_ids = list(sorted(by_class))
    while len(selected) < count:
        made_progress = False
        rng.shuffle(class_ids)
        for class_id in class_ids:
            while by_class[class_id]:
                candidate = by_class[class_id].pop()
                if candidate in selected_set:
                    continue
                selected.append(candidate)
                selected_set.add(candidate)
                made_progress = True
                break
            if len(selected) >= count:
                break
        if not made_progress:
            break
    return sorted(selected)


def write_list(path: Path, items: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{item.as_posix()}\n" for item in items),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out)
    if args.clean and out_dir.exists():
        for path in [out_dir / "train.txt", out_dir / "val.txt"]:
            if path.exists():
                path.unlink()
    rng = random.Random(args.seed)

    cashsnap_train_all = images_under(ROOT / "data" / "cashsnap_v1" / "images" / "train")
    cashsnap_val_all = images_under(ROOT / "data" / "cashsnap_v1" / "images" / "val")
    _, train_backgrounds = split_foreground_background(cashsnap_train_all)
    _, val_backgrounds = split_foreground_background(cashsnap_val_all)
    train_background_count = min(args.cashsnap_background_count, max(0, args.cashsnap_train_count // 5))
    val_background_count = min(args.cashsnap_background_count, max(0, args.cashsnap_val_count // 5))
    cashsnap_train = balanced_foreground_sample(
        cashsnap_train_all,
        max(0, args.cashsnap_train_count - train_background_count),
        rng,
    ) + sample(train_backgrounds, train_background_count, rng)
    cashsnap_val = balanced_foreground_sample(
        cashsnap_val_all,
        max(0, args.cashsnap_val_count - val_background_count),
        rng,
    ) + sample(val_backgrounds, val_background_count, rng)
    default_synthetic_roots = [] if args.no_default_synthetic else [
        ROOT / "data" / "synthetic" / "khr_current_clean_v1",
        ROOT / "data" / "synthetic" / "khr_current_thin_radial_slice_probe_v1",
    ]
    synthetic_roots = default_synthetic_roots + [resolve_path(path) for path in args.extra_synthetic_root]
    synthetic_train = [image for root in synthetic_roots for image in images_under(root / "images" / "train")]
    synthetic_val = [image for root in synthetic_roots for image in images_under(root / "images" / "val")]
    if args.synthetic_train_count is not None:
        synthetic_train = balanced_foreground_sample(synthetic_train, args.synthetic_train_count, rng)
    if args.synthetic_val_count is not None:
        synthetic_val = balanced_foreground_sample(synthetic_val, args.synthetic_val_count, rng)

    train = cashsnap_train + synthetic_train
    val = cashsnap_val + synthetic_val
    rng.shuffle(train)
    rng.shuffle(val)
    write_list(out_dir / "train.txt", train)
    write_list(out_dir / "val.txt", val)
    print(f"wrote {out_dir.relative_to(ROOT)}")
    train_fg = sum(1 for image_path in cashsnap_train if class_ids_for_image(image_path))
    val_fg = sum(1 for image_path in cashsnap_val if class_ids_for_image(image_path))
    print(
        f"train: cashsnap={len(cashsnap_train)} cashsnap_foreground={train_fg} "
        f"synthetic={len(synthetic_train)} total={len(train)}"
    )
    print(
        f"val: cashsnap={len(cashsnap_val)} cashsnap_foreground={val_fg} "
        f"synthetic={len(synthetic_val)} total={len(val)}"
    )


if __name__ == "__main__":
    main()
