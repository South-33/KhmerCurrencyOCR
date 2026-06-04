from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map denomination ImageFolder crops into KHR/USD/background classes.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["hardlink", "copy"], default="hardlink")
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


def gate_class(class_name: str) -> str:
    if class_name.startswith("KHR_"):
        return "KHR"
    if class_name.startswith("USD_"):
        return "USD"
    return "background"


def materialize(source: Path, target: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            target.hardlink_to(source)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    source = resolve(args.source)
    out_dir = resolve(args.out)
    if args.clean:
        safe_clean(out_dir)
    rows: list[dict[str, str]] = []
    counters: dict[tuple[str, str], int] = {}
    for split_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        split = split_dir.name
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            target_class = gate_class(class_dir.name)
            key = (split, target_class)
            for image_path in sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
                index = counters.get(key, 0)
                counters[key] = index + 1
                target = out_dir / split / target_class / f"{target_class}_{index:06d}_{class_dir.name}_{image_path.name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                materialize(image_path, target, args.mode)
                rows.append(
                    {
                        "split": split,
                        "source_class": class_dir.name,
                        "gate_class": target_class,
                        "source_path": str(image_path.relative_to(ROOT)),
                        "image_path": str(target.relative_to(ROOT)),
                    }
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} gate crops to {out_dir.relative_to(ROOT)}")
    for (split, class_name), count in sorted(counters.items()):
        print(f"{split} {class_name}: {count}")


if __name__ == "__main__":
    main()
