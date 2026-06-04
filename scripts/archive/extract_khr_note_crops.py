from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import yaml


KHR_CLASSES = {
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_20000",
    "KHR_50000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract KHR note crops from a YOLO dataset.")
    parser.add_argument("--data", default="data/cashsnap_v1/data.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data/reference/khr_real_crops")
    parser.add_argument("--min-side", type=int, default=48)
    parser.add_argument("--pad", type=float, default=0.03)
    return parser.parse_args()


def load_dataset(data_path: Path) -> tuple[Path, dict[int, str], list[Path]]:
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    names_raw = data["names"]
    if isinstance(names_raw, dict):
        names = {int(k): str(v) for k, v in names_raw.items()}
    else:
        names = {i: str(v) for i, v in enumerate(names_raw)}
    split_value = data["train"]
    if isinstance(split_value, str):
        image_dirs = [root / split_value]
    else:
        image_dirs = [root / value for value in split_value]
    return root, names, image_dirs


def label_path_for(root: Path, image_path: Path) -> Path:
    parts = list(image_path.parts)
    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            break
    label_path = Path(*parts).with_suffix(".txt")
    if label_path.exists():
        return label_path
    return root / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    root, names, image_dirs = load_dataset(data_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    counts = {name: 0 for name in sorted(KHR_CLASSES)}
    for image_dir in image_dirs:
        for image_path in sorted(image_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            label_path = label_path_for(root, image_path)
            if not label_path.exists():
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except OSError:
                continue
            for line_i, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))
                class_name = names.get(class_id)
                if class_name not in KHR_CLASSES:
                    continue
                cx, cy, bw, bh = map(float, parts[1:5])
                w = bw * image.width
                h = bh * image.height
                if min(w, h) < args.min_side:
                    continue
                pad_x = w * args.pad
                pad_y = h * args.pad
                x1 = max(0, int((cx * image.width) - w / 2 - pad_x))
                y1 = max(0, int((cy * image.height) - h / 2 - pad_y))
                x2 = min(image.width, int((cx * image.width) + w / 2 + pad_x))
                y2 = min(image.height, int((cy * image.height) + h / 2 + pad_y))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image.crop((x1, y1, x2, y2))
                denom = class_name.removeprefix("KHR_")
                crop.save(out / f"{denom}_real_{written:06d}_{line_i}.jpg", quality=92)
                written += 1
                counts[class_name] += 1

    for class_name, count in counts.items():
        print(f"{class_name}: {count}")
    print(f"wrote {written} crops to {out}")


if __name__ == "__main__":
    main()
