from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image
from local_runtime import configure_project_cache


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAMES = {
    0: "USD_1",
    1: "USD_5",
    2: "USD_10",
    3: "USD_20",
    4: "USD_50",
    5: "USD_100",
    6: "KHR_500",
    7: "KHR_1000",
    8: "KHR_2000",
    9: "KHR_5000",
    10: "KHR_10000",
    11: "KHR_20000",
    12: "KHR_50000",
}
REGIONS = {
    "full": (0.0, 0.0, 1.0, 1.0),
    "top": (0.0, 0.0, 1.0, 0.38),
    "middle": (0.0, 0.28, 1.0, 0.72),
    "bottom": (0.0, 0.62, 1.0, 1.0),
    "left": (0.0, 0.0, 0.42, 1.0),
    "right": (0.58, 0.0, 1.0, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Khmer OCR as a denomination cue on visible banknote boxes.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="YOLO detect labels for visible note boxes.")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--crop-dir", type=Path, default=None, help="Optional folder for saved OCR probe crops.")
    parser.add_argument("--pydeps", type=Path, default=ROOT / ".cache_runtime" / "pydeps")
    parser.add_argument("--padding", type=float, default=0.02, help="Box padding as a fraction of image max side.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def add_optional_pydeps(path: Path) -> None:
    resolved = resolve(path)
    if resolved.exists():
        sys.path.insert(0, str(resolved))


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}: line {line_number} has {len(parts)} fields; expected 5")
        class_id = int(parts[0])
        cx, cy, width, height = [float(value) for value in parts[1:]]
        rows.append((class_id, cx, cy, width, height))
    return rows


def crop_box(image: Image.Image, label: tuple[int, float, float, float, float], padding: float) -> Image.Image:
    _, cx, cy, width, height = label
    image_width, image_height = image.size
    pad = padding * max(image_width, image_height)
    x1 = max(0, round((cx - width / 2) * image_width - pad))
    y1 = max(0, round((cy - height / 2) * image_height - pad))
    x2 = min(image_width, round((cx + width / 2) * image_width + pad))
    y2 = min(image_height, round((cy + height / 2) * image_height + pad))
    return image.crop((x1, y1, x2, y2))


def region_crop(note_crop: Image.Image, region: tuple[float, float, float, float]) -> Image.Image:
    width, height = note_crop.size
    x1, y1, x2, y2 = region
    return note_crop.crop((round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)))


def main() -> None:
    configure_project_cache()
    args = parse_args()
    add_optional_pydeps(args.pydeps)
    try:
        from mer import Mer
    except ModuleNotFoundError as exc:
        install = (
            "python -m pip install --target .cache_runtime/pydeps mer==1.2.1 "
            "--no-warn-script-location"
        )
        raise SystemExit(f"Missing optional OCR dependency 'mer'. Install it repo-locally with: {install}") from exc

    image_path = resolve(args.image)
    label_path = resolve(args.labels)
    out_path = resolve(args.out)
    crop_dir = resolve(args.crop_dir) if args.crop_dir else None
    cache_dir = ROOT / ".cache_runtime" / "mer"

    ocr = Mer(cache_dir=cache_dir, device=args.device, postprocess=True, json_result=True)
    rows: list[dict[str, str]] = []
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    for box_index, label in enumerate(read_labels(label_path)):
        class_id = label[0]
        note_crop = crop_box(image, label, args.padding)
        for region_name, region in REGIONS.items():
            crop = region_crop(note_crop, region)
            crop_path = ""
            if crop_dir is not None:
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop_path_obj = crop_dir / f"{image_path.stem}_box{box_index:02d}_{region_name}.jpg"
                crop.save(crop_path_obj, quality=92)
                crop_path = crop_path_obj.relative_to(ROOT).as_posix()
            try:
                result = ocr.recognize_line(crop, json_result=True)
                text = str(result.get("text", "")).strip()
            except Exception as exc:
                text = f"ERROR: {exc}"
            rows.append(
                {
                    "image": image_path.relative_to(ROOT).as_posix(),
                    "labels": label_path.relative_to(ROOT).as_posix(),
                    "box_index": str(box_index),
                    "class_id": str(class_id),
                    "class_name": DEFAULT_NAMES.get(class_id, f"class_{class_id}"),
                    "region": region_name,
                    "crop_path": crop_path,
                    "ocr_text": text,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path.relative_to(ROOT)} rows={len(rows)}")


if __name__ == "__main__":
    main()
