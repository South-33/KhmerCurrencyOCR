#!/usr/bin/env python
"""Build train-only ImageFolder reject crops from reviewed unknown/out-of-scope images."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_runtime import configure_project_cache

configure_project_cache()

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT_CLASSES = ("target", "reject")
DEFAULT_VARIANTS = ("full", "center80", "left65", "right65", "top65", "bottom65")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-list", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--label", default="reject", choices=OUT_CLASSES)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--max-images", type=int, default=0, help="Use at most this many images; 0 means all.")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed = (ROOT / "data" / "proposal_gate").resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise SystemExit(f"Refusing to clean outside {allowed}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def read_image_list(path: Path) -> list[Path]:
    images: list[Path] = []
    for raw_line in resolve(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image = Path(line)
        images.append(image if image.is_absolute() else ROOT / image)
    return images


def variant_box(name: str, width: int, height: int) -> tuple[int, int, int, int]:
    if name == "full":
        return (0, 0, width, height)
    if name == "center80":
        crop_w = int(width * 0.80)
        crop_h = int(height * 0.80)
        x1 = (width - crop_w) // 2
        y1 = (height - crop_h) // 2
        return (x1, y1, x1 + crop_w, y1 + crop_h)
    if name == "left65":
        return (0, 0, int(width * 0.65), height)
    if name == "right65":
        return (int(width * 0.35), 0, width, height)
    if name == "top65":
        return (0, 0, width, int(height * 0.65))
    if name == "bottom65":
        return (0, int(height * 0.35), width, height)
    raise SystemExit(f"unknown crop variant {name!r}; known={', '.join(DEFAULT_VARIANTS)}")


def main() -> int:
    args = parse_args()
    variants = [item.strip() for item in args.variants.replace(";", ",").split(",") if item.strip()]
    if not variants:
        raise SystemExit("--variants must include at least one crop variant")

    out_dir = resolve(args.out)
    if args.clean:
        safe_clean(out_dir)
    for split in ("train", "val", "test"):
        for label in OUT_CLASSES:
            (out_dir / split / label).mkdir(parents=True, exist_ok=True)

    images = read_image_list(args.image_list)
    if args.max_images > 0:
        images = images[: args.max_images]

    target_dir = out_dir / args.split / args.label
    manifest_rows: list[dict[str, Any]] = []
    skips: Counter[str] = Counter()
    crop_index = 0
    for image_path in images:
        if not image_path.exists():
            skips["missing_image"] += 1
            continue
        try:
            with Image.open(image_path) as raw_image:
                image = ImageOps.exif_transpose(raw_image).convert("RGB")
                width, height = image.size
                for variant in variants:
                    box = variant_box(variant, width, height)
                    crop = image.crop(box)
                    crop_path = target_dir / f"{image_path.stem}_{variant}_{crop_index:05d}.jpg"
                    crop.save(crop_path, quality=args.jpeg_quality)
                    manifest_rows.append(
                        {
                            "split": args.split,
                            "label": args.label,
                            "variant": variant,
                            "crop_path": repo_rel(crop_path),
                            "image_path": repo_rel(image_path),
                            "box_xyxy": ",".join(str(value) for value in box),
                            "width": width,
                            "height": height,
                        }
                    )
                    crop_index += 1
        except OSError:
            skips["image_open_error"] += 1

    manifest_path = out_dir / "manifest.csv"
    fieldnames = ["split", "label", "variant", "crop_path", "image_path", "box_xyxy", "width", "height"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "schema": "cashsnap_imagefolder_unknown_fullfrag_crops_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image_list": repo_rel(resolve(args.image_list)),
        "out": repo_rel(out_dir),
        "split": args.split,
        "label": args.label,
        "variants": variants,
        "images_requested": len(images),
        "images_used": len({row["image_path"] for row in manifest_rows}),
        "rows": len(manifest_rows),
        "counts": dict(Counter(f"{row['split']}/{row['label']}" for row in manifest_rows)),
        "skips": dict(sorted(skips.items())),
        "note": (
            f"{args.split}-split reject crops from reviewed current-schema unknown/out-of-scope images. "
            "Use as proposal-gate unknown supervision, not as YOLO detector background data."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest_rows)} crops to {repo_rel(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
