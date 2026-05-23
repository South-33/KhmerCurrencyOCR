from __future__ import annotations

import argparse
import re
from collections import deque
from pathlib import Path

import numpy as np
import torch
from ben2 import BEN_Base
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BEN2 background removal over PicWish-style batch folders.")
    parser.add_argument("--inputs", default="data/picwish_upload_batches", help="Folder containing batch_### input folders.")
    parser.add_argument("--out", default="data/asset_candidates/ben2_output", help="Output folder for transparent PNGs.")
    parser.add_argument("--only-stems-file", help="Optional text file of input stems to process, one per line.")
    parser.add_argument("--refine-foreground", action="store_true", help="Enable BEN2 foreground refinement.")
    parser.add_argument("--dilate", type=int, default=1, help="Alpha max-filter radius after inference; 0 disables dilation.")
    parser.add_argument("--no-fill-holes", action="store_true", help="Disable alpha hole filling.")
    return parser.parse_args()


def read_stems(path: str | None) -> set[str] | None:
    if not path:
        return None
    stems_path = (ROOT / path).resolve()
    if not stems_path.exists():
        raise SystemExit(f"Missing stems file: {stems_path}")
    stems = {
        line.strip()
        for line in stems_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not stems:
        raise SystemExit(f"No stems found in {stems_path}")
    return stems


def gather_inputs(inputs_dir: Path, only_stems: set[str] | None) -> list[Path]:
    paths: list[Path] = []
    for batch_path in sorted(inputs_dir.glob("batch_*")):
        if not batch_path.is_dir() or not re.fullmatch(r"batch_\d{3}", batch_path.name):
            continue
        for path in sorted(batch_path.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if only_stems and path.stem not in only_stems:
                continue
            paths.append(path)
    return paths


def fill_alpha_holes(alpha: Image.Image) -> Image.Image:
    mask = np.array(alpha) > 16
    height, width = mask.shape
    outside = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for x in range(width):
        for y in (0, height - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and not mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                queue.append((ny, nx))

    filled = mask | (~mask & ~outside)
    return Image.fromarray((filled.astype(np.uint8) * 255), mode="L")


def postprocess_alpha(image: Image.Image, fill_holes: bool, dilate: int) -> Image.Image:
    image = image.convert("RGBA")
    alpha = image.getchannel("A")
    if fill_holes:
        alpha = fill_alpha_holes(alpha)
    if dilate > 0:
        alpha = alpha.filter(ImageFilter.MaxFilter(dilate * 2 + 1))
    image.putalpha(alpha)
    return image


def main() -> None:
    args = parse_args()
    inputs_dir = (ROOT / args.inputs).resolve()
    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    only_stems = read_stems(args.only_stems_file)
    inputs = gather_inputs(inputs_dir, only_stems)

    to_process = [path for path in inputs if not (out_dir / f"{path.stem}.png").exists()]
    print(f"Found {len(inputs)} BEN2 inputs; {len(inputs) - len(to_process)} already processed; {len(to_process)} left.")
    if not to_process:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading BEN2 on {device}...")
    model = BEN_Base.from_pretrained("PramaLLC/BEN2")
    model.to(device).eval()

    for index, path in enumerate(to_process, start=1):
        image = Image.open(path).convert("RGB")
        foreground = model.inference(image, refine_foreground=args.refine_foreground)
        foreground = postprocess_alpha(foreground, fill_holes=not args.no_fill_holes, dilate=args.dilate)
        target = out_dir / f"{path.stem}.png"
        foreground.save(target)
        print(f"[{index}/{len(to_process)}] saved {target.name}")


if __name__ == "__main__":
    main()
