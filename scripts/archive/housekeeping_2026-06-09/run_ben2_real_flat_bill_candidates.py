#!/usr/bin/env python
"""Run BEN2 background removal over real flat-bill cutout candidates."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from local_runtime import configure_project_cache  # noqa: E402

configure_project_cache()

import torch  # noqa: E402
from ben2 import BEN_Base  # noqa: E402


DEFAULT_OUT = ROOT / "data" / "asset_candidates" / "cashsnap_real_flat_bill_cutout_bank_probe_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--candidate-manifest", type=Path, default=None)
    parser.add_argument("--transparent-root", type=Path, default=None)
    parser.add_argument("--only-stems-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--refine-foreground", action="store_true")
    parser.add_argument("--dilate", type=int, default=1, help="Alpha max-filter radius after inference; 0 disables dilation.")
    parser.add_argument("--no-fill-holes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_stems(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    path = resolve(path)
    if not path.exists():
        raise SystemExit(f"missing stems file: {repo_rel(path)}")
    stems = {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not stems:
        raise SystemExit(f"no stems found in {repo_rel(path)}")
    return stems


def read_candidates(path: Path, only_stems: set[str] | None) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing candidate manifest: {repo_rel(path)}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        stem = row.get("candidate_id", "").strip()
        crop_path = resolve(Path(row.get("crop_path", "")))
        if only_stems is not None and stem not in only_stems:
            continue
        if not stem:
            raise SystemExit(f"candidate row missing candidate_id: {row}")
        if not crop_path.exists():
            raise SystemExit(f"{stem} missing crop_path: {repo_rel(crop_path)}")
        selected.append(row)
    return selected


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


def postprocess_alpha(image: Image.Image, *, fill_holes: bool, dilate: int) -> Image.Image:
    image = image.convert("RGBA")
    alpha = image.getchannel("A")
    if fill_holes:
        alpha = fill_alpha_holes(alpha)
    if dilate > 0:
        alpha = alpha.filter(ImageFilter.MaxFilter(dilate * 2 + 1))
    image.putalpha(alpha)
    return image


def choose_device(value: str) -> torch.device:
    if value == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> int:
    args = parse_args()
    out_root = resolve(args.out_root)
    candidate_manifest = resolve(args.candidate_manifest) if args.candidate_manifest else out_root / "candidates" / "candidate_manifest.csv"
    transparent_root = resolve(args.transparent_root) if args.transparent_root else out_root / "ben2_output"
    transparent_root.mkdir(parents=True, exist_ok=True)
    only_stems = read_stems(args.only_stems_file)
    candidates = read_candidates(candidate_manifest, only_stems)

    pending: list[dict[str, str]] = []
    for row in candidates:
        target = transparent_root / f"{row['candidate_id']}.png"
        if target.exists() and not args.overwrite:
            continue
        pending.append(row)
    if args.limit > 0:
        pending = pending[: args.limit]

    print(
        f"candidates={len(candidates)} pending={len(pending)} "
        f"transparent_root={repo_rel(transparent_root)}"
    )
    if not pending:
        return 0

    device = choose_device(args.device)
    print(f"loading BEN2 on {device}...")
    model = BEN_Base.from_pretrained("PramaLLC/BEN2")
    model.to(device).eval()

    for index, row in enumerate(pending, start=1):
        crop_path = resolve(Path(row["crop_path"]))
        with Image.open(crop_path).convert("RGB") as image:
            foreground = model.inference(image, refine_foreground=args.refine_foreground)
        foreground = postprocess_alpha(
            foreground,
            fill_holes=not args.no_fill_holes,
            dilate=args.dilate,
        )
        target = transparent_root / f"{row['candidate_id']}.png"
        foreground.save(target)
        print(f"[{index}/{len(pending)}] {row['candidate_id']} -> {repo_rel(target)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
