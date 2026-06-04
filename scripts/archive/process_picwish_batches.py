from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

from picwish import PicWish


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process PicWish upload batches with bounded concurrency.")
    parser.add_argument("--inputs", default="data/picwish_upload_batches", help="Folder containing batch_### input folders.")
    parser.add_argument("--out", default="data/asset_candidates/picwish_output", help="Output folder for transparent PNGs.")
    parser.add_argument("--concurrency", type=int, default=10, help="Maximum active PicWish requests.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per image.")
    parser.add_argument("--sleep-duration", type=float, default=0.1, help="PicWish polling sleep duration.")
    parser.add_argument("--retry-after", type=float, default=0.2, help="PicWish retry-after duration.")
    parser.add_argument("--quiet-success", action="store_true", help="Do not print one success line per image.")
    return parser.parse_args()


async def process_image_with_retry(
    pw: PicWish,
    input_path: Path,
    output_path: Path,
    semaphore: asyncio.Semaphore,
    max_retries: int,
    quiet_success: bool,
) -> bool:
    """Process a single image with bounded active PicWish requests."""
    async with semaphore:
        delay = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                start_time = time.time()
                result = await pw.remove_background(str(input_path))
                await result.download(str(output_path))
                elapsed = time.time() - start_time
                if not quiet_success:
                    print(f"[SUCCESS] {input_path.name} processed in {elapsed:.2f}s")
                return True
            except Exception as e:
                if attempt == max_retries:
                    print(f"[ERROR] {input_path.name} failed after {max_retries} retries: {e}", file=sys.stderr)
                    return False
                sleep_time = delay * (2 ** (attempt - 1))
                print(
                    f"[RETRY] {input_path.name} failed: {e}. "
                    f"Retrying in {sleep_time:.1f}s ({attempt}/{max_retries})",
                    flush=True,
                )
                await asyncio.sleep(sleep_time)
        return False


def gather_inputs(batches_dir: Path, output_dir: Path) -> list[tuple[Path, Path]]:
    input_images: list[tuple[Path, Path]] = []
    for batch_path in sorted(batches_dir.glob("batch_*")):
        if not batch_path.is_dir() or not re.fullmatch(r"batch_\d{3}", batch_path.name):
            continue
        for file_path in sorted(batch_path.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES:
                input_images.append((file_path, output_dir / f"{file_path.stem}.png"))
    return input_images


async def main() -> None:
    args = parse_args()
    batches_dir = (ROOT / args.inputs).resolve()
    output_dir = (ROOT / args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")

    input_images = gather_inputs(batches_dir, output_dir)
    total_images = len(input_images)
    print(f"Found {total_images} total images in upload batches.")

    to_process = []
    for src, dst in input_images:
        if dst.exists() and dst.stat().st_size > 0:
            continue
        to_process.append((src, dst))

    already_done = total_images - len(to_process)
    print(f"Skipping {already_done} already processed images. {len(to_process)} left to process.")

    if not to_process:
        print("All images are already processed! Nothing to do.")
        return

    print(f"\nStarting background removal with {args.concurrency} concurrent workers...")
    print("--------------------------------------------------------------------------------")

    pw = PicWish(sleep_duration=args.sleep_duration, retry_after=args.retry_after)
    sem = asyncio.Semaphore(args.concurrency)

    start_time = time.time()
    tasks = [
        process_image_with_retry(pw, src, dst, sem, args.max_retries, args.quiet_success)
        for src, dst in to_process
    ]
    results = await asyncio.gather(*tasks)

    success_count = sum(1 for r in results if r)
    failure_count = len(results) - success_count
    total_elapsed = time.time() - start_time

    print("\n--------------------------------------------------------------------------------")
    print("Batch processing finished!")
    print(f"Total time elapsed: {total_elapsed:.2f} seconds")
    print(f"Successfully processed in this run: {success_count}")
    print(f"Failed to process: {failure_count}")
    print(f"Total files in output directory: {len(list(output_dir.glob('*.png')))}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess paused by user.")
