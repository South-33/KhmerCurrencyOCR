from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image

from local_runtime import ROOT, configure_project_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded CUDA-only SD-Turbo img2img refiner smoke on synthetic "
            "source images. Outputs must still pass protected composition and "
            "label-preservation gates before use."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--model", default="stabilityai/sd-turbo")
    parser.add_argument(
        "--prompt",
        default="realistic phone photo of banknotes on a retail counter, natural store lighting, realistic camera noise",
    )
    parser.add_argument(
        "--negative-prompt",
        default="cartoon, illustration, fake texture, melted details, text artifacts, watermark, low quality",
    )
    parser.add_argument("--strength", type=float, default=0.18)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument(
        "--resolution",
        type=int,
        default=384,
        help="Resize the refiner input so the long side is at most this many pixels; <=0 keeps source size.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing manifest: {repo_rel(path)}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{repo_rel(path)}:{line_no}: invalid JSON: {exc}") from exc
    if not rows:
        raise SystemExit(f"empty manifest: {repo_rel(path)}")
    return rows


def load_pipeline(args: argparse.Namespace) -> AutoPipelineForImage2Image:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; refusing CPU fallback")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("--device must be a CUDA device; refusing CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = True
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def resize_for_refiner(image: Image.Image, resolution: int) -> tuple[Image.Image, bool]:
    if resolution <= 0:
        return image, False
    width, height = image.size
    long_side = max(width, height)
    if long_side <= resolution:
        return image, False
    scale = resolution / float(long_side)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS), True


def main() -> int:
    configure_project_cache()
    args = parse_args()
    if args.strength <= 0 or args.strength > 1:
        raise SystemExit("--strength must be in (0, 1]")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if int(args.steps * args.strength) < 1:
        raise SystemExit("--strength * --steps is too low for at least one img2img denoise step")
    manifest = resolve_repo_path(args.manifest)
    out_root = resolve_repo_path(args.out_root)
    if args.clean and out_root.exists():
        import shutil

        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(manifest)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]

    pipe = load_pipeline(args)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    outputs: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        source_path = resolve_repo_path(Path(str(row["source_image"])))
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        refiner_input, resized_for_refiner = resize_for_refiner(source, args.resolution)
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=refiner_input,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        )
        refined = result.images[0].convert("RGB")
        resized = False
        if refined.size != source.size:
            refined = refined.resize(source.size, Image.Resampling.LANCZOS)
            resized = True
        out_path = out_root / f"{source_path.stem}.png"
        refined.save(out_path)
        outputs.append(
            {
                "index": idx,
                "id": row.get("id", source_path.stem),
                "class_name": row.get("class_name", ""),
                "source_image": repo_rel(source_path),
                "output_image": repo_rel(out_path),
                "source_size": list(source.size),
                "refiner_input_size": list(refiner_input.size),
                "output_size": list(refined.size),
                "resized_for_refiner": resized_for_refiner,
                "resized_to_source": resized,
            }
        )

    summary = {
        "schema": "cashsnap_sd_turbo_img2img_refiner_v1",
        "manifest": repo_rel(manifest),
        "out_root": repo_rel(out_root),
        "model": args.model,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "strength": args.strength,
        "steps": args.steps,
        "resolution": args.resolution,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "device": args.device,
        "rows": len(outputs),
        "outputs": outputs,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_root": repo_rel(out_root), "rows": len(outputs), "summary": repo_rel(out_root / "summary.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
