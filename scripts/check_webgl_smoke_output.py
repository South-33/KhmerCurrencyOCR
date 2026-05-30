#!/usr/bin/env python
"""Validate the minimal WebGL renderer smoke output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/synthetic/cashsnap_webgl_smoke"),
        help="Renderer output directory to validate.",
    )
    parser.add_argument(
        "--min-visual-colors",
        type=int,
        default=1000,
        help="Minimum distinct RGB colors required in the visual render.",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    return Image.open(path).convert("RGB")


def read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_labels(path: Path) -> list[list[float]]:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")
    rows: list[list[float]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{path}:{line_no}: expected 5 YOLO columns, got {len(parts)}")
        try:
            class_index = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid numeric label: {line}") from exc
        if any(value < 0.0 or value > 1.0 for value in values):
            raise SystemExit(f"{path}:{line_no}: normalized coordinates out of range: {line}")
        rows.append([float(class_index), *values])
    return rows


def assert_close(left: float, right: float, label: str, tolerance: float = 1e-6) -> None:
    if abs(left - right) > tolerance:
        raise SystemExit(f"{label}: expected {right:.6f}, got {left:.6f}")


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir

    visual = load_rgb(out_dir / "visual.png")
    id_image = load_rgb(out_dir / "id.png")
    boxes_doc = read_json(out_dir / "visible_boxes.json")
    layer_audit = read_json(out_dir / "layer_audit.json")
    metadata = read_json(out_dir / "metadata.json")
    labels = read_labels(out_dir / "labels_visible.txt")

    visual_colors = visual.getcolors(maxcolors=10_000_000)
    if visual_colors is None or len(visual_colors) < args.min_visual_colors:
        count = 0 if visual_colors is None else len(visual_colors)
        raise SystemExit(f"visual render looks blank or posterized: {count} colors")

    boxes = boxes_doc.get("boxes", [])
    if not boxes:
        raise SystemExit("visible_boxes.json has no boxes")
    if len(labels) != len(boxes):
        raise SystemExit(f"label count mismatch: {len(labels)} labels for {len(boxes)} boxes")

    expected_colors = {tuple(asset["idColor"]) for asset in metadata.get("assets", [])}
    allowed_colors = expected_colors | {(0, 0, 0)}
    id_colors = id_image.getcolors(maxcolors=10_000_000)
    if id_colors is None:
        raise SystemExit("ID pass has too many colors to be an exact instance mask")
    actual_colors = {color for _, color in id_colors}
    unexpected_colors = actual_colors - allowed_colors
    if unexpected_colors:
        preview = sorted(unexpected_colors)[:8]
        raise SystemExit(f"ID pass has unexpected colors, likely antialiasing or tone mapping: {preview}")

    for box, label in zip(boxes, labels):
        class_index = int(box["classIndex"])
        if int(label[0]) != class_index:
            raise SystemExit(f"class mismatch for {box['className']}: label={int(label[0])} box={class_index}")
        if box["pixels"] <= 0:
            raise SystemExit(f"empty visible mask for {box['className']}")
        for idx, (got, expected) in enumerate(zip(label[1:], box["yolo"]), start=1):
            assert_close(got, float(expected), f"{box['className']} yolo[{idx}]")

    if layer_audit.get("violations") != 0:
        raise SystemExit(f"layer-order audit has violations: {layer_audit}")
    if layer_audit.get("overlapPixels", 0) <= 0:
        raise SystemExit("layer-order audit did not exercise any overlapping pixels")
    if layer_audit.get("occluderPixels", 0) <= 0:
        raise SystemExit("layer-order audit did not exercise any occluder pixels")

    print(
        f"ok: {visual.size[0]}x{visual.size[1]} visual, "
        f"{len(actual_colors)} exact ID colors, {len(boxes)} boxes, "
        f"{layer_audit['overlapPixels']} audited overlap pixels, "
        f"{layer_audit['occluderPixels']} occluder pixels"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
