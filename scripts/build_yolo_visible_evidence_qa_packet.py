#!/usr/bin/env python
"""Build full-size visual QA previews for YOLO image-list rows."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument(
        "--exclude-list",
        type=Path,
        action="append",
        default=[],
        help="Rows to subtract from --image-list, for isolating add-on examples.",
    )
    parser.add_argument("--data", type=Path, default=Path("data/cashsnap_v1/data.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-preview-side", type=int, default=1600)
    parser.add_argument("--edge-epsilon", type=float, default=0.03)
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows after exclusion.")
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


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def read_rows(path: Path) -> list[str]:
    rows: list[str] = []
    for raw in resolve(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip().replace("\\", "/")
        if line and not line.startswith("#"):
            rows.append(line)
    return rows


def label_path_for_image(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def source_group(image: str) -> str:
    name = Path(image).name
    prefixes = {
        "asian_currency_": "asian_currency",
        "billsbank_": "billsbank",
        "cambodia_currency_project_": "cambodia_currency_project",
        "cashcountingxl_": "cashcountingxl",
        "khmer_scan_": "khmer",
        "khmer_us_currency_": "khmer_us_currency",
        "usd_total_": "usd_total",
    }
    for prefix, group in prefixes.items():
        if name.startswith(prefix):
            return group
    return "unknown"


def names_from_data(path: Path) -> dict[int, str]:
    data = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(resolve(path))}: expected YAML mapping")
    raw_names = data.get("names")
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    raise SystemExit(f"{repo_rel(resolve(path))}: missing names")


def read_labels(image: str, names: dict[int, str], width: int, height: int, edge_epsilon: float) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    label_path = resolve(label_path_for_image(image))
    if not label_path.exists():
        return labels
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        cx, cy, bw, bh = (float(value) for value in parts[1:5])
        x1 = max(0.0, (cx - bw / 2) * width)
        y1 = max(0.0, (cy - bh / 2) * height)
        x2 = min(float(width), (cx + bw / 2) * width)
        y2 = min(float(height), (cy + bh / 2) * height)
        edge_touch = (
            cx - bw / 2 <= edge_epsilon
            or cy - bh / 2 <= edge_epsilon
            or cx + bw / 2 >= 1.0 - edge_epsilon
            or cy + bh / 2 >= 1.0 - edge_epsilon
        )
        labels.append(
            {
                "line_no": line_no,
                "class_id": class_id,
                "class_name": names.get(class_id, str(class_id)),
                "cx": cx,
                "cy": cy,
                "bw": bw,
                "bh": bh,
                "area": bw * bh,
                "edge_touch": edge_touch,
                "xyxy": (x1, y1, x2, y2),
            }
        )
    return labels


def color_for(class_id: int) -> tuple[int, int, int]:
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (0, 128, 128),
        (240, 50, 230),
        (170, 110, 40),
        (0, 0, 128),
        (128, 128, 0),
        (128, 0, 0),
        (70, 70, 70),
    ]
    return palette[class_id % len(palette)]


def draw_preview(image: str, labels: list[dict[str, Any]], out_path: Path, max_side: int) -> None:
    with Image.open(resolve(image)) as loaded:
        original = ImageOps.exif_transpose(loaded.convert("RGB"))
    scale = min(1.0, max_side / max(original.size))
    if scale < 1.0:
        preview = original.resize((round(original.width * scale), round(original.height * scale)), Image.Resampling.LANCZOS)
    else:
        preview = original.copy()
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    for label in labels:
        x1, y1, x2, y2 = label["xyxy"]
        box = tuple(round(value * scale) for value in (x1, y1, x2, y2))
        color = color_for(int(label["class_id"]))
        width = max(3, round(4 * scale))
        draw.rectangle(box, outline=color, width=width)
        caption = f"{label['class_name']} area={label['area']:.3f}"
        if label["edge_touch"]:
            caption += " edge"
        draw.rectangle((box[0], max(0, box[1] - 18), box[0] + 220, max(16, box[1])), fill=(255, 255, 255))
        draw.text((box[0] + 4, max(0, box[1] - 16)), caption, fill=color, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    preview_dir = out_dir / "full_previews"
    names = names_from_data(args.data)
    excluded: set[str] = set()
    for path in args.exclude_list:
        excluded.update(read_rows(path))
    rows = [row for row in read_rows(args.image_list) if row not in excluded]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no rows selected after exclusions")

    packet_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    edge_rows = 0
    for index, image in enumerate(rows, start=1):
        with Image.open(resolve(image)) as loaded:
            image_size = ImageOps.exif_transpose(loaded).size
        labels = read_labels(image, names, image_size[0], image_size[1], args.edge_epsilon)
        for label in labels:
            class_counts[str(label["class_name"])] += 1
        source_counts[source_group(image)] += 1
        if any(bool(label["edge_touch"]) for label in labels):
            edge_rows += 1
        preview = preview_dir / f"{index:04d}_{Path(image).stem}.jpg"
        draw_preview(image, labels, preview, args.max_preview_side)
        areas = [float(label["area"]) for label in labels]
        packet_rows.append(
            {
                "review_decision": "",
                "usable_as": "",
                "final_class_or_route": "",
                "review_notes": "",
                "image": image,
                "label": repo_rel(resolve(label_path_for_image(image))),
                "full_preview": rel_between(out_dir, preview),
                "source_group": source_group(image),
                "classes": ",".join(str(label["class_name"]) for label in labels),
                "boxes": len(labels),
                "edge_touch": "1" if any(bool(label["edge_touch"]) for label in labels) else "0",
                "min_label_area": f"{min(areas):.6f}" if areas else "",
                "max_label_area": f"{max(areas):.6f}" if areas else "",
                "label_policy_prompt": "accept only if visible evidence is human-identifiable and countable; exclude ambiguous texture/slivers/duplicate-label risks",
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(packet_rows[0].keys())
    with (out_dir / "packet.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(packet_rows)
    summary = {
        "schema": "cashsnap_yolo_visible_evidence_qa_packet_v1",
        "image_list": repo_rel(resolve(args.image_list)),
        "exclude_lists": [repo_rel(resolve(path)) for path in args.exclude_list],
        "rows": len(rows),
        "edge_touch_rows": edge_rows,
        "source_counts": dict(sorted(source_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "packet_csv": repo_rel(out_dir / "packet.csv"),
        "preview_dir": repo_rel(preview_dir),
    }
    (out_dir / "summary.json").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(f"visible_evidence_qa={repo_rel(out_dir / 'packet.csv')} rows={len(rows)} edge_touch_rows={edge_rows}")


if __name__ == "__main__":
    main()
