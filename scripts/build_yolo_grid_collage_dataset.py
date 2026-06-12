#!/usr/bin/env python
"""Build a YOLO grid-collage diagnostic dataset from existing labeled rows.

The first intended use is a 3x3 p24 probe: six real one-note tiles plus three
synthetic one-note tiles per image, with a nine-real-tile control. This is a
diagnostic for multi-bill exposure and real+synth blending, not a real-scene
overlap replacement.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class Label:
    class_id: int
    xc: float
    yc: float
    width: float
    height: float


@dataclass(frozen=True)
class SourceRow:
    image: str
    labels: tuple[Label, ...]
    origin: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True, help="YOLO data YAML/config to sample train rows from.")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--real-tiles", type=int, default=6)
    parser.add_argument("--synthetic-tiles", type=int, default=3)
    parser.add_argument("--tile-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def read_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(resolved)}: expected YAML mapping")
    return payload


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw = Path(str(config.get("path", "."))).expanduser()
    return raw if raw.is_absolute() else (config_path.parent / raw).resolve()


def split_root(dataset_root: Path, split_path: str) -> Path:
    path = Path(split_path)
    return path if path.is_absolute() else dataset_root / path


def read_image_list(path: Path) -> list[str]:
    rows: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            rows.append(repo_rel(resolve(line)))
    return rows


def image_rows(root: Path) -> list[str]:
    image_dir = resolve(root)
    if not image_dir.exists():
        raise SystemExit(f"missing image dir: {repo_rel(image_dir)}")
    return [
        repo_rel(path)
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def train_rows(config_path: Path, config: dict[str, Any]) -> list[str]:
    root = data_root(config_path, config)
    train = config.get("train")
    if isinstance(train, str):
        train_items = [train]
    elif isinstance(train, list) and all(isinstance(item, str) for item in train):
        train_items = [str(item) for item in train]
    else:
        raise SystemExit(f"{repo_rel(config_path)} train split must be a string or list of strings")

    rows: list[str] = []
    for item in train_items:
        path = split_root(root, item)
        if path.suffix.lower() == ".txt":
            rows.extend(read_image_list(path))
        elif path.is_dir():
            rows.extend(image_rows(path))
        else:
            raise SystemExit(f"{repo_rel(config_path)} train item must point to a .txt list or image directory: {item}")
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


def read_labels(image: str) -> tuple[Label, ...]:
    label_path = resolve(label_path_for_image(image))
    if not label_path.exists():
        return ()
    labels: list[Label] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise SystemExit(f"{repo_rel(label_path)} has malformed YOLO label: {line}")
        labels.append(Label(int(float(parts[0])), *(float(value) for value in parts[1:5])))
    return tuple(labels)


def normalize_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names]
    if isinstance(raw_names, dict):
        return [str(value) for _, value in sorted((int(key), value) for key, value in raw_names.items())]
    raise SystemExit("source data YAML must include names as a list or mapping")


def origin_for_row(image: str) -> str:
    normalized = image.replace("\\", "/").lower()
    return "synthetic" if "/synthetic/" in f"/{normalized}" else "real"


def collect_one_box_pools(rows: list[str]) -> dict[str, dict[int, list[SourceRow]]]:
    pools: dict[str, dict[int, list[SourceRow]]] = {"real": defaultdict(list), "synthetic": defaultdict(list)}
    skipped = Counter()
    for row in rows:
        labels = read_labels(row)
        if len(labels) != 1:
            skipped[len(labels)] += 1
            continue
        origin = origin_for_row(row)
        source = SourceRow(image=row, labels=labels, origin=origin)
        pools[origin][labels[0].class_id].append(source)

    for origin in ("real", "synthetic"):
        if not pools[origin]:
            raise SystemExit(f"no one-box {origin} rows found in source train data")
    if skipped:
        print(f"skipped non-one-box rows: {dict(sorted(skipped.items()))}")
    return pools


class CyclingSampler:
    def __init__(self, items: list[Any], rng: random.Random):
        if not items:
            raise ValueError("cannot sample from empty item list")
        self.items = list(items)
        self.rng = rng
        self.index = 0
        self.rng.shuffle(self.items)

    def next(self) -> Any:
        if self.index >= len(self.items):
            self.index = 0
            self.rng.shuffle(self.items)
        item = self.items[self.index]
        self.index += 1
        return item


def make_samplers(pools: dict[str, dict[int, list[SourceRow]]], rng: random.Random) -> dict[tuple[str, int], CyclingSampler]:
    samplers: dict[tuple[str, int], CyclingSampler] = {}
    all_class_ids = sorted(set(pools["real"]) | set(pools["synthetic"]))
    for origin in ("real", "synthetic"):
        missing = [class_id for class_id in all_class_ids if not pools[origin].get(class_id)]
        if missing:
            raise SystemExit(f"{origin} pool missing one-box rows for class ids: {missing}")
        for class_id in all_class_ids:
            samplers[(origin, class_id)] = CyclingSampler(pools[origin][class_id], rng)
    return samplers


def paste_letterboxed(
    canvas: Image.Image,
    source: SourceRow,
    cell_x: int,
    cell_y: int,
    tile_size: int,
) -> list[Label]:
    with Image.open(resolve(source.image)) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        src_w, src_h = image.size
        scale = min(tile_size / src_w, tile_size / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    pad_x = cell_x + (tile_size - new_w) // 2
    pad_y = cell_y + (tile_size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    remapped: list[Label] = []
    canvas_w, canvas_h = canvas.size
    for label in source.labels:
        x1 = (label.xc - label.width / 2.0) * src_w
        y1 = (label.yc - label.height / 2.0) * src_h
        x2 = (label.xc + label.width / 2.0) * src_w
        y2 = (label.yc + label.height / 2.0) * src_h
        x1 = max(0.0, min(float(canvas_w), pad_x + x1 * scale))
        y1 = max(0.0, min(float(canvas_h), pad_y + y1 * scale))
        x2 = max(0.0, min(float(canvas_w), pad_x + x2 * scale))
        y2 = max(0.0, min(float(canvas_h), pad_y + y2 * scale))
        if x2 <= x1 or y2 <= y1:
            continue
        remapped.append(
            Label(
                class_id=label.class_id,
                xc=((x1 + x2) / 2.0) / canvas_w,
                yc=((y1 + y2) / 2.0) / canvas_h,
                width=(x2 - x1) / canvas_w,
                height=(y2 - y1) / canvas_h,
            )
        )
    return remapped


def write_label_file(path: Path, labels: list[Label]) -> None:
    lines = [
        f"{label.class_id} {label.xc:.6f} {label.yc:.6f} {label.width:.6f} {label.height:.6f}"
        for label in labels
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_contact_sheet(image_paths: list[Path], out_path: Path, *, thumb_size: int = 240, cols: int = 4) -> None:
    if not image_paths:
        return
    rows = (len(image_paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (32, 32, 32))
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as raw:
            thumb = ImageOps.contain(ImageOps.exif_transpose(raw).convert("RGB"), (thumb_size, thumb_size))
        x = (index % cols) * thumb_size + (thumb_size - thumb.width) // 2
        y = (index // cols) * thumb_size + (thumb_size - thumb.height) // 2
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)


def write_data_yaml(out_root: Path, names: list[str]) -> None:
    payload = {
        "path": rel_between(out_root, ROOT),
        "train": repo_rel(out_root / "images" / "train"),
        "val": "data/cashsnap_v1/images/val",
        "test": "data/cashsnap_v1/images/test",
        "names": {index: name for index, name in enumerate(names)},
    }
    (out_root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.grid_size <= 0:
        raise SystemExit("--grid-size must be positive")
    tile_count = args.grid_size * args.grid_size
    if args.real_tiles < 0 or args.synthetic_tiles < 0 or args.real_tiles + args.synthetic_tiles != tile_count:
        raise SystemExit("--real-tiles + --synthetic-tiles must equal --grid-size squared")
    if args.tile_size < 64:
        raise SystemExit("--tile-size must be at least 64")

    source_path = resolve(args.source_data)
    source_config = read_yaml(source_path)
    names = normalize_names(source_config.get("names"))
    rows = train_rows(source_path, source_config)
    pools = collect_one_box_pools(rows)

    rng = random.Random(args.seed)
    class_ids = sorted(set(pools["real"]) | set(pools["synthetic"]))
    class_cycle = CyclingSampler(class_ids, rng)
    samplers = make_samplers(pools, rng)

    out_root = resolve(args.out_root)
    if out_root.exists() and args.clean:
        shutil.rmtree(out_root)
    image_dir = out_root / "images" / "train"
    label_dir = out_root / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_root / "manifest.jsonl"
    preview_paths: list[Path] = []
    class_counts: Counter[int] = Counter()
    origin_counts: Counter[str] = Counter()
    label_count_hist: Counter[int] = Counter()
    canvas_size = args.grid_size * args.tile_size

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for image_index in range(args.count):
            origins = ["real"] * args.real_tiles + ["synthetic"] * args.synthetic_tiles
            rng.shuffle(origins)
            canvas = Image.new("RGB", (canvas_size, canvas_size), (24, 24, 24))
            labels: list[Label] = []
            tiles: list[dict[str, Any]] = []
            for tile_index, origin in enumerate(origins):
                class_id = class_cycle.next()
                source = samplers[(origin, class_id)].next()
                row = tile_index // args.grid_size
                col = tile_index % args.grid_size
                cell_x = col * args.tile_size
                cell_y = row * args.tile_size
                remapped = paste_letterboxed(canvas, source, cell_x, cell_y, args.tile_size)
                labels.extend(remapped)
                origin_counts[origin] += 1
                for label in remapped:
                    class_counts[label.class_id] += 1
                tiles.append(
                    {
                        "tile_index": tile_index,
                        "origin": origin,
                        "class_id": class_id,
                        "class_name": names[class_id] if class_id < len(names) else str(class_id),
                        "source_image": source.image,
                        "output_labels": len(remapped),
                    }
                )

            stem = f"grid3x3_{image_index:05d}"
            image_path = image_dir / f"{stem}.jpg"
            label_path = label_dir / f"{stem}.txt"
            canvas.save(image_path, quality=args.jpeg_quality)
            write_label_file(label_path, labels)
            label_count_hist[len(labels)] += 1
            if len(preview_paths) < args.preview_count:
                preview_paths.append(image_path)
            manifest_file.write(
                json.dumps(
                    {
                        "image": repo_rel(image_path),
                        "label": repo_rel(label_path),
                        "grid_size": args.grid_size,
                        "tile_size": args.tile_size,
                        "labels": len(labels),
                        "tiles": tiles,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    write_data_yaml(out_root, names)
    write_contact_sheet(preview_paths, out_root / "qa" / "contact_sheet.jpg")

    summary = {
        "source_data": repo_rel(source_path),
        "out_root": repo_rel(out_root),
        "count": args.count,
        "grid_size": args.grid_size,
        "tile_size": args.tile_size,
        "real_tiles": args.real_tiles,
        "synthetic_tiles": args.synthetic_tiles,
        "seed": args.seed,
        "images": args.count,
        "boxes": sum(class_counts.values()),
        "origin_tile_counts": dict(sorted(origin_counts.items())),
        "label_count_hist": dict(sorted(label_count_hist.items())),
        "class_counts": {names[class_id]: class_counts[class_id] for class_id in sorted(class_counts)},
        "train": repo_rel(image_dir),
        "data_yaml": repo_rel(out_root / "data.yaml"),
        "manifest": repo_rel(manifest_path),
        "contact_sheet": repo_rel(out_root / "qa" / "contact_sheet.jpg"),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
