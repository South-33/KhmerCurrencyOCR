#!/usr/bin/env python
"""Build a YOLO config with label-preserving styled positive support rows."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_BASE = (
    ROOT
    / "configs"
    / "webgl_ablation"
    / "cashsnap_target_anchor_transplant_alpha_contact_geoscale205_minshort190_bal20_probe_puresynth_realval_v1.yaml"
)
STYLE_RANGES = {
    "mined_fp_dark_mild_v1": {
        "brightness": (0.82, 0.95),
        "contrast": (1.02, 1.18),
        "saturation": (1.00, 1.22),
        "vignette": (0.04, 0.12),
        "grain_sigma": (0.2, 1.2),
        "gamma": (0.98, 1.04),
    },
    "mined_fp_dark_medium_v1": {
        "brightness": (0.68, 0.84),
        "contrast": (1.10, 1.35),
        "saturation": (1.05, 1.50),
        "vignette": (0.10, 0.24),
        "grain_sigma": (0.8, 3.0),
        "gamma": (0.96, 1.06),
    },
    "mined_fp_dark_v1": {
        "brightness": (0.50, 0.72),
        "contrast": (1.32, 1.92),
        "saturation": (1.30, 2.35),
        "vignette": (0.24, 0.46),
        "grain_sigma": (2.0, 6.5),
        "gamma": (0.92, 1.08),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--target-classes", default="", help="Comma/space separated class names; defaults to all classes.")
    parser.add_argument("--rows-per-class", type=int, default=1)
    parser.add_argument("--selection", choices=("spread", "random"), default="spread")
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--style", choices=tuple(STYLE_RANGES), default="mined_fp_dark_v1")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-config", type=Path, required=True)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument(
        "--intended-use",
        default=(
            "pure-synth TSTR probe adding label-preserving dark camera-style "
            "positive support so dark currency-like pixels are not only seen as background"
        ),
    )
    parser.add_argument(
        "--promotion-rule",
        default=(
            "reject unless fixed-step self-eval preserves the alpha/geoscale reference "
            "before combining with any zero-label negative dose"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"[,;\s]+", value.strip()) if item.strip()]


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(resolve(path))}: expected YAML mapping")
    return data


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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
            rows.append(line.replace("\\", "/"))
    return rows


def image_rows(root: Path) -> list[str]:
    image_dir = resolve(root)
    return [
        repo_rel(path)
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def train_rows(config_path: Path, config: dict[str, Any]) -> tuple[list[str], list[str]]:
    root = data_root(config_path, config)
    train = config.get("train")
    if isinstance(train, str):
        train_items = [train]
    elif isinstance(train, list) and all(isinstance(item, str) for item in train):
        train_items = [str(item) for item in train]
    else:
        raise SystemExit(f"{repo_rel(config_path)} train split must be a string or list of strings")

    rows: list[str] = []
    sources: list[str] = []
    for item in train_items:
        path = split_root(root, item)
        if path.suffix.lower() == ".txt":
            rows.extend(read_image_list(path))
        elif path.is_dir():
            rows.extend(image_rows(path))
        else:
            raise SystemExit(f"{repo_rel(config_path)} train item must point to a .txt list or image directory: {item}")
        sources.append(repo_rel(path))
    return rows, sources


def write_image_list(path: Path, rows: list[str]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def names_by_id(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names", {})
    if not isinstance(names, dict):
        raise SystemExit("base config names must be a mapping")
    return {int(class_id): str(class_name) for class_id, class_name in names.items()}


def label_path_for_image(image: str) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def single_label_class(image: str, names: dict[int, str]) -> str | None:
    label = resolve(label_path_for_image(image))
    if not label.exists():
        return None
    lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    parts = lines[0].split()
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0])
    except ValueError:
        return None
    return names.get(class_id)


def select_rows(rows: list[str], count: int, selection: str, rng: random.Random) -> list[str]:
    if count < 1:
        raise SystemExit("--rows-per-class must be at least 1")
    rows = sorted(dict.fromkeys(rows))
    if len(rows) < count:
        raise SystemExit(f"not enough candidate rows: {len(rows)} < {count}")
    if selection == "random":
        return sorted(rng.sample(rows, count))
    if count == 1:
        return [rows[len(rows) // 2]]
    indexes = sorted({round(index * (len(rows) - 1) / (count - 1)) for index in range(count)})
    selected = [rows[index] for index in indexes]
    cursor = 0
    while len(selected) < count and cursor < len(rows):
        row = rows[cursor]
        if row not in selected:
            selected.append(row)
        cursor += 1
    return sorted(selected)


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def style_params(style: str, rng: random.Random) -> dict[str, float]:
    ranges = STYLE_RANGES.get(style)
    if ranges is None:
        raise SystemExit(f"unsupported style: {style}")
    return {
        key: rng.uniform(float(bounds[0]), float(bounds[1]))
        for key, bounds in ranges.items()
    }


def apply_style(image: Image.Image, params: dict[str, float], rng: random.Random) -> Image.Image:
    styled = image.convert("RGB")
    styled = ImageEnhance.Contrast(styled).enhance(float(params["contrast"]))
    styled = ImageEnhance.Color(styled).enhance(float(params["saturation"]))
    styled = ImageEnhance.Brightness(styled).enhance(float(params["brightness"]))

    arr = np.asarray(styled, dtype=np.float32) / 255.0
    gamma = float(params["gamma"])
    arr = np.clip(arr, 0.0, 1.0) ** gamma
    height, width = arr.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / max(width / 2.0, 1.0)
    ny = (yy - height / 2.0) / max(height / 2.0, 1.0)
    radius = np.sqrt(nx * nx + ny * ny)
    vignette = 1.0 - np.clip(radius - 0.18, 0.0, 1.0) ** 1.35 * float(params["vignette"])
    arr *= vignette[..., None]
    sigma = float(params["grain_sigma"]) / 255.0
    noise_rng = np.random.default_rng(rng.randrange(0, 2**32))
    arr += noise_rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray(np.round(arr * 255.0).astype(np.uint8), mode="RGB")


def write_support_root(
    out_root: Path,
    selected_by_class: dict[str, list[str]],
    style: str,
    seed: int,
    names: dict[int, str],
    dry_run: bool,
) -> tuple[list[str], list[dict[str, Any]]]:
    image_out = out_root / "images" / "train"
    label_out = out_root / "labels" / "train"
    support_rows: list[str] = []
    manifest_rows: list[dict[str, Any]] = []
    if not dry_run:
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

    for class_index, class_name in sorted(names.items()):
        rows = selected_by_class.get(class_name, [])
        for local_index, image_row in enumerate(rows):
            source_image = resolve(Path(image_row))
            source_label = resolve(label_path_for_image(image_row))
            stem = f"{safe_stem(class_name)}_{local_index:02d}_{safe_stem(source_image.stem)}_{style}"
            out_image = image_out / f"{stem}.png"
            out_label = label_out / f"{stem}.txt"
            rng = random.Random(seed + class_index * 1009 + local_index * 9173)
            params = style_params(style, rng)
            if not dry_run:
                with Image.open(source_image) as image:
                    styled = apply_style(image, params, rng)
                styled.save(out_image)
                shutil.copyfile(source_label, out_label)
            row = repo_rel(out_image)
            support_rows.append(row)
            manifest_rows.append(
                {
                    "image": row,
                    "label": repo_rel(out_label),
                    "source_image": image_row,
                    "source_label": repo_rel(source_label),
                    "class_name": class_name,
                    "class_id": class_index,
                    "style": style,
                    "style_params": {key: round(float(value), 6) for key, value in params.items()},
                }
            )
    return support_rows, manifest_rows


def main() -> int:
    args = parse_args()
    base_path = resolve(args.base)
    base_config = read_yaml(base_path)
    base_rows, base_sources = train_rows(base_path, base_config)
    names = names_by_id(base_config)
    target_classes = parse_csv(args.target_classes) or [names[index] for index in sorted(names)]
    unknown = [class_name for class_name in target_classes if class_name not in set(names.values())]
    if unknown:
        raise SystemExit(f"unknown target classes: {', '.join(unknown)}")

    rng = random.Random(args.seed)
    candidates_by_class: dict[str, list[str]] = defaultdict(list)
    for image in base_rows:
        class_name = single_label_class(image, names)
        if class_name in target_classes:
            candidates_by_class[class_name].append(image)

    selected_by_class: dict[str, list[str]] = {}
    for class_name in target_classes:
        selected_by_class[class_name] = select_rows(
            candidates_by_class[class_name],
            args.rows_per_class,
            args.selection,
            rng,
        )

    out_root = resolve(args.out_root)
    support_rows, manifest_rows = write_support_root(
        out_root,
        selected_by_class,
        args.style,
        int(args.seed),
        names,
        args.dry_run,
    )
    combined_rows = [*base_rows, *support_rows]
    if len(set(combined_rows)) != len(combined_rows):
        raise SystemExit("support selection introduced duplicate train rows")

    report = {
        "schema": "cashsnap_yolo_style_positive_support_v1",
        "base_config": repo_rel(base_path),
        "base_train_sources": base_sources,
        "target_classes": target_classes,
        "rows_per_class": int(args.rows_per_class),
        "selection": str(args.selection),
        "seed": int(args.seed),
        "style": str(args.style),
        "out_root": repo_rel(out_root),
        "base_images": len(base_rows),
        "support_images": len(support_rows),
        "combined_images": len(combined_rows),
        "available_candidates": {
            class_name: len(candidates_by_class[class_name]) for class_name in target_classes
        },
        "selected_source_rows": selected_by_class,
        "support_image_rows": support_rows,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0

    write_image_list(args.out_list, combined_rows)
    manifest = {
        "schema": "cashsnap_yolo_style_positive_support_manifest_v1",
        "summary": report,
        "rows": manifest_rows,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_yaml(
        out_root / "data.yaml",
        {
            "path": ".",
            "train": "images/train",
            "val": "images/train",
            "names": copy.deepcopy(base_config.get("names", {})),
        },
    )

    out_config = resolve(args.out_config)
    config = copy.deepcopy(base_config)
    config["path"] = rel_between(out_config.parent, ROOT)
    config["train"] = repo_rel(resolve(args.out_list))
    sources = copy.deepcopy(config.get("cashsnap_sources", {}))
    if not isinstance(sources, dict):
        sources = {}
    sources["style_positive_support_base_config"] = repo_rel(base_path)
    sources["style_positive_support_root"] = repo_rel(out_root)
    config["cashsnap_sources"] = sources
    config["cashsnap_style_positive_support"] = report
    policy = copy.deepcopy(config.get("cashsnap_policy", {}))
    if not isinstance(policy, dict):
        policy = {}
    policy["intended_use"] = args.intended_use
    policy["promotion_rule"] = args.promotion_rule
    config["cashsnap_policy"] = policy
    write_yaml(out_config, config)
    print(json.dumps(report, indent=2))
    print(f"wrote_root={repo_rel(out_root)}")
    print(f"wrote_list={repo_rel(resolve(args.out_list))}")
    print(f"wrote_config={repo_rel(out_config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
