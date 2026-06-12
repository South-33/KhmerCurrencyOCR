#!/usr/bin/env python
"""Materialize a YOLO dataset view where every labeled box is one class."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", required=True, type=Path)
    parser.add_argument("--eval-config", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--out-config", required=True, type=Path)
    parser.add_argument("--single-class-name", default="BANKNOTE")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT / candidate


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(resolved)} must be a YAML mapping")
    return data


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    root_value = config.get("path", ".")
    root = Path(str(root_value)).expanduser()
    if root.is_absolute():
        return root
    return (resolve(config_path).parent / root).resolve()


def split_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def read_split_rows(config_path: Path, config: dict[str, Any], split: str) -> list[Path]:
    root = data_root(config_path, config)
    value = config.get(split)
    if value is None:
        raise SystemExit(f"{repo_rel(resolve(config_path))} missing split {split!r}")
    values = value if isinstance(value, list) else [value]
    rows: list[Path] = []
    for raw_item in values:
        if not isinstance(raw_item, str):
            raise SystemExit(f"{repo_rel(resolve(config_path))} split {split!r} must contain strings")
        item_path = split_path(root, raw_item)
        if item_path.suffix.lower() == ".txt":
            for raw_line in item_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                image_path = Path(line).expanduser()
                rows.append(image_path if image_path.is_absolute() else root / image_path)
            continue
        if item_path.is_dir():
            rows.extend(sorted(path for path in item_path.iterdir() if path.suffix.lower() in IMAGE_EXTS))
            continue
        raise SystemExit(f"unsupported {split} split path: {repo_rel(item_path)}")
    if not rows:
        raise SystemExit(f"{repo_rel(resolve(config_path))} split {split!r} has no images")
    return rows


def names_by_id(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names", {})
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def unique_name(split: str, index: int, image_path: Path) -> str:
    normalized = repo_rel(image_path)
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    safe_stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in image_path.stem)
    safe_stem = safe_stem[:80].strip("_") or "image"
    return f"{split}_{index:06d}_{safe_stem}_{digest}{image_path.suffix.lower()}"


def remap_label_file(
    source_label: Path,
    out_label: Path,
    source_names: dict[int, str],
) -> tuple[int, Counter[str]]:
    out_label.parent.mkdir(parents=True, exist_ok=True)
    class_counts: Counter[str] = Counter()
    out_lines: list[str] = []
    if source_label.exists():
        for line_no, raw_line in enumerate(source_label.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                raise SystemExit(f"{repo_rel(source_label)}:{line_no} expected YOLO class plus box fields")
            try:
                source_class_id = int(float(parts[0]))
            except ValueError as exc:
                raise SystemExit(f"{repo_rel(source_label)}:{line_no} invalid class id {parts[0]!r}") from exc
            class_counts[source_names.get(source_class_id, f"class_{source_class_id}")] += 1
            out_lines.append(" ".join(["0", *parts[1:]]))
    out_label.write_text(("\n".join(out_lines) + "\n") if out_lines else "", encoding="utf-8")
    return len(out_lines), class_counts


def materialize_split(
    *,
    split: str,
    rows: list[Path],
    out_root: Path,
    source_names: dict[int, str],
) -> dict[str, Any]:
    image_dir = out_root / "images" / split
    label_dir = out_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    source_class_counts: Counter[str] = Counter()
    rows_out: list[dict[str, Any]] = []
    labeled_images = 0
    empty_images = 0
    boxes = 0
    for index, source_image in enumerate(rows):
        if not source_image.exists():
            raise SystemExit(f"missing source image: {repo_rel(source_image)}")
        out_image = image_dir / unique_name(split, index, source_image)
        out_label = label_dir / f"{out_image.stem}.txt"
        source_label = label_path_for_image(source_image)
        shutil.copy2(source_image, out_image)
        label_count, counts = remap_label_file(source_label, out_label, source_names)
        boxes += label_count
        source_class_counts.update(counts)
        if label_count:
            labeled_images += 1
        else:
            empty_images += 1
        rows_out.append(
            {
                "split": split,
                "source_image": repo_rel(source_image),
                "source_label": repo_rel(source_label),
                "image": rel_between(out_root, out_image),
                "label": rel_between(out_root, out_label),
                "boxes": label_count,
            }
        )

    manifest_path = out_root / "metadata" / f"{split}.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows_out) + "\n",
        encoding="utf-8",
    )
    return {
        "images": len(rows),
        "labeled_images": labeled_images,
        "empty_images": empty_images,
        "boxes": boxes,
        "source_class_counts": dict(sorted(source_class_counts.items())),
        "manifest": repo_rel(manifest_path),
    }


def main() -> int:
    args = parse_args()
    out_root = resolve(args.out_root)
    if args.clean and out_root.exists():
        allowed_root = (ROOT / "data" / "processed").resolve()
        if not out_root.resolve().is_relative_to(allowed_root):
            raise SystemExit(f"--clean target must stay under data/processed: {repo_rel(out_root)}")
        shutil.rmtree(out_root)

    train_config = load_yaml(args.train_config)
    eval_config = load_yaml(args.eval_config)
    train_names = names_by_id(train_config)
    eval_names = names_by_id(eval_config)
    if not train_names:
        raise SystemExit("train config has no class names")
    if not eval_names:
        raise SystemExit("eval config has no class names")

    split_sources = {
        "train": (read_split_rows(args.train_config, train_config, "train"), train_names),
        "val": (read_split_rows(args.eval_config, eval_config, "val"), eval_names),
        "test": (read_split_rows(args.eval_config, eval_config, "test"), eval_names),
    }
    summaries: dict[str, Any] = {}
    for split, (rows, source_names) in split_sources.items():
        summaries[split] = materialize_split(
            split=split,
            rows=rows,
            out_root=out_root,
            source_names=source_names,
        )

    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: args.single_class_name},
        "cashsnap_single_class_view": {
            "schema": "cashsnap_single_class_view_v1",
            "source_train_config": repo_rel(resolve(args.train_config)),
            "source_eval_config": repo_rel(resolve(args.eval_config)),
            "single_class_name": args.single_class_name,
            "summary": summaries,
        },
    }
    write_yaml(out_root / "data.yaml", data_yaml)

    external_yaml = {
        **data_yaml,
        "path": rel_between(resolve(args.out_config).parent, out_root),
    }
    write_yaml(args.out_config, external_yaml)
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(data_yaml["cashsnap_single_class_view"], indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"out_root": repo_rel(out_root), "out_config": repo_rel(resolve(args.out_config)), "summary": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
