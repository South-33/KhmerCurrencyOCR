#!/usr/bin/env python
"""Materialize paired target+foreign rows with an explicit unknown-note class."""

from __future__ import annotations

import argparse
import copy
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
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--accepted-list", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-pair-config", type=Path, required=True)
    parser.add_argument("--out-pair-list", type=Path, required=True)
    parser.add_argument("--out-mix-config", type=Path, required=True)
    parser.add_argument("--out-mix-list", type=Path, required=True)
    parser.add_argument("--unknown-class-name", default="UNKNOWN_FOREIGN_NOTE")
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
    data = yaml.safe_load(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_rel(resolve(path))}: expected YAML mapping")
    return data


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def class_names(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names", {})
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    raise SystemExit("YOLO config names must be a mapping or list")


def label_path_for_image(image: str | Path) -> Path:
    path = Path(image)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def image_key(source_root: Path, image_path: Path) -> str:
    try:
        return image_path.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError:
        return repo_rel(image_path)


def read_image_list(path: Path) -> list[Path]:
    rows: list[Path] = []
    for raw_line in resolve(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            rows.append(resolve(line))
    if not rows:
        raise SystemExit(f"empty accepted list: {repo_rel(resolve(path))}")
    return rows


def load_metadata(source_root: Path) -> dict[str, dict[str, Any]]:
    metadata_path = source_root / "metadata" / "train.jsonl"
    if not metadata_path.exists():
        raise SystemExit(f"missing metadata: {repo_rel(metadata_path)}")
    records: dict[str, dict[str, Any]] = {}
    for line_no, raw_line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        image = str(record.get("image", ""))
        if not image:
            raise SystemExit(f"{repo_rel(metadata_path)}:{line_no} missing image")
        records[image.replace("\\", "/")] = record
    return records


def xyxy_to_yolo(
    xyxy: list[float],
    canvas_size: list[int],
    class_id: int,
) -> str:
    width, height = float(canvas_size[0]), float(canvas_size[1])
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    x1 = max(0.0, min(width, x1))
    x2 = max(0.0, min(width, x2))
    y1 = max(0.0, min(height, y1))
    y2 = max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise SystemExit(f"invalid unknown xyxy after clipping: {xyxy}")
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    raw = Path(str(config.get("path", "."))).expanduser()
    return raw if raw.is_absolute() else (config_path.parent / raw).resolve()


def split_rows(config_path: Path, config: dict[str, Any], split_name: str) -> list[str]:
    root = data_root(config_path, config)
    split_value = config.get(split_name)
    values = split_value if isinstance(split_value, list) else [split_value]
    rows: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise SystemExit(f"{repo_rel(config_path)} {split_name} split must contain strings")
        split_path = Path(value)
        path = split_path if split_path.is_absolute() else root / split_path
        if path.suffix.lower() == ".txt":
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line and not line.startswith("#"):
                    rows.append(repo_rel(resolve(line)))
        elif path.is_dir():
            rows.extend(
                repo_rel(item)
                for item in sorted(path.iterdir())
                if item.is_file() and item.suffix.lower() in IMAGE_EXTS
            )
        else:
            raise SystemExit(f"unsupported {split_name} split path: {repo_rel(path)}")
    return rows


def label_class_counts(rows: list[str], names: dict[int, str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        label_path = resolve(label_path_for_image(row))
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            class_id = int(float(parts[0]))
            counts[names.get(class_id, f"class_{class_id}")] += 1
    return dict(sorted(counts.items()))


def main() -> int:
    args = parse_args()
    source_root = resolve(args.source_root)
    out_root = resolve(args.out_root)
    if args.clean and out_root.exists():
        if not out_root.resolve().is_relative_to((ROOT / "data" / "synthetic").resolve()):
            raise SystemExit(f"--clean target must stay under data/synthetic: {repo_rel(out_root)}")
        shutil.rmtree(out_root)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "metadata").mkdir(parents=True, exist_ok=True)

    source_config = read_yaml(source_root / "data.yaml")
    names = class_names(source_config)
    unknown_class_id = max(names) + 1
    names[unknown_class_id] = args.unknown_class_name

    records_by_image = load_metadata(source_root)
    accepted_images = read_image_list(args.accepted_list)
    out_rows: list[str] = []
    out_records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for source_image in accepted_images:
        key = image_key(source_root, source_image)
        record = records_by_image.get(key)
        if record is None:
            raise SystemExit(f"missing metadata for accepted image: {repo_rel(source_image)}")
        unknown_xyxy = record.get("background_avoid_xyxy")
        canvas_size = record.get("canvas_size")
        if not isinstance(unknown_xyxy, list) or len(unknown_xyxy) != 4:
            raise SystemExit(f"metadata missing background_avoid_xyxy for {key}")
        if not isinstance(canvas_size, list) or len(canvas_size) != 2:
            raise SystemExit(f"metadata missing canvas_size for {key}")

        out_image = out_root / "images" / "train" / source_image.name
        out_label = out_root / "labels" / "train" / f"{source_image.stem}.txt"
        source_label = source_root / label_path_for_image(key)
        if not source_label.exists():
            raise SystemExit(f"missing source label: {repo_rel(source_label)}")
        shutil.copy2(source_image, out_image)
        label_lines = [
            line.strip()
            for line in source_label.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        label_lines.append(xyxy_to_yolo(unknown_xyxy, canvas_size, unknown_class_id))
        out_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        out_row = repo_rel(out_image)
        out_rows.append(out_row)
        counts[str(record.get("class_name", "unknown_target"))] += 1
        out_record = copy.deepcopy(record)
        out_record.update(
            {
                "image": rel_between(out_root, out_image),
                "label": rel_between(out_root, out_label),
                "source_image": repo_rel(source_image),
                "source_label": repo_rel(source_label),
                "unknown_class_id": unknown_class_id,
                "unknown_class_name": args.unknown_class_name,
                "unknown_xyxy": unknown_xyxy,
            }
        )
        out_records.append(out_record)

    write_text(args.out_pair_list, "\n".join(out_rows) + "\n")
    (out_root / "metadata" / "train.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in out_records) + "\n",
        encoding="utf-8",
    )
    pair_config = copy.deepcopy(read_yaml(args.out_pair_config) if resolve(args.out_pair_config).exists() else {})
    if not pair_config:
        pair_config = {
            "path": "../..",
            "train": repo_rel(resolve(args.out_pair_list)),
            "val": "data/cashsnap_v1/images/val",
            "test": "data/cashsnap_v1/images/test",
        }
    pair_config["train"] = repo_rel(resolve(args.out_pair_list))
    pair_config["names"] = names
    pair_config["cashsnap_policy"] = {
        "intended_use": "14-class paired target+foreign unknown-class view; diagnostic only.",
        "phase": "target-vs-unknown structural objective probe",
        "source_dataset": repo_rel(source_root),
        "unknown_class_id": unknown_class_id,
        "unknown_class_name": args.unknown_class_name,
        "teacher_accepted_rows": len(out_rows),
        "target_class_counts": dict(sorted(counts.items())),
        "promotion_rule": "Reject unless unknown-class training improves strict full real transfer without protected-class or background-FP regressions after filtering unknown predictions.",
    }
    write_yaml(args.out_pair_config, pair_config)
    write_yaml(
        out_root / "data.yaml",
        {
            "path": out_root.as_posix(),
            "train": "images/train",
            "val": "images/train",
            "test": "images/train",
            "names": names,
        },
    )

    base_config_path = resolve(args.base_config)
    base_config = read_yaml(base_config_path)
    base_rows = split_rows(base_config_path, base_config, "train")
    mix_rows = base_rows + out_rows
    write_text(args.out_mix_list, "\n".join(mix_rows) + "\n")
    mix_config = copy.deepcopy(base_config)
    mix_config["train"] = repo_rel(resolve(args.out_mix_list))
    mix_config["names"] = names
    policy = copy.deepcopy(base_config.get("cashsnap_policy", {}))
    policy.update(
        {
            "intended_use": "Fair strict-best append probe with explicit UNKNOWN_FOREIGN_NOTE labels for paired foreign-note distractors.",
            "phase": "target-vs-unknown structural objective probe",
            "unknown_pair_rows": len(out_rows),
            "unknown_class_id": unknown_class_id,
            "unknown_class_name": args.unknown_class_name,
            "unknown_pair_source_dataset": repo_rel(source_root),
            "promotion_rule": "Reject unless full real transfer improves strict best without protected-class regressions and lightweight eval improves after filtering UNKNOWN_FOREIGN_NOTE predictions.",
        }
    )
    mix_config["cashsnap_policy"] = policy
    write_yaml(args.out_mix_config, mix_config)

    summary = {
        "source_root": repo_rel(source_root),
        "out_root": repo_rel(out_root),
        "accepted_rows": len(out_rows),
        "unknown_class_id": unknown_class_id,
        "unknown_class_name": args.unknown_class_name,
        "out_pair_list": repo_rel(resolve(args.out_pair_list)),
        "out_pair_config": repo_rel(resolve(args.out_pair_config)),
        "out_mix_list": repo_rel(resolve(args.out_mix_list)),
        "out_mix_config": repo_rel(resolve(args.out_mix_config)),
        "base_rows": len(base_rows),
        "mix_rows": len(mix_rows),
        "target_class_counts": dict(sorted(counts.items())),
        "mix_label_class_counts": label_class_counts(mix_rows, names),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
