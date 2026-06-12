#!/usr/bin/env python
"""Build an ImageFolder dataset for a target-vs-reject proposal gate."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from local_runtime import configure_project_cache

configure_project_cache()

from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
OUT_CLASSES = ("target", "reject")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--source",
        nargs=4,
        action="append",
        metavar=("DATA_YAML", "SOURCE_SPLIT", "OUT_SPLIT", "MAX_IMAGES"),
        default=[],
        help="Add a YOLO source split using --unlabeled-image-policy.",
    )
    parser.add_argument(
        "--source-policy",
        nargs=5,
        action="append",
        metavar=("DATA_YAML", "SOURCE_SPLIT", "OUT_SPLIT", "MAX_IMAGES", "UNLABELED_POLICY"),
        default=[],
        help=(
            "Add a YOLO source split with a per-source unlabeled policy "
            "(skip, reject_proposals, or random_only)."
        ),
    )
    parser.add_argument("--target-class", action="append", default=[], help="Target class id/name. Default: all non-reject classes.")
    parser.add_argument("--reject-class", action="append", default=[], help="Reject class id/name, e.g. UNKNOWN_FOREIGN_NOTE.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--det-iou", type=float, default=0.70)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--agnostic-nms", action="store_true")
    parser.add_argument("--target-iou", type=float, default=0.45)
    parser.add_argument("--reject-iou", type=float, default=0.35)
    parser.add_argument("--negative-iou", type=float, default=0.05)
    parser.add_argument("--crop-padding", type=float, default=0.06)
    parser.add_argument(
        "--gt-target-crops-per-label",
        type=int,
        default=0,
        help="Add this many full GT target crops per target label before detector proposal crops.",
    )
    parser.add_argument(
        "--edge-target-crops-per-label",
        type=int,
        default=0,
        help="Add this many thin/partial GT target crops per target label before detector proposal crops.",
    )
    parser.add_argument("--random-rejects-per-image", type=int, default=1)
    parser.add_argument("--max-per-split-class", type=int, default=3000)
    parser.add_argument(
        "--unlabeled-image-policy",
        choices=("skip", "reject_proposals", "random_only"),
        default="skip",
        help="How to handle images with no target or reject labels.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean", action="store_true")
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


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed = (ROOT / "data").resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise SystemExit(f"Refusing to clean outside {allowed}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"YOLO data YAML must be a mapping: {repo_rel(path)}")
    return config


def parse_names(config: dict[str, Any]) -> dict[int, str]:
    raw_names = config.get("names") or {}
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    raise SystemExit("YOLO data names must be a list or mapping")


def parse_class_ids(raw_values: list[str], names: dict[int, str]) -> set[int]:
    ids: set[int] = set()
    name_to_id = {name: class_id for class_id, name in names.items()}
    for raw_value in raw_values:
        for token in raw_value.replace(",", " ").split():
            if token in name_to_id:
                ids.add(name_to_id[token])
                continue
            try:
                ids.add(int(token))
            except ValueError as exc:
                raise SystemExit(f"unknown class token {token!r}; names={sorted(name_to_id)}") from exc
    return ids


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    root = Path(str(config.get("path", "."))).expanduser()
    return root if root.is_absolute() else (config_path.parent / root).resolve()


def split_root(root: Path, split_path: str) -> Path:
    path = Path(split_path)
    return path if path.is_absolute() else root / path


def read_split_list(root: Path, split_path: str) -> list[Path]:
    list_path = split_root(root, split_path)
    images: list[Path] = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image = Path(line)
        images.append(image if image.is_absolute() else root / image)
    return images


def split_images(config_path: Path, config: dict[str, Any], split: str) -> list[Path]:
    root = data_root(config_path, config)
    split_value = config.get(split)
    if split_value is None:
        raise SystemExit(f"{repo_rel(config_path)} has no split {split!r}")
    values = split_value if isinstance(split_value, list) else [split_value]
    images: list[Path] = []
    for value in values:
        resolved = split_root(root, str(value))
        if resolved.suffix.lower() == ".txt":
            images.extend(read_split_list(root, str(value)))
        else:
            images.extend(sorted(path for path in resolved.glob("*") if path.suffix.lower() in IMAGE_EXTS))
    return images


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image_path.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_labels(label_path: Path, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    width, height = image_size
    labels: list[dict[str, Any]] = []
    if not label_path.exists():
        return labels
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected 5 YOLO fields")
        class_id = int(parts[0])
        cx, cy, bw, bh = [float(value) for value in parts[1:]]
        labels.append(
            {
                "class_id": class_id,
                "xyxy": [
                    (cx - bw / 2.0) * width,
                    (cy - bh / 2.0) * height,
                    (cx + bw / 2.0) * width,
                    (cy + bh / 2.0) * height,
                ],
            }
        )
    return labels


def read_label_class_ids(label_path: Path) -> set[int]:
    class_ids: set[int] = set()
    if not label_path.exists():
        return class_ids
    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{repo_rel(label_path)}:{line_no} expected 5 YOLO fields")
        class_ids.add(int(parts[0]))
    return class_ids


def image_from_yolo_result(result: Any, fallback_path: Path) -> Image.Image:
    orig_img = getattr(result, "orig_img", None)
    if orig_img is not None and getattr(orig_img, "ndim", 0) == 3 and orig_img.shape[2] >= 3:
        return Image.fromarray(orig_img[:, :, :3][:, :, ::-1].copy())
    with Image.open(fallback_path) as image:
        return image.convert("RGB")


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def max_iou(box: list[float], boxes: list[list[float]]) -> float:
    return max((box_iou(box, other) for other in boxes), default=0.0)


def padded_box(box: list[float], image_size: tuple[int, int], padding: float) -> tuple[int, int, int, int]:
    width, height = image_size
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0, int(x1 - pad_x)),
        max(0, int(y1 - pad_y)),
        min(width, int(x2 + pad_x)),
        min(height, int(y2 + pad_y)),
    )


def random_box(image_size: tuple[int, int], avoid_boxes: list[list[float]], max_overlap: float, rng: random.Random) -> list[float] | None:
    width, height = image_size
    for _ in range(80):
        crop_w = rng.uniform(0.12, 0.42) * width
        crop_h = rng.uniform(0.12, 0.42) * height
        if crop_w < 24 or crop_h < 24:
            continue
        x1 = rng.uniform(0, max(1.0, width - crop_w))
        y1 = rng.uniform(0, max(1.0, height - crop_h))
        box = [x1, y1, x1 + crop_w, y1 + crop_h]
        if max_iou(box, avoid_boxes) <= max_overlap:
            return box
    return None


def edge_target_box(box: list[float], image_size: tuple[int, int], rng: random.Random) -> list[float]:
    width, height = image_size
    x1, y1, x2, y2 = box
    x1 = min(max(0.0, x1), float(width))
    x2 = min(max(0.0, x2), float(width))
    y1 = min(max(0.0, y1), float(height))
    y2 = min(max(0.0, y2), float(height))
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    fraction = rng.uniform(0.30, 0.55)
    anchor = rng.choice(("start", "middle", "end"))
    if box_w >= box_h:
        strip_h = max(8.0, box_h * fraction)
        if anchor == "start":
            top = y1
        elif anchor == "end":
            top = y2 - strip_h
        else:
            top = y1 + (box_h - strip_h) * rng.uniform(0.35, 0.65)
        return [x1, max(0.0, top), x2, min(float(height), top + strip_h)]
    strip_w = max(8.0, box_w * fraction)
    if anchor == "start":
        left = x1
    elif anchor == "end":
        left = x2 - strip_w
    else:
        left = x1 + (box_w - strip_w) * rng.uniform(0.35, 0.65)
    return [max(0.0, left), y1, min(float(width), left + strip_w), y2]


def write_crop(
    image: Image.Image,
    box: list[float],
    label: str,
    out_dir: Path,
    split: str,
    stem: str,
    index: int,
    padding: float,
) -> Path:
    target_dir = out_dir / split / label
    target_dir.mkdir(parents=True, exist_ok=True)
    crop = image.crop(padded_box(box, image.size, padding))
    target = target_dir / f"{stem}_{index:05d}.jpg"
    crop.save(target, quality=92)
    return target


def batched(items: list[Path], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def proposal_label(
    box: list[float],
    target_boxes: list[list[float]],
    reject_boxes: list[list[float]],
    args: argparse.Namespace,
) -> tuple[str, float, float, str]:
    target_overlap = max_iou(box, target_boxes)
    reject_overlap = max_iou(box, reject_boxes)
    any_overlap = max(target_overlap, reject_overlap)
    if target_overlap >= args.target_iou and target_overlap >= reject_overlap:
        return "target", target_overlap, reject_overlap, "target_iou"
    if reject_overlap >= args.reject_iou and reject_overlap > target_overlap:
        return "reject", target_overlap, reject_overlap, "reject_iou"
    if any_overlap <= args.negative_iou:
        return "reject", target_overlap, reject_overlap, "low_iou"
    return "ignore", target_overlap, reject_overlap, "ambiguous"


def main() -> None:
    args = parse_args()
    valid_unlabeled_policies = {"skip", "reject_proposals", "random_only"}
    source_specs = [
        (data_text, source_split, out_split, max_images_text, args.unlabeled_image_policy)
        for data_text, source_split, out_split, max_images_text in args.source
    ]
    source_specs.extend(tuple(source) for source in args.source_policy)
    if not source_specs:
        raise SystemExit("At least one --source or --source-policy is required")
    for spec in source_specs:
        if spec[4] not in valid_unlabeled_policies:
            raise SystemExit(
                f"unknown source unlabeled policy {spec[4]!r}; "
                f"expected one of {sorted(valid_unlabeled_policies)}"
            )

    out_dir = resolve(args.out)
    if args.clean:
        safe_clean(out_dir)
    for split in ("train", "val", "test"):
        for class_name in OUT_CLASSES:
            (out_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    detector = YOLO(str(resolve(args.detector)))
    counters: Counter[tuple[str, str]] = Counter()
    source_counters: Counter[tuple[str, str, str]] = Counter()
    skip_counters: Counter[str] = Counter()
    rows: list[dict[str, str]] = []

    for source_index, source in enumerate(source_specs):
        data_text, source_split, out_split, max_images_text, source_unlabeled_policy = source
        data_path = resolve(data_text)
        config = load_config(data_path)
        names = parse_names(config)
        reject_ids = parse_class_ids(args.reject_class, names)
        target_ids = parse_class_ids(args.target_class, names) if args.target_class else set(names) - reject_ids
        max_images = int(max_images_text)
        images = split_images(data_path, config, source_split)
        if max_images > 0:
            rng.shuffle(images)
            images = images[:max_images]
        if source_unlabeled_policy == "skip":
            avoid_class_ids = target_ids | reject_ids
            filtered_images: list[Path] = []
            for image_path in images:
                class_ids = read_label_class_ids(label_path_for_image(image_path))
                if class_ids & avoid_class_ids:
                    filtered_images.append(image_path)
                else:
                    skip_counters["unlabeled_skip_pre_detector"] += 1
            images = filtered_images

        for batch in batched(images, max(1, args.batch)):
            results = detector.predict(
                source=[str(path) for path in batch],
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.det_iou,
                agnostic_nms=args.agnostic_nms,
                batch=len(batch),
                device=args.device,
                verbose=False,
            )
            for image_path, result in zip(batch, results, strict=False):
                image = image_from_yolo_result(result, image_path)
                try:
                    labels = read_labels(label_path_for_image(image_path), image.size)
                    target_boxes = [row["xyxy"] for row in labels if int(row["class_id"]) in target_ids]
                    reject_boxes = [row["xyxy"] for row in labels if int(row["class_id"]) in reject_ids]
                    avoid_boxes = target_boxes + reject_boxes
                    if not avoid_boxes and source_unlabeled_policy == "skip":
                        skip_counters["unlabeled_skip"] += 1
                        continue

                    crop_index = 0
                    for target_index, target_box in enumerate(target_boxes):
                        for gt_index in range(args.gt_target_crops_per_label):
                            if counters[(out_split, "target")] >= args.max_per_split_class:
                                skip_counters[f"cap_{out_split}_target"] += 1
                                break
                            crop_path = write_crop(
                                image,
                                target_box,
                                "target",
                                out_dir,
                                out_split,
                                f"s{source_index}_{image_path.stem}_gt{target_index}_{gt_index}",
                                crop_index,
                                args.crop_padding,
                            )
                            crop_index += 1
                            counters[(out_split, "target")] += 1
                            source_counters[(data_text, out_split, "target")] += 1
                            rows.append(
                                {
                                    "split": out_split,
                                    "label": "target",
                                    "crop_path": repo_rel(crop_path),
                                    "image_path": repo_rel(image_path),
                                    "label_path": repo_rel(label_path_for_image(image_path)),
                                    "data": data_text,
                                    "source_split": source_split,
                                    "detector_class": "",
                                    "detector_conf": "",
                                    "target_iou": "1.000000",
                                    "reject_iou": "0.000000",
                                    "reason": "gt_target",
                                    "kind": "gt_target",
                                    "source_unlabeled_policy": source_unlabeled_policy,
                                }
                            )
                        for edge_index in range(args.edge_target_crops_per_label):
                            if counters[(out_split, "target")] >= args.max_per_split_class:
                                skip_counters[f"cap_{out_split}_target"] += 1
                                break
                            edge_box = edge_target_box(target_box, image.size, rng)
                            crop_path = write_crop(
                                image,
                                edge_box,
                                "target",
                                out_dir,
                                out_split,
                                f"s{source_index}_{image_path.stem}_edge{target_index}_{edge_index}",
                                crop_index,
                                args.crop_padding,
                            )
                            crop_index += 1
                            counters[(out_split, "target")] += 1
                            source_counters[(data_text, out_split, "target")] += 1
                            rows.append(
                                {
                                    "split": out_split,
                                    "label": "target",
                                    "crop_path": repo_rel(crop_path),
                                    "image_path": repo_rel(image_path),
                                    "label_path": repo_rel(label_path_for_image(image_path)),
                                    "data": data_text,
                                    "source_split": source_split,
                                    "detector_class": "",
                                    "detector_conf": "",
                                    "target_iou": "1.000000",
                                    "reject_iou": "0.000000",
                                    "reason": "edge_target",
                                    "kind": "edge_target",
                                    "source_unlabeled_policy": source_unlabeled_policy,
                                }
                            )
                    if result.boxes is not None and source_unlabeled_policy != "random_only":
                        xyxy = result.boxes.xyxy.cpu().numpy()
                        cls = result.boxes.cls.cpu().numpy()
                        conf = result.boxes.conf.cpu().numpy()
                        for proposal_index, (box, detector_class, detector_conf) in enumerate(zip(xyxy, cls, conf, strict=False)):
                            xyxy_box = [float(value) for value in box.tolist()]
                            label, target_overlap, reject_overlap, reason = proposal_label(xyxy_box, target_boxes, reject_boxes, args)
                            if label == "ignore":
                                skip_counters[f"proposal_{reason}"] += 1
                                continue
                            if counters[(out_split, label)] >= args.max_per_split_class:
                                skip_counters[f"cap_{out_split}_{label}"] += 1
                                continue
                            crop_path = write_crop(
                                image,
                                xyxy_box,
                                label,
                                out_dir,
                                out_split,
                                f"s{source_index}_{image_path.stem}_p{proposal_index}",
                                crop_index,
                                args.crop_padding,
                            )
                            crop_index += 1
                            counters[(out_split, label)] += 1
                            source_counters[(data_text, out_split, label)] += 1
                            rows.append(
                                {
                                    "split": out_split,
                                    "label": label,
                                    "crop_path": repo_rel(crop_path),
                                    "image_path": repo_rel(image_path),
                                    "label_path": repo_rel(label_path_for_image(image_path)),
                                    "data": data_text,
                                    "source_split": source_split,
                                    "detector_class": str(int(detector_class)),
                                    "detector_conf": f"{float(detector_conf):.6f}",
                                    "target_iou": f"{target_overlap:.6f}",
                                    "reject_iou": f"{reject_overlap:.6f}",
                                    "reason": reason,
                                    "kind": "proposal",
                                    "source_unlabeled_policy": source_unlabeled_policy,
                                }
                            )

                    for random_index in range(args.random_rejects_per_image):
                        if counters[(out_split, "reject")] >= args.max_per_split_class:
                            skip_counters[f"cap_{out_split}_reject"] += 1
                            break
                        box = random_box(image.size, avoid_boxes, args.negative_iou, rng)
                        if box is None:
                            skip_counters["random_no_box"] += 1
                            continue
                        crop_path = write_crop(
                            image,
                            box,
                            "reject",
                            out_dir,
                            out_split,
                            f"s{source_index}_{image_path.stem}_r{random_index}",
                            crop_index,
                            args.crop_padding,
                        )
                        crop_index += 1
                        counters[(out_split, "reject")] += 1
                        source_counters[(data_text, out_split, "reject")] += 1
                        rows.append(
                            {
                                "split": out_split,
                                "label": "reject",
                                "crop_path": repo_rel(crop_path),
                                "image_path": repo_rel(image_path),
                                "label_path": repo_rel(label_path_for_image(image_path)),
                                "data": data_text,
                                "source_split": source_split,
                                "detector_class": "",
                                "detector_conf": "",
                                "target_iou": "0.000000",
                                "reject_iou": "0.000000",
                                "reason": "random_low_iou",
                                "kind": "random",
                                "source_unlabeled_policy": source_unlabeled_policy,
                            }
                        )
                finally:
                    image.close()

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "split",
            "label",
            "crop_path",
            "image_path",
            "label_path",
            "data",
            "source_split",
            "detector_class",
            "detector_conf",
            "target_iou",
            "reject_iou",
            "reason",
            "kind",
            "source_unlabeled_policy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema": "cashsnap_yolo_proposal_gate_dataset_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "detector": repo_rel(resolve(args.detector)),
        "out": repo_rel(out_dir),
        "sources": args.source,
        "source_policy": args.source_policy,
        "source_specs": source_specs,
        "settings": {
            "target_class": args.target_class,
            "reject_class": args.reject_class,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "det_iou": args.det_iou,
            "target_iou": args.target_iou,
            "reject_iou": args.reject_iou,
            "negative_iou": args.negative_iou,
            "crop_padding": args.crop_padding,
            "gt_target_crops_per_label": args.gt_target_crops_per_label,
            "edge_target_crops_per_label": args.edge_target_crops_per_label,
            "random_rejects_per_image": args.random_rejects_per_image,
            "max_per_split_class": args.max_per_split_class,
            "unlabeled_image_policy": args.unlabeled_image_policy,
            "seed": args.seed,
        },
        "rows": len(rows),
        "counts": {f"{split}/{label}": count for (split, label), count in sorted(counters.items())},
        "source_counts": {
            f"{data}|{split}|{label}": count for (data, split, label), count in sorted(source_counters.items())
        },
        "skips": dict(sorted(skip_counters.items())),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} proposal-gate crops to {repo_rel(out_dir)}")
    for key, count in sorted(counters.items()):
        print(f"{key[0]} {key[1]}: {count}")


if __name__ == "__main__":
    main()
