#!/usr/bin/env python
"""Mine train-only real analogs for uncovered representation-gap examples."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from probe_yolo_representation_domain_gap import (
    ROOT,
    choose_device,
    dataset_records,
    extract_features,
    repo_rel,
    resolve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap-summary", required=True, type=Path, help="representation-gap summary.json")
    parser.add_argument(
        "--model-label",
        default="",
        help="For activation-microscope summaries, select this model label.",
    )
    parser.add_argument("--candidate-data", required=True, type=Path, help="Real YOLO dataset YAML for anchors.")
    parser.add_argument("--candidate-split", default="train")
    parser.add_argument("--model", type=Path, default=None, help="Override model from --gap-summary.")
    parser.add_argument("--layer", default=None, help="Override nearest layer from --gap-summary.")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-query", type=int, default=40)
    parser.add_argument("--per-query", type=int, default=5)
    parser.add_argument("--max-candidates-per-class", type=int, default=250)
    parser.add_argument("--allow-cross-class", action="store_true")
    parser.add_argument("--sheet-width", type=int, default=320)
    parser.add_argument("--sheet-columns", type=int, default=2)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed = (ROOT / "runs").resolve()
    if not (resolved == allowed or allowed in resolved.parents):
        raise SystemExit(f"Refusing to clean outside runs/: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def load_gap(path: Path, model_label: str = "") -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") == "cashsnap_yolo_representation_domain_gap_v1":
        return payload
    if payload.get("schema") == "cashsnap_yolo_activation_microscope_v1":
        models = payload.get("models", [])
        if not model_label:
            labels = ", ".join(str(model.get("label")) for model in models)
            raise SystemExit(f"--model-label is required for activation microscope summaries; labels: {labels}")
        selected = next((model for model in models if str(model.get("label")) == model_label), None)
        if selected is None:
            labels = ", ".join(str(model.get("label")) for model in models)
            raise SystemExit(f"model label {model_label!r} not found; labels: {labels}")
        return {
            "schema": "cashsnap_yolo_representation_domain_gap_v1",
            "model": selected["path"],
            "nearest_layer": payload["nearest_layer"],
            "imgsz": payload["imgsz"],
            "top_uncovered_real": selected.get("top_uncovered_real", []),
        }
    raise SystemExit(f"Unsupported gap summary schema: {payload.get('schema')}")
    return payload


def path_from_repo_rel(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def query_records_from_gap(gap: dict[str, Any], top_query: int) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(gap.get("top_uncovered_real", [])[:top_query], start=1):
        image = path_from_repo_rel(str(row["image"]))
        if not image.exists():
            raise SystemExit(f"Missing query image from gap summary: {image}")
        rows.append(
            {
                "domain": "query",
                "image": image,
                "image_rel": repo_rel(image),
                "class_name": str(row.get("class_name", "")),
                "class_id": None,
                "box_count": int(row.get("box_count", 0) or 0),
                "query_rank": index,
                "query_nearest_l2": float(row.get("nearest_l2", 0.0) or 0.0),
            }
        )
    if not rows:
        raise SystemExit("No top_uncovered_real rows in gap summary")
    return rows


def cap_candidates(
    candidates: list[dict[str, Any]],
    *,
    wanted_classes: set[str],
    max_per_class: int,
) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        class_name = str(row["class_name"])
        if wanted_classes and class_name not in wanted_classes:
            continue
        by_class[class_name].append(row)

    capped = []
    for class_name in sorted(by_class):
        rows = sorted(by_class[class_name], key=lambda row: row["image_rel"])
        capped.extend(rows[:max_per_class] if max_per_class > 0 else rows)
    return capped


def remap_class_ids(query_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> None:
    class_id_by_name = {str(row["class_name"]): int(row["class_id"]) for row in candidate_rows}
    for row in query_rows:
        row["class_id"] = class_id_by_name.get(str(row["class_name"]), -1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def image_thumb(path: Path, width: int) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    ratio = width / max(image.width, 1)
    height = max(1, int(round(image.height * ratio)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def draw_labeled_thumb(path: Path, title: str, subtitle: str, width: int) -> Image.Image:
    thumb = image_thumb(path, width)
    label_h = 50
    canvas = Image.new("RGB", (thumb.width, thumb.height + label_h), (245, 245, 245))
    canvas.paste(thumb, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        small = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    draw.text((8, 6), title[:48], fill=(20, 20, 20), font=font)
    draw.text((8, 28), subtitle[:70], fill=(70, 70, 70), font=small)
    return canvas


def write_pair_sheet(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    query_by_image: dict[str, dict[str, Any]],
    width: int,
    columns: int,
) -> None:
    first_matches = [row for row in rows if int(row["rank_for_query"]) == 1]
    if not first_matches:
        return
    cells: list[Image.Image] = []
    for row in first_matches:
        query = query_by_image[row["query_image"]]
        query_thumb = draw_labeled_thumb(
            path_from_repo_rel(row["query_image"]),
            f"query {query['query_rank']} {row['query_class']}",
            f"gap {float(query['query_nearest_l2']):.2f}",
            width,
        )
        candidate_thumb = draw_labeled_thumb(
            path_from_repo_rel(row["candidate_image"]),
            f"train analog {row['candidate_class']}",
            f"dist {float(row['distance_l2']):.2f}",
            width,
        )
        pair = Image.new("RGB", (width * 2, max(query_thumb.height, candidate_thumb.height)), (235, 235, 235))
        pair.paste(query_thumb, (0, 0))
        pair.paste(candidate_thumb, (width, 0))
        cells.append(pair)

    columns = max(1, columns)
    rows_count = int(np.ceil(len(cells) / columns))
    cell_w = max(cell.width for cell in cells)
    cell_h = max(cell.height for cell in cells)
    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows_count), (230, 230, 230))
    for index, cell in enumerate(cells):
        x = (index % columns) * cell_w
        y = (index // columns) * cell_h
        sheet.paste(cell, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gap_path = resolve(args.gap_summary)
    gap = load_gap(gap_path, args.model_label)
    model_path = resolve(args.model) if args.model else path_from_repo_rel(str(gap["model"]))
    layer = str(args.layer if args.layer is not None else gap["nearest_layer"])
    imgsz = int(args.imgsz if args.imgsz is not None else gap.get("imgsz", 416))
    candidate_data = resolve(args.candidate_data)

    query_rows = query_records_from_gap(gap, args.top_query)
    wanted_classes = {str(row["class_name"]) for row in query_rows}
    candidate_rows, _names = dataset_records(
        data_path=candidate_data,
        split=args.candidate_split,
        domain="candidate",
        include_background=False,
    )
    candidate_rows = cap_candidates(
        candidate_rows,
        wanted_classes=wanted_classes,
        max_per_class=args.max_candidates_per_class,
    )
    remap_class_ids(query_rows, candidate_rows)
    if not candidate_rows:
        raise SystemExit("No candidate train rows after class filtering")

    records = query_rows + candidate_rows
    device = choose_device(args.device)
    print(
        f"[analogs] queries={len(query_rows)} candidates={len(candidate_rows)} "
        f"layer={layer} imgsz={imgsz} batch={args.batch} device={device}",
        flush=True,
    )
    features = extract_features(
        model_path=model_path,
        records=records,
        layers=[int(layer)],
        imgsz=imgsz,
        batch_size=args.batch,
        device=device,
    )[layer]
    x = StandardScaler().fit_transform(features)
    query_x = x[: len(query_rows)]
    candidate_x = x[len(query_rows) :]

    distances = pairwise_distances(query_x, candidate_x, metric="euclidean")
    candidate_indexes_by_class: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(candidate_rows):
        candidate_indexes_by_class[str(row["class_name"])].append(index)

    match_rows: list[dict[str, Any]] = []
    unique_candidates: dict[str, dict[str, Any]] = {}
    for query_index, query in enumerate(query_rows):
        if args.allow_cross_class:
            candidate_indexes = list(range(len(candidate_rows)))
        else:
            candidate_indexes = candidate_indexes_by_class.get(str(query["class_name"]), [])
        if not candidate_indexes:
            continue
        ordered = sorted(candidate_indexes, key=lambda idx: float(distances[query_index, idx]))
        for rank, candidate_index in enumerate(ordered[: args.per_query], start=1):
            candidate = candidate_rows[candidate_index]
            row = {
                "query_rank": query["query_rank"],
                "rank_for_query": rank,
                "query_image": query["image_rel"],
                "query_class": query["class_name"],
                "query_gap_nearest_l2": round(float(query["query_nearest_l2"]), 6),
                "candidate_image": candidate["image_rel"],
                "candidate_class": candidate["class_name"],
                "candidate_box_count": candidate["box_count"],
                "distance_l2": round(float(distances[query_index, candidate_index]), 6),
            }
            match_rows.append(row)
            key = str(candidate["image_rel"])
            current = unique_candidates.get(key)
            if current is None or float(row["distance_l2"]) < float(current["nearest_query_distance_l2"]):
                unique_candidates[key] = {
                    "image": candidate["image_rel"],
                    "class_id": candidate["class_id"],
                    "class_name": candidate["class_name"],
                    "box_count": candidate["box_count"],
                    "nearest_query_image": query["image_rel"],
                    "nearest_query_class": query["class_name"],
                    "nearest_query_rank": query["query_rank"],
                    "nearest_query_distance_l2": row["distance_l2"],
                }

    anchor_rows = sorted(
        unique_candidates.values(),
        key=lambda row: (str(row["class_name"]), float(row["nearest_query_distance_l2"]), str(row["image"])),
    )
    manifest_path = out_dir / "train_anchor_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in anchor_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    write_csv(out_dir / "query_to_train_analogs.csv", match_rows)
    write_csv(out_dir / "train_anchor_manifest.csv", anchor_rows)
    write_pair_sheet(
        path=out_dir / "query_train_analog_pairs.jpg",
        rows=match_rows,
        query_by_image={str(row["image_rel"]): row for row in query_rows},
        width=args.sheet_width,
        columns=args.sheet_columns,
    )

    class_counts = Counter(row["class_name"] for row in anchor_rows)
    summary = {
        "schema": "cashsnap_representation_gap_train_analogs_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gap_summary": repo_rel(gap_path),
        "model": repo_rel(model_path),
        "candidate_data": repo_rel(candidate_data),
        "candidate_split": args.candidate_split,
        "layer": layer,
        "imgsz": imgsz,
        "queries": len(query_rows),
        "candidate_pool": len(candidate_rows),
        "matches": len(match_rows),
        "unique_train_anchors": len(anchor_rows),
        "anchor_counts_by_class": dict(sorted(class_counts.items())),
        "top_train_anchors": anchor_rows[:30],
        "outputs": {
            "query_to_train_analogs_csv": repo_rel(out_dir / "query_to_train_analogs.csv"),
            "train_anchor_manifest_jsonl": repo_rel(manifest_path),
            "pair_sheet": repo_rel(out_dir / "query_train_analog_pairs.jpg"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"ok: anchors={len(anchor_rows)} matches={len(match_rows)} "
        f"summary={repo_rel(out_dir / 'summary.json')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
