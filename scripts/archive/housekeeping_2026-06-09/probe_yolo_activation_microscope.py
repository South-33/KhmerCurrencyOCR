#!/usr/bin/env python
"""Compare YOLO real-vs-synthetic activations across models.

This is the mechanism-focused follow-up to probe_yolo_representation_domain_gap.py.
It keeps one sampled real/synthetic record set fixed, probes several checkpoints,
then exports layer separability, worst classes, nearest-neighbor gaps, and simple
domain-separator evidence overlays for the most uncovered real examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probe_yolo_representation_domain_gap import (
    ROOT,
    choose_device,
    dataset_records,
    extract_features,
    layer_metrics,
    load_letterboxed_tensor,
    nearest_exports,
    parse_layers,
    per_class_gaps,
    repo_rel,
    resolve,
    sample_records,
    write_csv,
)
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="label=weights.pt; repeatable.")
    parser.add_argument("--real-data", required=True, type=Path)
    parser.add_argument("--real-split", default="test")
    parser.add_argument("--synthetic-data", required=True, type=Path)
    parser.add_argument("--synthetic-split", default="train")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--layers", default="0,2,6,10,22")
    parser.add_argument("--nearest-layer", default=None)
    parser.add_argument("--heatmap-layer", default=None)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-per-class", type=int, default=6)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--min-per-class", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--heatmap-top-k", type=int, default=4)
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument("--no-class-balance", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def safe_clean(path: Path) -> None:
    resolved = path.resolve()
    allowed = (ROOT / "runs").resolve()
    if not (resolved == allowed or allowed in resolved.parents):
        raise SystemExit(f"Refusing to clean outside runs/: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def parse_model_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = resolve(Path(raw))
        return path.stem, path
    label, path_raw = raw.split("=", 1)
    label = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label.strip()).strip("._")
    if not label:
        raise SystemExit(f"Empty model label in {raw!r}")
    return label, resolve(Path(path_raw))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def spatial_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, (list, tuple)):
        tensors = [item for item in output if isinstance(item, torch.Tensor)]
        if not tensors:
            return None
        output = tensors[0]
    if not isinstance(output, torch.Tensor):
        return None
    tensor = output.detach().float()
    return tensor if tensor.ndim == 4 else None


def extract_layer_maps(
    *,
    model_path: Path,
    records: list[dict[str, Any]],
    layer: int,
    imgsz: int,
    device: torch.device,
) -> list[np.ndarray | None]:
    yolo = YOLO(str(model_path))
    model = yolo.model.to(device).eval()
    modules = list(model.model)
    if layer < 0 or layer >= len(modules):
        raise SystemExit(f"Layer {layer} outside model range 0..{len(modules) - 1}")

    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor = spatial_tensor(output)
        if tensor is not None:
            captured["map"] = tensor.cpu()

    handle = modules[layer].register_forward_hook(hook)
    maps: list[np.ndarray | None] = []
    try:
        with torch.inference_mode():
            for row in records:
                batch = load_letterboxed_tensor(Path(row["image"]), imgsz).unsqueeze(0).to(device)
                captured.clear()
                _ = model(batch)
                if "map" not in captured:
                    maps.append(None)
                else:
                    maps.append(captured["map"][0].numpy())
    finally:
        handle.remove()
    return maps


def fit_separator(features: np.ndarray, records: list[dict[str, Any]], seed: int) -> tuple[StandardScaler, LogisticRegression, np.ndarray]:
    domains = np.array([0 if row["domain"] == "real" else 1 for row in records], dtype=np.int64)
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(x, domains)
    return scaler, clf, x


def channel_rows(clf: LogisticRegression, x: np.ndarray, records: list[dict[str, Any]], top_k: int = 20) -> list[dict[str, Any]]:
    domains = np.array([0 if row["domain"] == "real" else 1 for row in records], dtype=np.int64)
    coef = clf.coef_[0]
    real_mean = x[domains == 0].mean(axis=0)
    synth_mean = x[domains == 1].mean(axis=0)
    rows = []
    for index, weight in enumerate(coef):
        rows.append(
            {
                "channel": index,
                "separator_weight_synth_positive": float(weight),
                "real_mean_z": float(real_mean[index]),
                "synthetic_mean_z": float(synth_mean[index]),
                "mean_delta_synth_minus_real": float(synth_mean[index] - real_mean[index]),
                "abs_weight": float(abs(weight)),
                "direction": "synthetic" if weight > 0 else "real",
            }
        )
    rows.sort(key=lambda row: row["abs_weight"], reverse=True)
    return rows[:top_k]


def evidence_overlay(
    *,
    image_path: Path,
    fmap: np.ndarray,
    scaler: StandardScaler,
    clf: LogisticRegression,
    imgsz: int,
) -> Image.Image | None:
    if fmap.ndim != 3:
        return None
    channels = fmap.shape[0]
    coef = clf.coef_[0]
    if channels != len(coef):
        return None
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    scale = np.where(scale == 0, 1.0, scale)
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    fmap_z = (fmap - mean[:, None, None]) / scale[:, None, None]
    evidence = np.tensordot(coef.astype(np.float32), fmap_z, axes=(0, 0))
    evidence = evidence - float(np.median(evidence))
    max_abs = max(float(np.max(np.abs(evidence))), 1e-6)
    evidence = np.clip(evidence / max_abs, -1.0, 1.0)

    heat = Image.fromarray(((evidence + 1.0) * 127.5).astype(np.uint8), mode="L").resize(
        (imgsz, imgsz),
        Image.Resampling.BILINEAR,
    )
    heat_arr = (np.asarray(heat, dtype=np.float32) / 127.5) - 1.0
    pos = np.clip(heat_arr, 0.0, 1.0)
    neg = np.clip(-heat_arr, 0.0, 1.0)

    base = load_letterboxed_tensor(image_path, imgsz).permute(1, 2, 0).numpy()
    color = np.zeros_like(base)
    color[..., 0] = pos
    color[..., 2] = neg
    alpha = np.clip(np.abs(heat_arr)[..., None] * 0.55, 0.0, 0.55)
    blended = (base * (1.0 - alpha) + color * alpha) * 255.0
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def labeled_canvas(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    label_h = 48
    canvas = Image.new("RGB", (image.width, image.height + label_h), (245, 245, 245))
    canvas.paste(image, (0, label_h))
    return _draw_text(canvas, title, subtitle)


def _draw_text(canvas: Image.Image, title: str, subtitle: str) -> Image.Image:
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("arial.ttf", 15)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    draw.text((8, 6), title[:78], fill=(20, 20, 20), font=title_font)
    draw.text((8, 27), subtitle[:95], fill=(70, 70, 70), font=small_font)
    return canvas


def write_pair_overlay(
    *,
    out_path: Path,
    real_overlay: Image.Image,
    synthetic_overlay: Image.Image,
    real_title: str,
    real_subtitle: str,
    synthetic_title: str,
    synthetic_subtitle: str,
) -> None:
    real_cell = labeled_canvas(real_overlay, real_title, real_subtitle)
    synth_cell = labeled_canvas(synthetic_overlay, synthetic_title, synthetic_subtitle)
    canvas = Image.new("RGB", (real_cell.width + synth_cell.width, max(real_cell.height, synth_cell.height)), (255, 255, 255))
    canvas.paste(real_cell, (0, 0))
    canvas.paste(synth_cell, (real_cell.width, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def record_by_rel(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["image_rel"]): row for row in records}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# YOLO Activation Microscope",
        "",
        "Red overlay is synthetic-domain evidence from the layer separator; blue overlay is real-domain evidence.",
        "The overlays explain the separator, not the class prediction directly.",
        "",
        "## Models",
        "",
        "| Model | Path |",
        "| --- | --- |",
    ]
    for model in payload["models"]:
        lines.append(f"| {model['label']} | `{model['path']}` |")

    lines.extend(["", "## Layer Domain Accuracy", "", "| Model | Layer | Domain acc | Proxy A-distance | Centroid/within |", "| --- | ---: | ---: | ---: | ---: |"])
    for model in payload["models"]:
        for row in model["layer_table"]:
            acc = row["domain_accuracy"]["mean"]
            acc_text = "" if acc is None else f"{acc:.3f}"
            lines.append(
                f"| {model['label']} | {row['layer']} | {acc_text} | "
                f"{row['domain_accuracy']['proxy_a_distance']:.3f} | {row['centroid_l2_over_within']:.3f} |"
            )

    lines.extend(["", "## Worst Classes At Nearest Layer", "", "| Model | Class | Real->Synth NN mean | Centroid L2 |", "| --- | --- | ---: | ---: |"])
    for model in payload["models"]:
        for row in model["per_class_gaps"][:8]:
            lines.append(
                f"| {model['label']} | {row['class_name']} | "
                f"{row['real_to_synthetic_nearest_l2_mean']:.3f} | {row['centroid_l2']:.3f} |"
            )

    lines.extend(["", "## Heatmap Pairs", ""])
    for model in payload["models"]:
        pairs = model.get("heatmap_pairs", [])
        if not pairs:
            continue
        lines.append(f"### {model['label']}")
        for pair in pairs:
            lines.append(f"- `{pair['pair_image']}`: {pair['real_class']} real vs nearest {pair['synthetic_class']} synth, gap `{pair['nearest_l2']:.3f}`")
        lines.append("")

    lines.extend(
        [
            "## Kill Criteria",
            "",
            "- If this only restates high real-vs-synth separability without stable visual obligations, do not keep expanding it.",
            "- If separator evidence does not correlate with real misses or FP modes, pivot to real capture/adaptation.",
            "- If the real-trained control remains similarly separable, separability alone is not the bottleneck; compare failure-linked activations instead.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    if args.clean:
        safe_clean(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_specs = [parse_model_spec(raw) for raw in args.model]
    for label, path in model_specs:
        if not path.exists():
            raise SystemExit(f"Missing checkpoint for {label}: {path}")

    layers = parse_layers(args.layers)
    nearest_layer = str(args.nearest_layer if args.nearest_layer is not None else layers[-1])
    heatmap_layer = int(args.heatmap_layer if args.heatmap_layer is not None else nearest_layer)
    if int(nearest_layer) not in layers:
        raise SystemExit("--nearest-layer must be included in --layers")
    if heatmap_layer not in layers:
        raise SystemExit("--heatmap-layer must be included in --layers")

    real_records, _ = dataset_records(
        data_path=resolve(args.real_data),
        split=args.real_split,
        domain="real",
        include_background=args.include_background,
    )
    synthetic_records, _ = dataset_records(
        data_path=resolve(args.synthetic_data),
        split=args.synthetic_split,
        domain="synthetic",
        include_background=args.include_background,
    )
    records, sample_info = sample_records(
        real_records,
        synthetic_records,
        seed=args.seed,
        max_per_class=args.max_per_class,
        max_total=args.max_total,
        class_balance=not args.no_class_balance,
    )
    if not records:
        raise SystemExit("No sampled records")
    write_rows(out_dir / "sample_records.csv", [{k: v for k, v in row.items() if k != "labels"} for row in records])

    device = choose_device(args.device)
    summary: dict[str, Any] = {
        "schema": "cashsnap_yolo_activation_microscope_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "real_data": repo_rel(resolve(args.real_data)),
        "real_split": args.real_split,
        "synthetic_data": repo_rel(resolve(args.synthetic_data)),
        "synthetic_split": args.synthetic_split,
        "imgsz": args.imgsz,
        "layers": layers,
        "nearest_layer": nearest_layer,
        "heatmap_layer": str(heatmap_layer),
        "sample_info": sample_info,
        "models": [],
    }

    records_by_rel = record_by_rel(records)
    for label, model_path in model_specs:
        print(f"model={label} extracting features", flush=True)
        raw_features = extract_features(
            model_path=model_path,
            records=records,
            layers=layers,
            imgsz=args.imgsz,
            batch_size=args.batch,
            device=device,
        )

        model_out = out_dir / label
        model_out.mkdir(parents=True, exist_ok=True)
        model_payload: dict[str, Any] = {
            "label": label,
            "path": repo_rel(model_path),
            "layer_table": [],
            "per_class_gaps": [],
            "top_uncovered_real": [],
            "top_uncovered_synthetic": [],
            "heatmap_pairs": [],
        }

        standardized_by_layer: dict[str, np.ndarray] = {}
        for layer in layers:
            layer_key = str(layer)
            metrics, x = layer_metrics(features=raw_features[layer_key], records=records, seed=args.seed)
            metrics["layer"] = layer_key
            model_payload["layer_table"].append(metrics)
            standardized_by_layer[layer_key] = x
        write_json(model_out / "layer_metrics.json", model_payload["layer_table"])

        nearest_x = standardized_by_layer[nearest_layer]
        class_gaps = per_class_gaps(x=nearest_x, records=records, min_per_class=args.min_per_class)
        top_real, top_synthetic = nearest_exports(x=nearest_x, records=records, top_k=args.top_k)
        model_payload["per_class_gaps"] = class_gaps
        model_payload["top_uncovered_real"] = top_real
        model_payload["top_uncovered_synthetic"] = top_synthetic
        write_csv(model_out / "per_class_gaps.csv", class_gaps)
        write_csv(model_out / "top_uncovered_real.csv", top_real)
        write_csv(model_out / "top_uncovered_synthetic.csv", top_synthetic)

        scaler, clf, x = fit_separator(raw_features[str(heatmap_layer)], records, args.seed)
        channels = channel_rows(clf, x, records)
        model_payload["top_separator_channels"] = channels
        write_csv(model_out / "top_separator_channels.csv", channels)

        heatmap_rows: list[dict[str, Any]] = []
        heatmap_records: list[dict[str, Any]] = []
        for rank, row in enumerate(top_real[: args.heatmap_top_k], start=1):
            real_record = records_by_rel[row["image"]]
            synth_record = records_by_rel[row["nearest_synthetic"]]
            heatmap_rows.append({"rank": rank, **row})
            heatmap_records.extend([real_record, synth_record])

        if heatmap_records:
            maps = extract_layer_maps(
                model_path=model_path,
                records=heatmap_records,
                layer=heatmap_layer,
                imgsz=args.imgsz,
                device=device,
            )
            for index, row in enumerate(heatmap_rows):
                real_record = heatmap_records[index * 2]
                synth_record = heatmap_records[index * 2 + 1]
                real_map = maps[index * 2]
                synth_map = maps[index * 2 + 1]
                if real_map is None or synth_map is None:
                    continue
                real_overlay = evidence_overlay(
                    image_path=Path(real_record["image"]),
                    fmap=real_map,
                    scaler=scaler,
                    clf=clf,
                    imgsz=args.imgsz,
                )
                synth_overlay = evidence_overlay(
                    image_path=Path(synth_record["image"]),
                    fmap=synth_map,
                    scaler=scaler,
                    clf=clf,
                    imgsz=args.imgsz,
                )
                if real_overlay is None or synth_overlay is None:
                    continue
                pair_path = model_out / "heatmaps" / f"rank{int(row['rank']):02d}_{row['class_name']}_pair.jpg"
                write_pair_overlay(
                    out_path=pair_path,
                    real_overlay=real_overlay,
                    synthetic_overlay=synth_overlay,
                    real_title=f"real {row['class_name']}",
                    real_subtitle=f"rank {row['rank']} gap {float(row['nearest_l2']):.3f}",
                    synthetic_title=f"nearest synth {row['nearest_synthetic_class']}",
                    synthetic_subtitle=row["nearest_synthetic"],
                )
                model_payload["heatmap_pairs"].append(
                    {
                        "rank": int(row["rank"]),
                        "pair_image": repo_rel(pair_path),
                        "real_image": row["image"],
                        "real_class": row["class_name"],
                        "synthetic_image": row["nearest_synthetic"],
                        "synthetic_class": row["nearest_synthetic_class"],
                        "nearest_l2": float(row["nearest_l2"]),
                    }
                )

        summary["models"].append(model_payload)

    write_json(out_dir / "summary.json", summary)
    write_markdown(out_dir / "summary.md", summary)
    print(f"wrote_summary={repo_rel(out_dir / 'summary.json')}", flush=True)
    print(f"wrote_markdown={repo_rel(out_dir / 'summary.md')}", flush=True)


if __name__ == "__main__":
    main()
