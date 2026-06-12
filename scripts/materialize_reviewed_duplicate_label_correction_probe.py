#!/usr/bin/env python
"""Materialize paired YOLO train lists for reviewed duplicate-label corrections."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = Path(
    "configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_v1.yaml"
)
DEFAULT_BASE_TRAIN_LIST = Path(
    "configs/generated_lists/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_v1_train.txt"
)
DEFAULT_REVIEW_CSV = Path(
    "runs/cashsnap/real_visible_evidence_review_packet_v1/"
    "visible_evidence_review_packet_v1_manual_obvioussafe_reviewed.csv"
)
DEFAULT_OUT_DIR = Path("runs/cashsnap/real_visible_evidence_duplicate_label_correction_probe_v1")
DEFAULT_CANDIDATE_CONFIG = Path(
    "configs/webgl_ablation/"
    "cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_dupcount_correctedlabels_v1.yaml"
)
DEFAULT_CONTROL_CONFIG = Path(
    "configs/webgl_ablation/"
    "cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_dupcount_originallabels_control_v1.yaml"
)
DEFAULT_CANDIDATE_LIST = Path(
    "configs/generated_lists/webgl_ablation/"
    "cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_dupcount_correctedlabels_v1_train.txt"
)
DEFAULT_CONTROL_LIST = Path(
    "configs/generated_lists/webgl_ablation/"
    "cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_dupcount_originallabels_control_v1_train.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base-train-list", type=Path, default=DEFAULT_BASE_TRAIN_LIST)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--candidate-config", type=Path, default=DEFAULT_CANDIDATE_CONFIG)
    parser.add_argument("--control-config", type=Path, default=DEFAULT_CONTROL_CONFIG)
    parser.add_argument("--candidate-train-list", type=Path, default=DEFAULT_CANDIDATE_LIST)
    parser.add_argument("--control-train-list", type=Path, default=DEFAULT_CONTROL_LIST)
    parser.add_argument("--row-id", action="append", required=True, help="Review row id such as VE-046.")
    parser.add_argument("--iou-threshold", type=float, default=0.85)
    parser.add_argument(
        "--allow-new-images",
        action="store_true",
        help="Allow reviewed rows that are not already present in the base train list.",
    )
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


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def safe_clean_dir(path: Path) -> None:
    resolved = path.resolve()
    allowed = (ROOT / "runs" / "cashsnap").resolve()
    if not resolved.is_relative_to(allowed):
        raise SystemExit(f"--clean target must stay under {repo_rel(allowed)}: {repo_rel(resolved)}")
    if resolved.exists():
        shutil.rmtree(resolved)


def read_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(resolved)} must be a YAML mapping")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    resolved = resolve(path)
    with resolved.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_list(path: Path) -> list[str]:
    resolved = resolve(path)
    return [
        line.strip().replace("\\", "/")
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def write_list(path: Path, rows: list[str]) -> None:
    resolved = resolve(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def normalized(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def split_variants(raw: str, fallback: str) -> list[str]:
    values = [value.strip().replace("\\", "/") for value in raw.split("|") if value.strip()]
    if not values and fallback.strip():
        values = [fallback.strip().replace("\\", "/")]
    return list(dict.fromkeys(values))


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    for index, part in enumerate(parts):
        if part == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def read_labels(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise SystemExit(f"missing source label: {repo_rel(path)}")
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            raise SystemExit(f"{repo_rel(path)}:{line_no} expected YOLO class plus box fields")
        try:
            class_id = int(float(parts[0]))
            x, y, w, h = (float(value) for value in parts[1:5])
        except ValueError as exc:
            raise SystemExit(f"{repo_rel(path)}:{line_no} invalid YOLO label") from exc
        rows.append(
            {
                "class_id": class_id,
                "xywh": (x, y, w, h),
                "tail": parts[5:],
                "raw": " ".join(parts),
                "line_no": line_no,
                "area": w * h,
            }
        )
    return rows


def xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def dedupe_same_class(labels: list[dict[str, Any]], iou_threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for label in sorted(labels, key=lambda item: (-float(item["area"]), int(item["line_no"]))):
        duplicate_of: dict[str, Any] | None = None
        duplicate_iou = 0.0
        for existing in kept:
            if int(existing["class_id"]) != int(label["class_id"]):
                continue
            iou = box_iou(label["xywh"], existing["xywh"])
            if iou >= iou_threshold:
                duplicate_of = existing
                duplicate_iou = iou
                break
        if duplicate_of is None:
            kept.append(label)
        else:
            removed.append(
                {
                    "removed_line_no": label["line_no"],
                    "kept_line_no": duplicate_of["line_no"],
                    "class_id": label["class_id"],
                    "iou": round(duplicate_iou, 6),
                    "removed_raw": label["raw"],
                    "kept_raw": duplicate_of["raw"],
                }
            )
    return sorted(kept, key=lambda item: int(item["line_no"])), removed


def label_lines(labels: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for label in labels:
        x, y, w, h = label["xywh"]
        parts = [str(label["class_id"]), f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}", *label["tail"]]
        lines.append(" ".join(parts).rstrip())
    return lines


def unique_image_name(row_id: str, source_image: Path) -> str:
    digest = hashlib.sha1(repo_rel(source_image).encode("utf-8")).hexdigest()[:10]
    safe_row = "".join(ch if ch.isalnum() else "_" for ch in row_id).strip("_") or "row"
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source_image.stem)
    return f"{safe_row}_{safe_stem[:80]}_{digest}{source_image.suffix.lower()}"


def review_row_id(row: dict[str, str]) -> str:
    for key in ("review_id", "visible_review_id", "packet_id"):
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def select_review_images(
    rows: list[dict[str, str]],
    row_ids: set[str],
    base_rows: set[str],
    allow_new_images: bool,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    by_id = {review_row_id(row): row for row in rows}
    missing = sorted(row_ids - set(by_id))
    if missing:
        raise SystemExit(f"review row ids not found: {', '.join(missing)}")
    for row_id in sorted(row_ids):
        row = by_id[row_id]
        decision = normalized(row.get("review_decision", ""))
        route = normalized(row.get("final_route", "") or row.get("usable_as", ""))
        notes = normalized(row.get("review_notes", ""))
        if decision not in {"accept", "accepted", "approved", "reviewed", "use", "usable"}:
            raise SystemExit(f"{row_id} is not accepted review_decision={row.get('review_decision', '')!r}")
        if route not in {"exclude_duplicate_or_flat", "exclude"}:
            raise SystemExit(f"{row_id} must be an excluded duplicate/count row, got route={route!r}")
        if "duplicate_count_trap" not in notes:
            raise SystemExit(f"{row_id} review_notes must contain duplicate-count trap")
        for image in split_variants(row.get("variant_images", ""), row.get("image", "")):
            if image not in base_rows and not allow_new_images:
                continue
            if image in seen:
                continue
            seen.add(image)
            selected.append(
                {
                    "review_id": row_id,
                    "source_image": image,
                    "source_group": row.get("source_group", ""),
                    "present_in_base_train": image in base_rows,
                }
            )
    if not selected:
        raise SystemExit("no selected review images were present in the base train list")
    return selected


def materialize_clones(
    *,
    selected: list[dict[str, Any]],
    out_dir: Path,
    iou_threshold: float,
) -> tuple[list[str], list[str], list[dict[str, Any]], Counter[str]]:
    candidate_rows: list[str] = []
    control_rows: list[str] = []
    manifest: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    for item in selected:
        source_image = resolve(item["source_image"])
        if not source_image.exists():
            raise SystemExit(f"missing source image: {repo_rel(source_image)}")
        source_label = label_path_for_image(source_image)
        labels = read_labels(source_label)
        kept, removed = dedupe_same_class(labels, iou_threshold)
        if not removed:
            raise SystemExit(f"{repo_rel(source_label)} had no duplicate labels removed")

        out_name = unique_image_name(str(item["review_id"]), source_image)
        candidate_image = out_dir / "candidate_yolo" / "images" / "train" / out_name
        candidate_label = out_dir / "candidate_yolo" / "labels" / "train" / f"{candidate_image.stem}.txt"
        control_image = out_dir / "control_yolo" / "images" / "train" / out_name
        control_label = out_dir / "control_yolo" / "labels" / "train" / f"{control_image.stem}.txt"
        candidate_image.parent.mkdir(parents=True, exist_ok=True)
        candidate_label.parent.mkdir(parents=True, exist_ok=True)
        control_image.parent.mkdir(parents=True, exist_ok=True)
        control_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, candidate_image)
        shutil.copy2(source_image, control_image)
        candidate_label.write_text("\n".join(label_lines(kept)) + "\n", encoding="utf-8")
        control_label.write_text("\n".join(label["raw"] for label in labels) + "\n", encoding="utf-8")

        candidate_rows.append(repo_rel(candidate_image))
        control_rows.append(repo_rel(control_image))
        for label in kept:
            class_counts[str(label["class_id"])] += 1
        manifest.append(
            {
                "review_id": item["review_id"],
                "source_group": item["source_group"],
                "source_image": item["source_image"],
                "source_label": repo_rel(source_label),
                "candidate_image": repo_rel(candidate_image),
                "candidate_label": repo_rel(candidate_label),
                "control_image": repo_rel(control_image),
                "control_label": repo_rel(control_label),
                "original_labels": len(labels),
                "corrected_labels": len(kept),
                "removed_labels": removed,
            }
        )
    return candidate_rows, control_rows, manifest, class_counts


def write_config(
    *,
    source_config: dict[str, Any],
    source_config_path: Path,
    out_config: Path,
    train_list: Path,
    role: str,
    summary: dict[str, Any],
) -> None:
    payload = dict(source_config)
    payload["train"] = repo_rel(resolve(train_list))
    meta = {
        "schema": "cashsnap_reviewed_duplicate_label_correction_probe_v1",
        "role": role,
        "source_config": repo_rel(resolve(source_config_path)),
        "train_list": repo_rel(resolve(train_list)),
        **summary,
    }
    payload["cashsnap_duplicate_label_correction_probe"] = meta
    write_yaml(out_config, payload)


def main() -> int:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    if args.clean:
        safe_clean_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config = read_yaml(args.base_config)
    base_rows = read_list(args.base_train_list)
    base_set = set(base_rows)
    review_rows = read_csv(args.review_csv)
    selected = select_review_images(review_rows, set(args.row_id), base_set, args.allow_new_images)
    selected_source_rows = [item["source_image"] for item in selected]
    selected_set = set(selected_source_rows)
    removed_original_rows = sum(1 for item in selected if item["present_in_base_train"])

    candidate_clones, control_clones, manifest, class_counts = materialize_clones(
        selected=selected,
        out_dir=out_dir,
        iou_threshold=args.iou_threshold,
    )
    remaining = [row for row in base_rows if row not in selected_set]
    candidate_train = remaining + candidate_clones
    control_train = remaining + control_clones
    if len(candidate_train) != len(control_train):
        raise SystemExit("candidate/control train row counts must match each other")

    write_list(args.candidate_train_list, candidate_train)
    write_list(args.control_train_list, control_train)
    manifest_path = out_dir / "correction_manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in manifest) + "\n", encoding="utf-8")
    corrected_view_data = out_dir / "corrected_duplicate_view_data.yaml"
    corrected_view_payload = {
        "path": rel_between(out_dir, ROOT),
        "train": rel_between(ROOT, out_dir / "candidate_yolo" / "images" / "train"),
        "val": rel_between(ROOT, out_dir / "candidate_yolo" / "images" / "train"),
        "test": rel_between(ROOT, out_dir / "candidate_yolo" / "images" / "train"),
        "names": base_config.get("names", {}),
        "cashsnap_view": {
            "schema": "cashsnap_reviewed_duplicate_label_correction_view_v1",
            "source_summary": repo_rel(out_dir / "summary.json"),
            "label_policy": "candidate corrected labels; one countable KHR_500 per reviewed duplicate-count image",
        },
    }
    write_yaml(corrected_view_data, corrected_view_payload)

    summary = {
        "schema": "cashsnap_reviewed_duplicate_label_correction_probe_summary_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config": repo_rel(resolve(args.base_config)),
        "base_train_list": repo_rel(resolve(args.base_train_list)),
        "review_csv": repo_rel(resolve(args.review_csv)),
        "review_row_ids": sorted(set(args.row_id)),
        "iou_threshold": args.iou_threshold,
        "base_train_rows": len(base_rows),
        "allow_new_images": bool(args.allow_new_images),
        "removed_original_rows": removed_original_rows,
        "selected_review_images": len(selected_source_rows),
        "candidate_train_rows": len(candidate_train),
        "control_train_rows": len(control_train),
        "candidate_clone_rows": len(candidate_clones),
        "control_clone_rows": len(control_clones),
        "corrected_class_counts_by_id": dict(sorted(class_counts.items())),
        "manifest": repo_rel(manifest_path),
        "corrected_view_data": repo_rel(corrected_view_data),
        "candidate_train_list": repo_rel(resolve(args.candidate_train_list)),
        "control_train_list": repo_rel(resolve(args.control_train_list)),
        "candidate_config": repo_rel(resolve(args.candidate_config)),
        "control_config": repo_rel(resolve(args.control_config)),
        "intended_use": (
            "Paired diagnostic train probe: replace reviewed duplicate-count source labels with "
            "same-image corrected single-label clones, versus same-image original-label clones."
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config_summary = {
        key: summary[key]
        for key in [
            "schema",
            "review_row_ids",
            "iou_threshold",
            "removed_original_rows",
            "selected_review_images",
            "candidate_clone_rows",
            "control_clone_rows",
            "intended_use",
        ]
    }
    write_config(
        source_config=base_config,
        source_config_path=args.base_config,
        out_config=args.candidate_config,
        train_list=args.candidate_train_list,
        role="candidate_corrected_labels",
        summary=config_summary,
    )
    write_config(
        source_config=base_config,
        source_config_path=args.base_config,
        out_config=args.control_config,
        train_list=args.control_train_list,
        role="control_original_labels",
        summary=config_summary,
    )
    print(
        "duplicate_label_correction_probe="
        f"{repo_rel(out_dir)} selected={len(selected_source_rows)} removed={removed_original_rows} "
        f"candidate_rows={len(candidate_train)} control_rows={len(control_train)}"
    )
    print(f"summary={repo_rel(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
