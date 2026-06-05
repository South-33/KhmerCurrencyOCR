#!/usr/bin/env python
"""Prepare mined real-dataset stress candidates for benchmark-style visual review.

This intentionally writes a separate review package instead of promoting rows into
the official real fan benchmark. Existing cashsnap_v1 labels are useful draft
anchors, but they are not protected holdout proof until the image source/split,
visible labels, and per-box quality rows are reviewed.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_CSV = ROOT / "runs" / "cashsnap" / "real_dataset_stress_candidate_review_latest.csv"
DEFAULT_SOURCES_OUT = ROOT / "runs" / "cashsnap" / "mined_real_benchmark_review_sources_latest.csv"
DEFAULT_TASKS_OUT = ROOT / "runs" / "cashsnap" / "mined_real_benchmark_review_label_tasks_latest.csv"
DEFAULT_DRAFT_LABEL_DIR = ROOT / "data" / "real_fan_benchmark" / "mined_cashsnap_v1" / "drafts"
DEFAULT_LABEL_DIR = ROOT / "data" / "real_fan_benchmark" / "mined_cashsnap_v1" / "labels" / "val"
DEFAULT_PREVIEW_DIR = ROOT / "data" / "real_fan_benchmark" / "mined_cashsnap_v1" / "previews"
DEFAULT_REVIEW_INDEX = ROOT / "data" / "real_fan_benchmark" / "mined_cashsnap_v1" / "review_index.html"
DEFAULT_SUMMARY_OUT = ROOT / "runs" / "cashsnap" / "mined_real_benchmark_review_latest.json"

DEFAULT_SCENES = [
    "khr_5000_face_number_overlap",
    "simple_overlap",
    "same_denomination_fan",
    "thin_slice_khr_5000",
    "thin_slice_khr_20000",
    "weak_khr_20000",
    "weak_khr_50000",
]

ROLE_BY_SCENE = {
    "khr_5000_face_number_overlap": "dense_overlap_stress",
    "simple_overlap": "dense_overlap_stress",
    "same_denomination_fan": "fan_stress",
    "thin_slice_khr_5000": "thin_edge_stress",
    "thin_slice_khr_20000": "thin_edge_stress",
    "weak_khr_20000": "weak_class_stress",
    "weak_khr_50000": "weak_class_stress",
}

PRIORITY_BY_SCENE = {
    "khr_5000_face_number_overlap": 1,
    "simple_overlap": 1,
    "same_denomination_fan": 2,
    "thin_slice_khr_5000": 2,
    "thin_slice_khr_20000": 2,
    "weak_khr_20000": 3,
    "weak_khr_50000": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--sources-out", type=Path, default=DEFAULT_SOURCES_OUT)
    parser.add_argument("--tasks-out", type=Path, default=DEFAULT_TASKS_OUT)
    parser.add_argument("--draft-label-dir", type=Path, default=DEFAULT_DRAFT_LABEL_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument("--review-index-out", type=Path, default=DEFAULT_REVIEW_INDEX)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--scene", action="append", dest="scenes", help="Scene type to include; repeatable.")
    parser.add_argument("--max-per-scene", type=int, default=3)
    parser.add_argument(
        "--prefer-splits",
        default="test,val,train",
        help="Comma-separated split preference before score sorting.",
    )
    parser.add_argument("--allow-duplicate-origins", action="store_true")
    parser.add_argument("--allow-duplicate-images", action="store_true")
    parser.add_argument("--skip-previews", action="store_true")
    parser.add_argument("--skip-check", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return resolve(path).relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolve(path))


def read_csv(path: Path) -> list[dict[str, str]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output = resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def slug(value: str, *, max_len: int = 56) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return (cleaned or "unknown")[:max_len].strip("_")


def float_value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except ValueError:
        return 0.0


def int_value(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "") or 0))
    except ValueError:
        return 0


def split_ranker(prefer_splits: str) -> dict[str, int]:
    return {split.strip(): index for index, split in enumerate(prefer_splits.split(",")) if split.strip()}


def selected_candidates(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    scenes = args.scenes or DEFAULT_SCENES
    split_rank = split_ranker(args.prefer_splits)
    selected: list[dict[str, str]] = []
    selected_images: set[str] = set()
    selected_origins: set[str] = set()

    for scene in scenes:
        scene_rows = [
            row
            for row in rows
            if row.get("scene_type") == scene
            and int_value(row, "box_count") > 0
            and resolve(Path(row.get("image", ""))).exists()
            and resolve(Path(row.get("label", ""))).exists()
        ]
        scene_rows.sort(
            key=lambda row: (
                -float_value(row, "score"),
                split_rank.get(row.get("split", ""), 999),
                -int_value(row, "box_count"),
                row.get("origin_key", ""),
                row.get("image", ""),
            )
        )

        scene_selected = 0
        for row in scene_rows:
            origin_key = row.get("origin_key", "")
            image_path = row.get("image", "")
            if not args.allow_duplicate_images and image_path in selected_images:
                continue
            if not args.allow_duplicate_origins and origin_key in selected_origins:
                continue
            rank = scene_selected + 1
            image_id = f"cashsnapv1_{slug(scene, max_len=34)}_{rank:02d}_{slug(origin_key)}"
            selected.append({**row, "review_image_id": image_id})
            selected_images.add(image_path)
            selected_origins.add(origin_key)
            scene_selected += 1
            if scene_selected >= args.max_per_scene:
                break

    return selected


def source_row(row: dict[str, str]) -> dict[str, str]:
    scene = row["scene_type"]
    return {
        "image_id": row["review_image_id"],
        "local_path": repo_path(Path(row["image"])),
        "source_page": "data/cashsnap_v1/data.yaml",
        "source_image": repo_path(Path(row["image"])),
        "source_credit": "existing local cashsnap_v1 dataset",
        "license_status": "local_dataset_review_required",
        "benchmark_status": "candidate_from_existing_dataset",
        "label_status": "draft_from_existing_label",
        "benchmark_role": ROLE_BY_SCENE.get(scene, scene),
        "notes": (
            f"Mined {scene} candidate from cashsnap_v1 {row.get('split', '')} split; "
            "draft labels are existing dataset labels and require visual/per-box quality review. "
            "Do not count as protected real holdout proof if this image can appear in training configs."
        ),
    }


def task_row(row: dict[str, str]) -> dict[str, str]:
    scene = row["scene_type"]
    denominations = row.get("denominations", "")
    return {
        "image_id": row["review_image_id"],
        "local_path": repo_path(Path(row["image"])),
        "priority": str(PRIORITY_BY_SCENE.get(scene, 4)),
        "label_status": "draft_from_existing_label",
        "benchmark_role": ROLE_BY_SCENE.get(scene, scene),
        "annotation_rule": (
            "Audit every existing YOLO box as a visible-region denomination label; "
            "drop boxes that are not human-legible or not in the CashSnap class schema."
        ),
        "notes": (
            f"Scene={scene}; split={row.get('split', '')}; denominations={denominations or 'unknown'}; "
            f"score={row.get('score', '')}; boxes={row.get('box_count', '')}; "
            f"overlap_pairs={row.get('overlap_pair_count', '')}; thin_or_small={row.get('thin_or_small_count', '')}."
        ),
    }


def render_preview(image: Path, labels: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/render_yolo_label_preview.py",
            "--image",
            repo_path(image),
            "--labels",
            repo_path(labels),
            "--out",
            repo_path(out),
        ],
        cwd=ROOT,
        check=True,
    )


def prune_stale_outputs(selected_ids: set[str], draft_dir: Path, preview_dir: Path) -> None:
    for label_path in draft_dir.glob("cashsnapv1_*.txt"):
        if label_path.stem not in selected_ids:
            label_path.unlink()
    for preview_path in preview_dir.glob("cashsnapv1_*.jpg"):
        if preview_path.stem not in selected_ids:
            preview_path.unlink()


def local_href(path: Path, out_dir: Path) -> str:
    return Path(os.path.relpath(resolve(path), out_dir)).as_posix()


def write_review_index(path: Path, selected: list[dict[str, str]], draft_dir: Path, preview_dir: Path) -> None:
    output = resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output.parent
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        groups[row["scene_type"]].append(row)

    sections: list[str] = []
    scene_order = [scene for scene in DEFAULT_SCENES if scene in groups] + sorted(set(groups) - set(DEFAULT_SCENES))
    for scene in scene_order:
        rows = groups.get(scene, [])
        if not rows:
            continue
        cards: list[str] = []
        for row in rows:
            image_id = row["review_image_id"]
            image_path = resolve(Path(row["image"]))
            label_path = resolve(draft_dir) / f"{image_id}.txt"
            preview_path = resolve(preview_dir) / f"{image_id}.jpg"
            preview = preview_path if preview_path.exists() else image_path
            cards.append(
                f"""
          <article class="item">
            <a class="thumb" href="{html.escape(local_href(preview, out_dir))}">
              <img src="{html.escape(local_href(preview, out_dir))}" alt="{html.escape(image_id)} preview" />
            </a>
            <div class="body">
              <div class="badge">{html.escape(row.get("split", ""))} / score {html.escape(row.get("score", ""))}</div>
              <h2>{html.escape(image_id)}</h2>
              <p><strong>Role:</strong> {html.escape(ROLE_BY_SCENE.get(scene, scene))}</p>
              <p><strong>Origin:</strong> {html.escape(row.get("origin_key", ""))}</p>
              <p><strong>Denoms:</strong> {html.escape(row.get("denominations", "") or "unknown")} / <strong>Boxes:</strong> {html.escape(row.get("box_count", ""))}</p>
              <p><strong>Scene hints:</strong> overlap={html.escape(row.get("overlap_pair_count", ""))}, thin/small={html.escape(row.get("thin_or_small_count", ""))}, same-class-max={html.escape(row.get("same_class_max", ""))}</p>
              <p class="warn">Draft labels are copied from cashsnap_v1. Review boxes, classes, rights, and train/holdout contamination before promotion.</p>
              <div class="actions">
                <a class="button" href="{html.escape(local_href(preview, out_dir))}">Open overlay</a>
                <a class="button secondary" href="{html.escape(local_href(image_path, out_dir))}">Open original</a>
                <a class="button secondary" href="{html.escape(local_href(label_path, out_dir))}">Open draft label</a>
              </div>
            </div>
          </article>
                """
            )
        sections.append(f"<section><h2>{html.escape(scene)}</h2>{''.join(cards)}</section>")

    output.write_text(
        f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Mined CashSnap Real Stress Review</title>
    <style>
      :root {{ font-family: "Segoe UI", system-ui, sans-serif; color: #161917; background: #f4f6f3; }}
      body {{ margin: 0; padding: 18px; }}
      main {{ width: min(1240px, 100%); margin: 0 auto; }}
      h1 {{ margin: 0 0 8px; font-size: 26px; letter-spacing: 0; }}
      main > p {{ margin: 0 0 18px; max-width: 920px; color: #4b544e; }}
      section {{ margin: 0 0 22px; }}
      section > h2 {{ font-size: 20px; margin: 0 0 10px; }}
      .item {{ display: grid; grid-template-columns: minmax(280px, 430px) 1fr; gap: 16px; padding: 14px; margin-bottom: 12px; border: 1px solid #cdd5cf; border-radius: 8px; background: white; }}
      .thumb img {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #eef1ee; }}
      .body h2 {{ margin: 0 0 8px; font-size: 17px; letter-spacing: 0; overflow-wrap: anywhere; }}
      p {{ margin: 0 0 8px; color: #4b544e; }}
      .warn {{ color: #6c3f12; }}
      .badge {{ display: inline-block; margin-bottom: 8px; border-radius: 6px; background: #e7f1ea; color: #205f35; padding: 5px 8px; font-weight: 700; font-size: 13px; }}
      .button {{ display: inline-block; border-radius: 6px; background: #161917; color: white; padding: 8px 11px; text-decoration: none; margin: 0 6px 6px 0; }}
      .button.secondary {{ background: #e8ece9; color: #161917; }}
      @media (max-width: 760px) {{ .item {{ grid-template-columns: 1fr; }} }}
    </style>
  </head>
  <body>
    <main>
      <h1>Mined CashSnap Real Stress Review</h1>
      <p>This is a diagnostic review queue built from existing cashsnap_v1 labels. It is useful for finding visual stress anchors, but it is not protected real-transfer proof until each image is visually audited, quality-scored, rights-reviewed, and excluded from training where needed.</p>
      {''.join(sections)}
    </main>
  </body>
</html>
""",
        encoding="utf-8",
    )


def run_benchmark_check(args: argparse.Namespace) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/check_real_fan_benchmark.py",
            "--sources",
            str(resolve(args.sources_out)),
            "--tasks",
            str(resolve(args.tasks_out)),
            "--draft-label-dir",
            str(resolve(args.draft_label_dir)),
            "--label-dir",
            str(resolve(args.label_dir)),
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    args = parse_args()
    if args.max_per_scene < 1:
        raise SystemExit("--max-per-scene must be positive")

    rows = read_csv(args.review_csv)
    selected = selected_candidates(args, rows)
    if not selected:
        raise SystemExit("no mined candidates selected")

    draft_dir = resolve(args.draft_label_dir)
    label_dir = resolve(args.label_dir)
    preview_dir = resolve(args.preview_dir)
    draft_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = {row["review_image_id"] for row in selected}
    prune_stale_outputs(selected_ids, draft_dir, preview_dir)

    sources = [source_row(row) for row in selected]
    tasks = [task_row(row) for row in selected]
    write_csv(
        args.sources_out,
        ["image_id", "local_path", "source_page", "source_image", "source_credit", "license_status", "benchmark_status", "label_status", "benchmark_role", "notes"],
        sources,
    )
    write_csv(
        args.tasks_out,
        ["image_id", "local_path", "priority", "label_status", "benchmark_role", "annotation_rule", "notes"],
        tasks,
    )

    for row in selected:
        image_id = row["review_image_id"]
        source_label = resolve(Path(row["label"]))
        draft_label = draft_dir / f"{image_id}.txt"
        shutil.copyfile(source_label, draft_label)
        if not args.skip_previews:
            render_preview(resolve(Path(row["image"])), draft_label, preview_dir / f"{image_id}.jpg")

    write_review_index(args.review_index_out, selected, draft_dir, preview_dir)
    if not args.skip_check:
        run_benchmark_check(args)

    counts = Counter(row["scene_type"] for row in selected)
    summary = {
        "review_csv": repo_path(args.review_csv),
        "sources_out": repo_path(args.sources_out),
        "tasks_out": repo_path(args.tasks_out),
        "draft_label_dir": repo_path(args.draft_label_dir),
        "label_dir": repo_path(args.label_dir),
        "preview_dir": repo_path(args.preview_dir),
        "review_index": repo_path(args.review_index_out),
        "selected_total": len(selected),
        "selected_by_scene": dict(sorted(counts.items())),
        "policy": {
            "promotion_status": "diagnostic_review_only",
            "reason": "Existing cashsnap_v1 labels are draft review anchors, not protected real-transfer proof without visual quality rows and train/holdout separation.",
        },
    }
    summary_out = resolve(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"ok: prepared {len(selected)} mined real review candidate(s)")
    for scene, count in sorted(counts.items()):
        print(f"- {scene}: {count}")
    print(f"sources={repo_path(args.sources_out)}")
    print(f"tasks={repo_path(args.tasks_out)}")
    print(f"review_index={repo_path(args.review_index_out)}")
    print(f"summary={repo_path(args.summary_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
