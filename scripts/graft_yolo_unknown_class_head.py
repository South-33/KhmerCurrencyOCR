#!/usr/bin/env python
"""Graft target-class logits from a 13-class YOLO checkpoint into a 14-class checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from local_runtime import configure_project_cache

configure_project_cache()

import torch


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-checkpoint", required=True, type=Path)
    parser.add_argument("--unknown-checkpoint", required=True, type=Path)
    parser.add_argument("--out-checkpoint", required=True, type=Path)
    parser.add_argument("--target-class-count", type=int, default=13)
    parser.add_argument("--unknown-class-id", type=int, default=13)
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(resolve(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise SystemExit(f"{repo_rel(resolve(path))}: expected Ultralytics checkpoint with a model")
    return checkpoint


def is_class_logits_key(key: str, target_shape: torch.Size, unknown_shape: torch.Size, target_class_count: int) -> bool:
    if not (key.endswith(".weight") or key.endswith(".bias")):
        return False
    if len(target_shape) not in (1, 4) or len(unknown_shape) != len(target_shape):
        return False
    if target_shape[0] != target_class_count or unknown_shape[0] <= target_class_count:
        return False
    return ".cv3." in key or ".one2one_cv3." in key


def graft_state_dict(
    target_state: dict[str, torch.Tensor],
    unknown_state: dict[str, torch.Tensor],
    *,
    target_class_count: int,
    unknown_class_id: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    grafted = copy.deepcopy(unknown_state)
    copied_target_rows: list[str] = []
    preserved_unknown_rows: list[str] = []
    skipped_shape_mismatches: dict[str, dict[str, list[int]]] = {}

    for key, unknown_tensor in unknown_state.items():
        target_tensor = target_state.get(key)
        if target_tensor is None:
            continue
        if target_tensor.shape == unknown_tensor.shape:
            continue
        if not is_class_logits_key(key, target_tensor.shape, unknown_tensor.shape, target_class_count):
            skipped_shape_mismatches[key] = {
                "target_shape": list(target_tensor.shape),
                "unknown_shape": list(unknown_tensor.shape),
            }
            continue
        if unknown_class_id >= unknown_tensor.shape[0]:
            raise SystemExit(f"{key}: unknown class id {unknown_class_id} outside shape {list(unknown_tensor.shape)}")
        updated = unknown_tensor.clone()
        updated[:target_class_count] = target_tensor[:target_class_count].to(updated.dtype)
        grafted[key] = updated
        copied_target_rows.append(key)
        preserved_unknown_rows.append(f"{key}[{unknown_class_id}]")

    if not copied_target_rows:
        raise SystemExit("no class-logit tensors were grafted")
    summary = {
        "schema": "cashsnap_yolo_unknown_class_head_graft_v1",
        "target_class_count": target_class_count,
        "unknown_class_id": unknown_class_id,
        "grafted_tensors": copied_target_rows,
        "preserved_unknown_rows": preserved_unknown_rows,
        "skipped_shape_mismatches": skipped_shape_mismatches,
    }
    return grafted, summary


def main() -> int:
    args = parse_args()
    target_checkpoint_path = resolve(args.target_checkpoint)
    unknown_checkpoint_path = resolve(args.unknown_checkpoint)
    out_checkpoint_path = resolve(args.out_checkpoint)

    target_checkpoint = load_checkpoint(target_checkpoint_path)
    out_checkpoint = load_checkpoint(unknown_checkpoint_path)
    target_model = target_checkpoint["model"].float()
    out_model = out_checkpoint["model"].float()

    grafted_state, summary = graft_state_dict(
        target_model.state_dict(),
        out_model.state_dict(),
        target_class_count=args.target_class_count,
        unknown_class_id=args.unknown_class_id,
    )
    out_model.load_state_dict(grafted_state, strict=True)
    out_checkpoint["model"] = out_model
    out_checkpoint["ema"] = None
    out_checkpoint["optimizer"] = None
    out_checkpoint["train_args"] = {
        **dict(out_checkpoint.get("train_args") or {}),
        "cashsnap_head_graft": "target rows from 13-class checkpoint, unknown row preserved from 14-class checkpoint",
    }

    out_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_checkpoint, out_checkpoint_path)

    summary.update(
        {
            "target_checkpoint": repo_rel(target_checkpoint_path),
            "unknown_checkpoint": repo_rel(unknown_checkpoint_path),
            "out_checkpoint": repo_rel(out_checkpoint_path),
        }
    )
    summary_path = resolve(args.summary_json) if args.summary_json else out_checkpoint_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
