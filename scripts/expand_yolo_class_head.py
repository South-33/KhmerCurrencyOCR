#!/usr/bin/env python
"""Expand a YOLO detect head by appending class-logit rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from local_runtime import configure_project_cache


configure_project_cache()

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-checkpoint", type=Path, required=True)
    parser.add_argument("--new-class-name", default="UNKNOWN_FOREIGN_NOTE")
    parser.add_argument("--new-class-id", type=int, default=13)
    parser.add_argument("--old-class-count", type=int, default=13)
    parser.add_argument("--summary-json", type=Path, default=None)
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
    resolved = resolve(path)
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise SystemExit(f"{repo_rel(resolved)}: expected Ultralytics checkpoint with a model")
    return checkpoint


def normalized_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    return {}


def expanded_conv(old: nn.Conv2d, *, old_class_count: int, new_class_id: int) -> nn.Conv2d:
    if old.out_channels != old_class_count:
        raise SystemExit(f"expected class conv with {old_class_count} outputs, got {old.out_channels}")
    if new_class_id != old_class_count:
        raise SystemExit("this expander appends one class; --new-class-id must equal --old-class-count")
    new = nn.Conv2d(
        in_channels=old.in_channels,
        out_channels=old_class_count + 1,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )
    new = new.to(device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight[:old_class_count].copy_(old.weight)
        new.weight[new_class_id].copy_(old.weight.mean(dim=0))
        if old.bias is not None and new.bias is not None:
            new.bias[:old_class_count].copy_(old.bias)
            new.bias[new_class_id].copy_(old.bias.min() - 1.0)
    return new


def expand_detect_head(model: Any, *, old_class_count: int, new_class_id: int) -> dict[str, Any]:
    detect = model.model[-1]
    touched: list[str] = []
    for attr_name in ("cv3", "one2one_cv3"):
        branches = getattr(detect, attr_name, None)
        if branches is None:
            continue
        for branch_index, branch in enumerate(branches):
            old_layer = branch[-1]
            if not isinstance(old_layer, nn.Conv2d):
                raise SystemExit(f"{attr_name}.{branch_index} final layer is not Conv2d: {type(old_layer)}")
            branch[-1] = expanded_conv(
                old_layer,
                old_class_count=old_class_count,
                new_class_id=new_class_id,
            )
            touched.append(f"model.{len(model.model) - 1}.{attr_name}.{branch_index}.2")
    if not touched:
        raise SystemExit("no detect class heads were expanded")

    new_class_count = old_class_count + 1
    model.nc = new_class_count
    detect.nc = new_class_count
    if hasattr(detect, "no"):
        detect.no = int(getattr(detect, "no")) + 1
    names = normalized_names(getattr(model, "names", {}))
    names[new_class_id] = str(getattr(model, "new_class_name", "UNKNOWN_FOREIGN_NOTE"))
    model.names = names
    if isinstance(getattr(model, "yaml", None), dict):
        model.yaml["nc"] = new_class_count
        model.yaml["names"] = names
    return {
        "expanded_layers": touched,
        "new_class_count": new_class_count,
        "unknown_weight_init": "mean_target_class_weights",
        "unknown_bias_init": "min_target_bias_minus_1",
    }


def main() -> int:
    args = parse_args()
    if args.new_class_id != args.old_class_count:
        raise SystemExit("--new-class-id must equal --old-class-count for append-only expansion")

    checkpoint_path = resolve(args.checkpoint)
    out_checkpoint = resolve(args.out_checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    model = checkpoint["model"].float()
    setattr(model, "new_class_name", args.new_class_name)
    names = normalized_names(getattr(model, "names", {}))
    if len(names) != args.old_class_count:
        raise SystemExit(f"expected {args.old_class_count} class names, got {len(names)}")
    if args.new_class_id in names:
        raise SystemExit(f"class id {args.new_class_id} already exists")

    summary = expand_detect_head(
        model,
        old_class_count=args.old_class_count,
        new_class_id=args.new_class_id,
    )
    names = normalized_names(getattr(model, "names", {}))
    names[args.new_class_id] = args.new_class_name
    model.names = names
    if isinstance(getattr(model, "yaml", None), dict):
        model.yaml["names"] = names

    checkpoint["model"] = model
    checkpoint["ema"] = None
    checkpoint["optimizer"] = None
    checkpoint["train_args"] = {
        **dict(checkpoint.get("train_args") or {}),
        "cashsnap_class_head_expansion": f"appended {args.new_class_id}:{args.new_class_name}",
    }

    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_checkpoint)
    summary.update(
        {
            "schema": "cashsnap_yolo_class_head_expand_v1",
            "checkpoint": repo_rel(checkpoint_path),
            "out_checkpoint": repo_rel(out_checkpoint),
            "new_class_id": args.new_class_id,
            "new_class_name": args.new_class_name,
            "names": names,
        }
    )
    summary_path = resolve(args.summary_json) if args.summary_json else out_checkpoint.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
