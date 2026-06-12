#!/usr/bin/env python
"""Build a name-mapped official21 YOLO checkpoint from a core13 checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from local_runtime import configure_project_cache


configure_project_cache()

import torch
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]
CLASS_HEAD_SUFFIXES = (".weight", ".bias")
CLASS_HEAD_KEYS = (
    "model.23.cv3.0.2",
    "model.23.cv3.1.2",
    "model.23.cv3.2.2",
    "model.23.one2one_cv3.0.2",
    "model.23.one2one_cv3.1.2",
    "model.23.one2one_cv3.2.2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Core13 YOLO .pt checkpoint.")
    parser.add_argument("--official21-schema", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--missing-init",
        choices=("nearest_value", "mean", "random"),
        default="nearest_value",
        help="How to initialize official classes absent from the source checkpoint.",
    )
    parser.add_argument(
        "--missing-bias-offset",
        type=float,
        default=-2.0,
        help="Add this value to borrowed/mean missing-class classifier biases.",
    )
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


def load_yaml(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(resolved)} must be a YAML mapping")
    return payload


def names_by_id(payload: dict[str, Any]) -> dict[int, str]:
    raw = payload.get("names")
    if isinstance(raw, dict):
        names = {int(key): str(value) for key, value in raw.items()}
    elif isinstance(raw, list):
        names = {index: str(value) for index, value in enumerate(raw)}
    else:
        raise SystemExit("missing names list or mapping")
    expected = set(range(len(names)))
    if set(names) != expected:
        raise SystemExit(f"names must cover contiguous ids 0..{len(names) - 1}")
    return names


def class_value(name: str) -> tuple[str, float] | None:
    if "_" not in name:
        return None
    currency, raw_value = name.split("_", 1)
    try:
        return currency, float(raw_value)
    except ValueError:
        return None


def nearest_source_class(target_name: str, source_names: dict[int, str]) -> int | None:
    target = class_value(target_name)
    if target is None:
        return None
    target_currency, target_value = target
    candidates: list[tuple[float, int]] = []
    for source_id, source_name in source_names.items():
        parsed = class_value(source_name)
        if parsed is None:
            continue
        source_currency, source_value = parsed
        if source_currency != target_currency:
            continue
        candidates.append((abs(source_value - target_value), source_id))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def class_head_key(key: str) -> bool:
    return any(key == f"{prefix}{suffix}" for prefix in CLASS_HEAD_KEYS for suffix in CLASS_HEAD_SUFFIXES)


def mapped_class_tensor(
    *,
    old_tensor: torch.Tensor,
    new_tensor: torch.Tensor,
    source_names: dict[int, str],
    target_names: dict[int, str],
    target_to_source: dict[int, int],
    missing_init: str,
    missing_bias_offset: float,
    is_bias: bool,
) -> tuple[torch.Tensor, dict[int, dict[str, Any]]]:
    out = new_tensor.clone()
    missing_rows: dict[int, dict[str, Any]] = {}
    source_by_name = {name: class_id for class_id, name in source_names.items()}
    copied_sources = set(target_to_source.values())

    source_mean = old_tensor[list(sorted(copied_sources))].mean(dim=0) if copied_sources else old_tensor.mean(dim=0)

    for target_id, target_name in target_names.items():
        source_id = source_by_name.get(target_name)
        if source_id is not None:
            out[target_id] = old_tensor[source_id]
            target_to_source[target_id] = source_id
            continue

        if missing_init == "random":
            source_note: dict[str, Any] = {"init": "random_model_default", "source_id": None, "source_name": None}
        elif missing_init == "mean":
            out[target_id] = source_mean
            source_note = {"init": "mean_existing_source_classes", "source_id": None, "source_name": None}
        else:
            nearest_id = nearest_source_class(target_name, source_names)
            if nearest_id is None:
                out[target_id] = source_mean
                source_note = {"init": "mean_existing_source_classes", "source_id": None, "source_name": None}
            else:
                out[target_id] = old_tensor[nearest_id]
                source_note = {
                    "init": "nearest_value",
                    "source_id": int(nearest_id),
                    "source_name": source_names[nearest_id],
                }

        if is_bias and missing_init != "random":
            out[target_id] = out[target_id] + float(missing_bias_offset)
        missing_rows[target_id] = source_note

    return out, missing_rows


def main() -> None:
    args = parse_args()
    source_path = resolve(args.source)
    out_path = resolve(args.out)
    schema_path = resolve(args.official21_schema)
    schema = load_yaml(schema_path)
    target_names = names_by_id(schema)

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise SystemExit(f"{repo_rel(source_path)} is not an Ultralytics checkpoint with a model")
    source_model = checkpoint["model"].float()
    source_names = {int(class_id): str(name) for class_id, name in getattr(source_model, "names", {}).items()}
    if not source_names:
        source_names = names_by_id({"names": source_model.yaml.get("names")})

    model_cfg = copy.deepcopy(source_model.yaml)
    model_cfg["nc"] = len(target_names)
    target_model = DetectionModel(cfg=model_cfg, ch=3, nc=len(target_names), verbose=False)
    target_model.float()

    source_state = source_model.state_dict()
    target_state = target_model.state_dict()
    next_state: dict[str, torch.Tensor] = {}
    copied_exact: list[str] = []
    remapped_class_heads: list[str] = []
    skipped: list[dict[str, Any]] = []
    target_to_source: dict[int, int] = {}
    missing_by_head: dict[str, dict[str, Any]] = {}

    for key, target_tensor in target_state.items():
        source_tensor = source_state.get(key)
        if source_tensor is None:
            next_state[key] = target_tensor
            skipped.append({"key": key, "reason": "missing_from_source"})
            continue
        if source_tensor.shape == target_tensor.shape:
            next_state[key] = source_tensor
            copied_exact.append(key)
            continue
        if class_head_key(key):
            remapped, missing_rows = mapped_class_tensor(
                old_tensor=source_tensor,
                new_tensor=target_tensor,
                source_names=source_names,
                target_names=target_names,
                target_to_source=target_to_source,
                missing_init=args.missing_init,
                missing_bias_offset=args.missing_bias_offset,
                is_bias=key.endswith(".bias"),
            )
            next_state[key] = remapped
            remapped_class_heads.append(key)
            if missing_rows:
                missing_by_head[key] = {
                    target_names[target_id]: note for target_id, note in sorted(missing_rows.items())
                }
            continue
        next_state[key] = target_tensor
        skipped.append(
            {
                "key": key,
                "reason": "shape_mismatch_not_class_head",
                "source_shape": list(source_tensor.shape),
                "target_shape": list(target_tensor.shape),
            }
        )

    missing, unexpected = target_model.load_state_dict(next_state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"load_state_dict mismatch missing={missing} unexpected={unexpected}")

    target_model.names = target_names
    target_model.yaml["nc"] = len(target_names)
    target_model.yaml["names"] = target_names
    target_model.args = copy.deepcopy(getattr(source_model, "args", {}))

    out_checkpoint = copy.deepcopy(checkpoint)
    out_checkpoint["model"] = target_model
    out_checkpoint["ema"] = None
    out_checkpoint["optimizer"] = None
    out_checkpoint["scaler"] = None
    out_checkpoint["epoch"] = -1
    out_checkpoint["best_fitness"] = None
    out_checkpoint["date"] = datetime.now(timezone.utc).isoformat()
    out_checkpoint["cashsnap_official21_mapping"] = {
        "schema": "cashsnap_official21_mapped_checkpoint_v1",
        "source": repo_rel(source_path),
        "official21_schema": repo_rel(schema_path),
        "missing_init": args.missing_init,
        "missing_bias_offset": args.missing_bias_offset,
        "source_names": source_names,
        "target_names": target_names,
        "target_to_source": {
            target_names[target_id]: source_names[source_id]
            for target_id, source_id in sorted(target_to_source.items())
        },
        "missing_by_head": missing_by_head,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_checkpoint, out_path)

    summary = {
        "schema": "cashsnap_official21_mapped_checkpoint_v1",
        "created_utc": out_checkpoint["date"],
        "source": repo_rel(source_path),
        "out": repo_rel(out_path),
        "official21_schema": repo_rel(schema_path),
        "source_class_count": len(source_names),
        "target_class_count": len(target_names),
        "copied_exact_tensors": len(copied_exact),
        "remapped_class_head_tensors": len(remapped_class_heads),
        "skipped_tensors": skipped,
        "missing_init": args.missing_init,
        "missing_bias_offset": args.missing_bias_offset,
        "target_to_source": out_checkpoint["cashsnap_official21_mapping"]["target_to_source"],
        "missing_by_head": missing_by_head,
    }
    summary_path = resolve(args.summary_json) if args.summary_json else out_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "wrote_mapped_checkpoint="
        f"{repo_rel(out_path)} copied_exact={len(copied_exact)} "
        f"remapped_class_heads={len(remapped_class_heads)} summary={repo_rel(summary_path)}"
    )


if __name__ == "__main__":
    main()
