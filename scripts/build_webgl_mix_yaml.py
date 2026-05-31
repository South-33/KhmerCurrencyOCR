#!/usr/bin/env python
"""Build a YOLO data YAML from gated WebGL recipe package outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_smoke_suite_v1.json"
DEFAULT_OUT = ROOT / "configs" / "cashsnap_webgl_smoke_suite_mix.yaml"
VALID_GATE_KINDS = {"smoke", "trainable-candidate", "none"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gate-kind", choices=sorted(VALID_GATE_KINDS), default="smoke")
    parser.add_argument("--skip-gates", action="store_true", help="Do not re-run package gates before writing YAML.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def rel_between(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), base.resolve()).replace("\\", "/")


def read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing suite config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing YOLO data YAML: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_train_views(value: object) -> set[str]:
    if isinstance(value, list):
        return {str(item).strip() for item in value if str(item).strip()}
    if isinstance(value, str):
        return {item for item in re.split(r"[,;\s]+", value) if item}
    return {"detect"}


def run_gate(row: dict, root: Path, gate_kind: str) -> None:
    recipe_id = str(row["recipe_id"])
    scene_mode = str(row["scene_mode"])
    if gate_kind == "none":
        return
    if gate_kind == "smoke":
        subprocess.run(
            [
                sys.executable,
                "scripts/check_webgl_smoke_gate.py",
                "--root",
                str(root),
                "--require-recipe",
                recipe_id,
                "--require-scene-mode",
                scene_mode,
            ],
            cwd=ROOT,
            check=True,
        )
        return
    if gate_kind == "trainable-candidate":
        train_views = parse_train_views(row.get("train_views", ["detect"]))
        if "detect" not in train_views:
            raise SystemExit(f"{recipe_id}: build_webgl_mix_yaml writes detect data.yaml mixes, but train_views={sorted(train_views)}")
        command = [
            sys.executable,
            "scripts/check_webgl_trainable_candidate_gate.py",
            "--root",
            str(root),
            "--require-recipe",
            recipe_id,
            "--require-scene-mode",
            scene_mode,
            "--train-views",
            ",".join(sorted(train_views)),
        ]
        if bool(row.get("allow_zero_visible")):
            command.append("--allow-zero-visible")
        subprocess.run(command, cwd=ROOT, check=True)
        return
    raise SystemExit(f"unknown gate kind: {gate_kind}")


def split_path(dataset_root: Path, split: str | list[str]) -> str | list[str]:
    if isinstance(split, list):
        return [split_path(dataset_root, item) for item in split]
    path = Path(split)
    resolved = path if path.is_absolute() else dataset_root / path
    return rel(resolved)


def main() -> int:
    args = parse_args()
    suite = read_json(resolve(args.suite))
    rows = suite.get("recipes", [])
    if not isinstance(rows, list) or not rows:
        raise SystemExit("suite recipes must be a non-empty list")

    train: list[str] = []
    val: list[str] = []
    names: dict | None = None
    sources: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("suite recipe rows must be objects")
        recipe_id = str(row["recipe_id"])
        scene_mode = str(row["scene_mode"])
        root = resolve(Path(str(row["out_root"])))
        if not args.skip_gates:
            run_gate(row, root, args.gate_kind)
        data = read_yaml(root / "data.yaml")
        data_root = Path(data["path"])
        if not data_root.is_absolute():
            data_root = (root / data_root).resolve()
        if names is None:
            names = data["names"]
        elif data["names"] != names:
            raise SystemExit(f"{recipe_id}: class names differ from previous package")
        train_split = split_path(data_root, data["train"])
        val_split = split_path(data_root, data.get("val", data["train"]))
        if isinstance(train_split, list):
            train.extend(train_split)
        else:
            train.append(train_split)
        if isinstance(val_split, list):
            val.extend(val_split)
        else:
            val.append(val_split)
        sources.append(
            {
                "recipe_id": recipe_id,
                "scene_mode": scene_mode,
                "root": rel(root),
                "data_yaml": rel(root / "data.yaml"),
                "gate_kind": args.gate_kind,
                "train_views": sorted(parse_train_views(row.get("train_views", ["detect"]))),
            }
        )

    out_path = resolve(args.out)
    output = {
        "path": rel_between(out_path.parent, ROOT),
        "train": train,
        "val": val,
        "names": names,
        "cashsnap_sources": sources,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)} with {len(train)} train splits and {len(val)} val splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
