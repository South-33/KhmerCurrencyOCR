#!/usr/bin/env python
"""Write a YOLO data YAML whose validation/test splits point at train.

Ultralytics scans the val split during train setup even when the trainer skips
validation. For fixed-step no-val probes on memory-constrained machines, this
helper keeps training setup focused on the intended train rows; evaluate later
with the original data YAML.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve(args.data)
    out_path = resolve(args.out)
    config: dict[str, Any] = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit(f"YOLO data YAML must be a mapping: {repo_rel(data_path)}")
    train = config.get("train")
    if train is None:
        raise SystemExit(f"YOLO data YAML has no train split: {repo_rel(data_path)}")

    config["val"] = train
    config["test"] = train
    policy = dict(config.get("cashsnap_policy") or {})
    policy["trainonly_runtime_yaml"] = True
    policy["trainonly_source_data"] = repo_rel(data_path)
    policy["trainonly_reason"] = (
        "No-val training helper: avoid scanning the real val/test splits during "
        "Ultralytics train setup; evaluate with the original data YAML."
    )
    config["cashsnap_policy"] = policy

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"wrote={repo_rel(out_path)}")


if __name__ == "__main__":
    main()
