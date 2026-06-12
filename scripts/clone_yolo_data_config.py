#!/usr/bin/env python
"""Clone a YOLO data YAML with explicit split overrides."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out-config", required=True, type=Path)
    parser.add_argument("--path-mode", choices=("preserve", "repo-root"), default="preserve")
    parser.add_argument("--train", default="")
    parser.add_argument("--val", default="")
    parser.add_argument("--test", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--intended-use", default="")
    parser.add_argument(
        "--extra-name",
        action="append",
        default=[],
        metavar="INDEX:NAME",
        help="Append/override a YOLO class name in the output config, e.g. 13:UNKNOWN_FOREIGN_NOTE.",
    )
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rel_between(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/")


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(path)} must be a YAML mapping")
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def parse_extra_name(value: str) -> tuple[int, str]:
    if ":" not in value:
        raise SystemExit(f"--extra-name expected INDEX:NAME, got {value!r}")
    raw_index, name = value.split(":", 1)
    try:
        index = int(raw_index)
    except ValueError as exc:
        raise SystemExit(f"invalid --extra-name index {raw_index!r}") from exc
    name = name.strip()
    if not name:
        raise SystemExit(f"empty --extra-name class name in {value!r}")
    return index, name


def names_mapping(payload: dict[str, Any]) -> dict[int, str]:
    raw_names = payload.get("names", {})
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, list):
        return {index: str(value) for index, value in enumerate(raw_names)}
    raise SystemExit("YOLO data config has no names mapping")


def main() -> int:
    args = parse_args()
    source = resolve(args.data)
    out_config = resolve(args.out_config)
    payload = copy.deepcopy(read_yaml(source))

    if args.path_mode == "repo-root":
        payload["path"] = rel_between(out_config.parent, ROOT)

    overrides = {}
    for key in ("train", "val", "test"):
        value = getattr(args, key)
        if value:
            payload[key] = value
            overrides[key] = value

    if args.extra_name:
        names = names_mapping(payload)
        for value in args.extra_name:
            index, name = parse_extra_name(value)
            names[index] = name
        payload["names"] = {index: names[index] for index in sorted(names)}

    sources = copy.deepcopy(payload.get("cashsnap_sources", {}))
    if not isinstance(sources, dict):
        sources = {}
    sources["cloned_from_config"] = repo_rel(source)
    sources["clone_split_overrides"] = overrides
    payload["cashsnap_sources"] = sources

    if args.tag or args.intended_use:
        clone_meta = {
            "schema": "cashsnap_yolo_data_config_clone_v1",
            "source_config": repo_rel(source),
            "split_overrides": overrides,
        }
        if args.tag:
            clone_meta["tag"] = args.tag
        if args.intended_use:
            clone_meta["intended_use"] = args.intended_use
        payload["cashsnap_data_config_clone"] = clone_meta

    write_yaml(out_config, payload)
    print(f"wrote_config={repo_rel(out_config)}")
    if overrides:
        print("overrides=" + ",".join(f"{key}={value}" for key, value in sorted(overrides.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
