from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def resolve_ultralytics_data_yaml(data_path: Path) -> Path:
    """Write a repo-local Ultralytics data YAML with an absolute dataset root."""
    data_path = data_path.resolve()
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Dataset config must be a mapping: {data_path}")

    dataset_root = config.get("path")
    if dataset_root is not None:
        root_path = Path(str(dataset_root)).expanduser()
        if not root_path.is_absolute():
            root_path = (data_path.parent / root_path).resolve()
        config["path"] = root_path.as_posix()

    digest = hashlib.sha256(data_path.as_posix().encode("utf-8")).hexdigest()[:12]
    out_dir = ROOT / ".cache_runtime" / "ultralytics_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{data_path.stem}.{digest}.yaml"
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return out_path
