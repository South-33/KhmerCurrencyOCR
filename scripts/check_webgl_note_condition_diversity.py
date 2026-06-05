#!/usr/bin/env python
"""Gate per-note wear, dirt, crinkle, and wetness diversity in WebGL packages."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CONDITION_FIELDS = ("dirtiness", "crinkle", "wetness", "edgeWear")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL dataset root.")
    parser.add_argument("--allow-missing", action="store_true", help="Allow legacy packages with no condition metadata.")
    parser.add_argument("--min-notes", type=int, default=1)
    parser.add_argument("--min-profiles", type=int, default=None)
    parser.add_argument("--min-dirtiness-range", type=float, default=None)
    parser.add_argument("--min-crinkle-range", type=float, default=None)
    parser.add_argument("--min-wetness-range", type=float, default=None)
    parser.add_argument("--min-dirty-notes", type=int, default=None)
    parser.add_argument("--min-pristine-notes", type=int, default=None)
    parser.add_argument("--min-wet-notes", type=int, default=None)
    parser.add_argument("--expected-policy", choices=["mixed", "pristine_only", "heavy_wear", "wet_stress"], default="")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> object:
    if not path.exists():
        raise SystemExit(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def numeric_range(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return max(rows) - min(rows)


def auto_unique_threshold(notes: int) -> int:
    if notes < 4:
        return 1
    if notes < 16:
        return 2
    return 3


def auto_range_threshold(notes: int, small: float, large: float) -> float:
    if notes < 4:
        return 0.0
    if notes < 16:
        return small
    return large


def threshold(value: int | float | None, fallback: int | float) -> int | float:
    return fallback if value is None else value


def source_metadata_path(dataset_root: Path, row: dict) -> Path:
    raw = str(row.get("source_metadata", "")).strip()
    require(bool(raw), f"manifest row for variant {row.get('variant', '<unknown>')} is missing source_metadata")
    return dataset_root / raw


def main() -> int:
    args = parse_args()
    dataset_root = resolve(args.root)
    manifest = read_json(dataset_root / "manifest.json")
    require(isinstance(manifest, list), "manifest.json must be a list")

    total_notes = 0
    missing_conditions = 0
    policies: Counter[str] = Counter()
    profiles: Counter[str] = Counter()
    values: dict[str, list[float]] = {field: [] for field in CONDITION_FIELDS}

    for row in manifest:
        require(isinstance(row, dict), "manifest rows must be objects")
        meta = read_json(source_metadata_path(dataset_root, row))
        require(isinstance(meta, dict), f"{row.get('source_metadata')}: metadata must be an object")
        scene_config = meta.get("sceneConfig", {})
        require(isinstance(scene_config, dict), f"{row.get('source_metadata')}: sceneConfig must be an object")
        policies[str(scene_config.get("noteConditionPolicy", "mixed"))] += 1
        assets = meta.get("assets", [])
        require(isinstance(assets, list), f"{row.get('source_metadata')}: metadata.assets must be a list")
        for asset in assets:
            require(isinstance(asset, dict), f"{row.get('source_metadata')}: asset rows must be objects")
            total_notes += 1
            condition = asset.get("condition")
            if not isinstance(condition, dict):
                missing_conditions += 1
                continue
            profile = str(condition.get("profile", "")).strip()
            require(profile, f"{row.get('source_metadata')}: condition profile must be non-empty")
            profiles[profile] += 1
            for field in CONDITION_FIELDS:
                values[field].append(float(condition.get(field, 0.0)))
            require(int(condition.get("speckleCount", 0)) >= 0, f"{row.get('source_metadata')}: speckleCount must be non-negative")
            require(int(condition.get("stainCount", 0)) >= 0, f"{row.get('source_metadata')}: stainCount must be non-negative")
            require(int(condition.get("creaseCount", 0)) >= 0, f"{row.get('source_metadata')}: creaseCount must be non-negative")

    if total_notes == 0:
        print("ok: WebGL note condition diversity skipped (no notes)")
        return 0
    if missing_conditions == total_notes and args.allow_missing:
        print(f"ok: WebGL note condition diversity skipped (legacy package: {total_notes} notes without condition metadata)")
        return 0
    require(missing_conditions == 0, f"{missing_conditions}/{total_notes} notes are missing condition metadata")
    require(total_notes >= args.min_notes, f"expected at least {args.min_notes} notes, got {total_notes}")

    profile_count = len(profiles)
    dirty_notes = sum(1 for value in values["dirtiness"] if value >= 0.55)
    pristine_notes = int(profiles.get("pristine", 0))
    wet_notes = sum(1 for value in values["wetness"] if value >= 0.10)
    policy = next(iter(policies)) if len(policies) == 1 else "mixed"
    if args.expected_policy:
        require(
            len(policies) == 1 and args.expected_policy in policies,
            f"expected note condition policy {args.expected_policy}, got policies={dict(sorted(policies.items()))}",
        )
    stats = {
        "dirtiness_range": numeric_range(values["dirtiness"]),
        "crinkle_range": numeric_range(values["crinkle"]),
        "wetness_range": numeric_range(values["wetness"]),
        "edge_wear_range": numeric_range(values["edgeWear"]),
    }

    if policy == "pristine_only":
        default_min_profiles = 1
        default_min_dirtiness_range = 0.0
        default_min_crinkle_range = 0.0
        default_min_wetness_range = 0.0
        default_min_dirty_notes = 0
        default_min_pristine_notes = total_notes
        default_min_wet_notes = 0
    elif policy == "heavy_wear":
        default_min_profiles = 1
        default_min_dirtiness_range = auto_range_threshold(total_notes, 0.12, 0.25)
        default_min_crinkle_range = auto_range_threshold(total_notes, 0.16, 0.35)
        default_min_wetness_range = 0.0
        default_min_dirty_notes = max(1, int(total_notes * 0.70))
        default_min_pristine_notes = 0
        default_min_wet_notes = 0
    elif policy == "wet_stress":
        default_min_profiles = 1
        default_min_dirtiness_range = auto_range_threshold(total_notes, 0.16, 0.30)
        default_min_crinkle_range = auto_range_threshold(total_notes, 0.16, 0.30)
        default_min_wetness_range = auto_range_threshold(total_notes, 0.18, 0.35)
        default_min_dirty_notes = 0
        default_min_pristine_notes = 0
        default_min_wet_notes = max(1, int(total_notes * 0.85))
    else:
        default_min_profiles = auto_unique_threshold(total_notes)
        default_min_dirtiness_range = auto_range_threshold(total_notes, 0.20, 0.45)
        default_min_crinkle_range = auto_range_threshold(total_notes, 0.20, 0.40)
        default_min_wetness_range = 0.10 if total_notes >= 16 else 0.0
        default_min_dirty_notes = 1 if total_notes >= 8 else 0
        default_min_pristine_notes = 1 if total_notes >= 16 else 0
        default_min_wet_notes = 1 if total_notes >= 16 else 0

    min_profiles = int(threshold(args.min_profiles, default_min_profiles))
    min_dirtiness_range = float(threshold(args.min_dirtiness_range, default_min_dirtiness_range))
    min_crinkle_range = float(threshold(args.min_crinkle_range, default_min_crinkle_range))
    min_wetness_range = float(threshold(args.min_wetness_range, default_min_wetness_range))
    min_dirty_notes = int(threshold(args.min_dirty_notes, default_min_dirty_notes))
    min_pristine_notes = int(threshold(args.min_pristine_notes, default_min_pristine_notes))
    min_wet_notes = int(threshold(args.min_wet_notes, default_min_wet_notes))

    require(profile_count >= min_profiles, f"expected at least {min_profiles} condition profiles, got {profile_count}: {dict(profiles)}")
    require(stats["dirtiness_range"] >= min_dirtiness_range, f"dirtiness range {stats['dirtiness_range']:.4f} below {min_dirtiness_range:.4f}")
    require(stats["crinkle_range"] >= min_crinkle_range, f"crinkle range {stats['crinkle_range']:.4f} below {min_crinkle_range:.4f}")
    require(stats["wetness_range"] >= min_wetness_range, f"wetness range {stats['wetness_range']:.4f} below {min_wetness_range:.4f}")
    require(dirty_notes >= min_dirty_notes, f"expected at least {min_dirty_notes} dirty notes, got {dirty_notes}")
    require(pristine_notes >= min_pristine_notes, f"expected at least {min_pristine_notes} pristine notes, got {pristine_notes}")
    require(wet_notes >= min_wet_notes, f"expected at least {min_wet_notes} wet/damp notes, got {wet_notes}")

    print(
        "ok: WebGL note condition diversity passed "
        f"({total_notes} notes, policy={policy}, profiles={dict(sorted(profiles.items()))}, dirty={dirty_notes}, "
        f"pristine={pristine_notes}, wet={wet_notes}, dirtiness_range={stats['dirtiness_range']:.2f}, "
        f"crinkle_range={stats['crinkle_range']:.2f}, wetness_range={stats['wetness_range']:.2f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
