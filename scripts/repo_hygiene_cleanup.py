from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

OBSOLETE_FRAGMENT_DIRS = [
    "fragment_classifier_cashsnap_realfrag_plus_legacy_backfocus_v1",
    "fragment_classifier_cashsnap_realfrag_plus_legacy_v1",
    "fragment_classifier_cashsnap_realfrag_v1",
]

MINOR_FRAGMENT_DIRS = [
    "fragment_classifier_v1_candidate",
    "fragment_classifier_review_pack_smoke_v1",
    "fragment_classifier_roboflow_partial_khr_diag_v1",
    "fragment_classifier_roboflow_partial_khr_oldcommon_eval_v1",
    "fragment_classifier_smoke_v1",
    "fragment_classifier_p1_oldcommon_focus_unreviewed_diag_v1",
    "picwish_upload_batches_cashsnap_smoke",
    "picwish_upload_batches",
]

OBSOLETE_SYNTHETIC_DIRS = [
    "khr_messy_v1",
    "khr_messy_v2",
    "khr_messy_v3",
    "khr_messy_v4",
    "khr_messy_v5_real_crops",
    "khr_fan_v1",
    "khr_old_common_focus_v1",
    "khr_radial_slice_probe_v1",
    "khr_messy_clean_smoke",
    "khr_fan_smoke",
    "khr_current_cutout_obb_smoke",
    "khr_current_cutout_detect_smoke",
]

MINOR_SYNTHETIC_DIRS = [
    "khr_current_thin_radial_slice_probe_v1_obb",
    "khr_current_thin_radial_slice_probe_v1",
    "khr_p1_thin_hand_smoke_v1",
    "khr_p1_oldcommon_thin_hand_smoke_v1",
    "khr_p1_oldcommon_thin_only_smoke_v1",
    "khr_p1_oldcommon_microthin_smoke_v1",
    "khr_rare_gold_probe_v1",
    "khr_rare_gold_clean_v1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run local cleanup for obsolete generated CashSnap datasets."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete/move the listed generated directories. Default is dry-run.",
    )
    return parser.parse_args()


def require_inside_data(path: Path) -> Path:
    resolved = path.resolve()
    data_root = DATA_DIR.resolve()
    if resolved == data_root or data_root not in resolved.parents:
        raise SystemExit(f"Refusing to operate outside {data_root}: {resolved}")
    return resolved


def delete_dir(path: Path, apply: bool) -> None:
    resolved = require_inside_data(path)
    if not resolved.exists():
        return
    print(f"{'delete' if apply else 'would delete'}: {resolved.relative_to(ROOT)}")
    if apply:
        shutil.rmtree(resolved)


def move_dir(src: Path, dst: Path, apply: bool) -> None:
    resolved_src = require_inside_data(src)
    resolved_dst = require_inside_data(dst)
    if not resolved_src.exists():
        return
    print(
        f"{'move' if apply else 'would move'}: "
        f"{resolved_src.relative_to(ROOT)} -> {resolved_dst.relative_to(ROOT)}"
    )
    if apply:
        resolved_dst.parent.mkdir(parents=True, exist_ok=True)
        if resolved_dst.exists():
            raise SystemExit(f"Refusing to overwrite existing destination: {resolved_dst}")
        shutil.move(str(resolved_src), str(resolved_dst))


def main() -> None:
    args = parse_args()
    deprecated_dir = DATA_DIR / "deprecated"
    synthetic_dir = DATA_DIR / "synthetic"
    deprecated_synthetic_dir = deprecated_dir / "synthetic"

    for name in OBSOLETE_FRAGMENT_DIRS:
        delete_dir(DATA_DIR / name, args.apply)
    for name in MINOR_FRAGMENT_DIRS:
        move_dir(DATA_DIR / name, deprecated_dir / name, args.apply)
    for name in OBSOLETE_SYNTHETIC_DIRS:
        delete_dir(synthetic_dir / name, args.apply)
    for name in MINOR_SYNTHETIC_DIRS:
        move_dir(synthetic_dir / name, deprecated_synthetic_dir / name, args.apply)

    if not args.apply:
        print("dry-run only; pass --apply to make changes")


if __name__ == "__main__":
    main()
