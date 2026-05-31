#!/usr/bin/env python
"""Run the current WebGL P1 diagnostic pipeline end to end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCOREABLE_LABELS = ROOT / "data" / "audit" / "real_overlap_0003_commons_shop_5k_10k_20k.scoreable.txt"


def rel_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Render scenes instead of repackaging existing smoke variants.")
    parser.add_argument("--train-smoke", action="store_true", help="Run the tiny headroom training smoke.")
    parser.add_argument("--skip-alpha-eval", action="store_true", help="Skip current alpha draft diagnostic evaluation.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    suite_cmd = [sys.executable, "scripts/run_webgl_smoke_suite.py"]
    if not args.render:
        suite_cmd.append("--skip-render")
    run(suite_cmd, args.dry_run)

    run(
        [
            sys.executable,
            "scripts/build_webgl_mix_yaml.py",
            "--out",
            "configs/cashsnap_webgl_smoke_suite_mix.yaml",
        ],
        args.dry_run,
    )
    run([sys.executable, "scripts/check_webgl_p1_readiness.py"], args.dry_run)
    run(
        [
            sys.executable,
            "scripts/filter_yolo_labels_by_quality.py",
            "--labels",
            "data/real_fan_benchmark/drafts/real_overlap_0003_commons_shop_5k_10k_20k.txt",
            "--out",
            rel_path(DEFAULT_SCOREABLE_LABELS),
        ],
        args.dry_run,
    )
    if not args.skip_alpha_eval:
        run(
            [
                sys.executable,
                "scripts/evaluate_real_draft_labels.py",
                "--image",
                "data/real_fan_benchmark/images/candidates/real_overlap_0003_commons_shop_5k_10k_20k.png",
                "--labels",
                rel_path(DEFAULT_SCOREABLE_LABELS),
                "--model",
                "runs/cashsnap/yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8/weights/best.pt",
                "--imgsz",
                "416,640",
                "--conf",
                "0.03,0.05,0.25",
                "--nms-iou",
                "0.70",
                "--out",
                "data/audit/real_overlap_0003_alpha_scoreable_eval.csv",
            ],
            args.dry_run,
        )
    if args.train_smoke:
        run(
            [
                sys.executable,
                "scripts/bench_train_with_headroom.py",
                "--data",
                "configs/cashsnap_webgl_smoke_suite_mix.yaml",
                "--name",
                "webgl_smoke_mix_tiny_train_probe",
                "--epochs",
                "1",
                "--max-train-batches",
                "2",
                "--imgsz",
                "416",
                "--batch",
                "2",
                "--workers",
                "0",
                "--quiet",
                "--exist-ok",
            ],
            args.dry_run,
        )
    print("ok: WebGL P1 diagnostic pipeline command sequence completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
