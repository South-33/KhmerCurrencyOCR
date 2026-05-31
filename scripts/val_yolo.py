from __future__ import annotations

import argparse
from pathlib import Path

from local_runtime import configure_project_cache
from yolo_data_config import resolve_ultralytics_data_yaml

configure_project_cache()

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def resolve_from_root(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def resolve_local_model(model_value: str) -> str:
    path = Path(model_value).expanduser()
    if path.is_absolute():
        return str(path)

    repo_path = ROOT / path
    if repo_path.exists():
        return str(repo_path)

    return model_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO detector for CashSnap.")
    parser.add_argument("--model", required=True, help="YOLO weights path or model name.")
    parser.add_argument("--data", required=True, help="YOLO dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--project", default=str(ROOT / "runs" / "cashsnap"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default=None, help="Ultralytics device selector, e.g. cpu, 0, or 0,1.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override.")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold override.")
    parser.add_argument("--no-plots", action="store_true", help="Skip Ultralytics validation plots.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    parser.add_argument("--quiet", action="store_true", help="Reduce Ultralytics validation log output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_from_root(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    data_path = resolve_ultralytics_data_yaml(data_path)

    project_path = resolve_from_root(args.project)
    val_args = {
        "data": str(data_path),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "split": args.split,
        "project": str(project_path),
        "name": args.name,
        "plots": not args.no_plots,
        "exist_ok": args.exist_ok,
        "verbose": not args.quiet,
    }
    if args.device is not None:
        val_args["device"] = args.device
    if args.conf is not None:
        val_args["conf"] = args.conf
    if args.iou is not None:
        val_args["iou"] = args.iou

    model = YOLO(resolve_local_model(args.model))
    model.val(**val_args)


if __name__ == "__main__":
    main()
