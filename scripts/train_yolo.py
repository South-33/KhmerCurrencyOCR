from __future__ import annotations

import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Train a YOLO detector for CashSnap.")
    parser.add_argument("--data", default="configs/cashsnap_v1.yaml", help="YOLO dataset YAML path.")
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained YOLO weights.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs" / "cashsnap"))
    parser.add_argument("--name", default="yolo26n_v1")
    parser.add_argument("--device", default=None, help="Ultralytics device selector, e.g. cpu, 0, or 0,1.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training data to use.")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Stop after this many train batches.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    parser.add_argument("--quiet", action="store_true", help="Reduce Ultralytics training log output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = resolve_from_root(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    project_path = resolve_from_root(args.project)

    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project_path),
        "name": args.name,
        "pretrained": True,
        "plots": True,
        "val": True,
        "workers": args.workers,
        "fraction": args.fraction,
        "exist_ok": args.exist_ok,
        "verbose": not args.quiet,
    }
    if args.device is not None:
        train_args["device"] = args.device

    model = YOLO(resolve_local_model(args.model))
    if args.max_train_batches is not None:
        if args.max_train_batches < 1:
            raise ValueError("--max-train-batches must be at least 1 when set.")

        batch_counter = {"seen": 0}

        def stop_after_max_train_batches(trainer) -> None:
            batch_counter["seen"] += 1
            if batch_counter["seen"] >= args.max_train_batches:
                trainer.stop = True

        model.add_callback("on_train_batch_end", stop_after_max_train_batches)

    model.train(
        **train_args,
    )


if __name__ == "__main__":
    main()
