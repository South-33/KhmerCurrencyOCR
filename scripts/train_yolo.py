from __future__ import annotations

import argparse
from pathlib import Path

from local_runtime import configure_project_cache
from hardware_profile import parse_auto_int, recommended_device, recommended_train_batch, recommended_workers
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


def no_validation_trainer(base_trainer):
    class CashSnapNoValidationTrainer(base_trainer):
        def validate(self):
            return {}, 0.0

        def final_eval(self):
            from ultralytics.engine.trainer import LOCAL_RANK, RANK, torch_distributed_zero_first
            from ultralytics.utils.torch_utils import strip_optimizer

            with torch_distributed_zero_first(LOCAL_RANK):
                if RANK in {-1, 0}:
                    checkpoint = strip_optimizer(self.last) if self.last.exists() else {}
                    if self.best.exists():
                        strip_optimizer(self.best, updates={"train_results": checkpoint.get("train_results")})

    return CashSnapNoValidationTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detector for CashSnap.")
    parser.add_argument("--data", default="configs/cashsnap_v1.yaml", help="YOLO dataset YAML path.")
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained YOLO weights.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", default="auto", help="Integer batch size or 'auto' for the local hardware profile.")
    parser.add_argument("--workers", default="auto", help="Integer worker count or 'auto' for the local hardware profile.")
    parser.add_argument("--project", default=str(ROOT / "runs" / "cashsnap"))
    parser.add_argument("--name", default="yolo26n_v1")
    parser.add_argument("--device", default="auto", help="Ultralytics device selector, e.g. cpu, 0, 0,1, or auto.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate override.")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning rate fraction override.")
    parser.add_argument("--optimizer", default=None, help="Ultralytics optimizer override.")
    parser.add_argument("--box", type=float, default=None, help="Ultralytics box loss gain override.")
    parser.add_argument("--cls", type=float, default=None, help="Ultralytics class loss gain override.")
    parser.add_argument("--dfl", type=float, default=None, help="Ultralytics DFL loss gain override.")
    parser.add_argument("--warmup-epochs", type=float, default=None, help="Warmup epochs override.")
    parser.add_argument("--warmup-bias-lr", type=float, default=None, help="Warmup bias learning rate override.")
    parser.add_argument("--warmup-momentum", type=float, default=None, help="Warmup momentum override.")
    parser.add_argument("--seed", type=int, default=None, help="Ultralytics training seed override.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training data to use.")
    parser.add_argument(
        "--cache",
        default="false",
        choices=["false", "ram", "disk", "true"],
        help="Ultralytics image cache mode. Prefer 'disk' over 'ram' on this 16 GB laptop.",
    )
    parser.add_argument(
        "--compile",
        nargs="?",
        const="default",
        default=None,
        help="Optional Ultralytics torch.compile mode, e.g. default or reduce-overhead.",
    )
    parser.add_argument("--max-train-batches", type=int, default=None, help="Stop after this many train batches.")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze the first N YOLO layers during training.")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation probability override.")
    parser.add_argument("--erasing", type=float, default=None, help="Random erasing augmentation probability override.")
    parser.add_argument("--hsv-h", type=float, default=None, help="HSV hue augmentation override.")
    parser.add_argument("--hsv-s", type=float, default=None, help="HSV saturation augmentation override.")
    parser.add_argument("--hsv-v", type=float, default=None, help="HSV value augmentation override.")
    parser.add_argument("--translate", type=float, default=None, help="Translate augmentation override.")
    parser.add_argument("--scale", type=float, default=None, help="Scale augmentation override.")
    parser.add_argument("--fliplr", type=float, default=None, help="Horizontal flip augmentation probability override.")
    parser.add_argument("--degrees", type=float, default=None, help="Rotation augmentation override in degrees.")
    parser.add_argument("--shear", type=float, default=None, help="Shear augmentation override.")
    parser.add_argument("--perspective", type=float, default=None, help="Perspective augmentation override.")
    parser.add_argument("--close-mosaic", type=int, default=None, help="Disable mosaic for final N epochs.")
    parser.add_argument("--no-amp", action="store_true", help="Disable Ultralytics AMP checks/training.")
    parser.add_argument("--no-val", action="store_true", help="Skip epoch and final validation; use val_yolo.py explicitly.")
    parser.add_argument("--no-plots", action="store_true", help="Skip Ultralytics training plots.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    parser.add_argument("--quiet", action="store_true", help="Reduce Ultralytics training log output.")
    return parser.parse_args()


def parse_cache(value: str) -> bool | str:
    if value == "false":
        return False
    if value == "true":
        return True
    return value


def remove_stale_no_val_checkpoints(project_path: Path, run_name: str) -> None:
    weights_dir = project_path / run_name / "weights"
    for filename in ("best.pt", "last.pt"):
        checkpoint = weights_dir / filename
        try:
            checkpoint.unlink()
        except FileNotFoundError:
            continue
        print(f"removed_stale_checkpoint={checkpoint}", flush=True)


def main() -> None:
    args = parse_args()
    data_path = resolve_from_root(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    data_path = resolve_ultralytics_data_yaml(data_path)
    project_path = resolve_from_root(args.project)
    args.batch = parse_auto_int(args.batch, recommended_train_batch(args.imgsz), "batch", allow_zero=False)
    args.workers = parse_auto_int(args.workers, recommended_workers("train"), "workers")
    args.device = recommended_device(args.device)
    print(
        f"[hardware] train device={args.device} batch={args.batch} workers={args.workers} imgsz={args.imgsz}",
        flush=True,
    )

    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project_path),
        "name": args.name,
        "pretrained": True,
        "plots": not args.no_plots,
        "val": not args.no_val,
        "amp": not args.no_amp,
        "workers": args.workers,
        "fraction": args.fraction,
        "cache": parse_cache(args.cache),
        "exist_ok": args.exist_ok,
        "verbose": not args.quiet,
    }
    train_args["device"] = args.device
    if args.lr0 is not None:
        train_args["lr0"] = args.lr0
    if args.lrf is not None:
        train_args["lrf"] = args.lrf
    if args.optimizer is not None:
        train_args["optimizer"] = args.optimizer
    for key in ("box", "cls", "dfl"):
        value = getattr(args, key)
        if value is not None:
            train_args[key] = value
    if args.compile is not None:
        train_args["compile"] = args.compile
    if args.freeze is not None:
        train_args["freeze"] = args.freeze
    if args.warmup_epochs is not None:
        train_args["warmup_epochs"] = args.warmup_epochs
    if args.warmup_bias_lr is not None:
        train_args["warmup_bias_lr"] = args.warmup_bias_lr
    if args.warmup_momentum is not None:
        train_args["warmup_momentum"] = args.warmup_momentum
    if args.seed is not None:
        train_args["seed"] = args.seed
    for key in (
        "mosaic",
        "erasing",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "translate",
        "scale",
        "fliplr",
        "degrees",
        "shear",
        "perspective",
        "close_mosaic",
    ):
        value = getattr(args, key)
        if value is not None:
            train_args[key] = value

    if args.no_val and args.exist_ok:
        remove_stale_no_val_checkpoints(project_path, args.name)

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

    trainer = no_validation_trainer(model._smart_load("trainer")) if args.no_val else None
    model.train(trainer=trainer, **train_args)


if __name__ == "__main__":
    main()
