from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_runtime import configure_project_cache
from hardware_profile import parse_auto_int, recommended_device, recommended_val_batch, recommended_workers
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
    parser.add_argument("--batch", default="auto", help="Integer batch size or 'auto' for the local hardware profile.")
    parser.add_argument("--workers", default="auto", help="Integer worker count or 'auto' for the local hardware profile.")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--project", default=str(ROOT / "runs" / "cashsnap"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="auto", help="Ultralytics device selector, e.g. cpu, 0, 0,1, or auto.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold override.")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold override.")
    parser.add_argument(
        "--classes",
        default="",
        help="Optional comma/space-separated class ids to evaluate, e.g. '0,1,2'.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip Ultralytics validation plots.")
    parser.add_argument("--metrics-json", default=None, help="Optional path for machine-readable validation metrics.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    parser.add_argument("--quiet", action="store_true", help="Reduce Ultralytics validation log output.")
    return parser.parse_args()


def parse_classes(value: str) -> list[int] | None:
    tokens = [token for token in value.replace(",", " ").split() if token]
    if not tokens:
        return None
    class_ids: list[int] = []
    for token in tokens:
        try:
            class_ids.append(int(token))
        except ValueError as exc:
            raise SystemExit(f"--classes accepts integer class ids only, got {token!r}") from exc
    return class_ids


def jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def numeric_at(values, index: int) -> float | None:
    if values is None:
        return None
    values = jsonable(values)
    if not isinstance(values, list) or index >= len(values):
        return None
    try:
        return float(values[index])
    except (TypeError, ValueError):
        return None


def class_name(names: dict | list, index: int) -> str:
    if isinstance(names, dict):
        return str(names.get(index, names.get(str(index), index)))
    if isinstance(names, list) and index < len(names):
        return str(names[index])
    return str(index)


def metrics_payload(metrics, args: argparse.Namespace) -> dict:
    names = getattr(metrics, "names", {}) or {}
    box = getattr(metrics, "box", None)
    results_dict = getattr(metrics, "results_dict", {})
    payload = {
        "model": args.model,
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "results": jsonable(results_dict),
        "speed": jsonable(getattr(metrics, "speed", {})),
        "names": jsonable(names),
        "classes": args.classes,
    }
    if box is None:
        return payload

    payload["box"] = {
        "precision": jsonable(getattr(box, "mp", None)),
        "recall": jsonable(getattr(box, "mr", None)),
        "map50": jsonable(getattr(box, "map50", None)),
        "map50_95": jsonable(getattr(box, "map", None)),
        "map75": jsonable(getattr(box, "map75", None)),
        "ap_class_index": jsonable(getattr(box, "ap_class_index", None)),
    }

    class_count = len(names)
    per_class = []
    all_ap = jsonable(getattr(box, "all_ap", None))
    class_indexes = jsonable(getattr(box, "ap_class_index", list(range(class_count))))
    if not isinstance(class_indexes, list):
        class_indexes = list(range(class_count))
    metric_position_by_class: dict[int, int] = {}
    for position, class_id in enumerate(class_indexes):
        try:
            metric_position_by_class[int(class_id)] = position
        except (TypeError, ValueError):
            continue

    maps = jsonable(getattr(box, "maps", None))
    for index in range(class_count):
        position = metric_position_by_class.get(index)
        row = {
            "class_id": index,
            "class_name": class_name(names, index),
            "precision": numeric_at(getattr(box, "p", None), position) if position is not None else None,
            "recall": numeric_at(getattr(box, "r", None), position) if position is not None else None,
            "f1": numeric_at(getattr(box, "f1", None), position) if position is not None else None,
            "map50_95": numeric_at(maps, index) if position is not None else None,
        }
        if (
            position is not None
            and isinstance(all_ap, list)
            and position < len(all_ap)
            and isinstance(all_ap[position], list)
            and all_ap[position]
        ):
            try:
                row["map50"] = float(all_ap[position][0])
            except (TypeError, ValueError):
                pass
        per_class.append(row)
    payload["per_class"] = per_class
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_path = resolve_from_root(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    data_path = resolve_ultralytics_data_yaml(data_path)
    args.batch = parse_auto_int(args.batch, recommended_val_batch(args.imgsz), "batch", allow_zero=False)
    args.workers = parse_auto_int(args.workers, recommended_workers("val"), "workers")
    args.device = recommended_device(args.device)
    print(
        f"[hardware] val device={args.device} batch={args.batch} workers={args.workers} imgsz={args.imgsz}",
        flush=True,
    )

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
    val_args["device"] = args.device
    if args.conf is not None:
        val_args["conf"] = args.conf
    if args.iou is not None:
        val_args["iou"] = args.iou
    class_ids = parse_classes(args.classes)
    if class_ids is not None:
        val_args["classes"] = class_ids

    model = YOLO(resolve_local_model(args.model))
    metrics = model.val(**val_args)
    if args.metrics_json:
        metrics_path = resolve_from_root(args.metrics_json)
        write_json(metrics_path, metrics_payload(metrics, args))
        print(f"wrote_metrics={metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
