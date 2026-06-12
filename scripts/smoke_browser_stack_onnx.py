from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from local_runtime import configure_project_cache

configure_project_cache()

import onnxruntime as ort
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_LIST = (
    ROOT
    / "configs"
    / "generated_lists"
    / "audit"
    / "cashsnap_v1_semantic_plus_leakage_clean_no_khmer_us_currency_test_v1.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test CashSnap browser stack ONNX artifacts.")
    parser.add_argument("--config", action="append", required=True, type=Path, help="Browser stack JSON config.")
    parser.add_argument("--image", type=Path, default=None, help="Image to smoke. Defaults to --image-list index.")
    parser.add_argument("--image-list", type=Path, default=DEFAULT_IMAGE_LIST)
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--max-proposals", type=int, default=5)
    parser.add_argument("--json-out", required=True, type=Path)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{repo_rel(path)}: expected JSON object")
    return payload


def choose_image(args: argparse.Namespace) -> Path:
    if args.image is not None:
        return resolve(args.image)
    list_path = resolve(args.image_list)
    rows = [
        line.strip()
        for line in list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not rows:
        raise SystemExit(f"{repo_rel(list_path)}: no image rows")
    if args.image_index < 0 or args.image_index >= len(rows):
        raise SystemExit(f"--image-index {args.image_index} outside list length {len(rows)}")
    return resolve(rows[args.image_index])


def crop_with_padding(image: Image.Image, box: list[float], padding: float) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return image.crop(
        (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(width, int(x2 + pad_x)),
            min(height, int(y2 + pad_y)),
        )
    ).copy()


def preprocess_crop(crop: Image.Image, input_size: int, normalization: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(normalization.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.asarray(normalization.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    image = crop.convert("RGB").resize((input_size, input_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    return np.transpose(array, (2, 0, 1))[None, :, :, :].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def gate_predictions(
    *,
    session: ort.InferenceSession,
    classes: list[str],
    crops: list[Image.Image],
    input_size: int,
    normalization: dict[str, Any],
) -> list[dict[str, Any]]:
    if not crops:
        return []
    input_name = session.get_inputs()[0].name
    outputs: list[np.ndarray] = []
    for crop in crops:
        logits = session.run(None, {input_name: preprocess_crop(crop, input_size, normalization)})[0]
        outputs.append(logits)
    probabilities = softmax(np.concatenate(outputs, axis=0))
    rows: list[dict[str, Any]] = []
    for probs in probabilities:
        best_index = int(probs.argmax())
        rows.append(
            {
                "gate_class": classes[best_index],
                "gate_conf": round(float(probs[best_index]), 6),
                "gate_probs": {
                    classes[index]: round(float(probs[index]), 6)
                    for index in range(len(classes))
                },
            }
        )
    return rows


def ort_providers_without_tensorrt() -> list[str]:
    providers = [provider for provider in ort.get_available_providers() if provider != "TensorrtExecutionProvider"]
    return providers or ["CPUExecutionProvider"]


def smoke_config(config_path: Path, image_path: Path, max_proposals: int) -> dict[str, Any]:
    config = load_json(config_path)
    detector = config.get("detector", {})
    fragment = config.get("fragment_classifier", {})
    fusion = config.get("fusion", {})
    if not isinstance(detector, dict) or not isinstance(fragment, dict) or not isinstance(fusion, dict):
        raise SystemExit(f"{repo_rel(config_path)}: detector, fragment_classifier, and fusion must be objects")

    detector_path = resolve(str(detector.get("path", "")))
    gate_path = resolve(str(fragment.get("path", "")))
    detector_classes = [str(name) for name in detector.get("classes", [])]
    gate_classes = [str(name) for name in fragment.get("classes", [])]
    if not detector_classes or not gate_classes:
        raise SystemExit(f"{repo_rel(config_path)}: detector/classes and fragment_classifier/classes are required")
    detector_class_min_conf = {
        str(class_name): float(value)
        for class_name, value in detector.get("class_min_conf", {}).items()
    }

    model = YOLO(str(detector_path), task="detect")
    results = model.predict(
        source=str(image_path),
        imgsz=int(detector.get("input_size", 416)),
        conf=float(detector.get("proposal_confidence", 0.05)),
        iou=float(detector.get("proposal_iou", 0.7)),
        agnostic_nms=bool(detector.get("agnostic_nms", False)),
        verbose=False,
    )
    result = results[0]
    proposals: list[dict[str, Any]] = []
    crops: list[Image.Image] = []
    with Image.open(image_path) as image_raw:
        image = image_raw.convert("RGB")
        boxes = result.boxes
        if boxes is not None:
            for box in boxes[: max(0, max_proposals)]:
                class_id = int(box.cls[0].item())
                class_name = detector_classes[class_id] if class_id < len(detector_classes) else str(class_id)
                confidence = float(box.conf[0].item())
                if class_name in detector_class_min_conf and confidence < detector_class_min_conf[class_name]:
                    continue
                xyxy = [float(value) for value in box.xyxy[0].tolist()]
                proposals.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 6),
                        "xyxy": [round(value, 2) for value in xyxy],
                    }
                )
                crops.append(crop_with_padding(image, xyxy, float(fragment.get("crop_padding", 0.06))))

    gate_session = ort.InferenceSession(str(gate_path), providers=ort_providers_without_tensorrt())
    gate_rows = gate_predictions(
        session=gate_session,
        classes=gate_classes,
        crops=crops,
        input_size=int(fragment.get("input_size", 224)),
        normalization=fragment.get("normalization", {}),
    )

    reject_classes = set(str(name) for name in fusion.get("reject_fragment_classes", []))
    reject_min_conf = float(fusion.get("reject_fragment_class_min_conf", 1.0))
    class_reject_min_conf = {
        str(class_name): float(value)
        for class_name, value in fusion.get("reject_fragment_class_min_conf_by_detector_class", {}).items()
    }
    class_reject_min_detector_conf = fusion.get("reject_fragment_class_override_min_detector_conf")
    if class_reject_min_detector_conf is not None:
        class_reject_min_detector_conf = float(class_reject_min_detector_conf)
    fused_rows = []
    rejected = 0
    for proposal, gate_row in zip(proposals, gate_rows, strict=False):
        proposal_reject_min_conf = reject_min_conf
        proposal_class = str(proposal.get("class_name"))
        if proposal_class in class_reject_min_conf and (
            class_reject_min_detector_conf is None
            or float(proposal.get("confidence", 0.0)) >= class_reject_min_detector_conf
        ):
            proposal_reject_min_conf = class_reject_min_conf[proposal_class]
        is_rejected = (
            gate_row["gate_class"] in reject_classes
            and float(gate_row["gate_conf"]) >= proposal_reject_min_conf
        )
        rejected += int(is_rejected)
        fused_rows.append(
            {
                **proposal,
                **gate_row,
                "reject_min_conf_effective": proposal_reject_min_conf,
                "rejected": is_rejected,
            }
        )

    return {
        "config": repo_rel(config_path),
        "name": str(config.get("name", config_path.stem)),
        "status": str(config.get("status", "")),
        "detector": repo_rel(detector_path),
        "detector_class_min_conf": detector_class_min_conf,
        "gate": repo_rel(gate_path),
        "prediction_count": len(proposals),
        "rejected_count": rejected,
        "gate_execution_providers": gate_session.get_providers(),
        "top_predictions": fused_rows,
    }


def main() -> None:
    args = parse_args()
    image_path = choose_image(args)
    output = {
        "schema": "cashsnap_browser_stack_onnx_smoke_v1",
        "image": repo_rel(image_path),
        "configs": [smoke_config(resolve(config_path), image_path, args.max_proposals) for config_path in args.config],
    }
    out_path = resolve(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    print(f"wrote={repo_rel(out_path)}")


if __name__ == "__main__":
    main()
