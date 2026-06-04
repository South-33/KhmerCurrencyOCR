from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "USD_1",
    "USD_5",
    "USD_10",
    "USD_20",
    "USD_50",
    "USD_100",
    "KHR_500",
    "KHR_1000",
    "KHR_2000",
    "KHR_5000",
    "KHR_10000",
    "KHR_20000",
    "KHR_50000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug ONNX detector proposal drift from preprocessing choices.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--mode", choices=["cv2", "pil"], default="cv2")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def letterbox_pil(image_path: Path, size: int) -> tuple[np.ndarray, dict[str, float | int | tuple[int, int]]]:
    image = Image.open(image_path).convert("RGB")
    scale = min(size / image.width, size / image.height)
    resized = (round(image.width * scale), round(image.height * scale))
    pad = ((size - resized[0]) // 2, (size - resized[1]) // 2)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(image.resize(resized, Image.Resampling.BILINEAR), pad)
    array = np.asarray(canvas).astype("float32") / 255.0
    return np.transpose(array, (2, 0, 1))[None], {"scale": scale, "pad_x": pad[0], "pad_y": pad[1], "resized": resized}


def letterbox_cv2(image_path: Path, size: int) -> tuple[np.ndarray, dict[str, float | int | tuple[int, int]]]:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized = (round(width * scale), round(height * scale))
    pad = ((size - resized[0]) // 2, (size - resized[1]) // 2)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad[1] : pad[1] + resized[1], pad[0] : pad[0] + resized[0]] = cv2.resize(
        image,
        resized,
        interpolation=cv2.INTER_LINEAR,
    )
    array = canvas.astype("float32") / 255.0
    return np.transpose(array, (2, 0, 1))[None], {"scale": scale, "pad_x": pad[0], "pad_y": pad[1], "resized": resized}


def run_detector(model_path: Path, tensor: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: tensor})[0]


def proposal_rows(output: np.ndarray, meta: dict[str, float | int | tuple[int, int]], conf: float) -> list[dict[str, str]]:
    scale = float(meta["scale"])
    pad_x = float(meta["pad_x"])
    pad_y = float(meta["pad_y"])
    rows: list[dict[str, str]] = []
    for index, raw in enumerate(output.reshape(-1, 6)):
        score = float(raw[4])
        if score < conf:
            continue
        class_id = int(round(float(raw[5])))
        rows.append(
            {
                "index": str(index),
                "x1": f"{(float(raw[0]) - pad_x) / scale:.1f}",
                "y1": f"{(float(raw[1]) - pad_y) / scale:.1f}",
                "x2": f"{(float(raw[2]) - pad_x) / scale:.1f}",
                "y2": f"{(float(raw[3]) - pad_y) / scale:.1f}",
                "class_id": str(class_id),
                "class_name": CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else str(class_id),
                "confidence": f"{score:.6f}",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    model_path = resolve(args.model)
    image_path = resolve(args.image)
    preprocess = letterbox_cv2 if args.mode == "cv2" else letterbox_pil
    tensor, meta = preprocess(image_path, args.imgsz)
    output = run_detector(model_path, tensor)
    rows = proposal_rows(output, meta, args.conf)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["class_name"]] = counts.get(row["class_name"], 0) + 1
    print(f"mode={args.mode} output_shape={tuple(output.shape)} resized={meta['resized']} pad=({meta['pad_x']},{meta['pad_y']}) proposals={len(rows)}")
    print("classes=" + ";".join(f"{name}:{count}" for name, count in sorted(counts.items())))
    if args.out:
        out = resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["index"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
