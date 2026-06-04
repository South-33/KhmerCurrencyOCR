from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GALLERY = [
    ROOT / "data" / "asset_candidates" / "khr_nbc_circulation_legacy_cutout_bank_v1",
    ROOT / "data" / "asset_candidates" / "numista_khr_1990plus_cutout_bank_v1",
]
DEFAULT_MANIFEST = ROOT / "data" / "review" / "p1_focus_v2_oldcommon_failure_review_v1" / "manifest.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class FeatureRef:
    path: Path
    class_name: str
    keypoints: int
    descriptors: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether local template features can classify partial banknote crops.",
    )
    parser.add_argument("--gallery", nargs="+", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "audit" / "template_feature_probe.csv")
    parser.add_argument("--classes", default="KHR_5000,KHR_10000,KHR_20000")
    parser.add_argument("--method", choices=["sift", "akaze", "orb"], default="sift")
    parser.add_argument("--ratio", type=float, default=0.76, help="Lowe ratio-test threshold.")
    parser.add_argument("--max-side", type=int, default=720)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def class_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if part.startswith("KHR_") or part.startswith("USD_"):
            return part
    return path.parent.name


def iter_images(root: Path, classes: set[str]) -> list[Path]:
    paths: list[Path] = []
    for path in resolve(root).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        lower_parts = {part.lower() for part in path.parts}
        if "masks" in lower_parts or "audit" in lower_parts:
            continue
        if class_from_path(path) in classes:
            paths.append(path)
    return sorted(paths)


def read_gray(path: Path, max_side: int) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        image = (rgb * alpha[:, :, None] + white * (1.0 - alpha[:, :, None])).astype(np.uint8)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    height, width = gray.shape[:2]
    scale = max_side / max(height, width)
    if scale < 1:
        gray = cv2.resize(gray, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def make_detector(method: str):
    if method == "sift":
        return cv2.SIFT_create(nfeatures=900, contrastThreshold=0.018)
    if method == "akaze":
        return cv2.AKAZE_create()
    return cv2.ORB_create(nfeatures=900, fastThreshold=8)


def norm_for_method(method: str) -> int:
    return cv2.NORM_L2 if method == "sift" else cv2.NORM_HAMMING


def extract(path: Path, detector, max_side: int) -> tuple[int, np.ndarray | None]:
    gray = read_gray(path, max_side)
    if gray is None:
        return 0, None
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return len(keypoints), descriptors


def load_gallery(paths: list[Path], detector, max_side: int) -> list[FeatureRef]:
    refs: list[FeatureRef] = []
    for path in paths:
        keypoints, descriptors = extract(path, detector, max_side)
        if descriptors is None or keypoints < 8:
            continue
        refs.append(FeatureRef(path=path, class_name=class_from_path(path), keypoints=keypoints, descriptors=descriptors))
    return refs


def read_rows(path: Path) -> list[dict[str, str]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def target_class(row: dict[str, str]) -> str:
    for key in ["review_class", "target", "canonical_class", "class_name"]:
        value = row.get(key, "").strip()
        if value:
            return value
    return ""


def match_score(query_desc: np.ndarray, query_kp: int, ref: FeatureRef, matcher, ratio: float) -> tuple[int, float]:
    if query_desc is None or len(query_desc) < 2 or len(ref.descriptors) < 2:
        return 0, 0.0
    matches = matcher.knnMatch(query_desc, ref.descriptors, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    good_count = len(good)
    norm = good_count / max(1.0, math.sqrt(query_kp * ref.keypoints))
    return good_count, norm


def main() -> None:
    args = parse_args()
    classes = {item.strip() for item in args.classes.split(",") if item.strip()}
    detector = make_detector(args.method)
    matcher = cv2.BFMatcher(norm_for_method(args.method))
    gallery_paths = [path for root in args.gallery for path in iter_images(root, classes)]
    refs = load_gallery(gallery_paths, detector, args.max_side)
    if not refs:
        raise SystemExit("No usable gallery features found.")

    rows: list[dict[str, str]] = []
    correct = 0
    attempted = 0
    class_counts: Counter[str] = Counter()
    for row in read_rows(args.manifest):
        crop_path = row.get("crop_path", "").strip()
        target = target_class(row)
        if not crop_path or target not in classes:
            continue
        query_kp, query_desc = extract(resolve(Path(crop_path)), detector, args.max_side)
        best: tuple[str, str, int, float] = ("", "", 0, 0.0)
        second_score = 0.0
        for ref in refs:
            good, score = match_score(query_desc, query_kp, ref, matcher, args.ratio)
            if score > best[3]:
                second_score = best[3]
                best = (ref.class_name, repo_path(ref.path), good, score)
            elif score > second_score:
                second_score = score
        pred = best[0]
        attempted += 1
        correct += int(pred == target)
        class_counts[target] += 1
        rows.append(
            {
                "crop_id": row.get("crop_id", ""),
                "crop_path": crop_path,
                "target": target,
                "prediction": pred,
                "correct": str(pred == target),
                "query_keypoints": query_kp,
                "best_good_matches": best[2],
                "best_score": f"{best[3]:.6f}",
                "score_margin": f"{(best[3] - second_score):.6f}",
                "best_asset": best[1],
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "crop_id",
        "crop_path",
        "target",
        "prediction",
        "correct",
        "query_keypoints",
        "best_good_matches",
        "best_score",
        "score_margin",
        "best_asset",
    ]
    with resolve(args.out).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    accuracy = correct / attempted if attempted else 0.0
    print(f"gallery_refs={len(refs)} attempted={attempted} accuracy={accuracy:.3f} out={repo_path(resolve(args.out))}")
    print(f"targets={dict(sorted(class_counts.items()))}")


if __name__ == "__main__":
    main()
