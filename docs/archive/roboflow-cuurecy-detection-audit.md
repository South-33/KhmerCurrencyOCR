# Roboflow cuurecy-detection-is Audit

Dataset: [cuurecy-detection-is](https://universe.roboflow.com/ddd8889/cuurecy-detection-is)

Local path: `data/raw_datasets/roboflow_cuurecy_detection_is/`

Status: promising internal data lead, not ready for blind training or public release.

## What It Has

- 2,329 images and 5,689 YOLOv8 segmentation labels.
- 22 raw classes: KHR/USD front and back labels such as `10000-riel-f`, `20000-riel-b`, `5-US-f`, and `100-us-b`.
- Manifested as 5,067 CashSnap-core objects plus 622 non-core `KHR_100` objects.
- 3,438 edge-touching objects and 294 tiny-box objects, which is useful for partial/off-frame recognition.
- Front/back balance: 3,003 front objects and 2,686 back objects.

## Audit Results

All heavy-ish scans were run through `scripts/run_with_headroom.py`.

- Exact duplicate/split check: `scripts/check_duplicates.py --dataset cuurecy_detection_is --exact-only`
  - 0 exact duplicate groups.
  - 0 exact cross-split groups.
- Near duplicate check: `scripts/check_duplicates.py --dataset cuurecy_detection_is --threshold 4`
  - 6,432 dHash lookalike pairs.
  - 3,003 cross-split lookalike pairs.
  - 0 same-original Roboflow stem pairs.
  - Interpretation: no obvious same-source leakage, but the provided train/valid/test splits are too lookalike-heavy for strong validation claims.
- Segmentation geometry: `scripts/audit_yolo_segmentation_geometry.py`
  - 5,689 valid segmentation rows.
  - 0 malformed, out-of-bounds, or zero-area polygon issues.
- Visual QA:
  - Random/edge/tiny contact sheets show real phone, partial, off-frame, hand-held, and overlap examples.
  - Tiny masks mostly look like real far/partial/off-frame notes rather than broken labels.
  - Repeated layouts and hand occlusion are common enough to require curation.

## Review Packs

- `data/review/roboflow_cuurecy_detection_is_khr_20k_50k_partial_review_v1/`
  - 185 edge-touching/small-area `20000-riel-b/f` and `50000-riel-b/f` crops.
  - `review_class` is canonicalized to `KHR_20000` or `KHR_50000`; `side` preserves front/back.
- `data/review/roboflow_cuurecy_detection_is_khr_5k_10k_partial_review_v1/`
  - 189 edge-touching/small-area `5000-riel-b/f` and `10000-riel-b/f` crops.
  - Useful for the old/common KHR confusion path.

## Release Caution

Local export metadata says `License: CC BY 4.0`, and Roboflow Universe exposes dataset licenses in its cite block. CC BY helps with attribution, but it does not settle currency reproduction rules or other rights.

Treat public release separately from internal training:

- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) allows sharing/adaptation with attribution, subject to other rights.
- [U.S. BEP currency-image rules](https://www.bep.gov/currency/currency-image-use) constrain USD banknote illustrations.
- [Cambodia National Bank law Article 48](https://library.ncdd.gov.kh/detail/8656/download) forbids reproduction of notes, coins, checks, securities, or payment cards without prior written Central Bank authorization.

## Current Decision

Use this source for curated internal review and possible fragment-classifier data, especially backs/partials of `KHR_5000`, `KHR_10000`, `KHR_20000`, and `KHR_50000`.

Do not use the dataset's original splits as proof of generalization. Also do not treat its denomination labels as proof that a note design is in the current product scope; verify issue era/circulation against Numista `in_circulation` folders or the circulation-scope docs before promoting examples into trusted training. Use rights-clear phone photos and reviewed local validation for final model claims.
