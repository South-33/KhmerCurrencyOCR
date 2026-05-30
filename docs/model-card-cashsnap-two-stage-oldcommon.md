# CashSnap Two-Stage Old/Common KHR Diagnostic Model Card

## Purpose

Small phone/browser-oriented diagnostic stack for partial and overlapped KHR banknote counting experiments. This is not a production CashSnap model.

## Artifacts

- Stack config: `configs/cashsnap_two_stage_oldcommon_browser_stack.json`
- Detector ONNX: `runs/cashsnap/yolo26n_legacy_clean_plus_realcutout_low_skin_ft_e6_i416_b8/weights/best.onnx`
- Fragment classifier ONNX: `runs/fragment_classifier/mobilenet_v3_old_common_khr_realbox_pretrained_balanced_e12/best.onnx`
- Browser demo: `demo/browser/`
- Size gate: `scripts/check_browser_stack_artifacts.py` defaults to a 20 MB total ONNX artifact budget; current detector + classifier are about 15.1 MB.

## Classes

Detector classes follow the 13-class CashSnap v1 denomination list: USD 1/5/10/20/50/100 and KHR 500/1000/2000/5000/10000/20000/50000.

Fragment classifier classes are only `KHR_1000`, `KHR_5000`, `KHR_10000`, and `KHR_20000`.
The browser fusion path skips this KHR-only fragment classifier for detector proposals outside those four classes; `manifests/browser_smoke_cases.csv` includes USD_1 plus detector-only `KHR_500`, `KHR_2000`, and `KHR_50000` sanity cases to guard totals outside the classifier class list.

## Fusion

Current diagnostic fusion, refreshed on 2026-05-30:

- Detector proposal confidence: `0.05`
- Fragment crop padding: `0.0`
- Detector override confidence: `0.17` (`0.175` ties on the current draft)
- Class-agnostic NMS IoU: `0.85` (`0.75-0.90` ties on the current draft)
- NMS ranking score: detector confidence

## Current Evidence

On draft labels for `real_overlap_0003_commons_shop_5k_10k_20k`, the focused old/common KHR classifier plus detector-threshold fusion reaches:

- `6/6` any-class visible-region matches
- `5/6` same-class matches
- `6` predictions for `6` draft visible notes

The deployable browser path should be checked with `scripts/smoke_browser_demo_cdp.cjs`, because Edge ONNX Runtime Web/canvas preprocessing is the deployment truth and can differ from PyTorch-generated proposal CSVs around borderline detector scores. Current Edge smoke-suite refresh on the same shop-overlap image predicts `6` bills with `KHR 76,000`, `USD 0`, and classes `KHR_1000:1;KHR_10000:1;KHR_20000:3;KHR_5000:1`. Run the smoke with `--labels data/real_fan_benchmark/drafts/real_overlap_0003_commons_shop_5k_10k_20k.txt`; its JSON reports `6/6` any-class matches, `4/6` same-class matches, a `+6000` KHR value error, and matched-pair confusions `KHR_10000->KHR_1000` plus `KHR_5000->KHR_20000`, plus per-source recall showing final/detector labels at `4/6` and fragment-only labels at `3/6`.

The 2026-05-30 `scripts/run_browser_smoke_cases.py` suite also passed the USD_1 and detector-only `KHR_500`, `KHR_2000`, and `KHR_50000` sanity cases, so the current KHR-only fragment classifier is not overwriting denominations outside its class list in those guards.

Temporary browser override `--detector-override 0.20` keeps the shop-overlap count and 4/6 same-class score but changes the value error from `+6000` KHR to `-4000` KHR by turning the weak false `KHR_20000` into `KHR_10000`; the same override still passes the USD_1 and detector-only KHR guard cases. Do not change the default config from one image; use the smoke-runner override flags for calibration sweeps once more labeled real cases exist.

`scripts/inspect_two_stage_matches.py` now makes row-level fusion failures easier to inspect. On the PyTorch-side `det0.17/nms0.85/detconf` fused shop-overlap CSV, the remaining same-class miss is a high-confidence fragment error: visible `KHR_5000` is predicted as `KHR_10000` with fragment confidence `0.9826`, while the detector label is also wrong (`KHR_20000`). This is not a simple low-confidence rejection problem; it needs reviewed real `KHR_5000`/`KHR_20000` partial crops or a stronger local verifier.

Diagnostic duplicate-class relabeling is a limited calibration clue, not a browser default. On a permissive alpha-scan proposal CSV, `fuse_two_stage_csv.py --prefer-supported-detector-duplicates --det-threshold 0.03 --nms-iou 0.30 --nms-score-column detector_conf` recovers a lower-confidence same-box `KHR_5000` detector alternative, yielding `5/6` same-class matches with `5` predictions. On the configured browser-stack detector at `416/conf=0.05` and `det0.17/nms0.85`, the same option makes no relabels and the stack remains `5/6`; at permissive `416/conf=0.03` it can regress. Treat it as a debugging lens, not a production rule.

Browser smoke debug currently reports detector output dims `[1,300,6]`, `13` browser proposals, and `6` final detections. Ultralytics ONNX Runtime on the same detector artifact reaches `6/6` detector same-class recall at `416/conf=0.05`, so remaining browser work should focus on classifier/crop parity and data quality rather than blaming the ONNX file itself.

`scripts/debug_onnx_detector_preprocess.py` isolates the sensitivity: on the shop-overlap image, the detector ONNX produces `13` proposals with `cv2` resize and `8` proposals with PIL-style resize at the same `416/conf=0.05`, including a PIL-side `USD_100` proposal. Treat browser canvas preprocessing as a first-class deployment variable for thin/partial slices.

Broad 14-class fragment classifiers and a 3-class KHR/USD/background gate did not transfer to this real shop-overlap probe; both remained too confused for browser/mobile deployment.

Raw detector-only thresholding is not enough. In the same refresh, the detector can reach full same-class recall only at `416/conf=0.03` with `23` predictions for `6` notes, while `416/conf=0.05` gives `5/6` same-class and `14` predictions. Existing Khmer OCR (`mer`) also failed as a shortcut on the same crops, returning scattered text and wrong/partial denomination digits, so OCR remains optional auxiliary evidence rather than the core path.

## Known Limits

- Tuned against one draft-labeled real image, so treat metrics as calibration evidence, not a benchmark result.
- Does not cover all KHR denominations, rare current notes, USD fragment re-reading, or reliable old/new issue disambiguation.
- Real fan, hand occlusion, off-frame, worn-note, and mixed KHR/USD performance is not solved.
- Production claims require a rights-clear reviewed phone-photo benchmark and non-benchmark reviewed training crops.
- All main crop-review packs currently have blank `review_include` fields. Run `scripts/summarize_review_manifests.py`, curate with the `demo/review/` presets, and merge exports with `scripts/apply_review_export.py` before building trusted fragment-classifier data.

## Required Next Data

Track new captures in `manifests/real_partial_capture_inventory.csv` and run:

```powershell
rl python scripts/check_capture_requirements.py
```

The highest-value missing data is rights-clear phone photos with human-identifiable partial KHR slices, especially hand fan plus thin/edge `KHR_5000` and `KHR_20000` front/back views that target the current old/common confusion pairs. Synthetic microthin `KHR_5000,KHR_20000` smoke data can be generated with `--thin-strip-min-frac 0.04 --thin-strip-max-frac 0.12`, but it is still artificial and should not replace reviewed real captures.
