# Fragment Classifier Plan

Goal: recognize the correct denomination from partially visible banknote slices while keeping the final CashSnap model practical for phone/browser deployment.

## Why This Path

Fan scenes are not full-note detection scenes. Many visible regions are narrow strips, backs, corners, or grip-occluded pieces. A single detector currently entangles two jobs:

- find the visible bill fragment
- identify the denomination from incomplete evidence

Splitting those jobs should reduce class confusion:

1. A small YOLO detector proposes visible bill fragments or slices.
2. A tiny classifier re-reads each proposed crop and predicts denomination/currency/background.
3. The app counts only high-confidence, non-duplicate denomination predictions.

This is still deployable: YOLO26n/YOLO11n-style detector plus a MobileNetV3/EfficientNet-Lite-size classifier can run in browser or phone runtimes after ONNX/NCNN export.

## Dataset

Create `data/fragment_classifier_v1/`:

```text
data/fragment_classifier_v1/
  train/KHR_500/*.jpg
  train/KHR_1000/*.jpg
  ...
  train/background/*.jpg
  val/KHR_500/*.jpg
  val/background/*.jpg
  manifest.csv
```

Sources:

- visible crops from verified real fan/overlap labels
- synthetic visible-mask crops from the compositor
- clean note assets cropped into strips, corners, backs, and partial numerals
- hard negatives from fingers, red cloth/table, jewelry, wallets, and non-bill background
- confusing currency negatives, especially USD crops, to suppress USD false positives on KHR-only scenes

Each crop should preserve enough context to include print texture and edge cues, but not so much that the classifier learns background shortcuts.

## Generation Rules

For each verified full-note or transparent note asset:

- sample strip widths from about 8-45% of the note
- sample front/back/corner/center/denomination-number regions
- rotate/perspective-warp mildly
- add glare, blur, compression, exposure shift, and paper wear
- add partial finger/hand occlusion only if the denomination remains human-identifiable
- save ambiguous fragments to `background` or skip them, not to a denomination class

For synthetic fan scenes:

- use the compositor's visible masks
- crop each visible instance box with padding
- drop crops below a minimum visible area or with too little class evidence
- include detector false positives as hard negatives after manual review

## Validation Gates

Before using a classifier in the app:

- normal classifier val accuracy per class, not just overall accuracy
- confusion matrix focused on `KHR_5000`, `KHR_10000`, `KHR_20000`, and `KHR_50000`
- real overlap/fan crop set scored separately
- false-positive rate on USD/background hard negatives
- detector-plus-classifier end-to-end count score on `data/real_fan_benchmark/drafts/`

## First Implementation Step

Use `scripts/build_fragment_classifier_dataset.py` to generate fragment crops from:

- `data/asset_candidates/khr_nbc_current_cutout_bank_v1/`
- human-reviewed PicWish cutouts
- synthetic visible masks under `data/synthetic/*/masks/`
- optional YOLO proposal crops from real benchmark drafts, kept outside official benchmark labels

The script writes an ImageFolder-style dataset plus `manifest.csv` and `contact_sheet.jpg`. A smoke dataset at `data/fragment_classifier_smoke_v1/` verifies the builder, but it is not training data.

Do not train this from unreviewed PicWish gold assets or specimen-marked reference crops alone. The recent `shape_skin30_current` detector run showed that cleaner automatic shape filters still regressed validation and introduced USD false positives.

## Current Probe Results

- `scripts/train_fragment_classifier.py` trains MobileNetV3-small classifiers and exports ONNX using the legacy PyTorch exporter path.
- Candidate reference/strict-cutout fragments (`data/fragment_classifier_v1_candidate/`) reached `0.888` held-out synthetic test accuracy with ImageNet-pretrained MobileNetV3, but failed on the real overlap photo because the visual domain was too reference-like.
- Real CashSnap YOLO-box fragments (`data/fragment_classifier_cashsnap_realfrag_v1/`) plus balanced loss/sampling reached `0.922` test accuracy, but still misclassified the Commons old-overlap image as USD-heavy.
- Real fragments plus legacy NBC supplementation (`data/fragment_classifier_cashsnap_realfrag_plus_legacy_v1/`) reached `0.910` test accuracy. In the two-stage diagnostic, class-agnostic YOLO proposals plus this classifier correctly recovered the top-left old `KHR_20000`, but still confused the right-side `KHR_20000` back as `KHR_1000` and lower old notes as `KHR_500/50000`.
- Backfocused legacy reverse-side oversampling (`data/fragment_classifier_cashsnap_realfrag_plus_legacy_backfocus_v1`, `mobilenet_v3_realfrag_legacy_backfocus_pretrained_balanced_e8`) reached `0.908` test accuracy and exported ONNX, with small test gains for `KHR_20000`/`KHR_50000`. It did not transfer to the shop overlap probe: classifier replacement scored 0/6 same-class matches, while the detector's own classes on the same proposals scored 2/6 to 3/6. KHR-only probability gating also failed, mostly shifting USD mistakes into `KHR_1000`/`KHR_10000`.
- `scripts/build_yolo_crop_review_pack.py` now builds auditable crop sheets from non-benchmark YOLO labels, and `scripts/build_fragment_classifier_from_review_pack.py` converts rows marked with `review_include` into an ImageFolder training set. The first focused pack (`data/review/cashsnap_old_common_khr_crop_review_v1`) has 844 crops for `KHR_1000`, `KHR_5000`, `KHR_10000`, and `KHR_20000`; only 55 are `KHR_20000`, while `KHR_1000` has many back-like blue/gray real crops that visually explain the current `KHR_20000` back confusion.
- A focused old/common KHR real-box classifier (`data/fragment_classifier_cashsnap_old_common_khr_realbox_v1`, `mobilenet_v3_old_common_khr_realbox_pretrained_balanced_e12`) reached `0.948` test accuracy on those four classes. Classifier replacement alone matched the detector at 3/6 same-class on the shop overlap draft labels, but `scripts/fuse_two_stage_csv.py` with detector overrides around `det_conf >= 0.17` plus detector-confidence NMS reached 6/6 region coverage, 6 predictions for 6 draft notes, and 5/6 same-class recall.
- Adding modest legacy reference fragments to the focused real-box classifier (`data/fragment_classifier_cashsnap_old_common_khr_realbox_plus_legacyfrag_v1`, `mobilenet_v3_old_common_khr_realbox_legacyfrag_pretrained_balanced_e10`) did not beat that fused result. It reached `0.922` test accuracy and the same 5/6 fused shop-overlap score, so the remaining gap should be handled by more real reviewed crops or calibration, not more reference fragments.
- Broad 14-class MobileNetV3 classifiers are not ready to replace the focused KHR specialist in the two-stage app. `mobilenet_v3_cashsnap_realfrag_pretrained_balanced_e10` reached decent held-out crop metrics but only 2/6 same-class on the shop-overlap fusion sweep; `mobilenet_v3_realfrag_plus_legacy_pretrained_balanced_e8` also stayed at 2/6, with KHR slices drifting into USD/low-KHR classes. Keep the browser diagnostic on the focused KHR classifier until a reviewed real partial-note set supports a general USD+KHR classifier.
- A 3-class currency gate (`data/fragment_currency_gate_realfrag_v1`, `mobilenet_v3_currency_gate_realfrag_pretrained_balanced_e6`) reached `0.953` held-out test accuracy and exported ONNX, but failed on the shop-overlap proposals: most KHR proposal crops were routed as `USD` with high confidence. Do not rely on `cashsnap_v1` crop accuracy as a proxy for real old/overlapped slice robustness. If regenerating the gate dataset, use `scripts/build_currency_gate_dataset.py`'s default hardlink mode to avoid duplicating crop storage.
- Unreviewed Roboflow partial-crop diagnostic (`data/fragment_classifier_roboflow_partial_khr_diag_v1`, `mobilenet_v3_roboflow_partial_khr_unreviewed_diag_e4`) combined the 5k/10k and 20k/50k review packs and exported ONNX, but best val accuracy was only `0.325` after 4 CPU epochs. Treat this as a pipeline smoke only: the source splits/lookalikes and unreviewed fragments are not training-trustworthy yet.
- Current focused old/common classifier on the Roboflow overlap subset (`data/fragment_classifier_roboflow_partial_khr_oldcommon_eval_v1`) reached only `0.615` val and `0.593` test accuracy. `KHR_10000` is strong (`40/40` val, `26/30` test), but `KHR_20000` and `KHR_5000` mostly collapse into `KHR_10000`; prediction CSVs under `data/audit/roboflow_cuurecy_detection_is/oldcommon_classifier_*_predictions.csv` identify the high-confidence misses, and `data/review/roboflow_cuurecy_detection_is_oldcommon_highconf_failure_review_v1/` packages the top 31 for review.
- Rebuilt P1 focus queue (`data/fragment_classifier_p1_oldcommon_focus_unreviewed_diag_v2`) is an even harsher diagnostic for the current old/common classifier: val accuracy is `0.088` and test accuracy is `0.067`; `KHR_20000` and `KHR_5000` almost entirely predict as `KHR_10000`, while the tiny `KHR_10000` val slice is 2/2. Treat this as confirmation that the missing data is reviewed real thin/edge `KHR_5000` and `KHR_20000`, not another broad classifier architecture change.
- A diagnostic refresh that mixed the old/common real-box train split with unreviewed P1 train crops (`mobilenet_v3_oldcommon_realbox_plus_p1_unreviewed_diag_e6`) improved P1 val/test to `0.588`/`0.667` and preserved the original old/common test split at `0.991`, but it regressed the real shop-overlap fusion to `3/6` same-class at detector override `0.17` and `2/6` at `0.20`. Do not promote it; the P1 queue needs human review and a separate real overlap/fan validation set.
- Existing Khmer OCR is worth probing only as a weak cue. `scripts/probe_khmer_ocr_cues.py` ran `mer` on full/top/middle/bottom/left/right crops for the shop-overlap draft; outputs contained scattered Khmer/Latin fragments and wrong/partial digits (`KHR_20000` examples included `1974` and `7`), so OCR should not replace the fragment classifier path.

Conclusion: the detector-plus-classifier architecture is viable and browser/phone-exportable, but reference-derived reverse fragments are not enough for worn real old notes. The next useful classifier data should be human-verified real old/common KHR front/back crops, especially backs of `KHR_5000`, `KHR_10000`, and `KHR_20000`, plus a separate real fan/overlap crop validation set.
