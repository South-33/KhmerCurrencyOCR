# CashSnap YOLO26n Baseline

Current reset note: treat e2/e4 and related synthetic/reference fine-tunes in this document as contaminated historical baselines only. They are useful comparison points, but should not be continued for first-pass current-KHR training because their KHR sources likely mixed modern notes with older, rare, specimen-heavy, collector-like, or low-priority designs.

## Run

- Model: `yolo26n.pt`
- Dataset: `data/cashsnap_v1/data.yaml`
- Command: `python scripts/train_yolo.py --model yolo26n.pt --data data/cashsnap_v1/data.yaml --epochs 20 --imgsz 416 --batch 16 --name yolo26n_baseline_e20_i416`
- Saved run: `D:\Project\KhmerCurrencyOCR\runs\ultralytics_migrated\detect\runs\cashsnap\yolo26n_baseline_e20_i4162`
- Best weights: `D:\Project\KhmerCurrencyOCR\runs\ultralytics_migrated\detect\runs\cashsnap\yolo26n_baseline_e20_i4162\weights\best.pt`

Note: early runs were created under `C:\Users\Venom\runs`; they were moved into `runs/ultralytics_migrated/` to keep training artifacts on the D drive.

## Validation

Final validation on the curated validation split:

- Precision: `0.930`
- Recall: `0.913`
- mAP50: `0.962`
- mAP50-95: `0.920`

Weak validation classes:

- `KHR_20000`: recall `0.795`, mAP50 `0.868`
- `KHR_50000`: recall `0.591`, mAP50 `0.841`

## Test Split

Held-out test split:

- Precision: `0.958`
- Recall: `0.903`
- mAP50: `0.972`
- mAP50-95: `0.935`

Weak test classes:

- `KHR_20000`: recall `0.567`, mAP50 `0.884`
- `KHR_50000`: recall `0.524`, mAP50 `0.912`

## Fan Photo Check

Image: `D:\Download\Banknotes_of_Cambodian_Khmer_Riel.jpg`

- At `imgsz=416`, `conf=0.25`: no detections.
- At `imgsz=640`, `conf=0.05`: 3 low-confidence detections, not enough for counting.

Conclusion: the baseline is good for isolated/normal banknote photos in the curated datasets, but it is not ready for the real target scene: fanned, overlapping, hand-occluded KHR notes. The next dataset step should be a small real fan/overlap validation set plus targeted synthetic fan augmentation from clean note crops.

## Synthetic Fan Experiments

Generated KHR synthetic overlap/fan data with `scripts/generate_synthetic_fan_dataset.py`.

- v2 mixed synthetic: `yolo26n_messy_synth_e10_i416`
  - Validation: precision `0.955`, recall `0.929`, mAP50 `0.976`, mAP50-95 `0.933`
  - Test: precision `0.930`, recall `0.961`, mAP50 `0.973`, mAP50-95 `0.934`
  - Fan image: `0` detections at `416/conf=0.25`; `4` detections at `640/conf=0.05`
- v3 dense fan synthetic: `yolo26n_messy_synth_v3_e8_i416`
  - Best early checkpoint improved the fan image: `1` detection at `416/conf=0.25`; `8` detections at `640/conf=0.05`
  - Later epochs trended sideways/down on clean validation, so the run was stopped after epoch 4/early 5.
  - Current best fan checkpoint: `D:\Project\KhmerCurrencyOCR\runs\ultralytics_migrated\detect\runs\cashsnap\yolo26n_messy_synth_v3_e8_i416\weights\best.pt`
- v4 partial-slice synthetic: `yolo26n_messy_synth_v4_e4_i416_w0`
  - Validation: precision `0.926`, recall `0.942`, mAP50 `0.975`, mAP50-95 `0.925`
  - Test: precision `0.960`, recall `0.924`, mAP50 `0.981`, mAP50-95 `0.937`
  - Fan image regressed versus v3: `0` detections at `416/conf=0.25`; `6` detections at `640/conf=0.05`
- v3 plus pristine rare overlap: `yolo26n_messy_v3_pristine_overlap_e2_i416_b8`
  - Normal validation: mAP50-95 `0.930`; rare-overlap synthetic validation: mAP50 `0.584`
  - Held-out test: precision `0.963`, recall `0.962`, mAP50 `0.985`, mAP50-95 `0.939`
  - Weak rare-class test rows: `KHR_20000` recall `0.941`, mAP50-95 `0.934`; `KHR_50000` recall `0.949`, mAP50-95 `0.958`
  - Fan image: `2` detections at `416/conf=0.25`; `6` detections at `640/conf=0.05`
- v3 plus pristine rare overlap continued: `yolo26n_messy_v3_pristine_overlap_e4_i416_b8`
  - Normal validation: mAP50-95 `0.925`; rare-overlap synthetic validation: mAP50 `0.602`
  - Held-out test: precision `0.957`, recall `0.970`, mAP50 `0.985`, mAP50-95 `0.942`
  - Weak rare-class test rows: `KHR_20000` recall `0.933`, mAP50-95 `0.940`; `KHR_50000` recall `0.965`, mAP50-95 `0.968`
  - Fan image: `2` detections at `416/conf=0.25`; `7` detections at `640/conf=0.05`, including `KHR_50000` up to confidence `0.433`

Conclusion: dense synthetic overlap helps the target failure case, and pristine rare-overlap continuation improves rare-slice behavior on the stress image. The e2 checkpoint is still the normal-validation leader, while e4 leads held-out test, rare-overlap synthetic validation, and the one real fan stress image. Synthetic-only data still has not solved fanned KHR counting. The next highest-value data step is real fanned/overlapped phone photos with labels, especially for `KHR_20000` and `KHR_50000`. Real YOLO crops were extracted, but the generated v5 audit showed rectangular background artifacts; do not train on real-crop synthetic until masking/segmentation is improved.

## Current-KHR Reset Probes

Fresh reset probes use current NBC cutouts and fresh YOLO26n-family weights, not e2/e4 continuation.

- `yolo26n_current_fan_slice_v1_e5_i416_b4`: synthetic validation stayed weak (`mAP50-95` about `0.050`), and the hard fan image still had `0` detections at `conf>=0.05`.
- `yolo26n_current_thin_radial_slice_probe_v1_e5_i416_b4`: `thin_radial_slice` reduced synthetic median label area to about `5.2%`, but thin-only detect training remained too weak; hard fan only appeared below useful confidence.
- `yolo26n_current_clean_plus_thin_radial_e5_i416_b4`: best detect reset signal so far. Hard fan reaches `1` detection at `640/conf=0.05`, but the box is still oversized (`~26%` of image); overlap candidate reaches `15` detections at `960/conf=0.05`.
- `yolo26n_obb_current_thin_radial_slice_probe_v1_e5_i416_b4`: OBB is materially different, with plausible small rotated boxes on the overlap candidate (`19` detections at `960/conf=0.25`, top confidence `0.85`), but class labels are noisy and the hard fan remains unsolved.
- `yolo26n_obb_current_clean_plus_thin_radial_e5_i416_b4`: synthetic OBB validation improved (`mAP50-95` about `0.096`) but real-candidate signal weakened versus thin-only OBB, so clean+thin hybridization alone is not the missing piece.

Current conclusion: the pipeline now supports clean current-KHR detect/OBB probes, but the next high-value step is a rights-clear real fan/overlap benchmark with human-identifiable visible denominations. `real_fan_0001_voa_commons` remains a useful stress probe, not a sufficient denomination-counting scoreboard.

## Circulated-Design Overlap Probe

The Commons shop photo `real_overlap_0003_commons_shop_5k_10k_20k` is not a hand fan, but it is a useful rights-clear overlap candidate with six visually identifiable old-design notes. Its labels are kept as draft review labels under `data/real_fan_benchmark/drafts/`, not official benchmark ground truth.

- Added `scripts/render_yolo_label_preview.py` to render draft YOLO labels for human review.
- Added `scripts/evaluate_real_draft_labels.py` to score count/recall against private draft labels without moving them into the official benchmark label directory.
- `yolo26n_cashsnap_current_thin_balanced_v1_e20_i416_b8`: fresh YOLO26n detect trained from a class-balanced CashSnap subset plus current clean/thin synthetic. Validation mAP50-95 reached about `0.694`, but the draft real overlap image was mostly class-confused.
- `yolo26n_cashsnap_current_thin_legacy_v1_e20_i416_b8`: fresh YOLO26n detect trained from the same real subset plus current/thin synthetic and an explicit `target_modern_common + legacy_or_low_priority` NBC synthetic set. Validation mAP50-95 fell to about `0.551`, but real-overlap draft recall improved materially: at `416/conf=0.05`, `4/6` same-class matches and `6/6` any-class region matches, with too many extra boxes; at `640/conf=0.03`, the model predicts exactly `6` boxes but only `2/6` same-class matches.
- `yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8`: fresh YOLO26n detect trained with clean 1-3 note circulated-design synthetic instead of dense legacy overlap synthetic. Validation mAP50-95 improved to about `0.738`; old-design draft recall is lower than the dense legacy model at `416`, but less overfit and better overall. At `640/conf=0.03`, it gets `3/6` same-class matches and `5/6` any-class matches with `9` predictions.
- Historical contaminated `yolo26n_messy_v3_pristine_overlap_e2_i416_b8` remains stronger on this old-design draft image, but it is still diagnostic only and should not be used as a continuation parent.
- `yolo26n_legacy_clean_old_common_focus_ft_e6_i416_headroom_v2`: gentle continuation from the clean-legacy alpha on `data/sampled/cashsnap_old_common_focus_probe_v1/`, trained through `scripts/bench_train_with_headroom.py` at `batch=1/workers=0`. Mixed validation mAP50-95 reached `0.381`, far below the clean-legacy alpha. On the draft overlap image it improved any-class region coverage at larger image sizes (`640/conf=0.10`: `6/6` any-class, `2/6` same-class, `9` predictions; `960/conf=0.03`: `6/6` any-class, `4/6` same-class, `18` predictions), but at the most useful `416/conf=0.05` setting it reached only `4/6` same-class with `12` predictions and visibly over-predicted `KHR_1000/KHR_10000/KHR_2000`. Treat this as evidence that old/common designs are learnable, not as an alpha.

Current conclusion: explicit circulated-design coverage helps denomination recall on old KHR photos, but it is not solved. More old-common synthetic in the current style mostly increases region coverage and class confusion; the next useful step is better human-verified cutout assets/class balance and calibration pressure. The current preferred fresh-weight mobile/export candidate remains `runs/cashsnap/yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8/weights/best.pt`; treat it as an overlap-counting alpha, not a reliable denomination totaler.

## Real-Cutout PicWish Probe

The all-KHR real-photo cutout path is now end-to-end:

- `scripts/package_cashsnap_picwish_inputs.py` packaged 720 capped KHR crops across all seven KHR classes into `data/picwish_upload_batches_cashsnap_khr_v1/`.
- `scripts/process_picwish_batches.py --quiet-success` processed 720/720 PicWish removals into `data/asset_candidates/cashsnap_khr_picwish_output_v1/`.
- Alpha scoring found 320 gold, 387 review, and 13 reject outputs, but visual review showed that gold is not enough for hand-held crops because fingers can merge into the foreground mask.
- `scripts/build_cutout_bank_from_scores.py --max-skin-ratio 0.08` produced the conservative candidate bank `data/asset_candidates/cashsnap_khr_picwish_strict_low_skin_bank_v1/` with 40 cutouts; it is cleaner but drops `KHR_500`, so the synthetic probe mixed it with the current NBC cutout bank.
- `data/synthetic/khr_realcutout_low_skin_mix_v1/` generated 600 visible-mask scenes from those sources: train 484 images/4,303 boxes, val 46/436, test 70/599.

Fresh YOLO26n probe `runs/cashsnap/yolo26n_cashsnap_current_thin_realcutout_low_skin_v1_e12_i416_b8/weights/best.pt` reached mixed validation mAP50-95 `0.572`, below the clean-legacy alpha. On `real_overlap_0003_commons_shop_5k_10k_20k`, it improves permissive any-class coverage (`960/conf=0.05`: 6/6 any-class draft regions, 2/6 same-class, 11 predictions), but class labels remain noisy. Conclusion: the real-cutout path is promising for coverage, but needs human-verified cutout assets or better hand-contamination filtering before it should drive the mobile/browser alpha.

A gentle continuation from the clean-legacy alpha, `runs/cashsnap/yolo26n_legacy_clean_plus_realcutout_low_skin_ft_e6_i416_b8/weights/best.pt`, improved the overlap draft at permissive settings (`416/conf=0.05`: 5/6 same-class, 6/6 any-class, 14 predictions; `416/conf=0.03`: 6/6 same-class, 23 predictions). This proves the classes are learnable, but also shows confidence calibration and false positives are still not product-ready.

Compared against the old-common focus continuation above, the real-cutout continuation remains the stronger diagnostic candidate for the rights-clear overlap image at `416/conf=0.05`, despite its false positives. Do not promote either continuation to the browser/mobile alpha without a normal validation recovery pass and a better false-positive strategy.

Added stricter transparent-cutout QA metrics to `scripts/score_transparent_cutouts.py` and `scripts/build_cutout_bank_from_scores.py`: convex hull fill, rotated-rectangle fill, and row/column span checks. The resulting candidate banks are useful for visual triage (`cashsnap_khr_picwish_shape_strict_bank_v1`, `cashsnap_khr_picwish_shape_skin30_bank_v1`), but not enough to trust automatically. A clean curriculum from `cashsnap_khr_picwish_shape_skin30_bank_v1 + khr_nbc_current_cutout_bank_v1` produced `data/synthetic/khr_shape_skin30_current_clean_v1`, mixed via `configs/cashsnap_v1_shape_skin30_current_clean_probe.yaml`.

Continuation `runs/cashsnap/yolo26n_legacy_clean_shape_skin30_current_ft_e4_i416_headroom/weights/best.pt` completed through the headroom harness (`batch=1/workers=0`) and reached mixed validation mAP50-95 `0.443`. It is not an upgrade: on the draft overlap image, `416/conf=0.05` gets only `2/6` same-class and `3/6` any-class matches with `11` predictions; `640/conf=0.03` reaches `6/6` any-class but only `3/6` same-class and predicts six `USD_100` false positives. Conclusion: automatic shape filtering improves the asset bank but still needs human review or stronger contamination controls before training.

## Scan 2.5D Synthetic Probe

Built `data/synthetic/cashsnap_scan_2p5d_fan_v1/` from Numista scan cutouts with visible-region labels, unknown-fragment crops, soft shadows, hand primitives, class-balanced sampling, and per-note perspective warps. The 2,000-scene artifact has 9,623 exported denomination labels and 9,372 `banknote_unknown` crops; backgrounds are procedural because mining `cashsnap_v1` for real backgrounds leaked note fragments during contact-sheet QA.

Fresh YOLO26n probe `runs/cashsnap/yolo26n_cashsnap_scan_2p5d_probe_e4_i416_b2/weights/best.pt` trained 4 epochs through the headroom harness. Mixed validation improved over the run but stayed weak (`mAP50=0.185`, `mAP50-95=0.132`), so it is not an alpha. On `real_overlap_0003_commons_shop_5k_10k_20k`, it reaches useful region coverage only at permissive thresholds (`640/conf=0.05`: 6/6 any-class, 2/6 same-class, 17 predictions; `640/conf=0.03`: 6/6 any-class, 3/6 same-class, 23 predictions). Conclusion: scan 2.5D geometry increases proposal coverage but does not solve denomination identity or false positives; next work should improve real/photoreal texture, finger realism, and verifier calibration rather than scaling this exact recipe blindly.

Follow-up old/common-focus probe `data/synthetic/cashsnap_scan_2p5d_oldcommon_focus_v1/` used a Numista 1990+ KHR bank filtered to `KHR_5000`, `KHR_10000`, and `KHR_20000`. Fresh run `runs/cashsnap/yolo26n_cashsnap_scan_2p5d_oldcommon_probe_e3_i416_b2/weights/best.pt` reached mixed validation `mAP50=0.160`, `mAP50-95=0.111`. On the same shop-overlap draft it did not beat the broader scan-2.5D probe: best 416px result was `2/6` same-class and `2/6` any-class at `conf=0.03`/`0.05`, while 640px introduced many `KHR_500`/`KHR_1000` false positives and stayed at `1/6` same-class. Conclusion: old/common design pressure alone is insufficient; the missing signal is phone-domain partial texture and calibrated fragment identity, not just more scan-sourced overlap geometry.

## Fragment Classifier Branch

Added a two-stage diagnostic path:

- `scripts/build_fragment_classifier_dataset.py`: builds ImageFolder-style synthetic/reference fragment crops.
- `scripts/build_fragment_classifier_from_yolo.py`: builds real-photo fragment crops from YOLO labels.
- `scripts/train_fragment_classifier.py`: trains MobileNetV3-small and exports ONNX.
- `scripts/classify_yolo_proposals.py`: runs YOLO proposals, then reclassifies each crop.
- `scripts/evaluate_two_stage_csv.py`: scores two-stage CSV outputs against private draft labels.

Pretrained MobileNetV3 on `data/fragment_classifier_cashsnap_realfrag_v1/` reached held-out test accuracy `0.922`, but old-overlap two-stage predictions remained USD-biased. Adding legacy NBC fragments in `data/fragment_classifier_cashsnap_realfrag_plus_legacy_v1/` produced `runs/fragment_classifier/mobilenet_v3_realfrag_plus_legacy_pretrained_balanced_e8/` with held-out test accuracy `0.910`. On `real_overlap_0003_commons_shop_5k_10k_20k`, class-agnostic YOLO proposals plus this classifier reduced duplicate predictions and correctly recovered the top-left old `KHR_20000`, but same-class draft recall is still only `1/6`; the right-side `KHR_20000` back is still confused with `KHR_1000`, and lower notes remain confused. This branch is promising for phone/browser because `best.onnx` exports, but it needs targeted old front/back fragment data before it can replace detector-only denomination labels.

The repaired P1 old/common partial-focus queue confirms the same data bottleneck. Evaluating `mobilenet_v3_old_common_khr_realbox_pretrained_balanced_e12` on `data/fragment_classifier_p1_oldcommon_focus_unreviewed_diag_v2` gives val accuracy `0.088` and test accuracy `0.067`; `KHR_5000` and `KHR_20000` mostly collapse into `KHR_10000`. This is not a trusted benchmark because the queue is unreviewed, but it is a useful failure-localization diagnostic.

Short diagnostic refresh `runs/fragment_classifier/mobilenet_v3_oldcommon_realbox_plus_p1_unreviewed_diag_e6/` used the existing old/common real-box train split plus unreviewed P1 train crops. It improves the P1 diagnostic split (`0.588` val, `0.667` test) and keeps old/common real-box test accuracy high (`0.991`), but hurts the real shop-overlap fusion (`3/6` same-class at detector override `0.17`, `2/6` at `0.20`). Conclusion: unreviewed P1 augmentation overfits the crop diagnostic and should not replace the current browser classifier.

## Rare KHR Coverage Probe

The original merged train split has only `17` `KHR_20000` boxes and `18` `KHR_50000` boxes. Based on the rare-KHR research PDFs, the minimum visual families to cover are:

- `KHR_20000`: 1995, 2008, and 2017-dated / 2018-issued.
- `KHR_50000`: 1995/1998-style, 2001, and 2013-dated / 2014-issued.

Added a bridge synthetic dataset at `data/synthetic/khr_rare_v1/` using NBC and Numista references:

- Source filter kept `13` whole-note references after aspect-ratio and specimen filtering.
- Train: `950` images, `3,876` `KHR_20000` boxes, `6,234` `KHR_50000` boxes.
- Val: `130` images, `522` `KHR_20000` boxes, `830` `KHR_50000` boxes.
- Test: `120` images, `451` `KHR_20000` boxes, `741` `KHR_50000` boxes.

Visual audit improved after excluding square-ish Numista security-detail closeups, but the dataset still has jagged catalog-mask artifacts. Treat `khr_rare_v1` as a bridge for short rare-class probes only. The better path remains PicWish/manual transparent cutouts scored by `scripts/score_transparent_cutouts.py`.

Attempted `configs/cashsnap_v1_plus_khr_rare.yaml` fine-tuning from the baseline at `416`, but stopped early because the full combined run was too slow for quick iteration. Use a smaller probe or wait for scored PicWish cutouts before a longer combined fine-tune.

## Roboflow Model Comparison

Checked the relevant Roboflow Universe projects:

- `Khmer-US-currency`: public page shows `Model 1`, but direct hosted endpoints tested at versions `1`, `2`, `3`, and `10` returned `403 Forbidden` with the local key.
- `Cambodia Currency Project`: public page shows `Model 1`; direct hosted endpoint `cambodia-currency-project/2` was accessible.
- `KHMER SCAN`: public page shows `Model 0`; direct hosted endpoints tested at versions `1` and `2` returned `403 Forbidden`.

The accessible Roboflow endpoint, `cambodia-currency-project/2`, returned zero predictions on:

- the real fan photo, even at `confidence=5`
- 155 held-out KHR test images at `confidence=25`
- several images originating from the Cambodia Currency Project subset, even with confidence values from `0` through `25`

Because the endpoint returns no predictions even on its own-source images, treat it as not operational or not comparable through the direct hosted API path. It should not be interpreted as a reliable quality benchmark for the underlying Roboflow training run.

On the same 155-image KHR test subset for the six classes shared with the accessible Roboflow endpoint (`KHR_500`, `KHR_1000`, `KHR_5000`, `KHR_10000`, `KHR_20000`, `KHR_50000`), the local YOLO26n baseline at `conf=0.25` produced:

- True positives: `148`
- False positives: `8`
- False negatives: `9`
- Precision: `0.949`
- Recall: `0.943`
- F1: `0.946`
