# CashSnap Ideas

Living doc for high-value ideas, experiments, and results. Keep this short: only add ideas worth revisiting.

## Strong Ideas To Try

- Collect the minimum real partial-note phone pack from `manifests/real_partial_capture_requirements.csv`, run `scripts/run_capture_review_pipeline.py`, review crops in `demo/review/`, then retrain/calibrate only from accepted rows. This is now higher leverage than more synthetic scaling or broad public-dataset search.
- Audit the downloaded Roboflow `cuurecy-detection-is` segmentation export: it has KHR/USD front/back masks and promising real phone partial/overlap examples. Exact duplicate/split-leakage and label-geometry audits passed; license/reproduction constraints, near-duplicates, note versions, and tiny-mask visual QA still need review before training.
- Build the mask-based synthetic compositor described in `docs/synthetic-compositor-plan.md`: verified real-camera cutouts, z-order compositing, and visible-pixel label regeneration for detect/OBB/segmentation.
- Build a curated transparent banknote asset library from the best public/Roboflow examples: tight crop, remove background, QA visually, tag denomination/front/back/version.
- Use `scripts/package_cashsnap_picwish_inputs.py` to expand cutout candidates beyond rare KHR; `data/picwish_upload_batches_cashsnap_khr_v1/` currently has 720 all-KHR upload images ready for background removal.
- Use strong external/background-removal tools to create cleaner transparent PNG assets. Group cutouts are useful for background replacement and style realism, but per-bill synthetic labels still need individual note masks/cutouts.
- Add automatic cutout QA after background removal: keep assets whose alpha mask is mostly rectangular/quadrilateral, flag curved or multi-lobed shapes as likely hands, merged objects, or bad cutouts.
- Use masks internally for synthetic composition, then export the right training label type: upright visible-pixel boxes for normal YOLO, rotated boxes for YOLO OBB, masks for segmentation.
- Add a detector-plus-fragment-classifier path for partial fan scenes: YOLO proposes visible slices, then a tiny classifier predicts denomination from each crop; see `docs/fragment-classifier-plan.md` and `scripts/build_fragment_classifier_dataset.py`.
- Train synthetic data as a curriculum: clean single notes, two-note crosses, small overlaps, partial off-frame notes, hand occlusion, then hard fan stacks.
- Use real table/concrete/counter backgrounds and realistic phone effects: shadows, blur, compression, exposure shifts, paper wear, and color grading.
- Create a small real fanned/overlapped validation set before tuning more. One stress image is useful but not enough to tell if a generator really works.
- Explore YOLO26n OBB after the synthetic asset bank exists; local weights load, and rotated labels become cheap once masks/corners are known.
- Try context-first background removal for tight crop failures: send a larger parent-image crop around the YOLO box to the remover, then crop/score the transparent bill afterward.
- Try local open-source removers after PicWish triage: RMBG-2.0 first, BEN2 second, with fill-holes plus slight dilation so masks do not erase printed details inside the note.

## Tried And Learned

- Baseline YOLO26n is strong on curated splits but fails the real fanned KHR photo at normal confidence.
- Reset first-pass training around current KHR designs only; e2/e4 and related rare-overlap checkpoints are now contaminated historical baselines, not foundations for continuation.
- `rare_pristine_asset_bank_v1` is not clean enough to trust by folder name: scope audit finds 13 legacy/reference assets, 26 scene crops needing version review, and only 2 automatically current common assets.
- Synthetic v2 helped slightly: fan image improved only at permissive `640/conf=0.05`.
- Dense fan synthetic v3 was a useful historical fan probe, but it is not a clean current-KHR foundation.
- Partial-slice-heavy v4 improved normal test metrics but regressed on the real fan photo, so pure slice training overcorrects.
- Real YOLO crop compositing needs better masks first; naive real crops carried rectangular background artifacts and should not be trained as-is.
- `khr_rare_v1` adds minimum synthetic coverage for `KHR_20000`/`KHR_50000`, but is only a bridge; catalog masks still create artifacts, so PicWish-quality transparent assets should replace it for serious training.
- Mixed Asian-currency datasets are useful for general rotation/background augmentation, but should not be counted as proof of Cambodian old/new KHR version coverage.
- Gold cutouts alone were still too noisy for first-stage synthesis; filtering by alpha rectangularity/component/aspect produced a cleaner `rare_pristine_asset_bank_v1` for curriculum probes.
- Pristine clean synthetic smoke (`yolo26n_pristine_clean_synth_smoke_e2_i416_b4_mb20`) wrote correctly under repo `runs/` and reached synthetic-val mAP50 0.764 after 20 train batches; `KHR_20000` recall (0.487) still lags `KHR_50000` (0.714).
- Rare post-train (`yolo26n_pristine_clean_rare_e4_i416_b8`) is the first completed 20k/50k fine-tune: normal-val mAP50-95 0.926, `KHR_20000` recall 0.990, `KHR_50000` recall 0.566.
- Rare overlap v1 plus 2-epoch fine-tune (`yolo26n_pristine_overlap_e2_i416_b8`) improved synthetic overlap-val mAP50 from 0.418 to 0.568 and `KHR_50000` overlap recall from 0.262 to 0.439, while normal-val mAP50-95 dipped from 0.926 to 0.920.
- On rare-overlap synthetic validation, `yolo26n_pristine_overlap_e2_i416_b8` beats older fan checkpoints v3/v4 (mAP50 0.568 vs 0.422/0.446), so pristine overlap composition is worth extending beyond rare notes.
- Combining broad fan v3 with pristine rare overlap (`yolo26n_messy_v3_pristine_overlap_e2_i416_b8`) improved both scoreboards: normal-val mAP50-95 0.930 and rare-overlap mAP50 0.584, with `KHR_50000` overlap recall 0.475.
- Continuing that combined checkpoint for 2 more epochs (`yolo26n_messy_v3_pristine_overlap_e4_i416_b8`) improved rare-overlap mAP50 to 0.602 (`KHR_20000`/`KHR_50000` recall 0.535/0.515) and held-out test mAP50-95 to 0.942, but normal-val mAP50-95 fell to 0.925.
- Capped gentle-LR continuation from the combined e2 checkpoint (`yolo26n_messy_v3_pristine_overlap_gentle_lr_probe3_mb100_i416_b4`) avoided the collapse seen in the 300-batch probe, but did not beat e2/e4: normal-val mAP50-95 0.927 and rare-overlap mAP50 0.575.
- On the real fan stress image, combined e2/e4 both improved normal-confidence detection over dense v3 (`2` vs `1` detections at `416/conf=0.25`); e4 is the better rare-slice stress candidate at permissive confidence (`7` detections with `KHR_50000` up to 0.433), but still not reliable counting.
- Raising real fan inference to `imgsz=960` did not help v3/e2/e4; detections dropped versus `640/conf=0.05`, so prioritize data/labels over resolution-only tuning.
- Real fan failure analysis points to label/geometry mismatch, not NMS or tiling: e4 box counts are unchanged across NMS IoU `0.30-0.90`, crops do not recover per-slice detections, and current synthetic labels have large median areas (`23-27%` of image).
- Current NBC cutout bank generation is now reproducible: `scripts/build_current_khr_cutout_bank.py --clean` writes transparent PNGs, binary masks, a manifest, and a contact sheet from `target_modern_common`; use it before fresh YOLO26n detect/OBB probes.
- Tiny capped smoke runs proved Ultralytics accepts both current-cutout detect and OBB datasets from fresh `yolo26n.pt`/`yolo26n-obb.pt`; use larger runs only after the real fan benchmark is fixed.
- Fresh current-clean probe (`yolo26n_current_clean_v1_e2_i416_b8_mb80`) trained from `yolo26n.pt` on 600 clean current-cutout scenes but still produced 0 detections on the hard fan image at `416/640` and `conf=0.25/0.05`; next data must add fan/slice geometry, not just clean-note coverage.
- Fresh current fan-slice e5 still failed the hard fan image at `conf>=0.05`; adding `thin_radial_slice` dropped synthetic median box area from about 18-20% to 5.2%, but thin-only training was too weak and lost overlap recall.
- Hybrid clean + thin-radial e5 is the best reset-path signal so far: hard fan reaches 1 detection at `640/conf=0.05` but with a huge 26% box, while the overlap image reaches 15 detections at `960/conf=0.05`; see `data/real_fan_benchmark/diagnostics/current_clean_plus_thin_radial_e5_fan_overlap_diag.csv`.
- OBB is available and materially different: thin-only OBB gives plausible small rotated boxes on the overlap image (`19` detections at `960/conf=0.25`, top confidence 0.85) but class labels are biased/noisy; clean+thin OBB is synthetically healthier but weaker on the real candidates, so OBB needs real labeled fan/overlap validation before more architecture optimism.
- The current VOA hard fan image is a stress probe, not a fair full scoreboard: many visible slices are backs or ambiguous fragments, so the next benchmark needs rights-clear phone photos where humans can actually assign denomination labels to visible regions.
- Draft-label tooling now exists for real candidates without benchmark poisoning: use `scripts/render_yolo_label_preview.py` to review candidate labels and `scripts/evaluate_real_draft_labels.py` to score count/recall before promoting anything to official `labels/val`.
- Fresh class-balanced real+current-thin training improves validation (`yolo26n_cashsnap_current_thin_balanced_v1_e20_i416_b8`, mAP50-95 about 0.694) but still misclassifies old KHR designs on the shop overlap candidate.
- Explicit circulated-design support from `target_modern_common + legacy_or_low_priority` NBC cutouts helps old-design overlap recall. The cleaner support variant (`yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8`) is the preferred fresh alpha: stronger validation and ONNX/NCNN export, but still not reliable denomination totals.
- All-KHR PicWish cutout path works technically: 720/720 background removals succeeded, but alpha `gold` still admits hand-contaminated assets. A low-skin strict bank plus NBC fill improved real-overlap any-class coverage at `960/conf=0.05` (6/6 regions) but not denomination accuracy, so the next cutout work needs human QA or stronger hand filtering.
- Old-common focus continuation trained safely through the headroom harness, but it did not become an alpha: it improves any-class overlap coverage while increasing `KHR_1000/KHR_10000/KHR_2000` confusion. Do better asset QA/class balance before scaling that recipe.
- `scripts/bench_train_with_headroom.py` now handles memory pressure by relaunching smaller, then switching to pause/resume at the floor; a `batch=1/workers=0` six-epoch YOLO run completed with RAM near 90%.
- Shape-filtered PicWish banks are cleaner for visual triage but still not training-trustworthy: `shape_skin30_current` continuation reached only mAP50-95 0.443 and added USD false positives on a KHR-only overlap image.
- Fragment classifier branch is viable but data-limited: real-fragment MobileNetV3 reaches 0.922 test accuracy and exports ONNX; legacy supplementation recovers one old 20k overlap region, but right-side old backs remain confused, so target old front/back fragments next.
- Backfocused synthetic legacy-reverse fragments (`mobilenet_v3_realfrag_legacy_backfocus_pretrained_balanced_e8`) exported ONNX and held `0.908` test accuracy, but failed the shop overlap probe: classifier replacement scored 0/6 same-class while detector labels on the same proposals scored up to 3/6. Do not keep scaling reference-derived back fragments without human-verified real old/back crops.
- `data/review/cashsnap_old_common_khr_crop_review_v1` is the first real-crop review pack for the confusing old/common classes, with curation columns and an ImageFolder converter. It exposes the practical gap: `KHR_20000` has only 55 real labeled crops, while `KHR_1000` has many back-like blue/gray examples that explain the 20k-back -> 1k confusion.
- Focused old/common KHR real-box classification is the best two-stage signal so far: detector proposals + `mobilenet_v3_old_common_khr_realbox_pretrained_balanced_e12` + detector-threshold fusion around `0.17` + detector-conf NMS reached 6 predictions for 6 draft notes, 6/6 any-class coverage, and 5/6 same-class on the shop overlap candidate.
- Modest legacy reference-fragment augmentation of that focused classifier did not improve the fused shop-overlap score beyond 5/6, so prioritize real reviewed partial/back crops and calibration over more reference-fragment scaling.
- Existing Khmer OCR is not a shortcut on the real shop-overlap draft: `mer` returns scattered text and wrong/partial digits from note crops, so keep OCR as an optional cue after detection rather than replacing the visual classifier path.
- Roboflow `cuurecy-detection-is` downloaded successfully to `data/raw_datasets/roboflow_cuurecy_detection_is/`: 2,329 images and 5,689 YOLOv8 segmentation labels across KHR/USD front/back classes. Exact duplicate/split-leakage found 0 groups, geometry audit found no malformed mask rows, filtered tiny/edge sheets show useful real partial/off-frame cases, and the manifest has 5,067 core CashSnap objects plus front/back/partial metadata for curation.

## Data Gaps

- `KHR_20000` and `KHR_50000` remain rare/weak for current-design coverage; do not fill this gap with old/collector variants unless testing legacy support.
- Current target asset coverage should come from `target_modern_common` cutouts first, with optional rare KHR kept separate from the common fan-counting benchmark.
