# CashSnap Script Map

This directory is intentionally script-heavy because CashSnap has a lot of historical probes. `model.md` is the source of truth for which scripts are active.

## Current First-Line Commands

- `run_with_headroom.py`: wrap heavy CPU/RAM/GPU work so the machine stays below resource limits.
- `bench_train_with_headroom.py`: safer YOLO training wrapper for longer detector probes.
- `run_browser_smoke_cases.py`: browser/mobile deployment sanity suite.
- `check_browser_stack_artifacts.py`: ONNX artifact size gate for the two-stage browser stack.
- `summarize_review_manifests.py`: tells whether review packs are actually curated.
- `apply_review_export.py`: merges `demo/review/` CSV exports back into review manifests.
- `build_fragment_classifier_from_review_pack.py`: builds trusted ImageFolder datasets from reviewed crops.
- `build_imagefolder_mix.py`: creates controlled train-only augmentation mixes without contaminating validation/test splits.

## Review And Capture

- `init_capture_inbox.py`, `register_capture_photos.py`, `check_capture_requirements.py`: local phone-photo capture bookkeeping.
- `run_capture_review_pipeline.py`: detector -> classifier -> fusion -> review-pack pipeline for non-benchmark captures.
- `smoke_review_ui_cdp.cjs`: headless Edge smoke for the static crop-review UI and local draft restore.
- `build_proposal_review_pack.py`: create crop review packs from proposal CSVs.
- `build_prediction_failure_review_pack.py`: turn classifier prediction failures into focused review queues.
- `build_partial_focus_review_queue.py`: rebuild the current P1 old/common focus queue.
- `build_benchmark_review_index.py`, `render_yolo_label_preview.py`, `evaluate_real_draft_labels.py`: benchmark labeling helpers. Never train on `data/real_fan_benchmark/`.

## Fragment Classifier

- `build_fragment_classifier_dataset.py`: synthetic/reference fragment crop builder.
- `build_fragment_classifier_from_yolo.py`: YOLO-box real fragment crop builder.
- `build_yolo_crop_review_pack.py`: auditable review packs from YOLO detection or segmentation labels.
- `train_fragment_classifier.py`: MobileNetV3-small fragment classifier training and ONNX export.
- `evaluate_fragment_classifier.py`: ImageFolder evaluation and per-crop prediction CSVs.
- `classify_yolo_proposals.py`, `fuse_two_stage_csv.py`, `sweep_two_stage_fusion.py`, `evaluate_two_stage_csv.py`: detector/classifier fusion diagnostics; keep `fuse_two_stage_csv.py --prefer-supported-detector-duplicates` diagnostic-only until guard cases pass.
- `inspect_two_stage_matches.py`: row-level detector/classifier confidence inspection against draft labels.
- `build_currency_gate_dataset.py`: KHR/USD/background gate experiments; historical gate did not transfer to hard shop-overlap.
- `probe_template_feature_verifier.py`: diagnostic-only SIFT/ORB/AKAZE template matcher for partial crops; current P1 results are poor, so do not treat it as a production verifier path.

## Synthetic And Assets

- `validate_3d_pipeline_config.py`: strict renderer-agnostic validator for the active 3D proof configs.
- `render_3d_pipeline_probe.py`: P0 renderer scaffold that emits visual renders, ID masks, labels, metadata, and QA sheets from the active 3D configs.
- `render_webgl_variant_batch.py`, `check_webgl_smoke_output.py`, `check_webgl_label_views.py`: WebGL batch render/package/QA path for detect, OBB, fragment, recipe, and QA-summary outputs.
- `check_synthetic_recipe_catalog.py`: validates the synthetic target matrix and recipe catalog coverage.
- `curate_reference_images.py`: bucket KHR reference assets by circulation priority.
- `build_current_khr_cutout_bank.py`, `build_numista_cutout_bank.py`: reproducible scan/reference cutout banks.
- `audit_cutout_bank.py`, `render_cutout_contact_sheet.py`, `score_transparent_cutouts.py`, `select_best_cutouts.py`: cutout QA.
- `generate_synthetic_fan_dataset.py`: current 2.5D synthetic fan/overlap generator.
- `summarize_synthetic_metadata.py`, `check_yolo_dataset.py`: synthetic and YOLO dataset sanity checks.
- `extract_yolo_background_patches.py`: mine real background patches; require contact-sheet QA before training with them.

## Public Source Prep And Audit

- `download_public_sources.py`, `download_roboflow_datasets.py`, `download_rare_khr_numista.py`: source acquisition helpers.
- `prepare_cashsnap_dataset.py`, `prepare_hf_usd_yolo.py`, `prepare_real_fan_benchmark.py`: dataset preparation.
- `build_cuurecy_detection_manifest.py`, `audit_yolo_source_classes.py`, `audit_yolo_segmentation_geometry.py`, `visual_audit_yolo_dataset.py`, `check_duplicates.py`: Roboflow/public-source audits.
- `repair_yolo_segment_labels.py`, `summarize_yolo_labels.py`: YOLO label maintenance.

## Model Export And Debug

- `train_yolo.py`, `export_yolo.py`: detector training/export entry points.
- `debug_onnx_detector_preprocess.py`: compare ONNX preprocessing paths.
- `diagnose_fan_image.py`: local stress-image detector diagnostics.
- `probe_khmer_ocr_cues.py`: optional OCR cue probe; current evidence says OCR is auxiliary, not the main CashSnap path.

## Hygiene

- `repo_hygiene_cleanup.py`: local generated-data cleanup helper. Review and dry-run before deleting or moving anything.
- `local_runtime.py`: repo-local runtime/cache helpers for scripts.
- `log_research_result.py`: deprecated TSV scratch helper; prefer the curated `model.md` result ledger so project memory stays in one place.
