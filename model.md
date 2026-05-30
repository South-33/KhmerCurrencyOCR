# CashSnap Model Brain

This is the one living document for model work. If the plan changes, update this file first. Keep it clean, current, and useful for the next agent dropped into the repo.

## Mission

Build a small phone/browser-deployable model that counts mixed USD and Khmer Riel from one casual retail photo. The hard requirement is partial-note recognition: half, quarter, thin edge, overlapped, fanned, and hand-occluded notes must still be identified when humans can see enough denomination evidence.

Counterfeit detection is out of scope.

## Current North Star

Active bet: make synthetic data generation good enough to be the scaling unit.

Current phase: 3D synthetic-pipeline reset before the next training push.

Do not start another training run until the P0 renderer smoke proof can produce:

- plausible phone-like visual renders
- exact ID-pass visible masks
- visible-only detect boxes and OBB boxes
- metadata rows
- contact sheets and mask overlays
- deterministic reruns from a seed

2.5D remains useful as a matched baseline and fallback, but the next main effort is the 3D renderer proof.

## Work Loop

1. Read this file.
2. Check git status.
3. Work only on the current bottleneck.
4. Run bounded checks.
5. Update this file when a result changes the plan.
6. Commit coherent durable changes, not every tiny scratch edit.
7. Keep `results.tsv` local and untracked for experiment rows.

Heavy CPU/RAM/GPU work must go through a headroom wrapper. Never train directly when a safe wrapper exists.

## Local Machine Profile

- Machine: Lenovo 82Y9 laptop.
- CPU: AMD Ryzen 5 7640HS, 6 cores / 12 logical processors, 4.3 GHz reported max clock.
- RAM: about 16 GB physical memory.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM, driver 596.21.
- Re-scan command: `rl python scripts\profile_system.py --out .cache_runtime\system_profile.json`.
- Current operating rule: laptop usability matters. Heavy jobs should prefer 90% max CPU/RAM/GPU/VRAM caps, resume around 82%, and never set caps above 95%.
- If this hardware is unexpectedly slow, suspect RAM pressure first, then laptop GPU power/thermal limits, then data-loader worker count.
- `scripts/run_with_headroom.py` now refuses caps above 95%, watches free RAM as well as RAM percent/VRAM/GPU/CPU, lowers child priority, and waits for initial headroom before launching a heavy child.
- `scripts/bench_train_with_headroom.py` dry-run currently selects `batch=2`, `workers=0` on this laptop and passes `--min-free-ram-gb 4.0` to the live wrapper.
- Balance speed and headroom. Prefer GPU for training/inference when it is the faster engine and has room, but do not force GPU for CPU-native prep/rendering if the headroom wrapper keeps the laptop responsive.

## Active Commands

Initialize or append the local experiment ledger:

```powershell
rl python scripts\log_research_result.py --init
```

Validate active 3D configs:

```powershell
rl python scripts\validate_3d_pipeline_config.py configs\3d_pipeline\proof_p0_renderer_smoke.json configs\3d_pipeline\proof_p1_transfer.json
```

Render P0 proof scenes:

```powershell
rl python scripts\render_3d_pipeline_probe.py --config configs\3d_pipeline\proof_p0_renderer_smoke.json --clean
```

Render the minimal WebGL proof:

```powershell
rl python scripts\run_with_headroom.py --max-percent 90 --resume-percent 82 --max-ram-percent 90 --max-gpu-mem-percent 90 --min-free-ram-gb 3 --preflight-timeout 120 -- node renderers\webgl\src\render-smoke.mjs
```

Check WebGL smoke output:

```powershell
rl python scripts\check_webgl_smoke_output.py --out-dir data\synthetic\cashsnap_webgl_smoke
```

Protect the real benchmark boundary:

```powershell
rl python scripts\check_real_fan_benchmark.py
```

Dry-run safe training only after the renderer proof exists:

```powershell
rl python scripts\bench_train_with_headroom.py --data configs\cashsnap_v1.yaml --name dry_run --dry-run --quiet
```

## Current Active Assets

Current top-level `data/` table after cleanup:

- `asset_candidates/`
- `audit/`
- `backgrounds/`
- `cashsnap_v1/`
- `curated/`
- `inbox/`
- `numista_raw/`
- `picwish_upload_batches_cashsnap_khr_v1/`
- `raw_datasets/`
- `real_fan_benchmark/`
- `reference/`
- `review/`

Generated scratch folders removed from the table: old `data/synthetic/`, generated `data/fragment_classifier*/`, `data/deprecated/`, `data/dedup/`, `data/processed/`, `data/sampled/`, `data/diagnostics/`, and `tmp/` contents. Regenerate them only from a current config or reviewed need.

### Tier 0: Canonical Backbone

- `data/numista_raw/`: trusted raw scan metadata/source cache. Numista `in_circulation` is the best scan backbone in this repo.
- `data/asset_candidates/numista_current_cutout_bank_v1/`: current best scan cutout bank. It has a manifest, masks, classes, and front/back metadata.
- `configs/3d_pipeline/proof_p0_renderer_smoke.json`: active renderer smoke config.
- `configs/3d_pipeline/proof_p1_transfer.json`: active transfer-proof config.

### Tier 1: Required Evaluation And Guardrails

- `data/cashsnap_v1/`: verified local YOLO dataset, 13 classes, 9,048 boxes. Keep for clean/base validation.
- `data/real_fan_benchmark/`: scoreable/stress evaluation only. Never train on it.
- `manifests/real_fan_benchmark_label_quality.csv`: decides which real labels are scoreable.
- `runs/cashsnap/yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8/weights/best.pt`: current overlap-counting alpha checkpoint for diagnostics/export, not a solved final model.

### Tier 2: Useful But Conditional

- `data/raw_datasets/roboflow_cuurecy_detection_is/`: useful real phone partial/overlap material, but public/reproduction and split caveats mean it is review/domain-stress data until curated.
- `data/review/`: keep reviewed or actively reviewed packs; they are more valuable than unreviewed generated classifier datasets.
- `data/backgrounds/`: use only after contact-sheet QA proves no note fragments leaked into backgrounds.
- `data/asset_candidates/*picwish*`: cutout candidates only; hand/skin contamination has repeatedly hurt transfer.

### Tier 3: Archive Or Regenerate

- `data/synthetic/`: generated experiments. Keep only named artifacts referenced by a live result; otherwise regenerate from configs/scripts.
- `data/fragment_classifier*/`: generated classifier datasets/results. Keep only if tied to a live reviewed-data path.
- `data/audit/`, `data/diagnostics/`, `data/sampled/`, `data/processed/`, `data/dedup/`, `tmp/`: scratch/diagnostic areas. Clean aggressively when they are not tied to an active result.
- `data/deprecated/`: delete unless a specific file is promoted back into Tier 1 or Tier 2.

## Data Ranking

1. Numista in-circulation scans: best canonical design and side metadata.
2. Rights-clear Cambodian phone captures: best real transfer evidence once reviewed.
3. Curated real benchmark labels: best evaluation ruler, never training material.
4. Reviewed Roboflow/public partial crops: useful diagnostic supplements, not final proof.
5. NBC/reference/current cutouts: useful for coverage but specimen/old/current scope must be audited.
6. PicWish/BEN2 cutouts: useful only after visual QA and hand/skin filtering.
7. Broad synthetic/probe outputs: disposable unless the result is promoted in this file.

## Config Ranking

Keep visible in `configs/`:

- `cashsnap_v1.yaml`: clean/base YOLO dataset.
- `cashsnap_two_stage_oldcommon_browser_stack.json`: current browser diagnostic stack.
- `3d_pipeline/`: active renderer proof configs.

Everything else belongs under `configs/archive/` unless it is promoted back to active. Old configs are not deleted because they explain prior results, but they should not clutter the first view.

## Script Ranking

Use these first:

- Harness and safety: `log_research_result.py`, `run_with_headroom.py`, `bench_train_with_headroom.py`, `local_runtime.py`.
- 3D/synthetic proof: `render_3d_pipeline_probe.py`, `validate_3d_pipeline_config.py`, `build_numista_cutout_bank.py`, `generate_synthetic_fan_dataset.py`, `summarize_synthetic_metadata.py`, `check_yolo_dataset.py`.
- WebGL proof: `renderers/webgl/src/render-smoke.mjs` uses Three.js + `puppeteer-core` against local Microsoft Edge; `check_webgl_smoke_output.py` validates nonblank RGB, exact ID colors, visible labels, primitive finger occlusion, and layer-order audit output.
- Evaluation and deploy guards: `check_real_fan_benchmark.py`, `run_browser_smoke_cases.py`, `check_browser_stack_artifacts.py`, `val_yolo.py`, `export_yolo.py`.
- Real capture/review: `check_capture_requirements.py`, `run_capture_review_pipeline.py`, `apply_review_export.py`, `summarize_review_manifests.py`, `render_yolo_label_preview.py`, `evaluate_real_draft_labels.py`.
- Fragment diagnostics: `build_fragment_classifier_from_review_pack.py`, `train_fragment_classifier.py`, `evaluate_fragment_classifier.py`, `classify_yolo_proposals.py`, `fuse_two_stage_csv.py`, `sweep_two_stage_fusion.py`, `inspect_two_stage_matches.py`.

The scripts folder is still large because it records past probes. Before adding a new script, either reuse one of these or make the new entry clearly active in this section.

## P0 3D Renderer Smoke

Goal: prove renderer and labels, not model quality.

Config: `configs/3d_pipeline/proof_p0_renderer_smoke.json`

Scope:

- 20 scenes
- `KHR_5000` and `KHR_10000`
- front and back
- single, simple overlap, and shop stack
- basic bends/curls
- visual pass and ID pass
- detect labels, OBB labels, masks, overlays

Current scaffold:

- `scripts/render_3d_pipeline_probe.py` consumes the P0/P1 JSON config style.
- It renders perspective-warped Numista notes with simple contact shadows, visual images, flat ID masks, visible-only YOLO detect labels, OBB sidecar labels, OBB metadata, scene metadata, `data.yaml`, label stats, contact sheets, and mask overlays.
- P0 smoke output path: `data/synthetic/cashsnap_3d_p0_renderer_smoke/`.
- Last P0 smoke: 20 scenes in about 20-30 seconds; deterministic contact-sheet hash matched across reruns; `check_yolo_dataset.py --data data\synthetic\cashsnap_3d_p0_renderer_smoke\data.yaml` passed with 16 train images / 47 boxes and 4 val images / 13 boxes; `qa/label_stats.json` reports 66 instances and 60 exported labels.
- This scaffold is not the final 3D renderer. It proves the label/QA contract before WebGL/PBR/material realism.

Success:

- nonblank renders
- textures aligned
- ID colors map one-to-one to instances
- labels are visible-only, not amodal hidden-note extents
- same seed reruns deterministically
- contact shadows or depth cues are visible in overlap scenes

## P1 3D Transfer Proof

Goal: test whether 3D beats matched 2.5D on real scoreable partials.

Config: `configs/3d_pipeline/proof_p1_transfer.json`

Scope:

- 100-300 scenes
- targeted `KHR_5000` portrait-plus-number overlap
- confusing `KHR_10000` front/back views
- `KHR_20000` thin/edge cases
- several phone/browser camera profiles
- realistic postprocess

Compare against:

- matched 2.5D dataset
- current two-stage baseline
- scoreable real shop-overlap labels
- clean `cashsnap_v1` validation
- browser/export smoke

Only scale to thousands of scenes if P1 improves real partial metrics without materially regressing clean validation.

Current P1 render smoke:

- Command ran through `scripts/run_with_headroom.py` so it could preflight/pause for laptop safety.
- Output path: `data/synthetic/cashsnap_3d_p1_transfer_proof/`.
- 240 scenes rendered; `check_yolo_dataset.py --data data\synthetic\cashsnap_3d_p1_transfer_proof\data.yaml` passed with 192 train images / 682 boxes and 48 val images / 164 boxes.
- `qa/label_stats.json` reports 1,006 instances and 846 exported labels across `KHR_5000`, `KHR_10000`, and `KHR_20000`.
- The current Python/OpenCV renderer is useful for label-contract proof, but it is CPU-native and visually simple. Do not train from this scaffold as if it were final data.

## WebGL Renderer Smoke

Goal: prove Windows/headless Edge + Three.js can render realistic-looking RGB, exact ID colors, and visible boxes before building the full synthetic factory.

Current proof:

- Package: `renderers/webgl/` with `three` and `puppeteer-core`; it uses local Microsoft Edge rather than downloading Chromium.
- Command: `rl python scripts\run_with_headroom.py --max-percent 90 --resume-percent 82 --max-ram-percent 90 --max-gpu-mem-percent 90 --min-free-ram-gb 3 --preflight-timeout 120 -- node renderers\webgl\src\render-smoke.mjs`.
- Output path: `data/synthetic/cashsnap_webgl_smoke/`.
- Last smoke rendered `visual.png`, `id.png`, `visible_boxes.json`, `labels_visible.txt`, `layer_audit.json`, `metadata.json`, and the temporary `smoke.html`.
- ID pass is exact after disabling WebGL antialiasing: black background plus one RGB ID color per visible note, no blended edge colors.
- RGB note rendering includes an opaque paper backing behind each textured note so visual scenes read as stacked paper sheets instead of transparent scan planes.
- Primitive finger occluders are top-layer capsule meshes. They render as skin-colored geometry in RGB and black in the ID pass, so covered bill pixels are removed from visible labels.
- Note visibility uses explicit layer order (`renderOrder` with depth test/write disabled for banknote planes) so tilted sheets or fingers do not phase through upper layers and poison visible masks. The current smoke audited 88,156 overlapping pixels and 15,732 occluder pixels with zero layer-order violations. This is an intentional topological stacking shortcut until a real contact/physics solver exists.
- Texture loading should stay file-backed (`file://`) instead of base64-inlining full scan PNGs into the HTML. Base64 inlining caused headroom pauses; file-backed textures completed the smoke quickly under the same wrapper caps.
- Even after trimming shadow-map size, the local laptop can still pause WebGL smoke runs when baseline RAM is high. Batch rendering should reuse one browser/page and design texture downscaling/cache behavior before scaling scene counts.

## Known Results

- Clean isolated-note detection is strong, but dense overlap/fan counting is not solved.
- Current alpha detector exports ONNX/NCNN and is useful diagnostically, but denomination totals remain unreliable on old-design overlap photos.
- Real fan failure is data/geometry, not NMS, tiling, or resolution. NMS sweeps and higher inference sizes did not solve it.
- OBB thin-slice probes can produce plausible small rotated boxes, but class labels are noisy; OBB is not chosen until a real labeled fan/overlap benchmark exists.
- The hard VOA fan image is a stress probe, not a fair scoreboard, because many visible slices are backs or ambiguous fragments.
- Current best PyTorch-side shop-overlap diagnostic is old/common KHR detector/classifier fusion around detector threshold 0.17, reaching 5/6 same-class on draft labels.
- Browser stack is deployable-size but not quality-solved. It protects USD/KHR guard cases but still miscounts the shop-overlap draft.
- OCR is not a shortcut. Khmer OCR cues were noisy and wrong on shop-overlap crops.
- Template matching is not a shortcut. SIFT/ORB/AKAZE performed poorly on the compact P1 failure queue.
- Broad dataset searches in May 2026 did not find a better public KHR partial/fan dataset. Stop broad hunting unless a specific new lead appears.
- Roboflow `cuurecy-detection-is` is useful but not clean proof: it has real partial/overlap examples, front/back masks, and repeated layouts/split caveats.
- PicWish can generate lots of cutouts, but skin/hand contamination repeatedly caused regressions.
- Numista clean scan fragments alone did not fix `KHR_5000 -> KHR_10000` partial confusion.
- Light-dose Numista 2.5D calibration was the best 2.5D direction so far: cleaner than broad synthetic, but still overcounts and confuses backs.
- The remaining high-confidence shop-overlap miss is a real partial-evidence problem, not just confidence calibration.

## Untested Or Parked Ideas

- Full 3D WebGL renderer with ID pass, contact shadows, curls, hands/fingers, and phone postprocess.
- Minimal browser-less Python/OpenCV renderer as a P0 stepping stone if it can satisfy exact ID masks faster.
- Primitive finger occluders before a full hand mesh.
- Detector plus fragment classifier remains viable if fed reviewed real partial crops.
- Segmentation/OBB labels should be preserved from generated masks even when training detect-only.
- More rights-clear phone captures of `KHR_5000` face+number overlaps and `KHR_20000` thin/edge backs are high leverage.
- A tiny `banknote_unknown`/ignore policy for fragments without enough denomination evidence is better than forcing labels.

## Data Cleanup Policy

Keep:

- canonical raw/cutout assets
- clean validation data
- real benchmark evaluation data
- reviewed real captures/review packs
- tiny configs/manifests/QA summaries that reproduce a result

Delete or regenerate:

- one-off synthetic datasets with no promoted result
- stale fragment-classifier generated folders
- stale diagnostics under `tmp/`, `data/diagnostics/`, `data/sampled/`, `data/processed/`, `data/dedup/`, `data/deprecated/`
- old browser screenshots/logs unless attached to an active bug

Before deleting a large data directory, verify it is under `D:\Project\KhmerCurrencyOCR` and not Tier 0/Tier 1.

## Documentation Policy

This file is the living plan. Root `AGENTS.md` should stay short and point here. `docs/` is archive/reference only; do not add another active plan there. If something matters for model work, put it here.
