# CashSnap Model Brain

This is the one living document for model work. If the plan changes, update this file first. Keep it clean, current, and useful for the next agent dropped into the repo.

## Mission

Build a small phone/browser-deployable model that counts mixed USD and Khmer Riel from one casual retail photo. The hard requirement is partial-note recognition: half, quarter, thin edge, overlapped, fanned, and hand-occluded notes must still be identified when humans can see enough denomination evidence.

Counterfeit detection is out of scope.

## Current North Star

Active bet: make synthetic data generation good enough to be the scaling unit.

Current phase: 3D synthetic-pipeline reset before the next training push.

Current model direction after split-label QA: keep the lightweight detector path for clean/mostly-visible notes, add a fragment/evidence detector or classifier path for partial OCR/denomination cues, then fuse fragments into physical-note counts. Do not treat OBB as the primary hard-fan solution; use OBB only for compact visible instances where the rotated box is honest.

Do not start another training run until the P0 renderer smoke proof can produce:

- plausible phone-like visual renders
- exact ID-pass visible masks
- visible-only detect boxes and OBB boxes
- metadata rows
- contact sheets and mask overlays
- deterministic reruns from a seed

2.5D remains useful as a matched baseline and fallback, but the next main effort is the 3D renderer proof.

Label policy: exact visible masks/ID colors are authoritative. Detect AABBs are compatibility labels for detect-only probes; OBB labels are useful only when the visible mask is compact enough that a rotated rectangle does not cover large hidden regions. Fragment labels mean visible connected evidence components, not physical bill counts. If one physical bill is visible as disconnected islands, keep it one counted instance in mask metadata, skip/flag unsafe OBB, and use fragment labels plus downstream fusion rather than projecting a hidden-paper box. Barely visible fragments should get a denomination label only when a human can identify the denomination from visible evidence; otherwise use ignore/unknown instead of forcing a guess.

## Work Loop

1. Read this file.
2. Check git status.
3. Work only on the current bottleneck.
4. Run bounded checks.
5. Update this file when a result changes the plan.
6. Commit coherent durable changes, not every tiny scratch edit.
7. Keep `results.tsv` local and untracked for experiment rows.

Heavy CPU/RAM/GPU work must go through a headroom wrapper. Never train directly when a safe wrapper exists.

Harnesses are configurable working tools, not permanent doctrine. If a harness, config, or workflow creates friction, hides a bad assumption, or slows the path to a better model, inspect it and improve it instead of working around it forever.

## Local Machine Profile

- Machine: Lenovo 82Y9 laptop.
- CPU: AMD Ryzen 5 7640HS, 6 cores / 12 logical processors, 4.3 GHz reported max clock.
- RAM: about 16 GB physical memory.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM, driver 596.21.
- Re-scan command: `rl python scripts\profile_system.py --out .cache_runtime\system_profile.json`.
- Current operating rule: laptop usability matters. Heavy jobs should prefer 90% max CPU/RAM/GPU/VRAM caps, resume around 82%, and never set caps above 95%.
- If this hardware is unexpectedly slow, suspect RAM pressure first, then laptop GPU power/thermal limits, then data-loader worker count.
- `scripts/run_with_headroom.py` now refuses caps above 95%, lowers child priority, and waits for initial headroom before launching a heavy child. The free-RAM floor is a preflight gate and runtime warning; after launch, hard pauses/exits follow the explicit RAM/VRAM max caps so pause-sensitive browser jobs do not trip wall-clock timeouts.
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
- WebGL proof: `renderers/webgl/src/render-smoke.mjs` uses Three.js + `puppeteer-core` against local Microsoft Edge; `check_webgl_smoke_output.py` validates nonblank RGB, exact ID colors, visible labels, primitive finger occlusion, and layer-order audit output; `check_webgl_label_views.py` validates packaged detect/OBB/fragment views and prevents diagnostic-only OBB labels from being mistaken for trainable data.
- Evaluation and deploy guards: `check_real_fan_benchmark.py`, `run_browser_smoke_cases.py`, `check_browser_stack_artifacts.py`, `val_yolo.py`, `export_yolo.py`.
- Real capture/review: `check_capture_requirements.py`, `run_capture_review_pipeline.py`, `summarize_camera_metadata.py`, `apply_review_export.py`, `summarize_review_manifests.py`, `render_yolo_label_preview.py`, `evaluate_real_draft_labels.py`.
- Fragment diagnostics: `build_fragment_classifier_from_review_pack.py`, `train_fragment_classifier.py`, `evaluate_fragment_classifier.py`, `classify_yolo_proposals.py`, `fuse_two_stage_csv.py`, `sweep_two_stage_fusion.py`, `inspect_two_stage_matches.py`, `probe_fragment_count_fusion.py`.

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
- RGB table rendering now uses a deterministic procedural grain texture. Existing `data/backgrounds/smoke_cashsnap_no_note_patches_v4/` contact sheet includes hands/note fragments/grass, so do not use real background patches until a clean QA subset exists.
- `render-smoke.mjs --background-dir DIR` can texture the table from a reviewed-clean image directory while keeping the ID pass black/background-free. Smoke path `data/backgrounds/webgl_reviewed_clean_smoke_v1/` is only a two-image proof subset and is not a final background bank.
- Scene variation now includes deterministic camera FOV/pose jitter, multiple procedural table/counter surface palettes, key-light color/intensity/position jitter, and metadata capture of the sampled profile. These are label-safe because they happen inside the WebGL render before RGB and ID extraction.
- WebGL output defaults to 1440x1080 with RGB-only 2x visual supersampling. RGB uses an antialiased visual renderer, while the ID pass uses a separate non-antialiased hidden renderer and is saved from its canvas data URL to preserve exact colors.
- RGB visual pass now includes deterministic non-geometric camera postprocess: mild contrast/saturation/brightness plus grain/vignette overlay. ID pass and visible labels remain unfiltered/exact; geometric lens distortion/crop must wait until masks/boxes are transformed too.
- Primitive finger occluders are top-layer capsule meshes. They render as skin-colored geometry in RGB and black in the ID pass, so covered bill pixels are removed from visible labels.
- Keep primitive fingers smooth high-segment capsules until a proper hand/skin mesh is available. A quick tapered-lathe experiment looked faceted in visual QA and was reverted.
- Note visibility uses explicit layer order (`renderOrder` with depth test/write disabled for banknote planes) so tilted sheets or fingers do not phase through upper layers and poison visible masks. The current smoke audited 88,156 overlapping pixels and 15,732 occluder pixels with zero layer-order violations. Variants >0 now use unique instance ID colors and can emit duplicate denominations in the same scene. This is an intentional topological stacking shortcut until a real contact/physics solver exists.
- ID materials must build colors as sRGB byte values (`Color.setRGB(..., THREE.SRGBColorSpace)`). A mid-channel instance color such as `[255,128,0]` was previously color-managed into `[255,188,0]` and correctly tripped the layer audit.
- Banknote bending should remain smooth macro curl only. The earlier sinusoidal ripple made paper look corrugated under light; default ripple is now zero unless there is a real crease/fold model. Note faces do not receive self-shadows because shadow acne created diagonal artifacts on printed bills.
- Banknote plane height is derived from each loaded scan's pixel aspect ratio (`height = width / aspect`) so USD and KHR assets are not squeezed into one fixed rectangle. Plane width is also scaled by approximate physical denomination width: USD uses the 6.14in/156mm small-note baseline from `https://www.uscurrency.gov/node/45`, and KHR uses current NBC note-size listings from `https://www.nbc.gov.kh/english/about_the_bank/banknotes_in_circulation.php`.
- Texture loading should stay file-backed (`file://`) instead of base64-inlining full scan PNGs into the HTML. Base64 inlining caused headroom pauses; file-backed textures completed the smoke quickly under the same wrapper caps.
- Batch rendering should still reuse one browser/page and design texture downscaling/cache behavior before scaling scene counts, but the headroom wrapper no longer suspends browser jobs merely because free RAM dips below the soft floor.
- `render-smoke.mjs --variant N` is a real deterministic variation hook. Variant 0 is the fixed inspected smoke; variants >0 jitter pose/layer/finger placement and sample front/back/older/current textures from the Numista class pools. Variants 0-3 rendered under the headroom wrapper and passed `check_webgl_smoke_output.py`; contact sheet: `data/synthetic/cashsnap_webgl_variant_contact_v0_3.png`.
- `scripts/render_webgl_variant_batch.py` renders/checks deterministic WebGL variants, writes a contact sheet, packages YOLO detect, OBB, and visible-fragment dataset views, then runs `check_yolo_dataset.py` on the detect view and `check_webgl_label_views.py` on all packaged views by default. Smoke command: `rl python scripts\render_webgl_variant_batch.py --out-root data\synthetic\cashsnap_webgl_variant_batch_smoke --count 4`; variants 0-3 passed and wrote `data/synthetic/cashsnap_webgl_variant_batch_smoke/contact_sheet.png`.
- WebGL asset pools now use the full 13-class CashSnap schema from `data/cashsnap_v1/data.yaml` and load available scan PNGs from `data/asset_candidates/numista_current_cutout_bank_v1/`. Tiny visible slivers below `--min-visible-pixels` (default 500 at 1440p) stay in the ID image but are not exported as class labels.
- Variant class selection uses a deterministic co-prime stride instead of consecutive class IDs, so mixed-currency scenes look less like ordered dataset rows while staying reproducible.
- P0-sized WebGL variant batch: `rl python scripts\render_webgl_variant_batch.py --out-root data\synthetic\cashsnap_webgl_variant_batch_p0 --count 20` completed. `check_yolo_dataset.py --data data\synthetic\cashsnap_webgl_variant_batch_p0\data.yaml` passed with 20 train/val images and 90 boxes balanced across `KHR_5000`, `KHR_10000`, and `KHR_20000`; contact sheet: `data/synthetic/cashsnap_webgl_variant_batch_p0/contact_sheet.png`.
- First fan-mode smoke batch: `rl python scripts\render_webgl_variant_batch.py --out-root data\synthetic\cashsnap_webgl_fan_batch_smoke --start-variant 100 --count 8 --scene-mode fan` passed after sRGB-correct ID colors and zero paper ripple. It wrote 8 train/val images and 48 boxes (`KHR_5000`: 15, `KHR_10000`: 16, `KHR_20000`: 17); contact sheet: `data/synthetic/cashsnap_webgl_fan_batch_smoke/contact_sheet.png`. Best current visual QA sample after 1440p/split-renderer/supersampling changes: `data/synthetic/cashsnap_webgl_fan_probe_v8/visual.png`.
- Real-background smoke: `rl python scripts\run_with_headroom.py ... -- node renderers\webgl\src\render-smoke.mjs --variant 107 --scene-mode fan --background-dir data\backgrounds\webgl_reviewed_clean_smoke_v1 --out-dir data\synthetic\cashsnap_webgl_realbg_probe_v0` passed with 7 boxes, but visual QA shows the proof backgrounds are still too low-quality for training.
- Full-schema fan smoke: `rl python scripts\render_webgl_variant_batch.py --out-root data\synthetic\cashsnap_webgl_fullschema_fan_batch_smoke --start-variant 100 --count 4 --scene-mode fan` passed. It wrote 4 train/val images and 23 boxes under the 13-class schema; contact sheet: `data/synthetic/cashsnap_webgl_fullschema_fan_batch_smoke/contact_sheet.png`.
- Latest aspect-preserving full-schema fan smoke: `rl python scripts\render_webgl_variant_batch.py --out-root data\synthetic\cashsnap_webgl_fullschema_fan_batch_latest_smoke --start-variant 100 --count 4 --scene-mode fan` passed with 4 images and 23 boxes; contact sheet: `data/synthetic/cashsnap_webgl_fullschema_fan_batch_latest_smoke/contact_sheet.png`.
- P0 full-schema fan smoke: `rl python scripts\render_webgl_variant_batch.py --out-root data\synthetic\cashsnap_webgl_fullschema_fan_batch_p0 --start-variant 100 --count 13 --scene-mode fan` passed with 13 images and 67 detect boxes after the co-prime class mixer and visible-sliver floor. Class counts: `USD_1` 6, `USD_5` 4, `USD_10` 6, `USD_20` 6, `USD_50` 7, `USD_100` 3, `KHR_500` 4, `KHR_1000` 6, `KHR_2000` 4, `KHR_5000` 5, `KHR_10000` 5, `KHR_20000` 5, `KHR_50000` 6; contact sheet: `data/synthetic/cashsnap_webgl_fullschema_fan_batch_p0/contact_sheet.png`; OBB dataset sidecar: `data/synthetic/cashsnap_webgl_fullschema_fan_batch_p0/data_obb.yaml`.
- OBB export now audits connected components and rectangle fill before writing labels. On `cashsnap_webgl_fullschema_fan_batch_p0`, all 13 fan images were rejected for trainable OBB because each image had at least one unsafe visible instance; diagnostic metadata still records 34/67 compact instance OBBs, 25 loose min-area rectangles, and 8 fragmented visible masks. This all-or-nothing image policy avoids training OBB with missing labels where skipped visible bills become background.
- Visible-fragment export writes `data_fragments.yaml`, `fragments/labels/train/`, and parent/component metadata. These labels are for evidence detection/OCR, not direct counting. On `cashsnap_webgl_fullschema_fan_batch_p0`, the fragment view has 120 boxes across the same 13 images; on the 3-bill QA set it has 4 boxes because one physical bill splits into two visible components.
- Fragment count-fusion probe: `scripts/probe_fragment_count_fusion.py --root data\synthetic\cashsnap_webgl_qa3_split_composition_v0 --per-image` shows 3 physical bills, 5 visible fragments, and a naive fragment overcount of 2; `--root data\synthetic\cashsnap_webgl_fullschema_fan_batch_p0` shows 67 physical visible instances, 120 fragments, and a naive overcount of 53. Parent-fused synthetic counts match physical counts exactly, so real inference needs a learned or heuristic fusion layer before fragment outputs can become totals.
- Three-bill label QA set: `data/synthetic/cashsnap_webgl_3bill_label_qa_v0/` was packaged from the fixed stack render without relaunching WebGL while RAM was high. It has 3 detect boxes; the trainable OBB split is empty because `KHR_5000` had low rectangle fill (`0.2761`), while the partial OBB polygons are kept only under `obb/rejected_labels/train/` for visual diagnosis. Preview files: `variant_0000_detect_preview.jpg`, `variant_0000_obb_diagnostic_preview.jpg`, and `variant_0000_fragment_preview.jpg`.
- Dedicated split-label QA composition: `render-smoke.mjs --scene-mode qa3` renders a 3-note `KHR_1000/KHR_500/KHR_2000` stack matching the visible-label policy. Output `data/synthetic/cashsnap_webgl_qa3_split_composition_v0/` has 3 physical detect boxes, 5 fragment/evidence boxes, and 0 trainable OBB images because `KHR_500` and `KHR_2000` are fragmented. `check_webgl_smoke_output.py --allow-no-occluder` is required because this QA scene intentionally omits finger occluders.
- Fixed stack variant schema smoke: `data/synthetic/cashsnap_webgl_stack_schema_probe_v0/` passed and emits labels `9/10/11` for `KHR_5000/KHR_10000/KHR_20000` under the full CashSnap schema.
- Aspect-preserving full-schema fan probe: `data/synthetic/cashsnap_webgl_fullschema_fan_probe_v2/` passed `check_webgl_smoke_output.py` at 1440x1080 with 6 exported boxes after the tiny-sliver filter.
- Physical-width full-schema fan probe: `data/synthetic/cashsnap_webgl_fullschema_fan_probe_v3/` passed at 1440x1080 with 6 exported boxes after per-class physical width scaling.
- Mixed-class stride fan probe: `data/synthetic/cashsnap_webgl_fullschema_fan_probe_v4/` passed with `USD_1`, `USD_10`, `USD_100`, `KHR_1000`, `KHR_2000`, and `KHR_50000` visible in one scene.

## Research Checkpoint

2026-05-30 bounded synthesis-data review:

- Synthetic data is useful because it can provide pixel-perfect labels, but domain gap remains the core risk; mixing synthetic with real data is repeatedly favored over treating synthetic as a full replacement. Source: [Eversberg and Lambrecht 2021](https://www.mdpi.com/1424-8220/21/23/7901).
- There is no universal winner between photorealistic rendering and domain randomization. Domain knowledge about the target scene and object-specific realism can matter more than global prettiness. Source: [Eversberg and Lambrecht 2021](https://www.mdpi.com/1424-8220/21/23/7901).
- Domain randomization studies point to object/texture diversity, camera positioning, and occlusion as meaningful parameters; random noise alone is lower leverage. Source: [Domain randomization review, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8570318/).
- Hand-occlusion literature treats occlusion-aware augmentation as a valid first step, and papers cite black solid shapes or object segments pasted over people as established augmentation. This supports primitive finger occluders before a full hand mesh. Source: [HandOccNet CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_HandOccNet_Occlusion-Robust_3D_Hand_Mesh_Estimation_Network_CVPR_2022_paper.pdf).
- Recent synthetic-generation surveys frame 3D generators around either photorealistic rendering or domain randomization, but both still need target-domain validation. Source: [Artificial Intelligence Review 2025 synthetic generation survey](https://link.springer.com/article/10.1007/s10462-025-11431-3).

Decision from this pass: do not chase full PBR or a full hand mesh yet. Build a small WebGL batch proof with controlled banknote layer geometry, real-ish table/background variation, primitive fingers, exact visible masks, and measured comparison against the Python/2.5D baselines plus real partial/fan gates.

2026-05-30 camera/lens decision:

- Three.js still holds for the active renderer proof because it can render fast RGB scenes and exact ID passes locally, and its render-target/readback APIs are enough for future mask extraction. Source: [Three.js render targets](https://threejs.org/manual/en/rendertargets.html), [WebGLRenderer readback](https://threejs.org/docs/pages/WebGLRenderer.html).
- The renderer needs explicit phone-camera profiles, not generic perspective defaults. Seed profiles from common phone FOV/focal-length specs, but prefer EXIF from CashSnap capture images and calibration shots when available.
- Use an OpenCV-style pinhole camera plus radial/tangential distortion model for postprocess: camera matrix plus distortion coefficients `k1`, `k2`, `p1`, `p2`, `k3`. Source: [OpenCV camera calibration](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html).
- Postprocess should be profile-driven: radial distortion, perspective/crop/resize, motion or focus blur, sensor noise/grain, JPEG/WebP compression, white balance, exposure/flash glare, and mild sharpening. Do not randomize these blindly; sample ranges from real captures once enough EXIF/images exist.
- Current public real benchmark candidates are weak camera-profile sources: two have no EXIF and `real_overlap_0002_commons_museum.jpg` only reports Canon EOS 600D make/model. Future phone capture/review pipelines should preserve EXIF so renderer camera profiles come from CashSnap target devices.
- `scripts/run_capture_review_pipeline.py` writes `camera_metadata.jsonl` beside review outputs; use that file to build phone/lens/postprocess priors before tuning WebGL camera profiles.

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
