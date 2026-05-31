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

## Synthetic Data Completion Plan

Goal: make synthetic data a controlled experiment generator, not just an image generator. A future model failure should map to a specific missing condition, label issue, domain gap, training mix, or fusion bug instead of the vague answer "we need better data."

Definition of done for the synthetic pipeline:

- [x] Real-target matrix exists: required photo conditions, parked conditions, scoreable benchmark slices, and success metrics are named in `configs/synthetic_targets/cashsnap_real_target_matrix_v1.json`.
- [x] Asset bank is audited by class, side, design generation, circulation priority, source, license/usage status, alpha quality, dimensions, and visual QA.
- [x] Numista cutout audit writes `audit/summary.json` with class counts, front/back coverage, source-status counts, year ranges, suspect flags, and audit output paths.
- [ ] Scene generator covers normal notes, loose stacks, dense overlap, fans, partial edges, repeated denominations, front/back mixes, hand/finger occlusion, reviewed backgrounds, and phone-like framing.
- [ ] Camera and postprocess profiles are derived from real captures where possible: lens/FOV, perspective, crop, blur, noise, compression, exposure, white balance, glare, and lighting.
- [x] Label truth separates physical parents from visible fragments and exports exact ID masks plus visible-only detect labels.
- [x] OBB labels are audited and rejected when the visible mask is fragmented or too loose for an honest rotated rectangle.
- [x] Fragment/evidence labels are exported separately from physical count truth.
- [x] Below-threshold visible fragments are recorded as ignored metadata instead of forced training labels.
- [x] Kept fragment labels carry `trainable` vs `review_required` evidence metadata so small/low-fraction fragments are visible to promotion gates.
- [x] Ambiguous human-unidentifiable fragments have a full ignore/unknown policy beyond the current pixel threshold.
- [x] Label transforms remain exact after any crop, resize, distortion, or other geometric postprocess.
- [x] Batch QA writes a machine-readable summary for class balance, visible area, fragment counts, fragments per parent, OBB rejection reasons, layer-audit totals, and deterministic file hashes.
- [x] Batch QA writes detect and fragment preview overlays under `qa/previews/` and validates their existence.
- [x] Batch QA writes visual+ID mask overlays under `qa/previews/` and validates their existence.
- [x] Batch QA writes `qa/quarantine.json` for trainable-view exclusions and ignored below-threshold fragment components.
- [x] Batch QA writes `qa/contact_index.json` mapping contact-sheet cells back to variants.
- [x] Batch QA writes deterministic visual-quality metrics and smoke gates reject blank/exposure-failed artifacts.
- [x] Smoke artifacts have mode-specific promotion/quarantine gates that run from the recipe wrapper.
- [x] Hard-negative scene mode emits valid zero-box/zero-fragment packages with empty YOLO labels and full smoke QA.
- [x] Thin-edge scene mode emits visible sliver packages with card/paper occluders and records unsafe OBB/fragment exclusions.
- [x] Broader hand-occlusion scene mode emits multi-finger split-fragment packages and quarantines unsafe OBB views.
- [x] Full human visual QA suite includes realism snapshots and bad-scene review rules beyond deterministic blankness/exposure gates.
- [x] Named synthetic recipe slots exist for clean/base, overlap, fan, hand occlusion, thin-edge partials, back-side confusion, rare-class support, hard negatives, and calibration mixes in `configs/synthetic_recipes/cashsnap_webgl_recipe_catalog_v1.json`.
- [x] Batch outputs include `recipe.json` with recipe name, smoke/diagnostic/trainable-candidate status, variant seed range, intended use, checks, outputs, and trainability policy.
- [x] Smoke-ready WebGL recipes can be run or repackaged as a single gated suite.
- [x] Trainable-candidate WebGL artifacts have a gate for visual rejects, layer violations, selected train views, OBB rejection, and review-required fragments.
- [x] A bounded trainable-candidate suite manifest defines recipe seed ranges, output paths, train views, intended use, and required generated QA/manifest paths.
- [ ] Each trainable recipe has config, seed range, asset manifest, output path, QA summary, intended use, and a clear trainable-vs-diagnostic marker.
- [ ] P1 transfer proof compares WebGL synthetic against no-synthetic and matched 2.5D baselines on clean validation, real partial/fan labels, count metrics, and browser smoke.
- [ ] Fragment-to-physical-note fusion exists for real inference and handles split notes, repeated same-denomination notes, ambiguous backs, and count totals.
- [ ] Operations are one-command reproducible: render, QA/package, train under headroom, evaluate clean/real/browser guards, and clean scratch outputs.
- [ ] Promotion rules require real-scoreboard improvement, clean-validation guardrails, browser/deploy guardrails, and enough metadata to diagnose regressions.

Current completion status: asset-bank audit coverage, renderer, and label contract are proven at P0; target/recipe coverage is explicit; and WebGL packages carry QA, recipe, ignored-fragment metadata, fragment evidence-review metadata, fragment ambiguity ignore policy, geometric postprocess label guards, deterministic visual-quality gates, human visual-review packs, explicit front/back asset-side policy, explicit phone camera-profile policy, background-bank review gates, smoke/trainable-candidate gates, clean-scene smoke, hard-negative zero-box smoke, thin-edge sliver smoke, and hand-occlusion fragment smoke. The production training-data factory is still not complete. The next bottleneck is promoting smoke-ready recipes through real-gated P1 training experiments, then improving promotion rules and full operations.

## Work Loop

1. Read this file.
2. Check git status.
3. Work only on the current bottleneck.
4. Run bounded checks.
5. Update this file when a result changes the plan.
6. Commit coherent durable changes, not every tiny scratch edit.
7. Record durable experiment results in the result ledger below; do not rely on sidecar ledgers as project memory.

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

Validate active 3D configs:

```powershell
rl python scripts\validate_3d_pipeline_config.py configs\3d_pipeline\proof_p0_renderer_smoke.json configs\3d_pipeline\proof_p1_transfer.json
```

Validate synthetic target/recipe coverage:

```powershell
rl python scripts\check_synthetic_recipe_catalog.py
```

Render P0 proof scenes:

```powershell
rl python scripts\render_3d_pipeline_probe.py --config configs\3d_pipeline\proof_p0_renderer_smoke.json --clean
```

Render the minimal WebGL proof:

```powershell
rl python scripts\run_with_headroom.py --max-percent 90 --resume-percent 82 --max-ram-percent 90 --max-gpu-mem-percent 90 --min-free-ram-gb 3 --preflight-timeout 120 -- node renderers\webgl\src\render-smoke.mjs
```

Run a named WebGL synthetic recipe:

```powershell
rl python scripts\run_webgl_recipe.py --recipe-id webgl_clean_base_v1 --count 3 --min-free-ram-gb 2
```

Run the gated WebGL smoke suite:

```powershell
rl python scripts\run_webgl_smoke_suite.py --skip-render
```

Build the gated smoke-suite YOLO mix:

```powershell
rl python scripts\build_webgl_mix_yaml.py --out configs\cashsnap_webgl_smoke_suite_mix.yaml
```

Check WebGL P1 transfer readiness:

```powershell
rl python scripts\check_webgl_p1_readiness.py
```

Run the current WebGL P1 diagnostic pipeline:

```powershell
rl python scripts\run_webgl_p1_diagnostic_pipeline.py
```

Check WebGL smoke output:

```powershell
rl python scripts\check_webgl_smoke_output.py --out-dir data\synthetic\cashsnap_webgl_smoke
```

Gate a packaged WebGL smoke artifact:

```powershell
rl python scripts\check_webgl_smoke_gate.py --root data\synthetic\cashsnap_webgl_clean_batch_smoke --require-recipe webgl_clean_base_v1 --require-scene-mode clean
```

Gate a packaged WebGL trainable candidate:

```powershell
rl python scripts\check_webgl_trainable_candidate_gate.py --root data\synthetic\candidate_root --require-recipe recipe_id --train-views detect
```

Validate the bounded WebGL trainable-candidate suite:

```powershell
rl python scripts\check_webgl_trainable_candidate_suite.py
rl python scripts\run_webgl_trainable_candidate_suite.py --dry-run
rl python scripts\build_webgl_mix_yaml.py --suite configs\synthetic_recipes\cashsnap_webgl_trainable_candidates_v1.json --gate-kind trainable-candidate --out configs\cashsnap_webgl_trainable_candidates_mix.yaml
```

Build and check a WebGL visual review pack:

```powershell
rl python scripts\make_webgl_visual_review_pack.py --suite configs\synthetic_recipes\cashsnap_webgl_smoke_suite_v1.json --out-dir data\review\webgl_smoke_visual_review_v1
rl python scripts\check_webgl_visual_review.py --review-csv data\review\webgl_smoke_visual_review_v1\review.csv
```

Dry-run the WebGL trainable-candidate operations sequence:

```powershell
rl python scripts\run_webgl_trainable_candidate_pipeline.py --dry-run --train-smoke
```

Protect the real benchmark boundary:

```powershell
rl python scripts\check_real_fan_benchmark.py
```

Dry-run reviewed real-label promotion:

```powershell
rl python scripts\promote_real_benchmark_label.py --image-id real_overlap_0003_commons_shop_5k_10k_20k
```

Dry-run safe training only after the renderer proof exists:

```powershell
rl python scripts\bench_train_with_headroom.py --data configs\cashsnap_v1.yaml --name dry_run --dry-run --quiet
```

## Result Ledger

Keep this table curated. Add rows only for results that change what a future agent should trust, avoid, or run next.

| Date UTC | Area | Status | Result |
| --- | --- | --- | --- |
| 2026-05-30 13:35 | harness | keep | Adapted the autoresearch program/results harness and validated active 3D configs. |
| 2026-05-30 13:53 | renderer | keep | P0 renderer scaffold emitted 20 visual/ID-pass scenes and YOLO labels; `check_yolo_dataset.py` passed. |
| 2026-05-30 14:03 | renderer | keep | P0 renderer deformation plus pre-resize reruns are deterministic and emit label stats/OBB sidecars. |
| 2026-05-30 14:14 | harness | keep | Headroom wrapper records the local rig, refuses caps over 95, preflight-waits, and auto-selects GPU device 0 for YOLO. |
| 2026-05-30 14:22 | renderer | keep | P1 renderer proof emitted 240 scenes under the headroom wrapper and YOLO dataset check passed. |
| 2026-05-30 19:57 | renderer | note | WebGL split-label QA: `qa3` has 3 physical notes vs 5 fragments; full-schema fan P0 has 67 physical visible instances vs 120 fragments, so fragment evidence requires fusion before count totals. |
| 2026-05-30 20:22 | renderer | keep | WebGL batch packager writes and validates `qa/summary.json` with class, visibility, fragment, OBB, layer-audit, and hash stats. |
| 2026-05-30 20:23 | renderer | keep | WebGL batch outputs include `recipe.json` with recipe name/status/seed range/checks/outputs so smoke, diagnostic, and trainable-candidate artifacts are explicit. |
| 2026-05-30 20:24 | data | keep | Numista cutout audit writes `summary.json`; current bank has 76 assets, full 13-class front/back coverage, and 3 red-mark visual-review suspects. |
| 2026-05-30 20:27 | data | keep | Added `cashsnap_real_target_matrix_v1` with 11 named real conditions, required/parked scope, synthetic recipe needs, benchmarks, and success gates. |
| 2026-05-30 20:28 | data | keep | Added WebGL recipe catalog with 9 named recipe slots mapped to real target conditions, promotion gates, and blockers. |
| 2026-05-30 20:29 | harness | keep | Added `check_synthetic_recipe_catalog.py` to validate target/recipe JSON coverage and required-condition mapping. |
| 2026-05-30 20:31 | renderer | keep | Fragment packaging records below-threshold components as ignored metadata and validates ignored counts in label-view QA. |
| 2026-05-30 20:45 | renderer | keep | WebGL `clean` scene mode passed a 3-image smoke (`webgl_clean_smoke_v0_2`) with 6 detect boxes, 6 fragments, and 3/3 trainable OBB images; packager headroom thresholds are now configurable and recorded in `recipe.json`. |
| 2026-05-30 20:49 | harness | keep | Added `run_webgl_recipe.py`; `webgl_clean_base_v1` can now be run by recipe id and repackaged the clean smoke successfully with full label-view checks. |
| 2026-05-30 20:53 | renderer | keep | WebGL batch packaging writes detect and fragment preview overlays under `qa/previews/`, records their hashes in `qa/summary.json`, and validates preview existence in label-view QA. |
| 2026-05-30 20:57 | renderer | keep | WebGL batch packaging writes `qa/quarantine.json`; stack smoke records 3 OBB trainable exclusions and 2 ignored below-threshold fragment components with explicit policies. |
| 2026-05-30 21:00 | renderer | keep | WebGL batch packaging writes visual+ID mask overlay previews, hashes them in `qa/summary.json`, and validates them in label-view QA. |
| 2026-05-30 21:03 | renderer | keep | WebGL batch packaging writes `qa/contact_index.json` so contact-sheet visual and ID cells map back to variants; label-view QA validates the index. |
| 2026-05-30 21:06 | renderer | keep | `check_webgl_label_views.py` recomputes `qa/summary.json` file hashes for visual, ID, labels, and previews; clean and stack smoke packages pass after repackaging. |
| 2026-05-30 21:17 | renderer | keep | WebGL `negative` scene mode passed a 2-image hard-negative smoke with 0 boxes/fragments; renderer emits zero-byte `labels_visible.txt`, dataset checks accept blank YOLO labels, and full label-view QA passes. |
| 2026-05-30 21:21 | renderer | keep | WebGL `thin_edge` scene mode passed a 3-image smoke with 10 detect boxes, 11 fragments, 4 ignored below-threshold components, 1/3 trainable OBB images, and zero layer-order violations; treat as smoke only until ambiguity policy and real transfer gates exist. |
| 2026-05-30 21:26 | renderer | keep | WebGL `hand_occlusion` scene mode passed a 3-image smoke with 13 detect boxes, 32 fragments, 10 split parents, 6 ignored tiny components, 0/3 trainable OBB images, and zero layer-order violations; use for fragment diagnostics, not OBB training. |
| 2026-05-30 21:31 | harness | keep | Added `check_webgl_smoke_gate.py` and wired `run_webgl_recipe.py` to gate smoke artifacts; negative, thin-edge, hand-occlusion, and clean smoke packages pass, while older fan artifacts need repackaging because they lack `recipe.json`. |
| 2026-05-30 21:36 | harness | keep | Repackaged older fan and stack smoke through `run_webgl_recipe.py --skip-render`; packager now backfills missing `sceneMode` from the requested scene mode, and both `webgl_fan_fullschema_v1` plus `webgl_overlap_stack_v1` pass smoke gates. |
| 2026-05-30 21:40 | harness | keep | Added `cashsnap_webgl_smoke_suite_v1` plus `run_webgl_smoke_suite.py`; `--skip-render` suite ran all 6 smoke-ready recipes (clean, stack, fan, thin-edge, hand-occlusion, negative) through smoke/YOLO/label-view/gate checks. |
| 2026-05-30 21:46 | harness | keep | Added `build_webgl_mix_yaml.py` and `configs/cashsnap_webgl_smoke_suite_mix.yaml`; gated smoke mix validates as 19 images / 64 boxes, diagnostic-only and not a training-performance claim. |
| 2026-05-30 21:49 | training | note | `bench_train_with_headroom.py --data configs/cashsnap_webgl_smoke_suite_mix.yaml --dry-run` accepts the gated mix and selects GPU 0, batch 2, workers 0, repo-local `runs/cashsnap`; do not launch while free RAM is below the configured 4GB floor. |
| 2026-05-30 21:51 | evaluation | note | P1 real/deploy guardrails are structurally present but not full scoreboards: `check_real_fan_benchmark.py` passes with 3 candidate sources, 0 promoted labeled images, and 1 draft-labeled 6-box overlap image; browser smoke manifest validates 5 cases. |
| 2026-05-30 21:52 | evaluation | note | `filter_yolo_labels_by_quality.py` keeps 6/6 labels for the draft shop-overlap diagnostic at `data/audit/real_overlap_0003_commons_shop_5k_10k_20k.scoreable.txt`; this is local diagnostic material, not promoted benchmark truth. |
| 2026-05-30 21:54 | evaluation | note | Current alpha `yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8` on the scoreable shop-overlap draft tops out at 3/6 same-class and 5/6 any-class (`416/conf=0.03` or `640/conf=0.03`); use this as a local diagnostic hurdle, not a final benchmark. |
| 2026-05-30 21:58 | evaluation | note | Added `check_webgl_p1_readiness.py`; smoke mix, real-benchmark structure, and browser manifest pass, but full P1 transfer is blocked by 0 promoted real benchmark labels (6 draft scoreable boxes remain diagnostic-only). |
| 2026-05-31 14:00 | training | keep | `train_yolo.py`/`val_yolo.py` now normalize YOLO data YAML roots into `.cache_runtime/ultralytics_data/` before calling Ultralytics; this fixed the smoke mix path bug and a 1-epoch/2-batch headroom train completed, writing `runs/cashsnap/webgl_smoke_mix_tiny_train_probe/weights/best.pt` with diagnostic mAP50-95 `0.00214`. |
| 2026-05-31 14:06 | evaluation | keep | Added `run_webgl_p1_diagnostic_pipeline.py`; real skip-render run passed the smoke suite, gated mix build, P1 readiness check, draft scoreable-label refresh, and alpha draft eval. It remains diagnostic-only because P1 readiness still reports 0 promoted real benchmark labels. |
| 2026-05-31 14:13 | evaluation | keep | Added guarded `promote_real_benchmark_label.py`; default dry-run validates label format and scoreable quality rows, while `--confirm-reviewed --reviewed-by NAME` copies reviewed labels, updates manifests, logs hashes, renders a preview, and reruns the real benchmark check. Temp self-test promoted the 6-box draft into `.cache_runtime/` only. |
| 2026-05-31 14:15 | evaluation | note | `check_webgl_p1_readiness.py` now reports review-ready draft image ids; current blocker is still 0 promoted labels, but `real_overlap_0003_commons_shop_5k_10k_20k` is identified as a 6-box draft candidate ready for explicit human-review promotion. |
| 2026-05-31 14:19 | training | note | `run_webgl_p1_diagnostic_pipeline.py --train-smoke --skip-alpha-eval` passed end-to-end, including headroom tiny training; headroom paused/resumed once on CPU pressure, train CSV still reports diagnostic mAP50-95 `0.00214`, and final best-checkpoint validation printed mAP50-95 `0.00231`. |
| 2026-05-31 14:24 | renderer | keep | Added deterministic WebGL visual-quality QA: packager writes `qa/visual_quality.json`, summary status counts, and visual quarantine rows; label-view QA validates the file and smoke gates require zero visual rejects. Repacked smoke suite shows all 19 smoke images accepted with no visual-quality failures. |
| 2026-05-31 14:28 | renderer | keep | Added fragment evidence-review metadata: kept fragments now record `trainable` or `review_required` plus warning reasons for small/low-parent-fraction evidence, summaries count these statuses, and quarantine marks affected images. Repacked smoke suite flags 8 review-required fragments while keeping smoke labels intact. |
| 2026-05-31 14:32 | harness | keep | Added `check_webgl_trainable_candidate_gate.py` and wired it into `run_webgl_recipe.py` for `artifact_status=trainable-candidate`; gate validates visual rejects, layer violations, selected train views, OBB rejection, and review-required fragments. End-to-end clean probe passed under `.cache_runtime/webgl_trainable_gate_clean`. |
| 2026-05-31 21:39 | harness | keep | Added `cashsnap_webgl_trainable_candidates_v1` plus suite checker/runner; structural check now declares 7 candidate recipes / 304 images with detect on all rows, fragment on clean+stack, and OBB only on clean. `run_webgl_trainable_candidate_suite.py --dry-run`, default smoke mix rebuild, and a one-image trainable-gated mix probe passed; actual candidate artifacts are not rendered yet. |
| 2026-05-31 21:45 | renderer | keep | Added WebGL human visual-review rules plus `make_webgl_visual_review_pack.py` and `check_webgl_visual_review.py`; a smoke-suite probe wrote 12 review rows with RGB/detect/fragment/ID previews, and both pending-review validation plus synthetic all-accepted strict validation passed. |
| 2026-05-31 21:51 | renderer | keep | Added `--fragment-review-policy`; diagnostic packages keep review-required fragment labels, while trainable fragment views can move ambiguous components to ignored metadata with `requires_human_review`. Hand-occlusion trainable fragment probe passed with 28 trainable fragments, 10 ignored fragments, and 0 review-required labels. |
| 2026-05-31 21:54 | renderer | keep | Added a label-transform guard to `check_webgl_label_views.py`: geometric postprocess keys such as crop, resize, radial distortion, perspective, homography, warp, or scale now fail unless an exact shared RGB/ID/label transform path is implemented. Current WebGL packages declare non-geometric RGB-only postprocess. |
| 2026-05-31 21:57 | harness | keep | Added `run_webgl_trainable_candidate_pipeline.py` to chain trainable suite checks/renders, trainable-gated mix build, WebGL visual review pack/check, P1 readiness, and optional tiny train smoke from one audited command sequence. |
| 2026-05-31 22:00 | assets | keep | Extended `audit_cutout_bank.py` to summarize design generation, circulation priority, license/usage status, alpha quality, dimensions, and visual QA status. Current Numista bank remains internal/reference until usage review is complete; the script now makes that blocker explicit. |
| 2026-05-31 22:08 | renderer | keep | Added explicit WebGL `asset_side_policy` plumbing (`any`, `front_only`, `back_only`, `front_back_mix`) through renderer, packager, recipe runner, suite, mix builder, and gates. A one-image `.cache_runtime/webgl_front_back_mix_probe` passed with `front_back_mix` satisfied (2 front, 1 back). |
| 2026-05-31 22:16 | renderer | keep | Added explicit WebGL `camera_profile` plumbing (`generic_phone_jitter`, `phone_auto`, and named phone/browser profiles from `configs/3d_pipeline/proof_p1_transfer.json`) through renderer, packager, recipes, suite, mix builder, and gates. A one-image `.cache_runtime/webgl_phone_profile_probe` passed with `phone_auto` selecting `iphone_8_like`; geometric lens distortion remains blocked until RGB/ID/labels share the exact transform. |
| 2026-05-31 22:23 | renderer | keep | Added `cashsnap_webgl_background_banks_v1` plus `check_webgl_background_banks.py`; WebGL packager now rejects `--background-dir` when the bank review status does not allow the requested artifact status. Current registered banks: 1 proof-only smoke/diagnostic subset, 5 pending-review banks, and 1 rejected historical smoke bank. No trainable background bank is accepted yet. |
| 2026-05-31 22:33 | renderer | keep | Rendered first actual trainable-candidate artifact: `data/synthetic/cashsnap_webgl_clean_base_candidate_v1` (32 images, 65 detect/fragment boxes, 32/32 trainable OBB images, 0 visual rejects, `phone_auto` profile spread across all 4 named profiles). `check_webgl_trainable_candidate_gate.py` passed for detect+fragment+OBB. |
| 2026-05-31 22:38 | renderer | keep | Headless Edge WebGL reports `ANGLE (NVIDIA, NVIDIA GeForce RTX 4060 Laptop GPU ... D3D11)`, so rendering is already GPU-backed; RAM pressure comes from browser/canvas/image packaging. Added pass-through `--width`, `--height`, `--visual-scale`, and headroom cap knobs to WebGL recipe/suite/pipeline runners for Chrome-friendly renders. |

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

- Harness and safety: `run_with_headroom.py`, `bench_train_with_headroom.py`, `local_runtime.py`.
- 3D/synthetic proof: `render_3d_pipeline_probe.py`, `validate_3d_pipeline_config.py`, `build_numista_cutout_bank.py`, `generate_synthetic_fan_dataset.py`, `summarize_synthetic_metadata.py`, `check_yolo_dataset.py`.
- WebGL proof: `renderers/webgl/src/render-smoke.mjs` uses Three.js + `puppeteer-core` against local Microsoft Edge; `run_webgl_recipe.py` resolves named recipe ids from the catalog; `check_webgl_smoke_output.py` validates nonblank RGB, exact ID colors, visible labels, primitive finger occlusion, and layer-order audit output; `check_webgl_label_views.py` validates packaged detect/OBB/fragment views and prevents diagnostic-only OBB labels from being mistaken for trainable data; `check_synthetic_recipe_catalog.py` validates target/recipe config coverage.
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
- RGB table rendering now uses a deterministic procedural grain texture. Existing `data/backgrounds/smoke_cashsnap_no_note_patches_v4/` contact sheet includes hands/note fragments/grass, and `configs/synthetic_recipes/cashsnap_webgl_background_banks_v1.json` marks it rejected.
- `render-smoke.mjs --background-dir DIR` can texture the table from a reviewed-clean image directory while keeping the ID pass black/background-free. `render_webgl_variant_batch.py` gates this through `check_webgl_background_banks.py`; smoke path `data/backgrounds/webgl_reviewed_clean_smoke_v1/` is only a two-image proof subset and is blocked from trainable-candidate use.
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
- `render-smoke.mjs --asset-side-policy any|front_only|back_only|front_back_mix` constrains scan-side sampling before render. Packaged QA records `asset_selection`, and smoke/trainable gates can require the selected policy so front/back confusion recipes do not depend on accidental sampling.
- `render-smoke.mjs --camera-profile generic_phone_jitter|phone_auto|iphone_8_like|iphone_12_wide_like|budget_android_wide_like|browser_upload_resized` selects auditable FOV/framing ranges before RGB/ID extraction. `phone_auto` samples the named phone/browser profiles from the older P1 config; lens distortion is recorded as not applied until label-safe geometric transforms exist.
- Headless Edge WebGL is GPU-backed on the local rig (`ANGLE` on RTX 4060 via D3D11). If Chrome/RAM is tight during render batches, prefer lowering `--visual-scale` and passing stricter headroom caps through `run_webgl_trainable_candidate_suite.py` or `run_webgl_trainable_candidate_pipeline.py`; this reduces browser/canvas RAM pressure more directly than trying to move packaging to VRAM.
- `scripts/render_webgl_variant_batch.py` renders/checks deterministic WebGL variants, writes a contact sheet, packages YOLO detect, OBB, and visible-fragment dataset views, then runs `check_yolo_dataset.py` on the detect view and `check_webgl_label_views.py` on all packaged views by default. Smoke command: `rl python scripts\render_webgl_variant_batch.py --out-root data\synthetic\cashsnap_webgl_variant_batch_smoke --count 4`; variants 0-3 passed and wrote `data/synthetic/cashsnap_webgl_variant_batch_smoke/contact_sheet.png`.
- The WebGL batch packager now writes `qa/summary.json` with class counts, visible-pixel stats, fragment-per-parent stats, OBB rejection reasons, layer-audit totals, and SHA-256 hashes for reproducibility checks. `check_webgl_label_views.py` validates this summary against the manifest so the QA artifact cannot silently drift.
- `check_webgl_label_views.py` recomputes `qa/summary.json` hashes for visual images, ID masks, label files, and preview images, so QA summaries function as lightweight regression snapshots.
- WebGL batch packaging writes detect and fragment label-preview overlays under `qa/previews/`; manifest rows point to them, `qa/summary.json` stores their hashes, and `check_webgl_label_views.py` validates that they exist.
- WebGL batch packaging writes visual+ID mask overlay previews under `qa/previews/`; these make ID-mask alignment visually reviewable without opening separate visual/mask files.
- WebGL batch packaging writes `qa/quarantine.json` with explicit policies for trainable OBB exclusions and ignored below-threshold fragments. Stack smoke currently records 3 OBB exclusions and 2 ignored tiny fragment components.
- WebGL batch packaging writes `qa/contact_index.json` to map contact-sheet visual/ID cells back to variants, and `check_webgl_label_views.py` validates the index.
- Fragment packaging now writes `fragments/ignored_metadata/` for connected components below `FRAGMENT_MIN_PIXELS`; `fragments/summary.json`, `qa/summary.json`, and `check_webgl_label_views.py` validate ignored counts so tiny evidence is not silently forced into denomination labels.
- `check_webgl_smoke_gate.py` applies mode-specific smoke gates after label-view validation. New smoke artifacts must include `recipe.json`; older fan/stack artifacts without it need repackaging before they can be treated as gateable evidence.
- `render_webgl_variant_batch.py --skip-render --scene-mode MODE` backfills missing `sceneMode` in copied source metadata, so older rendered variants can be repackaged into current gateable smoke artifacts without relaunching Edge.
- WebGL batch outputs now include `recipe.json` with recipe name, artifact status (`smoke`, `diagnostic`, or `trainable-candidate`), variant seed range, checks, output paths, and trainability policy. Smoke verification used `--recipe-name webgl_stack_smoke_v0_3 --artifact-status smoke --intended-use "renderer and label-view smoke proof"` with `--skip-render`.
- WebGL `clean` scene mode now supports separated/single-note smoke data. `webgl_clean_smoke_v0_2` passed with 3 images, 6 detect boxes, 6 fragments, and 3/3 trainable OBB images after running the packager with `--min-free-ram-gb 2`; the hard RAM cap stayed at 90%.
- WebGL asset pools now use the full 13-class CashSnap schema from `data/cashsnap_v1/data.yaml` and load available scan PNGs from `data/asset_candidates/numista_current_cutout_bank_v1/`. Tiny visible slivers below `--min-visible-pixels` (default 500 at 1440p) stay in the ID image but are not exported as class labels.
- Current Numista cutout audit summary: 76 assets, all 13 classes have front/back coverage, all source rows are `in_circulation`, and 3 assets remain `large_red_mark_suspect` visual-review candidates. Summary path: `data/asset_candidates/numista_current_cutout_bank_v1/audit/summary.json`.
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
