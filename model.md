# CashSnap Model Brain

This is the active desk for model and synthetic-data work. Keep it compact and useful.
Move old detail to archive instead of letting this file become a wall of text.

Full pre-cleanup history snapshot:
`docs/archive/model_brain_full_history_2026-06-06.md`.

## Mission

Build a small phone/browser-deployable model that counts mixed USD and Khmer
Riel from one casual retail photo.

The hard requirement is partial-note recognition: half notes, quarters, thin
edges, overlaps, fans, stacks, and hand occlusions must be identified when a
human can see enough denomination evidence.

Counterfeit detection and authenticity classification are out of scope.

## Current Read

- WebGL rendering and package QA are strong enough for bounded model probes.
  They are not a "perfect/done" pipeline yet.
- The main blocker is validation and curriculum, not prettier render mechanics.
  Real fan/overlap/hand/no-note stress proof is still missing.
- The current scorecard is blocked: `4` pass / `12` blocked axes after adding
  the currency-taxonomy scope check.
- Current model taxonomy is an operational 13-class subset, not full current
  USD/KHR currency coverage. Missing official current classes are `USD_2`,
  `KHR_50`, `KHR_100`, `KHR_200`, `KHR_15000`, `KHR_30000`,
  `KHR_100000`, and `KHR_200000`.
- Browser failures are mostly detector-proposal/curriculum limited. Fragment
  or final-count heuristics alone are not enough.
- The latest base-first clean probe failed hard. A fresh `yolo26n.pt` trained
  for 20 epochs on p24 real + 32 WebGL clean-base images reached only
  `mAP50-95=0.17524`. Do not stage overlap/fan from that checkpoint.
- The 512-image clean WebGL curriculum root now passes the trainable gate after
  making note-condition diversity scene-aware. It is accepted for a bounded
  clean-checkpoint ablation, not for blind scale or overlap/fan staging.
- The seed0 clean-checkpoint ablation is not promotable: p24 real + 512 clean
  WebGL beats the matched p24 real-only fine-tune (`0.841867` vs `0.835861`,
  `+0.006006` mAP50-95), but still fails the stronger clean checkpoint
  (`0.883801`, delta `-0.041934`). Worst clean-checkpoint drops are
  `KHR_50000=-0.246982` and `KHR_20000=-0.074754`.

## North Star

Synthetic data should be a controlled experiment generator, not just an image
generator.

A future failure should map to one of these causes:

- missing real condition or label bridge
- bad label semantics
- domain/geometry/appearance gap
- curriculum or row-order problem
- model/fusion/deploy bug
- governance or taxonomy gap

Scale only after a synthetic recipe improves a real stress scoreboard under
matched controls and seed stability.

## Immediate Plan

1. Keep the desk clean:
   `model.md` stays current and concise; old details go to `docs/archive/`.
2. Do not promote the 512 clean root. Use its ablation result as signal that
   synthetic exposure can help p24-style scarcity, while `KHR_50000` and
   `KHR_20000` still need a better real/synthetic bridge.
3. Improve render throughput before more 500+ image runs:
   the current batch path launches/checks one WebGL render at a time and took
   about 50 minutes for 512 images. That is expected from the current harness
   but too slow for iteration on this laptop.
4. Only revisit fresh-from-`yolo26n.pt` base-first training if the checkpoint
   ablation is not harmful. Do not stage overlap/fan/hand from a weak clean
   stage.
5. In parallel, improve the real bridge:
   promoted real fan/overlap/hand/no-note labels are still the biggest blocker.

## Promotion Rules

For CashSnap, "perfect synth data" means operational fit, not raw realism.

Promote synthetic data only when all relevant checks pass:

- real P1 exact-count accuracy improves by about `+3pp`, or total-value MAE
  drops by about `10%`
- no high-value or rare class drops by more than about `2pp` / `2 AP`
- hard-negative, unknown, and non-identifiable sliver hallucination does not
  materially increase
- at least 3 seeds are positive; use 5 seeds for large promotions
- matched row-count and class-exposure controls do not explain the gain
- browser/mobile model size, latency, and quantized accuracy stay within budget
- qualitative error review shows real fixes, not shortcut learning

## Label Policy

Visible evidence is authoritative.

- `visible_polygon` / visible mask is the detection truth.
- `visible_bbox` is a compatibility label for detect-only probes.
- `oriented_bbox` is allowed only when the visible region is compact and
  rectangle-like.
- `physical_note_id` is the count target. One note split into several visible
  islands is still one physical note.
- `fragment_id` labels local visible evidence and must be fused back to the
  parent before counting.
- `human_identifiable=false` or `uncertain` should train/evaluate abstention or
  ignore behavior, not forced denomination guesses.
- Preserve side/design, visibility fraction, cue types, occlusion order, and
  ignore reason whenever labels support it.

## Current Assets

Active backbone:

- `data/cashsnap_v1/` is the canonical clean/base YOLO dataset.
  It has the current 13 bill classes and 9,048 labeled boxes.
- `data/asset_candidates/numista_current_cutout_bank_v1/` is the current clean
  scan/cutout bank for WebGL. It covers front/back for the 13 operational
  classes and still has usage/release limits.

Active diagnostic real bridge:

- `runs/cashsnap/mined_real_scoreable_dataset_latest/`
- `runs/cashsnap/mined_real_scoreable_dataset_holdout_latest/`
- `runs/cashsnap/mined_real_browser_cases_latest.csv`

Active WebGL trainable-candidate roots:

- `data/synthetic/cashsnap_webgl_clean_base_candidate_v1`
- `data/synthetic/cashsnap_webgl_overlap_stack_candidate_v1`
- `data/synthetic/cashsnap_webgl_fan_fullschema_candidate_v1`
- `data/synthetic/cashsnap_webgl_hand_occlusion_candidate_v1`
- `data/synthetic/cashsnap_webgl_thin_edge_partial_candidate_v1`
- `data/synthetic/cashsnap_webgl_hard_negative_diversity_catalog_gate_v1`
- `data/synthetic/cashsnap_webgl_back_side_confusion_candidate_v1`

New clean-curriculum probe root:

- `data/synthetic/cashsnap_webgl_clean_base_curriculum_probe_v1`
- Status: rendered, label-checked, and trainable-gate accepted.
- Evidence: 512 images, 1,023 physical count targets, roughly 78-79 boxes per
  current class, appearance diversity passed, OBB trainable for all images.
- Note-condition diversity: clean-scene mixed policy has `pristine=197`,
  `circulated=826`, `dirtiness_range=0.45`, `crinkle_range=0.42`,
  `wetness_range=0.28`. Heavy/wet stress belongs in dedicated stress recipes.
- Model result: one-epoch clean-checkpoint ablation with seed0 passes the
  matched p24 real-only comparison but rejects versus the clean checkpoint.

## Current Models

Default browser detector:

- `runs/cashsnap/yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8/weights/best.pt`
- ONNX is wired through `configs/cashsnap_two_stage_oldcommon_browser_stack.json`.
- Useful for current browser diagnostics, not a solved final model.

Useful curriculum signal:

- `runs/cashsnap/yolo26n_pristine_overlap_e2_i416_b8/weights/best.pt`
- Improves mined-real browser same-class/fan/overlap recall versus default, but
  overcounts and has value noise. Treat as evidence for staged curriculum, not
  as a promotion candidate.

Rejected latest base clean probe:

- `runs/cashsnap/yolo26n_base_clean_webgl_clean_base_v1_e20_i416_b8_seed0/`
- Best/last val: `mAP50-95=0.17524`, `mAP50=0.19789`, precision `0.20579`,
  recall `0.31653`, train class loss `3.4868`.

## Known Results

- Accepted WebGL blend is not stable across seeds and rejects on mined held-out
  rare/edge diagnostic utility. Worst issue remains `KHR_50000`.
- The 512 clean WebGL root is a useful scarcity-control signal but is not
  promotable. It improves over matched p24 real-only by `+0.006006` mAP50-95
  and fails clean-checkpoint guardrails by `-0.041934`, led by `KHR_50000`.
- Real-only p48/bg48 is the best expanded real-only control on the mined
  held-out rare/edge slice, but it is not promotable and still misses the p24
  clean gate.
- Duplicate-only rare oversampling, p96 scaling, and background removal are not
  sufficient repairs.
- Mined-real browser default stack gets `9/17` strict perfect cases on the
  scoreable stress set.
- The old staged overlap detector gets `12/17` perfect on the same mined-real
  browser set and improves fan/dense-overlap recall, but shifts errors into
  overcount/value noise.
- Disagreement/unclassified browser veto flags are diagnostic only. They clear
  some synthetic hard negatives but do not improve mined-real stress.

## Hard Blockers

- No protected/promoted real P1 fan/overlap/hand stress benchmark yet.
- No real repeated-denomination fan labels for same-class counting/fusion.
- No reviewed real no-note/non-banknote paper capture inventory.
- No real mixed USD+KHR rare/common stack captures, especially with
  `KHR_50000`.
- Current official currency taxonomy is incomplete relative to full USD/KHR.
- Clean-real train split still has thin rare anchors:
  `KHR_20000=14` and `KHR_50000=15` unique train images.
- Renderer throughput is too slow for 500+ image iteration on this laptop.

## Commands

Keep this list short. Use script help or archive history for older workflows.

Scorecard:

```powershell
rl python scripts\check_synthetic_dataset_scorecard.py
```

Readiness:

```powershell
rl python scripts\check_synthetic_pipeline_readiness.py --check-existing --json-out runs\cashsnap\synthetic_pipeline_readiness_latest.json
```

Trainable-candidate suite:

```powershell
rl python scripts\check_webgl_trainable_candidate_suite.py --check-existing
```

Current clean-curriculum root checks:

```powershell
rl python scripts\check_webgl_note_condition_diversity.py --root data\synthetic\cashsnap_webgl_clean_base_curriculum_probe_v1 --allow-missing
rl python scripts\check_webgl_trainable_candidate_gate.py --root data\synthetic\cashsnap_webgl_clean_base_curriculum_probe_v1 --require-recipe webgl_clean_base_v1 --train-views detect,fragment,obb --require-scene-mode clean --require-asset-side-policy any --require-camera-profile phone_auto
```

Mined real bridge:

```powershell
rl python scripts\check_mined_real_review_quality.py
rl python scripts\build_mined_real_scoreable_dataset.py --min-images 17 --min-stress-images 17 --min-boxes 35
```

Capture gaps:

```powershell
rl python scripts\check_capture_requirements.py --json-out runs\cashsnap\capture_requirements_latest.json --shot-list-out runs\cashsnap\real_capture_shot_list_latest.md
```

## Laptop Runtime

Machine profile:

- Lenovo 82Y9 laptop
- Ryzen 5 7640HS, 6 cores / 12 logical processors
- 16 GB RAM
- RTX 4060 Laptop GPU, 8 GB VRAM

Current operating posture:

- Heavy training/rendering should use `scripts/run_with_headroom.py`.
- Prefer GPU for training/inference when VRAM is available.
- RAM pressure is the main bottleneck. Keep data-loader workers low.
- Browser smoke/render jobs should not run in parallel unless the Edge profile
  and cache paths are isolated.
- For WebGL scale, fix batch throughput before another 500+ image render.
  Launching Edge/Node per variant is correct but too slow.

## Repo Hygiene

- Work on `master` unless the user asks for a branch.
- Keep generated training and browser outputs under ignored `runs/`.
- Keep generated synthetic roots under ignored `data/synthetic/`.
- Keep active planning in this `model.md`; archive old model history under
  `docs/archive/`.
- Do not add new active model-planning docs under `docs/`.
- Leave unrelated dirty/untracked files alone unless they support the current
  task.

## Research Sources

- `docs/research/CashSnap_Fact_Checked_Dataset_Strategy.pdf`
- `docs/research/What Makes a Dataset Perfect for Synthetic Data Pipelines.pdf`
- USD denominations: https://www.uscurrency.gov/denominations
- KHR banknotes: https://www.nbc.gov.kh/english/about_the_bank/banknotes_in_circulation.php
