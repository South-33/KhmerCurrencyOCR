# CashSnap Model Brain

This is the one living document for model work. If the plan changes, update this file first. Keep it clean, current, and useful for the next agent dropped into the repo.

## Mission

Build a small phone/browser-deployable model that counts mixed USD and Khmer Riel from one casual retail photo. The hard requirement is partial-note recognition: half, quarter, thin edge, overlapped, fanned, and hand-occluded notes must still be identified when humans can see enough denomination evidence.

Counterfeit detection is out of scope.

## Current North Star

Active bet: make synthetic data generation good enough to be the scaling unit, but judge it only by real partial/fan transfer.

Current phase: WebGL synthetic generation is mechanically auditable, but training/curriculum robustness is the blocker. The full-real real-only control also collapses, so the immediate failure is the all-target/background-heavy curriculum and rare-class exposure shape, not WebGL texture by itself. Capped real-only probes recover much of the all-target loss but are not promotable yet: p48/bg48 tests at `0.833190`, p48/bg48 with rare duplicate oversampling at `0.832952`, p96/bg96 at `0.819153`, versus the p24 real-only gate at `0.835861` and the clean checkpoint at `0.883801`. The accepted WebGL blend is not stable across seeds (`0.836855` seed0, `0.833769` seed1, paired mean delta `-0.004986` vs p24), and tiny-dose probes are row-count/order sensitive: neutral row-control dose1 tests at `0.809414`, while neutral row-control dose2 tests at `0.847043` and drops to `0.842007` when capped to the accepted base's 223 train batches. Clean-single KHR synthetic dose1 beats the collapsed row-control dose1 but still misses accepted seed1; clean-single KHR dose2 fails its fixed-step neutral row-control for both `KHR_10000` (`0.817981`, `-0.024026`) and `KHR_50000` (`0.818783`, `-0.023224`), yet beats same-class duplicate controls for both denominations under full-epoch probes. The renderer is good enough to run controlled experiments; the model-side probe harness, curriculum, and real validation bridge are not yet safe enough to scale into the actual model build.

Current model direction after split-label QA: keep the lightweight detector path for clean/mostly-visible notes, add a fragment/evidence detector or classifier path for partial OCR/denomination cues, then fuse fragments into physical-note counts. Do not treat OBB as the primary hard-fan solution; use OBB only for compact visible instances where the rotated box is honest.

Training-smoke results prove the harness, not model quality. The renderer/package contract now exists; the missing evidence is scoreable real fan/overlap stress labels for P1 transfer proof.

Latest model comparison: the old 621-image accepted-WebGL blend remains rejected (`0.810` held-out clean test mAP50-95 vs `0.884` clean checkpoint). A stricter 445-image blend of individually accepted WebGL recipes passed one matched real-only clean-test seed (`0.836855` vs `0.835861`, delta `+0.000994`) but fails stability: seed1 tests at `0.833769`, and the paired seed0/seed1 gate rejects it against p24 real-only (`mean_delta -0.004986`, worst `KHR_2000 -0.141446`). Whole-recipe leave-one-out probes did not fix it. Rare-support staged dose 2 is useful diagnostic signal but not a win: fan seed0 passed (`0.838473`, `+0.002612` vs p24 seed0 real-only), but fan seed1 rejected against p24 seed1 real-only (`0.826076`, `-0.018658`, worst `KHR_50000 -0.218713`); stack seed0 passed (`0.837617`, `+0.001756`), but stack seed1 rejected against p24 seed1 real-only (`0.826476`, `-0.018259`, worst `KHR_50000 -0.217202`). Clean-single KHR dose probes prove that direct accepted-base comparisons are confounded: row-control dose1 collapses to `0.809414`, so KHR_50000/KHR_10000 dose1 are not worse than a one-row neutral perturbation (`0.814629`/`0.815215`), but row-control dose2 is strong at `0.847043`; a fixed 223-batch rerun lowers it to `0.842007` but it still beats KHR_10000 and KHR_50000 clean dose2 fixed-step (`0.817981`/`0.818783`) by about `+0.024`/`+0.023`, with `KHR_50000` still the worst row-control-relative class drop for the KHR_50000 dose. Class-duplicate controls narrow the blame: clean-single synthetic dose2 beats duplicated existing same-class exposure for `KHR_10000` (`+0.006871`) and `KHR_50000` (`+0.002929`) in full-epoch probes. Do not call any tiny synthetic dose promotable or defective without a matched row-count/class-exposure control or fixed-step/order-stable harness. Real fan/overlap transfer proof is still missing.

2.5D remains useful as a matched baseline and fallback, but WebGL synthetic is the main candidate path.

Label policy: exact visible masks/ID colors are authoritative. Detect AABBs are compatibility labels for detect-only probes; OBB labels are useful only when the visible mask is compact enough that a rotated rectangle does not cover large hidden regions. Fragment labels mean visible connected evidence components, not physical bill counts. If one physical bill is visible as disconnected islands, keep it one counted instance in mask metadata, skip/flag unsafe OBB, and use fragment labels plus downstream fusion rather than projecting a hidden-paper box. Barely visible fragments should get a denomination label only when a human can identify the denomination from visible evidence; otherwise use ignore/unknown instead of forcing a guess.

## Promising Next Moves

The best-supported path right now is a scoreable P1 transfer test:

1. Promote or capture rights-clean real fan/overlap stress labels, especially `KHR_5000` face+number overlaps and `KHR_20000` thin/edge backs.
2. Repair the model-side probe harness and real curriculum before more synthetic scale: keep p24/p48-style caps, but do not rely on duplicate-only rare oversampling or direct accepted-base comparisons for tiny synthetic doses. Every tiny-dose synthetic probe needs a matched row-count control, or better, a fixed-step/order-stable training harness so one extra row cannot change the causal question. The next useful data probe should add genuine rare-denom variation through curated real examples, weighted loss/sampling, or synthetic rare-heavy class exposure with a real validation bridge, and any expanded real-only control must beat the p24 gate without eroding `KHR_50000`. Whole-recipe accepted-blend deletion is exhausted; rare-support dose 2 fan and stack variants each have one positive seed and one seed-matched rejection, and lower-LR seed1 smoothing made the fan variant worse, so the next synthetic-side work needs a clearer rare-class curriculum/validation strategy rather than blind scale, simple LR tuning, or swapping fan for stack geometry.
3. Probe fragment-to-physical-note fusion once the evaluation set can expose whether fragment evidence helps or just overcounts.
4. Promote synthetic changes when they improve the real scoreboard without materially regressing clean/base validation or browser deploy guards.

Current evidence gap: P1 transfer has 0 promoted real fan/overlap stress labels. Public/raw candidates are useful diagnostics, but not scoreable substitutes.

## Synthetic Pipeline State

Goal: make synthetic data a controlled experiment generator, not just an image generator. A future failure should map to a missing condition, label issue, domain gap, training mix, or fusion bug instead of "we need better data."

Done and trusted:

- Real-target matrix and WebGL recipe catalog map required conditions, promotion gates, and blockers.
- Numista in-circulation cutout bank is the canonical clean asset source for now.
- WebGL packages separate physical parents from visible fragments, export exact ID masks, visible-only detect labels, guarded OBB labels, fragment labels, ignored-fragment metadata, preview overlays, contact indexes, quarantine metadata, and deterministic QA hashes.
- Smoke/trainable-candidate gates cover visual rejects, layer violations, selected train views, unsafe OBB exclusions, review-required fragments, count/parent-fusion contract consistency, and blank/exposure failures.
- Trainable-candidate gates enforce a high-resolution render floor (`1280x960` minimum, current accepted suite is `1440x1080` at `visual_scale: 1`) so quick low-res smokes cannot be accidentally promoted into model data.
- WebGL clean/stack/fan/hand-occlusion/thin-edge modes support `--class-sequence` for targeted weak-denomination smoke/dose tests; `webgl_rare_class_support_v1` renders a 20-image fan pool and packages an exact balanced 16-image physical-visible subset for rare-KHR distribution audits, but still needs real rare-class validation before trainable-candidate approval.
- The first bounded WebGL trainable-candidate suite is refreshed at 1440x1080 with `visual_scale: 1`, visually accepted, and train-smoke proven under headroom. Its active mix validates as 304 images / 1160 boxes; rejected probes stay cataloged but out of `cashsnap_webgl_trainable_candidates_v1`.
- Recipe-isolated WebGL ablation configs now exist under `configs/webgl_ablation/` with train lists under `configs/generated_lists/webgl_ablation/`; use these before scaling blended synthetic again.
- `compare_yolo_metrics.py --max-per-class-drop` now turns hidden denomination regressions into comparison failures; `run_webgl_recipe_ablation.py` passes a `0.05` per-class guard by default and summarizes worst-class deltas/failure counts. Accepted-blend selection keeps global mAP failure reasons separate from per-class drop failures and preserves the post-train blend gate when the selected recipe set is unchanged.
- `compare_yolo_metric_replicates.py` turns repeated model probes into promotion gates by checking candidate mean, worst seed, and worst per-class seed against a baseline; pass `--paired` for seed-matched baseline/candidate sweeps before calling any synthetic dose/curriculum change stable.
- `run_yolo_fixed_step_probe.py` is the generic two-config model probe wrapper for staged-dose work: it derives `--max-train-batches` from a reference train list, trains/evals baseline and candidate, then writes the JSON comparison and probe manifest.
- `check_webgl_class_distribution.py` gates targeted WebGL dose packages from `counts/summary.json`, so class-sequence probes can fail automatically on class leakage, missing expected classes, or excessive class imbalance.
- `check_webgl_count_stress.py` gates per-image count stress from `counts/targets.jsonl`, including same-class repeat images, max same-class count, split-parent counts, and fragment-overcount pressure.
- `probe_fragment_count_fusion.py` reports naive fragment overcount, oracle parent-fused counts, and same-class proposal-fused counts with JSON output and optional gates; use it before claiming fragment evidence can preserve physical-note counts.
- Browser CDP smoke summaries include and enforce a `countContract`: UI totals must equal final post-NMS detections, class totals must sum to final detections, and the app must report `post_nms_detector_proposals_with_fragment_class_override` / `final_nms_detections` as the count source.
- `build_browser_synthetic_stress_cases.py` regenerates a compact synthetic browser stress manifest from WebGL roots by selection policy; use it to test deploy behavior on clean, overlap, fan, repeated-same-class fan, hand occlusion, thin edge, and hard-negative configurations.
- Diagnostic recipe gates can live in `cashsnap_webgl_recipe_catalog_v1.json` and run via `check_webgl_recipe_diagnostic_gates.py`; rare-class and same-class-repeat recipes now declare their accepted audit thresholds.
- `check_webgl_appearance_diversity.py` is wired into the trainable-candidate gate so accepted packages must keep meaningful camera-profile, surface, lighting, luma, blur, grain/vignette, and non-geometric RGB postprocess spread instead of collapsing into one synthetic look.
- WebGL now gives each rendered bill deterministic individual condition metadata and visual/geometry treatment for pristine/circulated/heavily circulated/damp notes: dirt speckles, stains, edge wear, crease overlays, wet patches/sheen, and subtle crinkle displacement. `check_webgl_note_condition_diversity.py` is wired into the trainable-candidate gate with legacy tolerance, so newly rendered condition-aware packages must keep note-level wear spread.
- `--note-condition-policy` supports targeted condition curricula/audits: `mixed`, `pristine_only`, `heavy_wear`, and `wet_stress`; batch and catalog recipe wrappers forward it and record it in `recipe.json`. The catalog now has diagnostic wet/heavy/pristine condition recipes with policy-aware `note_condition_diversity` gates, and `check_webgl_note_condition_policy_delta.py` catches policy visual-delta collapse on paired same-variant packages.
- WebGL environment maps are review-gated: packaged renders with `--environment-dir` must pass `check_webgl_environment_banks.py` against the environment-bank registry, so unregistered or planned-only HDRI/LDR banks cannot slip into trainable-candidate data.
- `download_polyhaven_environment_bank.py` can reproduce the planned Poly Haven HDRI proof bank with a unique API User-Agent, md5/size verification, per-asset source/license/credit metadata, and a standalone proof-only config under ignored `data/environment_maps/`.
- `check_webgl_environment_review.py` verifies that an environment-bank review render covered every asset listed in the bank config and still has valid batch count metadata.
- `check_synthetic_pipeline_readiness.py` is the mission-level scale audit: it joins target-matrix conditions, recipe catalog rows, the active trainable-candidate suite, rendered package metadata, real benchmark roles, and real capture inventory so synthetic scale decisions fail on explicit condition/validation gaps instead of vibes.
- `audit_yolo_domain_gap.py` now has opt-in domain-gap gates for real-vs-synthetic image statistics, box geometry, and synthetic dose ratios; `build_webgl_accepted_blend_config.py` runs the `accepted_blend_v1` preset by default after building the accepted blend so bad blends fail before training spend.
- `build_webgl_accepted_blend_variant_configs.py` builds leave-one-out accepted-blend configs/lists and annotates each with domain-gate pass/fail so interaction probes are reproducible and not hand-edited; all four whole-recipe removals are now tested and rejected as the fix for the accepted blend's rare-class edge.
- `build_webgl_staged_rare_dose_configs.py` builds accepted-blend + rare-support staged dose configs/lists, supports `--stem-prefix` so alternate dose roots do not overwrite each other, and can write matched row-count controls with `--write-row-count-controls`. `--control-source class --control-class <name>` duplicates existing same-class base rows so probes can separate row-count, class-exposure, and synthetic-visual effects. Fan and stack dose 1/2 pass the accepted-blend domain-gap preset; dose 4/8/16 are written for audit context but are domain-flagged before training.
- `configs/cashsnap_v1_plus_webgl_accepted_nowarmup_probe.yaml` is the current bounded accepted-blend ablation: global clean-test transfer passes matched real-only by `+0.000994` mAP50-95, but its `KHR_2000` per-class drop blocks blind scale.
- `scripts/build_yolo_balanced_subset.py --always-max-per-class` can cap labeled always-included synthetic dose per class; this is useful tooling, but the capped full-real accepted seed is also rejected and should not be promoted.
- `configs/cashsnap_v1_full_real_only_seed.yaml` proves the all-target real curriculum is itself unsafe: 1 epoch from the clean checkpoint tests at `0.783482` mAP50-95 vs `0.883801` clean checkpoint, nearly identical to the capped accepted-synthetic failure.
- `configs/cashsnap_v1_real_only_p48_bg48_seed.yaml`, `configs/cashsnap_v1_real_only_p48_bg48_rare48_seed.yaml`, and `configs/cashsnap_v1_real_only_p96_bg96_seed.yaml` prove capped real-only expansion is safer than full all-target but still not enough: p48/bg48 tests at `0.833190`, duplicate rare p48/bg48/rare48 at `0.832952`, and p96/bg96 at `0.819153`, while the p24 real-only gate remains `0.835861`.

Still missing:

- Scoreable real fan/overlap labels for P1 transfer.
- Scoreable real repeated-denomination fan labels for same-class counting/fusion transfer.
- Real-derived camera/postprocess profiles from CashSnap captures and EXIF.
- Accepted HDRI/equirectangular environment-map bank for realistic lighting/reflection variation; WebGL can load and gate environment maps, but the default registry is still planned-only with no reviewed CC0 asset bank accepted yet.
- Exact shared geometric postprocess for lens distortion and true depth-dependent blur; current WebGL blur is label-safe RGB blur, and lens distortion is intentionally not applied until RGB, ID masks, and labels share the same transform.
- Promoted real stress labels proving the browser's post-NMS detector-proposal count with fragment class override transfers to repeated-fan/overlap/hand scenes; the browser smoke now gates count-source consistency, not real transfer.
- Accepted trainable real background banks.
- A `KHR_2000`/`KHR_50000`-safe accepted blend from staged dose/class shaping, or a justified per-class tolerance for rare-class test variance.
- A promotable real-only curriculum expansion that beats the p24 gate while preserving rare `KHR_50000`/`KHR_20000` transfer.
- Consistent fixed-step/order-stable model probes in promotion decisions; `run_yolo_fixed_step_probe.py` exists, but tiny-dose results still need matched row-count/class-exposure controls before interpretation.
- End-to-end promotion rules that combine real-scoreboard improvement, clean-validation/test guardrails, per-class metric guardrails, browser/deploy guardrails, and enough metadata to diagnose regressions.

## Operating Principle

Use this file as context, not a cage. Any agent should pursue the path that best advances reliable CashSnap counting, including changing harnesses, configs, scripts, or strategy when the current setup slows the goal or encodes a bad assumption.

Update this file only when something becomes durable working memory: a decision, a trusted/rejected data source, a result that changes what to run next, or a cleanup rule that prevents future wasted effort. Do not record every passing command or scratch experiment.

## Practical Constraints

This is a laptop workflow, so keep heavy CPU/RAM/GPU work behind a headroom wrapper when practical. If the wrapper or a harness gets in the way of a better experiment, improve the harness instead of treating it as doctrine.

Default repo posture: work on `master`/mainline, keep generated YOLO/training outputs under repo-local ignored `runs/`, and treat `results.tsv` only as deprecated scratch rather than project memory.

## Local Machine Profile

- Machine: Lenovo 82Y9 laptop.
- CPU: AMD Ryzen 5 7640HS, 6 cores / 12 logical processors, 4.3 GHz reported max clock.
- RAM: about 16 GB physical memory.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB VRAM, driver 596.21.
- Re-scan command: `rl python scripts\profile_system.py --out .cache_runtime\system_profile.json`.
- Current operating rule: laptop usability matters. Heavy jobs should prefer 90% max CPU/RAM/GPU/VRAM caps, resume around 82%, and never set caps above 95%.
- If this hardware is unexpectedly slow, suspect RAM pressure first, then laptop GPU power/thermal limits, then data-loader worker count.
- `scripts/run_with_headroom.py` now refuses caps above 95%, lowers child priority, and waits for initial headroom before launching a heavy child. The free-RAM floor is a preflight gate and runtime warning; after launch, hard pauses/exits follow the explicit RAM/VRAM max caps so pause-sensitive browser jobs do not trip wall-clock timeouts.
- `scripts/bench_train_with_headroom.py` dry-run currently selects `batch=4`, `workers=1`, `device=0` on this laptop and passes `--min-free-ram-gb 4.0` to the live wrapper.
- Balance speed and headroom. Prefer GPU for training/inference when it is the faster engine and has room, but do not force GPU for CPU-native prep/rendering if the headroom wrapper keeps the laptop responsive.

## Command Posture

Do not let this file become a command catalog. Keep only the few commands that express the current model decision; script discovery and older workflows belong in the code, CLI help, or archived notes.

Current useful checks:

- P1 readiness: `rl python scripts\check_webgl_p1_readiness.py --smoke-mix configs\cashsnap_webgl_trainable_candidates_mix.yaml`
- Mission-level synthetic scale readiness: `rl python scripts\check_synthetic_pipeline_readiness.py --check-existing --json-out runs\cashsnap\synthetic_pipeline_readiness_latest.json`
- Accepted-blend domain gap gate: `rl python scripts\audit_yolo_domain_gap.py --data configs\cashsnap_v1_plus_webgl_accepted_nowarmup_probe.yaml --split train --json-out runs\cashsnap\domain_gap_accepted_nowarmup_train.json --gate-preset accepted_blend_v1 --fail-on-gap`
- Accepted-blend leave-one-out variants: `rl python scripts\build_webgl_accepted_blend_variant_configs.py`
- Staged rare-support dose configs: `rl python scripts\build_webgl_staged_rare_dose_configs.py`
- Fixed-step two-config probe: `rl python scripts\run_yolo_fixed_step_probe.py --baseline-data <control.yaml> --candidate-data <candidate.yaml> --step-reference-data configs\cashsnap_v1_plus_webgl_accepted_nowarmup_probe.yaml --seed <n>`
- Replicated model gate: `rl python scripts\compare_yolo_metric_replicates.py --baseline <baseline_metrics.json> --candidate <seed0_metrics.json> --candidate <seed1_metrics.json> --min-mean-delta 0 --max-worst-drop 0 --max-per-class-drop 0.05`
- Paired seed gate: `rl python scripts\compare_yolo_metric_replicates.py --paired --baseline <baseline_seed0_metrics.json> --baseline <baseline_seed1_metrics.json> --candidate <candidate_seed0_metrics.json> --candidate <candidate_seed1_metrics.json> --min-mean-delta 0 --max-worst-drop 0 --max-per-class-drop 0.05`
- Trainable-candidate dry run: `rl python scripts\run_webgl_trainable_candidate_pipeline.py --dry-run --train-smoke`
- Targeted class-dose gate: `rl python scripts\check_webgl_class_distribution.py --root <webgl_root> --expected-classes '<CSV>' --min-images <n> --min-per-class <n> --max-class-spread <n>`
- Count-stress gate: `rl python scripts\check_webgl_count_stress.py --root <webgl_root> --min-repeat-images <n> --min-max-same-class <n>`
- Fragment/count fusion probe: `rl python scripts\probe_fragment_count_fusion.py --root <webgl_root> --json-out runs\cashsnap\<name>.json --require-proposal-fused-kept-match --require-proposal-fused-all-match`
- Recipe diagnostic gates: `rl python scripts\check_webgl_recipe_diagnostic_gates.py --root <webgl_root> --recipe-id <recipe_id>`
- Appearance diversity gate: `rl python scripts\check_webgl_appearance_diversity.py --root <webgl_root> --min-images <n>`
- Note-condition policy visual-delta audit: `rl python scripts\check_webgl_note_condition_policy_delta.py --base-root <control_root> --stress-root <stress_root> --min-images <n> --min-note-mean-abs <value>`
- Real benchmark boundary: `rl python scripts\check_real_fan_benchmark.py`
- Capture gaps: `rl python scripts\check_capture_requirements.py`
- Training wrapper dry run: `rl python scripts\bench_train_with_headroom.py --data configs\cashsnap_v1.yaml --name dry_run --dry-run --quiet`

If a different path is more promising, use or improve the relevant script instead of preserving this list.

## Result Ledger

Keep this table curated. Add rows only for results that change what a future agent should trust, avoid, or run next. Do not add a row for a passing command unless it changes a decision.

| Date Local | Area | Status | Result |
| --- | --- | --- | --- |
| 2026-05-30 | renderer | keep | P0/P1 renderer proofs are deterministic enough for experiments: RGB/ID passes, visible-only labels, metadata, OBB sidecars, and YOLO dataset checks all pass under headroom. |
| 2026-05-30 | labels | keep | Split-label QA proves the counting risk: `qa3` has 3 physical notes vs 5 fragments, and full-schema fan smoke has 67 physical visible instances vs 120 fragments. Fragment evidence must be fused back to physical parents before count totals; OBB is only honest for compact visible masks. |
| 2026-05-30 | data | keep | Numista cutout bank is the canonical clean asset source for now: 76 assets with full 13-class front/back coverage. It remains internal/reference until usage review is complete, and the audit still flags 3 red-mark visual suspects. |
| 2026-05-30 | targets | keep | `cashsnap_real_target_matrix_v1` and the WebGL recipe catalog map real target conditions to synthetic recipe slots, promotion gates, and blockers; use `check_synthetic_recipe_catalog.py` before changing coverage. |
| 2026-05-30 | packaging | keep | WebGL packages are auditable: `qa/summary.json`, `recipe.json`, preview overlays, visual+ID mask overlays, `qa/quarantine.json`, `qa/contact_index.json`, deterministic hashes, and label-view QA are all part of the contract. |
| 2026-05-30 | smoke | keep | Smoke-ready WebGL modes are clean, stack, fan, negative, thin-edge, and hand-occlusion. The refreshed gated smoke mix is diagnostic-only (19 images / 67 boxes), not a training-performance claim. |
| 2026-05-31 | evaluation | blocked | `real_overlap_0003_commons_shop_5k_10k_20k` is promoted as a 6-box visible-denomination/mild-overlap sanity label by Codex visual audit. P1 transfer is still blocked by 0 promoted real fan/overlap stress labels. |
| 2026-05-31 | training | note | The smoke tiny train is only a harness proof: after YAML root normalization, 1 epoch under headroom completed with diagnostic mAP50-95 around `0.002`. Do not read it as model quality. |
| 2026-05-31 | training | note | The 304-image WebGL trainable-candidate mix runs through YOLO training under headroom. Batch 2 hit the 90% RAM cap, the wrapper relaunched at batch 1/pause-resume, and the diagnostic 1-epoch smoke ended around mAP50-95 `0.000385`; this is trainability evidence only. |
| 2026-05-31 | gates | keep | Trainable-candidate gates pass for all 7 roots after the 1440x1080 `visual_scale: 1` refresh; the gated mix validates as 304 images / 1160 boxes, package QA verifies physical count targets separately from fragment evidence, and the sampled 84-row visual audit pack is accepted. Full P1 remains blocked by 0 promoted real stress labels. |
| 2026-05-31 | policy | keep | Current trainable policy includes deterministic visual QA, fragment review/ignore metadata, geometric transform guards, explicit asset-side policy, explicit phone camera profiles, and background-bank review gates. No trainable background bank is accepted yet. |
| 2026-05-31 | backgrounds | reject | Existing extracted no-note background banks are rejected by Codex contact-sheet audit; a stricter reject-model probe yielded only two crops and one was outdoor/cushion context, so do not retry that path blindly without curated table-source photos. |
| 2026-05-31 | rendering | adjust | Full-suite refresh at 1440x1080 hit the 3 GB free-RAM launch gate during supersampled WebGL renders; trainable-candidate defaults now use `visual_scale: 1` to keep output resolution while avoiding high system-RAM canvas overhead on the laptop. |
| 2026-05-31 | operations | keep | Headless Edge is already GPU-backed on this laptop (`ANGLE` on RTX 4060/D3D11). Suite rows own `visual_scale`; global `--visual-scale` is only an explicit override. Hard-negative mode uses primitive non-banknote props so zero-box frames are not blank tables. |
| 2026-05-31 | real data | blocked | Existing real/raw stress candidates do not unblock P1: `real_fan_0001_voa_commons` is too ambiguous to promote from model hints, `real_overlap_0002_commons_museum` is not a fan/dense-overlap stress substitute, and `data/review/roboflow_cuurecy_detection_is_multinote_probe_v1/contact_sheet.jpg` shows useful Roboflow hand/fan stress references but not scoreable labels because of public/reproduction/split and visible-region policy caveats. Next unblocker is rights-clean phone fan/overlap capture(s) or human-audited stress labels. |

| 2026-06-04 | training | reject | Do not promote the blended accepted-WebGL fine-tunes (`webgl_candidate_balanced_probe_from_clean_e2_i416_noamp`, `webgl_candidate_balanced_low_lr_from_clean_e1_i416_noamp_v2`): the 621-image balanced real+accepted-WebGL subset trained cleanly and produced val mAP50-95 around `0.807`, but held-out clean test was `0.810` versus `0.884` for the prior clean checkpoint on the same split (`compare_yolo_metrics.py` delta `-0.073706`). Synthetic training must pass baseline comparison gates before scaling. |
| 2026-06-04 | evaluation | keep | `val_yolo.py --metrics-json` plus `compare_yolo_metrics.py` is the active comparison path for candidate checkpoints; use JSON all/per-class metrics and max-drop/min-delta/per-class-drop gates instead of scraping terminal logs. |
| 2026-06-04 | operations | note | Laptop RAM headroom is enough after foreground cleanup, but Ultralytics `optimizer=auto` can ignore requested `lr0` (it selected AdamW around `0.000588` despite `--lr0 0.00025`). For true low-LR probes, pass an explicit optimizer such as `--optimizer AdamW`; still gate by held-out JSON comparison. |
| 2026-06-04 | pipeline | keep | `build_webgl_recipe_ablation_configs.py` writes one balanced real-only config plus one balanced real+single-WebGL-recipe config per trainable recipe; all 8 generated YAMLs passed `check_yolo_dataset.py`, and the fan ablation config reached YOLO training end-to-end under headroom. Use recipe ablations before another blended synthetic run. |
| 2026-06-04 | smoke | note | `--max-train-batches` sets Ultralytics `trainer.stop`, which still triggers validation and final eval even when `--no-val` is passed. For cheap train-path smoke, prefer tiny list-backed smoke YAMLs over early-stopping a full-val config. |
| 2026-06-04 | synthetic | reject | `webgl_clean_closeup_v1` and `webgl_clean_closeup_v2` both failed strict matched real-only ablations despite valid labels. V1 scored `0.823222` mAP50-95 vs real-only `0.835861` (delta `-0.012639`); v2 improved global geometry but scored `0.821181` (delta `-0.014680`), worst on `KHR_2000`/`KHR_50000`. They are `rejected_probe` catalog entries and must stay out of active trainable mixes until background/geometry repair changes the evidence. |
| 2026-06-04 | synthetic | provisional | The 445-image accepted blend (`clean_base`, `thin_edge`, `hard_negative`, `back_side`) is the best current scale seed: held-out clean test `0.836855` vs matched real-only `0.835861` (delta `+0.000994`). It fails the per-class guard on `KHR_2000` (`-0.052433`, threshold `-0.05`), so scale only with `KHR_2000`/`KHR_50000` watch metrics or staged dose tests. |
| 2026-06-04 | synthetic | reject | The 413-image no-thin-edge core blend is not safer: held-out clean test `0.832365` vs real-only `0.835861` (delta `-0.003496`), with worse `KHR_50000` (`-0.059880`) and `KHR_2000` (`-0.050304`) deltas. Do not prefer it over the full accepted blend. |
| 2026-06-04 | training | reject | Full clean-real + uncapped accepted synthetic is a hard reject. The 5-epoch low-LR/no-AMP seed had best val mAP50-95 at epoch 1 (`0.84559`), but held-out clean test was `0.807251` vs `0.883801` clean checkpoint (delta `-0.076550`), with rare-class collapse on `KHR_50000` (`-0.587183`) and `KHR_20000` (`-0.176160`). Do not promote. |
| 2026-06-04 | training | reject | Capping accepted synthetic with `--always-max-per-class 4` does not fix the full-real seed. The 1-epoch capped probe tested at `0.785412` vs `0.883801` clean checkpoint (delta `-0.098389`), with `KHR_50000` (`-0.690631`) and `KHR_20000` (`-0.296286`) worse. The cap is useful sampler tooling, but the next step is isolating full-real real-only/background-heavy curriculum before more synthetic scale. |
| 2026-06-04 | training | reject | Full clean-real real-only all-target seed is also a hard reject. The 1-epoch low-LR/no-AMP control validates at `0.83542`, but held-out clean test is `0.783482` vs `0.883801` clean checkpoint (delta `-0.100319`) and `-0.001930` below the capped accepted-synthetic seed. This implicates the all-target/background-heavy real curriculum and rare-class exposure shape as the primary collapse; do not use full-real all-target seeds until the real-only curriculum is repaired. |
| 2026-06-05 | training | reject | Real-only p96/bg96 is safer than full all-target but not promotable. The 1181-image 1-epoch low-LR/no-AMP control validates at `0.82894` and tests at `0.819153`: `+0.035671` over the full-real all-target reject, but `-0.064648` vs clean and `-0.016708` vs the p24 real-only gate. `KHR_50000` remains the main damage (`0.485681` mAP50-95, `-0.429641` vs clean). |
| 2026-06-05 | training | reject | Real-only p48/bg48 is the best expanded real-only control so far but still misses promotion. The 605-image 1-epoch low-LR/no-AMP control validates at `0.80733` and tests at `0.833190`: `+0.049708` over the full-real all-target reject and `+0.014037` over p96, but `-0.050611` vs clean and `-0.002671` vs the p24 real-only gate. Next repair is richer rare variation or weighting, not larger common-class/background caps. |
| 2026-06-05 | training | reject | Duplicate-only rare oversampling does not promote p48/bg48. `configs/cashsnap_v1_real_only_p48_bg48_rare48_seed.yaml` repeats selected rare images until every class has at least 48 train-list appearances (`+67` duplicates, triggered only by `KHR_20000`/`KHR_50000`), validates at `0.82772`, and tests at `0.832952`: `-0.000237` vs plain p48 and `-0.002909` vs the p24 gate. It helps several classes but drops `KHR_50000` by `-0.110185` vs plain p48, so duplicate exposure alone is rejected; use richer rare variation or weighting next. |
| 2026-06-05 | evaluation | keep | Global-only comparison is now hardened with opt-in per-class guardrails. `compare_yolo_metrics.py --max-per-class-drop 0.05` correctly fails the accepted blend despite global `+0.000994` vs real-only, flagging `KHR_2000` at `-0.052433`; `run_webgl_recipe_ablation.py` applies this guard by default and records worst-class/failure counts in summaries. `build_webgl_accepted_blend_config.py --dry-run` still selects the same four accepted recipes, while rejected reasons now distinguish global mAP failures from per-class drop failures. |
| 2026-06-05 | synthetic | keep | Generic WebGL clean/stack/fan paths now accept a `--class-sequence` override, threaded through `render_webgl_variant_batch.py`, `run_webgl_recipe.py`, suite runners, mix metadata, and recipe metadata. A tiny `webgl_rare_class_support_v1` fan smoke rendered and passed checks: direct 2-image pack had 11 boxes only in `KHR_2000`/`KHR_50000`/`KHR_20000`/`KHR_10000`/`KHR_5000`, and catalog-driven 1-image pack had one box per requested class plus valid label views. Use this for rare-denom diagnostic dose tests, not blind scale. |
| 2026-06-05 | synthetic | keep | `webgl_rare_class_support_v1` now uses visible-count subset packaging: a 20-image fan pool had physical visible rare-KHR counts spread 18-22, then exact subset search packaged 16 images with 80 targets and perfect class balance (`KHR_2000`/`KHR_5000`/`KHR_10000`/`KHR_20000`/`KHR_50000`: 16 each). Label-view QA passed with 113 kept fragments and all 16 OBB views rejected as unsafe, which is correct for heavily fragmented fan masks. This proves targeted rare-KHR dose packaging, not real-transfer safety. |
| 2026-06-05 | synthetic | keep | `webgl_fan_fullschema_v1` has strong fan/fragment pressure but no same-denomination repeats (`max_same_class_per_image=1` across 64 images). `webgl_same_class_repeat_fan_v1` now renders a 20-image pool and packages a balanced 12-image subset because a direct contiguous 12-image rerender can miss the class-count gate. The balanced audit passed with 60 physical targets, 100 kept fragments, all 12 images containing same-class repeats, max same-class count 3, exact targeted class balance (`KHR_5000`/`KHR_10000`/`KHR_20000`/`KHR_50000`: 15 each), and parent-fused all-fragment counts matching physical targets. Keep diagnostic until real repeat-fan validation. |
| 2026-06-05 | synthetic | keep | `--class-sequence` now applies to hand-occlusion and thin-edge modes, not just clean/stack/fan. Tiny smokes passed: hand-occlusion forced `KHR_50000` only (1 image / 4 physical targets / 9 kept fragments), and thin-edge forced `KHR_50000` + `KHR_2000` (1 image / 3 physical targets). Use this for targeted weak-denom occlusion/sliver diagnostics; still gate real transfer separately. |
| 2026-06-05 | synthetic | keep | Trainable-candidate packages now run `check_webgl_appearance_diversity.py`, which gates camera-profile, surface, luma, blur, grain, vignette, saturation, and brightness spread from package metadata. Existing 7-root trainable-candidate suite passes through `check_webgl_trainable_candidate_suite.py --check-existing` with 304 images, so this catches collapsed synthetic appearance without invalidating current accepted roots. |
| 2026-06-05 | synthetic | keep | `check_synthetic_pipeline_readiness.py --check-existing` gives the step-back scale audit: all 9 required target conditions have active trainable-candidate suite coverage, but the system is not ready for synthetic scale because only clean baseline is unblocked; real role labels are ready for 1/5 role-gated conditions, usable capture inventory is 0, and remaining blockers are explicit per condition. |
| 2026-06-05 | synthetic | keep | `audit_yolo_domain_gap.py` can now fail real-vs-synthetic domain gaps with named presets or explicit thresholds. With `--gate-preset accepted_blend_v1`, the 445-image accepted blend passes, while the full 621-image real+all-trainable-candidate probe fails in report-only mode on global synthetic dose and per-class synthetic/real box ratios; this catches synthetic dose drift before another expensive training run. |
| 2026-06-05 | validation | keep | Real capture requirements now include explicit `same_denomination_fan` scenes. `check_synthetic_pipeline_readiness.py` maps `repeated_same_denomination` to that bucket instead of generic `hand_fan`, so same-class count/fusion validation cannot be accidentally satisfied by ordinary fan photos. |
| 2026-06-05 | synthetic | keep | `build_webgl_accepted_blend_config.py` now runs `audit_yolo_domain_gap.py --gate-preset accepted_blend_v1 --fail-on-gap` after writing the accepted blend and annotates `cashsnap_domain_gap_gate` on success. The current accepted blend was rebuilt and passed, so future accepted-blend generation cannot skip the domain/dose guard by accident. |
| 2026-06-05 | evaluation | note | Existing recipe-isolated per-class metrics do not identify one bad selected recipe for the accepted-blend `KHR_2000` failure: `clean_base`, `thin_edge`, `hard_negative`, and `back_side` each improve `KHR_2000` versus matched real-only individually. Treat the blend failure as an interaction/dose problem; whole-recipe deletion is now tested, so the next bounded model probe should be staged-dose or class-shaped accepted blends. |
| 2026-06-05 | evaluation | keep | `build_webgl_accepted_blend_variant_configs.py` generated four leave-one-out accepted-blend configs under `configs/webgl_blend_variants/` with train lists under `configs/generated_lists/webgl_blend_variants/`; all passed YOLO dataset checks. Domain-gate status is a training priority signal: minus `hard_negative` passes, while minus `clean_base`, minus `thin_edge`, and minus `back_side` are valid but domain-flagged. |
| 2026-06-05 | evaluation | reject | Completed the accepted-blend leave-one-out model probes; none beat the full accepted blend or matched p24 real-only. Full blend remains best global (`0.836855`, `+0.000994` vs p24 real-only) but misses `KHR_2000` (`-0.052433`); minus `thin_edge` is closest (`0.832365`, `-0.003496`, `KHR_2000 -0.050304`, `KHR_50000 -0.059880`), while minus `clean_base`/`hard_negative`/`back_side` score `0.813584`/`0.813922`/`0.810394` and damage `KHR_50000` by about `-0.22`. Do not spend more on whole-recipe removal; next synthetic probe is staged dose/class shaping. |
| 2026-06-05 | synthetic | keep | `build_webgl_staged_rare_dose_configs.py` generates rare-support staged dose configs from the accepted blend. Dose 1 and dose 2 pass `accepted_blend_v1` domain-gap gates; dose 4 first fails (`KHR_20000` synthetic/real box ratio `3.0588 > 3.0`), dose 8 worsens that ratio, and dose 16 also exceeds the global synthetic/real box-ratio cap. Do not train dose 4+ until the dose/domain shape is repaired. |
| 2026-06-05 | evaluation | reject | Rare-support staged dose 2 is seed-fragile, not promotable. Seed 0 passed (`0.838473`, `+0.002612` vs p24 seed0 real-only, no per-class failures, worst `KHR_50000 -0.043206`), but seed 1 rejected against p24 seed1 real-only (`0.826076`, `-0.018658`, worst `KHR_50000 -0.218713`, two per-class failures). Dose 1 is also rejected (`0.825533`, `-0.010328` vs p24 seed0, `KHR_50000 -0.143900`). Keep the class-balanced rare-support package as diagnostic signal, but do not scale it until the seed sensitivity is understood or real-stress transfer validates it. |
| 2026-06-05 | evaluation | keep | `compare_yolo_metric_replicates.py` adds a multi-run gate for model-side synthetic promotion. Baseline-mean mode rejects the two fan dose2 seeds (`mean 0.832275`, `-0.003586` vs p24 seed0; worst seed `0.826076`, `-0.009785`; worst per-class `KHR_50000 -0.158238`), and `--paired` supports seed-matched controls for replicated gates. |
| 2026-06-05 | training | reject | Lowering rare-support dose2 seed1 from `lr0=0.00005` to `0.000025` does not stabilize it. Train-time val stayed weak around `0.826`, held-out test fell to `0.821231` (`-0.014630` vs p24 real-only), and `KHR_50000` worsened to `-0.201808`; do not spend more on simple LR smoothing for this dose shape. |
| 2026-06-05 | synthetic | keep | Stack-mode rare-support audit package `data/synthetic/cashsnap_webgl_rare_class_support_stack_audit_v1` rendered and packaged cleanly: 16 selected 960x720 images, 74 physical targets, class counts `14`-`15`, 96 kept fragments, 2 trainable OBB images, and the explicit class gate passed (`spread=1`, ratio `1.071`). `build_webgl_staged_rare_dose_configs.py --stem-prefix` generated separate stack dose configs; dose1/2 pass `accepted_blend_v1`, while dose4+ fail on `KHR_20000` synthetic/real ratio like the fan dose. |
| 2026-06-05 | evaluation | reject | Stack rare-support dose2 does not fix seed fragility. Seed0 passes (`0.837617`, `+0.001756` vs p24 seed0, worst `KHR_50000 -0.049608` just inside the per-class guard), but seed1 rejects against p24 seed1 (`0.826476`, `-0.018259`, worst `KHR_50000 -0.217202`, two per-class failures); the paired replicate gate rejects the pair (`mean_delta -0.008251`). The issue is not only hard fan fragmentation. |
| 2026-06-05 | evaluation | keep | The p24 real-only seed1 control is strong on held-out test (`0.844735`, `+0.008874` vs p24 seed0, per-class guard passes) even though train-time val looked weak around `0.819`. Use held-out metrics JSON, not train-time val, for promotion gates, and compare synthetic seed1 to this seed1 baseline. |
| 2026-06-05 | evaluation | reject | Paired replicate gates confirm the rare-support dose2 rejection is not a single-baseline artifact. Fan dose2 pair deltas are `+0.002612` and `-0.018658` (`mean_delta -0.008023`, worst per-class `KHR_50000 -0.218713`); stack dose2 pair deltas are `+0.001756` and `-0.018259` (`mean_delta -0.008251`, worst per-class `KHR_50000 -0.217202`). |
| 2026-06-05 | evaluation | reject | Accepted-blend seed1 removes the old single-seed comfort. Seed1 tests at `0.833769`; paired with seed0, the accepted blend rejects against p24 real-only (`baseline_mean 0.840298`, `candidate_mean 0.835312`, `mean_delta -0.004986`, worst `KHR_2000 -0.141446`). Treat accepted blend as a useful diagnostic base, not a stable promoted foundation. |
| 2026-06-05 | synthetic | keep | Clean-single WebGL audit roots for `KHR_50000` and `KHR_10000` each passed visual QA, class distribution, label-view, OBB trainability, appearance-diversity, YOLO dataset, and `accepted_blend_v1` domain-gap gates. Configs under `configs/webgl_staged_dose/` and train lists under `configs/generated_lists/webgl_staged_dose/` preserve KHR_50000 doses `1/2/4/8/12`, KHR_10000 doses `1/2`, and matched row-count controls `1/2/4/8/12`. These are diagnostic model-probe inputs, not promoted training data. |
| 2026-06-05 | evaluation | keep | Matched row-count controls expose the tiny-dose probe confound. Row-control dose1 collapses to `0.809414`; KHR_50000/KHR_10000 clean dose1 score `0.814629`/`0.815215` and pass versus that matched control, but still fail accepted seed1. Row-control dose2 is strong at `0.847043` (`+0.013274` vs accepted seed1), while KHR_50000/KHR_10000 clean dose2 score `0.825640`/`0.825716` and both reject against the matched control by about `-0.0213`, with `KHR_50000` the worst drop. Tiny synthetic doses now require matched row-count controls or a fixed-step/order-stable harness before interpretation. |
| 2026-06-05 | evaluation | keep | Class-duplicate controls separate same-class exposure from new synthetic visuals. KHR_10000 duplicate-control dose2 tests at `0.818846` and KHR_50000 duplicate-control dose2 at `0.822711`; both fail the neutral row-control dose2, but clean-single synthetic dose2 beats the same-class duplicate control for KHR_10000 (`0.825716`, `+0.006871`) and KHR_50000 (`0.825640`, `+0.002929`) with no per-class guard failure. This shifts blame toward class-exposure/curriculum/order effects, not uniquely bad clean synthetic note images. |
| 2026-06-05 | evaluation | keep | Fixed-step dose2 probe (`--max-train-batches 223`, matching the 445-row accepted base at batch 2) confirms one extra optimizer batch is not the whole dose2 confound. Neutral row-control dose2 drops from `0.847043` to `0.842007` (`-0.005035`, `KHR_50000 -0.090657` vs full epoch), but fixed-step KHR_10000 clean dose2 drops to `0.817981` and still fails the fixed-step neutral control by `-0.024026` with three per-class failures. Future tiny-dose model probes should use both matched row/class controls and explicit fixed-step budgets. |
| 2026-06-05 | evaluation | keep | KHR_50000 clean dose2 fixed-step repeats the KHR_10000 conclusion. With the same 223-batch cap it tests at `0.818783`, fails fixed-step neutral row-control by `-0.023224` (worst `KHR_50000 -0.218492`, three per-class failures), and is `-0.006857` below its own full-epoch result (`0.825640`). Fixed-step removes only a small row-count effect; the remaining gap points to curriculum/order/class interaction rather than a uniquely bad synthetic image row. |
| 2026-06-05 | gates | keep | `check_webgl_trainable_candidate_gate.py` now requires every package to carry a valid count contract: `counts/summary.json` image totals must match QA, `counts/targets.jsonl` must have one row per image, physical totals must match QA visible instances, and `parent_fused_all_fragments` must match physical counts. The existing 7-root trainable-candidate suite passes with 304 images and parent-fused counts matching physical counts; this hardens synthetic count truth but does not replace the still-missing real inference fusion path. |
| 2026-06-05 | fusion | keep | `probe_fragment_count_fusion.py` now separates naive fragment counting, oracle parent fusion, and same-class proposal fusion. On `webgl_same_class_repeat_fan_v1` balanced audit, naive kept fragments overcount by `+40` and kept+ignored fragments by `+122`, while parent fusion and same-class proposal fusion both recover the 60 physical targets exactly with zero unassigned fragments. This validates the synthetic benchmark and suggests the inference path should count detector proposals with fragment evidence, not raw fragments. |
| 2026-06-05 | deploy/fusion | keep | Browser smoke now makes that inference count source executable: the app reports `post_nms_detector_proposals_with_fragment_class_override` / `final_nms_detections`, and `smoke_browser_demo_cdp.cjs` fails if UI totals, class totals, debug final counts, or proposal-count bounds drift. The waiter now accepts autorun completion with zero detections, and the 6-case browser smoke passes with a synthetic hard-negative zero-note guard. This is deploy consistency, not the missing real repeated-fan transfer proof. |
| 2026-06-05 | synthetic-test | keep | `manifests/browser_synthetic_stress_cases.csv` is generated by `build_browser_synthetic_stress_cases.py` from WebGL roots. Browser stress run `runs/cashsnap/browser_synthetic_stress_cases_v1.json` exposes deploy weak spots: clean count is correct but USD notes are overwritten by KHR, overlap/hand undercount by 2, fan undercounts by 6, same-class fan undercounts by 4, thin-edge detects 0/4, and hard-negative stays 0/0. This is a synthetic failure surface for training/curriculum probes, not real transfer proof. |
| 2026-06-05 | gates | keep | The WebGL appearance-diversity gate now also measures focus-blur range, key-light color count, and hemi/key intensity ranges. The 7-root trainable-candidate suite still passes with 304 images; current WebGL nuisance spread is strong for label-safe camera/lighting/postprocess variation, while exact lens distortion and true depth blur remain future renderer work. |
| 2026-06-05 | renderer | keep | WebGL now accepts `--environment-dir` for `.hdr`/LDR equirectangular environment maps, uses them only in the visual pass for `scene.environment` lighting/reflections, disables them in the ID pass, and records `sceneConfig.environment`. Direct renderer smoke and one-image batch smoke passed with a generated LDR environment map; this proves plumbing, not an accepted HDRI asset bank. |
| 2026-06-05 | gates | keep | `check_webgl_environment_banks.py` and `cashsnap_webgl_environment_banks_v1.json` gate WebGL environment-map use by review status, source/license metadata, and artifact status. The default Poly Haven bank is planned-only, unregistered env dirs reject, a generated proof-only smoke bank passed, and a one-image gated batch smoke proved the wrapper calls the bank check before rendering. |
| 2026-06-05 | assets | keep | `download_polyhaven_environment_bank.py --review-status proof_only` downloaded the 8-HDRI `cashsnap_retail_lighting_v1` preset from Poly Haven at 1k (12.6 MB) with md5/size verification and wrote an ignored standalone proof config. Smoke use passes the environment-bank checker, trainable-candidate use rejects as intended, and a WebGL batch smoke with the downloaded HDR bank passes label/count checks. The renderer now uses Three `HDRLoader` instead of deprecated `RGBELoader`. |
| 2026-06-05 | assets | keep | The downloaded Poly Haven proof bank has an 8-image review render at `data/synthetic/cashsnap_webgl_polyhaven_environment_review_v1`: one stack scene per HDRI, 34 physical count targets, valid label views, and visual review showed usable luma/color variation without leaking environment imagery into the background. `check_webgl_environment_review.py` wrote `runs/cashsnap/polyhaven_environment_review_v1.json` and confirmed all 8 bank assets were covered. Keep it proof-only until an accepted bank config is deliberately generated/used in the local environment. |
| 2026-06-05 | renderer | keep | Per-bill condition randomization is active in WebGL: every note gets `condition` metadata for dirtiness, crinkle, wetness, edge wear, stains, speckles, and crease shape; the visual pass applies masked dirt/wet/crease overlays and lowers roughness for damp sheen, while geometry gets subtle wrinkles/creases before the ID pass so labels stay honest. An 8-image stack package passed note-condition diversity (`36` notes: `3` pristine, `18` circulated, `10` heavily circulated, `5` damp; wetness range `0.76`), trainable-candidate gate, and existing legacy suite. A 1440x1080 HQ smoke passed ID QA and visual review for sharp edges/no old valley-texture regression. |
| 2026-06-05 | gates | keep | `check_webgl_trainable_candidate_gate.py` now rejects trainable-candidate packages below `1280x960` or `visual_scale < 1`, while still allowing smaller smoke/diagnostic artifacts when explicitly allowed. Existing 7-root trainable-candidate suite passes; the condition-aware 640x480 proof remains smoke-only. |
| 2026-06-05 | renderer | keep | High-resolution condition audit `data/synthetic/cashsnap_webgl_note_condition_hq_audit_v1` rendered 16 stack images at `1440x1080`, `visual_scale: 1`, and passed the diagnostic trainable gate: 71 physical targets, 99 fragments, appearance diversity, count contract, and note-condition diversity (`72` notes: `3` pristine, `34` circulated, `29` heavily circulated, `6` damp; wet notes `11`; wetness range `0.76`). Contact sheet visual review found no obvious low-quality edge or valley-texture regression. This proves condition realism works at trainable resolution, not just low-res smoke, but it is still diagnostic until real transfer validates it. |
| 2026-06-05 | renderer | keep | Per-note condition policies are now controllable instead of purely stochastic. `wet_stress` rendered an 8-image diagnostic smoke (`data/synthetic/cashsnap_webgl_wet_stress_policy_smoke_v1`) that passed label QA, appearance diversity, count contract, and policy-aware condition diversity: 36/36 notes wet, 30 damp profiles, wetness range `0.56`. `heavy_wear` and `pristine_only` 2-image smokes also passed condition gates. Use these as targeted stress/curriculum knobs; trainability still needs high-res render plus real transfer validation. |
| 2026-06-05 | gates | keep | Condition policies are now catalog-driven diagnostics: `webgl_wet_note_condition_stress_v1`, `webgl_heavy_wear_note_condition_stress_v1`, and `webgl_pristine_note_condition_control_v1` render through `run_webgl_recipe.py --run-diagnostic-gates` with policy-aware note-condition gates. Catalog proofs passed at 1440x1080/`visual_scale: 1`: pristine control 8/8 pristine notes, heavy-wear 36 dirty notes with dirtiness range `0.42` and crinkle range `0.61`, and strengthened wet-stress 36/36 wet notes with wetness range `0.56`. Paired wet-vs-heavy visual-delta audit passed on note-ID pixels (`mean_note_abs=11.91`, min image `6.82`), so wet stress is not metadata-only. Still diagnostic until real damp/worn-note transfer exists. |
| 2026-06-05 | research | keep | Paper pass for best dataset construction supports the current proof-harness direction: domain randomization should cover target nuisance variables, structured randomization should sample plausible scene/context distributions, copy-paste/synthetic augmentation can help rare classes but must be guarded for context/artifacts, synthetic-real mixes beat blind replacement in object detection studies, and curriculum/order matters. Translate this into acceptance criteria: every synthetic scale step needs condition coverage, exact-label QA, domain/dose gates, matched row/class controls, and real-only stress validation. |

## Trainable Candidate Artifacts

Generated artifacts live under ignored `data/synthetic/` roots and should not be committed unless explicitly requested. Update this table only when a candidate root becomes trusted, rejected, or its intended train view changes.

| Recipe | Root | Train views | Gate status | Use / caveat |
| --- | --- | --- | --- | --- |
| `webgl_clean_base_v1` | `data/synthetic/cashsnap_webgl_clean_base_candidate_v1` | detect, fragment, OBB | Passed: 32 accepted images, 65 detect/fragment boxes, 32/32 trainable OBB images, 0 visual rejects, `phone_auto` spread across all 4 named profiles. | Clean synthetic baseline candidate. |
| `webgl_clean_closeup_v1` | `data/synthetic/cashsnap_webgl_clean_closeup_candidate_v1` | detect, OBB | Rejected by strict real-only ablation: `0.823222` mAP50-95 vs real-only `0.835861` (delta `-0.012639`). | Diagnostic only and excluded from the active trainable suite; broad KHR rotations made boxes too square/large, led by `KHR_50000`. |
| `webgl_clean_closeup_v2` | `data/synthetic/cashsnap_webgl_clean_closeup_candidate_v2` | detect, OBB | Rejected by strict real-only ablation: `0.821181` mAP50-95 vs real-only `0.835861` (delta `-0.014680`). | Diagnostic only and excluded from the active trainable suite; geometry improved globally, but `KHR_2000`/`KHR_50000` still regress. |
| `webgl_overlap_stack_v1` | `data/synthetic/cashsnap_webgl_overlap_stack_candidate_v1` | detect, fragment | Passed at 1440x1080 `visual_scale: 1`: 64 accepted images, 283 detect boxes, 373 trainable fragments, 120 ignored fragments, OBB mostly rejected (8 accepted / 56 rejected). | Dense-overlap and fragment-fusion stress; not an OBB training source. |
| `webgl_fan_fullschema_v1` | `data/synthetic/cashsnap_webgl_fan_fullschema_candidate_v1` | detect | Passed at 1440x1080 `visual_scale: 1`: 64 accepted images, 349 detect boxes, 576 fragment components in metadata, 376 ignored fragments, 83 review-required fragments, OBB rejected on all 64 images. Count-stress audit shows no same-denomination repeat images (`max_same_class_per_image=1`). | Handheld-fan detect candidate; fragment labels are diagnostic until fan fragment/fusion policy improves, and same-class repeat counting belongs to `webgl_same_class_repeat_fan_v1`. |
| `webgl_hand_occlusion_fragments_v1` | `data/synthetic/cashsnap_webgl_hand_occlusion_candidate_v1` | detect | Passed at 1440x1080 `visual_scale: 1`: 48 accepted images, 216 detect boxes, 402 fragment components in metadata, 183 ignored fragments, 31 review-required fragments, OBB mostly rejected (1 accepted / 47 rejected). | Hand/finger occlusion detect candidate; class-skewed to 5 denominations, so do not use as balanced hand coverage. |
| `webgl_thin_edge_partial_v1` | `data/synthetic/cashsnap_webgl_thin_edge_partial_candidate_v1` | detect | Passed at 1440x1080 `visual_scale: 1`: 32 accepted images, 105 detect boxes, 121 fragment components in metadata, 183 ignored fragments, 17 review-required fragments, OBB mostly rejected (1 accepted / 31 rejected). | Thin-edge KHR sliver candidate; individually passes clean guard and improves the full accepted blend globally, but scale with `KHR_2000`/`KHR_50000` watch metrics because the accepted blend fails the `KHR_2000` per-class gate. |
| `webgl_hard_negative_replay_v1` | `data/synthetic/cashsnap_webgl_hard_negative_candidate_v1` | detect | Passed at `--visual-scale 1`: 32 accepted zero-box images, 0 visible banknotes, 0 fragments, 0 visual rejects. | False-positive guardrail candidate with primitive non-banknote props; still not a substitute for a reviewed real/background prop library. |
| `webgl_back_side_confusion_v1` | `data/synthetic/cashsnap_webgl_back_side_confusion_candidate_v1` | detect | Passed at 1440x1080 `visual_scale: 1`: 32 accepted images, 142 detect boxes, 192 fragment components in metadata, 45 ignored fragments, 5 review-required fragments, OBB mostly rejected (5 accepted / 27 rejected). | Balanced front/back stack candidate; `front_back_mix` satisfied on all 32 images. |
| `webgl_rare_class_support_v1` | `data/synthetic/cashsnap_webgl_rare_class_support_audit_v1` | detect, fragment diagnostic | Diagnostic distribution audit: 16 fan images at 960x720 `visual_scale: 1` produced 80 physical targets only in targeted KHR support classes, exactly 16 per class; label-view QA passed with 113 kept fragments, and OBB rejected all 16 images as unsafe for fragmented fan masks. | Proves class-sequence rare-KHR dose control works. Not a trainable candidate until real rare-class validation passes. |
| `webgl_same_class_repeat_fan_v1` | `data/synthetic/cashsnap_webgl_same_class_repeat_fan_balanced_audit_v1` | detect, fragment diagnostic | Diagnostic count-stress audit: 20 rendered fan images at 960x720 `visual_scale: 1`, balanced subset packaged 12 images with 60 physical targets, 100 kept fragments, all 12 images with same-class repeats, max same-class count 3, exact targeted class counts of 15 each, and parent-fused all-fragment counts matching physical targets. | Proves same-denomination fan repeat generation works with reproducible subset selection. Not a trainable candidate until real repeated-fan validation passes. |

## Current Active Assets

Treat this as the durable `data/` map. Named generated outputs may be kept when they support a live result; otherwise prefer regeneration from configs/scripts over keeping stale folders around.

### Tier 0: Canonical Backbone

- `data/numista_raw/`: trusted raw scan metadata/source cache. Numista `in_circulation` is the best scan backbone in this repo.
- `data/asset_candidates/numista_current_cutout_bank_v1/`: current best scan cutout bank. It has a manifest, masks, classes, and front/back metadata.
- `configs/3d_pipeline/proof_p0_renderer_smoke.json`: active renderer smoke config.
- `configs/3d_pipeline/proof_p1_transfer.json`: active transfer-proof config.

### Tier 1: Required Evaluation And Guardrails

- `data/cashsnap_v1/`: verified clean/base YOLO dataset, 13 bill classes and 9,048 labeled boxes. Use it for validation, guardrails, diagnostics, and baseline runs; it is not evidence that hard fan/overlap counting is solved.
- `data/real_fan_benchmark/`: scoreable/stress evaluation only. Never train on it.
- `manifests/real_fan_benchmark_label_quality.csv`: decides which real labels are scoreable.
- `runs/cashsnap/yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8/weights/best.pt`: current overlap-counting alpha checkpoint for diagnostics/export, not a solved final model.

### Tier 2: Useful But Conditional

- `data/raw_datasets/roboflow_cuurecy_detection_is/`: useful real phone partial/overlap material; `data/review/roboflow_cuurecy_detection_is_multinote_probe_v1/contact_sheet.jpg` samples dense hand/fan references, but public/reproduction, split, and visible-region label-policy caveats mean it stays review/domain-stress data until curated.
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
- WebGL proof: `renderers/webgl/src/render-smoke.mjs` uses Three.js + `puppeteer-core` against local Microsoft Edge and supports optional label-safe environment maps through `--environment-dir`; `run_webgl_recipe.py` resolves named recipe ids from the catalog; `check_webgl_smoke_output.py` validates nonblank RGB, exact ID colors, visible labels, primitive finger occlusion, and layer-order audit output; `check_webgl_label_views.py` validates packaged detect/OBB/fragment views and prevents diagnostic-only OBB labels from being mistaken for trainable data; `check_webgl_appearance_diversity.py` gates camera/lighting/surface/postprocess spread for trainable packages; `check_synthetic_recipe_catalog.py` validates target/recipe config coverage.
- Evaluation and deploy guards: `check_real_fan_benchmark.py`, `run_browser_smoke_cases.py`, `check_browser_stack_artifacts.py`, `build_webgl_recipe_ablation_configs.py`, `run_webgl_recipe_ablation.py`, `val_yolo.py`, `compare_yolo_metrics.py`, `export_yolo.py`.
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

Goal: keep the synthetic factory auditable: realistic-enough RGB, exact ID colors, visible-only labels, reproducible variants, and package QA that prevents diagnostic labels from becoming training data by accident.

Current contract:

- Renderer: `renderers/webgl/` uses Three.js and `puppeteer-core` against local Microsoft Edge by default; Edge is verified GPU-backed on this laptop through ANGLE/D3D11. Use `CASHSNAP_WEBGL_BROWSER` or `--browser-executable` only for deliberate Chromium A/B tests.
- Output: default RGB is 1440x1080 with RGB-only supersampling; ID pass is a separate non-antialiased render with black background and exact one-color-per-instance masks.
- Labels: detect boxes are visible-only compatibility labels; OBB export is allowed only for compact honest masks; visible-fragment labels are evidence/OCR inputs and must be fused back to physical parents before counting.
- Package QA: WebGL packages must include `recipe.json`, `qa/summary.json`, preview overlays, visual+ID overlays, `qa/quarantine.json`, and `qa/contact_index.json`; gates validate hashes, label views, trainability policy, ignored fragments, and OBB exclusions.
- Assets: WebGL uses the 13-class schema from `data/cashsnap_v1/data.yaml` and scan PNGs from `data/asset_candidates/numista_current_cutout_bank_v1/`; the current Numista audit has 76 assets, full front/back coverage for all 13 classes, and 3 red-mark suspects.
- Scene controls: variants are deterministic; class selection uses a co-prime stride; `asset-side-policy` controls front/back sampling; `camera-profile` records phone/browser framing assumptions before RGB/ID extraction.
- Occlusion and stacking: primitive finger capsules are the accepted hand-occlusion placeholder; note planes use explicit render order to keep visible masks label-safe. This is a controlled topology shortcut, not a physical contact solver.
- Backgrounds: no trainable real background bank is currently accepted. Existing extracted no-note banks were rejected for leaked note fragments, people/arms, outdoor/vertical contexts, or mask artifacts; `data/backgrounds/webgl_reviewed_clean_smoke_v1/` is only a two-image smoke proof.
- Performance: keep textures file-backed instead of base64-inlined. If render batches hit Chrome/RAM pressure, lower `visual_scale` or pass stricter headroom caps through the suite/pipeline wrappers.

Important caveats:

- Geometric lens distortion/crop is still parked until masks and boxes are transformed with it; current postprocess is non-geometric only.
- Smooth macro curl is acceptable; sinusoidal ripple looked corrugated and should stay off unless backed by a real crease/fold model.
- Tiny visible slivers below `--min-visible-pixels` stay in the ID image but are ignored as class labels.
- Dedicated split-label QA proved the counting trap: 3 physical bills can create 5 visible fragments, and full-schema fan smoke had 67 physical visible instances vs 120 fragments. Fragment outputs cannot be summed directly.

## Research Checkpoint

2026-05-30 bounded synthesis-data review:

- Synthetic data is useful because it can provide pixel-perfect labels, but domain gap remains the core risk; mixing synthetic with real data is repeatedly favored over treating synthetic as a full replacement. Source: [Eversberg and Lambrecht 2021](https://www.mdpi.com/1424-8220/21/23/7901).
- There is no universal winner between photorealistic rendering and domain randomization. Domain knowledge about the target scene and object-specific realism can matter more than global prettiness. Source: [Eversberg and Lambrecht 2021](https://www.mdpi.com/1424-8220/21/23/7901).
- Domain randomization studies point to object/texture diversity, camera positioning, and occlusion as meaningful parameters; random noise alone is lower leverage. Source: [Domain randomization review, PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8570318/).
- Hand-occlusion literature treats occlusion-aware augmentation as a valid first step, and papers cite black solid shapes or object segments pasted over people as established augmentation. This supports primitive finger occluders before a full hand mesh. Source: [HandOccNet CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Park_HandOccNet_Occlusion-Robust_3D_Hand_Mesh_Estimation_Network_CVPR_2022_paper.pdf).
- Recent synthetic-generation surveys frame 3D generators around either photorealistic rendering or domain randomization, but both still need target-domain validation. Source: [Artificial Intelligence Review 2025 synthetic generation survey](https://link.springer.com/article/10.1007/s10462-025-11431-3).

Decision from this pass: do not chase full PBR or a full hand mesh yet. Build a small WebGL batch proof with controlled banknote layer geometry, real-ish table/background variation, primitive fingers, exact visible masks, and measured comparison against the Python/2.5D baselines plus real partial/fan gates.

2026-06-05 dataset-construction paper pass:

- Domain randomization is useful when the simulator spans target nuisance variables enough that real data becomes an in-support variation, not when randomness is unbounded noise. Source: [Tobin et al. 2017](https://arxiv.org/abs/1703.06907).
- Structured domain randomization outperforms uniform random placement by sampling scene layouts from problem-specific context distributions; for CashSnap, randomization must be bill/table/hand/camera plausible. Source: [Prakash et al. 2018/2020](https://arxiv.org/abs/1810.10093).
- Copy-paste and synthetic instance generation can work, including for rare categories, but artifacts and context are part of the training signal; context-aware pasting can help object detection and random copy-paste can still be strong when validated. Sources: [Dwibedi et al. 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Dwibedi_Cut_Paste_and_ICCV_2017_paper.pdf), [Dvornik et al. 2018](https://arxiv.org/abs/1807.07428), [Ghiasi et al. 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.html).
- Realism/refinement methods still need annotation preservation and artifact guards; realistic-looking data is not automatically useful data. Source: [Shrivastava et al. 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Shrivastava_Learning_From_Simulated_CVPR_2017_paper.html).
- Synthetic dataset studies emphasize dataset properties and staged learning schedules, while object-detection studies report strongest savings from mixed synthetic+real sets and targeted enrichment of underrepresented classes. Sources: [Mayer et al. 2018](https://arxiv.org/abs/1801.06397), [Nowruzi et al. 2019](https://arxiv.org/abs/1907.07061), [Burdorf et al. 2022](https://arxiv.org/abs/2202.00632).
- Bias-controlled evaluation matters: a dataset can look good on ordinary benchmarks while failing under controlled viewpoint/background/rotation shifts. Source: [ObjectNet, Barbu et al. 2019](https://papers.neurips.cc/paper_files/paper/2019/hash/97af07a14cacba681feacf3012730892-Abstract.html).

Decision from this pass: define "perfect synthetic data" as a validated experiment generator, not infinite photoreal images. The CashSnap synthetic contract must include exact labels, structured condition coverage, target-domain nuisance spread, domain/dose gates, matched row-count and class-exposure controls, staged/curriculum probes, and real-only partial/fan stress validation before scale.

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
- Broad dataset searches in May 2026 found useful public KHR/USD partial/fan references, but no rights-clean scoreable P1 benchmark substitute. Stop broad hunting unless a specific new lead appears.
- Roboflow `cuurecy-detection-is` is useful but not clean proof: it has real partial/overlap examples, front/back masks, and repeated layouts/split caveats.
- PicWish can generate lots of cutouts, but skin/hand contamination repeatedly caused regressions.
- Numista clean scan fragments alone did not fix `KHR_5000 -> KHR_10000` partial confusion.
- Light-dose Numista 2.5D calibration was the best 2.5D direction so far: cleaner than broad synthetic, but still overcounts and confuses backs.
- The remaining high-confidence shop-overlap miss is a real partial-evidence problem, not just confidence calibration.

## Untested Or Parked Ideas

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

Keep this file honest during normal work. When a phase is achieved, abandoned, or superseded, rename or remove the old section instead of leaving stale labels like active tasks. Collapse completed checklists into current facts, delete parked ideas that became real features, and keep only the details that help the next agent make a better model decision.

Preferred documentation shape: one project `AGENTS.md`, this working `model.md`, and one user-facing `README.md`. Folder-level READMEs or extra docs should be temporary, archived reference, or deleted once their useful content is folded into the main three files.

Script organization should stay active too. When `scripts/` or other tool folders become noisy, group tools by current state such as `active/`, `archive/`, or domain folders only after checking imports, CLI references, and workflows that would break from a move.

Current script shape: root `scripts/` holds active helpers and wired workflows; `scripts/archive/` holds standalone historical probes/prep tools that are not referenced by the active docs/configs/code path. Restore an archived script to root only when it becomes part of the live workflow again.
