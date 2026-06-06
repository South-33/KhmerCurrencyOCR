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
- The current scorecard is blocked: `4` pass / `13` blocked axes after adding
  the project-level real-capture requirement axis.
- Taxonomy and asset coverage must be read by layer, not assumed. Official
  current USD/KHR scope is 21 classes. The active model schema and active WebGL
  cutout bank are still 13-class operational subsets; raw Numista current-status
  cache has front/back for 20/21 classes; raw Numista any-status cache has
  front/back for all 21. `KHR_50` is the only official class missing from the
  current-status raw cache, while any-status raw has 6 front/back pairs.
- `scripts/build_currency_taxonomy_gap_plan.py` turns that scope gap into a
  promotion queue. Current full-scope Numista candidate cutouts cover the seven
  active/model-missing current-status classes (`USD_2`, `KHR_100`, `KHR_200`,
  `KHR_15000`, `KHR_30000`, `KHR_100000`, `KHR_200000`); `KHR_50` is any-status
  only and needs explicit status review before trainable current-currency use.
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
- Older stack selected-geometry roots were confounded by synthetic finger
  occluders. Treat stack overlap and hand/finger occlusion as separate recipe
  variables; no-hand stack probes now require explicit occluder-policy gates.
- No-hand stack dose1 has a tiny positive detector signal but fails deploy
  usefulness: seed0 exact class-mix control probe improved test mAP50-95 by
  `+0.002033` and `KHR_50000` by `+0.022303`, but browser mined-real stress
  regressed on dense overlap, thin edge, and weak-class value behavior.
- Prop-diverse hard negatives are more realistic than the old weak root, but
  not a solved path: diversity8 loses `-0.003090` mAP50-95 versus old8, and
  browser rejection/threshold tweaks clear hard negatives by killing positives.
- Background false-positive probing explains the detector/browser mismatch:
  no-hand dose1 and both hard-negative fine-tunes hallucinate far more than the
  default detector on zero-label synthetic prop scenes. Do not scale those
  recipes until proposal false positives improve under a real/deploy guardrail.
- Existing model families split the problem: the old overlap-stage detector
  improves mined-real overlap/fan recall but is noisy on prop-diverse negatives;
  the full-real-only detector is background-saner but loses mined-real and
  synthetic stress recall. The next synthetic/curriculum probe must preserve
  overlap recall while borrowing the full-real/background restraint.
- Current WebGL overlap-stack training does not transfer like the old staged
  overlap detector: it is worse than default on mined-real dense overlap and
  thin-edge browser stress, and it is worse than default on synthetic prop
  false positives. The useful old-overlap signal is curriculum/data-shape
  evidence, not proof that the current WebGL overlap recipe is right.
- Tiny prop-diverse hard-negative repair of the old-overlap detector reduces
  raw prop false positives, but browser behavior shifts into duplicate proposal
  overcount. The next repair needs duplicate/proposal calibration, not simply
  more no-note hard negatives.
- Browser NMS is a real deploy lever for the old-overlap recall donor. NMS
  `0.65` keeps mined-real perfect cases at `12/17` while reducing total
  predicted count `44 -> 39` and absolute count error `11 -> 8`; adding the
  unclassified/disagreement guard clears synthetic hard negatives but drops
  mined-real perfect cases to `11/17`. Lowering the guard threshold from
  `0.14` to `0.10` does not recover those real matches.
- The existing broad 14-class real-fragment classifier is not the missing
  hard-negative gate. With old-overlap + NMS `0.65`, no rejection still fails
  synthetic card/phone and wallet/sticky negatives; disagreement rejection
  clears negatives by deleting almost all synthetic positives.
- Broad real-fragment `background` predictions are useful evidence but not a
  solved gate. Old-overlap + NMS `0.65` + broad background-only rejection after
  NMS clears all three synthetic hard negatives, but mined-real does not improve
  over plain NMS (`12/17` perfect still, `same 31/35` vs `32/35`, abs KHR
  `152000` vs `92000`) and synthetic hand evidence is deleted.
- A binary old-overlap proposal gate is the strongest calibration signal so
  far. V1 trained on detector proposal crops improves mined-real to `13/17`
  perfect and clears synthetic hard negatives; v2 adds WebGL partial-positive
  proposal crops and improves aggregate mined-real further (`pred=35`,
  `same=31/35`, `any=32/35`, abs count `4`, abs KHR `21000`) while still
  clearing the three synthetic hard negatives. This is diagnostic only until
  real no-note/non-banknote and protected real partial validation exist.
- Proposal-gate promotion is now explicitly blocked by capture buckets for
  `khr_50000_hard_positive_partials` and `usd_hard_positive_partials`; these
  target high-confidence banknote->background errors from the v2 failure pack.
- Browser-count comparison is now a first-class detector-probe artifact. Use
  `scripts/run_browser_detector_probe.py` after bounded synth detector probes
  when an ONNX export is available; it runs/reuses synthetic and mined-real
  browser reports and compares them with `scripts/compare_browser_reports.py`.
  The comparator caught no-hand stack dose1 as blocked despite its tiny detector
  mAP gain: synthetic stress worsened abs count by `+3`, abs KHR by `+26000`,
  and hard-negative predictions by `+1`; mined-real stress lost `3` perfect
  cases, `5` same-class matches, and added `+166000` abs KHR error.
- Strict browser comparison says proposal-gate v2 is the best diagnostic stack
  but still not promotable: mined-real passes versus default (`+4` perfect,
  `+8` same-class, `-6` abs count, `-62500` abs KHR, `-80` abs USD), while
  synthetic stress still blocks on positive evidence (`-1` same/any,
  `+2` abs count, `+66000` abs KHR) even though hard-negative predictions drop
  by `5` and failures drop by `3`.
- Proposal-gate v2 case effects are now reviewable with
  `scripts/summarize_browser_gate_effects.py`. The browser smoke summary now
  exposes proposal, classified, and clustered diagnostic sources, so the
  analyzer no longer infers gate effects from counters alone. At NMS `0.65`,
  synthetic stress has `2` hard-negative helps, `1` hard-negative same,
  `1` positive NMS harm, `1` positive reject harm, and `4` positive same.
  Mined-real has `14` positive same, `1` positive reject safe, and `2`
  positive reject harms. The concrete next target is protected thin/weak
  positives such as `cashsnapv1_weak_khr_50000_03_khmer_scan_img_9189_jpeg_jpg`,
  not generic hard-negative scale.
- Browser gate review targets are now generated with
  `scripts/build_browser_gate_review_targets.py` and embedded in detector-probe
  summaries as `gate_review`. For proposal-gate v2, the P1 review/capture
  targets are mined KHR_50000 hard-positive partials, mined KHR_20000 thin
  slices, synthetic hand-fan NMS/fusion harm, and synthetic same-denomination
  fan reject harm.
- `scripts/check_capture_requirements.py --gate-review-targets <targets.csv>`
  annotates the real shot list with those browser failure examples; use it when
  refreshing `runs/cashsnap/real_capture_shot_list_latest.md` so capture work is
  tied to concrete deploy failures.
- Raising proposal-gate v2 browser NMS from `0.65` to `0.80` is not the fix.
  It keeps mined-real strict comparison passing but worsens the v2 aggregate
  (`pred 35->38`, abs count `4->5`, abs KHR `21000->26000`) and synthetic KHR
  error (`474500->531500`) despite reducing synthetic abs count (`19->17`).
  The corrected gate-effect categories still point at the same protected-positive
  reject harms, so this needs proposal-gate calibration/validation, not a
  simple NMS threshold bump.
- Strict accepted-blend visible-note geometry is now diagnosable with
  `scripts/summarize_domain_gap_geometry_targets.py`. Current accepted-nowarmup
  synthetic boxes are much smaller than real (`area -0.372`, `width -0.417`,
  `height -0.381`). Thin-edge partials should remain a protected partial slice,
  not evidence that general visible-note scale is correct; clean/base and
  back-side confusion rows need larger visible-note framing or selected-geometry
  filtering. Worst class aspect gaps are `USD_10`, `USD_20`, and `KHR_10000`.
- Selection alone does not repair accepted general-visible geometry. Bounded
  selectors over clean-base + back-side roots still fail strict geometry at
  32 images (`area -0.352`) and 16 images (`area -0.341`). Clean-closeup v2
  proves aggregate scale is fixable (`area -0.009`, `width +0.027`,
  `height -0.039`) but still fails five per-class aspect checks and already
  regressed the real-only ablation, so a v3 needs class-conditioned pose/aspect
  calibration plus model proof, not just larger notes.

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

## Local Machine Posture

- Laptop: AMD Ryzen 5 7640HS, 6 cores / 12 threads, about 16 GB RAM, and an
  NVIDIA GeForce RTX 4060 Laptop GPU with about 8 GB VRAM; PyTorch sees CUDA
  device `0`.
- Prefer GPU-backed PyTorch/Ultralytics work with `--device 0` when VRAM has
  room. Heavy jobs should go through `scripts/run_with_headroom.py` when
  practical: caps may go up to 95% but never above 95%; resume around 82%.
- Do not let the headroom wrapper wait forever if it cannot find launch
  headroom. If it stalls, shrink the job first: lower batch/workers, lower RAM
  pressure, adjust preflight floors, or split the run.
- RAM pressure is the main bottleneck. Keep CPU/RAM-heavy jobs bounded with
  modest workers/batches, especially while the laptop is in active use. Browser
  smoke validation is ONNX Runtime Web/WASM and is still CPU-side.
- WebGL recipe rendering has an opt-in `--render-jobs` knob. Keep default `1`
  for safest laptop use; try `2` for bounded smoke/probe batches only with
  headroom caps at or below 95% and a short preflight timeout.

## Immediate Plan

1. Keep the desk clean:
   `model.md` stays current and concise; old details go to `docs/archive/`.
2. Do not promote the 512 clean root. Use its ablation result as signal that
   synthetic exposure can help p24-style scarcity, while `KHR_50000` and
   `KHR_20000` still need a better real/synthetic bridge.
3. Do not scale no-hand stack yet. Its first exact class-mix detector probe is
   slightly positive, but browser stress says the signal is not deploy-useful.
   Next work should explain detector-vs-browser mismatch before more renders.
4. Treat the next repo-only bet as a proposal-calibration/curriculum probe, not
   a renderer expansion: combine overlap/fan recall pressure with no-note or
   background restraint and require both mined-real browser stress and synthetic
   hard-negative stress to avoid regression.
5. Do not repeat broad old-overlap hard-negative fine-tunes as-is. A focused
   hardneg8 repair improved raw background FPs but increased browser duplicate
   proposals; any next variant must explicitly constrain duplicate/overcount
   behavior.
6. Keep old-overlap + browser NMS `0.65` as the current best diagnostic deploy
   calibration, not a promotion. It improves mined-real count noise without
   losing recall, but it still fails synthetic hard-negative safety unless the
   rejection guard is added, which costs real recall.
7. Advance the binary proposal gate, not broad denomination rejection. The v2
   synthetic-partial proposal gate is the best current diagnostic stack, but it
   still needs real no-note/non-banknote validation and hard-positive partial
   review before promotion.
8. Use render concurrency carefully before more 500+ image runs:
   `render_webgl_variant_batch.py` and `run_webgl_recipe.py` support
   `--render-jobs`; validation/package assembly remains sequential. Keep
   laptop probes bounded until larger batches prove headroom is stable.
9. For every detector-side synth probe that gets trained, export/reuse ONNX and
   run `scripts/run_browser_detector_probe.py` before treating mAP as useful.
   The pass/fail comparison and, when artifacts exist, gate-effect counts should
   travel with the training summary.
10. Use `scripts/summarize_browser_gate_effects.py` and
   `scripts/build_browser_gate_review_targets.py` on two-stage browser reports
   before proposal-gate changes. Gate deltas must separate hard-negative help,
   positive reject harm, and NMS/duplicate harm, then map those rows to real
   capture/review buckets.
11. Only revisit fresh-from-`yolo26n.pt` base-first training if the checkpoint
   ablation is not harmful. Do not stage overlap/fan/hand from a weak clean
   stage.
12. In parallel, improve the real bridge:
   promoted real fan/overlap/hand/no-note/hard-positive partial labels are
   still the biggest blocker.
13. For geometry repairs, run `scripts/summarize_domain_gap_geometry_targets.py`
   on the latest strict geometry CSVs before rendering more. Separate
   protected partial slices from general visible-note scale and repair the
   clean/back-side source groups first. Do not treat a strict geometry pass as
   enough; clean-closeup and selected-stack history show geometry can pass while
   model/browser utility still regresses.

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
- Raw Numista coverage is broader than the active bank. Diagnostic probe banks
  under ignored `data/asset_candidates/` show current-status full-scope cutouts
  cover 20/21 official classes, and any-status full-scope cutouts cover 21/21.
  Do not use the any-status bank for trainable current-currency recipes without
  design/status review.

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

No-hand stack diagnostic:

- `data/synthetic/cashsnap_webgl_no_hand_real_scale_stack_print_tone_selected_geometry_v1`
- Status: diagnostic only. It isolates note-on-note stack overlap from
  hand/finger occlusion and passes strict geometry after selecting 20 images
  from a 40-image no-hand pool.
- Evidence: selected20 has 83 YOLO boxes, occluder policy `no_hand`, zero
  occluders, local dynamic-range print tone passing, and strict geometry pass.
- Model result: dose1 selected variant 7 and beat an exact class-mix exposure
  control by `+0.002033` mAP50-95 at seed0; `KHR_50000` improved `+0.022303`,
  worst class drop was `USD_20=-0.009805`.
- Deploy result: the same detector is not promotable. Browser synthetic stress
  still fails hard negatives and overcounts overlap; mined-real browser stress
  worsens dense-overlap recall/count, thin-edge count/value, and weak-class
  value behavior versus the default detector.
- Remaining blockers: class balance is loose, trainable-candidate appearance
  diversity is too narrow for promotion, exact real class-mix controls are
  unavailable from current repo data, and deploy behavior does not transfer.

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
- Negative model probes from older selected stack roots are not pure no-hand
  overlap evidence because every stack image had synthetic finger capsules.
- No-hand stack dose1 is a weak positive detector result, not a promotion:
  exact class-mix mAP improved, but browser mined-real stress regressed.
- Hard-negative diversity8 is not promotable by itself: it improves some KHR
  value noise on synthetic stress, but loses detector AP versus old8 and
  threshold/disagreement rejection trades false positives for missed bills.
- Background FP comparison (`runs/cashsnap/background_fp_default_nohand_hardneg_comparison.json`):
  at `conf=0.05`, default has `13/32` detections on the old hard-negative root
  and `9/8` on the prop-diverse root; no-hand dose1 jumps to `56/32` and
  `22/8`, while hardneg old8/diversity8 remain high (`48/32`/`18/8` and
  `46/32`/`16/8`). The no-hand AP lift is likely proposal hallucination, not a
  deploy-useful overlap fix.
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
- Current WebGL overlap-stack ablation is rejected in deploy terms. Exported
  `webgl_ablation_overlap_stack_from_clean` gets mined-real dense-overlap
  final same recall `3/9` versus default `5/9` and old-overlap `8/9`, and
  synthetic hard-negative browser cases still all fail. Raw prop FP is also
  worse than default (`43/32` and `12/8` detections at `conf=0.05`).
- Old-overlap prop-diverse hardneg8 repair is diagnostic only. Full one-epoch
  repair cuts raw prop FPs (`11 -> 6` detections on the 8 prop-diverse frames)
  but overcounts badly in browser (`synthetic_overlap_stack pred=21/6`;
  mined-real fan `pred=13/8`, thin-edge `pred=18/10`). A gentler 32-batch
  repair still overcounts synthetic overlap (`pred=11/6`) while only partly
  reducing hard-negative FPs.
- Old-overlap browser NMS `0.65` is a useful calibration result. It keeps the
  old-overlap mined-real strict score at `12/17` but improves aggregate
  stability (`pred 44 -> 39`, abs count error `11 -> 8`, abs KHR error
  `107000 -> 92000`) versus default NMS `0.85`. NMS `0.65` plus
  disagreement/unclassified rejection clears all three synthetic hard-negative
  browser cases, but mined-real drops to `11/17` and loses fan/thin matches;
  unclassified-only leaves wallet/sticky false positives, disagreement-only
  leaves card/phone false positives.
- Existing broad real-fragment classifier diagnostic:
  `configs/cashsnap_two_stage_broad_realfrag_browser_stack.json` plus
  old-overlap/NMS `0.65` is rejected. Without disagreement rejection, synthetic
  card/phone and wallet/sticky still false-positive; with rejection, synthetic
  hard negatives clear but positive stress collapses to near zero recall.
- Background-only broad real-fragment gate is also diagnostic only. The browser
  app can reject configured fragment classes and optionally run NMS before
  rejection; `configs/cashsnap_two_stage_oldoverlap_broad_realfrag_bg_reject_browser_stack.json`
  uses this to reject `background` while preserving detector denominations.
  It clears synthetic hard negatives (`failures 2 -> 0`, perfect `1/9 -> 3/9`)
  but does not beat plain old-overlap NMS on mined-real (`12/17` perfect,
  `same 31/35`, `any 32/35`, abs count `7`, abs KHR `152000`, one rejected
  real proposal). Use it as proof that proposal-level background evidence
  exists, not as a default deploy stack.
- Binary proposal-gate diagnostics are promising. `scripts/build_proposal_gate_dataset.py`
  auto-labels old-overlap detector proposal crops as `banknote`/`background`
  by YOLO-label IoU, adds random safe background crops, and writes ImageFolder
  data. V1 (`mobilenet_v3_proposal_gate_oldoverlap_v1_e6`) clears synthetic
  hard negatives and improves mined-real to `13/17` perfect, abs count `5`,
  abs KHR `26000`, but lowers recall (`same 30/35`). V2
  (`mobilenet_v3_proposal_gate_oldoverlap_synpartial_v2_e6`) adds WebGL
  clean/overlap/fan/hand/thin positive proposals; it keeps synthetic
  hard-negative failures at zero and gets the best mined-real aggregate so far:
  `13/17` perfect, `pred=35`, `same 31/35`, `any 32/35`, abs count `4`, abs KHR
  `21000`. Still diagnostic: the validation set is tiny, and real no-note plus
  reviewed real partial hard positives are missing. V2 validation errors are
  concentrated in real proposal crops (`cashsnap_v1` USD banknotes and mined
  weak `KHR_50000`) being rejected as background, so the next repair is hard
  real positives plus reviewed real no-note props, not a simple threshold sweep.
  Failure review pack: `data/review/proposal_gate_synpartial_v2_failures/`
  (`9` background->banknote, `11` banknote->background high-confidence errors).
- Full-real-only is a useful negative/proposal reference, not a browser base.
  It is cleanest on synthetic prop false positives (`4/32` and `1/8`
  detections at `conf=0.05`, and zero detections by `conf=0.18`), but browser
  replay regresses mined-real same-class recall versus default/old-overlap and
  collapses positive synthetic stress. Use it to shape restraint, not to replace
  the current detector.
- Disagreement/unclassified browser veto flags are diagnostic only. They clear
  some synthetic hard negatives but do not improve mined-real stress.

## Hard Blockers

- No protected/promoted real P1 fan/overlap/hand stress benchmark yet.
- No real repeated-denomination fan labels for same-class counting/fusion.
- No reviewed real no-note/non-banknote paper capture inventory.
- No real mixed USD+KHR rare/common stack captures, especially with
  `KHR_50000`.
- Current model schema and active WebGL cutout bank are incomplete relative to
  full USD/KHR. Verify raw/current/active/model coverage with the taxonomy
  coverage check before saying a class exists or is missing.
- Clean-real train split still has thin rare anchors:
  `KHR_20000=14` and `KHR_50000=15` unique train images.
- Renderer throughput is too slow for 500+ image iteration on this laptop.

## Commands

Keep this list short. Use script help or archive history for older workflows.

Scorecard:

```powershell
rl python scripts\check_synthetic_dataset_scorecard.py
```

Currency coverage:

```powershell
rl python scripts\check_currency_taxonomy_coverage.py
```

Readiness:

```powershell
rl python scripts\check_synthetic_pipeline_readiness.py --check-existing --json-out runs\cashsnap\synthetic_pipeline_readiness_latest.json
```

Trainable-candidate suite:

```powershell
rl python scripts\check_webgl_trainable_candidate_suite.py --check-existing
```

Occluder policy:

```powershell
rl python scripts\check_webgl_occluder_policy.py --root <root> --expected-policy no_hand --forbid-hand-occluders
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

- Heavy training/rendering should use `scripts/run_with_headroom.py` when
  practical. The wrapper refuses caps above 95%, lowers child priority, waits
  for initial headroom, and treats free RAM as preflight/runtime pressure.
- Current cap posture: using up to 95% max CPU/RAM/GPU/VRAM is acceptable when
  needed, but never set any cap above 95%; resume around 82%.
- If the wrapper cannot find initial headroom, do not wait indefinitely. Reduce
  batch/workers, lower RAM pressure, relax only safe preflight floors, or split
  the run into smaller chunks.
- Prefer GPU for PyTorch/Ultralytics training/inference when VRAM is available;
  pass `--device 0` instead of letting long jobs drift onto CPU.
- RAM pressure is the main bottleneck. Keep data-loader workers low; the
  archived dry-run selected `batch=4`, `workers=1`, `device=0`, with
  `--min-free-ram-gb 4.0` on this laptop.
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
