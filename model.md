# CashSnap Model Brain

This is the living working memory for model and synthetic-data decisions. Keep it
short, current, and decision-oriented. Old detail belongs in `docs/archive/`,
registries, or the folder structure itself.

Major history snapshots:
- `docs/archive/model_brain_pre_production_pilot_cleanup_2026-06-11.md`
- `docs/archive/model_brain_pre_housekeeping_2026-06-09.md`
- `docs/archive/model_brain_pre_housekeeping_2026-06-08.md`
- `docs/archive/model_brain_pre_cleanup_2026-06-07.md`
- `docs/archive/model_brain_pre_compact_2026-06-07.md`
- `docs/archive/model_brain_full_history_2026-06-06.md`

Strategy reference:
- `docs/research/cashsnap_real_synth_blend_strategy_final.pdf`

## Shape Rule

Keep this shape simple and stable:

1. Yardstick And Posture.
2. Research Frame: Current State, Tested Ideas, Untested Ideas.
3. Promotion Gates.
4. Validation, Labels, And Scope.
5. Repo Hygiene.

Do not split the live context into separate "current read", "current bet",
"next move", and "evidence" sections. Keep those inside `Research Frame` as
current state, tested ideas, and untested ideas. If an idea cannot name the
expected effect, the guardrail, and what would kill it, leave the space open
instead of adding filler.

Do not use this file as an artifact index. Folder placement, archive folders,
JSON registries, generated-list locations, and `rg` should answer "where is that
file?" This file should answer what we believe, what is blocked, what not to
repeat without a new reason, which ideas look promising, and what gates decide
promotion.

Keep `model.md` live. Whenever direction, evidence, blockers, or candidate ideas
change, update this file in the same pass: prune stale advice, remove achieved or
rejected ideas, and rewrite the research frame instead of appending a mini
changelog. A stale `model.md` is a repo bug, not harmless history.

This file is context, not a command queue. A future agent should read it,
challenge it, inspect the current repo/results, and choose the best next step by
their own judgment.

## Yardstick And Posture

North star: build one small phone/browser-deployable detector that can count
mixed USD and Khmer riel from one casual retail photo, preserving clean
non-overlap performance while becoming materially better on countable partial,
overlap, fan, hand, cutoff, and edge-visible evidence.

Current phase posture: build a single production-pilot detector recipe. The old
"best clean detector" and "best overlap/partial clue" are no longer separate
deliverables. The clean champion is the guardrail to protect; the partial
candidate is the best available initialization/signal; the next model should
combine the durable lessons into one checkpoint.

Do not launch another tiny p12/p24/filter/scheduler probe unless it directly
de-risks the production-pilot blend, label policy, or promotion gates. Tiny
probes are useful only when they answer a specific failure question.

Clean/non-overlap yardstick: seed0 p24 balanced-real + strictbest-synth is the
strongest clean/non-overlap detector recipe we can honestly justify today:
`runs/cashsnap/fixed_step_real_p24_plus_strictbest_synth_p24_from_clean_e1_i416_b2_w0_adamw_lr5e5_nowarmup_noamp_cachefalse_steps318_seed0/weights/last.pt`.
Baselines: full real mAP50-95 `0.852767`, strict semantic+leakage-clean
`0.860743`, source-excluded strict-clean around `0.78`. It is the clean
foundation to protect, not the final partial/overlap answer.

Pilot initialization posture: do not restart from raw `yolo26n.pt` unless both
strong initializations fail or a control is explicitly needed. First use the
existing learned CashSnap foundations. The conservative init is the clean p24
synth+real champion above; the visible-evidence challenger init is the p24 vis70
candidate:
`runs/cashsnap/fixed_step_countsafe_vis70_p24_from_last_e50_i416_b2_w0_adamw_lr5e6_nowarmup_noamp_cachefalse_freeze22_steps318_seed0/weights/last.pt`.
It was only a 318-batch head tune over `927` unique rows (`323` original real,
`312` strictbest synthetic, `292` source-clean `vis0p7` partial crops), but it is
the best current compromise signal: full real `0.854178`, strict clean
`0.865306`, source-excluded clean `0.795336`, unfiltered partial test
`0.660102`, filtered countable-partial test recall/precision `0.8857/0.5569`.
It is not production-safe by itself because source/unknown-money and
wrong-denomination proposal issues remain. If time allows, run both inits on the
same pilot blend: champion-init protects clean behavior, vis70-init tests whether
the partial clue survives hard-negative and clean replay pressure.

Production Pilot v1 artifact exists but is untrained:
`configs/webgl_ablation/cashsnap_production_pilot_v1.yaml`, built by
`scripts/build_cashsnap_production_pilot_config.py`, has `2665` exposure rows
and `986` unique images: clean real `35.68%`, strictbest synth `23.86%`,
countable partial `23.79%`, train-safe hard negatives `9.91%`, and high-risk
class protectors `6.75%`. It uses existing list-backed/train-split rows only,
blocks unreviewed corner-50 partial rows, and keeps eval-mined hard negatives out
unless they are train-split analogs.

Browser/gate posture: current detector+gate/browser stacks are diagnostic
product clues, not proof that the detector learned visible-evidence reasoning.
Proposal gates can trim background/unknown-money leakage, but they do not rescue
a detector that creates duplicate or wrong-denomination boxes.

## Research Frame

### Current State

The live goal is one production-pilot detector, but the experiment should compare
reasonable initializations instead of assuming one. First-run matrix: clean
champion init, p24 vis70 init, and a YOLO26s capacity check if browser/model-size
budget allows. The first serious runs should train long enough to see the curated
blend, roughly `5k-8k` image presentations at `imgsz=416`, `batch=2`, low LR, no
AMP, cache false, with clean replay and source/unknown pressure. A
freeze-then-unfreeze schedule is allowed if it produces one final checkpoint and
does not hide a phase-confounded result.

The pilot is successful only if partial/overlap recall improves for the right
reason: recognizing countable visible evidence. It is a failure if the gain comes
from duplicate same-note boxes, wrong-denomination boxes, or target-class
predictions on unknown/foreign/non-banknote money.

Use the p24 synth+real clean champion as the fallback and guardrail. Use the p24
vis70 candidate as the best clue that partial-visible positives can transfer.
Use the filtered countable-partial eval bridge
`configs/audit/cashsnap_real_countablepartial_sourceclean_vis70_plus_center50_eval_v1.yaml`
as a cleaner partial yardstick than the old unfiltered vis50/70 split.

The old unfiltered partial eval contained policy-poison rows: exact USD100
"misses" that were not human-countable and `corner_*_vis0p5` fragments that were
often denomination-ambiguous. Future partial rows must be human-countable from
visible evidence, ignored/excluded if ambiguous, and never silently converted
into forced denomination labels.

Current hard blockers for promotion are proposal quality and source policy:
duplicate boxes, wrong-denomination overlaps, unknown/foreign/non-banknote money
leaking into target classes, and possible multi-instance label gaps. The source
FP review queue for the p24 vis70 candidate remains useful:
`runs/cashsnap/countsafe_vis70_p24_v1/source_fp_review_candidate_vs_dupctrl_v1/`.

### Tested Ideas

- **Clean p24 synth+real is the clean yardstick.** Controlled balanced-real p24
  plus strictbest synth p24 beat balanced real duplication on full/strict/source
  clean checks. Protect it during pilot work.
- **p24 vis70 is a real but unsafe visible-evidence signal.** It improves the
  filtered countable-partial test slice and preserves clean AP better than many
  probes, but source-FP review shows duplicate, wrong-class, unknown-money, and
  multi-instance issues. It is an init/teacher clue, not a promotion.
- **Positive-only partial dosing is not enough.** Border partials, bbox
  blockers, mined edge/cutoff rows, strict KHR partials, and center/corner
  shuffles either failed partial scorecards, broke clean/source guardrails, or
  increased FP/prediction counts. Keep the visual QA policy; pair partial
  positives with hard negatives, clean replay, and proposal/objectness pressure.
- **Filtered vis70+center50 is useful as eval policy, not as a positive-only
  training win.** The filtered eval bridge keeps `vis0p7` plus center-strip 50%
  rows and excludes corner-50 rows. The corresponding p24 train mix lost to its
  duplicate control, so do not repeat center/corner-positive shuffles without a
  broader pilot recipe.
- **Reviewed real overlap/fan anchors are valuable but source-heavy.** The first
  39-row reviewed overlap anchor dose passed some clean guards but mostly added
  proposals and failed duplicate-control/held-out scorecards. Include reviewed
  anchors only as low-exposure pilot ingredients unless the eval pocket grows and
  source/class protection improves.
- **Naive synthetic overlap/fan assets have not transferred yet.** Rectangular
  real-crop fan composites and small WebGL stack/fan doses hurt clean/KHR guards.
  WebGL/masked assets are still useful for diagnostics and future audited
  label-preserving generation, but unsafe synthetic stack/fan labels should not
  be blindly added to the pilot.
- **Hard negatives help only with policy and balance.** Tiny coin/foreign/empty
  doses and the broad source-policy positive/negative mix did not repair the
  detector; some bought recall by spending count safety. The pilot uses only
  train-safe zero-label hard negatives and should be judged on count/value and
  source-FP behavior, not just AP.
- **Thresholds, NMS, and gates are diagnostic, not detector proof.** Narrow KHR
  class floors can improve a filtered slice but are not broadly safe. Lowering
  class-aware YOLO NMS did not change filtered partial results, so ordinary
  same-class NMS is not the remaining fix. Broad class-agnostic NMS can hide
  duplicates by spending recall.
- **Head-only AP continuations are not enough.** The June 10 duplicate-control
  continuation improved AP but failed low-confidence proposal and hard-slice
  count/value guards. Do not silently promote AP-only improvements.
- **Crop/reclassifier/browser stacks remain adjacent.** Reclassification and
  proposal gates are useful product architecture clues, but the current work
  should still deliver a detector checkpoint unless the phase explicitly switches
  to product-stack selection.
- **Official21/KHR100 work is schema diagnostic.** Official21 probes show staged
  missing-class learning is possible, but the operational detector remains the
  current 13-class schema. Do not mix official21 claims into pilot promotion
  without a compatible evaluation/mapping harness.

### Untested Ideas

- **Train Production Pilot v1 init/capacity matrix.** Run at least A and B under
  the same pilot config and promotion gates. A starts from the clean p24
  synth+real champion; B starts from the p24 vis70 candidate. This separates
  "clean foundation plus curated blend" from "visible-evidence init plus curated
  blend." Add C for YOLO26s if compute and deployment budget allow. A direct
  `yolo26s.pt -> pilot` run is only a capacity smoke because it lacks the
  CashSnap clean foundation; the fair YOLO26s comparison is `yolo26s.pt ->
  clean p24 synth+real foundation -> pilot`. Judge YOLO26s on accuracy plus
  phone/browser costs: PyTorch size, ONNX size, CPU/WebGL latency, and whether it
  actually fixes visible-evidence/counting errors. Do not regenerate all synth
  data or audit the entire universe from zero; do a targeted audit of pilot
  ingredients and generate only missing, policy-clear evidence. Suggested
  vis70-init dry-run command already validated:
  `scripts/bench_train_with_headroom.py --model <p24-vis70-last.pt> --data configs/webgl_ablation/cashsnap_production_pilot_v1.yaml --epochs 3 --imgsz 416 --batch 2 --workers 0 --device 0 --optimizer AdamW --lr0 0.000005 --lrf 0.2 --warmup-epochs 0 --seed 0 --cache false --freeze 22 --no-amp --no-val --quiet --memory-clean-task CashSnapHiddenMemReduct`.
  Clean-champion init dry-run is the same command with `--model
  <clean-champion-last.pt>` and run name
  `production_pilot_v1_from_cleanchampion_e3_i416_b2_w0_adamw_lr5e6_nowarmup_noamp_cachefalse_freeze22_seed0`.
  YOLO26s smoke dry-runs also resolve, but `yolo26s.pt` is not repo-local yet;
  use `lr0=0.00005`, no freeze, and treat direct `yolo26s.pt -> pilot` as a
  smoke, not the fair capacity verdict. The fair YOLO26s path first builds a
  clean p24 synth+real foundation from `yolo26s.pt`, then runs the pilot blend
  from that checkpoint.
- **Evaluate the pilot as one model.** Compare against the clean champion and
  p24 vis70 clue on full real, strict clean, source-excluded clean, filtered
  countable partial, source-FP review queue, hard-slice count/value, and
  per-class guards. Do not call the pilot a win until duplicate/wrong-denom and
  unknown-money behavior are inspected.
- **If Pilot v1 fails, diagnose by mechanism.** Split failures into clean
  regression, partial recall miss, duplicate overproposal, wrong-denomination
  overlap, unknown/foreign/non-banknote leakage, and protected-class collapse.
  The next recipe should target the mechanism, not add another generic row dose.
- **Real phone capture bridge remains high-value.** Own-photo capture is still
  empty. Highest-value captures: hand fans, same-denomination fans,
  `KHR_5000`/`KHR_20000` thin slices, `KHR_5000` face/number overlap,
  `KHR_50000` hard positives, mixed USD+KHR stacks, no-note backgrounds, coins,
  and non-banknote paper props.
- **Audited label-preserving half-synth remains plausible.** Use masked/audited
  real note assets or real captures, account for all notes in the scene, include
  source-aware unknowns, and protect weak KHR classes. Kill it if real recall and
  empty/source-FP behavior do not improve together.
- **Unknown-aware proposal/objectness objective may be needed.** If the pilot
  repeats the same overproposal pattern, a detector-side objective or sampling
  scheme that separates target recall from unknown rejection may matter more than
  further positive-row curation.

Small supporting tactics, not big ideas: failure-led obligation sets,
train-side mined-real near-negatives, audited source-context replacement,
multi-instance replacement, convergence-control checks, class-aware teacher row
filters, crop visual-gap gates, and camera/ISP/tone ablations. Use them only if
they serve one of the big questions above.

## Promotion Gates

A detector-foundation candidate is credible only when it improves or preserves:
- full real val/test;
- semantic-clean and semantic+leakage-clean audit slices, alongside full real
  val/test;
- strict clean source-excluded slices;
- filtered countable-partial val/test;
- source-FP review queues and train-safe hard-negative probes;
- hard-slice count/value behavior with final browser-style postprocess when the
  question is product selection;
- protected classes, especially riel and high-value USD;
- real empty-frame FP detections and images-with-FP at `conf=0.05`, `imgsz=416`,
  `batch=1`, `device=0`;
- max per-class mAP50-95 drop `<=0.05`, unless explicitly waived;
- at least one seed repeat for serious promotion, more for large claims.

Synthetic package gates are necessary filters, not promotion authority. Self-eval
preservation is not enough. For low-memory probes, use lightweight transfer
scorecards over multiple confidence thresholds and require no recall regression
plus no FP/background regression.

The clean base can move toward overlap/fan/hand only when the chosen foundation
survives the live detector gates: current-champion comparison, strict-clean and
source diagnostics, protected riel/USD stability, real-empty FPs no worse than
control, low-confidence behavior understood, and at least a seed repeat or a
slow-promotion run.

## Validation, Labels, And Scope

Validation:
- Full real val/test includes many empty-label images; always pair aggregate AP
  with empty-frame FP probes.
- The filtered countable-partial bridge is the current partial-visible yardstick
  because it removes non-human-countable and corner-50 ambiguous rows.
- Mined-real stress slices are warning slices, not release proof.
- Roboflow core-13 bridge is a positive KHR/USD judge for the current detector,
  but it is stretched and lacks background pressure.
- Roboflow official21 partial bridge preserves official classes present in the
  source, including `KHR_100`, but current 13-class weights cannot evaluate it.

Labels and class scope:
- Visible evidence is authoritative.
- Detector labels are visible-instance AABBs.
- OBB/quadrilateral metadata is for audits and future oriented/fusion work, not
  today's direct YOLO detect label.
- Fragment/evidence labels are not physical-note counts; count fusion must map
  fragments back to parent notes.
- Human-unidentifiable slivers should be ignore/unknown, not forced
  denominations.
- Zero-label hard-negative roots must remain zero-label; do not silently turn
  foreign/unknown notes into target classes.
- Current active detector scope is 13 operational classes, not all official
  USD/KHR. Run `scripts/check_currency_taxonomy_coverage.py` before class-scope
  claims.
- `KHR_100` is official KHR but outside the current core-13 detector.
- `KHR_50` remains blocked for v1 operational training unless real retail/bank
  capture evidence or an explicit product requirement justifies it.
- Trainable WebGL target-note renders must pass the approved texture-asset gate.

## Repo Hygiene

Documentation:
- Preferred doc shape is one project `AGENTS.md`, this working `model.md`, and
  one user-facing `README.md`.
- No long path inventories, append-only changelogs, stale "active" labels, or
  command dumps here.
- Archive/reference material can live under `docs/archive/`; active model memory
  belongs here.
- When a script, config, or dataset is no longer active, make that visible in the
  folder or registry. Before moving code/configs, check imports, CLI references,
  docs, and workflow callers with `rg`.

Runtime and harness:
- Work on `master` unless the user asks for a branch.
- Prefer `rl` command prefixes in RunLong.
- Use repo-local runtime storage through `scripts/local_runtime.py`.
- Keep YOLO train/eval caches and generated outputs under repo-local ignored
  paths.
- Import/call `scripts/local_runtime.py::configure_project_cache()` before
  Ultralytics/Torch-heavy imports in ML entry points.
- YOLO promotion posture: train/eval `cache=false`; use `workers=0` for train on
  this laptop unless explicitly running a heavier parity pass.
- Run long/big training, rendering, and broad eval jobs through the headroom
  guard (`scripts/run_with_headroom.py`, or `scripts/bench_train_with_headroom.py`
  for YOLO training). Prefer `--memory-clean-task CashSnapHiddenMemReduct` or the
  quiet WinMemoryCleaner task path under RAM pressure; do not use automation
  tasks that show user notifications.
- While the laptop is being used interactively, keep probes GPU-targeted
  (`device=0`) but CPU/RAM-light: no parallel GPU jobs, `workers=0`,
  `cache=false`, and smaller eval/train batches unless explicitly running a
  promotion-parity pass.
- List-backed YOLO runs can write mixed-image cache files; delete stale
  `data/cashsnap_v1/labels/train.cache`, `data/cashsnap_v1/labels/test.cache`,
  and partial-eval label caches after mixed probes.
- Fixed-step `--max-train-batches` is a stop cap, not a data repeater. Set enough
  `--epochs` to reach the cap.
- Fixed-step preflight reports train-phase summaries for unequal row counts. Use
  `--fail-on-train-phase-mismatch` for clean A/Bs, or label unequal-row runs as
  phase-confounded diagnostics.
- WebGL default posture remains `--render-jobs 2 --renderer-batch-size 32
  --check-jobs 4`.
- `cache=disk` is rejected for YOLO probes because it created large `.npy` caches
  and slowed throughput.

Canonical checks:

```powershell
rl python scripts\check_currency_taxonomy_coverage.py
rl python scripts\check_data_lifecycle_registry.py
rl python scripts\check_synthetic_pipeline_readiness.py --check-existing --json-out runs\cashsnap\synthetic_pipeline_readiness_latest.json
rl python scripts\check_webgl_trainable_candidate_suite.py --check-existing
rl python scripts\check_yolo_transfer_guardrails.py --help
```
