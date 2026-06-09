# CashSnap Model Brain

This is the living working memory for model and synthetic-data decisions. Keep it
short, current, and decision-oriented. Old detail belongs in `docs/archive/`,
registries, or the folder structure itself.

Major history snapshots:
- `docs/archive/model_brain_pre_housekeeping_2026-06-09.md`
- `docs/archive/model_brain_pre_housekeeping_2026-06-08.md`
- `docs/archive/model_brain_pre_cleanup_2026-06-07.md`
- `docs/archive/model_brain_pre_compact_2026-06-07.md`
- `docs/archive/model_brain_full_history_2026-06-06.md`

strategy reference:
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

Keep `model.md` live. Whenever direction, evidence, blockers, or candidate
ideas change, update this file in the same pass: prune stale advice, remove
achieved or rejected ideas, and rewrite the research frame instead of appending a
new mini-changelog. A stale `model.md` is a repo bug, not harmless history.

This file is context, not a command queue. A future agent should read it,
challenge it, inspect the current repo/results, and choose the best next step by
their own judgment.

## Yardstick And Posture

North star: build a small detector foundation that can eventually count mixed
USD and Khmer riel from one casual retail photo on phone/browser hardware.

Current phase posture: clean/non-overlap foundation work is parked unless new
reviewed labels, official21 missing-class labels, or a materially different
data source appears. The promoted p24 synth+real detector already sits around
the old aggregate target, and the remaining non-overlap risks are now
data/schema/product-policy bottlenecks rather than obvious training-recipe
knobs. Do not start another clean non-overlap training sweep just to chase a
small aggregate AP move.

End-goal premise: there is currently no real overlap/fan/hand data in the
trusted training set. The reason to push synth+real for non-overlap is to build
a calibrated real-domain base that can plausibly learn overlap later from
synthetic or real-derived half-synthetic overlap data. Do not treat clean
non-overlap as the final task; treat it as the foundation that must survive a
future synthetic-overlap fine-tune.

Parked non-overlap deliverable: seed0 p24 balanced-real + strictbest-synth is
the strongest clean/non-overlap detector recipe we can honestly justify today,
with repeat evidence from seed1 and clear failure pockets documented below.
Future non-overlap changes should be tied to reviewed real/source policy,
official21 schema expansion, or new eval evidence, not another blind blend.

Historical entry target: `0.82-0.85` full real test mAP50-95. Treat this as the
old "is the detector plausible?" bar, not the current goal. Clean-real controls
proved the evaluator and model capacity can reach the zone: a near-size
real-trained control reached `0.819153`, and the later high clean-real checkpoint
reached `0.883801`. The current goal is a robust, explainable foundation, not a
single aggregate AP number.

Current promoted detector-only non-overlap yardstick: controlled balanced-real p24 +
strictbest-synth p24 seed0 from the clean checkpoint:
`runs/cashsnap/fixed_step_real_p24_plus_strictbest_synth_p24_from_clean_e1_i416_b2_w0_adamw_lr5e5_nowarmup_noamp_cachefalse_steps318_seed0/weights/last.pt`.
It is the current detector/AP winner: full real mAP50-95 `0.852767`, strict
semantic+leakage-clean `0.860743`, and source-excluded strict-clean `0.769331`,
versus balanced real p24 `0.835861` / `0.855619` / `0.760870`. This proves the
current strictbest synth rows can improve the detector over real duplication. It
is the model-side recipe to beat; detector+gate/browser behavior is adjacent
diagnostic evidence only unless the phase explicitly switches back to product
selection.

Deployability smoke: the same checkpoint has a compact ONNX export
`weights/last.onnx` and passes a CPU ONNX detector smoke on
`asian_currency_IMG_6451...` with one `KHR_5000` proposal matching the CUDA smoke
closely (`0.907686` confidence,
`runs/cashsnap/browser_stack_onnx_detector_cpu_smoke_real_synth_p24_latest.json`).
This only proves the foundation can load on a CPU-style ONNX path; it is not a
browser/product promotion and does not address overlap/fan/hand readiness.

Seed repeat strengthens the synth+real recipe but does not promote seed1 as the
checkpoint. Seed1 fixed-step A/B, same b2/e1/steps318 recipe, beats duplicate
real control on full real `0.827864 -> 0.858237` (`+0.030374`, worst protected
class `USD_100 -0.017923`, pass):
`runs/cashsnap/real_data_label_audit_v1/fixed_step_real_p24_exposure_control_vs_real_p24_plus_strictbest_synth_p24_seed1_adaptive_summary_v1.json`
and
`runs/cashsnap/fixed_step_real_p24_exposure_control_vs_real_p24_plus_strictbest_synth_p24_steps318_seed1/summary.json`.
It also beats seed0 on semantic-clean (`0.883299 -> 0.888936`) and strict-clean
(`0.860743 -> 0.870491`), but fails the source-excluded strict-clean guard
against both seed0 (`0.769331 -> 0.713638`) and balanced real p24
(`0.760870 -> 0.713638`), with `KHR_500`, `KHR_10000`, and `USD_100` drops.
Therefore seed1 is recipe-repeat evidence, not the broad-guardrail model to
carry forward.

Overlap pivot call: seed0 synth+real is the detector checkpoint to use when the
work pivots into overlap/fan/hand experiments. This is not a product green
light; carry forward the known risks explicitly: seed-level source-excluded
instability, USD/high-value overproposal at low confidence, mixed-source label
trust, empty/unknown-money ambiguity, and official21 missing-class scope. If the
overlap phase needs all-riel coverage, start from the official21 review bridge
and reviewed missing-class labels before training a 21-class model.

Housekeeping read before overlap: `model.md` now treats non-overlap as parked,
not solved, and the overlap pivot as blocked first by validation quality. The
registered real geometry slice has only `5` val/test multi-note images with `11`
boxes, so it can expose failures but cannot promote a model. Use it as a smoke
alarm while building a larger real/semi-real overlap bridge.

Current strict synthetic-only best:
- Detector:
  `runs/cashsnap/fixed_step_scaled_foreignhardneg6_from_yolo26n_e50_i416_b64_w0_auto_lr1e2_warmup3_amp_cachefalse_steps1000_seed0/weights/best.pt`
- Full real test mAP50-95: `0.5035036831091516`.
- Historical gap to old entry target: `+0.316496` to `0.82`, `+0.346496` to
  `0.85`.
- Evidence bundle:
  `runs/cashsnap/final_synth_only_nonoverlap_phase_evidence_v1.json`.

What this proves: target-anchor scale/contact rendering plus six vetted
foreign-note hard negatives can move strict `yolo26n.pt` synthetic-only transfer
above the old floor (`0.420709 -> 0.503504`) while passing the current
per-class guard. It is a real mechanism clue, not a solved model.

What this does not prove: the detector is not close to the old clean-base entry
target and not product-ready. At `conf=0.05`, the strictbest lightweight eval still has
recall `0.6438`, precision `0.2321`, and background FPs on `516/748`
empty-label test images. Positive-only transfer is much better than older
blend185/hardnegold8 results (`0.696549` clean-visible, `0.610140` labeled-all),
but the aggregate full real test is still far short.

Active target clarification: this phase is about the best synth+real
non-overlap detector model, not the detector+gate product stack. Gate/browser
artifacts under `runs/cashsnap/product_threshold_sweep_v1/` are adjacent
diagnostics only. Do not use them as the yardstick for promoting the current
model recipe. The model yardstick is detector mAP/guardrails on full, semantic,
strict-clean, source-excluded, per-class, and low-confidence/background slices.

Audit-clean source-diverse real p24 is a serious detector challenger, but not
the current detector/AP winner: full `0.849920`, semantic-clean `0.884686`,
strict-clean `0.852600`, and strict no-`khmer_us_currency` `0.755911`. It beats
balanced real p24 on full/semantic-clean and even edges the synth+real detector
on semantic-clean, but it trails the synth+real champion on full, strict-clean,
and source-excluded strict-clean.

Do not use the old strict synthetic-only gap as the active progress meter for
the synth+real phase. It remains useful historical context for synthetic-data
quality: clean-checkpoint synthetic-data repair reached `0.747316` seed0 but was
guard-failing and not seed-repeated, while strict base-init generation remained
roughly `+0.32` to `+0.35` below the old aggregate target. Those numbers explain
why synthetic-only is not the current foundation; they do not define success for
the active p24 synth+real/real-label-cleanup work.

A useful candidate direction should answer at least one of these:
- Does it reduce real positive misses at usable confidence?
- Does it reduce giant/full-frame or empty-frame false positives?
- Does it protect weak/high-value classes instead of trading them away?
- Does it reduce real-vs-synth representation/domain separability for the right
  reason?
- Does it expose a missing real validation bridge, label policy, or harness
  limitation that must be fixed before scale?

Working posture:
- Be a good researcher, not a comfortable executor.
- Be bold but bounded. Prefer experiments that can fail loudly and teach the next
  direction over safe micro-tweaks that only make one proxy look tidier.
- Be willing to redo, restructure, or replace the harness when the harness blocks
  the right question.
- Start every non-trivial next step from the live yardstick: current promoted
  foundation, matched control, source/strict/low-confidence guardrails, and the
  final mixed-cash/overlap-readiness goal. Treat small aggregate gains as
  bottleneck clues, not wins, unless they also reduce a real failure mode without
  weakening protected classes or background behavior.
- Before running another safe continuation, propose then attack it: name the
  assumption, prior warning, proxy-failure risk, kill condition, and what would
  count as a real step-change. Prefer brave, testable mechanism bets over
  polishing a weak path.
- Evaluate ideas as go/stop bets, not indefinite optimization tracks. Try an
  idea hard enough to see whether it can produce a big-step mechanism; if it is
  merely promising-but-slow, or only yields small local improvements, leave it
  documented and move on.
- For long training/render/eval jobs, choose command posture deliberately:
  use `rl <cmd>` by default when the wrapper does not mangle arguments or stdin.
  For long runs, launch the real command in the background with stdout/stderr
  logs, then use quick poll/check commands while doing housekeeping; avoid
  foreground `Start-Sleep` polling except as a separate idle wait. Known `rl`
  exceptions: PowerShell command-boundary quirks, stdin scripts, and comma-valued
  args that the wrapper rewrites. Do not stack competing GPU-heavy jobs unless
  the experiment explicitly needs it.
- Visual QA note: use vision deliberately. Prefer opening several clear,
  full-size/simple-scene images over relying on one compressed contact sheet,
  because small contact-sheet tiles hide rendering flaws and make visual
  reasoning harder.
- Research first when the path is fuzzy: read the code, docs, prior artifacts,
  papers, or web sources as needed. Preserve only conclusions that change a
  decision.
- Run a Builder/Skeptic pass before non-trivial direction changes: why could this
  create a regime change, and why might it be too small, misleading, already
  disproven, or proxy work?
- Ask the uncomfortable questions: What would make this fail? Has that already
  happened? Are we measuring what is easy instead of what matters? What simple
  obvious idea are we avoiding? What would collapse the most uncertainty?
- Do not chase pretty contact sheets, row-count comfort, or one-seed wins.
  Promotion is real/deploy utility under guardrails.

## Research Frame

### Current State

Active model line: p24 balanced-real + strictbest-synth seed0 is the promoted
clean/non-overlap detector foundation and checkpoint to beat. The latest
challenger, USD-total empty24, is killed for promotion: it improved full and
strict AP but failed source-excluded and low-confidence recall guardrails. Keep
it as a clue that USD/unknown-money negatives can reduce overproposal, not as a
foundation upgrade.

High-priority next exploration: augment reviewed real notes/contexts into
label-preserving half-synthetic material, including multi-note collage/fan/overlap
layouts. Do not let this get buried behind more row-dose tweaks. The cheap
built-in YOLO mosaic probe was neutral, and the first rectangular real-crop fan
probe plus a small accepted WebGL stack/fan dose were harmful, but those do not
kill the underlying bridge; the next version needs masked/audited note assets or
real captures, source-aware unknowns, a schedule that does not hammer weak KHR
classes, and overlap/counting validation. Step-back stress guardrails on the
WebGL stack/fan cap6 candidate show only a tiny geometry-stress AP clue, not a
foundation rescue: geometry-stress test `0.891355 -> 0.892408`, but
protected-riel test `0.651747 -> 0.644157` versus the champion.

Historical synth-only context: the final synth-only non-overlap phase result is
chosen. Strictbest foreign-hardneg6 remains the best verified guardrailed
detector. The updated handoff is
`runs/cashsnap/final_synth_only_nonoverlap_handoff_ready_v1.md`.

The latest directly connected `usd_total` support attempt is diagnostic, not
promoted. The b64 isolate had fair preflight parity but hit the RAM guard at
epoch `2/50`. A corrected memory-safe `yolo26n.pt` b48/e38 A/B completed with
row/phase parity and lifted its own baseline `0.497016 -> 0.513584`, but failed
promotion: `KHR_10000 -0.064772`, `KHR_50000 -0.343211`, and low-conf
background-FP images worsened `516/748 -> 569/748`. Its product stack reached
`0.8140`/`0.4419`/`1010`, versus strictbest stack
`0.7809`/`0.4620`/`1021`. Source split shows the trade: `usd_total` recall
improves `0.4921 -> 0.7165` and `billsbank` `0.6038 -> 0.6792`, while
`khmer_us_currency` recall drops `0.9383 -> 0.8333` and empty-source overfire
worsens. Use this as a USD/source-context clue, not a final detector.

Controlled synth+real p24 is the first real blend signal worth chasing. Config:
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_v1.yaml`
adds current strictbest synthetic p24/class plus six hard negatives to balanced
real p24/bg24. Its exact duplicated-real exposure control is
`configs/webgl_ablation/cashsnap_balanced_real_p24_exposure_control_for_strictbest_synth_p24_v1.yaml`.
Phase-matched b2/e1/steps318 from the clean checkpoint beat the duplicated-real
control on full real test mAP50-95 `0.830854 -> 0.852767` and beat the original
balanced real-only reference `0.835861 -> 0.852767`, with worst per-class AP
drop inside the `0.05` guard. Lightweight `conf=0.05` guard passed versus the
exposure control (`recall -0.008568`, FP `-75`, background-FP images `-26`) but
failed versus original balanced real-only because background-FP images worsened
`151 -> 169`. Against balanced real on audit slices, semantic-clean recall is
unchanged but FP rises `+27`, mostly high-value/US-dollar classes; strict-clean
recall improves `+0.0066` while FP rises `+7` and background-FP images drop `-5`.
So the next detector-side repair is USD overproposal/background cleanup while
keeping the synthetic rows, not removing them. FP-delta review packs:
`runs/cashsnap/real_data_label_audit_v1/light_fp_delta_semantic_clean_balanced_real_p24_vs_real_synth_p24_conf005_v1.{json,csv,jpg}`
and
`runs/cashsnap/real_data_label_audit_v1/light_fp_delta_strict_clean_balanced_real_p24_vs_real_synth_p24_conf005_v1.{json,csv,jpg}`.
They show semantic-clean FP growth is concentrated in `usd_total` (`+29`) and
USD classes (`USD_50 +20`, `USD_5 +12`, `USD_20 +11`, `USD_1 +11`,
`USD_100 +9`); strict-clean is smaller (`USD_50 +10`, `USD_20 +8`, `USD_1 +6`,
`USD_100 +4`) while reducing background-FP images. Visual sheets show many rows
are duplicate/localization/class-fragment proposals around real notes plus some
unknown/coin/background overfires. Standard YOLO NMS is killed as a repair:
`nms_iou=0.45` and `0.20` produced identical raw predictions/TP/FP/FN on both
semantic-clean and strict-clean synth+real light evals. The next repair should
move beyond row-dose empties toward reviewed unknown/out-of-scope labels,
source-aware objectives, or proposal architecture; generic and source-specific
empty hard negatives have both failed as promotion paths. Adjacent
product/gate diagnostic only: full real gate/reclassifier probes trailed
balanced real-only
(`1209 -> 1199` gate-only exact-value, `1184 -> 1173` post-reclassifier
exact-value), but this is not the active detector-model yardstick.

USD-high-value real-dup swap is killed as a detector improvement. Config:
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_usdhi_realdup_v1.yaml`
replaces the `USD_50`/`USD_100` strictbest-synth extra rows with duplicated real
high-value USD rows under the same 635-row exposure. Fixed-step summary:
`runs/cashsnap/real_data_label_audit_v1/fixed_step_real_p24_plus_strictbest_synth_p24_usdhi_realdup_vs_dupctrl_v1_summary.json`.
It still beats duplicate-real control (`0.830854 -> 0.844567`, worst class
`USD_100 -0.029293`, guard passes) but trails the synth+real champion
`0.852767`. Lesson: do not protect USD_50/USD_100 by simply removing their synth
rows; the all-class strictbest synth mix remains the detector model to beat.

Audit-filtered eval confirms the synth+real detector signal as the current
model yardstick, not just a noisy full-test bump. Detector mAP50-95 by slice:
full real `0.835861/0.852767/0.846769`, semantic-clean
`0.878936/0.883299/0.880504`, and semantic+leakage-clean
`0.855619/0.860743/0.857153` for balanced real p24 / synth+real p24 /
synth+real then real recalibration. Pairwise JSONs live under
`runs/cashsnap/real_data_label_audit_v1/eval_compare_*_v1.json`. Current model
call: synth+real p24 is the best detector/AP candidate. Do not run more blind
blend tweaks unless they are judged against this detector champion on full,
semantic, strict, source-excluded, per-class, and low-conf/background slices.

Seed1 repeat confirms the p24 synth+real recipe is useful but not source-stable
enough to promote over seed0. Seed1 beat duplicate-real control on full
(`0.827864 -> 0.858237`) and seed0 on strict-clean (`0.860743 -> 0.870491`),
but failed no-`khmer_us_currency` test (`0.769331 -> 0.713638`). That test slice
has only `151` boxes with one `KHR_500` and one `KHR_10000`; the combined
val+test diagnostic
`configs/audit/cashsnap_v1_semantic_plus_leakage_clean_no_khmer_us_currency_valtest_eval_v1.yaml`
is less jumpy and gives balanced/seed0/seed1 `0.787549/0.782187/0.786081`.
Treat source-excluded AP as a class/source diagnostic, not a solo promotion gate;
the stable actionable weakness is USD_10, while KHR_500 is too tiny to steer
training by itself.

USD_10-only real-dup protection is killed as a promotion path. Config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_usd10_realdup_v1.yaml`
replaces the `24` synthetic USD_10 rows with duplicated real USD_10 rows. It
beats matched duplicate-real control on full test (`0.830854 -> 0.845872`) and
nudges strict-clean over seed0 (`0.860743 -> 0.863296`), but trails the p24
champion on full (`-0.006895`) and source-excluded combined (`0.782187 -> 0.779782`);
USD_10 improves only `+0.010995` vs seed0 there and remains below balanced by
`-0.038681`. Summary:
`runs/cashsnap/real_data_label_audit_v1/fixed_step_real_p24_plus_strictbest_synth_p24_usd10_realdup_vs_dupctrl_v1_summary.json`.
Lesson: replacing one vulnerable synth class with real duplicates is too narrow
and shifts damage into KHR/USD_50 rather than fixing source stability.

Half-dose strictbest synth p12 is killed. Config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p12_v1.yaml`
uses the same synthetic source capped at `12` rows/class plus `6` empty rows and
was trained for the same `318` optimizer steps as p24 against exact duplicate-real
control
`configs/webgl_ablation/cashsnap_balanced_real_p24_p12_exposure_control_for_strictbest_synth_p12_v1.yaml`.
It fails even that matched control on full test (`0.836945 -> 0.835922`) and is
far below the p24 champion, so do not chase lower synth dose as the next repair
unless the sampling/source design changes materially. Summary:
`runs/cashsnap/real_data_label_audit_v1/fixed_step_real_p24_plus_strictbest_synth_p12_vs_dupctrl_steps318_v1_summary.json`.

Low-risk empty24 hard-negative dose is killed. Config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_lowrisk_empty24_v1.yaml`
adds `24` train-only zero-label rows from
`runs/cashsnap/real_data_label_audit_v1/candidate_empty_train_lowrisk_no_teacher_unmatched_v1.txt`
to the p24 synth+real champion. The matched control
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_empty24_exposure_control_v1.yaml`
duplicates `24` existing empty rows so the empty count is equal. The candidate
beats that duplicate-empty control on full test (`0.847448 -> 0.849864`) but
fails the per-class guard (`KHR_50000 -0.064949`) and trails the actual p24
champion (`0.852767 -> 0.849864`, `KHR_50000 -0.098940`). Summaries:
`runs/cashsnap/real_data_label_audit_v1/fixed_step_real_synth_p24_lowrisk_empty24_vs_emptydupctrl_v1_summary.json`
and
`runs/cashsnap/real_data_label_audit_v1/eval_compare_full_real_synth_p24_seed0_vs_lowrisk_empty24_seed0_v1.json`.
Lesson: generic low-risk empty/background pressure can help slightly over
duplicated empty exposure, but it is too blunt and damages weak high-value KHR.
Future background/objectness repair should be source- or failure-mode-specific,
with `KHR_50000` guarded explicitly.

USD-total empty24 hard-negative dose is killed as a foundation upgrade, but kept
as a mechanism clue. Config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_usdtotal_empty24_v1.yaml`
adds `24` train-only zero-label `usd_total` rows from the low-risk empty pool;
matched control
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_usdtotal_empty24_exposure_control_v1.yaml`
duplicates existing empty rows. It passes matched duplicate-empty control
(`0.847448 -> 0.856213`), direct full comparison to seed0
(`0.852767 -> 0.856213`), and strict-clean (`0.860743 -> 0.865608`), but fails
source-excluded combined (`0.782187 -> 0.759560`, `KHR_2000 -0.254823`,
`USD_100 -0.083230`). Low-conf scorecard versus seed0 also fails on recall:
semantic-clean recall/precision/bg-FP images `0.9515/0.6511/36` ->
`0.9403/0.6762/30`; strict-clean `0.9639/0.6296/20` ->
`0.9607/0.6356/21`. Summary/scorecard:
`runs/cashsnap/real_data_label_audit_v1/fixed_step_real_synth_p24_usdtotal_empty24_vs_emptydupctrl_v1_summary.json`
and
`runs/cashsnap/real_data_label_audit_v1/light_scorecard_real_synth_p24_seed0_vs_usdtotal_empty24_conf005_v1.json`.
Lesson: source-specific non-target USD pressure can cut false positives and
raise AP, but the row-dose form trades away source/KHR robustness and recall.
Do not promote or sweep more empty-dose variants without a structural
target-vs-unknown objective or reviewed labels.

Basic same-data YOLO mosaic/collage augmentation is neutral, not a promotion
path. Fixed-step A/B kept the p24 synth+real config unchanged but forced the
candidate to `mosaic=1.0`/`close_mosaic=0`:
`runs/cashsnap/real_data_label_audit_v1/fixed_step_real_synth_p24_mosaic_active_vs_closedmosaic_v1_summary.json`.
Candidate weights differed from the champion, but full box metrics and every
per-class mAP50-95 delta were exactly `0.0`
(`0.8527673041457657` both ways). Lesson: random built-in mosaic on the current
mostly single-note p24 rows is not enough to expose the multi-bill/overlap
potential; a serious collage test needs generated/audited half-synth scenes and
counting/overlap-specific validation.

Literal 3x3 image-grid packing is killed as a foundation upgrade, but it sharpens
the half-synth read. Registered diagnostic roots
`data/processed/cashsnap_grid3x3_real6_synth3_p24_diagnostic_v1` and
`data/processed/cashsnap_grid3x3_real9_p24_control_v1` were generated by
`scripts/build_yolo_grid_collage_dataset.py`: `24` grid images each, `216` boxes
each, one-box p24 tiles only. The `6 real + 3 synth` grid did not beat the
matched `9 real` grid control on full real test (`0.854262 -> 0.852556`) and did
not beat the p24 champion directly (`0.852767 -> 0.852556`). The `9 real` grid
control showed a tiny full-test bump over champion (`0.852767 -> 0.854262`) but
failed guardrails: strict-clean `0.860743 -> 0.858194`, and source-excluded
combined `0.782187 -> 0.750058` with `KHR_2000 -0.227770` and three per-class
failures. Summaries:
`runs/cashsnap/real_data_label_audit_v1/fixed_step_real_synth_p24_grid3x3_real6synth3_vs_real9ctrl_v1_summary.json`,
`runs/cashsnap/real_data_label_audit_v1/eval_compare_full_real_synth_p24_seed0_vs_grid3x3_real6synth3_seed0_v1.json`,
`runs/cashsnap/real_data_label_audit_v1/eval_compare_full_real_synth_p24_seed0_vs_grid3x3_real9_control_seed0_v1.json`,
`runs/cashsnap/real_data_label_audit_v1/eval_compare_strict_clean_real_synth_p24_seed0_vs_grid3x3_real9_control_seed0_v1.json`,
and
`runs/cashsnap/real_data_label_audit_v1/eval_compare_strict_clean_no_khmer_us_currency_valtest_real_synth_p24_seed0_vs_grid3x3_real9_control_seed0_v1.json`.
Lesson: packing more bills per training image can move a full-test proxy, but
literal grids create scale/seam/source artifacts and the synthetic-in-grid blend
does not hide synth domain cues usefully. Do not chase 4x4/5x5 grids. Rework the
idea as audited, real-anchored half-synth scenes with natural spatial jitter,
partial overlap/fan geometry, and a validation slice that actually contains
multi-note/counting stress.

Rectangular real-crop fan/overlap composites are killed as a foundation upgrade.
Dataset `data/synthetic/cashsnap_realcrop_fan_overlap_train24_v1` was generated
from the p24 real train crop bank plus strict no-note backgrounds after adding
`--source-alpha-policy opaque` to `scripts/generate_synthetic_fan_dataset.py`.
Visual QA no longer had torn alpha holes, but the scenes still looked like
rectangular crop patches. The `48`-image fan mix lost to its duplicate-exposure
control (`0.835053 -> 0.830359`, `KHR_50000 -0.094243`) and lost badly against
the p24 champion (`0.852767 -> 0.830359`, `KHR_50000 -0.224346`). Even the
duplicate-control exposure variant trailed the champion (`0.852767 -> 0.835053`).
Summaries:
`runs/cashsnap/real_data_label_audit_v1/fixed_step_real_synth_p24_realcrop_fan48_vs_dupctrl_v1_summary.json`,
`runs/cashsnap/real_data_label_audit_v1/eval_compare_full_real_synth_p24_seed0_vs_realcrop_fan48_seed0_v1.json`,
and
`runs/cashsnap/real_data_label_audit_v1/eval_compare_full_real_synth_p24_seed0_vs_realcrop_fan48_dupctrl_seed0_v1.json`.
Lesson: simple rectangular crop compositing teaches the wrong seam/scale/context
signals and can damage weak high-value KHR. Keep the generator patch for
controlled diagnostics, but do not scale this asset form; the half-synth bridge
requires real captures, better cut masks, or an explicitly reviewed source-note
policy.

Small accepted WebGL stack/fan dose on top of the current p24 synth+real champion
is also killed for this schedule. Registered diagnostic roots
`data/synthetic/cashsnap_webgl_overlap_stack_candidate_v1` and
`data/synthetic/cashsnap_webgl_fan_fullschema_candidate_v1` pass the WebGL
trainable-candidate suite, but a cap6 stack+fan mix added only `16` images and
still failed. Candidate config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_webgl_stackfan_cap6_v1.yaml`
lost to its duplicate-exposure control (`0.829316 -> 0.822214`,
`KHR_50000 -0.086328`) and to the p24 champion (`0.852767 -> 0.822214`,
`KHR_50000 -0.262497`). The duplicate-exposure control itself also trailed the
champion (`0.852767 -> 0.829316`). Summaries:
`runs/cashsnap/real_data_label_audit_v1/fixed_step_real_synth_p24_webgl_stackfan_cap6_vs_dupctrl_v1_summary.json`,
`runs/cashsnap/real_data_label_audit_v1/eval_compare_full_real_synth_p24_seed0_vs_webgl_stackfan_cap6_seed0_v1.json`,
and
`runs/cashsnap/real_data_label_audit_v1/eval_compare_full_real_synth_p24_seed0_vs_webgl_stackfan_cap6_dupctrl_seed0_v1.json`.
Lesson: exact-mask WebGL overlap/fan assets are better than crop rectangles, but
naively dosing them into the current clean-checkpoint p24 blend still trips the
same weak-KHR failure mode. Do not keep scaling stack/fan rows on this schedule;
use WebGL next only with a real overlap validation bridge, KHR-protected
curriculum/sampling, or a staged objective that is explicitly judged on counting
stress rather than clean AP alone.

Stress guardrail follow-up confirms this is not just over-optimizing clean AP.
On mined real geometry/protected-riel slices, WebGL stack/fan cap6 barely edges
the champion on geometry-stress test mAP50-95 (`0.891355 -> 0.892408`) but loses
protected-riel test (`0.651747 -> 0.644157`, `KHR_50000 -0.033339`). Against its
duplicate-exposure control it does show a weak positive stress signal
(`0.889530 -> 0.892408` geometry, `0.619795 -> 0.644157` protected riel), so
keep the asset idea as a staged/KHR-protected clue, not as current-foundation
training material. Artifacts:
`runs/cashsnap/real_geometry_stress_slices_v1/webgl_stackfan_cap6_vs_real_synth_p24_stress_guardrail_v1.json`
and
`runs/cashsnap/real_geometry_stress_slices_v1/webgl_stackfan_cap6_vs_dupctrl_stress_guardrail_v1.json`.

Tiny real multi-note smoke scorecard
`runs/cashsnap/real_geometry_stress_slices_v1/light_scorecard_multi_note_overlap_smoke_v1.json`
scores only `5` val/test images / `11` boxes, so it is not promotion authority.
At `conf=0.05`, p24 synth+real and balanced-real both hit combined recall
`7/11 = 0.6364`; p24 synth+real has better precision (`0.3889` vs `0.3500`).
The old WebGL overlap-stack candidate is worse (`6/11 = 0.5455`, precision
`0.3000`). The val half is KHR-heavy and exposes fragile fanned/stacked KHR
behavior, while test is less bad. Current read: do not move to overlap training
until the validation bridge grows beyond this smoke slice, and do not assume
existing WebGL overlap assets transfer to real multi-note behavior.

Real overlap/fan review bridge v1
`scripts/build_real_overlap_review_queue.py` ranks real CashSnap images by
multi-note, bbox-overlap, tight-pair, partial-edge, protected-Riel, and repeated
class signals, then writes both raw-image and canonical-cluster queues with
visual sheets. Run
`runs/cashsnap/real_overlap_review_queue_v1/summary.json` found `6043` raw
candidate image variants but only `3205` canonical clusters; raw `94`
multi-note variants collapse to `48` canonical multi-note clusters, `52`
bbox-overlap variants to `21` clusters, and `70` tight-pair variants to `36`
clusters. The top cluster sheet visually contains real fanned/stacked notes,
hands, table/context shots, and partial-edge cases, but also duplicate
augmentation variants and some ordinary flat/repeated notes. Treat
`review_clusters.csv` as the next overlap-validation review entrypoint, not
training data and not a promotion set. Rows need visual decisions such as
`trusted_overlap_eval`, `train_anchor_candidate`, `partial_policy_unclear`, or
`exclude_duplicate_or_flat` before they can become an eval bridge or
half-synthetic anchors.

The first balanced review packet is
`runs/cashsnap/real_overlap_review_queue_v1/first_review_clusters_balanced_v1.csv`
with sheet
`runs/cashsnap/real_overlap_review_queue_v1/first_review_clusters_balanced_v1_sheet.jpg`.
The queue builder also writes
`first_review_clusters_balanced_v1_images.txt` and diagnostic
`first_review_clusters_balanced_v1_data.yaml` so model-error review can run over
the packet without hand-built configs. `scripts/build_real_overlap_review_diagnostic_views.py`
then splits an annotated review CSV into representative-only diagnostic YOLO
views by held-out split and packet bucket; these views are still not promotion
configs, but they prevent train-row triage from being confused with held-out
evidence.
After review, materialize it through
`scripts/materialize_real_overlap_review.py`: it ignores blank rows, fails by
default when nothing reviewed is materialized, and only writes empty dry
manifests with `--allow-empty`. Default materialization keeps
`trusted_overlap_eval` representative-only and held out to val/test, while
`train_anchor_candidate` and `hard_negative_context` expand only train variants.
Smoke check
`runs/cashsnap/real_overlap_review_materialized_smoke_v1/summary.json`
materialized `2` reviewed clusters into `4` images (`3` train anchors and `1`
held-out eval representative) and wrote diagnostic YOLO views including
`active_eval_data.yaml` for `eval_yolo_lightweight_real_recall.py --split test`.
Mock-materializing the unreviewed `val_test_multi_note_smoke` packet through the
same bridge reproduced the old 5-image recall (`7/11 = 0.6364`) for the current
p24 synth+real champion, with precision `0.3684` (`12` FP). Against balanced
real p24 on the same bridge, aggregate recall ties and FP drops by `1`, but the
scorecard still fails per-class because the single `KHR_1000` GT goes from
`1/1` to `0/1`; tiny sample, useful warning. This verifies the bridge plumbing,
not model promotion. Visual error triage is in
`runs/cashsnap/real_overlap_review_materialized_valtest_mock_v1/positive_error_review_balanced_vs_champion/`;
use it to inspect KHR partial/fan denomination confusions before designing any
new overlap dose. `scripts/merge_overlap_review_model_errors.py` can merge those
error rows back into a review CSV; the mock merge wrote
`runs/cashsnap/real_overlap_review_queue_v1/first_review_clusters_balanced_v1_model_error_triage_mock.csv`
with `4/120` rows carrying model-error triage columns.
Scoring the old WebGL stack/fan cap6 and real-crop fan48 checkpoints through the
same 5-image mock bridge did not revive them: both tie champion recall at
`0.6364` but add `+1` FP/prediction and fail the scorecard
`runs/cashsnap/real_overlap_review_materialized_valtest_mock_v1/scorecard_champion_vs_old_overlap_candidates_valtest_mock.json`.
Full first-packet diagnostic triage over all 120 images remains useful but is
split-mixed, not a held-out verdict: balanced-real p24 has `157/186` TP, `288`
predictions, `18` missed GT, `120` unmatched FP, and `11` wrong-class errors;
p24 synth+real has `143/186` TP, `229` predictions, `32` missed GT, `75`
unmatched FP, and `11` wrong-class errors, with weak `KHR_5000` recall (`5/12`).
The merged triage CSV
`runs/cashsnap/real_overlap_review_queue_v1/first_review_clusters_balanced_v1_model_error_triage.csv`
annotates `79/120` first-packet rows.

Diagnostic split views under
`runs/cashsnap/real_overlap_review_diagnostic_views_v1/` correct the read. On
held-out representatives (`64` val/test images, `70` boxes), balanced-real and
p24 synth+real both hit `62/70` TP (`0.8857` recall), while p24 synth+real is
quieter (`40` FP vs `62`, precision `0.6078` vs `0.5000`). The held-out
model-error subset also ties recall (`33/41`) with fewer p24 synth+real FPs
(`40` vs `62`). The scorecard still fails per-class on tiny pockets
(`KHR_1000`/`USD_5` one-box drops and `KHR_50000` `17/19 -> 16/19` on the
held-out error subset), so this is not promotion evidence.

The real recall hole is train-only hard geometry that still needs visual
review/materialization before training: train representatives are `95/116` TP
for balanced-real versus `81/116` for p24 synth+real; bbox-overlap is `28/37 ->
24/37`, tight-pair is `36/41 -> 27/41`, and the worst class pockets include
`KHR_5000` plus USD tight-pair rows such as `USD_20`. Held-out partial-edge and
remaining-overlap views are recall-safe or better for p24 synth+real; the small
held-out cashcountingxl USD context remains a recall warning (`5/7 -> 4/7`,
with fewer FPs). Read: do not train or promote from the unreviewed queue, and do
not treat the all-120 diagnostic as a held-out failure. The next detector dose
needs reviewed bbox/tight-pair train anchors and a reviewed USD cash-counting
eval pocket, while preserving the champion's lower FP behavior.

Focused follow-up packet
`runs/cashsnap/real_overlap_focus_review_packet_v1/focus_review_packet_v1.csv`
is built by `scripts/build_real_overlap_focus_review_packet.py` from the
annotated first-packet triage. It has `43` rows: `14` train bbox-overlap anchor
candidates, `9` train tight-pair anchor candidates, `8` `khmer_us_currency`
tight-pair flat-source policy rows suggested as `exclude_duplicate_or_flat`, `7`
held-out cashcountingxl USD eval candidates, `4` held-out multi-note smoke eval
candidates, and `1` held-out `khmer_us_currency` multi-note flat-source policy
row, with sheet `focus_review_packet_v1_sheet.jpg`. The script adds
`suggested_usable_as` and `suggested_final_route` but leaves
`review_decision`/`usable_as` blank; the materializer correctly refuses it unless
rows receive explicit accepted review decisions. Use this as the next review
entrypoint before any overlap dose, and do not let flat front/back catalog pairs
stand in for real overlap/fan anchors.

The same builder writes diagnostic route views, still not accepted materialized
data: `suggested_eval_view_v1_data.yaml` (`11` held-out candidates),
`suggested_train_anchor_view_v1_data.yaml` (`23` train-anchor candidates), and
`flat_source_policy_view_v1_data.yaml` (`9` likely flat/catalog policy rows).
On suggested eval, balanced-real versus p24 synth+real is `10/16` TP, `25` FP
versus `9/16` TP, `16` FP, so the champion is quieter but loses one TP
(`KHR_1000`, `USD_1`, `USD_5` pockets). On suggested train anchors, the recall
hole survives flat-source removal: `48/60` TP, `39` FP versus `41/60` TP, `29`
FP, led by `KHR_5000` (`11/12 -> 5/12`). This makes reviewed `KHR_5000`
overlap/fan anchors the highest-value train-side question.
Threshold probing on the same packet says this is not fixed by a simple global
confidence knob: champion `conf=0.05` gives recall/precision `0.7688/0.6245`,
`0.03` gives `0.8172/0.5547`, `0.02` gives `0.8333/0.4874`, and `0.01` gives
`0.8763/0.3638`, versus balanced-real p24 at `0.8441/0.5451` for `conf=0.05`.
Lowering only `KHR_5000` to the `0.01` floor recovers little (`0.7849/0.6160`);
lowering the weak set (`KHR_1000`, `KHR_5000`, `KHR_10000`, `USD_20`, `USD_5`)
gets to `0.8333/0.5065` but creates heavy `KHR_1000/KHR_10000` FP pressure.
Use thresholds as diagnostics/product knobs only; the training/data issue is
recall-safe overlap/partial evidence without uncontrolled FP expansion.
Do not train or promote from the queue before review and this strict
materialization step.

Audit-clean sourcecap48 real p24 proves data trust/source diversity can rival
the synth+real detector without adding synthetic rows. Config:
`configs/audit/cashsnap_v1_auditclean_real_p24_bg24_sourcecap48_v1.yaml`, built
by `scripts/build_audit_clean_balanced_real_config.py` from provisional clean
positive anchors plus low-risk empty candidates, with a soft `48` positive/source
cap. The selected train list is `316` images/`24` backgrounds/`308` boxes; source
counts are `billsbank 60`, `cashcountingxl 58`, `khmer_us_currency 49`,
`usd_total 49`, `cambodia_currency_project 40`, `khmer 32`, `asian_currency 28`.
Detector mAP50-95 by slice is full `0.849920`, semantic-clean `0.884686`,
strict semantic+leakage-clean `0.852600`, and strict-clean without
`khmer_us_currency` `0.755911`. This beats balanced p24 on full and
semantic-clean, trails synth+real on full/strict/no-khmer, and trails balanced
p24 on strict/no-khmer. Low-conf `conf=0.05` is product-mixed: full
recall/precision/bg-FP `0.9510`/`0.5734`/`148`, strict-clean
`0.9639`/`0.6176`/`34`, so raw strict-clean overfire is worse than balanced
(`25`) and synth+real (`19`).

Adjacent system note only: sourcecap48 plus the true-empty gate is a count/value
tradeoff, not the active model yardstick. Scorecard:
`runs/cashsnap/real_data_label_audit_v1/auditclean_sourcecap48_detector_gate_scorecard_v1.json`.
Full real gate-only is recall/precision/bg-FP/exact-value
`0.9510`/`0.6167`/`85`/`1216`; balanced p24 is
`0.9486`/`0.6215`/`79`/`1209`, and synth+real is
`0.9474`/`0.5991`/`88`/`1199`. Strict-clean gate-only is
`0.9639`/`0.6667`/`16`/`671`; balanced is
`0.9574`/`0.6606`/`13`/`664`, and synth+real is
`0.9639`/`0.6419`/`13`/`665`. Per-image strict-clean post-gate comparison to
balanced p24 gives exact net `+7` and weighted net `+6` over `102` changed rows;
`khmer_us_currency` is `+3` exact / `+5` weighted, while `cashcountingxl` is
`+2` exact / `-6` weighted. This says source-diverse audit-clean real sampling
is a real mechanism, but it belongs to detector+gate system selection. The threshold sweep
`runs/cashsnap/product_threshold_sweep_v1/full_real_balanced_vs_sourcecap48_threshold_sweep_v1.json`
shows sourcecap48 `reject>=0.80` can beat tuned balanced on exact-value
(`1243` vs `1232`) and KHR MAE (`813` vs `1302`) at similar bg-FP (`59` vs
`58`), but tuned balanced is safer on full USD MAE (`7.03` vs `8.34`) and
`USD_100` recall (`0.874` vs `0.857`). Keep this out of model promotion unless
the phase explicitly switches back to product-stack selection.

Adding current strictbest synth to sourcecap48 real p24 is killed as tested, and
is useful mainly as a mechanism clue. Config:
`configs/audit/cashsnap_auditclean_sourcecap48_real_p24_plus_strictbest_synth_p24_v1.yaml`
was compared against exact duplicate-real exposure control
`configs/audit/cashsnap_auditclean_sourcecap48_real_p24_exposure_control_for_strictbest_synth_p24_v1.yaml`
with fixed b2/e1/steps317 from the clean checkpoint. Summary:
`runs/cashsnap/real_data_label_audit_v1/fixed_step_auditclean_sourcecap48_real_plus_synth_vs_dupctrl_v1_summary.json`.
Full real mAP50-95 moved only `0.839567 -> 0.840959` (`+0.001392`) and the
protected per-class guard failed on `KHR_50000` (`0.728696 -> 0.636870`,
`-0.091826`). Do not run more sourcecap48+synth row-mix probes until the rare
KHR_50000 risk and product bg-FP trade are designed for explicitly.

Cleaner product probe nuance: on strict semantic+leakage-clean test, the
true-empty gate/reclassifier stack made synth+real look stack-compatible but not
clearly better. Balanced real p24: pre-gate recall/precision/bg-FP
`0.9574`/`0.6320`/`25`, post-gate `0.9574`/`0.6606`/`13`, post-reclassifier
`0.9311`/`0.6425` with `661` exact-value images. Synth+real p24:
pre-gate `0.9639`/`0.6296`/`19`, post-gate `0.9639`/`0.6419`/`13`,
post-reclassifier `0.9377`/`0.6245` with `663` exact-value images. This keeps
synth+real alive for a cleaned, product-gated bridge, but it is too small and
slice-specific to override the full-test product call. Per-image comparison
(`runs/cashsnap/real_data_label_audit_v1/proposal_gate_strict_clean_balanced_real_p24_vs_real_synth_p24_per_image_compare_v1.json`)
shows the exact-value edge is a churny `34` fixes vs `32` breaks, with
`khmer_us_currency` contributing both the most exact wins (`10`) and exact
losses (`15`); inspect the companion changed-image CSV/sheet before trusting the
edge. Treat this as evidence that source cleanup/class audit must come before
another model-promotion claim. Source exclusion sharpens the read: excluding
`khmer_us_currency`, synth+real improves strict-clean post-reclassifier
exact-value `543 -> 550`; within `khmer_us_currency`, it regresses
`118 -> 113`. The blend may be useful, but the mixed-source label problem is
actively masking or distorting the product signal.
Focused `khmer_us_currency` churn artifacts are
`proposal_gate_strict_clean_khmer_us_currency_balanced_real_p24_vs_real_synth_p24_per_image_compare_v1.json`,
the matching changed-image CSV, and the matching sheet in the same audit folder;
they show synth+real exact net `-5` and weighted net `-4` over `153` shared
source images.
Detector AP agrees directionally on the same source-excluded view:
`configs/audit/cashsnap_v1_semantic_plus_leakage_clean_no_khmer_us_currency_eval_v1.yaml`
gives synth+real mAP50-95 `0.760870 -> 0.769331`, but with a precision/recall
trade (`0.673`/`0.818` vs `0.870`/`0.701`). This is a thin diagnostic slice
(`151` boxes, no `KHR_2000` test boxes), not promotion authority.
Outside `khmer_us_currency`, the strict-clean stack mechanism is: synth+real
has better raw proposals (`543` vs `532` exact-value, bg-FP `19/471` vs
`25/471`), the gate equalizes bg-FP at `13/471`, and post-reclassifier exact is
`550` vs `543` while precision remains lower (`0.5134` vs `0.5240`). The
promising part is proposal/background behavior, not denomination
reclassification.
The next product-bridge review entrypoint is
`runs/cashsnap/real_data_label_audit_v1/product_bridge_review_queue_v1/`:
`162` deduped rows with a review-ready CSV and sheet, seeded from strict-stack
`khmer_us_currency` churn, KHR_100/schema rows, empty-label target suspects, and
mixed-source ranked review rows.
Do not waste a run just dropping `khmer_us_currency` from the current p24
synth+real train mix: only `13/635` train rows come from that source and they
are all `KHR_2000`. The measured problem is primarily eval/source-label trust
and product-stack churn, not heavy train-source exposure in this recipe.

High-value/protected positive error review v1 is superseded. It exposed a real
harness bug in `scripts/build_yolo_positive_error_review.py`: feeding Ultralytics
a text-file image list let results return in a different order than the source
list, so zipping `images` to `results` created false missed-GT rows. The script
now runs explicit batches and preserves batch order; `py_compile` passes. Do not
use `positive_error_review_highvalue_khr_compare_v1/` or
`shared_error_eval_slice_highvalue_khr_v1/` as evidence.

Corrected high-value/protected review v2 changes the bottleneck diagnosis. Review
pack `runs/cashsnap/real_data_label_audit_v1/positive_error_review_highvalue_khr_compare_v2/`
compares the p24 synth+real champion, balanced-real p24, and audit-clean
sourcecap48 at `conf=0.05` on full val+test. Weak-KHR recall is mostly strong
once the harness is fixed: p24 synth+real gets test `KHR_2000 20/20`,
`KHR_10000 26/27`, `KHR_20000 17/17`, `KHR_50000 10/10`, and val
`KHR_2000 45/45`, `KHR_10000 52/52`, `KHR_20000 21/21`, `KHR_50000 34/37`.
The corrected dominant issue is low-confidence overproposal, especially
`KHR_50000/KHR_20000` unmatched FPs on `asian_currency` and
`khmer_us_currency`, plus high-value USD misses/overproposal in
`usd_total`/`billsbank`.

Corrected triage and guardrail artifacts:
`positive_error_review_highvalue_khr_compare_v2/triage_queue_highvalue_khr_compare_v2.csv`,
`positive_error_review_highvalue_khr_compare_v2/triage_queue_highvalue_khr_compare_v2_sheet.jpg`,
`shared_error_eval_slice_highvalue_khr_v2/data.yaml`, and
`shared_error_eval_slice_highvalue_khr_v2/light_scorecard_shared_error_highvalue_khr_v2.json`.
On the v2 mined failure slice at `conf=0.05`, balanced-real p24 is strongest:
test recall/precision/bg-FP images `0.9000/0.1875/29` vs p24 synth+real
`0.1500/0.0357/33`, and val `0.6000/0.0573/63` vs p24 synth+real
`0.2000/0.0158/71`. Treat this as a low-confidence FP/source guardrail, not a
standalone promotion gate because it is intentionally mined from failures.
Confidence sweep artifact
`shared_error_eval_slice_highvalue_khr_v2/light_conf_sweep_real_synth_p24_v2.json`
shows a global threshold raise is only a suppression knob: from `conf=0.05` to
`0.30`, synth+real test recall stays `0.1500` while FP drops `81 -> 28` and
bg-FP images `33 -> 20`; val recall falls `0.2000 -> 0.0667` while FP drops
`187 -> 70` and bg-FP images `71 -> 53`. Do not "fix" this pocket by simply
raising confidence; repair class/source calibration with reviewed hard
negatives, unknown-currency routing, or staged KHR-protected training.
The existing synth+real -> balanced-real recalibration checkpoint also does not
rescue this pocket:
`shared_error_eval_slice_highvalue_khr_v2/light_scorecard_shared_error_highvalue_khr_v2_with_recal_v1.json`
keeps test recall at `0.1500` with worse FP (`81 -> 86`) and bg-FP unchanged
`33/33`, while val recall only improves `0.2000 -> 0.2667` with FP `187 -> 194`
and bg-FP images `71 -> 73`. Balanced-real p24 remains the local reference here.
Balanced-vs-synth delta sheets
`fp_delta_balanced_real_p24_vs_real_synth_p24_{test,val}_conf005_v1.{json,csv,jpg}`
show the extra synth+real FP pressure is mainly `KHR_50000/KHR_20000` on
`asian_currency`/`khmer_us_currency`, but the larger model gap is missed positives
that balanced-real gets on this mined slice.
Review-only bridge queue
`shared_error_eval_slice_highvalue_khr_v2/synth_real_calibration_bridge_review_queue_v1.csv`
(`.json` summary and `.jpg` sheet beside it) merges corrected v2 shared triage,
synth+real-only positive errors, and balanced-vs-synth extra-FP rows. It has
`315` rows: `220` corrected shared triage, `56` extra-FP rows, and `39`
synth+real-only positive errors. It is explicitly not training data; each row
must be adjudicated into trusted positive, unknown/out-of-scope, or reviewed hard
negative before any calibration run. Because those rows come from val/test
diagnostics, do not train on them directly; use them to set source/class policy,
then mine and review train-split analogs.
Train-split analog queue
`shared_error_eval_slice_highvalue_khr_v2/synth_real_calibration_train_analog_review_queue_v1.csv`
(`.json` summary and `.jpg` sheet beside it) pulls `4902` train rows from the
existing audit queue that match the corrected bottleneck sources/actions:
`asian_currency 3782`, `khmer_us_currency 971`, `usd_total 140`, and `khmer 9`.
Top rows visibly include many `khmer_us_currency` 100-riel/current-schema rows;
`scripts/check_currency_taxonomy_coverage.py` confirms `KHR_100` is missing from
the current model schema, so route those as unknown/out-of-scope unless the model
schema expands.
Clustered train analogs
`shared_error_eval_slice_highvalue_khr_v2/synth_real_calibration_train_analog_review_clusters_v1.csv`
compress those `4902` rows to `2347` canonical review units (`asian_currency
1275`, `khmer_us_currency 971`, `usd_total 92`, `khmer 9`) with a top-100 sheet.
Review clusters first, then expand decisions to rows; otherwise the label cleanup
will drown in near-duplicate `KHR_100`/high-risk-source variants.
First review packet
`shared_error_eval_slice_highvalue_khr_v2/synth_real_calibration_first_review_clusters_v1.csv`
(`.json` summary and `.jpg` sheet beside it) selects `200` balanced train-split
clusters: `50` `KHR_100` current-schema policy, `45` `asian_currency`
predicted-money empty reviews, `45` `asian_currency` high-risk reviews, `35`
`khmer_us_currency` mixed-class audits, and `25` `usd_total` high-value reviews.
Use this packet as the next manual/agent review entrypoint before creating any
new calibration train mix.
Materialization guard:
`scripts/materialize_synth_real_calibration_review.py` converts reviewed
train-cluster decisions into explicit train-only lists, but refuses blank
decisions by default. It accepts only reviewed/accepted clusters with `usable_as`
in `trusted_positive`, `hard_negative`, `unknown_out_of_scope`, or `exclude`, and
keeps `unknown_out_of_scope` separate from hard negatives because schema scope is
a product decision. Dry check
`shared_error_eval_slice_highvalue_khr_v2/materialized_review_drycheck_v1/summary.json`
correctly materialized `0` rows from the current unreviewed packet.
Smoke fixture
`shared_error_eval_slice_highvalue_khr_v2/materializer_smoke_v1/materialized/summary.json`
proves the positive path: one explicitly `reviewed` current-schema `KHR_100`
cluster expands to exactly one train image in `unknown_out_of_scope` and no YOLO
config.
Current-13 prefill
`shared_error_eval_slice_highvalue_khr_v2/synth_real_calibration_first_review_clusters_current13_prefill_v1.csv`
marks the `50` obvious first-packet `KHR_100` policy rows as
`unknown_out_of_scope` but uses `review_decision=policy_prefill_needs_visual_confirm`,
so the materializer still refuses them until explicit review acceptance. This is
the safe current-schema path.
Taxonomy scope note
`shared_error_eval_slice_highvalue_khr_v2/taxonomy_scope_note_v1.json` records
the current official-scope blocker: the 13-class YOLO schema and active cutout
bank both miss `USD_2`, `KHR_50`, `KHR_100`, `KHR_200`, `KHR_15000`,
`KHR_30000`, `KHR_100000`, and `KHR_200000`. Under the current schema, `KHR_100`
rows are unknown/out-of-scope; for final "all Khmer riel" counting, schema and
asset expansion is a product/model requirement rather than label cleanup.
Schema expansion inventory
`shared_error_eval_slice_highvalue_khr_v2/taxonomy_schema_expansion_inventory_v1.csv`
turns that blocker into an eight-class worklist. All missing classes except
`KHR_50` already have raw current front/back coverage but need active cutouts and
schema wiring; `KHR_50` also needs current raw front/back sourcing. The train
audit has `89` `KHR_100` policy hits, so this is not an edge-case cleanup detail.
Focused eval slice
`current_schema_unknown_khr100_eval_v1/data.yaml` contains `38` val/test
`khmer_us_currency` `KHR_100` current-schema unknown rows (`26` val, `12` test).
Scorecard `current_schema_unknown_khr100_eval_v1/light_scorecard_current_schema_unknown_khr100_v1.json`
shows all main detector foundations fire on every image at `conf=0.05`.
Synth+real p24 has FP `28` test / `84` val, balanced-real p24 `22` / `64`, and
audit-clean sourcecap48 `24` / `56`; predictions are mostly
`KHR_20000/KHR_50000` plus smaller Riel classes. This proves `KHR_100` is a
current-schema unknown rejection problem for all foundations and a schema
expansion blocker for final all-riel CashSnap.
Expansion plan artifacts:
`current_schema_unknown_khr100_eval_v1/currency_taxonomy_gap_plan_official_v1.{json,md}`
and
`current_schema_unknown_khr100_eval_v1/schema_expansion_candidate_cutout_audit_v1.json`.
Candidate full-scope cutout banks already cover `USD_2`, `KHR_100`, `KHR_200`,
`KHR_15000`, `KHR_30000`, `KHR_100000`, and `KHR_200000` pending rights/status
and red-mark review; `KHR_50` is the lone missing class that needs current raw
front/back sourcing or status review. Audits found `92` assets / `5` suspects in
the current full-scope candidate bank and `158` assets / `17` suspects in the
any-status candidate, all suspect flags are red-mark style.
Non-active schema draft
`configs/taxonomy/cashsnap_official21_schema_draft_v1.yaml` defines the 21-class
official/current USD+KHR order and a verified current-core13 -> official21 class
mapping. It must not be used as a training config until labels, cutouts, and eval
slices are migrated; it exists so schema-expansion work has a precise target.
KHR_100 annotation proposal queue
`runs/cashsnap/real_data_label_audit_v1/khr100_schema_expansion_annotation_proposals_v1/`
uses the three main 13-class detectors as localization teachers on `89`
train-split `KHR_100` policy rows. It wrote `590` proposal rows and a best-box
sheet; `87/89` images have a `conf>=0.20` proposal. Predictions are mostly
wrong-class `KHR_20000`/`KHR_50000`/`KHR_500`, but the boxes visually cover the
note well enough to speed annotation. This is not training data; it is a review
queue for adding `KHR_100` boxes under the official21 schema.
Official21 review bridge
`scripts/materialize_cashsnap_official21_review_bridge.py` is the safe promotion
path from reviewed proposal hints to a YOLO-readable official21 dataset. It
remaps current core-13 labels through
`configs/taxonomy/cashsnap_official21_schema_draft_v1.yaml`, accepts proposal
boxes only when normalized `review_decision=accepted_box`, keeps accepted missing
classes train-only by default, dedupes duplicate teacher boxes by IoU, and writes
hardlinked/copied `images/<split>` plus remapped `labels/<split>`. Current dry
check
`khr100_schema_expansion_annotation_proposals_v1/official21_bridge_drycheck_v1/summary.json`
materializes `0` rows from the unreviewed proposal CSV, while the explicit fail
check exits with "no accepted proposal boxes found". Smoke fixture
`khr100_schema_expansion_annotation_proposals_v1/official21_bridge_smoke_v1/materialized/`
proves one reviewed KHR_100 proposal becomes an official21 YOLO label with class
id `8`. Do not train from a full bridge root until the proposal rows are
reviewed, the output root is registered/classified in the data lifecycle
registry, and `scripts/check_data_lifecycle_registry.py` passes.
Full dry-run
`khr100_schema_expansion_annotation_proposals_v1/official21_bridge_full_drycheck_v1/summary.json`
walks the current base train/val/test labels through the official21 mapping
without writing image/label mirrors: train `14036`, val `2103`, test `1562`,
with `0` accepted proposal boxes because the source proposal CSV is still
unreviewed. This verifies the core13 -> official21 label migration path before
any full materialization.
Do not assume existing WebGL "fullschema" artifacts solve official21 support:
`current_schema_unknown_khr100_eval_v1/webgl_fan_fullschema_candidate_schema_check_v1.json`
shows `data/synthetic/cashsnap_webgl_fan_fullschema_candidate_v1/data.yaml` is
still core-13 (`nc=13`) and omits all eight missing official/current classes,
including `KHR_100`.
Class-specific high-Riel confidence thresholds are a diagnostic/product knob, not
a foundation repair. Probe artifact
`shared_error_eval_slice_highvalue_khr_v2/light_class_threshold_probe_real_synth_p24_v1.json`
shows `KHR_50000/KHR_20000 >=0.20` cuts mined-slice FP (`test -25`, val `-44`)
and bg-FP images (`test -6`, val `-8`) without hurting that mined-slice recall,
but protected-Riel test recall drops `0.9643 -> 0.9286`. Stop threshold sweeping
here unless the active question is product operating-point calibration; the model
needs better reviewed real/source policy, not a prettier threshold.

Real-data quality is now the gating bottleneck before more training. Full
`data/cashsnap_v1` inventory: train `14036` images/`7240` boxes/`6918` empty,
val `2103`/`991`/`1115`, test `1562`/`817`/`748`. Train box counts by class:
`USD_1 627`, `USD_5 753`, `USD_10 1009`, `USD_20 894`, `USD_50 911`,
`USD_100 976`, `KHR_500 284`, `KHR_1000 635`, `KHR_2000 153`,
`KHR_5000 614`, `KHR_10000 349`, `KHR_20000 17`, `KHR_50000 18`; rare
high-value KHR train images are only `14`/`15` for `KHR_20000`/`KHR_50000`.
Val/test class counts are in `runs/cashsnap/real_data_audit_check_yolo_dataset_v1.json`.

Real-label audit v1 lives at `runs/cashsnap/real_data_label_audit_v1/`, built by
`scripts/audit_cashsnap_real_dataset_labels.py`. It scanned `17701` real images,
found `3372` ranked issue signals, and wrote `inventory.csv`, `issues.csv`,
`cross_split_leakage.csv`, visual sheets, and `audit_rollup_v1.json`. Important
review lists: `review_empty_label_target_suspects.txt` (`1026` empty-label rows
with high-confidence unmatched target predictions), `review_highrisk_source_teacher_rejects.txt`
(`4760` high-risk-source rejects), `review_first_union.txt` (`5110` rows), and
provisional `trusted_positive_train_candidates_v1.txt` (`6692` train positives).
Derived training/triage buckets: provisional clean positive train anchors
`6692`, low-risk empty-label train candidates `2472`, high-risk-source empty train rows
`3989`, train empty rows with unmatched teacher predictions `780`,
KHR_100-looking schema-out-of-scope rows `127`, and `khmer_us_currency` positive
train rows needing source/class audit `1114`. Visual sample of the low-risk
empty-label candidate bucket still contains coins/dark/background/non-target
money-like objects, so treat it as hard-negative/background candidate material,
not verified true-empty.
Use `review_queue_ranked_v1.csv` (`6158` rows) as the cleanup entrypoint; it
prioritizes empty-label predicted-money review/relabel, high-risk-source visual
review, mixed-currency source/class audit, and KHR_100 schema routing.

Visual QA changed the data policy: `asian_currency` empty-label samples visibly
contain banknotes, including target-looking KHR, so the empty bucket is not safe
background. `khmer_us_currency` is a mixed USD/KHR source and must be source/class
audited before any row is used as trusted real, negative, or half-synthetic
anchor. `cashcountingxl` empty samples are mostly office/coin/foreign/unknown
scenes but still include money-like objects, so they are useful hard negatives
only after semantic triage.

Mixed-currency visual QA pack
`runs/cashsnap/real_data_label_audit_v1/khmer_us_currency_teacher_rejects_conf025.jpg`
shows many 100-riel/KHR_100-looking rows. Current 13-class schema cannot encode
`KHR_100`, so those rows are not true empty and not current target positives;
route them as current-schema-out-of-scope/unknown currency until the taxonomy and
model schema are expanded.

Detector-assisted audit with the recalibrated real+synth p24 teacher at
`conf=0.25` found `1414` images with unmatched predictions across full splits:
train `1076`, val `228`, test `110`. Top suspect sources are train
`asian_currency 340`, `usd_total 311`, `billsbank 122`, `khmer_us_currency 119`,
`cashcountingxl 103`, `cambodia_currency_project 72`. Top unmatched predicted
classes are `KHR_50000 335`, `USD_100 235`, `USD_50 221`, `USD_1 184`,
`KHR_20000 132`, `KHR_1000 121`. Treat these as review priority, not automatic
truth.

Cross-split leakage is real: audit v1 found `881` canonical-base/exact-hash
leakage groups, mostly Roboflow-style `.rf.*` variants of the same source image
appearing across train/val/test. This is not necessarily a bad training row, but
it weakens random-split evaluation. Promotion claims should prefer source/session
or canonical-base split checks once the labels are cleaned.

The "empty-label" bucket is semantically mixed, not a clean background split.
Semantic bridge v1 found likely true empty frames, suspect unlabeled targets,
foreign/currency-review rows, student-overfire rows, and model-review rows in
train/val/test. A class-balanced active label queue exists at
`runs/cashsnap/semantic_bridge_active_label_queue_train_v1/queue.csv`; use it
for review/manual relabeling only, not pseudo-label training.

The current detector bottleneck is split between missing proposals and wrong
denominations. Strictbest proposal audit has same-class recall `0.6438` but
class-agnostic localization recall `0.8421`: `129` no-box FNs and `162`
wrong-class FNs before reclassification. Clean-real control reaches
pre-gate recall/localization `0.9731`/`0.9853` with only `12` no-box FNs, so the
strict detector's miss pattern is a data/objective problem, not model capacity.

Source/context accounting is now essential. `usd_total`/`billsbank` are broad
USD trouble domains; `asian_currency`/`cashcountingxl` carry most empty-label
and foreign/unknown pressure. Source-context support can improve one source and
destroy another unless every detectable banknote is labeled, removed, or routed
as unknown.

WebGL UNKNOWN exporters, source-group audits, teacher-agreement checks, visual
gap audits, and fixed-step phase preflights are valuable harness infrastructure.
They should support the next structural mechanism; they are not by themselves a
reason to run more row-dose sweeps.

### Tested Ideas

- **Target-anchor plus six vetted foreign hard negatives is the current strict
  best.** It is promoted only as the final result for this phase, not as a target
  success.
- **Clean-checkpoint repair is promising but not promoted.** Poisson/contact plus
  train-only FP-mined negatives reached `0.747316` seed0 but failed `USD_50` and
  lacks seed repeat. It moved toward the old entry target but is not a release
  claim.
- **Unknown/near-negative pressure is powerful and too blunt when applied as row
  dosing.** Some obligation/unknown mixes raised aggregate AP and cut background
  FPs hard, but suppressed recall or protected KHR. Do not continue pure dose
  tuning without a structural target-vs-unknown objective.
- **Pseudo-label and true-empty shortcuts are rejected.** Blind teacher-positive
  suspect-target rows, semantic true-empty background replacement, and naive
  zero-label pressure all damaged real transfer or protected classes.
- **Broad source-context replacement is rejected as a final detector path.**
  Accepted USD source rows and `usd_total` support are useful clues, but the
  tested packs either regressed full AP/product behavior or traded USD recall
  against KHR/background safety.
- **Replacement-in-real-phone-context is not a transfer fix by itself.**
  Single-box and multi-instance source-context replacement audits remain useful
  QA, but scaling them directly failed strict transfer badly.
- **Direct 14-class UNKNOWN detection is killed for now.** Initial gains were
  phase-confounded; phase-matched direct unknown-prop reruns collapsed and
  worsened lightweight behavior. Keep the exporter, not the row objective.
- **Proposal gating is the strongest product-architecture signal.** The
  true-empty gate preserves recall while cutting background proposals. Naive gate
  training variants over-rejected targets or collapsed, so future gate work must
  be reviewed/source-balanced and judged on count/value errors, not only mAP.
- **Crop denomination reclassification is useful but not single-stage proof.**
  Synthetic crop training alone is weak; synthetic plus a tiny real crop anchor
  and fragment-shaped training approaches the real-full crop upper bound. This
  argues for architecture/data calibration, not more full-scene detector rows.
- **Strictbest synthetic checkpoint -> tiny real p24 calibration is killed as
  tested.** One epoch/b2/lr5e-5/no-warmup fine-tune from the strict synth-only
  detector to balanced real p24 collapsed full real test mAP50-95
  `0.835861 -> 0.487777` versus the balanced real-only clean-checkpoint
  reference, with `12` per-class guard failures. Do not repeat this schedule;
  only revisit staged transfer with an explicit longer/safer real calibration,
  lower LR, partial reset/freeze plan, or architecture reason.
- **Real-flat asset replacement and one-class BANKNOTE factorization are
  rejected as no-box repairs.** Both hurt the recall/localization bottleneck
  under current training mixes, even when they improved some precision/counting
  proxies.
- **USD visible extent is real but insufficient.** Extent-heavy target-anchor
  rows improved localization/product bottlenecks but failed strict full-test AP
  and KHR guards. Extent must be coupled with source/context accounting,
  unknown-note pressure, and KHR protection.
- **Tone/refiner/stylization work is harness learning, not trainable data yet.**
  Local-tone and SD-Turbo note-edge locking produced useful QA and preservation
  gates, but did not show a regime-changing camera-domain fix. Visual QA caught
  risky edge/context changes that metric sheets alone would hide.
- **Built-in random mosaic on the current p24 blend is neutral.** Activating
  YOLO `mosaic=1.0`/`close_mosaic=0` for the same p24 synth+real rows and same
  `318` steps produced different weights but identical full/per-class AP. Do not
  count this as a real half-synth/collage test; it only says random same-row
  mosaic did not move the current clean AP yardstick.
- **Literal 3x3 grid-collage packing is killed as tested.** The `6 real + 3
  synth` grid failed its matched `9 real` grid control and was slightly below
  the p24 champion. The all-real grid control's tiny full AP gain failed strict
  and source-excluded guardrails. Keep the builder for diagnostics; do not scale
  to 4x4/5x5 or promote artificial grid seams as overlap training.
- **Rectangular real-crop fan composites are killed as tested.** The `48`-image
  fan/overlap mix from p24 crop-classifier tiles failed its duplicate-exposure
  control and was far below the p24 champion, with a large `KHR_50000` drop.
  Keep only the alpha-policy/generator lesson; next half-synth work needs masked
  note assets, real captures, or explicitly audited source-note handling.
- **Accepted WebGL stack/fan cap6 on the p24 synth+real champion is killed as
  tested.** The exact-mask WebGL roots pass suite gates, but adding `16`
  stack/fan images lost to duplicate exposure and cratered versus the champion,
  again led by `KHR_50000`. Do not scale this schedule; any WebGL revisit needs
  KHR-protected staging and a real overlap/counting validation bridge.

### Untested Ideas

Do not preserve small probes here just because they are reasonable. A worthwhile
untested idea should plausibly change the transfer regime, expose why the current
harness is misleading, or materially improve the promoted clean foundation on a
real bottleneck under guardrails. Otherwise it is a tactic, not the research
frame.

- **Real+synth schedule bakeoff.** The June 2026 blend-strategy PDF argues for
  comparing real-only, naive mix, synthetic pretrain -> real fine-tune, real ->
  targeted synthetic repair -> real calibration, and only later detector+gate
  behavior when the active phase explicitly returns to product-stack selection.
  The new p24 blend shows synthetic variation can beat real duplication; next
  kill/promote decision needs staged calibration and low-conf/background/source
  guards, not just another mixed-row run. Do not run it on unreviewed
  `asian_currency`, `khmer_us_currency`, or other empty-label/mixed-source rows
  as if they were clean negatives. The sourcecap48+strictbest-synth duplicate
  exposure-control probe is already killed as tested (`+0.001392` full mAP,
  `KHR_50000 -0.091826`), so the next synth blend must protect rare KHR classes
  by construction rather than just adding rows to the audit-clean sourcecap base.
- **High-value USD real-dup swap is diagnostic, not a promotion.** Replacing the
  `USD_50`/`USD_100` strictbest-synth extras with duplicated real exposure kept
  row/class parity but dropped full mAP versus the synth+real champion
  (`0.844567` vs `0.852767`). This suggests the high-value USD synth rows should
  not be blindly removed; the next model-side repair needs a subtler USD protect
  mechanism than swapping synth out for duplicate real.
- **Detector-label cleanup before another blend run.** The smartest next
  move is to turn `review_queue_ranked_v1.csv` into a small trusted bridge:
  reviewed positives, explicit unknown/out-of-scope rows, and vetted hard
  negatives, then rerun detector scorecards on full, semantic, strict,
  source-excluded, per-class, and low-conf/background slices. Kill more blend
  training if it improves aggregate AP while weak classes, source robustness, or
  low-confidence background behavior degrade. Start with the corrected v2
  shared-error queue and scorecard, not the superseded v1 queue: the live problem
  is `KHR_50000/KHR_20000` overproposal on `asian_currency`/`khmer_us_currency`
  plus high-value USD source misses/FPs, so the next bridge should prioritize
  reviewed hard negatives/unknown-currency routing and source/class calibration
  rather than assuming weak KHR positives are missing from the model. The
  corrected confidence sweep already showed global thresholding mostly suppresses
  proposals while leaving the miss pattern unresolved.
- **URGENT: real-augmented half-synth bridge.** Because there is no trusted
  trainable real overlap/fan/hand set, the plausible path is: clean/audit real
  non-overlap, build a synth+real calibrated base, then augment reviewed real
  notes/contexts into label-preserving half-synthetic scenes: multi-note
  collages, fans, partial overlaps, hand/table/phone-context variants, and
  source-aware unknown-money negatives. The promoted real fan benchmark currently
  has only `3` candidate sources and `1` labeled mild-overlap image with `6`
  boxes, and the current real multi-note geometry slice has only `5` val/test
  images / `11` boxes, so use both as smoke/rights-review prompts, not release
  proof. The new `real_overlap_review_queue_v1/review_clusters.csv` gives the
  first real review entrypoint: adjudicate canonical fanned/stacked/hand/partial
  clusters before generating more synthetic overlap. The rectangular p24 crop
  fan probe and the accepted WebGL stack/fan cap6 dose are both killed, so the
  next bridge must use masked/audited note assets or real captures with
  KHR-protected staging and explicit unknown-note policy rather than
  crop-classifier rectangles or naive WebGL row dosing. Use only audited
  positives or reviewed high-risk-source rows as anchors; never use empty-label
  money scenes as backgrounds. Kill it if overlap gains require sacrificing
  clean non-overlap recall, true-empty behavior, or mixed USD/KHR denomination
  accuracy.
- **Manual/class-balanced real-label bridge, not blind pseudo rows.** The
  semantic split is useful as a validator. The next bridge should manually
  relabel a small class-balanced subset of suspect targets/currency-review rows
  or use calibrated, class-balanced adaptation. Kill it if it improves bridge
  proxies while true-empty FPs or weak classes degrade.
- **Real capture/validation bridge as the next big missing signal.** Capture or
  label mixed USD+KHR stacks, hard `KHR_50000`, `KHR_5000/KHR_20000` thin
  slices, same-denomination fans, no-note backgrounds, and non-banknote paper
  props. Kill it if it becomes another tiny slice with no new failure
  separation.
- **Real-context, unknown-aware synthetic rebuild.** A serious rebuild must
  account for all banknotes in a scene, preserve/remove/source-label source
  notes, represent unknown/foreign notes explicitly, fit camera/ISP variation,
  and protect KHR while repairing broad USD source modes. Kill it if
  teacher/proxy gains do not improve real recall and empty-frame FPs together.
- **Class/source-aware obligation objective, not row-dose repair.** The next
  test must separate target recall from unknown rejection structurally: sampling,
  loss weighting, proposal gating, or validation-driven curriculum. Kill it if
  background FP suppression disappears, protected KHR/weak USD still fail, or the
  only win is another small aggregate bump.
- **Controlled label-preserving refiner, only if it beats the SD-Turbo smoke.**
  A useful refiner must make an obvious camera/context-domain improvement while
  preserving teacher agreement. Pixel preservation and background stats alone
  are not enough.
- **Unknown-aware counting architecture branch.** Direct 13/14-class detection is
  not the only viable product shape. The serious next version is a
  reviewed/source-balanced banknote/background/unknown proposal gate or
  equivalent detector objective trained from strict-best proposals, true-empty
  rows, vetted foreign-note/unknown rows, and protected weak-class positives.

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
- clean-visible val/test when available;
- labeled-positive and geometry-stress slices;
- protected classes, especially riel and high-value notes;
- real empty-frame FP detections and images-with-FP at `conf=0.05`,
  `imgsz=416`, `batch=1`, `device=0`;
- detector + true-empty gate count/value behavior when the active question is
  product selection, with crop/reclassifier only if it beats the gate-only target
  on the chosen product metric;
- max per-class mAP50-95 drop `<=0.05`, unless explicitly waived;
- at least one seed repeat for serious promotion, more for large claims.

Synthetic package gates are necessary filters, not promotion authority. Self-eval
preservation is not enough. For low-memory probes, use the lightweight transfer
scorecard over multiple confidence thresholds and require no recall regression
plus no FP/background regression.

The clean base can move toward overlap/fan/hand only when the chosen foundation
survives the live detector gates: current-champion comparison, strict-clean and
source diagnostics, protected riel/USD stability, real-empty FPs no worse than
control, low-confidence behavior understood, and at least seed repeat or a
slow-promotion run. Synthetic-only `yolo26n` reaching the old aggregate target is
historical context, not the active blocker.

## Validation, Labels, And Scope

Validation:
- Full real val/test includes many empty-label images; always pair aggregate AP
  with empty-frame FP probes.
- Roboflow core-13 bridge is a positive KHR/USD judge for the current detector,
  but it is stretched and lacks background pressure.
- Eval sanity checked: the Roboflow core-13 model scores `0.946364` on its own
  bridge test and `0.445459` on CashSnap test with the same 13-class order,
  while the same CashSnap evaluator gives the clean-real control `0.883801`; do
  not treat the old core-13 `~0.5` cross-test result as an eval-bug signal.
- Roboflow official21 partial bridge preserves official classes present in the
  source, including `KHR_100`, but current 13-class weights cannot evaluate it.
- Mined-real stress is a warning slice, not release proof. It currently has `17`
  ready stress images and `35` scoreable boxes with narrow class coverage.
- Own-photo capture bridge is empty. High-value gaps are hand fan,
  same-denomination fan, `KHR_5000`/`KHR_20000` thin slices,
  `KHR_5000` face/number overlap, `KHR_50000` hard positives, mixed USD+KHR
  stacks, no-note backgrounds, and non-banknote paper props.

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
- When a script, config, or dataset is no longer active, make that visible in the
  folder or registry. Before moving code/configs, check imports, CLI references,
  docs, and workflow callers with `rg`.

Runtime and harness:
- Work on `master` unless the user asks for a branch.
- Prefer `rl` command prefixes in RunLong.
- Use repo-local runtime storage through `scripts/local_runtime.py`.
- Keep YOLO train/eval caches and generated outputs under repo-local ignored
  paths.
- Fixed-step train run names now encode the init model (`from_clean`,
  `from_yolo26n`, etc.); pre-fix custom-init runs may still say `from_clean`, so
  check `source_model` in the summary/manifest when auditing old runs.
- YOLO promotion posture: train `batch=64`, `workers=0`, `device=0`,
  `cache=false`; eval `batch=64`, `workers=2`; background-FP guardrail
  `batch=1`.
- In Codex/RunLong memory pressure, use smaller diagnostic batches but label them
  clearly. Do not compare low-batch diagnostics to b64 promotion parity.
- YOLO headroom default now requires about 4 GB free system RAM before launch;
  this prevents doomed runs on the 16 GB laptop where Torch/Ultralytics overhead
  alone can trip the 95% RAM guard even at tiny batches.
- If YOLO training hits the RAM guard at b4/b8, stop reducing batch and inspect
  host memory or harness behavior; that is a runtime blocker, not a data result.
- List-backed YOLO runs can write a mixed-image `data/cashsnap_v1/labels/train.cache`;
  delete that cache after mixed synthetic/real train-list probes so future real
  runs do not inherit stale cache state.
- Fixed-step `--max-train-batches` is a stop cap, not a data repeater. Set
  enough `--epochs` to reach the cap; `epochs=1` on a 40-row dataset is just one
  pass even if the run name says `steps200`.
- Fixed-step preflight now reports train-phase summaries and warnings for
  unequal row counts: scheduler progress at stop and post-close-mosaic step
  exposure can differ even with equal `--max-train-batches`. Use
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
