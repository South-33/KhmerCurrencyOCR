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

North star: build one small phone/browser-deployable detector that can count
mixed USD and Khmer riel from one casual retail photo, preserving clean
non-overlap performance while becoming materially better on countable partial,
overlap, fan, hand, cutoff, and edge-visible evidence.

Current phase posture: pivot from small probes to a single production-pilot
detector recipe. Clean/non-overlap foundation work remains parked as a
standalone AP chase; the next serious training run should be a curated one-model
blend with clean replay, visible-evidence positives, and explicit hard-negative
/ unknown-money pressure. Do not launch another tiny p12/p24/filter/scheduler
probe unless it directly de-risks the production-pilot blend or promotion gates.

End-goal premise: there is currently no real overlap/fan/hand data in the
trusted training set. The reason to push synth+real for non-overlap is to build
a calibrated real-domain base that can plausibly learn overlap later from
synthetic or real-derived half-synthetic overlap data. Do not treat clean
non-overlap as the final task; treat it as the foundation that must survive a
future synthetic-overlap fine-tune.

Parked non-overlap deliverable and clean yardstick: seed0 p24 balanced-real +
strictbest-synth is the strongest clean/non-overlap detector recipe we can
honestly justify today, with repeat evidence from seed1 and clear failure
pockets documented below. It is the clean foundation to protect, not the final
partial/overlap answer.

Historical entry target: `0.82-0.85` full real test mAP50-95. Treat this as the
old "is the detector plausible?" bar, not the current goal. Clean-real controls
proved the evaluator and model capacity can reach the zone: a near-size
real-trained control reached `0.819153`, and the later high clean-real checkpoint
reached `0.883801`. The current goal is a robust, explainable foundation, not a
single aggregate AP number.

Current promoted detector-only non-overlap yardstick remains: controlled balanced-real p24 +
strictbest-synth p24 seed0 from the clean checkpoint:
`runs/cashsnap/fixed_step_real_p24_plus_strictbest_synth_p24_from_clean_e1_i416_b2_w0_adamw_lr5e5_nowarmup_noamp_cachefalse_steps318_seed0/weights/last.pt`.
Its promoted AP baselines are: full real mAP50-95 `0.852767`, strict
semantic+leakage-clean `0.860743`, and source-excluded strict-clean `0.769331`,
versus balanced real p24 `0.835861` / `0.855619` / `0.760870`. This proves the
current strictbest synth rows can improve the detector over real duplication. It
is the model-side recipe to beat; detector+gate/browser behavior is adjacent
diagnostic evidence only unless the phase explicitly switches back to product
selection.

Current one-model pilot starting point, if a single detector must balance clean
and partial evidence today, is the p24 vis70 candidate:
`runs/cashsnap/fixed_step_countsafe_vis70_p24_from_last_e50_i416_b2_w0_adamw_lr5e6_nowarmup_noamp_cachefalse_freeze22_steps318_seed0/weights/last.pt`.
It was only a small 318-batch head tune over `927` unique rows (`323` original
real, `312` strictbest synthetic, `292` source-clean `vis0p7` partial crops),
but it is the best current compromise signal: full real mAP50-95 `0.854178`,
strict clean `0.865306`, source-excluded clean `0.795336`, unfiltered partial
test mAP50-95 `0.660102`, and filtered countable-partial test recall/precision
`0.8857/0.5569`. It is not production-safe by itself because source/unknown
money and wrong-denomination proposal issues remain; use it as the init or
teacher clue for Production Pilot v1, not as a silent promotion.

Do not silently promote the June 10 head-only duplicate-control continuation from
that checkpoint. It improves AP versus the promoted yardstick on full
(`0.852767 -> 0.854594`), strict-clean (`0.860743 -> 0.868202`), and combined
source-excluded (`0.782187 -> 0.798750`) comparisons, all with protected
per-class AP drops within `0.05`, but it fails low-confidence proposal guards:
raw `conf=0.05` FP deltas are `+73` full, `+31` strict, `+32` source-excluded;
with current KHR floors they are still `+70`, `+19`, `+19`. Treat it as an
AP/calibration candidate only, not proof of visible-evidence learning or a
browser-stack replacement. The v4-mined count-risk slice kills it as a product
candidate too: at `conf=0.15` it gets `0.8772` recall but only `0.3788`
precision, `20/22` background-FP images, and scorecard `+58` FP versus champion.
The repaired KHR100-aware proposal gate at `reject>=0.80` trims background FPs
to `5/22` without recall loss, but duplicate-control+gate is still far behind
champion+gate on count/value (`32/79` exact-value images versus `63/79`, exact
net `-31`, weighted net `-36`). Evidence:
`runs/cashsnap/real_data_label_audit_v1/compare_full_champion_vs_currentbest_dupctrl_steps80_v1.json`,
`runs/cashsnap/real_data_label_audit_v1/compare_strict_clean_champion_vs_currentbest_dupctrl_steps80_v1.json`,
`runs/cashsnap/real_data_label_audit_v1/compare_source_excluded_champion_vs_currentbest_dupctrl_steps80_v1.json`,
and
`runs/cashsnap/currentbest_dupctrl_steps80_guard_v1/light_scorecard_champion_vs_currentbest_dupctrl_steps80_khr_floor_conf005.json`,
plus
`runs/cashsnap/ve_v4_trainanchors_guard_v2/scorecard_challenge_slice_champion_vs_currentbest_dupctrl_conf015.json`
and
`runs/cashsnap/ve_v4_trainanchors_guard_v2/compare_challenge_slice_champion_gate_vs_currentbest_dupctrl_gate_rej080_conf015.json`.

Currentbest USD/high-value target-anchor alpha smoke is killed as tested. The
matched fixed-step A/B against duplicate exposure failed
(`0.854594 -> 0.852004`, delta `-0.002590`) and the worst protected AP drops were
`KHR_50000 -0.020655` and `USD_50 -0.007578`; do not scale that render recipe
without a clearer human-countable partial-evidence policy and visual QA. Evidence:
`runs/cashsnap/fixed_step_currentbest_usd_overproposal_alpha_smoke_vs_dupctrl_steps80_seed0/summary.json`.

Reviewed visible-evidence v4 train anchors are killed as a promotion path. The
exact object-exposure control cannot also match train phase for this dose
because `24` multi-note anchor images add `67` labeled objects; its preflight is
rightly phase-confounded (`702` vs `659` rows). A new row-count nearest
class-mix control matches scheduler geometry (`659` rows) but is explicitly
inexact (`missing USD_1 x1`, `KHR_10000 x4`). Against that weaker rowmix
control, the candidate only gains `+0.000472` full-test mAP50-95 and remains
below the promoted champion (`0.852532` vs `0.852767`). On the cleaned v4
visible-evidence eval, recall is saturated for all (`13/13`), while the
candidate adds a raw `conf=0.05` FP versus champion/rowmix (`8` vs `7`): a
wrong `KHR_50000` box over the true `KHR_20000` on `VE-012`. Browser KHR floors
suppress that FP and all three tie (`5` FP), but this is not count-safe
visible-evidence learning. Evidence:
`runs/cashsnap/fixed_step_visibleevidence_review_v4_trainanchors_vs_rowmixctrl_steps80_seed0/summary.json`,
`runs/cashsnap/real_visible_evidence_eval_extension_v4/scorecard_champion_vs_ve_v4_trainanchors_conf005.json`,
and
`runs/cashsnap/real_visible_evidence_eval_extension_v4/VE-012_candidate_wrong_khr50000_overlay.jpg`.

The full-test low-confidence and deployment-threshold audit reinforces killing
v4 train anchors. Rerunning champion and candidate on canonical
`configs/cashsnap_v1.yaml` with full FP JSONL export shows at `conf=0.05`
champion recall/precision/background-FP images `0.9510/0.5468/170` versus
candidate `0.9572/0.5245/173`; strict candidate-only FP matching finds `122`
extra boxes. At `conf=0.15`, candidate still gains recall (`0.9229` vs
`0.8984`) but fails count-safety guardrails with precision drop
(`0.6892` vs `0.7175`), `+51` total FP, `+8` background-FP images, and `82`
candidate-only FP boxes. Lowering YOLO NMS to `0.45` changes none of this.
Full-size QA of the surviving high-confidence extras shows duplicate boxes over
one note, catalog/collage unlabeled bill faces, coin/background images predicted
as USD, KHR_100/foreign notes predicted as `KHR_50000`, and a `KHR_10000`
predicted as `KHR_2000`; this is not acceptable partial-visible/count-safe
learning. Evidence:
`runs/cashsnap/ve_v4_trainanchors_guard_v2/fp_audit_summary.md`,
`runs/cashsnap/ve_v4_trainanchors_guard_v2/candidate_only_fp_queue_conf015.json`,
and
`runs/cashsnap/ve_v4_trainanchors_guard_v2/scorecard_champion_vs_ve_v4_trainanchors_full_test_conf015_nms045.json`.
The mined hard slice
`configs/audit/cashsnap_ve_v4_trainanchors_candidate_only_fp_conf015_slice_v1.yaml`
is now available as a count-risk regression guard: on its `79` images at
`conf=0.15`, champion is already hard (`0.8421` recall, `0.6667` precision,
`11/22` background-FP images), while v4 collapses precision (`0.9123` recall,
`0.3377` precision, `22/22` background-FP images, scorecard `+78` FP). Use this
slice to catch future count-risk regressions, not as a standalone promotion
metric because it is candidate-mined.
The repaired KHR100-aware proposal gate is useful but does not rescue v4: on the
hard slice at `conf=0.15`, champion+gate `reject>=0.80` preserves recall and
improves precision/background FPs (`0.8421/0.7619`, `5/22` bg FP images), while
v4+the same gate preserves v4 recall but only reaches `0.4160` precision
(`5/22` bg FP images). Treat the gate as a hard-negative/product mitigation, not
as a fix for detector duplicate/wrong-denomination behavior. Exact-value
comparison is decisive: champion+gate has `63/79` exact-value images on the hard
slice, while v4+gate has `19/79` (`-44` net, `46` exact losses, `2` wins).
The AP-hot duplicate-control continuation also fails this gated count/value
check: it reaches `32/79` exact-value images, exact net `-31`, weighted net
`-36`, with most losses from `usd_total`, `billsbank`, and
`khmer_us_currency`. More recall from head-only continuation is therefore not
the next bottleneck; detector-side duplicate suppression and denomination
evidence quality must improve before another proposal-heavy checkpoint can be
trusted.
A focused hard-slice sweep of plausible existing alternatives also keeps
champion+gate on top. Seed1 p24 synth+real is the closest recall clue
(`0.9123` pre-gate recall, same `11/22` pre-gate bg FP as champion) but
seed1+gate drops to `51/79` exact-value images versus champion+gate `63/79`
(`-12` exact net). USD-total empty24+gate is closer on background FPs
(`3/22`) but still worse on exact value (`56/79`, `-7` exact net). Bbox
occlusion, real-overlap review39, mined partial-stress, duplicate-control, and
v4 train-anchor checkpoints all show the same proposal-heavy failure at
`conf=0.15`. Evidence:
`runs/cashsnap/ve_v4_trainanchors_guard_v2/fp_audit_summary.md`,
`scorecard_challenge_slice_champion_vs_seed1_conf015.json`,
`scorecard_challenge_slice_champion_vs_usdtotal_empty24_conf015.json`,
`compare_challenge_slice_champion_gate_vs_seed1_gate_rej080_conf015.json`, and
`compare_challenge_slice_champion_gate_vs_usdtotal_empty24_gate_rej080_conf015.json`.
The direct combination hypothesis is also killed: seed1 -> USD-total empty24
80-batch head calibration loses full-test AP to the seed1 -> base-p24 control
(`0.864407 -> 0.860212`) and is worse on hard-slice count/value. With the same
gate, seed1 base-control80 gets `48/79` exact-value images and the empty24
calibration gets `45/79`, both behind champion+gate `63/79`. Evidence:
`runs/cashsnap/fixed_step_seed1_usdtotal_empty24_calib_vs_seed1_base_steps80/summary.json`
and
`runs/cashsnap/ve_v4_trainanchors_guard_v2/compare_challenge_slice_champion_gate_vs_seed1_usdtotal_empty24_calib80_gate_rej080_conf015.json`.
Champion+gate's own remaining hard-slice errors are now materialized in
`runs/cashsnap/ve_v4_trainanchors_guard_v2/champion_gate_exact_error_queue_conf015.json`
and show the next bottleneck: high-value USD partial/source misses, duplicate or
off-target USD overcounts, KHR_100/unknown notes leaking as high-value KHR,
coin/background leakage, and one `KHR_10000 -> KHR_2000` wrong-denomination
case. A gate threshold sweep from saved champion proposals finds
`reject>=0.70/0.72` improves hard-slice exact-value images (`63/79 -> 65/79`)
and background-FP images (`5 -> 4`) but drops recall (`0.8421 -> 0.8246`).
This is a product threshold tradeoff, not a detector-side solution. Use the
exact-error queue to build source-safe train analogs and reviewed capture
targets; do not train on the candidate-mined test slice itself.
Train-only analog queues now exist beside that audit. The positive error pack
`champion_train_positive_error_analogs_from_exact_queue_v1/` shows train-split
weak pockets that align with hard-slice failures: `KHR_10000` recall is only
`38/60 = 0.6333`, `USD_5` is `34/48 = 0.7083`, and common wrong-class pairs
include `KHR_20000->KHR_2000` (`8`) and `KHR_10000->KHR_50000` (`5`). The
empty-label train probe
`champion_train_empty_fp_analogs_allclasses_conf015_v1.json` finds `10/30`
train empty rows with FPs at `conf=0.15`, all `KHR_1000` on `asian_currency`
Korean-won images; the train split does not yet cover the hard-slice
`KHR_100`/high-value KHR and coin/source-policy leaks well. Prior tiny coin and
reviewed foreign-hardneg detector doses already failed, and the broader
source-policy row-mix is now killed too. The source-policy mix
`cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_sourcepolicy_posneg_v1`
adds `154` net reviewed rows (`USD_50` positives, KHR partial positives,
`asian_currency` hard negatives, and coin hard negatives; `KHR_100` unknowns are
excluded) and has an exact class/empty exposure control at `789` rows / `119`
empty images. The candidate loses full-test AP to that control
(`0.852462 -> 0.852261`) and remains below the champion. On the hard slice at
`conf=0.15`, it buys recall by spending count safety (`0.9123` recall,
`0.3688` precision, `21/22` background-FP images; scorecard `+65` FP versus
champion). The raw proposal-gate probe at `reject>=0.80` still gets only
`32/79` exact-value images, but that comparison omits browser final NMS and
overstates postprocess-recoverable duplicate damage. After materializing the
browser-style final class-agnostic NMS (`nms_iou=0.70`), champion/candidate are
`66/79` exact at `reject>=0.80` and `68/79` vs `60/79` at the current
`reject>=0.72`; the source-policy candidate ties or loses to its exposure
control (`59/79` vs `59/79` at `0.80`, `60/79` vs `61/79` at `0.72`). Full-size
QA of
`sourcepolicy_posneg_candidate_only_fp_queue_conf015` shows the mechanism:
USD strips/backs and same-note boxes become extra `USD_100`/USD duplicates,
out-of-schema or foreign banknotes become target classes such as `KHR_50000`,
and coins still fire as USD. Do not repeat broader 13-class empty-negative row
dosing as the detector fix. The next move needs an explicit duplicate/objectness
or count-aware objective, stronger unknown/schema routing, or a proposal-gate
architecture change. Harness lesson: use final-NMS materialization for
browser-stack count/value comparisons, while raw gate-only metrics remain useful
for detector proposal diagnosis.

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
reviewed synth+real stack/fan bridge is killed, including the gentler 40-step
repeat: the 17-row cap6 mix can pass the champion full-real/per-class guard at
low dose (`0.852767 -> 0.853184`, `KHR_50000 -0.0351`), but it loses to exact
duplicate exposure by `-0.000036` and still adds false-positive pressure on
active-train and feathered-overlap diagnostics. Splitting the pack shows the
full-real clue is from the 12 WebGL stack/fan rows (`+0.001437` vs duplicate
control, `+0.000048` vs champion), not the 5 reviewed-real anchors, but WebGL
still fails active-train and feathered-overlap scorecards. Treat this as
evidence that masked/audited stack/fan scenes have useful signal only when
paired with duplicate/count or FP control; do not scale tiny row-dose variants
as-is.

Production-pilot ingredient: augment reviewed real notes/contexts into
label-preserving half-synthetic material, including multi-note collage/fan/overlap
layouts, but only as part of the curated one-detector blend. The cheap built-in
YOLO mosaic probe was neutral, and the first rectangular real-crop fan probe plus
a small accepted WebGL stack/fan dose were harmful; those failures kill the
naive schedules, not the underlying bridge. Any next use needs masked/audited
note assets or real captures, source-aware unknowns, a schedule that protects
weak KHR classes, and overlap/counting validation. Step-back stress guardrails on
the WebGL stack/fan cap6 candidate show only a tiny geometry-stress AP clue, not
a foundation rescue: geometry-stress test `0.891355 -> 0.892408`, but
protected-riel test `0.651747 -> 0.644157` versus the champion.

Latest partial/overlap read: keep the seed0 p24 synth+real checkpoint as the
foundation. The user-reviewed focus packet
`runs/cashsnap/real_overlap_focus_review_packet_v1/focus_review_packet_v1_reviewed.csv`
materialized into `43` clusters (`39` train-anchor variants, `4` trusted eval,
`24` excluded rows) under
`runs/cashsnap/real_overlap_focus_materialized_reviewed_v1/`. On the tiny reviewed
held-out eval, p24 synth+real ties balanced-real recall (`0.5000`) with better
precision (`0.5000` vs `0.3333`); on reviewed train-anchor diagnostics it is
also slightly better (`0.6190/0.7471` recall/precision vs `0.6095/0.7191`).
This is validation evidence for the current base, not a release proof.

The first reviewed real-overlap train-anchor dose is not promotable as-is.
Adding the `39` materialized train-anchor rows to the p24 synth+real list and
fine-tuning only the head (`freeze=22`, one epoch, lr `1e-5`) keeps the clean
guard alive but mainly increases proposals. The candidate full-test AP is
`0.857446` versus champion `0.852767` and matched duplicate control `0.855847`,
with per-class guards passing. Direct reviewed held-out eval stays tied for all
three at `0.5000/0.5000` recall/precision; train-anchor recall rises
(`0.6190 -> 0.6476`) but FP also rises (`22 -> 25`), while duplicate control
gets the same recall and worse FP (`27`). Broader held-out diagnostic views show
the real-anchor signal exists but is noisy: candidate gains one TP on
heldout-representatives (`0.8857 -> 0.9000`) and model-error (`0.8049 ->
0.8293`) slices, yet adds `+9` and `+8` FP versus champion. Treat reviewed real
anchors as a useful direction only with proposal/noise control, hard negatives,
or a better objective; do not promote or simply scale the review39 dose.
A constrained repeat (`freeze=22`, lr `5e-6`, `80` train batches) does not save
the branch: reviewed eval remains tied, train-anchor recall gain is duplicated
by the matched control (`0.6381` for both, candidate has one more FP), and broad
held-out representatives/model-error slices have no recall gain while still
adding `+7/+6` FP versus champion. This kills "just make the head update
smaller" as the review39 repair.

The narrower reviewed-overlap `KHR_5000` staging probe is also killed. Adding
only the `8` reviewed `KHR_5000` anchor images (`24` labels) with the same
freeze-22/lr5e-6/80-step posture improves the tiny four-image trusted eval from
`0.5000/0.5000` to `0.7500/0.6000`, but that pocket is USD-only and the
scorecards still flag prediction/FP growth versus champion and duplicate
control. On the reviewed train-anchor view, candidate and duplicate-control tie
recall (`0.6476`) while candidate precision is worse (`0.7234` vs `0.7391`);
for the actual `KHR_5000` anchors, champion/candidate/control all detect
`11/24`, with the control having `0` FP and the candidate/champion having `1`
FP. Do not infer visible-evidence learning for `KHR_5000` from this probe; the
next overlap move needs more diverse reviewed evidence plus proposal control,
not a narrower positive-only anchor dose.

Partial-visible augmentation now includes both image-edge cuts and in-bounding-box
blockers. The diagnostic roots
`data/processed/cashsnap_real_borderpartial_visibility_p8_v1` and
`data/processed/cashsnap_real_bboxocclusion_visibility_p8_v1` are registered
stress/curriculum sources. The old p24 synth+real champion is already strong on
50%-visible bbox blockers (`val/test recall 0.8889/0.6842`) but weak on 20-30%
visible fragments (`val 0.3810/0.5333`, `test 0.3636/0.4565`), with
`KHR_50000`, `KHR_20000`, `USD_100`, and `USD_50` doing much of the damage.
Training on the synthetic partial rows has not produced a promotable detector:
border-partial micro passed full-test AP but failed partial scorecards; bbox
cap16/20 was clean-identical and had no bbox recall gain; bbox cap16/40 passed
clean guard versus champion (`0.852767 -> 0.850462`) but failed bbox scorecards
(`val recall 0.6154 -> 0.5812`, `test 0.5043 -> 0.5128` with more FP); bbox
cap16/80 improved bbox test recall (`0.5043 -> 0.5556`) while failing clean
(`0.852767 -> 0.842349`, `KHR_50000 -0.199998`) and bbox scorecards; micro4/20
had no measurable effect and micro4/80 hurt clean without beating champion on
partial stress. A freeze-22/head-only bbox cap16/80 schedule is a clean
consolidation clue, not a partial fix: full test improves versus champion
(`0.852767 -> 0.857658`, matched duplicate-control `0.856853`) with per-class
guard passing, but bbox scorecards still fail (`val recall 0.6154 -> 0.5812`,
`test 0.5043 -> 0.5299` with `+14` FP; candidate only `+0.0085` recall over
control on test). Do not keep sweeping cap16/micro4 schedules without a new
mechanism; the next partial move should be real captured/reviewed partials or a
more context-preserving, KHR-protected curriculum aimed at the 20-30% fragment
failure mode.
Resolution is not the hidden fix for bbox-occlusion stress: evaluating the
champion at `imgsz=512/640` does not improve the aggregate (`416` val/test
`0.6154/0.5043`, `512` `0.5812/0.4530`, `640` `0.4359/0.4957`). The hard split
is label/evidence shape: border-cut partials are nearly solved at `50%` visible
(`val/test 0.9565/0.9545`) but poor at `20%` (`0.2963/0.2581`), with many
20%-visible rows visually closer to ambiguous note texture than reliable
counting evidence. A filtered bbox curriculum that removed all `vis0p2` rows and
trained only `30/50%` rows (`min30_p8_cap16`, 40 batches) is also killed:
candidate/control/champion bbox val recall `0.5812/0.5641/0.6154`, test
`0.5385/0.5299/0.5043`, but candidate fails scorecards versus both champion and
control; on the intended 30% slice, duplicate control beats candidate on test
(`0.5435` vs `0.5217`) and val is worse than champion. Do not repeat 20%-row
dosing, higher eval size, or simple `>=30%` filtering as the next partial fix.
Lowering the partial-stress matching IoU is also killed as an explanation:
`iou=0.25` only nudges the hardest bbox 20%-visible test slice (`0.3636 ->
0.3939`) and leaves the broader bbox/border partial read essentially unchanged.

Reset visual QA for partial-visible rows found a sharper rule: trainable partial
evidence must be a single human-countable note, not a stock/texture layout.
Full-size QA showed `billsbank` USD partial rows can contain an unlabeled full
bill plus a labeled fragment, and bbox-occlusion rows often use artificial
rectangular blockers; both are unsafe as bulk curriculum. A source-filtered
KHR-only border/off-frame micro-dose (`24` rows, no `billsbank`, no artificial
blockers, no 20% fragments) still failed: held-out borderpartial
champion/candidate/control recall is val `0.5897/0.5897/0.6026`, test
`0.5526/0.5789/0.5789`; candidate-vs-champion adds FP (`+5/+3`) and
candidate-vs-control loses val recall. Keep the visual filter policy, but do not
promote or scale this KHR-only borderpartial dose.

The next countable-center reset is diagnostic, not promotable. The source-clean
real-derived center/corner root exposed a real champion weakness on source-clean
partials (`conf=0.05` val/test recall `0.6344/0.6667`), but full-size QA found
50%-visible corners can still be non-denomination texture. A stricter center-only
50%-visible view (`52/21/21` train/val/test) is the safer stress slice: champion
scores `0.7143/0.6818` val and `0.8571/0.6207` test. A freeze-22, 80-batch
synth+real micro-dose with those `52` rows gained recall but only by adding FP
and failed strict scorecards versus both champion and duplicate exposure:
candidate/control/champion center50 val `0.8095/0.8095/0.7143` recall and
`0.4595/0.4857/0.6818` precision; test `0.9048/0.9048/0.8571` recall and
`0.4872/0.4872/0.6207` precision. On the reviewed active slice, candidate and
control both keep active-eval recall at `0.5000` but precision falls from
champion `0.2857` to `0.2000`; train-anchor recall rises (`0.7018` candidate,
`0.6842` control, champion `0.6667`) while precision falls badly (`0.5195` /
`0.5065` vs champion `0.6441`). At `conf=0.10`, the center50 recall gain mostly
vanishes and candidate still trails champion/control precision. Class-agnostic
NMS is not a rescue: center50 candidate/control/champion agnostic val recall is
`0.7619/0.7619/0.6667`, but candidate still adds FP versus champion; on test it
trails duplicate-control recall (`0.8571` vs `0.9048`). The visual error pack
`runs/cashsnap/countablepartial_center50_p6_eval_v1/positive_error_review_center50_conf005`
shows the clutter is duplicate/near-denom proposals over real note evidence, not
random background. A true-empty proposal gate and denomination reclassifier also
do not rescue the branch: the gate only gives modest precision moves and can
lose recall, while the candidate post-reclassifier drops center50 val/test to
`0.7143/0.4054` and `0.8571/0.4615`. Class-floor v2 is the only useful clue:
after per-class floors (`KHR_10000=0.50`, `KHR_500=0.15`, `KHR_2000=0.30`,
`KHR_20000=0.40`, `USD_5=0.15`, `USD_100=0.70`, `USD_20=0.15`,
`KHR_5000=0.20`), candidate center50 val/test become `0.8095/0.6538` and
`0.9048/0.5938`, and active eval improves to `0.5000/0.4444`. However, the
duplicate-exposure control matches center50 val/test and active eval exactly,
and is close on active train (`0.6491/0.6066` vs candidate `0.6667/0.6129`);
strict scorecards still fail versus champion because FP rises on center50 and
active train. Applying the same floors to the untouched champion loses recall
(`0.5714/0.8000` center50 val, `0.3750/0.4286` active eval), so this is a
score-distribution/calibration clue, not a safe champion-only knob. Combining
class-floor v2 with class-agnostic NMS is cleaner but still not a detector win:
candidate center50 val/test become `0.7619/0.7619` and `0.8571/0.7200`, active
eval/train `0.5000/0.5000` and `0.6491/0.6491`; the duplicate-control matches
val/active and beats candidate on center50 test (`0.9048/0.7308`), so the
candidate fails the combined-postprocess scorecard against control. Raising the
YOLO class-loss gain (`cls=2.0`) in the same freeze-22/lr5e-6/80-step no-aug
head tune is also killed: valid no-aug candidate center50 val/test is
`0.8095/0.4722` and `0.9048/0.4872`, active eval/train `0.5000/0.2000` and
`0.7018/0.5195`; the valid no-aug duplicate-control is equal or better on
center50 precision and scorecards still fail versus both control and champion.
Ignore `fixed_step_cls2_loss_probe_summary.json` for conclusions because the
first fixed-step attempt was augmentation-confounded; use the direct no-aug
`countablepartial_center50_p6_*_cls2_noaug_*` runs instead. Keep
`configs/webgl_ablation/cashsnap_real_countablepartial_sourceclean_center50_p6_eval_v1.yaml`
as a cleaner diagnostic slice, but do not promote or scale center-crop positives
without accepted no-note/unknown-money negatives, calibrated objectness/class
suppression, or a training objective that beats duplicate exposure.

Countable-center plus mined train-empty FP negatives is QA-only, not an approved
detector dose. Configs
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_countablepartial_center50_p6_fpneg32_v1.yaml`
and duplicate-empty control
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_countablepartial_center50_p6_fpneg32_dupemptyctrl_v1.yaml`
were generated with clean fixed-step preflight (`719/719` rows, 80 steps, no
phase warnings), but the added-row QA packet
`runs/cashsnap/countablepartial_center50_p6_fpneg32_eval_v1/fpneg32_added_negative_qa_v1/`
shows the 32 zero-label rows are semantically mixed unknown/out-of-schema money:
`usd_total 10`, `khmer_us_currency 6`, `asian_currency 3`,
`cambodia_currency_project 2`, and `cashcountingxl 11`. Do not train this as
plain background without review decisions or an explicit unknown/schema route.
The current browser proposal gate already handles most of the slice:
`proposal_gate_fpneg32_added_negative_rej072_summary.json` trims detector
background-FP images `22 -> 5/32` and predictions `32 -> 7`; the residual kept
overlay sheet shows mostly `USD_2` rows plus one KHR-like out-of-schema
Cambodian note, not random background. `scripts/check_currency_taxonomy_coverage.py`
confirms the 13-class model schema is still missing `USD_2`, `KHR_50`,
`KHR_100`, `KHR_200`, `KHR_15000`, `KHR_30000`, `KHR_100000`, and
`KHR_200000`. Treat this as evidence for official21/unknown-routing or a
proposal-gate/schema fix, not for teaching the 13-class detector to suppress
real banknotes as empty labels.
The residuals now have a tiny reviewed official21 seed packet:
`runs/cashsnap/countablepartial_center50_p6_fpneg32_eval_v1/official21_residual_missing_schema_review_v1/`
contains accepted proposal CSVs for `USD_2` (`4` boxes) and `KHR_200` (`1`
box), plus overlay/crop sheets. Materializer dry-runs pass for both
class-specific CSVs with `--scope proposal_images --splits train`:
`official21_residual_usd2_proposal_images_dryrun_v1` accepts `4/4` boxes and
`official21_residual_khr200_proposal_images_dryrun_v1` accepts `1/1`. This is
not enough to train, but it is the correct direction for the five surviving
gate failures: accumulate reviewed missing-schema labels, then rebuild an
official21 bridge under registry control, instead of adding them as 13-class
empty negatives.
The larger USD_2 missing-schema queue now exists as review evidence, not labels:
`configs/generated_lists/audit/cashsnap_missing_schema_usd2_top52_from_bgfp_v1.txt`
selects all `52` unique `usd_total_2Dollar` rows from the top-200 train-empty
FP probe, and
`runs/cashsnap/missing_schema_usd2_top52_review_v1/proposal_gate_usd2_top52_rej072_summary.json`
shows the current browser gate only partially suppresses them (`84` detector
proposals, `21` rejected, `63` kept; background-FP images `52 -> 39/52`). The
blank-decision official21 packet
`runs/cashsnap/missing_schema_usd2_top52_review_v1/official21_usd2_top52_review_queue_v1/overlays/`
renders `52` USD_2 proposal overlays/crops with no skipped images; top proposal
current-class mix is `USD_50 29`, `USD_100 15`, `USD_1 8`. Use
`scripts/build_official21_proposals_from_gate_json.py` to regenerate similar
proposal CSVs from gate JSON. Do not materialize or train from this queue until
boxes are reviewed and any new official21 root is registered. Saved-proposal
sweeps (`proposal_gate_usd2_top52_threshold_sweep_low_v1.csv`) show threshold
tuning alone cannot solve USD_2: even rejecting every reject-top crop
(`reject>=0.01`) leaves `34/52` images and `55` predictions, because most
surviving crops are gate-`target`. This needs explicit schema expansion,
unknown-money routing, or a stronger gate trained on reviewed missing-schema
banknotes. Materializer safety dry-run
`official21_usd2_top52_blank_materializer_dryrun_v1` accepts `0` boxes from the
blank-review queue, as intended. Prior visual acceptances from the tiny residual
packet were transferred into
`review_queue_prior_accept4_v1.csv` via
`scripts/transfer_review_acceptances_by_iou.py` only after box IoU matched the
new queue at `>=0.9999`; dry-run `official21_usd2_top52_prior_accept4_materializer_dryrun_v1`
accepts exactly `4` train boxes, leaving the other `48` USD_2 rows blank.
Codex visual review expanded the USD_2 queue conservatively in
`review_queue_codex_visual_accept37_v1.csv`: `37` rows are `accepted_box`, while
`15` are held because they are portrait-only/low-evidence or composite images
with a second visible USD_2 surface outside the proposed box. The registered
diagnostic bridge
`runs/cashsnap/missing_schema_usd2_top52_review_v1/official21_usd2_top52_codex_visual_accept37_v1/materialized`
has config
`configs/official21/cashsnap_official21_usd2_top52_codex_visual_accept37_v1.yaml`
and registry id
`cashsnap_official21_usd2_top52_codex_visual_accept37_review_bridge_v1`; it is
train-only review evidence, not a replacement root.
The same missing-schema review route now covers the top Cambodia-project FP
cluster without pretending every row is `KHR_200`:
`configs/generated_lists/audit/cashsnap_missing_schema_cambodia_project_top17_from_bgfp_v1.txt`
selects `17` `cambodia_currency_project_IMG_57xx` current-schema empty rows, and
`runs/cashsnap/missing_schema_cambodia_project_top17_review_v1/proposal_gate_cambodia_top17_rej072_summary.json`
shows browser-gate leakage `17 -> 10/17` background-FP images (`21` detector
proposals, `13` kept at `reject>=0.72`). The generic denomination-triage packet
`runs/cashsnap/missing_schema_cambodia_project_top17_review_v1/official21_cambodia_top17_review_queue_v1/overlays/`
has `17` overlays/crops, `proposed_new_class` intentionally blank, and no skipped
images; reviewers must fill the verified official21 denomination before any
`accepted_box`. Threshold sweep lower bound (`reject>=0.01`) still leaves `6/17`
images and `8` predictions, so gate threshold alone is not a clean schema fix
and could over-reject real targets. Blank materializer dry-run
`official21_cambodia_top17_blank_materializer_dryrun_v1` accepts `0` boxes.
The prior visual KHR_200 acceptance transfers into
`review_queue_prior_khr200_accept1_v1.csv` via the same IoU helper at box IoU
`0.9999`; dry-run
`official21_cambodia_top17_prior_khr200_accept1_materializer_dryrun_v1` accepts
exactly `1` train box while leaving the other `16` rows unresolved.
Mixed missing-class seed consolidation now exists but is still only a seed:
`runs/cashsnap/official21_missing_schema_seed_accept11_v1/combined_accepted_proposals.csv`
combines reviewed `KHR_100 6`, `USD_2 4`, and `KHR_200 1` proposal boxes.
`scripts/materialize_cashsnap_official21_review_bridge.py` supports opt-in
`--proposal-class any` for such mixed official21 accepted CSVs while preserving
strict single-class mode by default. The registered diagnostic root
`runs/cashsnap/official21_missing_schema_seed_accept11_v1/materialized` is now
materialized with config
`configs/official21/cashsnap_official21_missing_schema_seed_accept11_v1.yaml`;
registry check passes with this root as
`cashsnap_official21_missing_schema_seed_accept11_review_bridge_v1`. It contains
only `11` train images/boxes (`KHR_100 6`, `USD_2 4`, `KHR_200 1`), so it proves
the mixed review-to-official21 path but is still far too small/unbalanced for a
replacement detector.
Future official21 training handle:
`configs/official21/cashsnap_official21_roboflow_plus_current_accept6_cap180_empty360_plus_accept11_usd2_khr200_v1.yaml`
extends the existing Roboflow+current accept6 balanced replay with only the
reviewed `USD_2 x4` and `KHR_200 x1` rows from accept11 (filters out duplicate
KHR_100 rows), producing `4370` train rows. This config is not run evidence and
is not promotable without many more reviewed USD_2/KHR_200 labels plus
current-domain/count guards. Preflight
`runs/cashsnap/official21_missing_schema_seed_accept11_v1/fixed_step_accept11_mix_preflight_v1.json`
records the expected `+5` candidate row delta versus the base accept6 balanced
config and warns that fixed steps stop at different scheduler phases; use it as
a future training handle, not an exact A/B unless a matched control is built.
Scheduler-only control
`configs/official21/cashsnap_official21_roboflow_plus_current_accept6_cap180_empty360_plus_accept11_usd2_khr200_rowcountctrl_v1.yaml`
adds `5` empty rows to match row count (`4370/4370`) because no exact
`USD_2/KHR_200` duplicate exposure exists in the base. Preflight
`runs/cashsnap/official21_missing_schema_seed_accept11_v1/fixed_step_accept11_vs_rowcountctrl_preflight_v1.json`
has no train-phase warnings, but the control is explicitly class-mix inexact and
should only bound scheduler/row-count effects.
Ran the initial accept11 seed A/B on 2026-06-11:
`runs/cashsnap/fixed_step_accept11_rowcountctrl_vs_accept11_usd2_khr200_steps160_seed0/summary.json`
has official21 test `mAP50-95` delta `-0.000028`, and reviewed-box probes under
`runs/cashsnap/official21_missing_schema_seed_accept11_v1/probe_accept11_*`
show `0/4` `USD_2` and `0/1` `KHR_200` target hits at normal `conf=0.05`.
At `conf=0.001`, both the row-count control and candidate emit weak target boxes,
so the low-confidence signal is not evidence that the seed learned usable new
classes.
Repeat24 dosing of the same reviewed rows also failed:
`configs/official21/cashsnap_official21_roboflow_plus_current_accept6_cap180_empty360_plus_accept11_usd2_khr200_repeat24_v1.yaml`
adds `96` extra `USD_2` and `24` extra `KHR_200` exposures, but
`runs/cashsnap/fixed_step_accept11_repeat24_emptyctrl_vs_accept11_repeat24_usd2_khr200_steps160_seed0/summary.json`
drops official21 test `mAP50-95` by `-0.002945` with worst real class `USD_10`
`-0.048236`, while reviewed-box probes still show `0/4` `USD_2` and `0/1`
`KHR_200` target hits at `conf=0.05`. Do not promote or rerun simple accept11
repeat dosing; the next official21 expansion needs more reviewed labels, a
safer head/class-expansion recipe, or a separate new-class adapter/gate.
Accept11-only overfit diagnostic
`runs/cashsnap/overfit_accept11_seed_official21_from_last_e100_i416_b2_w0_adamw_lr1e3_nowarmup_noamp_cachefalse_steps300_seed0`
trained on clone config
`configs/official21/cashsnap_official21_missing_schema_seed_accept11_overfit_trainval_v1.yaml`
and proves the labels/class mapping can be learned: review probes at `conf=0.05`
hit `USD_2 3/4` and `KHR_200 1/1` (`KHR_200` confidence `0.234428`; `USD_2`
hit confidences `0.086734-0.202842`). One `USD_2` row still has no target
prediction. Read: the failure is mixed-replay transfer/dosing/calibration, not a
broken label materialization path.
The larger accept37 USD_2 bridge confirms and sharpens that lesson. A matched
head-only A/B from the champion
(`fixed_step_usd2_accept37_emptyctrl_headonly_vs_usd2_accept37_headonly_steps160_seed0`)
slightly improves official21 test mAP50-95 (`+0.003380`, worst real class
`USD_100 -0.015354`) but still gets `0/37` reviewed USD_2 target hits at
`conf=0.05`; low-confidence matched target confidence only shifts
`0.001400 -> 0.001474` with `19` rows up and `18` down. Repeat12 dosing
(`cashsnap_official21_roboflow_plus_current_accept6_cap180_empty360_plus_usd2_accept37_repeat12_v1.yaml`,
`481` USD_2 exposures total) also gets `0/37` normal-threshold hits and fails the
real-class guard (`mAP50-95 -0.005737`, worst `USD_10 -0.125885`). A frozen
head-only overfit/prewarm on the `37` accepted rows
(`overfit_usd2_accept37_headonly_from_last_e100_i416_b2_w0_adamw_lr1e3_nowarmup_noamp_cachefalse_freeze23_steps300_seed0`)
does hit `37/37` at `conf=0.05`, proving the labels are learnable without
changing the backbone. But replaying from that prewarmed head with repeat12 only
keeps `19/37` hits and leaves official21 old-class performance unacceptable
(`fixed_step_usd2prewarm_repeat12_emptyctrl_headonly_vs_usd2prewarm_repeat12_headonly_steps160_seed0`,
candidate mAP50-95 `0.090638`, delta `-0.007419`, worst `USD_10 -0.072700`).
The opt-in missing-class micro-adaptation path in
`scripts/build_yolo_micro_adaptation_config.py` can now build row-matched
candidate/control configs when official21 names include classes absent from the
base (`--extra-classes-only --allow-missing-base-classes
--missing-extra-control empty`). The first balanced micro probe
(`cashsnap_official21_micro_usd2_accept37_bal16_empty64_v1.yaml`) is also a
negative result: `325` rows with old-class rehearsal and `USD_2 x37` gets only
`1/37` reviewed USD_2 hits at `conf=0.05`, while official21 mAP50-95 is lower
than the empty control (`0.094626`, delta `-0.004572`, worst
`USD_100 -0.104597`). Do not rerun simple accept37 append/repeat/prewarm or tiny balanced
micro recipes. The next official21/USD_2 attempt needs either a materially
larger official21 training regime, a class-expansion recipe that avoids
destroying existing head calibration, or a separate crop-level new-class
adapter/gate; more copies of the reviewed USD_2 rows are not enough.
The source-named CashSnap v1 USD_2 pseudo-label branch gives a larger reservoir
but does not rescue the scratch-expanded official21 head by itself. The
registered diagnostic root
`runs/cashsnap/missing_schema_usd2_filename_pseudolabel_v1/usd2_filename_conf015_area025_pseudo_official21_v1/materialized`
contains `USD_2` pseudo boxes from current detector top proposals on
`usd_total_2Dollar` images (`232` train, `65` val, `27` test) selected with
`confidence>=0.15` and area `>=0.25`; overlay samples show clear single USD_2
surfaces, but labels are source-name/detector pseudo labels, not per-row manual
ground truth. The 300-step head-only A/B from the champion
(`fixed_step_usd2_filename_pseudo232_emptyctrl_headonly_vs_usd2_filename_pseudo232_headonly_steps300_seed0`)
is not promotable: official21 test mAP50-95 is flat (`+0.000468`) while
old-class safety fails (`USD_100 -0.074909`, `USD_10 -0.047294`), and the
deploy-threshold review probe on the `65` pseudo-val rows gets `0/65` USD_2
target hits for both candidate and empty control despite `59/65` any-class
localization hits. Read: the detector already localizes many USD_2 notes, but
the scratch-expanded head still calls them old USD classes; AP on the pseudo
bridge is a calibration/ranking proxy and must not be treated as deployable
USD_2 recovery.
`scripts/build_yolo_official21_mapped_checkpoint.py` is the new class-safe
official21 initialization path. It builds
`runs/cashsnap/official21_mapped_init_v1/weights/official21_mapped_from_champion_nearestlowbias.pt`
by copying `696` tensors exactly and remapping the `12` class-output tensors by
class name (`core13 -> official21` ids); missing official classes borrow the
nearest same-currency row with bias offset `-2.0`. This unmixed mapped
checkpoint scores official21 base test `mAP50-95=0.766` and already gets
`10/65` USD_2 pseudo-val target hits at `conf=0.05`, versus about `0.210`
official21 test mAP and `0/65` USD_2 hits for the scratch-expanded 300-step
pseudo232 heads. Next USD_2 detector training should start from the mapped
checkpoint and be judged by deploy-threshold target hits plus old-class/FP
guards; do not launch more scratch-expanded official21 head-only doses. First
mapped-init low-LR probe
(`fixed_step_usd2_filename_pseudo232_emptyctrl_mappedinit_headonly_vs_usd2_filename_pseudo232_mappedinit_headonly_steps160_seed0`,
`lr0=3e-4`, `freeze=23`) is a useful but incomplete positive: official21 test
mAP50-95 is essentially flat versus the empty control (`0.740196 -> 0.739968`,
delta `-0.000228`, worst class `KHR_2000 -0.017072`, no `-0.05` class breach),
and deploy-threshold pseudo-val USD_2 hits improve from mapped-init baseline
`10/65` and empty-control `8/65` to candidate `15/65` while all `65/65` rows
still have any-class localization. This is the first missing-class run that
moves USD_2 in the right direction without destroying old-class calibration, but
coverage remains far below deployable; next steps should improve USD_2 class
confidence/precision from the mapped init and inspect duplicate/FP growth before
any product-stack claim. The follow-up safety audit makes the current blocker
explicit: on official21 non-USD_2 test rows, both the mapped-init empty control
and candidate emit `9` USD_2 predictions on `8` images after the proposal gate,
and a USD_2 confidence sweep cannot separate signal from leakage (`15/65`
pseudo-val hits with `9` non-USD_2 FPs at `0.05`, falling to `0/65` hits while
still leaving `6` FPs by `0.50`). Full-size overlays in
`runs/cashsnap/missing_schema_usd2_filename_pseudolabel_v1/visual_qa_official21_test_usd2_fp_candidate_v1/`
show clear USD_1 bills predicted as USD_2 at high confidence (`0.90`, `0.99`,
`0.65`, gate target near `1.00`), plus a low-confidence hand-occluded USD_5
fragment predicted as USD_2. Do not promote the pseudo232 mapped candidate or
try to fix it with a simple USD_2 threshold; the next missing-class attempt
needs explicit USD_1-vs-USD_2 contrast/rehearsal or a class-calibration adapter.
Prepared the first contrastive mapped-init A/B but did not launch training yet:
candidate
`configs/official21/cashsnap_official21_usd2_pseudo232_plus_usd1repeat1_v1.yaml`
starts from the USD_2 pseudo232 mix and appends one extra copy of all `234`
USD_1 train rows (`237` USD_1 labels, combined `4831` rows); control
`configs/official21/cashsnap_official21_usd2_pseudo232_plus_empty234ctrl_v1.yaml`
appends `234` repeated empty-label rows instead, also `4831` rows. Summaries are
under `runs/cashsnap/missing_schema_usd2_filename_pseudolabel_v1/usd2_pseudo232_plus_*_config_summary.json`.
Preflight
`runs/cashsnap/fixed_step_usd2p232_empty234ctrl_vs_usd1repeat_mappedinit_steps160_seed0/preflight.json`
passed with equal row count, `160` fixed batches, and no train-phase warnings.
Use the same mapped-init head-only settings as the prior pseudo232 probe
(`freeze=23`, `lr0=3e-4`, no warmup, no AMP/cache, batch `2`) if launching it.
The trained result is killed for USD_2 recovery despite a better official21
aggregate: candidate/control official21 test mAP50-95 is `0.747968/0.741426`,
but pseudo-val USD_2 hits move the wrong way (`13/65` candidate vs `15/65`
empty control) and official21 non-USD_2 test leakage is unchanged (`9` USD_2
FPs on `8` images for both, mostly USD_1/USD_5). Do not promote or scale the
USD_1-repeat branch; it confirms that simple USD_1 rehearsal is not enough to
calibrate USD_2.
`scripts/check_official21_accept11_artifacts.py` is the cheap resume guard for
this bridge; it verifies summary/materialized counts, registry entry, train-list
existence, and the exact candidate/control delta (`USD_2 x4`, `KHR_200 x1`).
`scripts/check_cashsnap_detector_launch_readiness.py` wraps that guard with
browser-stack artifact checks, taxonomy status, matched train-row counts, and live
RAM; it writes
`runs/cashsnap/official21_missing_schema_seed_accept11_v1/launch_readiness_latest.json`
for the default accept11 handle and custom reports for later candidate/control
pairs. Use `--try-memory-cleaners` to run the configured scheduled memory
cleaners once and record the before/after RAM in the report before deciding
whether to launch.
The top-200 train-empty FP audit
`runs/cashsnap/missing_schema_false_positive_audit_v1/top200_bucket_summary.json`
confirms the broader pattern: the largest buckets are non-target money or
missing-schema notes, not ordinary empty backgrounds (`cashcountingxl`/coins
`54`, `USD_2` `52`, KHR_100 candidates `35`, `asian_currency` foreign money
`33`, Cambodia-project denomination triage `17`). This argues against another
blind 13-class empty-negative dose from train-empty labels; split the problem
into official21 expansion, reviewed unknown-money/gate training, and true-empty
background calibration. Consolidated gate scorecard:
`runs/cashsnap/missing_schema_false_positive_audit_v1/top200_bucket_gate_scorecard_v1.json`.
The current browser gate mostly helps the largest
non-banknote-money bucket:
`runs/cashsnap/nonbanknote_cashcountingxl_top54_gate_eval_v1/proposal_gate_cashcountingxl_top54_rej072_summary.json`
reruns the top `54` `cashcountingxl` rows and trims browser-stack detections
from `18 -> 7/54` background-FP images (`32 -> 13` predictions). Threshold
sweep lower bound (`reject>=0.01`) still leaves the same `7/54` images and `12`
predictions because the residuals are gate-`target`; this is a reviewed
unknown/gate-training gap, not proof that broad detector empty-negative dosing
is safe.
For KHR_100 candidates, the current product workaround is much stronger:
`runs/cashsnap/missing_schema_khr100_top35_gate_eval_v1/proposal_gate_khr100_top35_rej072_summary.json`
runs the `35` top `khmer_us_currency_100-riel` rows and suppresses all `49`
detector proposals (`KHR_20000 35`, `KHR_50000 14`) at `reject>=0.72`; the
threshold sweep stays at `0/35` background-FP images through `reject>=0.80`.
That is acceptable only as a current 13-class unknown-route behavior. It also
proves the detector still has no KHR_100 denomination support; official21
training must turn reviewed KHR_100 evidence into target labels rather than
relying on this rejection path.
Foreign-money `asian_currency` top-FP rows are already gate-controlled:
`runs/cashsnap/foreign_asian_currency_top33_gate_eval_v1/proposal_gate_asian_currency_top33_rej072_summary.json`
suppresses all `36` detector proposals on the `33` top rows at `reject>=0.72`
(`KHR_1000 8`, `KHR_2000 12`, `KHR_50000 14`, plus two USD proposals), and the
sweep remains `0/33` background-FP images through `reject>=0.90`. Do not repeat
the earlier broad foreign-hard-negative detector dose unless new evidence shows
foreign-money leaks outside the current gate's coverage.
The small Khmer-source tail is thresholdable but not solved at the current
operating point:
`runs/cashsnap/khmer_source_other_empty_top9_gate_eval_v1/proposal_gate_khmer_other_top9_rej072_summary.json`
has `15` detector proposals on `9` images, all gate-`reject`; `reject>=0.72`
leaves `3/9` background-FP images, while `reject<=0.50` clears them. Treat this
as calibration/review evidence, not a detector-empty-negative recipe.
`scripts/build_official21_proposal_review_overlays.py` now derives review IDs,
README wording, and summary scope from each row's `proposed_new_class`; older
packets may still show a hardcoded `khr100_*` prefix even for non-`KHR_100`
classes.

Mining existing real train images for off-frame/partial stress is a real clue,
not a promotion. The cap8 candidate adds `88` train-split stress rows across
`11` classes and has a matched `88`-row duplicate-exposure control; it still
does not add new `KHR_20000/KHR_50000` stress coverage. The unfrozen
`lr=1e-5`, `40`-batch update improved real partial-edge recall at `conf=0.05`
versus control (`val/test 0.9610/0.9545` vs `0.9509/0.9364`) and kept full
clean AP alive (`0.854053` vs champion `0.852767`), but failed the
source-excluded guard (`0.771022` vs champion `0.782187`; duplicate control
also failed at `0.761500`), so the update size was too loose. A reduced
head-only repeat (`freeze=22`, `lr=5e-6`, `80` batches) protects the clean
foundation better: full clean `0.858924`, semantic-clean `0.863698`, and
source-excluded `0.788748` all pass versus the champion, while duplicate control
also passes (`0.854199` / `0.863126` / `0.793873`). It preserves the real
partial-edge recall signal at `conf=0.05` (`candidate/control/champion` val
`0.9698/0.9622/0.9534`, test `0.9576/0.9424/0.9424`) and the signal survives a
`conf=0.10-0.20` sweep, but every strict transfer scorecard still fails because
candidate FP and images-with-FP rise versus both champion and duplicate control
(`conf=0.05` FP deltas `+64/+71` vs champion and `+46/+42` vs control on
val/test). Candidate-vs-control FP-delta sheets show the extra FPs are mostly
duplicate/class-confusion proposals on real banknote evidence, not random
background clutter, which is still bad for counting because it inflates notes.
Shared USD class-floor probing (`USD_50=0.30`, other common USD classes `0.20`)
reduces raw overfire but still fails the same candidate-vs-control scorecard
(`+39/+29` FP on val/test), so this is not just a low-confidence tail.
Seed1 repeats the same shape with weaker upside: clean full-test passes versus
champion and duplicate control (`0.854215` vs `0.852767`/`0.852288`, worst
per-class drop `KHR_50000 -0.0334`), but real partial-edge scorecards still fail
FP gates (`+41/+54` FP versus champion and `+14/+16` versus matched control on
val/test) for small recall lifts (`+0.0038/+0.0045` over control).
Pairing the mined partial rows with a small train-empty FP-negative dose is also
not a rescue. A fresh champion probe on train empty-label rows found
`1841/6918` images with low-conf FPs, then a `32`-image spread dose was added on
top of mined-real partialstress cap8 with a duplicate-empty control. The
candidate still fails partial-edge scorecards versus champion on val/test
(`+26/+25` FP) and loses recall to its duplicate-empty control on val
(`0.9622` vs `0.9673`); it only ties control recall on test while trimming `5`
FP. Empty val/test background behavior is mixed rather than fixed: val worsens
versus champion/control (`337` images with FP vs `326/331`), while test images
with FP improve (`165` vs `171/172`) but total detections remain slightly above
champion (`279` vs `274`). Treat mined train-empty FPs as useful calibration
evidence, not as a proven antidote to partial-row overproposal.
The reviewed-real edge-clamp micro-dose is also killed. Automatic exact-edge
geometry was too weak after full-size visual QA because it admitted clean
full-note tabletop crops, so a curated `10`-row list kept only countable
phone/hand/occluded/off-frame rows and used a matched duplicate-exposure
control. The candidate raised real partial-edge recall versus champion
(`val/test 0.9534/0.9424 -> 0.9647/0.9500`) but failed FP gates (`+43/+50` FP,
`+30/+36` images with FP); against the duplicate control it had zero recall gain
and worse FP/precision (`val/test FP +9/+31`, precision `0.6070/0.6437` vs
control `0.6113/0.6649`). Empty-background probes also disfavor it: val/test
detections `535/274 -> 561/285`, while duplicate control is `545/274`. Do not
scale visually reviewed positive-only real partials without explicit proposal
control or harder no-note/unknown-money supervision. Class-agnostic NMS only
softens the duplicate-proposal damage and is not a rescue: candidate-vs-champion
still fails partial-edge FP gates (`+22/+35` FP), and candidate-vs-control still
fails with tiny recall deltas (`+0.0038/+0.0015`) and extra FP (`+1/+21`).
Treat this as evidence that mined real partial/off-frame examples can teach
visible-evidence recovery, while the current unreviewed row policy also teaches
overproposal. The next partial move needs visual QA of the mined rows,
human-identifiable fragment policy, class/source-protected sampling, and/or
paired hard negatives/objectness calibration; do not simply scale cap8 mined
partials.

The strict reviewed KHR partial micro-dose is killed as a detector update but
sharpens the data policy. Full-size QA packet
`runs/cashsnap/visible_evidence_qa_mined_partialstress_cap8_v1/` shows the cap8
mined-partial source is contaminated: edge-touch rows include clean full-note
tabletop crops and unsafe one-label multi-note USD strip/crop scenes. A Codex
reviewed strict subset
`strict_partial_khr_codex_reviewed_v1.csv` keeps only `6` single-note,
human-countable KHR frame-cut rows (`KHR_500 x1`, `KHR_2000 x1`, `KHR_5000 x2`,
`KHR_10000 x2`) and excludes full-note/multi-note ambiguity. Candidate config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_strictpartial_khr6_v1.yaml`
and exact duplicate-control config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_strictpartial_khr6_dupctrl_v1.yaml`
were trained from the champion with the standard freeze-22/lr5e-6/80-step
posture. Full-test mAP still loses to duplicate exposure (`0.855949 ->
0.855379`, worst `KHR_2000 -0.005251`). On the six reviewed rows,
champion/control/candidate recall is tied at `0.8333`; candidate only recovers
champion precision (`0.5556`) while control is lower (`0.5000`). Broader
visible-evidence scorecards fail: candidate-vs-control
`scorecard_strictpartial_khr6_candidate_vs_dupctrl_conf005_v1.json` has no
recall gain and fails FP/prediction gates on active eval and center50 val;
candidate-vs-champion
`scorecard_strictpartial_khr6_candidate_vs_champion_conf005_v1.json` has no
recall gain on any focused view and adds FPs on active eval/train and center50
val. Lesson: visual QA is necessary but not sufficient; tiny positive-only
partial positives are now exhausted under this schedule. Pairing the same `6`
strict KHR positives with `12` visually reviewed US coin/non-banknote
hard-negative rows is also killed. Candidate config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_strictpartial_khr6_coinhardneg12_v1.yaml`
loses to the exact duplicate control on full test (`0.852049 -> 0.851214`,
worst `KHR_2000 -0.013758`) and loses harder versus the champion (`0.852767 ->
0.851214`, worst `KHR_50000 -0.053262`). Focused scorecards
`scorecard_strictpartial_khr6_coinhardneg12_candidate_vs_dupctrl_conf005_v1.json`
and
`scorecard_strictpartial_khr6_coinhardneg12_candidate_vs_champion_conf005_v1.json`
fail with no partial recall gain, extra FP/predictions on strict KHR, active,
and center50 views, and no coin false-positive relief: coin background images
with FP are champion/control/candidate `8/9/9` of `12`. The next move needs
proposal-mined no-note/unknown-money supervision, a stronger objectness/proposal
objective, or larger reviewed real captures; do not repeat tiny random
coin/empty doses or another edge-touch positive filter. A stricter
proposal-mined coin packet confirmed the same boundary: the champion low-risk
empty probe found `23` visually safe `cashcountingxl` coin rows from `158/2472`
FP images, excluding `USD_2`/out-of-schema notes, but
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_strictpartial_khr6_cointopfp23_v1.yaml`
still loses to exact duplicate exposure on full test (`0.853755 -> 0.852730`)
and just barely loses to the champion (`0.852767 -> 0.852730`). Focused
scorecard
`scorecard_strictpartial_khr6_cointopfp23_candidate_vs_dupctrl_conf005_v1.json`
shows small proposal thinning versus control (`coin_topfp23` detections `-4`,
strict/active eval FP `-1`) but no real coin suppression: champion/control/
candidate all have FP images on `23/23` top-FP coins. Versus the champion,
`scorecard_strictpartial_khr6_cointopfp23_candidate_vs_champion_conf005_v1.json`
still fails active, center50, and coin FP gates. Conclusion: coin rows alone can
nudge duplicate proposals but do not teach the detector to abstain on full-frame
non-banknote money; use a learned proposal gate/unknown route or larger,
reviewed non-banknote/background policy before spending another detector dose.
The existing MobileNetV3 true-empty proposal gate has only a small diagnostic
save here. Uncapped `reject>=0.50` trims coin FPs but rejects true center50
partials; adding a detector-confidence cap is safer, with
`reject>=0.70`, `det_conf<=0.25-0.40` preserving strict/active/center50 recall
while reducing combined coin FP images by `4` and coin detections by `5`
(`gate_e4_lowdetcap_policy_selection_summary_conf005_v1.json`). More aggressive
coin cleanup immediately costs center50 recall. Treat this as evidence for a
better reviewed gate/unknown branch, not a product promotion.

The reviewed Korean-won foreign hard-negative micro-dose is killed as a promotion
candidate. First-pass visual QA routed official-but-currently-unknown
`KHR_100`/`USD_2` rows to unknown/out-of-scope and used `24` Korean-won clusters
as zero-label foreign hard negatives; after duplicate removal the train dose was
`+22` rows, matched by a duplicate-empty exposure control. Fresh shared-error v2
evals show no recall gain over the champion (`val/test 0.5333/0.5500` for both)
and extra FP (`+4/+1`), while candidate versus duplicate control is effectively
flat (`val recall tie, test +0.0500`) with no background-FP-image relief.
Empty-label val/test also does not improve (`325/173` images with FP versus
champion `325/171`; detections `540/278` versus `535/274`). Partial-edge recall
ticks up (`0.9559/0.9500 -> 0.9584/0.9561`) but only by adding proposals
(`+31/+21` FP and `+19/+13` images with FP), and the duplicate control shadows
it. Source-aware foreign hard negatives are still conceptually useful, but this
tiny zero-label dose mainly behaves like empty exposure, not learned
unknown-money suppression.

Real-bbox paste overlap is now a registered diagnostic eval bridge, per the
user's suggested half-synth direction. `scripts/build_yolo_bbox_paste_overlap_dataset.py`
creates crude rectangular crops from real YOLO boxes, pastes them into
overlap/fan/off-frame scenes, and labels only visible instance masks. The first
root is `data/processed/cashsnap_real_bboxpaste_overlap_eval_v1` (`120/80/80`
train/val/test scenes, `1109` visible labels) and is registered as diagnostic,
not working/trainable. Visual QA says the rectangle seams are obvious but useful
as an overlap/counting stressor; do not train from it until a filter/policy says
the scenes are human-identifiable enough. On this bridge at `conf=0.05`,
champion/candidate/control recall is val `0.3870/0.3777/0.3839`, test
`0.3675/0.3841/0.3775`. All strict scorecards fail; the mined-real candidate is
not a robust pasted-overlap winner, with per-class drops such as val
`KHR_1000 -0.1212` and test `KHR_20000 -0.1429` versus champion. Keep the bridge
for eval and generator iteration; `iou=0.25` only raises everyone by roughly
`0.04-0.05` recall, so the stress is not just strict visible-box matching. The
next useful version should reduce pasted rectangle artifacts, balance
class/source pressure, and maybe include reviewable real contexts before any
training dose.

The class-balanced follow-up
`data/processed/cashsnap_real_bboxpaste_overlap_balanced_eval_v1` reduces the
USD/source skew (`1058` labels, roughly `72-99` per class) and gives a cleaner
partial-overlap denominator. On this balanced bridge at `conf=0.05`,
champion/candidate/control recall is val `0.4271/0.4305/0.4271`, test
`0.3885/0.4088/0.4020`. Candidate-vs-champion still fails FP gates (`+7/+24`
FP on val/test), but candidate-vs-duplicate-control is more encouraging: val
has `+0.0034` recall with `-12` FP and fails only one per-class guard
(`KHR_1000 -0.0526`), while test passes with `+0.0068` recall and `-10` FP.
Candidate-vs-champion FP sheets show the unsafe deltas are mostly large
duplicate/class-echo boxes on visible pasted notes, not background clutter.
Class-agnostic NMS is diagnostic but not a rescue: it makes the balanced
bbox-paste candidate-vs-champion val split pass (`+0.0068` recall, `-2` FP),
but bbox-paste test still fails (`+0.0101` recall, `+6` FP plus per-class
blocker) and real partial-edge still fails FP gates (`+32/+47` FP versus
champion on val/test). Keep agnostic NMS as a deploy/postprocess knob to test,
not as evidence the detector branch is promotion-safe.
Seed1 confirms the overlap proxy is not stable enough for promotion: class-aware
bbox-paste candidate recall is val/test `0.4373/0.3986` versus champion
`0.4271/0.3885` and matched control `0.4271/0.3986`. Candidate-vs-control passes
val (`+0.0102` recall, `-16` FP) but test fails per-class despite `-15` FP and
equal aggregate recall; candidate-vs-champion still fails FP gates (`+8/+15`).
With agnostic NMS, candidate loses to control on val recall (`-0.0034`) and only
ties control on test while reducing FP, so suppression is a product knob, not a
detector proof.
Interpretation: mined-real partial rows may help visible-evidence recovery
beyond duplicate exposure on a balanced overlap proxy, but the effect is small
and not champion-safe; use this bridge for focused follow-up, not promotion.

The feathered bbox-paste bridge
`data/processed/cashsnap_real_bboxpaste_overlap_balanced_feather_eval_v1` is a
cleaner diagnostic denominator, not trainable evidence. It uses the same balanced
real-box paste idea with `--cutout-feather-frac 0.035` so crop seams are less
hard, and visual QA says full-size rows are less jarring than the hard-edge
bridge but still synthetic stressors. Champion `conf=0.05` recall/precision is
val `0.4155/0.3555` and test `0.3658/0.3375`, versus the old balanced bridge
val `0.4271/0.3452` and test `0.3885/0.3363`. Feathering trims FP/prediction
pressure (`-16/-19` val FP/preds and `-13/-19` test FP/preds versus old bridge)
but also makes recall slightly lower. Keep it for eval/generator iteration only;
do not train from it until the synthetic scenes are visibly human-countable and
validated against real fan/hand captures.
Postprocess does not solve this feathered visible-evidence stress. Applying the
current browser-stack KHR echo floors
(`KHR_1000/KHR_2000/KHR_20000/KHR_50000 >= 0.15`) improves precision but spends
recall: raw champion val/test `0.4155/0.3658` recall and `0.3555/0.3375`
precision becomes `0.3986/0.3289` recall and `0.4436/0.4153` precision, with
the biggest recall drops on `KHR_1000`, `KHR_2000`, and `KHR_50000`.
Class-agnostic NMS is no rescue either (`0.3986/0.3221` recall,
`0.4199/0.3664` precision). This keeps KHR floors/agnostic NMS as product
calibration knobs for background/count risk, not evidence that the detector has
learned partial-visible notes.
Legacy checkpoint backfill on the same feathered bridge is also negative. New
artifact root:
`runs/cashsnap/legacy_visible_evidence_feather_bridge_leaderboard_v1`. The only
large recall lift is the already-killed countable-center branch: candidate val
`0.4662/0.2788` and test `0.4094/0.2536`, while duplicate control ties val and
beats test recall (`0.4161/0.2495`) with even more FP, so the signal is generic
overproposal/continuation. Mined partial-stress, partial-stress+FP-neg,
realoverlap review39, and bbox-occlusion cap16 are split-noisy and add FP; their
candidate test recall gains over champion are at most `+0.0201`, except
countable-center, and paired duplicate controls often match or beat them. The
best small WebGL-filtered feather row remains the three-row no-`USD_50`/no-
`KHR_10000` subset (`0.4257/0.3792` val/test recall, `+9/+13` FP versus
champion), but it already loses to duplicate exposure on full real and raises
active-train FP. Verdict: no existing trained checkpoint proves held-out
visible-evidence learning on this bridge; the next detector attempt needs new
reviewed real/mask-based evidence, duplicate/count-aware pressure, or a stronger
gate/objective rather than another continuation of these partial-positive doses.
Browser-stack gating on the feathered bridge confirms the same bottleneck.
`runs/cashsnap/feather_bridge_browser_gate_eval_v1` uses the current KHR floors
plus the repaired KHR_100-aware reject gate at `reject>=0.72`; the harness now
supports detector class confidence floors in `scripts/probe_yolo_proposal_gate.py`.
Before gate rejection, class-floored detector val/test is `0.3986/0.4436` and
`0.3289/0.4170` recall/precision, with only `1/80` exact-value images on each
split. The gate alone barely changes val and hurts test slightly: it rejects one
val FP, and on test rejects three proposals including two true-positive-like
partial boxes (`0.3289 -> 0.3255` recall). Browser-style final class-agnostic
NMS is the main product tradeoff: gate+final-NMS becomes val/test
`0.3784/0.6188` and `0.2987/0.5669`, exact-value `4/80` and `3/80`; final-NMS
without the gate is nearly the same and slightly better on test (`0.3020`
recall, `4/80` exact-value). Read: the current stack can prune duplicate/echo
boxes, but it is doing so by spending partial-visible recall, not by solving
overlap counting. Do not tune more thresholds on this bridge as detector
progress; use it to demand better learned localization/count behavior. A quick
high-IoU final-NMS sweep is also negative: `0.85/0.90/0.95` leaves the existing
hard-slice champion+gate count/value result unchanged (`68/79` exact-value at
`reject>=0.72`) and on feather only recovers at most one test TP while adding FP
or reducing exact-value at `0.95`. Keep browser final NMS at the documented
`0.70` operating point until a broader product sweep gives a real reason to
move it. Containment-style suppression is killed too:
`runs/cashsnap/postprocess_duplicate_suppression_iom_probe_v1/summary.json`
shows intersection-over-min-area thresholds `0.50-0.95` are identical to IoU NMS
on the hard slice and usually lose more recall on the feather/countable-feather
bridges.
The stricter countable feather bridge is now registered and generated:
`data/processed/cashsnap_real_bboxpaste_overlap_countable_feather_eval_v1`
(`120/80/80`, `579` labels, class-balanced, blurred-source backgrounds,
`cutout_feather_frac=0.035`, `2-4` notes, visibility ratio `>=0.35`, visible
area `>=0.025`). Metadata says it is less ambiguous than the earlier feather
root: val/test median label count is `2`, min kept visibility ratio is
`0.3509/0.3552`, and min visible area is `0.0343/0.0252`. Champion raw
val/test `conf=0.05` recall/precision is `0.5687/0.4483` and `0.5476/0.4381`;
KHR floors trade recall for precision (`0.5375/0.5089`,
`0.5417/0.5084`). Browser-style gate+final-NMS on this cleaner bridge raises
precision but still spends recall: val/test `0.5063/0.7168` and
`0.4940/0.6917`, exact-value `26/80` and `22/80`; gate alone is tiny
(`-1` val TP, no test recall change), so final NMS remains the main product
tradeoff. Selected legacy backfill does not change the detector conclusion.
Countable-center has the highest recall (`0.6250/0.6548`) but adds `+94/+97`
FP versus champion and its duplicate control ties test recall, so it is still
overproposal/continuation. Bbox-occlusion cap16, mined partial-stress seed1,
and filtered WebGL-3 produce smaller, split-mixed recall gains (`<= +0.0417`
test except countable-center) with FP growth and controls close by. Use this
root as a cleaner overlap/counting scorecard; do not promote a checkpoint from
it without beating duplicate exposure and FP/count-value guards.

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
rows receive explicit accepted review decisions. The reviewed copy is now
`runs/cashsnap/real_overlap_focus_review_packet_v1/focus_review_packet_v1_reviewed.csv`
and should be the authoritative input for this packet; do not let flat
front/back catalog pairs stand in for real overlap/fan anchors.
`scripts/build_real_overlap_review_html.py` writes the local reviewer
`runs/cashsnap/real_overlap_focus_review_packet_v1/focus_review_packet_v1_review.html`;
use it only if the packet needs a new review pass, then pass the reviewed CSV to
`scripts/materialize_real_overlap_review.py`.

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

Real capture bridge status: `scripts/check_capture_requirements.py` currently
reports `0` inventory rows and writes
`runs/cashsnap/real_capture_requirements_latest.json` plus
`runs/cashsnap/real_capture_shot_list_latest.md`. The P1 capture gaps align with
the model bottleneck: `hand_fan`, `khr_5000_face_number_overlap`,
`thin_slice_khr_5000`, `mixed_usd_khr_rare_common`, hard `KHR_50000` partials,
and no-note/non-banknote hard negatives. `scripts/init_capture_inbox.py
--write-guides` created ignored drop folders under
`data/inbox/real_partial_photos/`; use those real phone captures, then register
and review them, before any new overlap/fan training dose.
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

The first reviewed materialization is now concrete and registered:
`runs/cashsnap/real_overlap_focus_materialized_reviewed_v1` has `39` train-anchor
images (`105` boxes; mostly `cambodia_currency_project` KHR overlap/fan rows),
`4` trusted held-out eval representatives, and `24` accepted exclude/flat-policy
images. On the trusted eval pocket, champion and reviewed-anchor candidate/control
all tie at recall/precision `0.5000/0.5000`; the unrelated seed1 partial-stress
candidate reaches `0.7500/0.6000`, but the pocket is only four images and the
strict scorecard still flags prediction growth rather than an FP-free promotion.
On reviewed train anchors, champion is `0.6286/0.7416`, seed1 partial-stress is
`0.6381/0.7204`, and the reviewed-anchor dose/control are `0.6381/0.7283` versus
`0.6476/0.7312`. Configs
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_overlapanchors_v1.yaml`
and
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_overlapanchors_exposure_control_v1.yaml`
are therefore killed for this freeze-22/lr5e-6/80-step schedule: the anchor dose
does not improve held-out eval, loses to duplicate exposure on its own train-anchor
view, and is too source/KHR-heavy to justify clean guard time. Next overlap work
needs either a larger/more diverse reviewed eval pocket first, or class/source
protected staging rather than dumping all reviewed anchors at once.

Visible-evidence reset packet v1 makes the review policy sharper but is killed
as a detector dose. `scripts/build_real_visible_evidence_review_packet.py` builds
`runs/cashsnap/real_visible_evidence_review_packet_v1` from
`real_overlap_review_queue_v1/review_clusters.csv`: `85` balanced review rows,
per-row full-size label previews, and a local HTML reviewer. The manual
obvious-safe review file
`visible_evidence_review_packet_v1_manual_obvioussafe_reviewed.csv` accepted only
`10` visually obvious train-anchor clusters (`20` materialized train images) and
`4` trusted eval rows, while explicitly excluding `25` flat/source-policy,
duplicate-folded-note, hidden-sliver, or label-risk rows. Full-size review caught
bad examples that the sheet/metrics would have let through: `VE-002`/`VE-040`
flat front/back layouts, `VE-046`/`VE-048`/`VE-049` duplicate boxes on one folded
note, `VE-051`/`VE-058`/`VE-059` hidden/sliver labels, and `VE-052` likely
wrong-denomination labeling. This reinforces the rule: do not auto-scale from
overlap tags; train anchors must be human-countable from visible evidence, with
no duplicate-label or label-mismatch ambiguity.

The materialized obvious-safe bridge
`runs/cashsnap/real_visible_evidence_materialized_obvioussafe_v1` is useful as a
diagnostic slice, not a promotion source. Champion baseline at `conf=0.05` is
weak on the accepted active eval pocket (`0.5000` recall / `0.2857` precision)
and train-anchor view (`0.6667` / `0.6441`), but the fixed 80-step freeze-22
micro-dose
`fixed_step_reviewed_visibleevidence_obvioussafe_candidate_headfreeze22_lr5e6_steps80_from_champion_i416_b2_w0_adamw_nowarmup_noamp_cachefalse_seed0`
does not improve it: active eval stays `0.5000` / `0.2857`, train anchors stay
`0.6667` recall and drop to `0.6129` precision. The exact class-exposure control
matches the candidate on both slices, and scorecards
`scorecard_candidate_vs_champion_active_conf005.json` /
`scorecard_candidate_vs_dupctrl_active_conf005.json` fail. Verdict: keep the
review/previews and explicit excludes, but do not promote or rerun this tiny
obvious-safe dose under the same schedule. Next visible-evidence work needs
more genuinely reviewed phone/fan/hand captures or a label-corrected larger
anchor set, not another small tag-selected row dose.

Failure anatomy on the accepted bridge says the next mechanism is not global
NMS/threshold tuning. Raising NMS to `0.95` leaves champion active eval/train
unchanged. Dropping `conf` to `0.01` with `nms=0.95` only moves active eval
`0.5000 -> 0.6250` recall and active train anchors `0.6667 -> 0.7193`, while
precision collapses (`0.1351` eval, `0.3254` train). Per-image errors show
`VE-001` is mainly denomination confusion (`KHR_50000` predicted as
`KHR_1000`/other KHR), while fanned train rows often emit too few true boxes or
many same-denom duplicates. Existing nearby checkpoints do not solve this:
partial-stress seed1 ties recall and worsens precision, auditclean sourcecap48
only nudges active-train recall to `0.6842` without eval gain, and the
available synthpretrain-to-balanced-real p24 checkpoint falls to `0.3509`
active-train recall. Read: champion remains the base; next work should target
denomination disambiguation plus duplicate/count proposal behavior on reviewed
fan/hand rows, not lower thresholds or another partial-edge dose.

Reset eval extension v1 adds a better real-visible diagnostic before more
training. Full-size review extended the trusted held-out pocket from `4` to `10`
images in
`runs/cashsnap/real_visible_evidence_materialized_eval_extension_v1/`, accepting
only hand/off-frame/folded or close-up partial notes with clear denomination
evidence (`VE-014`, `VE-015`, `VE-020`, `VE-029`, `VE-031`, `VE-037`) and routing
clean full-note tabletop edge-tag rows (`VE-005`-`VE-007`) to
`partial_policy_unclear`. Champion scores `0.7143/0.4348` recall/precision on
this pocket at `conf=0.05`; WebGL s30 candidate/control stay exactly neutral,
while no-`USD_50` WebGL s40 candidate keeps recall but drops precision to
`0.3846` with `+3` FP versus both champion and duplicate control
(`scorecard_no_usd50_candidate_vs_{champion,dupctrl}_conf005.json`). Read: the
small safe test confirms WebGL stack/fan positives are currently teaching
overproposal on real partial-visible rows. Class-agnostic NMS is a product clue
but not a detector rescue on this pocket: champion improves to `0.7143/0.5000`
(`10` TP / `10` FP) while the no-`USD_50` candidate only recovers to the
class-aware champion level (`0.7143/0.4348`, `10` TP / `13` FP) and still fails
versus champion under the same agnostic setting with `+3` FP. Grow reviewed real
eval pockets and proposal/duplicate controls before any new detector dose.

June 10 reset spot-QA keeps that conclusion, and also fixes the eval pocket.
Full-size previews confirm accepted rows such as `VE-014`, `VE-015`, `VE-020`,
`VE-031`, and `VE-037` are genuinely human-countable partial-visible notes, while
`VE-005`, `VE-008`, `VE-019`, and `VE-026` are mostly clean tabletop notes with
weak partial-teaching value. `VE-032` is flat/front-back or collage-like USD and
belongs outside trusted partial eval. Two old trusted rows were poisoning the
signal: `VE-001` has denomination-label risk (model `KHR_1000` predictions may be
right, not misses), and `VE-003` includes a tiny ambiguous `KHR_500` edge sliver.
They now route to `partial_policy_unclear` in
`visible_evidence_review_packet_v1_codex_eval_extension_reviewed.csv`.

The AP-hot duplicate-control continuation from the promoted checkpoint does not
improve this pocket. On v1 it kept recall at `10/14` and added one FP versus
champion in class-aware and agnostic-NMS views, while the current KHR-floor view
only tied champion (`10` TP / `10` FP). Scorecards:
`runs/cashsnap/currentbest_dupctrl_steps80_guard_v1/scorecard_visible_evidence_extension_champion_vs_dupctrl_classaware_conf005.json`,
`runs/cashsnap/currentbest_dupctrl_steps80_guard_v1/scorecard_visible_evidence_extension_champion_vs_dupctrl_agnosticnms_conf005.json`,
and
`runs/cashsnap/currentbest_dupctrl_steps80_guard_v1/scorecard_visible_evidence_extension_champion_vs_dupctrl_khrfloor_conf005.json`.
The cleaned v3 pocket
`runs/cashsnap/real_visible_evidence_materialized_eval_extension_v3/` has `12`
trusted images / `13` boxes; champion already gets `13/13` TP at `conf=0.05`.
KHR floors cut champion FP from `7` to `5`, and duplicate-control ties only under
that floor view. V3 scorecards:
`runs/cashsnap/real_visible_evidence_eval_extension_v3/scorecard_champion_vs_currentbest_dupctrl_steps80_conf005.json`
and
`runs/cashsnap/real_visible_evidence_eval_extension_v3/scorecard_champion_vs_currentbest_dupctrl_steps80_khrfloor_conf005.json`.
Production-pilot evidence still needs more reviewed real fan/hand/partial rows
that expose recall/counting failures, plus duplicate-aware proposal control; do
not run another AP-only continuation as a substitute.

KHR_50000 disambiguation reweight v1 is killed as a visible-evidence repair.
The focused packet
`runs/cashsnap/real_khr50000_disambiguation_review_packet_v1` reviewed `6`
clean single-label real/table/hand `KHR_50000` rows, but all six were already
present in the p24 synth+real base train list, so the only valid probe was
deliberate duplicate reweighting. Candidate/config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_khr50000_disambig6_reweight_v1.yaml`
and matched duplicate-control config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_khr50000_disambig6_reweight_dupctrl_v1.yaml`
were trained from the champion with the standard freeze-22/lr5e-6/80-step
schedule. Candidate full val mAP50-95 was `0.83379`; control was `0.83603`.
On the accepted visible-evidence eval, champion/candidate/control recall stays
`0.5000`, while precision moves `0.2857 -> 0.2667 -> 0.2667`. On accepted
train anchors, recall stays `0.6667`, while precision moves
`0.6441 -> 0.6333 -> 0.6333`. On the six KHR_50000 rows, candidate/control
match at `1.0000` recall / `0.7500` precision versus champion `1.0000` /
`0.6667`, removing the wrong `KHR_20000` FP on one row but leaving duplicate
`KHR_50000` proposals. Scorecards
`real_khr50000_disambiguation_materialized_v1/scorecard_candidate_vs_champion_conf005.json`
and `scorecard_candidate_vs_dupctrl_conf005.json` fail. Verdict: the tiny
KHR_50000 single-note reweight nudges generic exposure, not real-context
visible-evidence behavior; next work needs reviewed duplicate/count negatives,
paired same-image label correction, or postprocess/calibration, not more copies
of already-solved single-note positives.

Combined reviewed-real + WebGL stack/fan synth+real cap6 is killed as a
promotion, but it is the right diagnostic shape. Config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_visibleevidence_plus_webgl_stackfan_cap6_v1.yaml`
adds `17` visually QA'd rows (`12` WebGL stack/fan, `5` reviewed real
visible-evidence anchors) to the p24 synth+real champion; exact class-exposure
control is
`cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_visibleevidence_plus_webgl_stackfan_cap6_dupctrl_v1.yaml`.
The freeze-22/lr5e-6/80-step candidate beats duplicate control on full real test
mAP50-95 (`0.852370 -> 0.854260`, worst class `KHR_50000 -0.0164`) but fails
against the champion's protected-class guard despite a tiny aggregate bump
(`0.852767 -> 0.854260`, `KHR_50000 -0.0511`). On the accepted active eval,
champion/control/candidate all stay `0.5000` recall / `0.2857` precision. On
reviewed train anchors, candidate keeps recall `0.6667` while reducing FP versus
control/champion (`20` FP vs `24`/`21`). On the feathered bbox-paste bridge,
candidate versus control is split-noisy and not safe: val
`0.4223/0.3531 -> 0.4189/0.3473` with `+4` FP, test
`0.3658/0.3313 -> 0.3725/0.3313` with `+4` FP; versus champion it adds `+10`
FP on both bridge splits. Scorecards
`reviewed_real_plus_webgl_stackfan_cap6_eval_v1/scorecard_candidate_vs_dupctrl_conf005.json`
and `scorecard_candidate_vs_champion_conf005.json` fail. Read: synth+real is the
right direction, but this tiny mix mostly gives duplicate-proposal cleanup on
seen reviewed anchors, not held-out visible-evidence recall. Next synth+real
move should protect `KHR_50000` and train/eval on larger reviewed real fan/hand
captures or cleaner mask-based synthetic scenes, with explicit duplicate/count
loss or postprocess pressure.

The simple `KHR_50000`-protected follow-up is also killed. Config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_visibleevidence_plus_webgl_stackfan_khr50000protect_v1.yaml`
reuses the reviewed-real + WebGL add-on but blocks the five selected variants
containing `KHR_50000`, leaving `12` QA'd add-ons and zero new `KHR_50000`
exposure; exact duplicate control
`cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_visibleevidence_plus_webgl_stackfan_khr50000protect_dupctrl_v1.yaml`
duplicates `51` base rows. This protects the champion per-class guard but loses
the detector race: full real test mAP50-95 is champion/control/candidate
`0.852767/0.852880/0.850852`, and candidate loses to duplicate control by
`-0.002028`. Purpose slices are mixed rather than promotable: active eval stays
`0.5000` recall but precision drops control->candidate `0.2857 -> 0.2667`;
active train improves `0.6491/0.5968 -> 0.6667/0.6333`; feathered bbox-paste
val ties recall while reducing FP (`235 -> 223`), but test loses recall
`0.3725 -> 0.3658`. Scorecards
`reviewed_real_plus_webgl_stackfan_khr50000protect_eval_v1/scorecard_candidate_vs_dupctrl_conf005.json`
and `scorecard_candidate_vs_champion_conf005.json` fail. Read: protecting
`KHR_50000` by deletion avoids the obvious class cliff but removes too much
useful overlap signal and still does not produce held-out visible-evidence
recall. Next synth+real work needs better selected/masked scenes or a staged
objective, not another coarse class-blocked subset.

The gentler 40-step repeat of the original cap6 mix is killed as promotion but
kept as a clue. Same 17 add-ons and exact duplicate-control configs, trained
from the champion with freeze-22/lr5e-6/40 steps, produce full real
champion/control/candidate mAP50-95 `0.852767/0.853220/0.853184`; candidate
passes the champion aggregate/per-class guard but loses to duplicate control by
`-0.000036`. Active held-out stays tied at `0.5000/0.2857`; active train buys
recall at a precision cost versus control (`0.6667/0.6333 -> 0.6842/0.6000`);
feathered bbox-paste val improves (`0.4155/0.3388 -> 0.4223/0.3521`) but test
ties recall and loses precision (`0.3792/0.3314 -> 0.3792/0.3219`). Scorecards
`reviewed_real_plus_webgl_stackfan_cap6_s40_eval_v1/scorecard_candidate_vs_dupctrl_conf005.json`
and `scorecard_candidate_vs_champion_conf005.json` fail. Read: the softer dose
avoids the `KHR_50000` cliff, but the measurable aggregate lift is duplicate
cleanup, not trustworthy new visible-evidence learning.

Split-cap6 says the WebGL half carries the only useful signal, while the
reviewed-real-anchor half is harmful at this dose. WebGL-only config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_webgl_stackfan12_v1.yaml`
adds the 12 rendered stack/fan rows and beats exact duplicate control on full
real test (`0.851379 -> 0.852815`, worst class `USD_1 -0.0087`) and barely beats
the champion (`0.852767 -> 0.852815`, worst `KHR_50000 -0.0171`). It is still
not promotable: active eval passes and improves over its control (`0.5000`
recall, precision `0.2667 -> 0.2857`), but active-train recall ties while FP
rise (`+4` FP), feather-val loses recall (`0.4155 -> 0.4122`), and feather-test
loses recall versus control despite fewer FP (`0.3826 -> 0.3758`). Scorecards
`webgl_stackfan12_s40_eval_v1/scorecard_candidate_vs_dupctrl_conf005.json` and
`scorecard_candidate_vs_champion_conf005.json` fail. The 5 reviewed-real-anchor
split
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_realanchors5_v1.yaml`
loses badly to its duplicate control (`0.854239 -> 0.851411`) and to the
champion (`0.852767 -> 0.851411`), while its duplicate control is the one that
improves full real (`0.854239`). Read: use WebGL stack/fan selection/masking as
the next synthetic lever, and stop treating the reviewed real-anchor microdose
as positive data.
Schedule probe: WebGL-only 20 steps is completely neutral on full real
candidate/control/champion (`0.852767/0.852767/0.852767`), so the 40-step signal
has not been preserved at lower dose. WebGL-only 30 steps is also killed as a
candidate-data clue after rerunning the held-out eval with the WinMemoryCleaner
headroom wrapper: duplicate control and candidate tie exactly on full real
(`0.853149/0.853149`, both `+0.000382` over champion; worst per-class drop
`USD_1 -0.000686`) and tie exactly on all four conf-0.05 transfer slices
(`webgl_stackfan12_s30_eval_v1/scorecard_candidate_vs_dupctrl_conf005.json`).
Champion scorecards only carry the pre-existing active-train split-metadata
blocker, with zero metric deltas on active eval/train and feather val/test. Read:
the s30 lift is continuation/duplicate-exposure cleanup, not trustworthy WebGL
stack/fan learning. The no-`USD_50` seven-row WebGL subset is now killed too:
full real duplicate-control/candidate mAP50-95 is `0.852591/0.851937`
(`-0.000654`, worst `USD_1 -0.012318`) in
`fixed_step_webgl_stackfan_no_usd50_7_dupctrl_s40_vs_webgl_stackfan_no_usd50_7_candidate_s40_steps40_seed0/summary.json`.
Full-size visual QA of the seven source rows found the same unsafe teaching
shape: CGI stack/fan scenes with artificial fingers, many overlapping full-note
boxes, and some tiny/sliver visible labels. On the 10-image real-visible eval
extension, the candidate ties recall but adds `+3` FP versus both champion and
duplicate control. Do not train more WebGL stack/fan positives from this label
policy. The no-`USD_50` config remains only as a killed diagnostic artifact:
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_webgl_stackfan_no_usd50_7_v1.yaml`
with exact control
`cashsnap_balanced_real_p24_plus_strictbest_synth_p24_webgl_stackfan_no_usd50_7_dupctrl_v1.yaml`.
A stricter no-`USD_50`/no-`KHR_10000` three-row subset
(`variant_1120`, `variant_1234`, `variant_1253`) is also killed at 40 steps.
It loses to duplicate exposure on full real (`0.853870 -> 0.853149`,
`-0.000721`) while both candidate and control pass the champion aggregate/per-
class guard (`0.852767 -> 0.853149/0.853870`). Stress slices show a real but
unsafe clue: candidate beats control on active eval precision (`0.2667 ->
0.2857`) and feather val/test recall (`0.4189 -> 0.4257`, `0.3725 -> 0.3792`)
with two fewer FP on each feather split, but active-train FP rises (`+3`) and
scorecards still fail versus both control and champion. Visual filtering helped
remove the worst WebGL classes, but the remaining tiny positive-only dose still
does not beat duplicate exposure or solve count/proposal risk.

Visible-evidence class-threshold probe is only a bounded product knob. On the
accepted obvious-safe bridge, champion plus `--class-min-conf KHR_2000=0.50`
passes `scorecard_champion_classmin_khr2000_50_conf005.json`: active eval recall
stays `0.5000` while precision improves `0.2857 -> 0.3333` (`-2` FP), and
active train anchors stay `0.6667` / `0.6441`. Broader KHR floors are unsafe:
`KHR_1000=0.20` does not improve eval and drops train recall to `0.6316`;
`KHR_10000=0.20` drops train recall to `0.6491`; `KHR_5000=0.20` drops train
recall to `0.6491`; the combined noisy-KHR floor at `0.20` reaches eval
precision `0.4444` but fails train recall (`0.5965`). Keep `KHR_2000=0.50` as
a tiny calibration clue only; it does not fix VE-001 denomination confusion or
duplicate/count proposals.
Champion agnostic NMS is killed as a broad default despite tiny visible-evidence
precision clues. Full-real test at `conf=0.05` improves precision
`0.5536 -> 0.6038` and reduces total FP `624 -> 498`, but drops recall
`0.9474 -> 0.9290` and leaves empty-label FP images unchanged (`169/748`);
strict semantic+leakage-clean repeats the pattern (`0.9639 -> 0.9541`, empty
FP images `19/471` unchanged). The scorecard
`runs/cashsnap/agnostic_nms_champion_guard_v1/scorecard_champion_agnosticnms_vs_classaware_conf005.json`
fails on recall/per-class drops, especially `USD_5` and strict-clean `USD_10`.
FN examples show higher-confidence wrong-denomination boxes winning suppression
(`USD_50` over `USD_5`, `USD_1`/`USD_50` or `USD_20` over `USD_10`), so the
repair is class calibration/reclassification or label/source cleanup, not naive
cross-class suppression.
Use agnostic NMS only as a duplicate-proposal diagnostic or local product knob,
not as the detector's default postprocess.
Champion KHR echo-floor v1 is the safer adjacent postprocess clue: keeping
class-aware NMS but applying `--class-min-conf KHR_1000=0.15`,
`KHR_2000=0.15`, `KHR_20000=0.15`, and `KHR_50000=0.15` preserves recall on
full-real test (`0.9474`) and strict semantic+leakage-clean (`0.9639`) while
reducing total FP by `73`/`16` and empty-label FP images by `26`/`4`. It also
preserves the 10-image visible-evidence extension recall (`0.7143`) while
cutting `3` FP. Scorecards:
`agnostic_nms_champion_guard_v1/scorecard_champion_khr_echo_floor_v1_vs_classaware_conf005.json`
and
`agnostic_nms_champion_guard_v1/scorecard_champion_khr_echo_floor_v1_visible_evidence_conf005.json`.
The knob is wired into `detector.class_min_conf` in the current browser-stack
config and demo fallback; `check_browser_stack_artifacts.py`, `smoke_browser_stack_onnx.py`,
JS syntax, and static-server artifact fetches pass.
Do not broaden the floor set blindly: adding `USD_50=0.20`, `USD_1=0.15`, and
`KHR_500=0.10` cut more FP but failed full-real recall (`0.9474 -> 0.9327`),
especially `USD_50` and `KHR_500`.
Current KHR-gate plus the older fragment-aware denomination reclassifier is
killed as the agnostic-NMS repair path. At `reject>=0.72` with the real+synth
p24 champion, post-reclassifier versus post-gate drops full-real
recall/precision/exact-value by `-0.0588`/`-0.0419`/`-28`; strict-clean by
`-0.0230`/`-0.0164`/`-2`; and source-excluded strict-clean by
`-0.0331`/`-0.0217`/`0`, while adding wrong-class FNs on every slice. Summary:
`runs/cashsnap/proposal_gate_real_synth_p24_khr100_gate_rej072_reclass_comparison_v1.json`.
Use crop reclassification only as a future architecture/calibration research
thread; do not bolt the current reclassifier onto the browser stack.

Reviewed duplicate-label correction v1 is killed under the same micro-dose
schedule. `scripts/materialize_reviewed_duplicate_label_correction_probe.py`
materializes a paired add-on from visually excluded duplicate-count rows
`VE-046`/`VE-048`/`VE-049`: `9` KHR_500 folded-note train clones with corrected
single-box labels, plus a same-image control retaining the original two duplicate
labels. The rows were not in the p24 synth+real base train list, so this was an
add-on label-correction-vs-original-label exposure test, not a replacement.
Candidate/control configs
`cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_dupcount_correctedlabels_v1.yaml`
and
`cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_dupcount_originallabels_control_v1.yaml`
train to identical quick full-val mAP50-95 (`0.83526`). On accepted
visible-evidence eval, both are `0.5000` recall / `0.2667` precision versus
champion `0.5000` / `0.2857`; on accepted train anchors, both are `0.6667` /
`0.6230` versus champion `0.6667` / `0.6441`; on the corrected duplicate-count
view, both are `1.0000` / `0.7500` versus champion `1.0000` / `0.8182`.
Scorecards
`real_visible_evidence_duplicate_label_correction_probe_v1/scorecard_candidate_vs_champion_conf005.json`
and `scorecard_candidate_vs_originallabels_control_conf005.json` fail. Verdict:
this tiny corrected duplicate-label add-on only adds exposure/FP pressure; fixing
duplicate/count behavior likely needs a larger reviewed set with true replacement
of bad labels, an explicit duplicate-suppression objective/postprocess, or fresh
phone captures where countable-note labels are clean from the start.

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
As of 2026-06-10, `queue.csv` is still unreviewed (`review_status` /
`review_decision` blank) and `data/inbox/real_partial_photos/` contains only
folder scaffolding plus `.capture_guide.txt`, no capture images. Do not train
from this queue or claim phone/overlap improvement until rows are visually
reviewed or new captures are registered.
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
The `35` `khmer_us_currency` mixed-class audit rows have been separately
reviewed and tested as trusted `USD_50` positives; the `KHR_100`,
`asian_currency`, and `usd_total` buckets still need explicit visual review
before any new calibration train mix.
Materialization guard:
`scripts/materialize_synth_real_calibration_review.py` converts reviewed
train-cluster decisions into explicit train-only lists, but refuses blank
decisions by default. It accepts only reviewed/accepted clusters with `usable_as`
in `trusted_positive`, `hard_negative`, `unknown_out_of_scope`, or `exclude`, and
keeps `unknown_out_of_scope` separate from hard negatives because schema scope is
a product decision. Dry check
`shared_error_eval_slice_highvalue_khr_v2/materialized_review_drycheck_v1/summary.json`
correctly materialized `0` rows from the current unreviewed packet.
Reviewed `USD_50` mixed-source subset
`shared_error_eval_slice_highvalue_khr_v2/synth_real_calibration_first_review_clusters_codex_reviewed_usd50_mixed_v1.csv`
accepts those `35` mixed-class audit rows as trusted `USD_50` positives after
visual QA; train-only materialization lives under
`shared_error_eval_slice_highvalue_khr_v2/materialized_codex_reviewed_usd50_mixed_v1/`.
The candidate config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_usd50_mixed_v1.yaml`
adds the `35` unique rows, and the matched duplicate-control config
`configs/webgl_ablation/cashsnap_balanced_real_p24_plus_strictbest_synth_p24_reviewed_usd50_mixed_dupctrl_v1.yaml`
duplicates `35` `USD_50` rows instead. Fixed-step head-only A/B from the
champion (`freeze=22`, `lr=5e-6`, `80` batches) failed promotion:
`runs/cashsnap/fixed_step_usd50mix_dupctrl_vs_usd50mix_candidate_steps80_seed0/summary.json`
reports full-test mAP `0.852734 -> 0.852490`, worst class `KHR_20000 -0.018869`,
and `USD_50` only `+0.001984`. Low-conf shared-error recall improves versus
duplicate exposure (`test 0.1500 -> 0.2500`, `val 0.2000 -> 0.2667`), but strict
scorecards fail: candidate-vs-duplicate-control
`scorecard_usd50mix_candidate_vs_dupctrl_conf005_v1.json` trips test prediction
increase and val background-FP-image increase, while candidate-vs-champion
`scorecard_usd50mix_candidate_vs_champion_conf005_v1.json` trips total-FP,
prediction, and val background-FP gates. Do not scale this positive-only
`USD_50` bridge unless it is paired with proposal/objectness control or broader
reviewed source policy.
Reviewed foreign/unknown hard-negative rows are also killed as a detector dose.
`shared_error_eval_slice_highvalue_khr_v2/materialized_codex_reviewed_foreignhardneg24_v1/`
materializes `80` train rows from `32` reviewed clusters (`72` `asian_currency`
hard negatives plus `8` `khmer_us_currency` unknown/out-of-scope rows). The
head-only `reviewed_foreignhardneg_koreanwon24_from_champion_light` probe keeps
shared-error recall flat but does not suppress low-confidence proposals:
candidate-vs-duplicate-control fails with val `+2` FP / `+2` predictions and
test `+1` prediction, while candidate-vs-champion has no recall gain and
adds `+4/+1` FP on val/test. Do not repeat tiny reviewed foreign-note hard
negative dosing as a YOLO-only repair; use it as supervision for a stronger
unknown/proposal gate or a larger source-balanced objective.
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
Codex visual QA accepted that first `KHR_100` policy block as current-schema
unknowns:
`shared_error_eval_slice_highvalue_khr_v2/synth_real_calibration_first_review_clusters_codex_reviewed_khr100_unknown50_v1.csv`
and materialized list directory
`shared_error_eval_slice_highvalue_khr_v2/materialized_codex_reviewed_khr100_unknown50_v1/`
contain `50` reviewed train rows, all `khmer_us_currency`, all
`unknown_out_of_scope`, with no skipped decisions/rows. This is not a YOLO config
and should feed an unknown/proposal gate or explicit schema decision, not a
background-negative detector dose.
Proposal-gate unknown supervision now has a CPU-built crop pack:
`data/proposal_gate/cashsnap_khr100_unknown50_fullfrag_reject_v1/` contains
`300` train-only reject crops from those `50` reviewed `KHR_100` images (`full`,
`center80`, and four side/top/bottom fragments). Mixed gate root
`data/proposal_gate/cashsnap_strictbest_banknote_background_edgepos_trueempty_center50_coin_khr100_unknown_v1/`
adds those crops to the current true-empty/center50/coin gate base, changing
train reject `2950 -> 3250` while keeping train target `5832`, val reject
`1421`, and val target `2537`. Treat this as proposal-gate unknown-currency
calibration only, not YOLO detector background training data.
The broader product-bridge KHR100 policy route is now materialized too:
`product_bridge_review_queue_v1/materialized_khr100_policy_v1/` contains `80`
current-schema `KHR_100` unknown/out-of-scope rows (`68` train, `12` test), using
the narrow policy gate `khmer_us_currency` + label-empty + `100-riel` filename +
KHR100 schema-routing queue membership. The train list fully contains the
earlier reviewed `50` and adds `18` new train images. Crop root
`data/proposal_gate/cashsnap_khr100_productbridge_unknown68_fullfrag_reject_v1/`
has `408` train-only reject crops, and mixed gate root
`data/proposal_gate/cashsnap_strictbest_banknote_background_edgepos_trueempty_center50_coin_khr100_productbridge_unknown_v1/`
changes train reject to `3358` while keeping train target `5832`, val reject
`1421`, and val target `2537`. Prefer this broader root over the earlier
50-image KHR100 gate mix for the next unknown-aware proposal-gate retrain, but
still do not use these official `KHR_100` notes as YOLO detector background
negatives.
Held-out KHR100 unknown gate evaluation root
`data/proposal_gate/cashsnap_khr100_productbridge_unknown12_test_fullfrag_reject_v1/`
contains `72` reject crops from the `12` product-bridge KHR100 test images.
`scripts/eval_imagefolder_classifier.py` now uses `allow_empty=True` when
available so reject-only/target-empty diagnostic splits can evaluate without
fabricating dummy target crops.
Torch-free ONNX gate eval script
`scripts/eval_imagefolder_onnx_classifier.py` shows the current true-empty/
center50/coin gate is weak on that held-out KHR100 unknown set:
`runs/fragment_classifier/eval_gate_trueempty_center50_coin_on_khr100_productbridge_unknown12_test_onnx_v1/summary.json`
has reject accuracy `28/72 = 0.3889`, with `44/72` official-but-currently-unknown
KHR100 crops misrouted as `target`, many at `>0.98` confidence. This validates
the broader KHR100 unknown gate root as a real training target. Base-val
preservation target from the same ONNX evaluator is
`eval_gate_trueempty_center50_coin_on_base_val_onnx_v1/summary.json`: accuracy
`0.9462` over `3958` crops, with reject `1327/1421` and target `2418/2537`.
`scripts/train_imagefolder_classifier.py` now has `--freeze-features` and
`--init-checkpoint` so the next low-memory gate repair can fine-tune the current
gate head from
`runs/fragment_classifier/mobilenet_v3_banknote_background_edgepos_trueempty_strictbest_current_v1_e4_pre_lr3e4_b64w2/best.pt`
rather than restarting from ImageNet. After registering the elevated quiet
WinMemoryCleaner task, the preferred run completed:
`runs/fragment_classifier/mobilenet_v3_banknote_background_edgepos_trueempty_center50_coin_khr100_productbridge_unknown_v1_e4_initprev_freezefeat_lr3e4_b32w0/`
fine-tuned the classifier head from the prior gate for `4` epochs, exported
`best.onnx`, and reached base val accuracy `0.94821` (reject `1356/1421`,
target `2397/2537`). The repaired gate scores `72/72` on held-out KHR100 unknown
crops:
`eval_gate_khr100_productbridge_unknown_v1_on_khr100_unknown12_test_onnx_v1/summary.json`.
Diagnostic stack
`configs/cashsnap_two_stage_real_synth_p24_khr100_unknown_gate_browser_stack.json`
now uses reject threshold `0.72`. KHR100 detector-proposal threshold sweep
`runs/cashsnap/proposal_gate_eval_real_synth_p24_khr100_productbridge_unknown12_threshold_sweep_v1/summary.json`
shows the repaired gate clears the `12` held-out KHR100 unknown images at
`0.72` (`0` kept FPs) while `0.80` leaves `2` images / `8` proposals. Full-real
and strict-clean scorecards with the same real+synth p24 detector are in
`runs/cashsnap/proposal_gate_real_synth_p24_khr100_gate_threshold_sweep_v1/`:
at `0.72`, full-real exact-value is `1261/1562`, background-FP images `37`,
recall `0.9388`, precision `0.6699`; strict-clean exact-value is `679/774`,
background-FP images `3`, recall `0.9574`, precision `0.6854`. This beats the
balanced-real `rej080` product clue on full-real exact value/background FPs and
is the strongest current browser-stack candidate. Per-image comparisons against
balanced-real `rej080` are net-positive but churny:
`balanced_rej080_vs_real_synth_khrgate_rej072_full_per_image_compare_v1.json`
has exact net `+29` (`128` wins, `99` losses), and
`balanced_rej080_vs_real_synth_khrgate_rej072_strict_clean_per_image_compare_v1.json`
has exact net `+11` (`38` wins, `27` losses). This is still not evidence that
the detector itself learned `KHR_100` or visible-evidence localization.
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
Full-size review pack
`khr100_schema_expansion_annotation_proposals_v1/fullsize_review_top24_v1/`
selects the top-confidence proposal per image for 24 images and writes
`review_queue.csv`, `overlay_sheet.jpg`, `crop_sheet.jpg`, full-size overlays,
and padded crops. `scripts/build_official21_proposal_review_overlays.py` leaves
`review_decision` blank by default so proposal hints cannot silently become
labels. Three boxes explicitly opened full-size and judged clean single-note
KHR_100 examples (`khr100_001`, `khr100_008`, `khr100_024`) were copied to
`review_queue_codex_accept3_v1.csv` with `review_decision=accepted_box` via
`scripts/apply_cashsnap_review_decisions.py`; do not infer that the other 21
top24 rows are accepted without full-size review.
Full queue review surface
`khr100_schema_expansion_annotation_proposals_v1/fullsize_review_all89_v1/`
extends the same full-size overlay/crop workflow to all `89` proposal images
with top-per-image selection and no skipped images. Its top hints are
`KHR_20000 x52`, `KHR_500 x15`, `KHR_2000 x10`, `KHR_50000 x9`,
`KHR_1000 x2`, `USD_50 x1`, all treated only as localization hints. Tail QA
opened rows `khr100_086`, `khr100_088`, and `khr100_089` full-size; they are
clean KHR_100 examples, with `088/089` useful hand-occluded visible-evidence
cases. Superset reviewed CSV
`fullsize_review_all89_v1/review_queue_codex_accept6_v1.csv` accepts only the
six opened-full-size rows (`001`, `008`, `024`, `086`, `088`, `089`); the
remaining `83` rows are still blank review candidates.
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
id `8`. Codex accept3 bridge
`khr100_schema_expansion_annotation_proposals_v1/official21_bridge_codex_accept3_v1/materialized/`
materializes 3 train-only proposal images with 3 official21 `KHR_100` labels
(class id `8`), previewed at
`official21_bridge_codex_accept3_v1/preview_train_000000.jpg`; it is registered
as diagnostic
`cashsnap_official21_khr100_codex_accept3_review_bridge_v1`. Superset
`official21_bridge_codex_accept6_v1/materialized/` materializes 6 train-only
proposal images with 6 official21 `KHR_100` labels, including hand-occluded
visible-evidence rows; `preview_train_000001_hand.jpg` verifies the rendered
official21 label. It is registered as diagnostic
`cashsnap_official21_khr100_codex_accept6_review_bridge_v1`, and
`scripts/check_data_lifecycle_registry.py` passes. This proves the review-to-
official21-label path, but it is far too small to train a replacement detector.
Do not train from a full bridge root until more proposal rows are reviewed, the
output root is registered/classified in the data lifecycle registry, and
`scripts/check_data_lifecycle_registry.py` passes.
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
Latest taxonomy guard
`scripts/check_currency_taxonomy_coverage.py` still reports blocked coverage:
official target `21`, raw-current `20`, raw-any `21`, cutout `13`, model `13`;
active model/schema still miss `USD_2`, `KHR_50`, `KHR_100`, `KHR_200`,
`KHR_15000`, `KHR_30000`, `KHR_100000`, and `KHR_200000`.
Roboflow official21 partial bridge
`data/processed/roboflow_khmer_us_currency_official21_partial_bridge_v1/`
is the broadest current train/eval source for a `KHR_100`-aware detector: it is
a deduped CC BY v10+v3 Roboflow bridge with `exclude_image` for unsupported
classes, full train/val/test splits, and official21 ids. It has KHR_100 boxes
train/val/test `171/24/19`, but still only covers 14 of the 21 official names
and misses `USD_2`, `KHR_50`, `KHR_200`, `KHR_15000`, `KHR_30000`,
`KHR_100000`, and `KHR_200000`.
Official21 bootstrap detector
`runs/cashsnap/official21_roboflow_partial_yolo26n_e1_i416_b2_w0_adamw_lr1e3_noamp_seed0/`
is a one-epoch `yolo26n.pt` bootstrap on that partial bridge (`imgsz=416`,
`batch=2`, `workers=0`, `AdamW`, `lr0=0.001`, no AMP). It is not a promotion
candidate and cannot be compared directly to the 13-class champion, but it
proves the schema can train a small KHR_100-aware head. Aggregate val/test
mAP50-95 is only `0.1415/0.1243`; KHR_100 val/test is precision
`0.1490/0.1206`, recall `0.5000/0.5263`, mAP50-95 `0.1981/0.1550`.
`bootstrap_summary.json` captures the metrics. Visual smoke on the accepted
hand-occluded bridge row
`runs/cashsnap/official21_e1_accept6_hand_predict_v1/...jpg` fires `KHR_100`
at low confidence (`0.09`). The accept6 transfer probe
`accept6_predict_summary.json` finds KHR_100 predictions on `5/6` reviewed
bridge rows at `conf=0.05`, but confidences are low, several rows have duplicate
KHR_100 boxes, and the close high-resolution `100kh-1` row is missed. Next
official21 training should use more epochs, duplicate/count controls,
class/source calibration, and a core13-compatible evaluation/mapping harness
before any browser or detector promotion claim. The current-core13 remapped
val/test bridge confirms that caution: `current_core13_transfer_summary.json`
shows only `0.0653/0.0704` val/test mAP50-95 and `0.0960/0.0959` recall, with
nonzero recall only for `KHR_500` and `KHR_10000`. Treat the Roboflow-only
bootstrap as a schema/KHR_100 sanity check, not a transferable detector.
Official21 joint-mix probe
`runs/cashsnap/official21_current_accept6_roboflow_mix_from_rfbootstrap_fixsplit_steps1600_i416_b2_w0_adamw_lr5e4_noamp_seed0/`
is also a negative result. It fine-tuned the Roboflow official21 bootstrap for
`1600` batches on
`configs/official21/cashsnap_official21_current_accept6_plus_roboflow_partial_v1.yaml`
(current core13 train/val/test remapped to official21 with accept6 `KHR_100`
train boxes, plus `2018` Roboflow official21 train positives). Current val/test
mAP50-95 stayed tiny (`0.0675/0.0650`), Roboflow val/test fell to
`0.0885/0.0700` from the bootstrap `0.1415/0.1243`, Roboflow `KHR_100` recall
went to `0.0/0.0`, and the accept6 probe found `KHR_100` predictions on `0/6`
reviewed rows at `conf=0.05`. `mix_probe_summary.json` has the class rows. Do
not extend this exact balance blindly; the next official21 attempt needs an
explicit replay/balance policy for `KHR_100` and current positives/empties, not
just appending Roboflow to the full current bridge.
Balanced official21 replay probe
`runs/cashsnap/official21_roboflow_currentcap180_empty360_from_yolo26n_e1_i416_b2_w0_adamw_lr1e3_noamp_seed0/`
uses the opposite balance: Roboflow official21 train as the base plus a capped
current-core13-accept6 replay (`2347` current rows, max `180` per class, `360`
empties) from
`configs/official21/cashsnap_official21_roboflow_plus_current_accept6_cap180_empty360_v1.yaml`.
It is better than the full-current joint adaptation but still not a candidate:
current val/test mAP50-95 `0.1014/0.1117`, Roboflow val/test `0.1578/0.1576`,
accepted KHR_100 probe `1/6`, Roboflow KHR_100 recall `0.0/0.0526`. This says
official21 can improve with balance, but missing-class recall needs explicit
KHR_100 replay/oversampling or staged learning; do not treat aggregate Roboflow
gains as solving visible-evidence KHR_100.
Official21 KHR_100 replay plus balanced recovery is the first promising
missing-class schedule, but still only a diagnostic branch. Repeating KHR_100
rows on
`configs/official21/cashsnap_official21_roboflow_currentcap180_empty360_khr100repeat3_current24_v1.yaml`
from the Roboflow bootstrap gets accepted KHR_100 probe `5/6` and Roboflow
KHR_100 recall val/test `0.7083/0.8421`, but current no-KHR100-unknown
mAP50-95 falls to `0.0879/0.0977`. A 1000-batch balanced recovery from that
checkpoint,
`runs/cashsnap/official21_khr100repeat3_current24_balanced_recover1000_from_repeat_i416_b2_w0_adamw_lr1e4_nowarmup_noamp_seed0/`,
keeps accepted KHR_100 at `6/6` through `conf=0.25`, improves Roboflow val/test
mAP50-95 to `0.2413/0.2438`, and improves current no-KHR100-unknown val/test to
`0.1227/0.1284` (`0.1224/0.1279` unfiltered). The caution is calibration:
current unmatched predictions at `conf=0.05` rise to val/test `1223/692`
(`KHR_100` `197/107`), though at `conf=0.15` they drop to `224/102` while the
accept6 KHR_100 probe stays `6/6`. Treat this as evidence for staged
missing-class learning plus threshold/gate calibration, not a browser/product
promotion: the operational champion is still the 13-class p24 synth+real
detector, and official21 needs broader reviewed labels plus current-domain
calibration before it can replace it.
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
  On corrected shared-error v2 with the current synth+real champion, the existing
  true-empty gate at `reject>=0.99` keeps strict recall fixed while cutting
  background-FP images val/test `71->37` and `33->17`; threshold sweep suggests
  `reject>=0.70` preserves recall there and cuts further to `29/12`, but partial
  visible guards still matter (`0.99` partial-edge val recall `0.9572->0.9559`).
  Coin/center50 diagnostics sharpen the rule: low thresholds can suppress
  non-banknote money only when capped to low detector-confidence proposals, and
  otherwise reject real partial notes. Current best zero-recall-loss coin policy
  is only a modest trim (`4` fewer combined coin FP images), so do not present the
  existing gate as a robust unknown-money solution.
  Treat the gate as product/proposal control, not proof the detector learned
  partial-visible evidence.
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
- **Real-bbox paste overlap is useful as eval scaffolding, not a win yet.**
  The first rectangular bbox-paste root gives a larger partial/overlap/counting
  stress bridge and is registered as diagnostic. Its visual seams are obvious,
  which is acceptable for eval but not training by default. The unbalanced root
  is source/USD-skewed and does not show a clean mined-real candidate win. The
  balanced follow-up is more useful, but seed1 shows the candidate/control edge
  is not stable: one class-aware split beats duplicate control with fewer FPs,
  another ties aggregate recall and fails per-class. Class-agnostic NMS suppresses
  some echoes but does not separate the candidate from control or fix real
  partial-edge FP gates. Iterate the bridge for evaluation quality before
  treating pasted crops as trainable.
- **Reviewed real-overlap anchor dump is killed as tested.** The first accepted
  focus packet materialization is real reviewed data, not raw queue scratch, but
  adding all `39` source/KHR-heavy train-anchor images with the freeze-22/lr5e-6
  schedule ties the tiny held-out eval and loses to duplicate exposure on the
  train-anchor view. Do not repeat all-anchor dumping; grow/diversify the reviewed
  eval pocket or stage anchors with class/source protection.
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
- **Naive partial-visibility row dosing is killed as a promotion path.**
  Real-derived edge partials, bbox-blocker occlusions, and mined off-frame real
  rows expose genuine partial-visible stress signal, but the tried schedules
  either failed partial scorecards (border-partial micro, freeze-22 bbox
  cap16/80), improved partial recall while breaking the clean/source guard
  (bbox-blocker cap16/80, unfrozen mined-real cap8), or preserved clean guards
  while adding too many partial-edge FPs (reduced mined-real cap8). The reduced
  mined-real branch is the best clue because it beats champion/control recall on
  real partial-edge val/test, but it is not promotable until row review,
  fragment-label policy, hard negatives, or calibration makes the recall
  FP-safe. Keep the generators/miners and held-out stress sets; do not keep
  doing cap/step/filter shuffles without a materially different evidence policy.
- **Strict visual QA positive-only partials are still not enough.** Full-size QA
  of the mined partialstress cap8 rows found the edge-touch heuristic admits
  full-note crops and one-label multi-note USD strip scenes. A six-row reviewed
  KHR frame-cut subset is much cleaner, but the freeze-22/lr5e-6/80-step probe
  still loses full AP to exact duplicate exposure and has no recall gain on the
  reviewed rows, active visible-evidence views, or center50 val/test; scorecards
  fail on FP/prediction growth. This kills "just filter positives harder" as the
  next detector update unless paired with proposal/objectness or hard-negative
  supervision.
  The source-clean `vis50_70` reset reinforces the same rule. Full-size QA found
  the exact USD100 partial-test "miss" is not human-countable, and 50%-visible
  corner crops often show denomination-ambiguous texture, while center-strip
  50% rows are safer. A filtered `vis70 + center50` p24 mix excluded all
  `corner_*_vis0p5` rows and kept 24/class, with an exact duplicate-exposure
  control, but still failed: full-test AP lost to control (`0.854521` vs
  `0.856717`) and partial-test `conf=0.05` recall lost too (`0.8101` vs
  `0.8165`, precision `0.4523` vs `0.4448`). Do not train the full unfiltered
  `vis50_70` mix or repeat center/corner-positive shuffles without first fixing
  the human-countable label policy and adding explicit proposal/hard-negative
  pressure. Evidence:
  `runs/cashsnap/fixed_step_countsafe_vis70_center50_p24_dupctrl_vs_candidate_from_champion_steps318_seed0/summary.json`
  and `runs/cashsnap/countsafe_vis70_plus_center50_p24_v1/`.
  The same visual reset produced a better partial eval bridge:
  `configs/audit/cashsnap_real_countablepartial_sourceclean_vis70_plus_center50_eval_v1.yaml`
  keeps `vis0p7` plus `center_[xy]_vis0p5` and excludes all `corner_*_vis0p5`
  rows. On that filtered slice, the older p24 vis70 candidate shows real but
  still unsafe signal: test `conf=0.05` improves over champion/control
  (`0.8857/0.5569` recall/precision versus champion `0.8667/0.5652`, control
  `0.8667/0.5230`), while val improves over champion but trails duplicate
  control recall (`0.8926/0.5373` versus champion `0.8760/0.5638`, control
  `0.9008/0.5317`). A narrow `KHR_2000=0.10,KHR_50000=0.10` floor preserves
  the filtered test recall and improves precision (`0.8857/0.5813`), but it is
  not broadly safe: source-excluded total FP drops and TP rises, yet background
  FP images spread to `44/1036` versus raw candidate/control/champion
  `41/38/42`. Treat this as a denomination-arbitration/unknown-money blocker,
  not a product setting or detector promotion. Raising global detector
  confidence to `0.10` is also only a cleanup tradeoff: it keeps a small
  candidate recall edge over champion on the filtered slice, but absolute
  recall falls hard and candidate still loses filtered-val recall to duplicate
  control (`0.8182` vs `0.8264`). Lowering class-aware YOLO NMS is not a fix
  either: `nms_iou=0.45` and `0.25` are identical to `0.70` on the filtered
  partial val/test results, so the remaining proposal damage is cross-class
  arbitration/evidence policy rather than ordinary same-class NMS. Evidence:
  `runs/cashsnap/countsafe_vis70_p24_v1/filtered_eval_vis70_center50/` and
  `light_eval_source_excluded_p24_candidate_khr2000_50000_floor010_conf005.json`.
  Source-FP visual review reinforces the blocker: the raw p24 vis70
  candidate-only source queue has `32` boxes versus duplicate control, mostly
  `cashcountingxl`, and sampled full-size rows split into duplicate same-class
  boxes, wrong-denomination overlaps, unknown/foreign/non-banknote money, and
  one likely missing second-instance label. Use
  `runs/cashsnap/countsafe_vis70_p24_v1/source_fp_review_candidate_vs_dupctrl_v1/`
  as a review/eval-policy artifact; it argues for proposal arbitration and
  unknown-money/multi-instance policy before another positive-only partial dose.
- **Small FP-mined train-empty negatives do not repair mined partial positives.**
  Adding `32` champion train-empty FP negatives to mined-real partialstress cap8
  fails partial-edge scorecards versus champion, loses val recall to a
  duplicate-empty control, and only gives a mixed empty-background trade on
  val/test. This keeps hard negatives in the recipe, but they need stronger
  source/schema policy or an explicit objectness/calibration objective rather
  than a tiny appended empty-row dose.
- **Reviewed real-overlap review39 anchors need proposal control before scaling.**
  The head-only review39 anchor dose passes clean AP/per-class guards and shows a
  tiny true real-anchor recall signal versus duplicate exposure, but held-out
  overlap diagnostics fail scorecards because FP/pred counts expand. The next
  attempt should pair reviewed anchors with hard negatives, tighter sampling, or
  calibration/objectness constraints instead of treating more reviewed positives
  as the fix. The lower-lr/shorter head-only repeat killed the easy scheduler
  repair: it removed the held-out TP gain while retaining FP growth.
- **KHR_5000-only reviewed-overlap staging is killed as tested.** Restricting the
  reviewed anchor dose to `8` `KHR_5000` images/`24` labels did not improve
  `KHR_5000` anchor recall (`11/24` for champion, candidate, and duplicate
  control), added one `KHR_5000` FP versus control, and cost `KHR_1000` recall.
  The tiny trusted-eval gain is USD-only and fails FP/prediction scorecards, so
  this is not the missing visible-evidence mechanism.

### Untested Ideas

Do not preserve small probes here just because they are reasonable. A worthwhile
untested idea should plausibly change the transfer regime, expose why the current
harness is misleading, or materially improve the promoted clean foundation on a
real bottleneck under guardrails. Otherwise it is a tactic, not the research
frame.

- **Production Pilot v1: one detector, curated blend.** Build one YOLO26n
  detector for phone/browser testing, initialized from the p24 vis70 candidate
  unless a preflight shows its FP tendencies are too hard to regularize. Target
  roughly `2.4k-3.0k` training rows, not a micro-dose: about `35-40%` clean real
  replay, `20-25%` strictbest synthetic/base support, `20-25%`
  countable-partial/overlap positives, `10-15%` train-safe hard negatives
  (true empty, coins, foreign/unknown money, source-FP analogs), and `5-10%`
  high-risk class protectors for `KHR_50000/KHR_20000/USD_50/USD_100`. Use
  only human-countable partial positives: include p24 `vis0p7`, reviewed
  hand/edge/fan/cutoff rows, and visually safe center-strip rows; exclude
  unreviewed `corner_*_vis0p5`, 20% texture fragments, unsafe WebGL stack/fan
  labels, and val/test-mined FP images as train rows. Train long enough to see
  the blend, e.g. `5k-8k` image presentations at `imgsz=416`, `batch=2`, low LR,
  and clean replay; staged freeze/unfreeze is acceptable as long as the output
  is one final checkpoint. Promotion must beat or preserve champion on full
  real, strict clean, source-excluded clean, filtered countable-partial, source
  FP queue, and per-class guards; if it only improves partial recall by adding
  duplicate/wrong-denom boxes, kill it.
  Materialized first pilot artifact:
  `configs/webgl_ablation/cashsnap_production_pilot_v1.yaml`, built by
  `scripts/build_cashsnap_production_pilot_config.py`, has `2665` exposure rows
  and `986` unique images: clean real `35.68%`, strictbest synth `23.86%`,
  countable partial `23.79%`, train-safe hard negatives `9.91%`, and high-risk
  class protectors `6.75%`. It uses existing list-backed/train-split rows only,
  blocks unreviewed corner-50 partial rows, and keeps eval-mined hard negatives
  out unless they are train-split analogs. It is an untrained config, not a
  checkpoint; start from the p24 vis70 candidate and evaluate against the full
  promotion stack above.
- **Real+synth schedule lesson for Production Pilot v1.** Do not spend the next
  serious run on a broad bakeoff. Use the June 2026 blend-strategy PDF only to
  shape the one-detector pilot: staged calibration, low-conf/background/source
  guards, clean replay, and a final product-stack pass only after the detector is
  plausible. The p24 blend shows synthetic variation can beat real duplication,
  but unreviewed `asian_currency`, `khmer_us_currency`, and other
  empty-label/mixed-source rows are not clean negatives by default. The
  sourcecap48+strictbest-synth duplicate exposure-control probe is already killed
  as tested (`+0.001392` full mAP, `KHR_50000 -0.091826`), so the pilot blend
  must protect rare KHR classes by construction rather than just adding rows to
  the audit-clean sourcecap base.
- **High-value USD real-dup swap is diagnostic, not a promotion.** Replacing the
  `USD_50`/`USD_100` strictbest-synth extras with duplicated real exposure kept
  row/class parity but dropped full mAP versus the synth+real champion
  (`0.844567` vs `0.852767`). This suggests the high-value USD synth rows should
  not be blindly removed; the next model-side repair needs a subtler USD protect
  mechanism than swapping synth out for duplicate real.
- **Detector-label cleanup feeds the pilot.** Turn `review_queue_ranked_v1.csv`
  and the corrected shared-error v2 queue into trusted pilot ingredients:
  reviewed positives, explicit unknown/out-of-scope rows, and vetted hard
  negatives. Rerun detector scorecards on full, semantic, strict,
  source-excluded, per-class, and low-conf/background slices before promotion.
  Kill pilot training if it improves aggregate AP while weak classes, source
  robustness, or low-confidence background behavior degrade. Start with the
  corrected v2 shared-error queue and scorecard, not the superseded v1 queue: the
  live problem is `KHR_50000/KHR_20000` overproposal on
  `asian_currency`/`khmer_us_currency`
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
- Run long/big training, rendering, and broad eval jobs through the headroom
  guard (`scripts/run_with_headroom.py`, or `scripts/bench_train_with_headroom.py`
  for YOLO training) so CPU, RAM, and GPU memory stay below the laptop freeze
  thresholds.
- Optional RAM cleanup trigger: set `HEADROOM_MEMORY_CLEAN_EXE` plus args or
  `HEADROOM_MEMORY_CLEAN_TASK`; the guard runs it only on hard RAM pressure or
  below the separate `HEADROOM_MEMORY_CLEAN_MIN_FREE_RAM_GB` emergency floor
  (default 1.5 GB), with a default 10-minute cooldown and never for CUDA VRAM
  pressure.
- WinMemoryCleaner `3.0.8` portable is downloaded, SHA256-verified, and source-
  cloned under `.cache_runtime/`; `run_with_headroom.py` has direct
  `winmemorycleaner` and scheduled-task `winmemorycleaner-task` presets. Direct
  non-elevated cleanup fails cleanly with WinError 740 and no notification, but
  elevated `scripts/install_winmemorycleaner_task.ps1` now successfully
  registers `CashSnapWinMemoryCleaner`. `schtasks /Run /TN CashSnapWinMemoryCleaner`
  freed RAM from about `1.05 GB` to `4.44 GB`, and the headroom guard
  successfully triggered it mid-training. This archive predates the current
  cleaner posture: active runs now prefer
  `HEADROOM_MEMORY_CLEAN_PRESET=memreduct`, which maps to installed
  `memreductTask=-clean` for `C:\Program Files\Mem Reduct\memreduct.exe`;
  `winmemorycleaner-task` remains a fallback.
- While the laptop is being used interactively, keep probes GPU-targeted
  (`device=0`) but CPU/RAM-light: no parallel GPU jobs, `workers=0`,
  `cache=false`, and smaller eval/train batches unless explicitly running a
  promotion-parity pass.
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
