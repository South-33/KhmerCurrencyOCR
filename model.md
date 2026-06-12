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

Pilot initialization result: clean champion init and p24 vis70 init were
effectively tied once trained through the same pilot blend, so initialization is
not the main bottleneck right now. The conservative init is still the clean p24
synth+real champion above; the visible-evidence challenger init was the p24
vis70 candidate:
`runs/cashsnap/fixed_step_countsafe_vis70_p24_from_last_e50_i416_b2_w0_adamw_lr5e6_nowarmup_noamp_cachefalse_freeze22_steps318_seed0/weights/last.pt`.
It was only a 318-batch head tune over `927` unique rows (`323` original real,
`312` strictbest synthetic, `292` source-clean `vis0p7` partial crops), but it is
the best current compromise signal: full real `0.854178`, strict clean
`0.865306`, source-excluded clean `0.795336`, unfiltered partial test
`0.660102`, filtered countable-partial test recall/precision `0.8857/0.5569`.
It is not production-safe by itself because source/unknown-money and
wrong-denomination proposal issues remain.

Current 13-class pilot frontier:
Production Pilot v2 hard-negative guard remains the conservative fallback:
`runs/cashsnap/production_pilot_v2_hardneg_guard_from_cleanchampion_e3_i416_b2_w0_adamw_lr5e6_nowarmup_noamp_cachefalse_freeze22_seed0/weights/last.pt`.
It starts from the clean champion and uses
`configs/webgl_ablation/cashsnap_production_pilot_v2_hardneg_guard.yaml`, a
v1-style clean/strictbest/partial blend plus train-split exact-failure
hard-negative pressure. V9 is the current simple 13-class candidate-to-beat:
`runs/cashsnap/production_pilot_v9_v2schedule13_from_v2_e3_i416_b2_w0_adamw_lr2e5_nowarmup_noamp_cachefalse_freeze22_seed0/weights/last.pt`.
It continues v2 on the unchanged v2 blend with the v6-style `lr0=2e-5`,
`lrf=0.01` schedule. V9 improves full/source AP and held-out unknown/empty
safety versus v2 while staying browser-simple, but its partial-val precision
cost and weaker partial-test `conf=0.25` recall versus v6 keep it unpromoted.
The hard-slice browser-stack check now makes V9 the strongest current
product-count clue: on the 79-image count-risk slice with product-style
`conf=0.05`, KHR floors, gate `reject>=0.72`, and final NMS `0.70`, V9 reaches
`61/79` exact-value images and `3/22` background-FP images versus the old clean
champion's `59/79` and `5/22`. Broad browser-stack final-NMS checks also keep
V9 competitive: full-real exact-value/background-FP images are V9 `1437/1562`,
`36/748` versus champion `1432/1562`, `37/748`; strict-clean ties exact value
at `745/774` with one extra V9 background-FP image (`4/471` vs `3/471`).
V10 no-erasing keeps the same V9 path but sets `erasing=0.0`; it improves some
partial precision but loses partial recall and worsens unknown-money balance, so
keep it as a diagnostic result, not the frontier candidate.
V6 remains the strongest partial-recall challenger:
`runs/cashsnap/production_pilot_v6_realunknown14_pseudo81_from_v2expanded_e3_i416_b2_w0_adamw_lr2e5_nowarmup_noamp_cachefalse_freeze22_seed0/weights/last.pt`.
V6 improves full/source AP and true-foreign held-out money suppression, and it
raises filtered partial test recall, but it spends partial-val precision and the
new `UNKNOWN_FOREIGN_NOTE` class never fires on held-out unknown money.

Browser/gate posture: current detector+gate/browser stacks are diagnostic
product clues, not proof that the detector learned visible-evidence reasoning.
Proposal gates can trim background/unknown-money leakage, but they do not rescue
a detector that creates duplicate or wrong-denomination boxes. On the filtered
countable-partial bridge, the product gate also trims real target recall, so do
not use it as a partial-recall substitute.
V9 now has a diagnostic ONNX browser-stack config at
`configs/cashsnap_two_stage_v9_khr100_unknown_gate_browser_stack.json`: detector
ONNX is `9.22 MB`, gate ONNX is `5.81 MB`, artifact check passes at `15.03 MB`,
and `runs/cashsnap/browser_stack_onnx_smoke_v9_usdrisk15_gate_rej072_v1.json`
confirms CPU ONNX execution for the USD-risk15 policy. The smoke image is an
unknown-money `asian_currency` row and still produces a kept `KHR_5000` target
proposal, so deployability is proven but unknown-money rejection is not solved.

## Research Frame

### Current State

The live goal is still one production-pilot detector, and the scorecard now
needs to be blended like the product: clean/non-overlap positives, countable
partial/overlap positives, held-out unknown-money negatives, and ordinary
true-empty negatives. The current V9/V15/V16 current-list scorecard is
`runs/cashsnap/production_pilot_eval_suite_v1/scorecard_summary_V9_V15_V16_current.json`.
Do not compare newer source-excluded metrics to the older saved V9
`0.7938` artifact without checking the list version; the current source-excluded
list gives V9 full/strict/source `0.8629`/`0.8669`/`0.7394`. The reusable
v2/v6/v9-v13 historical scorecard artifact is
`runs/cashsnap/production_pilot_eval_suite_v1/scorecard_summary_V2_V6_V9_V10_V11_V12_V13_contrastive.json`;
the previous v2/v6/v9-v12 source-clean snapshot is
`runs/cashsnap/production_pilot_eval_suite_v1/scorecard_summary_V2_V6_V9_V10_V11_V12_sourceclean_seed1.json`;
the previous v2/v6/v9-v11 snapshot is
`runs/cashsnap/production_pilot_eval_suite_v1/scorecard_summary_V2_V6_V9_V10_V11_seed1.json`,
and the scorecard script is
`scripts/summarize_cashsnap_production_pilot_scorecard.py`.
Do not treat train-split hard-negative rows as held-out proof after adding them
to a blend.

Hard-slice browser count/value gate has been pulled back into pilot selection.
On `configs/audit/cashsnap_ve_v4_trainanchors_candidate_only_fp_conf015_slice_v1.yaml`
with the historical setting (`conf=0.15`, gate `reject>=0.72`, final NMS
`0.70`), exact-value/background-FP images are clean champion `68/79`, `4/22`;
V9 `67/79`, `3/22`; V15 `62/79`, `4/22`; V16 `68/79`, `4/22`. Under the
product-style setting (`conf=0.05`, detector IoU `0.70`, KHR floors `0.15`,
gate `reject>=0.72`, final NMS `0.70`), clean champion is `59/79`, `5/22`, V9
is `61/79`, `3/22`, and V16 is `59/79`, `6/22`. Evidence lives under
`runs/cashsnap/production_pilot_eval_suite_v1/challenge_slice_*gate_rej072*`.
Read this as a mechanism result: V9 survives the browser count-risk slice better
than V15/V16 despite weaker AP, so do not promote AP/source gains without this
gate.

Broad product-stack final-NMS summary is
`runs/cashsnap/production_pilot_eval_suite_v1/browser_gate_finalnms070_summary_champion_v9_v16_current.json`.
With product-style `conf=0.05`, KHR floors, gate `reject>=0.72`, and final NMS
`0.70`, full-real exact-value/background-FP images are champion `1432/1562`,
`37/748`; V9 `1437/1562`, `36/748`; V16 `1436/1562`, `39/748`. Strict-clean is
champion `745/774`, `3/471`; V9 `745/774`, `4/471`; V16 `746/774`, `4/471`.
Read this with the hard slice: V9 and V16 are broad-stack near-ties, but V9 is
the current product-stack tie-breaker because V16 leaks more on the hard
count-risk slice and full-real background. Do not overstate the V9 broad gain:
full-real is `22` exact wins versus `17` losses with churn across `billsbank`
and `usd_total`, while strict-clean is `4` wins and `4` losses with losses
concentrated in `khmer`/`khmer_us_currency`.

Partial product-stack summary is
`runs/cashsnap/production_pilot_eval_suite_v1/partial_product_gate_rej072_finalnms070_summary_champion_v9_v16_current.json`.
With the same product-style setting on the filtered countable-partial bridge,
the gate trims proposal recall on every detector: champion test/val
`0.8667 -> 0.8095` / `0.8347 -> 0.8099`, V9
`0.8952 -> 0.8381` / `0.8678 -> 0.8430`, and V16
`0.9143 -> 0.8571` / `0.8430 -> 0.8182`. After final NMS, exact-value images
are champion `74/105` test and `90/121` val, V9 `78/105` and `92/121`, and
V16 `78/105` and `92/121`, with `0` background-FP images because the slice has
no background rows. Read this as a mechanism warning: the product gate improves
some precision/count behavior but clips partial visible-evidence recall, and
V9/V16 remain tied on partial exact value under the stack.

V9 USD-risk15 calibration is the current deployable-policy clue, not a new
detector checkpoint:
`runs/cashsnap/production_pilot_eval_suite_v1/v9_usdrisk15_policy_summary_current.json`.
It keeps product `conf=0.05`, KHR floors `0.15`, gate `reject>=0.72`, and final
NMS `0.70`, then adds `USD_1`/`USD_50`/`USD_100` detector floors at `0.15`.
Detector-only unknown/true-empty FPs fall from KHR-only `102/237` and `44/441`
to `76/237` and `26/441`. The split explains the win: USD2 improves from
`32/41`, `55` FPs to `23/41`, `34` FPs, and foreign Asian currency improves
from `58/184`, `64` FPs to `41/184`, `46` FPs, while KHR100 is unchanged at
`12/12`, `21` FPs. The product stack is better than detector-only on missing/
foreign money: after gate+final NMS, USD2 is still `17/41` images-with-FP, but
foreign Asian falls to `3/184` and KHR100 falls to `0/12`. Hard-slice product
final NMS improves from V9 `61/79`, `3/22` to `64/79`, `3/22`; strict-clean
improves from `745/774`, `4/471` to `747/774`, `3/471`; partial product
exact-value stays `78/105` test and `92/121` val. The tradeoff is real:
full-real product exact-value falls from `1437/1562` to `1434/1562` while
background-FP images improve `36/748` to `31/748`, and partial val post-gate
recall drops versus KHR-only. Use it as the best current browser calibration
candidate, not as proof the detector learned unknown-money reasoning; USD2 is
now the bigger missing-denomination leak in the gated stack.

V9 vs V16 USD-risk20 calibration provides a fair comparison under the same product policy floors:
`runs/cashsnap/production_pilot_eval_suite_v1/usdrisk20_policy_summary_current.json`.
It raises USD floors to `0.20`. In the final NMS product stack:
- Full-real exact value / background FP is V16 `1440/1562`, `28/748` vs V9 `1435/1562`, `29/748` (V16 wins broad-stack counts).
- Strict clean is V16 `748/774`, `3/471` vs V9 `747/774`, `3/471` (V16 wins).
- Hard slice is V16 `65/79`, `4/22` vs V9 `65/79`, `3/22` (Tied on exact value, V9 wins by one background-FP image).
- Partial test / val exact matches at `78/105` / `91/121` for both models.
- Split unknown checks: foreign Asian FP is V16 `2/184` vs V9 `3/184` (V16 wins); KHR100 FP is V16 `1/12` vs V9 `0/12` (V9 wins); USD2 FP is V16 `15/41` (15 total FP) vs V9 `15/41` (16 total FP) (Tied on images, V16 wins on total FP count).
This calibration positions V16 as a highly competitive contender, beating V9 on broad/strict exact counts while V9 preserves a narrow advantage on hard-slice background-FP and KHR100 suppression.

The pilot is successful only if partial/overlap recall improves for the right
reason: recognizing countable visible evidence. It is a failure if the gain comes
from duplicate same-note boxes, wrong-denomination boxes, or target-class
predictions on unknown/foreign/non-banknote money.

Use the p24 synth+real clean champion as the fallback and guardrail. Use Pilot
v2 as the conservative production-pilot fallback and V9 as the current simple
candidate-to-beat. Use the p24 vis70 candidate as a clue, not an init priority.
Use the filtered countable-partial eval bridge
`configs/audit/cashsnap_real_countablepartial_sourceclean_vis70_plus_center50_eval_v1.yaml`
as a cleaner partial yardstick than the old unfiltered vis50/70 split.

Fair held-out unknown-money eval now exists:
`configs/audit/cashsnap_heldout_zero_label_money_guardrails_v1_*_eval.yaml`
with val/test rows from zero-label `asian_currency`, `usd_total_2Dollar`, and
current-schema-unknown `khmer_us_currency_100-riel` splits. The matching
train-only broad hard-negative sample is
`configs/generated_lists/audit/cashsnap_zero_label_money_train_hardneg_broad240_v1.txt`;
use it for training pressure, not promotion proof.

The old unfiltered partial eval contained policy-poison rows: exact USD100
"misses" that were not human-countable and `corner_*_vis0p5` fragments that were
often denomination-ambiguous. Future partial rows must be human-countable from
visible evidence, ignored/excluded if ambiguous, and never silently converted
into forced denomination labels.

Current hard blockers for promotion are still count safety and source policy:
duplicate boxes, wrong-denomination overlaps, unknown/foreign/non-banknote money
leaking into target classes, and possible multi-instance label gaps. The v2
checkpoint reduces but does not solve held-out unknown-money hallucination: at
`conf=0.25`, held-out unknown-money combined detections are `51/237` images
versus `72` for Pilot A; true-empty detections are `4/441` versus `10`.
The source FP review queue for the p24 vis70 candidate remains useful:
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
- **Production Pilot v2 is the current balanced leader.** Pilot A
  (clean-champion init) and Pilot B (p24 vis70 init) both reached full real
  `0.8549`, strict clean about `0.866`, source-excluded about `0.792`, and
  filtered partial test recall `0.9048`; init was not decisive. V2, from the
  clean champion with exact train-split hard-negative guard pressure, improved
  full real to `0.8573`, strict clean to `0.8688`, filtered partial test
  recall/precision to `0.9238/0.5575`, and reduced held-out unknown-money
  detections at `conf=0.25` from Pilot A `72` to `51` while reducing true-empty
  detections from `10` to `4`. The cost is source-excluded clean `0.7889`, a
  small drop from A/B, and unknown-money hallucination remains too high for
  launch without further work.
- **Broad unknown negatives need the right budget.** V3 used the broad 240-row
  train unknown-money hard-negative sample at a high total dose and is killed:
  it improved some negative counts but dropped full real to `0.8532`, strict
  clean to `0.8646`, source-excluded to `0.7865`, and partial test recall to
  `0.8952`. V4 kept roughly v2's hard-negative exposure budget with broader
  diversity and is also killed as the main candidate: partial recall rose to
  `0.9429`, but held-out unknown-money detections worsened versus v2 (`63` vs
  `51` at `conf=0.25`), true-empty worsened (`8` vs `4`), full real stayed
  `0.853`, and source-excluded fell to `0.785`.
- **Synthetic UNKNOWN class did not transfer.** V5 remapped paired synthetic
  foreign notes and broad WebGL unknown notes into class 13. A direct 14-class
  head reset cratered target mAP (`0.439` full real target-filtered). Head-graft
  recovery restored target rows but not launch behavior: grafted target mAP was
  `0.824`, fine-tuned graft was `0.848`, and held-out real unknown-money/empty
  FPs were v2-level or worse. Product-filtered probes showed
  `ignored_detections=0` for `UNKNOWN_FOREIGN_NOTE` even at low confidence, so
  synthetic unknown labels did not absorb real unknown-money hallucinations.
- **Real pseudo-UNKNOWN boxes are a suppression clue, not solved routing.** V6
  expanded the v2 head to 14 classes and added 71 train-split real unknown-money
  FP rows as `UNKNOWN_FOREIGN_NOTE`: target-filtered full/source improved
  (`0.8633`/`0.7934` vs v2 `0.8573`/`0.7889`) and held-out unknown-money at
  `conf=0.25` improved (`42/237`, `46` detections vs v2 `49/237`, `51`), but
  `ignored_detections=0`, partial-test precision fell (`0.5294`), and partial-val
  precision fell hard (`0.4615`). V7 repeated those pseudo rows 3x and is killed
  as a mechanism: held-out unknown-money worsened versus v6 (`47/237`, `53`
  detections at `conf=0.25`; `203` detections at `conf=0.05`) and
  `UNKNOWN_FOREIGN_NOTE` still never fired. Do not continue a simple UNKNOWN-dose
  ladder without a different objective or better real reviewed evidence.
- **The same 71 real unknown rows as empty negatives are also not enough.** V8
  kept a 13-class head, continued v2 with those FP-focused train rows as empty
  hard negatives, and slightly improved true-empty at `conf=0.25` (`3/441`), but
  held-out unknown-money stayed worse than v6 (`47/237`, `53` detections at
  `conf=0.25`; `195` detections at `conf=0.05`). Kill the simple "pseudo box vs
  empty row" ladder as the main unknown-money repair path.
- **V9 shows v6's schedule was a major part of the gain.** V9 keeps the 13-class
  v2 data/head, continues from v2 with the v6-style `lr0=2e-5/lrf=0.01` schedule,
  and on the original source-excluded list reached full/source
  `0.8629`/`0.7938` vs v2 `0.8573`/`0.7889` and v6 `0.8633`/`0.7934`. On the
  current V15/V16 source-excluded list, V9 is full/strict/source
  `0.8629`/`0.8669`/`0.7394`. Held-out unknown-money at `conf=0.25`
  improves to `39/237`, `46` detections, and true-empty to `3/441`, but
  filtered partial is mixed:
  test `0.9333/0.5297`, `0.8571/0.6870`, `0.7429/0.7358` at conf
  `0.05/0.15/0.25`; val `0.8843/0.4672`, `0.7934/0.6000`, `0.7686/0.6691`.
  FP-delta review versus v2 on filtered partial test shows the gain is not free:
  at conf `0.15`, recall `+0.0381` costs `+5` FPs; at conf `0.25`, recall
  `+0.0095` costs `+3` FPs, with sampled risks led by wrong-class overlap and
  duplicate same-class overlap. Treat V9 as the current simple candidate-to-beat
  pending seed repeat and count-safety repair/review, not final production.
- **Disabling random erasing is not the promotion fix.** V10 keeps the V9
  13-class v2 schedule/data/checkpoint path but sets `erasing=0.0`:
  `runs/cashsnap/production_pilot_v10_v2schedule13_noerase_from_v2_e3_i416_b2_w0_adamw_lr2e5_nowarmup_noamp_cachefalse_freeze22_seed0/weights/last.pt`.
  Full/strict-clean stay in the V9 band (`0.8626`/`0.8669` mAP50-95), and the
  earlier apparent source-excluded drop was a stale-list comparison against V9
  `0.7938`, not a current-list proof. Correct filtered partial eval is still a
  threshold trade, not a win:
  test `0.9333/0.5414`, `0.8381/0.7097`, `0.7238/0.7451`; val
  `0.8843/0.5071`, `0.8017/0.6736`, `0.7355/0.7479` at conf
  `0.05/0.15/0.25`. V10 cuts some partial FPs versus V9 (`-5` at conf `0.15`,
  `-2` at `0.25`) but loses `0.019` recall on partial test at both thresholds.
  Held-out unknown-money is slightly worse than V9 at conf `0.25`
  (`41/237`, `47` detections vs `39/237`, `46`), true-empty is slightly better
  (`2/441`, `3` detections), and split unknown rows match V9 on foreign/KHR100
  while worsening USD2. Keep the signal that excessive erasing may encourage
  some FP proposals, but do not continue a no-erasing ladder; the model still
  needs clean/source transfer and count safety together.
- **The V9-style recipe is not seed-stable enough for promotion.** V11 repeats
  V9's v2-schedule13 recipe with seed `1`:
  `runs/cashsnap/production_pilot_v11_v2schedule13_from_v2_e3_i416_b2_w0_adamw_lr2e5_nowarmup_noamp_cachefalse_freeze22_seed1/weights/last.pt`.
  It keeps some partial/count-safety benefits, especially conf `0.25` partial
  test/val (`0.7524/0.7524` and `0.7686/0.7381`, with fewer FPs than V9), but
  the broader detector balance does not reproduce: full real drops to `0.8584`,
  strict clean to `0.8645`, and source-excluded clean collapses to `0.7099`.
  Unknown-money also worsens at conf `0.25` (`46/237`, `50` detections) and
  true-empty returns to `4/441`, `5`. Treat V9 seed0 as a useful high-water mark,
  not a validated recipe; the next recipe must explicitly stabilize source
  transfer while preserving the partial/count-safety gains.
- **Small source-clean replay does not stabilize the V9-style recipe.** V12 adds
  12/class train-side source-clean replay rows to the V11 seed-1 setup:
  `runs/cashsnap/production_pilot_v12_seed1_sourceclean_replay12_from_v2_e3_i416_b2_w0_adamw_lr2e5_nowarmup_noamp_cachefalse_freeze22_seed1/weights/last.pt`.
  It only partly repairs source-excluded clean versus V11 (`0.7346` vs
  `0.7099`), while full real stays V11-like (`0.8587`) and strict clean stays
  acceptable (`0.8662`). Do not use the old V9 `0.7938` source number as this
  branch's current comparator without checking the list version. Its best signal
  is partial-test conf `0.25` (`0.7714/0.7714`, FP delta `-4`, recall delta
  `+0.0286` versus V9), but conf `0.15` loses recall and the guardrails worsen:
  unknown-money conf `0.25` is `46/237`, `50` detections, true-empty is
  `5/441`, `6`, USD2 is `22/41`, `23`, and KHR100 is `9/12`, `11`. Kill this
  replay path as a promotion candidate; keep only the clue that the
  high-confidence partial slice can improve without solving source/unknown-money
  safety.
- **13-class contrastive pos/neg continuation is not the promotion path.** V13
  adds source-policy positives plus train-safe out-of-schema/unknown/empty
  negatives to V9. It improves full/strict AP (`0.8670`/`0.8671`) and slightly
  improves true-foreign suppression, but the old source-excluded comparison
  against V9 `0.7938` is list-stale; USD2 remains worse (`21/41`, `23`
  detections), and KHR100 is not repaired. Use it as evidence that true-foreign
  suppression is learnable, not as a row-balance recipe to repeat.
- **Source-balanced contrastive replay does not unlock promotion.** V14 adds
  audit-clean source-excluded replay to the V13-style contrastive mix and still
  sits around source-excluded `0.738` mAP50-95 with weak precision (`0.594`).
  Kill this branch unless a genuinely different objective or schema policy is
  being tested.
- **Gentle contrastive continuation is a scored tradeoff, not promotion.** V15
  starts from V9 with the V13-style contrastive config for one low-LR epoch
  (`lr0=3e-6`). On the current-list scorecard it beats V9 AP:
  full/strict/source `0.8667`/`0.8688`/`0.7803` vs V9
  `0.8629`/`0.8669`/`0.7394`. Filtered partial is threshold-mixed: test
  `0.9429/0.5103`, `0.8476/0.6794`, `0.7905/0.7477`; val
  `0.8678/0.4565`, `0.7934/0.6115`, `0.7521/0.6842` at conf
  `0.05/0.15/0.25`. Background safety is not a clean win: at conf `0.25`,
  combined unknown-money is `42/237`, `47` detections vs V9 `39/237`, `46`,
  true-empty is `4/441`, `4` vs V9 `3/441`, `3`, USD2 worsens, foreign slightly
  worsens by images-with-FP, and KHR100 ties. Keep V15 as the AP/source and
  high-conf partial clue, but the browser hard-slice gate is a kill signal for
  product-count selection: historical `conf=0.15` exact-value falls to `62/79`
  versus champion `68/79` and V9 `67/79`.
- **Low-LR base replay can also improve current-list AP, but not the full gate.**
  V16 is the control: one low-LR epoch from V9 on the original V2
  hard-negative-guard data, without V13/V15 contrastive extras. It scores
  full/strict/source `0.8664`/`0.8698`/`0.7786`, wins the strict-clean AP and
  partial-test conf `0.15` slice, and matches V9 true-empty at conf `0.25`
  (`3/441`, `3`). Under the KHR-only product-style stack, the hard-slice/background
  tie-break favored V9 (V9 reached `61/79` exact and `3/22` background-FP vs V16 `59/79`
  and `6/22`). However, under calibrated USD-risk20 product stack settings, V16
  re-enters the frontier: it ties V9 on hard-slice exact (`65/79`), beats V9 on broad
  full-real exact (`1440/1562` vs `1435/1562`) and strict clean exact (`748/774` vs `747/774`),
  holds partial exact matches, and performs comparably on split unknown money splits
  (V16 wins foreign_asian by 1 FP, V9 wins KHR100 by 1 FP, and USD2 is tied).
  This makes V16 a highly competitive product-stack contender.
- **Thin V9 obligation replay is not the missing objectness fix.** V18 starts
  from V9 for one low-LR epoch on
  `configs/webgl_ablation/cashsnap_production_pilot_v18_v9_obligation_thin.yaml`,
  adding train-split analog positives from V9 full-test misses plus source-shaped
  safe empty/unknown negatives. It does not promote:
  full/strict/source `0.8625`/`0.8687`/`0.7430`; filtered partial improves on
  test but spends val recall (`test conf 0.05/0.15/0.25` =
  `0.9524/0.4950`, `0.8762/0.6715`, `0.8095/0.7589`; val =
  `0.8595/0.4561`, `0.7851/0.6169`, `0.7521/0.7054`). The intended
  count-safety gain fails: held-out unknown-money worsens to `44/237`, `52`
  detections at conf `0.25`, true-empty worsens to `5/441`, `5`, V9-lightweight
  full-test precision drops to `0.4962` with `173/748` background-FP images, and
  the product-style hard slice falls to `57/79` exact-value and `6/22`
  background-FP images after gate+final NMS. Evidence:
  `runs/cashsnap/production_pilot_eval_suite_v1/scorecard_summary_V9_V16_V18_current.json`.
  Kill direct train-analog obligation replay as a count-safety mechanism unless
  paired with a genuinely different detector objective or schema policy.
- **Split unknown-money by product policy.** V6's combined held-out improvement
  comes from true foreign-money suppression, not missing-schema official
  denominations: at `conf=0.25`, foreign Asian currency improves v2 `19/184`,
  `20` detections to v6 `16/184`, `17`, while USD2 stays `18/41` and KHR100 stays
  `8/12` with class-13 ignored detections still zero. V9's split is similar but
  slightly better on true foreign money: foreign Asian `15/184`, `17` detections,
  USD2 `16/41`, `18`, and KHR100 `8/12`, `11`. Treat USD2/KHR100 as a
  schema-compatibility problem, not proof that more generic unknown rejection is
  the right detector objective. Under product `conf=0.05`, V9 USD-risk15 floors
  reduce USD2 and foreign Asian detector FPs but leave detector-only KHR100
  unchanged at `12/12` images-with-FP; the current gate+final-NMS stack then
  rejects KHR100 to `0/12` and true foreign to `3/184`, while USD2 remains
  `17/41`. That is a product-stack schema guard, not one-detector learning.
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
- **Official21/KHR100 work is schema diagnostic, not a pilot replacement.**
  Official21 probes show staged missing-class learning is possible, but the
  operational detector remains the current 13-class schema. Prior official21
  checkpoints are far below the core13 pilot on current-class AP; KHR100 can be
  forced only with severe core-regression, and USD2 pseudo runs light up pseudo
  val with poor precision/calibration. Mapped official21 init preserves more
  old-class signal but is still not a compatible production-pilot result.

### Untested Ideas

- **Stabilize before promotion, but not by more blind 13-class row balancing.**
  V9 seed0 remains the simple candidate-to-beat, while V15/V16 show low-LR V9
  continuation can improve current-list AP/source without solving product
  count-safety. A narrow V16 -> KHR100/foreign empty-negative V17 bump was
  considered, materialized, then discarded before training because it did not
  answer the broader count/value bottleneck. Do not recreate it just because the
  ingredients are available. The next useful test should change the mechanism:
  source-aware/validation-weighted sampling, a detector-side objectness/proposal
  objective, or an explicit schema/policy bridge for official out-of-schema
  notes. Kill it if current-list source-excluded clean cannot beat or hold V9
  and V9 browser hard-slice/unknown-money/true-empty safety is not matched.
- **Fix unknown-money rejection without bluntly spending recall.** V3 and v4
  show broad hard negatives are not enough by themselves: too much dose hurts
  clean/partial, and same-budget diversity gives up v2's count safety. V6/V7 show
  train-split real unknown-money FP boxes relabeled as `UNKNOWN_FOREIGN_NOTE`
  suppress some target FPs but do not teach held-out UNKNOWN routing; V8 shows
  the same rows as empty negatives are also insufficient. V18 shows thin
  train-split full-test obligation replay also fails the held-out unknown/empty
  and hard-slice count gates. The next unknown-money test needs a different
  objective or schema/policy structure, not more direct replay of the same
  source-shaped rows.
- **YOLO26s is a capacity question, not the current bottleneck.** Do not run it
  just because it is available. Run it only if v2-style data/objectness pressure
  plateaus and the remaining errors look like capacity/feature limits. A direct
  `yolo26s.pt -> pilot` run is only a smoke because it lacks the CashSnap clean
  foundation; the fair comparison is `yolo26s.pt -> clean p24 synth+real
  foundation -> pilot`, judged on the same blended scorecard plus browser/phone
  model size and latency.
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
  source, including `KHR_100`, but it still lacks `USD_2`; current 13-class
  weights cannot evaluate it directly.

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
- `USD_2` and `KHR_100` are official money but outside the current core-13
  detector; treating them as empty negatives is a product-policy choice, not a
  detector truth.
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
  for YOLO training). Prefer `--memory-clean-preset memreduct`; it triggers the
  installed COM task path `memreductTask=-clean` for
  `C:\Program Files\Mem Reduct\memreduct.exe`, then ends the task so it does not
  stay resident. Upstream project: `https://github.com/henrypp/memreduct`. Do
  not call plain `schtasks /Run` because it can launch literal `$(Arg0)`.
  `CashSnapWinMemoryCleaner` remains a working fallback.
- For headroom-wrapped training, do not override the memory cleaner to
  `--memory-clean-min-free-ram-gb 4.0 --memory-clean-cooldown-seconds 0` during
  normal interactive use; that can loop the cleaner every interval when baseline
  RAM is below 4 GB. Prefer the script defaults unless a run proves otherwise.
- `batch=auto` can speed harness diagnostics when RAM is healthy, but YOLO train
  batch size is part of the model recipe; do not compare an auto-batch run
  directly against the `batch=2` pilot line unless the batch change is the tested
  variable and the scorecard says it preserved behavior.
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
