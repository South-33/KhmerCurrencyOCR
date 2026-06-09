# CashSnap Model Brain

This is the living working memory for model and synthetic-data decisions. Keep it
short, current, and decision-oriented. Old detail belongs in `docs/archive/`,
registries, or the folder structure itself.

Major history snapshots:
- `docs/archive/model_brain_pre_housekeeping_2026-06-08.md`
- `docs/archive/model_brain_pre_cleanup_2026-06-07.md`
- `docs/archive/model_brain_pre_compact_2026-06-07.md`
- `docs/archive/model_brain_full_history_2026-06-06.md`

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

North star: build a small phone/browser-deployable model that counts mixed USD
and Khmer riel from one casual retail photo.

Current phase: synthetic data must transfer to real clean-visible notes before
the project seriously moves into overlap/fan/hand curricula.

Hard clean-base target: `0.82-0.85` full real test mAP50-95. This is realistic
because a near-size real-trained control has already reached `0.819153` on full
real test.

Current reality:
- Legacy weak TSTR references: filtered185 fast fixed-step `0.075907`,
  filtered185 slow `0.142098`, and old target-anchor latest `0.144740`.
- Clean-checkpoint synthetic-data-only repair is now much stronger but not
  promoted: poisson/contact plus 32 train-only FP-mined negatives reached
  `0.747316` seed0, with `USD_50` guard failure and no seed repeat yet.
- Strict `yolo26n.pt`-init transfer is no longer at the old `0.15` floor:
  scaled poisson/contact `bal96` reached `0.420709` at a fair 1,000-step
  budget, and six manually vetted foreign-note hard negatives pushed it to
  `0.503504` while passing the `0.05` per-class-drop guard.
- Positive-only slice verification for the current strictbest is much stronger
  than older blend185 audits: clean-visible test mAP50-95 is `0.697` and
  labeled-all test is `0.610` (`0.667`/`0.587` without hardneg6), so the old
  `~0.22` clean-positive result applies to earlier blend185/hardnegold8 models,
  not this final strict detector.
- The strict-base hard-negative gain is a mechanism clue, not a win. The target
  gap is still `0.316-0.346` absolute mAP50-95, and the mining audit exposed a
  broken "empty-label" bucket mixing true empty frames, valid out-of-taxonomy
  foreign-note negatives, and mislabeled in-taxonomy KHR notes.
- Empty-label semantic bridge v1 confirms the bucket problem: train empty-label
  rows split into likely true empty `3619`, suspect unlabeled target `221`,
  currency review `174`, student-overfire review `1501`, and model review
  `1403`; val/test also contain suspect unlabeled targets (`58`/`22`) and
  currency-review rows (`35`/`20`).
- Naive teacher-pseudo target anchoring is rejected. Appending 221
  teacher-labeled suspect target positives to the current strict best crashed
  real test mAP50-95 `0.503504 -> 0.339228` with 10 per-class guard failures;
  the target/non-target/empty split is still useful, but pseudo rows are not
  clean labels and cannot be treated as a simple positive dose.
- A class-balanced active label queue now exists for the safer bridge path:
  `runs/cashsnap/semantic_bridge_active_label_queue_train_v1/queue.csv` selects
  160 train-split suspect-target/currency-review rows from 395 available
  candidates and writes a blank `review_template.csv`. Use it for review/manual
  relabeling only; do not train on it as pseudo-labeled data.
- Semantic true-empty background replacement is rejected as a positive-transfer
  path: the fair hardneg6 A/B scored `0.503504 -> 0.332511` with 10 class guard
  failures. Background diversity alone, when sampled from likely-empty real
  frames, can destroy the positive decision boundary.
- Single-box phone-context same-class replacement is rejected as a direct
  strict-base positive path: the fair hardneg6 A/B scored
  `0.503504 -> 0.186959`, delta `-0.316545`, with all 13 classes failing the
  per-class guard. Keep its source reuse reports and teacher/remnant audits as
  QA infrastructure, but do not tune this replacement family without a new
  mechanism.
- Current strict-best obligation ledger:
  `runs/cashsnap/synthetic_obligation_ledger_current_strict_best_v1.md`. Its
  lightweight eval shows recall `0.6438`, precision `0.2321`, background FP
  `516/748`, with USD misses concentrated in `billsbank` and large/unknown FPs
  in `asian_currency`; next synth package must pair target positives with
  unknown/near-negative scenes and visible-extent geometry, not just add
  positives.
- The first obligation-mix probe found a real but unsafe mechanism.
  Teacher-accepted weak-USD alpha positives plus 320 likely-empty/unknown negatives
  raised strict full-test mAP50-95 `0.503504 -> 0.589059` and cut lightweight
  background FPs `516/748 -> 22/748`, but failed promotion by suppressing recall
  (`0.6438 -> 0.5557`) and dropping `KHR_1000 -0.126492`, `KHR_5000 -0.120518`.
  A reduced 80-negative dose improved aggregate AP a little more (`0.595659`)
  but kept the same lightweight recall (`0.5557`), worsened `KHR_1000`
  (`-0.160297`), and relaxed background FPs to `47/748`. Treat this as proof
  that unknown pressure is powerful and too blunt; do not continue pure dose
  tuning without a new mechanism.
- Teacher-accepted recall-repair positives do not rescue the obligation mix by
  simple dosing or staging. A `safeunknown80` failure-led repair render accepted
  `698/768` rows under the real teacher, but appending them to `safeunknown320`
  only moved full mAP50-95 `0.589059 -> 0.594999` while dropping lightweight
  recall to `0.5312`. A short positive-only rescue from the `safeunknown320`
  checkpoint recovered some low-confidence recall (`0.5912`) but crashed full
  AP to `0.478540` and reopened background FPs to `178/748`. The issue is not
  "more positives after negatives"; it needs class/source-aware sampling or a
  different objective, if this branch is revisited at all.
- Paired target+foreign scenes are also only a partial mechanism. Free-side
  layout fixed the first border/overlap issue and teacher accepted `154/208`
  rows, but appending those rows to the strict foreign-hardneg6 best scored
  `0.503504 -> 0.524653` while failing per-class guards on `KHR_1000`,
  `KHR_500`, `USD_1`, and `USD_50`. Lightweight scorecard moved recall
  `0.6438 -> 0.6181`, precision `0.2321 -> 0.2590`, and background-FP images
  `516/748 -> 478/748`: a small boundary improvement, nowhere near the
  safeunknown `22/748` and not enough to scale this six-background pack.
- The first explicit unknown-class view is rejected. Rematerializing the same
  154 paired rows with class `13: UNKNOWN_FOREIGN_NOTE` dropped full AP
  `0.503504 -> 0.453351` with 8 guard failures; after filtering unknown
  predictions, lightweight recall was `0.6193`, precision `0.1887`, and
  background-FP images worsened to `548/748`. A tiny repeated six-background
  unknown class confuses target learning and does not teach real unknown
  routing.
- Broad labeled WebGL unknown props are useful harness infrastructure, but the
  direct 14-class row objective is killed for now. The original 510-row
  direct14 run looked strong (`0.503504 -> 0.583225`) and lightly reduced
  background FPs (`516/748 -> 456/748` when ignoring UNKNOWN predictions), but a
  phase audit showed it stopped at LR progress `0.714` with `0` post-close
  steps while the baseline reached LR progress `1.000` with `200` post-close
  steps. A phase-matched 510 rerun (`36` epochs, `close_mosaic=7`) collapsed to
  mAP50-95 `0.437491`, failed nine real-class guards led by `KHR_50000`
  (`-0.286593`), and worsened lightweight filtered recall/precision/bgFP to
  `0.5814`/`0.2059`/`532/748` (`574/748` unfiltered). The 256-row half dose was
  also bad (`0.460125`, recall `0.5851`). Verdict: keep the WebGL UNKNOWN
  exporter and the train-phase audit, but do not run more direct unknown-dose or
  phase-tuning sweeps without a new structural mechanism such as class/source-
  aware sampling, an unknown/proposal gate, or a different loss objective.
- A binary proposal gate is the best structural signal after the direct UNKNOWN
  collapse. `scripts/probe_yolo_proposal_gate.py` evaluates a crop-level gate on
  YOLO proposals while preserving detector denomination classes. The archived
  old-overlap gate was the first proof: it preserved test recall (`0.6438 ->
  0.6438`), improved precision (`0.2316 -> 0.3076`), cut background-FP images
  (`533/748 -> 329/748`), and raised exact-value images `491 -> 703`. The new
  current-domain best is the pretrained edge-positive plus semantic true-empty
  gate at `reject>=0.99`: val/test recall is exactly preserved (`0.6751 ->
  0.6751`, `0.6438 -> 0.6438`), test precision improves `0.2316 -> 0.3809`,
  test background-FP images fall `533/748 -> 87/748`, count MAE improves
  `1.0423 -> 0.4763`, and exact-value images rise `491 -> 945` with zero
  TP-like rejections. The diagnostic stack is
  `configs/cashsnap_two_stage_strictbest_trueempty_proposal_gate_browser_stack.json`.
  Naive gate training is still killed: target-vs-reject, banknote/background,
  edge-positive-only, and from-scratch true-empty training either over-rejected
  target proposals or collapsed to all-reject. Treat proposal gating as a real
  product architecture branch, not synthetic success or detector mAP progress.
- Weak-USD positive-only support after the gate is rejected as a promotion path
  but kept as a source-specific clue. Appending 317 teacher-accepted weak-USD
  alpha positives to the strict best, phase-matched to 1,000 steps, dropped
  standalone test mAP50-95 to `0.420` and under the true-empty gate lowered test
  recall `0.6438 -> 0.6267` and exact-value images `945 -> 938`. It did improve
  test `USD_5`/`USD_50`/`USD_100` recall and `billsbank` source recall, so the
  useful mechanism is localized USD/source support, not blind positive-only
  appending; any revisit needs protected KHR and `usd_total` source guards.
- Post-gate detector misses are mostly denomination, not localization. The
  refreshed proposal-gate harness reports strictbest test same-class recall
  `0.6438` but class-agnostic localization recall `0.8421`: `162` GT notes have
  an IoU-matched wrong-denomination proposal and `129` have no usable box. A
  real-train denomination crop reclassifier is the upper-bound product branch
  (`0.972` real-val crop accuracy; test recall `0.8262`, precision `0.4888`,
  exact-value `1031`). Current synthetic-only crop training does not transfer
  without stronger augmentation (`0.548` real-val crop accuracy; proposal recall
  `0.4933`). Crop-transfer augmentation raises synthetic-only to `0.700` val and
  gives a modest pure-synth product lift at confidence-gated override
  (`0.6659` recall, exact-value `960`). Adding the 317 weak-USD alpha positives
  to the pure-synth crop set raises crop val to `0.758` and exact-value to
  `973` at the same recall/precision, mostly by improving weak USD value errors.
  Synthetic becomes much more useful with a small real style anchor:
  real-24/class e8 alone reaches `0.628` val / `0.4982` proposal recall, while
  weak-USD-augmented synthetic + real-24/class with train-only synthetic/real
  edge fragments reaches `0.936` val and, with `KHR_50000` reclassifier override
  blocked, real-test proposal recall/precision/exact-value `0.7809`/`0.4620`/
  `1021`; this is now close to the real-full crop upper bound
  `0.8262`/`0.4888`/`1031`. Treat the crop branch as the strongest current synth
  clue: target crop/denomination transfer plus tiny real calibration and
  fragment-shaped training, not more full-scene detector row dosing. Lowering
  the detector proposal threshold is not a usable no-box fix: at `conf=0.01` the
  previous three-stage stack cuts no-box FNs `129 -> 66` and raises recall to
  `0.8005`, but precision falls to `0.2465`, FPs jump `773 -> 1999`, and
  exact-value drops `1007 -> 805`.
- Full real-flat asset-style detector repair is rejected. A class-diverse
  train-split real-flat bank with 90 rectified assets across all 13 classes,
  current poisson/contact/background-tone rendering, and the same hardneg6 train
  mix scored far worse than strictbest at the fair 1,000-step `yolo26n.pt`
  budget: full real test mAP50-95 `0.503504 -> 0.322847` with 11 class guard
  failures. Under the current true-empty gate plus fragment-trained reclassifier
  it also worsened the bottleneck it targeted: recall `0.7809 -> 0.6867`,
  localization recall `0.8421 -> 0.7368`, no-box FNs `129 -> 215`, and
  exact-value images `1021 -> 1011`. Do not keep tuning real-flat asset
  diversity; the missing detector mechanism is not fixed by replacing Numista
  assets with train-split flat assets under the same target-anchor renderer.
- Class-agnostic BANKNOTE detector factorization is rejected as a no-box repair
  under the current strictbest train mix. A one-class view trained for the same
  1,000-step `yolo26n.pt` budget scored one-class real-test mAP50-95 `0.4401`
  and, under the true-empty gate plus fragment reclassifier, traded cleaner
  outputs for worse product recall: recall `0.7809 -> 0.6952`, precision
  `0.4620 -> 0.6194`, localization recall `0.8421 -> 0.7601`, no-box FNs
  `129 -> 196`, and exact-value images `1021 -> 1167`. It doubled `billsbank`
  no-box misses (`51 -> 104`) while leaving `usd_total` about flat (`62 -> 64`);
  useful as a precision/counting clue, but not the needed detector coverage
  mechanism.
- The clean-real detector control proves the strict detector's no-box problem is
  a data/objective miss, not model capacity. The high clean-real checkpoint
  scores full-test mAP50-95 `0.883801`; through the same proposal/gate audit it
  reaches pre-gate recall/localization `0.9731`/`0.9853` with only `12` no-box
  FNs, while strictbest is `0.6438`/`0.8421` with `129` no-box FNs. A train-split
  teacher/student audit (`scripts/audit_detector_teacher_student_gap.py`) shows
  the clean teacher localizes `1025` GT boxes the strict student misses and
  same-classes `2249` GT boxes the strict student does not, concentrated in
  `usd_total`/`billsbank` and `USD_5`/`USD_50`/`USD_100`/`USD_20`.
- USD visible extent is a real but insufficient target-anchor mechanism.
  Strictbest synthetic boxes average area `0.3579` with only `256/1248` boxes
  `>=0.50` and `14/1248` `>=0.90`, while real train/test are around
  `0.51-0.53` mean with many full-frame notes. A same-budget USD extent-heavy
  replacement mix moved the intended proposal bottleneck in the product audit
  (localization recall `0.8421 -> 0.8568`, no-box FNs `129 -> 118`,
  `usd_total` localization `0.7559 -> 0.8031`, post-reclassifier recall
  `0.7809 -> 0.8005`) but failed strict transfer: full-test mAP50-95
  `0.503504 -> 0.497547`, guard failures on `KHR_1000` and `KHR_10000`,
  pre-gate background FPs `533 -> 597`, and exact-value images
  `1021 -> 1016`. Verdict: stop scaling clean-background extent-heavy
  target-anchor rows; keep the lesson that extent must be coupled with
  source/context accounting, unknown-note pressure, and protected KHR behavior.

Distance to target is now two tracks: clean-checkpoint repair is roughly
`+0.07` to `+0.10` from the hard line but guard-failing; strict base-init
generation remains roughly `+0.32` to `+0.35` away for guard-passing models.
The best strict diagnostic gains above `0.503504` are non-promotable bottleneck
clues, not a narrowed finish line. Base-init gains are useful only when they
reveal a scalable mechanism; do not celebrate relative improvement from a weak
baseline.

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
- Start every non-trivial next step from the yardstick: current best real score,
  hard target, and remaining gap. Treat small gains as bottleneck clues, not
  wins; if ten repeats of that gain would not approach `0.82-0.85`, do not
  confuse it with a path to done.
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

Use this section as working state, not a pep talk. Keep untested ideas short and
sharp; if the next good idea is unclear, say that instead of padding the list.

### Current State

- Synthetic-to-real clean-visible transfer is the blocker. Synthetic self-eval,
  package QA, geometry gates, and visual contact sheets can all pass while real
  transfer still fails.
- Clean-context tone repair now has a direct proxy-vs-transfer warning:
  off/default WebGL failed the crop visual gate on compressed highlights,
  bridge-tone overcorrected contrast/saturation, and local-tone passed the
  8-row and 64-row crop visual gates, but a 64-row pure-synth TSTR still scored
  only val/test mAP50-95 `0.023897`/`0.026258`, worse than the already-weak
  clean-topdown reference (`0.028250`) and far below handled96 (`0.054845`).
  Do not scale clean-context tone/prop polish; renderer work needs a structural
  scene-realism or real-context change.
- The old target-anchor latest TSTR leader (`0.144740`) is no longer the
  clean-checkpoint repair leader, but it remains the warning example: it misses
  `669/817` real-test GT boxes at `conf=0.05` and fires on `174/748` empty test
  images.
- The current clean-checkpoint repair clue is meaningful (`0.747316`), but the
  strict base-init yardstick is the honest synthetic-generator yardstick:
  current best is `0.503504` from scaled poisson/contact `bal96` plus six
  curated foreign-note hard negatives, still far below the `0.82-0.85` target.
- Reverse-transfer asymmetry is the core warning. A real-trained model reads the
  synthetic blend well (`0.866141`), while the pure-synth model learns synthetic
  self-eval (`0.914675`) but transfers poorly (`0.142098`).
- Representation probes show synthetic-vs-real separability at early and late
  layers. Camera/image-formation statistics, context, extent, and background
  rejection are still the main suspects.
- Training convergence is a real audit risk, but not a sufficient explanation.
  Root scripts already support promotion posture (`batch=64`, `workers=0`);
  low-batch fixed-step runs are diagnostic probes, not proof that the whole
  synthetic gap is just undertraining.
- Promotion-style continuation can preserve far more real performance than
  low-batch diagnostics. Target-anchor latest balanced20 at
  `i416/b64/steps150` scored `0.623422`; useful as a control, but still far
  below the clean real checkpoint and not synthetic success.
- Best promotion-style synthetic repair axis so far is balanced20
  poisson/contact/geoscale/min-short appearance plus calibrated train-only
  FP-mined negatives: 32 spread rows reached `0.747316` and cut empty-frame test
  FPs hard, but it still fails promotion on `USD_50` and needs a seed repeat.
- Best strict-base synthetic axis so far is scaled poisson/contact `bal96` from
  `yolo26n.pt`: base positives reached `0.420709`, and curated foreign-note
  hard negatives reached `0.503504` without per-class guard failures. The lesson
  is target-vs-non-target pressure plus positive appearance, but the gap is too
  large to keep dose-tuning without first fixing real target/non-target/empty
  semantics.
- The current strict diagnostic leader is not promotable:
  weak-USD alpha/preserve-luma positives on clean no-note backgrounds, filtered
  by teacher agreement (`317/384` accepted), plus 320 train-only likely-empty
  unknown negatives. It raised full real test `0.503504 -> 0.589059` and
  precision `0.625462 -> 0.770639`, but failed the no-regression guard on
  `KHR_1000` and `KHR_5000`. Lightweight scorecards show the mechanism clearly:
  background FPs almost vanish (`516/748 -> 22/748`) and precision jumps
  (`0.2321 -> 0.5463`), while recall falls (`0.6438 -> 0.5557`). The safe
  negative bucket is valid as pressure/review, not as positive-render
  backgrounds, and the next experiment must balance this pressure with
  `KHR_1000`/`KHR_5000` and weak-USD recall repair.
- A reduced negative-dose check does not rescue the obligation mix. The
  `safeunknown80` variant reached `0.595659`, slightly above `safeunknown320`,
  but still failed promotion with `KHR_1000 -0.160297` and `USD_1 -0.054681`.
  Its lightweight scorecard kept the same recall as `safeunknown320` (`0.5557`)
  while allowing more background FPs (`47/748` vs `22/748`). This kills running
  the already-built `safeunknown160` dose as a default continuation; only run it
  with a new reason, such as paired recall-repair positives or a different loss/
  sampling setup.
- Failure-led repair positives are now tested and are not enough as a row dose.
  The `safeunknown80` lightweight failures generated a repair queue, rendered
  clean no-note alpha positives for `KHR_1000`, `KHR_10000`, `KHR_5000`, and
  weak USD classes, and accepted `698/768` under the real teacher. Appending
  those rows to the `safeunknown320` mix barely changed full AP (`0.589059 ->
  0.594999`), improved `KHR_5000` but hurt `KHR_50000`, `USD_1`, `USD_50`, and
  the lightweight scorecard (`recall 0.5312`, `bgFP 40/748`). A staged
  positive-only rescue from the `safeunknown320` checkpoint failed harder:
  `0.478540` full AP, with `KHR_1000 -0.163256` and `KHR_50000 -0.552450`
  versus safeunknown320. It raised lightweight recall to `0.5912` by relaxing
  the detector (`precision 0.3788`, `bgFP 178/748`, `KHR_1000` precision
  `0.1871`). Kill simple recall-repair dosing/staging; only revisit this
  mechanism with class/source-aware sampling, loss weighting, or a two-head/
  unknown-gated objective.
- A naive two-model rescue gate is also only a clue. Using `safeunknown320` as
  the base and adding high-confidence strict-best boxes for selected recall
  classes on test improved lightweight recall `0.5288 -> 0.5618` under the gate
  harness and helped `KHR_1000`, `KHR_5000`, `KHR_2000`, and `USD_10`, but
  background FPs doubled `22/748 -> 46/748` and precision dropped
  `0.7036 -> 0.6820`. The conflict is class-separable enough to justify a
  real proposal/unknown-gate or objective test, but not enough for a simple
  inference-time box-union threshold sweep.
- Paired target+foreign scenes are a better-shaped question than more
  unknown-only negatives, but the first six-background hardneg6 pack is not a
  promotable continuation. The renderer now supports avoid boxes plus free-side
  placement, which fixed the first overlap/border artifact. Teacher accepted
  `154/208` free-side rows and the fair strict A/B improved aggregate AP
  `0.503504 -> 0.524653`, but failed guards on `KHR_1000`, `KHR_500`,
  `USD_1`, and `USD_50`; lightweight recall fell `0.6438 -> 0.6181`, and
  background-FP images only improved `516/748 -> 478/748`. Verdict: keep the
  layout/avoid-box harness and the mechanism clue, but do not scale the current
  hardneg6 pair pack without much broader unknown diversity or a structural
  target-vs-unknown objective.
- The simplest explicit unknown-class structural test is also killed. Copying
  the same `154` teacher-accepted free-side rows into a 14-class view with one
  `UNKNOWN_FOREIGN_NOTE` box per foreign note made the fair strict A/B worse
  (`0.503504 -> 0.453351`, 8 guard failures). Product-filtered lightweight eval
  ignored 152 unknown predictions but still worsened target precision and
  background FPs (`548/748`). This does not kill a serious proposal/unknown
  architecture, but it kills "add one unknown class from six repeated foreign
  backgrounds" as the next jump.
- Renderer support for broader unknown-objective experiments is real harness
  infrastructure, but the direct14 row objective is not a promotion path. WebGL
  negative `unknown_banknote` props can be opt-in exported as class
  `13: UNKNOWN_FOREIGN_NOTE` with
  `--negative-prop-label-policy unknown_banknote`; the smoke pack passed
  4-image/6-box render, YOLO, visual, and registry checks, and the broad pack
  kept 510/512 images after dropping two ID-sliver QA failures. The first 510-row
  win was epoch/LR/close-mosaic phase-confounded, the 256-row half dose
  collapsed, and the phase-matched 510 rerun collapsed harder (`0.437491`,
  filtered bgFP `532/748`). This supports a structural class/source-aware
  unknown objective or proposal gate, not blind scale-up/down dose sweeps.
- Activation-guided real-train bridging changed the strict base-init regime but
  is diagnostic only because selection used test failure modes. The compact
  bridge raised current scaled synth `0.148345 -> 0.224808` and cut empty-frame
  test FPs `358->40`; adding a 24/class real floor raised it to `0.305717` and
  still cut test FPs `358->66`, but both variants fail no-regression on
  `KHR_5000` and `USD_1`. Treat this as proof that the missing mechanism is
  real scene/capture support plus background pressure, not as a promotable model
  or a reason to dose-tune test-guided rows.
- Real flat-bill extraction is useful diagnostic infrastructure, but the first
  transfer ablation is rejected. The stricter train-only scan-source pass mined
  94 shaped, single-bill candidates, BEN2/rectified 90 raw assets across all 13
  operational classes, and manually accepted only 17 scan-like full-bill cutouts
  across 7 classes. A full-row `manual17+Numista-missing` alpha/no-tone render
  lost a fair 1,000-step `yolo26n.pt` A/B against the scaled poisson/contact
  reference (`0.420709 -> 0.310841`, delta `-0.109868`, 7 guard failures). It
  modestly reduced empty-frame FPs, but not enough to offset positive collapse.
  Do not dose-tune this small manual bank.
- Real-trained and hybrid controls prove the architecture can work, but naive
  real+synthetic is not magic. Full real+accepted-WebGL scored `0.807251` and
  failed against the clean checkpoint by `-0.076550`, led by `KHR_50000`.
- The own-photo capture bridge is empty, so rare/high-value class claims and
  mixed USD/KHR retail scenes are still under-validated.

### Tested Ideas

- **Target-anchor latest-design transplant: useful clue, not foundation.** It is
  the current synthetic-only leader because it combines train-only CashSnap
  no-note pixels, real CashSnap geometry, and latest-design target assets. The
  result is too weak to scale directly.
- **Camera/ISP context train260: partial clue, not a better pack.** Against an
  older filtered185 control it improved aggregate transfer by `+0.017795` but
  failed `USD_10` and traded empty-frame FPs. Against the same-size target-anchor
  latest balanced20 control at `i416/b64/steps150`, it lost
  `0.623422 -> 0.524175`, delta `-0.099247`, with 9/13 per-class guard failures
  and worst `KHR_500 -0.551811`. Keep camera/ISP as an isolated anti-shortcut
  axis, not as this full context/geometry package.
- **Poisson/contact image formation: strongest supporting axis, unsafe alone.**
  Low-batch transfer improved (`0.028350 -> 0.042401`). A fair same-size
  `i416/b64/steps150` balanced20 probe improved full real test
  `0.623422 -> 0.701477`, delta `+0.078054`, with large gains on weak
  high-value riel (`KHR_50000 +0.581077`, `KHR_20000 +0.543044`). It still
  fails promotion because `KHR_2000 -0.164626` and `USD_1 -0.051688` trip
  per-class guards, and empty-frame FPs spike at `conf=0.05`
  (val `491->728` images with FP, test `244->449`; detections
  `846->1459` and `421->915`). Keep it high-ranked as an appearance/contact
  layer, but only with background/unknown pressure and class repair.
- **Closed-loop FP-mined negatives: best current clue, not promotion.** On the
  poisson/contact leader, train-only empty-frame FP-mined doses form a real
  non-monotonic curve. Spread-8 improved full real test
  `0.701477 -> 0.722377` and cut test empty-frame detections `915->432` but
  failed `KHR_500 -0.061697`. Spread-16 cut FPs further (`915->357`) but dipped
  to `0.693158` and failed four classes. Spread-32 is the best point so far:
  `0.747316`, test images-with-FP `449->152`, detections `915->201`, val
  `728->266` and `1459->353`, with one blocker `USD_50 -0.063762` and residual
  USD false-positive skew (`USD_1 +62`, `USD_20 +1`). Spread-64 over-suppressed:
  `0.658296`, though FPs fell to `449->94` and `915->136`. This validates
  Agent5's failure-mining loop as useful, but the current winner still needs
  USD_50 repair and seed repeat; the first seed-1 parity repeat stopped during
  baseline training under system RAM pressure, not due a measured model failure.
- **Strict-base hard negatives: useful only after semantic QA.** In the stronger
  `bal96`/1,000-step strict regime, raw empty-label FP-mined doses produced big
  aggregate gains but guard failures: FP32 `0.420709 -> 0.496266`, FP16
  `0.420709 -> 0.524592`, FP8 `0.420709 -> 0.520952`. Visual audit showed the
  FP8 rows mixed valid foreign-note negatives with actual in-taxonomy KHR notes
  mislabeled as background. A six-row manually vetted foreign-note subset passed
  (`0.503504`, worst `KHR_2000 -0.039574`). Clean train no-note patches improved
  less (`0.436412`) and failed six guards. Next work must split true empty,
  out-of-taxonomy note negatives, and suspect unlabeled target positives before
  any more mining.
- **Current-obligation weak-USD plus unknown pressure: mechanism, not
  promotion.** The train-only obligation queue selected weak USD positives from
  `billsbank` and likely-empty unknown pressure from `asian_currency`/
  `cashcountingxl`. Rendering positives directly on those "safe" unknown
  backgrounds failed visual QA because many rows still contain visible currency,
  people, or props; use them as negatives/review/style pressure only. Clean
  no-note alpha/preserve-luma positives were safer than Poisson (`25/32` teacher
  accepted in smoke; scaled pack `317/384` accepted). The fair strict A/B with
  the teacher-accepted positives plus 320 unknown negatives improved aggregate
  mAP50-95 by `+0.085556`, but suppressed operating recall and failed protected
  KHR classes. Reducing unknown negatives to 80 improved aggregate delta to
  `+0.092156`, but did not recover operating recall and made the worst
  protected-class failure worse (`KHR_1000 -0.160297`). Verdict: keep the
  obligation ledger/queue and teacher filtering, but stop pure negative-dose
  tuning; the next target/unknown curriculum needs explicit recall repair or a
  different training objective.
- **Obligation recall-repair positives: simple mix and staged rescue killed.**
  The `safeunknown80` failure queue selected 1,293 train positives and rendered
  768 clean no-note alpha repair rows across `KHR_1000`, `KHR_10000`,
  `KHR_5000`, and weak USD classes; the real teacher accepted `698/768`.
  Visual QA was plausible but still alpha-cutout-like. Appending the accepted
  rows to `safeunknown320` gave only a tiny full-AP change (`0.589059 ->
  0.594999`), failed `KHR_1000`, and further reduced low-confidence recall
  (`0.5312`) despite a small precision gain. A short staged positive-only
  rescue from the `safeunknown320` checkpoint recovered recall by becoming
  permissive again (`0.5912` recall), but full AP fell to `0.478540`,
  background FPs rose to `178/748`, and precision fell to `0.3788`. Verdict:
  the safeunknown branch is a real bottleneck clue, but row-dose recall repair
  and positive-only staging are not the mechanism.
- **Two-model box rescue gate: partial separation, not a breakthrough.** Added
  `scripts/probe_yolo_lightweight_ensemble_gate.py` to test whether the
  high-precision safeunknown model can be a base while a high-recall model adds
  class-limited rescue boxes. On val, staged-rescue boxes reopened FPs quickly.
  Strict-best rescue boxes were cleaner only at high confidence. On test,
  `safeunknown320 + strictbest` at rescue `conf=0.8` raised gated lightweight
  recall `0.5288 -> 0.5618` and repaired `KHR_1000` (`+0.1714`),
  `KHR_5000` (`+0.1622`), `KHR_2000` (`+0.1500`), and `USD_10` (`+0.0781`),
  but background FPs rose `22/748 -> 46/748`. Do not polish this simple gate;
  use it as evidence that any architecture branch needs a trained
  proposal/unknown gate or class/source-aware objective, not box union.
- **Binary proposal gate: real structural signal, true-empty pretrained gate is
  the current best.** Added `scripts/probe_yolo_proposal_gate.py` to evaluate a
  crop-level gate on detector proposals and added per-source unlabeled policies
  to `scripts/build_yolo_proposal_gate_dataset.py` so semantic likely-true-empty
  rows can supply reject proposals while ordinary CashSnap empty-label rows stay
  skipped. Earlier current-domain gates are killed despite high crop-val
  accuracy: target-vs-reject (`0.952`) lost val recall (`0.6751 -> 0.6589`),
  banknote/background (`0.954`) still lost recall (`0.6620`) while barely
  cleaning background FPs (`730/1115`), and edge-positive-only still lost recall
  (`0.6609`; at `reject>=0.95`, `0.6700` but bgFP `715/1115`). From-scratch
  true-empty training collapsed to all-reject on val (`reject acc 1.0`, target
  acc 0.0). The successful recipe is pretrained MobileNetV3, lower LR
  (`0.0003`), edge-positive target support, UNKNOWN-prop banknote positives, and
  semantic likely-true-empty reject proposals. At `reject>=0.99`, val/test
  recall is preserved exactly, test precision rises `0.2316 -> 0.3809`,
  background-FP images fall `533/748 -> 87/748`, count MAE improves
  `1.0423 -> 0.4763`, KHR-total MAE improves `12537.77 -> 1279.77`, and
  exact-value test images rise `491 -> 945`; zero TP-like proposals were
  rejected on val/test. `reject>=0.95` is more aggressive (test bgFP `64/748`)
  but rejected one TP-like proposal on each split. This is a product
  architecture win and should be judged on count/value/unknown behavior, not as
  synthetic detector transfer. A first detector-swap check did not beat the
  strict-best detector: scaled poisson/contact positives without foreign hard
  negatives had lower val recall under the gate (`0.6650`) and worse post-gate
  background FPs (`196/1115`).
- **Weak-USD positive-only append: localized signal, product reject.** Built
  `cashsnap_target_anchor_transplant_poisson_contact_geoscale205_minshort190_bal96_scaled_foreignhardneg6_usdweakalpha317_posonly_probe_puresynth_realval_v1`
  by appending 317 teacher-accepted weak-USD alpha positives without an unknown
  negative dose, then phase-matched training to the strict-best 1,000 steps
  (`40` epochs, `close_mosaic=8`). Standalone val/test mAP50-95 was
  `0.430`/`0.420`, below the strict-best test `0.503504`. With the current
  true-empty gate at `reject>=0.99`, val/test recall fell to
  `0.6357`/`0.6267`; test precision was slightly higher (`0.4009`), bgFP was
  comparable (`89/748`), and the gate rejected zero TP-like proposals, so the
  recall loss is detector-side. Test `USD_5` (`0.1919 -> 0.2929`), `USD_50`
  (`0.3365 -> 0.4231`), `USD_100` (`0.5294 -> 0.6050`), and `billsbank`
  (`0.6006 -> 0.6541`) improved, but `KHR_20000`, `KHR_10000`, `KHR_2000`,
  `usd_total`, and exact-value images regressed. This is useful evidence for
  class/source-aware USD support, not a row-dose continuation.
- **Denomination proposal reclassifier: architecture upper bound plus first
  useful small-anchor synth signal.** Extended `scripts/probe_yolo_proposal_gate.py`
  with class-agnostic localization, wrong-class/no-box FN decomposition, and an
  optional MobileNetV3 reclassifier for kept proposals. Strictbest+true-empty
  gate on real test has same-class recall `0.6438` but localization recall
  `0.8421` (`162` wrong-denomination FNs, `129` no-box FNs), so a classifier can
  theoretically recover a large part of the gap. Built
  `scripts/build_yolo_crop_imagefolder_dataset.py` and four diagnostic crop
  roots. Results: current strictbest synthetic crops alone trained to only
  `0.548` real-val accuracy and hurt proposal recall (`0.4933`; confidence
  `0.90` merely no-ops to `0.6389`). Adding `--aug-profile crop_transfer_v1`
  to `scripts/train_imagefolder_classifier.py` moves synthetic-only to `0.700`
  real-val accuracy; all-override still hurts (`0.6157`), but confidence-gated
  override at `0.90` gives the first pure-synth proposal lift
  (`0.6659` recall, `0.3939` precision, exact-value `960`). Appending the 317
  teacher-accepted weak-USD alpha positives to the pure-synth crop train set
  raises crop val to `0.758` and, with all overrides, keeps recall/precision
  `0.6659`/`0.3939` while improving exact-value `960 -> 973`, USD MAE
  `14.70 -> 12.71`, and weak `USD_5`/`USD_50` recall. Real full-train
  crops are the upper bound: `0.972` real-val accuracy and proposal
  recall/precision/exact-value `0.8262`/`0.4888`/`1031`, reducing wrong-class
  FNs to `13`. The important stronger synthetic signal is small-anchor mixing:
  real-24/class e8 alone scored `0.628` val and damaged proposals (`0.4982`,
  exact `900`). Strictbest synthetic + real-24/class e8 with the crop profile
  scored `0.847` val and improved the gated product to recall `0.7417`,
  precision `0.4388`, exact-value `1006`; adding the weak-USD alpha positives
  raised val to `0.859` and product to `0.7442`/`0.4403`/`1007`, with better
  `USD_5`, `USD_100`, and KHR MAE but worse `USD_20` than the non-USDweak
  small-anchor stack. The review sheet from
  `runs/fragment_classifier/eval_denoms13_usdweak_synth_real24_augcropv1_realval_confusion_v1`
  showed the reclassifier sees detector-like fragments (backs, edges, corners,
  vertical strips), so `scripts/build_yolo_crop_imagefolder_dataset.py` now has
  `full_and_fragments_v1` train-only fragment variants. The fragment-trained
  weak-USD small-anchor set scored `0.936` real-val crop accuracy and improved
  the gated product to `0.7809` recall, `0.4620` precision, and exact-value
  `1020`; blocking `KHR_50000` as a reclassifier override preserved recall,
  raised exact-value to `1021`, and cut KHR MAE `1744.56 -> 766.01`. This is
  the current three-stage leader, still below the real-full upper bound
  (`0.8262` recall, exact `1031`) and still dependent on a small real anchor.
  Pure-synth fragment training improves crop val (`0.758 -> 0.784`) but not the
  product stack (`0.6659`/`973` becomes `0.6646`/`969`), so fragment shape alone
  is not enough without real style anchoring.
  The no-box review sheet
  `runs/cashsnap/no_box_fn_review_strictbest_trueempty_fragtrain_reclass_v1.jpg`
  shows remaining no-box FNs are mostly `billsbank`/`usd_total` USD notes; many
  have same-class detector predictions with bad extent/IoU and others have no
  proposal. The next bottleneck is detector localization/source coverage, not
  crop reclassifier class logic.
  One low-confidence proposal check is killed as a product route: dropping
  detector `conf` to `0.01` raises localization recall to `0.9192` and
  same-class recall to `0.8005`, but precision collapses to `0.2465`, FPs rise
  `773 -> 1999`, count MAE rises `0.4763 -> 1.2074`, and exact-value images
  fall `1007 -> 805`. A cue-preserving augmentation tweak (`denom_cue_v1`) is
  also killed as a replacement (`0.7307` recall, exact-value `998`); the scalable
  mechanism was fragment-shaped training, not gentler augmentation.
- **Paired target+foreign hardneg6 scenes: promising mechanism, bad first
  layout.** Added background avoid-box support to
  `scripts/build_cashsnap_target_anchor_transplant.py` and manually boxed the
  six vetted foreign hard negatives so target renders can avoid covering the
  unknown note. The smoke (`2/class`) rendered clean paired scenes and matched
  `20/26` target labels under the real teacher when extra foreign-note
  predictions were allowed. The `16/class` diagnostic rendered 208 rows but
  accepted only `148/208`; teacher extras are mostly `KHR_1000` and `KHR_2000`
  on the foreign note, and weak USD target labels are not preserved well
  (`USD_20 4/16`, `USD_5 6/16`, `USD_50 8/16`, `USD_1`/`USD_100 9/16`).
  Visual QA shows many targets pushed against image borders by the avoid
  constraint. Verdict: do not run the fair append training probe on this pack
  unchanged; fix placement margins or build a cleaner unknown-note asset/layout
  generator first.
- **Semantic empty-frame guard: keep.** Added
  `scripts/build_empty_label_semantic_bridge.py` and list-backed probing. On
  likely-true-empty test rows at `conf=0.25`, the scaled baseline fires on
  `230/441` images with `303` detections, while the curated foreign-hardneg6
  model fires on only `17/441` images with `19` detections; val similarly
  improves `313/621` and `404` detections to `17/621` and `18`. This is the
  first clean evidence that target-vs-non-target pressure can fix true empty FPs
  without being credited for suppressing suspect target positives.
- **Teacher pseudo-target bridge: hard reject, useful warning.** Materialized
  221 train-split suspect unlabeled targets with the real-only teacher
  (`data/processed/cashsnap_empty_label_teacher_pseudo_targets_train_v1`) and
  appended them to the strict foreign-hardneg6 best. The anchor was skewed
  (`KHR_20000:80`, `KHR_500:42`, `USD_100:36`, `USD_50:27`, missing `USD_5`,
  `USD_10`, `USD_20`, `KHR_5000`) and failed the fair 1,000-batch
  `yolo26n.pt` A/B: `0.503504 -> 0.339228`, delta `-0.164275`, 10 guard
  failures, worst `KHR_50000 -0.393055`; only `KHR_2000` improved
  (`+0.100166`). It also broke the cleanest empty guard at usable confidence:
  likely-true-empty test `conf=0.25` worsened `17/441` images and `19`
  detections to `159/441` and `196`; val worsened `17/621` and `18` to
  `212/621` and `262`. Do not repeat blind teacher-pseudo positive dosing; use
  the bridge for review/manual relabeling, calibrated adaptation, or
  class-balanced active learning instead.
- **Semantic true-empty backgrounds: hard reject for transfer.** Rendered
  `cashsnap_target_anchor_transplant_poisson_contact_geoscale205_minshort190_bal96_semtrueemptybg_probe_v1`
  with 1,248 balanced positives and 1,061 unique train-split likely-true-empty
  backgrounds, then built the fair `foreignhardneg6` train-list so only the
  background bank differed from the current strict best. Once host RAM freed up,
  the exact b64/1,000-step `yolo26n.pt` A/B failed badly:
  `0.503504 -> 0.332511`, delta `-0.170993`, with 10 guard failures and worst
  `USD_1 -0.476505`, `KHR_5000 -0.449778`, `USD_10 -0.412497`. This kills
  "replace the tiny no-note patch bank with generic likely-empty frames" as a
  big-step path. True-empty FP behavior at `conf=0.25` was only mixed: test
  worsened `29/441 -> 33/441` images with FP and `31 -> 34` detections, while
  val improved `36/621 -> 21/621` and `38 -> 22`; not enough to offset positive
  collapse. It may still be useful only as a background realism/refiner reference
  after positive/camera semantics are controlled.
- **Broad phone-ISP postprocess: reject as a monolithic recipe.** Added
  `--camera-isp-policy phone_isp_v1` to the target-anchor renderer and rendered
  `cashsnap_target_anchor_transplant_poisson_contact_geoscale205_minshort190_bal96_phoneisp_probe_v1`
  plus the fair `foreignhardneg6` train list. The exact 1,000-step A/B scored
  `0.503504 -> 0.446671`, delta `-0.056832`, with six guard failures
  (`KHR_10000`, `KHR_2000`, `KHR_5000`, `USD_1`, `USD_10`, `USD_20`).
  It did help `KHR_50000` (`+0.271769`) and `KHR_20000` (`+0.237197`), so
  camera stress may be useful as class/slice-targeted augmentation, but broad
  full-image ISP noise/blur/JPEG is too destructive for the clean-transfer base.
- **Scaled poisson/contact positives: row budget helps, not enough.** Generated
  diagnostic root
  `data/synthetic/cashsnap_target_anchor_transplant_poisson_contact_geoscale205_minshort190_bal96_scaled_probe_v1`
  with `per_class=96`/1248 rows and registered it in the lifecycle registry.
  Under fair strict `--model yolo26n.pt` fixed-step training with 1,000 batches,
  it reached `0.420709` but still left large weak-class gaps and high
  empty-frame FPs. This falsifies "just scale the repair generator" as the
  base-init fix; the next generator rethink needs real-context/source
  replacement, explicit non-target currency pressure, or teacher filtering
  rather than more of the same rows.
- **Activation microscope v1: useful, but raw domain separability is not the
  target.** Added `scripts/probe_yolo_activation_microscope.py` and ran capped
  current-synth-vs-real probe at
  `runs/cashsnap/activation_microscope_poisson_contact_bal96_realtest_v1`.
  All checkpoints remain highly domain-separable, including the real-trained
  control (mean layer accuracy roughly `0.92-0.93`), so "make real/synth
  inseparable" is not enough. The useful evidence is failure-linked: weak/strict
  synth models still miss hundreds of GT boxes (`tp=74/115` at `conf=0.05`),
  repair jumps to `tp=559` but still collapses `KHR_2000`, `KHR_1000`, and
  `KHR_10000`, and heatmap pairs show wrong nearest synthetic analogs plus
  background/edge/pose evidence. Next microscope work must correlate activation
  cues with misses/FPs, not add more separability tables.
- **Activation failure links: two failure buckets, not one generic domain gap.**
  Added `scripts/summarize_activation_failure_links.py` to join microscope gaps
  with positive-error review. Base-init synthetic models are dominated by USD
  failures (`strict0149` priority: `USD_50`, `USD_5`, `USD_100`, `USD_10`,
  `USD_20`), while repaired/real-control models still expose persistent KHR
  collapses (`KHR_2000`, `KHR_10000`, partly `KHR_1000`). Treat future
  obligations separately: base-init needs broad USD denomination/edge/context
  repair, but the product/real-control path needs focused KHR class repair and
  unknown/FP discipline.
- **Activation microscope and failure bridge: useful mechanism, not promotion.**
  After capped `mpc5`, larger `mpc15`, and old-target-anchor-vs-current-poisson
  synth comparisons, top priority buckets were stable enough to trust: wrinkled
  multi-note USD collages, long thin/partial KHR backs/sides on dark contexts,
  and persistent denomination confusion. Train-analog mining showed
  `KHR_10000` is train-coverable, `KHR_2000` is sparse/distant, and broad USD
  modes need real scene support. A test-guided bridge built from those analogs
  plus 128x2 real empty backgrounds raised strict current synth
  `0.148345 -> 0.224808` and passed background-FP guardrails, but failed
  `KHR_5000`/`USD_1`. Adding a 24/class real floor raised it again to
  `0.305717` and still passed empty-frame FP guardrails, but still failed
  `KHR_5000 -0.160638` and `USD_1 -0.078897`, with FP pressure shifting toward
  `KHR_20000`/`KHR_5000`. A non-test-selected val-guided version kept only a
  small held-out test gain (`0.148345 -> 0.185609`) while still passing
  empty-frame FP guardrails (`358->54` test images with FP) and failing
  `KHR_5000 -0.319418` plus `KHR_1000 -0.072505`. Verdict: keep the microscope
  as a data-selection gate and obligation generator; stop bridge dose tuning.
  Later full real-flat asset extraction also failed, so the next legitimate
  action is a scene/objective rebuild, not another mined-row or asset-bank dose.
- **Real flat-bill cutout mining: useful tooling, tiny-bank transfer rejected.**
  Added staged tooling to mine train-split YOLO bill crops, run local BEN2, fit
  alpha quads, remove boundary-connected black border pixels before/after warp,
  remove alpha islands, rectify to transparent landscape cutouts, and
  audit/label the result. The corrected scan-focused pass used source shape and
  single-box filters first, then individual/checker/render visual QA: 94
  candidates, 90 raw rectified assets, 23 audit suspects, and 17 manually
  accepted scan-like cutouts in `manifest_easy_flat_manual_scanlike_v2.csv`
  (`KHR_500:3`, `KHR_1000:3`, `KHR_2000:3`, `KHR_5000:4`, `KHR_10000:2`,
  `KHR_20000:1`, `USD_20:1`). Rejected hand-held, stock-watermarked, duplicate,
  black-border, hidden-alpha-edge, and speckle/crop-border assets; `USD_10` was
  pruned after checker/render QA exposed an outside-alpha strip. Full-row hybrid
  `manifest_realflat_manual17_plus_numista_missing_v2.csv` with alpha/no-tone
  rendering lost to the scaled poisson/contact reference at 1,000 train batches:
  mAP50-95 `0.420709 -> 0.310841`, worst `KHR_2000 -0.479055`, failures
  `KHR_10000`, `KHR_2000`, `KHR_500`, `KHR_5000`, `KHR_50000`, `USD_10`,
  `USD_20`. Empty-frame FPs improved only modestly at `conf=0.05`
  (val images/detections `953/2031 -> 932/1788`, test `658/1453 -> 620/1318`).
  A later full-coverage 90-asset version with the current poisson/contact/tone
  renderer also failed (`0.503504 -> 0.322847`) and increased no-box misses
  (`129 -> 215`). Verdict: keep extraction/checker QA tooling, but stop tuning
  real-flat assets unless paired with a genuinely different renderer/objective.
- **Source-context and multi-instance replacement: plausible mechanism, unsafe
  artifacts.** They are the strongest representation mechanism so far, but
  source remnants, inpaint scars, label safety, and real-transfer proof still
  block promotion. A real 200-batch tiny-dose probe of the 40-image
  multi-instance notiny package failed badly against a 39-row target-anchor
  control: `0.671194 -> 0.424416` mAP50-95, delta `-0.246778`, with 12/13
  per-class guard failures and worst `KHR_10000 -0.691737`. Treat raw
  multi-instance replacement as recall-suppressive until class-aware teacher
  filters, source-remnant audits, and label-density controls repair it.
- **Auditclean teacher-proxy filtering: cleanup gate, not enough.** Existing
  `auditclean` source-context packs use the clean real model to remove rows with
  unmatched/oversized predictions, but they do not require class-aware teacher
  agreement. Tiny low-batch probes looked positive (`0.007368 -> 0.011406` for
  USD50/100 fallback), yet the fair same-size `i416/b64/steps150` comparison
  lost `0.623422 -> 0.522899`, delta `-0.100523`, with 8/13 per-class failures
  and worst `KHR_2000 -0.305649`. Keep auditclean as a QA filter; do not treat
  it as proof that teacher-filtered row selection works.
- **Class-aware teacher agreement exposes the source-context failure mode.**
  Added `scripts/audit_yolo_teacher_label_agreement.py` to require each
  synthetic label to be detected by the real-clean teacher as the same class and
  to flag extra teacher detections. It now supports
  `--extra-prediction-policy ignore_label_overlap` because most "extra"
  predictions in replacement roots were duplicate teacher boxes overlapping an
  existing label, not true source remnants. On the 40-row balanced-cycle
  multi-instance notiny pack, strict agreement was only `4/40`; `34/40` rows had
  missing or wrong-class label matches. The same context-phone recipe with
  same-source class policy was much healthier (`17/40` strict, `33/40` all
  labels matched), but source-class coverage was badly skewed (`KHR_500:42`,
  `KHR_1000:20`, `KHR_10000:16`, only a few USD/KHR high-value rows). Preserve
  source class first, then solve balance via source inventory, class-aware
  selection, capture, or a bridge; arbitrary balanced-cycle replacement teaches
  visually implausible labels.
- **Source-group geometry/accounting is now explicit, but all-banknote safety is
  still the gate.** Added `scripts/audit_cashsnap_source_group_geometry.py` and
  source-group filters plus `balanced_source_group_class` selection in
  `scripts/build_cashsnap_multi_instance_replacement.py`. The real inventory
  confirms `usd_total`/`billsbank` are the broad USD trouble domains
  (`usd_total` train area p50 `0.745`, `657` boxes `>=0.90`; `billsbank` p50
  `0.468`) while `asian_currency`/`cashcountingxl` supply most label-empty
  pressure. A 24-row USD trouble-source smoke balanced `2` rows per USD class per
  source group, but clean-teacher acceptance was only `18/24` with
  `ignore_label_overlap` because `6` rows had true extra detections; visual QA
  also exposed unaccounted source notes/watermarks. The full real-train
  `usd_total`/`billsbank` clean-teacher prefilter accepted `3831/4965`;
  `billsbank` was cleaner (`2378/2859`) than `usd_total` (`1453/2106`), with
  rejections led by extra detections on large `USD_50`/`USD_100` rows. Accepted
  source rows are still plentiful for each USD class/source group. Rerendering
  the same 24-row smoke from the accepted-list manifest improved generated-row
  teacher acceptance `18/24 -> 23/24`; keep that source-manifest gate, but still
  require visual/source-shape QA because some accepted rows are compound
  front/back banknote displays. Do not scale source-group replacement until every
  detectable banknote is labeled, removed, or explicitly routed as unknown.
- **Accepted-source USD trouble-context replacement is rejected as the final
  synth-only detector path.** A medium support pack balanced `16` rows per USD
  class per source group (`192` rows) and passed the clean-teacher gate at
  `184/192`; a teacher-accepted same-budget mix used `29` support rows plus `67`
  strict rows per USD class, preserving KHR and hard negatives. The fair
  1,000-step `yolo26n.pt` A/B failed full real test mAP50-95
  `0.503504 -> 0.467260`, with guard failures on `KHR_50000`, `USD_10`,
  `KHR_1000`, `USD_20`, and `KHR_500`. Product audit confirmed it is not a
  hidden win: post-reclassifier recall fell `0.7809 -> 0.5973` and no-box FNs
  rose `129 -> 292`. It did improve `usd_total` localization
  `0.7559 -> 0.8071`, but destroyed `billsbank` localization
  `0.8396 -> 0.3239`; source-context support must not be scaled without a
  source-family-specific design and KHR protection. A `usd_total`-only
  `mix81_15` isolate had fair 1,000-step b64 preflight parity, but training hit
  the RAM guard at epoch `2/50` and produced no manifest/summary. A corrected
  memory-safe `yolo26n.pt` b48/e38 A/B completed with row/phase parity and a
  small aggregate lift (`0.497016 -> 0.513584`), but it still fails promotion:
  `KHR_10000` drops `-0.064772`, `KHR_50000` collapses `-0.343211`, and
  low-conf background-FP images worsen `516/748 -> 569/748` despite recall
  improving to `0.7197`. The product stack is also only a clue:
  gate+reclassifier recall/precision/exact-value becomes
  `0.8140`/`0.4419`/`1010` versus strictbest stack
  `0.7809`/`0.4620`/`1021`. Source split shows the trade: `usd_total` recall
  improves `0.4921 -> 0.7165` and `billsbank` `0.6038 -> 0.6792`, but
  `khmer_us_currency` recall drops `0.9383 -> 0.8333` and empty-source overfire
  worsens (`asian_currency` FP `261 -> 304`, `cashcountingxl` background-FP
  images `308 -> 361`). Treat it as a USD/source-context clue, not a final
  detector.
- **Single-box phone-context replacement: direct path rejected, audits useful.**
  Multi-note/fanned source-context rows fight the non-overlap base phase: the
  balanced same-class min1/minshort80 probe kept only `52/78` rows under
  overlap-duplicate-exempt teacher agreement and lost `KHR_1000` coverage
  because source rows still had occluded/fanned labels. Single-box phone
  inventory at minshort80 has `1813` rows and all 13 classes, but rare classes
  cap a no-reuse balanced set (`USD_20:8`, `KHR_20000:9`, `KHR_50000:9`). The
  rendered bal8 single-box probe is exactly balanced (`104` labels) and accepts
  `83/104` rows under the overlap-duplicate-exempt audit while preserving all
  classes, but `USD_20` keeps only `1/8`; the same clean teacher accepts only
  `2/8` of the original real USD20 source photos, so do not use teacher
  agreement as a hard USD20 label filter. Scaling with source reuse plus the six
  vetted foreign hard negatives failed the fair strict-base test:
  `0.503504 -> 0.186959`, delta `-0.316545`, with all classes dropping beyond
  guard (`USD_10` worst at `-0.559228`; `KHR_500` still best candidate class at
  `0.647075`). The 17 unmatched-teacher source-remnant filter and reuse reports
  are good QA, but replacement-in-real-phone-context is not a transfer fix by
  itself.
- **Reduced mosaic: curriculum clue only.** `mosaic=0.75` improved small
  bounded-real behavior, but class/threshold guards still failed.
- **Broad stat matching, strict geometry matching, and accepted-blend polishing:
  not enough.** These improved proxies and contact sheets but did not prove real
  transfer. Do not revisit without a specific failure mechanism.
- **Clean-context local-tone: useful audit knob, transfer path killed.** Added
  bounded sampling to `scripts/audit_yolo_crop_visual_domain_gap.py`, then
  tested WebGL clean-context tone policies against sampled real-train crops.
  Off/default failed on low high-tail luma (`luma_p95 -0.113730`), bridge-tone
  overcorrected (`luma_std +0.100`, `luma_p05 -0.164`,
  `saturation_std +0.087`), and local-tone passed both smoke and 64-row gates
  (`luma_std +0.009`, `luma_p05 -0.049`, `luma_p95 -0.041`,
  `saturation_std +0.011`). The transfer result killed it anyway:
  `yolo26n.pt` pure-synth local-tone64 scored val/test mAP50-95
  `0.023897`/`0.026258`. Keep the crop audit and tone policy as QA/harness
  tools; do not run more clean-context toy-world tone sweeps.
- **Naive real+synthetic mixing: product-relevant, but not proof by itself.**
  Real data is the backbone for any practical model, but prior full
  real+accepted-WebGL runs did not beat the clean real checkpoint and hurt
  protected high-value KHR. Future hybrid work must improve a real baseline, not
  merely exceed weak synthetic-only scores.
- **Stylized, mined, and unknown hard-negative rows: diagnostic, not a fix.**
  Blunt zero-label pressure can be too easy or suppressive. FP-mined real doses
  gave tiny aggregate deltas (`+0.001666` for spread-8, `+0.000180` for top-25)
  but failed lower-confidence scorecards on per-class recall; top-25 also lost
  recall at `conf=0.01`. New dark/spread WebGL unknown roots are visual-gap
  diagnostics, not trainable paired target/unknown curricula yet.
- **Overlap/fragment/two-stage detectors: not current clean-base work.** They
  improve some mined-real fan/overlap recall but overcount and hallucinate more.
  Keep them archived until clean-visible transfer is credible.
- **Refiner/editor outputs: harness lesson, not trainable data yet.** SD-Turbo
  note-edge locking proved label-preservation mechanics can work, and
  `runs/cashsnap/refiner_readiness_poisson_contact_bal96_scaled_smoke_v1`
  now provides a balanced 65-row mask/manifest smoke pack with real-target and
  background references. On the teacher-passable 49-row subset, raw SD-Turbo
  `strength=0.35/steps=8` failed label preservation on all 49 rows, while
  note-edge locking passed pixel preservation and background-realism stats but
  dropped teacher agreement from source `49/49` to refined `46/49`
  (`USD_5` flipped toward `KHR_1000`, `KHR_20000` flipped toward
  `KHR_5000`/`KHR_500`, and one `KHR_5000` disappeared). Visual QA showed mostly
  tone/blur smoothing with risky edge/context changes, not a regime-changing
  camera-domain fix. Do not train or scale this SD-Turbo setting; keep the
  manifest filter, note-edge lock, teacher audit, and visual-sheet harness as
  gates for any more controlled refiner.

### Untested Ideas

Do not preserve small probes here just because they are reasonable. A worthwhile
untested idea should plausibly change the transfer regime, expose why the current
harness is misleading, or close a large part of the current `+0.32` to `+0.35`
gap. Otherwise it is a tactic, not the research frame.

- **Manual/class-balanced real-label bridge, not blind pseudo rows.** The
  semantic split is now proven useful as a validator, and blind teacher-positive
  dosing is dead. The next real-data bridge must either manually relabel a small
  class-balanced subset of suspect targets/currency-review rows or use a proper
  adaptation recipe with confidence calibration and class balance. Kill it if it
  improves suspect-target/currency-review proxies while true-empty FPs or weak
  target classes degrade.
- **Real capture/validation bridge as the next big missing signal.** The project
  may be stuck because it is trying to infer the missing retail domain from
  synthetic tweaks and weak external bridges. Capturing or labeling mixed
  USD+KHR stacks, hard `KHR_50000`, `KHR_5000/KHR_20000` thin slices,
  same-denomination fans, and no-note paper props could reveal whether the real
  blocker is camera/context/background/label policy rather than another renderer
  parameter. If this only produces another tiny slice with no new failure
  separation, it is not enough.
- **Real-context, unknown-aware synthetic rebuild.** Another row dose, texture
  polish, or contact-sheet fix is unlikely to move the strict base-init yardstick
  from `0.503504` toward `0.82-0.85`. The serious synthetic bet is no longer
  clean target-anchor geometry; the failed USD extent probe showed localization
  can improve while class/background behavior worsens. A serious rebuild must
  account for all banknotes in a scene, preserve or explicitly remove/source-label
  source notes, represent unknown/foreign notes instead of treating them as
  accidental clutter, fit camera/ISP variation from real train photos, and protect
  KHR while repairing broad USD source modes. The first required gate is an
  all-banknotes-accounted source prefilter using source-group geometry plus
  clean-teacher extra-detection audits; any row with unaccounted detectable notes
  must be rejected, relabeled, or moved into an explicit unknown objective. Kill
  it if teacher/proxy gains do not improve real recall and empty-frame FPs
  together.
- **Class/source-aware obligation objective, not row-dose repair.** The
  `0.589059-0.594999` obligation family proves unknown pressure can fix false
  positives, but simple negative-dose changes, appended repair positives, and
  positive-only staging are now killed. Broad direct14 unknown-prop labels are
  also killed as a row objective after the phase-matched 510 collapse. The next
  test must separate target recall from unknown rejection structurally:
  class/source sampling, loss weighting, a trained proposal gate, or a
  validation-driven curriculum. Kill it quickly if background FP suppression
  disappears, `KHR_5000`/`KHR_10000`/weak USD still fail, or the only win is
  another small aggregate bump.
- **Controlled label-preserving refiner, only if it beats the SD-Turbo smoke.**
  A useful refiner must make an obvious camera/context-domain improvement while
  preserving teacher agreement on source-accepted rows; pixel preservation and
  background stats alone are not enough. Prefer a controlled/trained refiner or
  a sharper semantic gate over another blind SD-Turbo dose. Kill it if the
  change is cosmetic, if teacher agreement drops, or if the only safe setting is
  too weak to plausibly close the `+0.32` strict-base gap.
- **Unknown-aware counting architecture branch.** Direct 13/14-class detection
  is not the only viable product shape. The narrow repeated unknown-class view
  failed, broad direct14 labels collapsed under phase matching, and a naive
  two-model box union reintroduced FPs. The current best binary proposal gate
  proves a source-aware second stage can cut real background proposals with no
  measured recall loss, while naive current-domain target-vs-reject and
  banknote/background retrains over-reject true positives. The serious next
  version is a reviewed/source-balanced banknote/background/unknown proposal
  gate or equivalent detector objective trained from strict-best proposals,
  true-empty rows, vetted foreign-note/unknown rows, and protected weak-class
  positives. This is a product-architecture pivot, not a clean-base shortcut;
  judge it on real count/value errors, weak-class protection, unknown routing,
  and empty-frame behavior, not only mAP.
- **Real-flat source bank tuning is killed.** Both the tiny manual hybrid and
  the later class-diverse 90-asset bank failed transfer under target-anchor
  rendering. Only revisit if the proposal is a different scene/objective
  mechanism, not more extracted flat assets.

Small supporting tactics, not big ideas: failure-led obligation sets,
train-side mined-real near-negatives, audited source-context replacement,
multi-instance replacement, convergence-control sweeps, class-aware teacher row
filters, crop visual-gap gates, and camera/ISP/tone ablations. Use them only if
they serve one of the big questions above. For near-negatives, require paired
target/unknown positives and negatives or a calibrated loss/sampling plan before
another zero-label dose. Do not run another blunt zero-label pack or mined-row
bridge until target vs non-target vs true-empty semantics are explicit.

## Promotion Gates

A synthetic axis is credible only when it improves or preserves:
- full real val/test;
- clean-visible val/test;
- labeled-positive and geometry-stress slices;
- protected classes, especially riel and high-value notes;
- real empty-frame FP detections and images-with-FP at `conf=0.05`,
  `imgsz=416`, `batch=1`, `device=0`;
- max per-class mAP50-95 drop `<=0.05`, unless explicitly waived;
- at least one seed repeat for serious promotion, more for large claims.

Synthetic package gates are necessary filters, not promotion authority. Self-eval
preservation is not enough. For low-memory probes, use the lightweight transfer
scorecard over multiple confidence thresholds and require no recall regression
plus no FP/background regression.

Clean-base can move toward overlap/fan/hand only when synthetic-only `yolo26n`
is near the target line, clean-visible and labeled-positive test are `>=0.75`,
protected riel passes, real-empty FPs are no worse than control, and the result
survives seed repeat or a slow-promotion run.

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
  same-denomination fan, KHR_5000/KHR_20000 thin slices, KHR_5000 face/number
  overlap, KHR_50000 hard positives, mixed USD+KHR stacks, no-note backgrounds,
  and non-banknote paper props.

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
