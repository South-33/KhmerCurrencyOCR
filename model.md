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
because clean-real controls already prove the evaluator and model capacity can
reach the zone: a near-size real-trained control reached `0.819153`, and the
later high clean-real checkpoint reached `0.883801`.

Current strict synthetic-only best:
- Detector:
  `runs/cashsnap/fixed_step_scaled_foreignhardneg6_from_yolo26n_e50_i416_b64_w0_auto_lr1e2_warmup3_amp_cachefalse_steps1000_seed0/weights/best.pt`
- Full real test mAP50-95: `0.5035036831091516`.
- Gap to target: `+0.316496` to `0.82`, `+0.346496` to `0.85`.
- Evidence bundle:
  `runs/cashsnap/final_synth_only_nonoverlap_phase_evidence_v1.json`.

What this proves: target-anchor scale/contact rendering plus six vetted
foreign-note hard negatives can move strict `yolo26n.pt` synthetic-only transfer
above the old floor (`0.420709 -> 0.503504`) while passing the current
per-class guard. It is a real mechanism clue, not a solved model.

What this does not prove: the detector is not close to the clean-base target and
not product-ready. At `conf=0.05`, the strictbest lightweight eval still has
recall `0.6438`, precision `0.2321`, and background FPs on `516/748`
empty-label test images. Positive-only transfer is much better than older
blend185/hardnegold8 results (`0.696549` clean-visible, `0.610140` labeled-all),
but the aggregate full real test is still far short.

Current best product clue: strictbest proposals plus the true-empty proposal
gate plus the fragment-trained denomination reclassifier reach
recall/precision/exact-value `0.7809`/`0.4620`/`1021` on real test, with the
important caveat that the reclassifier uses a small real labeled crop anchor.
Treat this as an architecture clue, not proof that single-stage synthetic
detection has reached target.

Distance to target is still two tracks:
- Clean-checkpoint synthetic-data repair is much closer (`0.747316` seed0) but
  guard-failing and not seed-repeated.
- Strict base-init generation remains roughly `+0.32` to `+0.35` away for
  guard-passing models. Gains above `0.503504` are useful only when they reveal
  a scalable mechanism and protect weak/high-value classes.

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

### Current State

The final synth-only non-overlap phase result is chosen: strictbest
foreign-hardneg6 remains the best verified guardrailed detector. The updated
handoff is `runs/cashsnap/final_synth_only_nonoverlap_handoff_ready_v1.md`.

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
  lacks seed repeat. It is closer to the target line but not a release claim.
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

### Untested Ideas

Do not preserve small probes here just because they are reasonable. A worthwhile
untested idea should plausibly change the transfer regime, expose why the current
harness is misleading, or close a large part of the current `+0.32` to `+0.35`
strict-base gap. Otherwise it is a tactic, not the research frame.

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
