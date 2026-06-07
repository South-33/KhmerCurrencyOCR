# CashSnap Model Brain

This is the active working memory for model and synthetic-data decisions. Keep
it compact, current, and decision-oriented. Move old evidence to
`docs/archive/` instead of letting this file become a changelog.

Archives:
- `docs/archive/model_brain_full_history_2026-06-06.md`
- `docs/archive/model_brain_pre_compact_2026-06-07.md`
- `docs/archive/model_brain_pre_cleanup_2026-06-07.md`

## North Star

Build a small phone/browser-deployable model that counts mixed USD and Khmer
riel from one casual retail photo.

Synthetic transfer is the current bottleneck. Do not move seriously into
overlap/fan/hand until a synthetic-only `yolo26n` can learn normal visible notes
well enough to transfer to real photos.

Hard product requirement later: partial notes, thin edges, overlaps, fans,
stacks, and hand occlusions must count when a human can see denomination
evidence. Counterfeit/authenticity detection is out of scope.

Clean-base target: `0.82-0.85` full real test mAP50-95.

Why that target is real: the near-same-size real-trained `p96_bg96` probe
reaches `0.819153` on full real test and `0.866141` on the 185-image synthetic
blend. If synthetic data is truly camera-real and broad enough, synthetic-only
training should approach that transfer scale.

Current reality:
- Slow synthetic-only filtered185 full real test: `0.142098`.
- Fast fixed-step filtered185 full real test: `0.075907`.
- Current synthetic-only fixed-step aggregate leader: target-anchor
  latest-design transplant `0.144740`.
- Strongest current low-batch mechanism clue: isolated Poisson/contact
  target-anchor `0.028350 -> 0.042401` in the quiet b8/AMP 150-update A/B, but
  it fails empty-frame FP guardrails.

Perspective: we are around `9-14` trying to reach `82-85`, not `80` trying to
reach `85`. Small `+0.01` or `+0.02` wins below `0.25` are clues about a
mechanism, not a trajectory toward done. Stay in step-change mode.

## Current Decision

Stop iterating small negative doses around the same Poisson/contact candidate.
The best next synth-data bet is no longer "prettier background/refiner" alone.
It is an obligation-driven sim-to-real rebuild:

1. Keep Poisson/contact as the current image-formation base.
2. Drive new recipes from real failure clusters and detector-representation
   gaps, not from renderer knobs that only improve proxy stats.
3. Add a bounded label-preserving learned/AI refiner only when it reduces the
   representation gap and transfer failures, not just visual surface style.
4. Build or source a realistic target-domain near-negative bank, not more
   stylized WebGL line-prop negatives.
5. Promote only through real-transfer, background-FP, per-class, and seed-repeat
   guardrails.

Refiner contract: raw learned outputs are not trainable data. Any learned/AI
refiner output must pass through protected composition first, with note+edge
pixels restored from the trusted source unless a stricter label-preservation
gate proves another policy safe.

The right first learned-refiner path is `FastCUT/CUT` as the low-VRAM sanity
check, then `CycleGAN-Turbo/img2img-turbo` if the harness and memory fit. Use
SD1.5 ControlNet/img2img only as a low-denoise baseline. Do not start with
Qwen/FLUX prompt editing: currency labels and security details are too easy to
mutate, and this laptop has only an RTX 4060 Laptop 8GB VRAM with about 16GB RAM.

Do not blindly scale synthetic yet. Scaling target-anchor latest, realfgstyle,
poseclose, SD-Turbo note-edge, or the existing negative roots as-is is very
unlikely to close the gap to `0.82-0.85`.

Representation-gap read:
- The current `0.144740` synthetic leader is still easy to separate from real
  positives after class-balanced sampling: layer `0` domain accuracy `0.823`,
  layer `8` `0.950`, layer `22` `0.938`.
- Source Poisson/contact m260 is worse through a real-trained lens: layer `0`
  `0.869`, layer `8` `0.977`, layer `22` `0.950`.
- SD-Turbo note-edge did not reduce the gap; it was equal or worse than source
  m260 in most layers (`0.869/0.981/0.965` at layers `0/8/22`).
- Research-grounded diagnostics now available: proxy A-distance/domain
  classifier, MMD, and CORAL-style covariance gap. Treat them as warning
  lights tied to real errors, not standalone promotion metrics.
- Top uncovered real modes are multi-note/scan-like bills, tiny notes in deep
  clutter, flat full-frame rotated note captures, weak KHR back/rare-class
  views, and realistic empty-frame near-negatives. The current generator
  over-represents one-note-on-surface scenes.

Current refiner-readiness status:
- `scripts/build_refiner_readiness_pack.py` prepares the first controlled
  synthetic-to-real refiner smoke pack from isolated Poisson/contact.
- Pack: `runs/cashsnap/refiner_readiness_poisson_contact_v1/`
- It exports note masks, interior/detail lock masks, edge freedom masks,
  train-only real target manifests, strict no-note background manifests, CUDA
  readiness, and 95% headroom-wrapper guidance.
- `scripts/check_refiner_label_preservation.py` is the pre-trainable gate for
  refined outputs. Identity check against the original source images passes:
  `runs/cashsnap/refiner_readiness_poisson_contact_v1/identity_preservation_check.json`.
- `scripts/materialize_refiner_unaligned_dataset.py` turns that pack into a
  CUT/FastCUT-style smoke dataset with `trainA`, `trainB`, labels, and masks.
  Current dataset:
  `runs/cashsnap/refiner_readiness_poisson_contact_v1/cut_unaligned_smoke/`.
  It has 52 synthetic `trainA`/`train_A` rows, 160 train-only real
  `trainB`/`train_B` rows, 8 train-only `test_A`/`test_B` rows for refiner
  smoke validation, fixed CycleGAN-Turbo prompts, and its own identity
  preservation check passes.
- CUT/FastCUT repo was cloned into ignored
  `.cache_runtime/third_party/contrastive-unpaired-translation`; dependencies
  are installed after downgrading `setuptools` to restore `pkg_resources` for
  `visdom` on Python 3.14. The cached CUT argparse bug for `--CUT_mode` was
  patched locally.
- Tiny FastCUT GPU/headroom proof succeeded:
  `runs/cashsnap/refiner_checkpoints/cashsnap_fastcut_poisson_smoke_e1_m8_128/`.
  It used CUDA `gpu_ids=0`, 8 images, 1 epoch, 128px, batch 1, smaller
  `resnet_6blocks/ngf32/ndf32`, no flip, no HTML/visdom, and checkpoints under
  `runs/cashsnap`.
- Shape-preserving inference succeeded:
  `runs/cashsnap/refiner_results/cashsnap_fastcut_poisson_smoke_e1_m8_128/train_latest/images/fake_B/`
  emits 640x640 outputs. It is not a usable refiner candidate: label
  preservation on first 8 fails all rows, with detail L1 mean `51.104` and max
  `88.984`.
- `scripts/apply_refiner_detail_lock.py` can restore source note pixels after
  refiner inference. Detail-lock recomposite cuts detail L1 to mean `0.692` but
  still fails 4/8 on edge/note thresholds. Full-note lock gives detail/note L1
  `0.0` but still has one edge-only failure and visual output remains poor on
  the toy run.
- Cached CUT now has a local `--lambda_detail_identity` mask-aware training
  loss wired from `masks/detail_lock`. A tiny FastCUT 8-image smoke at lambda
  `10` still failed all rows (`detail_l1` mean `58.064`); a stronger lambda
  `200`, 5-epoch smoke also failed all rows (`detail_l1` mean `52.125`). Read:
  mask-aware loss is wired but not enough to trust raw FastCUT outputs at toy
  scale.
- `scripts/apply_refiner_detail_lock.py --lock-mask note_edge --feather-px 0`
  restores the note plus contact edge band after refiner inference. On the
  stronger FastCUT smoke it passes the strict first-8 preservation gate with
  `0/8` failures and detail/note/edge L1 all `0.0`. This is the current safe
  refiner contract: learned model may alter background/broad tone, but protected
  note and edge pixels come from source unless a future gate proves relaxation
  safe.
- A 52-image note-edge-locked FastCUT candidate was materialized as a normal
  YOLO root:
  `runs/cashsnap/refiner_yolo_candidates/fastcut_maskid200_note_edge_e5_m52_128/`.
  It passes strict preservation (`0/52` failures), `check_yolo_dataset.py`, and
  numeric visual QA, but human sheet review rejects it: backgrounds collapse
  into gray/green/purple translator artifacts.
- `scripts/check_refiner_background_realism.py` now compares candidate
  background pixels outside note+edge masks to strict no-note CashSnap
  backgrounds. The original Poisson/contact `trainA` source passes; the
  note-edge-locked FastCUT candidate fails with `luma_std_ratio=0.176884`
  against the default `>=0.45` guard. Do not promote this FastCUT candidate.
- `scripts/run_sd_turbo_img2img_refiner.py` is the first local AI-image-refiner
  proof path. It runs `stabilityai/sd-turbo` img2img on CUDA/fp16 and refuses
  CPU fallback. Low denoise `strength=0.25`, `steps=4`, `guidance=0` is one
  effective denoise step on this setup.
- Raw SD-Turbo is not label-safe: on the 260-image audit pack it failed
  preservation on `257/260` rows (`detail_l1` mean `22.012`, `note_l1` mean
  `20.450`). The prompt does not reliably preserve denomination detail.
- Note-edge-locked SD-Turbo is the first viable AI-refiner candidate branch:
  `runs/cashsnap/refiner_yolo_candidates/sd_turbo_note_edge_s025_steps4_m260/`
  has 260 images, 20 per class, passes strict preservation (`0/260` failures),
  background realism (`luma_std_ratio=1.060475`, no violations),
  `check_yolo_dataset.py`, visual QA numeric gates, and composite-edge audit
  (`boundary_ratio_mean=1.2708`, `edge_color_step_mean=0.0759`).
- Full-size visual QA on SD-Turbo m260 is mixed: several clean-counter images
  are plausible, but some outputs have pastel/blocky AI background patches,
  washed/haloed notes from the hard edge lock, or bad source backgrounds such as
  speckled black patches. Treat it as a promising audit candidate, not a
  trainable/done package.
- Model-side b4 fixed-step source-vs-SD m260 same-root learnability rejected
  this exact SD setting: source `0.736527`, SD-Turbo `0.672436`, delta
  `-0.064091`, with 7 classes worse than the `0.05` per-class guard and worst
  `KHR_10000 -0.176763`. The b8 run hit the 95% RAM guard on the SD candidate,
  so use b4 for this bounded comparison. This is not real transfer proof; it is
  an early "do not promote as-is" signal.
- Better-model research branch to keep current: test ICEdit, Qwen-Image-Edit,
  FLUX.1/FLUX.2 editing variants, Step1X-Edit, and OmniGen2 only through this
  same preservation-first harness. Do not trust model-card claims of detail
  preservation; raw output must pass masks/OCR/detail checks or be used only
  with note+edge recomposition.
- CycleGAN-Turbo/img2img-turbo repo was cloned into ignored
  `.cache_runtime/third_party/img2img-turbo`. Core deps now import
  (`accelerate`, `diffusers`, `transformers`, `peft`, `lpips`, `clean-fid`,
  `vision_aided_loss`, `wandb`, `open_clip`); `xformers` is still absent and
  must not be enabled.
- Cached CycleGAN-Turbo patches for laptop smoke: `--report_to none`, no
  validation/FID/DINO when `--validation_steps 0 --validation_num_images 0`,
  skip LPIPS/VGG when LPIPS weights are zero, and add `resize_128/resize_192`.
- CycleGAN-Turbo 256px one-step crossed the guard and was killed at GPU memory
  `96%`; do not use 256px default on this laptop yet.
- Optimized CycleGAN-Turbo 128px one-step completed under headroom with
  batch `1`, workers `0`, low LoRA rank, no validation, no LPIPS:
  `runs/cashsnap/cyclegan_turbo_smoke_s1_128_noval_nolpips/`. It did not save a
  checkpoint because the upstream save condition misses step 1 when
  `checkpointing_steps=1`.
- Rerunning the 128px smoke with a step-1 checkpoint request hit the 95% GPU
  memory guard during setup. Treat CycleGAN-Turbo as barely fitting: usable for
  tiny proof runs only until the harness gets a real lower-memory/autocast or
  offload path.
- An fp16 attempt failed with `Attempting to unscale FP16 gradients`; do not use
  `--mixed_precision fp16` for this cached unpaired trainer until the loop is
  rewritten around proper autocast instead of casting trainable weights.
- Latest hardware read inside the pack: CUDA OK on `NVIDIA GeForce RTX 4060
  Laptop GPU`; low available RAM warned and recommended `256px`, batch `1`,
  fp16/AMP, workers `0`. Use `scripts/run_with_headroom.py` for long refiner
  training/inference so CPU/RAM/GPU-memory caps stay at or below 95%.

## Current Read

- Renderer mechanics and package QA are strong enough for bounded probes, but
  renderer gates are not transfer proof.
- The current `0.144740` synthetic leader misses `669/817` real-test GT boxes
  at `conf=0.05` (`81.88%` miss rate) and still fires on `174/748` empty test
  images (`292` detections). This is not close; it is a distribution failure.
- The detector can identify synthetic-vs-real positives from internal
  activations even after class balancing. The gap appears immediately at early
  layers, so camera/image-formation statistics and context modes are major
  suspects, not only late denomination semantics.
- The main blocker is real transfer: target-domain foreground/camera pixels,
  pasted-edge/contact/focus consistency, realistic target-vs-non-target
  banknote pressure, rare-class/capture gaps, and validation obligations tied to
  real failures.
- Target-anchor latest was the first clear recent mechanism signal because it
  combines real CashSnap no-note pixels, real CashSnap geometry, and
  latest-design target assets. It is a direction signal, not a solved package.
- Latest-design texture policy matters. The all-manifest target-anchor MVP
  improved aggregate but failed class guards, likely by mixing old/current class
  appearances.
- Poisson/contact image formation is useful. Isolated Poisson/contact improved
  low-batch positive transfer, but it increased real-empty false positives.
- Close-up pose/scale did not transfer. It reduced empty-frame FPs but broke
  positive transfer, especially `KHR_10000`.
- Existing synthetic negative pools are too stylized/easy. At `conf=0.05`, the
  Poisson/contact b8 model had zero FPs on the current unknown/hard-negative
  synthetic roots; they do not match real empty-frame failures.
- Full CashSnap empty-label frames are hard negatives, not clean positive
  canvases. Many contain visible foreign/unknown/target-like notes. Use strict
  no-note patches for positive transplants; use full empty frames for FP
  diagnostics and target-vs-non-target pressure.
- Bridge-calibrated selected-v1 proved that prettier summary stats can still
  train a worse detector. Domain separators and visual-gap audits are warning
  lights, not judges.
- Missing promoted real fan/overlap/hand stress labels still blocks final proof
  for the ultimate product behavior.
- Before class-scope claims, run `scripts/check_currency_taxonomy_coverage.py`.

## Latest Scoreboard

| Candidate | Result | Read |
| --- | --- | --- |
| Fast filtered185 control | `0.075907` full real test | Fixed-step reference. |
| Target-anchor latest | `0.144740`; passes per-class guard vs fast filtered185 | Current aggregate leader and seed distribution. |
| Target-anchor all-manifest MVP | `0.119088`; fails `KHR_5000`, `USD_10` | Rejected. Latest-design policy is important. |
| Realfgstyle v1 | Dataset/crop stats mixed; b64/b32/b16 runs hit RAM | Diagnostic only. Visual QA still shows crisp sticker-like notes. |
| Isolated Poisson/contact | b8 A/B `0.028350 -> 0.042401`; no class guard failures | Useful compositor clue. Fails empty-frame FPs: val `88/59 -> 168/100`, test `61/38 -> 72/48`. |
| Poisson/contact poseclose | b8 A/B `0.028350 -> 0.027729`; `KHR_10000 -0.0546` | Rejected as replacement. Empty FPs improved but positive transfer failed. |
| Poisson/contact hardnegdiv8 | b8 A/B `0.028350 -> 0.026388` | Rejected. Stylized WebGL negatives worsened FPs badly. |
| Poisson/contact realbgneg25 | b8 A/B `0.028350 -> 0.030660` | Rejected. Small positive gain, but real-empty FPs worsened badly. |
| Poisson/contact unknownsoftfp8lowconf | b8 A/B `0.028350 -> 0.031460` | Least-bad negative dose; test FPs improved but val FPs worsened. Not promotion. |
| Target-anchor + repairmix50 | b32 A/B `0.071116 -> 0.042483` | Rejected. Naive fusion collapsed precision. |
| Bridge-calibrated selected-v1 | CashSnap `0.075907 -> 0.048918`; bridge `0.223646/0.225793 -> 0.126261/0.121071` | Rejected. Zero USD recall on bridge review. |

Key isolated Poisson/contact details:
- Config: `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_puresynth_realval_v1.yaml`
- Root: `data/synthetic/cashsnap_target_anchor_transplant_poisson_contact_v1/`
- Build: 1,248 images, 96/class, all 26 latest-design assets, all 25 strict
  no-note train backgrounds, 1,134 unique geometry sources.
- Visual QA: `p50_short_px=150.8`, `p05=83.2`, tiny `0.0008`, small `0.0393`,
  soft `0.1002`.
- Edge audit vs old realfgstyle: boundary ratio `1.2549`, color step `0.0761`,
  quad mean `365.6x245.9`, area `73.4k`.
- Model read: useful image formation, not promotable without realistic negative
  pressure and stronger proof.

## Next Bet: Representation-Guided Synthesis

Goal: rebuild the synthetic curriculum around real modes that the detector says
are uncovered, then prove the new branch reduces both real errors and
representation separability.

Immediate obligations from the current leader:
- Broad positive recall: `669/817` missed real-test GTs at `conf=0.05`.
- Weak classes: `USD_5`, `USD_50`, `USD_100`, `KHR_1000`, `KHR_2000`,
  `KHR_5000`, `KHR_10000`, `KHR_20000`.
- Empty-frame pressure: `174/748` empty test images with FPs, dominated by
  `USD_20`, `USD_10`, `KHR_50000`, `KHR_2000`, `KHR_500`, and `USD_1`.
- Image-formation modes: multi-note scan-like bills, tiny notes in deep clutter,
  flat full-frame rotated captures, weak KHR backs, and rare-class views.

Next engineering move:
1. Turn the inpaint-context mechanism signal into a cleaner generator by
   coupling each background to the same train-anchor geometry it came from, so
   the synthetic note covers the erased real note region instead of landing on a
   random inpaint scar.
2. Then rerun representation-gap, positive-error, background-FP, and a clean
   fixed-step A/B under lower memory pressure before any scale-up.
3. Keep the analog-geometry branch as evidence that geometry alone is not
   enough; context/image formation is the active bottleneck.

Train-only analog mining status:
- `scripts/mine_representation_gap_train_analogs.py` maps the top uncovered
  real-test representation rows to same-class train-split anchors.
- Current run:
  `runs/cashsnap/representation_gap_synthleader_train_analogs_v1/summary.json`.
- It found `151` unique train anchors for `40` uncovered test queries from a
  `2,813`-image train-positive candidate pool, with visible analogs for
  scan-like/multi-note USD/KHR, flat rotated notes, hand-held views, tiny dark
  background shots, KHR backs, and weak rare-class views.
- Use these train anchors as the next generator target/geometry context source;
  do not use the uncovered test images themselves for training data.

Targeted branch status:
- Analog-geometry branch:
  `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_analogs_v1/`.
  Dataset, visual QA, and edge audit pass, but representation gap is mixed:
  early layers got worse (`layer0 0.823 -> 0.877`) while late layer improved
  (`layer22 0.938 -> 0.904`). Read: geometry helps semantics but strict
  no-note patch contexts still scream synthetic.
- Inpaint-context branch:
  `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_inpaintctx_v1/`.
  Built from `58` filtered train-only inpainted anchor backgrounds and the same
  train-anchor geometry. Dataset and numeric visual QA pass; edge color step is
  higher (`0.0996`), and full-size review shows some inpaint seams/halos.
- Inpaint-context representation result is the first strong mechanism signal:
  layer `0` domain accuracy `0.823 -> 0.808`, layer `1` `0.877 -> 0.823`,
  layer `22` `0.938 -> 0.862`, and late MMD `0.0992 -> 0.0414`. Read: real
  train-context canvases directly attack the measured image-formation gap.
- Coupled-context branch:
  `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_couplectx_v1/`.
  It reuses each inpainted background's source-image geometry so the synthetic
  note covers the erased real note region. Dataset/QA pass, but edge boundary
  ratio is high (`1.4016`) and representation is mixed: early layers improve
  strongly (`layer0 0.808 -> 0.773`, MMD `0.1050 -> 0.0624` versus inpaintctx),
  while late domain accuracy worsens (`layer22 0.862 -> 0.888`). Read: coupling
  reduces camera/context distance but exposes a note-edge/presentation shortcut.
- Strict inpaint filtering (`max_mask_fraction=0.35`) is rejected. It kept only
  `32` contexts, collapsed class-specific source coverage, worsened edge audit
  (`boundary_ratio=1.5176`, `color_step=0.1122`), and worsened representation
  separation (`layer22=0.9538`).
- Alpha-feathered coupled branch:
  `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_couplectx_feather08_v1/`.
  MMD/centroids improve further and `layer19` domain accuracy reaches `0.846`,
  but edge boundary ratio rises (`1.6120`) and `layer22` domain accuracy worsens
  to `0.9077`. Keep as a diagnostic knob, not a promotion path.
- Dynamic source-context branch:
  `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_sourcectx_dyninpaint_v1/`.
  It uses the real train anchor image itself, couples source geometry, and
  inpaints only under the warped synthetic note. This gives the strongest
  representation signal so far (`layer22 0.938 -> 0.731`, MMD `0.0992 -> 0.0241`)
  and better edge stats (`boundary=1.2876`, `color_step=0.0680`), but visual QA
  shows unlabeled real-note remnants/extra notes. Treat it as mechanism proof,
  not training data.
- Source-box/singlebox variant:
  `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_v1/`.
  It also inpaints the sampled source YOLO box and filters manifest backgrounds
  to `<=1` source label. The filter kept all `151` anchors, meaning source
  labels do not expose all visually present currency. Visual leakage is reduced
  but inpaint bands remain. Representation is still strong (`layer22=0.7769`,
  `layer0=0.6538`), so source-context replacement is now the best mechanism
  direction, with label-safety/erase quality as the blocker.
- Detector label-safety audit now quantifies that blocker. Using the current
  real-clean detector at `conf=0.05` and unmatched IoU `<0.10`, inpaint-context
  has `6/260` suspect images and `8` unmatched predictions, while
  sourcectx-singlebox has `23/260` suspect images and `26` unmatched
  predictions. The representation win is partly buying real-note leakage or
  detector-visible source remnants, so source-context must be filtered,
  re-erased, or gated before it can become training data.
- Audit-clean sourcectx-singlebox:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_auditclean_puresynth_realval_v1.yaml`.
  It excludes the `23` detector-suspect composites and keeps `237/260` images;
  rerunning the same unlabeled-prediction audit gives `0` suspect images and
  `0` unmatched predictions. The representation win mostly survives but is no
  longer as extreme: layer `0=0.6846`, layer `1=0.7346`, layer `8=0.9077`,
  layer `22=0.8346`, late MMD `0.0227`. Read: leakage helped the unfiltered
  score, but source-context replacement is still materially closer to real than
  inpaintctx (`layer22=0.8615`) and the current leader (`0.9385`). Treat this
  as a safer diagnostic branch, not a trainable promotion, because USD_50 and
  USD_100 are down to `13` images and USD_20/KHR_5000 to `16`.
- Overgen-filter-rebalanced sourcectx-singlebox:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen40_auditclean_balanced20_puresynth_realval_v1.yaml`.
  Generated `520` images at `40/class`, detector-audited out `67` suspects,
  kept `453` clean images with a `22` image minimum class floor, then selected a
  balanced `260` image, `20/class` train list. Final detector audit is `0`
  suspect images and `0` unmatched predictions. Representation remains the best
  label-clean branch so far: layer `0=0.6846`, `1=0.7538`, `8=0.9231`,
  `13=0.8346`, `19=0.8000`, `22=0.8154`, late MMD `0.0205`. Read: source
  context is still the strongest measured mechanism after removing obvious
  source-note leakage and restoring class balance.
- Strict coverage audit caught a second leakage mode: large leftover source
  notes that overlap the intended label enough to pass the IoU-only audit. With
  `--min-prediction-coverage 0.50`, the overgen40 balanced candidate still had
  `18/260` suspect images and `24` unmatched predictions. Do not use IoU-only
  unlabeled audits for source-context promotion.
- Strict overgen60 sourcectx-singlebox:
  `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen60_strictclean_balanced20_puresynth_realval_v1.yaml`.
  Generated `780` images at `60/class`, strict-audited out `117` suspects, kept
  `663` clean images, and selected a balanced `260` image, `20/class` train
  list. Final strict audit is `0` suspect images and `0` unmatched predictions.
  Visual sheet is improved on giant source-note remnants, but still shows
  hard inpaint panels/context scars, so this is bounded-probe material, not a
  perfect/done pipeline. Representation is label-safe but less dramatic than
  the IoU-only branch: layer `0=0.7231`, `1=0.7654`, `8=0.9308`, `13=0.8769`,
  `19=0.8231`, `22=0.8577`, late MMD `0.0293`. Read: strict source-context is
  about tied with inpaintctx on late domain accuracy, better on late MMD, and
  much safer on unlabeled-target leakage.
- Detector-erased source-background diagnostic:
  `scripts/build_yolo_inpainted_background_bank.py` can now add detector boxes
  to the erase mask before inpainting source backgrounds. The first detector
  bank kept `88/151` train anchors and skipped `63` high-mask images, but the
  resulting `detectorerasectx_v1` composite is rejected: dataset check passes,
  yet strict composite audit still finds `26/260` suspect images and visual QA
  shows large source-note remnants. Read: pre-erasing with the current detector
  is not sufficient because the detector misses some visible source notes; the
  final strict composite audit remains mandatory.
- Source-anchor strict audit diagnostic: `scripts/audit_yolo_unlabeled_predictions.py`
  now accepts `--manifest`, and `scripts/filter_jsonl_manifest_by_unlabeled_audit.py`
  can remove suspect JSONL rows. The mined train-anchor manifest has `18/151`
  strict source-level suspects; filtering keeps `133`. Building
  `sourcectx_sourceclean_v1` from that manifest still fails final strict
  composite audit (`23/260` suspect images, `32` unmatched predictions). Read:
  source-level filtering helps explain the leak but does not replace final
  composite filtering; source context can become unsafe after placement/overlap.
- Model-side status: real-transfer proof is still blocked by RAM headroom, not
  by data wiring. Full latest-baseline `416/b2` train-only probe failed at the
  RAM guard while scanning/training the `1,248` row baseline. A row-matched
  latest `260` image, `20/class` baseline was built, but `416/b2` still hit the
  guard at train start. A reduced `320/b1`, `150`-batch row-matched train did
  complete and produced weights for both arms, but its built-in comparison used
  train-only YAML self-eval, not real transfer: candidate synthetic self
  mAP50-95 `0.005647` vs baseline `0.005095` (`+0.000552`). Real CashSnap test
  validation of those weights was attempted and blocked immediately by RAM
  (`~97%`, `~0.5GB` free). Do not cite the `320/b1` self-eval as transfer
  proof; rerun real-test eval when memory headroom is available.
- Fixed-step model A/B is not completed. b64/b32/b16/b8 attempts hit the 95%
  RAM guard while RunLong/Codex were resident. Also, one failed b64 attempt
  reused the old leader run name with the wrapper's real-clean default before
  this was caught; do not trust that run directory's current weights. Use the
  previously written metrics JSONs as historical evidence or rerun clean with
  explicit `--model yolo26n.pt` under lower memory pressure.
- Latest model-side retry: b8 `target_anchor_latest_yolo_b8_s150_ctxprobe_v1`
  trained the baseline from `yolo26n.pt`, but candidate inpaintctx hit the 95%
  RAM guard; b4 and b2/no-AMP retries also hit the guard during setup/scanning.
  A train-only YAML helper avoided scanning the real val split during no-val
  setup, but b8 inpaintctx still hit the RAM guard mid-epoch and wrote no
  weights. No fair model A/B exists for inpaintctx/couplectx yet in this
  RunLong session.
- Do not promote either branch yet. Next best move is a clean fixed-step model
  A/B/eval for the strict overgen60 balanced candidate when memory headroom is
  available, using original real test data for evaluation. While RAM is blocked,
  the safer synth-data direction is stronger source-note removal than current
  detector pre-erase, e.g. multi-pass/segmentation/full-region source erasure
  plus final strict composite audit.

Success signal is not a prettier sheet. A real step-change branch should reduce
early-layer domain separability, recover broad real positive recall, and avoid
new empty-frame FPs. If domain accuracy stays above about `0.90` at mid/late
layers and real misses stay broad, the branch is still teaching shortcuts.

## Refiner Smoke

Goal: test whether a label-preserving learned/refiner path can remove the
remaining composited-camera shortcut without mutating denomination evidence.

Starting point:
- Use isolated Poisson/contact target-anchor output as the synthetic source.
- Use train-only CashSnap images/crops as target-domain appearance anchors.
- Do not use CashSnap val/test for refiner training, discriminator training, or
  prompt tuning.
- Keep boxes, quads, class ids, and metadata unchanged by design.
- Use `runs/cashsnap/refiner_readiness_poisson_contact_m260_v1/synthetic_manifest.jsonl`
  as the current audit manifest for AI-refiner work unless a stronger source
  distribution replaces isolated Poisson/contact.

Preferred order:
1. Low-denoise SD-Turbo/img2img with hard note+edge recomposition is the current
   local baseline because it passed the 260-image audit gates.
2. Bake off newer controlled editors/inpainters (ICEdit, Qwen-Image-Edit,
   FLUX editing variants, Step1X-Edit, OmniGen2) on the same 20-50 image pack
   before spending larger GPU/API time.
3. Inpainting/background-only variants are preferred over whole-image editing
   because hallucination is harmless only outside the protected note/edge mask.
4. FastCUT/CUT and CycleGAN-Turbo are diagnostic only unless a future patch
   fixes visual quality and 8GB memory pressure.

Smoke stages:
- `20-50` image visual/label smoke.
- `200-500` image audit probe with edge/crop/domain checks.
- `1k+` trainable candidate only after the small probes pass.

Required gates before any refined dataset is trainable:
- Exact label/metadata preservation.
- Full-size visual QA, not just contact sheets. Open several clear,
  full-size/simple-scene images with vision; compressed contact-sheet tiles hide
  rendering flaws and are not enough for visual reasoning.
- Composite-edge audit.
- Crop/geometry audit.
- Real-trained detector consistency on synthetic labels.
- Class-detail spot review for weak/protected classes.
- Domain separator as a warning light.
- Fixed-step `yolo26n` transfer.
- Background-FP guardrail.
- Per-class guard.
- At least one seed repeat before serious promotion.

Immediate next concrete step: keep the refiner harness available, but do not
spend more runs on SD-Turbo prompt/strength tweaks until a candidate reduces the
representation gap or directly attacks the obligation ledger. The current
note-edge SD-Turbo m260 candidate is a harness proof, not a promotion path.

Kill criteria:
- Denomination text, portraits, numerals, colors, or security details mutate.
- Refiner erases small visible class evidence.
- Edges still look pasted after tone improves.
- Domain stats improve but real transfer does not.
- Empty-frame FPs worsen without a compensating real-transfer result strong
  enough to justify a new branch.

## Negative Bank

Realistic near-negatives matter now. The real background-FP review shows
empty-label false positives are often full non-target banknotes or target-like
paper on the same retail surfaces as positives.

Current state:
- Existing WebGL unknown/hard-negative roots are too stylized/easy.
- Hardnegdiv8, realbgneg25, and unknownsoftfp8lowconf did not clear guardrails.
- `configs/synthetic_recipes/cashsnap_external_negative_banks_v1.json` is
  planned/registry work, not an accepted trainable bank.

Next realistic bank should include reviewed foreign/unknown currencies,
target-lookalike partial notes, receipts, cards, patterned paper, and retail
clutter rendered or composited through the same camera-domain policy. Do not
mine CashSnap val/test empty frames into training.

## Promotion Gates

A synthetic axis is credible only with:
- Full real val/test improvement or preservation.
- Clean-visible val/test preservation.
- Labeled-positive and geometry-stress preservation.
- Protected-class preservation, especially riel.
- Real empty-frame FP detections and images-with-FP not worse at `conf=0.05`,
  `imgsz=416`, `batch=1`, `device=0`.
- Max per-class mAP50-95 drop `<=0.05` unless explicitly waived.
- At least one seed repeat for serious promotion, more for large claims.

Synthetic package gates are necessary filters, not promotion authority.

Clean-base can move toward overlap/fan/hand only when synthetic-only `yolo26n`
is near the target line, clean-visible and labeled-positive test are `>=0.75`,
protected riel passes, real-empty FPs are no worse than control, and the result
survives at least two seeds or a slow-promotion run.

## Validation Slices

- Full real val/test includes many empty-label images; always pair aggregate AP
  with empty-frame FP probes.
- Roboflow core-13 bridge is the cleaner positive KHR/USD judge for the current
  detector. Pair it with CashSnap empty-frame FP diagnostics; neither replaces
  the other.
- Roboflow official21 partial bridge preserves official classes present in the
  source, including `KHR_100`, but is not evaluable with current 13-class
  weights.
- `runs/cashsnap/real_geometry_stress_slices_v1` is the current provisional real
  geometry slice set.
- `labeled_all`: positive-only val/test, `988/814` images.
- `geometry_stress`: broad non-clean labeled val/test, `791/659` images.
- True `multi_note` is only `2/3` images and is not fan/overlap proof.
- Protected-riel slices are small; use scoped checks and do not over-read one
  noisy class row.

## Label And Class Policy

- Detector labels are visible-instance AABBs derived from the renderer/CV
  visible mask or source annotation, one box per visible class instance.
- OBB/quadrilateral metadata preserves side/pose information for audits and
  future oriented/fusion work; it is not a direct YOLO detect label today.
- Fragment/evidence labels are for disconnected visible evidence and future
  count fusion. They are not direct physical-note count labels.
- Zero-label hard-negative roots must remain zero-label. Do not silently turn
  unknown/foreign note props into target classes.
- Raw Roboflow exports are intake. Use bridge builders to convert them into
  schema-aware processed roots.
- `KHR_100` is official KHR, not garbage. It is excluded only from the current
  core-13 bridge because the active detector cannot predict it yet.
- Current active detector scope is 13 operational classes, not all official
  KHR/USD. Official current scope is 21 classes.
- `KHR_50` remains blocked for v1 operational training unless real retail/bank
  capture evidence or an explicit product requirement justifies it.
- Trainable WebGL target-note renders must pass
  `scripts/check_webgl_texture_asset_policy.py` against the approved texture
  bank. USD_20 must remain the reviewed `2004-2021` current design.

## Active Artifacts

Key configs:
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_latest_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_poseclose_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_hardnegdiv8_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_realbgneg25_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_poisson_contact_unknownsoftfp8lowconf_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_realfgstyle_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_latest_balanced20_puresynth_realval_v1.yaml`
- `configs/generated_lists/webgl_ablation/cashsnap_target_anchor_latest_balanced20_v1_train.txt`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_analogs_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_inpaintctx_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_couplectx_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_couplectx_feather08_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_dyninpaint_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_auditclean_puresynth_realval_v1.yaml`
- `configs/generated_lists/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_auditclean_v1_train.txt`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen40_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen40_auditclean_puresynth_realval_v1.yaml`
- `configs/generated_lists/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen40_auditclean_v1_train.txt`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen40_auditclean_balanced20_puresynth_realval_v1.yaml`
- `configs/generated_lists/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen40_auditclean_balanced20_v1_train.txt`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen60_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen60_strictclean_puresynth_realval_v1.yaml`
- `configs/generated_lists/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen60_strictclean_v1_train.txt`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen60_strictclean_balanced20_puresynth_realval_v1.yaml`
- `configs/generated_lists/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen60_strictclean_balanced20_v1_train.txt`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_detectorerasectx_puresynth_realval_v1.yaml`
- `configs/webgl_ablation/cashsnap_target_anchor_transplant_rep_gap_sourcectx_sourceclean_puresynth_realval_v1.yaml`
- `configs/synthetic_recipes/cashsnap_external_negative_banks_v1.json`
- `configs/synthetic_recipes/cashsnap_webgl_recipe_catalog_v1.json`
- `configs/synthetic_recipes/cashsnap_synthetic_governance_v1.json`
- `configs/synthetic_recipes/cashsnap_data_lifecycle_registry_v1.json`
- `configs/synthetic_recipes/cashsnap_webgl_approved_texture_bank_v1.json`

Key roots:
- `data/synthetic/cashsnap_target_anchor_transplant_latest_v1/`
- `data/synthetic/cashsnap_target_anchor_transplant_poisson_contact_v1/`
- `data/synthetic/cashsnap_target_anchor_transplant_poisson_contact_poseclose_v1/`
- `data/backgrounds/cashsnap_v1_no_note_patches_strict_v1/`
- `data/backgrounds/cashsnap_rep_gap_train_anchor_inpainted_filtered_v1/`
- `data/backgrounds/cashsnap_rep_gap_train_anchor_detector_erased_v1/`
- `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_analogs_v1/`
- `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_inpaintctx_v1/`
- `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_couplectx_v1/`
- `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_couplectx_feather08_v1/`
- `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_sourcectx_dyninpaint_v1/`
- `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_v1/`
- `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen40_v1/`
- `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_overgen60_v1/`
- `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_detectorerasectx_v1/`
- `data/synthetic/cashsnap_target_anchor_transplant_rep_gap_sourcectx_sourceclean_v1/`
- `data/synthetic/cashsnap_webgl_unknown_currency_soft_negative_smoke_v1/`
- `data/processed/roboflow_khmer_us_currency_core13_bridge_v1/`
- `data/processed/roboflow_khmer_us_currency_official21_partial_bridge_v1/`

Key run artifacts:
- `runs/cashsnap/synthetic_obligation_ledger_latest.json`
- `runs/cashsnap/synthetic_obligation_ledger_latest.md`
- `runs/cashsnap/composite_edge_audit_target_anchor_poisson_contact_v1.json`
- `runs/cashsnap/fixed_step_target_anchor_latest_vs_poisson_contact_b8_e50_s150_seed0_summary.json`
- `runs/cashsnap/transfer_guardrails_target_anchor_latest_vs_poisson_contact_b8.json`
- `runs/cashsnap/transfer_guardrails_target_anchor_latest_vs_poisson_contact_unknownsoftfp8lowconf_b8.json`
- `runs/cashsnap/background_fp_poisson_contact_b8_unknownsoft_conf0005.json`
- `runs/cashsnap/currency_taxonomy_gap_plan_official_latest.md`
- `runs/cashsnap/refiner_readiness_poisson_contact_v1/summary.json`
- `runs/cashsnap/refiner_readiness_poisson_contact_v1/synthetic_manifest.jsonl`
- `runs/cashsnap/refiner_readiness_poisson_contact_v1/preview/mask_lock_contact.jpg`
- `runs/cashsnap/refiner_readiness_poisson_contact_v1/identity_preservation_check.json`
- `runs/cashsnap/refiner_readiness_poisson_contact_v1/cut_unaligned_smoke/dataset_manifest.json`
- `runs/cashsnap/refiner_readiness_poisson_contact_v1/cut_unaligned_smoke/synthetic_manifest.jsonl`
- `runs/cashsnap/refiner_readiness_poisson_contact_v1/cut_unaligned_smoke/identity_preservation_check.json`
- `runs/cashsnap/refiner_checkpoints/cashsnap_fastcut_poisson_smoke_e1_m8_128/`
- `runs/cashsnap/refiner_results/cashsnap_fastcut_poisson_smoke_e1_m8_128/label_preservation_fake_B_first8.json`
- `runs/cashsnap/refiner_results/cashsnap_fastcut_poisson_smoke_e1_m8_128/label_preservation_detail_locked_first8.json`
- `runs/cashsnap/refiner_results/cashsnap_fastcut_poisson_smoke_e1_m8_128/label_preservation_note_locked_hard_first8.json`
- `runs/cashsnap/refiner_checkpoints/cashsnap_fastcut_maskid200_poisson_smoke_e5_m8_128/`
- `runs/cashsnap/refiner_results/cashsnap_fastcut_maskid200_poisson_smoke_e5_m8_128/label_preservation_fake_B_first8.json`
- `runs/cashsnap/refiner_results/cashsnap_fastcut_maskid200_poisson_smoke_e5_m8_128/label_preservation_note_edge_locked_first8.json`
- `runs/cashsnap/refiner_checkpoints/cashsnap_fastcut_maskid200_poisson_candidate_e5_m52_128/`
- `runs/cashsnap/refiner_results/cashsnap_fastcut_maskid200_poisson_candidate_e5_m52_128/label_preservation_note_edge_locked.json`
- `runs/cashsnap/refiner_results/cashsnap_fastcut_maskid200_poisson_candidate_e5_m52_128/background_realism_note_edge_locked.json`
- `runs/cashsnap/refiner_yolo_candidates/fastcut_maskid200_note_edge_e5_m52_128/summary.json`
- `runs/cashsnap/refiner_yolo_candidates/fastcut_maskid200_note_edge_e5_m52_128/edge_audit.json`
- `runs/cashsnap/refiner_yolo_candidates/fastcut_maskid200_note_edge_e5_m52_128/qa/random_blend.png`
- `runs/cashsnap/refiner_readiness_poisson_contact_v1/background_realism_source_trainA.json`
- `runs/cashsnap/refiner_readiness_poisson_contact_m260_v1/summary.json`
- `runs/cashsnap/refiner_readiness_poisson_contact_m260_v1/identity_preservation_check.json`
- `runs/cashsnap/refiner_readiness_poisson_contact_m260_v1/background_realism_source.json`
- `runs/cashsnap/sd_turbo_img2img_s025_steps4_m52/label_preservation_raw.json`
- `runs/cashsnap/sd_turbo_img2img_s025_steps4_m52_note_edge_locked/label_preservation.json`
- `runs/cashsnap/sd_turbo_img2img_s025_steps4_m52_note_edge_locked/background_realism.json`
- `runs/cashsnap/sd_turbo_img2img_s025_steps4_m260/label_preservation_raw.json`
- `runs/cashsnap/sd_turbo_img2img_s025_steps4_m260_note_edge_locked/label_preservation.json`
- `runs/cashsnap/sd_turbo_img2img_s025_steps4_m260_note_edge_locked/background_realism.json`
- `runs/cashsnap/refiner_yolo_candidates/sd_turbo_note_edge_s025_steps4_m260/summary.json`
- `runs/cashsnap/refiner_yolo_candidates/sd_turbo_note_edge_s025_steps4_m260/edge_audit.json`
- `runs/cashsnap/refiner_yolo_candidates/sd_turbo_note_edge_s025_steps4_m260/qa/random_blend.png`
- `runs/cashsnap/fixed_step_source_vs_sd_turbo_m260_b4_e50_s150_seed0_summary.json`
- `runs/cashsnap/fixed_step_source_poisson_contact_m260_vs_sd_turbo_note_edge_s025_m260_steps150/summary.json`
- `runs/cashsnap/representation_gap_synthleader_target_anchor_latest_test_v1/summary.json`
- `runs/cashsnap/representation_gap_realclean_source_poisson_m260_test_v1/summary.json`
- `runs/cashsnap/representation_gap_realclean_sd_turbo_note_edge_m260_test_v1/summary.json`
- `runs/cashsnap/positive_error_review_synthleader_real_test_v1/summary.json`
- `runs/cashsnap/background_fp_synthleader_real_test_v1.json`
- `runs/cashsnap/synthetic_obligation_ledger_synthleader_rep_gap_v1.md`
- `runs/cashsnap/representation_gap_synthleader_train_analogs_v1/summary.json`
- `runs/cashsnap/representation_gap_synthleader_train_analogs_v1/train_anchor_manifest.jsonl`
- `runs/cashsnap/representation_gap_synthleader_train_analogs_v1/query_train_analog_pairs.jpg`
- `runs/cashsnap/dataset_check_rep_gap_analogs_v1.json`
- `runs/cashsnap/visual_qa_rep_gap_analogs_v1/`
- `runs/cashsnap/composite_edge_audit_rep_gap_analogs_v1.json`
- `runs/cashsnap/representation_gap_synthleader_rep_gap_analogs_test_v1/summary.json`
- `runs/cashsnap/dataset_check_rep_gap_inpaintctx_v1.json`
- `runs/cashsnap/visual_qa_rep_gap_inpaintctx_v1/`
- `runs/cashsnap/composite_edge_audit_rep_gap_inpaintctx_v1.json`
- `runs/cashsnap/representation_gap_synthleader_rep_gap_inpaintctx_test_v1/summary.json`
- `runs/cashsnap/dataset_check_rep_gap_couplectx_v1.json`
- `runs/cashsnap/visual_qa_rep_gap_couplectx_v1/`
- `runs/cashsnap/composite_edge_audit_rep_gap_couplectx_v1.json`
- `runs/cashsnap/representation_gap_synthleader_rep_gap_couplectx_test_v1/summary.json`
- `runs/cashsnap/dataset_check_rep_gap_couplectx_feather08_v1.json`
- `runs/cashsnap/visual_qa_rep_gap_couplectx_feather08_v1/`
- `runs/cashsnap/composite_edge_audit_rep_gap_couplectx_feather08_v1.json`
- `runs/cashsnap/representation_gap_synthleader_rep_gap_couplectx_feather08_test_v1/summary.json`
- `runs/cashsnap/dataset_check_rep_gap_sourcectx_dyninpaint_v1.json`
- `runs/cashsnap/visual_qa_rep_gap_sourcectx_dyninpaint_v1/`
- `runs/cashsnap/composite_edge_audit_rep_gap_sourcectx_dyninpaint_v1.json`
- `runs/cashsnap/representation_gap_synthleader_rep_gap_sourcectx_dyninpaint_test_v1/summary.json`
- `runs/cashsnap/dataset_check_rep_gap_sourcectx_singlebox_v1.json`
- `runs/cashsnap/visual_qa_rep_gap_sourcectx_singlebox_v1/`
- `runs/cashsnap/composite_edge_audit_rep_gap_sourcectx_singlebox_v1.json`
- `runs/cashsnap/representation_gap_synthleader_rep_gap_sourcectx_singlebox_test_v1/summary.json`
- `runs/cashsnap/unlabeled_prediction_audit_rep_gap_inpaintctx_v1.json`
- `runs/cashsnap/unlabeled_prediction_audit_rep_gap_inpaintctx_v1.jpg`
- `runs/cashsnap/unlabeled_prediction_audit_rep_gap_sourcectx_singlebox_v1.json`
- `runs/cashsnap/unlabeled_prediction_audit_rep_gap_sourcectx_singlebox_v1.jpg`
- `runs/cashsnap/dataset_check_rep_gap_sourcectx_singlebox_auditclean_v1.json`
- `runs/cashsnap/representation_gap_synthleader_rep_gap_sourcectx_singlebox_auditclean_test_v1/summary.json`
- `runs/cashsnap/unlabeled_prediction_audit_rep_gap_sourcectx_singlebox_auditclean_v1.json`
- `runs/cashsnap/dataset_check_rep_gap_sourcectx_singlebox_overgen40_v1.json`
- `runs/cashsnap/unlabeled_prediction_audit_rep_gap_sourcectx_singlebox_overgen40_v1.json`
- `runs/cashsnap/dataset_check_rep_gap_sourcectx_singlebox_overgen40_auditclean_v1.json`
- `runs/cashsnap/dataset_check_rep_gap_sourcectx_singlebox_overgen40_auditclean_balanced20_v1.json`
- `runs/cashsnap/unlabeled_prediction_audit_rep_gap_sourcectx_singlebox_overgen40_auditclean_balanced20_v1.json`
- `runs/cashsnap/representation_gap_synthleader_rep_gap_sourcectx_singlebox_overgen40_auditclean_balanced20_test_v1/summary.json`
- `runs/cashsnap/unlabeled_prediction_audit_rep_gap_sourcectx_singlebox_overgen40_auditclean_balanced20_strictcov50_v1.json`
- `runs/cashsnap/visual_qa_rep_gap_sourcectx_singlebox_overgen40_auditclean_balanced20_v1/per_class_sheet.jpg`
- `runs/cashsnap/dataset_check_rep_gap_sourcectx_singlebox_overgen60_v1.json`
- `runs/cashsnap/unlabeled_prediction_audit_rep_gap_sourcectx_singlebox_overgen60_strictcov50_v1.json`
- `runs/cashsnap/dataset_check_rep_gap_sourcectx_singlebox_overgen60_strictclean_v1.json`
- `runs/cashsnap/dataset_check_rep_gap_sourcectx_singlebox_overgen60_strictclean_balanced20_v1.json`
- `runs/cashsnap/unlabeled_prediction_audit_rep_gap_sourcectx_singlebox_overgen60_strictclean_balanced20_strictcov50_v1.json`
- `runs/cashsnap/visual_qa_rep_gap_sourcectx_singlebox_overgen60_strictclean_balanced20_v1/per_class_sheet.jpg`
- `runs/cashsnap/representation_gap_synthleader_rep_gap_sourcectx_singlebox_overgen60_strictclean_balanced20_test_v1/summary.json`
- `runs/cashsnap/dataset_check_target_anchor_latest_balanced20_v1.json`
- `runs/cashsnap/fixed_step_target_anchor_latest_bal20_vs_sourcectx_strictclean_b20_b1_s150_i320_v1_preflight.json`
- `runs/cashsnap/fixed_step_target_anchor_latest_bal20_vs_sourcectx_strictclean_b20_b1_s150_i320_v1_summary.json`
- `runs/cashsnap/fixed_step_target_anchor_latest_balanced20_trainonly_vs_sourcectx_strictclean_b20_trainonly_steps150/summary.json`
- `runs/cashsnap/system_profile_after_i320_train_before_realtest_guard_v1.json`
- `runs/cashsnap/dataset_check_rep_gap_detectorerasectx_v1.json`
- `runs/cashsnap/unlabeled_prediction_audit_rep_gap_detectorerasectx_strictcov50_v1.json`
- `runs/cashsnap/visual_qa_rep_gap_detectorerasectx_v1/per_class_sheet.jpg`
- `runs/cashsnap/unlabeled_prediction_audit_rep_gap_train_anchors_strictcov50_v1.json`
- `runs/cashsnap/representation_gap_synthleader_train_analogs_v1/train_anchor_manifest_strict_sourceclean_v1.jsonl`
- `runs/cashsnap/representation_gap_synthleader_train_analogs_v1/train_anchor_manifest_strict_sourceclean_v1.summary.json`
- `runs/cashsnap/dataset_check_rep_gap_sourcectx_sourceclean_v1.json`
- `runs/cashsnap/unlabeled_prediction_audit_rep_gap_sourcectx_sourceclean_strictcov50_v1.json`
- `runs/cashsnap/visual_qa_rep_gap_sourcectx_sourceclean_v1/per_class_sheet.jpg`
- `runs/cashsnap/fixed_step_target_anchor_latest_vs_rep_gap_inpaintctx_b8_s150_ctxprobe_v1_preflight.json`
- `runs/cashsnap/system_profile_after_inpaintctx_b32_guard.json`
- `runs/cashsnap/system_profile_after_b8_inpaintctx_guard_v1.json`
- `runs/cashsnap/cyclegan_turbo_smoke_s1_128_noval_nolpips/`

Key scripts:
- `scripts/build_cashsnap_target_anchor_transplant.py`
- `scripts/build_yolo_inpainted_background_bank.py`
- `scripts/build_yolo_balanced_subset.py`
- `scripts/materialize_yolo_trainonly_data_yaml.py`
- `scripts/materialize_yolo_unlabeled_audit_filtered_config.py`
- `scripts/build_yolo_split_visual_qa_sheet.py`
- `scripts/filter_jsonl_manifest_by_unlabeled_audit.py`
- `scripts/audit_synthetic_composite_edges.py`
- `scripts/build_synthetic_obligation_ledger.py`
- `scripts/build_webgl_hard_negative_dose_config.py`
- `scripts/build_fp_mined_negative_dose_config.py`
- `scripts/build_external_negative_dose_config.py`
- `scripts/build_synthetic_visual_qa_pack.py`
- `scripts/build_refiner_readiness_pack.py`
- `scripts/check_refiner_label_preservation.py`
- `scripts/check_refiner_background_realism.py`
- `scripts/materialize_refiner_unaligned_dataset.py`
- `scripts/materialize_refiner_yolo_candidate.py`
- `scripts/apply_refiner_detail_lock.py`
- `scripts/run_sd_turbo_img2img_refiner.py`
- `scripts/probe_yolo_representation_domain_gap.py`
- `scripts/mine_representation_gap_train_analogs.py`
- `scripts/audit_yolo_unlabeled_predictions.py`
- `scripts/run_yolo_fixed_step_probe.py`
- `scripts/probe_yolo_background_false_positives.py`
- `scripts/check_yolo_transfer_guardrails.py`
- `scripts/check_yolo_dataset.py`
- `scripts/check_currency_taxonomy_coverage.py`

Script notes:
- `build_webgl_hard_negative_dose_config.py` accepts directory train splits as
  well as `.txt` lists and supports `--filename-contains`.
- `build_fp_mined_negative_dose_config.py` accepts directory train splits as
  well as `.txt` lists.
- `build_cashsnap_target_anchor_transplant.py` accepts
  `--geometry-manifest` with `--geometry-manifest-mode prefer|only`; `prefer`
  uses train-anchor geometry where present and falls back per missing class.
- `build_cashsnap_target_anchor_transplant.py` also accepts
  `--couple-background-geometry`, `--geometry-size-jitter`,
  `--position-jitter-fraction`, and `--warp-alpha-feather-px` for train-anchor
  inpaint/context diagnostics.
- `build_cashsnap_target_anchor_transplant.py` can use train-positive source
  images via `--background-manifest`, dynamic erase via
  `--inpaint-under-foreground-px`, and source-box erase via
  `--inpaint-source-box-pad-fraction`; these modes are mechanism probes until
  visual label-safety is proven.
- `build_yolo_inpainted_background_bank.py` creates train-only empty-label
  inpainted canvases from labeled YOLO images; use `--max-mask-fraction` to
  reject full-frame erasures that make obvious scars.
- `materialize_yolo_trainonly_data_yaml.py` rewrites `val`/`test` to `train`
  for no-val training setup only. Evaluate with the original data YAML; do not
  treat train-only YAML metrics as real transfer.

## Repo And Harness Rules

- RunLong mode: prefix terminal commands with `rl`.
- Work directly on `master` unless asked for a branch.
- Repo-local runtime storage is enforced through `scripts/local_runtime.py`.
- YOLO promotion posture: `batch=64`, `workers=0`, `device=0`, `cache=false`;
  eval `batch=64`, `workers=2`.
- Background-FP guardrail uses `batch=1`.
- In RunLong/Codex memory pressure, use quiet/log-redirection and b8/b16
  diagnostics. Do not compare lower-batch diagnostics to b64 promotion parity
  without saying so.
- Long refiner training/inference must run through `scripts/run_with_headroom.py`
  unless there is a specific reason not to. The wrapper enforces the repo's
  95% CPU/RAM/GPU-memory ceiling and keeps the machine usable.
- WebGL posture: `--render-jobs 2 --renderer-batch-size 32 --check-jobs 4`.
- `cache=disk` is rejected; it created many `.npy` files, used about 12 GB, and
  slowed throughput.
- Active docs are `README.md`, `AGENTS.md`, and this file.
- Old working memory belongs in `docs/archive/`.
- Generated synthetic roots stay under ignored `data/synthetic/`.
- External negatives stay under ignored `data/external_negatives/`.
- Training/eval outputs stay under ignored `runs/`.
- Browser/temp/cache state stays under ignored `.cache_runtime/` or `tmp/`.

## Useful Commands

```powershell
rl python scripts\check_currency_taxonomy_coverage.py
rl python scripts\check_webgl_trainable_candidate_suite.py --check-existing
rl python scripts\check_synthetic_pipeline_readiness.py --check-existing --json-out runs\cashsnap\synthetic_pipeline_readiness_latest.json
rl python scripts\check_yolo_dataset.py --data configs\webgl_ablation\cashsnap_target_anchor_transplant_poisson_contact_puresynth_realval_v1.yaml --min-train-class-images 96 --min-train-class-boxes 96 --fail-on-problems
rl python scripts\build_synthetic_visual_qa_pack.py --data configs\webgl_ablation\cashsnap_target_anchor_transplant_poisson_contact_puresynth_realval_v1.yaml --split train --max-images 128 --out-dir runs\cashsnap\visual_qa_target_anchor_poisson_contact_v1
rl python scripts\build_refiner_readiness_pack.py --synthetic-root data\synthetic\cashsnap_target_anchor_transplant_poisson_contact_v1 --out-dir runs\cashsnap\refiner_readiness_poisson_contact_v1 --max-synthetic 52 --max-real 160 --max-backgrounds 64 --edge-band-px 10 --detail-erode-px 10 --min-free-vram-gb 1.5 --min-free-ram-gb 1.0
rl python scripts\materialize_refiner_unaligned_dataset.py --readiness-dir runs\cashsnap\refiner_readiness_poisson_contact_v1 --clean
rl python scripts\check_refiner_label_preservation.py --manifest runs\cashsnap\refiner_readiness_poisson_contact_v1\synthetic_manifest.jsonl --refined-root <refined-root> --json-out runs\cashsnap\<refiner>_label_preservation.json --fail-on-violations
rl python scripts\check_refiner_label_preservation.py --manifest runs\cashsnap\refiner_readiness_poisson_contact_v1\cut_unaligned_smoke\synthetic_manifest.jsonl --refined-root <cut-output-root> --json-out runs\cashsnap\<refiner>_cut_label_preservation.json --fail-on-violations
rl python scripts\apply_refiner_detail_lock.py --manifest runs\cashsnap\refiner_readiness_poisson_contact_v1\cut_unaligned_smoke\synthetic_manifest.jsonl --refined-root <cut-output-root> --out-root runs\cashsnap\<refiner>_detail_locked --json-out runs\cashsnap\<refiner>_detail_lock_apply.json --lock-mask detail_lock --feather-px 1.5
rl python scripts\run_with_headroom.py --memory-action pause -- python <refiner_train_or_infer.py>
rl python scripts\run_with_headroom.py --memory-action pause -- python .cache_runtime\third_party\contrastive-unpaired-translation\train.py --dataroot runs\cashsnap\refiner_readiness_poisson_contact_v1\cut_unaligned_smoke --name cashsnap_fastcut_poisson_smoke_e1_m8_128 --CUT_mode FastCUT --gpu_ids 0 --checkpoints_dir runs\cashsnap\refiner_checkpoints --dataset_mode unaligned --direction AtoB --batch_size 1 --num_threads 0 --load_size 128 --crop_size 128 --preprocess resize_and_crop --max_dataset_size 8 --n_epochs 1 --n_epochs_decay 0 --netG resnet_6blocks --ngf 32 --ndf 32 --display_id -1 --no_html --print_freq 4 --save_latest_freq 100 --save_epoch_freq 1 --no_flip
rl python scripts\run_with_headroom.py --memory-action exit -- python .cache_runtime\third_party\img2img-turbo\src\train_cyclegan_turbo.py --dataset_folder runs\cashsnap\refiner_readiness_poisson_contact_v1\cut_unaligned_smoke --train_img_prep resize_128 --val_img_prep resize_128 --dataloader_num_workers 0 --train_batch_size 1 --max_train_steps 1 --max_train_epochs 1 --pretrained_model_name_or_path stabilityai/sd-turbo --output_dir runs\cashsnap\cyclegan_turbo_smoke_s1_128_noval_nolpips --report_to none --tracker_project_name cashsnap_cyclegan_turbo_smoke --validation_steps 0 --validation_num_images 0 --checkpointing_steps 1 --learning_rate 1e-5 --gradient_accumulation_steps 1 --allow_tf32 --gradient_checkpointing --lora_rank_unet 4 --lora_rank_vae 2 --lambda_gan 0.5 --lambda_idt 1 --lambda_cycle 1 --lambda_cycle_lpips 0 --lambda_idt_lpips 0
rl python scripts\audit_synthetic_composite_edges.py --help
rl python scripts\probe_yolo_representation_domain_gap.py --model runs\cashsnap\fixed_step_target_anchor_transplant_latest_v1_from_clean_e50_i416_b64_w0_auto_lr1e2_warmup3_amp_cachefalse_steps150_seed0\weights\best.pt --real-data data\cashsnap_v1\data.yaml --real-split test --synthetic-data configs\webgl_ablation\cashsnap_target_anchor_transplant_latest_puresynth_realval_v1.yaml --synthetic-split train --out-dir runs\cashsnap\representation_gap_synthleader_target_anchor_latest_test_v1 --imgsz 416 --batch 8 --device 0 --max-per-class 10 --top-k 40 --clean
rl python scripts\mine_representation_gap_train_analogs.py --gap-summary runs\cashsnap\representation_gap_synthleader_target_anchor_latest_test_v1\summary.json --candidate-data data\cashsnap_v1\data.yaml --candidate-split train --out-dir runs\cashsnap\representation_gap_synthleader_train_analogs_v1 --batch 16 --device 0 --top-query 40 --per-query 4 --max-candidates-per-class 300 --clean
rl python scripts\build_yolo_inpainted_background_bank.py --manifest runs\cashsnap\representation_gap_synthleader_train_analogs_v1\train_anchor_manifest.jsonl --out-root data\backgrounds\cashsnap_rep_gap_train_anchor_inpainted_filtered_v1 --suffix train --pad-fraction 0.08 --mask-dilate-px 5 --inpaint-radius 7 --max-mask-fraction 0.45 --clean
rl python scripts\materialize_yolo_trainonly_data_yaml.py --data configs\webgl_ablation\cashsnap_target_anchor_transplant_rep_gap_inpaintctx_puresynth_realval_v1.yaml --out .cache_runtime\ultralytics_data\cashsnap_target_anchor_transplant_rep_gap_inpaintctx_trainonly_ctxprobe_v1.yaml
rl python scripts\build_cashsnap_target_anchor_transplant.py --background-root data\backgrounds\cashsnap_rep_gap_train_anchor_inpainted_filtered_v1 --background-split train --geometry-manifest runs\cashsnap\representation_gap_synthleader_train_analogs_v1\train_anchor_manifest.jsonl --geometry-manifest-mode prefer --foreground-style-policy real_crop_stats --composite-policy poisson_mixed --shadow-policy contact --pose-policy aabb_aspect_repair --box-scale-jitter 0.12 --min-class-geometry-samples 1 --per-class 20 --seed 29 --out-root data\synthetic\cashsnap_target_anchor_transplant_rep_gap_inpaintctx_v1 --out-config configs\webgl_ablation\cashsnap_target_anchor_transplant_rep_gap_inpaintctx_puresynth_realval_v1.yaml --preview-count 30 --clean
rl python scripts\build_cashsnap_target_anchor_transplant.py --background-root data\backgrounds\cashsnap_rep_gap_train_anchor_inpainted_filtered_v1 --background-split train --geometry-manifest runs\cashsnap\representation_gap_synthleader_train_analogs_v1\train_anchor_manifest.jsonl --geometry-manifest-mode prefer --foreground-style-policy real_crop_stats --composite-policy poisson_mixed --shadow-policy contact --pose-policy aabb_aspect_repair --box-scale 1.06 --box-scale-jitter 0.04 --geometry-size-jitter 0.05 --position-jitter-fraction 0.012 --couple-background-geometry --min-class-geometry-samples 1 --per-class 20 --seed 31 --out-root data\synthetic\cashsnap_target_anchor_transplant_rep_gap_couplectx_v1 --out-config configs\webgl_ablation\cashsnap_target_anchor_transplant_rep_gap_couplectx_puresynth_realval_v1.yaml --preview-count 30 --clean
rl python scripts\build_cashsnap_target_anchor_transplant.py --background-manifest runs\cashsnap\representation_gap_synthleader_train_analogs_v1\train_anchor_manifest.jsonl --background-max-source-boxes 1 --geometry-manifest runs\cashsnap\representation_gap_synthleader_train_analogs_v1\train_anchor_manifest.jsonl --geometry-manifest-mode prefer --foreground-style-policy real_crop_stats --composite-policy poisson_mixed --shadow-policy contact --pose-policy aabb_aspect_repair --box-scale 1.10 --box-scale-jitter 0.03 --geometry-size-jitter 0.04 --position-jitter-fraction 0.006 --inpaint-under-foreground-px 8 --inpaint-under-foreground-radius 5 --inpaint-source-box-pad-fraction 0.02 --couple-background-geometry --min-class-geometry-samples 1 --per-class 20 --seed 53 --out-root data\synthetic\cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_v1 --out-config configs\webgl_ablation\cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_puresynth_realval_v1.yaml --preview-count 30 --clean
rl python scripts\probe_yolo_representation_domain_gap.py --model runs\cashsnap\fixed_step_target_anchor_transplant_latest_v1_from_clean_e50_i416_b64_w0_auto_lr1e2_warmup3_amp_cachefalse_steps150_seed0\weights\best.pt --real-data data\cashsnap_v1\data.yaml --real-split test --synthetic-data configs\webgl_ablation\cashsnap_target_anchor_transplant_rep_gap_inpaintctx_puresynth_realval_v1.yaml --synthetic-split train --out-dir runs\cashsnap\representation_gap_synthleader_rep_gap_inpaintctx_test_v1 --imgsz 416 --batch 8 --device 0 --max-per-class 10 --top-k 40 --clean
rl python scripts\probe_yolo_representation_domain_gap.py --model runs\cashsnap\fixed_step_target_anchor_transplant_latest_v1_from_clean_e50_i416_b64_w0_auto_lr1e2_warmup3_amp_cachefalse_steps150_seed0\weights\best.pt --real-data data\cashsnap_v1\data.yaml --real-split test --synthetic-data configs\webgl_ablation\cashsnap_target_anchor_transplant_rep_gap_couplectx_puresynth_realval_v1.yaml --synthetic-split train --out-dir runs\cashsnap\representation_gap_synthleader_rep_gap_couplectx_test_v1 --imgsz 416 --batch 8 --device 0 --max-per-class 10 --top-k 40 --clean
rl python scripts\probe_yolo_representation_domain_gap.py --model runs\cashsnap\fixed_step_target_anchor_transplant_latest_v1_from_clean_e50_i416_b64_w0_auto_lr1e2_warmup3_amp_cachefalse_steps150_seed0\weights\best.pt --real-data data\cashsnap_v1\data.yaml --real-split test --synthetic-data configs\webgl_ablation\cashsnap_target_anchor_transplant_rep_gap_sourcectx_singlebox_puresynth_realval_v1.yaml --synthetic-split train --out-dir runs\cashsnap\representation_gap_synthleader_rep_gap_sourcectx_singlebox_test_v1 --imgsz 416 --batch 8 --device 0 --max-per-class 10 --top-k 40 --clean
rl python scripts\build_synthetic_obligation_ledger.py --no-default-evidence --positive-error-review runs\cashsnap\positive_error_review_synthleader_real_test_v1\summary.json --background-fp runs\cashsnap\background_fp_synthleader_real_test_v1.json --visual-failure "representation_gap|real_test|Current synthetic leader remains highly domain-separable after class balancing." --json-out runs\cashsnap\synthetic_obligation_ledger_synthleader_rep_gap_v1.json --md-out runs\cashsnap\synthetic_obligation_ledger_synthleader_rep_gap_v1.md
rl python scripts\run_yolo_fixed_step_probe.py --help
rl python scripts\check_yolo_transfer_guardrails.py --help
rl python scripts\probe_yolo_background_false_positives.py --help
```
