# CashSnap Active Plan

Read this first when the repo feels noisy.

## North Star

Build a small phone/browser-deployable CashSnap model that counts mixed USD and Khmer Riel banknotes from real mobile photos. The model must keep clean-note accuracy while learning rare KHR classes, old/common circulated notes, and eventually partial, overlapped, fanned, and hand-occluded notes.

## Current Decision

The active path is base-model strength first, then reviewed partial/fan specialization.

Use the 2.5D synthetic harness to make the base detector better before scaling hard overlap scenes. The next synthetic work should focus on clean and near-clean 2.5D scenes with realistic camera/domain knobs:

- lens/aspect/crop variation
- color temperature and tint
- exposure, contrast, blur, grain, sharpening, and JPEG compression
- real or visually QA'd backgrounds
- class balancing for weak KHR classes, especially `KHR_20000` and `KHR_50000`
- clean separation between current KHR, current rare KHR, old/common circulated KHR, and USD

The first Numista 2.5D calibration probes improved real-overlap region coverage but still overcounted and confused old/common KHR backs. Do not scale the same synthetic recipe blindly. The immediate bottleneck is reviewed real fragment evidence for thin/edge `KHR_5000` and `KHR_20000` crops, especially the compact P1 failure queue in `docs/p1-fragment-curation-runbook.md`.

After reviewed fragments improve denomination identity without clean-validation regression, scale into:

- simple overlap
- shop-counter spreads
- partial off-frame notes
- thin visible slices
- hand-held fans and finger occlusion

## Parked But Not Deleted

The original WebGL/3D plan is parked behind a proof gate. Keep `docs/3d-scene-composition-pipeline.md` as a design reference, but do not treat it as the next implementation path unless a small Windows-stable ID-pass proof beats a matched 2.5D dataset on reviewed real labels.

Do not spend more broad effort on public dataset hunting unless a targeted lead appears. Recent sweeps did not find a better KHR partial/fan source than the existing Roboflow lead plus rights-clear phone captures.

OCR is optional auxiliary evidence only. Current Khmer OCR cues are too noisy to drive the detector path.

## Working Rules

- Use `ideas.md` as the short living board. Keep the top dashboard current.
- Use `docs/synthetic-harness-runbook.md` for 2.5D generation commands.
- Use `scripts/bench_train_with_headroom.py` or another headroom wrapper for long/heavy jobs.
- Never train on `data/real_fan_benchmark/`.
- Do not pass a mined background bank into training until its contact sheet has no visible banknote fragments.
- Treat ambiguous synthetic fragments as `banknote_unknown` or ignore, not forced denomination labels.

## Immediate Milestones

### M1: Clean Asset Atlas

Rebuild or audit scan/cutout assets for current, rare, and old/common KHR scopes. The base model needs enough clean visual evidence for weak classes before hard occlusion is useful.

### M2: 2.5D Base Domain Data

Generate a modest clean/near-clean 2.5D dataset with phone-style scene augmentation and QA'd backgrounds. This is for base recognition strength, not fan counting yet.

### M3: Base Detector Probe

Train from fresh YOLO26n-family weights under the headroom harness. Evaluate clean validation, weak KHR classes, and deployable ONNX/browser smoke before any hard-case fine-tune.

### M4: Reviewed Fragment Loop

Review the compact P1 old/common failure queue first, build a trusted fragment-classifier refresh only from accepted rows, and verify against both the P1 diagnostic set and the real shop-overlap draft. Keep clean and near-clean detector data in the mix so the model does not forget full-note recognition.

### M5: 3D Decision

If 2.5D transfer stalls for geometry-specific reasons, build the minimal WebGL/3D proof: one curled note, two overlapping notes, one fan, one occluder, visual pass, ID pass, visible boxes, OBB, masks, crops, and deterministic Windows reruns.
