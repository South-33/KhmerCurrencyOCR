# Real Fan Failure Analysis

Goal: understand why the current CashSnap detector still fails the real hand-held KHR fan image before adding more synthetic data.

## Current Evidence

Hard image:

- `data/real_fan_benchmark/images/candidates/real_fan_0001_voa_commons.jpg`
- Size: `1023x575`
- Best current stress candidate: `runs/cashsnap/yolo26n_messy_v3_pristine_overlap_e4_i416_b8/weights/best.pt`

Diagnostic CSVs are local/ignored under `data/real_fan_benchmark/diagnostics/`.

Key observations:

- At `imgsz=640/conf=0.25`, e4 returns only `4` detections.
- At `imgsz=640/conf=0.05`, e4 returns `7` detections.
- At `imgsz=640/conf=0.01`, e4 returns `20` detections, but many are large, weak boxes rather than clean bill-slice boxes.
- Median predicted box area is about `17-20%` of the full image across the useful settings, which is too large for many individual visible slices in a dense fan.
- Increasing to `imgsz=960` makes detections worse, not better.
- Changing NMS IoU from `0.30` through `0.90` does not change the detection counts, so NMS is not the main failure.
- Left/center/right crops do not recover dense per-slice predictions; the top-fan crop helps confidence slightly but still produces only a few large boxes.

## Working Hypotheses

### H1: Label/Geometry Mismatch Is The Main Failure

The detector has learned large visible-note or fan-region boxes, not narrow visible bill slices in a hand-held radial fan.

Evidence:

- Current predictions cover broad fan chunks.
- Synthetic train labels are large: `khr_messy_v3` median box area is `23.096%`; `khr_rare_pristine_overlap_v1` median box area is `26.574%`.
- The real fan has many narrow, parallel slices with severe overlap and repeated back-side patterns.

Next knockdown test:

- Build a small synthetic `radial_slice_v1` set with many narrow visible slices, pivoted like a hand-held fan, and visible-region labels.
- Validate against the fixed real fan stress image and the normal CashSnap val/test split.

### H2: Synthetic Appearance Is Still Too Fake

The current synthetic images are useful, but many look like flat piles with cutout edges, specimen artifacts, artificial fingers, and random rotations. The real target is a photographed, perspective-compressed hand fan with fingers and many similar backs.

Evidence:

- Visual audit of `khr_messy_v3` and `khr_rare_pristine_overlap_v1` shows pile/collage composition more than ordered hand fan composition.
- e4 improves rare-overlap synthetic validation but still fails real fan counting.

Next knockdown test:

- Generate a radial fan curriculum from the best transparent assets: ordered pivots, shared bottom grip point, perspective/shear, finger masks near the pivot, and muted phone-photo color.

### H3: Confidence Calibration Is Secondary

Lowering confidence reveals more candidates, but not enough reliable per-slice boxes.

Evidence:

- e4 has `20` detections at `conf=0.01`, `7` at `0.05`, and `4` at `0.25`.
- Low-confidence boxes are still broad and class-confused, not merely suppressed correct slices.

Next knockdown test:

- Keep low-confidence diagnostic outputs as annotation hints, but do not treat threshold tuning as a production fix.

### H4: Resolution/Tiling Is Not The Primary Fix

If resolution were the main issue, larger `imgsz` or local crops would improve detection density.

Evidence:

- `imgsz=960` performs worse than `640`.
- Left/center/right crops return fewer boxes and oversized regions.
- Top-fan crop improves confidence slightly but not enough for counting.

Next knockdown test:

- Do not prioritize browser/mobile tiling until slice-level training improves.

### H5: Deployment Is Not The Blocking Problem Yet

ONNX/NCNN export already works for the balanced checkpoint. The browser/phone path matters, but the present blocker is recognition quality on real fan geometry.

Evidence:

- `docs/mobile-export.md` records ONNX and NCNN export smoke results.
- The same failure appears before export, directly in PyTorch inference.

Next knockdown test:

- Once real fan recall improves in PyTorch, rerun the same diagnostic script on ONNX/NCNN/mobile exports.

## Immediate Plan

1. Freeze a tiny real benchmark: manually label `real_fan_0001_voa_commons` first using visible-region boxes, then add a few more real phone fan scenes.
2. Add a radial fan synthetic generator mode that deliberately creates narrow, ordered, hand-held bill slices instead of random overlap piles.
3. Train a small capped probe from e4 or e2 and evaluate in this order: normal val/test, synthetic radial-slice val, real fan stress diagnostics.
4. Only then revisit browser/phone-specific optimizations.
