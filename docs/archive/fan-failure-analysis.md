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
- Fresh current-KHR reset probes confirm the geometry issue: `current_fan_slice_v1_e5` still has zero hard-fan detections at `conf>=0.05`, and its synthetic labels remain large (`~18-20%` median area).
- `thin_radial_slice` lowers synthetic median box area to about `5.2%`; detect hybrid training gets one hard-fan detection at `640/conf=0.05`, but it is still an oversized `26%` fan-region box.
- OBB thin-slice training gives plausible small rotated boxes on the lower-priority overlap candidate, but the class labels are noisy and the hard fan remains unsolved; OBB is a promising probe, not a selected architecture.
- The hard VOA fan image itself has many backs/ambiguous fragments, so treat it as a stress probe rather than the only denomination-counting scoreboard.

## Working Hypotheses

### H1: Label/Geometry Mismatch Is The Main Failure

The detector has learned large visible-note or fan-region boxes, not narrow visible bill slices in a hand-held radial fan.

Evidence:

- Current predictions cover broad fan chunks.
- Synthetic train labels are large: `khr_messy_v3` median box area is `23.096%`; `khr_rare_pristine_overlap_v1` median box area is `26.574%`.
- The real fan has many narrow, parallel slices with severe overlap and repeated back-side patterns.

Next knockdown test:

- Build or capture a rights-clear real benchmark where visible regions are human-identifiable by denomination, then keep using thin/OBB synthetic probes against that benchmark.

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

ONNX/NCNN export already works for the balanced and circulated-design checkpoints. The browser/phone path matters, but the present blocker is recognition quality on real fan/overlap geometry.

Evidence:

- `docs/mobile-export.md` records ONNX and NCNN export smoke results.
- The same failure appears before export, directly in PyTorch inference.
- `yolo26n_cashsnap_current_thin_legacy_v1_e20_i416_b8` exports to ONNX and NCNN and smoke-predicts, but it still misclassifies several old-design notes on `real_overlap_0003_commons_shop_5k_10k_20k`.

Next knockdown test:

- Once real fan/overlap denomination recall improves in PyTorch, rerun the same diagnostic script on ONNX/NCNN/mobile exports.

### H6: Current-Only KHR Scope Is Too Narrow For Circulated Old Notes

The clean current-bank reset is the right foundation for first-pass modern KHR support, but real users may photograph older NBC-listed designs still seen in circulation.

Evidence:

- Current-only and current+thin fresh probes count broad regions but heavily confuse the old-design shop overlap candidate.
- Historical contaminated e2/e4 checkpoints are much better on that candidate, implying broader design coverage matters.
- A controlled legacy-support rebuild from `target_modern_common + legacy_or_low_priority` NBC cutouts improved draft real-overlap recall, while still not reaching reliable denomination counts.

Next knockdown test:

- Keep current-only and circulated-design experiments separate, then add rights-clear real photos for each visible design family instead of letting old/reference variants leak silently into the primary current model.

## Immediate Plan

1. Freeze a tiny real benchmark with rights-clear phone photos whose visible bill regions are human-identifiable; keep `real_fan_0001_voa_commons` as a stress probe unless labels can be assigned confidently.
2. Keep draft-label tooling (`render_yolo_label_preview.py` and `evaluate_real_draft_labels.py`) available for candidate review, but do not train on `data/real_fan_benchmark/`.
3. Use `scripts/build_current_khr_cutout_bank.py --clean` for first-pass current KHR probes; use a separately named bank when explicitly testing circulated legacy support.
4. Generate paired detect/OBB probe datasets from clean banks with `scripts/generate_synthetic_fan_dataset.py`; use `thin_radial_slice` for narrow visible-slice geometry and `--label-format obb --save-visible-masks` for the OBB copy.
5. Train small probes from fresh YOLO26n-family base weights and evaluate in this order: normal val/test, synthetic radial-slice val, real fan/overlap draft diagnostics, export smoke.
6. Compare e2/e4 only as contaminated historical baselines, not as starting checkpoints.
