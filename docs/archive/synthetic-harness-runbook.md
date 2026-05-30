# Synthetic Harness Runbook

This is the current short path for CashSnap partial-banknote synthetic work. Keep generated data under ignored `data/` paths and train/evaluate through the existing headroom scripts.

## 1. Build A Clean Scan Cutout Bank

```powershell
rl python scripts/build_numista_cutout_bank.py --out data/asset_candidates/numista_current_cutout_bank_v1 --clean
rl python scripts/audit_cutout_bank.py --bank data/asset_candidates/numista_current_cutout_bank_v1
```

Review:

- `data/asset_candidates/numista_current_cutout_bank_v1/contact_sheet.jpg`
- `data/asset_candidates/numista_current_cutout_bank_v1/audit/suspect_contact.jpg`
- `data/asset_candidates/numista_current_cutout_bank_v1/audit/suspects.csv`

Do not treat a red/pink note as a specimen automatically. KHR 500 designs naturally contain large red areas; visual audit is the authority. `build_numista_cutout_bank.py` defaults to rectangular scan alpha so white note regions stay opaque; use `--alpha-mode content` only for non-tight source images.

## 2. Mine Real Background Patches

Use labeled real images to extract square patches that avoid every YOLO note label, then optionally reject candidates that the current detector still recognizes as banknotes. These patches can reduce the synthetic-to-real gap, but they are not trustworthy until the contact sheet is visually clean.

```powershell
rl python scripts/extract_yolo_background_patches.py `
  --data data/cashsnap_v1/data.yaml `
  --out data/backgrounds/cashsnap_v1_no_note_patches `
  --count 500 `
  --output-size 640 `
  --seed 20260530 `
  --max-label-overlap-frac 0 `
  --box-pad-frac 0.45 `
  --reject-model runs/cashsnap/yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8/weights/best.pt `
  --reject-conf 0.02 `
  --reject-imgsz 640 `
  --clean
```

Review:

- `data/backgrounds/cashsnap_v1_no_note_patches/contact_sheet.jpg`
- `data/backgrounds/cashsnap_v1_no_note_patches/manifest.csv`

If the contact sheet includes accidental note fragments, do not use that bank for training. Lower `--reject-conf`, lower `--max-label-overlap-frac`, raise `--box-pad-frac`, or switch to a true note-free background source.

Recent smoke finding: geometry-only strict mining can still leak cropped currency fragments. Detector-rejected patches looked cleaner, but the safe rule is still contact-sheet review before training.

## 3. Generate A Small 2.5D Base Smoke Dataset

The immediate phase is base detector strength, not hard fan counting. Keep this smoke clean or near-clean: 1-3 notes, no synthetic fingers, and only mild overlap. Use the phone-style postprocess knobs to cover camera domain variation without confusing the label geometry.

```powershell
rl python scripts/generate_synthetic_fan_dataset.py `
  --out data/synthetic/smoke_base_scan_phone_v1 `
  --count 48 `
  --image-size 416 `
  --seed 532 `
  --clean `
  --min-notes 1 `
  --max-notes 3 `
  --layout-modes "row,scattered,crossed" `
  --drop-unknown-denom-labels `
  --note-shadow-prob 0.45 `
  --hand-prob 0.0 `
  --balance-classes `
  --perspective-prob 0.55 `
  --camera-geom-prob 0.45 `
  --lens-distort-prob 0.35 `
  --scene-aug-prob 0.85 `
  --jpeg-quality-min 50 `
  --jpeg-quality-max 90 `
  --save-visible-crops `
  --crop-include-unknown
```

Add `--background-dir data/backgrounds/cashsnap_v1_no_note_patches` only after the background contact sheet is clean.

Check:

```powershell
rl python scripts/summarize_synthetic_metadata.py data/synthetic/smoke_base_scan_phone_v1
rl python scripts/check_yolo_dataset.py --data data/synthetic/smoke_base_scan_phone_v1/data.yaml
```

Inspect a few images under `images/train/` and verifier crops under `crops/train/`.

## 4. Scale Only After QA Passes

For the first serious scan-based base probe, keep it modest and mostly clean:

```powershell
rl python scripts/generate_synthetic_fan_dataset.py `
  --out data/synthetic/cashsnap_scan_2p5d_base_phone_v1 `
  --count 3000 `
  --image-size 416 `
  --seed 20260530 `
  --clean `
  --min-notes 1 `
  --max-notes 4 `
  --layout-modes "row,scattered,crossed" `
  --drop-unknown-denom-labels `
  --note-shadow-prob 0.55 `
  --hand-prob 0.0 `
  --balance-classes `
  --perspective-prob 0.55 `
  --camera-geom-prob 0.45 `
  --lens-distort-prob 0.30 `
  --scene-aug-prob 0.75 `
  --jpeg-quality-min 55 `
  --jpeg-quality-max 92 `
  --save-visible-crops `
  --crop-include-unknown
```

Add the reviewed `--background-dir ...` argument when the background bank passes contact-sheet QA.

Use the metadata summary to verify:

- `banknote_unknown` exists but does not dominate denomination labels.
- Class balance is not collapsing toward USD or easy KHR classes.
- Real backgrounds are being sampled from multiple source patches, not one repeated image.
- Visible-area quantiles are closer to partial/fan reality than broad full-note boxes.
- Crops include both identifiable fragments and `banknote_unknown` examples for verifier calibration.

Build the mixed real + scan-synthetic probe lists without silently reusing older synthetic roots:

```powershell
rl python scripts/build_mixed_cashsnap_probe_lists.py `
  --out data/sampled/cashsnap_scan_2p5d_base_probe_v1 `
  --cashsnap-train-count 2400 `
  --cashsnap-val-count 600 `
  --extra-synthetic-root data/synthetic/cashsnap_scan_2p5d_base_phone_v1 `
  --no-default-synthetic `
  --clean
rl python scripts/check_yolo_dataset.py --data configs/cashsnap_v1_scan_2p5d_probe.yaml
```

After the base checkpoint is stable, use the same harness to add harder layouts in stages: `crossed,row`, then `strip_fan`, then `thin_radial_slice`, then `tight_fan,fan` with grip-aware hand occlusion.

## 5. Training Direction

Train detector probes from fresh YOLO26n-family weights, not from contaminated historical e2/e4 checkpoints. Use `scripts/bench_train_with_headroom.py` for long runs so CPU, RAM, and GPU stay under 90%.

Evaluate in this order:

1. Clean validation.
2. Synthetic fan/overlap validation.
3. Reviewed real fan/overlap draft labels.
4. Detector-plus-fragment-verifier fusion.
5. Browser/mobile export smoke only after PyTorch quality improves.

Do not train on `data/real_fan_benchmark/`.
