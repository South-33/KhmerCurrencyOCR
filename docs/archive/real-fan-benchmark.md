# CashSnap Real Fan Benchmark

Purpose: measure whether CashSnap can count fanned, overlapped, hand-held currency photos, not just curated single-note or synthetic validation scenes.

Capture requirements for new rights-clear phone photos live in `docs/real-fan-capture-guide.md`.

## Current Seed

- `real_fan_0001_voa_commons`: copied locally to `data/real_fan_benchmark/images/candidates/real_fan_0001_voa_commons.jpg`
- Source page: https://commons.wikimedia.org/wiki/File:Banknotes_of_Cambodian_Khmer_Riel.jpg
- Source image: https://upload.wikimedia.org/wikipedia/commons/7/77/Banknotes_of_Cambodian_Khmer_Riel.jpg
- Status: candidate, unlabeled.
- Rights caveat: the Wikimedia page marks the file public domain as Voice of America material, but also shows a 2026 deletion request related to Cambodian banknote copyright. Keep it as a local benchmark seed unless rights are rechecked.
- Labeling caveat: many slices show backs or ambiguous fragments, so this is a hard stress image but not a sufficient denomination-counting oracle by itself. Skip any slice a human cannot confidently identify.
- `real_overlap_0002_commons_museum`: lower-priority real photographed multi-note scene, copied locally to `data/real_fan_benchmark/images/candidates/real_overlap_0002_commons_museum.jpg`
- Source page: https://commons.wikimedia.org/wiki/File:Cambodian_Riel.jpg
- Source image: https://upload.wikimedia.org/wikipedia/commons/3/30/Cambodian_Riel.jpg
- Status: candidate, unlabeled.
- Rights caveat: the Wikimedia page marks the file CC0, but also shows the same 2026 deletion request family related to Cambodian banknote copyright. Use locally until rights are rechecked.
- Labeling caveat: this is a museum/display mix with several legacy or out-of-class denominations, so do not turn it into a current-KHR scoreboard image except for clear in-scope visible notes.
- `real_overlap_0003_commons_shop_5k_10k_20k`: real shop photo with visible 5k/10k/20k notes, copied locally to `data/real_fan_benchmark/images/candidates/real_overlap_0003_commons_shop_5k_10k_20k.png`
- Source page: https://commons.wikimedia.org/wiki/File:Campuchia_-_Ti%E1%BB%81n_Gi%E1%BA%A5y_5000,_10_000,_20_000_rial.png
- Source image: https://upload.wikimedia.org/wikipedia/commons/5/50/Campuchia_-_Ti%E1%BB%81n_Gi%E1%BA%A5y_5000%2C_10_000%2C_20_000_rial.png
- Status: candidate, unlabeled.
- Rights caveat: the Wikimedia page marks the file public domain by self-release, but also shows the same 2026 deletion request family related to Cambodian banknote copyright. Use locally until rights are rechecked.
- Label-quality caveat: draft boxes are human-legible visible regions, but several issue years/designs are older shop notes; use it as a hard local diagnostic, not as the final current-circulation scoreboard until design scope is reviewed.

## Labeling Rule

Use modal/visible-region boxes for the current YOLO detector:

- One box per visible bill slice.
- Class is denomination only, using the existing CashSnap class IDs.
- Tight box around visible pixels, not the estimated full hidden note.
- If a slice is too ambiguous to identify by denomination, skip it and record the ambiguity in notes rather than adding noisy labels.

For draft benchmark labels, maintain `manifests/real_fan_benchmark_label_quality.csv` with one row per box. Use `quality=clear` or `partial_clear` only when a human can identify the denomination from visible evidence; use `ambiguous` or `ignore` when the box should not count toward model scoring. To produce a fair-to-score label file for evaluation, run:

```powershell
rl python scripts/filter_yolo_labels_by_quality.py --labels data/real_fan_benchmark/drafts/real_overlap_0003_commons_shop_5k_10k_20k.txt --out data/audit/real_overlap_0003_commons_shop_5k_10k_20k.scoreable.txt
```

Do not train on benchmark images. Keep them as validation/test-only assets.

Run `scripts/check_real_fan_benchmark.py` after adding candidates or labels. It verifies local image readability, catches manifest/label-status mismatches, and validates YOLO visible-region label files under `data/real_fan_benchmark/labels/val/` once an image is promoted to `labeled`.

## Promotion Criteria

Move a candidate image into the benchmark only when:

- Source and rights are recorded in `manifests/real_fan_benchmark_sources.csv`.
- Labels are manually checked, not copied directly from model predictions.
- The image adds coverage: fan, overlap, hand occlusion, off-frame notes, mixed USD/KHR, or rare KHR denominations.
- The visible regions contain enough denomination evidence for a human to label them; a fan of mostly ambiguous backs is useful as a stress probe, not as the main scoreboard.

## Current Model Read

- e2: `runs/cashsnap/yolo26n_messy_v3_pristine_overlap_e2_i416_b8/weights/best.pt`
- e4: `runs/cashsnap/yolo26n_messy_v3_pristine_overlap_e4_i416_b8/weights/best.pt`
- e4 is the better stress-image candidate so far, but neither e2 nor e4 solves real fan counting.
- Draft e4 hints for the current candidates live under `data/real_fan_benchmark/drafts/e4_i640_c0p05_candidates/`; use them only as annotation starting points, not ground truth.
