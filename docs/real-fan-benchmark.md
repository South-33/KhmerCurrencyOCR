# CashSnap Real Fan Benchmark

Purpose: measure whether CashSnap can count fanned, overlapped, hand-held currency photos, not just curated single-note or synthetic validation scenes.

## Current Seed

- `real_fan_0001_voa_commons`: copied locally to `data/real_fan_benchmark/images/candidates/real_fan_0001_voa_commons.jpg`
- Source page: https://commons.wikimedia.org/wiki/File:Banknotes_of_Cambodian_Khmer_Riel.jpg
- Source image: https://upload.wikimedia.org/wikipedia/commons/7/77/Banknotes_of_Cambodian_Khmer_Riel.jpg
- Status: candidate, unlabeled.
- Rights caveat: the Wikimedia page marks the file public domain as Voice of America material, but also shows a 2026 deletion request related to Cambodian banknote copyright. Keep it as a local benchmark seed unless rights are rechecked.
- `real_overlap_0002_commons_museum`: lower-priority real photographed multi-note scene, copied locally to `data/real_fan_benchmark/images/candidates/real_overlap_0002_commons_museum.jpg`
- Source page: https://commons.wikimedia.org/wiki/File:Cambodian_Riel.jpg
- Source image: https://upload.wikimedia.org/wikipedia/commons/3/30/Cambodian_Riel.jpg
- Status: candidate, unlabeled.
- Rights caveat: the Wikimedia page marks the file CC0, but also shows the same 2026 deletion request family related to Cambodian banknote copyright. Use locally until rights are rechecked.

## Labeling Rule

Use modal/visible-region boxes for the current YOLO detector:

- One box per visible bill slice.
- Class is denomination only, using the existing CashSnap class IDs.
- Tight box around visible pixels, not the estimated full hidden note.
- If a slice is too ambiguous to identify by denomination, skip it and record the ambiguity in notes rather than adding noisy labels.

Do not train on benchmark images. Keep them as validation/test-only assets.

## Promotion Criteria

Move a candidate image into the benchmark only when:

- Source and rights are recorded in `manifests/real_fan_benchmark_sources.csv`.
- Labels are manually checked, not copied directly from model predictions.
- The image adds coverage: fan, overlap, hand occlusion, off-frame notes, mixed USD/KHR, or rare KHR denominations.

## Current Model Read

- e2: `runs/cashsnap/yolo26n_messy_v3_pristine_overlap_e2_i416_b8/weights/best.pt`
- e4: `runs/cashsnap/yolo26n_messy_v3_pristine_overlap_e4_i416_b8/weights/best.pt`
- e4 is the better stress-image candidate so far, but neither e2 nor e4 solves real fan counting.
- Draft e4 hints for the current candidates live under `data/real_fan_benchmark/drafts/e4_i640_c0p05_candidates/`; use them only as annotation starting points, not ground truth.
