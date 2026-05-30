# Real Fan Capture Guide

Goal: collect a small rights-clear benchmark that tests CashSnap on phone photos of current KHR/USD bills without training on those benchmark images.

## Shot List

Capture each scene with normal phone distance and lighting:

- 5 single-note KHR photos: front/back mix, normal table background.
- 5 simple overlap photos: 2-4 notes, denomination numerals still visible.
- 5 hand-held fan photos: 6-15 notes, at least some visible slices must show denomination-specific text, numerals, portrait, color, or landmarks.
- 3 hand-occlusion photos: fingers partly cover notes but do not hide all denomination evidence.
- 3 partial off-frame photos: notes clipped by the image border.
- 3 thin/edge partial `KHR_5000` slices and 3 thin/edge partial `KHR_20000` slices, with a front/back mix if possible. These directly target the current old/common classifier's high-confidence partial-slice confusions.
- 5 `KHR_5000` overlap/partial photos where the visible region includes the portrait plus `5000` numerals. This directly targets the current shop-overlap row-6 miss, whose embedding neighbors are all `KHR_10000`.
- Optional mixed scenes: KHR plus USD only after the KHR-only set is captured.

For visual examples of the target thin/edge failure shape, regenerate `data/review/cashsnap_p1_oldcommon_partial_focus_review_v1/contact_sheet.jpg` with `scripts/build_partial_focus_review_queue.py --clean`. Use it as capture inspiration only; collect rights-clear photos rather than copying those public-data crops into the benchmark.

## Capture Rules

- Use current everyday notes first; avoid old collector notes, specimen marks, souvenirs, and museum displays for the main scoreboard.
- Keep faces, IDs, cards, receipts, screens, GPS signs, and other private details out of frame.
- Take the photo yourself or record rights/source clearly; do not scrape random copyrighted images into the benchmark.
- Include at least one image for `KHR_20000` and `KHR_50000` if those bills are available, because they remain weak classes.
- Track new photos in `manifests/real_partial_capture_inventory.csv` and run `scripts/check_capture_requirements.py` to see priority-ranked scene/denomination gaps; it also reports dropped inbox images that still need registration. Missing scene rows print the matching `data/inbox/real_partial_photos/` drop folder. Use `init_capture_inbox.py --write-guides` when setting up folders so the root inbox and each ignored drop folder include local capture and registration notes.

```powershell
rl python scripts/init_capture_inbox.py --dry-run
rl python scripts/init_capture_inbox.py --write-guides
rl python scripts/register_capture_photos.py --images-dir data/inbox/real_partial_photos --scene-type hand_fan --denominations "KHR_5000;KHR_10000" --dry-run
rl python scripts/register_capture_photos.py --images-dir data/inbox/real_partial_photos --scene-type hand_fan --denominations "KHR_5000;KHR_10000"
rl python scripts/register_capture_photos.py --images-dir data/inbox/real_partial_photos/thin_slice_khr_5000 --scene-type thin_slice_khr_5000 --dry-run
rl python scripts/register_capture_photos.py --images-dir data/inbox/real_partial_photos/khr_5000_face_number_overlap --scene-type khr_5000_face_number_overlap --denominations "KHR_5000" --dry-run
rl python scripts/register_capture_photos.py --images-dir data/inbox/real_partial_photos/thin_slice_khr_20000 --scene-type thin_slice_khr_20000 --dry-run
rl python scripts/register_capture_photos.py --images-dir data/inbox/real_partial_photos --recursive --scene-type-from-parent --dry-run
rl python scripts/check_capture_requirements.py
```

Registration validates scene types against `manifests/real_partial_capture_requirements.csv`; pass `--allow-unknown-scene-type` only for unusual captures. When registering recursively from the inbox root, `thin_slice_khr_5000`, `khr_5000_face_number_overlap`, and `thin_slice_khr_20000` folders automatically fill `denominations` if no shared `--denominations` value is supplied.

## Label Rules

- One label per visible bill region, not the hidden full bill.
- Label only if a human can identify the denomination from the visible region.
- Skip ambiguous backs or tiny slices and note the ambiguity instead of guessing.
- Keep labels under `data/real_fan_benchmark/labels/val/` and run `scripts/check_real_fan_benchmark.py`.
- Never train on these benchmark images.
- For quick draft labels, serve the repo and open `http://localhost:8787/demo/labeler/`; draw visible-note boxes, export YOLO TXT, then render/check the labels before promoting them.
- To test existing Khmer OCR as an auxiliary cue, run `scripts/probe_khmer_ocr_cues.py` against reviewed/draft visible-note boxes and inspect the CSV before adding OCR to the model path.

## Proposal Review Loop

Use model proposals as review hints, not as ground truth. Keep draft outputs under `data/real_fan_benchmark/drafts/` and review packs under `data/review/`.

For a folder of new, non-benchmark phone photos, run the full detector -> classifier -> fusion -> review-pack pipeline:

```powershell
rl python scripts/run_capture_review_pipeline.py --images-dir data/inbox/real_partial_photos --recursive --out-dir data/review/real_partial_proposal_review_v1
```

The pipeline points common ML cache environment variables at the repo-local `.cache_runtime/` folder on `D:` so it does not recreate default Torch/Hugging Face caches on `C:`.

```powershell
rl python scripts/build_proposal_review_pack.py --item data/real_fan_benchmark/images/candidates/example.jpg data/real_fan_benchmark/drafts/example_proposals.csv --out-dir data/review/example_proposal_review_v1
```

The generated `review.csv` has blank `review_include`, `review_class`, and `review_notes` columns for human curation. Only reviewed rows should become training or calibration crops, and benchmark images still must not be trained on.

For faster curation, serve the repo and open the static review UI:

```powershell
rl python scripts/build_benchmark_review_index.py
rl python -m http.server 8787
```

Open `http://localhost:8787/data/real_fan_benchmark/review_index.html` for benchmark candidate links, then use `http://localhost:8787/demo/labeler/` to edit visible-region draft boxes.

Open `http://localhost:8787/demo/review/`, load a review CSV such as `/data/review/real_partial_proposal_review_v1/review_pack/review.csv`, mark usable crops, and export the edited CSV.

After review, convert selected rows from non-benchmark capture packs into an ImageFolder classifier dataset:

```powershell
rl python scripts/build_fragment_classifier_from_review_pack.py --manifest data/review/example_proposal_review_v1/review.csv --out data/fragment_classifier_real_partial_reviewed_v1 --clean
```

Use `--include-unreviewed` only for diagnostics or smoke tests, never for final training data.
