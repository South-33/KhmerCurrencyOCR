# CashSnap Data Prep

## Downloaded Buckets

- `data/raw_datasets/hf_usd_side_coco_annotations/`: Hugging Face USD Side Detection Dataset. This is the main downloadable USD detection seed and still needs label remapping from front/back/authentic/counterfeit variants into denomination-only classes.
- `data/reference/usd_wikimedia/`: clean USD front/back reference scans from Wikimedia Commons, intended for synthetic generation experiments. Public release must follow U.S. currency reproduction rules.
- `data/reference/khr_nbc/`: official National Bank of Cambodia reference images from the Banknotes in Circulation page. Use for visual reference and synthetic experiments only after checking reuse rights; some notes may contain SPECIMEN watermarks.
- `data/cashsnap_v1/`: merged local YOLO dataset from the multi-source preparation pipeline. This directory is intentionally ignored by git because it is large/generated.

## Manual Or Blocked Sources

See `manifests/blocked_or_manual_sources.csv`.

Roboflow datasets are useful but need manual export or an API key:

- Khmer-US-currency
- Cambodia Currency Project
- KHMER SCAN

Recommended export format is YOLOv8/YOLO26 TXT or COCO JSON. After export, place each source under `data/raw_datasets/roboflow_<source_name>/`.

Numista is not bulk-downloaded because licensing appears mixed/unclear. Use it for research and source discovery until rights are verified.

## Prep Rules

- Keep raw downloads unchanged.
- Put derived files under `data/processed/`.
- Keep synthetic generated scenes separate from real validation/test photos.
- Do not train directly on USD raw reference scans if a public dataset release is planned; generate size-compliant, one-sided derivative images and document compliance.
- Do not train heavily on KHR images with SPECIMEN watermark unless the watermark is removed or strongly augmented away.

## Real Fan Benchmark Rules

- Before more synthetic tuning, create a small real fanned/overlapped phone-photo validation set and keep it out of training.
- Label each visible bill slice separately by denomination. Use tight boxes around the visible region for the current YOLO detector; do not estimate hidden full-note extents unless the project deliberately switches to amodal labels.
- Include easy, medium, and hard scenes: flat overlaps, hand-held fans, hand occlusion, partial off-frame notes, mixed KHR/USD, and weak rare classes (`KHR_20000`, `KHR_50000`).
- Track e2/e4 and future checkpoints on this fixed set before treating synthetic validation gains as real progress.
- Consider YOLO OBB or segmentation only after the real benchmark exists; rotated/mask labels are promising for fan slices, but they need consistent labels and a comparable scoreboard.

## Current Verification Notes

- `scripts/check_yolo_dataset.py --data configs/cashsnap_v1.yaml` reports 9,048 valid boxes after repairing segmentation rows into detection boxes.
- The merged dataset has 14,036 train images, 2,103 validation images, and 1,562 test images.
- Weakest v1 classes by box count are `KHR_20000` and `KHR_50000`; keep them in v1, but collect or synthesize more examples before trusting production-like results.
- `data/audit/` contact sheets are generated for visual QA and intentionally ignored by git.
