# CashSnap Data Source Leads

Purpose: keep dataset research actionable. Do not download large datasets just because they exist; use them only when they fill a specific gap in the current benchmark or training mix.

## Highest-Value Current Leads

### Local Numista Raw Cache

- Path: `data/numista_raw/`
- Use: clean scan texture atlas for synthetic generation.
- Builder: `scripts/build_numista_cutout_bank.py`
- Caveat: scans are clean/reference-like, so they need phone-style degradation, hand/fan geometry, shadows, and real benchmark validation before training claims.

### Roboflow `cuurecy-detection-is`

- Repo memory: `data/raw_datasets/roboflow_cuurecy_detection_is/`
- Use: real phone partial/overlap examples and KHR/USD masks.
- Caveat: read `docs/roboflow-cuurecy-detection-audit.md` first; split/lookalike caveats mean it is not a clean validation oracle without curation.

### Hugging Face `ebowwa/usd-side-coco-annotations`

- Link: https://huggingface.co/datasets/ebowwa/usd-side-coco-annotations
- Published summary: 3,618 images, 3,746 annotations, COCO + JSONL, 24 classes for USD denomination/side/authenticity.
- Use: USD side/denomination coverage and USD hard negatives for KHR confusion control.
- Caveat: large at about 4.15 GB and includes counterfeit classes; map to CashSnap denomination/side carefully and avoid counterfeit-specific objectives.

### Zenodo USD Missing Object Dataset

- Link: https://zenodo.org/records/15692324
- Published summary: 232.4 MB dataset zip, CC BY 4.0.
- Use: possible USD occlusion/missing-object patterns.
- Caveat: inspect before integration; title suggests missing-object framing, not necessarily denomination detection or mobile fan scenes.

### BankNote-Net

- Paper: https://arxiv.org/abs/2204.03738
- Repo: https://github.com/microsoft/banknote-net
- Published summary: 24,816 banknote embeddings/images in assistive recognition scenarios, spanning 17 currencies and 112 denominations; embeddings/encoder are the main reusable artifact.
- Use: architectural guidance for detector-plus-embedding/verifier and maybe generic banknote representation.
- Caveat: not a multi-instance detector and likely not a direct KHR/USD partial-fan dataset.

## Lower-Priority Leads

### Kaggle USD Bill Classification Dataset

- Link: https://www.kaggle.com/datasets/aishwaryatechie/usd-bill-classification-dataset
- Use: extra USD crop classifier data if licensing/access is acceptable.
- Caveat: classification crops only; less valuable than detection/partial data.

### Wikimedia Commons Cambodia Banknotes

- Link: https://commons.wikimedia.org/wiki/Category:Banknotes_of_Cambodia
- 2026-05-30 re-check: category lists 29 media files plus a few subcategories, including reference-style banknote scans and collection photos rather than mobile overlap/fan scenes.
- Use: visual design checks and possible reference seeds after rights review.
- Caveat: Cambodian banknote reproduction/copyright concerns exist; use only with explicit rights review before public release or model packaging claims.

### Roboflow Cambodia Currency Project

- Link: https://universe.roboflow.com/khmer-riel-classification-computer-vision/cambodia-currency-project
- 2026-05-30 re-check: object-detection project page reports 552 images, seven KHR classes (`100_Riel`, `500_Riel`, `1000_Riel`, `5000_Riel`, `10000_Riel`, `20000_Riel`, `50000_Riel`), and CC BY 4.0 metadata.
- Use: possible small KHR sanity check or historical/low-denomination reference if its dataset export is audited.
- Caveat: page preview does not establish the targeted `KHR_5000` portrait/number overlap coverage needed now, and it does not replace rights-clear partial/fan phone captures.

### Generic Roboflow Banknote Detection

- Link: https://universe.roboflow.com/valute/banknote-detection-2amdt-l24cn/dataset/3
- Published search summary: 1,595-image generic banknote detection dataset with recent YOLO exports, including a v3 page dated 2026-01-19.
- Use: generic banknote/background robustness or detector preflight only.
- Caveat: not KHR/USD denomination-specific and likely weak for CashSnap counting unless labels and license are manually audited.

## Current Recommendation

Current web re-check on 2026-05-30 did not surface a better public KHR partial/fan source. Hugging Face search did not reveal a KHR banknote dataset, Roboflow search still centers on Cambodia Currency Project / Khmer-US-currency style projects and generic banknote datasets, and Wikimedia Commons remains reference/collector imagery. Keep the next cycle focused on rights-clear KHR phone captures, especially the `khr_5000_face_number_overlap` gap, plus carefully reviewed local Roboflow partial crops.

Do not spend the next cycle downloading broad public banknote datasets. The best path is:

1. Use the local Numista scan bank for controlled synthetic texture coverage.
2. Use Roboflow and reviewed phone captures for real partial/overlap calibration.
3. Add the HF USD side dataset only if USD confusion or side coverage becomes the active bottleneck.
4. Prioritize rights-clear Cambodian phone photos over more reference scans.
5. Treat adjacent public Roboflow banknote datasets as audit candidates, not as the next default download.
