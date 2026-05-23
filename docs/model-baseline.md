# CashSnap YOLO26n Baseline

## Run

- Model: `yolo26n.pt`
- Dataset: `data/cashsnap_v1/data.yaml`
- Command: `python scripts/train_yolo.py --model yolo26n.pt --data data/cashsnap_v1/data.yaml --epochs 20 --imgsz 416 --batch 16 --name yolo26n_baseline_e20_i416`
- Saved run: `D:\Project\KhmerCurrencyOCR\runs\ultralytics_migrated\detect\runs\cashsnap\yolo26n_baseline_e20_i4162`
- Best weights: `D:\Project\KhmerCurrencyOCR\runs\ultralytics_migrated\detect\runs\cashsnap\yolo26n_baseline_e20_i4162\weights\best.pt`

Note: early runs were created under `C:\Users\Venom\runs`; they were moved into `runs/ultralytics_migrated/` to keep training artifacts on the D drive.

## Validation

Final validation on the curated validation split:

- Precision: `0.930`
- Recall: `0.913`
- mAP50: `0.962`
- mAP50-95: `0.920`

Weak validation classes:

- `KHR_20000`: recall `0.795`, mAP50 `0.868`
- `KHR_50000`: recall `0.591`, mAP50 `0.841`

## Test Split

Held-out test split:

- Precision: `0.958`
- Recall: `0.903`
- mAP50: `0.972`
- mAP50-95: `0.935`

Weak test classes:

- `KHR_20000`: recall `0.567`, mAP50 `0.884`
- `KHR_50000`: recall `0.524`, mAP50 `0.912`

## Fan Photo Check

Image: `D:\Download\Banknotes_of_Cambodian_Khmer_Riel.jpg`

- At `imgsz=416`, `conf=0.25`: no detections.
- At `imgsz=640`, `conf=0.05`: 3 low-confidence detections, not enough for counting.

Conclusion: the baseline is good for isolated/normal banknote photos in the curated datasets, but it is not ready for the real target scene: fanned, overlapping, hand-occluded KHR notes. The next dataset step should be a small real fan/overlap validation set plus targeted synthetic fan augmentation from clean note crops.

## Synthetic Fan Experiments

Generated KHR synthetic overlap/fan data with `scripts/generate_synthetic_fan_dataset.py`.

- v2 mixed synthetic: `yolo26n_messy_synth_e10_i416`
  - Validation: precision `0.955`, recall `0.929`, mAP50 `0.976`, mAP50-95 `0.933`
  - Test: precision `0.930`, recall `0.961`, mAP50 `0.973`, mAP50-95 `0.934`
  - Fan image: `0` detections at `416/conf=0.25`; `4` detections at `640/conf=0.05`
- v3 dense fan synthetic: `yolo26n_messy_synth_v3_e8_i416`
  - Best early checkpoint improved the fan image: `1` detection at `416/conf=0.25`; `8` detections at `640/conf=0.05`
  - Later epochs trended sideways/down on clean validation, so the run was stopped after epoch 4/early 5.
  - Current best fan checkpoint: `D:\Project\KhmerCurrencyOCR\runs\ultralytics_migrated\detect\runs\cashsnap\yolo26n_messy_synth_v3_e8_i416\weights\best.pt`
- v4 partial-slice synthetic: `yolo26n_messy_synth_v4_e4_i416_w0`
  - Validation: precision `0.926`, recall `0.942`, mAP50 `0.975`, mAP50-95 `0.925`
  - Test: precision `0.960`, recall `0.924`, mAP50 `0.981`, mAP50-95 `0.937`
  - Fan image regressed versus v3: `0` detections at `416/conf=0.25`; `6` detections at `640/conf=0.05`

Conclusion: dense synthetic overlap helps the target failure case, but synthetic-only data has not solved fanned KHR counting. The next highest-value data step is real fanned/overlapped phone photos with labels, especially for `KHR_20000` and `KHR_50000`. Real YOLO crops were extracted, but the generated v5 audit showed rectangular background artifacts; do not train on real-crop synthetic until masking/segmentation is improved.

## Rare KHR Coverage Probe

The original merged train split has only `17` `KHR_20000` boxes and `18` `KHR_50000` boxes. Based on the rare-KHR research PDFs, the minimum visual families to cover are:

- `KHR_20000`: 1995, 2008, and 2017-dated / 2018-issued.
- `KHR_50000`: 1995/1998-style, 2001, and 2013-dated / 2014-issued.

Added a bridge synthetic dataset at `data/synthetic/khr_rare_v1/` using NBC and Numista references:

- Source filter kept `13` whole-note references after aspect-ratio and specimen filtering.
- Train: `950` images, `3,876` `KHR_20000` boxes, `6,234` `KHR_50000` boxes.
- Val: `130` images, `522` `KHR_20000` boxes, `830` `KHR_50000` boxes.
- Test: `120` images, `451` `KHR_20000` boxes, `741` `KHR_50000` boxes.

Visual audit improved after excluding square-ish Numista security-detail closeups, but the dataset still has jagged catalog-mask artifacts. Treat `khr_rare_v1` as a bridge for short rare-class probes only. The better path remains PicWish/manual transparent cutouts scored by `scripts/score_transparent_cutouts.py`.

Attempted `configs/cashsnap_v1_plus_khr_rare.yaml` fine-tuning from the baseline at `416`, but stopped early because the full combined run was too slow for quick iteration. Use a smaller probe or wait for scored PicWish cutouts before a longer combined fine-tune.

## Roboflow Model Comparison

Checked the relevant Roboflow Universe projects:

- `Khmer-US-currency`: public page shows `Model 1`, but direct hosted endpoints tested at versions `1`, `2`, `3`, and `10` returned `403 Forbidden` with the local key.
- `Cambodia Currency Project`: public page shows `Model 1`; direct hosted endpoint `cambodia-currency-project/2` was accessible.
- `KHMER SCAN`: public page shows `Model 0`; direct hosted endpoints tested at versions `1` and `2` returned `403 Forbidden`.

The accessible Roboflow endpoint, `cambodia-currency-project/2`, returned zero predictions on:

- the real fan photo, even at `confidence=5`
- 155 held-out KHR test images at `confidence=25`
- several images originating from the Cambodia Currency Project subset, even with confidence values from `0` through `25`

Because the endpoint returns no predictions even on its own-source images, treat it as not operational or not comparable through the direct hosted API path. It should not be interpreted as a reliable quality benchmark for the underlying Roboflow training run.

On the same 155-image KHR test subset for the six classes shared with the accessible Roboflow endpoint (`KHR_500`, `KHR_1000`, `KHR_5000`, `KHR_10000`, `KHR_20000`, `KHR_50000`), the local YOLO26n baseline at `conf=0.25` produced:

- True positives: `148`
- False positives: `8`
- False negatives: `9`
- Precision: `0.949`
- Recall: `0.943`
- F1: `0.946`
