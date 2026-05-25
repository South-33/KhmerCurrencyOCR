# CashSnap Ideas

Living doc for high-value ideas, experiments, and results. Keep this short: only add ideas worth revisiting.

## Strong Ideas To Try

- Build a curated transparent banknote asset library from the best public/Roboflow examples: tight crop, remove background, QA visually, tag denomination/front/back/version.
- Use strong external/background-removal tools to create cleaner transparent PNG assets. Group cutouts are useful for background replacement and style realism, but per-bill synthetic labels still need individual note masks/cutouts.
- Add automatic cutout QA after background removal: keep assets whose alpha mask is mostly rectangular/quadrilateral, flag curved or multi-lobed shapes as likely hands, merged objects, or bad cutouts.
- Use masks internally for synthetic composition, then export the right training label type: upright visible-pixel boxes for normal YOLO, rotated boxes for YOLO OBB, masks for segmentation.
- Train synthetic data as a curriculum: clean single notes, two-note crosses, small overlaps, partial off-frame notes, hand occlusion, then hard fan stacks.
- Use real table/concrete/counter backgrounds and realistic phone effects: shadows, blur, compression, exposure shifts, paper wear, and color grading.
- Create a small real fanned/overlapped validation set before tuning more. One stress image is useful but not enough to tell if a generator really works.
- Explore YOLO OBB after the synthetic asset bank exists; rotated labels become cheap once masks/corners are known.
- Try context-first background removal for tight crop failures: send a larger parent-image crop around the YOLO box to the remover, then crop/score the transparent bill afterward.
- Try local open-source removers after PicWish triage: RMBG-2.0 first, BEN2 second, with fill-holes plus slight dilation so masks do not erase printed details inside the note.

## Tried And Learned

- Baseline YOLO26n is strong on curated splits but fails the real fanned KHR photo at normal confidence.
- Synthetic v2 helped slightly: fan image improved only at permissive `640/conf=0.05`.
- Dense fan synthetic v3 is the best fan checkpoint so far: still weak, but clearly better than baseline/v2/v4 on the real fan photo.
- Partial-slice-heavy v4 improved normal test metrics but regressed on the real fan photo, so pure slice training overcorrects.
- Real YOLO crop compositing needs better masks first; naive real crops carried rectangular background artifacts and should not be trained as-is.
- `khr_rare_v1` adds minimum synthetic coverage for `KHR_20000`/`KHR_50000`, but is only a bridge; catalog masks still create artifacts, so PicWish-quality transparent assets should replace it for serious training.
- Mixed Asian-currency datasets are useful for general rotation/background augmentation, but should not be counted as proof of Cambodian old/new KHR version coverage.
- Gold cutouts alone were still too noisy for first-stage synthesis; filtering by alpha rectangularity/component/aspect produced a cleaner `rare_pristine_asset_bank_v1` for curriculum probes.
- Pristine clean synthetic smoke (`yolo26n_pristine_clean_synth_smoke_e2_i416_b4_mb20`) wrote correctly under repo `runs/` and reached synthetic-val mAP50 0.764 after 20 train batches; `KHR_20000` recall (0.487) still lags `KHR_50000` (0.714).
- Rare post-train (`yolo26n_pristine_clean_rare_e4_i416_b8`) is the first completed 20k/50k fine-tune: normal-val mAP50-95 0.926, `KHR_20000` recall 0.990, `KHR_50000` recall 0.566.
- Rare overlap v1 plus 2-epoch fine-tune (`yolo26n_pristine_overlap_e2_i416_b8`) improved synthetic overlap-val mAP50 from 0.418 to 0.568 and `KHR_50000` overlap recall from 0.262 to 0.439, while normal-val mAP50-95 dipped from 0.926 to 0.920.
- On rare-overlap synthetic validation, `yolo26n_pristine_overlap_e2_i416_b8` beats older fan checkpoints v3/v4 (mAP50 0.568 vs 0.422/0.446), so pristine overlap composition is worth extending beyond rare notes.
- Combining broad fan v3 with pristine rare overlap (`yolo26n_messy_v3_pristine_overlap_e2_i416_b8`) improved both scoreboards: normal-val mAP50-95 0.930 and rare-overlap mAP50 0.584, with `KHR_50000` overlap recall 0.475.
- Continuing that combined checkpoint for 2 more epochs (`yolo26n_messy_v3_pristine_overlap_e4_i416_b8`) improved rare-overlap mAP50 to 0.602 (`KHR_20000`/`KHR_50000` recall 0.535/0.515) and held-out test mAP50-95 to 0.942, but normal-val mAP50-95 fell to 0.925.
- Capped gentle-LR continuation from the combined e2 checkpoint (`yolo26n_messy_v3_pristine_overlap_gentle_lr_probe3_mb100_i416_b4`) avoided the collapse seen in the 300-batch probe, but did not beat e2/e4: normal-val mAP50-95 0.927 and rare-overlap mAP50 0.575.
- On the real fan stress image, combined e2/e4 both improved normal-confidence detection over dense v3 (`2` vs `1` detections at `416/conf=0.25`); e4 is the better rare-slice stress candidate at permissive confidence (`7` detections with `KHR_50000` up to 0.433), but still not reliable counting.
- Raising real fan inference to `imgsz=960` did not help v3/e2/e4; detections dropped versus `640/conf=0.05`, so prioritize data/labels over resolution-only tuning.
- Real fan failure analysis points to label/geometry mismatch, not NMS or tiling: e4 box counts are unchanged across NMS IoU `0.30-0.90`, crops do not recover per-slice detections, and current synthetic labels have large median areas (`23-27%` of image).

## Data Gaps

- `KHR_20000` and `KHR_50000` remain rare/weak, especially old/new version coverage.
- Research PDFs in the repo root indicate `KHR_20000` 1995 and `KHR_50000` 2001 are notable hard-to-source designs.
