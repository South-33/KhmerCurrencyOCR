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

## Tried And Learned

- Baseline YOLO26n is strong on curated splits but fails the real fanned KHR photo at normal confidence.
- Synthetic v2 helped slightly: fan image improved only at permissive `640/conf=0.05`.
- Dense fan synthetic v3 is the best fan checkpoint so far: still weak, but clearly better than baseline/v2/v4 on the real fan photo.
- Partial-slice-heavy v4 improved normal test metrics but regressed on the real fan photo, so pure slice training overcorrects.
- Real YOLO crop compositing needs better masks first; naive real crops carried rectangular background artifacts and should not be trained as-is.
- `khr_rare_v1` adds minimum synthetic coverage for `KHR_20000`/`KHR_50000`, but is only a bridge; catalog masks still create artifacts, so PicWish-quality transparent assets should replace it for serious training.
- Mixed Asian-currency datasets are useful for general rotation/background augmentation, but should not be counted as proof of Cambodian old/new KHR version coverage.

## Data Gaps

- `KHR_20000` and `KHR_50000` remain rare/weak, especially old/new version coverage.
- Research PDFs in the repo root indicate `KHR_20000` 1995 and `KHR_50000` 2001 are notable hard-to-source designs.
