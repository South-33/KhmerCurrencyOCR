# CashSnap 3D Synthetic Data Pipeline

Status: active design reset as of 2026-05-30.

## Goal

Build a scalable 3D synthetic data factory for CashSnap before the next training push. The purpose is not "make pretty renders"; it is to generate large amounts of realistic, perfectly labeled phone-like banknote scenes from trusted Numista scans so the model can learn visible partial notes, overlaps, fans, and hand occlusions without hand-labeling every case.

The pipeline should become the project's scaling unit:

1. Numista/current scans define canonical note designs, sides, issue metadata, and clean texture source.
2. 3D scene generation creates physical camera views with bends, folds, dirt, occlusion, lighting, and backgrounds.
3. ID/mask passes create exact visible labels.
4. QA checks prove that labels and rendered images are sane.
5. Only after the generator passes those gates do we train from a fresh base model.

Real phone photos remain the ruler, not the primary data source. They are used to calibrate and validate the 3D generator, not to replace it.

## North-Star Bet

If the 3D generator can model the real distribution well enough, then scaling generated scenes should scale model quality. That gives CashSnap a repeatable path instead of relying on small, fragile public datasets.

The generator is worth scaling only if a small proof shows transfer to scoreable real photos. Perfect labels are necessary but not sufficient; the rendered image distribution also has to be close enough to real phone photos.

## What "Perfect Synthetic Data" Means Here

Perfect data does not mean every render is photorealistic in a cinematic sense. It means every generated sample has:

- trusted note identity and side metadata
- correct current/legacy scope metadata
- physically plausible note pose and camera projection
- visible-only labels, not amodal hidden-note labels
- exact instance masks from an ID pass
- exact detect boxes, OBB boxes, and optional segmentation masks
- no forced denomination labels for visually ambiguous fragments
- camera/postprocess variation that resembles real phone captures
- QA artifacts that let us reject bad scenes before training

## Inputs

### Banknote Assets

Primary source:

- `data/numista_raw/`
- `data/asset_candidates/numista_current_cutout_bank_v1/`

Rules:

- Prefer Numista `in_circulation` folders for issue year, side, and design metadata.
- Keep front/back explicit.
- Keep issue year/design explicit even though CashSnap labels are denomination-only.
- Do not silently mix collector, specimen, out-of-scope, or unknown public-data notes into canonical training.
- Roboflow/Commons/public phone data is review/domain-stress data until design and circulation scope are checked.

### Scene Assets

Useful scene sources:

- real tabletop/background photos already in the repo after contact-sheet QA
- 360/HDRI environment maps for lighting
- downloaded or local 3D rooms/tables/counters if their license is acceptable
- simple procedural room/table primitives when full 3D rooms are unnecessary

Scene assets must have source/license metadata if committed or released. For internal local probes, keep downloaded heavy assets under ignored `data/` paths.

### Camera Profiles

The generator should support named camera profiles, starting with approximate profiles rather than pretending exact proprietary phone optics are known:

- `iphone_8_like`
- `iphone_12_wide_like`
- `budget_android_wide_like`
- `browser_upload_resized`

Each profile should define:

- image resolution or resize target
- focal length / field of view
- sensor aspect ratio
- radial distortion coefficients
- rolling/handheld rotation jitter
- noise/sharpening/JPEG profile
- white balance and exposure behavior

If exact public camera calibration data is found later, add it as metadata. Until then, profiles are calibrated approximations.

## Renderer Requirements

### Geometry

Represent each note as a subdivided rectangular mesh, not a single rigid quad.

Local coordinates:

```text
x in [-L/2, L/2]
y in [-W/2, W/2]
u = (x / L) + 0.5
v = (y / W) + 0.5
```

Base displacement:

```text
z_base(x, y) =
  curl_x * (abs(x) / (L/2))^p_x +
  curl_y * (abs(y) / (W/2))^p_y
```

Crease displacement:

```text
d = dot([x, y] - crease_origin, crease_normal)
z_crease = crease_amp * exp(-(d / crease_width)^2) * crease_side_falloff
```

Low-frequency paper ripple:

```text
z_ripple =
  a1 * sin(f1*x + phase1) * sin(g1*y + phase2) +
  a2 * noise2d(x, y)
```

Final vertex:

```text
P_local = [x, y, z_base + z_crease + z_ripple]
P_world = R * P_local + T
```

The first proof does not need full cloth simulation. Algebraic bends, local creases, and z-order are enough to test transfer. Cloth/soft body physics can come later if the proof passes.

### Layouts

Minimum layout modes:

- `single_table`: 1-3 mostly visible notes for clean/base replay.
- `simple_overlap`: 2-5 notes with realistic partial occlusion.
- `shop_stack`: 4-10 notes in rows or piles, similar to counter photos.
- `radial_fan`: notes share a grip/pivot point and rotate through an angle sweep.
- `hand_fan`: radial fan plus finger/hand occlusion near the grip.
- `edge_partial`: notes clipped by image borders.

Each scene must store layout metadata:

- note instance id
- class
- side
- issue/year/design
- full mesh corners
- visibility ratio
- occlusion source
- label eligibility tier

### Materials

Each banknote needs material variation:

- diffuse texture from scan
- roughness map
- normal/bump map from scan luminance plus procedural paper fibers
- optional dirt/stain overlays
- optional fold/crease darkening
- slight edge wear and corner rounding
- color fade / saturation loss
- specular wash for glossy/laminated or worn paper areas

Material parameters should be sampled from named presets:

- `pristine_scan_like`
- `normal_circulated`
- `worn_shop_note`
- `dirty_folded_note`
- `washed_out_flash`

Do not assume pristine scans are enough. Clean scan renders are useful for base recognition, but the hard partial cases need circulated-note domain variation.

### Hands And Fingers

Use this order of complexity:

1. Real segmented hand/finger patches composited into the render with depth and shadow metadata.
2. Simple 3D finger primitives with skin material, nail/crease texture, and soft shadows.
3. Parameterized 3D hand mesh only after the simpler occluders fail a real benchmark gate.

Finger occlusions must be label-aware: a finger can hide a note region, but hidden pixels must not contribute to the visible mask.

### Lighting And Environments

Lighting modes:

- diffuse indoor room light
- warm shop bulb
- daylight window side light
- phone flash / harsh specular
- mixed color temperature

Use HDRI/360 environment maps where practical, but do not require them for every proof scene. A simple table plane plus calibrated lights is acceptable for the first proof if it produces useful transfer.

The renderer must produce contact shadows between notes and between fingers and notes. Contact shadows are one of the main reasons to use 3D instead of pure 2D layering.

### Camera And Phone Postprocess

Each render should pass through a camera/postprocess stage:

- perspective projection from phone-like FOV
- optional radial distortion
- slight motion blur
- depth/focus blur when useful
- exposure and white balance shifts
- sensor noise / ISO grain
- sharpening/denoising artifacts
- JPEG compression
- resize/canvas preprocessing profiles matching the browser stack

Keep raw render, postprocessed image, and metadata links so failures can be diagnosed.

## Label Pipeline

Render at least two passes:

### Visual Pass

Full texture/material/lighting/camera render used as the training image.

### ID Pass

Flat, unlit, no-antialias instance-color render. Each note instance and hand occluder gets a stable unique id color.

The ID pass is the source of truth for:

- visible instance masks
- visible pixel area
- visible bounding boxes
- OBB/min-area boxes
- segmentation polygons
- crop exports for fragment verifier
- unknown/ambiguous fragment candidates

Rules:

- Never label hidden full-note extents as visible detections.
- Drop or mark instances below a minimum visible-pixel threshold.
- If a visible region lacks denomination evidence, route to `banknote_unknown`/ignore for verifier work, not to a denomination class.
- Preserve masks even when training a detect-only model so later OBB/segmentation/verifier paths can reuse the exact same scenes.

## Output Dataset Contract

Each generated dataset should include:

```text
data/synthetic/<name>/
  data.yaml
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  masks/train/*_id.png
  masks/val/*_id.png
  crops/train/<class>/*.jpg
  metadata.csv
  scene_config.json
  qa/
    contact_sheet.jpg
    mask_overlay_contact.jpg
    label_stats.json
```

`metadata.csv` should include at least:

- image path
- instance id
- class
- side
- issue/year/design
- source asset
- layout mode
- camera profile
- material preset
- visibility ratio
- occlusion tier
- label eligibility
- bbox
- obb
- mask path

## QA Gates Before Training

No generated dataset should be used for training until these pass:

1. Contact sheet looks like plausible phone photos, not flat scan stickers.
2. Mask overlay contact sheet has correct visible-only masks.
3. Label stats show no extreme class imbalance unless intentional.
4. Visibility histogram includes clean, partial, and hard slices in intended ratios.
5. Ambiguous fragments are ignored or unknown, not forced denomination labels.
6. Dataset checker passes.
7. A small sample is manually inspected.
8. The proof dataset beats or ties a matched 2.5D dataset on scoreable real labels without regressing clean validation.

## Proof Milestone

Before any large generation or training:

### P0: Renderer Smoke

Generate 20 scenes:

- `KHR_5000` and `KHR_10000`
- front and back
- single, simple overlap, and shop stack
- basic bends/curls
- visual pass
- ID pass
- detect labels
- mask overlays

Success:

- nonblank renders
- aligned textures
- visible masks match the visible note pixels
- labels are not amodal
- deterministic rerun with same seed

### P1: Transfer Proof Dataset

Generate 100-300 scenes:

- targeted `KHR_5000` portrait-plus-5000 overlap
- confusing `KHR_10000` front/back views
- `KHR_20000` thin/edge cases
- camera profiles: iPhone-like wide, older phone-like, browser-resized
- realistic postprocess

Compare:

- matched 2.5D dataset
- current two-stage baseline
- scoreable real shop-overlap labels
- clean/base validation

Success:

- real scoreable partial metrics improve
- clean validation does not materially regress
- browser/export smoke still passes

Only after P1 succeeds should the project generate thousands of scenes.

## Scaling Plan

If P1 passes:

1. Expand class coverage to all CashSnap USD/KHR classes.
2. Add more Numista issue/year variants from trusted `in_circulation` folders.
3. Add more camera profiles and real background/HDRI environments.
4. Add hand/finger occlusion curriculum.
5. Add OBB and segmentation exports.
6. Generate a curriculum:
   - 35% clean/near-clean current notes
   - 25% simple overlaps
   - 20% shop stacks/counter spreads
   - 10% hand fan/finger occlusion
   - 10% hard partial/edge/off-frame slices
7. Train from a fresh base model, not contaminated historical checkpoints.

## Open Implementation Choice

Preferred first implementation:

- Python or Node renderer is acceptable.
- The first proof may use browser-less Python/OpenCV projection if it can produce exact ID masks and deterministic labels faster.
- Promote to Three.js/WebGL when lighting/shadows/materials become the bottleneck.

Do not choose a renderer for prestige. Choose the smallest renderer that proves transfer and can scale cleanly.
