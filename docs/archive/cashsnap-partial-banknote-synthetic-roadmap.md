# CashSnap Partial Banknote Synthetic Data Roadmap

## Goal

Build CashSnap into a reliable mixed USD + Khmer Riel banknote counter for real phone photos where notes are overlapped, fanned, partially off-frame, worn, and hand-occluded.

The central bet is that high-quality synthetic data can solve most of the geometry and occlusion problem, but only if the generator is validated against real partial-note photos and only if ambiguous fragments are allowed to stay unknown.

## Decision

Build a renderer-agnostic synthetic evidence harness.

Start with a 2.5D CPU/OpenCV-style renderer because it is easier to debug, deterministic, and already fits the current Python-heavy pipeline. Promote a full 3D/WebGL renderer only if the 2.5D harness fails a real benchmark gate and a WebGL proof of concept proves correct labels, stable rendering, and useful transfer.

The desired product is not "a 3D renderer." The desired product is a data engine that creates images, visible-pixel labels, partial-crop verifier data, QA reports, and real-benchmark gains.

The current sequence is base-model strength first, partial/fan specialization second. Use 2.5D to improve clean and near-clean recognition, weak KHR class coverage, and phone-domain robustness before scaling dense overlaps and hand-held fans.

## What To Believe From The Research

Use these points as strong guidance:

- Visible-region supervision should come first. Train on what the camera can actually see, not hallucinated full-note boxes hidden behind other notes.
- A detector-plus-verifier architecture is the most practical path for partial notes. The detector finds visible note fragments; a tiny classifier or embedding verifier decides denomination or unknown.
- Not every visible fragment deserves a denomination label. A blank edge, backside strip, motion-blurred corner, or finger-covered sliver should be ignored or routed to unknown.
- Classical homography and local-feature work is still useful as a design clue: denomination identity often comes from local high-entropy motifs, not from the whole banknote rectangle.
- Mixed real-and-synthetic data is required. Synthetic can cover geometry breadth, but real phone photos are the ruler.

Treat these points as unproven or secondary:

- "YOLOv11/seg is the answer" is not a project-level conclusion. CashSnap already has YOLO26n-family export paths; architecture should follow benchmark results.
- Full instance segmentation may help, but it should earn its cost against visible boxes/OBB on real fan and overlap scenes.
- Perfect scans do not automatically make perfect training data. They are clean canonical textures, but they must be degraded, warped, worn, recolored, blurred, and mixed with real camera cutouts.
- PBR normals, roughness maps, HDRIs, and path-traced realism are lower priority than correct geometry, labels, hand occlusion, and domain calibration.

## Label Policy

Use modal, visible-region labels for detector training.

- Label the visible pixels of each note fragment, not the estimated full hidden note.
- Export detect boxes, OBB boxes, and optional segmentation masks from the same visible instance mask.
- Store synthetic amodal corners and full-note geometry as metadata only. Do not use them as the primary detector target.
- Store `visibility_ratio`, `side`, `series_or_design_family`, `source_asset`, `scene_type`, `occlusion_type`, and `evidence_tier` per instance.

Use three evidence tiers:

- `identifiable`: enough denomination-specific evidence is visible for a human to label it.
- `banknote_unknown`: a banknote is visible, but denomination evidence is insufficient.
- `ignore`: too tiny, blurred, occluded, or noisy to supervise safely.

If the current detector class schema cannot support `banknote_unknown`, keep ambiguous fragments out of denomination training and use them for verifier/background calibration instead.

## Synthetic Harness Requirements

The harness should produce a complete training artifact, not just rendered JPEGs:

- rendered RGB image
- per-instance visible mask
- YOLO detect labels
- YOLO OBB labels
- optional segmentation labels
- visible crop dataset for the fragment verifier
- metadata CSV/JSONL for every rendered instance
- QA contact sheets
- class, side, visibility, overlap, and crop-evidence statistics

Core scene families:

- clean isolated notes
- table spreads with light overlap
- dense shop-counter overlaps
- ordered radial fans with a shared grip point
- hand-held fans with finger/palm occlusion near the pivot
- off-frame partial notes
- mixed USD/KHR scenes and hard negatives

Core geometry:

- per-note homography and perspective projection
- physical note dimensions and shared camera model
- z-order with visible-mask recomputation
- grid-based paper curl, edge lift, creases, and mild crumple
- variable scale, distance, rotation, and off-frame cropping
- synthetic visibility distribution matched to real benchmark slices

Core optics:

- real table/shop/wallet/background textures
- contact shadows between notes
- grip/finger shadows
- glare and specular wash
- motion blur and defocus blur
- lens distortion and vignette
- exposure, contrast, color temperature, tint, sensor noise, sharpening, and JPEG compression

Core occluders:

- grip-aware capsule/fingertip primitives for cheap variation
- segmented real hand/finger patches for transfer realism
- optional 3D hand mesh only after real-benchmark evidence shows 2D hand patches are the blocker

## Asset Strategy

Use perfect scans as the canonical texture atlas, not as the only training source.

For each denomination and side:

- keep the clean scan or orthophoto
- keep issue/design family metadata
- create phone-style variants: downsampled, compressed, color-shifted, worn, folded, stained, and blurred
- keep real-camera cutouts where available, especially for old/common circulated KHR notes
- reject specimen-marked, copyright-risky, or design-mismatched assets from serious training unless they are explicitly tagged as diagnostic only

KHR support should stay split into named scopes:

- current modern common notes
- current rare notes
- circulated old/common notes
- legacy or low-priority notes

Do not let those scopes leak silently into one "KHR" model. Mix them only when the experiment says so.

## Model Strategy

Train and evaluate these paths in order:

1. Visible-region YOLO26n-family detector.
   - Detect visible banknote fragments.
   - Use denomination labels only for identifiable fragments.
   - Keep OBB as an active probe because fan slices are rotated and narrow.

2. Detector plus fragment verifier.
   - Crop detector proposals with padding.
   - Use a MobileNetV3/EfficientNet-Lite-size classifier or embedding verifier.
   - Predict denomination, currency, background, or unknown.
   - Fuse detector and verifier scores with explicit unknown thresholds.

3. Short-video aggregation.
   - Optional but high leverage for phone UX.
   - Track proposals over 5-15 frames and aggregate verifier evidence when different frames reveal different note regions.

4. Segmentation or full 3D.
   - Promote only if real failure analysis shows boxes/OBB cannot separate visible regions or 2.5D geometry cannot transfer.

OCR and template matching should remain auxiliary cues. Use them only for crops where the relevant numeral, text, portrait, or motif is visibly present.

## Training Curriculum

Keep clean-note skill alive while adding hard geometry.

For the immediate base-strength phase, use a gentler curriculum:

- 60% clean or near-clean notes with realistic phone degradation.
- 25% light table spreads and simple two-note overlaps.
- 15% mild off-frame/partial notes, with hard fans and heavy fingers kept mostly out.

Bias this phase toward weak classes and sides, especially `KHR_20000`, `KHR_50000`, and old/common circulated KHR only when the experiment explicitly includes that scope.

Recommended batch mix for the main robust checkpoint:

- 45% clean or near-clean notes
- 30% simple overlap and shop-counter spreads
- 25% complex fans, dense stacks, off-frame notes, and hand occlusion

Use a short fan-focused fine-tune only after the main checkpoint is stable:

- 35% clean or near-clean notes
- 25% simple overlap
- 40% complex fans and hand-held partials

Include hard negatives and confusing currency/background crops in verifier training. Balance rare KHR classes and sides explicitly, especially `KHR_5000`, `KHR_20000`, and `KHR_50000`.

## Evaluation Gates

Never train on the real fan benchmark.

Evaluate every serious dataset or model against:

- normal clean validation
- synthetic overlap/fan validation
- reviewed real fan/overlap benchmark labels
- real fragment verifier crops
- hard negatives and USD/KHR confusion cases
- mobile/browser export smoke checks after PyTorch quality improves

Primary metrics:

- visible-fragment recall at lenient IoU
- denomination accuracy only on identifiable fragments
- unknown precision on ambiguous fragments
- count error per currency and per denomination
- clean-validation regression from the previous stable checkpoint
- per-class confusion for KHR old/common backs and thin edges

Stop or roll back a synthetic recipe if it improves synthetic validation but regresses clean real validation or increases wrong-denomination confidence on ambiguous real fragments.

## Renderer Promotion Gates

2.5D is enough if it improves real fan/overlap metrics while keeping clean validation stable.

Promote WebGL/3D only if all are true:

- the 2.5D harness has correct visible labels, realistic hand/fan geometry, and calibrated real-style optics
- the real benchmark still shows geometry-specific failures
- a WebGL/3D proof of concept renders nonblank visual frames, exact ID masks, stable z-order, and reproducible labels on Windows
- training on the 3D dataset improves real benchmark results beyond the 2.5D dataset
- the runtime cost and dependency burden do not slow the experiment loop too much

The first WebGL/3D proof should render only:

- one curled note
- two overlapping notes
- one radial fan with a hand occluder
- visual pass plus ID pass
- detect, OBB, segmentation, and crop exports
- contact sheets and label-overlay previews

## Milestones

### M0: Ground Truth Gate

- Freeze a small rights-clear real benchmark with human-identifiable visible regions.
- Keep ambiguous stress images separate from the main denominator-counting scoreboard.
- Define evidence tiers and unknown handling.

### M1: Asset Atlas

- Build a clean scan/orthophoto texture atlas with side, denomination, issue family, and source metadata.
- Add real-camera cutouts for circulated KHR where available.
- Generate phone-style texture variants without losing metadata.

### M2: 2.5D Synthetic Harness

- Implement full-note placement, homography, mesh curl, z-order masks, contact shadows, and grip-aware hand occlusion.
- Export detect, OBB, masks, crops, metadata, and QA sheets.
- Match generated visibility and scene distributions to the real benchmark.

### M3: First Real Transfer Loop

- Train small YOLO26n-family probes under the headroom harness.
- Compare clean validation, synthetic validation, and real fan/overlap metrics.
- Keep current-KHR and circulated-design experiments separate.

### M4: Fragment Verifier

- Build verifier crops from synthetic visible masks and reviewed real proposals.
- Train a tiny classifier or embedding verifier with denomination, currency, background, and unknown outputs.
- Fuse detector and verifier outputs and evaluate count accuracy.

### M5: Renderer Decision

- If 2.5D transfer is strong, continue scaling and QA.
- If 2.5D transfer stalls for geometry reasons, build the minimal WebGL/3D proof and run a direct dataset comparison.

## Open Risks

- Public or scanned banknote imagery may have rights and reproduction constraints. Keep source metadata and avoid using questionable assets in public releases.
- Perfect scans can create a clean-reference domain that does not match worn phone photos.
- Ambiguous real fragments may never be classifiable from a single image. Product UX must allow unknowns or request a better angle/video.
- Heavy synthetic scale can hide bad assumptions. Small, audited probes should precede large dataset generation.
- Browser/mobile deployment is not the main blocker yet. Recognition quality on real partial/fan geometry is.

## Reviewed Inputs

- `D:\Download\Banknote Detection Research Handoff.pdf`
- `D:\Download\Practical Research Handoff for Partial Khmer Riel Banknote Detection and Identification.pdf`
- `docs/3d-scene-composition-pipeline.md`
- `docs/archive/synthetic-compositor-plan.md`
- `docs/fan-failure-analysis.md`
- `docs/fragment-classifier-plan.md`
- `docs/real-fan-benchmark.md`

External references checked:

- Real-Time Identification of Mixed and Partly Covered Foreign Currency Using YOLOv11 Object Detection: https://www.mdpi.com/2673-2688/6/10/241
- MBDM: Multinational Banknote Detecting Model: https://www.mdpi.com/2227-7390/11/6/1392
- BankNote-Net: https://github.com/microsoft/banknote-net
- Euro banknote recognition with homography and local features: https://carlosmccosta.github.io/Currency-Recognition/
- SAHI small-object slicing: https://arxiv.org/abs/2202.06934
- BCNet occlusion-aware instance segmentation: https://arxiv.org/abs/2103.12340
