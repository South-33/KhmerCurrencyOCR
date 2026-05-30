# CashSnap Synthetic Strategy Evaluation

## Recommendation

Stay on the renderer-agnostic 2.5D path for the next iteration. The current failure mode is no longer just "the detector cannot find a note region"; the latest probes show that synthetic geometry can raise proposal coverage, but denomination identity remains weak on real old/common shop-overlap slices.

Treat full WebGL/3D as a gated experiment, not the main road yet. Build it only after a minimal proof renders stable visual frames, exact ID masks, and a direct A/B dataset gain over the 2.5D generator on reviewed real partial-note benchmarks.

The highest-value work is:

1. Scoreable, rights-clear real partial/fan phone photos and reviewed fragment crops, especially `KHR_5000` portrait-plus-number overlap and `KHR_20000` thin/edge cases.
2. Better 2.5D domain realism from trusted Numista `in_circulation` scans: phone-style texture degradation, real hand/finger patches, contact shadows, glare, and calibrated unknown fragments.
3. Detector-plus-fragment-verifier fusion, because the deployable browser stack is already small enough and closer to the hard overlap behavior than detector-only training.

## Current Evidence

- Clean/flat banknotes are mostly solved; the hard gap is dense overlap, fanned slices, old/common KHR backs, and hand occlusion.
- `cashsnap_scan_2p5d_fan_v1` improved permissive real-overlap region coverage but not denomination identity (`640/conf=0.05`: 6/6 any-class, 2/6 same-class, 17 predictions).
- The old/common scan-focus follow-up did not fix the identity problem: best 416px result on the same draft was 2/6 same-class and 2/6 any-class, while 640px added `KHR_500`/`KHR_1000` false positives.
- The current two-stage browser stack remains the best phone/browser-shaped diagnostic: 6/6 any-class, 4/6 same-class in Edge, with exact sanity checks for USD_1 and detector-only KHR classes.
- Targeted clean Numista `KHR_5000` face/number crops preserved base crop accuracy but still topped at 4/6 same-class on the shop-overlap diagnostic, so clean scan fragments alone are not the missing evidence.

## 2D/2.5D Compositor Critique

The classic 2D layer compositor is limited when it treats each banknote as a rigid sticker. It can randomize position and z-order, but it does not naturally model camera projection, paper bending, contact shadows, finger grip geometry, or the way visible note evidence changes in thin fan slices.

Core limitations:

- Geometry: affine rotation/scale cannot represent near-camera perspective, curved paper edges, lifted corners, or fanned notes sharing a physical pivot.
- Occlusion labels: full-note labels on hidden notes teach the detector to hallucinate. Labels must be regenerated from visible masks after every occlusion.
- Optics: alpha-blended shadows are often too uniform; real overlaps have directional contact shadows, specular wash, defocus, motion blur, sensor sharpening, and JPEG artifacts.
- Semantics: a visible blank edge or generic back-side strip is not always a denomination label. These should become `banknote_unknown` or verifier calibration crops.
- Domain: scan textures are too clean unless aggressively degraded toward phone photos, worn notes, and old/common circulated designs.

Low-hanging upgrades that are worth more than jumping straight to 3D:

- Per-note homography with visible-mask recomputation.
- Thin-slice and radial-fan modes with explicit shared grip points.
- Mesh-style 2D displacement fields for curl, crease, and edge lift.
- Directional contact shadows derived from the visible mask and z-order.
- Real background and real hand/finger patch banks, but only after contact-sheet QA.
- Unknown-fragment crops for the verifier, not denomination labels.
- A real-benchmark visibility histogram and class/side sampler that matches the failure cases instead of blindly increasing synthetic volume.

## 3D WebGL Critique

Full 3D solves a real class of problems: perspective-consistent camera projection, shadows from actual depth order, curved surfaces, and ID-pass masks from the same rendered scene. Those are hard to reproduce perfectly in a 2D stack.

But 3D does not automatically solve CashSnap's current bottleneck. If the textures are still pristine scans, fingers are synthetic-looking, and old/worn phone-domain fragments are missing, a 3D renderer will produce more beautiful synthetic data with the same transfer gap.

Use WebGL/3D for:

- Pixel-exact visible masks when many curled notes overlap.
- Camera-consistent perspective and shadow geometry.
- Controlled A/B experiments on OBB/segmentation labels.
- Generating verifier crops with known visibility and evidence tiers.

Do not use WebGL/3D yet for:

- Broad 10k-scene training before proving transfer.
- PBR polish before fixing hand occlusion and real texture domain.
- Replacing the real partial-photo benchmark.

## Runtime Stack Risks

Headless Three.js through Chrome/Edge/Puppeteer is plausible on Windows, but it is not risk-free:

- Chrome may fall back to SwiftShader/software WebGL in headless or unsupported GPU environments, which changes speed and sometimes render details. Chromium documents SwiftShader as a WebGL fallback for headless or unsupported GPU cases.
- GPU acceleration, antialiasing, color management, and canvas readback can differ between headed, headless, Edge, Chrome, and CI-like environments.
- Windows process cleanup is fragile: orphaned browser, HTTP, or DevTools processes can leave ports locked.
- WebGL contexts can fail or be lost under GPU/RAM pressure.
- ID-pass extraction must disable antialiasing/dithering or use integer-safe color encoding; otherwise instance colors can bleed at edges.

Alternative stacks:

- Pure Python/OpenCV/NumPy 2.5D: best current experiment loop. Deterministic masks, easy QA, no GPU/WebGL dependency.
- Blender `bpy` headless: highest realism and real cloth/lighting options, and Blender supports background command-line rendering, but it is a heavy binary/runtime dependency and slower to iterate.
- `pyrender`/Trimesh: useful for pure Python 3D, but offscreen rendering needs EGL or OSMesa in headless environments, which is another brittle OpenGL setup path.
- Browser-less pure math projection: good middle ground. Project subdivided note meshes into 2D, rasterize masks with OpenCV, and synthesize shadows/post effects without a WebGL runtime.

## Paper Curl Equations

Represent each banknote as a rectangular grid in local coordinates:

```text
x in [-L/2, L/2]
y in [-W/2, W/2]
u = x / L
v = y / W
```

A stable first curl model is cylindrical bend plus edge lift:

```text
theta = kx * x
if abs(kx) < eps:
    xb = x
    zb = 0
else:
    xb = sin(theta) / kx
    zb = (1 - cos(theta)) / kx

edge_y = max(abs(v) - edge_start, 0) / max(1 - edge_start, eps)
z_edge = ay * edge_y^2 * sign_or_mode
```

Then add a crease ridge along a random line:

```text
p = (x, y)
n = (cos(phi), sin(phi))
d = dot(n, p - p0)
z_crease = ac * exp(-(d*d) / (2*sigma*sigma)) * (1 + 0.25 * sin(freq * dot(t, p)))
```

And low-frequency crumple:

```text
z_noise = an * smooth_noise(fx * u, fy * v)
```

Final local mesh:

```text
P_local = (xb, y, zb + z_edge + z_crease + z_noise)
```

For a hand-held fan, pin deformation near the grip point `g`:

```text
r = length((x, y) - g)
w = clamp((r - grip_radius) / feather_radius, 0, 1)
P_local = P_flat * (1 - w) + P_curled * w
```

This keeps the note nearly flat under the fingers and lets the free edge curl upward.

## Hand And Finger Occlusion

Start with cheap occluders, but make them grip-aware:

- Capsules and rounded rectangles arranged around the fan pivot.
- Fingertip ellipses with knuckle-width variation.
- Soft masks, directional shadows, and skin-tone jitter.
- Occluders constrained to plausible palm/finger geometry instead of random blobs.

This is useful for teaching "do not hallucinate through a finger" and for visible-mask robustness, but it will not match real hands well enough for final transfer.

Real segmented hand/finger patches are the next better step. They preserve skin texture, wrinkles, motion blur, camera noise, and natural edge softness. They are also cheaper than a 3D hand mesh.

A parameterized 3D hand mesh should wait until two conditions are true:

- Real hand patches still fail to cover the benchmark failure mode.
- The 3D renderer already has proven label correctness and transfer value.

3D hands are powerful for pose consistency and finger depth, but they bring rigging, pose sampling, material realism, and uncanny-domain risks. For CashSnap, real hand patches plus grip-aware primitives are the better next investment.

## Dataset Mix

For a robust main checkpoint:

- 45% clean or near-clean notes.
- 30% simple overlaps and shop-counter spreads.
- 25% complex fans, dense stacks, off-frame notes, and hand occlusion.

For a short hard-case fine-tune after the main checkpoint is stable:

- 35% clean or near-clean notes.
- 25% simple overlaps.
- 40% complex fans and hand-held partials.

Do not let synthetic fan data dominate until clean validation and browser sanity cases stay stable. Keep old/common KHR, current KHR, and USD mixtures explicit so the model does not silently trade clean denomination skill for overlap recall.

## Promotion Gates

Promote a synthetic recipe only if it passes all of these:

- Clean validation does not regress meaningfully.
- Reviewed real partial/fan benchmark improves count and same-class recall.
- Browser/phone export smoke still passes size and sanity checks.
- False positive count stays controlled at deployable thresholds.
- Unknown/ambiguous fragments are not pushed into confident wrong denominations.

Promote WebGL/3D only after a 100-300 scene proof beats a matched 2.5D dataset on the real benchmark. The proof must include visual pass, ID pass, visible boxes, OBB, masks, verifier crops, contact sheets, and deterministic reruns on Windows.

## Source Notes

- Chromium SwiftShader fallback: https://chromium.googlesource.com/chromium/src.git/+/refs/heads/main/docs/gpu/swiftshader.md
- Blender command-line background rendering: https://docs.blender.org/manual/en/latest/advanced/command_line/index.html
- Pyrender offscreen EGL/OSMesa requirements: https://pyrender.readthedocs.io/en/latest/examples/offscreen.html
- BankNote-Net dataset: https://github.com/microsoft/banknote-net
- Cambodia Currency Project dataset lead: https://universe.roboflow.com/khmer-riel-classification-computer-vision/cambodia-currency-project
- Banknotes of Cambodia image lead: https://commons.wikimedia.org/wiki/Category:Banknotes_of_Cambodia
