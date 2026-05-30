# KhmerCurrencyOCR

KhmerCurrencyOCR is the research repo behind CashSnap, a lightweight computer-vision banknote counter for mixed Cambodian riel (KHR) and US dollar (USD) photos.

The goal is practical retail counting from one casual phone image: separated notes, overlapping stacks, handheld fans, partial notes, and hand/finger occlusion. Counterfeit detection and authenticity checks are intentionally out of scope.

## Current Status

This project is still research/prototype work. Clean visible-note detection is strong enough to be useful diagnostically, but dense overlap, fan layouts, and partial-note counting are not solved yet.

The active direction is a small detector plus fragment/evidence handling:

- visible-only labels for detector compatibility
- exact ID masks from synthetic renders
- fragment labels for disconnected visible evidence
- separate physical-note count metadata
- future fusion from fragments back to physical bill totals

Synthetic data is being built as a controlled experiment generator, not as a shortcut around real validation. Any synthetic recipe must improve real partial/fan benchmarks before it can be promoted.

## Repository Map

- `model.md` is the main working memory: active plan, current results, data ranking, commands, and known failure modes.
- `AGENTS.md` contains short project rules for future coding agents.
- `configs/` contains active dataset, target, recipe, and renderer-proof configs.
- `scripts/` contains data prep, synthetic rendering, QA, training, export, and evaluation utilities.
- `renderers/webgl/` contains the Three.js/Edge synthetic renderer proof.
- `docs/` is reference/archive material, not the active plan.

Most datasets, generated synthetic images, model weights, and run outputs are intentionally git-ignored.

## Synthetic Pipeline

The current WebGL pipeline can render banknote scenes through local Microsoft Edge using Three.js. It emits:

- RGB visual render
- exact flat-color ID mask
- visible-only YOLO detect labels
- OBB sidecar labels with rejection metadata
- fragment/evidence labels
- ignored-fragment metadata for below-threshold components
- per-batch `qa/summary.json`
- per-batch `recipe.json`

Important checks:

```powershell
rl python scripts\check_synthetic_recipe_catalog.py
rl python scripts\render_webgl_variant_batch.py --out-root data\synthetic\cashsnap_webgl_variant_batch_smoke --count 4 --skip-render
```

Long or heavy jobs should run through the headroom wrappers so the laptop remains usable.

## Quick Start

This repo is developed on Windows with Python and Node tooling. Prefer `pnpm` for Node work.

```powershell
python -m pip install -r requirements.txt
cd renderers\webgl
pnpm install
cd ..\..
rl python scripts\check_synthetic_recipe_catalog.py
```

For model training or larger rendering jobs, read `model.md` first and use the listed headroom-safe commands.

## Public Data Note

Currency imagery and public datasets can have licensing, reproduction, split-leakage, and current-design caveats. This repo treats public and synthetic data as research inputs only; final quality claims need reviewed real phone captures and real held-out benchmarks.

## Project Scope

In scope:

- KHR + USD denomination detection/counting
- phone/browser-deployable model paths
- synthetic data with exact labels
- real partial/fan validation

Out of scope:

- counterfeit detection
- authentication/security claims
- training on the real fan benchmark
- broad unreviewed data scraping as a substitute for validation
