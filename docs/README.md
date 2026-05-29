# CashSnap Docs Map

Use this file as the first stop when the repo feels noisy. The project has many historical probes; the active path is narrower.

## Current Active Path

1. Review real partial KHR fragments, especially thin/edge `KHR_5000` and `KHR_20000` front/back crops.
2. Train small fragment-classifier probes only from reviewed rows, mixed carefully with the existing old/common real-box base.
3. Verify in the browser stack and on real fan/overlap draft labels before changing default model config.
4. Use 2.5D synthetic data as a controlled geometry supplement; promote WebGL/3D only after a matched benchmark gate.

## Start Here

- `docs/model-card-cashsnap-two-stage-oldcommon.md`: current deployable diagnostic stack, known limits, and browser smoke evidence.
- `docs/p1-fragment-curation-runbook.md`: next concrete data loop for the old/common `KHR_5000`/`KHR_20000` collapse.
- `docs/synthetic-strategy-evaluation.md`: current 2D/2.5D vs 3D decision memo.
- `docs/synthetic-harness-runbook.md`: scan-bank audit, 2.5D generation, QA, and detector-probe commands.
- `docs/data-prep.md`: source inventory, data rules, and generated artifact notes.

## Decision Memos

- `docs/fan-failure-analysis.md`: why the hard fan failure is mainly slice geometry/data, not a simple NMS or tiling issue.
- `docs/fragment-classifier-plan.md`: detector-plus-fragment-classifier architecture and historical classifier results.
- `docs/cashsnap-partial-banknote-synthetic-roadmap.md`: renderer-agnostic roadmap and promotion gates.
- `docs/3d-scene-composition-pipeline.md`: draft WebGL/Three.js renderer plan.
- `docs/archive/`: older baseline/model-first/compositor plans; useful for history, not the next command source.

## Data And Source References

- `docs/khr-circulation-scope.md`: KHR design/version scope.
- `docs/roboflow-cuurecy-detection-audit.md`: public Roboflow segmentation lead audit.
- `docs/data-source-leads.md`: external data leads and blocked/manual sources.
- `docs/research/README.md`: index for research handoff PDFs.
- `docs/real-fan-benchmark.md`: fixed benchmark rules.
- `docs/real-fan-capture-guide.md`: capture and proposal-review workflow.
- `docs/mobile-export.md`: ONNX/NCNN/TFLite export notes.

## Generated Artifacts

Most model runs, datasets, review packs, contact sheets, and downloaded source exports live under ignored `data/`, `runs/`, and `data/audit/` paths. Treat those as reproducible or local working artifacts unless a doc explicitly says they are the current benchmark or deployed stack.

Do not train on `data/real_fan_benchmark/`. Benchmark images can produce draft labels and review hints, but training/calibration data must come from non-benchmark captures or reviewed public-source crops.
