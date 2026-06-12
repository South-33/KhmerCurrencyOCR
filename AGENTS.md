This is the project's AGENTS.md

## Notes
- `model.md` is the working brain for plans, active decisions, durable asset/result rankings, and command posture; keep exhaustive script/config/run status in folder layout, registries, or archives instead.
- Keep `model.md` honest: prune stale phase labels, achieved TODOs, and outdated advice during normal work instead of letting old plans accumulate.
- Preferred doc shape is one project `AGENTS.md`, one working `model.md`, and one user-facing `README.md`; fold scattered active docs back into those files or archive/delete them when they stop earning their keep.
- Work directly on `master`/mainline for this repo unless the user explicitly asks for a branch.
- Keep generated YOLO/training outputs under repo-local ignored `runs/`, not user-home fallback paths.
- Import/call `scripts/local_runtime.py::configure_project_cache()` before Ultralytics/Torch-heavy imports in ML entry points so caches/temp stay under repo-local `.cache_runtime/`.
- For long runs under RAM pressure, prefer `scripts/run_with_headroom.py --memory-clean-preset memreduct`; it uses COM task `memreductTask=-clean`, then ends the task, for `C:\Program Files\Mem Reduct\memreduct.exe` from `https://github.com/henrypp/memreduct`; plain `schtasks /Run /TN memreductTask` can launch literal `$(Arg0)`.
- Do not add new active model-planning docs under `docs/`; archive/reference material can live there, but active model memory belongs in `model.md`.
- Before using a new data root for training/rendering, register or classify it in `configs/synthetic_recipes/cashsnap_data_lifecycle_registry_v1.json` and run `scripts/check_data_lifecycle_registry.py`.
- Before claiming a currency class exists or is missing, run `scripts/check_currency_taxonomy_coverage.py`; raw Numista, active cutout bank, and YOLO schema coverage intentionally differ.
