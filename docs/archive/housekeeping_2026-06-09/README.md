# Housekeeping 2026-06-09

This archive groups files removed from active repo surfaces during the
post-synth-only-result cleanup. Items here are not active plans or current
rerun entry points.

Moved out of the repository root:
- `root_weights/inception-2015-12-05.pt` - unreferenced downloaded Inception
  checkpoint; kept out of root in case a past metric probe needs it.
- `root_weights/yolo26n-obb.pt` - unreferenced OBB YOLO checkpoint; keep
  `yolo26n.pt` in root because active training scripts default to it.

Moved out of active config/script surfaces:
- Activation microscope configs and generated lists moved to
  `configs/archive/housekeeping_2026-06-09/`.
- Real-flat detector repair configs and generated lists moved to
  `configs/archive/housekeeping_2026-06-09/`.
- Legacy browser stack configs moved to
  `configs/archive/housekeeping_2026-06-09/browser_stacks/`; the demo now
  defaults to the current strictbest true-empty proposal gate stack.
- One-off activation microscope and real-flat helper scripts moved to
  `scripts/archive/housekeeping_2026-06-09/`.
- Rejected teacher pseudo-target bridge script/config/list moved out of active
  `scripts/` and `configs/webgl_ablation/`.
- Rejected direct UNKNOWN / `unknownprop` detector configs and lists moved out
  of active `configs/webgl_ablation/`.
- Rejected semantic true-empty background-replacement configs/lists moved out
  of active `configs/webgl_ablation/`; the lifecycle registry entry is marked
  `rejected`.
- Blunt unknown-pressure / `safeunknown` / recall-repair / positive-only dose
  configs and lists moved out of active `configs/webgl_ablation/`; keep the
  data registry entries as diagnostic memory, not active continuation points.
- Older `current_obligation_usdweak_cleanbg*` detector configs/lists moved out
  of active `configs/webgl_ablation/`; current `usdweakalpha317` product-stack
  reclassifier evidence remains active separately.
- Dark/style positive-support probe configs and generated lists moved out of
  active `configs/webgl_ablation/`; they remain as visual-domain research
  history, not current trainable data. The matching style-positive config
  builder moved to `scripts/archive/housekeeping_2026-06-09/`.
- Rejected source-context replacement training configs/lists (`single_*`,
  `multi_instance_*`, and `rep_gap_sourcectx*`) moved out of active
  `configs/webgl_ablation/`; reusable audit/materialization scripts stayed
  active.
- Older clean/WebGL baseline, dose, `hardnegold`, topdown-support, and tone probe
  configs/lists (`cashsnap_v1_plus_webgl*`, `cashsnap_webgl_clean_*`) moved out
  of active `configs/webgl_ablation/`; active builder defaults were retargeted
  to the current strictbest foreign-hardneg config.
- Early target-anchor launchers (`latest`, `mvp`, smoke/pose/luma/style,
  phone-ISP, pair/unknown, extent-heavy, and exploratory bridge/camera configs)
  moved out of active `configs/webgl_ablation/`; the active surface now keeps the
  strictbest base, current foreign-hardneg config, mined-negative variants, and
  real baseline configs.
- Orphan generated lists from refiner/smoke diagnostics moved out of active
  `configs/generated_lists/webgl_ablation/`; remaining active lists are referenced
  by active YAMLs.
- Standalone clean-curriculum probe root YAML and recipe JSON moved to
  `configs/archive/housekeeping_2026-06-09/root_configs/` and
  `configs/archive/housekeeping_2026-06-09/synthetic_recipes/`; the broader
  trainable-candidate suite remains active because check/run scripts use it.
- Older accepted-WebGL top-level probes, staged-dose configs, blend-minus
  variants, old-overlap repair configs, and their generated lists moved to
  matching `configs/archive/housekeeping_2026-06-09/` folders; now-empty active
  subfolders were removed.
- Legacy oldcommon/realfrag/old-overlap fragment-classifier runs moved to
  `runs/archive/housekeeping_2026-06-09/fragment_classifier_legacy/`, with
  archived browser-stack JSON paths updated to match.
- Old top-level eval run folders moved to
  `runs/archive/housekeeping_2026-06-09/legacy_eval/`; empty
  `runs/baseline_pretrained/` was deleted.
- Old-overlap proposal-gate classifier datasets moved to
  `data/archive/housekeeping_2026-06-09/fragment_classifier_proposal_gate_legacy/`.

Kept active:
- `model.md`, `AGENTS.md`, `README.md`, `configs/synthetic_recipes/`,
  `scripts/` harnesses still referenced by current model memory, `data/`,
  `runs/`, and root `yolo26n.pt`.
