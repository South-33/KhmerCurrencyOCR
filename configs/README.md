# Configs

Active configs stay visible here:

- `cashsnap_v1.yaml`: clean/base YOLO dataset.
- `cashsnap_two_stage_oldcommon_browser_stack.json`: current browser diagnostic stack.
- `3d_pipeline/`: active 3D renderer proof configs.
- `synthetic_targets/cashsnap_real_target_matrix_v1.json`: required real-world conditions and gates for synthetic recipe coverage.
- `synthetic_recipes/cashsnap_webgl_recipe_catalog_v1.json`: named synthetic recipe slots mapped to target conditions, promotion gates, and current blockers.
- `synthetic_recipes/cashsnap_webgl_smoke_suite_v1.json`: one-command smoke suite for all smoke-ready WebGL recipes.
- `synthetic_recipes/cashsnap_webgl_trainable_candidates_v1.json`: bounded trainable-candidate suite with recipe seed ranges, output roots, train views, and required generated QA paths.
- `synthetic_recipes/cashsnap_webgl_visual_review_rules_v1.json`: human review statuses and bad-scene reasons for WebGL visual QA packs.
- `cashsnap_webgl_smoke_suite_mix.yaml`: generated YOLO mix YAML for the gated smoke suite; diagnostic only, not a training claim.

Old probe configs live in `archive/`. Do not promote one back to root unless `model.md` says it is active again.
