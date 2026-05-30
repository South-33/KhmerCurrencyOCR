This is the project's AGENTS.md

## Notes
- `model.md` is the single living source for CashSnap model plans, ideas, data rankings, config rankings, active results, and cleanup rules; update it whenever direction changes.
- Keep this file lean: only durable repo rules that a future agent needs before reading `model.md`.
- Current mission: small phone/browser USD+KHR banknote counter; counterfeit detection is out of scope.
- Current active phase: 3D synthetic-pipeline reset before the next training push; validate the renderer proof configs before training.
- Never train on `data/real_fan_benchmark/`; it is evaluation/stress data only.
- Use `rl` for terminal work in LongRun/RunLong mode, and route heavy CPU/RAM/GPU jobs through `scripts/run_with_headroom.py` or `scripts/bench_train_with_headroom.py`.
- Keep YOLO runs under repo-local ignored `runs/`, not `C:\Users\Venom\runs`.
- Keep `results.tsv` untracked and append experiment rows with `scripts/log_research_result.py`.
- Prefer Numista `in_circulation` scans and `data/asset_candidates/numista_current_cutout_bank_v1/` as canonical banknote assets; treat public/Roboflow/PicWish data as review or domain-stress material until curated.
- Do not add new active model docs under `docs/`; archive/reference docs can live there, but `model.md` is the working brain.
