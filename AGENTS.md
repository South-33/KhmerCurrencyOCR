This is the project's AGENTS.md

## Notes
- `model.md` is the single living source for CashSnap model plans, ideas, data rankings, config rankings, active results, and cleanup rules; update it whenever direction changes.
- Keep this file lean: only durable repo rules that a future agent needs before reading `model.md`.
- Current mission: small phone/browser USD+KHR banknote counter; counterfeit detection is out of scope.
- Current active phase: 3D synthetic-pipeline reset before the next training push; validate the renderer proof configs before training.
- Local rig profile: Lenovo 82Y9 laptop, AMD Ryzen 5 7640HS (6 cores / 12 threads), about 16 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM, driver 596.21.
- Re-scan the rig with `scripts/profile_system.py` before major training/rendering decisions if performance looks odd.
- Treat this as a constrained laptop, not a workstation: default heavy-job caps should stay below 95% CPU/RAM/GPU/VRAM, with 90% preferred and 82% resume thresholds unless `model.md` says otherwise.
- Harnesses, configs, and workflows are adjustable tools, not fixed rules: inspect and improve them whenever they slow the goal, add clutter, or encode a bad assumption.
- `scripts/run_with_headroom.py` has a preflight headroom wait and refuses caps above 95%; free-RAM floor is a launch gate/runtime warning, while hard RAM/VRAM pauses follow the explicit max caps. Use it for generic heavy work and `scripts/bench_train_with_headroom.py` for YOLO training.
- Balance speed and headroom: prefer GPU for training/inference when it is the faster engine and has room, but do not force GPU for CPU-native prep/rendering if the headroom wrapper keeps the laptop responsive.
- Never train on `data/real_fan_benchmark/`; it is evaluation/stress data only.
- Use `rl` for terminal work in LongRun/RunLong mode, and route heavy CPU/RAM/GPU jobs through `scripts/run_with_headroom.py` or `scripts/bench_train_with_headroom.py`.
- Keep YOLO runs under repo-local ignored `runs/`, not `C:\Users\Venom\runs`.
- Keep `results.tsv` untracked and append experiment rows with `scripts/log_research_result.py`.
- Prefer Numista `in_circulation` scans and `data/asset_candidates/numista_current_cutout_bank_v1/` as canonical banknote assets; treat public/Roboflow/PicWish data as review or domain-stress material until curated.
- Do not add new active model docs under `docs/`; archive/reference docs can live there, but `model.md` is the working brain.
