This is the project's AGENTS.md

## Notes
- Phase 1 scope is documentation for CashSnap: a USD + KHR banknote denomination counter using computer vision and a Hugging Face API flow; counterfeit detection is intentionally out of scope.
- Current technical direction is a lightweight object detector, currently YOLO26n first with YOLO11n/YOLOv8n fallbacks, using denomination-only labels and custom KHR phone-photo data expected because public KHR datasets are incomplete.
- Data prep has a verified local YOLO dataset at `data/cashsnap_v1/` with 13 classes and 9,048 boxes; `KHR_20000` and `KHR_50000` are the weakest classes and need more real/synthetic examples before strong claims.
- YOLO runs must be written under the repo's ignored `runs/` directory on D; avoid `C:\Users\Venom\runs` because the user profile drive is space-constrained.
- Real fan-photo stress image still is not solved; combined e2/e4 checkpoints beat dense synthetic v3 at `416/conf=0.25`, while e4 is the better rare-slice stress candidate.
- Use `ideas.md` as the short living board for high-value CashSnap experiment ideas and results; keep it curated, not append-only.
- Background removal can be automated for free using `scripts/process_picwish_batches.py` with the `picwish` PyPI library; keep concurrency under 15 (using `asyncio.Semaphore(10)`) with short sleeps to avoid Cloudflare rate blocks.
- Current best cutout set comes from scoring PicWish and BEN2 outputs, then selecting via `scripts/select_best_cutouts.py`; use the stricter `data/asset_candidates/rare_pristine_asset_bank_v1/` subset for first synthetic probes.
- First rare-cutout synthetic stage should be clean curriculum (`data/synthetic/khr_rare_pristine_clean_v1/`): 1-3 notes, no synthetic fingers, no strip/fan chaos; only scale up to harder fan/overlap after this probe is evaluated.
- Avoid `--fraction` for mixed CashSnap smoke probes: Ultralytics takes an ordered slice, which can select only empty/background `cashsnap_v1` labels; use synthetic-only YAML or `--max-train-batches` instead.
- Rare overlap probe data is `data/synthetic/khr_rare_pristine_overlap_v1/` from the pristine asset bank; it improves synthetic overlap validation but slightly trades off normal validation, so evaluate both.
- Current best checkpoint depends on the scoreboard: e2 (`runs/cashsnap/yolo26n_messy_v3_pristine_overlap_e2_i416_b8/weights/best.pt`) leads normal validation, while e4 (`runs/cashsnap/yolo26n_messy_v3_pristine_overlap_e4_i416_b8/weights/best.pt`) leads held-out test, rare-overlap synthetic, and the real fan stress image.
- Real fan failure is currently a slice-geometry/data problem, not NMS/tiling/resolution; see `docs/fan-failure-analysis.md` before adding more broad synthetic fan data.
- Before more KHR synthetic training, audit note designs against `docs/khr-circulation-scope.md`; current e2/e4 fine-tunes likely mixed modern notes with older/low-priority KHR variants.
- Rare KHR research PDFs live under `docs/research/`; use them for version/source checks instead of leaving reference docs in the repo root.
- Mobile export smoke notes live in `docs/mobile-export.md`; ONNX/NCNN export works from the balanced checkpoint, while TFLite needs a separate Python 3.11/3.12 TensorFlow environment because this repo is currently on Python 3.14.
- When the user is gaming or asks for headroom, run long jobs through `scripts/run_with_headroom.py --max-percent 90 --resume-percent 82 -- ...`; it throttles active CPU/GPU and reports RAM/VRAM, so also keep memory low with `--batch` 2-4, `--workers` 0-1, and `--max-train-batches` when possible.
