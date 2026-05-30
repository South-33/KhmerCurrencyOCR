# Headroom Bench Harness

Use `scripts/bench_train_with_headroom.py` for long YOLO probe runs when the computer must remain usable.

The harness:

- probes CPU, RAM, and NVIDIA GPU/VRAM before launching
- chooses conservative `batch` and `workers` when RAM is already high
- disables Ultralytics plots by default to save memory
- refuses to launch by default when available system RAM is below `--min-free-ram-gb 4`
- runs training through `scripts/run_with_headroom.py`
- pauses/resumes on CPU/GPU utilization
- relaunches with smaller `batch`/`workers` if RAM or VRAM reaches the configured hard limit
- switches to pause/resume mode at the smallest restart settings instead of abandoning the job by default

Default guardrails are `--max-percent 90`, `--resume-percent 82`, `--max-ram-percent 90`, `--max-gpu-mem-percent 90`, and `--min-free-ram-gb 4`.

CPU/GPU utilization can be throttled live by pausing and resuming the child process. Batch size cannot be changed inside a running Ultralytics process, so memory pressure is handled by an adaptive relaunch: the wrapper asks the child to stop, lowers workers/batch, and restarts in the same run name with `--exist-ok`. Once there is nothing smaller to relaunch, `--floor-memory-action pause` keeps the job parked until RAM/VRAM pressure clears. Use `--floor-memory-action exit` when you would rather stop than wait.

Example:

```powershell
rl python scripts\bench_train_with_headroom.py `
  --model runs/cashsnap/yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8/weights/best.pt `
  --data configs/cashsnap_v1_current_thin_realcutout_low_skin_probe.yaml `
  --epochs 6 `
  --imgsz 416 `
  --name yolo26n_safe_probe_e6 `
  --optimizer AdamW `
  --lr0 0.0002 `
  --lrf 0.2 `
  --warmup-epochs 1 `
  --quiet
```

Run a dry plan first:

```powershell
rl python scripts\bench_train_with_headroom.py --data configs/cashsnap_v1_current_thin_realcutout_low_skin_probe.yaml --name dry_run --dry-run --quiet
```

Use `--batch` or `--workers` to override auto choices only when the machine is idle. Use `--plots` only when plots are needed; leaving it off is safer for RAM. Use `--min-free-ram-gb 0` only for a deliberate diagnostic after checking the machine is safe to load.

The current smoke test selected `batch=2`, `workers=0`, disabled plots, and completed under the 90% memory guard.
