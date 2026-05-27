# CashSnap Browser Probe

Static ONNX Runtime Web demo for the current two-stage old/common KHR diagnostic stack.

Run from the repository root so the page can fetch the model artifact:

```powershell
lr python -m http.server 8787
```

Open:

```text
http://localhost:8787/demo/browser/
```

For repeatable smoke checks with the shop-overlap candidate:

```text
http://localhost:8787/demo/browser/?image=/data/real_fan_benchmark/images/candidates/real_overlap_0003_commons_shop_5k_10k_20k.png&autorun=1
```

Or run the headless Edge smoke from the repo root:

```powershell
lr python scripts/run_with_headroom.py --interval 2 --max-percent 90 --resume-percent 82 --max-ram-percent 90 --max-gpu-mem-percent 90 -- node scripts/smoke_browser_demo_cdp.cjs --labels data/real_fan_benchmark/drafts/real_overlap_0003_commons_shop_5k_10k_20k.txt --min-same-class 4 --min-any-class 6 --screenshot .agent/cashsnap-browser-smoke-cdp.png --out-csv .agent/cashsnap-browser-smoke-cdp.csv
```

To run every labeled browser smoke case in `manifests/browser_smoke_cases.csv`:

```powershell
lr python scripts/run_with_headroom.py --interval 2 --max-percent 90 --resume-percent 82 --max-ram-percent 90 --max-gpu-mem-percent 90 -- python scripts/run_browser_smoke_cases.py
```

This writes per-case screenshots, detection CSVs, and `summary.json` under `.agent/browser_smoke_cases/`.

The demo reads `configs/cashsnap_two_stage_oldcommon_browser_stack.json`, then loads:

- Detector: `runs/cashsnap/yolo26n_legacy_clean_plus_realcutout_low_skin_ft_e6_i416_b8/weights/best.onnx`
- Fragment classifier: `runs/fragment_classifier/mobilenet_v3_old_common_khr_realbox_pretrained_balanced_e12/best.onnx`

It fuses low-confidence detector proposals with the KHR 1k/5k/10k/20k crop classifier and uses detector-confidence NMS. It is suitable for browser plumbing and review hints, not reliable denomination totals yet.

Smoke note: the autorun shop-overlap URL loads the ONNX stack in Edge and predicts 6 bills, but the denomination total is still wrong (`KHR 76,000`, `USD 0` vs the draft-label total of `KHR 70,000`, `USD 0`). Current browser classes are `KHR_1000:1`, `KHR_5000:1`, `KHR_10000:1`, and `KHR_20000:3`; the smoke JSON reports `6/6` any-class matches, `4/6` same-class matches, expected label totals, value errors, and matched-pair confusion counts when `--labels` is provided. It also reports per-source recall: current final/detector labels are `4/6` same-class, while fragment-only is `3/6`. Treat this as improved browser preprocessing parity, not solved counting.

The smoke JSON includes debug counters such as detector output dims, browser proposal count, classified count, and final count. Current debug output is `[1,300,6]`, `13` proposals, `13` classified crops, and `6` final detections.

For detector preprocessing parity checks, run `scripts/debug_onnx_detector_preprocess.py`. On the shop-overlap image, the same detector ONNX produces `13` proposals with `--mode cv2` but only `8` with `--mode pil`, with a `USD_100` class appearing in the PIL-style path.
