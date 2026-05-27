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
lr node scripts/smoke_browser_demo_cdp.cjs --screenshot .agent/cashsnap-browser-smoke-cdp.png --out-csv .agent/cashsnap-browser-smoke-cdp.csv
```

The demo reads `configs/cashsnap_two_stage_oldcommon_browser_stack.json`, then loads:

- Detector: `runs/cashsnap/yolo26n_legacy_clean_plus_realcutout_low_skin_ft_e6_i416_b8/weights/best.onnx`
- Fragment classifier: `runs/fragment_classifier/mobilenet_v3_old_common_khr_realbox_pretrained_balanced_e12/best.onnx`

It fuses low-confidence detector proposals with the KHR 1k/5k/10k/20k crop classifier and uses detector-confidence NMS. It is suitable for browser plumbing and review hints, not reliable denomination totals yet.

Smoke note: the autorun shop-overlap URL loads the ONNX stack in Edge and predicts 6 bills, but the denomination total is still wrong (`KHR 56,000`, `USD 0` vs the draft-label total of `KHR 70,000`, `USD 0`). Current browser classes are `KHR_1000:1`, `KHR_5000:1`, `KHR_10000:3`, and `KHR_20000:1`; treat this as working browser plumbing, not solved counting.
