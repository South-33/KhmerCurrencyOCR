# CashSnap Mobile Export

Current balanced checkpoint:

```powershell
runs/cashsnap/yolo26n_messy_v3_pristine_overlap_e2_i416_b8/weights/best.pt
```

Export commands:

```powershell
rl python scripts/export_yolo.py --model runs/cashsnap/yolo26n_messy_v3_pristine_overlap_e2_i416_b8/weights/best.pt --format onnx --imgsz 416 --simplify --opset 12
rl python scripts/export_yolo.py --model runs/cashsnap/yolo26n_messy_v3_pristine_overlap_e2_i416_b8/weights/best.pt --format ncnn --imgsz 416
```

Observed outputs:

- ONNX: `best.onnx`, 9.3 MB, `(1, 300, 6)` output. Ultralytics ONNX Runtime normal-val smoke reached mAP50 0.976 and mAP50-95 0.921.
- NCNN: `best_ncnn_model/`, 9.2 MB. Ultralytics disables the YOLO26 end-to-end branch for NCNN; rare-overlap smoke reached mAP50 0.631 and mAP50-95 0.406.

TFLite is blocked in the current Python 3.14 environment because Ultralytics requires `tensorflow>=2.0.0,<=2.19.0`, and no compatible TensorFlow wheel is available for this interpreter. Use a separate Python 3.11 or 3.12 environment for TFLite export.

## Fresh Circulated-Design Candidate

Preferred fresh-weight overlap-counting alpha:

```powershell
runs/cashsnap/yolo26n_cashsnap_current_thin_legacy_clean_v1_e20_i416_b8/weights/best.pt
```

Export smoke:

- ONNX export succeeded at `imgsz=416`, opset 12: `best.onnx`, about `9.2 MB`, output shape `(1, 300, 6)`.
- ONNX Runtime smoke on `real_overlap_0003_commons_shop_5k_10k_20k` matched PyTorch-style predictions at `416/conf=0.25`: `2 KHR_20000`.
- NCNN export succeeded: `best_ncnn_model/`, about `9.2 MB`; Ultralytics disables YOLO26 end-to-end branch for NCNN.
- NCNN smoke on the same image produced a slightly different count/class set (`2 KHR_20000`, `1 KHR_50000`), so treat NCNN parity as not yet validated.

This checkpoint is suitable for browser/phone integration plumbing tests, but not yet for reliable denomination totals.

## Two-Stage Old/Common KHR Diagnostic Stack

Best current shop-overlap diagnostic stack:

```powershell
# Detector
runs/cashsnap/yolo26n_legacy_clean_plus_realcutout_low_skin_ft_e6_i416_b8/weights/best.pt

# KHR 1k/5k/10k/20k fragment classifier
runs/fragment_classifier/mobilenet_v3_old_common_khr_realbox_pretrained_balanced_e12/best.pt
```

Export state:

- Detector ONNX export succeeded at `imgsz=416`, opset 12: `best.onnx`, about `9.3 MB`, output shape `(1, 300, 6)`.
- Detector NCNN export succeeded: `best_ncnn_model/`, about `9.2 MB`; Ultralytics disables the YOLO26 end-to-end branch for NCNN.
- Classifier ONNX export exists at `runs/fragment_classifier/mobilenet_v3_old_common_khr_realbox_pretrained_balanced_e12/best.onnx`, about `5.8 MB`, with class order in `classes.json`.

Check current browser-stack artifact paths and total ONNX size with:

```powershell
rl python scripts/check_browser_stack_artifacts.py
```

Current check reports `15.09 MB` total for the detector plus old/common fragment classifier ONNX files.

Current diagnostic fusion recipe:

```powershell
rl python scripts/fuse_two_stage_csv.py --csv data/real_fan_benchmark/drafts/two_stage_realcutout_oldcommon_realboxcls_i416_c0p05_agnostic_pad0.csv --out data/real_fan_benchmark/drafts/two_stage_realcutout_oldcommon_realboxcls_fuse_det0p17_nms0p85_detconf.csv --det-threshold 0.17 --nms-iou 0.85 --nms-score-column detector_conf --image data/real_fan_benchmark/images/candidates/real_overlap_0003_commons_shop_5k_10k_20k.png --out-preview data/real_fan_benchmark/previews/two_stage_realcutout_oldcommon_realboxcls_fuse_det0p17_nms0p85_detconf.jpg
```

On the draft-labeled `real_overlap_0003_commons_shop_5k_10k_20k` probe this reaches `5/6` same-class matches and `6/6` any-class matches after detector-threshold fusion and detector-confidence NMS. Treat this as a promising calibration point, not a production benchmark, because it is tuned against one draft-labeled real image.

Browser smoke demo:

```powershell
rl python -m http.server 8787
```

Open `http://localhost:8787/demo/browser/`. The page loads `configs/cashsnap_two_stage_oldcommon_browser_stack.json`, runs the detector plus old/common fragment classifier with ONNX Runtime Web, and draws/counts detections on an uploaded image. Use the headroom-wrapped `scripts/smoke_browser_demo_cdp.cjs --labels ...` command in `demo/browser/README.md` for repeatable Edge checks with inline draft-label evaluation; the 2026-05-30 browser smoke suite passes the USD_1 and detector-only KHR sanity cases, while the shop-overlap case remains 6 bills, KHR 76,000, 6/6 any-class, 4/6 same-class, and a `+6000` KHR value error against draft labels.
