# P1 Fragment Curation Runbook

This is the shortest useful loop for the current CashSnap blocker: the focused old/common KHR classifier collapses many thin `KHR_5000` and `KHR_20000` crops into `KHR_10000`, and unreviewed P1 augmentation improves crop diagnostics while hurting the real shop-overlap fusion. The next win is reviewed real fragments, not another blind classifier trick.

## Inputs

- Broad P1 queue: `data/review/cashsnap_p1_oldcommon_partial_focus_review_v1/manifest.csv`
- Compact failure queue: `data/review/p1_focus_v2_oldcommon_failure_review_v1/manifest.csv`
- Current browser classifier: `runs/fragment_classifier/mobilenet_v3_old_common_khr_realbox_pretrained_balanced_e12/best.onnx`
- Current diagnostic P1 augmentation to avoid promoting: `runs/fragment_classifier/mobilenet_v3_oldcommon_realbox_plus_p1_unreviewed_diag_e6/`

The compact 43-row queue is the best first pass because it focuses on high-confidence `KHR_5000/20000 -> KHR_10000` and `KHR_20000 -> KHR_5000` failures with zero missing crop paths.

## 1. Rebuild And Summarize The Queues

```powershell
rl python scripts/build_partial_focus_review_queue.py --clean
rl python scripts/summarize_review_manifests.py
```

The summary should report blank `review_include` fields before human curation. If crop paths are missing, fix that before review.

## 2. Review In The Static UI

```powershell
rl python -m http.server 8787
```

Open `http://localhost:8787/demo/review/` and choose the `P1 old/common failure queue` preset first. Mark only human-identifiable crops:

- `review_include=1` for usable crops.
- `review_class` as the visible denomination, not the model prediction.
- `review_notes` for ambiguity, backside uncertainty, occluding fingers, motion blur, or old/new design clues.

Use the per-card quick actions to accept the visible label, mark `banknote_unknown`, or mark `background`; use the `Needs review` filter to keep the queue focused on untouched rows.

Leave ambiguous slices blank instead of forcing a class. Ambiguous crops are useful later as verifier hard negatives, but they should not enter trusted denomination training.

Use `banknote_unknown` when the crop clearly contains a banknote fragment but lacks human-identifiable denomination evidence. Use `background` only for non-banknote false positives. Do not include texture-only edge strips in denomination-class training just because the source dataset label names a note.

## 3. Merge The Export

Save the browser export under an ignored path such as `data/review_exports/p1_failure_review_export.csv`, then dry-run the merge:

```powershell
rl python scripts/apply_review_export.py `
  --source data/review/p1_focus_v2_oldcommon_failure_review_v1/manifest.csv `
  --export data/review_exports/p1_failure_review_export.csv `
  --dry-run
```

If the matched row count and selected row count look right, write a reviewed copy instead of overwriting the source:

```powershell
rl python scripts/apply_review_export.py `
  --source data/review/p1_focus_v2_oldcommon_failure_review_v1/manifest.csv `
  --export data/review_exports/p1_failure_review_export.csv `
  --out data/review/p1_focus_v2_oldcommon_failure_review_v1/reviewed.csv
```

## 4. Build A Trusted Small Dataset

Build only from reviewed rows. Keep `--include-unreviewed` out of trusted runs.

```powershell
rl python scripts/build_fragment_classifier_from_review_pack.py `
  --manifest data/review/p1_focus_v2_oldcommon_failure_review_v1/reviewed.csv `
  --out data/fragment_classifier_p1_oldcommon_failure_reviewed_v1 `
  --classes KHR_5000,KHR_10000,KHR_20000 `
  --ensure-classes KHR_1000,KHR_5000,KHR_10000,KHR_20000 `
  --clean
```

This command intentionally excludes `banknote_unknown` and `background` for the current focused denomination classifier. Keep those rows in the reviewed CSV for a later verifier/unknown probe. If there are too few reviewed denomination crops, stop and collect/review more. Do not stretch a tiny set with heavy training claims.

## 5. Mix Without Contaminating Validation

Use the reviewed P1 rows as train-only extras against the existing old/common real-box base, preserving base validation/test splits.

```powershell
rl python scripts/build_imagefolder_mix.py `
  --base data/fragment_classifier_cashsnap_old_common_khr_realbox_v1 `
  --train-extra data/fragment_classifier_p1_oldcommon_failure_reviewed_v1 `
  --out data/fragment_classifier_oldcommon_realbox_plus_p1_reviewed_v1 `
  --train-extra-splits train,val,test `
  --clean
```

## 6. Train With Headroom

```powershell
rl python scripts/run_with_headroom.py `
  --interval 2 `
  --max-percent 90 `
  --resume-percent 82 `
  --max-ram-percent 90 `
  --max-gpu-mem-percent 90 `
  -- python scripts/train_fragment_classifier.py `
    --data data/fragment_classifier_oldcommon_realbox_plus_p1_reviewed_v1 `
    --name mobilenet_v3_oldcommon_realbox_plus_p1_reviewed_e6 `
    --epochs 6 `
    --batch 16 `
    --workers 0 `
    --pretrained `
    --balanced-loss `
    --balanced-sampler `
    --export-onnx
```

## 7. Gate Before Promotion

Promotion requires all of these:

- Base old/common real-box test accuracy stays near the current focused classifier.
- P1 reviewed diagnostic improves specifically on `KHR_5000` and `KHR_20000` without collapsing `KHR_10000`.
- `scripts/evaluate_two_stage_csv.py` on real shop-overlap draft labels beats or ties the current classifier.
- `scripts/run_browser_smoke_cases.py` passes USD and detector-only KHR guard cases.
- `scripts/check_browser_stack_artifacts.py` keeps detector + classifier under the mobile/browser size budget.

Do not replace the browser default from crop accuracy alone. The rejected unreviewed P1 run is the warning example: better P1 crop numbers, worse real overlap fusion.

The agent-reviewed broad P1 focus probes are the second warning example. A clean 13-crop `KHR_5000/20000` supplement and a balanced 25-crop supplement with `KHR_10000` replay both preserved old/common crop test accuracy, but both still topped at `4/6` same-class on the shop-overlap draft labels. Treat crop accuracy as necessary but insufficient; promotion needs row-level fusion gains on real partial/overlap cases.
