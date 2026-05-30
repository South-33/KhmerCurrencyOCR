# CashSnap Research PDFs

This folder holds compact research handoffs and source-check PDFs that are useful for project direction but should not replace `model.md`.

## Current Takeaways

- `Practical Research Handoff for Partial Khmer Riel Banknote Detection and Identification.pdf`: supports the current detector-plus-fragment-verifier path, with explicit unknown/low-confidence handling for ambiguous visible pieces. It argues against starting with full-note crop classification or amodal segmentation before real error analysis proves those are needed.
- `Banknote Detection Research Handoff.pdf`: broader technical survey covering keypoint/homography systems, YOLO-family detectors, amodal/instance segmentation, and synthetic occlusion ideas. It is useful for later verifier or segmentation experiments, but the main risk it names for KHR fans is still repetitive local texture plus curved/non-planar notes.
- `Cambodian Riel Banknote Version Checklist for Denomination Detection.pdf`: denomination/version reference for deciding whether labels stay denomination-only or include note series.
- `Research on Additional Image Sources for Cambodian 20,000 and 50,000 Riel Banknotes.pdf`: rare-class source research for `KHR_20000` and `KHR_50000`.

## How To Use

Use these PDFs as evidence and source context. Put active model decisions in root `model.md`.
