# KHR Circulation Scope

Purpose: keep CashSnap training focused on banknotes a user is likely to photograph in daily use, instead of mixing modern notes with old, collector, commemorative, or low-priority legal-tender variants.

Official source of truth:

- National Bank of Cambodia, Banknotes in Circulation: https://www.nbc.gov.kh/english/about_the_bank/banknotes_in_circulation.php
- Checked on 2026-05-25.

## Current Problem

The existing CashSnap KHR classes are denomination-only:

- `KHR_500`
- `KHR_1000`
- `KHR_2000`
- `KHR_5000`
- `KHR_10000`
- `KHR_20000`
- `KHR_50000`

The current synthetic/reference pipeline has mixed multiple issue eras inside each denomination. For example, the local NBC reference folder includes modern and older variants such as 2008/1995-era `KHR_20000`, older `KHR_50000`, and multiple older `KHR_1000`/`KHR_2000`/`KHR_5000` designs. Some pristine cutout assets also come from Numista or older scene crops.

That means the current e2/e4 checkpoints were not base-pretrained on old KHR notes, but they were fine-tuned with synthetic/reference data that likely includes old or low-priority designs.

## Scope Rule

Before more fan training, split KHR assets into explicit buckets:

- `target_modern_common`: designs we want the app to recognize first in real phone photos.
- `target_modern_rare`: valid/current designs that are uncommon but worth optional coverage.
- `legacy_or_low_priority`: older legal-tender or collector-like designs that should not drive first-pass training.
- `junk_or_unusable`: site assets, specimen-heavy images, watermarked images, bad cutouts, or unclear references.

Do not generate new synthetic training data from `legacy_or_low_priority` unless the experiment explicitly says it is testing legacy support.

`scripts/curate_reference_images.py` applies this rule to the local NBC reference download and writes `manifests/khr_nbc_curated_manifest.csv`. Its current bucket counts are:

- `target_modern_common`: 18 assets.
- `target_modern_rare`: 14 assets.
- `legacy_or_low_priority`: 36 assets.
- `junk_or_unusable`: 14 assets.

For first-pass KHR synthesis, use only `target_modern_common` unless the experiment explicitly adds optional rare denominations.

`scripts/audit_asset_bank_scope.py` can audit transparent asset-bank manifests against these buckets. On `rare_pristine_asset_bank_v1`, it currently finds only 2 automatically scoped `target_modern_common` assets, 13 `legacy_or_low_priority` assets, and 26 scene crops that still need visual design/version review before training.

`scripts/build_current_khr_cutout_bank.py --clean` turns the 18 `target_modern_common` NBC assets into transparent PNGs, binary masks, a manifest, and a contact sheet at `data/asset_candidates/khr_nbc_current_cutout_bank_v1/`. Use this bank, not the mixed raw NBC folder, for fresh first-pass current-KHR synthetic probes.

## Current NBC Scope Read

NBC's Banknotes in Circulation page lists multiple issue eras for some denominations. CashSnap should not interpret "in circulation" as "good first-pass training source" without product filtering.

| Product bucket | Denominations / issues | Training rule |
| --- | --- | --- |
| `target_modern_common` | KHR 500 (2015), 1,000 (2013/2017), 2,000 (2013/2022), 5,000 (2017), 10,000 (2015), 20,000 (2018), 50,000 (2014) | First-pass KHR target set. Use front/back assets that match these issues. |
| `target_modern_rare` | KHR 50, 100 (2015), 200 (2022), 15,000 (2019), 30,000 (2021), 100,000 (2013), 200,000 (2024) | Keep as optional current coverage; do not let these distract from common KHR fan counting. |
| `legacy_or_low_priority` | Older NBC-listed variants such as 1995, 1999, 2001, 2003, 2005, and 2008 issues | Keep for visual audit or explicit legacy experiments only. |
| `junk_or_unusable` | Site assets, specimen-marked images, unclear references, bad masks/crops | Do not train from these. |

## Candidate Product Scope

The NBC circulation page currently includes denominations beyond the existing class list, including `KHR_50`, `KHR_100`, `KHR_200`, `KHR_15000`, `KHR_30000`, `KHR_100000`, and `KHR_200000`.

For the first mobile/browser detector, prefer this scope:

- Keep common daily-use KHR denominations in the primary model.
- Add missing modern classes only after collecting/curating enough real or clean modern references.
- Keep commemorative/rare denominations out of the first fan-counting benchmark unless the user confirms they matter for the app.

## Immediate Remediation

1. Audit `data/reference/khr_nbc/`, `data/curated/reference/khr_nbc/`, and `data/asset_candidates/*` into the four buckets above.
2. Rebuild the synthetic asset bank from `target_modern_common` only.
3. Regenerate fan/radial synthetic data from the cleaned bank.
4. Retrain/evaluate against normal validation plus the real fan benchmark.
