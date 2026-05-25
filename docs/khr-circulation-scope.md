# KHR Circulation Scope

Purpose: keep CashSnap training focused on banknotes a user is likely to photograph in daily use, instead of mixing modern notes with old, collector, commemorative, or low-priority legal-tender variants.

Official source of truth:

- National Bank of Cambodia, Banknotes in Circulation: https://www.nbc.gov.kh/english/about_the_bank/banknotes_in_circulation.php
- Checked on 2026-05-24.

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
