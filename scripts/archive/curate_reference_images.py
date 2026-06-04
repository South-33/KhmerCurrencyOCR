from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIRST_PASS_KHR = {"500", "1000", "2000", "5000", "10000", "20000", "50000"}
OPTIONAL_KHR = {"50", "100", "200", "15000", "30000", "100000", "200000"}

TARGET_MODERN_COMMON_STEMS = {
    "005_10000_b_2015",
    "006_10000_f_2015",
    "011_1000_back_2013",
    "013_1000_front_2013",
    "016_1000_front_new_2017",
    "020_1000_reverse_new_2017",
    "033_20000_front_new_2018",
    "036_20000_reverse_new_2018",
    "037_2000_back_2013",
    "038_2000_back_2022",
    "041_2000_front_2013",
    "042_2000_front_2022",
    "053_50000_new_back",
    "054_50000_new_front",
    "059_5000_front_new_2017",
    "062_5000_reverse_new_2017",
    "063_500_back_new_2015",
    "066_500_front_new_2015",
}

TARGET_MODERN_RARE_STEMS = {
    "001_100000_front",
    "004_100000_reverse",
    "021_100_back_new_2015",
    "024_100_front_new_2015",
    "027_15000_front_new_2019",
    "028_15000_reverse_new_2019",
    "029_200000_front_new_2024",
    "030_200000_reverse_new_2024",
    "045_200_back_2022",
    "046_200_front_2022",
    "049_30000_front_new_2021",
    "050_30000_reverse_new_2021",
    "069_50_front_new_2002",
    "070_50_reverse_new_2002",
}

BUCKETS = [
    "target_modern_common",
    "target_modern_rare",
    "legacy_or_low_priority",
    "junk_or_unusable",
]


def parse_nbc_filename(path: Path) -> tuple[str, str, str]:
    name = path.stem.lower()
    match = re.search(r"_(\d{2,6})(?:_|$)", name)
    if not match:
        return "junk_or_unusable", "", "site asset or non-banknote file"
    denomination = match.group(1)
    if "specimen" in name:
        return "junk_or_unusable", denomination, "specimen-marked reference image"
    if name in TARGET_MODERN_COMMON_STEMS:
        return "target_modern_common", denomination, "current first-pass KHR issue"
    if name in TARGET_MODERN_RARE_STEMS:
        return "target_modern_rare", denomination, "current optional/rare/low-value KHR issue"
    if denomination in FIRST_PASS_KHR or denomination in OPTIONAL_KHR:
        return "legacy_or_low_priority", denomination, "older circulation variant; exclude from first-pass synthesis"
    return "legacy_or_low_priority", denomination, "banknote outside first-pass scope"


def image_ok(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as img:
            width, height = img.size
    except Exception as exc:
        return False, f"unreadable: {exc}"
    if width < 80 or height < 30:
        return False, f"too small {width}x{height}"
    return True, f"{width}x{height}"


def curate_nbc() -> list[dict[str, str]]:
    source_dir = ROOT / "data" / "reference" / "khr_nbc"
    curated_dir = ROOT / "data" / "curated" / "reference" / "khr_nbc"
    for bucket in BUCKETS:
        bucket_dir = curated_dir / bucket
        if bucket_dir.exists():
            shutil.rmtree(bucket_dir)
    rows: list[dict[str, str]] = []
    for path in sorted(source_dir.glob("*")):
        if not path.is_file():
            continue
        ok, note = image_ok(path)
        bucket, denomination, scope_note = parse_nbc_filename(path) if ok else ("junk_or_unusable", "", note)
        target_dir = curated_dir / bucket / (f"KHR_{denomination}" if denomination else "site_assets")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        shutil.copy2(path, target)
        rows.append(
            {
                "source": str(path.relative_to(ROOT)),
                "curated_path": str(target.relative_to(ROOT)),
                "bucket": bucket,
                "denomination": denomination,
                "note": note,
                "scope_note": scope_note,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(image_paths: list[Path], output: Path, title: str, thumb_w: int = 180) -> None:
    if not image_paths:
        return
    thumbs = []
    for path in image_paths:
        with Image.open(path).convert("RGB") as img:
            ratio = thumb_w / img.width
            thumb_h = max(1, int(img.height * ratio))
            img = img.resize((thumb_w, thumb_h))
            thumbs.append((path, img))
    cols = 4
    label_h = 36
    row_h = max(img.height for _, img in thumbs) + label_h
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * row_h + 40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="black")
    for index, (path, img) in enumerate(thumbs):
        x = (index % cols) * thumb_w
        y = 40 + (index // cols) * row_h
        sheet.paste(img, (x, y))
        draw.text((x + 4, y + img.height + 4), path.name[:28], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    rows = curate_nbc()
    write_csv(ROOT / "manifests" / "khr_nbc_curated_manifest.csv", rows)
    for bucket in BUCKETS:
        paths = [
            ROOT / row["curated_path"]
            for row in rows
            if row["bucket"] == bucket and Path(row["curated_path"]).suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        make_contact_sheet(paths, ROOT / "data" / "curated" / "reference" / f"khr_nbc_{bucket}_contact.jpg", f"KHR NBC {bucket}")
    print("Curated counts:")
    for bucket in BUCKETS:
        print(f"{bucket}: {sum(1 for row in rows if row['bucket'] == bucket)}")


if __name__ == "__main__":
    main()
