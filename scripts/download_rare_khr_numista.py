from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]

PAGES = [
    {
        "label": "KHR_20000",
        "family": "1995",
        "url": "https://en.numista.com/217619",
        "license_note": "Mixed credits on page; treat as internal/reference unless individual image license is verified.",
    },
    {
        "label": "KHR_20000",
        "family": "2008",
        "url": "https://en.numista.com/206056",
        "license_note": "CC BY-SA credited on Numista page.",
    },
    {
        "label": "KHR_20000",
        "family": "2017_issued_2018",
        "url": "https://en.numista.com/208877",
        "license_note": "CC BY-NC credited on Numista page; internal/academic prototype only.",
    },
    {
        "label": "KHR_50000",
        "family": "2013_issued_2014",
        "url": "https://en.numista.com/215642",
        "license_note": "CC BY-SA credited on Numista page.",
    },
    {
        "label": "KHR_50000",
        "family": "2001",
        "url": "https://en.numista.com/207992",
        "license_note": "Copyright attribution only on Numista page; use as internal reference unless permission is obtained.",
    },
    {
        "label": "KHR_50000",
        "family": "1995_1998",
        "url": "https://en.numista.com/211651",
        "license_note": "Mixed credits on page, including CC0 for obverse; verify before public release.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download rare KHR Numista reference images.")
    parser.add_argument("--out", default="data/reference/numista_rare_khr")
    return parser.parse_args()


def request(url: str) -> requests.Response:
    response = requests.get(url, headers={"User-Agent": "CashSnap research downloader/0.1"}, timeout=45)
    response.raise_for_status()
    return response


def original_photo_urls(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    urls: list[tuple[str, str]] = []
    seen: set[str] = set()
    for img in soup.find_all("img"):
        src = img.get("src") or ""
        alt = img.get("alt") or ""
        if "/catalogue/photos/" not in src:
            continue
        original = re.sub(r"-\d+\.jpg$", "-original.jpg", src)
        if original not in seen:
            seen.add(original)
            urls.append((original, alt))
    return urls


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheet(paths: list[Path], output: Path) -> None:
    if not paths:
        return
    cols, thumb_w, thumb_h = 4, 240, 160
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 34) + 34), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), "Numista rare KHR references", fill="black")
    for index, path in enumerate(paths):
        with Image.open(path).convert("RGB") as image:
            image.thumbnail((thumb_w, thumb_h))
            x = (index % cols) * thumb_w
            y = 34 + (index // cols) * (thumb_h + 34)
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + thumb_h + 4), path.name[:34], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)


def main() -> None:
    args = parse_args()
    out_root = (ROOT / args.out).resolve()
    manifest_rows: list[dict[str, str]] = []
    downloaded_paths: list[Path] = []
    for page in PAGES:
        html = request(page["url"]).text
        for index, (image_url, alt) in enumerate(original_photo_urls(html), start=1):
            suffix = Path(image_url.split("?", 1)[0]).suffix or ".jpg"
            side = "front" if "obverse" in alt.lower() else "back" if "reverse" in alt.lower() else f"extra_{index}"
            target = out_root / page["label"] / page["family"] / f"{index:02d}_{side}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size == 0:
                target.write_bytes(request(image_url).content)
            downloaded_paths.append(target)
            manifest_rows.append(
                {
                    "label": page["label"],
                    "family": page["family"],
                    "page_url": page["url"],
                    "image_url": image_url,
                    "alt": alt,
                    "path": str(target.relative_to(ROOT)),
                    "license_note": page["license_note"],
                }
            )

    write_csv(out_root / "manifest.csv", manifest_rows)
    make_contact_sheet(downloaded_paths, out_root / "contact.jpg")
    print(f"Downloaded/verified {len(downloaded_paths)} Numista rare KHR reference images")
    print(f"Output: {out_root}")
    print(f"Manifest: {out_root / 'manifest.csv'}")


if __name__ == "__main__":
    main()
