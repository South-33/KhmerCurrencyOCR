from __future__ import annotations

import csv
import argparse
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFESTS = ROOT / "manifests"

USD_COMMONS = [
    ("USD_1", "front", "https://commons.wikimedia.org/wiki/File:United_States_one_dollar_bill,_obverse.jpg"),
    ("USD_1", "back", "https://commons.wikimedia.org/wiki/File:United_States_one_dollar_bill,_reverse.jpg"),
    ("USD_5", "front", "https://commons.wikimedia.org/wiki/File:US_$5_Series_2006_obverse.jpg"),
    ("USD_5", "back", "https://commons.wikimedia.org/wiki/File:US_$5_Series_2006_reverse.jpg"),
    ("USD_5", "front", "https://commons.wikimedia.org/wiki/File:Obverse_of_the_series_2021_$5_Federal_Reserve_Note.jpg"),
    ("USD_5", "back", "https://commons.wikimedia.org/wiki/File:Reverse_of_the_series_2021_$5_Federal_Reserve_Note.jpg"),
    ("USD_10", "front", "https://commons.wikimedia.org/wiki/File:US10dollarbill-Series_2004A.jpg"),
    ("USD_10", "back", "https://commons.wikimedia.org/wiki/File:US_$10_Series_2004_reverse.jpg"),
    ("USD_10", "front", "https://commons.wikimedia.org/wiki/File:Obverse_of_the_series_2021_$10_Federal_Reserve_Note.jpg"),
    ("USD_10", "back", "https://commons.wikimedia.org/wiki/File:Reverse_of_the_series_2021_$10_Federal_Reserve_Note.jpg"),
    ("USD_20", "front", "https://commons.wikimedia.org/wiki/File:US_$20_Series_2006_Obverse.jpg"),
    ("USD_20", "back", "https://commons.wikimedia.org/wiki/File:US_$20_Series_2006_Reverse.jpg"),
    ("USD_20", "front", "https://commons.wikimedia.org/wiki/File:Obverse_of_the_series_2017A_$20_Federal_Reserve_Note.jpg"),
    ("USD_20", "back", "https://commons.wikimedia.org/wiki/File:Reverse_of_the_series_2017A_$20_Federal_Reserve_Note.jpg"),
    ("USD_50", "front", "https://commons.wikimedia.org/wiki/File:$50_Dollar_Bill_Series_1969C_Front.jpg"),
    ("USD_50", "back", "https://commons.wikimedia.org/wiki/File:$50_Dollar_Bill_Series_1969C_Back.jpg"),
    ("USD_50", "front", "https://commons.wikimedia.org/wiki/File:50_USD_Series_2004_Note_Front.jpg"),
    ("USD_50", "back", "https://commons.wikimedia.org/wiki/File:50_USD_Series_2004_Note_Back.jpg"),
    ("USD_50", "front", "https://commons.wikimedia.org/wiki/File:Obverse_of_the_$50_Federal_Reserve_Note.jpg"),
    ("USD_100", "front", "https://commons.wikimedia.org/wiki/File:U.S._hundred_dollar_bill,_1999.jpg"),
    ("USD_100", "back", "https://commons.wikimedia.org/wiki/File:POSTERIOR-100DOLLARSNOTES-SERIE1996.jpg"),
    ("USD_100", "front", "https://commons.wikimedia.org/wiki/File:Usdollar100front.jpg"),
    ("USD_100", "back", "https://commons.wikimedia.org/wiki/File:USA_100_Dollar_Bill_Series2009_Reverse.png"),
    ("USD_100", "front", "https://commons.wikimedia.org/wiki/File:Small-head_$100s_front.jpg"),
    ("USD_100", "back", "https://commons.wikimedia.org/wiki/File:Small-head_$100s_back.jpg"),
]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def request(url: str) -> requests.Response:
    response = requests.get(url, headers={"User-Agent": "CashSnap research downloader/0.1"}, timeout=45)
    response.raise_for_status()
    return response


def resolve_commons_original(file_page: str) -> tuple[str, str]:
    title = file_page.rsplit("/wiki/", 1)[1]
    api = "https://commons.wikimedia.org/w/api.php"
    response = request(
        api
        + "?action=query&format=json&prop=imageinfo&iiprop=url|mime|size|extmetadata&titles="
        + title
    )
    pages = response.json()["query"]["pages"]
    imageinfo = next(iter(pages.values()))["imageinfo"][0]
    return imageinfo["url"], imageinfo.get("mime", "")


def download_file(url: str, target: Path) -> tuple[bool, str]:
    if target.exists() and target.stat().st_size > 0:
        return False, "exists"
    response = request(url)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return True, f"downloaded {len(response.content)} bytes"


def download_usd_commons() -> None:
    rows = []
    out_dir = DATA / "reference" / "usd_wikimedia"
    for index, (label, side, page_url) in enumerate(USD_COMMONS, start=1):
        try:
            original_url, mime = resolve_commons_original(page_url)
            ext = Path(original_url.split("?", 1)[0]).suffix or ".jpg"
            target = out_dir / label / f"{index:02d}_{side}_{safe_name(Path(page_url).name)}{ext}"
            changed, status = download_file(original_url, target)
            rows.append(
                {
                    "label": label,
                    "side": side,
                    "page_url": page_url,
                    "download_url": original_url,
                    "path": str(target.relative_to(ROOT)),
                    "mime": mime,
                    "status": status,
                    "usage_note": "USD currency images have reproduction restrictions; do not publish raw high-resolution two-sided datasets.",
                }
            )
            print(f"USD {label} {side}: {status}")
        except Exception as exc:
            rows.append({"label": label, "side": side, "page_url": page_url, "status": f"failed: {exc}"})
            print(f"USD {label} {side}: failed: {exc}")
    write_csv(MANIFESTS / "usd_wikimedia_manifest.csv", rows)


def download_nbc_khr() -> None:
    url = "https://www.nbc.gov.kh/english/about_the_bank/banknotes_in_circulation.php"
    out_dir = DATA / "reference" / "khr_nbc"
    rows = []
    html = request(url).text
    soup = BeautifulSoup(html, "html.parser")
    image_urls = []
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        absolute = urljoin(url, src)
        if any(token in absolute.lower() for token in ["banknote", "currency", "riel", "images"]):
            image_urls.append(absolute)
    image_urls = sorted(set(image_urls))

    for index, image_url in enumerate(image_urls, start=1):
        try:
            ext = Path(image_url.split("?", 1)[0]).suffix or ".jpg"
            target = out_dir / f"{index:03d}_{safe_name(Path(image_url).stem)}{ext}"
            changed, status = download_file(image_url, target)
            rows.append(
                {
                    "source_page": url,
                    "download_url": image_url,
                    "path": str(target.relative_to(ROOT)),
                    "status": status,
                    "usage_note": "Official NBC reference image; verify reuse rights before public dataset/model release. Some notes may include SPECIMEN watermarks.",
                }
            )
            print(f"NBC KHR {index}: {status}")
        except Exception as exc:
            rows.append({"source_page": url, "download_url": image_url, "status": f"failed: {exc}"})
            print(f"NBC KHR {index}: failed: {exc}")
    write_csv(MANIFESTS / "khr_nbc_manifest.csv", rows)


def download_hf_usd_side() -> None:
    target = DATA / "raw_datasets" / "hf_usd_side_coco_annotations"
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="ebowwa/usd-side-coco-annotations",
        repo_type="dataset",
        local_dir=target,
    )
    print(f"HF USD Side downloaded to {path}")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_blocked_sources() -> None:
    rows = [
        {
            "source": "Khmer-US-currency Roboflow",
            "url": "https://universe.roboflow.com/robot-yfusg/khmer-us-currency-jofw1",
            "status": "blocked/manual",
            "reason": "Roboflow dataset export requires a signed download/API-key flow from an account.",
            "wanted_format": "YOLOv8/YOLO26 or COCO JSON",
        },
        {
            "source": "Cambodia Currency Project Roboflow",
            "url": "https://universe.roboflow.com/khmer-riel-classification-computer-vision/cambodia-currency-project",
            "status": "blocked/manual",
            "reason": "Roboflow dataset export requires a signed download/API-key flow from an account.",
            "wanted_format": "YOLOv8/YOLO26 or COCO JSON",
        },
        {
            "source": "KHMER SCAN Roboflow",
            "url": "https://universe.roboflow.com/test-gl3sj/khmer-scan",
            "status": "blocked/manual",
            "reason": "Roboflow dataset export requires a signed download/API-key flow from an account.",
            "wanted_format": "YOLOv8/YOLO26 or COCO JSON",
        },
        {
            "source": "Numista Cambodia catalogue",
            "url": "https://en.numista.com/catalogue/cambodia_section-5.html",
            "status": "not downloaded",
            "reason": "Useful for reference discovery, but image reuse/licensing is mixed or unclear. Avoid bulk download until license is verified.",
            "wanted_format": "clean front/back reference images only if license permits",
        },
    ]
    write_csv(MANIFESTS / "blocked_or_manual_sources.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public CashSnap source data.")
    parser.add_argument("--skip-hf", action="store_true", help="Skip the slow full Hugging Face USD dataset snapshot.")
    parser.add_argument("--only-hf-metadata", action="store_true", help="Download only HF metadata/annotation files, not image files.")
    return parser.parse_args()


def download_hf_usd_side_metadata() -> None:
    target = DATA / "raw_datasets" / "hf_usd_side_coco_annotations"
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="ebowwa/usd-side-coco-annotations",
        repo_type="dataset",
        local_dir=target,
        allow_patterns=["*.md", "*.txt", "*.json"],
    )
    print(f"HF USD Side metadata downloaded to {path}")


def main() -> None:
    args = parse_args()
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    if args.only_hf_metadata:
        download_hf_usd_side_metadata()
    elif not args.skip_hf:
        download_hf_usd_side()
    download_usd_commons()
    download_nbc_khr()
    write_blocked_sources()


if __name__ == "__main__":
    main()
