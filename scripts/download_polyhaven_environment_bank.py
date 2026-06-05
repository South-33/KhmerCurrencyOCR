#!/usr/bin/env python
"""Download and register Poly Haven HDRI environment maps for WebGL."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://api.polyhaven.com"
LICENSE_URL = "https://polyhaven.com/license"
DEFAULT_USER_AGENT = "KhmerCurrencyOCR-synthetic-pipeline/0.1"
VALID_REVIEW_STATUSES = {"pending_review", "proof_only", "accepted", "rejected"}
VALID_ARTIFACT_STATUSES = {"smoke", "diagnostic", "trainable-candidate", "promoted"}
PRESETS = {
    "cashsnap_retail_lighting_v1": [
        "phone_shop",
        "comfy_cafe",
        "leadenhall_market",
        "poly_haven_studio",
        "small_empty_room_2",
        "small_workshop",
        "urban_street_03",
        "modern_evening_street",
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["none", *sorted(PRESETS)],
        default="cashsnap_retail_lighting_v1",
        help="Curated asset-id preset. Use 'none' with explicit --asset-id values.",
    )
    parser.add_argument("--asset-id", action="append", default=[], help="Poly Haven HDRI asset id/slug to include.")
    parser.add_argument("--resolution", default="1k", help="HDRI resolution to download, e.g. 1k, 2k, 4k.")
    parser.add_argument("--format", choices=["hdr"], default="hdr", help="Environment-map file format to download.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/environment_maps/polyhaven_cashsnap_environment_v1"))
    parser.add_argument(
        "--config-out",
        type=Path,
        default=Path("data/environment_maps/polyhaven_cashsnap_environment_v1_config.json"),
        help="Standalone bank config written after download. The project registry is not edited automatically.",
    )
    parser.add_argument("--bank-id", default="polyhaven_cashsnap_environment_v1")
    parser.add_argument("--config-name", default="polyhaven_cashsnap_environment_v1_config")
    parser.add_argument("--review-status", choices=sorted(VALID_REVIEW_STATUSES), default="pending_review")
    parser.add_argument("--allow-artifact-status", action="append", choices=sorted(VALID_ARTIFACT_STATUSES), default=[])
    parser.add_argument("--review-basis", default="", help="Required when writing an accepted bank.")
    parser.add_argument("--max-total-mb", type=float, default=64.0, help="Refuse planned downloads above this total size.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="Unique Poly Haven API User-Agent.")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--force", action="store_true", help="Redownload files even when a local file exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print the download/config plan without writing files.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    resolved = resolve(path).resolve()
    return resolved.relative_to(ROOT).as_posix()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.strip()
        if key and key not in seen:
            result.append(key)
            seen.add(key)
    return result


def fetch_json(url: str, user_agent: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def api_url(path: str, **query: str) -> str:
    suffix = path if path.startswith("/") else f"/{path}"
    encoded = urllib.parse.urlencode(query)
    return f"{API_ROOT}{suffix}" + (f"?{encoded}" if encoded else "")


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def credit_for(asset: dict[str, Any]) -> str:
    authors = asset.get("authors")
    if isinstance(authors, dict) and authors:
        return "Poly Haven; authors: " + ", ".join(str(name) for name in authors)
    return "Poly Haven"


def target_file_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    name = Path(urllib.parse.unquote(parsed.path)).name
    require(name, f"could not derive filename from download URL: {url}")
    return name


def file_entry(files: dict[str, Any], asset_id: str, resolution: str, file_format: str) -> dict[str, Any]:
    hdri = files.get("hdri")
    require(isinstance(hdri, dict), f"{asset_id}: /files response has no hdri section")
    resolution_row = hdri.get(resolution)
    require(isinstance(resolution_row, dict), f"{asset_id}: no hdri resolution {resolution!r}")
    entry = resolution_row.get(file_format)
    require(isinstance(entry, dict), f"{asset_id}: no {resolution} {file_format} file")
    require(str(entry.get("url", "")).strip(), f"{asset_id}: file entry missing url")
    require(str(entry.get("md5", "")).strip(), f"{asset_id}: file entry missing md5")
    require(int(entry.get("size", 0)) > 0, f"{asset_id}: file entry missing positive size")
    return entry


def plan_downloads(args: argparse.Namespace) -> list[dict[str, Any]]:
    asset_ids = []
    if args.preset != "none":
        asset_ids.extend(PRESETS[args.preset])
    asset_ids.extend(args.asset_id)
    asset_ids = unique(asset_ids)
    require(asset_ids, "no Poly Haven asset ids selected")

    assets = fetch_json(api_url("/assets", type="hdris"), args.user_agent, args.timeout)
    require(isinstance(assets, dict), "/assets response was not an object")

    plan: list[dict[str, Any]] = []
    for asset_id in asset_ids:
        asset = assets.get(asset_id)
        require(isinstance(asset, dict), f"{asset_id}: not found in Poly Haven HDRI assets")
        files = fetch_json(api_url(f"/files/{urllib.parse.quote(asset_id, safe='')}"), args.user_agent, args.timeout)
        require(isinstance(files, dict), f"{asset_id}: /files response was not an object")
        entry = file_entry(files, asset_id, args.resolution, args.format)
        filename = target_file_from_url(str(entry["url"]))
        plan.append(
            {
                "asset_id": asset_id,
                "name": asset.get("name", asset_id),
                "filename": filename,
                "url": str(entry["url"]),
                "md5": str(entry["md5"]),
                "size": int(entry["size"]),
                "categories": asset.get("categories", []),
                "tags": asset.get("tags", []),
                "authors": asset.get("authors", {}),
                "source_url": f"https://polyhaven.com/a/{asset_id}",
                "credit": credit_for(asset),
            }
        )
    return plan


def download_file(item: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> str:
    target = out_dir / str(item["filename"])
    expected_md5 = str(item["md5"])
    expected_size = int(item["size"])
    if target.exists() and not args.force:
        local_md5 = md5_file(target)
        require(local_md5 == expected_md5, f"{target}: existing file md5 {local_md5} != expected {expected_md5}; use --force")
        require(target.stat().st_size == expected_size, f"{target}: existing file size mismatch; use --force")
        return "exists_verified"

    tmp = target.with_name(f"{target.name}.tmp")
    request = urllib.request.Request(str(item["url"]), headers={"User-Agent": args.user_agent})
    digest = hashlib.md5()
    written = 0
    with urllib.request.urlopen(request, timeout=args.timeout) as response, tmp.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            written += len(chunk)

    actual_md5 = digest.hexdigest()
    if written != expected_size or actual_md5 != expected_md5:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"{item['asset_id']}: download verification failed "
            f"(bytes {written}/{expected_size}, md5 {actual_md5}/{expected_md5})"
        )
    tmp.replace(target)
    return "downloaded"


def default_allowed(review_status: str, explicit: list[str]) -> list[str]:
    if explicit:
        return unique(explicit)
    if review_status == "proof_only":
        return ["smoke", "diagnostic"]
    if review_status == "accepted":
        return ["trainable-candidate"]
    return []


def bank_config(args: argparse.Namespace, plan: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    allowed = default_allowed(args.review_status, args.allow_artifact_status)
    require(
        args.review_status == "accepted" or not ({"trainable-candidate", "promoted"} & set(allowed)),
        "only accepted banks may allow trainable-candidate or promoted artifacts",
    )
    require(args.review_status != "accepted" or args.review_basis.strip(), "--review-basis is required for accepted banks")

    assets = [
        {
            "file": item["filename"],
            "asset_id": item["asset_id"],
            "name": item["name"],
            "source_url": item["source_url"],
            "download_url": item["url"],
            "license": "CC0",
            "license_url": LICENSE_URL,
            "credit": item["credit"],
            "authors": item["authors"],
            "md5": item["md5"],
            "size_bytes": item["size"],
            "resolution": args.resolution,
            "format": args.format,
            "categories": item["categories"],
            "tags": item["tags"],
        }
        for item in plan
    ]
    bank: dict[str, Any] = {
        "id": args.bank_id,
        "path": repo_rel(out_dir),
        "review_status": args.review_status,
        "allowed_artifact_statuses": allowed,
        "assets": assets,
        "source_policy": "Downloaded from Poly Haven via its public API; assets are CC0 equirectangular HDRIs with md5/size verification.",
    }
    if args.review_basis.strip():
        bank["review_basis"] = args.review_basis.strip()
    elif args.review_status != "accepted":
        bank["blocker"] = "Downloaded metadata is present, but this bank still needs visual/lighting review before trainable use."

    return {
        "schema_version": 1,
        "name": args.config_name,
        "description": "Standalone Poly Haven environment-map bank config for WebGL lighting/reflection smokes and review.",
        "min_trainable_images": 8,
        "banks": [bank],
    }


def main() -> int:
    args = parse_args()
    out_dir = resolve(args.out_dir).resolve()
    try:
        out_dir.relative_to((ROOT / "data" / "environment_maps").resolve())
    except ValueError as exc:
        raise SystemExit("--out-dir must stay under data/environment_maps") from exc

    plan = plan_downloads(args)
    total_mb = sum(int(item["size"]) for item in plan) / (1024 * 1024)
    require(total_mb <= args.max_total_mb, f"planned download is {total_mb:.1f} MB > --max-total-mb {args.max_total_mb:.1f}")

    print(
        f"planned {len(plan)} Poly Haven HDRI(s), resolution={args.resolution}, "
        f"format={args.format}, total={total_mb:.1f} MB"
    )
    for item in plan:
        print(f"  - {item['asset_id']}: {item['filename']} ({int(item['size']) / (1024 * 1024):.1f} MB)")

    config = bank_config(args, plan, out_dir)
    if args.dry_run:
        print(json.dumps(config, indent=2)[:4000])
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    for item in plan:
        status = download_file(item, out_dir, args)
        print(f"{status}: {item['filename']}")

    config_out = resolve(args.config_out)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {repo_rel(config_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
