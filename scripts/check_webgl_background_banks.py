#!/usr/bin/env python
"""Validate the WebGL background-bank review registry."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_background_banks_v1.json"
VALID_REVIEW_STATUSES = {"pending_review", "proof_only", "accepted", "rejected"}
VALID_ARTIFACT_STATUSES = {"smoke", "diagnostic", "trainable-candidate", "promoted"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--require-path", type=Path, default=None, help="Require this background directory to be allowed.")
    parser.add_argument(
        "--artifact-status",
        choices=sorted(VALID_ARTIFACT_STATUSES),
        default="",
        help="Artifact status that wants to use --require-path.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    return resolve(path).resolve().relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict:
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing background bank config: {resolved}")
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"{resolved}: expected JSON object")
    return document


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def image_count(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    args = parse_args()
    config = read_json(args.config)
    min_trainable_images = int(config.get("min_trainable_images", 1))
    require(min_trainable_images > 0, "min_trainable_images must be positive")
    banks = config.get("banks", [])
    require(isinstance(banks, list) and banks, "background config must contain banks")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    rows: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()
    allowed_counts: Counter[str] = Counter()

    for row in banks:
        require(isinstance(row, dict), "background bank rows must be objects")
        bank_id = str(row.get("id", "")).strip()
        require(bank_id, "background bank id must be non-empty")
        require(bank_id not in seen_ids, f"duplicate background bank id: {bank_id}")
        seen_ids.add(bank_id)

        path_text = str(row.get("path", "")).strip()
        require(path_text, f"{bank_id}: missing path")
        path = resolve(Path(path_text)).resolve()
        try:
            path.relative_to((ROOT / "data" / "backgrounds").resolve())
        except ValueError as exc:
            raise SystemExit(f"{bank_id}: background path must stay under data/backgrounds") from exc
        require(path.exists() and path.is_dir(), f"{bank_id}: missing background directory {path}")
        path_key = path.relative_to(ROOT).as_posix()
        require(path_key not in seen_paths, f"duplicate background bank path: {path_key}")
        seen_paths.add(path_key)

        review_status = str(row.get("review_status", "")).strip()
        require(review_status in VALID_REVIEW_STATUSES, f"{bank_id}: invalid review_status {review_status!r}")
        allowed = [str(item) for item in row.get("allowed_artifact_statuses", [])]
        unknown_allowed = sorted(set(allowed) - VALID_ARTIFACT_STATUSES)
        require(not unknown_allowed, f"{bank_id}: unknown allowed artifact statuses {unknown_allowed}")
        count = image_count(path)
        require(count > 0, f"{bank_id}: no background images found")
        if review_status == "accepted":
            require(count >= min_trainable_images, f"{bank_id}: accepted trainable bank needs at least {min_trainable_images} images")
            require("trainable-candidate" in allowed, f"{bank_id}: accepted bank must allow trainable-candidate")
            require(str(row.get("review_basis", "")).strip(), f"{bank_id}: accepted bank must include review_basis")
        else:
            require("trainable-candidate" not in allowed, f"{bank_id}: only accepted banks may allow trainable-candidate")
            require("promoted" not in allowed, f"{bank_id}: only accepted banks may allow promoted")

        status_counts[review_status] += 1
        allowed_counts.update(allowed)
        rows.append(
            {
                "id": bank_id,
                "path": path_key,
                "review_status": review_status,
                "allowed_artifact_statuses": allowed,
                "images": count,
                "blocker": str(row.get("blocker", "")),
            }
        )

    if args.require_path is not None:
        require(args.artifact_status, "--artifact-status is required with --require-path")
        requested_path = repo_rel(args.require_path)
        matches = [row for row in rows if row["path"] == requested_path]
        require(matches, f"background directory is not registered: {requested_path}")
        bank = matches[0]
        allowed = set(bank["allowed_artifact_statuses"])
        require(
            args.artifact_status in allowed,
            (
                f"{bank['id']}: review_status={bank['review_status']} does not allow "
                f"artifact_status={args.artifact_status}; blocker={bank['blocker']}"
            ),
        )

    print(
        f"ok: {config.get('name')} has {len(rows)} bank(s), "
        f"statuses={dict(sorted(status_counts.items()))}, allowed={dict(sorted(allowed_counts.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
