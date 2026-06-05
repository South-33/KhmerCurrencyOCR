#!/usr/bin/env python
"""Check the CashSnap synthetic-data governance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "synthetic_recipes" / "cashsnap_synthetic_governance_v1.json"
DEFAULT_JSON_OUT = ROOT / "runs" / "cashsnap" / "synthetic_governance_latest.json"

RELEASE_STATUSES = {
    "internal_experiment_only",
    "trainable_candidate_internal",
    "release_candidate",
    "public_release",
}
SOURCE_KINDS = {"file", "directory"}
RELEASE_LIMITED_RIGHTS = {
    "blocked_manual",
    "local_capture_review_required",
    "local_dataset_review_required",
    "mixed_manual_review",
    "unknown",
    "usage_review_required",
}
REQUIRED_LIST_FIELDS = [
    "intended_uses",
    "prohibited_uses",
    "limitations",
    "privacy_controls",
    "label_policy",
    "promotion_requirements",
    "source_artifacts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--require-pass", action="store_true", help="Exit non-zero unless governance passes.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return resolve(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_listing_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        rel = repo_path(child)
        digest.update(rel.encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_path(path)}: expected JSON object")
    return data


def add_missing_list_blockers(manifest: dict[str, Any], blockers: list[str]) -> None:
    for field in REQUIRED_LIST_FIELDS:
        rows = manifest.get(field)
        if not isinstance(rows, list) or not rows:
            blockers.append(f"{field} must be a non-empty list")


def source_artifact_rows(manifest: dict[str, Any], blockers: list[str]) -> list[dict[str, Any]]:
    rows = manifest.get("source_artifacts", [])
    if not isinstance(rows, list):
        blockers.append("source_artifacts must be a list")
        return []

    seen: set[str] = set()
    evidence_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            blockers.append(f"source_artifacts[{index}] must be an object")
            continue
        artifact_id = str(row.get("id", "")).strip()
        if not artifact_id:
            blockers.append(f"source_artifacts[{index}] is missing id")
        elif artifact_id in seen:
            blockers.append(f"duplicate source_artifacts id {artifact_id!r}")
        seen.add(artifact_id)

        kind = str(row.get("kind", "")).strip()
        if kind not in SOURCE_KINDS:
            blockers.append(f"{artifact_id or f'source_artifacts[{index}]'}: unknown kind {kind!r}")

        path_text = str(row.get("path", "")).strip()
        if not path_text:
            blockers.append(f"{artifact_id or f'source_artifacts[{index}]'}: missing path")
            path = ROOT / "__missing__"
        else:
            path = resolve(Path(path_text))
            try:
                path.resolve().relative_to(ROOT)
            except ValueError:
                blockers.append(f"{artifact_id}: path must stay inside repo: {path_text}")

        rights_status = str(row.get("rights_status", "")).strip()
        release_policy = str(row.get("release_policy", "")).strip()
        purpose = str(row.get("purpose", "")).strip()
        if not rights_status:
            blockers.append(f"{artifact_id or f'source_artifacts[{index}]'}: missing rights_status")
        if not release_policy:
            blockers.append(f"{artifact_id or f'source_artifacts[{index}]'}: missing release_policy")
        if not purpose:
            blockers.append(f"{artifact_id or f'source_artifacts[{index}]'}: missing purpose")

        evidence = {
            "id": artifact_id,
            "kind": kind,
            "path": repo_path(path) if path_text else "",
            "exists": path.exists(),
            "rights_status": rights_status,
            "release_policy": release_policy,
            "purpose": purpose,
        }
        if not path.exists():
            blockers.append(f"{artifact_id or path_text}: missing source artifact path {path_text}")
        elif kind == "file":
            if not path.is_file():
                blockers.append(f"{artifact_id}: expected file at {path_text}")
            else:
                evidence["sha256"] = file_sha256(path)
                evidence["size_bytes"] = path.stat().st_size
        elif kind == "directory":
            if not path.is_dir():
                blockers.append(f"{artifact_id}: expected directory at {path_text}")
            else:
                listing_sha, file_count = directory_listing_sha256(path)
                evidence["listing_sha256"] = listing_sha
                evidence["file_count"] = file_count
        evidence_rows.append(evidence)
    return evidence_rows


def check_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    path = resolve(manifest_path)
    if not path.exists():
        return {
            "manifest": repo_path(path),
            "status": "missing",
            "blockers": [f"missing governance manifest: {repo_path(path)}"],
            "warnings": [],
            "source_artifacts": [],
        }

    manifest = read_json(path)
    blockers: list[str] = []
    warnings: list[str] = []

    release_status = str(manifest.get("release_status", "")).strip()
    if release_status not in RELEASE_STATUSES:
        blockers.append(f"release_status must be one of {sorted(RELEASE_STATUSES)}")
    for field in ["name", "scope", "public_release_blocker", "model_release_blocker"]:
        if not str(manifest.get(field, "")).strip():
            blockers.append(f"{field} must be present")
    for field in ["public_release_allowed", "model_release_allowed"]:
        if not isinstance(manifest.get(field), bool):
            blockers.append(f"{field} must be boolean")
    add_missing_list_blockers(manifest, blockers)

    evidence_rows = source_artifact_rows(manifest, blockers)
    required_ids = manifest.get("required_source_artifact_ids", [])
    if not isinstance(required_ids, list) or not required_ids:
        blockers.append("required_source_artifact_ids must be a non-empty list")
        required_ids = []
    present_ids = {str(row.get("id", "")).strip() for row in evidence_rows}
    for artifact_id in sorted(str(item).strip() for item in required_ids if str(item).strip()):
        if artifact_id not in present_ids:
            blockers.append(f"required source artifact is missing: {artifact_id}")

    rights_counts = Counter(str(row.get("rights_status", "")) for row in evidence_rows)
    release_limited = [
        str(row.get("id", "") or row.get("path", ""))
        for row in evidence_rows
        if str(row.get("rights_status", "")) in RELEASE_LIMITED_RIGHTS
    ]
    public_allowed = bool(manifest.get("public_release_allowed"))
    model_allowed = bool(manifest.get("model_release_allowed"))
    if (public_allowed or model_allowed or release_status == "public_release") and release_limited:
        blockers.append(
            "public/model release is allowed but rights-limited artifacts remain: "
            + ", ".join(sorted(release_limited))
        )
    if not public_allowed:
        warnings.append("public dataset release is explicitly disabled until a separate rights/privacy review passes")
    if not model_allowed:
        warnings.append("model release is explicitly disabled until transfer, clean/base, rare-class, and browser gates pass")

    return {
        "manifest": repo_path(path),
        "manifest_sha256": file_sha256(path),
        "name": manifest.get("name", ""),
        "scope": manifest.get("scope", ""),
        "release_status": release_status,
        "public_release_allowed": public_allowed,
        "model_release_allowed": model_allowed,
        "public_release_blocker": manifest.get("public_release_blocker", ""),
        "model_release_blocker": manifest.get("model_release_blocker", ""),
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "warnings": warnings,
        "rights_status_counts": dict(sorted(rights_counts.items())),
        "release_limited_source_artifacts": sorted(release_limited),
        "source_artifacts": evidence_rows,
        "check_counts": {
            "intended_uses": len(manifest.get("intended_uses", []) if isinstance(manifest.get("intended_uses"), list) else []),
            "prohibited_uses": len(manifest.get("prohibited_uses", []) if isinstance(manifest.get("prohibited_uses"), list) else []),
            "limitations": len(manifest.get("limitations", []) if isinstance(manifest.get("limitations"), list) else []),
            "privacy_controls": len(manifest.get("privacy_controls", []) if isinstance(manifest.get("privacy_controls"), list) else []),
            "label_policy": len(manifest.get("label_policy", []) if isinstance(manifest.get("label_policy"), list) else []),
            "promotion_requirements": len(
                manifest.get("promotion_requirements", []) if isinstance(manifest.get("promotion_requirements"), list) else []
            ),
            "source_artifacts": len(evidence_rows),
        },
    }


def main() -> int:
    args = parse_args()
    report = check_manifest(args.manifest)
    out = resolve(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "synthetic_governance="
        f"{report['status']} "
        f"source_artifacts={len(report.get('source_artifacts', []))} "
        f"blockers={len(report.get('blockers', []))} "
        f"warnings={len(report.get('warnings', []))}"
    )
    for blocker in report.get("blockers", []):
        print(f"- {blocker}")
    if args.require_pass and report["status"] != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
