#!/usr/bin/env python
"""Validate packaged WebGL detect, OBB, and fragment label views."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALID_ARTIFACT_STATUSES = {"smoke", "diagnostic", "trainable-candidate"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Packaged WebGL dataset root.")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> object:
    if not path.exists():
        raise SystemExit(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_label_rows(path: Path, expected_columns: int) -> list[list[float]]:
    if not path.exists():
        raise SystemExit(f"missing label file: {path}")
    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != expected_columns:
            raise SystemExit(f"{path}:{line_number}: expected {expected_columns} columns, got {len(parts)}")
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid numeric label: {line}") from exc
        if class_id < 0:
            raise SystemExit(f"{path}:{line_number}: negative class id")
        if any(value < 0.0 or value > 1.0 for value in values):
            raise SystemExit(f"{path}:{line_number}: normalized value out of range")
        rows.append([float(class_id), *values])
    return rows


def main() -> int:
    args = parse_args()
    dataset_root = resolve_path(args.root)
    manifest = read_json(dataset_root / "manifest.json")
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("manifest.json must be a non-empty list")
    manifest_by_variant = {}

    trainable_obb_images = 0
    rejected_obb_images = 0
    fragment_count = 0
    ignored_fragment_count = 0
    visible_instance_count = 0
    class_counts: Counter[str] = Counter()
    layer_audit_totals: Counter[str] = Counter()
    for row in manifest:
        if not isinstance(row, dict):
            raise SystemExit("manifest row must be an object")
        manifest_by_variant[int(row["variant"])] = row
        boxes_doc = read_json(dataset_root / row["visible_boxes"])
        visible_boxes = boxes_doc.get("boxes", [])
        visible_instance_count += len(visible_boxes)
        class_counts.update(str(box.get("className", "unknown")) for box in visible_boxes)
        layer_audit = read_json(dataset_root / row["layer_audit"])
        for key in ("visiblePixels", "overlapPixels", "occluderPixels", "violations"):
            layer_audit_totals[key] += int(layer_audit.get(key, 0))
        detect_rows = read_label_rows(dataset_root / row["label"], expected_columns=5)
        if len(detect_rows) != len(visible_boxes):
            raise SystemExit(f"{row['label']}: {len(detect_rows)} detect labels for {len(visible_boxes)} visible boxes")
        for preview_key in ("detect_preview", "fragment_preview", "id_overlay"):
            preview_path = dataset_root / row[preview_key]
            if not preview_path.exists():
                raise SystemExit(f"missing preview image: {preview_path}")

        fragment_rows = read_label_rows(dataset_root / row["fragment_label"], expected_columns=5)
        fragment_metadata = read_json(dataset_root / row["fragment_metadata"])
        if len(fragment_rows) != len(fragment_metadata):
            raise SystemExit(
                f"{row['fragment_label']}: {len(fragment_rows)} fragment labels for "
                f"{len(fragment_metadata)} metadata rows"
            )
        for fragment in fragment_metadata:
            parent_index = fragment.get("parentVisibleIndex")
            if not isinstance(parent_index, int) or parent_index < 0 or parent_index >= len(visible_boxes):
                raise SystemExit(f"{row['fragment_metadata']}: invalid parentVisibleIndex {parent_index}")
        fragment_count += len(fragment_rows)
        ignored_fragment_metadata = read_json(dataset_root / row["fragment_ignored_metadata"])
        if not isinstance(ignored_fragment_metadata, list):
            raise SystemExit(f"{row['fragment_ignored_metadata']}: ignored fragment metadata must be a list")
        for fragment in ignored_fragment_metadata:
            parent_index = fragment.get("parentVisibleIndex")
            if not isinstance(parent_index, int) or parent_index < 0 or parent_index >= len(visible_boxes):
                raise SystemExit(f"{row['fragment_ignored_metadata']}: invalid parentVisibleIndex {parent_index}")
            if not str(fragment.get("ignore_reason", "")).strip():
                raise SystemExit(f"{row['fragment_ignored_metadata']}: ignored fragment missing ignore_reason")
        ignored_fragment_count += len(ignored_fragment_metadata)

        obb_status = row.get("obb_status")
        if obb_status == "accepted":
            trainable_obb_images += 1
            read_label_rows(dataset_root / row["obb_label"], expected_columns=9)
            obb_metadata = read_json(dataset_root / row["obb_metadata"])
            bad_statuses = [item.get("status") for item in obb_metadata if item.get("status") != "exported"]
            if bad_statuses:
                raise SystemExit(f"{row['obb_metadata']}: accepted OBB row has rejected instances: {bad_statuses}")
        elif obb_status == "rejected":
            rejected_obb_images += 1
            read_label_rows(dataset_root / row["obb_diagnostic_label"], expected_columns=9)
            obb_metadata = read_json(dataset_root / row["obb_diagnostic_metadata"])
            if all(item.get("status") == "exported" for item in obb_metadata):
                raise SystemExit(f"{row['obb_diagnostic_metadata']}: rejected OBB row has no rejected instances")
        else:
            raise SystemExit(f"unknown obb_status: {obb_status}")

    obb_summary = read_json(dataset_root / "obb" / "summary.json")
    fragment_summary = read_json(dataset_root / "fragments" / "summary.json")
    if obb_summary.get("trainable_obb_images") != trainable_obb_images:
        raise SystemExit("obb summary trainable image count mismatch")
    if obb_summary.get("rejected_obb_images") != rejected_obb_images:
        raise SystemExit("obb summary rejected image count mismatch")
    if fragment_summary.get("fragments") != fragment_count:
        raise SystemExit("fragment summary count mismatch")
    if fragment_summary.get("ignored_fragments") != ignored_fragment_count:
        raise SystemExit("fragment summary ignored count mismatch")
    qa_summary = read_json(dataset_root / "qa" / "summary.json")
    quarantine = read_json(dataset_root / "qa" / "quarantine.json")
    contact_index = read_json(dataset_root / "qa" / "contact_index.json")
    visual_quality = read_json(dataset_root / "qa" / "visual_quality.json")
    if len(contact_index.get("rows", [])) != len(manifest):
        raise SystemExit("contact index row count mismatch")
    if not (dataset_root / contact_index.get("contact_sheet", "")).exists():
        raise SystemExit("contact index points to missing contact sheet")
    if not isinstance(quarantine.get("rows"), list):
        raise SystemExit("quarantine rows must be a list")
    quarantine_counts = Counter(str(row.get("action", "")) for row in quarantine["rows"])
    if quarantine.get("counts") != dict(sorted(quarantine_counts.items())):
        raise SystemExit("quarantine count mismatch")
    visual_quality_rows = visual_quality.get("rows", []) if isinstance(visual_quality, dict) else []
    if not isinstance(visual_quality_rows, list):
        raise SystemExit("visual_quality rows must be a list")
    if len(visual_quality_rows) != len(manifest):
        raise SystemExit("visual_quality row count mismatch")
    visual_quality_by_variant = {int(row["variant"]): row for row in visual_quality_rows}
    if set(visual_quality_by_variant) != set(manifest_by_variant):
        raise SystemExit("visual_quality variants do not match manifest")
    visual_quality_counts = Counter(str(row.get("status", "")) for row in visual_quality_rows)
    if visual_quality.get("counts") != dict(sorted(visual_quality_counts.items())):
        raise SystemExit("visual_quality count mismatch")
    if qa_summary.get("images") != len(manifest):
        raise SystemExit("qa summary image count mismatch")
    if qa_summary.get("visible_instances", {}).get("total") != visible_instance_count:
        raise SystemExit("qa summary visible instance count mismatch")
    if qa_summary.get("fragments", {}).get("total") != fragment_count:
        raise SystemExit("qa summary fragment count mismatch")
    if qa_summary.get("fragments", {}).get("ignored_total") != ignored_fragment_count:
        raise SystemExit("qa summary ignored fragment count mismatch")
    if qa_summary.get("class_counts") != dict(sorted(class_counts.items())):
        raise SystemExit("qa summary class count mismatch")
    if qa_summary.get("layer_audit_totals") != dict(sorted(layer_audit_totals.items())):
        raise SystemExit("qa summary layer audit totals mismatch")
    if qa_summary.get("visual_quality", {}).get("status_counts") != dict(sorted(visual_quality_counts.items())):
        raise SystemExit("qa summary visual_quality status count mismatch")
    hash_paths = {
        "visual": "image",
        "id": "id",
        "detect_label": "label",
        "fragment_label": "fragment_label",
        "detect_preview": "detect_preview",
        "fragment_preview": "fragment_preview",
        "id_overlay": "id_overlay",
    }
    for detail in qa_summary.get("images_detail", []):
        variant = int(detail["variant"])
        manifest_row = manifest_by_variant.get(variant)
        if manifest_row is None:
            raise SystemExit(f"qa summary references unknown variant {variant}")
        for hash_key, manifest_key in hash_paths.items():
            expected_hash = detail.get("sha256", {}).get(hash_key)
            actual_hash = sha256_file(dataset_root / manifest_row[manifest_key])
            if expected_hash != actual_hash:
                raise SystemExit(f"qa summary hash mismatch for variant {variant} {hash_key}")
        if detail.get("visual_quality", {}).get("status") != visual_quality_by_variant[variant].get("status"):
            raise SystemExit(f"qa summary visual_quality status mismatch for variant {variant}")
    recipe = read_json(dataset_root / "recipe.json")
    artifact_status = recipe.get("artifact_status")
    if artifact_status not in VALID_ARTIFACT_STATUSES:
        raise SystemExit(f"recipe artifact_status must be one of {sorted(VALID_ARTIFACT_STATUSES)}")
    if not str(recipe.get("recipe_name", "")).strip():
        raise SystemExit("recipe_name must be non-empty")
    outputs = recipe.get("outputs", {})
    if outputs.get("qa_summary"):
        expected = str((dataset_root / "qa" / "summary.json").relative_to(ROOT))
        if outputs["qa_summary"] != expected:
            raise SystemExit("recipe qa_summary output path mismatch")
    if outputs.get("visual_quality"):
        expected = str((dataset_root / "qa" / "visual_quality.json").relative_to(ROOT))
        if outputs["visual_quality"] != expected:
            raise SystemExit("recipe visual_quality output path mismatch")

    print(
        f"ok: {len(manifest)} images, {fragment_count} fragments, "
        f"{trainable_obb_images} trainable OBB images, {rejected_obb_images} rejected OBB images"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
