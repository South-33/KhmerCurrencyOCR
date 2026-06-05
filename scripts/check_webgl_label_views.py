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
VALID_IGNORED_FRAGMENT_REASONS = {
    "below_min_fragment_pixels",
    "requires_human_review",
    "requires_visual_audit",
}
REVIEW_IGNORED_FRAGMENT_REASONS = {
    "requires_human_review",
    "requires_visual_audit",
}
GEOMETRIC_POSTPROCESS_TOKENS = {
    "crop",
    "distort",
    "homography",
    "lens",
    "perspective",
    "radial",
    "resize",
    "rotate",
    "scale",
    "skew",
    "tangential",
    "translate",
    "warp",
}


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


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise SystemExit(f"missing JSONL file: {path}")
    rows = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number}: invalid JSONL row") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{line_number}: JSONL row must be an object")
        rows.append(row)
    return rows


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


def assert_no_geometric_postprocess(metadata: object, metadata_path: Path) -> None:
    if not isinstance(metadata, dict):
        raise SystemExit(f"{metadata_path}: source metadata must be an object")
    scene_config = metadata.get("sceneConfig", {})
    if not isinstance(scene_config, dict):
        return
    label_transform_policy = scene_config.get("labelTransformPolicy", {})
    shared_label_transform = (
        isinstance(label_transform_policy, dict)
        and bool(label_transform_policy.get("sharedGeometricPostprocess"))
        and label_transform_policy.get("labelsDerivedFrom") == "postprocessed_id_pixels"
    )
    camera = scene_config.get("camera", {})
    if isinstance(camera, dict):
        lens_distortion = camera.get("lensDistortion", {})
        if isinstance(lens_distortion, dict) and lens_distortion.get("applied"):
            if not shared_label_transform:
                raise SystemExit(
                    f"{metadata_path}: camera lensDistortion is applied but labelTransformPolicy does not prove a shared RGB/ID/label transform"
                )
    postprocess = scene_config.get("postprocess", {})
    if not isinstance(postprocess, dict):
        return
    geometric_keys = sorted(
        key
        for key in postprocess
        if any(token in str(key).lower() for token in GEOMETRIC_POSTPROCESS_TOKENS)
    )
    if geometric_keys:
        if shared_label_transform:
            return
        raise SystemExit(
            f"{metadata_path}: geometric postprocess keys {geometric_keys} require an exact shared RGB/ID/label transform path"
        )


def counter_payload(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def count_by_class(rows: list[dict[str, object]]) -> Counter[str]:
    return Counter(str(row.get("className", "unknown")) for row in rows)


def parent_fused_counts(fragment_rows: list[dict[str, object]]) -> Counter[str]:
    parent_keys = {
        (int(row["parentVisibleIndex"]), str(row.get("className", "unknown")))
        for row in fragment_rows
    }
    return Counter(class_name for _parent_index, class_name in parent_keys)


def split_parent_count(fragment_rows: list[dict[str, object]]) -> int:
    parent_fragment_counts: Counter[tuple[int, str]] = Counter(
        (int(row["parentVisibleIndex"]), str(row.get("className", "unknown")))
        for row in fragment_rows
    )
    return sum(1 for count in parent_fragment_counts.values() if count > 1)


def count_block(counter: Counter[str]) -> dict[str, object]:
    return {"total": int(sum(counter.values())), "by_class": counter_payload(counter)}


def build_count_target(
    *,
    variant: int,
    image: str,
    visible_boxes: list[dict[str, object]],
    fragment_metadata: list[dict[str, object]],
    ignored_fragment_metadata: list[dict[str, object]],
) -> dict[str, object]:
    physical_counts = count_by_class(visible_boxes)
    kept_fragment_counts = count_by_class(fragment_metadata)
    ignored_fragment_counts = count_by_class(ignored_fragment_metadata)
    all_fragment_metadata = [*fragment_metadata, *ignored_fragment_metadata]
    parent_fused_kept_counts = parent_fused_counts(fragment_metadata)
    parent_fused_all_counts = parent_fused_counts(all_fragment_metadata)
    physical_total = int(sum(physical_counts.values()))
    kept_total = len(fragment_metadata)
    all_total = len(all_fragment_metadata)
    return {
        "variant": variant,
        "image": image,
        "physical_visible_instances": count_block(physical_counts),
        "kept_fragments": count_block(kept_fragment_counts),
        "ignored_fragments": count_block(ignored_fragment_counts),
        "parent_fused_kept_fragments": count_block(parent_fused_kept_counts),
        "parent_fused_all_fragments": count_block(parent_fused_all_counts),
        "naive_kept_fragment_overcount": int(kept_total - physical_total),
        "naive_all_fragment_overcount": int(all_total - physical_total),
        "kept_split_parent_count": split_parent_count(fragment_metadata),
        "all_split_parent_count": split_parent_count(all_fragment_metadata),
        "policy": {
            "count_truth": "physical_visible_instances",
            "fragment_counts": "visible evidence components; do not use directly as bill totals",
            "parent_fused_all_fragments": "synthetic oracle fusion target that should match physical_visible_instances",
        },
    }


def merge_count_blocks(rows: list[dict[str, object]], key: str) -> Counter[str]:
    merged: Counter[str] = Counter()
    for row in rows:
        block = row.get(key, {})
        by_class = block.get("by_class", {}) if isinstance(block, dict) else {}
        if isinstance(by_class, dict):
            merged.update({str(name): int(count) for name, count in by_class.items()})
    return merged


def summarize_count_targets(rows: list[dict[str, object]]) -> dict[str, object]:
    physical_counts = merge_count_blocks(rows, "physical_visible_instances")
    kept_fragment_counts = merge_count_blocks(rows, "kept_fragments")
    ignored_fragment_counts = merge_count_blocks(rows, "ignored_fragments")
    parent_fused_kept_counts = merge_count_blocks(rows, "parent_fused_kept_fragments")
    parent_fused_all_counts = merge_count_blocks(rows, "parent_fused_all_fragments")
    return {
        "images": len(rows),
        "physical_visible_instances": count_block(physical_counts),
        "kept_fragments": count_block(kept_fragment_counts),
        "ignored_fragments": count_block(ignored_fragment_counts),
        "parent_fused_kept_fragments": count_block(parent_fused_kept_counts),
        "parent_fused_all_fragments": count_block(parent_fused_all_counts),
        "parent_fused_all_matches_physical": parent_fused_all_counts == physical_counts,
        "naive_kept_fragment_overcount": int(sum(int(row["naive_kept_fragment_overcount"]) for row in rows)),
        "naive_all_fragment_overcount": int(sum(int(row["naive_all_fragment_overcount"]) for row in rows)),
        "kept_split_parent_count": int(sum(int(row["kept_split_parent_count"]) for row in rows)),
        "all_split_parent_count": int(sum(int(row["all_split_parent_count"]) for row in rows)),
        "policy": {
            "count_truth": "physical_visible_instances",
            "fragment_counts": "visible evidence components; do not use directly as bill totals",
            "parent_fused_all_fragments": "synthetic oracle fusion target that should match physical_visible_instances",
        },
    }


def main() -> int:
    args = parse_args()
    dataset_root = resolve_path(args.root)
    manifest = read_json(dataset_root / "manifest.json")
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("manifest.json must be a non-empty list")
    manifest_by_variant = {}
    count_target_rows = read_jsonl(dataset_root / "counts" / "targets.jsonl")
    if len(count_target_rows) != len(manifest):
        raise SystemExit("count target row count mismatch")
    count_targets_by_variant = {int(row["variant"]): row for row in count_target_rows}
    expected_count_targets: list[dict[str, object]] = []

    trainable_obb_images = 0
    rejected_obb_images = 0
    fragment_count = 0
    ignored_fragment_count = 0
    fragment_evidence_status_counts: Counter[str] = Counter()
    fragment_evidence_warning_counts: Counter[str] = Counter()
    visible_instance_count = 0
    class_counts: Counter[str] = Counter()
    layer_audit_totals: Counter[str] = Counter()
    for row in manifest:
        if not isinstance(row, dict):
            raise SystemExit("manifest row must be an object")
        variant = int(row["variant"])
        manifest_by_variant[variant] = row
        boxes_doc = read_json(dataset_root / row["visible_boxes"])
        source_metadata_path = dataset_root / row["source_metadata"]
        assert_no_geometric_postprocess(read_json(source_metadata_path), source_metadata_path)
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
            evidence_status = str(fragment.get("evidence_status", ""))
            if evidence_status not in {"trainable", "review_required"}:
                raise SystemExit(f"{row['fragment_metadata']}: invalid evidence_status {evidence_status!r}")
            evidence_warnings = fragment.get("evidence_warnings", [])
            if not isinstance(evidence_warnings, list):
                raise SystemExit(f"{row['fragment_metadata']}: evidence_warnings must be a list")
            if evidence_status == "review_required" and not evidence_warnings:
                raise SystemExit(f"{row['fragment_metadata']}: review_required fragment missing evidence_warnings")
            if evidence_status == "trainable" and evidence_warnings:
                raise SystemExit(f"{row['fragment_metadata']}: trainable fragment has evidence_warnings")
            fragment_evidence_status_counts[evidence_status] += 1
            fragment_evidence_warning_counts.update(str(reason) for reason in evidence_warnings)
        fragment_count += len(fragment_rows)
        ignored_fragment_metadata = read_json(dataset_root / row["fragment_ignored_metadata"])
        if not isinstance(ignored_fragment_metadata, list):
            raise SystemExit(f"{row['fragment_ignored_metadata']}: ignored fragment metadata must be a list")
        for fragment in ignored_fragment_metadata:
            parent_index = fragment.get("parentVisibleIndex")
            if not isinstance(parent_index, int) or parent_index < 0 or parent_index >= len(visible_boxes):
                raise SystemExit(f"{row['fragment_ignored_metadata']}: invalid parentVisibleIndex {parent_index}")
            ignore_reason = str(fragment.get("ignore_reason", "")).strip()
            if not ignore_reason:
                raise SystemExit(f"{row['fragment_ignored_metadata']}: ignored fragment missing ignore_reason")
            if ignore_reason not in VALID_IGNORED_FRAGMENT_REASONS:
                raise SystemExit(f"{row['fragment_ignored_metadata']}: invalid ignore_reason {ignore_reason!r}")
            evidence_warnings = fragment.get("evidence_warnings", [])
            if not isinstance(evidence_warnings, list):
                raise SystemExit(f"{row['fragment_ignored_metadata']}: evidence_warnings must be a list")
            if ignore_reason in REVIEW_IGNORED_FRAGMENT_REASONS and not evidence_warnings:
                raise SystemExit(f"{row['fragment_ignored_metadata']}: review-ignored fragment missing evidence_warnings")
            if fragment.get("evidence_status") != "ignored":
                raise SystemExit(f"{row['fragment_ignored_metadata']}: ignored fragment evidence_status must be ignored")
        ignored_fragment_count += len(ignored_fragment_metadata)
        expected_count_target = build_count_target(
            variant=variant,
            image=str(row["image"]),
            visible_boxes=visible_boxes,
            fragment_metadata=fragment_metadata,
            ignored_fragment_metadata=ignored_fragment_metadata,
        )
        expected_count_targets.append(expected_count_target)
        if count_targets_by_variant.get(variant) != expected_count_target:
            raise SystemExit(f"counts/targets.jsonl: count target mismatch for variant {variant}")

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
    count_summary = read_json(dataset_root / "counts" / "summary.json")
    expected_count_summary = summarize_count_targets(expected_count_targets)
    if set(count_targets_by_variant) != set(manifest_by_variant):
        raise SystemExit("count target variants do not match manifest")
    if count_summary != expected_count_summary:
        raise SystemExit("count summary mismatch")
    if not expected_count_summary["parent_fused_all_matches_physical"]:
        raise SystemExit("parent-fused all-fragment counts do not match physical counts")
    if obb_summary.get("trainable_obb_images") != trainable_obb_images:
        raise SystemExit("obb summary trainable image count mismatch")
    if obb_summary.get("rejected_obb_images") != rejected_obb_images:
        raise SystemExit("obb summary rejected image count mismatch")
    if fragment_summary.get("fragments") != fragment_count:
        raise SystemExit("fragment summary count mismatch")
    if fragment_summary.get("ignored_fragments") != ignored_fragment_count:
        raise SystemExit("fragment summary ignored count mismatch")
    if fragment_summary.get("evidence_status_counts") != dict(sorted(fragment_evidence_status_counts.items())):
        raise SystemExit("fragment summary evidence status count mismatch")
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
    if qa_summary.get("fragments", {}).get("evidence_status_counts") != dict(sorted(fragment_evidence_status_counts.items())):
        raise SystemExit("qa summary fragment evidence status count mismatch")
    if qa_summary.get("fragments", {}).get("evidence_warning_counts") != dict(sorted(fragment_evidence_warning_counts.items())):
        raise SystemExit("qa summary fragment evidence warning count mismatch")
    if qa_summary.get("count_targets") != expected_count_summary:
        raise SystemExit("qa summary count target mismatch")
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
    if outputs.get("count_targets"):
        expected = str((dataset_root / "counts" / "targets.jsonl").relative_to(ROOT))
        if outputs["count_targets"] != expected:
            raise SystemExit("recipe count_targets output path mismatch")
    if outputs.get("count_summary"):
        expected = str((dataset_root / "counts" / "summary.json").relative_to(ROOT))
        if outputs["count_summary"] != expected:
            raise SystemExit("recipe count_summary output path mismatch")

    print(
        f"ok: {len(manifest)} images, {fragment_count} fragments, "
        f"{trainable_obb_images} trainable OBB images, {rejected_obb_images} rejected OBB images, "
        f"{expected_count_summary['physical_visible_instances']['total']} physical count targets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
