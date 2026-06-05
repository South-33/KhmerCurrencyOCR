#!/usr/bin/env python
"""Audit mission-level readiness of the CashSnap synthetic pipeline.

This sits above the renderer/package gates. It answers a different question:
for each required real-world condition, do we have synthetic recipe coverage,
an active generated trainable-candidate package, and enough real validation or
capture inventory to trust a scale decision?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "configs" / "synthetic_targets" / "cashsnap_real_target_matrix_v1.json"
DEFAULT_CATALOG = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_recipe_catalog_v1.json"
DEFAULT_SUITE = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_trainable_candidates_v1.json"
DEFAULT_SOURCES = ROOT / "manifests" / "real_fan_benchmark_sources.csv"
DEFAULT_QUALITY = ROOT / "manifests" / "real_fan_benchmark_label_quality.csv"
DEFAULT_CAPTURE_INVENTORY = ROOT / "manifests" / "real_partial_capture_inventory.csv"
DEFAULT_CAPTURE_REQUIREMENTS = ROOT / "manifests" / "real_partial_capture_requirements.csv"
DEFAULT_REAL_DATASET_CANDIDATES = ROOT / "runs" / "cashsnap" / "real_dataset_stress_candidates_latest.json"

RECIPE_STATUS_RANK = {
    "rejected_probe": -1,
    "planned": 0,
    "smoke_ready": 1,
    "label_policy_ready": 2,
    "diagnostic": 3,
    "trainable-candidate": 4,
    "promoted": 5,
}
ACTIVE_RECIPE_STATUSES = {"smoke_ready", "label_policy_ready", "diagnostic", "trainable-candidate", "promoted"}
TRAINABLE_PACKAGE_STATUSES = {"trainable-candidate", "promoted"}

CONDITION_REAL_ROLES = {
    "loose_counter_stack_overlap": {"visible_denoms_mild_overlap"},
    "dense_shop_overlap": {"dense_overlap_stress"},
    "handheld_fan": {"fan_stress"},
    "finger_or_hand_split_occlusion": {"hand_occlusion_stress"},
    "repeated_same_denomination": {"fan_stress"},
}

CONDITION_CAPTURE_REQUIREMENTS = {
    "loose_counter_stack_overlap": {"simple_overlap", "khr_5000_face_number_overlap"},
    "dense_shop_overlap": {"simple_overlap", "khr_5000_face_number_overlap"},
    "handheld_fan": {"hand_fan"},
    "finger_or_hand_split_occlusion": {"hand_occlusion"},
    "thin_edge_partial_fragments": {"thin_slice_khr_5000", "thin_slice_khr_20000"},
    "front_back_and_old_common_confusion": {"weak_khr_20000", "weak_khr_50000"},
    "mixed_rare_common_cross_currency_stack": {"mixed_usd_khr_rare_common_stack"},
    "repeated_same_denomination": {"same_denomination_fan"},
    "hard_negatives_and_non_banknote_paper": {"no_note_background", "non_banknote_paper_props"},
}

CONDITION_REAL_DATASET_CANDIDATES = {
    "loose_counter_stack_overlap": {"simple_overlap", "khr_5000_face_number_overlap"},
    "dense_shop_overlap": {"simple_overlap", "khr_5000_face_number_overlap"},
    "handheld_fan": {"hand_fan"},
    "finger_or_hand_split_occlusion": set(),
    "thin_edge_partial_fragments": {"thin_slice_khr_5000", "thin_slice_khr_20000", "partial_off_frame"},
    "front_back_and_old_common_confusion": {"weak_khr_20000", "weak_khr_50000"},
    "mixed_rare_common_cross_currency_stack": {"mixed_usd_khr_rare_common"},
    "repeated_same_denomination": {"same_denomination_fan"},
    "hard_negatives_and_non_banknote_paper": set(),
}

BLOCKING_TARGET_STATUSES = {
    "not_solved",
    "renderer_smoke_only",
    "known_failure_mode",
    "metadata_support_needed_for_fusion",
    "background_qa_needed",
    "real_anchor_missing",
}

USABLE_RIGHTS = {"own_photo", "rights_clear", "public_domain", "cc0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--capture-inventory", type=Path, default=DEFAULT_CAPTURE_INVENTORY)
    parser.add_argument("--capture-requirements", type=Path, default=DEFAULT_CAPTURE_REQUIREMENTS)
    parser.add_argument(
        "--real-dataset-candidates",
        type=Path,
        default=DEFAULT_REAL_DATASET_CANDIDATES,
        help="Optional stress-candidate summary from mine_real_dataset_stress_candidates.py.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="Also inspect rendered suite roots and flag package metadata/count mismatches.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any required condition is blocked.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    try:
        return resolve(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing JSON file: {repo_path(resolved)}")
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_path(resolved)}: expected JSON object")
    return data


def read_optional_json(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    if not resolved.exists():
        return {}
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_path(resolved)}: expected JSON object")
    return data


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = resolve(path)
    row: dict[str, Any] = {"path": repo_path(resolved), "exists": resolved.exists()}
    if resolved.exists() and resolved.is_file():
        row["sha256"] = file_sha256(resolved)
        row["size_bytes"] = resolved.stat().st_size
    return row


def read_csv(path: Path) -> list[dict[str, str]]:
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing CSV file: {repo_path(resolved)}")
    with resolved.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "score", "keep"}


def split_values(value: str) -> set[str]:
    return {part.strip() for part in value.replace(";", ",").split(",") if part.strip()}


def row_matches(row: dict[str, str], column: str, value: str) -> bool:
    if column == "denominations":
        return value in split_values(row.get(column, ""))
    return row.get(column, "").strip() == value


def recipe_rank(status: str) -> int:
    return RECIPE_STATUS_RANK.get(status, -2)


def best_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "missing"
    return max((str(row.get("artifact_status", "")) for row in rows), key=recipe_rank)


def promoted_real_roles(source_rows: list[dict[str, str]]) -> Counter[str]:
    roles: Counter[str] = Counter()
    for row in source_rows:
        if row.get("benchmark_status", "").strip() != "labeled" and row.get("label_status", "").strip() != "labeled":
            continue
        role = row.get("benchmark_role", "").strip()
        if role:
            roles[role] += 1
    return roles


def scoreable_real_images(quality_rows: list[dict[str, str]]) -> set[str]:
    image_ids: set[str] = set()
    for row in quality_rows:
        if row.get("quality", "").strip() not in {"clear", "partial_clear"}:
            continue
        if truthy(row.get("count_for_score", "")):
            image_ids.add(row.get("image_id", "").strip())
    return {image_id for image_id in image_ids if image_id}


def scoreable_real_images_by_role(source_rows: list[dict[str, str]], scoreable_images: set[str]) -> dict[str, list[str]]:
    role_images: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        image_id = row.get("image_id", "").strip()
        if image_id not in scoreable_images:
            continue
        role = row.get("benchmark_role", "").strip()
        if role:
            role_images[role].add(image_id)
    return {role: sorted(image_ids) for role, image_ids in sorted(role_images.items())}


def usable_capture_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    usable: list[dict[str, str]] = []
    for row in rows:
        if row.get("rights_status", "").strip().lower() not in USABLE_RIGHTS:
            continue
        local_path = row.get("local_path", "").strip()
        if not local_path or not resolve(Path(local_path)).exists():
            continue
        usable.append(row)
    return usable


def capture_requirement_reports(
    condition_id: str,
    requirement_rows: list[dict[str, str]],
    usable_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    wanted = CONDITION_CAPTURE_REQUIREMENTS.get(condition_id, set())
    reports: list[dict[str, Any]] = []
    for req in requirement_rows:
        requirement_id = req.get("requirement_id", "").strip()
        if requirement_id not in wanted:
            continue
        minimum = int(req.get("min_images", "0") or "0")
        count = sum(1 for row in usable_rows if row_matches(row, req["match_column"], req["match_value"]))
        required = req.get("required", "yes").strip().lower() not in {"0", "false", "no", "optional"}
        reports.append(
            {
                "requirement_id": requirement_id,
                "description": req.get("description", ""),
                "minimum": minimum,
                "count": count,
                "missing": max(0, minimum - count),
                "required": required,
                "status": "ok" if count >= minimum else ("missing" if required else "optional_missing"),
            }
        )
    return reports


def real_dataset_candidate_reports(condition_id: str, candidate_summary: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = CONDITION_REAL_DATASET_CANDIDATES.get(condition_id, set())
    counts = candidate_summary.get("scene_candidate_counts", {})
    unique_counts = candidate_summary.get("scene_unique_origin_counts", {})
    if not isinstance(counts, dict) or not isinstance(unique_counts, dict):
        return []
    reports: list[dict[str, Any]] = []
    for scene_type in sorted(wanted):
        count = int(counts.get(scene_type, 0) or 0)
        unique_origins = int(unique_counts.get(scene_type, 0) or 0)
        reports.append(
            {
                "scene_type": scene_type,
                "candidate_count": count,
                "unique_origin_count": unique_origins,
                "status": "candidate_needs_visual_review"
                if count > 0 or unique_origins > 0
                else "missing_candidate_hint",
            }
        )
    return reports


def package_report(row: dict[str, Any]) -> dict[str, Any]:
    root = resolve(Path(str(row.get("out_root", ""))))
    report: dict[str, Any] = {
        "recipe_id": row.get("recipe_id", ""),
        "root": repo_path(root),
        "declared_images": int(row.get("count", 0) or 0),
        "exists": root.exists(),
        "status": "missing",
        "blockers": [],
    }
    if not root.exists():
        report["blockers"].append("rendered package root is missing")
        return report

    recipe_path = root / "recipe.json"
    summary_path = root / "qa" / "summary.json"
    for required_path in (recipe_path, summary_path, root / "data.yaml", root / "manifest.json"):
        if not required_path.exists():
            report["blockers"].append(f"missing {repo_path(required_path)}")
    if report["blockers"]:
        return report

    recipe = read_json(recipe_path)
    summary = read_json(summary_path)
    rendered_status = str(recipe.get("artifact_status", ""))
    rendered_images = int(summary.get("images", 0) or 0)
    visible_instances = summary.get("visible_instances", {})
    if not isinstance(visible_instances, dict):
        visible_instances = {}
    report.update(
        {
            "status": rendered_status,
            "rendered_images": rendered_images,
            "visible_instances": int(visible_instances.get("total", 0) or 0),
            "scene_modes": summary.get("scene_modes", {}),
            "train_views": row.get("train_views", []),
            "fingerprints": {
                "recipe_json": file_fingerprint(recipe_path),
                "qa_summary": file_fingerprint(summary_path),
                "data_yaml": file_fingerprint(root / "data.yaml"),
                "manifest": file_fingerprint(root / "manifest.json"),
            },
        }
    )
    if rendered_status not in TRAINABLE_PACKAGE_STATUSES:
        report["blockers"].append(f"rendered artifact_status {rendered_status!r} is not trainable")
    if rendered_images != report["declared_images"]:
        report["blockers"].append(f"rendered image count {rendered_images} != suite count {report['declared_images']}")
    if str(recipe.get("recipe_name", "")) != str(row.get("recipe_id", "")):
        report["blockers"].append("rendered recipe_name does not match suite recipe_id")
    return report


def suite_recipe_rows(suite: dict[str, Any]) -> list[dict[str, Any]]:
    rows = suite.get("recipes", [])
    if not isinstance(rows, list):
        raise SystemExit("suite recipes must be a list")
    return [row for row in rows if isinstance(row, dict)]


def condition_state(
    target: dict[str, Any],
    recipes: list[dict[str, Any]],
    suite_rows: list[dict[str, Any]],
    package_reports: dict[str, dict[str, Any]],
    promoted_roles: Counter[str],
    scoreable_images: set[str],
    scoreable_images_by_role: dict[str, list[str]],
    requirement_rows: list[dict[str, str]],
    usable_captures: list[dict[str, str]],
    real_dataset_candidates: dict[str, Any],
    check_existing: bool,
) -> dict[str, Any]:
    condition_id = str(target["id"])
    active_recipes = [row for row in recipes if str(row.get("artifact_status", "")) in ACTIVE_RECIPE_STATUSES]
    suite_image_count = sum(int(row.get("count", 0) or 0) for row in suite_rows)
    blockers: list[str] = []
    notes: list[str] = []

    if not recipes:
        blockers.append("no recipe in synthetic catalog")
    elif not active_recipes:
        blockers.append("only rejected/planned recipes cover this condition")
    if not suite_rows:
        blockers.append("no active trainable-candidate package covers this condition")

    target_status = str(target.get("status", "")).strip()
    if target_status in BLOCKING_TARGET_STATUSES:
        blockers.append(f"target matrix status is still {target_status!r}")

    package_blockers: list[str] = []
    if check_existing:
        for row in suite_rows:
            report = package_reports.get(str(row.get("recipe_id", "")), {})
            package_blockers.extend(str(blocker) for blocker in report.get("blockers", []))
        blockers.extend(package_blockers)

    required_roles = CONDITION_REAL_ROLES.get(condition_id, set())
    missing_roles = sorted(role for role in required_roles if promoted_roles.get(role, 0) <= 0)
    if missing_roles:
        blockers.append(f"no promoted real benchmark labels for roles {missing_roles}")

    capture_reports = capture_requirement_reports(condition_id, requirement_rows, usable_captures)
    missing_capture = [
        f"{row['requirement_id']} {row['count']}/{row['minimum']}"
        for row in capture_reports
        if row["required"] and row["count"] < row["minimum"]
    ]
    if missing_capture:
        blockers.append(f"capture inventory gaps: {', '.join(missing_capture)}")

    candidate_reports = real_dataset_candidate_reports(condition_id, real_dataset_candidates)
    positive_candidate_reports = [
        row for row in candidate_reports if int(row["candidate_count"] or 0) > 0 or int(row["unique_origin_count"] or 0) > 0
    ]
    missing_candidate_reports = [
        row for row in candidate_reports if int(row["candidate_count"] or 0) <= 0 and int(row["unique_origin_count"] or 0) <= 0
    ]
    if positive_candidate_reports:
        notes.append(
            "real dataset review candidates: "
            + ", ".join(
                f"{row['scene_type']} {row['candidate_count']} candidates/{row['unique_origin_count']} origins"
                for row in positive_candidate_reports
            )
        )
    if missing_candidate_reports:
        notes.append(
            "missing real dataset candidate hints: "
            + ", ".join(str(row["scene_type"]) for row in missing_candidate_reports)
        )

    scoreable_role_images = sorted(
        {
            image_id
            for role in required_roles
            for image_id in scoreable_images_by_role.get(role, [])
            if image_id in scoreable_images
        }
    )
    if scoreable_role_images:
        notes.append(
            f"{len(scoreable_role_images)} scoreable real image(s) for required roles: "
            + ", ".join(scoreable_role_images[:3])
        )

    if any(str(row.get("artifact_status", "")) == "promoted" for row in recipes):
        state = "promoted"
    elif suite_rows and not blockers:
        state = "trainable_candidate_ready_for_bounded_probe"
    elif suite_rows:
        state = "trainable_candidate_blocked"
    elif best_status(active_recipes) == "diagnostic":
        state = "diagnostic_only"
    elif active_recipes:
        state = "renderer_or_label_policy_only"
    else:
        state = "missing"

    return {
        "condition_id": condition_id,
        "priority": target.get("priority", ""),
        "required_for_v1": bool(target.get("required_for_v1")),
        "target_status": target.get("status", ""),
        "success_gate": target.get("success_gate", ""),
        "catalog_recipe_ids": [str(row.get("id", "")) for row in recipes],
        "catalog_status_counts": dict(Counter(str(row.get("artifact_status", "")) for row in recipes)),
        "best_catalog_status": best_status(recipes),
        "active_suite_recipe_ids": [str(row.get("recipe_id", "")) for row in suite_rows],
        "suite_image_count": suite_image_count,
        "required_real_roles": sorted(required_roles),
        "missing_real_roles": missing_roles,
        "capture_requirements": capture_reports,
        "real_dataset_review_candidates": candidate_reports,
        "state": state,
        "blockers": blockers,
        "notes": notes,
    }


def main() -> int:
    args = parse_args()
    targets = read_json(args.targets)
    catalog = read_json(args.catalog)
    suite = read_json(args.suite)
    source_rows = read_csv(args.sources)
    quality_rows = read_csv(args.quality)
    capture_inventory = read_csv(args.capture_inventory)
    capture_requirements = read_csv(args.capture_requirements)
    real_dataset_candidates = read_optional_json(args.real_dataset_candidates)

    target_rows = targets.get("conditions", [])
    catalog_rows = catalog.get("recipes", [])
    if not isinstance(target_rows, list) or not target_rows:
        raise SystemExit("target matrix must contain conditions")
    if not isinstance(catalog_rows, list) or not catalog_rows:
        raise SystemExit("recipe catalog must contain recipes")

    recipes_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_ids = {str(row.get("id", "")) for row in target_rows if isinstance(row, dict)}
    for recipe in catalog_rows:
        if not isinstance(recipe, dict):
            continue
        for condition_id in recipe.get("target_condition_ids", []):
            condition_text = str(condition_id)
            if condition_text in target_ids:
                recipes_by_condition[condition_text].append(recipe)

    suite_rows = suite_recipe_rows(suite)
    catalog_by_recipe_id = {str(row.get("id", "")): row for row in catalog_rows if isinstance(row, dict)}
    suite_rows_by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in suite_rows:
        recipe_id = str(row.get("recipe_id", ""))
        recipe = catalog_by_recipe_id.get(recipe_id, {})
        for condition_id in recipe.get("target_condition_ids", []):
            if str(condition_id) in target_ids:
                suite_rows_by_condition[str(condition_id)].append(row)

    package_reports = {}
    if args.check_existing:
        package_reports = {str(row.get("recipe_id", "")): package_report(row) for row in suite_rows}

    promoted_roles = promoted_real_roles(source_rows)
    scoreable_images = scoreable_real_images(quality_rows)
    scoreable_role_images = scoreable_real_images_by_role(source_rows, scoreable_images)
    usable_captures = usable_capture_rows(capture_inventory)

    condition_reports = [
        condition_state(
            target=row,
            recipes=recipes_by_condition.get(str(row["id"]), []),
            suite_rows=suite_rows_by_condition.get(str(row["id"]), []),
            package_reports=package_reports,
            promoted_roles=promoted_roles,
            scoreable_images=scoreable_images,
            scoreable_images_by_role=scoreable_role_images,
            requirement_rows=capture_requirements,
            usable_captures=usable_captures,
            real_dataset_candidates=real_dataset_candidates,
            check_existing=args.check_existing,
        )
        for row in target_rows
        if isinstance(row, dict) and row.get("priority") != "out_of_scope"
    ]

    required_reports = [row for row in condition_reports if row["required_for_v1"]]
    blocked_required = [row for row in required_reports if row["blockers"]]
    trainable_required = [row for row in required_reports if row["active_suite_recipe_ids"]]
    real_role_required = [row for row in required_reports if row["required_real_roles"]]
    real_role_ready = [row for row in real_role_required if not row["missing_real_roles"]]
    candidate_mapped_required = [row for row in required_reports if row["real_dataset_review_candidates"]]
    candidate_hint_ready = [
        row
        for row in candidate_mapped_required
        if any(
            int(candidate.get("candidate_count", 0) or 0) > 0
            or int(candidate.get("unique_origin_count", 0) or 0) > 0
            for candidate in row["real_dataset_review_candidates"]
        )
    ]
    candidate_hint_gaps = [
        {
            "condition_id": row["condition_id"],
            "scene_type": candidate["scene_type"],
            "candidate_count": candidate["candidate_count"],
            "unique_origin_count": candidate["unique_origin_count"],
        }
        for row in candidate_mapped_required
        for candidate in row["real_dataset_review_candidates"]
        if int(candidate.get("candidate_count", 0) or 0) <= 0
        and int(candidate.get("unique_origin_count", 0) or 0) <= 0
    ]

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "targets": repo_path(args.targets),
        "catalog": repo_path(args.catalog),
        "suite": repo_path(args.suite),
        "input_fingerprints": {
            "targets": file_fingerprint(args.targets),
            "catalog": file_fingerprint(args.catalog),
            "suite": file_fingerprint(args.suite),
            "sources": file_fingerprint(args.sources),
            "quality": file_fingerprint(args.quality),
            "capture_inventory": file_fingerprint(args.capture_inventory),
            "capture_requirements": file_fingerprint(args.capture_requirements),
            "real_dataset_candidates": file_fingerprint(args.real_dataset_candidates),
        },
        "check_existing": args.check_existing,
        "required_conditions": len(required_reports),
        "required_with_trainable_candidate": len(trainable_required),
        "required_with_real_role_labels": len(real_role_ready),
        "required_real_role_conditions": len(real_role_required),
        "required_real_dataset_candidate_rule_conditions": len(candidate_mapped_required),
        "required_with_real_dataset_candidate_hints": len(candidate_hint_ready),
        "real_dataset_candidate_hint_gaps": candidate_hint_gaps,
        "scoreable_real_images": sorted(scoreable_images),
        "scoreable_real_images_by_role": scoreable_role_images,
        "promoted_real_role_counts": dict(sorted(promoted_roles.items())),
        "usable_capture_images": len(usable_captures),
        "real_dataset_candidates": {
            "path": repo_path(args.real_dataset_candidates),
            "loaded": bool(real_dataset_candidates),
            "generated_at_utc": real_dataset_candidates.get("generated_at_utc", ""),
            "data": real_dataset_candidates.get("data", ""),
            "data_config_sha256": real_dataset_candidates.get("data_config_sha256", ""),
            "splits": real_dataset_candidates.get("splits", []),
            "split_fingerprints": real_dataset_candidates.get("split_fingerprints", {}),
            "scene_candidate_counts": real_dataset_candidates.get("scene_candidate_counts", {}),
            "scene_unique_origin_counts": real_dataset_candidates.get("scene_unique_origin_counts", {}),
        },
        "suite_package_reports": package_reports,
        "ready_for_synthetic_scale": not blocked_required,
        "blocked_required_conditions": [row["condition_id"] for row in blocked_required],
        "conditions": condition_reports,
    }

    if args.json_out is not None:
        out = resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        "synthetic_pipeline_ready="
        f"{str(report['ready_for_synthetic_scale']).lower()} "
        f"required={len(required_reports)} "
        f"trainable={len(trainable_required)} "
        f"real_role_ready={len(real_role_ready)}/{len(real_role_required)} "
        f"candidate_hints={len(candidate_hint_ready)}/{len(candidate_mapped_required)} "
        f"usable_captures={len(usable_captures)}"
    )
    for row in required_reports:
        blocker_suffix = f" blockers={len(row['blockers'])}" if row["blockers"] else ""
        print(
            f"{row['state']}: {row['condition_id']} "
            f"priority={row['priority']} "
            f"best={row['best_catalog_status']} "
            f"suite_images={row['suite_image_count']}"
            f"{blocker_suffix}"
        )
        for blocker in row["blockers"][:4]:
            print(f"  - {blocker}")
        if len(row["blockers"]) > 4:
            print(f"  - ... {len(row['blockers']) - 4} more")
        for note in row["notes"][:2]:
            print(f"  note: {note}")

    if args.json_out is not None:
        print(f"wrote_json={repo_path(args.json_out)}")
    if args.strict and blocked_required:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
