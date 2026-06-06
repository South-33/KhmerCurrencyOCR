#!/usr/bin/env python
"""Build accepted-blend configs with staged rare-class WebGL support doses."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "configs" / "cashsnap_v1_plus_webgl_accepted_nowarmup_probe.yaml"
DEFAULT_DOSE_ROOT = ROOT / "data" / "synthetic" / "cashsnap_webgl_rare_class_support_audit_v1"
DEFAULT_OUT_DIR = ROOT / "configs" / "webgl_staged_dose"
DEFAULT_LIST_DIR = ROOT / "configs" / "generated_lists" / "webgl_staged_dose"
DEFAULT_DOMAIN_GAP_DIR = ROOT / "runs" / "cashsnap"
DEFAULT_CLASSES = "KHR_2000,KHR_50000,KHR_20000,KHR_10000,KHR_5000"
DEFAULT_STEM_PREFIX = "cashsnap_v1_plus_webgl_accepted_rare_support"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--dose-root", type=Path, default=DEFAULT_DOSE_ROOT)
    parser.add_argument("--recipe-id", default="webgl_rare_class_support_v1")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--list-dir", type=Path, default=DEFAULT_LIST_DIR)
    parser.add_argument("--domain-gap-dir", type=Path, default=DEFAULT_DOMAIN_GAP_DIR)
    parser.add_argument(
        "--stem-prefix",
        default=DEFAULT_STEM_PREFIX,
        help="Output stem prefix; dose number is appended as '_dose<N>'.",
    )
    parser.add_argument("--doses", default="1,2,4,8,16", help="Comma/space-separated dose image counts.")
    parser.add_argument("--dose-classes", default=DEFAULT_CLASSES)
    parser.add_argument("--max-combinations", type=int, default=500_000)
    parser.add_argument("--domain-gap-preset", default="accepted_blend_v1")
    parser.add_argument("--skip-domain-gap-gate", action="store_true")
    parser.add_argument("--fail-on-domain-gap", action="store_true")
    parser.add_argument("--skip-yolo-check", action="store_true")
    parser.add_argument(
        "--write-row-count-controls",
        action="store_true",
        help="Also write matched configs that append duplicate base rows instead of synthetic dose rows.",
    )
    parser.add_argument(
        "--control-stem-prefix",
        default=None,
        help="Output stem prefix for row-count controls; defaults to '<stem-prefix>_row_control'.",
    )
    parser.add_argument(
        "--control-source",
        choices=("background", "first", "class", "class_mix", "real_class_mix"),
        default="background",
        help="Base rows to duplicate for matched row-count controls.",
    )
    parser.add_argument(
        "--control-class",
        default=None,
        help="Class name/id to duplicate when --control-source class.",
    )
    parser.add_argument(
        "--fail-on-inexact-real-class-mix-control",
        action="store_true",
        help="Fail if --control-source real_class_mix cannot exactly match every selected synthetic row.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def rel_between(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), base.resolve()).replace("\\", "/")


def read_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"expected YAML mapping: {repo_rel(path)}")
    return document


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"expected JSON mapping: {repo_rel(path)}")
    return document


def data_root(config_path: Path, config: dict[str, Any]) -> Path:
    root = Path(str(config.get("path", "."))).expanduser()
    return root if root.is_absolute() else (config_path.parent / root).resolve()


def split_root(dataset_root: Path, split_path: str) -> Path:
    path = Path(split_path)
    return path if path.is_absolute() else dataset_root / path


def train_list_path(config_path: Path, config: dict[str, Any]) -> Path:
    train_value = config.get("train")
    if not isinstance(train_value, str):
        raise SystemExit(f"{repo_rel(config_path)} train split must be a list path string")
    path = split_root(data_root(config_path, config), train_value)
    if path.suffix.lower() != ".txt":
        raise SystemExit(f"{repo_rel(config_path)} train split must point to a .txt list")
    return path


def read_image_list(path: Path) -> list[str]:
    rows: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            rows.append(line.replace("\\", "/"))
    return rows


def write_image_list(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.replace(";", ",").replace(" ", ",").split(",") if item.strip()]


def parse_doses(value: str) -> list[int]:
    doses = sorted({int(item) for item in parse_csv(value)})
    if not doses or any(dose < 1 for dose in doses):
        raise SystemExit("--doses must contain positive integers")
    return doses


def manifest_rows(dose_root: Path) -> list[dict[str, Any]]:
    manifest_path = dose_root / "manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise SystemExit(f"expected manifest list: {repo_rel(manifest_path)}")
    rows = [row for row in document if isinstance(row, dict)]
    if not rows:
        raise SystemExit(f"manifest has no rows: {repo_rel(manifest_path)}")
    return rows


def visible_counts_for_row(dose_root: Path, row: dict[str, Any]) -> Counter[str]:
    visible_boxes = str(row.get("visible_boxes", "")).strip()
    if not visible_boxes:
        raise SystemExit("dose manifest row missing visible_boxes")
    boxes_doc = read_json(dose_root / visible_boxes)
    boxes = boxes_doc.get("boxes", [])
    if not isinstance(boxes, list):
        raise SystemExit(f"{visible_boxes}: boxes must be a list")
    return Counter(str(box.get("className", "unknown")) for box in boxes if isinstance(box, dict))


def balance_metrics(counter: Counter[str], classes: list[str]) -> dict[str, Any]:
    by_class = {class_name: int(counter.get(class_name, 0)) for class_name in classes}
    values = list(by_class.values())
    minimum = min(values) if values else 0
    maximum = max(values) if values else 0
    return {
        "total": int(sum(values)),
        "by_class": by_class,
        "covered_classes": int(sum(1 for value in values if value > 0)),
        "min_per_class": int(minimum),
        "max_per_class": int(maximum),
        "class_spread": int(maximum - minimum),
        "max_to_min_ratio": round(float(maximum / minimum), 6) if minimum else None,
    }


def balance_score(metrics: dict[str, Any], classes: list[str]) -> tuple[float, ...]:
    by_class = metrics["by_class"]
    values = [int(by_class[class_name]) for class_name in classes]
    mean = sum(values) / len(values) if values else 0.0
    mean_abs_error = sum(abs(value - mean) for value in values)
    return (
        -float(metrics["covered_classes"]),
        float(metrics["class_spread"]),
        -float(metrics["min_per_class"]),
        -float(metrics["total"]),
        round(mean_abs_error, 6),
    )


def selected_dose_rows(
    rows: list[dict[str, Any]],
    row_counts: list[Counter[str]],
    dose: int,
    classes: list[str],
    max_combinations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dose > len(rows):
        raise SystemExit(f"dose {dose} exceeds dose pool size {len(rows)}")
    combination_count = math.comb(len(rows), dose)
    if combination_count > max_combinations:
        raise SystemExit(
            f"dose {dose} would check {combination_count:,} combinations, "
            f"above --max-combinations {max_combinations:,}"
        )
    best_indexes: tuple[int, ...] | None = None
    best_metrics: dict[str, Any] | None = None
    best_score: tuple[float, ...] | None = None
    for indexes in itertools.combinations(range(len(rows)), dose):
        counts: Counter[str] = Counter()
        for index in indexes:
            counts.update(row_counts[index])
        metrics = balance_metrics(counts, classes)
        score = balance_score(metrics, classes)
        if best_score is None or score < best_score:
            best_indexes = indexes
            best_metrics = metrics
            best_score = score
    if best_indexes is None or best_metrics is None:
        raise RuntimeError("dose selection produced no candidates")
    return [rows[index] for index in best_indexes], {
        "source": "visible_boxes.physical_visible_instances",
        "classes": classes,
        "dose_images": int(dose),
        "search": {
            "method": "exact_combinations",
            "combinations_checked": int(combination_count),
            "max_combinations": int(max_combinations),
        },
        "selected_variants": [int(rows[index].get("variant")) for index in best_indexes],
        "selected_counts": best_metrics,
    }


def image_path_from_repo_line(dataset_root: Path, row: str) -> Path:
    path = Path(row)
    return path if path.is_absolute() else dataset_root / path


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError:
        return image.with_suffix(".txt")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def label_classes(image: Path) -> list[int]:
    label = label_path_for_image(image)
    classes: list[int] = []
    for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise SystemExit(f"{label}:{line_no} expected 5 YOLO fields, found {len(parts)}")
        classes.append(int(parts[0]))
    return classes


def label_class_counts(image: Path) -> Counter[int]:
    return Counter(label_classes(image))


def is_synthetic_row(row: str) -> bool:
    normalized = row.replace("\\", "/").lower()
    return normalized.startswith("data/synthetic/") or "/data/synthetic/" in normalized


def image_class_summary(dataset_root: Path, rows: list[str], names: dict[int, str]) -> dict[str, Any]:
    class_counts: Counter[int] = Counter()
    background_count = 0
    for row in rows:
        classes = label_classes(image_path_from_repo_line(dataset_root, row))
        if not classes:
            background_count += 1
        for class_id in set(classes):
            if class_id in names:
                class_counts[class_id] += 1
    return {
        "images": len(rows),
        "unique_images": len(set(rows)),
        "backgrounds": int(background_count),
        "class_images": {names[class_id]: int(class_counts[class_id]) for class_id in sorted(names)},
    }


def class_id_from_value(names: dict[int, str], value: str | None) -> int:
    if value is None or not value.strip():
        raise SystemExit("--control-class is required when --control-source class")
    token = value.strip()
    if token.isdigit() or (token.startswith("-") and token[1:].isdigit()):
        class_id = int(token)
        if class_id not in names:
            raise SystemExit(f"--control-class id is not in config names: {class_id}")
        return class_id
    matches = [class_id for class_id, class_name in names.items() if class_name == token]
    if not matches:
        valid = ", ".join(names[class_id] for class_id in sorted(names))
        raise SystemExit(f"--control-class must be one of: {valid}")
    return matches[0]


def row_count_control_rows(
    dataset_root: Path,
    base_rows: list[str],
    count: int,
    source: str,
    class_id: int | None = None,
    target_rows: list[str] | None = None,
    selection_report: list[dict[str, Any]] | None = None,
) -> list[str]:
    if count < 1:
        raise SystemExit("row-count control count must be positive")
    if source == "background":
        pool = [
            row
            for row in base_rows
            if not label_classes(image_path_from_repo_line(dataset_root, row))
        ]
        if not pool:
            raise SystemExit("base train list has no empty-label background rows for row-count controls")
    elif source == "first":
        pool = list(base_rows)
    elif source == "class":
        if class_id is None:
            raise SystemExit("--control-class is required when --control-source class")
        pool = [
            row
            for row in base_rows
            if class_id in label_classes(image_path_from_repo_line(dataset_root, row))
        ]
        if not pool:
            raise SystemExit(f"base train list has no rows for class id {class_id}")
    elif source in {"class_mix", "real_class_mix"}:
        if not target_rows:
            raise SystemExit(f"--control-source {source} requires selected dose rows")
        base_counts = [
            (row, label_class_counts(image_path_from_repo_line(dataset_root, row)))
            for row in base_rows
        ]
        real_only = source == "real_class_mix"
        pool = [
            item
            for item in base_counts
            if item[1] and (not real_only or not is_synthetic_row(item[0]))
        ]
        if not pool:
            qualifier = "real labeled" if real_only else "labeled"
            raise SystemExit(f"base train list has no {qualifier} rows for {source} row-count controls")
        control_rows: list[str] = []
        for target_row in target_rows:
            target_counts = label_class_counts(image_path_from_repo_line(dataset_root, target_row))
            if not target_counts:
                control_rows.append(row_count_control_rows(dataset_root, base_rows, 1, "background")[0])
                continue
            target_classes = set(target_counts)

            def match_details(row: str, counts: Counter[int]) -> dict[str, Any]:
                missing = sum(max(0, target_counts[class_id] - counts.get(class_id, 0)) for class_id in target_classes)
                extra = sum(count for class_id, count in counts.items() if class_id not in target_classes)
                surplus = sum(max(0, counts.get(class_id, 0) - target_counts[class_id]) for class_id in target_classes)
                overlap = sum(min(target_counts[class_id], counts.get(class_id, 0)) for class_id in target_classes)
                return {
                    "target_row": target_row,
                    "control_row": row,
                    "real_only": real_only,
                    "missing": int(missing),
                    "extra": int(extra),
                    "surplus": int(surplus),
                    "overlap": int(overlap),
                    "exact": missing == 0 and extra == 0 and surplus == 0,
                    "target_class_counts": dict(sorted((int(key), int(value)) for key, value in target_counts.items())),
                    "control_class_counts": dict(sorted((int(key), int(value)) for key, value in counts.items())),
                }

            def score(item: tuple[str, Counter[int]]) -> tuple[int, int, int, str]:
                row, counts = item
                details = match_details(row, counts)
                return (
                    int(details["missing"]),
                    int(details["extra"]) + int(details["surplus"]),
                    -int(details["overlap"]),
                    row,
                )

            selected_row, selected_counts = min(pool, key=score)
            control_rows.append(selected_row)
            if selection_report is not None:
                selection_report.append(match_details(selected_row, selected_counts))
        if len(control_rows) != count:
            raise SystemExit(f"{source} produced {len(control_rows)} controls for requested count {count}")
        return control_rows
    else:
        raise SystemExit(f"unsupported row-count control source: {source}")
    if not pool:
        raise SystemExit("base train list is empty")
    return [pool[index % len(pool)] for index in range(count)]


def names_by_id(config: dict[str, Any]) -> dict[int, str]:
    names = config.get("names", {})
    if not isinstance(names, dict):
        raise SystemExit("base config names must be a mapping")
    return {int(class_id): str(name) for class_id, name in names.items()}


def domain_gap_command(out_path: Path, json_out: Path, preset: str, skip: bool, fail_on_gap: bool) -> list[str]:
    if skip or not preset.strip():
        return []
    command = [
        sys.executable,
        "scripts/audit_yolo_domain_gap.py",
        "--data",
        repo_rel(out_path),
        "--split",
        "train",
        "--json-out",
        repo_rel(json_out),
        "--gate-preset",
        preset,
    ]
    if fail_on_gap:
        command.append("--fail-on-gap")
    return command


def yolo_check_command(out_path: Path, skip: bool) -> list[str]:
    if skip:
        return []
    return [sys.executable, "scripts/check_yolo_dataset.py", "--data", repo_rel(out_path)]


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def annotate_domain_gap(config_path: Path, json_out: Path, preset: str) -> None:
    config = read_yaml(config_path)
    domain_gap = read_json(json_out).get("domain_gap_gate", {})
    passed = bool(domain_gap.get("passed")) if isinstance(domain_gap, dict) else False
    config["cashsnap_domain_gap_gate"] = {
        "status": "pass" if passed else "fail",
        "preset": preset,
        "summary": repo_rel(json_out),
    }
    write_yaml(config_path, config)


def main() -> int:
    args = parse_args()
    base_path = resolve(args.base)
    dose_root = resolve(args.dose_root)
    out_dir = resolve(args.out_dir)
    list_dir = resolve(args.list_dir)
    domain_gap_dir = resolve(args.domain_gap_dir)
    doses = parse_doses(args.doses)
    dose_classes = parse_csv(args.dose_classes)

    base_config = read_yaml(base_path)
    dataset_root = data_root(base_path, base_config)
    base_train_list = train_list_path(base_path, base_config)
    base_rows = read_image_list(base_train_list)
    class_names = names_by_id(base_config)
    dose_manifest = manifest_rows(dose_root)
    row_counts = [visible_counts_for_row(dose_root, row) for row in dose_manifest]
    control_stem_prefix = args.control_stem_prefix or f"{args.stem_prefix}_row_control"
    control_class_id = class_id_from_value(class_names, args.control_class) if args.control_source == "class" else None

    def emit_config(
        *,
        stem: str,
        combined_rows: list[str],
        extra_policy: dict[str, Any],
        extra_sources: dict[str, Any],
        staged_dose: dict[str, Any] | None = None,
        row_count_control: dict[str, Any] | None = None,
    ) -> None:
        out_path = out_dir / f"{stem}.yaml"
        train_list = list_dir / f"{stem}_train.txt"
        domain_gap_json = domain_gap_dir / f"domain_gap_{stem}_train.json"

        if args.dry_run:
            print(f"would_write {repo_rel(out_path)}")
            print(f"would_write {repo_rel(train_list)}")
        else:
            write_image_list(train_list, combined_rows)
            config = copy.deepcopy(base_config)
            config["path"] = rel_between(out_path.parent, ROOT)
            config["train"] = repo_rel(train_list)
            config.pop("cashsnap_webgl_blend_gate", None)
            config.pop("cashsnap_domain_gap_gate", None)
            config.pop("cashsnap_webgl_staged_dose", None)
            config.pop("cashsnap_row_count_control", None)
            policy = copy.deepcopy(config.get("cashsnap_subset_policy", {}))
            if not isinstance(policy, dict):
                policy = {}
            selected_summary = image_class_summary(dataset_root, combined_rows, class_names)
            policy["selected_unique_images"] = selected_summary["unique_images"]
            policy["selected_images"] = selected_summary["images"]
            policy["selected_backgrounds"] = selected_summary["backgrounds"]
            policy["selected_class_images"] = selected_summary["class_images"]
            policy.update(extra_policy)
            config["cashsnap_subset_policy"] = policy
            sources = copy.deepcopy(config.get("cashsnap_sources", {}))
            if not isinstance(sources, dict):
                sources = {}
            sources["base_config"] = repo_rel(base_path)
            sources["base_train_list"] = repo_rel(base_train_list)
            sources.update(extra_sources)
            config["cashsnap_sources"] = sources
            if staged_dose is not None:
                config["cashsnap_webgl_staged_dose"] = staged_dose
            if row_count_control is not None:
                config["cashsnap_row_count_control"] = row_count_control
            write_yaml(out_path, config)

        check_command = yolo_check_command(out_path, args.skip_yolo_check)
        if check_command:
            run(check_command, args.dry_run)
        gap_command = domain_gap_command(
            out_path,
            domain_gap_json,
            args.domain_gap_preset,
            args.skip_domain_gap_gate,
            args.fail_on_domain_gap,
        )
        if gap_command:
            run(gap_command, args.dry_run)
            if not args.dry_run:
                annotate_domain_gap(out_path, domain_gap_json, args.domain_gap_preset)
        if not args.dry_run:
            print(f"wrote {repo_rel(out_path)}")
            print(f"wrote {repo_rel(train_list)}")

    for dose in doses:
        selected_rows, dose_report = selected_dose_rows(
            dose_manifest,
            row_counts,
            dose,
            dose_classes,
            args.max_combinations,
        )
        dose_image_rows = [
            repo_rel(dose_root / str(row["image"]))
            for row in selected_rows
        ]
        if (
            args.write_row_count_controls
            and args.fail_on_inexact_real_class_mix_control
            and args.control_source == "real_class_mix"
        ):
            precheck_match_report: list[dict[str, Any]] = []
            row_count_control_rows(
                dataset_root,
                base_rows,
                dose,
                args.control_source,
                class_id=control_class_id,
                target_rows=dose_image_rows,
                selection_report=precheck_match_report,
            )
            precheck_exact_matches = sum(1 for row in precheck_match_report if row.get("exact"))
            if precheck_match_report and precheck_exact_matches != len(precheck_match_report):
                raise SystemExit(
                    "real_class_mix row-count control is not exact "
                    f"for dose {dose}: {precheck_exact_matches}/{len(precheck_match_report)} exact matches"
                )
        combined_rows = list(dict.fromkeys([*base_rows, *dose_image_rows]))
        stem = f"{args.stem_prefix}_dose{dose}"

        print(
            f"dose={dose} variants={','.join(str(v) for v in dose_report['selected_variants'])} "
            f"counts={json.dumps(dose_report['selected_counts']['by_class'], sort_keys=True)}",
            flush=True,
        )
        dose_summary = image_class_summary(dataset_root, dose_image_rows, class_names)
        emit_config(
            stem=stem,
            combined_rows=combined_rows,
            extra_policy={
                "staged_dose_images": dose_summary["images"],
                "staged_dose_class_images": dose_summary["class_images"],
            },
            extra_sources={
                "staged_dose_root": repo_rel(dose_root),
                "staged_dose_recipe_id": args.recipe_id,
            },
            staged_dose={
                "base_config": repo_rel(base_path),
                "base_train_list": repo_rel(base_train_list),
                "dose_root": repo_rel(dose_root),
                "dose_recipe_id": args.recipe_id,
                "stem_prefix": args.stem_prefix,
                "dose_image_rows": dose_image_rows,
                **dose_report,
            },
        )

        if args.write_row_count_controls:
            control_match_report: list[dict[str, Any]] = []
            control_rows = row_count_control_rows(
                dataset_root,
                base_rows,
                dose,
                args.control_source,
                class_id=control_class_id,
                target_rows=dose_image_rows,
                selection_report=control_match_report,
            )
            control_summary = image_class_summary(dataset_root, control_rows, class_names)
            control_stem = f"{control_stem_prefix}_dose{dose}"
            control_source_label = args.control_source
            if control_class_id is not None:
                control_source_label = f"{args.control_source}:{class_names[control_class_id]}"
            elif args.control_source == "class_mix":
                control_source_label = "class_mix:synthetic_dose_rows"
            elif args.control_source == "real_class_mix":
                control_source_label = "real_class_mix:synthetic_dose_rows"
            control_note = (
                "Duplicates base rows to match train-list row count without adding synthetic note exposure."
            )
            if control_class_id is not None:
                control_note = (
                    "Duplicates base rows to match train-list row count and labeled class exposure "
                    "without adding new synthetic images."
                )
            elif args.control_source == "class_mix":
                control_note = (
                    "Duplicates base rows selected to match each synthetic dose row's class mix; "
                    "rows may be real or synthetic depending on the base train list."
                )
            elif args.control_source == "real_class_mix":
                control_note = (
                    "Duplicates real base rows selected to approximate each synthetic dose row's class mix; "
                    "check class_mix_match_report before interpreting it as an exact real-control."
                )
            exact_matches = sum(1 for row in control_match_report if row.get("exact"))
            if (
                args.fail_on_inexact_real_class_mix_control
                and args.control_source == "real_class_mix"
                and control_match_report
                and exact_matches != len(control_match_report)
            ):
                raise SystemExit(
                    "real_class_mix row-count control is not exact "
                    f"for dose {dose}: {exact_matches}/{len(control_match_report)} exact matches"
                )
            match_suffix = (
                f" class_mix_exact={exact_matches}/{len(control_match_report)}"
                if control_match_report
                else ""
            )
            print(
                f"row_control dose={dose} source={control_source_label} "
                f"unique={control_summary['unique_images']} backgrounds={control_summary['backgrounds']}"
                f"{match_suffix}",
                flush=True,
            )
            row_count_control_payload = {
                "base_config": repo_rel(base_path),
                "base_train_list": repo_rel(base_train_list),
                "stem_prefix": control_stem_prefix,
                "source": control_source_label,
                "matched_dose_images": int(dose),
                "control_image_rows": control_rows,
                "note": control_note,
            }
            if control_match_report:
                row_count_control_payload["class_mix_match_report"] = control_match_report
                row_count_control_payload["class_mix_exact_matches"] = int(exact_matches)
                row_count_control_payload["class_mix_all_exact"] = exact_matches == len(control_match_report)
            emit_config(
                stem=control_stem,
                combined_rows=[*base_rows, *control_rows],
                extra_policy={
                    "row_count_control_images": control_summary["images"],
                    "row_count_control_unique_images": control_summary["unique_images"],
                    "row_count_control_backgrounds": control_summary["backgrounds"],
                    "row_count_control_class_images": control_summary["class_images"],
                },
                extra_sources={
                    "row_count_control_base_train_list": repo_rel(base_train_list),
                    "row_count_control_source": control_source_label,
                },
                row_count_control=row_count_control_payload,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
