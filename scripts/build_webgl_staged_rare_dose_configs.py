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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--dose-root", type=Path, default=DEFAULT_DOSE_ROOT)
    parser.add_argument("--recipe-id", default="webgl_rare_class_support_v1")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--list-dir", type=Path, default=DEFAULT_LIST_DIR)
    parser.add_argument("--domain-gap-dir", type=Path, default=DEFAULT_DOMAIN_GAP_DIR)
    parser.add_argument("--doses", default="1,2,4,8,16", help="Comma/space-separated dose image counts.")
    parser.add_argument("--dose-classes", default=DEFAULT_CLASSES)
    parser.add_argument("--max-combinations", type=int, default=500_000)
    parser.add_argument("--domain-gap-preset", default="accepted_blend_v1")
    parser.add_argument("--skip-domain-gap-gate", action="store_true")
    parser.add_argument("--fail-on-domain-gap", action="store_true")
    parser.add_argument("--skip-yolo-check", action="store_true")
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
        float(metrics["class_spread"]),
        round(mean_abs_error, 6),
        -float(metrics["min_per_class"]),
        -float(metrics["total"]),
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
        "backgrounds": int(background_count),
        "class_images": {names[class_id]: int(class_counts[class_id]) for class_id in sorted(names)},
    }


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
        combined_rows = list(dict.fromkeys([*base_rows, *dose_image_rows]))
        stem = f"cashsnap_v1_plus_webgl_accepted_rare_support_dose{dose}"
        out_path = out_dir / f"{stem}.yaml"
        train_list = list_dir / f"{stem}_train.txt"
        domain_gap_json = domain_gap_dir / f"domain_gap_{stem}_train.json"

        print(
            f"dose={dose} variants={','.join(str(v) for v in dose_report['selected_variants'])} "
            f"counts={json.dumps(dose_report['selected_counts']['by_class'], sort_keys=True)}",
            flush=True,
        )
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
            policy = copy.deepcopy(config.get("cashsnap_subset_policy", {}))
            if not isinstance(policy, dict):
                policy = {}
            selected_summary = image_class_summary(dataset_root, combined_rows, class_names)
            dose_summary = image_class_summary(dataset_root, dose_image_rows, class_names)
            policy["selected_unique_images"] = selected_summary["images"]
            policy["selected_images"] = selected_summary["images"]
            policy["selected_backgrounds"] = selected_summary["backgrounds"]
            policy["selected_class_images"] = selected_summary["class_images"]
            policy["staged_dose_images"] = dose_summary["images"]
            policy["staged_dose_class_images"] = dose_summary["class_images"]
            config["cashsnap_subset_policy"] = policy
            sources = copy.deepcopy(config.get("cashsnap_sources", {}))
            if not isinstance(sources, dict):
                sources = {}
            sources["base_config"] = repo_rel(base_path)
            sources["base_train_list"] = repo_rel(base_train_list)
            sources["staged_dose_root"] = repo_rel(dose_root)
            sources["staged_dose_recipe_id"] = args.recipe_id
            config["cashsnap_sources"] = sources
            config["cashsnap_webgl_staged_dose"] = {
                "base_config": repo_rel(base_path),
                "base_train_list": repo_rel(base_train_list),
                "dose_root": repo_rel(dose_root),
                "dose_recipe_id": args.recipe_id,
                "dose_image_rows": dose_image_rows,
                **dose_report,
            }
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
