from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from cashsnap_currency_taxonomy import (
    OFFICIAL_CURRENT_CLASS_NAMES,
    OFFICIAL_TAXONOMY_SOURCES,
    ROOT,
    class_names_for_scope,
    cutout_bank_coverage,
    numista_raw_coverage,
    repo_path,
    resolve_repo_path,
)


DEFAULT_METADATA = ROOT / "data" / "numista_raw" / "metadata.json"
DEFAULT_ACTIVE_CUTOUT_BANK = ROOT / "data" / "asset_candidates" / "numista_current_cutout_bank_v1"
DEFAULT_TAXONOMY_DATA = ROOT / "data" / "cashsnap_v1" / "data.yaml"
DEFAULT_CANDIDATE_BANKS = [
    ROOT / "data" / "asset_candidates" / "numista_current_fullscope_cutout_bank_probe_v1",
    ROOT / "data" / "asset_candidates" / "numista_official_fullscope_any_status_cutout_bank_probe_v1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a denomination-by-denomination plan for CashSnap official taxonomy gaps."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--active-cutout-bank", type=Path, default=DEFAULT_ACTIVE_CUTOUT_BANK)
    parser.add_argument("--taxonomy-data", type=Path, default=DEFAULT_TAXONOMY_DATA)
    parser.add_argument("--candidate-cutout-bank", type=Path, action="append", default=[])
    parser.add_argument("--class-scope", choices=["operational", "official"], default="official")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def read_yolo_class_names(path: Path) -> list[str]:
    resolved = resolve_repo_path(path).resolve()
    if not resolved.exists():
        return []
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    raw_names = data.get("names", {})
    if isinstance(raw_names, dict):
        rows: list[tuple[int, str]] = []
        for key, value in raw_names.items():
            try:
                rows.append((int(key), str(value)))
            except (TypeError, ValueError):
                continue
        return [name for _, name in sorted(rows)]
    if isinstance(raw_names, list):
        return [str(item) for item in raw_names]
    return []


def ready_classes(coverage: dict[str, Any]) -> set[str]:
    return set(str(name) for name in coverage.get("front_back_ready_class_names", []))


def raw_current_ready(raw: dict[str, Any], class_name: str) -> bool:
    return class_name in set(raw.get("current_front_back_ready_class_names", []))


def raw_any_ready(raw: dict[str, Any], class_name: str) -> bool:
    return class_name in set(raw.get("any_front_back_ready_class_names", []))


def class_examples(coverage: dict[str, Any], class_name: str) -> list[dict[str, Any]]:
    rows = coverage.get("coverage", {})
    row = rows.get(class_name, {}) if isinstance(rows, dict) else {}
    examples = row.get("examples", []) if isinstance(row, dict) else []
    return examples if isinstance(examples, list) else []


def candidate_banks(args: argparse.Namespace) -> list[Path]:
    if args.candidate_cutout_bank:
        return [resolve(path) for path in args.candidate_cutout_bank]
    return [path for path in DEFAULT_CANDIDATE_BANKS if path.exists()]


def target_action(
    *,
    class_name: str,
    raw_current: bool,
    raw_any: bool,
    active_ready: bool,
    model_ready: bool,
    candidate_ready: list[str],
) -> str:
    if active_ready and model_ready and raw_current:
        return "covered"
    if not raw_current and raw_any:
        return "status-review raw/any-status source before trainable current-currency use"
    if candidate_ready and not active_ready:
        return "review candidate cutouts, then promote into active bank if current/status/rights checks pass"
    if active_ready and not model_ready:
        return "schema/curriculum expansion needed after real labels and deployment scope are decided"
    if raw_current and not candidate_ready:
        return "build/review cutouts from raw current scans before schema expansion"
    return "decide whether this denomination is in product scope; otherwise document narrower scope"


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    target = class_names_for_scope(args.class_scope)
    raw = numista_raw_coverage(args.metadata, scope=args.class_scope)
    active = cutout_bank_coverage(args.active_cutout_bank, scope=args.class_scope)
    model_classes = set(read_yolo_class_names(args.taxonomy_data))
    candidate_reports = [
        cutout_bank_coverage(path, scope=args.class_scope)
        for path in candidate_banks(args)
    ]
    active_ready = ready_classes(active)
    rows: list[dict[str, Any]] = []
    for class_name in target:
        candidate_ready = [
            report["bank"]
            for report in candidate_reports
            if class_name in ready_classes(report)
        ]
        row_raw_current = raw_current_ready(raw, class_name)
        row_raw_any = raw_any_ready(raw, class_name)
        row_active = class_name in active_ready
        row_model = class_name in model_classes
        rows.append(
            {
                "class_name": class_name,
                "raw_current_front_back_ready": row_raw_current,
                "raw_any_front_back_ready": row_raw_any,
                "active_cutout_front_back_ready": row_active,
                "model_schema_ready": row_model,
                "candidate_cutout_banks_ready": candidate_ready,
                "active_examples": class_examples(active, class_name),
                "action": target_action(
                    class_name=class_name,
                    raw_current=row_raw_current,
                    raw_any=row_raw_any,
                    active_ready=row_active,
                    model_ready=row_model,
                    candidate_ready=candidate_ready,
                ),
            }
        )
    gaps = [row for row in rows if not (row["raw_current_front_back_ready"] and row["active_cutout_front_back_ready"] and row["model_schema_ready"])]
    return {
        "schema": "cashsnap_currency_taxonomy_gap_plan_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "class_scope": args.class_scope,
        "official_sources": OFFICIAL_TAXONOMY_SOURCES,
        "official_current_class_names": OFFICIAL_CURRENT_CLASS_NAMES,
        "inputs": {
            "metadata": repo_path(resolve_repo_path(args.metadata).resolve()),
            "active_cutout_bank": repo_path(resolve(args.active_cutout_bank)),
            "taxonomy_data": repo_path(resolve_repo_path(args.taxonomy_data).resolve()),
            "candidate_cutout_banks": [report["bank"] for report in candidate_reports],
        },
        "counts": {
            "target_classes": len(target),
            "complete_classes": len(rows) - len(gaps),
            "gap_classes": len(gaps),
            "raw_current_front_back_ready": len(raw.get("current_front_back_ready_class_names", [])),
            "raw_any_front_back_ready": len(raw.get("any_front_back_ready_class_names", [])),
            "active_cutout_front_back_ready": len(active.get("front_back_ready_class_names", [])),
            "model_schema_ready": len([name for name in target if name in model_classes]),
        },
        "rows": rows,
        "gaps": gaps,
    }


def clean_table(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "/").strip()


def mark(value: bool) -> str:
    return "yes" if value else "no"


def write_markdown(path: Path, plan: dict[str, Any]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = plan["counts"]
    lines = [
        "# CashSnap Currency Taxonomy Gap Plan",
        "",
        f"Scope: `{plan['class_scope']}`",
        f"Complete classes: {counts['complete_classes']}/{counts['target_classes']}",
        f"Gap classes: {counts['gap_classes']}",
        "",
        "Candidate cutout banks:",
        *[f"- `{path}`" for path in plan["inputs"]["candidate_cutout_banks"]],
        "",
        "| Class | Raw Current | Raw Any | Active Cutout | Model Schema | Candidate Banks | Action |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in plan["rows"]:
        candidate_banks = row["candidate_cutout_banks_ready"]
        candidate_text = (
            ""
            if row["action"] == "covered"
            else ", ".join(f"`{Path(path).name}`" for path in candidate_banks) if candidate_banks else ""
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['class_name']}`",
                    mark(bool(row["raw_current_front_back_ready"])),
                    mark(bool(row["raw_any_front_back_ready"])),
                    mark(bool(row["active_cutout_front_back_ready"])),
                    mark(bool(row["model_schema_ready"])),
                    candidate_text,
                    clean_table(row["action"]),
                ]
            )
            + " |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote_md={repo_path(out)}")


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    if args.json_out:
        out = resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_json={repo_path(out)}")
    if args.md_out:
        write_markdown(args.md_out, plan)
    counts = plan["counts"]
    print(
        "currency_taxonomy_gap_plan="
        f"complete={counts['complete_classes']}/{counts['target_classes']} "
        f"gaps={counts['gap_classes']} "
        f"active={counts['active_cutout_front_back_ready']} "
        f"model={counts['model_schema_ready']}"
    )


if __name__ == "__main__":
    main()
