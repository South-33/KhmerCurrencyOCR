from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BOX_METRICS = ["box_area", "box_width", "box_height", "box_aspect"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize strict real-vs-synthetic visible-note geometry failures into repair targets."
    )
    parser.add_argument("--domain-gap-json", required=True, type=Path)
    parser.add_argument("--box-csv", required=True, type=Path)
    parser.add_argument("--image-csv", type=Path, default=None)
    parser.add_argument("--top-classes", type=int, default=12)
    parser.add_argument("--top-groups", type=int, default=8)
    parser.add_argument("--top-images", type=int, default=16)
    parser.add_argument("--include-passing-classes", action="store_true")
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    resolved = resolve(path)
    try:
        return resolved.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved.resolve())


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(resolve(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{repo_path(path)}: expected JSON object")
    return data


def read_csv(path: Path) -> list[dict[str, str]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def float_value(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def source_family(row: dict[str, str]) -> str:
    return row.get("source_family", "").strip()


def is_real(row: dict[str, str]) -> bool:
    return source_family(row) == "real"


def is_synthetic(row: dict[str, str]) -> bool:
    return source_family(row) == "synthetic"


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_metric_rows(rows: list[dict[str, str]]) -> dict[str, float | None]:
    return {
        metric: mean([float_value(row.get(metric)) for row in rows if row.get(metric) not in {None, ""}])
        for metric in BOX_METRICS
    }


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "").strip()].append(row)
    return dict(grouped)


def gate_limits(payload: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    gate = payload.get("domain_gap_gate", {})
    limits = gate.get("limits", {}) if isinstance(gate, dict) else {}
    box_limits = limits.get("box", {}) if isinstance(limits, dict) else {}
    class_limits = limits.get("class_box", {}) if isinstance(limits, dict) else {}
    return (
        {metric: float_value(value) for metric, value in box_limits.items()},
        {metric: float_value(value) for metric, value in class_limits.items()},
    )


def metric_delta(row_stats: dict[str, float | None], real_stats: dict[str, float | None], metric: str) -> float | None:
    value = row_stats.get(metric)
    real_value = real_stats.get(metric)
    if value is None or real_value is None:
        return None
    return float(value) - float(real_value)


def normalized_severity(delta: float | None, limit: float | None) -> float:
    if delta is None or limit is None or limit <= 0:
        return 0.0
    return abs(delta) / max(limit, 1e-9)


def class_targets(
    *,
    real_boxes: list[dict[str, str]],
    synthetic_boxes: list[dict[str, str]],
    class_limits: dict[str, float],
    include_passing: bool,
) -> list[dict[str, Any]]:
    real_by_class = group_rows(real_boxes, "class_name")
    synthetic_by_class = group_rows(synthetic_boxes, "class_name")
    targets: list[dict[str, Any]] = []
    for class_name in sorted(set(real_by_class) & set(synthetic_by_class)):
        real_stats = summarize_metric_rows(real_by_class[class_name])
        synthetic_stats = summarize_metric_rows(synthetic_by_class[class_name])
        deltas = {metric: metric_delta(synthetic_stats, real_stats, metric) for metric in BOX_METRICS}
        failing_metrics = [
            metric
            for metric, delta in deltas.items()
            if metric in class_limits and delta is not None and abs(delta) > class_limits[metric]
        ]
        if not failing_metrics and not include_passing:
            continue
        dominant_metric = max(
            BOX_METRICS,
            key=lambda metric: normalized_severity(deltas.get(metric), class_limits.get(metric)),
        )
        targets.append(
            {
                "kind": "class",
                "class_name": class_name,
                "real_boxes": len(real_by_class[class_name]),
                "synthetic_boxes": len(synthetic_by_class[class_name]),
                "dominant_metric": dominant_metric,
                "severity": normalized_severity(deltas.get(dominant_metric), class_limits.get(dominant_metric)),
                "failing_metrics": ",".join(failing_metrics),
                **{f"real_{metric}": real_stats.get(metric) for metric in BOX_METRICS},
                **{f"synthetic_{metric}": synthetic_stats.get(metric) for metric in BOX_METRICS},
                **{f"delta_{metric}": deltas.get(metric) for metric in BOX_METRICS},
                "repair_hint": class_repair_hint(class_name, deltas, failing_metrics),
            }
        )
    return sorted(targets, key=lambda row: (-float(row["severity"]), str(row["class_name"])))


def class_repair_hint(class_name: str, deltas: dict[str, float | None], failing_metrics: list[str]) -> str:
    if not failing_metrics:
        return "Keep as regression coverage."
    if "box_aspect" in failing_metrics and class_name.startswith("USD_"):
        return "Review USD in-plane pose/aspect bands; area repair alone may keep aspect wrong."
    if "box_width" in failing_metrics or "box_height" in failing_metrics:
        return "Increase visible-note scale for this class and rerun strict geometry selection."
    return "Tune class-specific geometry selection before adding more training rows."


def group_action(group: str, deltas: dict[str, float | None]) -> str:
    lower = group.lower()
    if "thin_edge" in lower:
        return "Keep as protected partial-slice data; do not use it to satisfy general visible-note scale."
    if "clean_base" in lower:
        return "Render or select closer/larger clean notes before more clean-base scale."
    if "back_side" in lower:
        return "Back-side confusion rows need larger visible notes or stricter selected-geometry filtering."
    if "hard_negative" in lower:
        return "No-box hard negatives do not explain visible-note geometry; keep them on the hard-negative axis."
    if all((delta is not None and delta < 0) for metric, delta in deltas.items() if metric != "box_aspect"):
        return "Increase note scale or select larger-box rows before training."
    return "Review this source group before changing recipe scale."


def source_group_targets(
    *,
    real_boxes: list[dict[str, str]],
    synthetic_boxes: list[dict[str, str]],
    image_rows: list[dict[str, str]],
    box_limits: dict[str, float],
) -> list[dict[str, Any]]:
    real_stats = summarize_metric_rows(real_boxes)
    images_by_group = Counter(row.get("source_group", "").strip() for row in image_rows if is_synthetic(row))
    targets: list[dict[str, Any]] = []
    for group, rows in group_rows(synthetic_boxes, "source_group").items():
        stats = summarize_metric_rows(rows)
        deltas = {metric: metric_delta(stats, real_stats, metric) for metric in BOX_METRICS}
        dominant_metric = max(
            BOX_METRICS,
            key=lambda metric: normalized_severity(deltas.get(metric), box_limits.get(metric)),
        )
        targets.append(
            {
                "kind": "source_group",
                "source_group": group,
                "synthetic_images": int(images_by_group.get(group, 0)),
                "synthetic_boxes": len(rows),
                "dominant_metric": dominant_metric,
                "severity": normalized_severity(deltas.get(dominant_metric), box_limits.get(dominant_metric)),
                **{f"synthetic_{metric}": stats.get(metric) for metric in BOX_METRICS},
                **{f"delta_{metric}": deltas.get(metric) for metric in BOX_METRICS},
                "repair_hint": group_action(group, deltas),
            }
        )
    return sorted(targets, key=lambda row: (-float(row["severity"]), str(row["source_group"])))


def image_targets(synthetic_boxes: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for image, image_boxes in group_rows(synthetic_boxes, "image").items():
        stats = summarize_metric_rows(image_boxes)
        classes = sorted({row.get("class_name", "") for row in image_boxes if row.get("class_name", "")})
        rows.append(
            {
                "kind": "image",
                "image": image,
                "source_group": image_boxes[0].get("source_group", "") if image_boxes else "",
                "box_count": len(image_boxes),
                "classes": ",".join(classes),
                **{f"mean_{metric}": stats.get(metric) for metric in BOX_METRICS},
            }
        )
    return sorted(rows, key=lambda row: (float(row["mean_box_area"] or 0), float(row["mean_box_width"] or 0), str(row["image"])))


def round_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: round_value(value) for key, value in row.items()}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean_row(row) for row in rows)
    print(f"wrote_csv={repo_path(out)}")


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.3f}"
    return str(value)


def fmt_plain(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def clean_table(value: Any) -> str:
    return str(value).replace("\n", " ").replace("|", "/").strip()


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    out = resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    aggregate = payload["aggregate"]
    lines = [
        "# CashSnap Geometry Gap Targets",
        "",
        f"Domain gap: `{payload['domain_gap_json']}`",
        f"Box CSV: `{payload['box_csv']}`",
        "",
        f"Gate status: `{payload['gate_status']}`",
        "Aggregate synthetic-minus-real deltas: "
        + ", ".join(f"{metric}={fmt(aggregate.get(f'delta_{metric}'))}" for metric in BOX_METRICS),
        "",
        "## Source Groups",
        "",
        "| Source Group | Images | Boxes | Area | Width | Height | Aspect | Hint |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["source_group_targets"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{clean_table(row['source_group'])}`",
                    str(row["synthetic_images"]),
                    str(row["synthetic_boxes"]),
                    fmt(row.get("delta_box_area")),
                    fmt(row.get("delta_box_width")),
                    fmt(row.get("delta_box_height")),
                    fmt(row.get("delta_box_aspect")),
                    clean_table(row["repair_hint"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Classes",
            "",
            "| Class | Real Boxes | Synthetic Boxes | Dominant | Area | Width | Height | Aspect | Failing | Hint |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in payload["class_targets"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{clean_table(row['class_name'])}`",
                    str(row["real_boxes"]),
                    str(row["synthetic_boxes"]),
                    clean_table(row["dominant_metric"]),
                    fmt(row.get("delta_box_area")),
                    fmt(row.get("delta_box_width")),
                    fmt(row.get("delta_box_height")),
                    fmt(row.get("delta_box_aspect")),
                    clean_table(row["failing_metrics"]),
                    clean_table(row["repair_hint"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Smallest Synthetic Rows",
            "",
            "| Image | Group | Boxes | Classes | Mean Area | Mean Width | Mean Height |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in payload["image_targets"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{clean_table(row['image'])}`",
                    f"`{clean_table(row['source_group'])}`",
                    str(row["box_count"]),
                    clean_table(row["classes"]),
                    fmt_plain(row.get("mean_box_area")),
                    fmt_plain(row.get("mean_box_width")),
                    fmt_plain(row.get("mean_box_height")),
                ]
            )
            + " |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote_md={repo_path(out)}")


def main() -> None:
    args = parse_args()
    payload_json = read_json(args.domain_gap_json)
    box_rows = read_csv(args.box_csv)
    image_rows = read_csv(args.image_csv) if args.image_csv else []
    real_boxes = [row for row in box_rows if is_real(row)]
    synthetic_boxes = [row for row in box_rows if is_synthetic(row)]
    if not real_boxes or not synthetic_boxes:
        raise SystemExit("box CSV must contain both real and synthetic rows")

    box_limits, class_limits = gate_limits(payload_json)
    real_stats = summarize_metric_rows(real_boxes)
    synthetic_stats = summarize_metric_rows(synthetic_boxes)
    aggregate = {
        **{f"real_{metric}": real_stats.get(metric) for metric in BOX_METRICS},
        **{f"synthetic_{metric}": synthetic_stats.get(metric) for metric in BOX_METRICS},
        **{f"delta_{metric}": metric_delta(synthetic_stats, real_stats, metric) for metric in BOX_METRICS},
        "real_boxes": len(real_boxes),
        "synthetic_boxes": len(synthetic_boxes),
    }
    class_rows = class_targets(
        real_boxes=real_boxes,
        synthetic_boxes=synthetic_boxes,
        class_limits=class_limits,
        include_passing=args.include_passing_classes,
    )[: max(0, args.top_classes)]
    group_rows_out = source_group_targets(
        real_boxes=real_boxes,
        synthetic_boxes=synthetic_boxes,
        image_rows=image_rows,
        box_limits=box_limits,
    )[: max(0, args.top_groups)]
    image_rows_out = image_targets(synthetic_boxes)[: max(0, args.top_images)]
    gate = payload_json.get("domain_gap_gate", {})
    report = {
        "schema": "cashsnap_domain_gap_geometry_targets_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "domain_gap_json": repo_path(args.domain_gap_json),
        "box_csv": repo_path(args.box_csv),
        "image_csv": repo_path(args.image_csv) if args.image_csv else "",
        "gate_status": "pass" if isinstance(gate, dict) and gate.get("passed") else "blocked",
        "gate_preset": gate.get("limits", {}).get("preset", "") if isinstance(gate, dict) else "",
        "aggregate": clean_row(aggregate),
        "source_group_targets": [clean_row(row) for row in group_rows_out],
        "class_targets": [clean_row(row) for row in class_rows],
        "image_targets": [clean_row(row) for row in image_rows_out],
    }
    if args.csv_out:
        combined_rows = [
            {**row, "section": "source_group"} for row in report["source_group_targets"]
        ] + [
            {**row, "section": "class"} for row in report["class_targets"]
        ] + [
            {**row, "section": "image"} for row in report["image_targets"]
        ]
        fieldnames = sorted({key for row in combined_rows for key in row})
        write_csv(args.csv_out, combined_rows, fieldnames)
    if args.md_out:
        write_markdown(args.md_out, report)
    if args.json_out:
        out = resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_json={repo_path(out)}")
    print(
        "domain_gap_geometry_targets="
        f"groups={len(report['source_group_targets'])} "
        f"classes={len(report['class_targets'])} "
        f"images={len(report['image_targets'])} "
        f"gate={report['gate_status']}"
    )


if __name__ == "__main__":
    main()
