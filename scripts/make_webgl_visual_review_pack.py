#!/usr/bin/env python
"""Build a human visual-review pack for packaged WebGL synthetic outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_smoke_suite_v1.json"
DEFAULT_RULES = ROOT / "configs" / "synthetic_recipes" / "cashsnap_webgl_visual_review_rules_v1.json"
DEFAULT_OUT = ROOT / "data" / "review" / "webgl_visual_review_pack_v1"
FIELDNAMES = [
    "suite_name",
    "recipe_id",
    "scene_mode",
    "artifact_status",
    "variant",
    "package_root",
    "image_path",
    "detect_preview",
    "fragment_preview",
    "id_overlay",
    "contact_sheet",
    "visible_instances",
    "fragments",
    "ignored_fragments",
    "review_required_fragments",
    "obb_status",
    "visual_quality_status",
    "visual_quality_failures",
    "quarantine_actions",
    "review_status",
    "bad_scene_reasons",
    "reviewer",
    "reviewed_at",
    "review_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-images-per-recipe", type=int, default=0, help="0 means include every packaged image.")
    parser.add_argument("--seed", type=int, default=33)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_rel(path: Path) -> str:
    return resolve(path).resolve().relative_to(ROOT).as_posix()


def out_rel(out_dir: Path, target: Path) -> str:
    return os.path.relpath(resolve(target).resolve(), out_dir.resolve()).replace("\\", "/")


def read_json(path: Path) -> object:
    resolved = resolve(path)
    if not resolved.exists():
        raise SystemExit(f"missing JSON file: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def split_actions(quarantine: object) -> dict[int, list[str]]:
    actions: dict[int, list[str]] = {}
    if not isinstance(quarantine, dict):
        return actions
    rows = quarantine.get("rows", [])
    if not isinstance(rows, list):
        return actions
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            variant = int(row.get("variant", ""))
        except (TypeError, ValueError):
            continue
        action = str(row.get("action", "")).strip()
        if action:
            actions.setdefault(variant, []).append(action)
    return actions


def detail_by_variant(summary: object) -> dict[int, dict]:
    details: dict[int, dict] = {}
    if not isinstance(summary, dict):
        return details
    rows = summary.get("images_detail", [])
    if not isinstance(rows, list):
        return details
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            details[int(row.get("variant", ""))] = row
        except (TypeError, ValueError):
            continue
    return details


def choose_rows(rows: list[dict], limit: int, rng: random.Random) -> list[dict]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    return sorted(rng.sample(rows, limit), key=lambda row: int(row.get("variant", 0)))


def recipe_contact_sheet(root: Path, recipe: dict) -> Path:
    outputs = recipe.get("outputs", {})
    if isinstance(outputs, dict) and str(outputs.get("contact_sheet", "")).strip():
        return resolve(Path(str(outputs["contact_sheet"])))
    return root / "contact_sheet.png"


def manifest_path(root: Path, row: dict, field: str, recipe_id: str, variant: int) -> Path:
    value = str(row.get(field, "")).strip()
    require(value, f"{recipe_id} variant {variant}: missing manifest field {field}")
    path = root / value
    require(path.exists(), f"{recipe_id} variant {variant}: missing {field} file: {path}")
    return path


def collect_recipe_rows(suite_name: str, suite_row: dict, limit: int, rng: random.Random) -> list[dict[str, str]]:
    recipe_id = str(suite_row.get("recipe_id", ""))
    scene_mode = str(suite_row.get("scene_mode", ""))
    root = resolve(Path(str(suite_row.get("out_root", ""))))
    require(recipe_id, "suite recipe row missing recipe_id")
    require(scene_mode, f"{recipe_id}: suite recipe row missing scene_mode")
    require(root.exists(), f"{recipe_id}: package root does not exist: {root}")

    manifest = read_json(root / "manifest.json")
    recipe = read_json(root / "recipe.json")
    summary = read_json(root / "qa" / "summary.json")
    quarantine = read_json(root / "qa" / "quarantine.json")
    require(isinstance(manifest, list), f"{recipe_id}: manifest.json must be a list")
    require(isinstance(recipe, dict), f"{recipe_id}: recipe.json must be an object")

    details = detail_by_variant(summary)
    actions = split_actions(quarantine)
    artifact_status = str(recipe.get("artifact_status", ""))
    contact_sheet = recipe_contact_sheet(root, recipe)
    require(contact_sheet.exists(), f"{recipe_id}: missing contact sheet: {contact_sheet}")
    rows: list[dict[str, str]] = []
    for row in choose_rows([item for item in manifest if isinstance(item, dict)], limit, rng):
        try:
            variant = int(row.get("variant", ""))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{recipe_id}: manifest row missing numeric variant") from exc
        detail = details.get(variant, {})
        visual_quality = detail.get("visual_quality", {}) if isinstance(detail, dict) else {}
        if not isinstance(visual_quality, dict):
            visual_quality = {}
        image_path = manifest_path(root, row, "image", recipe_id, variant)
        detect_preview = manifest_path(root, row, "detect_preview", recipe_id, variant)
        fragment_preview = manifest_path(root, row, "fragment_preview", recipe_id, variant)
        id_overlay = manifest_path(root, row, "id_overlay", recipe_id, variant)
        rows.append(
            {
                "suite_name": suite_name,
                "recipe_id": recipe_id,
                "scene_mode": scene_mode,
                "artifact_status": artifact_status,
                "variant": str(variant),
                "package_root": repo_rel(root),
                "image_path": repo_rel(image_path),
                "detect_preview": repo_rel(detect_preview),
                "fragment_preview": repo_rel(fragment_preview),
                "id_overlay": repo_rel(id_overlay),
                "contact_sheet": repo_rel(contact_sheet),
                "visible_instances": str(detail.get("visible_instances", "")),
                "fragments": str(detail.get("fragments", "")),
                "ignored_fragments": str(detail.get("ignored_fragments", "")),
                "review_required_fragments": str(detail.get("review_required_fragments", "")),
                "obb_status": str(row.get("obb_status", "")),
                "visual_quality_status": str(visual_quality.get("status", "")),
                "visual_quality_failures": ";".join(str(item) for item in visual_quality.get("failures", [])),
                "quarantine_actions": ";".join(sorted(set(actions.get(variant, [])))),
                "review_status": "",
                "bad_scene_reasons": "",
                "reviewer": "",
                "reviewed_at": "",
                "review_notes": "",
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_html(rows: list[dict[str, str]], out_dir: Path, out_path: Path, rules_name: str) -> None:
    cards: list[str] = []
    for row in rows:
        title = f"{row['recipe_id']} / {row['scene_mode']} / variant {row['variant']}"
        meta = (
            f"visual={row['visual_quality_status'] or 'unknown'} | obb={row['obb_status'] or 'unknown'} | "
            f"visible={row['visible_instances']} | fragments={row['fragments']} | quarantine={row['quarantine_actions'] or 'none'}"
        )
        image_cells = []
        for label, field in (("RGB", "image_path"), ("Detect", "detect_preview"), ("Fragments", "fragment_preview"), ("ID", "id_overlay")):
            image_cells.append(
                "<figure>"
                f"<img src=\"{html.escape(out_rel(out_dir, ROOT / row[field]))}\" alt=\"{html.escape(label)} preview\">"
                f"<figcaption>{html.escape(label)}</figcaption>"
                "</figure>"
            )
        cards.append(
            "<section class=\"card\">"
            f"<h2>{html.escape(title)}</h2>"
            f"<p>{html.escape(meta)}</p>"
            "<div class=\"grid\">"
            + "\n".join(image_cells)
            + "</div>"
            "</section>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CashSnap WebGL Visual Review</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f7f8; color: #1c1f23; }}
h1 {{ font-size: 22px; margin: 0 0 8px; }}
.meta {{ color: #4a5563; margin-bottom: 20px; }}
.card {{ background: #fff; border: 1px solid #d8dde3; border-radius: 6px; padding: 14px; margin-bottom: 16px; }}
.card h2 {{ font-size: 16px; margin: 0 0 6px; }}
.card p {{ margin: 0 0 12px; color: #4a5563; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 10px; }}
figure {{ margin: 0; }}
img {{ width: 100%; height: auto; border: 1px solid #cfd6dd; background: #111; }}
figcaption {{ font-size: 12px; margin-top: 4px; color: #4a5563; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: repeat(2, minmax(160px, 1fr)); }} }}
</style>
</head>
<body>
<h1>CashSnap WebGL Visual Review</h1>
<p class="meta">{len(rows)} scenes | rules: {html.escape(rules_name)} | fill review columns in review.csv</p>
{''.join(cards)}
</body>
</html>
"""
    out_path.write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    suite = read_json(args.suite)
    rules = read_json(args.rules)
    require(isinstance(suite, dict), "suite must be a JSON object")
    require(isinstance(rules, dict), "rules must be a JSON object")
    suite_rows = suite.get("recipes", [])
    require(isinstance(suite_rows, list) and suite_rows, "suite recipes must be a non-empty list")

    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    for suite_row in suite_rows:
        if not isinstance(suite_row, dict):
            raise SystemExit("suite recipe rows must be objects")
        rows.extend(collect_recipe_rows(str(suite.get("name", "")), suite_row, args.max_images_per_recipe, rng))

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    review_csv = out_dir / "review.csv"
    write_csv(rows, review_csv)
    (out_dir / "rules.json").write_text(json.dumps(rules, indent=2) + "\n", encoding="utf-8")
    write_html(rows, out_dir, out_dir / "index.html", str(rules.get("name", args.rules.name)))
    print(f"wrote {repo_rel(review_csv)} with {len(rows)} review row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
