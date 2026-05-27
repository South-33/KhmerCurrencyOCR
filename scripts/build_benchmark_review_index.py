from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = ROOT / "manifests" / "real_fan_benchmark_sources.csv"
DEFAULT_TASKS = ROOT / "manifests" / "real_fan_benchmark_label_tasks.csv"
DEFAULT_DRAFT_LABEL_DIR = ROOT / "data" / "real_fan_benchmark" / "drafts"
DEFAULT_OUT = ROOT / "data" / "real_fan_benchmark" / "review_index.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local review index for CashSnap benchmark candidates.")
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--draft-label-dir", type=Path, default=DEFAULT_DRAFT_LABEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    return resolve(path).relative_to(ROOT).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_yolo_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#"))


def labeler_url(image_path: str, draft_label_path: str | None) -> str:
    url = f"/demo/labeler/?image=/{quote(image_path)}"
    if draft_label_path:
        url += f"&labels=/{quote(draft_label_path)}"
    return url


def row_html(source: dict[str, str], task: dict[str, str], draft_dir: Path) -> str:
    image_id = source["image_id"]
    image_path = source["local_path"]
    draft_label = draft_dir / f"{image_id}.txt"
    draft_path = repo_path(draft_label) if draft_label.exists() else None
    draft_count = count_yolo_rows(draft_label)
    status = source.get("label_status") or task.get("label_status", "")
    link = labeler_url(image_path, draft_path)
    return f"""
      <article class="item">
        <a class="thumb" href="{html.escape(link)}">
          <img src="/{html.escape(image_path)}" alt="" />
        </a>
        <div class="body">
          <h2>{html.escape(image_id)}</h2>
          <p><strong>Status:</strong> {html.escape(status)} / <strong>Draft boxes:</strong> {draft_count}</p>
          <p><strong>Priority:</strong> {html.escape(task.get("priority", ""))} / <strong>Benchmark:</strong> {html.escape(source.get("benchmark_status", ""))}</p>
          <p>{html.escape(task.get("notes", source.get("notes", "")))}</p>
          <a class="button" href="{html.escape(link)}">Open labeler</a>
        </div>
      </article>
    """


def main() -> None:
    args = parse_args()
    sources = read_csv(args.sources)
    tasks = {row["image_id"]: row for row in read_csv(args.tasks)}
    draft_dir = resolve(args.draft_label_dir)
    items = "\n".join(row_html(source, tasks.get(source["image_id"], {}), draft_dir) for source in sources)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CashSnap Benchmark Review</title>
    <style>
      :root {{ font-family: "Segoe UI", system-ui, sans-serif; color: #151817; background: #f3f5f2; }}
      body {{ margin: 0; padding: 18px; }}
      main {{ width: min(1200px, 100%); margin: 0 auto; }}
      h1 {{ margin: 0 0 14px; font-size: 26px; letter-spacing: 0; }}
      .item {{ display: grid; grid-template-columns: 280px 1fr; gap: 14px; padding: 12px; margin-bottom: 12px; border: 1px solid #cfd6d1; border-radius: 8px; background: white; }}
      .thumb img {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #eef1ee; }}
      h2 {{ margin: 0 0 8px; font-size: 18px; letter-spacing: 0; }}
      p {{ margin: 0 0 8px; color: #47524d; }}
      .button {{ display: inline-block; border-radius: 6px; background: #151817; color: white; padding: 8px 11px; text-decoration: none; }}
      @media (max-width: 760px) {{ .item {{ grid-template-columns: 1fr; }} }}
    </style>
  </head>
  <body>
    <main>
      <h1>CashSnap Benchmark Review</h1>
      {items}
    </main>
  </body>
</html>
""",
        encoding="utf-8",
    )
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
