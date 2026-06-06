from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTHETIC_CASES = ROOT / "manifests" / "browser_synthetic_stress_cases.csv"
DEFAULT_MINED_CASES = ROOT / "runs" / "cashsnap" / "mined_real_browser_cases_latest.csv"
DEFAULT_SYNTHETIC_BASELINE = ROOT / "runs" / "cashsnap" / "browser_synthetic_stress_cases_v1.json"
DEFAULT_MINED_BASELINE = ROOT / "runs" / "cashsnap" / "browser_mined_real_scoreable_default_latest.json"
DEFAULT_PROJECT = ROOT / "runs" / "cashsnap"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a detector if needed, run browser stress cases, and compare with baseline reports."
    )
    parser.add_argument("--detector-model", required=True, type=Path, help="Detector .pt or .onnx path under the repo.")
    parser.add_argument("--label", default="", help="Stable label for output report names. Defaults to detector run name.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--synthetic-cases", type=Path, default=DEFAULT_SYNTHETIC_CASES)
    parser.add_argument("--mined-cases", type=Path, default=DEFAULT_MINED_CASES)
    parser.add_argument("--synthetic-baseline", type=Path, default=DEFAULT_SYNTHETIC_BASELINE)
    parser.add_argument("--mined-baseline", type=Path, default=DEFAULT_MINED_BASELINE)
    parser.add_argument("--synthetic-report", type=Path, default=None, help="Reuse this synthetic browser report.")
    parser.add_argument("--mined-report", type=Path, default=None, help="Reuse this mined-real browser report.")
    parser.add_argument("--synthetic-artifacts-dir", type=Path, default=None)
    parser.add_argument("--mined-artifacts-dir", type=Path, default=None)
    parser.add_argument("--no-synthetic", action="store_true", help="Skip synthetic browser stress.")
    parser.add_argument("--no-mined", action="store_true", help="Skip mined-real browser stress.")
    parser.add_argument("--reuse-existing", action="store_true", default=True)
    parser.add_argument("--rerun-existing", action="store_false", dest="reuse_existing")
    parser.add_argument("--no-export", action="store_true", help="Require an existing ONNX path; do not export from .pt.")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="0")
    parser.add_argument("--timeout-ms", default="120000")
    parser.add_argument("--port-base", type=int, default=8877)
    parser.add_argument("--debug-port-base", type=int, default=9323)
    parser.add_argument("--stack-config", default="")
    parser.add_argument("--proposal-conf", default="")
    parser.add_argument("--detector-override", default="")
    parser.add_argument("--fragment-classifier-model", default="")
    parser.add_argument("--reject-fragment-disagreement", action="store_true")
    parser.add_argument("--fragment-disagreement-min-conf", default="")
    parser.add_argument("--unclassified-min-conf", default="")
    parser.add_argument("--nms-iou", default="")
    parser.add_argument("--crop-padding", default="")
    parser.add_argument("--edge", default="")
    parser.add_argument("--no-artifacts", action="store_true")
    parser.add_argument("--skip-gate-effects", action="store_true")
    parser.add_argument("--skip-gate-review", action="store_true")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 after writing the probe summary.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def repo_path(path: Path) -> str:
    resolved = resolve(path)
    try:
        return resolved.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved.resolve())


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def default_label(detector_model: Path) -> str:
    resolved = resolve(detector_model)
    if resolved.name in {"best.pt", "best.onnx"} and resolved.parent.name == "weights":
        return slug(resolved.parent.parent.name)
    return slug(resolved.stem)


def run(command: list[str], *, allow_nonzero_with_report: Path | None = None) -> subprocess.CompletedProcess[str]:
    print(" ".join(command), flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode and (allow_nonzero_with_report is None or not allow_nonzero_with_report.exists()):
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)
    return result


def ensure_onnx(detector_model: Path, args: argparse.Namespace) -> Path:
    model = resolve(detector_model)
    if model.suffix.lower() == ".onnx":
        if not model.exists():
            raise FileNotFoundError(f"detector ONNX not found: {repo_path(model)}")
        return model
    if model.suffix.lower() != ".pt":
        raise ValueError(f"detector model must be .pt or .onnx: {repo_path(model)}")
    out_path = model.with_suffix(".onnx")
    if out_path.exists():
        return out_path
    if args.no_export:
        raise FileNotFoundError(f"detector ONNX not found and --no-export was set: {repo_path(out_path)}")
    run(
        [
            sys.executable,
            "scripts/export_yolo.py",
            "--model",
            repo_path(model),
            "--format",
            "onnx",
            "--imgsz",
            str(args.imgsz),
            "--device",
            str(args.device),
        ]
    )
    if not out_path.exists():
        raise FileNotFoundError(f"expected ONNX export was not written: {repo_path(out_path)}")
    return out_path


def browser_report_path(project: Path, kind: str, label: str) -> Path:
    return resolve(project) / f"browser_{kind}_{label}.json"


def browser_out_dir(project: Path, kind: str, label: str) -> Path:
    return resolve(project) / "browser_artifacts" / f"{kind}_{label}"


def compare_path(project: Path, kind: str, label: str) -> Path:
    return resolve(project) / f"browser_compare_{kind}_default_vs_{label}.json"


def gate_effect_paths(project: Path, kind: str, label: str) -> tuple[Path, Path]:
    stem = f"browser_gate_effects_{kind}_{label}"
    return resolve(project) / f"{stem}.csv", resolve(project) / f"{stem}.json"


def gate_review_paths(project: Path, label: str) -> tuple[Path, Path, Path]:
    stem = f"browser_gate_review_targets_{label}"
    return (
        resolve(project) / f"{stem}.csv",
        resolve(project) / f"{stem}.json",
        resolve(project) / f"{stem}.md",
    )


def run_browser_report(kind: str, label: str, onnx: Path, args: argparse.Namespace) -> dict[str, Any]:
    report_override = args.synthetic_report if kind == "synthetic" else args.mined_report
    artifacts_override = args.synthetic_artifacts_dir if kind == "synthetic" else args.mined_artifacts_dir
    report = (
        resolve(report_override)
        if report_override
        else browser_report_path(args.project, "synthetic_stress" if kind == "synthetic" else "mined_real", label)
    )
    cases = args.synthetic_cases if kind == "synthetic" else args.mined_cases
    out_dir = resolve(artifacts_override) if artifacts_override else browser_out_dir(args.project, kind, label)
    if args.reuse_existing and report.exists():
        return {"report": repo_path(report), "artifacts_dir": repo_path(out_dir), "reused": True, "exit_code": 0}

    command = [
        sys.executable,
        "scripts/run_browser_smoke_cases.py",
        "--cases",
        repo_path(resolve(cases)),
        "--summary-json",
        repo_path(report),
        "--out-dir",
        repo_path(out_dir),
        "--timeout-ms",
        str(args.timeout_ms),
        "--port-base",
        str(args.port_base if kind == "synthetic" else args.port_base + 100),
        "--debug-port-base",
        str(args.debug_port_base if kind == "synthetic" else args.debug_port_base + 100),
        "--detector-model",
        repo_path(onnx),
    ]
    if args.no_artifacts:
        command.append("--no-artifacts")
    for value, flag in [
        (args.edge, "--edge"),
        (args.stack_config, "--stack-config"),
        (args.proposal_conf, "--proposal-conf"),
        (args.detector_override, "--detector-override"),
        (args.fragment_classifier_model, "--fragment-classifier-model"),
        (args.fragment_disagreement_min_conf, "--fragment-disagreement-min-conf"),
        (args.unclassified_min_conf, "--unclassified-min-conf"),
        (args.nms_iou, "--nms-iou"),
        (args.crop_padding, "--crop-padding"),
    ]:
        if str(value).strip():
            command.extend([flag, str(value)])
    if args.reject_fragment_disagreement:
        command.append("--reject-fragment-disagreement")
    result = run(command, allow_nonzero_with_report=report)
    return {"report": repo_path(report), "artifacts_dir": repo_path(out_dir), "reused": False, "exit_code": result.returncode}


def compare_report(kind: str, label: str, report: Path, args: argparse.Namespace) -> dict[str, Any]:
    baseline = args.synthetic_baseline if kind == "synthetic" else args.mined_baseline
    out_path = compare_path(args.project, "synthetic" if kind == "synthetic" else "mined_real", label)
    run(
        [
            sys.executable,
            "scripts/compare_browser_reports.py",
            "--baseline",
            repo_path(resolve(baseline)),
            "--candidate",
            repo_path(resolve(report)),
            "--json-out",
            repo_path(out_path),
            "--no-fail",
        ]
    )
    comparison = json.loads(out_path.read_text(encoding="utf-8"))
    return {"comparison": repo_path(out_path), "passed": bool(comparison.get("passed")), "deltas": comparison.get("deltas", {})}


def summarize_gate_effects(kind: str, label: str, report: Path, artifacts_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_gate_effects or args.no_artifacts or not artifacts_dir.exists():
        return {}
    output_kind = "synthetic" if kind == "synthetic" else "mined_real"
    csv_out, json_out = gate_effect_paths(args.project, output_kind, label)
    run(
        [
            sys.executable,
            "scripts/summarize_browser_gate_effects.py",
            "--report",
            repo_path(report),
            "--artifacts-dir",
            repo_path(artifacts_dir),
            "--csv-out",
            repo_path(csv_out),
            "--json-out",
            repo_path(json_out),
        ]
    )
    summary = json.loads(json_out.read_text(encoding="utf-8"))
    return {
        "csv": repo_path(csv_out),
        "json": repo_path(json_out),
        "effect_counts": summary.get("effect_counts", {}),
    }


def summarize_gate_review(label: str, gate_effect_csvs: list[Path], args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_gate_review or not gate_effect_csvs:
        return {}
    csv_out, json_out, md_out = gate_review_paths(args.project, label)
    command = [
        sys.executable,
        "scripts/build_browser_gate_review_targets.py",
        "--csv-out",
        repo_path(csv_out),
        "--json-out",
        repo_path(json_out),
        "--md-out",
        repo_path(md_out),
    ]
    for path in gate_effect_csvs:
        command.extend(["--gate-effects", repo_path(path)])
    run(command)
    summary = json.loads(json_out.read_text(encoding="utf-8"))
    return {
        "csv": repo_path(csv_out),
        "json": repo_path(json_out),
        "md": repo_path(md_out),
        "target_count": summary.get("target_count", 0),
        "effect_counts": summary.get("effect_counts", {}),
        "priority_counts": summary.get("priority_counts", {}),
    }


def main() -> int:
    args = parse_args()
    if args.no_synthetic and args.no_mined:
        raise SystemExit("At least one of synthetic or mined-real browser stress must be enabled.")
    args.project = resolve(args.project)
    args.project.mkdir(parents=True, exist_ok=True)
    label = slug(args.label) if args.label else default_label(args.detector_model)
    onnx = ensure_onnx(args.detector_model, args)
    results: dict[str, Any] = {
        "schema": "cashsnap_browser_detector_probe_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "label": label,
        "detector_model": repo_path(resolve(args.detector_model)),
        "detector_onnx": repo_path(onnx),
        "settings": {
            "imgsz": args.imgsz,
            "timeout_ms": int(args.timeout_ms),
            "stack_config": args.stack_config,
            "proposal_conf": args.proposal_conf,
            "detector_override": args.detector_override,
            "fragment_classifier_model": args.fragment_classifier_model,
            "reject_fragment_disagreement": args.reject_fragment_disagreement,
            "fragment_disagreement_min_conf": args.fragment_disagreement_min_conf,
            "unclassified_min_conf": args.unclassified_min_conf,
            "nms_iou": args.nms_iou,
            "crop_padding": args.crop_padding,
        },
        "reports": {},
        "comparisons": {},
        "gate_effects": {},
    }

    if not args.no_synthetic:
        synthetic = run_browser_report("synthetic", label, onnx, args)
        results["reports"]["synthetic"] = synthetic
        synthetic_report = resolve(Path(synthetic["report"]))
        synthetic_artifacts = resolve(Path(synthetic["artifacts_dir"]))
        results["comparisons"]["synthetic"] = compare_report("synthetic", label, synthetic_report, args)
        effects = summarize_gate_effects("synthetic", label, synthetic_report, synthetic_artifacts, args)
        if effects:
            results["gate_effects"]["synthetic"] = effects
    if not args.no_mined:
        mined = run_browser_report("mined", label, onnx, args)
        results["reports"]["mined_real"] = mined
        mined_report = resolve(Path(mined["report"]))
        mined_artifacts = resolve(Path(mined["artifacts_dir"]))
        results["comparisons"]["mined_real"] = compare_report("mined", label, mined_report, args)
        effects = summarize_gate_effects("mined", label, mined_report, mined_artifacts, args)
        if effects:
            results["gate_effects"]["mined_real"] = effects

    gate_effect_csvs = [
        resolve(Path(row["csv"]))
        for row in results["gate_effects"].values()
        if isinstance(row, dict) and row.get("csv")
    ]
    gate_review = summarize_gate_review(label, gate_effect_csvs, args)
    if gate_review:
        results["gate_review"] = gate_review

    comparisons = results["comparisons"]
    results["passed"] = bool(comparisons) and all(bool(row.get("passed")) for row in comparisons.values())
    summary_path = resolve(args.summary_json) if args.summary_json else args.project / f"browser_detector_probe_{label}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote_probe={repo_path(summary_path)}")
    print(f"browser_detector_probe={'pass' if results['passed'] else 'blocked'} label={label}")
    if results["passed"] or args.no_fail:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
