#!/usr/bin/env python3
"""
verify_interpretation_bundle.py

Deterministic verification for an interpretation bundle produced by:
  - dv8_agent.py --temporal
  - backfill_temporal_payloads.py

Checks:
  - timeseries.json exists
  - INPUT_INTERPRETATION/SINGLE_REVISION_ANALYSIS_DATA contains revision folders
  - Each revision has OutputData/interpretation_payload.json
  - Paths referenced by interpretation_payload.json exist (DSM, DRH, plots, summary HTML)
  - Evidence graph diff index files exist (if present)

Writes a Markdown verification report under INPUT_INTERPRETATION/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_revision_dirs(single_root: Path) -> List[Path]:
    revs = []
    for p in single_root.iterdir():
        if not p.is_dir():
            continue
        if len(p.name) >= 2 and p.name[:2].isdigit():
            revs.append(p)
    return sorted(revs)


def rev_number_from_dir(path: Path) -> Optional[int]:
    try:
        return int(path.name.split("_", 1)[0])
    except Exception:
        return None


@dataclass(frozen=True)
class Finding:
    severity: str  # ISSUE | WARN | INFO
    message: str


def _exists_rel(base: Path, rel: Optional[str]) -> bool:
    if not rel:
        return False
    try:
        p = (base / rel).resolve()
    except Exception:
        return False
    return p.exists()


def verify_revision(rev_dir: Path) -> Tuple[List[Finding], Dict[str, Any]]:
    findings: List[Finding] = []
    out_dir = rev_dir / "OutputData"
    payload_path = out_dir / "interpretation_payload.json"
    if not payload_path.exists():
        findings.append(Finding("ISSUE", f"Missing interpretation payload: {payload_path}"))
        return findings, {}

    payload = read_json(payload_path)
    if not payload:
        findings.append(Finding("ISSUE", f"interpretation_payload.json is unreadable/empty: {payload_path}"))
        return findings, {}

    # Core fields
    if "metrics" not in payload:
        findings.append(Finding("ISSUE", "Payload missing key: metrics"))
    if "dsm_paths" not in payload:
        findings.append(Finding("ISSUE", "Payload missing key: dsm_paths"))

    # DSM paths must exist if provided
    dsm_paths = payload.get("dsm_paths") or {}
    if isinstance(dsm_paths, dict):
        for k in ("matrix_json", "drh_clustering_json"):
            rel = dsm_paths.get(k)
            if rel and not _exists_rel(out_dir, rel):
                findings.append(Finding("ISSUE", f"dsm_paths.{k} does not exist: OutputData/{rel}"))
    else:
        findings.append(Finding("ISSUE", "dsm_paths is not an object"))

    # DRH plots
    drh_plots = payload.get("drh_plots") or {}
    if isinstance(drh_plots, dict):
        for k in ("png", "pdf"):
            rel = drh_plots.get(k)
            if rel and not _exists_rel(out_dir, rel):
                findings.append(Finding("WARN", f"drh_plots.{k} missing: OutputData/{rel}"))

    # DV8 summary HTML (dangerous_files.source)
    dangerous = payload.get("dangerous_files") or {}
    if isinstance(dangerous, dict):
        src = dangerous.get("source")
        if src and not _exists_rel(out_dir, src):
            findings.append(Finding("WARN", f"dangerous_files.source missing: OutputData/{src}"))

    # Structural hotspot source (usually matrix.json)
    structural = payload.get("structural_hotspots") or {}
    if isinstance(structural, dict):
        src = structural.get("source")
        if src and not _exists_rel(out_dir, src):
            findings.append(Finding("WARN", f"structural_hotspots.source missing: OutputData/{src}"))

    return findings, payload


def verify_bundle(temporal_root: Path) -> Tuple[List[Finding], Dict[str, Any]]:
    findings: List[Finding] = []
    ts_path = temporal_root / "timeseries.json"
    if not ts_path.exists():
        findings.append(Finding("ISSUE", f"Missing timeseries.json: {ts_path}"))
        return findings, {}

    ts = read_json(ts_path)
    if not ts:
        findings.append(Finding("ISSUE", f"timeseries.json unreadable/empty: {ts_path}"))
        return findings, {}

    interp_root = temporal_root / "INPUT_INTERPRETATION"
    single_root = interp_root / "SINGLE_REVISION_ANALYSIS_DATA"
    if not single_root.exists():
        findings.append(Finding("ISSUE", f"Missing SINGLE_REVISION_ANALYSIS_DATA: {single_root}"))
        return findings, ts

    rev_dirs = find_revision_dirs(single_root)
    if not rev_dirs:
        findings.append(Finding("ISSUE", f"No revision folders found under: {single_root}"))
        return findings, ts

    # Revision mapping sanity
    expected = sorted(
        [int(r["revision_number"]) for r in (ts.get("revisions") or []) if isinstance(r, dict) and r.get("revision_number")]
    )
    found = sorted([n for n in (rev_number_from_dir(p) for p in rev_dirs) if n is not None])
    if expected and found and expected != found:
        findings.append(Finding("WARN", f"Revision numbers mismatch: timeseries={expected} vs folders={found}"))

    # Per revision checks
    per_rev: List[Dict[str, Any]] = []
    for rev_dir in rev_dirs:
        n = rev_number_from_dir(rev_dir)
        rev_findings, payload = verify_revision(rev_dir)
        per_rev.append(
            {
                "revision_dir": rev_dir.name,
                "revision_number": n,
                "issues": [f.message for f in rev_findings if f.severity == "ISSUE"],
                "warnings": [f.message for f in rev_findings if f.severity == "WARN"],
            }
        )
        findings.extend(rev_findings)

    # Evidence graph diff index (optional but recommended)
    evidence_index_path = interp_root / "evidence_graph_diff_index.json"
    if evidence_index_path.exists():
        evidence_index = read_json(evidence_index_path)
        if isinstance(evidence_index, list):
            for row in evidence_index:
                if not isinstance(row, dict):
                    continue
                rel = row.get("path")
                if isinstance(rel, str) and rel:
                    if not (interp_root / rel).exists():
                        findings.append(Finding("WARN", f"Missing evidence graph diff file: INPUT_INTERPRETATION/{rel}"))
        else:
            findings.append(Finding("WARN", f"evidence_graph_diff_index.json is not a list: {evidence_index_path}"))
    else:
        findings.append(Finding("INFO", f"No evidence_graph_diff_index.json found (ok): {evidence_index_path}"))

    return findings, {"timeseries": ts, "per_revision": per_rev}


def format_markdown(temporal_root: Path, findings: List[Finding], summary: Dict[str, Any]) -> str:
    issues = [f for f in findings if f.severity == "ISSUE"]
    warns = [f for f in findings if f.severity == "WARN"]
    infos = [f for f in findings if f.severity == "INFO"]
    status = "PASS" if not issues else "FAIL"

    lines: List[str] = []
    lines.append("## Interpretation Bundle Verification")
    lines.append(f"- status: {status}")
    lines.append(f"- temporal_root: {temporal_root}")
    lines.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## Findings")
    if issues:
        for f in issues:
            lines.append(f"- ISSUE: {f.message}")
    if warns:
        for f in warns:
            lines.append(f"- WARN: {f.message}")
    if infos:
        for f in infos:
            lines.append(f"- INFO: {f.message}")
    if not issues and not warns and not infos:
        lines.append("- No issues found.")
    lines.append("")

    per_rev = (summary or {}).get("per_revision") or []
    if isinstance(per_rev, list) and per_rev:
        lines.append("## Per-Revision")
        for row in per_rev:
            if not isinstance(row, dict):
                continue
            rev_dir = row.get("revision_dir")
            n = row.get("revision_number")
            lines.append(f"- revision {n}: {rev_dir}")
            for msg in row.get("issues") or []:
                lines.append(f"  - ISSUE: {msg}")
            for msg in row.get("warnings") or []:
                lines.append(f"  - WARN: {msg}")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the INPUT_INTERPRETATION bundle for a temporal analysis folder.")
    ap.add_argument("--temporal-root", required=True, help="Path to temporal_analysis_* folder")
    ap.add_argument("--output", default=None, help="Output markdown path (default: INPUT_INTERPRETATION/bundle_verification.md)")
    args = ap.parse_args()

    temporal_root = Path(args.temporal_root).expanduser().resolve()
    findings, summary = verify_bundle(temporal_root)

    interp_root = temporal_root / "INPUT_INTERPRETATION"
    default_out = interp_root / "bundle_verification.md"
    out_path = Path(args.output).expanduser().resolve() if args.output else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(format_markdown(temporal_root, findings, summary), encoding="utf-8")
    print(f"Wrote bundle verification: {out_path}")

    return 0 if not any(f.severity == "ISSUE" for f in findings) else 1


if __name__ == "__main__":
    raise SystemExit(main())

