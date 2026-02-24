#!/usr/bin/env python3
"""
verify_interpretation_report.py

Lightweight verifier pass for interpret_drh_diff outputs.
Checks required headings, disallows code fences, and flags file references
outside the prompt's ALLOWED FILES list.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


REQUIRED_HEADINGS = [
    "## Comprehensive Summary",
    "## DRH Differences",
    "## Metrics & Evidence",
    "## Likely Drivers",
]

FILE_TOKEN_RE = re.compile(r"(?<![\\w/.-])([\\w./-]+\\.[\\w.-]+)(?![\\w/.-])")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def extract_allowed_files(prompt_text: str) -> List[str]:
    lines = prompt_text.splitlines()
    start = None
    end = None
    for i, line in enumerate(lines):
        if line.strip().startswith("ALLOWED FILES"):
            start = i + 1
            continue
        if start is not None and line.strip().startswith("FACTS"):
            end = i
            break
    if start is None:
        return []
    if end is None:
        end = len(lines)
    allowed = []
    for line in lines[start:end]:
        line = line.strip()
        if line.startswith("- "):
            name = line[2:].strip()
            if name and name != "(none)":
                allowed.append(name)
    return allowed


def first_heading(report_text: str) -> str:
    for line in report_text.splitlines():
        if line.startswith("## "):
            return line.strip()
    return ""


def section_text(report_text: str, heading: str) -> str:
    if not report_text or not heading:
        return ""
    lines = report_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i + 1
            break
    if start is None:
        return ""
    end = None
    for j in range(start, len(lines)):
        if lines[j].startswith("## "):
            end = j
            break
    if end is None:
        end = len(lines)
    return "\n".join(lines[start:end]).strip()


def find_file_mentions(report_text: str) -> List[str]:
    raw = FILE_TOKEN_RE.findall(report_text or "")
    mentions = []
    for token in raw:
        cleaned = token.strip("`*()[]{}.,:;")
        if not cleaned:
            continue
        # Filter out numeric-only tokens like "63.08".
        if not any(ch.isalpha() for ch in cleaned):
            continue
        mentions.append(cleaned)
    return mentions


def verify(report_text: str, prompt_text: str) -> Dict[str, List[str]]:
    issues: List[str] = []
    warnings: List[str] = []

    if "```" in report_text:
        issues.append("Report contains code fences (```), which are disallowed.")

    for heading in REQUIRED_HEADINGS:
        if heading not in report_text:
            issues.append(f"Missing required heading: {heading}")

    first = first_heading(report_text)
    if first and first != "## Comprehensive Summary":
        issues.append(f"First heading is '{first}', expected '## Comprehensive Summary'.")

    if report_text.count("## Comprehensive Summary") > 1:
        issues.append("Comprehensive Summary appears multiple times.")

    if "ALLOWED FILES" in report_text:
        warnings.append("Report repeats prompt scaffolding ('ALLOWED FILES').")

    # Section content sanity
    drh_text = section_text(report_text, "## DRH Differences")
    ev_text = section_text(report_text, "## Metrics & Evidence")
    drv_text = section_text(report_text, "## Likely Drivers")
    if drh_text and len(drh_text.splitlines()) < 2:
        warnings.append("DRH Differences section looks very short.")
    if ev_text and len(ev_text.splitlines()) < 2:
        warnings.append("Metrics & Evidence section looks very short.")
    if drv_text and len(drv_text.splitlines()) < 2:
        warnings.append("Likely Drivers section looks very short.")

    allowed_files = extract_allowed_files(prompt_text)
    if not allowed_files:
        warnings.append("Prompt ALLOWED FILES list missing or empty; file reference checks skipped.")
    else:
        allowed = set(allowed_files)
        allowed_basenames = {Path(a).name for a in allowed_files}
        mentions = find_file_mentions(report_text)
        unknown = []
        for m in mentions:
            if m in allowed or m in allowed_basenames:
                continue
            unknown.append(m)
        if unknown:
            sample = ", ".join(sorted(set(unknown))[:20])
            issues.append(f"Report references file names not in ALLOWED FILES: {sample}")

    return {"issues": issues, "warnings": warnings}


def format_report(
    report_path: Path, prompt_path: Path | None, result: Dict[str, List[str]]
) -> str:
    status = "PASS" if not result["issues"] else "FAIL"
    lines = [
        "## Verification Summary",
        f"- status: {status}",
        f"- report: {report_path}",
    ]
    if prompt_path:
        lines.append(f"- prompt: {prompt_path}")
    lines.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## Findings")
    if result["issues"]:
        for msg in result["issues"]:
            lines.append(f"- ISSUE: {msg}")
    if result["warnings"]:
        for msg in result["warnings"]:
            lines.append(f"- WARN: {msg}")
    if not result["issues"] and not result["warnings"]:
        lines.append("- No issues found.")

    return "\n".join(lines).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify interpret_drh_diff report output.")
    ap.add_argument("--report", required=True, help="Path to drh_diff_report_*.md")
    ap.add_argument("--prompt", default=None, help="Path to prompt file (optional)")
    ap.add_argument("--output", default=None, help="Output verification markdown path")
    args = ap.parse_args()

    report_path = Path(args.report).expanduser().resolve()
    prompt_path = Path(args.prompt).expanduser().resolve() if args.prompt else None

    report_text = read_text(report_path)
    prompt_text = read_text(prompt_path) if prompt_path else ""

    result = verify(report_text, prompt_text)
    default_out = report_path.with_name(report_path.stem + ".verify.md")
    out_path = Path(args.output).expanduser().resolve() if args.output else default_out

    out_path.write_text(format_report(report_path, prompt_path, result), encoding="utf-8")
    print(f"Wrote verification report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
