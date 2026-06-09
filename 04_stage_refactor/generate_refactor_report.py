#!/usr/bin/env python3
"""Generate a manager-friendly report summarizing the auto-refactor loop results.

Reads the temporal analysis folder and produces a Markdown report covering:
- Repository overview and analysis scope
- Baseline architecture metrics (from the newest real commit)
- Each refactoring iteration: what was identified, what action was taken, metric deltas
- Final summary with before/after comparison

Usage:
    python3 generate_refactor_report.py --temporal-root <path>
    python3 generate_refactor_report.py --temporal-root <path> --output report.md
"""

from __future__ import annotations
import argparse
import json
import pathlib
import re
import sys
from datetime import datetime


def load_timeseries(temporal_root: pathlib.Path) -> dict:
    for candidate in [
        temporal_root / "INPUT_INTERPRETATION" / "timeseries.json",
        temporal_root / "timeseries.json",
    ]:
        if candidate.exists():
            return json.loads(candidate.read_text())
    return {}


def find_interpretation_reports(temporal_root: pathlib.Path) -> list[pathlib.Path]:
    """Find DRH diff / temporal interpretation reports."""
    out_dir = temporal_root / "OUTPUT_INTERPRETATION"
    if not out_dir.is_dir():
        return []
    reports = []
    for run_dir in sorted(out_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for f in sorted(run_dir.iterdir()):
            if f.suffix == ".md" and f.name != "USER_ANSWER_20260525.md":
                reports.append(f)
    return reports


def find_loop_folders(temporal_root: pathlib.Path) -> list[pathlib.Path]:
    """Find refactoring loop iteration folders (003_..._loop1_..., etc)."""
    data_repos = temporal_root / "data_repositories"
    if not data_repos.is_dir():
        return []
    loop_dirs = []
    for d in sorted(data_repos.iterdir()):
        if d.is_dir() and "_loop" in d.name:
            loop_dirs.append(d)
    return loop_dirs


def find_real_revisions(temporal_root: pathlib.Path) -> list[pathlib.Path]:
    """Find real (non-loop) revision folders."""
    data_repos = temporal_root / "data_repositories"
    if not data_repos.is_dir():
        return []
    return sorted(
        [d for d in data_repos.iterdir()
         if d.is_dir() and "_loop" not in d.name
         and len(d.name) > 2 and d.name[:2].isdigit()],
        key=lambda p: p.name
    )


def read_metrics(rev_dir: pathlib.Path) -> dict:
    """Read metrics from a revision's OutputData."""
    metrics = {}
    metrics_dir = rev_dir / "OutputData" / "metrics"
    if not metrics_dir.is_dir():
        return metrics
    for mf in metrics_dir.iterdir():
        if mf.suffix == ".json":
            try:
                data = json.loads(mf.read_text())
                name = mf.stem
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, (int, float)):
                            metrics[name] = v
                            break
                elif isinstance(data, (int, float)):
                    metrics[name] = data
            except (json.JSONDecodeError, ValueError):
                pass
    return metrics


def read_interpretation_payload(rev_dir: pathlib.Path) -> str:
    """Read the interpretation payload markdown."""
    md_path = rev_dir / "OutputData" / "interpretation_payload.md"
    if md_path.exists():
        return md_path.read_text(encoding="utf-8", errors="replace")
    return ""


def extract_dangerous_files(payload_md: str) -> list[str]:
    """Extract dangerous file entries from interpretation payload."""
    files = []
    for line in payload_md.splitlines():
        if line.startswith("- Filename:"):
            # Extract just the filename part
            match = re.match(r"- Filename:\s*(.+?)(?:,\s*LOC:)", line)
            if match:
                files.append(match.group(1).strip())
    return files[:10]  # Top 10


def read_qa_conversation(temporal_root: pathlib.Path) -> list[dict]:
    """Read Q&A answers from OUTPUT_INTERPRETATION folders."""
    out_dir = temporal_root / "OUTPUT_INTERPRETATION"
    if not out_dir.is_dir():
        return []

    qa_entries = []
    for run_dir in sorted(out_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for f in sorted(run_dir.iterdir()):
            if f.name.startswith("USER_ANSWER_") and f.suffix == ".md":
                content = f.read_text(encoding="utf-8", errors="replace")
                qa_entries.append({"file": f.name, "content": content})
    return qa_entries


def extract_loop_action(loop_dir: pathlib.Path) -> str | None:
    """Try to extract what action was applied in a loop iteration."""
    # Check if there's a refactor log
    for candidate in [
        loop_dir / "OutputData" / "refactor_action.txt",
        loop_dir / "refactor_action.txt",
    ]:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="replace").strip()
    return None


def generate_report(temporal_root: pathlib.Path) -> str:
    """Generate the full manager report."""
    ts = load_timeseries(temporal_root)
    real_revs = find_real_revisions(temporal_root)
    loop_dirs = find_loop_folders(temporal_root)

    repo_name = ts.get("repo", temporal_root.parent.name)
    rev_count = len(real_revs)
    start_month = ts.get("start_month", "?")
    end_month = ts.get("end_month", "?")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Separate AI revisions from real ones in timeseries
    real_ts_revisions = [r for r in ts.get("revisions", [])
                         if not r.get("commit_hash", "").startswith("ai-")]
    ai_ts_revisions = [r for r in ts.get("revisions", [])
                       if r.get("commit_hash", "").startswith("ai-")]

    lines = []
    lines.append(f"# Architecture Refactoring Report")
    lines.append(f"**Repository:** {repo_name}")
    lines.append(f"**Analysis period:** {start_month} to {end_month}")
    lines.append(f"**Revisions analyzed:** {rev_count}")
    lines.append(f"**Refactoring iterations:** {len(loop_dirs)}")
    lines.append(f"**Generated:** {now}")
    lines.append("")

    # --- Section 1: Baseline ---
    lines.append("---")
    lines.append("## 1. Baseline Architecture (Before Refactoring)")
    lines.append("")

    if real_ts_revisions:
        newest = real_ts_revisions[0]
        oldest = real_ts_revisions[-1]
        m = newest.get("metrics", {})
        lines.append(f"**Latest revision:** {newest.get('commit_hash', '?')[:12]} "
                     f"({newest.get('commit_date', '?')[:10]})")
        lines.append(f"- **M-Score (Modularity):** {m.get('m-score', '?')}%")
        lines.append(f"- **Propagation Cost:** {m.get('propagation-cost', '?')}%")
        lines.append(f"- **Decoupling Level:** {m.get('decoupling-level', '?')}%")
        lines.append(f"- **Independence Level:** {m.get('independence-level', '?')}%")
        lines.append("")

        # Trend from oldest to newest
        if len(real_ts_revisions) > 1:
            om = oldest.get("metrics", {})
            lines.append("### Historical Trend")
            lines.append(f"| Metric | Oldest ({oldest.get('commit_date', '?')[:10]}) "
                         f"| Latest ({newest.get('commit_date', '?')[:10]}) | Delta |")
            lines.append("|--------|--------|--------|-------|")
            for metric_key in ["m-score", "propagation-cost", "decoupling-level", "independence-level"]:
                old_val = om.get(metric_key)
                new_val = m.get(metric_key)
                if old_val is not None and new_val is not None:
                    delta = new_val - old_val
                    sign = "+" if delta >= 0 else ""
                    lines.append(f"| {metric_key} | {old_val:.2f}% | {new_val:.2f}% | {sign}{delta:.2f}% |")
            lines.append("")

    # Dangerous files from newest revision
    if real_revs:
        payload = read_interpretation_payload(real_revs[0])
        dangerous = extract_dangerous_files(payload)
        if dangerous:
            lines.append("### Top Problematic Files (DV8 Analysis)")
            for i, f in enumerate(dangerous[:5], 1):
                lines.append(f"{i}. `{f}`")
            lines.append("")

    # --- Section 2: Interpretation Summary ---
    lines.append("---")
    lines.append("## 2. Architecture Interpretation")
    lines.append("")

    reports = find_interpretation_reports(temporal_root)
    temporal_reports = [r for r in reports if "temporal_interpretation" in r.name]
    drh_reports = [r for r in reports if "drh_diff" in r.name and r.suffix == ".md"
                   and ".prompt." not in r.name and ".verify." not in r.name]

    if temporal_reports:
        lines.append("### Overall Temporal Assessment")
        content = temporal_reports[0].read_text(encoding="utf-8", errors="replace")
        # Take first 500 chars as executive summary
        summary = content[:1500].strip()
        if len(content) > 1500:
            summary += "\n\n*(truncated — see full report)*"
        lines.append(summary)
        lines.append("")

    if drh_reports:
        lines.append(f"### Period-by-Period Analysis ({len(drh_reports)} transitions)")
        for dr in drh_reports:
            # Extract period from filename
            period_match = re.search(r"(\d{4}-\d{2})_to_(\d{4}-\d{2})", dr.name)
            period = f"{period_match.group(1)} → {period_match.group(2)}" if period_match else dr.stem
            lines.append(f"- **{period}** — see `{dr.name}`")
        lines.append("")

    # --- Section 3: Refactoring Iterations ---
    lines.append("---")
    lines.append("## 3. Automated Refactoring Results")
    lines.append("")

    if not loop_dirs and not ai_ts_revisions:
        lines.append("*No refactoring iterations were performed.*")
        lines.append("")
    else:
        # Build iteration table
        if ai_ts_revisions:
            lines.append("### Iteration Summary")
            lines.append("")
            lines.append("| Iteration | M-Score | Delta | Status |")
            lines.append("|-----------|---------|-------|--------|")

            baseline_mscore = None
            if real_ts_revisions:
                baseline_mscore = real_ts_revisions[0].get("metrics", {}).get("m-score")
                lines_baseline = f"| Baseline | {baseline_mscore:.2f}% | — | Original |"
                lines.append(lines_baseline)

            prev = baseline_mscore
            for ai_rev in reversed(ai_ts_revisions):  # oldest AI first
                m = ai_rev.get("metrics", {}).get("m-score")
                if m is not None:
                    delta = m - prev if prev is not None else 0
                    sign = "+" if delta >= 0 else ""
                    status = "Improved" if delta > 0.1 else ("No change" if abs(delta) <= 0.1 else "Regressed")
                    loop_num = re.search(r"loop(\d+)", ai_rev.get("commit_hash", ""))
                    iter_label = f"Loop {loop_num.group(1)}" if loop_num else ai_rev.get("commit_hash", "?")
                    lines.append(f"| {iter_label} | {m:.2f}% | {sign}{delta:.2f}% | {status} |")
                    prev = m
            lines.append("")

            # Final vs baseline
            if ai_ts_revisions and baseline_mscore is not None:
                final_mscore = ai_ts_revisions[0].get("metrics", {}).get("m-score")
                if final_mscore is not None:
                    total_delta = final_mscore - baseline_mscore
                    sign = "+" if total_delta >= 0 else ""
                    lines.append(f"**Net M-Score change: {sign}{total_delta:.2f}% "
                                 f"({baseline_mscore:.2f}% → {final_mscore:.2f}%)**")
                    lines.append("")

        # Per-iteration details
        for i, loop_dir in enumerate(loop_dirs, 1):
            lines.append(f"### Iteration {i}: `{loop_dir.name}`")
            payload = read_interpretation_payload(loop_dir)
            if payload:
                # Extract metrics from payload
                for metric_line in payload.splitlines():
                    if metric_line.startswith("- m_score:") or \
                       metric_line.startswith("- propagation_cost:") or \
                       metric_line.startswith("- decoupling_level:") or \
                       metric_line.startswith("- independence_level:"):
                        lines.append(f"  {metric_line.strip()}")

            action = extract_loop_action(loop_dir)
            if action:
                lines.append(f"\n**Action applied:**")
                lines.append(action[:500])

            dangerous = extract_dangerous_files(payload)
            if dangerous:
                lines.append(f"\n**Remaining problematic files:**")
                for f in dangerous[:3]:
                    lines.append(f"  - `{f}`")
            lines.append("")

    # --- Section 4: Recommendations ---
    lines.append("---")
    lines.append("## 4. Key Takeaways")
    lines.append("")

    if real_ts_revisions:
        m = real_ts_revisions[0].get("metrics", {})
        mscore = m.get("m-score")
        pcost = m.get("propagation-cost")
        if mscore is not None:
            if mscore >= 90:
                lines.append("- Architecture modularity is **good** (M-Score >= 90%).")
            elif mscore >= 80:
                lines.append("- Architecture modularity is **moderate** (M-Score 80-90%). "
                             "Targeted refactoring recommended.")
            else:
                lines.append("- Architecture modularity is **poor** (M-Score < 80%). "
                             "Significant refactoring needed.")
        if pcost is not None:
            if pcost <= 5:
                lines.append("- Change propagation risk is **low** (Propagation Cost <= 5%).")
            elif pcost <= 15:
                lines.append("- Change propagation risk is **moderate** (Propagation Cost 5-15%).")
            else:
                lines.append("- Change propagation risk is **high** (Propagation Cost > 15%). "
                             "Changes tend to cascade widely.")

    if ai_ts_revisions and baseline_mscore is not None:
        final_mscore = ai_ts_revisions[0].get("metrics", {}).get("m-score")
        if final_mscore is not None and final_mscore > baseline_mscore:
            lines.append(f"- Automated refactoring **improved** modularity by "
                         f"+{final_mscore - baseline_mscore:.2f}% across {len(ai_ts_revisions)} iteration(s).")
        elif final_mscore is not None:
            lines.append(f"- Automated refactoring did not significantly improve modularity. "
                         f"Manual architectural review recommended.")

    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated by ARCH_AGENT on {now}*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate manager-friendly refactoring report")
    parser.add_argument("--temporal-root", required=True, help="Path to temporal analysis folder")
    parser.add_argument("--output", help="Output file path (default: <temporal-root>/REFACTOR_REPORT.md)")
    args = parser.parse_args()

    temporal_root = pathlib.Path(args.temporal_root).expanduser().resolve()
    if not temporal_root.is_dir():
        print(f"Error: {temporal_root} is not a directory", file=sys.stderr)
        return 1

    report = generate_report(temporal_root)

    if args.output:
        out_path = pathlib.Path(args.output).expanduser().resolve()
    else:
        out_path = temporal_root / "REFACTOR_REPORT.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Report written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
