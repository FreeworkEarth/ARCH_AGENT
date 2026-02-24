#!/usr/bin/env python3
"""
interpret_metrics.py - Lightweight Metric Interpreter (Option A)

This script interprets metric changes by correlating them with git commits.
Uses Ollama (DeepSeek-R1 or similar) for reasoning.

Usage:
    python interpret_metrics.py --repo ../REPOS/pdfbox --timeseries ../REPOS/pdfbox/timeseries.json
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import subprocess

from commit_analyzer import CommitAnalyzer


class MetricInterpreter:
    """Interpret metric changes using git context and LLM reasoning"""

    def __init__(self, repo_path: str, model: str = "deepseek-r1:14b"):
        self.repo_path = Path(repo_path)
        self.model = model
        self.commit_analyzer = CommitAnalyzer(repo_path)

    def load_timeseries(self, timeseries_path: str) -> Dict:
        """Load timeseries.json from temporal analysis"""
        with open(timeseries_path, 'r') as f:
            return json.load(f)

    def query_ollama(self, prompt: str) -> str:
        """Query Ollama for reasoning"""
        try:
            cmd = [
                "ollama", "run", self.model,
                prompt
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            out = (result.stdout or "").strip()
            if out:
                return out
            return (result.stderr or "").strip()
        except subprocess.TimeoutExpired:
            return "LLM query timed out"
        except Exception as e:
            return f"Error querying LLM: {e}"

    def interpret_metric_change(
        self,
        metric_name: str,
        old_value: float,
        new_value: float,
        commit_summary: Dict
    ) -> str:
        """
        Use LLM to interpret a single metric change.

        Args:
            metric_name: Name of the metric
            old_value: Previous value
            new_value: Current value
            commit_summary: Summary of commits in this period

        Returns:
            Interpretation text
        """
        change = new_value - old_value
        change_pct = (change / old_value * 100) if old_value != 0 else 0

        # Build context prompt
        prompt = f"""You are an expert software architect analyzing architectural changes.

METRIC CHANGE:
- Metric: {metric_name}
- Previous value: {old_value:.2f}%
- Current value: {new_value:.2f}%
- Change: {change:+.2f}% ({change_pct:+.2f}%)

COMMIT CONTEXT ({commit_summary['total_commits']} commits):
Commit types:
{json.dumps(commit_summary['categories'], indent=2)}

Top commits:
{json.dumps(commit_summary['top_commits'], indent=2)}

Top contributors:
{json.dumps(commit_summary['top_authors'], indent=2)}

Sample commit messages:
{chr(10).join([f"- {c['message'][:100]}" for c in commit_summary['commit_samples'][:5]])}

TASK:
Explain in 2-3 sentences WHY this metric changed based on the commits.
Focus on architectural implications, not just code changes.
"""

        response = self.query_ollama(prompt)
        return response

    def generate_report(
        self,
        timeseries_path: str,
        output_path: str = None
    ) -> str:
        """
        Generate full interpretation report.

        Args:
            timeseries_path: Path to timeseries.json
            output_path: Where to save report (default: same dir as timeseries)

        Returns:
            Path to generated report
        """
        # Load data
        data = self.load_timeseries(timeseries_path)
        revisions = data['revisions']

        if len(revisions) < 2:
            print("Need at least 2 revisions to interpret changes")
            return None

        # Prepare output
        if output_path is None:
            ts_path = Path(timeseries_path)
            output_path = ts_path.parent / "interpretation_report.md"

        report_lines = []
        report_lines.append("# Metric Interpretation Report")
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"\nRepository: {data.get('repo', data.get('repository', 'Unknown'))}")
        report_lines.append(f"\nAnalyzed {len(revisions)} revisions\n")
        report_lines.append("---\n")

        # Analyze each transition
        for i in range(len(revisions) - 1):
            rev_old = revisions[i + 1]  # Older revision (later in list)
            rev_new = revisions[i]      # Newer revision (earlier in list)

            report_lines.append(f"## Revision {i+1}: {rev_old['commit_date'][:10]} → {rev_new['commit_date'][:10]}")
            report_lines.append(f"\n**Commits:** `{rev_old['commit_hash'][:8]}` → `{rev_new['commit_hash'][:8]}`\n")

            # Get commit context
            commit_summary = self.commit_analyzer.get_summary_between_revisions(
                rev_old['commit_date'][:10],
                rev_new['commit_date'][:10],
                limit=30
            )

            if commit_summary['total_commits'] == 0:
                report_lines.append("No commits found in this period\n")
                continue

            report_lines.append(f"**{commit_summary['total_commits']} commits** in this period")
            report_lines.append(f"- Refactoring: {commit_summary['categories'].get('refactoring', 0)}")
            report_lines.append(f"- Bug fixes: {commit_summary['categories'].get('bugfix', 0)}")
            report_lines.append(f"- New features: {commit_summary['categories'].get('feature', 0)}\n")

            # Analyze key metrics
            metrics_to_check = ['propagation-cost', 'm-score', 'decoupling-level', 'independence-level']

            for metric in metrics_to_check:
                old_val = rev_old['metrics'].get(metric)
                new_val = rev_new['metrics'].get(metric)

                # Skip if either value is None or if both are zero
                if old_val is None or new_val is None:
                    continue
                if old_val == 0 and new_val == 0:
                    continue
                if old_val == new_val:
                    continue  # Skip unchanged metrics

                change = new_val - old_val
                direction = "📈" if change > 0 else "📉"

                report_lines.append(f"\n### {direction} {metric.title()}")
                report_lines.append(f"`{old_val:.2f}%` → `{new_val:.2f}%` ({change:+.2f}%)\n")

                # Get LLM interpretation
                print(f"Interpreting {metric} change...")
                interpretation = self.interpret_metric_change(
                    metric, old_val, new_val, commit_summary
                )
                report_lines.append(f"**Interpretation:**\n{interpretation}\n")

            # Hotspots for this period (file-level patterns)
            if commit_summary.get('hotspot_files'):
                report_lines.append("\n#### Hotspot files in this period (top 5)")
                for hf in commit_summary['hotspot_files'][:5]:
                    report_lines.append(f"- {hf['file']} (changes: {hf['changes']})")

            report_lines.append("\n---\n")

        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"\nReport generated: {output_path}")
        return str(output_path)

    def generate_tool_report(self, temporal_dir: str, output_path: str) -> str | None:
        """Aggregate tool logs from each revision and create a troubleshooting report.

        Looks for ./<temporal_dir>/<NN_*>/OutputData/tool_logs/DEPENDS_DV8_OUTPUT.json
        and summarizes non-zero return codes and stderr lines.
        """
        from pathlib import Path
        import json
        import re as _re
        td = Path(temporal_dir)
        if not td.exists():
            return None

        issues = []
        for rev_dir in sorted(td.glob("[0-9][0-9]_*")):
            log_path = rev_dir / "OutputData" / "tool_logs" / "DEPENDS_DV8_OUTPUT.json"
            if not log_path.is_file():
                continue
            try:
                steps = json.loads(log_path.read_text())
            except Exception:
                continue
            bad = []
            for s in steps:
                rc = s.get("rc", 0)
                stderr = (s.get("stderr") or "")
                if rc != 0 or _re.search(r"error|exception|failed", stderr, _re.I):
                    bad.append({
                        "step": s.get("step"),
                        "cmd": s.get("cmd"),
                        "rc": rc,
                        "stderr": stderr.strip()[:1000],
                    })
            if bad:
                issues.append({
                    "revision": rev_dir.name,
                    "problems": bad
                })

        if not issues:
            Path(output_path).write_text("# DV8/Depends Tool Report\n\nNo issues detected in tool logs.\n")
            print(f"Tool report generated: {output_path}")
            return output_path

        # Build a concise, actionable report and ask LLM for root causes
        summary_lines = ["# DV8/Depends Tool Report", ""]
        summary_lines.append("This report summarizes errors and anomalies observed while extracting dependencies and computing metrics.\n")
        for item in issues:
            summary_lines.append(f"## {item['revision']}")
            for p in item['problems']:
                summary_lines.append(f"- Step: {p['step']} (rc={p['rc']})")
                summary_lines.append(f"  Cmd: {p['cmd']}")
                if p['stderr']:
                    summary_lines.append(f"  Stderr: {p['stderr']}")
            summary_lines.append("")

        # Ask LLM for likely root causes and quick checks
        prompt = (
            "You are a DV8/Depends troubleshooting assistant. Given the steps and errors below, "
            "list the most likely root causes and quick checks to resolve them. Focus on missing source path, parser limitations, "
            "non-Java files, encoding issues, or DV8 license problems if relevant. Provide bullet points.\n\n" + "\n".join(summary_lines)
        )
        advice = self.query_ollama(prompt)
        summary_lines.insert(1, "## Likely Root Causes and Quick Checks\n\n" + advice + "\n")

        Path(output_path).write_text("\n".join(summary_lines))
        print(f"Tool report generated: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Interpret metric changes with git context")
    parser.add_argument('--repo', required=True, help='Path to git repository')
    parser.add_argument('--timeseries', required=True, help='Path to timeseries.json')
    parser.add_argument('--model', default='deepseek-r1:14b', help='Ollama model to use')
    parser.add_argument('--output', help='Output path for report')

    args = parser.parse_args()

    print(f"Interpreting metrics for {args.repo}")
    print(f"Using model: {args.model}\n")

    interpreter = MetricInterpreter(args.repo, args.model)
    report_path = interpreter.generate_report(args.timeseries, args.output)

    # Also produce a tool troubleshooting report next to the timeseries
    ts_dir = str(Path(args.timeseries).parent)
    tool_report_path = str(Path(ts_dir) / "depends_DV8_TOOL_report.md")
    try:
        interpreter.generate_tool_report(ts_dir, tool_report_path)
    except Exception as e:
        print(f"Tool log aggregation failed: {e}")

    if report_path:
        print(f"\nRead the report:")
        print(f"   cat '{report_path}'")


if __name__ == "__main__":
    main()
