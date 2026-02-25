#!/usr/bin/env python3
"""
Temporal Architecture Analysis - Multi-revision DV8 analysis with time-series visualization.

This module extends dv8_agent.py to analyze multiple Git revisions, track metrics over time,
and generate matplotlib visualizations showing architecture evolution.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt

# Import functions from dv8_agent.py
sys.path.insert(0, str(Path(__file__).parent))
try:
    from dv8_agent import (
        run,
        ensure_dirs,
        guess_project_name,
        resolve_dv8_console,
        ensure_dv8_license,
        compute_all_metrics,
        CONFIG_HOME,
        run_depends_via_dv8,
        convert_to_dsm,
        write_params,
        run_arch_report,
        run_additional_dv8_tasks,
    )
except ImportError as e:
    print(f"Error: Cannot import from dv8_agent.py: {e}")
    print("Make sure dv8_agent.py is in the same directory.")
    sys.exit(1)


# --- Configuration ---
# Use current working directory for all output
TEMPORAL_WORKSPACE = Path.cwd()
METRICS_CACHE = TEMPORAL_WORKSPACE / "metrics_cache.json"


# --- Git Operations ---

def get_commit_history(
    repo_path: Path,
    branch: str = "main",
    count: int = 10,
    spacing_mode: str = "simple",
    min_months_apart: int = 0,
    min_commits_apart: int = 1,
) -> List[Dict[str, str]]:
    """
    Get commits from a Git repository, with optional spacing modes.

    spacing_mode:
      - "simple": newest-first, take first N
      - "alltime": evenly spaced across full history (newest, oldest, interpolated)
      - "recent": enforce a minimum month gap between samples
      - "commits": pick commits with a fixed commit-count gap (min_commits_apart)

    Returns list of dicts with: hash, date, author, message, files_changed
    """
    repo_path = repo_path.resolve()

    # Try main, master, or HEAD
    for br in [branch, "main", "master", "HEAD"]:
        try:
            result = subprocess.run(
                ["git", "log", f"{br}", f"-{count}", "--pretty=format:%H|%ai|%an|%s"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            break
        except subprocess.CalledProcessError:
            continue
    else:
        raise RuntimeError(f"Could not retrieve git log from {repo_path}. Is it a git repo?")

    commits = []
    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        parts = line.split('|', 3)
        if len(parts) == 4:
            commit_hash, date_str, author, message = parts
            commits.append({
                'hash': commit_hash,
                'short_hash': commit_hash[:8],
                'date': date_str,
                'author': author,
                'message': message.strip()
            })

    # Spacing strategies
    if spacing_mode in ("alltime", "uniform"):
        if len(commits) <= count:
            return commits
        selected = [commits[0]]  # newest
        if count > 2:
            total = len(commits)
            step = (total - 1) / (count - 1)
            for i in range(1, count - 1):
                idx = int(i * step)
                selected.append(commits[idx])
        if count > 1:
            selected.append(commits[-1])  # oldest
        return selected

    elif spacing_mode == "smart":
        # Select commits with the most files changed — captures real architectural movement.
        # Uses git log --numstat to count added+removed lines per commit across all files.
        result2 = subprocess.run(
            ["git", "log", branch, "--pretty=format:%H", "--numstat"],
            cwd=repo_path, capture_output=True, text=True
        )
        change_counts: Dict[str, int] = {}
        current_hash = None
        for line in result2.stdout.split('\n'):
            line = line.strip()
            if not line:
                continue
            if len(line) == 40 and all(c in '0123456789abcdef' for c in line):
                current_hash = line
                change_counts[current_hash] = 0
            elif current_hash and '\t' in line:
                parts = line.split('\t')
                try:
                    added = int(parts[0]) if parts[0] != '-' else 0
                    removed = int(parts[1]) if parts[1] != '-' else 0
                    change_counts[current_hash] += added + removed
                except (ValueError, IndexError):
                    pass
        # Rank by total lines changed, take top N, keep newest-first order
        ranked = sorted(commits, key=lambda c: change_counts.get(c['hash'], 0), reverse=True)
        selected = ranked[:count]
        selected.sort(key=lambda c: c['date'], reverse=True)  # newest-first like other modes
        return selected

    elif spacing_mode == "recent":
        min_days = max(0, min_months_apart) * 30
        selected = []
        for commit in commits:
            time_ok = True
            if selected and min_days > 0:
                try:
                    last_date = datetime.fromisoformat(selected[-1]['date'].split()[0])
                    curr_date = datetime.fromisoformat(commit['date'].split()[0])
                    days_diff = abs((last_date - curr_date).days)
                    time_ok = days_diff >= min_days
                except Exception:
                    time_ok = True
            if time_ok:
                selected.append(commit)
            if len(selected) >= count:
                break
        if len(selected) < count and commits:
            if commits[-1] not in selected:
                selected.append(commits[-1])
        return selected

    elif spacing_mode == "commits":
        selected = []
        gap = max(1, min_commits_apart)
        for i_gap in range(count):
            idx = i_gap * gap
            if idx < len(commits):
                selected.append(commits[idx])
            else:
                break
        if len(selected) < count and commits:
            if commits[-1] not in selected:
                selected.append(commits[-1])
        return selected

    # simple fallback: newest first
    return commits[:count]


def checkout_commit(repo_path: Path, commit_hash: str, commit_date: str, workspace: Path, revision_number: int) -> Path:
    """
    Checkout a specific commit to a new directory with numbered naming.

    Args:
        repo_path: Path to the repository
        commit_hash: Full commit hash
        commit_date: Commit date string (YYYY-MM-DD format)
        workspace: Parent workspace directory
        revision_number: Sequential number (1=newest, 2=second newest, etc.)

    Returns path to the checked-out directory.
    """
    repo_name = repo_path.name

    # Format date as DDMMYYYY
    # commit_date format from git: "2025-10-08 12:34:11 +0000"
    try:
        from datetime import datetime
        # Extract just the date part (YYYY-MM-DD)
        date_part = commit_date.split()[0]  # Get "2025-10-08"
        date_obj = datetime.strptime(date_part, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%d%m%Y")
    except Exception as e:
        # Fallback if date parsing fails - use first 8 chars of hash
        formatted_date = commit_hash[:8]

    # Create directory name: 01_pdfbox_08122024
    checkout_dir = workspace / f"{revision_number:02d}_{repo_name}_{formatted_date}"

    if checkout_dir.exists():
        print(f"  Checkout already exists: {checkout_dir.name}")
        return checkout_dir

    print(f"  Checking out to {checkout_dir.name}...")

    # Clone and checkout specific commit
    run(
        ["git", "clone", str(repo_path), str(checkout_dir)],
        display=f"git clone {repo_path} {checkout_dir.name}"
    )

    run(
        ["git", "checkout", commit_hash],
        cwd=checkout_dir,
        display=f"git checkout {commit_hash[:8]}"
    )

    return checkout_dir


def _is_empty_revision(metrics: dict) -> bool:
    """Return True if a revision produced no usable DV8 metrics (empty/trivial repo state)."""
    keys = ("m-score", "propagation-cost", "decoupling-level", "independence-level")
    vals = [metrics.get(k) for k in keys]
    if all(v is None for v in vals):
        return True
    numeric = [v for v in vals if isinstance(v, (int, float))]
    if numeric and all(v == 0.0 for v in numeric):
        return True
    return False


# --- DV8 Analysis ---

def analyze_revision(
    revision_path: Path,
    dv8_console: Path,
    env: dict,
    source_path: Optional[str] = None,
    full_report: bool = False,
) -> Dict[str, Any]:
    """
    Run complete DV8 analysis pipeline on a single revision and extract all metrics.

    Pipeline: Source → Depends → DSM → Metrics

    Returns dict with metric values or None on failure.
    """
    print(f"\n  Analyzing {revision_path.name}...")

    # Determine source root
    source_root = None
    if source_path:
        p = Path(source_path)
        if not p.is_absolute():
            p = revision_path / p
        if p.exists():
            source_root = p

    if source_root is None:
        # Try common patterns
        for rel in ("src/main/java", "src", "SourceCode"):
            cand = revision_path / rel
            if cand.exists():
                source_root = cand
                break

    if source_root is None:
        source_root = revision_path

    # Ensure directory structure
    input_data = revision_path / "InputData"
    depends_output = input_data / "DependsOutput" / "json"
    output_data = revision_path / "OutputData"
    ensure_dirs(input_data, depends_output, output_data)

    project_name = guess_project_name(revision_path)
    basename = f"{project_name}-depends"

    try:
        # Step 1: Run depends (dependency extraction)
        json_dep, mapping = run_depends_via_dv8(dv8_console, source_root, depends_output, basename, env)

        # Step 2: Convert to DSM
        dsm_path = output_data / "repo.dv8-dsm"
        convert_to_dsm(dv8_console, json_dep, mapping, dsm_path, env)

        # Step 3: Compute all metrics from DSM
        all_metrics = compute_all_metrics(dv8_console, dsm_path, output_data, env)

        # Step 4: Optional full architecture report (anti-patterns, hotspots, DRH)
        if full_report:
            params_dir = input_data
            params_dir.mkdir(parents=True, exist_ok=True)

            # First attempt: hotspot on, file stat off
            params_path = params_dir / "archreport.properties"
            write_params(
                params_path,
                project_name,
                dsm_path,
                output_data,
                run_file_stat=False,
                run_hotspot=True,
                run_archissue=True,
                run_archroot=True,
            )
            try:
                run_arch_report(dv8_console, params_path, revision_path, env)
            except SystemExit:
                # Second attempt: hotspot off
                params_path2 = params_dir / "archreport.nohotspot.properties"
                write_params(
                    params_path2,
                    project_name,
                    dsm_path,
                    output_data,
                    run_file_stat=False,
                    run_hotspot=False,
                    run_archissue=True,
                    run_archroot=True,
                    run_metrics=True,
                    run_clustering=False,
                    run_report_doc=True,
                )
                try:
                    run_arch_report(dv8_console, params_path2, revision_path, env)
                except SystemExit:
                    # Third attempt: arch-issue only
                    params_path3 = params_dir / "archreport.archissue_only.properties"
                    write_params(
                        params_path3,
                        project_name,
                        dsm_path,
                        output_data,
                        run_file_stat=False,
                        run_hotspot=False,
                        run_archissue=True,
                        run_archroot=False,
                        run_metrics=False,
                        run_clustering=False,
                        run_report_doc=True,
                    )
                    try:
                        run_arch_report(dv8_console, params_path3, revision_path, env)
                    except SystemExit:
                        pass

            # Also run direct DV8 tasks (anti-patterns, debt cost, hotspots, DRH, matrix export)
            run_additional_dv8_tasks(dv8_console, dsm_path, output_data, env)

        # Extract metric values
        metrics = {}
        metrics_dir = output_data / "metrics"

        for metric_name in ["propagation-cost", "m-score", "decoupling-level", "independence-level"]:
            metric_file = metrics_dir / f"{metric_name}.json"
            if metric_file.exists():
                with open(metric_file) as f:
                    data = json.load(f)
                    # Extract numeric value
                    if metric_name == "propagation-cost":
                        val = data.get("propagationCost")
                    elif metric_name == "m-score":
                        val = data.get("mScore")
                    elif metric_name == "decoupling-level":
                        val = data.get("decouplingLevel")
                    elif metric_name == "independence-level":
                        val = data.get("independenceLevel")
                    else:
                        val = data.get("value")

                    # Parse percentage strings
                    if isinstance(val, str) and val.endswith('%'):
                        val = float(val.rstrip('%'))
                    elif isinstance(val, (int, float)):
                        val = float(val)
                    else:
                        val = None

                    metrics[metric_name] = val
            else:
                metrics[metric_name] = None

        # --- Optional: export DSM/DRH to JSON and run exact M-Score + DRH stats ---
        try:
            # Locate DV8 analysis result (if compute_all_metrics produced it)
            arch_root = output_data / "Architecture-analysis-result"
            dv8_dsm = None
            dv8_drh = None
            if arch_root.exists():
                # Expect a single project folder under Architecture-analysis-result
                for proj_dir in arch_root.iterdir():
                    cand_dsm = proj_dir / "dv8-analysis-result" / "dsm"
                    if cand_dsm.exists():
                        dsm_bin = cand_dsm / "structure-dsm.dv8-dsm"
                        drh_bin = cand_dsm / "drh-clustering.dv8-clsx"
                        if dsm_bin.exists() and drh_bin.exists():
                            dv8_dsm = dsm_bin
                            dv8_drh = drh_bin
                            break
            if dv8_dsm and dv8_drh:
                # Export matrix and cluster to JSON
                dsm_json = dv8_dsm.with_suffix(".json")
                drh_json = dv8_drh.with_suffix(".json")
                if not dsm_json.exists():
                    subprocess.run(
                        [dv8_console, "core:export-matrix", "-outputFile", str(dsm_json), str(dv8_dsm)],
                        cwd=output_data,
                        env=env,
                        check=False,
                    )
                if not drh_json.exists():
                    subprocess.run(
                        [dv8_console, "core:export-cluster", "-outputFile", str(drh_json), str(dv8_drh)],
                        cwd=output_data,
                        env=env,
                        check=False,
                    )
                # Exact M-Score on exported JSON (with component rollups)
                try:
                    from mscore_dv8_exact import calculate_mscore_dv8_exact
                    mscore_result = calculate_mscore_dv8_exact(
                        dsm_json, drh_json, file_filter=None, include_details=True
                    )
                    mscore_components_path = output_data / "metrics" / "mscore_exact_components.json"
                    mscore_components_path.write_text(json.dumps(mscore_result, indent=2))
                    # Add to metrics dict for timeseries
                    metrics["m-score-exact"] = mscore_result.get("mscore_percentage")
                    comp = mscore_result.get("component_rollup", {})
                    metrics["m-score-components"] = {
                        "size_factor_total": comp.get("size_factor_total"),
                        "avg_clddf": comp.get("avg_clddf"),
                        "avg_imcf": comp.get("avg_imcf"),
                        "module_count": comp.get("module_count"),
                    }
                except Exception as exc:
                    print(f"    Warning: exact M-Score computation failed: {exc}")

                # Simple DRH stats (layers, modules per layer)
                try:
                    drh_data = json.load(open(drh_json))
                    layers = drh_data.get("structure", [])
                    per_layer = [len(l.get("nested", [])) for l in layers if l.get("@type") == "group"]
                    drh_stats = {
                        "num_layers": len(per_layer),
                        "modules_per_layer": per_layer,
                    }
                    stats_path = output_data / "metrics" / "drh_stats.json"
                    stats_path.write_text(json.dumps(drh_stats, indent=2))
                except Exception:
                    pass

            # Note: compute_all_metrics runs DV8 tasks; for <=10 revisions we assume arch report/anti-patterns are enabled.
        except Exception as e:
            print(f"    Warning: post-processing export/mscore failed in {revision_path.name}: {e}")

        return metrics

    except Exception as e:
        print(f"    ERROR analyzing {revision_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "propagation-cost": None,
            "m-score": None,
            "decoupling-level": None,
            "independence-level": None,
            "error": str(e)
        }


# --- Time Series Data Management ---

def save_timeseries_data(repo_name: str, data: List[Dict[str, Any]]) -> Path:
    """Save time-series data to JSON file."""
    ensure_dirs(TEMPORAL_WORKSPACE)

    output_file = TEMPORAL_WORKSPACE / f"{repo_name}_timeseries.json"
    with open(output_file, 'w') as f:
        json.dump({
            'repo': repo_name,
            'timestamp': datetime.now().isoformat(),
            'revisions': data
        }, f, indent=2)

    print(f"\nTime-series data saved to: {output_file}")
    return output_file


def load_timeseries_data(repo_name: str) -> Optional[List[Dict[str, Any]]]:
    """Load existing time-series data if available."""
    data_file = TEMPORAL_WORKSPACE / f"{repo_name}_timeseries.json"

    if not data_file.exists():
        return None

    with open(data_file) as f:
        data = json.load(f)

    return data.get('revisions', [])


# --- Main Temporal Analysis ---

def run_temporal_analysis(
    repo_path: str,
    revision_count: int = 10,
    branch: str = "main",
    source_path: Optional[str] = None,
    dv8_console_path: Optional[str] = None,
    force: bool = False,
    spacing_mode: str = "simple",
    min_months_apart: int = 0,
    min_commits_apart: int = 1,
) -> Path:
    """
    Run temporal analysis on a repository.

    Args:
        repo_path: Path to git repository
        revision_count: Number of revisions to analyze
        branch: Git branch name
        source_path: Optional source code subdirectory
        dv8_console_path: Optional explicit path to dv8-console
        force: Force re-analysis even if cached data exists

    Returns:
        Path to saved time-series JSON file
    """
    repo = Path(repo_path).expanduser().resolve()

    if not repo.exists():
        raise FileNotFoundError(f"Repository not found: {repo}")

    if not (repo / ".git").exists():
        raise RuntimeError(f"Not a git repository: {repo}")

    repo_name = repo.name

    # Check for cached data
    if not force:
        cached = load_timeseries_data(repo_name)
        if cached:
            print(f"Found cached temporal analysis for {repo_name}")
            ans = input("Use cached data? [Y/n]: ").strip().lower()
            if ans not in ('n', 'no'):
                print("Using cached data. Use --force to re-analyze.")
                return TEMPORAL_WORKSPACE / f"{repo_name}_timeseries.json"

    print(f"\n{'='*60}")
    print(f"TEMPORAL ANALYSIS: {repo_name}")
    print(f"  Revisions: {revision_count}")
    print(f"  Branch: {branch}")
    print(f"{'='*60}\n")

    # Get commit history
    print(f"Retrieving last {revision_count} commits...")
    commits = get_commit_history(
        repo,
        branch,
        revision_count,
        spacing_mode=spacing_mode,
        min_months_apart=min_months_apart,
        min_commits_apart=min_commits_apart,
    )
    print(f"Found {len(commits)} commits to analyze\n")

    # Setup DV8
    dv8_console = resolve_dv8_console(dv8_console_path)
    env = os.environ.copy()
    env["PATH"] = f"{dv8_console.parent}{os.pathsep}{env.get('PATH', '')}"
    ensure_dv8_license(dv8_console, env, None, None)  # Use cached credentials or prompt

    # Create workspace - directly under repo name
    workspace = TEMPORAL_WORKSPACE / repo_name
    ensure_dirs(workspace)

    # For small revision sets, run the full arch report/anti-pattern pipeline
    do_full_report = revision_count <= 10

    # Analyze each revision
    timeseries_data = []
    prep_script = Path(__file__).parent / "prepare_interpretation_payload.py"

    for i, commit in enumerate(commits, 1):
        print(f"\n[{i}/{len(commits)}] Commit {commit['short_hash']}: {commit['message'][:60]}")
        print(f"  Date: {commit['date']}")
        print(f"  Author: {commit['author']}")

        # Checkout revision with numbered naming
        revision_path = checkout_commit(repo, commit['hash'], commit['date'], workspace, i)

        # Analyze with DV8
        metrics = analyze_revision(revision_path, dv8_console, env, source_path, full_report=do_full_report)

        # Skip empty/trivial revisions (e.g. cvs2svn init commits, empty repo state)
        if _is_empty_revision(metrics):
            print(f"  ⚠  Skipping commit {commit['short_hash']} ({commit['date'][:10]}): "
                  f"all metrics are None/zero — likely an empty or trivial repository state.")
            continue

        # Store results
        result = {
            **commit,
            'metrics': metrics
        }
        timeseries_data.append(result)

        # Prepare interpretation payload (metrics, anti-patterns, churn) for this revision
        output_data_dir = revision_path / "OutputData"
        if output_data_dir.exists() and prep_script.exists():
            try:
                subprocess.run(
                    [
                        sys.executable,
                        str(prep_script),
                        str(output_data_dir),
                        "--repo-root",
                        str(revision_path),
                        "--meta-name",
                        revision_path.name,
                        "--meta-commit",
                        commit.get("hash", ""),
                        "--meta-date",
                        commit.get("date", ""),
                        "--meta-repo",
                        repo_name,
                    ],
                    cwd=output_data_dir,
                    check=False,
                )
            except Exception as exc:
                print(f"    Warning: interpretation payload prep failed: {exc}")

        # Print metrics
        print(f"  Metrics:")
        for name, value in metrics.items():
            if name != "error":
                print(f"    {name}: {value}")

        if "error" in metrics:
            print(f"    ERROR: {metrics['error']}")

    # Save results
    output_file = save_timeseries_data(repo_name, timeseries_data)

    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"  Analyzed: {len(timeseries_data)} revisions")
    print(f"  Output: {output_file}")
    print(f"{'='*60}\n")

    # Plot M-Score components over time if available
    try:
        dates = []
        mscore = []
        sf_total = []
        clddf = []
        imcf = []
        for rev in timeseries_data:
            dates.append(rev.get("date", rev.get("short_hash", "")))
            mscore.append(rev.get("metrics", {}).get("m-score-exact") or rev.get("metrics", {}).get("m-score"))
            comp = rev.get("metrics", {}).get("m-score-components", {})
            sf_total.append(comp.get("size_factor_total"))
            clddf.append(comp.get("avg_clddf"))
            imcf.append(comp.get("avg_imcf"))

        if len(dates) > 1 and any(mscore):
            plt.figure(figsize=(10, 6))
            plt.plot(dates, mscore, label="M-Score exact/overall", marker="o")
            if any(sf_total):
                plt.plot(dates, sf_total, label="Size factor total", linestyle="--", marker="x")
            if any(clddf):
                plt.plot(dates, clddf, label="Avg CLDDF (1 - cross penalty)", linestyle="--", marker="s")
            if any(imcf):
                plt.plot(dates, imcf, label="Avg IMCF (1 - internal penalty)", linestyle="--", marker="d")
            plt.xticks(rotation=45, ha="right")
            plt.xlabel("Revision (date)")
            plt.ylabel("Score")
            plt.title(f"{repo_name}: M-Score components over time")
            plt.legend()
            plt.tight_layout()
            plot_dir = workspace / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plot_dir / "mscore_components_over_time.png"
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"Saved M-Score component plot: {plot_path}")
    except Exception as exc:
        print(f"Warning: plotting M-Score components failed: {exc}")

    return output_file


# --- CLI ---

def parse_args():
    parser = argparse.ArgumentParser(
        description="Temporal architecture analysis with DV8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze last 10 commits
  python3 temporal_analyzer.py --repo ./pdfbox --count 10

  # Analyze last 20 commits from develop branch
  python3 temporal_analyzer.py --repo /path/to/repo --count 20 --branch develop

  # Force re-analysis (ignore cache)
  python3 temporal_analyzer.py --repo ./myproject --count 15 --force

  # Specify source path
  python3 temporal_analyzer.py --repo ./java-project --source-path src/main/java
        """
    )

    parser.add_argument(
        "--repo",
        required=True,
        help="Path to git repository"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of revisions to analyze (default: 10)"
    )
    parser.add_argument(
        "--mode",
        default="simple",
        choices=["simple", "alltime", "recent", "commits"],
        help="Commit spacing mode: simple (newest N), alltime (evenly spaced), recent (min months apart), commits (fixed commit gap)"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=0,
        help="Minimum months apart for --mode recent (default: 0)"
    )
    parser.add_argument(
        "--commit-gap",
        type=int,
        default=1,
        help="Commit gap for --mode commits (default: 1)"
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Git branch name (default: main)"
    )
    parser.add_argument(
        "--source-path",
        help="Source code subdirectory (e.g., src/main/java)"
    )
    parser.add_argument(
        "--dv8-console",
        help="Explicit path to dv8-console"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-analysis (ignore cached data)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots after analysis (requires metric_plotter.py)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        output_file = run_temporal_analysis(
            repo_path=args.repo,
            revision_count=args.count,
            branch=args.branch,
            source_path=args.source_path,
            dv8_console_path=args.dv8_console,
            force=args.force,
            spacing_mode=args.mode,
            min_months_apart=args.months,
            min_commits_apart=args.commit_gap,
        )

        if args.plot:
            print("\nGenerating plots...")
            try:
                import metric_plotter
                metric_plotter.plot_timeseries(output_file)
            except ImportError:
                print("ERROR: metric_plotter.py not found. Install it to generate plots.")
                print("Run without --plot flag to skip plotting.")

        return 0

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        return 130
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
