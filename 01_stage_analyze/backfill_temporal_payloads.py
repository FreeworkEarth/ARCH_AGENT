"""
Backfill exact M-Score components and regenerate interpretation payloads
for an existing temporal analysis (without re-running the full DV8 pipeline).

Steps per revision:
  - Locate DSM/DRH JSON.
  - Compute exact M-Score with component rollups.
  - Write metrics/mscore_exact_components.json.
  - Re-run prepare_interpretation_payload.py with meta info from timeseries.json.
"""

import argparse
import json
import sys
import subprocess
import os
import shutil
from pathlib import Path
from collections import defaultdict, deque

from mscore_dv8_exact import calculate_mscore_dv8_exact


def _resolve_git_root(temporal_root: Path, repo_name: str | None) -> Path:
    """
    Best-effort: find a real git repo root for churn/commit inspection.

    Normal case: temporal_root lives under the git clone folder, so temporal_root.parent is the git repo.
    Special cases:
      - analysis_tag outputs: <repo>_java/<temporal_analysis...> (git clone is <repo>)
      - toy repos: TEST_AUTO/000_TOY_EXAMPLES/<repo>
    """
    cand = temporal_root.parent
    if (cand / ".git").exists():
        return cand

    # analysis_tag: <repo>_java or <repo>_python
    parent_name = cand.name
    for suffix in ("_java", "_python"):
        if parent_name.endswith(suffix):
            sib = cand.parent / parent_name[: -len(suffix)]
            if (sib / ".git").exists():
                return sib

    # if repo_name is known, try TEST_AUTO/000_TOY_EXAMPLES/<repo_name>
    if repo_name:
        test_auto_dir = Path(__file__).resolve().parents[1]
        toy = test_auto_dir / "000_TOY_EXAMPLES" / repo_name
        if (toy / ".git").exists():
            return toy
        repos = test_auto_dir / "REPOS" / repo_name
        if (repos / ".git").exists():
            return repos

    return cand


def compute_pc_from_dsm(dsm_json: Path) -> float:
    """
    Compute Propagation Cost (PC) from a DSM JSON.
    PC = average over nodes of reachable_count/(N-1) via transitive closure.
    """
    data = json.loads(dsm_json.read_text())
    n = len(data.get("variables", []))
    if n <= 1:
        return 0.0

    adj = defaultdict(list)
    for cell in data.get("cells", []):
        if cell.get("values"):
            adj[cell["src"]].append(cell["dest"])

    def reachable(start: int) -> int:
        seen = set()
        dq = deque([start])
        while dq:
            u = dq.popleft()
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    dq.append(v)
        seen.discard(start)
        return len(seen)

    total = 0.0
    denom = n - 1
    for i in range(n):
        total += reachable(i) / denom
    return total / n


def compute_dl_approx(drh_json: Path) -> dict:
    """
    Rough DL approximation using DRH modules:
    - balance heuristic based on module size variance.
    """
    try:
        drh = json.loads(drh_json.read_text())
    except Exception:
        return {}
    structure = drh.get("structure", [])
    modules = []
    for layer in structure:
        for mod in layer.get("nested", []):
            files = []
            stack = [mod]
            while stack:
                item = stack.pop()
                if item.get("@type") == "item":
                    files.append(item["name"])
                elif item.get("@type") == "group":
                    stack.extend(item.get("nested", []))
            if files:
                modules.append(files)
    if not modules:
        return {}
    sizes = [len(m) for m in modules]
    mean_sz = sum(sizes) / len(sizes)
    var = sum((s - mean_sz) ** 2 for s in sizes) / len(sizes)
    balance = 1.0 / (1.0 + var)
    return {"approx_balance": balance, "module_count": len(modules)}


def find_rev_dirs(root: Path):
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name[:2].isdigit()])


def rev_number_from_dir(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[0])
    except Exception:
        return -1


def write_mscore_components(output_dir: Path):
    """
    Find dv8-analysis-result/dsm under output_dir (or its subfolders),
    compute exact M-Score components, and write metrics/mscore_exact_components.json
    next to the other metric files.
    """
    dsm = None
    drh = None
    for candidate in output_dir.glob("**/dv8-analysis-result/dsm/structure-dsm.json"):
        dsm = candidate
        drh = candidate.parent / "drh-clustering.json"
        if drh.exists():
            break
    if not (dsm and drh and dsm.exists() and drh.exists()):
        return False

    res = calculate_mscore_dv8_exact(dsm, drh, include_details=True)
    # compute PC/DL approximations
    res["pc_from_dsm"] = compute_pc_from_dsm(dsm)
    res["dl_approx"] = compute_dl_approx(drh)

    # metrics dir is at OutputData/metrics
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    # Write with the new name (preferred) and legacy name for compatibility.
    out_path_new = metrics_dir / "mscore_from_dsm_drh_components.json"
    with open(out_path_new, "w") as f:
        json.dump(res, f, indent=2)
    out_path_legacy = metrics_dir / "mscore_exact_components.json"
    with open(out_path_legacy, "w") as f:
        json.dump(res, f, indent=2)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("temporal_root", help="Path to temporal_analysis_alltime_* folder")
    ap.add_argument("--meta-repo", default=None, help="Repository name for metadata")
    ap.add_argument(
        "--issue-map",
        default=None,
        help="Optional issue map JSON (JIRA/GitHub) to enable typed churn in interpretation payloads",
    )
    args = ap.parse_args()

    root = Path(args.temporal_root).resolve()
    ts_path = root / "timeseries.json"
    timeseries = json.load(ts_path.open()) if ts_path.exists() else {}
    revisions_meta = timeseries.get("revisions", [])
    repo_name = timeseries.get("repo") if isinstance(timeseries, dict) else None

    # Prep interpretation bundle roots
    interp_root = root / "INPUT_INTERPRETATION"
    plots_root = interp_root / "plots"
    plots_root.mkdir(parents=True, exist_ok=True)
    single_root = interp_root / "SINGLE_REVISION_ANALYSIS_DATA"
    single_root.mkdir(parents=True, exist_ok=True)

    # Resolve issue map path (explicit or best-effort autodiscovery).
    issue_map_path = None
    if args.issue_map:
        cand = Path(args.issue_map).expanduser()
        if not cand.is_absolute():
            cand = (root / cand).resolve()
        if cand.exists():
            issue_map_path = cand
    if not issue_map_path:
        for cand in [
            interp_root / "issue_stats" / "issue_map.json",
            interp_root / "issue_map.json",
            root / "issue_map.json",
        ]:
            if cand.exists():
                issue_map_path = cand
                break
    # Normalize location: prefer INPUT_INTERPRETATION/issue_stats/issue_map.json
    if issue_map_path:
        issue_stats_dir = interp_root / "issue_stats"
        issue_stats_dir.mkdir(parents=True, exist_ok=True)
        normalized = issue_stats_dir / "issue_map.json"
        if issue_map_path.resolve() != normalized.resolve():
            try:
                shutil.copy2(issue_map_path, normalized)
                issue_map_path = normalized
            except Exception:
                # Keep original path if copy fails (e.g., permissions).
                pass

    rev_dirs = find_rev_dirs(root)
    for idx, rev_dir in enumerate(rev_dirs):
        print(f"\n=== {rev_dir.name} ===")
        out_dir = rev_dir / "OutputData"
        mscore_done = write_mscore_components(out_dir)
        if not mscore_done:
            print("  ! Skipped (no DSM/DRH JSON found)")
            continue

        # meta info
        meta = revisions_meta[idx] if idx < len(revisions_meta) else {}
        commit = meta.get("commit_hash")
        date = meta.get("commit_date")
        name = f"{args.meta_repo or timeseries.get('repo','')}_{date.split(' ')[0] if date else rev_dir.name}"
        churn_range = None
        if commit and idx + 1 < len(revisions_meta):
            prev_commit = revisions_meta[idx + 1].get("commit_hash")
            if prev_commit:
                # Git range older..newer to scope churn to this interval.
                churn_range = f"{prev_commit}..{commit}"

        # re-run payload prep
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "prepare_interpretation_payload.py"),
            str(out_dir),
            "--repo-root",
            str(rev_dir),
            "--git-root",
            str(_resolve_git_root(root, repo_name)),
            "--meta-name",
            name,
        ]
        if commit:
            cmd += ["--meta-commit", commit]
        if date:
            cmd += ["--meta-date", date]
        if args.meta_repo:
            cmd += ["--meta-repo", args.meta_repo]
        if churn_range:
            cmd += ["--churn-range", churn_range]
        if issue_map_path:
            cmd += ["--issue-map", str(issue_map_path)]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("  ! Payload prep failed")

        # Mirror InputData/OutputData into interpretation bundle
        dest_rev = single_root / rev_dir.name
        for sub in ["InputData", "OutputData"]:
            src = rev_dir / sub
            if not src.exists():
                continue
            dest = dest_rev / sub
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest)

    # Copy aggregate artifacts useful for the interpreter (plots + timeseries)
    plots_src = root / "plots"
    if plots_src.exists():
        for p in plots_src.iterdir():
            if p.is_file():
                shutil.copy2(p, plots_root / p.name)
            elif p.is_dir():
                dest_dir = plots_root / p.name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.copytree(p, dest_dir)

    if ts_path.exists():
        shutil.copy2(ts_path, interp_root / "timeseries.json")

    # Build pairwise evidence graph diffs (matrix.json) for adjacent revisions.
    evidence_dir = interp_root / "EVIDENCE_GRAPH_DIFF"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_index = []
    for i in range(len(rev_dirs) - 1):
        new_dir = rev_dirs[i]
        old_dir = rev_dirs[i + 1]
        new_out = new_dir / "OutputData"
        old_out = old_dir / "OutputData"
        new_n = rev_number_from_dir(new_dir)
        old_n = rev_number_from_dir(old_dir)
        out_path = evidence_dir / f"evidence_graph_diff_new{new_n}_old{old_n}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "compute_evidence_graph_diff.py"),
            "--new-output",
            str(new_out),
            "--old-output",
            str(old_out),
            "--out",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True)
            evidence_index.append(
                {
                    "new_revision_number": new_n,
                    "old_revision_number": old_n,
                    "new_dir": new_dir.name,
                    "old_dir": old_dir.name,
                    "path": str(out_path.relative_to(interp_root)),
                }
            )
        except subprocess.CalledProcessError:
            print(f"  ! Evidence diff failed for {new_dir.name} -> {old_dir.name}")

    if evidence_index:
        (interp_root / "evidence_graph_diff_index.json").write_text(
            json.dumps(evidence_index, indent=2), encoding="utf-8"
        )

    # Deterministic bundle verification (best-effort; should not fail the backfill run).
    verifier = Path(__file__).resolve().parents[1] / "02_STAGE_INTERPRET" / "verify_interpretation_bundle.py"
    if verifier.exists():
        try:
            subprocess.run(
                [sys.executable, str(verifier), "--temporal-root", str(root)],
                check=False,
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
