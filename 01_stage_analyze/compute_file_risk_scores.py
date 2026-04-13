#!/usr/bin/env python3
"""
compute_file_risk_scores.py
===========================
Post-processing script for the ARCH_AGENT temporal pipeline.

Reads a completed INPUT_INTERPRETATION/ folder and produces:
  - file_risk_scores.json   (ranked per-file signal table + composite risk score)
  - file_risk_scores.csv    (same data, spreadsheet-friendly)

Usage:
    python compute_file_risk_scores.py <INPUT_INTERPRETATION_DIR> [options]

Options:
    --git-root PATH      Path to the git repo (auto-detected from timeseries meta if omitted)
    --weights JSON       JSON dict overriding default signal weights, e.g.
                         '{"bug_churn":0.25,"total_churn":0.05,"anti_pattern":0.25,
                           "hotspot_fanin":0.2,"scc_membership":0.15,"co_change":0.1}'
    --co-change-threshold INT  Min co-occurrences to count a pair as coupled (default: 3)
    --top-n INT          Number of files to include in output (default: all)
    --verbose            Print progress messages

Input folder structure expected (produced by backfill_temporal_payloads.py):
    INPUT_INTERPRETATION/
        timeseries.json
        SINGLE_REVISION_ANALYSIS_DATA/
            <rev_name>/OutputData/
                interpretation_payload.json
                dsm/matrix.json
        EVIDENCE_GRAPH_DIFF/
            evidence_graph_diff_new<N>_old<M>.json

Output:
    INPUT_INTERPRETATION/file_risk_scores.json
    INPUT_INTERPRETATION/file_risk_scores.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_QUERY_DIR = Path(__file__).resolve().parent.parent / "03_stage_query"
if str(_QUERY_DIR) not in sys.path:
    sys.path.append(str(_QUERY_DIR))
try:
    from llm_backend import LLMBackend
except Exception:
    LLMBackend = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "bug_churn": 0.25,
    "total_churn": 0.05,
    "anti_pattern": 0.25,
    "hotspot_fanin": 0.20,
    "scc_membership": 0.15,
    "co_change": 0.10,
}
CO_CHANGE_THRESHOLD = 3  # minimum co-occurrences to count as coupled

# Keep aligned with dv8_agent.py deterministic prod scope exclusions.
PROD_EXCLUDE_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "build",
    "dist",
    "doc",
    "docs",
    "out",
    "test",
    "tests",
    "testing",
    "target",
    "node_modules",
}

SOURCE_FILE_EXTENSIONS = {
    ".java",
    ".py",
    ".kt",
    ".scala",
    ".groovy",
    ".js",
    ".ts",
    ".cs",
    ".cpp",
    ".c",
    ".go",
    ".rb",
}

ANTI_PATTERN_ALIASES = {
    "clique": "clique",
    "packagecycle": "package-cycle",
    "package-cycle": "package-cycle",
    "unhealthyinheritance": "unhealthy-inheritance",
    "unhealthy-inheritance": "unhealthy-inheritance",
    "unstableinterface": "unstable-interface",
    "unstable-interface": "unstable-interface",
    "crossing": "crossing",
    "modularityviolation": "modularity-violation",
    "modularity-violation": "modularity-violation",
    "modularity-violation-group": "modularity-violation",
}

HISTORY_ANTI_PATTERN_TYPES = {
    "unstable-interface",
    "modularity-violation",
    "crossing",
}

# ---------------------------------------------------------------------------
# .dv8-clsx parser (DV8 anti-pattern instance file membership)
# ---------------------------------------------------------------------------


def _parse_dv8_clsx(path: Path) -> List[str]:
    """
    Extract file paths from a DV8 .dv8-clsx clustering file.

    Format: custom header (b'dv8clust') + gzip-compressed binary body.
    The body contains length-prefixed UTF-8 strings of the form:
        \\x01\\x00\\xNN<string of NN bytes>
    where strings ending in '.java', '.py', '.kt', etc. are file paths.

    Returns a list of normalised file path strings (may be empty on parse error).
    """
    try:
        raw = path.read_bytes()
        gz_start = raw.find(b"\x1f\x8b")
        if gz_start < 0:
            return []
        content = gzip.decompress(raw[gz_start:])
    except Exception:
        return []

    paths: List[str] = []
    i = 0
    while i < len(content) - 3:
        if content[i] == 0x01 and content[i + 1] == 0x00:
            length = content[i + 2]
            end = i + 3 + length
            if end <= len(content):
                try:
                    s = content[i + 3:end].decode("utf-8", errors="replace").strip()
                    # Accept strings that look like source file paths
                    if s and ("." in s) and not s.startswith("\x00"):
                        ext = s.rsplit(".", 1)[-1].lower()
                        if ext in ("java", "py", "kt", "scala", "groovy", "js", "ts", "cs", "cpp", "c", "go", "rb"):
                            paths.append(_normalise_path(s))
                except Exception:
                    pass
                i = end
                continue
        i += 1
    return paths


def _canonicalise_antipattern_type(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return raw
    key = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    key = key.replace("--", "-")
    return ANTI_PATTERN_ALIASES.get(key, key)


def _collect_antipattern_membership(output_dir: Path) -> Dict[str, Set[str]]:
    """
    Parse all .dv8-clsx files under output_dir/arch-issue/ and return
    file → set(anti_pattern_type_names).

    Checks both arch-issue/ (new pipeline layout) and the nested
    dv8-analysis-result/anti-pattern/anti-pattern-instances/ layout.
    """
    membership: Dict[str, Set[str]] = collections.defaultdict(set)

    def _scan_dir(root: Path, ap_type: str) -> None:
        ap_type = _canonicalise_antipattern_type(ap_type)
        for clsx_file in root.rglob("*.dv8-clsx"):
            for fpath in _parse_dv8_clsx(clsx_file):
                if fpath:
                    membership[fpath].add(ap_type)

    # Layout 1: arch-issue/<ap-type>/<instance>/*.dv8-clsx
    arch_issue = output_dir / "arch-issue"
    if arch_issue.is_dir():
        for ap_dir in arch_issue.iterdir():
            if ap_dir.is_dir():
                ap_type = ap_dir.name  # e.g. "clique", "unhealthy-inheritance"
                _scan_dir(ap_dir, ap_type)

    # Layout 2: dv8-analysis-result/anti-pattern/anti-pattern-instances/<ap-type>/
    nested = output_dir.glob("dv8-analysis-result/anti-pattern/anti-pattern-instances")
    for instances_dir in nested:
        if instances_dir.is_dir():
            for ap_dir in instances_dir.iterdir():
                if ap_dir.is_dir():
                    _scan_dir(ap_dir, ap_dir.name)

    return membership


def _collect_antipattern_stats(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse DV8 anti-pattern instance files and return richer per-file stats:
      file -> {
        "types": set[str],
        "instance_count": int,
        "type_counts": dict[str, int],
      }
    """
    stats: Dict[str, Dict[str, Any]] = collections.defaultdict(
        lambda: {
            "types": set(),
            "instance_count": 0,
            "type_counts": collections.defaultdict(int),
        }
    )

    def _scan_dir(root: Path, ap_type: str) -> None:
        ap_type = _canonicalise_antipattern_type(ap_type)
        for clsx_file in root.rglob("*.dv8-clsx"):
            files = set(_parse_dv8_clsx(clsx_file))
            for fpath in files:
                if not fpath:
                    continue
                stats[fpath]["types"].add(ap_type)
                stats[fpath]["instance_count"] += 1
                stats[fpath]["type_counts"][ap_type] += 1

    arch_issue = output_dir / "arch-issue"
    if arch_issue.is_dir():
        for ap_dir in arch_issue.iterdir():
            if ap_dir.is_dir():
                _scan_dir(ap_dir, ap_dir.name)

    nested = output_dir.glob("dv8-analysis-result/anti-pattern/anti-pattern-instances")
    for instances_dir in nested:
        if instances_dir.is_dir():
            for ap_dir in instances_dir.iterdir():
                if ap_dir.is_dir():
                    _scan_dir(ap_dir, ap_dir.name)

    return stats


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _iter_revision_payloads(interp_root: Path):
    """Yield (rev_dir_name, payload_dict) for each revision payload found."""
    rev_data_dir = interp_root / "SINGLE_REVISION_ANALYSIS_DATA"
    if not rev_data_dir.is_dir():
        return
    for rev_dir in sorted(rev_data_dir.iterdir()):
        if not rev_dir.is_dir():
            continue
        payload_path = rev_dir / "OutputData" / "interpretation_payload.json"
        if payload_path.exists():
            yield rev_dir.name, _load_json(payload_path)


def _collect_structural_signals(
    interp_root: Path,
    verbose: bool,
) -> Tuple[
    Dict[str, int],       # anti_pattern_total  file → count-of-revisions with anti-pattern
    Dict[str, float],     # hotspot_fanin_sum   file → sum of FanIn across revisions
    Dict[str, int],       # rev_presence        file → count of revisions present
    Dict[str, int],       # total_churn         file → total churn lines
    Dict[str, int],       # bug_churn           file → bug-linked churn lines
    Dict[str, set],       # anti_patterns_seen  file → set of pattern type names
    Dict[str, int],       # anti_pattern_revision_count
    Dict[str, int],       # anti_pattern_instance_load
    Dict[str, Dict[str, int]],  # anti_pattern_type_counts
]:
    anti_pattern_total: Dict[str, int] = collections.defaultdict(int)
    hotspot_fanin_sum: Dict[str, float] = collections.defaultdict(float)
    rev_presence: Dict[str, int] = collections.defaultdict(int)
    total_churn: Dict[str, int] = collections.defaultdict(int)
    bug_churn: Dict[str, int] = collections.defaultdict(int)
    anti_patterns_seen: Dict[str, set] = collections.defaultdict(set)
    anti_pattern_revision_count: Dict[str, int] = collections.defaultdict(int)
    anti_pattern_instance_load: Dict[str, int] = collections.defaultdict(int)
    anti_pattern_type_counts: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )

    for rev_name, payload in _iter_revision_payloads(interp_root):
        if verbose:
            print(f"  [structural] {rev_name}", flush=True)

        # --- structural hotspots (fan-in) ---
        hotspots = payload.get("structural_hotspots", {})
        rows = hotspots.get("rows", [])
        for row in rows:
            fname = _normalise_path(row.get("Filename", ""))
            if not fname:
                continue
            try:
                fanin = float(row.get("FanIn", 0) or 0)
            except (ValueError, TypeError):
                fanin = 0.0
            hotspot_fanin_sum[fname] += fanin
            rev_presence[fname] += 1

        # --- anti-pattern per-file membership from .dv8-clsx files ---
        # These are the ground-truth file lists for each anti-pattern instance.
        rev_output_dir = interp_root / "SINGLE_REVISION_ANALYSIS_DATA" / rev_name / "OutputData"
        clsx_stats: Dict[str, Dict[str, Any]] = {}
        if rev_output_dir.is_dir():
            clsx_stats = _collect_antipattern_stats(rev_output_dir)

        if clsx_stats:
            # Use real per-file anti-pattern data
            for fname, ap_info in clsx_stats.items():
                ap_types = ap_info["types"]
                anti_pattern_total[fname] += len(ap_types)
                anti_patterns_seen[fname].update(ap_types)
                anti_pattern_revision_count[fname] += 1
                anti_pattern_instance_load[fname] += int(ap_info["instance_count"])
                for ap_type, count in dict(ap_info["type_counts"]).items():
                    anti_pattern_type_counts[fname][ap_type] += int(count)
                rev_presence[fname] = rev_presence.get(fname, 0)
        else:
            # Fallback: dangerous_files table from DV8 HTML summary
            dangerous = payload.get("dangerous_files", {})
            drows = dangerous.get("rows", [])
            for row in drows:
                fname = _normalise_path(row.get("Filename", ""))
                if not fname:
                    continue
                anti_pattern_total[fname] += 1
                anti_patterns_seen[fname].add("DV8-DangerousFile")
                anti_pattern_revision_count[fname] += 1
                anti_pattern_instance_load[fname] += 1
                anti_pattern_type_counts[fname]["DV8-DangerousFile"] += 1
                if fname not in rev_presence:
                    rev_presence[fname] += 1

        # --- churn ---
        for entry in payload.get("churn_top", []):
            try:
                fname, lines = _normalise_path(entry[0]), int(entry[1])
            except (IndexError, ValueError, TypeError):
                continue
            if fname:
                total_churn[fname] += lines
                rev_presence[fname] = rev_presence.get(fname, 0)  # ensure key exists

        # --- bug-linked churn ---
        itc = payload.get("issue_typed_churn", {})
        # typed churn: look for "bug" / "bugfix" / "hotfix" key in churn_top
        bug_churn_top = {}
        if isinstance(itc, dict):
            for key in ("bug", "bugfix", "hotfix", "fix"):
                if key in itc.get("churn_top", {}):
                    bug_churn_top = itc["churn_top"][key]
                    break
        for entry in bug_churn_top:
            try:
                fname, lines = _normalise_path(entry[0]), int(entry[1])
            except (IndexError, ValueError, TypeError):
                continue
            if fname:
                bug_churn[fname] += lines

    return (
        anti_pattern_total,
        hotspot_fanin_sum,
        rev_presence,
        total_churn,
        bug_churn,
        anti_patterns_seen,
        anti_pattern_revision_count,
        anti_pattern_instance_load,
        anti_pattern_type_counts,
    )


def _parse_matrix_cells(matrix: Dict[str, Any]) -> Tuple[List[str], List[Tuple[str, str]]]:
    variables = matrix.get("variables") or []
    cells = matrix.get("cells") or matrix.get("matrix") or []
    edges: List[Tuple[str, str]] = []
    if not isinstance(variables, list) or not isinstance(cells, list):
        return [], edges
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        src_idx = cell.get("src")
        dest_idx = cell.get("dest")
        values = cell.get("values") or {}
        if not isinstance(src_idx, int) or not isinstance(dest_idx, int):
            continue
        if src_idx < 0 or dest_idx < 0:
            continue
        if src_idx >= len(variables) or dest_idx >= len(variables):
            continue
        if not isinstance(values, dict) or not values:
            continue
        if not any(float(v) > 0 for v in values.values() if isinstance(v, (int, float))):
            continue
        src = _normalise_path(str(variables[src_idx]))
        dest = _normalise_path(str(variables[dest_idx]))
        if src and dest:
            edges.append((src, dest))
    return [_normalise_path(str(v)) for v in variables], edges


def _tarjan_scc(nodes: List[str], edges: List[Tuple[str, str]]) -> List[List[str]]:
    adj: Dict[str, List[str]] = {n: [] for n in nodes if n}
    for src, dest in edges:
        adj.setdefault(src, []).append(dest)

    index = 0
    stack: List[str] = []
    on_stack: Dict[str, bool] = {}
    idx_map: Dict[str, int] = {}
    low: Dict[str, int] = {}
    out: List[List[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        idx_map[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj.get(v, []):
            if w not in idx_map:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif on_stack.get(w):
                low[v] = min(low[v], idx_map[w])

        if low[v] == idx_map[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                out.append(comp)

    for v in nodes:
        if v and v not in idx_map:
            strongconnect(v)
    return out


def _collect_scc_signals(interp_root: Path, verbose: bool) -> Dict[str, int]:
    """Return file → #revisions_in_scc by recomputing SCCs from each revision matrix."""
    scc_count: Dict[str, int] = collections.defaultdict(int)
    rev_data_dir = interp_root / "SINGLE_REVISION_ANALYSIS_DATA"
    if not rev_data_dir.is_dir():
        return scc_count
    for rev_dir in sorted(rev_data_dir.iterdir()):
        if not rev_dir.is_dir():
            continue
        matrix_json = rev_dir / "OutputData" / "dsm" / "matrix.json"
        if not matrix_json.exists():
            continue
        if verbose:
            print(f"  [scc] {rev_dir.name}", flush=True)
        data = _load_json(matrix_json)
        nodes, edges = _parse_matrix_cells(data)
        for comp in _tarjan_scc(nodes, edges):
            for fpath in comp:
                if fpath:
                    scc_count[fpath] += 1
    return scc_count


def _load_dsm_edges(matrix_json: Path) -> Set[Tuple[str, str]]:
    """Return set of (src_path, dest_path) from a DSM matrix.json."""
    data = _load_json(matrix_json)
    variables: List[str] = data.get("variables", [])
    edges: Set[Tuple[str, str]] = set()
    for edge in data.get("cells") or data.get("matrix") or []:
        try:
            src = variables[edge["src"]]
            dest = variables[edge["dest"]]
            edges.add((_normalise_path(src), _normalise_path(dest)))
        except (KeyError, IndexError, TypeError):
            continue
    return edges


def _mine_cochange(
    interp_root: Path,
    timeseries: Dict[str, Any],
    git_root: Path | None,
    threshold: int,
    verbose: bool,
) -> Tuple[
    Dict[str, int],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
]:
    """
    Return file → number_of_non_structural_cochange_partners.

    For each revision window (commit range from timeseries), run git to get
    per-commit file lists, build a co-occurrence matrix, subtract DSM edges,
    and count partners above threshold.
    """
    cochange_partners: Dict[str, int] = collections.defaultdict(int)
    hidden_cochange: Dict[str, Dict[str, int]] = collections.defaultdict(dict)

    if git_root is None or not git_root.is_dir():
        if verbose:
            print("  [co-change] no git root available, skipping co-change mining", flush=True)
        return cochange_partners, {}, {}

    revisions: List[Dict] = timeseries.get("revisions", [])
    if len(revisions) < 2:
        return cochange_partners, {}, {}

    # Collect all (older, newer) commit ranges from timeseries.
    # revision_number 1 = newest commit; highest revision_number = oldest.
    # So sorted_revs[i] (lower number) is newer, sorted_revs[i+1] is older.
    # git log OLDER..NEWER lists commits reachable from NEWER but not OLDER.
    ranges: List[Tuple[str, str]] = []
    sorted_revs = sorted(revisions, key=lambda r: r["revision_number"])
    for i in range(len(sorted_revs) - 1):
        older = sorted_revs[i + 1]["commit_hash"]  # higher rev_number = older
        newer = sorted_revs[i]["commit_hash"]       # lower rev_number = newer
        ranges.append((older, newer))

    # Load DSM edges per revision for structural-vs-behavioral comparison
    # We use the newest revision's DSM as proxy for all ranges (simplification)
    dsm_edges: Set[Tuple[str, str]] = set()
    rev_data_dir = interp_root / "SINGLE_REVISION_ANALYSIS_DATA"
    rev_entries = sorted(rev_data_dir.iterdir()) if rev_data_dir.is_dir() else []
    if rev_entries:
        first_rev = rev_entries[0]
        matrix_json = first_rev / "OutputData" / "dsm" / "matrix.json"
        if matrix_json.exists():
            if verbose:
                print(f"  [co-change] loading DSM from {matrix_json.parent.parent.parent.name}", flush=True)
            dsm_edges = _load_dsm_edges(matrix_json)

    # Per-range co-occurrence accumulation
    global_cochange: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )

    for older, newer in ranges:
        commit_range = f"{older}..{newer}"
        if verbose:
            print(f"  [co-change] mining {commit_range[:20]}...", flush=True)

        # Get all commit hashes in this range
        try:
            result = subprocess.run(
                ["git", "log", "--format=%H", commit_range],
                cwd=str(git_root),
                capture_output=True,
                text=True,
                timeout=60,
            )
            commit_hashes = [h.strip() for h in result.stdout.splitlines() if h.strip()]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

        for chash in commit_hashes:
            # Get files changed in this commit
            try:
                result = subprocess.run(
                    ["git", "diff-tree", "--no-commit-id", "-r", "--name-only", chash],
                    cwd=str(git_root),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                changed_files = [
                    _normalise_path(f.strip())
                    for f in result.stdout.splitlines()
                    if f.strip()
                ]
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

            # Increment co-occurrence counts for all pairs
            for i, fa in enumerate(changed_files):
                for fb in changed_files[i + 1:]:
                    if fa and fb and fa != fb:
                        global_cochange[fa][fb] += 1
                        global_cochange[fb][fa] += 1

    # Compute co_change_without_dep: partners above threshold not in DSM
    for fa, partners in global_cochange.items():
        count = 0
        for fb, occurrences in partners.items():
            if occurrences >= threshold:
                # Check if this pair has a structural dependency
                if (fa, fb) not in dsm_edges and (fb, fa) not in dsm_edges:
                    count += 1
                    hidden_cochange[fa][fb] = occurrences
        if count > 0:
            cochange_partners[fa] = count

    # Convert nested defaultdicts to plain dicts for output stability
    raw_cochange = {fa: dict(partners) for fa, partners in global_cochange.items()}
    hidden_cochange = {fa: dict(partners) for fa, partners in hidden_cochange.items()}
    return cochange_partners, raw_cochange, hidden_cochange


def _bug_churn_from_commits(
    interp_root: Path,
    timeseries: Dict[str, Any],
    git_root: Path | None,
    verbose: bool,
) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
    """
    Fallback: compute bug-linked churn by keyword-matching commit messages.
    Used when issue_typed_churn is empty in the payloads.
    Keywords: fix, bug, hotfix, patch, defect, issue, error, correct, regress
    """
    bug_keywords = re.compile(
        r"\b(fix|bug|hotfix|patch|defect|issue|error|correct|regress)\b",
        re.IGNORECASE,
    )
    bug_churn: Dict[str, int] = collections.defaultdict(int)
    audit_commits: List[Dict[str, Any]] = []

    if git_root is None or not git_root.is_dir():
        return bug_churn, audit_commits

    revisions = timeseries.get("revisions", [])
    # Sort ascending by revision_number — rev_number 1 = newest, highest = oldest
    # so "sorted_revs[i+1]" (higher number = older commit) is the base,
    # "sorted_revs[i]" (lower number = newer commit) is the tip.
    sorted_revs = sorted(revisions, key=lambda r: r["revision_number"])
    ranges: List[Tuple[str, str]] = []
    for i in range(len(sorted_revs) - 1):
        # older (higher rev_number) → newer (lower rev_number)
        older_hash = sorted_revs[i + 1]["commit_hash"]
        newer_hash = sorted_revs[i]["commit_hash"]
        ranges.append((older_hash, newer_hash))

    for older, newer in ranges:
        commit_range = f"{older}..{newer}"
        if verbose:
            print(f"  [bug-churn-fallback] {commit_range[:20]}...", flush=True)
        try:
            result = subprocess.run(
                ["git", "log", "--format=%H %s", commit_range],
                cwd=str(git_root),
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

        bug_commits: List[Tuple[str, str, List[str]]] = []
        for line in result.stdout.splitlines():
            parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                continue
            chash, subject = parts[0], parts[1]
            matched_keywords = sorted({m.lower() for m in bug_keywords.findall(subject)})
            if matched_keywords:
                bug_commits.append((chash, subject, matched_keywords))

        for chash, subject, matched_keywords in bug_commits:
            try:
                result = subprocess.run(
                    ["git", "diff-tree", "--no-commit-id", "-r", "--numstat", chash],
                    cwd=str(git_root),
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

            changed_files: List[Dict[str, Any]] = []
            for line in result.stdout.splitlines():
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                try:
                    added = int(parts[0]) if parts[0] != "-" else 0
                    deleted = int(parts[1]) if parts[1] != "-" else 0
                    fname = _normalise_path(parts[2])
                    if fname:
                        bug_churn[fname] += added + deleted
                        changed_files.append(
                            {
                                "file": fname,
                                "added": added,
                                "deleted": deleted,
                                "churn": added + deleted,
                            }
                        )
                except (ValueError, IndexError):
                    continue

            if changed_files:
                audit_commits.append(
                    {
                        "range": commit_range,
                        "commit_hash": chash,
                        "subject": subject,
                        "matched_keywords": matched_keywords,
                        "total_churn": sum(item["churn"] for item in changed_files),
                        "files": changed_files,
                    }
                )

    return bug_churn, audit_commits


# ---------------------------------------------------------------------------
# Normalisation / utility
# ---------------------------------------------------------------------------


_SOURCE_ROOT_PREFIXES = (
    "src/main/java/",
    "src/main/kotlin/",
    "src/main/scala/",
    "src/main/groovy/",
    "src/main/",
    "src/java/",
    "main/java/",
    "main/",
    "source/",
    "sources/",
    "lib/",
)


def _normalise_path(p: str) -> str:
    """
    Strip leading/trailing whitespace, normalise separators, and remove
    common source-root prefixes so that git churn paths
    (e.g. 'src/main/java/org/apache/...')  match DSM variable paths
    (e.g. 'org/apache/...').
    """
    if not p:
        return ""
    p = p.strip().replace("\\", "/")
    for prefix in _SOURCE_ROOT_PREFIXES:
        if p.startswith(prefix):
            return p[len(prefix):]
    return p


def _auto_detect_git_root(interp_root: Path, timeseries: Dict[str, Any]) -> Path | None:
    """Try to find the git repo root from the revision folder structure."""
    rev_data_dir = interp_root / "SINGLE_REVISION_ANALYSIS_DATA"
    if not rev_data_dir.is_dir():
        return None
    for rev_dir in sorted(rev_data_dir.iterdir()):
        if not rev_dir.is_dir():
            continue
        # The revision folder often contains the actual repo checkout
        # InputData usually has the source; OutputData has analysis results.
        # Walk up from interp_root to find a .git directory.
        candidate = interp_root.parent
        while candidate != candidate.parent:
            if (candidate / ".git").is_dir():
                return candidate
            candidate = candidate.parent
    return None


def _normalise_values(values: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a dict of floats to [0, 1]."""
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    span = vmax - vmin
    if span == 0:
        return {k: 0.0 for k in values}
    return {k: (v - vmin) / span for k, v in values.items()}


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_risk_scores(
    anti_pattern_total: Dict[str, int],
    hotspot_fanin_sum: Dict[str, float],
    rev_presence: Dict[str, int],
    total_churn: Dict[str, int],
    bug_churn: Dict[str, int],
    scc_count: Dict[str, int],
    cochange_partners: Dict[str, int],
    weights: Dict[str, float],
    anti_patterns_seen: Dict[str, set] = None,
    anti_pattern_revision_count: Dict[str, int] = None,
    anti_pattern_instance_load: Dict[str, int] = None,
    anti_pattern_type_counts: Dict[str, Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Compute composite risk scores and return sorted list of file dicts."""

    # Universe of files: union of all signal dicts
    all_files: Set[str] = (
        set(anti_pattern_total)
        | set(hotspot_fanin_sum)
        | set(rev_presence)
        | set(total_churn)
        | set(bug_churn)
        | set(scc_count)
        | set(cochange_partners)
    )
    all_files = {f for f in all_files if f}  # drop empty strings

    # Raw signal dicts (with 0 defaults)
    def raw(d: Dict, f: str, default=0):
        return d.get(f, default) if d is not None else default

    anti_pattern_mean = (
        sum(anti_pattern_total.values()) / len(anti_pattern_total)
        if anti_pattern_total
        else 0.0
    )
    anti_pattern_std = (
        math.sqrt(
            sum((v - anti_pattern_mean) ** 2 for v in anti_pattern_total.values()) / len(anti_pattern_total)
        )
        if anti_pattern_total
        else 0.0
    )
    anti_pattern_instance_mean = (
        sum(anti_pattern_instance_load.values()) / len(anti_pattern_instance_load)
        if anti_pattern_instance_load
        else 0.0
    )
    anti_pattern_instance_std = (
        math.sqrt(
            sum((v - anti_pattern_instance_mean) ** 2 for v in anti_pattern_instance_load.values()) / len(anti_pattern_instance_load)
        )
        if anti_pattern_instance_load
        else 0.0
    )

    raw_signals = {
        f: {
            "anti_pattern_count": raw(anti_pattern_total, f),
            "anti_pattern_revision_count": raw(anti_pattern_revision_count, f),
            "anti_pattern_instance_load": raw(anti_pattern_instance_load, f),
            "anti_pattern_diversity": len(raw(anti_patterns_seen, f, set())),
            "hotspot_fanin_score": raw(hotspot_fanin_sum, f, 0.0),
            "rev_count": raw(rev_presence, f),
            "total_churn": raw(total_churn, f),
            "bug_churn_total": raw(bug_churn, f),
            "scc_membership_count": raw(scc_count, f),
            "co_change_without_dep": raw(cochange_partners, f),
            "anti_pattern_relative_to_mean": (
                round(raw(anti_pattern_total, f) / anti_pattern_mean, 6)
                if anti_pattern_mean > 0
                else 0.0
            ),
            "anti_pattern_zscore": (
                round((raw(anti_pattern_total, f) - anti_pattern_mean) / anti_pattern_std, 6)
                if anti_pattern_std > 0
                else 0.0
            ),
            "anti_pattern_instance_relative_to_mean": (
                round(raw(anti_pattern_instance_load, f) / anti_pattern_instance_mean, 6)
                if anti_pattern_instance_mean > 0
                else 0.0
            ),
            "anti_pattern_instance_zscore": (
                round((raw(anti_pattern_instance_load, f) - anti_pattern_instance_mean) / anti_pattern_instance_std, 6)
                if anti_pattern_instance_std > 0
                else 0.0
            ),
        }
        for f in all_files
    }

    # Normalise each signal independently
    norm: Dict[str, Dict[str, float]] = {}
    for sig_name in [
        "anti_pattern_count",
        "hotspot_fanin_score",
        "total_churn",
        "bug_churn_total",
        "scc_membership_count",
        "co_change_without_dep",
    ]:
        raw_vals = {f: raw_signals[f][sig_name] for f in all_files}
        norm_vals = _normalise_values(raw_vals)
        for f in all_files:
            norm.setdefault(f, {})[sig_name] = norm_vals.get(f, 0.0)

    # Composite score
    w = weights
    results: List[Dict[str, Any]] = []
    for f in all_files:
        n = norm[f]
        score = (
            w.get("bug_churn", 0.25) * n["bug_churn_total"]
            + w.get("total_churn", 0.05) * n["total_churn"]
            + w.get("anti_pattern", 0.25) * n["anti_pattern_count"]
            + w.get("hotspot_fanin", 0.20) * n["hotspot_fanin_score"]
            + w.get("scc_membership", 0.15) * n["scc_membership_count"]
            + w.get("co_change", 0.10) * n["co_change_without_dep"]
        )
        ap_seen = sorted(anti_patterns_seen.get(f, set())) if anti_patterns_seen else []
        ap_type_counts = dict(sorted(raw(anti_pattern_type_counts, f, {}).items()))
        ap_type_counts_structural = {
            k: v for k, v in ap_type_counts.items() if k not in HISTORY_ANTI_PATTERN_TYPES
        }
        ap_type_counts_history = {
            k: v for k, v in ap_type_counts.items() if k in HISTORY_ANTI_PATTERN_TYPES
        }
        results.append(
            {
                "file": f,
                "risk_score": round(score, 6),
                "signals": raw_signals[f],
                "signals_normalised": {k: round(v, 6) for k, v in n.items()},
                "anti_patterns_seen": ap_seen,
                "anti_pattern_type_counts": ap_type_counts,
                "anti_pattern_type_counts_structural": ap_type_counts_structural,
                "anti_pattern_type_counts_history": ap_type_counts_history,
            }
        )

    results.sort(key=lambda x: x["risk_score"], reverse=True)
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"  Written: {path}")


def _write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    signal_keys = list(records[0]["signals"].keys())
    fieldnames = ["rank", "risk_score", "file"] + signal_keys + ["anti_patterns_seen", "anti_pattern_type_counts"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {
                "rank": rec["rank"],
                "risk_score": rec["risk_score"],
                "file": rec["file"],
                "anti_patterns_seen": "|".join(rec.get("anti_patterns_seen", [])),
                "anti_pattern_type_counts": json.dumps(rec.get("anti_pattern_type_counts", {}), sort_keys=True),
            }
            row.update(rec["signals"])
            writer.writerow(row)
    print(f"  Written: {path}")


def _matrix_file_order(matrix_map: Dict[str, Dict[str, int]]) -> List[str]:
    files = set(matrix_map.keys())
    for row in matrix_map.values():
        files.update(row.keys())
    row_sum: Dict[str, int] = {}
    for f in files:
        row_sum[f] = sum(matrix_map.get(f, {}).values())
    return sorted(files, key=lambda f: (-row_sum.get(f, 0), f))


def _row_sum_map(matrix_map: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    files = set(matrix_map.keys())
    for row in matrix_map.values():
        files.update(row.keys())
    return {f: int(sum(matrix_map.get(f, {}).values())) for f in files}


def _load_drh_order(interp_root: Path) -> Dict[str, Tuple[int, int, int]]:
    rev_root = interp_root / "SINGLE_REVISION_ANALYSIS_DATA"
    if not rev_root.exists():
        return {}
    rev_dirs = sorted([p for p in rev_root.iterdir() if p.is_dir()])
    if not rev_dirs:
        return {}

    candidates = [
        rev_dirs[0] / "OutputData" / "dv8-analysis-result" / "dsm" / "drh-clustering.json",
    ]
    dv8_dirs = list((rev_dirs[0] / "OutputData").glob("*/dv8-analysis-result/dsm/drh-clustering.json"))
    candidates.extend(dv8_dirs)

    drh_path = next((p for p in candidates if p.exists()), None)
    if drh_path is None:
        return {}

    try:
        data = json.loads(drh_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    structure = data.get("structure", [])
    if not isinstance(structure, list):
        return {}

    order: Dict[str, Tuple[int, int, int]] = {}
    item_pos = 0
    for layer in structure:
        if not isinstance(layer, dict):
            continue
        layer_name = str(layer.get("name", ""))
        m_layer = re.match(r"L(\d+)", layer_name)
        layer_no = int(m_layer.group(1)) if m_layer else 999
        for module in layer.get("nested", []) or []:
            if not isinstance(module, dict):
                continue
            module_name = str(module.get("name", ""))
            m_mod = re.match(r"L\d+/M(\d+)", module_name)
            mod_no = int(m_mod.group(1)) if m_mod else 999
            for item in module.get("nested", []) or []:
                if not isinstance(item, dict) or item.get("@type") != "item":
                    continue
                fname = str(item.get("name", "")).strip()
                if fname:
                    order[fname] = (layer_no, mod_no, item_pos)
                    item_pos += 1
    return order


def _matrix_file_order_drh(
    matrix_map: Dict[str, Dict[str, int]],
    *,
    drh_order: Dict[str, Tuple[int, int, int]],
) -> List[str]:
    files = set(matrix_map.keys())
    for row in matrix_map.values():
        files.update(row.keys())
    row_sum = _row_sum_map(matrix_map)

    matched = [f for f in files if f in drh_order]
    unmatched = [f for f in files if f not in drh_order]

    matched_sorted = sorted(
        matched,
        key=lambda f: (
            drh_order[f][0],
            drh_order[f][1],
            -row_sum.get(f, 0),
            drh_order[f][2],
            f,
        ),
    )
    unmatched_sorted = sorted(unmatched, key=lambda f: (-row_sum.get(f, 0), f))
    return matched_sorted + unmatched_sorted


def _dense_matrix(matrix_map: Dict[str, Dict[str, int]], files: List[str]) -> List[List[int]]:
    return [
        [int(matrix_map.get(src, {}).get(dest, 0)) for dest in files]
        for src in files
    ]


def _is_source_file(path_str: str) -> bool:
    p = Path(path_str)
    return p.suffix.lower() in SOURCE_FILE_EXTENSIONS


def _has_excluded_segment_str(path_str: str) -> bool:
    p = Path(path_str)
    return any(part.lower() in PROD_EXCLUDE_DIR_NAMES for part in p.parts)


def _filter_matrix_map(
    matrix_map: Dict[str, Dict[str, int]],
    *,
    file_predicate,
) -> Dict[str, Dict[str, int]]:
    allowed = {
        f
        for f in set(matrix_map.keys()) | {g for row in matrix_map.values() for g in row.keys()}
        if file_predicate(f)
    }
    out: Dict[str, Dict[str, int]] = {}
    for src, row in matrix_map.items():
        if src not in allowed:
            continue
        kept = {dest: val for dest, val in row.items() if dest in allowed}
        if kept:
            out[src] = kept
    return out


def _write_matrix_json(
    path: Path,
    *,
    name: str,
    files: List[str],
    matrix: List[List[int]],
    meta: Dict[str, Any],
) -> None:
    payload = {
        "name": name,
        "files": files,
        "matrix": matrix,
        "meta": meta,
    }
    _write_json(path, payload)


def _write_matrix_csv(path: Path, *, files: List[str], matrix: List[List[int]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file"] + files)
        for src, row in zip(files, matrix):
            writer.writerow([src] + row)
    print(f"  Written: {path}")


def _write_matrix_index_csv(
    path: Path,
    *,
    files: List[str],
    drh_order: Dict[str, Tuple[int, int, int]],
    row_sum: Dict[str, int],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "file", "layer", "module", "row_sum"])
        for idx, fname in enumerate(files, start=1):
            layer, module, _ = drh_order.get(fname, (None, None, None))
            writer.writerow([idx, fname, layer, module, row_sum.get(fname, 0)])
    print(f"  Written: {path}")


def _write_matrix_index_md(
    path: Path,
    *,
    title: str,
    files: List[str],
    drh_order: Dict[str, Tuple[int, int, int]],
    row_sum: Dict[str, int],
) -> None:
    lines = [f"# {title}", "", "| # | File | Layer | Module | Row Sum |", "|---:|---|---:|---:|---:|"]
    for idx, fname in enumerate(files, start=1):
        layer, module, _ = drh_order.get(fname, (None, None, None))
        layer_s = "" if layer is None else str(layer)
        module_s = "" if module is None else str(module)
        lines.append(f"| {idx} | `{fname}` | {layer_s} | {module_s} | {row_sum.get(fname, 0)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Written: {path}")


def _plot_matrix_heatmap(
    path: Path,
    *,
    title: str,
    files: List[str],
    matrix: List[List[int]],
    max_files: int = 80,
    index_axes: bool = False,
) -> None:
    if not files or not matrix:
        return
    if max_files > 0 and len(files) > max_files:
        files = files[:max_files]
        matrix = [row[:max_files] for row in matrix[:max_files]]

    arr = np.array(matrix, dtype=float)
    if arr.size == 0:
        return

    plot_arr = np.log1p(arr)
    fig_w = max(8, min(28, 0.22 * len(files)))
    fig_h = max(7, min(24, 0.22 * len(files)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(plot_arr, cmap="YlOrRd", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Target file")
    ax.set_ylabel("Source file")

    if index_axes:
        ax.set_xticks(range(len(files)))
        ax.set_xticklabels([str(i) for i in range(1, len(files) + 1)], rotation=90, fontsize=5)
        ax.set_yticks(range(len(files)))
        ax.set_yticklabels([str(i) for i in range(1, len(files) + 1)], fontsize=5)
    elif len(files) <= 60:
        ax.set_xticks(range(len(files)))
        ax.set_xticklabels(files, rotation=90, fontsize=6)
        ax.set_yticks(range(len(files)))
        ax.set_yticklabels(files, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("log(1 + co-change count)")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {path}")


def _write_matrix_bundle(
    interp_root: Path,
    *,
    prefix: str,
    title: str,
    matrix_map: Dict[str, Dict[str, int]],
    repo: str,
    revision_count: int,
    co_change_threshold: int,
    meaning: str,
    drh_order: Dict[str, Tuple[int, int, int]] | None = None,
    write_top_labeled_n: int = 0,
) -> None:
    if not matrix_map:
        return
    cochange_root = interp_root / "cochange"
    cochange_plots_root = interp_root / "plots" / "cochange"
    cochange_root.mkdir(parents=True, exist_ok=True)
    cochange_plots_root.mkdir(parents=True, exist_ok=True)
    drh_order = drh_order or {}
    files = _matrix_file_order_drh(matrix_map, drh_order=drh_order) if drh_order else _matrix_file_order(matrix_map)
    dense = _dense_matrix(matrix_map, files)
    row_sum = _row_sum_map(matrix_map)
    _write_matrix_json(
        cochange_root / f"{prefix}.json",
        name=prefix,
        files=files,
        matrix=dense,
        meta={
            "repo": repo,
            "revision_count": revision_count,
            "co_change_threshold": co_change_threshold,
            "meaning": meaning,
        },
    )
    _write_matrix_csv(
        cochange_root / f"{prefix}.csv",
        files=files,
        matrix=dense,
    )
    _write_matrix_index_csv(
        cochange_root / f"{prefix}_index.csv",
        files=files,
        drh_order=drh_order,
        row_sum=row_sum,
    )
    _write_matrix_index_md(
        cochange_root / f"{prefix}_index.md",
        title=f"{title} — file index",
        files=files,
        drh_order=drh_order,
        row_sum=row_sum,
    )
    _plot_matrix_heatmap(
        cochange_plots_root / f"{prefix}_heatmap.png",
        title=title,
        files=files,
        matrix=dense,
        index_axes=True,
    )
    if write_top_labeled_n > 0:
        top_n = min(write_top_labeled_n, len(files))
        top_files = files[:top_n]
        top_dense = _dense_matrix(matrix_map, top_files)
        _write_matrix_json(
            cochange_root / f"{prefix}_top{top_n}.json",
            name=f"{prefix}_top{top_n}",
            files=top_files,
            matrix=top_dense,
            meta={
                "repo": repo,
                "revision_count": revision_count,
                "co_change_threshold": co_change_threshold,
                "meaning": f"top {top_n} files from {prefix} using DRH-aware ordering",
            },
        )
        _write_matrix_csv(
            cochange_root / f"{prefix}_top{top_n}.csv",
            files=top_files,
            matrix=top_dense,
        )
        _write_matrix_index_md(
            cochange_root / f"{prefix}_top{top_n}_index.md",
            title=f"{title} (Top {top_n}) — file index",
            files=top_files,
            drh_order=drh_order,
            row_sum=row_sum,
        )
        _plot_matrix_heatmap(
            cochange_plots_root / f"{prefix}_top{top_n}_heatmap.png",
            title=f"{title} (Top {top_n}, labeled)",
            files=top_files,
            matrix=top_dense,
            max_files=top_n,
            index_axes=False,
        )


def _extract_json_block(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _extract_json_list(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
    except Exception:
        return []
    return []


def _review_bug_churn_commits_with_llm(
    audit: Dict[str, Any],
    *,
    model: str,
    timeout_s: int,
) -> Dict[str, Any]:
    commits = audit.get("commits") or []
    if not commits:
        return {
            "meta": {
                "model": model,
                "source": audit.get("meta", {}).get("source"),
                "note": "No heuristic bug-churn commits available for LLM review.",
            },
            "commits": [],
        }
    if LLMBackend is None:
        return {
            "meta": {
                "model": model,
                "source": audit.get("meta", {}).get("source"),
                "error": "LLMBackend import failed.",
            },
            "commits": [],
        }

    llm = LLMBackend(model=model, num_ctx=8192)
    reviewed: List[Dict[str, Any]] = []
    system = (
        "You classify software commit messages for bug-fix relevance. "
        "Return only compact JSON. "
        "Valid llm_label values: bug_fix, not_bug_fix, uncertain."
    )
    batch_size = 10
    for i in range(0, len(commits), batch_size):
        batch = commits[i:i + batch_size]
        batch_payload = []
        for commit in batch:
            files = commit.get("files") or []
            top_files = sorted(files, key=lambda x: x.get("churn", 0), reverse=True)[:8]
            file_summary = [
                {
                    "file": item.get("file"),
                    "churn": item.get("churn"),
                }
                for item in top_files
            ]
            batch_payload.append(
                {
                    "commit_hash": commit.get("commit_hash"),
                    "subject": commit.get("subject"),
                    "matched_keywords": commit.get("matched_keywords"),
                    "top_changed_files": file_summary,
                }
            )
        prompt = (
            "Classify whether each commit is primarily a real bug-fix/defect-fix commit.\n"
            "Treat typo-only, formatting-only, docs-only, CI-only, or test-only maintenance as not_bug_fix unless the text clearly describes fixing a real defect.\n"
            "If unclear, return uncertain.\n\n"
            f"commits: {json.dumps(batch_payload, ensure_ascii=False)}\n\n"
            'Return JSON array only, one object per commit, with keys: commit_hash, llm_label, confidence, rationale.'
        )
        raw = llm.generate(prompt=prompt, system=system, timeout_s=timeout_s)
        parsed_list = _extract_json_list(raw)
        parsed_by_hash = {item.get("commit_hash"): item for item in parsed_list if item.get("commit_hash")}
        for commit in batch:
            files = commit.get("files") or []
            top_files = sorted(files, key=lambda x: x.get("churn", 0), reverse=True)[:8]
            file_summary = [
                {
                    "file": item.get("file"),
                    "churn": item.get("churn"),
                }
                for item in top_files
            ]
            parsed = parsed_by_hash.get(commit.get("commit_hash"), {})
            reviewed.append(
                {
                    "commit_hash": commit.get("commit_hash"),
                    "subject": commit.get("subject"),
                    "matched_keywords": commit.get("matched_keywords", []),
                    "heuristic_label": "bug_fix",
                    "llm_label": parsed.get("llm_label", "uncertain"),
                    "confidence": parsed.get("confidence"),
                    "rationale": parsed.get("rationale", raw[:500]),
                    "total_churn": commit.get("total_churn"),
                    "files": file_summary,
                }
            )
    return {
        "meta": {
            "model": model,
            "source": audit.get("meta", {}).get("source"),
            "reviewed_commit_count": len(reviewed),
        },
        "commits": reviewed,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute multi-signal per-file risk scores from a temporal analysis folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "interp_root",
        help="Path to the INPUT_INTERPRETATION/ folder from a temporal analysis run.",
    )
    ap.add_argument(
        "--git-root",
        help="Path to the git repository root (auto-detected if omitted).",
        default=None,
    )
    ap.add_argument(
        "--weights",
        help="JSON dict of signal weights, e.g. '{\"bug_churn\":0.4}'",
        default=None,
    )
    ap.add_argument(
        "--co-change-threshold",
        type=int,
        default=CO_CHANGE_THRESHOLD,
        help=f"Min co-occurrences to count a file pair as coupled (default: {CO_CHANGE_THRESHOLD}).",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Limit output to top N files by risk score (default: all).",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages.",
    )
    ap.add_argument(
        "--bug-churn-review-model",
        default=None,
        help="Optional LLM model for reviewing heuristic bug-churn commits, e.g. deepseek-r1:32b or qwen2.5:14b",
    )
    ap.add_argument(
        "--bug-churn-review-timeout-s",
        type=int,
        default=120,
        help="Timeout in seconds per LLM bug-churn review call (default: 120).",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    interp_root = Path(args.interp_root).resolve()
    if not interp_root.is_dir():
        print(f"ERROR: {interp_root} is not a directory.", file=sys.stderr)
        return 1

    # Resolve weights
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        try:
            overrides = json.loads(args.weights)
            weights.update(overrides)
        except json.JSONDecodeError as exc:
            print(f"ERROR: --weights is not valid JSON: {exc}", file=sys.stderr)
            return 1

    verbose = args.verbose

    # Load timeseries
    ts_path = interp_root / "timeseries.json"
    if not ts_path.exists():
        print(f"ERROR: timeseries.json not found at {ts_path}", file=sys.stderr)
        return 1
    timeseries = _load_json(ts_path)
    repo = timeseries.get("repo", "unknown")

    # Git root
    git_root: Path | None = None
    if args.git_root:
        git_root = Path(args.git_root).resolve()
    else:
        git_root = _auto_detect_git_root(interp_root, timeseries)

    if verbose:
        print(f"Repo: {repo}")
        print(f"Revisions: {timeseries.get('revision_count', '?')}")
        print(f"Git root: {git_root}")
        print(f"Weights: {weights}")
        print(f"Co-change threshold: {args.co_change_threshold}")

    # --- Phase 1: Structural signals from interpretation payloads ---
    if verbose:
        print("\n[1] Collecting structural signals from payloads...")
    (
        anti_pattern_total,
        hotspot_fanin_sum,
        rev_presence,
        total_churn_all,
        bug_churn_payload,
        anti_patterns_seen,
        anti_pattern_revision_count,
        anti_pattern_instance_load,
        anti_pattern_type_counts,
    ) = _collect_structural_signals(interp_root, verbose)

    # --- Phase 2: SCC membership from evidence graph diffs ---
    if verbose:
        print("\n[2] Collecting SCC signals from evidence graph diffs...")
    scc_count = _collect_scc_signals(interp_root, verbose)

    # --- Phase 3: Bug churn ---
    # Use payload typed churn if any file has it; otherwise use keyword fallback.
    if verbose:
        print("\n[3] Resolving bug churn...")
    if any(v > 0 for v in bug_churn_payload.values()):
        bug_churn = bug_churn_payload
        bug_churn_audit = {
            "meta": {
                "source": "issue_typed_churn_payload",
                "note": "Commit-level bug-churn audit is not available in this mode because the payload only provides aggregated per-file typed churn.",
            },
            "commits": [],
        }
        if verbose:
            print("  Using issue-typed churn from payloads.")
    else:
        if verbose:
            print("  No typed churn in payloads — using keyword-based commit fallback.")
        bug_churn, audit_commits = _bug_churn_from_commits(interp_root, timeseries, git_root, verbose)
        bug_churn_audit = {
            "meta": {
                "source": "keyword_commit_fallback",
                "keywords": ["fix", "bug", "hotfix", "patch", "defect", "issue", "error", "correct", "regress"],
            },
            "commits": audit_commits,
        }

    # --- Phase 4: Co-change mining ---
    if verbose:
        print("\n[4] Mining co-change signals from git history...")
    cochange_partners, raw_cochange_matrix, hidden_cochange_matrix = _mine_cochange(
        interp_root,
        timeseries,
        git_root,
        args.co_change_threshold,
        verbose,
    )
    drh_order = _load_drh_order(interp_root)

    # --- Phase 5: Compute composite risk scores ---
    if verbose:
        print("\n[5] Computing composite risk scores...")
    results = compute_risk_scores(
        anti_pattern_total=anti_pattern_total,
        hotspot_fanin_sum=hotspot_fanin_sum,
        rev_presence=rev_presence,
        total_churn=total_churn_all,
        bug_churn=bug_churn,
        scc_count=scc_count,
        cochange_partners=cochange_partners,
        weights=weights,
        anti_patterns_seen=anti_patterns_seen,
        anti_pattern_revision_count=anti_pattern_revision_count,
        anti_pattern_instance_load=anti_pattern_instance_load,
        anti_pattern_type_counts=anti_pattern_type_counts,
    )

    # Apply top-n limit
    if args.top_n > 0:
        results = results[: args.top_n]

    # --- Phase 6: Write outputs ---
    now = datetime.now().strftime("%Y-%m-%d")
    revisions = timeseries.get("revisions", [])
    dates = [r.get("commit_date", "") for r in revisions if r.get("commit_date")]
    date_range = f"{min(dates)[:10]} to {max(dates)[:10]}" if dates else "unknown"

    output_json = {
        "meta": {
            "repo": repo,
            "revision_count": timeseries.get("revision_count", len(revisions)),
            "date_range": date_range,
            "generated": now,
            "weights": weights,
            "co_change_threshold": args.co_change_threshold,
            "git_root": str(git_root) if git_root else None,
            "signal_sources": {
                "anti_pattern": "DV8 .dv8-clsx anti-pattern instance membership across structural and history-based types (fallback: dangerous_files from interpretation_payload.json)",
                "hotspot_fanin": "structural_hotspots.rows[].FanIn from interpretation_payload.json",
                "bug_churn": "issue_typed_churn or keyword-based commit fallback",
                "total_churn": "churn_top from interpretation_payload.json",
                "scc_membership": "full SCC membership recomputed from each revision OutputData/dsm/matrix.json",
                "co_change": "git diff-tree co-occurrence without DSM edge (filtered against newest revision DSM edges)",
            },
        },
        "files": results,
    }

    if verbose:
        print(f"\n[6] Writing outputs to {interp_root}...")
    _write_json(interp_root / "file_risk_scores.json", output_json)
    _write_csv(interp_root / "file_risk_scores.csv", results)
    _write_json(interp_root / "bug_churn_commits.json", bug_churn_audit)
    if args.bug_churn_review_model:
        if verbose:
            print(f"\n[6b] Reviewing bug-churn commits with {args.bug_churn_review_model}...")
        llm_review = _review_bug_churn_commits_with_llm(
            bug_churn_audit,
            model=args.bug_churn_review_model,
            timeout_s=args.bug_churn_review_timeout_s,
        )
        _write_json(interp_root / "bug_churn_commits_llm_review.json", llm_review)

    # --- Phase 7: Write co-change matrix artifacts ---
    if verbose:
        print(f"\n[7] Writing co-change matrix artifacts...")

    if raw_cochange_matrix:
        _write_matrix_bundle(
            interp_root,
            prefix="cochange_matrix",
            title=f"{repo}: Co-change Matrix (DRH-ordered, indexed axes)",
            matrix_map=raw_cochange_matrix,
            repo=repo,
            revision_count=timeseries.get("revision_count", len(revisions)),
            co_change_threshold=args.co_change_threshold,
            meaning="pairwise commit co-occurrence counts across the analyzed temporal window",
            drh_order=drh_order,
        )

        source_only = _filter_matrix_map(raw_cochange_matrix, file_predicate=_is_source_file)
        _write_matrix_bundle(
            interp_root,
            prefix="cochange_matrix_source_only",
            title=f"{repo}: Co-change Matrix (source-only)",
            matrix_map=source_only,
            repo=repo,
            revision_count=timeseries.get("revision_count", len(revisions)),
            co_change_threshold=args.co_change_threshold,
            meaning="pairwise commit co-occurrence counts across the analyzed temporal window, filtered to source-code files by extension",
            drh_order=drh_order,
        )

        prod_only = _filter_matrix_map(
            raw_cochange_matrix,
            file_predicate=lambda f: _is_source_file(f) and not _has_excluded_segment_str(f),
        )
        _write_matrix_bundle(
            interp_root,
            prefix="cochange_matrix_prod_only",
            title=f"{repo}: Co-change Matrix (prod-only)",
            matrix_map=prod_only,
            repo=repo,
            revision_count=timeseries.get("revision_count", len(revisions)),
            co_change_threshold=args.co_change_threshold,
            meaning="pairwise commit co-occurrence counts across the analyzed temporal window, filtered with deterministic prod-only exclusions",
            drh_order=drh_order,
        )

    if hidden_cochange_matrix:
        _write_matrix_bundle(
            interp_root,
            prefix="hidden_coupling_matrix",
            title=f"{repo}: Hidden Coupling Matrix (DRH-ordered, indexed axes, threshold >= {args.co_change_threshold})",
            matrix_map=hidden_cochange_matrix,
            repo=repo,
            revision_count=timeseries.get("revision_count", len(revisions)),
            co_change_threshold=args.co_change_threshold,
            meaning="co-change counts for file pairs with cochange >= threshold and no DSM edge in either direction",
            drh_order=drh_order,
        )

        hidden_source_only = _filter_matrix_map(hidden_cochange_matrix, file_predicate=_is_source_file)
        _write_matrix_bundle(
            interp_root,
            prefix="hidden_coupling_matrix_source_only",
            title=f"{repo}: Hidden Coupling Matrix (source-only, threshold >= {args.co_change_threshold})",
            matrix_map=hidden_source_only,
            repo=repo,
            revision_count=timeseries.get("revision_count", len(revisions)),
            co_change_threshold=args.co_change_threshold,
            meaning="hidden coupling matrix filtered to source-code files by extension",
            drh_order=drh_order,
        )

        hidden_prod_only = _filter_matrix_map(
            hidden_cochange_matrix,
            file_predicate=lambda f: _is_source_file(f) and not _has_excluded_segment_str(f),
        )
        _write_matrix_bundle(
            interp_root,
            prefix="hidden_coupling_matrix_prod_only",
            title=f"{repo}: Hidden Coupling Matrix (prod-only, DRH-ordered, indexed axes, threshold >= {args.co_change_threshold})",
            matrix_map=hidden_prod_only,
            repo=repo,
            revision_count=timeseries.get("revision_count", len(revisions)),
            co_change_threshold=args.co_change_threshold,
            meaning="hidden coupling matrix filtered with deterministic prod-only exclusions",
            drh_order=drh_order,
            write_top_labeled_n=30,
        )

    print(f"\nDone. {len(results)} files scored.")
    if results:
        print(f"Top-5 by risk score:")
        for item in results[:5]:
            print(
                f"  #{item['rank']:>3}  {item['risk_score']:.4f}  {item['file']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
