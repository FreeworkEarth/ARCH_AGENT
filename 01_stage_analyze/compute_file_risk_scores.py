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
                         '{"bug_churn":0.4,"anti_pattern":0.2,"hotspot_fanin":0.2,
                           "scc_membership":0.1,"co_change":0.1}'
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
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "bug_churn": 0.30,
    "anti_pattern": 0.25,
    "hotspot_fanin": 0.20,
    "scc_membership": 0.15,
    "co_change": 0.10,
}
CO_CHANGE_THRESHOLD = 3  # minimum co-occurrences to count as coupled

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


def _collect_antipattern_membership(output_dir: Path) -> Dict[str, Set[str]]:
    """
    Parse all .dv8-clsx files under output_dir/arch-issue/ and return
    file → set(anti_pattern_type_names).

    Checks both arch-issue/ (new pipeline layout) and the nested
    dv8-analysis-result/anti-pattern/anti-pattern-instances/ layout.
    """
    membership: Dict[str, Set[str]] = collections.defaultdict(set)

    def _scan_dir(root: Path, ap_type: str) -> None:
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
]:
    anti_pattern_total: Dict[str, int] = collections.defaultdict(int)
    hotspot_fanin_sum: Dict[str, float] = collections.defaultdict(float)
    rev_presence: Dict[str, int] = collections.defaultdict(int)
    total_churn: Dict[str, int] = collections.defaultdict(int)
    bug_churn: Dict[str, int] = collections.defaultdict(int)
    anti_patterns_seen: Dict[str, set] = collections.defaultdict(set)

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
        clsx_membership: Dict[str, Set[str]] = {}
        if rev_output_dir.is_dir():
            clsx_membership = _collect_antipattern_membership(rev_output_dir)

        if clsx_membership:
            # Use real per-file anti-pattern data
            for fname, ap_types in clsx_membership.items():
                anti_pattern_total[fname] += len(ap_types)
                anti_patterns_seen[fname].update(ap_types)
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
    )


def _collect_scc_signals(interp_root: Path, verbose: bool) -> Dict[str, int]:
    """Return file → #revisions_in_scc from evidence_graph_diff files."""
    scc_count: Dict[str, int] = collections.defaultdict(int)
    diff_dir = interp_root / "EVIDENCE_GRAPH_DIFF"
    if not diff_dir.is_dir():
        return scc_count
    for diff_file in sorted(diff_dir.glob("evidence_graph_diff_*.json")):
        if verbose:
            print(f"  [scc] {diff_file.name}", flush=True)
        data = _load_json(diff_file)
        for section in ("scc_new", "scc_old"):
            scc_section = data.get(section, {})
            for scc_members in scc_section.get("top_sccs", []):
                for fpath in scc_members:
                    fname = _normalise_path(fpath)
                    if fname:
                        scc_count[fname] += 1
    return scc_count


def _load_dsm_edges(matrix_json: Path) -> Set[Tuple[str, str]]:
    """Return set of (src_path, dest_path) from a DSM matrix.json."""
    data = _load_json(matrix_json)
    variables: List[str] = data.get("variables", [])
    edges: Set[Tuple[str, str]] = set()
    for edge in data.get("matrix", []):
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
) -> Dict[str, int]:
    """
    Return file → number_of_non_structural_cochange_partners.

    For each revision window (commit range from timeseries), run git to get
    per-commit file lists, build a co-occurrence matrix, subtract DSM edges,
    and count partners above threshold.
    """
    cochange_partners: Dict[str, int] = collections.defaultdict(int)

    if git_root is None or not git_root.is_dir():
        if verbose:
            print("  [co-change] no git root available, skipping co-change mining", flush=True)
        return cochange_partners

    revisions: List[Dict] = timeseries.get("revisions", [])
    if len(revisions) < 2:
        return cochange_partners

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
    if rev_data_dir.is_dir():
        first_rev = sorted(rev_data_dir.iterdir())[0]
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
        if count > 0:
            cochange_partners[fa] = count

    return cochange_partners


def _bug_churn_from_commits(
    interp_root: Path,
    timeseries: Dict[str, Any],
    git_root: Path | None,
    verbose: bool,
) -> Dict[str, int]:
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

    if git_root is None or not git_root.is_dir():
        return bug_churn

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

        bug_commits: List[str] = []
        for line in result.stdout.splitlines():
            parts = line.strip().split(" ", 1)
            if len(parts) < 2:
                continue
            chash, subject = parts[0], parts[1]
            if bug_keywords.search(subject):
                bug_commits.append(chash)

        for chash in bug_commits:
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
                except (ValueError, IndexError):
                    continue

    return bug_churn


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
        return d.get(f, default)

    raw_signals = {
        f: {
            "anti_pattern_count": raw(anti_pattern_total, f),
            "hotspot_fanin_score": raw(hotspot_fanin_sum, f, 0.0),
            "rev_count": raw(rev_presence, f),
            "total_churn": raw(total_churn, f),
            "bug_churn_total": raw(bug_churn, f),
            "scc_membership_count": raw(scc_count, f),
            "co_change_without_dep": raw(cochange_partners, f),
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
            w.get("bug_churn", 0.30) * n["bug_churn_total"]
            + w.get("anti_pattern", 0.25) * n["anti_pattern_count"]
            + w.get("hotspot_fanin", 0.20) * n["hotspot_fanin_score"]
            + w.get("scc_membership", 0.15) * n["scc_membership_count"]
            + w.get("co_change", 0.10) * n["co_change_without_dep"]
        )
        ap_seen = sorted(anti_patterns_seen.get(f, set())) if anti_patterns_seen else []
        results.append(
            {
                "file": f,
                "risk_score": round(score, 6),
                "signals": raw_signals[f],
                "signals_normalised": {k: round(v, 6) for k, v in n.items()},
                "anti_patterns_seen": ap_seen,
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
    fieldnames = ["rank", "risk_score", "file"] + signal_keys + ["anti_patterns_seen"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = {
                "rank": rec["rank"],
                "risk_score": rec["risk_score"],
                "file": rec["file"],
                "anti_patterns_seen": "|".join(rec.get("anti_patterns_seen", [])),
            }
            row.update(rec["signals"])
            writer.writerow(row)
    print(f"  Written: {path}")


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
        if verbose:
            print("  Using issue-typed churn from payloads.")
    else:
        if verbose:
            print("  No typed churn in payloads — using keyword-based commit fallback.")
        bug_churn = _bug_churn_from_commits(interp_root, timeseries, git_root, verbose)

    # --- Phase 4: Co-change mining ---
    if verbose:
        print("\n[4] Mining co-change signals from git history...")
    cochange_partners = _mine_cochange(
        interp_root,
        timeseries,
        git_root,
        args.co_change_threshold,
        verbose,
    )

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
                "anti_pattern": "dangerous_files from interpretation_payload.json (DV8 flagged)",
                "hotspot_fanin": "structural_hotspots.rows[].FanIn from interpretation_payload.json",
                "bug_churn": "issue_typed_churn or keyword-based commit fallback",
                "total_churn": "churn_top from interpretation_payload.json",
                "scc_membership": "evidence_graph_diff top_sccs membership",
                "co_change": "git diff-tree co-occurrence without DSM edge",
            },
        },
        "files": results,
    }

    if verbose:
        print(f"\n[6] Writing outputs to {interp_root}...")
    _write_json(interp_root / "file_risk_scores.json", output_json)
    _write_csv(interp_root / "file_risk_scores.csv", results)

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
