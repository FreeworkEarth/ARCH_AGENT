#!/usr/bin/env python3
"""
build_risk_calibration_dataset.py
=================================

Build a rolling temporal calibration dataset for learning/validating
file-level risk-score weights.

Initial target:
  - future_bug_churn_<H>windows

For each cutoff revision t, the script aggregates all signals observed up to t
and pairs them with the future bug churn over the next H temporal windows.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import compute_file_risk_scores as risk


BUG_KEYWORDS = re.compile(
    r"\b(fix|bug|hotfix|patch|defect|issue|error|correct|regress)\b",
    re.IGNORECASE,
)


@dataclass
class RevisionPoint:
    revision_number: int
    commit_hash: str
    commit_date: str
    rev_dir: Path


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  Written: {path}")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Written: {path}")


def _load_revision_points(interp_root: Path) -> Tuple[Dict[str, Any], List[RevisionPoint]]:
    ts = risk._load_json(interp_root / "timeseries.json")
    rev_root = interp_root / "SINGLE_REVISION_ANALYSIS_DATA"
    rev_dirs = {
        int(p.name.split("_", 1)[0]): p
        for p in rev_root.iterdir()
        if p.is_dir() and p.name.split("_", 1)[0].isdigit()
    }
    by_num = {int(r["revision_number"]): r for r in ts.get("revisions", [])}
    chronological: List[RevisionPoint] = []
    for rev_num in sorted(by_num.keys(), reverse=True):  # oldest -> newest
        if rev_num not in rev_dirs:
            continue
        r = by_num[rev_num]
        chronological.append(
            RevisionPoint(
                revision_number=rev_num,
                commit_hash=r["commit_hash"],
                commit_date=r["commit_date"],
                rev_dir=rev_dirs[rev_num],
            )
        )
    return ts, chronological


def _scan_range_stats(git_root: Path, older: str, newer: str) -> Dict[str, Any]:
    commit_range = f"{older}..{newer}"
    total_churn: Dict[str, int] = collections.defaultdict(int)
    bug_churn: Dict[str, int] = collections.defaultdict(int)
    cochange: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))

    result = subprocess.run(
        ["git", "log", "--format=%H %s", commit_range],
        cwd=str(git_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    commits = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) != 2:
            continue
        chash, subject = parts
        commits.append((chash, subject, sorted({m.lower() for m in BUG_KEYWORDS.findall(subject)})))

    for chash, _subject, matched in commits:
        diff = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "-r", "--numstat", chash],
            cwd=str(git_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        changed_files: List[str] = []
        for line in diff.stdout.splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                added = int(parts[0]) if parts[0] != "-" else 0
                deleted = int(parts[1]) if parts[1] != "-" else 0
            except ValueError:
                continue
            fname = risk._normalise_path(parts[2])
            if not fname:
                continue
            churn = added + deleted
            total_churn[fname] += churn
            if matched:
                bug_churn[fname] += churn
            changed_files.append(fname)
        for i, fa in enumerate(changed_files):
            for fb in changed_files[i + 1:]:
                if fa == fb:
                    continue
                cochange[fa][fb] += 1
                cochange[fb][fa] += 1

    return {
        "range": commit_range,
        "total_churn": dict(total_churn),
        "bug_churn": dict(bug_churn),
        "cochange": {fa: dict(row) for fa, row in cochange.items()},
    }


def _count_hidden_cochange(
    cumulative_cochange: Dict[str, Dict[str, int]],
    dsm_edges: set[tuple[str, str]],
    threshold: int,
) -> Dict[str, int]:
    out: Dict[str, int] = collections.defaultdict(int)
    for fa, row in cumulative_cochange.items():
        count = 0
        for fb, occurrences in row.items():
            if occurrences < threshold:
                continue
            if (fa, fb) in dsm_edges or (fb, fa) in dsm_edges:
                continue
            count += 1
        if count:
            out[fa] = count
    return out


def build_dataset(
    interp_root: Path,
    *,
    git_root: Path,
    horizon_windows: int,
    co_change_threshold: int,
) -> Dict[str, Any]:
    timeseries, chronological = _load_revision_points(interp_root)
    if len(chronological) < horizon_windows + 2:
        raise RuntimeError("Not enough revisions for requested horizon.")

    # Precompute adjacent window stats oldest -> newest
    windows: List[Dict[str, Any]] = []
    for i in range(len(chronological) - 1):
        older = chronological[i].commit_hash
        newer = chronological[i + 1].commit_hash
        windows.append(_scan_range_stats(git_root, older, newer))

    anti_pattern_total: Dict[str, int] = collections.defaultdict(int)
    hotspot_fanin_sum: Dict[str, float] = collections.defaultdict(float)
    rev_presence: Dict[str, int] = collections.defaultdict(int)
    total_churn_cum: Dict[str, int] = collections.defaultdict(int)
    bug_churn_cum: Dict[str, int] = collections.defaultdict(int)
    scc_count: Dict[str, int] = collections.defaultdict(int)
    anti_patterns_seen: Dict[str, set] = collections.defaultdict(set)
    anti_pattern_revision_count: Dict[str, int] = collections.defaultdict(int)
    anti_pattern_instance_load: Dict[str, int] = collections.defaultdict(int)
    anti_pattern_type_counts: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )
    cumulative_cochange: Dict[str, Dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int)
    )

    rows: List[Dict[str, Any]] = []

    for cutoff_idx, rev in enumerate(chronological):
        payload = risk._load_json(rev.rev_dir / "OutputData" / "interpretation_payload.json")
        rev_output_dir = rev.rev_dir / "OutputData"

        # Structural accumulation for current revision
        for row in payload.get("structural_hotspots", {}).get("rows", []):
            fname = risk._normalise_path(row.get("Filename", ""))
            if not fname:
                continue
            try:
                hotspot_fanin_sum[fname] += float(row.get("FanIn", 0) or 0)
            except (ValueError, TypeError):
                pass
            rev_presence[fname] += 1

        clsx_stats = risk._collect_antipattern_stats(rev_output_dir)
        if clsx_stats:
            for fname, ap_info in clsx_stats.items():
                anti_pattern_total[fname] += len(ap_info["types"])
                anti_patterns_seen[fname].update(ap_info["types"])
                anti_pattern_revision_count[fname] += 1
                anti_pattern_instance_load[fname] += int(ap_info["instance_count"])
                for ap_type, count in dict(ap_info["type_counts"]).items():
                    anti_pattern_type_counts[fname][ap_type] += int(count)
        else:
            for row in payload.get("dangerous_files", {}).get("rows", []):
                fname = risk._normalise_path(row.get("Filename", ""))
                if not fname:
                    continue
                anti_pattern_total[fname] += 1
                anti_patterns_seen[fname].add("DV8-DangerousFile")
                anti_pattern_revision_count[fname] += 1
                anti_pattern_instance_load[fname] += 1
                anti_pattern_type_counts[fname]["DV8-DangerousFile"] += 1

        # SCCs for current revision
        matrix_json = rev.rev_dir / "OutputData" / "dsm" / "matrix.json"
        dsm_edges = set()
        if matrix_json.exists():
            matrix_data = risk._load_json(matrix_json)
            nodes, edges = risk._parse_matrix_cells(matrix_data)
            dsm_edges = set(risk._load_dsm_edges(matrix_json))
            for comp in risk._tarjan_scc(nodes, edges):
                for fpath in comp:
                    if fpath:
                        scc_count[fpath] += 1

        # Add the previous window so features are "history up to current revision"
        if cutoff_idx > 0:
            win = windows[cutoff_idx - 1]
            for f, v in win["total_churn"].items():
                total_churn_cum[f] += v
            for f, v in win["bug_churn"].items():
                bug_churn_cum[f] += v
            for fa, row in win["cochange"].items():
                for fb, v in row.items():
                    cumulative_cochange[fa][fb] += v

        # Need enough future windows for target
        if cutoff_idx > len(chronological) - horizon_windows - 1:
            continue

        cochange_partners = _count_hidden_cochange(cumulative_cochange, dsm_edges, co_change_threshold)
        scored = risk.compute_risk_scores(
            anti_pattern_total=anti_pattern_total,
            hotspot_fanin_sum=hotspot_fanin_sum,
            rev_presence=rev_presence,
            total_churn=total_churn_cum,
            bug_churn=bug_churn_cum,
            scc_count=scc_count,
            cochange_partners=cochange_partners,
            weights=risk.DEFAULT_WEIGHTS,
            anti_patterns_seen=anti_patterns_seen,
            anti_pattern_revision_count=anti_pattern_revision_count,
            anti_pattern_instance_load=anti_pattern_instance_load,
            anti_pattern_type_counts=anti_pattern_type_counts,
        )
        score_by_file = {r["file"]: r for r in scored}

        future_bug: Dict[str, int] = collections.defaultdict(int)
        for future_w in windows[cutoff_idx:cutoff_idx + horizon_windows]:
            for f, v in future_w["bug_churn"].items():
                future_bug[f] += v

        all_files = set(score_by_file) | set(future_bug)
        for f in sorted(all_files):
            rec = score_by_file.get(f)
            signals = rec["signals"] if rec else {
                "anti_pattern_count": 0,
                "anti_pattern_revision_count": 0,
                "anti_pattern_instance_load": 0,
                "anti_pattern_diversity": 0,
                "hotspot_fanin_score": 0.0,
                "rev_count": 0,
                "total_churn": 0,
                "bug_churn_total": 0,
                "scc_membership_count": 0,
                "co_change_without_dep": 0,
                "anti_pattern_relative_to_mean": 0.0,
                "anti_pattern_zscore": 0.0,
            }
            rows.append(
                {
                    "repo": timeseries.get("repo", "unknown"),
                    "cutoff_revision_number": rev.revision_number,
                    "cutoff_commit_hash": rev.commit_hash,
                    "cutoff_commit_date": rev.commit_date,
                    "horizon_windows": horizon_windows,
                    "file": f,
                    "current_risk_score": rec["risk_score"] if rec else 0.0,
                    "anti_pattern_count": signals["anti_pattern_count"],
                    "anti_pattern_revision_count": signals["anti_pattern_revision_count"],
                    "anti_pattern_instance_load": signals["anti_pattern_instance_load"],
                    "anti_pattern_diversity": signals["anti_pattern_diversity"],
                    "hotspot_fanin_score": signals["hotspot_fanin_score"],
                    "rev_count": signals["rev_count"],
                    "total_churn": signals["total_churn"],
                    "bug_churn_total": signals["bug_churn_total"],
                    "scc_membership_count": signals["scc_membership_count"],
                    "co_change_without_dep": signals["co_change_without_dep"],
                    f"future_bug_churn_{horizon_windows}w": future_bug.get(f, 0),
                    f"future_bug_flag_{horizon_windows}w": 1 if future_bug.get(f, 0) > 0 else 0,
                }
            )

    return {
        "meta": {
            "repo": timeseries.get("repo", "unknown"),
            "revision_count": len(chronological),
            "horizon_windows": horizon_windows,
            "co_change_threshold": co_change_threshold,
            "target": f"future_bug_churn_{horizon_windows}w",
            "description": "Rolling cutoff dataset: features from history up to t, target from next horizon windows.",
        },
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a rolling calibration dataset for risk-score learning.")
    ap.add_argument("interp_root", help="Path to INPUT_INTERPRETATION")
    ap.add_argument("--git-root", required=True, help="Path to git repository root")
    ap.add_argument("--horizon-windows", type=int, default=3, help="Future horizon in adjacent temporal windows (default: 3)")
    ap.add_argument("--co-change-threshold", type=int, default=risk.CO_CHANGE_THRESHOLD)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    interp_root = Path(args.interp_root).resolve()
    git_root = Path(args.git_root).resolve()
    out_dir = interp_root / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        interp_root,
        git_root=git_root,
        horizon_windows=args.horizon_windows,
        co_change_threshold=args.co_change_threshold,
    )
    rows = dataset["rows"]
    _write_json(out_dir / f"future_bug_churn_{args.horizon_windows}w_dataset.json", dataset)
    _write_csv(out_dir / f"future_bug_churn_{args.horizon_windows}w_dataset.csv", rows)
    print(f"  Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
