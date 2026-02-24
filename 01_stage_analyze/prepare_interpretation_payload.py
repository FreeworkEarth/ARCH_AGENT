"""
Prepare a compact, model-friendly payload for temporal revision interpretation.

Inputs:
  - Path to a DV8 OutputData folder for a specific revision
    (e.g., .../temporal_analysis_alltime_2013-06_to_2025-11/01_rev/OutputData)
  - Optional repo root to compute churn (defaults to the revision checkout root)

Outputs:
  - interpretation_payload.json : structured summary (metrics, anti-patterns, churn)
  - interpretation_payload.md   : short Markdown snippet for an LLM

We avoid HTML parsing; instead we surface the structured artifacts DV8 already emits.
"""

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_metrics(output_dir: Path) -> Dict:
    mdir = output_dir / "metrics"
    data = {
        "m_score": load_json(mdir / "m-score.json"),
        "propagation_cost": load_json(mdir / "propagation-cost.json"),
        "decoupling_level": load_json(mdir / "decoupling-level.json"),
        "independence_level": load_json(mdir / "independence-level.json"),
    }
    # Prefer new filename; fall back to legacy.
    exact = load_json(mdir / "mscore_from_dsm_drh_components.json")
    if not exact:
        exact = load_json(mdir / "mscore_exact_components.json")
    if exact:
        # Keep only actionable pieces; no rollups/averages.
        mscore_entry = {
            "mscore": exact.get("mscore"),
            "mscore_percentage": exact.get("mscore_percentage"),
            "pc_from_dsm": exact.get("pc_from_dsm"),
            "dl_approx": exact.get("dl_approx"),
        }

        modules = exact.get("module_details", [])
        if modules:
            data["m_score_modules"] = modules
            # Layer -> module count summary to help spot where modules sit.
            layer_counts = {}
            for m in modules:
                layer = m.get("layer")
                if layer is None:
                    continue
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            mscore_entry["layer_module_counts"] = layer_counts
            mscore_entry["module_count"] = len(modules)

        data["m_score_from_dsm_drh"] = mscore_entry
    return data


def strip_tags(txt: str) -> str:
    # Very small helper to remove HTML tags and collapse whitespace.
    txt = re.sub(r"<[^>]+>", "", txt)
    return " ".join(txt.split())


def load_dangerous_files(output_dir: Path) -> Dict:
    """Extract the "Most Dangerous Files" table from analysis-summary.html if present.

    We avoid heavy dependencies; this uses a simple regex-based table grab. The
    output is a dict with headers and rows (list of dicts keyed by header).
    """

    # Find the analysis-summary.html (sometimes nested under a named folder).
    html_path = None
    candidate = output_dir / "dv8-analysis-result" / "analysis-summary.html"
    if candidate.exists():
        html_path = candidate
    else:
        found = list(output_dir.glob("**/dv8-analysis-result/analysis-summary.html"))
        if found:
            html_path = found[0]
    if not html_path:
        return {}

    try:
        html = html_path.read_text(errors="ignore")
    except Exception:
        return {}

    tables = re.findall(r"<table[^>]*>.*?</table>", html, re.IGNORECASE | re.DOTALL)
    chosen = None
    headers: List[str] = []
    rows: List[Dict[str, str]] = []

    for tbl in tables:
        hdrs = [strip_tags(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", tbl, re.IGNORECASE | re.DOTALL)]
        if not hdrs:
            continue
        # Heuristic: pick the first table whose headers include "File".
        if any("file" in h.lower() for h in hdrs):
            chosen = tbl
            headers = hdrs
            break

    if not chosen or not headers:
        return {}

    # Extract data rows.
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", chosen, re.IGNORECASE | re.DOTALL):
        cells = [strip_tags(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.IGNORECASE | re.DOTALL)]
        if len(cells) != len(headers):
            # Skip header or malformed rows.
            continue
        row_dict = {headers[i]: cells[i] for i in range(len(headers))}
        rows.append(row_dict)

    if not rows:
        return {}

    return {
        "source": str(html_path.relative_to(output_dir)),
        "headers": headers,
        "rows": rows,
    }


def find_matrix_json(output_dir: Path) -> Path | None:
    direct = output_dir / "dsm" / "matrix.json"
    if direct.exists():
        return direct
    found = list(output_dir.glob("**/dsm/matrix.json"))
    return found[0] if found else None


def find_drh_json(output_dir: Path) -> Path | None:
    direct = output_dir / "dv8-analysis-result" / "dsm" / "drh-clustering.json"
    if direct.exists():
        return direct
    found = list(output_dir.glob("**/dv8-analysis-result/dsm/drh-clustering.json"))
    return found[0] if found else None


def load_structural_hotspots(output_dir: Path, top_n: int = 10) -> Dict:
    """Fallback hotspot list from DSM matrix (fan-in/out weights)."""
    matrix_path = find_matrix_json(output_dir)
    if not matrix_path:
        return {}
    matrix = load_json(matrix_path)
    variables = matrix.get("variables") or []
    cells = matrix.get("cells") or []
    if not isinstance(variables, list) or not isinstance(cells, list) or not variables:
        return {}
    fan_in = [0.0] * len(variables)
    fan_out = [0.0] * len(variables)
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        src = cell.get("src")
        dest = cell.get("dest")
        values = cell.get("values") or {}
        if not isinstance(src, int) or not isinstance(dest, int):
            continue
        if src < 0 or dest < 0 or src >= len(variables) or dest >= len(variables):
            continue
        total = 0.0
        if isinstance(values, dict):
            for v in values.values():
                try:
                    total += float(v)
                except Exception:
                    continue
        if total <= 0:
            continue
        fan_out[src] += total
        fan_in[dest] += total

    rows = []
    for i, name in enumerate(variables):
        total = fan_in[i] + fan_out[i]
        rows.append(
            {
                "Filename": str(name),
                "FanIn": round(fan_in[i], 2),
                "FanOut": round(fan_out[i], 2),
                "TotalWeight": round(total, 2),
            }
        )
    rows.sort(key=lambda r: r.get("TotalWeight", 0.0), reverse=True)
    rows = rows[:top_n]
    if not rows:
        return {}
    return {
        "source": str(matrix_path.relative_to(output_dir)),
        "headers": ["Filename", "FanIn", "FanOut", "TotalWeight"],
        "rows": rows,
    }


def load_antipattern_summary(output_dir: Path) -> Tuple[List[Dict], Dict]:
    summary_path = output_dir / "arch-issue" / "anti-pattern-summary.csv"
    rows: List[Dict] = []
    counts: Dict[str, int] = {}
    if not summary_path.exists():
        return rows, counts
    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            typ = r.get("type")
            if typ:
                counts[typ] = counts.get(typ, 0) + 1
    return rows, counts


def git_churn(
    repo_root: Path, top_n: int = 10, commit_range: str = None
) -> List[Tuple[str, int]]:
    if not (repo_root / ".git").exists():
        return []
    try:
        cmd = ["git", "log"]
        if commit_range:
            cmd.append(commit_range)
        cmd += ["--numstat", "--pretty=format:--", "--no-renames"]
        # If a commit range is provided, we want churn over the full interval.
        # Otherwise, keep this lightweight by limiting to recent history.
        if not commit_range:
            cmd += ["-50"]
        res = subprocess.run(
            cmd, cwd=repo_root, capture_output=True, text=True, check=True
        )
        churn: Dict[str, int] = {}
        for line in res.stdout.splitlines():
            if line == "--" or not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            add, dele, path = parts
            try:
                a = 0 if add == "-" else int(add)
                d = 0 if dele == "-" else int(dele)
            except ValueError:
                continue
            churn[path] = churn.get(path, 0) + a + d
        return sorted(churn.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    except Exception:
        return []


def git_recent_commits(repo_root: Path, top_n: int = 5) -> List[Dict[str, str]]:
    if not (repo_root / ".git").exists():
        return []
    try:
        cmd = [
            "git",
            "log",
            f"-{top_n}",
            "--pretty=format:%h|%ad|%s",
            "--date=iso",
        ]
        res = subprocess.run(
            cmd, cwd=repo_root, capture_output=True, text=True, check=True
        )
        commits = []
        for line in res.stdout.splitlines():
            if "|" not in line:
                continue
            sh, dt, subj = line.split("|", 2)
            commits.append({"hash": sh, "date": dt, "subject": subj.strip()})
        return commits
    except Exception:
        return []


def load_issue_type_map(path: Path) -> Dict[str, str]:
    """
    Load an issue->type mapping from JSON.

    Supports both:
      - { "PROJ-123": "bug", ... }
      - { "meta": {...}, "issues": { "PROJ-123": "bug", ... } }
    """
    if not path or not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}

    if isinstance(raw, dict):
        if "issues" in raw and isinstance(raw["issues"], dict):
            return {str(k): str(v) for k, v in raw["issues"].items()}
        # Flat mapping
        flat = {}
        for k, v in raw.items():
            if isinstance(v, str) and "-" in k:
                flat[str(k)] = v
        return flat
    return {}


_JIRA_ISSUE_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
_GITHUB_ISSUE_RE = re.compile(r"(?<![A-Z0-9])#(\d+)\b")


def git_typed_issue_churn(
    repo_root: Path,
    commit_range: str,
    issue_type_map: Dict[str, str],
    top_n: int = 20,
) -> Dict:
    """
    Compute typed churn/frequency over a commit range by classifying commits via issue IDs in messages.

    - For Apache projects like Zeppelin, commit messages usually contain JIRA keys (e.g., ZEPPELIN-1234).
    - If no key is present or unknown, commits are categorized as "unknown".

    Returns a compact summary suitable for LLM interpretation.
    """
    if not commit_range or not (repo_root / ".git").exists():
        return {}

    # type -> file -> churn (added+deleted)
    churn_by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    freq_by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    commit_counts: Dict[str, int] = defaultdict(int)
    issue_counts: Dict[str, int] = defaultdict(int)
    issues_seen: Dict[str, set] = defaultdict(set)

    # Parse entire range in a single command for speed.
    # We mark commits with a sentinel line, then read their numstat lines.
    try:
        cmd = [
            "git",
            "log",
            commit_range,
            "--numstat",
            "--pretty=format:__COMMIT__%n%H%x09%s",
            "--no-renames",
        ]
        res = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
    except Exception:
        return {}

    cur_type = "unknown"
    cur_issue_keys: List[str] = []
    in_commit = False

    def classify_issue_keys(keys: List[str]) -> str:
        if not keys:
            return "unknown"
        # Choose the first key that maps to a known type; otherwise unknown.
        for k in keys:
            t = issue_type_map.get(k)
            if t:
                return t
        return "unknown"

    for line in res.stdout.splitlines():
        if line.startswith("__COMMIT__"):
            in_commit = True
            cur_type = "unknown"
            cur_issue_keys = []
            continue

        if in_commit and "\t" in line and len(line.split("\t", 1)[0]) >= 7:
            # Commit header line: "<sha>\t<subject>"
            sha, subj = line.split("\t", 1)
            jira_keys = _JIRA_ISSUE_RE.findall(subj or "")
            gh_nums = _GITHUB_ISSUE_RE.findall(subj or "")
            # Normalize GitHub refs to a pseudo key so they can be mapped if desired.
            gh_keys = [f"GH#{n}" for n in gh_nums]
            cur_issue_keys = jira_keys + gh_keys
            cur_type = classify_issue_keys(cur_issue_keys)
            commit_counts[cur_type] += 1
            for k in cur_issue_keys:
                issues_seen[cur_type].add(k)
            continue

        # Numstat line: "<added>\t<deleted>\t<path>"
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        add, dele, path = parts
        try:
            a = 0 if add == "-" else int(add)
            d = 0 if dele == "-" else int(dele)
        except ValueError:
            continue
        churn_by_type[cur_type][path] += a + d
        freq_by_type[cur_type][path] += 1

    for t, s in issues_seen.items():
        issue_counts[t] = len(s)

    def top_k(d: Dict[str, int]) -> List[List]:
        return [[k, v] for k, v in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]

    out = {
        "range": commit_range,
        "commit_count": dict(commit_counts),
        "issue_key_count": dict(issue_counts),
        "churn_total": {t: sum(m.values()) for t, m in churn_by_type.items()},
        "churn_top": {t: top_k(m) for t, m in churn_by_type.items()},
        "freq_total": {t: sum(m.values()) for t, m in freq_by_type.items()},
        "freq_top": {t: top_k(m) for t, m in freq_by_type.items()},
    }
    return out


def build_markdown(payload: Dict) -> str:
    lines = []
    meta = payload.get("meta", {})
    lines.append(f"# Revision Summary: {meta.get('name','(unknown)')}")
    if meta:
        lines.append(
            f"- Commit: {meta.get('commit','?')} | Date: {meta.get('date','?')} | Repo: {meta.get('repo','?')}"
        )
    metrics = payload.get("metrics", {})
    lines.append("## Metrics")
    for key, obj in metrics.items():
        if key == "m_score_from_dsm_drh":
            lines.append(f"- m_score_from_dsm_drh: {obj.get('mscore_percentage')}")
            if obj.get("layer_module_counts"):
                lm = obj["layer_module_counts"]
                lines.append("  - layer_module_counts:")
                for k in sorted(lm):
                    lines.append(f"    - layer {k}: {lm[k]} modules")
        elif isinstance(obj, dict):
            val = (
                obj.get("mScore")
                or obj.get("propagationCost")
                or obj.get("decouplingLevel")
                or obj.get("independenceLevel")
            )
            if val is None:
                continue
            lines.append(f"- {key}: {val}")
        else:
            # Ignore lists (e.g., m_score_modules) in the Markdown summary.
            continue
    # Do not list all modules in Markdown to keep it short.

    ap_counts = payload.get("anti_pattern_counts", {})
    if ap_counts:
        lines.append("## Anti-Patterns (counts)")
        for k, v in ap_counts.items():
            lines.append(f"- {k}: {v}")

    dangerous = payload.get("dangerous_files", {})
    if dangerous and dangerous.get("rows"):
        lines.append("## Dangerous files (from DV8 analysis-summary.html)")
        hdrs = dangerous.get("headers", [])
        for row in dangerous["rows"][:5]:
            if isinstance(row, dict):
                summary = ", ".join(f"{h}: {row.get(h,'')}" for h in hdrs)
                lines.append(f"- {summary}")
    else:
        structural = payload.get("structural_hotspots", {})
        if structural and structural.get("rows"):
            lines.append("## Structural hotspots (from DSM matrix)")
            hdrs = structural.get("headers", [])
            for row in structural["rows"][:5]:
                if isinstance(row, dict):
                    summary = ", ".join(f"{h}: {row.get(h,'')}" for h in hdrs)
                    lines.append(f"- {summary}")

    churn = payload.get("churn_top", [])
    if churn:
        lines.append("## Top churn files (recent git log)")
        for path, c in churn:
            lines.append(f"- {path}: {c} LOC changed")

    recent = payload.get("git_recent", [])
    if recent:
        lines.append("## Recent commits")
        for c in recent:
            lines.append(f"- {c.get('hash','')} | {c.get('date','')} | {c.get('subject','')}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Prepare interpretation payload from a DV8 OutputData folder."
    )
    ap.add_argument("output_dir", help="Path to revision OutputData folder")
    ap.add_argument(
        "--repo-root",
        help="Repo root for churn (defaults to parent of OutputData)",
        default=None,
    )
    ap.add_argument(
        "--git-root",
        help="Optional separate git root for churn/commits (e.g., original clone)",
        default=None,
    )
    ap.add_argument(
        "--meta-name", help="Friendly revision name", default=None
    )
    ap.add_argument(
        "--meta-commit", help="Commit hash for metadata", default=None
    )
    ap.add_argument(
        "--meta-date", help="Commit date for metadata", default=None
    )
    ap.add_argument(
        "--meta-repo", help="Repo identifier", default=None
    )
    ap.add_argument(
        "--churn-range",
        help="Optional git range (e.g., old..new) to scope churn to this interval",
        default=None,
    )
    ap.add_argument(
        "--issue-map",
        help="Optional issue type map JSON (JIRA/GitHub) to compute typed churn over churn-range",
        default=None,
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else out_dir.parent
    git_root = Path(args.git_root).resolve() if args.git_root else repo_root

    metrics = load_metrics(out_dir)
    ap_rows, ap_counts = load_antipattern_summary(out_dir)
    churn_top = git_churn(git_root, top_n=10, commit_range=args.churn_range)
    git_recent = git_recent_commits(git_root, top_n=5)
    drh_png = out_dir / "plots" / "drh_layers.png"
    drh_pdf = out_dir / "plots" / "drh_layers.pdf"
    dangerous_files = load_dangerous_files(out_dir)
    structural_hotspots = load_structural_hotspots(out_dir, top_n=10)
    matrix_json = find_matrix_json(out_dir)
    drh_json = find_drh_json(out_dir)
    issue_type_map = load_issue_type_map(Path(args.issue_map)) if args.issue_map else {}
    typed_issue_churn = git_typed_issue_churn(
        git_root, args.churn_range, issue_type_map, top_n=20
    ) if issue_type_map and args.churn_range else {}

    payload = {
        "meta": {
            "name": args.meta_name,
            "commit": args.meta_commit,
            "date": args.meta_date,
            "repo": args.meta_repo,
        },
        "organization_this_file": {
            "first_section": "DV8 modularity metrics",
            "second_section": "M-Score in-depth from DRH",
            "third_section": "churn top files (and typed churn if available)",
            "fourth_section": "locations of plots and supplemental files",
            "fifth_section": "dangerous_files from DV8 analysis summary",
        },
        "metrics": metrics,
        "anti_pattern_counts": ap_counts,
        "anti_pattern_rows": ap_rows,
        "churn_top": churn_top,
        "churn_range": args.churn_range,
        "issue_typed_churn": typed_issue_churn,
        "git_recent": git_recent,
        "drh_plots": {
            "png": str(drh_png.relative_to(out_dir)) if drh_png.exists() else None,
            "pdf": str(drh_pdf.relative_to(out_dir)) if drh_pdf.exists() else None,
        },
        "dsm_paths": {
            "matrix_json": str(matrix_json.relative_to(out_dir)) if matrix_json else None,
            "drh_clustering_json": str(drh_json.relative_to(out_dir)) if drh_json else None,
        },
        "dangerous_files": dangerous_files,
        "structural_hotspots": structural_hotspots,
    }

    # Lightweight schema/help for future readers/LLM prompts.
    schema_txt = out_dir / "interpretation_payload_schema.txt"
    schema_txt.write_text(
        """interpretation_payload.json structure (per revision):
meta: name, commit, date, repo
metrics: DV8 metrics (m-score, pc, dl, il) plus m_score_from_dsm_drh {mscore, modules[], layer_module_counts}
anti_pattern_counts/rows: counts and raw CSV rows
churn_top: top files by git log LOC change (optionally scoped to churn_range)
issue_typed_churn: typed churn/frequency over churn_range using issue IDs in commit messages (optional)
git_recent: last few commits (hash, date, subject)
drh_plots: paths to DRH layer plot png/pdf (relative to OutputData)
dsm_paths: matrix.json + drh-clustering.json paths (relative to OutputData)
dangerous_files: table scraped from analysis-summary.html (headers, rows)
structural_hotspots: fallback hotspot list from DSM matrix if no analysis-summary.html
""",
        encoding="utf-8",
    )

    json_path = out_dir / "interpretation_payload.json"
    md_path = out_dir / "interpretation_payload.md"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(md_path, "w") as f:
        f.write(build_markdown(payload))

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
