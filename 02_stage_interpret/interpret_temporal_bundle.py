#!/usr/bin/env python3
"""
interpret_temporal_bundle.py

End-to-end interpretation for a temporal analysis folder:
  - Generates per-transition DRH-diff reports using interpret_drh_diff.py
  - Writes a single combined report with an overall summary at the top

Example:
  python3 interpret_temporal_bundle.py \
    --temporal-root ../REPOS/zeppelin/temporal_analysis_alltime_2013-06_to_2025-11 \
    --model deepseek-r1:32b
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Stage 3 LLM backend (optional import — falls back to direct subprocess if not available)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "03_stage_query"))
    from llm_backend import LLMBackend as _LLMBackend
    _HAS_LLM_BACKEND = True
except ImportError:
    _HAS_LLM_BACKEND = False


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def normalize_model_name(model: str) -> str:
    return (model or "model").replace("/", "_").replace(":", "_")


def extract_managers_special(report_text: str) -> str:
    if not report_text:
        return ""
    m = re.search(r"^## Comprehensive Summary\s*$", report_text, re.M)
    if not m:
        return ""
    start = m.start()
    # End at next H2
    m2 = re.search(r"^##\s+", report_text[m.end() :], re.M)
    end = (m.end() + m2.start()) if m2 else len(report_text)
    return report_text[start:end].strip()


def _auto_num_ctx(prompt: str, answer_budget: int = 2048) -> int:
    """Return the smallest Ollama-supported power-of-2 context that fits prompt + answer_budget.

    Uses 3 chars/token (conservative — DeepSeek-R1 thinking tokens also consume context).
    answer_budget should include room for chain-of-thought tokens, not just the final answer.
    """
    prompt_tokens = len(prompt) // 3
    needed = prompt_tokens + answer_budget
    print(f"  [ctx] Prompt: {len(prompt):,} chars (~{prompt_tokens:,} tokens) + answer_budget={answer_budget} → need {needed:,} tokens", flush=True)
    for ctx in (4096, 8192, 16384, 32768):
        if ctx >= needed:
            return ctx
    return 32768


def query_ollama(model: str, prompt: str, timeout_s: int = 1800, num_ctx: int = 0) -> str:
    """Call LLM via LLMBackend (supports Ollama/vLLM/API via env vars).
    If num_ctx=0 (default), automatically selects the smallest sufficient context window."""
    if num_ctx == 0:
        num_ctx = _auto_num_ctx(prompt)
    est_tokens = len(prompt) // 4
    print(f"  [LLM] Prompt size: {len(prompt):,} chars (~{est_tokens:,} tokens) | num_ctx={num_ctx} | timeout={timeout_s}s | model={model}")
    if _HAS_LLM_BACKEND:
        llm = _LLMBackend(model=model, num_ctx=num_ctx)
        return llm.generate(prompt, timeout_s=timeout_s)
    # Fallback: direct subprocess (original behaviour)
    res = subprocess.run(["ollama", "run", model, prompt], capture_output=True, text=True, timeout=timeout_s)
    out = (res.stdout or "").strip()
    if out:
        return out
    return (res.stderr or "").strip()


def strip_thinking_and_fences(text: str) -> str:
    if not text:
        return text
    # Strip <think>...</think> blocks (Deepseek-R1 style)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip leading Thinking... block (older Ollama style)
    lines = text.splitlines()
    if lines and lines[0].strip().lower().startswith("thinking"):
        end_idx = None
        for i, ln in enumerate(lines[:200]):
            if "done thinking" in (ln or "").lower():
                end_idx = i
                break
        if end_idx is not None:
            text = "\n".join(lines[end_idx + 1 :]).lstrip()
    # Strip ``` fences (first fenced block)
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()
    return text.strip()


def load_bug_commit_context(temporal_root: Path, max_commits: int = 80) -> tuple[str, int, str]:
    """Load bug-linked commit context for Q&A.

    Preference order:
    1. issue_map.json commit_log if populated
    2. bug_churn_commits_llm_review.json
    3. bug_churn_commits.json
    """
    issue_map_path = next(
        (p for p in [
            temporal_root / "INPUT_INTERPRETATION" / "issue_map.json",
            temporal_root / "issue_map.json",  # legacy location
        ] if p.exists()),
        None,
    )
    if issue_map_path is not None:
        try:
            issue_data = json.loads(issue_map_path.read_text(encoding="utf-8"))
            summaries_map = issue_data.get("summaries", {})
            issues_map = issue_data.get("issues", {})
            commit_log = issue_data.get("commit_log", [])
            jira_re = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")
            bug_kw = re.compile(r"\b(fix|bug|hotfix|patch|defect|regress)\b", re.IGNORECASE)
            bug_commits = []
            for c in commit_log:
                subj = c.get("subject", "")
                jira_refs = jira_re.findall(subj)
                is_bug_jira = any(issues_map.get(k) == "bug" for k in jira_refs)
                is_bug_kw = bool(bug_kw.search(subj))
                if is_bug_jira or is_bug_kw:
                    issue_title = ""
                    for k in jira_refs:
                        if k in summaries_map:
                            issue_title = f" ({summaries_map[k]})"
                            break
                    bug_commits.append(f"- [{c.get('date','')[:10]}] {c.get('hash','')[:8]} {subj}{issue_title}")
            if bug_commits:
                return "\n".join(bug_commits[:max_commits]), len(bug_commits), "issue_map.json"
        except Exception:
            pass

    for rel in [
        "INPUT_INTERPRETATION/bug_churn_commits_llm_review.json",
        "INPUT_INTERPRETATION/bug_churn_commits.json",
    ]:
        path = temporal_root / rel
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data.get("commits", [])
            if not rows:
                continue
            lines: List[str] = []
            for c in rows[:max_commits]:
                subject = c.get("subject", "")
                commit_hash = (c.get("commit_hash") or "")[:8]
                matched = ", ".join(c.get("matched_keywords", []) or [])
                churn = c.get("total_churn", 0)
                extra = []
                if "llm_label" in c:
                    extra.append(f"llm={c.get('llm_label')}")
                if "heuristic_label" in c:
                    extra.append(f"heuristic={c.get('heuristic_label')}")
                if matched:
                    extra.append(f"kw={matched}")
                extra.append(f"churn={churn}")
                lines.append(f"- {commit_hash} {subject} ({', '.join(extra)})")
            return "\n".join(lines), len(rows), path.name
        except Exception:
            continue

    return "", 0, "none"


def load_hotspot_data(temporal_root: Path, top_n: int = 15) -> str:
    """Load ranked hotspot ROI data from the most recent revision's hotspot/ folder.

    DV8's arch-report with runHotspot=on produces active-hotspot-roi.csv with columns
    including FileName, Fi (fan-in), Churn, and a combined score. Returns a formatted
    string for LLM context injection.
    """
    hotspot_csv = None

    # Try SINGLE_REVISION_ANALYSIS_DATA first (mirrored copy from backfill)
    single_rev_root = temporal_root / "INPUT_INTERPRETATION" / "SINGLE_REVISION_ANALYSIS_DATA"
    if single_rev_root.exists():
        rev_dirs = sorted(d for d in single_rev_root.iterdir() if d.is_dir())
        if rev_dirs:
            newest_rev = rev_dirs[0]  # 01_* is most recent
            for candidate in [
                newest_rev / "OutputData" / "hotspot" / "active-hotspot-roi.csv",
                newest_rev / "OutputData" / "hotspot" / "active-hotspot-index.csv",
            ]:
                if candidate.exists():
                    hotspot_csv = candidate
                    break

    # Fallback: look directly in data_repositories/ (new pipeline layout)
    if hotspot_csv is None:
        data_repos = temporal_root / "data_repositories"
        if data_repos.is_dir():
            rev_dirs_dr = sorted(
                d for d in data_repos.iterdir() if d.is_dir() and d.name[:2].isdigit()
            )
            if rev_dirs_dr:
                newest_dr = rev_dirs_dr[0]
                for candidate in [
                    newest_dr / "OutputData" / "hotspot" / "active-hotspot-roi.csv",
                    newest_dr / "OutputData" / "hotspot" / "active-hotspot-index.csv",
                ]:
                    if candidate.exists():
                        hotspot_csv = candidate
                        break

    if hotspot_csv is None:
        return ""

    try:
        rows = []
        with open(hotspot_csv, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception:
        return ""

    if not rows:
        return ""

    # Detect score column — DV8 uses different names across versions
    headers = list(rows[0].keys())
    score_col = next(
        (c for c in ["NormScore", "Score", "HotspotScore", "NormChurn", "Churn"] if c in headers),
        None,
    )
    fi_col = next((c for c in ["Fi", "NormFi", "CurrFi"] if c in headers), None)
    churn_col = next((c for c in ["Churn", "NormChurn", "CurrChurn"] if c in headers), None)
    name_col = next((c for c in ["FileName", "File", "Name"] if c in headers), None)

    if not name_col:
        return ""

    # Sort by score descending if available, otherwise keep DV8's ordering (already ranked)
    if score_col:
        try:
            rows.sort(key=lambda r: float(r.get(score_col, 0) or 0), reverse=True)
        except Exception:
            pass

    lines_out = [f"## TOP HOTSPOTS BY ROI (most recent revision — files where high structural coupling × high change frequency = highest refactoring payoff):"]
    lines_out.append(f"(CSV: {hotspot_csv.name}, columns: {', '.join(headers)})\n")
    for i, row in enumerate(rows[:top_n], 1):
        name = (row.get(name_col) or "?").split("/")[-1]
        parts = [f"{i}. {name}"]
        if fi_col and row.get(fi_col):
            parts.append(f"Fi={row[fi_col]}")
        if churn_col and row.get(churn_col):
            parts.append(f"Churn={row[churn_col]}")
        if score_col and row.get(score_col):
            parts.append(f"Score={row[score_col]}")
        lines_out.append("  " + " | ".join(parts))

    return "\n".join(lines_out) + "\n\n"


def _load_ap_summary(rev_dir: Path) -> dict:
    """Read anti-pattern-summary.csv for a revision. Returns {type: instance_count}."""
    csv_path = rev_dir / "OutputData" / "arch-issue" / "anti-pattern-summary.csv"
    if not csv_path.exists():
        return {}
    import csv as _csv
    result = {}
    try:
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in _csv.DictReader(f):
                t = row.get("Type", "").strip()
                try:
                    result[t] = int(row.get("InstanceCount", 0) or 0)
                except ValueError:
                    pass
    except Exception:
        pass
    return result


def _load_ap_members(rev_dir: Path) -> dict:
    """Load per-instance member sets for all anti-pattern types.
    Returns {ap_type: {instance_id: set(file_basename)}}.
    ap_type is the folder name e.g. 'clique', 'modularity-violation'.
    """
    import csv as _csv
    arch_issue = rev_dir / "OutputData" / "arch-issue"
    result: dict = {}
    if not arch_issue.is_dir():
        return result
    for ap_dir in arch_issue.iterdir():
        if not ap_dir.is_dir():
            continue
        ap_type = ap_dir.name
        result[ap_type] = {}
        for inst_dir in ap_dir.iterdir():
            if not inst_dir.is_dir():
                continue
            inst_id = inst_dir.name
            members: set = set()
            csv_path = inst_dir / f"{inst_id}-clsx_files.csv"
            if csv_path.exists():
                try:
                    with open(csv_path, encoding="utf-8", newline="") as f:
                        for row in _csv.DictReader(f):
                            fp = row.get("file_path", "").strip()
                            if fp:
                                members.add(fp.split("/")[-1])
                except Exception:
                    pass
            if members:
                result[ap_type][inst_id] = members
    return result


def _load_drh_modules(rev_dir: Path) -> dict:
    """Load DRH cluster assignment. Returns {file_basename: module_name}."""
    drh_json = rev_dir / "OutputData" / "dv8-analysis-result" / "dsm" / "drh-clustering.json"
    if not drh_json.exists():
        return {}
    result = {}
    try:
        data = json.loads(drh_json.read_text(encoding="utf-8"))
        def _walk(node, module):
            if node.get("@type") == "group":
                m = node.get("name", module)
                for child in node.get("nested", []):
                    _walk(child, m)
            elif node.get("@type") == "item":
                fp = node.get("name", "")
                result[fp.split("/")[-1]] = module
        for top in data.get("structure", []):
            _walk(top, top.get("name", "?"))
    except Exception:
        pass
    return result


def load_evidence_graph_evolution(temporal_root: Path, max_transitions: int = 3) -> str:
    """Load pairwise evidence graph diffs and summarise dependency evolution.

    Returns a formatted string for LLM context injection covering the most recent
    max_transitions revision pairs (newest first). Key signals: fan-in/fan-out growth,
    newly added dependencies, SCC (cyclic dependency) changes.
    """
    interp_root = temporal_root / "INPUT_INTERPRETATION"
    index_path = interp_root / "evidence_graph_diff_index.json"
    if not index_path.exists():
        return ""
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not index:
        return ""

    # Sort by new_revision_number ascending (1 = most recent pair first)
    index_sorted = sorted(index, key=lambda x: x.get("new_revision_number", 99))

    sections = []
    for entry in index_sorted[:max_transitions]:
        diff_path = interp_root / entry["path"]
        if not diff_path.exists():
            continue
        try:
            d = json.loads(diff_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        new_n = entry["new_revision_number"]
        old_n = entry["old_revision_number"]
        new_dir = entry.get("new_dir", f"rev{new_n}")
        old_dir = entry.get("old_dir", f"rev{old_n}")
        # Extract date hint from dir name (e.g. 01_commons-io_23022026_1058 → 23022026)
        new_date = new_dir.split("_")[2] if new_dir.count("_") >= 2 else new_dir
        old_date = old_dir.split("_")[2] if old_dir.count("_") >= 2 else old_dir

        # Load anti-pattern instance counts for both revisions
        single_rev_root = interp_root / "SINGLE_REVISION_ANALYSIS_DATA"
        data_repos = temporal_root / "data_repositories"
        def _find_rev_dir(dir_name: str) -> Path | None:
            for base in [single_rev_root, data_repos]:
                if base.is_dir():
                    candidates = [p for p in base.iterdir() if p.is_dir() and p.name == dir_name]
                    if candidates:
                        return candidates[0]
                    # fallback: prefix match
                    prefix = dir_name.split("_")[0] + "_"
                    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
                    if candidates:
                        return sorted(candidates)[0]
            return None
        ap_new = _load_ap_summary(_find_rev_dir(new_dir)) if _find_rev_dir(new_dir) else {}
        ap_old = _load_ap_summary(_find_rev_dir(old_dir)) if _find_rev_dir(old_dir) else {}

        node_delta = d.get("nodes", {}).get("delta", 0)
        edges_new = d.get("edges", {}).get("new", {}).get("total_weight", 0)
        edges_old = d.get("edges", {}).get("old", {}).get("total_weight", 0)
        edge_delta = edges_new - edges_old

        lines = [f"### Transition rev{new_n} ({new_date}) ← rev{old_n} ({old_date})"]
        lines.append(f"  Files: delta={node_delta:+d}  |  Total edge weight: {edges_new:.0f} (was {edges_old:.0f}, delta={edge_delta:+.0f})")

        # Fan-in growth = files becoming MORE depended upon (centralisation risk)
        fi_top = d.get("fan_in_delta_top", [])[:8]
        if fi_top:
            lines.append("  TOP FAN-IN GROWTH (files gaining most incoming dependencies → centralisation risk):")
            for item in fi_top:
                node = item.get("node", "?").split("/")[-1]
                delta = item.get("delta", 0)
                lines.append(f"    {node}: +{delta}")

        # Fan-out growth = files coupling to more targets (fragility risk)
        fo_top = d.get("fan_out_delta_top", [])[:8]
        if fo_top:
            lines.append("  TOP FAN-OUT GROWTH (files coupling to more targets → fragility risk):")
            for item in fo_top:
                node = item.get("node", "?").split("/")[-1]
                delta = item.get("delta", 0)
                lines.append(f"    {node}: +{delta}")

        # Sample of newly added edges
        added = d.get("edges_added_sample", [])[:5]
        if added:
            lines.append("  SAMPLE NEW DEPENDENCIES ADDED:")
            for e in added:
                src = e.get("src", "?").split("/")[-1]
                dest = e.get("dest", "?").split("/")[-1]
                kind = e.get("kind", "?")
                lines.append(f"    {src} --[{kind}]--> {dest}")

        # SCC (cyclic dependency) changes
        scc_new = d.get("scc_new", {})
        scc_old = d.get("scc_old", {})
        scc_count_new = scc_new.get("scc_count", 0)
        scc_count_old = scc_old.get("scc_count", 0)
        largest_new = scc_new.get("largest_scc_size", 0)
        largest_old = scc_old.get("largest_scc_size", 0)
        lines.append(
            f"  SCCs (cyclic deps): {scc_count_old} → {scc_count_new} ({scc_count_new - scc_count_old:+d})"
            f"  |  Largest SCC: {largest_old} → {largest_new} files ({largest_new - largest_old:+d})"
        )

        # Load per-instance member sets and DRH modules for cross-referencing
        new_rev_dir = _find_rev_dir(new_dir)
        old_rev_dir = _find_rev_dir(old_dir)
        ap_members_new = _load_ap_members(new_rev_dir) if new_rev_dir else {}
        ap_members_old = _load_ap_members(old_rev_dir) if old_rev_dir else {}
        drh_new = _load_drh_modules(new_rev_dir) if new_rev_dir else {}
        drh_old = _load_drh_modules(old_rev_dir) if old_rev_dir else {}

        # Collect all files that had significant fan-in or fan-out growth this transition
        changed_files = set()
        for item in (fi_top + fo_top):
            f = item.get("node", "").split("/")[-1]
            if f and item.get("delta", 0) > 0:
                changed_files.add(f)

        # Anti-pattern instance count changes + per-file cross-reference
        all_ap_types = sorted(set(ap_new) | set(ap_old) | set(ap_members_new) | set(ap_members_old))
        if all_ap_types:
            lines.append("  ANTI-PATTERN CHANGES THIS TRANSITION:")
            for t in all_ap_types:
                n_count = ap_new.get(t, 0)
                o_count = ap_old.get(t, 0)
                delta_ap = n_count - o_count
                flag = " [NEW INSTANCES]" if delta_ap > 0 else (" [RESOLVED]" if delta_ap < 0 else "")
                lines.append(f"    {t}: {o_count} → {n_count} ({delta_ap:+d}){flag}")

                # For new instances: list which fan-in/fan-out growers are in them
                if delta_ap > 0 and t in ap_members_new and t in ap_members_old:
                    old_inst_ids = set(ap_members_old[t].keys())
                    for inst_id, members in ap_members_new[t].items():
                        if inst_id not in old_inst_ids:
                            # This is a genuinely new instance
                            growers_in_inst = changed_files & members
                            if growers_in_inst:
                                lines.append(f"      NEW {t} Instance {inst_id} contains dep-growth files: {', '.join(sorted(growers_in_inst))}")
                            else:
                                lines.append(f"      NEW {t} Instance {inst_id} ({len(members)} files)")

        # DRH module changes for files that grew in dependencies
        if drh_new and drh_old and changed_files:
            module_changes = []
            for f in sorted(changed_files):
                m_new = drh_new.get(f)
                m_old = drh_old.get(f)
                if m_new and m_old and m_new != m_old:
                    module_changes.append(f"    {f}: {m_old} → {m_new}")
            if module_changes:
                lines.append("  DRH MODULE CHANGES (files that gained deps and moved module):")
                lines.extend(module_changes)

        sections.append("\n".join(lines))

    if not sections:
        return ""

    header = "## DEPENDENCY EVOLUTION (pairwise evidence-graph diffs, newest transition first):\n"
    return header + "\n\n".join(sections) + "\n\n"


def load_antipattern_groups(temporal_root: Path, max_groups_per_type: int = 5):
    """Build a structured group-level summary from the most recent revision's arch-issue CSVs.

    For each anti-pattern type (Clique, MVG, Package-Cycle, Unhealthy-Inheritance) returns the
    top instances sorted by group-level combined bug_churn, with member file names and per-file
    risk scores — so the LLM can answer professor-style "which parts of the system..." questions.

    Returns a tuple (formatted_str, raw_groups_list) where raw_groups_list is a list of dicts
    {ap_type, id, size, total_bug_churn, members} for use by build_refactoring_context().
    """
    # Find the most recent revision folder (rev 01 = newest = lowest sort key)
    single_rev_root = temporal_root / "INPUT_INTERPRETATION" / "SINGLE_REVISION_ANALYSIS_DATA"
    if not single_rev_root.exists():
        return "", []
    rev_dirs = sorted(single_rev_root.iterdir())
    if not rev_dirs:
        return "", []
    newest_rev = rev_dirs[0]  # 01_* is most recent
    arch_issue_root = newest_rev / "OutputData" / "arch-issue"
    if not arch_issue_root.exists():
        return "", []

    # Load per-file risk scores for aggregation
    risk_lookup: Dict[str, Dict] = {}
    risk_json = temporal_root / "INPUT_INTERPRETATION" / "file_risk_scores.json"
    if risk_json.exists():
        try:
            rd = json.loads(risk_json.read_text(encoding="utf-8"))
            for f in rd.get("files", []):
                basename = f["file"].split("/")[-1]
                risk_lookup[basename] = f
        except Exception:
            pass

    # Load total file count for % of system calculation
    total_system_files = 0
    metrics_json = newest_rev / "OutputData" / "metrics" / "all-metrics.json"
    if metrics_json.exists():
        try:
            m = json.loads(metrics_json.read_text(encoding="utf-8"))
            # numberOfFiles is nested under "m-score" or "decoupling-level" sub-dict
            for v in m.values():
                if isinstance(v, dict) and "numberOfFiles" in v:
                    total_system_files = int(v["numberOfFiles"])
                    break
        except Exception:
            pass

    type_map = {
        "clique": "Clique (cyclic dependency — files mutually depend on each other, cannot be changed independently)",
        "modularity-violation": "Modularity Violation (hidden behavioral coupling — files change together despite no structural dependency)",
        "package-cycle": "Package Cycle (circular dependency across packages — prevents hierarchical decomposition)",
        "unhealthy-inheritance": "Unhealthy Inheritance (parent↔child coupling or client depends on both — violates Liskov/DIP)",
    }

    sections: List[str] = []
    all_top_groups: Dict[str, list] = {}
    for ap_type, ap_desc in type_map.items():
        ap_dir = arch_issue_root / ap_type
        if not ap_dir.exists():
            continue
        instance_dirs = sorted(
            [d for d in ap_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name
        )
        groups = []
        for inst_dir in instance_dirs:
            files_csv = inst_dir / f"{inst_dir.name}-clsx_files.csv"
            if not files_csv.exists():
                # CSV missing — try to generate it on-the-fly from the .dv8-clsx binary
                clsx_bin = inst_dir / f"{inst_dir.name}-clsx.dv8-clsx"
                if clsx_bin.exists():
                    try:
                        import sys as _sys_exp
                        import importlib.util as _ilu
                        _exp_path = Path(__file__).parent.parent / "01_stage_analyze" / "export_dv8_binary_files.py"
                        if not _exp_path.exists():
                            _exp_path = Path(__file__).parent / "export_dv8_binary_files.py"
                        _spec = _ilu.spec_from_file_location("export_dv8_binary_files", _exp_path)
                        _mod = _ilu.module_from_spec(_spec)
                        _spec.loader.exec_module(_mod)
                        _mod.export_clsx(clsx_bin)
                    except Exception as _e:
                        pass  # give up — skip this instance
                if not files_csv.exists():
                    continue
            try:
                lines = files_csv.read_text(encoding="utf-8").splitlines()
                # Skip header line "file_path"
                members = [l.strip().split("/")[-1] for l in lines[1:] if l.strip()]
            except Exception:
                continue
            if not members:
                continue
            # Aggregate risk signals across members
            total_bug_churn = 0
            total_ap_count = 0
            total_churn_all = 0
            member_risk = []
            for m in members:
                rf = risk_lookup.get(m)
                if rf:
                    s = rf.get("signals", {})
                    total_bug_churn += s.get("bug_churn_total", 0)
                    total_ap_count += s.get("anti_pattern_count", 0)
                    total_churn_all += s.get("total_churn", 0)
                    member_risk.append((m, rf.get("combined_signals", rf.get("risk_score", 0.0))))
            member_risk.sort(key=lambda x: -x[1])
            groups.append({
                "id": inst_dir.name,
                "size": len(members),
                "member_set": set(members),  # full set for accurate overlap
                "total_bug_churn": total_bug_churn,
                "total_nonbug_churn": total_churn_all - total_bug_churn,
                "total_churn_all": total_churn_all,
                "total_ap_count": total_ap_count,
                "top_members": member_risk[:8],
            })

        # Sort groups by combined bug_churn descending
        groups.sort(key=lambda g: -g["total_bug_churn"])
        top_groups = groups[:max_groups_per_type]
        if not top_groups:
            continue
        all_top_groups[ap_type] = top_groups

        # Compute cross-instance overlap using FULL member sets
        # So the LLM knows which instances are subsets vs independent clusters
        for i, g in enumerate(top_groups):
            g["overlap_notes"] = []
            for j, other in enumerate(top_groups):
                if i == j:
                    continue
                shared = g["member_set"] & other["member_set"]
                if not shared:
                    continue
                pct_of_g = round(100 * len(shared) / g["size"])
                pct_of_other = round(100 * len(shared) / other["size"])
                if pct_of_g >= 30 or pct_of_other >= 30:
                    g["overlap_notes"].append(
                        f"{len(shared)} files ({pct_of_g}% of this instance) also appear in Instance {other['id']} ({pct_of_other}% of that instance) — these are distinct co-change patterns discovered by DV8, not duplicates; the shared files are so architecturally central they belong to multiple coupling networks simultaneously"
                    )

        # Refactor guide HTML path (DV8-generated, always at this location)
        refactor_html = arch_issue_root / ap_type / f"refactor-{ap_type}.html"
        refactor_html_str = str(refactor_html) if refactor_html.exists() else None

        lines_out = [f"### {ap_type.upper()} GROUPS — {ap_desc}", f"(showing top {len(top_groups)} of {len(groups)} instances by combined bug_churn in most recent revision)\n"]
        if refactor_html_str:
            lines_out.append(f"  DV8 Refactoring Guide: {refactor_html_str}\n")
        # Collect all members across top groups for unique-to-this-instance annotation
        all_top_members: set = set()
        for g in top_groups:
            all_top_members |= g["member_set"]
        for idx, g in enumerate(top_groups):
            # Files unique to this group (not in any other shown group)
            others_union = set()
            for j, other in enumerate(top_groups):
                if j != idx:
                    others_union |= other["member_set"]
            unique_count = len(g["member_set"] - others_union)

            member_str = ", ".join(
                f"{name}(risk={score:.3f})" for name, score in g["top_members"]
            )
            if g["size"] > 8:
                member_str += f" ... +{g['size'] - 8} more files"
            if total_system_files:
                pct = round(100 * g['size'] / total_system_files, 1)
                if pct > 100:
                    pct_str = f" = {pct}% of {total_system_files} structural files (>100% is valid: this anti-pattern is detected on the merged struct+history DSM which includes cross-revision file paths beyond the current snapshot)"
                else:
                    pct_str = f" = {pct}% of {total_system_files} system files"
            else:
                pct_str = ""

            # Instance folder — prefer mirrored copy (has exported CSV/JSON), fall back to original
            inst_dir_mirrored = arch_issue_root / ap_type / str(g["id"])
            # Check if exported CSVs exist in the mirrored folder
            clsx_csv = inst_dir_mirrored / f"{g['id']}-clsx_files.csv"
            sdsm_csv = inst_dir_mirrored / f"{g['id']}-sdsm_deps.csv"
            instance_files_note = ""
            if clsx_csv.exists() or sdsm_csv.exists():
                csv_names = ", ".join(n for n, exists in [(clsx_csv.name, clsx_csv.exists()), (sdsm_csv.name, sdsm_csv.exists())] if exists)
                instance_files_note = f"\n    Instance data: {inst_dir_mirrored} [{csv_names}]"

            lines_out.append(
                f"  Instance {g['id']}: {g['size']} files total{pct_str} ({unique_count} unique to this instance) | bug_churn={g['total_bug_churn']} | nonbug_churn={g['total_nonbug_churn']} | total_churn={g['total_churn_all']} | combined_ap_count={g['total_ap_count']}"
            )
            lines_out.append(f"    Top members by risk: {member_str}")
            for note in g["overlap_notes"]:
                lines_out.append(f"    ⚠ OVERLAP: {note}")
            if instance_files_note:
                lines_out.append(instance_files_note)
        # Track largest group size in this anti-pattern type for cross-type sorting
        max_size_in_type = max((g["size"] for g in top_groups), default=0)
        sections.append((max_size_in_type, "\n".join(lines_out)))

    if not sections:
        return "", []
    # Sort anti-pattern types by their largest group size descending —
    # the type whose biggest instance affects most of the system comes first.
    sections.sort(key=lambda x: -x[0])

    # Build a pre-sorted global ranking header so the LLM reads groups in the correct order
    # for both "worst by scope" (% descending) and "worst by pain" (bug_churn descending).
    all_groups_flat = []
    for ap_type, grp_list in all_top_groups.items():
        for g in grp_list:
            pct = round(100 * g["size"] / total_system_files, 1) if total_system_files else 0
            all_groups_flat.append((ap_type, g["id"], g["size"], pct, g["total_bug_churn"]))

    # Diversity-aware selection: top-3 globally + top-1 per remaining anti-pattern type
    # so Package Cycle, Clique, Unhealthy Inheritance always appear alongside MV instances
    by_scope_all = sorted(all_groups_flat, key=lambda x: -x[3])
    scope_selected = list(by_scope_all[:3])
    seen_types_s = {x[0] for x in scope_selected}
    for entry in by_scope_all[3:]:
        if entry[0] not in seen_types_s:
            scope_selected.append(entry)
            seen_types_s.add(entry[0])
        if len(scope_selected) >= 8:
            break
    scope_selected.sort(key=lambda x: -x[3])
    by_scope = scope_selected

    by_pain_all = sorted(all_groups_flat, key=lambda x: -x[4])
    pain_selected = list(by_pain_all[:3])
    seen_types_p = {x[0] for x in pain_selected}
    for entry in by_pain_all[3:]:
        if entry[0] not in seen_types_p:
            pain_selected.append(entry)
            seen_types_p.add(entry[0])
        if len(pain_selected) >= 8:
            break
    pain_selected.sort(key=lambda x: -x[4])
    by_pain = pain_selected

    scope_lines = [f"  {ap_type} Instance {gid}: {pct}% of system, bug_churn={bc}"
                   for ap_type, gid, sz, pct, bc in by_scope]
    pain_lines  = [f"  {ap_type} Instance {gid}: bug_churn={bc}, {pct}% of system"
                   for ap_type, gid, sz, pct, bc in by_pain]

    ranking_header = (
        "GLOBAL RANKING (use these pre-sorted lists for your ranked summaries — do NOT re-sort):\n"
        "  WORST BY SCOPE (% of system, descending):\n" + "\n".join(scope_lines) + "\n"
        "  WORST BY PAIN (bug_churn, descending):\n" + "\n".join(pain_lines) + "\n\n"
    )

    # Build raw groups list for build_refactoring_context()
    raw_groups: list = []
    for ap_type, grp_list in all_top_groups.items():
        for g in grp_list:
            raw_groups.append({
                "ap_type": ap_type,
                "id": g["id"],
                "size": g["size"],
                "total_bug_churn": g["total_bug_churn"],
                "arch_issue_root": str(arch_issue_root),
            })

    return (
        "## ANTI-PATTERN GROUP MEMBERSHIPS (most recent revision, grouped by shared structural flaw, ordered by largest instance size):\n\n"
        + ranking_header
        + "\n\n".join(s for _, s in sections) + "\n\n",
        raw_groups,
    )


def build_refactoring_context(top_groups_raw: list, top_n: int = 3) -> str:
    """Load per-instance clsx_files.json + merge_deps.json for the top-churn groups and build
    a structured context string for Q2 refactoring strategy prompts.

    top_groups_raw: list of dicts from load_antipattern_groups() second return value.
    top_n: how many groups to include (sorted by total_bug_churn descending).
    """
    if not top_groups_raw:
        return ""

    sorted_groups = sorted(top_groups_raw, key=lambda g: -g["total_bug_churn"])[:top_n]
    blocks: list = []

    for g in sorted_groups:
        ap_type = g["ap_type"]
        inst_id = g["id"]
        bug_churn = g["total_bug_churn"]
        arch_issue_root = Path(g["arch_issue_root"])
        inst_dir = arch_issue_root / ap_type / str(inst_id)

        # Load member list from clsx_files.json
        clsx_path = inst_dir / f"{inst_id}-clsx_files.json"
        member_files: list = []
        if clsx_path.exists():
            try:
                clsx_data = json.loads(clsx_path.read_text(encoding="utf-8"))
                member_files = clsx_data.get("files", [])
            except Exception:
                pass

        # Load dependency graph from merge_deps.json
        merge_path = inst_dir / f"{inst_id}-merge_deps.json"
        dep_types: list = []
        cluster_files: Dict[str, list] = {}
        if merge_path.exists():
            try:
                merge_data = json.loads(merge_path.read_text(encoding="utf-8"))
                dep_types = merge_data.get("dep_types", [])
                for raw_path in merge_data.get("files", []):
                    # DV8 prefixes each path with a cluster letter (A, B, C, ...)
                    # Files with no letter prefix (or digit prefix) are the CORE cluster —
                    # the dense hub that co-changes with / directly depends on all other clusters.
                    if raw_path and raw_path[0].isupper() and not raw_path[0].isdigit():
                        label = raw_path[0]
                        clean = raw_path[1:]  # strip cluster letter prefix
                    else:
                        label = "CORE"
                        clean = raw_path
                    cluster_files.setdefault(label, []).append(clean.split("/")[-1])
            except Exception:
                pass

        # Semantic meaning of CORE cluster per anti-pattern type
        core_meaning = {
            "modularity-violation": "CORE = files that co-change with ALL other clusters despite no declared structural dependency — the hidden coupling hub",
            "clique": "CORE = the dense mutual-dependency hub (files in CORE directly depend on each other via Call/Import/Extend)",
            "package-cycle": "CORE = files participating in the circular import chain across package boundaries",
            "unhealthy-inheritance": "CORE = base classes / root of the problematic inheritance chain",
        }.get(ap_type, "CORE = the central coupling hub for this instance")

        # Build the block
        lines: list = [f"### {ap_type.upper()} Instance {inst_id} — {g['size']} files, bug_churn={bug_churn}"]

        if cluster_files:
            lines.append("Cluster layout (DV8 merge DSM cluster labels):")
            # Show CORE first, then alphabetical
            ordered_labels = (["CORE"] if "CORE" in cluster_files else []) + sorted(
                k for k in cluster_files if k != "CORE"
            )
            for label in ordered_labels:
                files_in_cluster = cluster_files[label][:8]  # cap at 8 per cluster
                extra = f" (+{len(cluster_files[label]) - 8} more)" if len(cluster_files[label]) > 8 else ""
                lines.append(f"  Cluster {label} ({len(cluster_files[label])} files): {', '.join(files_in_cluster)}{extra}")
            if "CORE" in cluster_files:
                lines.append(f"  Note: {core_meaning}")
        elif member_files:
            lines.append("Members (from clsx_files.json):")
            for f in member_files[:12]:
                lines.append(f"  {f.split('/')[-1]}")
            if len(member_files) > 12:
                lines.append(f"  ... +{len(member_files) - 12} more")

        if dep_types:
            lines.append(f"Dependency types present: {', '.join(dep_types)}")

        # Append DV8 generic refactoring guide path
        refactor_html = arch_issue_root / ap_type / f"refactor-{ap_type}.html"
        if refactor_html.exists():
            lines.append(f"DV8 Refactoring Guide: {refactor_html}")

        blocks.append("\n".join(lines))

    if not blocks:
        return ""

    return "## REFACTORING CONTEXT (per-instance structural dependency data from DV8):\n\n" + "\n\n".join(blocks) + "\n\n"


def _metric_delta_line(old_metrics: Dict[str, Any], new_metrics: Dict[str, Any], key: str) -> str | None:
    ov = old_metrics.get(key)
    nv = new_metrics.get(key)
    if not isinstance(ov, (int, float)) or not isinstance(nv, (int, float)):
        return None
    delta = round(float(nv) - float(ov), 2)
    rel = None
    if ov != 0:
        rel = round((delta / float(ov)) * 100.0, 2)
    base = f"{key}: {round(float(ov), 2)} → {round(float(nv), 2)} (Δ={delta:+.2f} points"
    if rel is not None:
        base += f", {rel:+.2f}% relative"
    base += ")"
    return base


def build_all_transitions_summary(
    timeseries: Dict[str, Any],
    transition_reports: List[Path],
) -> str:
    """Build a chronological summary of ALL transitions for Metric Trajectory context.

    Unlike detect_metric_peaks (which deduplicates to top-N pairs), this function
    emits every transition exactly once, ordered oldest→newest.
    """
    revs = timeseries.get("revisions") or []
    by_num: Dict[int, Dict[str, Any]] = {}
    for r in revs:
        if isinstance(r, dict) and r.get("revision_number"):
            try:
                by_num[int(r["revision_number"])] = r
            except Exception:
                continue

    nums = sorted(by_num.keys())
    if len(nums) < 2:
        return ""

    # Build all transitions as (newer_n, older_n); iterate oldest→newest
    transitions = [(nums[i], nums[i + 1]) for i in range(len(nums) - 1)]
    transitions_chrono = list(reversed(transitions))

    report_by_pair: Dict[Tuple[int, int], Path] = {}
    for p in transition_reports:
        m = re.search(r"_new(\d+)_old(\d+)\.md$", p.name)
        if m:
            report_by_pair[(int(m.group(1)), int(m.group(2)))] = p

    lines = ["## ALL TRANSITIONS (chronological, oldest→newest — use this for Metric Trajectory)"]
    for new_n, old_n in transitions_chrono:
        new_r = by_num.get(new_n) or {}
        old_r = by_num.get(old_n) or {}
        new_m = new_r.get("metrics") or {}
        old_m = old_r.get("metrics") or {}
        new_date = (new_r.get("commit_date") or "")[:10]
        old_date = (old_r.get("commit_date") or "")[:10]
        new_hash = (new_r.get("commit_hash") or "")[:7]
        old_hash = (old_r.get("commit_hash") or "")[:7]

        lines.append(f"\n### rev{old_n} ({old_date} `{old_hash}`) → rev{new_n} ({new_date} `{new_hash}`)")

        for key in ("m-score", "propagation-cost", "decoupling-level", "independence-level"):
            dl = _metric_delta_line(old_m, new_m, key)
            if dl:
                lines.append(f"  - {dl}")

        rp = report_by_pair.get((new_n, old_n))
        if rp:
            try:
                txt = rp.read_text(encoding="utf-8")
                m2 = re.search(r"DRH file count:\s*old=(\d+)\s*→\s*new=(\d+)", txt)
                if m2:
                    old_fc, new_fc = int(m2.group(1)), int(m2.group(2))
                    lines.append(f"  - DRH file count: {old_fc} → {new_fc} (Δ={new_fc - old_fc:+d})")
            except Exception:
                pass

    return "\n".join(lines)


def detect_metric_peaks(
    timeseries: Dict[str, Any],
    temporal_root: Path,
    transition_reports: List[Path],
    top_n: int = 3,
) -> str:
    """Find transitions with the largest metric jumps and cross-reference structural causes.

    For each metric (m-score, propagation-cost), finds the top_n transitions by absolute delta.
    Cross-references with:
    - DRH file count change (module growth/split)
    - Evidence graph fan-in/fan-out top growth
    - Edge delta (total new dependencies added)
    - SCC size change (cycle growth)
    Returns a formatted string for injection into the deterministic overview section.
    """
    revs = timeseries.get("revisions") or []
    by_num: Dict[int, Dict[str, Any]] = {}
    for r in revs:
        if isinstance(r, dict) and r.get("revision_number"):
            try:
                by_num[int(r["revision_number"])] = r
            except Exception:
                continue

    nums = sorted(by_num.keys())  # ascending: rev1=newest first, matching evidence graph index convention
    if len(nums) < 2:
        return ""

    # Build transitions as (newer_n, older_n) — lower number = more recent, matching evidence index
    transitions = [(nums[i], nums[i + 1]) for i in range(len(nums) - 1)]

    # Index DRH report by (new, old)
    report_by_pair: Dict[Tuple[int, int], Path] = {}
    for p in transition_reports:
        m = re.search(r"_new(\d+)_old(\d+)\.md$", p.name)
        if m:
            report_by_pair[(int(m.group(1)), int(m.group(2)))] = p

    def drh_delta(new_n: int, old_n: int) -> Optional[int]:
        rp = report_by_pair.get((new_n, old_n))
        if not rp:
            return None
        try:
            txt = rp.read_text(encoding="utf-8")
        except Exception:
            return None
        m2 = re.search(r"DRH file count:\s*old=(\d+)\s*→\s*new=(\d+)", txt)
        if m2:
            try:
                return int(m2.group(2)) - int(m2.group(1))
            except Exception:
                return None
        return None

    # Load evidence graph diffs indexed by (new_n, old_n)
    interp_root = temporal_root / "INPUT_INTERPRETATION"
    index_path = interp_root / "evidence_graph_diff_index.json"
    ev_by_pair: Dict[Tuple[int, int], Dict] = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for entry in index:
                diff_path = interp_root / entry["path"]
                if diff_path.exists():
                    try:
                        ev_by_pair[(entry["new_revision_number"], entry["old_revision_number"])] = \
                            json.loads(diff_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
        except Exception:
            pass

    # Compute per-transition deltas for each metric
    metric_peaks: Dict[str, List[Tuple[float, int, int]]] = {"m-score": [], "propagation-cost": []}
    for new_n, old_n in transitions:
        new_m = (by_num.get(new_n) or {}).get("metrics") or {}
        old_m = (by_num.get(old_n) or {}).get("metrics") or {}
        for key in ("m-score", "propagation-cost"):
            ov = old_m.get(key)
            nv = new_m.get(key)
            if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
                delta = float(nv) - float(ov)
                metric_peaks[key].append((abs(delta), new_n, old_n))

    # Take top_n by absolute delta across all transitions
    seen_pairs: set = set()
    peak_pairs: List[Tuple[int, int]] = []
    combined = []
    for key, peaks in metric_peaks.items():
        for abs_d, new_n, old_n in sorted(peaks, reverse=True)[:top_n]:
            combined.append((abs_d, new_n, old_n))
    combined.sort(reverse=True)
    for _, new_n, old_n in combined:
        if (new_n, old_n) not in seen_pairs:
            seen_pairs.add((new_n, old_n))
            peak_pairs.append((new_n, old_n))
        if len(peak_pairs) >= top_n:
            break

    if not peak_pairs:
        return ""

    lines = ["## Metric Peaks (largest metric jumps — structural cause cross-reference)"]
    lines.append(
        "Each peak shows: metric delta | DRH file count change | top fan-in/fan-out spikes | edge growth | SCC change"
    )

    for new_n, old_n in peak_pairs:
        new_r = by_num.get(new_n) or {}
        old_r = by_num.get(old_n) or {}
        new_m = new_r.get("metrics") or {}
        old_m = old_r.get("metrics") or {}
        new_date = (new_r.get("commit_date") or "")[:10]
        old_date = (old_r.get("commit_date") or "")[:10]

        lines.append(f"\n### PEAK: rev{old_n} ({old_date}) → rev{new_n} ({new_date})")

        # Metric deltas
        for key in ("m-score", "propagation-cost", "decoupling-level", "independence-level"):
            dl = _metric_delta_line(old_m, new_m, key)
            if dl:
                lines.append(f"  - {dl}")

        # DRH file count
        dd = drh_delta(new_n, old_n)
        if dd is not None:
            direction = "grew" if dd > 0 else "shrank"
            lines.append(f"  - DRH file count {direction} by {dd:+d} files (module growth/split signal)")

        # Evidence graph signals
        ev = ev_by_pair.get((new_n, old_n))
        if ev:
            edges_new = ev.get("edges", {}).get("new", {}).get("total_weight", 0)
            edges_old = ev.get("edges", {}).get("old", {}).get("total_weight", 0)
            edge_delta = edges_new - edges_old
            if edge_delta != 0:
                lines.append(f"  - Dependency edges: {edges_old:.0f} → {edges_new:.0f} (delta={edge_delta:+.0f})")

            fi_top = (ev.get("fan_in_delta_top") or [])[:5]
            if fi_top:
                fi_str = ", ".join(
                    f"{item.get('node','?').split('/')[-1]} (+{item.get('delta',0)})"
                    for item in fi_top
                )
                lines.append(f"  - Top fan-in growth: {fi_str}")

            fo_top = (ev.get("fan_out_delta_top") or [])[:5]
            if fo_top:
                fo_str = ", ".join(
                    f"{item.get('node','?').split('/')[-1]} (+{item.get('delta',0)})"
                    for item in fo_top
                )
                lines.append(f"  - Top fan-out growth: {fo_str}")

            scc_new = ev.get("scc_new", {})
            scc_old = ev.get("scc_old", {})
            scc_delta = scc_new.get("scc_count", 0) - scc_old.get("scc_count", 0)
            largest_delta = scc_new.get("largest_scc_size", 0) - scc_old.get("largest_scc_size", 0)
            if scc_delta != 0 or largest_delta != 0:
                lines.append(
                    f"  - SCC cycles: count delta={scc_delta:+d}, largest cluster delta={largest_delta:+d}"
                )
        else:
            lines.append("  - Evidence graph diff: not available for this transition")

    return "\n".join(lines)


def build_deterministic_overall(
    repo: str,
    temporal_root: Path,
    timeseries: Dict[str, Any],
    transition_reports: List[Path],
    transitions: List[Tuple[int, int]],
) -> str:
    revs = timeseries.get("revisions") or []
    by_num: Dict[int, Dict[str, Any]] = {}
    for r in revs:
        if isinstance(r, dict) and r.get("revision_number"):
            try:
                by_num[int(r["revision_number"])] = r
            except Exception:
                continue

    def drh_file_counts_from_report(p: Path) -> Tuple[int | None, int | None]:
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            return None, None
        m = re.search(r"DRH file count:\s*old=(\d+)\s*→\s*new=(\d+)", txt)
        if not m:
            return None, None
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            return None, None

    report_by_pair: Dict[Tuple[int, int], Path] = {}
    for p in transition_reports:
        m = re.search(r"_new(\d+)_old(\d+)\.md$", p.name)
        if m:
            report_by_pair[(int(m.group(1)), int(m.group(2)))] = p

    lines: List[str] = []
    lines.append("## Overall Summary")
    lines.append(f"- repo: {repo}")
    lines.append(f"- temporal_root: {temporal_root}")
    lines.append("- scope: explain-only (no refactor advice)")

    for new_n, old_n in transitions:
        new_r = by_num.get(new_n) or {}
        old_r = by_num.get(old_n) or {}
        new_m = new_r.get("metrics") or {}
        old_m = old_r.get("metrics") or {}

        parts = []
        for k in ("m-score", "propagation-cost", "decoupling-level", "independence-level"):
            dl = _metric_delta_line(old_m, new_m, k)
            if dl:
                parts.append(dl)

        old_fc = None
        new_fc = None
        rp = report_by_pair.get((new_n, old_n))
        if rp:
            old_fc, new_fc = drh_file_counts_from_report(rp)

        old_date = (old_r.get("commit_date") or "")[:10]
        new_date = (new_r.get("commit_date") or "")[:10]
        old_hash = (old_r.get("commit_hash") or "")[:7]
        new_hash = (new_r.get("commit_hash") or "")[:7]
        headline = f"- transition old=rev{old_n} ({old_date} `{old_hash}`) → new=rev{new_n} ({new_date} `{new_hash}`)"
        msg = new_r.get("commit_message") or ""
        if msg:
            headline += f": {msg.split(chr(10))[0][:80]}"
        lines.append(headline)
        if parts:
            lines.append("  - metrics: " + " | ".join(parts))
        if isinstance(old_fc, int) and isinstance(new_fc, int):
            lines.append(f"  - DRH file count: {old_fc} → {new_fc} (Δ={new_fc - old_fc:+d})")

    lines.append("")
    # Inject metric peaks cross-reference section
    peaks_section = detect_metric_peaks(timeseries, temporal_root, transition_reports, top_n=3)
    if peaks_section:
        lines.append(peaks_section)
        lines.append("")

    lines.append("## Comprehensive Summary")
    lines.append(
        "Across the sampled revisions, DV8 metrics change in lock-step with the DRH structure and evidence graph."
        " Use the per-transition reports to see the concrete DRH layer/module distributions and the matrix-based coupling deltas."
    )
    lines.append("")

    lines.append("## DRH → Metrics Narrative")
    for new_n, old_n in transitions:
        new_r = by_num.get(new_n) or {}
        old_r = by_num.get(old_n) or {}
        new_m = new_r.get("metrics") or {}
        old_m = old_r.get("metrics") or {}

        mscore = _metric_delta_line(old_m, new_m, "m-score")
        pcost = _metric_delta_line(old_m, new_m, "propagation-cost")

        rp = report_by_pair.get((new_n, old_n))
        old_fc = None
        new_fc = None
        if rp:
            old_fc, new_fc = drh_file_counts_from_report(rp)

        old_date = (old_r.get("commit_date") or "")[:10]
        new_date = (new_r.get("commit_date") or "")[:10]
        lines.append(f"- old=rev{old_n} ({old_date}) → new=rev{new_n} ({new_date}):")
        if mscore:
            lines.append(f"  - {mscore}")
        if pcost:
            lines.append(f"  - {pcost}")
        if isinstance(old_fc, int) and isinstance(new_fc, int):
            lines.append(
                f"  - DRH file count increased from {old_fc} to {new_fc}"
                " suggests responsibilities were split across more files; the per-transition DRH diff shows whether that was mostly additive or moved across layers/modules."
            )
        lines.append(
            "  - See the per-transition report for the evidence-graph delta (matrix.json) and the top hotspot/dangerous files driving these metrics."
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("- metrics source: `timeseries.json`")
    lines.append("- per-transition reports: `OUTPUT_INTERPRETATION/<run>/drh_diff_report_<model>_newX_oldY.md`")
    lines.append("- evidence graph diffs: `INPUT_INTERPRETATION/EVIDENCE_GRAPH_DIFF/`")
    return "\n".join(lines).rstrip() + "\n"


def _load_fanin_fanout(rev_folder: Path) -> Dict[str, Dict[str, int]]:
    """Load per-file FanIn/FanOut from interpretation_payload.json in the revision folder."""
    payload_path = rev_folder / "OutputData" / "interpretation_payload.json"
    if not payload_path.exists():
        return {}
    try:
        data = json.loads(payload_path.read_text(encoding="utf-8"))
        rows = (data.get("dangerous_files") or {}).get("rows") or []
        result: Dict[str, Dict[str, int]] = {}
        for r in rows:
            fname = re.sub(r'/self \(File\)$', '', r.get("Filename") or "")
            if fname:
                try:
                    result[fname] = {
                        "FanIn": int(r.get("FanIn", 0)),
                        "FanOut": int(r.get("FanOut", 0)),
                    }
                except (ValueError, TypeError):
                    pass
        return result
    except Exception:
        return {}


def load_worst_transition_drh(temporal_root: Path, timeseries: Dict[str, Any]) -> tuple:
    """Find the worst transition by propagation-cost delta and load its DRH diff report.

    Returns (content_str, file_path_str) — both empty strings if not found.
    Content is trimmed to the most useful sections: Likely Drivers, Structural Summary,
    Notable Layer/Module Moves, and SCC diff (largest SCC members).
    """
    revs = timeseries.get("revisions") or []
    by_num: Dict[int, Dict[str, Any]] = {}
    for r in revs:
        if isinstance(r, dict) and r.get("revision_number"):
            try:
                by_num[int(r["revision_number"])] = r
            except Exception:
                continue
    nums = sorted(by_num.keys())
    if len(nums) < 2:
        return "", ""

    # Find worst transition by propagation-cost delta (largest increase)
    worst_pair: Optional[tuple] = None
    worst_delta = 0.0
    for i in range(len(nums) - 1):
        new_n, old_n = nums[i], nums[i + 1]
        new_m = (by_num.get(new_n) or {}).get("metrics") or {}
        old_m = (by_num.get(old_n) or {}).get("metrics") or {}
        ov = old_m.get("propagation-cost")
        nv = new_m.get("propagation-cost")
        if isinstance(ov, (int, float)) and isinstance(nv, (int, float)):
            delta = float(nv) - float(ov)
            if delta > worst_delta:
                worst_delta = delta
                worst_pair = (new_n, old_n)

    if not worst_pair:
        return "", ""

    new_n, old_n = worst_pair

    # Search all output interpretation subfolders for the matching DRH diff report
    output_root = temporal_root / "OUTPUT_INTERPRETATION"
    if not output_root.exists():
        return "", ""

    candidates = list(output_root.glob(f"*/drh_diff_report_*_new{new_n}_old{old_n}.md"))
    if not candidates:
        return "", ""

    report_path = candidates[0]
    try:
        full_text = report_path.read_text(encoding="utf-8")
    except Exception:
        return "", ""

    # Extract the most useful sections only (keep context window lean)
    useful_sections = []
    # Always include the header (first ~5 lines)
    header_lines = full_text.split("\n")[:6]
    useful_sections.append("\n".join(header_lines))

    for section_title in (
        "## Likely Drivers",
        "### Structural Summary",
        "### Notable Layer Moves",
        "### Notable Module Moves",
        "### Commit Context",
    ):
        idx = full_text.find(section_title)
        if idx == -1:
            continue
        # Find end of section (next ## or ###)
        rest = full_text[idx:]
        end = len(rest)
        for marker in ("\n## ", "\n### "):
            pos = rest.find(marker, len(section_title))
            if pos != -1:
                end = min(end, pos)
        useful_sections.append(rest[:end].strip())

    # Include SCC size info from Metrics section (just the scc lines, not the full JSON)
    metrics_idx = full_text.find("## Metrics & Evidence")
    if metrics_idx != -1:
        metrics_block = full_text[metrics_idx:metrics_idx + 800]
        scc_lines = [ln for ln in metrics_block.split("\n") if "scc" in ln.lower() or "largest_scc" in ln.lower()]
        if scc_lines:
            useful_sections.append("### SCC Change (from Metrics & Evidence)\n" + "\n".join(scc_lines[:6]))

    content = "\n\n".join(useful_sections)
    # Cap at 6000 chars to avoid dominating context
    if len(content) > 6000:
        content = content[:6000] + "\n[...truncated — see full report at path below]"

    new_date = (by_num.get(new_n) or {}).get("commit_date", "")[:10]
    old_date = (by_num.get(old_n) or {}).get("commit_date", "")[:10]
    header = (
        f"## WORST TRANSITION DRH REPORT: rev{old_n} ({old_date}) → rev{new_n} ({new_date})\n"
        f"(Δ propagation-cost = +{worst_delta:.2f} points — the single largest structural degradation step)\n"
        f"Full report: {report_path}\n\n"
    )

    return header + content, str(report_path)


def load_recent_churn_top(temporal_root: Path, top_n: int = 10) -> str:
    """Load churn_top from the most recent revision's interpretation_payload.json.
    Returns a formatted string for Q&A context injection, or empty string if unavailable."""
    srd = temporal_root / "INPUT_INTERPRETATION" / "SINGLE_REVISION_ANALYSIS_DATA"
    if not srd.exists():
        return ""
    rev_dirs = sorted(srd.iterdir())
    if not rev_dirs:
        return ""
    newest_rev = rev_dirs[0]  # sorted: 01_ prefix = most recent
    payload_path = newest_rev / "OutputData" / "interpretation_payload.json"
    if not payload_path.exists():
        return ""
    try:
        data = json.loads(payload_path.read_text(encoding="utf-8"))
        churn_top = data.get("churn_top", [])
        if not churn_top:
            return ""
        meta = data.get("meta", {})
        rev_date = meta.get("date", "most recent revision")
        lines = [f"## MOST ACTIVE FILES (last revision: {rev_date} — total lines changed in that window):"]
        for i, entry in enumerate(churn_top[:top_n], 1):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                fname = str(entry[0]).split("/")[-1]
                lines.append(f"  {i:2d}. {fname}: {entry[1]} lines changed")
        return "\n".join(lines) + "\n"
    except Exception:
        return ""


def _load_clique_count(rev_folder: Path) -> int:
    """Return total distinct files participating in cliques for this revision."""
    csv_path = rev_folder / "OutputData" / "arch-issue" / "anti-pattern-summary.csv"
    if not csv_path.exists():
        return 0
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0].strip() == "Clique":
                    return int(float(row[2]))
    except Exception:
        pass
    return 0


def load_mscore_worst_modules(temporal_root: Path, top_n: int = 5, max_revisions: int = 5) -> str:
    """
    For each revision folder under temporal_root, load mscore_exact_components.json
    and return the top_n worst modules ranked by contribution (= cross_penalty × size_factor).
    Also loads FanIn/FanOut per file and clique count to give a multi-signal refactoring picture.
    Returns a formatted string ready to embed in an LLM prompt.

    max_revisions: only include the most recent N revisions (revision_number=1 is newest).
    Default 5 keeps the context budget manageable (~5000 chars vs 32k for all 36 revisions).
    """
    lines = []
    # Revision folders live in data_repositories/ (new standard) or directly in temporal_root (legacy)
    data_repos = temporal_root / "data_repositories"
    search_root = data_repos if data_repos.exists() else temporal_root
    component_files = sorted(search_root.glob("*/OutputData/metrics/mscore_exact_components.json"))
    rev_map: Dict[int, Path] = {}
    for p in component_files:
        folder_name = p.parts[-4]  # e.g. "01_commons-io_23022026_1058"
        try:
            rev_num = int(folder_name.split("_")[0])
            rev_map[rev_num] = p
        except (ValueError, IndexError):
            continue

    # revision_number=1 is newest — take the lowest revision numbers (most recent)
    all_rev_nums = sorted(rev_map.keys())
    recent_rev_nums = all_rev_nums[:max_revisions]

    for rev_n in recent_rev_nums:
        path = rev_map[rev_n]
        # path: temporal_root/NN_repo/OutputData/metrics/mscore_exact_components.json
        rev_folder = path.parents[2]  # NN_repo/ folder
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        modules = data.get("module_details", [])
        # Sort by contribution (= cross_penalty × size_factor) — captures both violation
        # severity AND module size. A large module with moderate cross_penalty outranks a
        # single file with high cross_penalty because it requires coordinated refactoring.
        worst = sorted(modules, key=lambda m: m.get("contribution", 0), reverse=True)[:top_n]
        if not worst:
            continue

        # Load supplementary signals
        fanin_fanout = _load_fanin_fanout(rev_folder)
        clique_count = _load_clique_count(rev_folder)
        clique_note = f", clique_files={clique_count}" if clique_count > 0 else ""

        lines.append(
            f"\nRevision {rev_n} "
            f"(mscore={data.get('mscore_percentage', 0):.1f}%, "
            f"layers={data.get('num_layers', 0)}, "
            f"modules={data.get('num_modules', 0)}{clique_note}):"
        )
        for m in worst:
            files = m.get("files", [])
            file_str_parts = []
            for fpath in files[:3]:
                fname = re.sub(r'/self \(File\)$', '', fpath) if isinstance(fpath, str) else str(fpath)
                fi_fo = fanin_fanout.get(fname)
                if fi_fo:
                    file_str_parts.append(
                        f"{fname}(FanIn={fi_fo['FanIn']},FanOut={fi_fo['FanOut']})"
                    )
                else:
                    file_str_parts.append(fname)
            if len(files) > 3:
                file_str_parts.append("...")
            file_str = ", ".join(file_str_parts)
            lines.append(
                f"  - Layer {m.get('layer')} / Module {m.get('module')}: "
                f"contribution={m.get('contribution', 0):.4f} "
                f"(cross_penalty={m.get('cross_penalty', 0):.3f}, size={m.get('module_size', 0)} files) | "
                f"files: {file_str}"
            )
    return "\n".join(lines) if lines else "(mscore components not available)"


def build_overall_prompt(repo: str, temporal_root: Path, summaries: List[Tuple[int, int, str]], timeseries: Dict[str, Any], mscore_breakdown: str = "", peaks_section: str = "") -> str:
    return f"""You are an expert software architect.

Hard rules:
- Output MUST be Markdown text (no code blocks).
- Do NOT output hidden reasoning or "thinking".
- Do NOT propose refactor plans or break-even analysis (explain-only).
- Start with: "## Overall Summary"
- Then: "## Comprehensive Summary"
- Then: "## DRH → Metrics Narrative"
- Then: "## Notes"
- When discussing file counts, treat added_files_count/removed_files_count as DRH-diff counters (not git add/delete). Prefer DRH file count (old→new) from the summaries.
- Copy/paste numbers from FACTS; do not invent or recompute values.
- When referencing M-score worst modules: modules are ranked by contribution (= cross_penalty × size_factor). Name specific files from the highest-contribution module as the primary refactoring target. FanIn/FanOut values per file are shown — high FanIn means many dependents break if refactored, high FanOut means the file is fragile and hard to isolate. The revision header shows clique_files= count indicating how many files are in circular dependency clusters (those require coordinated refactoring of the whole cluster). Note if the same module appears across multiple revisions (persistent hotspot). Do NOT rank by cross_penalty alone.
- METRIC PEAKS: The FACTS section contains a "Metric Peaks" block listing the transitions with the largest metric jumps, each with correlated structural signals (DRH file count change, fan-in/fan-out growth, edge delta, SCC change). In your "## Overall Summary" section, for each peak transition: name the metric that spiked, state the delta value, then explain WHAT structural change caused it — e.g. "DRH grew by +12 files (module split), IOUtils.java gained +102 fan-in, and edge count grew by +240, explaining the M-score increase of +3.2". Do NOT just repeat the numbers — explain the causal link between structural change and metric change. If no clear cause is identifiable from the data, say so explicitly. M-score is a purely structural metric — it rises when dependencies grow or cross layer boundaries more, and falls when the hierarchy becomes cleaner.

Context:
- repo: {repo}
- temporal_root: {temporal_root}
- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FACTS (timeseries.json excerpt):
{json.dumps(timeseries, indent=2)[:12000]}

{peaks_section}

M-SCORE WORST MODULES PER REVISION (top 5 by contribution = cross_penalty × size_factor; FanIn/FanOut per file; clique_files count in header):
{mscore_breakdown}

PER-TRANSITION COMPREHENSIVE SUMMARIES (chronological newest→older):
{chr(10).join([f'### new={n} old={o}{chr(10)}{ms}' for n,o,ms in summaries])}
"""


def answer_user_question(model: str, question: str, report_text: str,
                         mscore_breakdown: str = "", timeout_s: int = 900,
                         risk_score_context: str = "", commit_context: str = "",
                         antipattern_groups: str = "", hotspot_data: str = "",
                         evidence_evolution: str = "", peaks_section: str = "",
                         recent_churn_section: str = "", worst_drh_section: str = "",
                         all_transitions_section: str = "",
                         data_quality_warnings: list = None,
                         prior_answer: str = "",
                         refactoring_context: str = "") -> str:
    """Call the LLM to answer a specific user question using the combined report as context.

    Builds a priority context: M-score worst modules first (most useful for file-level questions),
    then risk score table (multi-signal: bug churn + anti-patterns + fan-in + SCC + co-change),
    then anti-pattern group memberships (for professor-style "which parts of the system" questions),
    then Comprehensive Summary blocks (each labelled with commit dates for date-range questions),
    then narrative up to a total of ~28000 chars.
    """
    # No artificial budget cap — include all available data.
    # num_ctx=32768 is fixed for Q&A calls; if the prompt exceeds that, _auto_num_ctx
    # will be called with a large answer_budget to select the right window.
    # Hard per-section limits only exist to prevent a single runaway section from
    # crowding out others; they are set high enough that real data never hits them.

    q_lower_pre = question.lower()

    # 0. Data quality warnings — prepended so LLM knows what's missing
    quality_block = ""
    if data_quality_warnings:
        quality_block = (
            "## DATA QUALITY WARNINGS\n"
            + "\n".join(f"- {w}" for w in data_quality_warnings)
            + "\n\nThese signals are unavailable for this run. "
            "Do NOT invent or estimate values for missing signals. "
            "Explicitly state in your answer which data is missing and that the analysis is partial.\n\n"
        )

    # 1. M-score worst modules
    priority_context = ""
    if mscore_breakdown and "(mscore components not available)" not in mscore_breakdown:
        priority_context = (
            "## M-SCORE WORST MODULES (most recent 5 revisions, top 5 by contribution = cross_penalty × size_factor; FanIn/FanOut per file):\n"
            + mscore_breakdown + "\n\n"
        )

    # 2. File hotspot table — observable signals (bug churn, anti-patterns, SCC, co-change)
    risk_section = ""
    if risk_score_context:
        risk_section = (
            "## FILE HOTSPOT SIGNALS (ranked by bug_churn; observable metrics only — no composite score):\n"
            + risk_score_context + "\n\n"
        )

    # 3. Anti-pattern group memberships
    group_section = antipattern_groups if antipattern_groups else ""

    # 3b. Hotspot ROI data
    hotspot_section = hotspot_data if hotspot_data else ""

    # 3c. Evidence graph evolution
    evolution_section = evidence_evolution if evidence_evolution else ""

    # 3d. Metric peaks cross-reference (pre-computed: worst metric jumps + structural causes)
    peaks_ctx = ""
    if peaks_section:
        peaks_ctx = (
            "## METRIC PEAKS (transitions with largest M-score/propagation-cost jumps,"
            " cross-referenced with structural signals — fan-in/fan-out growth, edge delta, SCC change):\n"
            + peaks_section + "\n\n"
        )

    # 3e. All transitions chronological (for Metric Trajectory — no deduplication)
    all_transitions_ctx = (all_transitions_section + "\n\n") if all_transitions_section else ""

    # 4. Bug-linked commits — cap at 800 commits (~12k chars) to avoid dominating
    commit_section = ""
    if commit_context:
        commit_section = (
            "## BUG-LINKED COMMITS (JIRA-typed or keyword-matched; most recent first):\n"
            + commit_context[:12000] + "\n\n"
        )

    # 5. Comprehensive Summary blocks — all blocks, no truncation
    summary_blocks = re.findall(
        r'(## Comprehensive Summary.*?)(?=\n## |\Z)', report_text, re.DOTALL
    )
    summary_text = "\n\n".join(summary_blocks)

    # 6. Remaining narrative (report text minus summary blocks already included)
    narrative = ""

    recent_churn_ctx = (recent_churn_section + "\n") if recent_churn_section else ""
    worst_drh_ctx = (worst_drh_section + "\n\n") if worst_drh_section else ""
    prior_block = f"## PRIOR ANSWER (Q1 analysis — use this as context for refactoring advice)\n{prior_answer}\n\n" if prior_answer else ""
    context = prior_block + quality_block + priority_context + risk_section + group_section + hotspot_section + evolution_section + peaks_ctx + all_transitions_ctx + worst_drh_ctx + recent_churn_ctx + commit_section + summary_text + ("\n\n" if summary_text else "") + narrative

    q_lower = q_lower_pre

    # Detect "most dangerous / worst files" questions — use risk score data
    is_danger_question = any(k in q_lower for k in [
        "dangerous", "most dangerous", "riskiest", "worst file", "bad file",
        "technical debt", "most debt", "debt", "problematic", "priority",
        "5 most", "top 5", "top file",
    ])

    # Detect hotspot-specific questions — use hotspot ROI data
    is_hotspot_question = any(k in q_lower for k in [
        "hotspot", "hotspots", "most changed", "changed most", "change most often",
        "changed most often", "highest churn", "most churn", "churn the most",
        "roi", "return on investment", "refactoring roi", "worth fixing",
        "most worth", "highest payoff", "fix first", "most impactful fix",
        "changed frequently", "frequently changed",
    ])

    # Detect group/area/part-of-system questions — use anti-pattern group memberships
    is_group_question = any(k in q_lower for k in [
        "part", "parts of", "area", "areas", "subsystem", "module", "group",
        "cluster", "which part", "which area", "system area", "whole subsystem",
        "rapidly", "decreasing in quality", "getting worse", "most rapidly",
        "increase in", "increased in",
        # Natural professor phrasings
        "technical debt", "debt", "anti-pattern", "antipattern",
        "coupling", "coupled", "clique", "modularity violation",
        "package cycle", "inheritance", "structural flaw",
        "component", "components", "layer", "layers",
        "where in", "which section", "hotspot", "hotspots",
        "smell", "code smell", "design flaw",
    ])

    # Detect M-score causality questions
    is_mscore_causality = any(k in q_lower for k in [
        "why did m-score", "why m-score", "why did the m-score", "m-score got worse",
        "m-score deteriorat", "m-score declin", "what caused", "what change caused",
        "linked to", "link to commit", "which commit", "which feature", "which bug",
        "quality decrease", "quality declin", "decreasing in quality", "decrease in quality",
    ])

    # Detect refactoring strategy questions — use per-instance structural data (Q2)
    is_refactoring_strategy = any(k in q_lower for k in [
        "how would you refactor", "how to refactor", "refactor this",
        "refactoring strategy", "refactoring plan", "how do i fix",
        "how should i refactor", "where to start refactoring",
        "how would i refactor", "concrete refactor",
    ])

    # Detect refactor/structural questions — use M-score module data
    is_files_question = any(k in q_lower for k in [
        "refactor", "specific file", "which file", "file to fix", "file to improve",
        "give me file", "files to", "file i should", "files i should",
    ])

    # Detect evolution/growth questions — use evidence graph diff data
    is_evolution_question = any(k in q_lower for k in [
        "evolution", "over time", "grew", "growth", "increased", "worsened", "got worse",
        "last 3 months", "last 6 months", "recent months", "trend", "changed over",
        "dependency growth", "new dependencies", "coupling increased", "fan-in",
        "fan-out", "scc", "cycl", "more dependen", "gaining depend",
    ])

    if is_refactoring_strategy and refactoring_context:
        prompt = f"""You are a software architect providing a prioritized, iterative refactoring plan based on DV8 structural analysis.

Hard rules:
- Output Markdown only. No <think> blocks.
- Do NOT invent file names, cluster labels, or dependency types not present in REFACTORING CONTEXT below.
- Every action must cite specific files by name, cluster labels (A/B/C/CORE), and dependency types (Call, Extend, Import, etc.) from REFACTORING CONTEXT.
- Output EXACTLY 3 to 5 prioritized actions total — NOT one section per instance. Rank across ALL instances by expected bug_churn reduction (highest impact first).
- Frame each action as one discrete step: do it, re-run DV8, measure if M-score or propagation-cost improved.

CORE cluster semantics (read before writing):
- In a modularity-violation: CORE = files that co-change with ALL other clusters despite no declared structural dependency
- In a clique: CORE = the dense mutual-dependency hub (every file in CORE directly depends on the others via Call/Import/Extend)
- In a package-cycle: CORE = files forming the circular import chain
- In an unhealthy-inheritance: CORE = base classes at the root of the problematic hierarchy

Output this exact format:

## Refactoring Priority List (do in order, re-measure DV8 after each)

### Action 1 — [short descriptive title] | targets: [instance name] | impact: highest
**What to do**: [name the specific files, which cluster they are in, which dep types to cut — e.g. "Extract interface IOReader from IOUtils.java (CORE cluster) so that Cluster A files (DeletingPathVisitor, FileTimes) depend on IOReader instead of IOUtils directly, severing the Call coupling from A→CORE"]
**Why first**: bug_churn=[X] — [explain what makes this the highest-impact first move, citing cluster sizes and dep types]
**How to measure**: re-run DV8 — expect [specific metric to improve, e.g. "modularity-violation Instance 1 shrinks or disappears", "propagation-cost drops"]
**Risk**: [what breaks if done wrong, and one mitigation step]

### Action 2 — ...
[same format]

[3 to 5 actions total]

## What to measure after each step
- Re-run the full pipeline on the same temporal root
- Check: did M-score increase? Did propagation-cost drop? Did any anti-pattern instance disappear or shrink?
- If yes: proceed to next action. If no: the cluster coupling was not fully severed — identify which remaining dep type still links the clusters.

If PRIOR ANSWER is present: reference the worst file and worst group named in Q1's conclusion when choosing action targets — do not repeat Q1's analysis, only build on it.

QUESTION: {question}

{refactoring_context}
{context}
"""

    elif is_group_question and group_section:
        prompt = f"""You are a software architect answering a professor's question about software quality. Answer by grouping files according to their shared structural flaw (anti-pattern instance).

CHURN DEFINITIONS (use these exact terms in your answer):
- bug_churn = lines of code changed in defect-fix commits — measures maintenance cost from bugs
- nonbug_churn = lines of code changed in feature/refactor commits — measures development pressure
- total_churn = bug_churn + nonbug_churn — full change pressure on this group

Hard rules:
- Do NOT output reasoning or <think> blocks.
- Do NOT use markdown tables. Write in flowing prose with bold field labels.
- Do NOT invent numbers. Use only values present in the data below.
- No word limit. Do NOT skip, abbreviate, or merge any group.

ANTI-PATTERN COVERAGE CHECK — do this BEFORE writing anything else:
  1. Scan the data for every distinct anti-pattern type present (modularity-violation, unhealthy-inheritance, clique, package-cycle, etc.).
  2. Every type with ANY instance (even size=1, bug_churn=0) MUST appear as a separate named entry in the ranked lists below AND in TIER 1 or TIER 2.
  3. If you omit any detected anti-pattern type from your answer, your answer is INCOMPLETE and WRONG.
  4. If ALL groups have bug_churn=0, that does NOT excuse omitting any type — list all types by scope descending.

Start with two SHORT ranked lists (one line per entry, no field labels):
  "Worst by scope" — top 5 groups by % of system descending. Each line: bold name, size/%, bug_churn. Mark any group that ALSO appears in the pain list with [HIGH SCOPE + HIGH CHURN].
  "Worst by pain" — top 5 groups by bug_churn descending. Each line: bold name, bug_churn, %. Mark any group that ALSO appears in the scope list with [HIGH SCOPE + HIGH CHURN].
  Keep these lists brief — full details follow.
  CRITICAL: Every distinct anti-pattern TYPE present in the data (modularity-violation, unhealthy-inheritance, clique, package-cycle, etc.) MUST appear at least once across the two lists — even if bug_churn=0 for all. If all groups have bug_churn=0, the pain list equals the scope list; in that case list by scope descending for both, but still include ALL distinct anti-pattern types. Do NOT omit any type.

Then write group details in two tiers:

TIER 1 — groups with HIGH SCOPE AND HIGH CHURN (appear in both ranked lists — large AND painful): write FULL details for each. Do not skip any field:
  **Group name**: anti-pattern type + instance ID
  **Size**: number of files and % of system. If % > 100%: explain that DV8 analyses the full history DSM across all revisions — the co-change coupling network is larger than the current code snapshot, meaning the architectural debt spans the entire project history.
  **Key members**: top 8 files by bug_churn (lines changed in bug-fix commits), each with their individual bug_churn value from the FILE HOTSPOT SIGNALS table. CRITICAL: do NOT use the group's bug_churn total as an individual file's bug_churn — the group total is the sum across all members. Individual per-file bug_churn values are in the FILE HOTSPOT SIGNALS table under the "bug=" column.
  **Churn breakdown**: bug_churn=X, nonbug_churn=X, total_churn=X. Then one sentence interpreting what this means for THIS specific group — not a generic definition. Good example: "2993 lines changed purely in defect-fix commits means IOUtils, FileUtils and FilenameUtils are touched in virtually every bug in the project." Bad example: "High bug_churn indicates maintenance cost." Be specific to these files.
  **Structural flaw**: 2-3 sentences analysing the co-change pattern of THESE specific files — what does it mean that these specific files change together despite no structural dependency? Why does THAT specific coupling cause ongoing maintenance cost? Do NOT write a generic definition of the anti-pattern type — analyse this instance's members specifically.
  **Refactoring direction**: one concrete action specific to this group (e.g. "Extract the path-manipulation methods from FilenameUtils into a new PathTokenizer class to sever the co-change coupling with IOUtils" — not "reduce coupling").
  **DV8 Refactoring Guide**: include the HTML path for the FIRST instance of each anti-pattern type only. Format: "DV8 Refactoring Guide: <path>"
  If overlap notes are shown: explain that the shared files anchor multiple independent coupling networks simultaneously. Do NOT list overlapping groups again.
  If a file spans DIFFERENT anti-pattern types: flag it as multiply problematic.
  If an instance data path is shown: include it as "Full member list and dependency CSV: <path>"

TIER 2 — groups that appear in only ONE list (large but low churn = structural debt not yet causing bugs; OR small but high churn = concentrated pain): write a SHORT summary for each (3-4 lines max):
  **Group name**: type + instance ID — label as "large/low-churn" or "small/high-churn" as appropriate
  **Size**: files and %. **Key members**: top 3 files. **Churn**: bug_churn only.
  **Why notable**: one sentence on what makes this group specifically worth watching.
  Include DV8 Refactoring Guide path if this is the first instance of that anti-pattern type.
  IMPORTANT: Include ALL groups here whose anti-pattern type was not fully covered in TIER 1 — even if bug_churn=0. A group with bug_churn=0 that represents a distinct anti-pattern type (e.g. unhealthy-inheritance when TIER 1 only covered modularity-violation) MUST appear in TIER 2.

Then write TWO sub-sections under "Files that got worse over time":

**A) Worst cumulative fan-in growth (structural centralisation)** — top 5 files by `fan_in_delta` from the FILE HOTSPOT SIGNALS table, sorted descending. For each: bold filename, `fan_in_delta` value, which anti-pattern instances it belongs to, one-sentence structural meaning (e.g. "became a dependency magnet — every new class now imports it"). This is the primary signal for files becoming structurally worse over time.

**B) Worst single-transition spike** — up to 3 files with the largest fan-in jump in a single transition window from DEPENDENCY EVOLUTION. For each: bold filename, exact delta and which transition (e.g. "gained +104 fan-in in rev2→rev1"), which anti-pattern instances it NEWLY JOINED in that same transition (if any).
Then write a "## Metric Trajectory" section showing ALL transitions in chronological order (oldest→newest). For EACH transition:
- State the revision window (dates)
- State the exact Δ M-score and Δ propagation-cost (use the Δ= values verbatim from ALL TRANSITIONS data — NOT from METRIC PEAKS)
- State DRH file count change
- Name the top 2 files with most fan-in growth that window (with Δ values)
- Note which AP instances were active or grew
- End with one-sentence architectural interpretation

For the WORST transition (largest propagation-cost jump): add a sub-section "#### Why this was the worst transition" using the WORST TRANSITION DRH REPORT data (if present). Explain specifically: which files joined the coupling network, what structural changes drove the metric spike (layer moves, SCC expansion, new dependency edges by type), and what the commit activity shows. Link to the full report: "Full DRH report: <path from WORST TRANSITION DRH REPORT header>".

If ALL TRANSITIONS data is absent, omit the Metric Trajectory section entirely.

Then write a "Worst files overall" section — top 5 files ranked by bug_churn (most lines changed in bug-fix commits). For each file include ALL of these observable signals:
- **bug_churn**: lines changed in bug-fix commits
- **ap_instances**: total anti-pattern instance memberships across revisions (from FILE HOTSPOT SIGNALS)
- **fan_in_delta**: cumulative fan-in growth across ALL transitions — use the `fan_in_delta` value from the FILE HOTSPOT SIGNALS table (e.g. "fan_in_delta=+182"). Do NOT use the absolute fan-in value from DEPENDENCY EVOLUTION for this field.
- **Number of distinct anti-pattern types**: e.g. "4 types: clique, modularity-violation, package-cycle, unhealthy-inheritance"
- **Specific instances**: names of the SPECIFIC anti-pattern instances it belongs to (e.g. "Clique Instance 1, Modularity Violation Instance 1 and 4")
Do NOT mention "risk score" — use observable signals only.

End with a short conclusion: worst group (name, %, bug_churn), worst file (name, bug_churn, fan-in), and one concrete first refactoring step.

If MOST ACTIVE FILES data is present in context: after "Worst files overall", add a short "### Most active in last revision" section — list the top 5 files by total lines changed in the most recent window, with their churn count. Note which of these also appear in anti-pattern groups or have high bug_churn (cross-reference with FILE HOTSPOT SIGNALS). This shows current development pressure, not historical pain.

QUESTION: {question}

{evolution_section}
{context}
"""

    elif is_danger_question and risk_section:
        prompt = f"""You are a software architect. Answer the question below using the FILE HOTSPOT SIGNALS as your primary evidence. Files are ranked by bug_churn (lines changed in bug-fix commits) — this is an observable, empirically grounded signal.

Observable signals used (no composite score — each is independently measurable):
- bug_churn_total: lines changed in bug-fix commits (JIRA-linked or keyword-matched) — primary ranking signal
- ap_instances (anti_pattern_instance_load): cumulative count of DV8 anti-pattern instances this file belonged to across revisions
- scc_membership_count: revisions where file is in a cyclic SCC (circular dependency)
- co_change_without_dep: behaviorally coupled partners with no declared structural dependency

METRIC SEMANTICS — you MUST interpret these for the reader, not just quote the numbers:
- bug_churn_total: total LOC changed in defect-fix commits — directly measures how much bug-fixing work this file has caused
- total_churn: ALL LOC changed (bug + feature + refactor) — the full change pressure on this file over its history
- anti_pattern_instance_load: cumulative count of CONCRETE anti-pattern instances this file belonged to across ALL revisions.
  HOW TO INTERPRET: (a) divide by anti_pattern_revision_count → avg instances per revision; (b) find the dominant type from anti_pattern_type_counts and compute its % of total load; (c) explain what that means.
  Example: "1,098 total instance memberships, avg 29 per revision. 918 of these (84%) are modularity-violation — meaning this file anchors dozens of independent hidden coupling networks across the codebase."
- anti_pattern_count: sum over revisions of distinct anti-pattern types per revision — measures breadth of anti-pattern participation over time, NOT raw instance count
- anti_pattern_diversity: number of distinct anti-pattern types ever seen (e.g. 4 = clique + MVG + PKC + UIH)
- anti_pattern_revision_count: how many analyzed revisions the file had at least one anti-pattern
- co_change_without_dep: how many other files this file co-changes with despite no declared structural dependency

Hard rules:
- Do NOT output reasoning or <think> blocks.
- Do NOT mention "risk_score" — it is not empirically validated. Use observable signals only.
- For each file: state its rank, bug_churn_total (lines changed in bug-fix commits), ap_instances (anti-pattern instance memberships), and fan_in (if available from evolution data).
- For each file include these 4 parts:
  1. **Bug-fix churn**: cite bug_churn_total and total_churn — explain what they mean in plain terms (e.g. "416 lines changed in bug-fix commits = high maintenance burden")
  2. **Anti-pattern breakdown**: list each type with its count from anti_pattern_type_counts; compute the dominant type as a % of anti_pattern_instance_load and interpret it (hidden coupling hub? structural bottleneck? brittle hierarchy?)
  3. **Structural meaning**: explain what the combination of anti-pattern types means for this specific file — not generic boilerplate
  4. **Refactoring direction**: one concrete specific action
- For anti_pattern_instance_load: MUST interpret — avg per revision, dominant type %, architectural meaning
- If both co_change_without_dep and modularity-violation appear, note: co_change_without_dep is the file-level co-change signal; modularity-violation is the DV8 group-level detected instance — related but not identical
- Anti-pattern type meanings:
  - clique = dense mutual dependency, files cannot be changed independently
  - package-cycle = circular dependency across packages, prevents layered decomposition
  - unhealthy-inheritance = brittle hierarchy, LSP/DIP violations
  - modularity-violation = hidden behavioral coupling — files change together despite no structural dependency
  - crossing = cuts across many modules or architectural boundaries
  - unstable-interface = volatile interface propagating changes to many dependents
- Cross-reference M-SCORE WORST MODULES if the file appears there — doubly confirmed structural problem
- No word limit — answer as completely as the data supports. Do not truncate.
- Do NOT invent numbers. Use only values from the data below.

QUESTION: {question}

{context}
"""
    elif is_mscore_causality and commit_section:
        prompt = f"""You are a software architect. Answer the question about WHY the M-score changed over time, linking metric changes to specific commits, bugs, and code changes.

Hard rules:
- Do NOT output reasoning or <think> blocks.
- For each M-score drop (negative Δ): find the time window from the COMPREHENSIVE SUMMARY blocks, then look in the COMMIT CONTEXT for commits in that window. Identify bug-fix commits or feature commits that touch files already in DV8 anti-patterns or SCCs.
- Causality chain: "M-score dropped Δ=-X at transition old=revN → new=revM (DATE). In that window, commit HASH (DATE) added/changed FILE which joined SCC Y / entered anti-pattern Z, increasing propagation cost by +P."
- If a JIRA issue is mentioned in a commit message, cite it and its type (bug/feature).
- Group: distinguish deterioration caused by (a) new features adding coupling vs (b) bug fixes patching heavily-coupled files vs (c) refactoring that accidentally increased coupling.
- Format: 1 paragraph per significant drop event, with commit evidence. Max 500 words.
- Do NOT invent numbers or files not in the data.

QUESTION: {question}

{context}
"""
    elif is_files_question and priority_context:
        prompt = f"""You are a software architect. Output ONLY a numbered list — no paragraphs, no headers, no conclusions, no thinking.

STRICT FORMAT (follow exactly, nothing else):
1. FileName.java — Layer N, contribution=X.XXXX (cross_penalty=Y.YYY, size=Z files), FanIn=A, FanOut=B; [one sentence: why this is the worst]
2. FileName.java — Layer N, contribution=X.XXXX (cross_penalty=Y.YYY, size=Z files), FanIn=A, FanOut=B; [one sentence reason]
3. FileName.java — ...
4. FileName.java — ...
5. FileName.java — ...

Rules:
- Use ONLY files from the M-SCORE WORST MODULES data below.
- Rank by contribution score (highest first). contribution = cross_penalty × size_factor.
- If FanIn/FanOut are shown in the data, include them. If not shown, omit those fields.
- Pick representative file names from the module's files list (prefer the first file listed per module, or pick distinct files across modules).
- Do NOT output anything before item 1 or after item 5.
- Do NOT output reasoning, thinking, headers, summaries, or explanations outside the numbered items.

QUESTION: {question}

M-SCORE WORST MODULES (most recent revision first, ranked by contribution):
{priority_context}
"""
    elif is_hotspot_question and hotspot_section:
        prompt = f"""You are a software architect answering a question about hotspots — files where high structural coupling AND high change frequency converge.

HOTSPOT DEFINITION: A hotspot is a file that is both:
1. Structurally coupled (high fan-in = many files depend on it, so changes propagate widely), AND
2. Frequently changed (high churn = actively modified, so structural coupling is constantly exercised)
The ROI score = coupling × churn — files with the highest ROI score give the most refactoring payoff.

Hard rules:
- Do NOT output reasoning or <think> blocks.
- Rank by ROI Score (highest first).
- For each file state: rank, filename, Fi (fan-in = how many files depend on it), Churn (how many times changed), Score.
- Interpret each metric: explain WHY a high Fi is problematic (blast radius), WHY high churn matters (maintenance cost).
- Explain the combined implication: "This file is changed often AND many files depend on it — every change risks cascading failures."
- Cross-reference with ANTI-PATTERN GROUP MEMBERSHIPS if the hotspot also belongs to a Clique or Modularity Violation — doubly confirmed.
- Provide ONE concrete refactoring suggestion per top file.
- Do NOT invent numbers. Use only values from the data below.

QUESTION: {question}

{hotspot_section}
{group_section}
{risk_section}
"""

    elif is_evolution_question and evolution_section:
        prompt = f"""You are a software architect answering a question about how the architecture EVOLVED over time.

EVOLUTION DATA DEFINITIONS:
- Fan-in growth: a file is gaining more incoming dependencies — more files now depend on it. This increases blast radius: any change to this file ripples to more consumers. HIGH RISK if growing fast.
- Fan-out growth: a file is depending on more other files — increasing its coupling surface. Makes the file harder to test and change in isolation.
- SCC (Strongly Connected Component): a cycle of files that mutually depend on each other. A GROWING SCC means more files are trapped in a circular dependency — any change can cascade unpredictably.
- Edge weight delta: total dependency coupling increased or decreased across the whole system.

Hard rules:
- Do NOT output reasoning or <think> blocks.
- Use DEPENDENCY EVOLUTION as your primary evidence — cite the specific transition (e.g. "rev1←rev2") and the exact delta values.
- For each flagged file explain: what changed (fan-in/fan-out delta), why it is dangerous (blast radius, fragility), and how it compares to prior revisions if multiple transitions show the same file worsening.
- Cross-reference with FILE HOTSPOT SIGNALS and ANTI-PATTERN GROUPS — a file that both grew in dependencies AND belongs to a Clique/Modularity Violation is multiply confirmed as a priority.
- If SCCs grew in count or size, explain what that means: a larger cyclic cluster means more files are architecturally trapped together.
- If METRIC PEAKS data is present: write a "## Metric Trajectory" section showing ALL transitions chronologically (oldest→newest). For each: state the revision window (dates), exact Δ M-score and Δ propagation-cost, DRH file count change, top 2 files with most fan-in growth (Δ values), total new dependency edges, which AP instances were active or grew, and a one-sentence architectural interpretation. For the WORST transition (largest propagation-cost jump): add a "#### Why this was the worst transition" sub-section using the WORST TRANSITION DRH REPORT data — explain which files joined the coupling network, what structural changes drove the spike (layer moves, SCC expansion, new edge types), what the commits show, and link to "Full DRH report: <path>". Omit only if METRIC PEAKS data is absent.
- Rank your findings: worst evolution first. For each: (1) filename, (2) what signal worsened and by how much, (3) architectural implication, (4) one concrete fix.
- Do NOT invent numbers. Use only values from the data below.

QUESTION: {question}

{evolution_section}
{peaks_ctx}
{risk_section}
{group_section}
"""

    else:
        prompt = f"""You are an expert software architect answering a specific question about a repository's architectural evolution.

Hard rules:
- Answer ONLY the question asked. Be direct and concise (max 400 words).
- Use ONLY facts from the report below. Do NOT invent numbers or files not mentioned in the report.
- Do NOT use background knowledge about library versions, release history, or project wikis — cite only what is in the report.
- Do NOT output reasoning or <think> blocks.
- Format: short header, bullet points for evidence, 1-sentence conclusion.
- For date-range questions: find transitions whose dates fall within the range asked and cite their specific metric deltas.
- For "technical debt" questions: combine the MULTI-SIGNAL RISK SCORES (which files have most bugs+churn+anti-patterns) with the M-SCORE WORST MODULES (structural coupling) to identify groups of files that are both structurally bad AND actively causing problems.
- For "most rapidly decreasing in quality" questions: find transitions with the largest negative M-score deltas (look at the transition summaries), identify which files changed in those windows (from COMMIT CONTEXT), and name the specific anti-patterns/SCCs that worsened.
- When citing metric changes, ALWAYS include both absolute delta and percentage.

QUESTION: {question}

REPORT (M-score data first, then risk scores, then transition summaries with dates, then narrative):
{context}
"""
    return query_ollama(model, prompt, timeout_s=timeout_s, num_ctx=_auto_num_ctx(prompt, answer_budget=6000))


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate per-transition DRH interpretation + one combined summary report.")
    ap.add_argument("--temporal-root", required=True, help="Path to temporal_analysis_* folder")
    ap.add_argument("--repo", default=None, help="Path to git repo (default: temporal_root/..)")
    ap.add_argument("--model", default="deepseek-r1:32b", help="Ollama model (default: deepseek-r1:32b)")
    ap.add_argument("--ollama-timeout-s", type=int, default=1800, help="Ollama timeout in seconds (default: 1800)")
    ap.add_argument("--no-llm", action="store_true", help="Generate per-transition prompt files only (no model calls)")
    ap.add_argument("--no-overall", action="store_true", help="Skip overall summary generation")
    ap.add_argument("--llm-overall", action="store_true", help="Also generate an LLM-written overall summary (optional; deterministic summary is always included unless --no-overall)")
    ap.add_argument("--verify", action="store_true", help="Run verifier pass for each per-transition report")
    ap.add_argument("--no-verify", action="store_true", help="Disable verifier pass (default: verify on)")
    ap.add_argument("--user-question", default=None,
                    help="Optional question to answer from the combined report (printed to terminal + saved as USER_ANSWER_*.md)")
    ap.add_argument("--qa-only", action="store_true",
                    help="Skip all per-transition processing and run Q&A only on the most recent existing combined report.")
    args = ap.parse_args()

    temporal_root = Path(args.temporal_root).expanduser().resolve()
    repo_path = Path(args.repo).expanduser().resolve() if args.repo else temporal_root.parent.resolve()

    # Support new location (INPUT_INTERPRETATION/timeseries.json) with legacy fallback
    ts_path = next(
        (p for p in [
            temporal_root / "INPUT_INTERPRETATION" / "timeseries.json",
            temporal_root / "timeseries.json",
        ] if p.exists()),
        temporal_root / "INPUT_INTERPRETATION" / "timeseries.json",  # default for error message
    )
    timeseries = read_json(ts_path)
    revisions = timeseries.get("revisions") or []
    if not revisions:
        if not ts_path.exists():
            hint = ""
            try:
                candidates = list(temporal_root.parent.glob("temporal_analysis*/INPUT_INTERPRETATION/timeseries.json"))
                candidates += list(temporal_root.parent.glob("temporal_analysis*/timeseries.json"))
                if candidates:
                    newest = max(candidates, key=lambda x: x.stat().st_mtime)
                    hint = f" (hint: try {newest.parent.parent if 'INPUT_INTERPRETATION' in str(newest) else newest.parent})"
            except Exception:
                pass
            raise RuntimeError(f"timeseries.json not found in {ts_path}{hint}")
        # File exists but doesn't contain revisions.
        raise RuntimeError(f"No revisions found in {ts_path} (revisions is empty or unreadable)")

    # revisions are newest-first with revision_number starting at 1
    nums = sorted([int(r["revision_number"]) for r in revisions if isinstance(r, dict) and r.get("revision_number")])
    if len(nums) < 2:
        raise RuntimeError("Need at least 2 revisions.")

    interp_root = temporal_root / "OUTPUT_INTERPRETATION"
    interp_root.mkdir(parents=True, exist_ok=True)

    # --- QA-only mode: skip all per-transition work, use existing report ---
    if args.qa_only:
        # Find the most recent combined report across all run subfolders
        existing_reports = sorted(
            interp_root.rglob("temporal_interpretation_report_*.md"),
            key=lambda p: p.stat().st_mtime,
        )
        if not existing_reports:
            print("ERROR: --qa-only specified but no existing temporal_interpretation_report_*.md found.", file=sys.stderr)
            return 1
        combined_path = existing_reports[-1]
        run_folder = combined_path.parent
        model_safe = normalize_model_name(args.model)
        # Reuse existing QA folder for this model if one exists, else create dated one
        existing_qa = sorted(interp_root.glob(f"*_QA_{model_safe}"))
        if existing_qa:
            qa_folder = existing_qa[-1]
        else:
            qa_folder = interp_root / f"{datetime.now().strftime('%y%m%d')}_QA_{model_safe}"
        qa_folder.mkdir(parents=True, exist_ok=True)
        print(f"[qa-only] Using existing report: {combined_path}")
        print(f"[qa-only] Answers will be saved to: {qa_folder / 'USER_ANSWERS.md'}")
        mscore_breakdown = load_mscore_worst_modules(temporal_root, max_revisions=5)
        # Load evidence_evolution first so fan-in deltas can enrich the risk score table
        evidence_evolution = load_evidence_graph_evolution(temporal_root)
        # Jump directly to Q&A section
        risk_score_context = ""
        top_files = []
        risk_json_path = temporal_root / "INPUT_INTERPRETATION" / "file_risk_scores.json"
        if risk_json_path.exists():
            try:
                risk_data = json.loads(risk_json_path.read_text(encoding="utf-8"))
                all_files_rs = risk_data.get("files", [])
                # Re-sort by bug_churn_total DESC (observable signal), tiebreak ap_instances DESC
                all_files_rs.sort(key=lambda x: (
                    -x.get("signals", {}).get("bug_churn_total", 0),
                    -x.get("signals", {}).get("anti_pattern_instance_load", 0),
                ))
                top_files = all_files_rs[:25]
                # Build cumulative fan-in delta per file from evidence_evolution
                # Evidence format: "  TOP FAN-IN GROWTH ...\n    File.java: +104.0\n    ..."
                _fanin_cumul: dict = {}
                _in_fanin_block = False
                for _line in (evidence_evolution or "").splitlines():
                    if "TOP FAN-IN GROWTH" in _line:
                        _in_fanin_block = True
                    elif _in_fanin_block:
                        _fm = re.match(r'^\s{4}(\S+\.(?:java|py|kt|ts|js)):\s+([+-]?\d+(?:\.\d+)?)', _line)
                        if _fm:
                            _fanin_cumul[_fm.group(1)] = _fanin_cumul.get(_fm.group(1), 0) + float(_fm.group(2))
                        elif _line.strip() == "" or _line.startswith("  "):
                            if "TOP FAN-" not in _line and "TOP FAN-IN" not in _line:
                                _in_fanin_block = False
                lines_rs = ["rank | file | bug_churn | ap_instances | fan_in(cumul_delta) | anti_patterns | scc_revisions | co_change | anti_pattern_types"]
                lines_rs.append("---" * 12)
                for f in top_files:
                    s = f.get("signals", {})
                    ap_counts = f.get("anti_pattern_type_counts", {})
                    aps = ", ".join(f"{k}:{v}" for k, v in list(ap_counts.items())[:4])
                    fname_short = f['file'].split('/')[-1]
                    _fi_delta = _fanin_cumul.get(fname_short)
                    fi_str = f"fan_in_delta={_fi_delta:+.0f}" if _fi_delta is not None else "fan_in_delta=n/a"
                    lines_rs.append(
                        f"#{f['rank']:2d} | {fname_short:40s} | "
                        f"bug={s.get('bug_churn_total',0):5d} | ap_instances={s.get('anti_pattern_instance_load',0):3d} | {fi_str} | "
                        f"ap={s.get('anti_pattern_count',0):3d} counts | "
                        f"scc={s.get('scc_membership_count',0):2d} revisions | co={s.get('co_change_without_dep',0):2d} | [{aps}]"
                    )
                risk_score_context = "\n".join(lines_rs)
                print(f"  [Q&A] Loaded risk scores for top {len(top_files)} files")
            except Exception as exc:
                print(f"  [Q&A] WARNING: Could not load risk scores: {exc}")
        commit_context, bug_commit_count, bug_commit_source = load_bug_commit_context(temporal_root)
        print(f"  [Q&A] Loaded {bug_commit_count} bug-linked commits from {bug_commit_source}")
        antipattern_groups, top_groups_raw = load_antipattern_groups(temporal_root)
        if antipattern_groups:
            print(f"  [Q&A] Loaded anti-pattern group memberships for most recent revision")
        refactoring_ctx = build_refactoring_context(top_groups_raw, top_n=3)
        if refactoring_ctx:
            print(f"  [Q&A] Loaded per-instance refactoring context for top-3 groups")
        hotspot_data = load_hotspot_data(temporal_root)
        if hotspot_data:
            print(f"  [Q&A] Loaded hotspot ROI data for most recent revision")
        else:
            print(f"  [Q&A] No hotspot ROI data found (run analysis with --fine-grain to generate it)")
        # evidence_evolution already loaded above for fan-in table enrichment
        if evidence_evolution:
            n_transitions = evidence_evolution.count("### Transition")
            print(f"  [Q&A] Loaded evidence graph evolution across {n_transitions} transition(s)")
        # Compute metric peaks for Q&A context (no DRH reports in qa-only mode — pass empty list)
        qa_peaks_section = detect_metric_peaks(timeseries, temporal_root, [], top_n=5)
        if qa_peaks_section:
            print(f"  [Q&A] Computed metric peaks cross-reference")
        # All-transitions summary (chronological, for Metric Trajectory — no deduplication)
        qa_all_transitions_section = build_all_transitions_summary(timeseries, [])
        if qa_all_transitions_section:
            print(f"  [Q&A] Built all-transitions summary")
        # Load worst transition DRH diff report for deep "why" explanation
        worst_drh_section, worst_drh_path = load_worst_transition_drh(temporal_root, timeseries)
        if worst_drh_section:
            print(f"  [Q&A] Loaded worst-transition DRH report: {worst_drh_path}")
        # Load most recent revision's raw churn_top (total activity last window)
        recent_churn_section = load_recent_churn_top(temporal_root)
        if recent_churn_section:
            print(f"  [Q&A] Loaded most-active files from last revision")

        # Build data quality warnings
        _dqw = []
        if not risk_json_path.exists():
            _dqw.append("file_risk_scores.json MISSING — bug_churn/ap_instances/fan-in data unavailable")
        elif not risk_score_context or (top_files and all(
            f.get("signals", {}).get("bug_churn_total", 0) == 0
            and f.get("signals", {}).get("anti_pattern_count", 0) == 0
            for f in top_files
        )):
            _dqw.append(
                "file_risk_scores.json exists but ALL signals are zero — "
                "DV8/backfill data was missing when risk scores were computed. "
                "Fix: delete INPUT_INTERPRETATION/SINGLE_REVISION_ANALYSIS_DATA and re-run backfill + compute_file_risk_scores.py"
            )
        if not antipattern_groups:
            _dqw.append("SINGLE_REVISION_ANALYSIS_DATA missing or empty — anti-pattern group memberships unavailable")
        if not hotspot_data:
            _dqw.append("hotspot ROI data unavailable (run with --fine-grain to generate it)")
        if not evidence_evolution:
            _dqw.append("evidence graph evolution unavailable — fan-in/fan-out deltas not tracked")
        if _dqw:
            print(f"  [Q&A] *** DATA QUALITY WARNINGS ({len(_dqw)}) ***")
            for w in _dqw:
                print(f"  [Q&A]   ⚠ {w}")
            if any("ALL signals are zero" in w for w in _dqw):
                print(f"  [Q&A] *** ANSWER WILL BE DEGRADED — run repair commands before Q&A ***")

        report_text = combined_path.read_text(encoding="utf-8")
        sep = "=" * 70
        conversation: List[str] = []
        current_question = args.user_question
        if not current_question:
            print(f"\n{sep}\n  INTERACTIVE Q&A (qa-only mode)\n{sep}")
            try:
                current_question = input("  Your question: ").strip()
            except EOFError:
                current_question = ""
            if not current_question or current_question.lower() in ("q", "quit", "exit"):
                return 0
        while current_question:
            print(f"\nAnswering: {current_question!r}")
            prior = "\n\n".join(conversation) if conversation else ""
            context_for_answer = (prior + "\n\n" + report_text) if prior else report_text
            prior_answer = conversation[-1] if conversation else ""
            raw = answer_user_question(args.model, current_question, context_for_answer,
                                       mscore_breakdown=mscore_breakdown,
                                       timeout_s=args.ollama_timeout_s,
                                       risk_score_context=risk_score_context,
                                       commit_context=commit_context,
                                       antipattern_groups=antipattern_groups,
                                       hotspot_data=hotspot_data,
                                       evidence_evolution=evidence_evolution,
                                       peaks_section=qa_peaks_section,
                                       recent_churn_section=recent_churn_section,
                                       worst_drh_section=worst_drh_section,
                                       all_transitions_section=qa_all_transitions_section,
                                       data_quality_warnings=_dqw or None,
                                       prior_answer=prior_answer,
                                       refactoring_context=refactoring_ctx)
            answer = strip_thinking_and_fences(raw)
            print(f"\n{sep}\n  ANSWER\n  Q: {current_question}\n{sep}")
            print(answer)
            print(sep)
            conversation.append(f"Q: {current_question}\nA: {answer}")
            now = datetime.now()
            time_str = now.strftime("%H:%M:%S")
            answer_path = qa_folder / "USER_ANSWERS.md"
            answer_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if answer_path.exists() else "w"
            with open(answer_path, mode, encoding="utf-8") as fh:
                if mode == "w":
                    fh.write(f"# Q&A — {args.model}\n\n---\n\n")
                fh.write(f"**Q ({now.strftime('%Y-%m-%d')} {time_str})**: {current_question}\n\n{answer}\n\n---\n\n")
            print(f"  Saved: {answer_path}")
            try:
                current_question = input(f"\n  Follow-up question (Enter to quit): ").strip()
            except EOFError:
                break
            if not current_question or current_question.lower() in ("q", "quit", "exit"):
                break
        return 0

    # Create a dated run subfolder — all LLM outputs go here for better organisation
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    run_folder = interp_root / f"{ts}_{normalize_model_name(args.model)}"
    run_folder.mkdir(parents=True, exist_ok=True)
    # Shared Q&A output folder — reuse existing dated folder or create new one
    _model_safe = normalize_model_name(args.model)
    _existing_qa = sorted(interp_root.glob(f"*_QA_{_model_safe}"))
    if _existing_qa:
        qa_folder = _existing_qa[-1]
    else:
        qa_folder = interp_root / f"{ts[:6]}_QA_{_model_safe}"  # ts[:6] = YYMMDD
    qa_folder.mkdir(parents=True, exist_ok=True)

    # Build rev→revision lookup for date-stamped filenames
    by_num_main: Dict[int, Dict[str, Any]] = {}
    for _r in revisions:
        if isinstance(_r, dict) and _r.get("revision_number"):
            try:
                by_num_main[int(_r["revision_number"])] = _r
            except Exception:
                pass

    # Generate per-transition reports (written into run_folder via --output)
    runner = Path(__file__).parent / "interpret_drh_diff.py"
    reports: List[Path] = []
    for old_n, new_n in zip(reversed(nums[1:]), reversed(nums[:-1])):
        model_safe = normalize_model_name(args.model)
        # Mirror the date+hash filename logic from interpret_drh_diff.py
        old_r_main = by_num_main.get(old_n) or {}
        new_r_main = by_num_main.get(new_n) or {}
        old_date_main = (old_r_main.get("commit_date") or "")[:7]   # "YYYY-MM"
        new_date_main = (new_r_main.get("commit_date") or "")[:7]   # "YYYY-MM"
        new_hash_main = (new_r_main.get("commit_hash") or "")[:7]   # 7-char short hash
        report_out = run_folder / f"drh_diff_report_{model_safe}_{old_date_main}_to_{new_date_main}_{new_hash_main}_new{new_n}_old{old_n}.md"
        cmd = [
            "python3",
            str(runner),
            "--temporal-root",
            str(temporal_root),
            "--repo",
            str(repo_path),
            "--new",
            str(new_n),
            "--old",
            str(old_n),
            "--model",
            args.model,
            "--ollama-timeout-s",
            str(args.ollama_timeout_s),
            "--output",
            str(report_out),
        ]
        if args.no_llm:
            cmd.append("--no-llm")
        if args.no_verify:
            cmd.append("--no-verify")
        elif args.verify:
            cmd.append("--verify")
        print("Running:", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
        # Find the actual written file (interpret_drh_diff may adjust the name)
        candidates = sorted(
            run_folder.glob(f"drh_diff_report_{model_safe}_*new{new_n}_old{old_n}.md"),
            key=lambda p: p.stat().st_mtime
        )
        report_path = candidates[-1] if candidates else report_out
        if report_path.exists():
            reports.append(report_path)

    # Combined report — also in the run subfolder
    combined_path = run_folder / f"temporal_interpretation_report_{normalize_model_name(args.model)}_{ts}.md"
    lines: List[str] = []
    mscore_breakdown: str = ""  # populated below; kept in scope for Q&A
    lines.append(f"# Temporal Interpretation Report")
    lines.append(f"- repo: {timeseries.get('repo') or repo_path.name}")
    lines.append(f"- temporal_root: {temporal_root}")
    lines.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- model: {args.model}")
    lines.append("")

    # Overall summary (optional)
    summaries: List[Tuple[int, int, str]] = []
    for p in reports:
        m = re.search(r"_new(\d+)_old(\d+)\.md$", p.name)
        if not m:
            continue
        new_n, old_n = int(m.group(1)), int(m.group(2))
        ms = extract_managers_special(p.read_text(encoding="utf-8"))
        if ms:
            summaries.append((new_n, old_n, ms))

    if not args.no_overall:
        transitions = list(zip(reversed(nums[:-1]), reversed(nums[1:])))
        mscore_breakdown = load_mscore_worst_modules(temporal_root, max_revisions=5)
        det_overall = build_deterministic_overall(timeseries.get("repo") or repo_path.name, temporal_root, timeseries, reports, transitions).strip()
        if mscore_breakdown and "(mscore components not available)" not in mscore_breakdown:
            det_overall += "\n\n## M-Score Worst Modules Per Revision\n" + mscore_breakdown
        lines.append(det_overall)
        lines.append("")
        if args.llm_overall and not args.no_llm:
            mscore_breakdown = load_mscore_worst_modules(temporal_root, max_revisions=5)
            peaks_section = detect_metric_peaks(timeseries, temporal_root, reports, top_n=3)
            prompt = build_overall_prompt(timeseries.get("repo") or repo_path.name, temporal_root, summaries, timeseries, mscore_breakdown, peaks_section=peaks_section)
            overall = query_ollama(args.model, prompt, timeout_s=args.ollama_timeout_s)
            overall = strip_thinking_and_fences(overall)
            if overall:
                lines.append("## LLM Overall Summary (optional)")
                lines.append(overall.strip())
                lines.append("")

    # Build rev→date lookup for section headers
    _by_num: Dict[int, Dict[str, Any]] = {}
    for _r in revisions:
        if isinstance(_r, dict) and _r.get("revision_number"):
            try:
                _by_num[int(_r["revision_number"])] = _r
            except Exception:
                pass

    # Append each report verbatim
    for p in reports:
        lines.append("---")
        # Enrich section header with dates if we can parse new/old from filename
        _m = re.search(r"_new(\d+)_old(\d+)\.md$", p.name)
        if _m:
            _new_n, _old_n = int(_m.group(1)), int(_m.group(2))
            _old_r = _by_num.get(_old_n) or {}
            _new_r = _by_num.get(_new_n) or {}
            _old_d = (_old_r.get("commit_date") or "")[:10]
            _new_d = (_new_r.get("commit_date") or "")[:10]
            lines.append(f"## Transition Report: old=rev{_old_n} ({_old_d}) → new=rev{_new_n} ({_new_d})")
        else:
            lines.append(f"## Transition Report: {p.name}")
        lines.append("")
        lines.append(p.read_text(encoding="utf-8").strip())
        lines.append("")

    combined_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"Wrote combined report: {combined_path}")

    # Ensure mscore_breakdown is always available for Q&A (may not be set if --no-overall)
    if not mscore_breakdown:
        mscore_breakdown = load_mscore_worst_modules(temporal_root, max_revisions=5)

    # --- Load enriched context for Q&A ---
    # Evidence graph evolution loaded first so fan-in deltas can enrich the risk score table
    evidence_evolution = load_evidence_graph_evolution(temporal_root)

    # Risk score table (multi-signal per-file composite)
    risk_score_context = ""
    risk_json_path = temporal_root / "INPUT_INTERPRETATION" / "file_risk_scores.json"
    if risk_json_path.exists():
        try:
            risk_data = json.loads(risk_json_path.read_text(encoding="utf-8"))
            top_files = risk_data.get("files", [])[:25]
            # Build cumulative fan-in delta per file from evidence_evolution
            _fanin_cumul: dict = {}
            _in_fanin_block = False
            for _line in (evidence_evolution or "").splitlines():
                if "TOP FAN-IN GROWTH" in _line:
                    _in_fanin_block = True
                elif _in_fanin_block:
                    _fm = re.match(r'^\s{4}(\S+\.(?:java|py|kt|ts|js)):\s+([+-]?\d+(?:\.\d+)?)', _line)
                    if _fm:
                        _fanin_cumul[_fm.group(1)] = _fanin_cumul.get(_fm.group(1), 0) + float(_fm.group(2))
                    elif _line.strip() == "" or _line.startswith("  "):
                        if "TOP FAN-" not in _line and "TOP FAN-IN" not in _line:
                            _in_fanin_block = False
            lines_rs = ["rank | file | bug_churn | ap_instances | fan_in(cumul_delta) | anti_patterns | scc_revisions | co_change | anti_pattern_types"]
            lines_rs.append("---" * 12)
            for f in top_files:
                s = f.get("signals", {})
                ap_counts = f.get("anti_pattern_type_counts", {})
                aps = ", ".join(f"{k}:{v}" for k, v in list(ap_counts.items())[:4])
                fname_short = f['file'].split('/')[-1]
                _fi_delta = _fanin_cumul.get(fname_short)
                fi_str = f"fan_in_delta={_fi_delta:+.0f}" if _fi_delta is not None else "fan_in_delta=n/a"
                lines_rs.append(
                    f"#{f['rank']:2d} | {fname_short:40s} | "
                    f"bug={s.get('bug_churn_total',0):5d} | ap_instances={s.get('anti_pattern_instance_load',0):3d} | {fi_str} | "
                    f"ap={s.get('anti_pattern_count',0):3d} counts | "
                    f"scc={s.get('scc_membership_count',0):2d} revisions | co={s.get('co_change_without_dep',0):2d} | [{aps}]"
                )
            risk_score_context = "\n".join(lines_rs)
            print(f"  [Q&A] Loaded risk scores for top {len(top_files)} files")
        except Exception as exc:
            print(f"  [Q&A] WARNING: Could not load risk scores: {exc}")

    # Commit log with bug-linked commits (for M-score causality)
    commit_context, bug_commit_count, bug_commit_source = load_bug_commit_context(temporal_root)
    print(f"  [Q&A] Loaded {bug_commit_count} bug-linked commits from {bug_commit_source}")

    # Anti-pattern group memberships (for professor-style "parts of the system" questions)
    antipattern_groups, top_groups_raw = load_antipattern_groups(temporal_root)
    if antipattern_groups:
        print(f"  [Q&A] Loaded anti-pattern group memberships for most recent revision")
    refactoring_ctx = build_refactoring_context(top_groups_raw, top_n=3)
    if refactoring_ctx:
        print(f"  [Q&A] Loaded per-instance refactoring context for top-3 groups")

    # Hotspot ROI data (for "which files are hotspots?" questions)
    hotspot_data = load_hotspot_data(temporal_root)
    if hotspot_data:
        print(f"  [Q&A] Loaded hotspot ROI data for most recent revision")
    else:
        print(f"  [Q&A] No hotspot ROI data found (run analysis with --fine-grain to generate it)")

    # Evidence graph evolution — already loaded above for fan-in table enrichment
    if evidence_evolution:
        n_transitions = evidence_evolution.count("### Transition")
        print(f"  [Q&A] Loaded evidence graph evolution across {n_transitions} transition(s)")

    # Metric peaks cross-reference: worst metric jumps linked to structural causes
    qa_peaks_section = detect_metric_peaks(timeseries, temporal_root, reports, top_n=5)
    if qa_peaks_section:
        print(f"  [Q&A] Computed metric peaks cross-reference")
    # All-transitions summary (chronological, for Metric Trajectory — no deduplication)
    qa_all_transitions_section = build_all_transitions_summary(timeseries, reports)
    if qa_all_transitions_section:
        print(f"  [Q&A] Built all-transitions summary")
    # Load worst transition DRH diff report for deep "why" explanation
    worst_drh_section, worst_drh_path = load_worst_transition_drh(temporal_root, timeseries)
    if worst_drh_section:
        print(f"  [Q&A] Loaded worst-transition DRH report: {worst_drh_path}")
    # Load most recent revision's raw churn_top
    recent_churn_section = load_recent_churn_top(temporal_root)
    if recent_churn_section:
        print(f"  [Q&A] Loaded most-active files from last revision")

    # Build data quality warnings
    _dqw = []
    if not risk_json_path.exists():
        _dqw.append("file_risk_scores.json MISSING — bug_churn/ap_instances/fan-in data unavailable")
    elif not risk_score_context:
        _dqw.append(
            "file_risk_scores.json exists but ALL signals are zero — "
            "DV8/backfill data was missing when risk scores were computed. "
            "Fix: delete INPUT_INTERPRETATION/SINGLE_REVISION_ANALYSIS_DATA and re-run backfill + compute_file_risk_scores.py"
        )
    if not antipattern_groups:
        _dqw.append("SINGLE_REVISION_ANALYSIS_DATA missing or empty — anti-pattern group memberships unavailable")
    if not hotspot_data:
        _dqw.append("hotspot ROI data unavailable (run with --fine-grain to generate it)")
    if not evidence_evolution:
        _dqw.append("evidence graph evolution unavailable — fan-in/fan-out deltas not tracked")
    if _dqw:
        print(f"  [Q&A] *** DATA QUALITY WARNINGS ({len(_dqw)}) ***")
        for w in _dqw:
            print(f"  [Q&A]   ⚠ {w}")
        if any("ALL signals are zero" in w for w in _dqw):
            print(f"  [Q&A] *** ANSWER WILL BE DEGRADED — run repair commands before Q&A ***")

    # --- Interactive Q&A loop ---
    # Start with the user's initial question (if any), then offer follow-up prompts.
    if not args.no_llm:
        report_text = combined_path.read_text(encoding="utf-8")
        sep = "=" * 70

        # Build conversation history so follow-ups have context
        conversation: List[str] = []

        # Seed with initial question, or prompt the user for a first question
        current_question = args.user_question
        if not current_question:
            print(f"\n{sep}")
            print("  INTERACTIVE Q&A")
            print("  Ask a question about this repository's architectural evolution.")
            print("  (Press Enter with no input to skip, type 'q' to quit.)")
            print(sep)
            try:
                current_question = input("  Your question: ").strip()
            except EOFError:
                current_question = ""
            if not current_question or current_question.lower() in ("q", "quit", "exit"):
                return 0
            # Reject junk input (single non-word chars like backslash, slash, etc.)
            if not any(c.isalpha() or c.isdigit() for c in current_question):
                return 0

        while current_question:
            print(f"\nAnswering: {current_question!r}")
            # Build context: report + prior Q&A turns
            prior = "\n\n".join(conversation) if conversation else ""
            context_for_answer = (prior + "\n\n" + report_text) if prior else report_text
            prior_answer = conversation[-1] if conversation else ""
            raw = answer_user_question(args.model, current_question, context_for_answer,
                                       mscore_breakdown=mscore_breakdown,
                                       timeout_s=args.ollama_timeout_s,
                                       risk_score_context=risk_score_context,
                                       commit_context=commit_context,
                                       antipattern_groups=antipattern_groups,
                                       hotspot_data=hotspot_data,
                                       evidence_evolution=evidence_evolution,
                                       peaks_section=qa_peaks_section,
                                       recent_churn_section=recent_churn_section,
                                       worst_drh_section=worst_drh_section,
                                       all_transitions_section=qa_all_transitions_section,
                                       data_quality_warnings=_dqw or None,
                                       prior_answer=prior_answer,
                                       refactoring_context=refactoring_ctx)
            answer = strip_thinking_and_fences(raw)

            print(f"\n{sep}")
            print(f"  ANSWER")
            print(f"  Q: {current_question}")
            print(sep)
            print(answer)
            print(sep)

            # Record this turn in conversation history
            conversation.append(f"Q: {current_question}\nA: {answer}")

            # Save answer — single append file in QA_<model>/ folder
            now = datetime.now()
            time_str = now.strftime("%H:%M:%S")
            answer_path = qa_folder / "USER_ANSWERS.md"
            answer_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if answer_path.exists() else "w"
            with open(answer_path, mode, encoding="utf-8") as f:
                if mode == "w":
                    f.write(f"# Q&A — {args.model}\n\n---\n\n")
                f.write(
                    f"**Q ({now.strftime('%Y-%m-%d')} {time_str})**: {current_question}\n\n"
                    f"{answer}\n\n---\n\n"
                )
            print(f"Saved answer: {answer_path}")

            # Offer follow-up
            print(f"\n{sep}")
            print("  FOLLOW-UP")
            print("  Type a follow-up question, or press Enter / 'q' to finish.")
            print(sep)
            try:
                next_q = input("  Your question: ").strip()
            except EOFError:
                next_q = ""
            if not next_q or next_q.lower() in ("q", "quit", "exit"):
                break
            # Reject junk input (single non-word chars like backslash)
            if not any(c.isalpha() or c.isdigit() for c in next_q):
                break
            current_question = next_q

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
