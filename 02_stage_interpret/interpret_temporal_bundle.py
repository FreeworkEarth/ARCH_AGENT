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


def query_ollama(model: str, prompt: str, timeout_s: int = 900, num_ctx: int = 32768) -> str:
    """Call LLM via LLMBackend (supports Ollama/vLLM/API via env vars)."""
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
            fname = (r.get("Filename") or "").split("/")[-1]
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
    # Revision folders live INSIDE temporal_root named NN_reponame_DDMMYYYY_HHMM
    search_root = temporal_root
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
                fname = fpath.split("/")[-1]
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


def build_overall_prompt(repo: str, temporal_root: Path, summaries: List[Tuple[int, int, str]], timeseries: Dict[str, Any], mscore_breakdown: str = "") -> str:
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

Context:
- repo: {repo}
- temporal_root: {temporal_root}
- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FACTS (timeseries.json excerpt):
{json.dumps(timeseries, indent=2)[:12000]}

M-SCORE WORST MODULES PER REVISION (top 5 by contribution = cross_penalty × size_factor; FanIn/FanOut per file; clique_files count in header):
{mscore_breakdown}

PER-TRANSITION COMPREHENSIVE SUMMARIES (chronological newest→older):
{chr(10).join([f'### new={n} old={o}{chr(10)}{ms}' for n,o,ms in summaries])}
"""


def answer_user_question(model: str, question: str, report_text: str,
                         mscore_breakdown: str = "", timeout_s: int = 900,
                         risk_score_context: str = "", commit_context: str = "") -> str:
    """Call the LLM to answer a specific user question using the combined report as context.

    Builds a priority context: M-score worst modules first (most useful for file-level questions),
    then risk score table (multi-signal: bug churn + anti-patterns + fan-in + SCC + co-change),
    then Comprehensive Summary blocks (each labelled with commit dates for date-range questions),
    then narrative up to a total of ~16000 chars.
    """
    # Context budget: 28000 chars (fits comfortably in 32k num_ctx with room for prompt boilerplate).
    # Hard cap every section so the total is ALWAYS within budget regardless of input size.
    CONTEXT_BUDGET = 28000

    # 1. M-score worst modules (capped to recent 5 revisions by caller; hard cap 6000 chars here too)
    priority_context = ""
    if mscore_breakdown and "(mscore components not available)" not in mscore_breakdown:
        priority_context = (
            "## M-SCORE WORST MODULES (most recent 5 revisions, top 5 by contribution = cross_penalty × size_factor; FanIn/FanOut per file):\n"
            + mscore_breakdown[:6000] + "\n\n"
        )

    # 2. Risk score table — multi-signal evidence (bug churn, anti-patterns, SCC, co-change)
    risk_section = ""
    if risk_score_context:
        risk_section = (
            "## MULTI-SIGNAL FILE RISK SCORES (bug_churn 30% + anti_pattern 25% + fan_in 20% + scc 15% + co_change 10%):\n"
            + risk_score_context[:4000] + "\n\n"
        )

    # 3. Commit context for M-score causality questions
    commit_section = ""
    if commit_context:
        commit_section = (
            "## BUG-LINKED COMMITS (JIRA-typed or keyword-matched; most recent first):\n"
            + commit_context[:6000] + "\n\n"
        )

    # 4. Extract Comprehensive Summary blocks — they carry date labels for date-range questions
    summary_blocks = re.findall(
        r'(## Comprehensive Summary.*?)(?=\n## |\Z)', report_text, re.DOTALL
    )
    summary_text = "\n\n".join(summary_blocks[:8])[:6000]

    # 5. Fill remaining budget with narrative (usually 0 — summaries already cover it)
    used = len(priority_context) + len(risk_section) + len(commit_section) + len(summary_text)
    remaining = max(0, CONTEXT_BUDGET - used)
    narrative = report_text[:remaining] if remaining > 500 else ""

    context = priority_context + risk_section + commit_section + summary_text + ("\n\n" if summary_text else "") + narrative

    q_lower = question.lower()

    # Detect "most dangerous / worst files" questions — use risk score data
    is_danger_question = any(k in q_lower for k in [
        "dangerous", "most dangerous", "riskiest", "worst file", "bad file",
        "technical debt", "most debt", "debt", "problematic", "priority",
        "5 most", "top 5", "top file",
    ])

    # Detect M-score causality questions
    is_mscore_causality = any(k in q_lower for k in [
        "why did m-score", "why m-score", "why did the m-score", "m-score got worse",
        "m-score deteriorat", "m-score declin", "what caused", "what change caused",
        "linked to", "link to commit", "which commit", "which feature", "which bug",
        "quality decrease", "quality declin", "decreasing in quality", "decrease in quality",
    ])

    # Detect refactor/structural questions — use M-score module data
    is_files_question = any(k in q_lower for k in [
        "refactor", "specific file", "which file", "file to fix", "file to improve",
        "give me file", "files to", "file i should", "files i should",
    ])

    if is_danger_question and risk_section:
        prompt = f"""You are a software architect. Answer the question below using the MULTI-SIGNAL FILE RISK SCORES as your primary evidence.

The risk score combines 5 signals measured across 36 monthly snapshots of the repository:
- bug_churn_total: lines changed in bug-fix commits (JIRA-linked or keyword-matched) — weight 30%
- anti_pattern_count: number of revisions where the file is in a DV8 anti-pattern (clique/cycle/unhealthy inheritance) — weight 25%
- hotspot_fanin_score: cumulative fan-in (blast radius if changed) — weight 20%
- scc_membership_count: revisions where file is in a cyclic SCC (circular dependency) — weight 15%
- co_change_without_dep: behaviorally coupled partners with no declared structural dependency — weight 10%

Hard rules:
- Do NOT output reasoning or <think> blocks.
- For each file: state its rank, risk_score, and cite at least 2 specific signal values (bug_churn, anti_pattern_count, etc.) from the data.
- Explain what each signal means architecturally (e.g. "high bug_churn means this file is actively being patched for defects").
- Also cross-reference with M-SCORE WORST MODULES data if relevant (file appears in both → doubly confirmed).
- Format: numbered list, one file per item. Max 400 words total.
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
    return query_ollama(model, prompt, timeout_s=timeout_s, num_ctx=32768)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate per-transition DRH interpretation + one combined summary report.")
    ap.add_argument("--temporal-root", required=True, help="Path to temporal_analysis_* folder")
    ap.add_argument("--repo", default=None, help="Path to git repo (default: temporal_root/..)")
    ap.add_argument("--model", default="deepseek-r1:32b", help="Ollama model (default: deepseek-r1:32b)")
    ap.add_argument("--ollama-timeout-s", type=int, default=900, help="Ollama timeout in seconds (default: 900)")
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

    ts_path = temporal_root / "timeseries.json"
    timeseries = read_json(ts_path)
    revisions = timeseries.get("revisions") or []
    if not revisions:
        if not ts_path.exists():
            hint = ""
            try:
                candidates = list(temporal_root.parent.glob("temporal_analysis*/timeseries.json"))
                if candidates:
                    newest = max(candidates, key=lambda x: x.stat().st_mtime)
                    hint = f" (hint: try {newest.parent})"
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
        print(f"[qa-only] Using existing report: {combined_path}")
        mscore_breakdown = load_mscore_worst_modules(temporal_root, max_revisions=5)
        # Jump directly to Q&A section
        risk_score_context = ""
        risk_json_path = temporal_root / "INPUT_INTERPRETATION" / "file_risk_scores.json"
        if risk_json_path.exists():
            try:
                risk_data = json.loads(risk_json_path.read_text(encoding="utf-8"))
                top_files = risk_data.get("files", [])[:25]
                lines_rs = ["rank | file | risk_score | bug_churn | anti_patterns | scc_revisions | co_change | anti_pattern_types"]
                lines_rs.append("---" * 12)
                for f in top_files:
                    s = f.get("signals", {})
                    aps = ", ".join(f.get("anti_patterns_seen", [])[:3])
                    lines_rs.append(
                        f"#{f['rank']:2d} | {f['file'].split('/')[-1]:40s} | {f['risk_score']:.3f} | "
                        f"bug={s.get('bug_churn_total',0):5d} | ap={s.get('anti_pattern_count',0):3d} revisions | "
                        f"scc={s.get('scc_membership_count',0):2d} revisions | co={s.get('co_change_without_dep',0):2d} | [{aps}]"
                    )
                risk_score_context = "\n".join(lines_rs)
                print(f"  [Q&A] Loaded risk scores for top {len(top_files)} files")
            except Exception as exc:
                print(f"  [Q&A] WARNING: Could not load risk scores: {exc}")
        commit_context = ""
        issue_map_path = temporal_root / "issue_map.json"
        if issue_map_path.exists():
            try:
                issue_data = json.loads(issue_map_path.read_text(encoding="utf-8"))
                summaries_map = issue_data.get("summaries", {})
                issues_map = issue_data.get("issues", {})
                commit_log = issue_data.get("commit_log", [])
                import re as _re
                _jira_re = _re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")
                _bug_kw = _re.compile(r"\b(fix|bug|hotfix|patch|defect|regress)\b", _re.IGNORECASE)
                bug_commits = []
                for c in commit_log:
                    subj = c.get("subject", "")
                    jira_refs = _jira_re.findall(subj)
                    is_bug_jira = any(issues_map.get(k) == "bug" for k in jira_refs)
                    is_bug_kw = bool(_bug_kw.search(subj))
                    if is_bug_jira or is_bug_kw:
                        issue_title = ""
                        for k in jira_refs:
                            if k in summaries_map:
                                issue_title = f" ({summaries_map[k]})"
                                break
                        bug_commits.append(f"- [{c.get('date','')[:10]}] {c.get('hash','')[:8]} {subj}{issue_title}")
                commit_context = "\n".join(bug_commits[:80])
                print(f"  [Q&A] Loaded {len(bug_commits)} bug-linked commits")
            except Exception as exc:
                print(f"  [Q&A] WARNING: Could not load commit context: {exc}")
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
            raw = answer_user_question(args.model, current_question, context_for_answer,
                                       mscore_breakdown=mscore_breakdown,
                                       timeout_s=args.ollama_timeout_s,
                                       risk_score_context=risk_score_context,
                                       commit_context=commit_context)
            answer = strip_thinking_and_fences(raw)
            print(f"\n{sep}\n  ANSWER\n  Q: {current_question}\n{sep}")
            print(answer)
            print(sep)
            conversation.append(f"Q: {current_question}\nA: {answer}")
            now = datetime.now()
            day_str = now.strftime("%Y%m%d")
            time_str = now.strftime("%H:%M:%S")
            answer_path = run_folder / f"USER_ANSWER_{day_str}.md"
            mode = "a" if answer_path.exists() else "w"
            with open(answer_path, mode, encoding="utf-8") as fh:
                if mode == "w":
                    fh.write(f"# Q&A Session — {day_str}\n\n**Model**: {args.model}\n\n---\n\n")
                fh.write(f"**Q ({time_str})**: {current_question}\n\n{answer}\n")
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
        transitions = list(zip(reversed(nums[1:]), reversed(nums[:-1])))
        mscore_breakdown = load_mscore_worst_modules(temporal_root, max_revisions=5)
        det_overall = build_deterministic_overall(timeseries.get("repo") or repo_path.name, temporal_root, timeseries, reports, transitions).strip()
        if mscore_breakdown and "(mscore components not available)" not in mscore_breakdown:
            det_overall += "\n\n## M-Score Worst Modules Per Revision\n" + mscore_breakdown
        lines.append(det_overall)
        lines.append("")
        if args.llm_overall and not args.no_llm:
            mscore_breakdown = load_mscore_worst_modules(temporal_root, max_revisions=5)
            prompt = build_overall_prompt(timeseries.get("repo") or repo_path.name, temporal_root, summaries, timeseries, mscore_breakdown)
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
    # Risk score table (multi-signal per-file composite)
    risk_score_context = ""
    risk_json_path = temporal_root / "INPUT_INTERPRETATION" / "file_risk_scores.json"
    if risk_json_path.exists():
        try:
            risk_data = json.loads(risk_json_path.read_text(encoding="utf-8"))
            top_files = risk_data.get("files", [])[:25]
            lines_rs = ["rank | file | risk_score | bug_churn | anti_patterns | scc_revisions | co_change | anti_pattern_types"]
            lines_rs.append("---" * 12)
            for f in top_files:
                s = f.get("signals", {})
                aps = ", ".join(f.get("anti_patterns_seen", [])[:3])
                lines_rs.append(
                    f"#{f['rank']:2d} | {f['file'].split('/')[-1]:40s} | {f['risk_score']:.3f} | "
                    f"bug={s.get('bug_churn_total',0):5d} | ap={s.get('anti_pattern_count',0):3d} revisions | "
                    f"scc={s.get('scc_membership_count',0):2d} revisions | co={s.get('co_change_without_dep',0):2d} | [{aps}]"
                )
            risk_score_context = "\n".join(lines_rs)
            print(f"  [Q&A] Loaded risk scores for top {len(top_files)} files")
        except Exception as exc:
            print(f"  [Q&A] WARNING: Could not load risk scores: {exc}")

    # Commit log with bug-linked commits (for M-score causality)
    commit_context = ""
    issue_map_path = temporal_root / "issue_map.json"
    if issue_map_path.exists():
        try:
            issue_data = json.loads(issue_map_path.read_text(encoding="utf-8"))
            summaries_map = issue_data.get("summaries", {})
            issues_map = issue_data.get("issues", {})
            commit_log = issue_data.get("commit_log", [])
            # Keep only bug-linked commits (JIRA bug reference or keyword match) — last 3 years
            import re as _re
            _jira_re = _re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")
            _bug_kw = _re.compile(r"\b(fix|bug|hotfix|patch|defect|regress)\b", _re.IGNORECASE)
            bug_commits = []
            for c in commit_log:
                subj = c.get("subject", "")
                # Check if references a known bug issue
                jira_refs = _jira_re.findall(subj)
                is_bug_jira = any(issues_map.get(k) == "bug" for k in jira_refs)
                is_bug_kw = bool(_bug_kw.search(subj))
                if is_bug_jira or is_bug_kw:
                    issue_title = ""
                    for k in jira_refs:
                        if k in summaries_map:
                            issue_title = f" ({summaries_map[k]})"
                            break
                    bug_commits.append(f"- [{c.get('date','')[:10]}] {c.get('hash','')[:8]} {subj}{issue_title}")
            # Limit to 80 most recent bug commits
            commit_context = "\n".join(bug_commits[:80])
            print(f"  [Q&A] Loaded {len(bug_commits)} bug-linked commits for M-score causality")
        except Exception as exc:
            print(f"  [Q&A] WARNING: Could not load commit context: {exc}")

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
            raw = answer_user_question(args.model, current_question, context_for_answer,
                                       mscore_breakdown=mscore_breakdown,
                                       timeout_s=args.ollama_timeout_s,
                                       risk_score_context=risk_score_context,
                                       commit_context=commit_context)
            answer = strip_thinking_and_fences(raw)

            print(f"\n{sep}")
            print(f"  ANSWER")
            print(f"  Q: {current_question}")
            print(sep)
            print(answer)
            print(sep)

            # Record this turn in conversation history
            conversation.append(f"Q: {current_question}\nA: {answer}")

            # Save answer to file — one file per day, append subsequent Q&A turns
            now = datetime.now()
            day_str = now.strftime("%Y%m%d")
            time_str = now.strftime("%H:%M:%S")
            answer_path = run_folder / f"USER_ANSWER_{day_str}.md"
            if answer_path.exists():
                with open(answer_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n---\n\n"
                        f"**Q ({time_str})**: {current_question}\n\n"
                        f"{answer}\n"
                    )
            else:
                answer_path.write_text(
                    f"# Q&A Session — {day_str}\n\n"
                    f"**Model**: {args.model}\n\n"
                    f"---\n\n"
                    f"**Q ({time_str})**: {current_question}\n\n"
                    f"{answer}\n",
                    encoding="utf-8"
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
