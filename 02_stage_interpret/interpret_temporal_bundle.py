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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def query_ollama(model: str, prompt: str, timeout_s: int = 900) -> str:
    res = subprocess.run(["ollama", "run", model, prompt], capture_output=True, text=True, timeout=timeout_s)
    out = (res.stdout or "").strip()
    if out:
        return out
    return (res.stderr or "").strip()


def strip_thinking_and_fences(text: str) -> str:
    if not text:
        return text
    # Strip leading Thinking... block
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
    lines.append("- per-transition reports: `INPUT_INTERPRETATION/drh_diff_report_<model>_newX_oldY.md`")
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


def load_mscore_worst_modules(temporal_root: Path, top_n: int = 5) -> str:
    """
    For each revision folder under temporal_root, load mscore_exact_components.json
    and return the top_n worst modules ranked by contribution (= cross_penalty × size_factor).
    Also loads FanIn/FanOut per file and clique count to give a multi-signal refactoring picture.
    Returns a formatted string ready to embed in an LLM prompt.
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

    for rev_n in sorted(rev_map.keys()):
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
                         mscore_breakdown: str = "", timeout_s: int = 900) -> str:
    """Call the LLM to answer a specific user question using the combined report as context.

    Builds a priority context: M-score worst modules first (most useful for file-level questions),
    then Comprehensive Summary blocks (each labelled with commit dates for date-range questions),
    then narrative up to a total of ~12000 chars.
    """
    # 1. M-score worst modules — put first so it's never truncated away
    priority_context = ""
    if mscore_breakdown and "(mscore components not available)" not in mscore_breakdown:
        priority_context = (
            "## M-SCORE WORST MODULES PER REVISION (top 5 by contribution = cross_penalty × size_factor; FanIn/FanOut per file; clique_files count in header):\n"
            + mscore_breakdown + "\n\n"
        )

    # 2. Extract Comprehensive Summary blocks — they carry date labels for date-range questions
    summary_blocks = re.findall(
        r'(## Comprehensive Summary.*?)(?=\n## |\Z)', report_text, re.DOTALL
    )
    summary_text = "\n\n".join(summary_blocks[:6])[:6000]

    # 3. Fill remaining budget with the narrative from the full report
    budget = 12000 - len(priority_context) - len(summary_text)
    narrative = report_text[:max(0, budget)]

    context = priority_context + summary_text + ("\n\n" if summary_text else "") + narrative

    # Detect file/refactor questions — use ultra-direct numbered-list prompt to prevent narrative drift
    q_lower = question.lower()
    is_files_question = any(k in q_lower for k in [
        "refactor", "specific file", "which file", "bad file", "worst file",
        "file to fix", "file to improve", "give me file", "5 file", "top file",
        "files to", "file i should", "files i should",
    ])

    if is_files_question and priority_context:
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
- Answer ONLY the question asked. Be direct and concise (max 300 words).
- Use ONLY facts from the report below. Do NOT invent numbers or files not mentioned in the report.
- Do NOT use background knowledge about library versions, release history, or project wikis — cite only what is in the report.
- Format: short header, bullet points for evidence, 1-sentence conclusion.
- Do NOT output reasoning or "thinking" blocks.
- For date-range questions (e.g. "from 2023 to 2024"): the Comprehensive Summary blocks above are labelled with transition dates in the format "old=revN (YYYY-MM-DD) → new=revM (YYYY-MM-DD)". Find the transition(s) whose dates fall within the range asked and cite their specific metric deltas. Do NOT give a generic answer about overall trends.
- For "which files to refactor" questions: use the M-SCORE WORST MODULES section above. Modules are ranked by contribution (cross_penalty × size_factor) — this is the correct refactoring priority because it weights both violation severity AND module size (larger coupled modules are harder to fix). List files from the highest-contribution modules first, citing their FanIn/FanOut values. Also note the clique_files count in the revision header — if high, mention that circular dependency clusters require coordinated refactoring. Do NOT use layer-movement data (files that moved between layers). Do NOT rank by cross_penalty alone.
- When citing metric changes, ALWAYS include both absolute delta and percentage (e.g. "+7.38 points, +15.02%").

QUESTION: {question}

REPORT (M-score data first, then transition summaries with dates, then narrative):
{context}
"""
    return query_ollama(model, prompt, timeout_s=timeout_s)


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

    interp_root = temporal_root / "INPUT_INTERPRETATION"
    interp_root.mkdir(parents=True, exist_ok=True)

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
        mscore_breakdown = load_mscore_worst_modules(temporal_root)
        det_overall = build_deterministic_overall(timeseries.get("repo") or repo_path.name, temporal_root, timeseries, reports, transitions).strip()
        if mscore_breakdown and "(mscore components not available)" not in mscore_breakdown:
            det_overall += "\n\n## M-Score Worst Modules Per Revision\n" + mscore_breakdown
        lines.append(det_overall)
        lines.append("")
        if args.llm_overall and not args.no_llm:
            mscore_breakdown = load_mscore_worst_modules(temporal_root)
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
        mscore_breakdown = load_mscore_worst_modules(temporal_root)

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
                                       timeout_s=args.ollama_timeout_s)
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
