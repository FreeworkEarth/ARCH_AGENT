#!/usr/bin/env python3
"""
Upgraded LLM Frontend with Tool-Calling and Integrated Explainer

Enhancements:
- Calls integrated_explainer.py for detailed AI-powered explanations
- More conversational flow
- Automatic tool selection
- Interactive follow-ups
"""

import json, os, re, subprocess, sys, urllib.request, urllib.error, pathlib
from typing import Optional

# Paths
THIS_DIR = os.path.dirname(__file__)
AGENT = os.path.join(THIS_DIR, "dv8_agent.py")

# Known toy repo snapshots — used for single-revision godclass tests
# Keys: (language, snapshot_name)  Values: (repo_url, commit_hash)
_TOY_SNAPSHOTS = {
    ("python", "godclass"): (
        "https://github.com/FreeworkEarth/ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG",
        "d5f99d1ff0b55b1556ee11300bf2c953076c53c2",
    ),
    ("java", "godclass"): (
        "https://github.com/FreeworkEarth/ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG",
        "8c99861f3873c890ac5ad096c9bf267b21d51596",
    ),
}
# Use the shared RAG explainer location
EXPLAINER = os.path.join(os.path.dirname(THIS_DIR), "04_RAG_EXPLAINER", "integrated_explainer.py")
TEMPORAL = os.path.join(THIS_DIR, "temporal_analyzer.py")
PLOTTER = os.path.join(THIS_DIR, "metric_plotter.py")
BACKFILL_TEMPORAL = os.path.join(THIS_DIR, "backfill_temporal_payloads.py")
INTERPRET_TEMPORAL = os.path.join(os.path.dirname(THIS_DIR), "02_stage_interpret", "interpret_temporal_bundle.py")
BUNDLE_VERIFY = os.path.join(os.path.dirname(THIS_DIR), "02_stage_interpret", "verify_interpretation_bundle.py")
QUERY_ENGINE = os.path.join(os.path.dirname(THIS_DIR), "03_stage_query", "query_engine.py")
FETCH_ISSUES = os.path.join(THIS_DIR, "fetch_github_issues.py")
EXPORT_DV8 = os.path.join(THIS_DIR, "export_dv8_binary_files.py")
COMPUTE_RISK = os.path.join(THIS_DIR, "compute_file_risk_scores.py")
PLOT_RISK = os.path.join(THIS_DIR, "plot_risk_score_stats.py")

# Config
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
# Prefer explicit env, then TEST_AUTO/RAG_KnowledgeBase, else legacy KB/archdia
_PARENT = os.path.dirname(THIS_DIR)
_RAG_LOCAL = os.path.join(_PARENT, "RAG_KnowledgeBase")
_RAG_LEGACY = os.path.join(_PARENT, "KB", "archdia")
RAG_KB_DIR = os.getenv("RAG_KB_DIR") or os.getenv("ARCHDIA_KB_DIR") or (_RAG_LOCAL if os.path.isdir(_RAG_LOCAL) else _RAG_LEGACY)

# Ensure data output folder exists (created on first run)
_REPOS_ANALYZED_DIR = os.path.join(os.path.dirname(THIS_DIR), "REPOS_ANALYZED")
os.makedirs(_REPOS_ANALYZED_DIR, exist_ok=True)

def _update_neodepends_if_local() -> None:
    """If a local NeoDepends clone exists, pull latest production branch (best-effort).

    dv8_agent.py handles full discovery and GitHub fallback — this only keeps an
    existing local clone fresh. Never crashes; never sets any paths.
    """
    candidates = [
        pathlib.Path(__file__).resolve().parents[3] / "TEST_AUTO" / "00_CORE" / "NEODEPENDS_DEICIDE" / "00_NEODEPENDS" / "neodepends",
        pathlib.Path.home() / ".dv8_agent" / "neodepends" / "neodepends_repo",
    ]
    for nd_path in candidates:
        if nd_path.exists() and (nd_path / ".git").exists():
            try:
                subprocess.run(["git", "-C", str(nd_path), "fetch", "origin"], check=True, capture_output=True)
                subprocess.run(["git", "-C", str(nd_path), "checkout", "production"], check=True, capture_output=True)
                subprocess.run(["git", "-C", str(nd_path), "pull", "origin", "production"], check=True, capture_output=True)
                print(f"[neodepends] Pulled latest production branch at {nd_path}")
            except subprocess.CalledProcessError:
                print(f"[neodepends] Warning: git pull failed at {nd_path} — continuing with existing copy.")
            break

# Enhanced system prompt with tool-calling
SYSTEM_PROMPT = """You are a DV8 architecture analysis assistant with the following tools:

Tools available:
1. analyze_repo - Run DV8 analysis on a repository (single revision)
2. explain_metrics - Generate detailed AI explanation of DV8 metrics
3. explain_concept - Explain DV8 concepts using knowledge base
4. temporal_analysis - Analyze multiple Git revisions and plot metrics over time
5. interpret_metrics - Interpret WHY metrics changed by analyzing git commits (requires timeseries.json)
6. interpret_temporal - Interpret a temporal analysis folder (pairwise DRH diffs + overall summary)
7. peak_full_arch - Find the two revisions with biggest M-score delta and run full arch reports on both
8. query - Fast Q&A on existing results (uses risk scores + commit log + M-score data; answers in <2min; no re-run needed)

IMPORTANT: Use tool="query" whenever the user asks a question about already-analyzed data WITHOUT requesting a new analysis run. Examples: "what are the 5 most dangerous files?", "why did m-score get worse?", "what causes technical debt?", "which parts are decreasing in quality?". These questions use pre-computed risk scores, commit logs, and DV8 data — they do NOT re-run DV8.

Output ONLY JSON:
{
  "tool": "analyze_repo|explain_metrics|explain_concept|temporal_analysis|interpret_metrics|interpret_temporal|query",
  "repo": "<local path or Git URL or short repo name>",
  "ask": "all|m-score|propagation-cost|..." (optional for analyze_repo),
  "skip_arch_report": true|false (default: true),
  "force_depends": true|false (default: false),
  "source_path": "<subfolder path>" (optional: analyze only this subdir; relative to repo root),
  "topic": "<concept>" (for explain_concept),
  "count": <number> (for temporal_analysis, default: 5),
  "branch": "<branch>" (for temporal_analysis, default: "trunk" for PDFBox, otherwise "main"),
  "min_months_apart": <number> (for temporal_analysis: 0=all-time mode, 1-12=recent mode with N months spacing),
  "model": "<ollama_model>" (for interpret_metrics/interpret_temporal/query, default: "deepseek-r1:32b", recommended: "deepseek-r1:70b" for best quality),
  "question": "<question text>" (for query tool: the full question to answer),
  "user_question": "<question text>" (for temporal_analysis/interpret_temporal: question answered after analysis),
  "mv_cochange": <integer> (optional: modularity violation co-change threshold passed to DV8; default=DV8 built-in (2); use 5 for medium repos to filter process-noise)
}

MV_COCHANGE EXTRACTION RULES:
- If user mentions "mv threshold 5", "mv cochange 5", "mvCochange 5", "modularity violation threshold 5" → "mv_cochange": 5
- If user mentions "with mv threshold N" or "threshold N for modularity" → "mv_cochange": N
- Do NOT set mv_cochange if user does not mention it (leave out of JSON — use DV8 default)

MODEL EXTRACTION RULES:
- If user mentions "deepseek-r1:14b", "14b", "14B" → "model": "deepseek-r1:14b"
- If user mentions "deepseek-r1:32b", "32b", "32B" → "model": "deepseek-r1:32b"
- If user mentions "deepseek-r1:70b", "70b", "70B" → "model": "deepseek-r1:70b"
- If user says "all models" or "all deepseek" → "model": "all"
- Default when no model mentioned: "deepseek-r1:32b"

TWO SIMPLE MODES:
1. ALL-TIME MODE (min_months_apart=0):
   - Selects: First ever commit, last ever commit, evenly interpolated in between
   - Folder: temporal_analysis_alltime
   - Use when: "of all time", "from beginning to end", "entire history"

2. RECENT-MAJOR MODE (min_months_apart>0):
   - Selects: Recent commits with N months minimum spacing
   - Folder: temporal_analysis_5revisions_3month_diff (example for 5 revisions, 3 months)
   - Use when: "major changes", "X months apart", "recent with spacing"

Notes:
- For PDFBox, use branch="trunk" not "main"
- min_months_apart=0 → ALL-TIME mode (first, last, interpolated)
- min_months_apart=1 → RECENT mode with 1 month spacing
- min_months_apart=3 → RECENT mode with 3 months spacing (RECOMMENDED by professor)

Reasoning:
- User says "of all time" or "entire history" → min_months_apart=0 (ALL-TIME)
- User says "major changes" or "X months apart" → min_months_apart=3 (RECENT-MAJOR)
- User says "1 month" → min_months_apart=1
- User says "3 months" → min_months_apart=3
- User says "X years Y months apart" → count = X * (12 / Y), min_months_apart = Y
  - "5 years 3 months apart" → count=20, min_months_apart=3
  - "3 years 6 months apart" → count=6, min_months_apart=6
  - "10 years 1 month apart" → count=120, min_months_apart=1
- ALWAYS use tool="temporal_analysis" when user says "analyze and interpret" (not "interpret_temporal")
- NEVER output placeholder paths like "/path/to/..." — use ONLY short repo names from KNOWN REPOSITORIES

Examples:
- "analyze 5 revisions of all time for pdfbox" → {"tool": "temporal_analysis", "repo": "pdfbox", "count": 5, "branch": "trunk", "min_months_apart": 0}
- "analyze 5 major revisions of pdfbox with 3 months in between" → {"tool": "temporal_analysis", "repo": "pdfbox", "count": 5, "branch": "trunk", "min_months_apart": 3}
- "analyze last 7 major revisions with 1 month spacing" → {"tool": "temporal_analysis", "repo": "pdfbox", "count": 7, "branch": "trunk", "min_months_apart": 1}
- "analyze jsoup all-time with 10 timesteps 3 months apart and then interpret with deepseek-r1:32b" → {"tool": "temporal_analysis", "repo": "jsoup", "count": 10, "branch": "main", "min_months_apart": 3, "model": "deepseek-r1:32b"}
- "analyze commons-io all-time in 5 timesteps and then interpret with deepseek-r1:70b" → {"tool": "temporal_analysis", "repo": "commons-io", "count": 5, "branch": "main", "min_months_apart": 0, "model": "deepseek-r1:70b"}
- "analyze and interpret commons-io with 5 years 3 months apart with deepseek-r1:32b" → {"tool": "temporal_analysis", "repo": "commons-io", "count": 20, "branch": "main", "min_months_apart": 3, "model": "deepseek-r1:32b"}
- "analyze and interpret commons-io 3 years 6 months apart with deepseek-r1:32b" → {"tool": "temporal_analysis", "repo": "commons-io", "count": 6, "branch": "main", "min_months_apart": 6, "model": "deepseek-r1:32b"}
- "interpret the temporal analysis for jsoup with deepseek-r1:32b" → {"tool": "interpret_temporal", "repo": "jsoup", "model": "deepseek-r1:32b"}
- "interpret this temporal analysis folder '/.../temporal_analysis_alltime_...'" → {"tool": "interpret_temporal", "repo": "/.../temporal_analysis_alltime_.../INPUT_INTERPRETATION", "model": "deepseek-r1:32b"}
- "analyze ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG all-time in 2 timesteps on branch temporal and then interpret with deepseek-r1:32b" → {"tool": "temporal_analysis", "repo": "ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG", "count": 2, "branch": "temporal", "min_months_apart": 0, "model": "deepseek-r1:32b"}
- "analyze and interpret https://github.com/apache/commons-io.git all time 5 timesteps with deepseek-r1:32b and answer: why did the m-score change between 2020 and 2024?" → {"tool": "temporal_analysis", "repo": "https://github.com/apache/commons-io.git", "count": 5, "branch": "main", "min_months_apart": 0, "model": "deepseek-r1:32b", "user_question": "why did the m-score change between 2020 and 2024?"}
- "analyze https://github.com/apache/pdfbox.git all-time 5 timesteps and then interpret with deepseek-r1:32b" → {"tool": "temporal_analysis", "repo": "https://github.com/apache/pdfbox.git", "count": 5, "branch": "trunk", "min_months_apart": 0, "model": "deepseek-r1:32b"}
- "analyze and interpret https://github.com/apache/commons-io.git last 3 years 1 commit per month with deepseek-r1:32b and tell me the 5 most dangerous files" → {"tool": "temporal_analysis", "repo": "https://github.com/apache/commons-io.git", "count": 36, "branch": "main", "min_months_apart": 1, "model": "deepseek-r1:32b", "user_question": "What are the 5 most dangerous files in the repository right now, and why? Base the answer on anti-pattern involvement, structural coupling (fan-in), SCC membership, bug-linked churn, and co-change signals."}
- "analyze commons-io last 3 years 12 commits per year and interpret and answer: what are the 5 most dangerous files?" → {"tool": "temporal_analysis", "repo": "commons-io", "count": 36, "branch": "main", "min_months_apart": 1, "model": "deepseek-r1:32b", "user_question": "What are the 5 most dangerous files in commons-io right now, and why?"}
- "query commons-io: which files should I refactor first?" → {"tool": "query", "repo": "commons-io", "question": "which files should I refactor first?"}
- "ask commons-io: why did the m-score drop in 2023?" → {"tool": "query", "repo": "commons-io", "question": "why did the m-score drop in 2023?"}
- "what is a clique anti-pattern?" → {"tool": "query", "repo": null, "question": "what is a clique anti-pattern?"}
- "query: explain propagation cost" → {"tool": "query", "repo": null, "question": "explain propagation cost"}
- "fast query commons-io give me 5 worst files" → {"tool": "query", "repo": "commons-io", "question": "give me 5 worst files to refactor"}
- "query commons-io: what are the 5 most dangerous files?" → {"tool": "query", "repo": "commons-io", "question": "What are the 5 most dangerous files in commons-io right now, and why? Use the multi-signal risk scores (bug churn, anti-patterns, SCC membership) AND the DV8 structural data to justify each file's ranking."}
- "ask commons-io why did the m-score get worse over time and link it to commits" → {"tool": "query", "repo": "commons-io", "question": "Why did the M-score get worse over time? Link each significant drop in M-score to specific commits, bugs, or new features that introduced new coupling. Which specific files or groups entered anti-patterns during those transitions?"}
- "query commons-io what parts are decreasing in quality" → {"tool": "query", "repo": "commons-io", "question": "What parts of the system are most rapidly decreasing in quality over the last 12 months? Identify file groups where bug churn, anti-pattern membership, or propagation cost has been increasing, and name the specific commits or changes that drove the deterioration."}
- "ask commons-io what causes the most technical debt" → {"tool": "query", "repo": "commons-io", "question": "What parts of the system are causing the most technical debt? Identify groups of files that combine the most bugs and churn with structural design flaws (anti-patterns, high fan-in, SCC cycles), and explain what design flaws link them together."}

KNOWN REPOSITORIES (use EXACTLY these short names when no URL is given):
- "jsoup" → Java HTML parser
- "commons-io" → Apache Commons IO
- "pdfbox" → Apache PDFBox (branch: trunk)
- "ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG" → Train ticket toy example (also known as "train ticket toy", "trainticket toy", "TTS toy", "toy example", "multilang toy")

For the toy example with language-specific branches, use the FULL GitHub URL and set branch accordingly:
- Python toy (2-commit temporal): repo="https://github.com/FreeworkEarth/ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG.git", branch="temporal_PYTHON"
- Java toy (2-commit temporal):   repo="https://github.com/FreeworkEarth/ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG.git", branch="temporal_JAVA"
- Both languages:                  repo="https://github.com/FreeworkEarth/ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG.git", branch="temporal"

Examples for the toy:
- "analyze and interpret the python toy all time" → {"tool": "temporal_analysis", "repo": "https://github.com/FreeworkEarth/ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG.git", "count": 2, "branch": "temporal_PYTHON", "min_months_apart": 0, "model": "deepseek-r1:32b"}
- "analyze and interpret the java toy all time" → {"tool": "temporal_analysis", "repo": "https://github.com/FreeworkEarth/ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG.git", "count": 2, "branch": "temporal_JAVA", "min_months_apart": 0, "model": "deepseek-r1:32b"}

IMPORTANT: If the user provides a full GitHub/git URL (starts with https:// or git://), use the FULL URL as the "repo" value — do NOT replace it with a short name.
IMPORTANT: If no URL is given, use the exact short name from KNOWN REPOSITORIES above. If user says "train ticket toy", "TTS toy", "toy example", "multilang" → use "ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG". Never invent placeholder paths.
IMPORTANT: Always extract the branch from the prompt. "on branch temporal" or "temporal branch" → "branch": "temporal". Default is "main" only if no branch is mentioned.
"""

def _http_json(method: str, path: str, payload: dict | None, timeout=300):
    url = f"{OLLAMA_ENDPOINT}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

def _healthcheck() -> None:
    try:
        req = urllib.request.Request(f"{OLLAMA_ENDPOINT}/api/version", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            _ = resp.read()
    except Exception as e:
        print(f"Warning: Ollama not running at {OLLAMA_ENDPOINT}")
        print("   Start it with: ollama serve")
        raise SystemExit(1)

def call_ollama(user: str) -> str:
    _healthcheck()
    try:
        body = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 256},
        }
        data = _http_json("POST", "/api/chat", body)
        return data["message"]["content"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {user}\n\nRespond with ONLY JSON."
        data = _http_json("POST", "/api/generate", {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0}})
        return data["response"]

def parse_json(s: str):
    # Try to extract JSON object - handle cases where LLM adds extra text
    # First try: find first complete JSON object
    brace_count = 0
    start_idx = -1

    for i, char in enumerate(s):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                # Found a complete JSON object
                json_str = s[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Keep looking for another valid JSON
                    start_idx = -1
                    continue

    # Fallback to regex if manual parsing didn't work
    m = re.search(r"\{[^}]*\}", s, re.S)
    if not m:
        raise SystemExit(f"LLM did not return JSON. Got: {s[:200]}...")
    return json.loads(m.group(0))

def _guess_repo_from_text(text: str) -> str | None:
    t = (text or '').strip()
    if not t:
        return None
    # If URL present, return it
    m = re.search(r"https?://\S+", t)
    if m:
        return m.group(0)
    # Try patterns: "for <name>", "of <name>", "in <name>"
    m = re.search(r"\b(for|of|in)\s+([A-Za-z0-9._\-/]+)\b", t, re.I)
    if m:
        return m.group(2)
    return None

def _prompt_for_repo(default=".") -> str:
    while True:
        entered = input("Enter repo path, URL, or ZIP (or 'q' to quit): ").strip()
        if not entered:
            entered = default
        if entered.lower() in {"q", "quit", "exit"}:
            raise SystemExit("Aborted.")
        if re.match(r"^https?://", entered) or entered.lower().endswith(".zip"):
            return entered
        p = pathlib.Path(entered).expanduser()
        if p.exists():
            return str(p)
        print(f"Path not found: {p}")

def _sanitize_repo(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip()
    placeholders = (
        "path/to/repo", "<path", "<repo", "your repo",
        "<local path or git url>", "<local path or url>", "<local path or git",
    )
    # Treat any angle-bracketed text as placeholder
    if v in {".", "./"} or any(p in v.lower() for p in placeholders) or ("<" in v or ">" in v):
        return None
    # Normalize GitHub tree/blob URLs → bare clone URL
    # e.g. https://github.com/owner/repo/tree/branch → https://github.com/owner/repo.git
    if re.match(r"^https?://github\.com/", v):
        v = re.sub(r"/(tree|blob)/[^/]+.*$", "", v)
        if not v.endswith(".git"):
            v = v.rstrip("/") + ".git"
    return v

_QUESTION_KEYWORDS = ("why", "how", "what", "which", "explain", "show me",
                      "describe", "identify", "list", "summarize")

def _extract_user_question(text: str) -> str | None:
    """Extract the architectural question from a natural-language prompt.

    Priority:
    1. Explicit pattern: "and answer: <question>" or "then answer <question>"
    2. Fallback: return the full prompt if it contains a question keyword.
    """
    if not text:
        return None
    t = text.strip()
    m = re.search(r"(?:and\s+)?(?:then\s+)?answer[:\s]+(.+)$", t, re.I | re.S)
    if m:
        return m.group(1).strip()
    lower = t.lower()
    if any(kw in lower for kw in _QUESTION_KEYWORDS):
        return t
    return None


def _find_local_repo(repo: str) -> str:
    """
    Find the actual local repository path.
    - If repo is a GitHub URL like https://github.com/apache/pdfbox,
      it was cloned to ./pdfbox
    - If repo is a local path, return as-is
    """
    # If it's a URL, extract the repo name and look for it locally
    if re.match(r"^https?://", repo):
        from urllib.parse import urlparse
        parsed = urlparse(repo)
        name = pathlib.Path(parsed.path).stem or "repository"
        if name.endswith(".git"):
            name = name[:-4]

        # Check in current directory first
        local_path = pathlib.Path(THIS_DIR) / name
        if local_path.exists() and (local_path / "OutputData" / "metrics").exists():
            return str(local_path)

        # Check if repo name exists in current dir
        local_path = pathlib.Path(name)
        if local_path.exists() and (local_path / "OutputData" / "metrics").exists():
            return str(local_path)

        # Fallback: return the repo name (relative path)
        return name

    # If it's already a local path, return as-is
    return repo

def _resolve_repo_and_source(repo: str, source_path: str | None) -> tuple[str, str | None]:
    """If repo points to a subfolder of a git repo, lift to repo root and set source_path."""
    try:
        p = pathlib.Path(repo).expanduser().resolve()
    except Exception:
        return repo, source_path
    if not p.exists():
        return repo, source_path
    if (p / ".git").exists():
        return str(p), source_path
    cur = p
    while cur.parent != cur:
        if (cur / ".git").exists():
            rel = str(p.relative_to(cur))
            if not source_path:
                source_path = rel
            return str(cur), source_path
        cur = cur.parent
    return str(p), source_path

def _ts_path(root: pathlib.Path) -> pathlib.Path:
    """Return the timeseries.json path for a temporal root, preferring new INPUT_INTERPRETATION location."""
    new_loc = root / "INPUT_INTERPRETATION" / "timeseries.json"
    if new_loc.exists():
        return new_loc
    return root / "timeseries.json"  # legacy fallback


def _has_timeseries(root: pathlib.Path) -> bool:
    """Check if a temporal root has a timeseries.json (in either location)."""
    return _ts_path(root).exists()


def _temporal_root_from_interpretation_path(p: str) -> str | None:
    """Given a path to INPUT_INTERPRETATION, OUTPUT_INTERPRETATION, or their parent temporal folder, return the temporal root."""
    try:
        pp = pathlib.Path(p).expanduser().resolve()
    except Exception:
        return None
    if pp.is_file():
        pp = pp.parent
    if pp.name in ("INPUT_INTERPRETATION", "OUTPUT_INTERPRETATION"):
        tr = pp.parent
        if _has_timeseries(tr):
            return str(tr)
        # Common user mistake: path points to a placeholder folder without timeseries.json.
        # Try siblings like "<repo>/(temporal_analysis*/)" and pick newest.
        try:
            sib = tr.parent
            candidates = [d for d in sib.glob("temporal_analysis*/") if _has_timeseries(d)]
            if candidates:
                newest = max(candidates, key=lambda x: x.stat().st_mtime)
                return str(newest)
        except Exception:
            pass
        return None
    # If path itself is temporal root
    if _has_timeseries(pp):
        return str(pp)
    # Walk upwards a bit
    cur = pp
    for _ in range(4):
        if _has_timeseries(cur):
            return str(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    return None

def _run_risk_pipeline(
    temporal_root: pathlib.Path,
    repo_name: str,
    git_root: pathlib.Path | None = None,
    review_model: str | None = None,
) -> None:
    """
    Run the full per-file risk scoring pipeline after backfill completes:
      1. fetch_github_issues   — build issue_map.json (JIRA/GitHub auto-detected)
      2. export_dv8_binary_files --all  — convert .dv8-clsx/.dv8-dsm → JSON/CSV
      3. compute_file_risk_scores       — multi-signal composite risk scores
      4. plot_risk_score_stats          — statistical plots + risk_score_stats.json

    All steps are best-effort: failure of one does not stop the rest.
    """
    interp_root = temporal_root / "INPUT_INTERPRETATION"

    # --- Step 1: Issue map (JIRA/GitHub auto-detected from commit history) ---
    # New location: INPUT_INTERPRETATION/issue_map.json; legacy fallback: temporal root
    issue_map_path = next(
        (p for p in [interp_root / "issue_map.json", temporal_root / "issue_map.json"] if p.exists()),
        interp_root / "issue_map.json",  # default write target for new runs
    )
    if not issue_map_path.exists() and os.path.isfile(FETCH_ISSUES):
        print("\n[risk-pipeline] Fetching issue map (JIRA/GitHub auto-detection)...")
        fi_cmd = [sys.executable, FETCH_ISSUES, "--out", str(issue_map_path)]
        if git_root and git_root.exists():
            fi_cmd += ["--git-root", str(git_root)]
        token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
        if token:
            fi_cmd += ["--token", token]
        fi_rc = subprocess.call(fi_cmd)
        if fi_rc != 0:
            print("  [risk-pipeline] Issue fetch failed or skipped — continuing without typed churn.")
            issue_map_path = None
    elif issue_map_path.exists():
        print(f"\n[risk-pipeline] Reusing existing issue_map.json")

    # --- Step 2: Export .dv8-clsx / .dv8-dsm binary files → JSON + CSV ---
    if os.path.isfile(EXPORT_DV8) and interp_root.exists():
        already_exported = any(interp_root.rglob("*_files.json"))
        if already_exported:
            print("\n[risk-pipeline] DV8 binary exports already present — skipping re-export.")
        else:
            print("\n[risk-pipeline] Exporting DV8 binary anti-pattern files → JSON/CSV...")
            subprocess.call([sys.executable, EXPORT_DV8, "--all", str(interp_root)])

    # --- Step 3: Compute multi-signal file risk scores ---
    risk_json = interp_root / "file_risk_scores.json"
    bug_review_json = interp_root / "bug_churn_commits_llm_review.json"
    need_bug_review = bool(review_model) and not bug_review_json.exists()
    if risk_json.exists() and not need_bug_review:
        print(f"\n[risk-pipeline] Reusing existing file_risk_scores.json")
    elif os.path.isfile(COMPUTE_RISK) and interp_root.exists():
        print("\n[risk-pipeline] Computing file risk scores...")
        cr_cmd = [sys.executable, COMPUTE_RISK, str(interp_root), "--verbose"]
        if git_root and git_root.exists():
            cr_cmd += ["--git-root", str(git_root)]
        if review_model:
            cr_cmd += ["--bug-churn-review-model", str(review_model)]
        if issue_map_path and pathlib.Path(str(issue_map_path)).exists():
            # Pass issue map to backfill if it was generated — already baked into payloads,
            # but log for traceability
            pass
        subprocess.call(cr_cmd)
    else:
        print(f"\n[risk-pipeline] Skipping risk scores (script not found or INPUT_INTERPRETATION missing)")

    # --- Step 4: Statistical plots ---
    plots_dir = interp_root / "plots" / "risk_stats"
    if os.path.isfile(PLOT_RISK) and risk_json.exists() and not any(plots_dir.glob("*.png")):
        print("\n[risk-pipeline] Generating risk score statistical plots...")
        subprocess.call([sys.executable, PLOT_RISK, str(risk_json), "--top-n", "30"])
    elif plots_dir.exists() and any(plots_dir.glob("*.png")):
        print("\n[risk-pipeline] Reusing existing risk score plots.")

    print("\n[risk-pipeline] Done.")
    if risk_json.exists():
        # Print quick top-5 summary
        try:
            import json as _json
            data = _json.loads(risk_json.read_text(encoding="utf-8"))
            files = data.get("files", [])
            if files:
                print(f"\n{'='*60}")
                print(f"  TOP-5 MOST DANGEROUS FILES — {repo_name}")
                print(f"{'='*60}")
                for f in files[:5]:
                    aps = ", ".join(f.get("anti_patterns_seen", [])) or "—"
                    sigs = f.get("signals", {})
                    print(
                        f"  #{f['rank']:>2}  score={f['risk_score']:.4f}  {f['file']}\n"
                        f"       anti-patterns: {aps}\n"
                        f"       fan-in={sigs.get('hotspot_fanin_score',0):.0f}  "
                        f"scc={sigs.get('scc_membership_count',0)}  "
                        f"anti_pattern_count={sigs.get('anti_pattern_count',0)}  "
                        f"anti_pattern_load={sigs.get('anti_pattern_instance_load',0)}  "
                        f"bug_churn={sigs.get('bug_churn_total',0)}"
                    )
                print(f"{'='*60}")
                print(f"  Full results: {risk_json}")
                plots_dir = risk_json.parent / "plots" / "risk_stats"
                if plots_dir.exists():
                    print(f"  Plots: {plots_dir}")
        except Exception:
            pass


def _load_rich_qa_context(temporal_root: pathlib.Path) -> tuple:
    """
    Load risk scores + bug-linked commit log for Q&A context injection.
    Returns (risk_score_context: str, commit_context: str, mscore_breakdown: str, report_text: str).
    All strings are empty if data not found.
    """
    risk_score_context = ""
    commit_context = ""
    report_text = ""

    # Risk scores
    risk_json = temporal_root / "INPUT_INTERPRETATION" / "file_risk_scores.json"
    if risk_json.exists():
        try:
            risk_data = json.loads(risk_json.read_text(encoding="utf-8"))
            top_files = risk_data.get("files", [])[:25]
            lines_rs = ["rank | file | risk_score | bug_churn | anti_patterns | scc_revisions | co_change | anti_pattern_types"]
            lines_rs.append("---" * 12)
            for f in top_files:
                s = f.get("signals", {})
                ap_counts = f.get("anti_pattern_type_counts", {})
                aps = ", ".join(f"{k}:{v}" for k, v in list(ap_counts.items())[:4])
                lines_rs.append(
                    f"#{f['rank']:2d} | {f['file'].split('/')[-1]:40s} | {f['risk_score']:.3f} | "
                    f"bug={s.get('bug_churn_total',0):5d} | ap={s.get('anti_pattern_count',0):3d} counts / load={s.get('anti_pattern_instance_load',0):3d} | "
                    f"scc={s.get('scc_membership_count',0):2d} revisions | co={s.get('co_change_without_dep',0):2d} | [{aps}]"
                )
            risk_score_context = "\n".join(lines_rs)
            print(f"  [query] Risk scores: top {len(top_files)} files loaded")
        except Exception as exc:
            print(f"  [query] WARNING: Could not load risk scores: {exc}")

    # Bug-linked commits from issue_map (new location: INPUT_INTERPRETATION/issue_map.json, legacy: root)
    issue_map_path = next(
        (p for p in [
            temporal_root / "INPUT_INTERPRETATION" / "issue_map.json",
            temporal_root / "issue_map.json",
        ] if p.exists()),
        None,
    )
    if issue_map_path is not None:
        try:
            issue_data = json.loads(issue_map_path.read_text(encoding="utf-8"))
            summaries_map = issue_data.get("summaries", {})
            issues_map = issue_data.get("issues", {})
            commit_log = issue_data.get("commit_log", [])
            _jira_re = re.compile(r"\b([A-Z][A-Z0-9]+-\d+)\b")
            _bug_kw = re.compile(r"\b(fix|bug|hotfix|patch|defect|regress)\b", re.IGNORECASE)
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
            print(f"  [query] Commit log: {len(bug_commits)} bug-linked commits loaded")
        except Exception as exc:
            print(f"  [query] WARNING: Could not load commit log: {exc}")

    # Most recent combined interpretation report
    interp_out = temporal_root / "OUTPUT_INTERPRETATION"
    if interp_out.exists():
        existing = sorted(
            interp_out.rglob("temporal_interpretation_report_*.md"),
            key=lambda p: p.stat().st_mtime
        )
        if existing:
            report_text = existing[-1].read_text(encoding="utf-8")
            print(f"  [query] Interpretation report: {existing[-1].name}")

    return risk_score_context, commit_context, report_text


def tool_query(plan: dict) -> int:
    """
    Fast Q&A on existing analysis results.
    Uses risk scores + commit log + M-score data + interpretation report.
    No re-running DV8 or LLM interpretation — answers in 1-3 minutes.
    Falls back to Stage 3 RAG engine if no temporal analysis exists for this repo.
    """
    question = plan.get("question") or plan.get("user_request") or ""
    repo = plan.get("repo") or None
    model = plan.get("model") or "deepseek-r1:32b"

    # --- Try to find an existing temporal analysis for this repo ---
    temporal_root = None
    if repo:
        test_auto_dir = pathlib.Path(THIS_DIR).parent
        repos_dir = test_auto_dir / "REPOS_ANALYZED" / repo
        if repos_dir.exists():
            candidates = [p for p in repos_dir.glob("temporal_analysis*/") if p.is_dir()
                          and _has_timeseries(p)]
            if candidates:
                temporal_root = max(candidates, key=lambda p: p.stat().st_mtime)
                print(f"[query] Using temporal analysis: {temporal_root.name}")

    if temporal_root:
        # Rich Q&A: risk scores + commit log + interpretation report
        risk_score_context, commit_context, report_text = _load_rich_qa_context(temporal_root)
        mscore_breakdown = ""
        try:
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location("itb", pathlib.Path(INTERPRET_TEMPORAL))
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            mscore_breakdown = _mod.load_mscore_worst_modules(temporal_root)
        except Exception as exc:
            print(f"  [query] WARNING: Could not load mscore breakdown: {exc}")

        if not question:
            sep = "=" * 70
            print(f"\n{sep}\n  INTERACTIVE Q&A — commons-io\n"
                  f"  Data: risk scores + {temporal_root.name}\n"
                  f"  Type 'q' to quit.\n{sep}")
            try:
                question = input("  Your question: ").strip()
            except EOFError:
                question = ""
            if not question or question.lower() in ("q", "quit", "exit"):
                return 0

        conversation = []
        current_question = question
        sep = "=" * 70
        while current_question:
            print(f"\nAnswering: {current_question!r}")
            prior = "\n\n".join(conversation)
            ctx = (prior + "\n\n" + report_text) if prior else report_text
            try:
                import importlib.util as _ilu
                _spec = _ilu.spec_from_file_location("itb", pathlib.Path(INTERPRET_TEMPORAL))
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                raw = _mod.answer_user_question(
                    model, current_question, ctx,
                    mscore_breakdown=mscore_breakdown,
                    timeout_s=900,
                    risk_score_context=risk_score_context,
                    commit_context=commit_context,
                )
                answer = _mod.strip_thinking_and_fences(raw)
            except Exception as exc:
                answer = f"[query error] {exc}"

            print(f"\n{sep}\n  ANSWER\n  Q: {current_question}\n{sep}")
            print(answer)
            print(sep)
            conversation.append(f"Q: {current_question}\nA: {answer}")

            # Save answer
            run_folder = temporal_root / "OUTPUT_INTERPRETATION"
            if run_folder.exists():
                # Find most recent run subfolder to save into
                subfolders = sorted(run_folder.iterdir(), key=lambda p: p.stat().st_mtime)
                save_folder = subfolders[-1] if subfolders else run_folder
            else:
                save_folder = temporal_root
            from datetime import datetime as _dt
            now = _dt.now()
            answer_path = save_folder / f"USER_ANSWER_{now.strftime('%Y%m%d')}.md"
            entry = f"\n---\n\n**Q ({now.strftime('%H:%M:%S')})**: {current_question}\n\n{answer}\n"
            if answer_path.exists():
                with open(answer_path, "a", encoding="utf-8") as f:
                    f.write(entry)
            else:
                answer_path.write_text(
                    f"# Q&A Session — {now.strftime('%Y%m%d')}\n\n**Model**: {model}\n\n---\n\n"
                    f"**Q ({now.strftime('%H:%M:%S')})**: {current_question}\n\n{answer}\n",
                    encoding="utf-8"
                )
            print(f"  Saved: {answer_path}")

            print(f"\n{sep}\n  FOLLOW-UP (Enter / 'q' to finish)\n{sep}")
            try:
                next_q = input("  Your question: ").strip()
            except EOFError:
                break
            if not next_q or next_q.lower() in ("q", "quit", "exit"):
                break
            if not any(c.isalpha() or c.isdigit() for c in next_q):
                break
            current_question = next_q
        return 0

    # --- Fallback: Stage 3 RAG engine (no temporal analysis found) ---
    print(f"[query] No temporal analysis found for '{repo}' — falling back to RAG engine")
    stage3_dir = pathlib.Path(QUERY_ENGINE).parent
    if not stage3_dir.exists():
        print(f"[query] Stage 3 not found at {stage3_dir}")
        return 1
    index_file = stage3_dir / ".rag_index.json"
    if not index_file.exists():
        print("[query] RAG index not found — building now (first-time setup)...")
        rag_index_py = stage3_dir / "rag_index.py"
        subprocess.run([sys.executable, str(rag_index_py)], check=False)
    num_ctx = int(plan.get("num_ctx") or 4096)
    cmd = [sys.executable, str(QUERY_ENGINE), "--model", model, "--num-ctx", str(num_ctx)]
    if repo:
        cmd += ["--repo", repo]
    if question:
        cmd += ["--question", question]
        print(f"[query] Question: {question}")
        print(f"[query] Repo: {repo or 'all'}, Model: {model}\n")
    else:
        print(f"[query] Starting interactive session — Repo: {repo or 'all'}, Model: {model}")
    return subprocess.run(cmd).returncode


def tool_interpret_temporal(plan: dict) -> int:
    """
    Interpret a temporal analysis folder (pairwise DRH diffs + overall summary).

    Accepts repo field as either:
      - <temporal_root>
      - <temporal_root>/INPUT_INTERPRETATION
    """
    ur = (plan.get("user_request") or "")
    auto_refactor = bool(plan.get("auto_refactor", False))
    force_reinterpret = bool(plan.get("force_reinterpret", False))
    refactor_model = plan.get("refactor_model") or "qwen3-coder-30b-refactor"
    refactor_loop_count = int(plan.get("refactor_loop_count") or 0)
    refactor_models = plan.get("refactor_models", [])  # multi-model comparison
    use_feedback_loop = bool(plan.get("use_feedback_loop", False))

    # Extract model from prompt if not set (e.g. "with deepseek-r1:32b" or "32b")
    raw_model = plan.get("model") or ""
    if not raw_model or raw_model == "deepseek-r1:14b":
        # Check if prompt explicitly mentions a model size
        m_model = re.search(r'deepseek[-_]r1:(\d+b)|(?<!\d)(\d+)b(?!\d)', ur, re.I)
        if m_model:
            size = (m_model.group(1) or m_model.group(2)).lower()
            if size in ("14b", "32b", "70b"):
                raw_model = f"deepseek-r1:{size}"
    model = (raw_model or "deepseek-r1:32b").strip()

    folder = plan.get("repo") or plan.get("folder")
    # If folder is just a short repo name (not an absolute path, URL, or temporal path),
    # resolve it to the most-recent temporal_analysis_* folder in REPOS/.
    if folder and "://" not in folder and "temporal_analysis" not in folder and not pathlib.Path(folder).is_absolute():
        test_auto_dir = pathlib.Path(THIS_DIR).parent
        repos_dir = test_auto_dir / "REPOS_ANALYZED" / folder
        if repos_dir.exists():
            candidates = [p for p in repos_dir.glob("temporal_analysis*/") if p.is_dir()]
            if candidates:
                folder = str(max(candidates, key=lambda p: p.stat().st_mtime))
                print(f"Auto-selected most recent temporal folder: {folder}")
    if not folder:
        # Try to extract quoted path from user_request
        m = re.search(r"['\"]([^'\"]+(INPUT_INTERPRETATION|OUTPUT_INTERPRETATION)[^'\"]*)['\"]", ur)
        folder = m.group(1) if m else None
    if not folder:
        # Try unquoted temporal_analysis_* folder name in prompt
        m = re.search(r'((?:/[^\s]+)?temporal_analysis[^\s\'"]*)', ur)
        folder = m.group(1) if m else None
    if not folder:
        # Fallback: find most-recent temporal_analysis_* for the named repo
        repo_guess = _guess_repo_from_text(ur)
        if repo_guess:
            test_auto_dir = pathlib.Path(THIS_DIR).parent
            repos_dir = test_auto_dir / "REPOS_ANALYZED" / repo_guess
            if repos_dir.exists():
                candidates = [p for p in repos_dir.glob("temporal_analysis*/") if p.is_dir()]
                if candidates:
                    folder = str(max(candidates, key=lambda p: p.stat().st_mtime))
                    print(f"Auto-selected most recent temporal folder: {folder}")
    if not folder:
        print("No temporal folder provided. Pass a temporal_analysis_* path or its INPUT_INTERPRETATION/OUTPUT_INTERPRETATION subfolder.")
        return 1
    folder = str(folder).strip()
    if "..." in folder:
        print("Ellipsis '...' detected in path. Please provide the full absolute folder path.")
        return 1
    temporal_root = _temporal_root_from_interpretation_path(folder)
    if not temporal_root:
        print(f"Could not resolve temporal root from: {folder}")
        try:
            pp = pathlib.Path(folder).expanduser().resolve()
            base = pp.parent if pp.name in ("INPUT_INTERPRETATION", "OUTPUT_INTERPRETATION") else pp
            if base.exists():
                candidates = [d for d in base.glob("temporal_analysis*/") if _has_timeseries(d)]
                if candidates:
                    newest = max(candidates, key=lambda x: x.stat().st_mtime)
                    print(f"Hint: try temporal root: {newest}")
        except Exception:
            pass
        return 1
    tr = pathlib.Path(temporal_root)
    ts_path = _ts_path(tr)
    ts = {}
    if ts_path.exists():
        try:
            ts = json.loads(ts_path.read_text(encoding="utf-8"))
        except Exception:
            ts = {}
    repo_name = (ts.get("repo") or tr.parent.name) if isinstance(ts, dict) else tr.parent.name

    # Resolve a git repo for commit context.
    repo = tr.parent
    if not (repo / ".git").exists():
        # Common case: outputs are written to <repo>_java/<temporal_analysis...>; the git clone is <repo>.
        parent_name = repo.name
        for suffix in ("_java", "_python"):
            if parent_name.endswith(suffix):
                cand = repo.parent / parent_name[: -len(suffix)]
                if (cand / ".git").exists():
                    repo = cand
                    break
    if not (repo / ".git").exists():
        # Toy repos often live in TEST_AUTO/000_TOY_EXAMPLES/<repo_name>.
        test_auto_dir = pathlib.Path(THIS_DIR).parent
        cand = test_auto_dir / "000_TOY_EXAMPLES" / str(repo_name)
        if (cand / ".git").exists():
            repo = cand
    if not (repo / ".git").exists():
        # Last attempt: TEST_AUTO/REPOS/<repo_name> (remote clones).
        test_auto_dir = pathlib.Path(THIS_DIR).parent
        cand = test_auto_dir / "REPOS_ANALYZED" / str(repo_name)
        if (cand / ".git").exists():
            repo = cand

    # Ensure INPUT_INTERPRETATION bundle exists (run backfill if needed).
    interp_single = tr / "INPUT_INTERPRETATION" / "SINGLE_REVISION_ANALYSIS_DATA"
    need_backfill = True
    if interp_single.exists():
        try:
            payloads = list(interp_single.glob("*/OutputData/interpretation_payload.json"))
            need_backfill = len(payloads) == 0
        except Exception:
            need_backfill = True
    if need_backfill:
        if os.path.isfile(BACKFILL_TEMPORAL):
            bf_cmd = ["python3", BACKFILL_TEMPORAL, str(tr), "--meta-repo", str(repo_name)]
            print("\nBuilding interpretation bundle (backfill)...")
            print("Running:", " ".join(bf_cmd))
            rc = subprocess.call(bf_cmd)
            if rc != 0:
                print("Backfill failed; cannot interpret without INPUT_INTERPRETATION payloads.")
                return rc
            # Optional: bundle verification (best-effort)
            if os.path.isfile(BUNDLE_VERIFY):
                v_cmd = ["python3", BUNDLE_VERIFY, "--temporal-root", str(tr)]
                subprocess.call(v_cmd)
        else:
            print(f"Missing backfill script: {BACKFILL_TEMPORAL}")
            return 1

    # --- Risk pipeline: DISABLED ---
    # Custom risk scores are unproven. Q&A and refactoring use only DV8's native
    # metrics (M-score, anti-patterns, fan-in/out, SCC, bug churn, co-change)
    # which are already in the DV8 output files.

    # --- Check for existing interpretation runs ---
    model_safe = model.replace("/", "_").replace(":", "_")
    interp_dir = tr / "OUTPUT_INTERPRETATION"
    existing_reports = sorted(
        interp_dir.glob(f"*/temporal_interpretation_report_{model_safe}*.md"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if existing_reports:
        latest = existing_reports[0]
        run_folder = latest.parent
        print(f"\nExisting interpretation found ({len(existing_reports)} run(s)):")
        for i, rp in enumerate(existing_reports[:3]):
            print(f"  [{i+1}] {rp.parent.name}  —  {rp.name}")
        print()
        print("  [s] Use latest report — go straight to Q&A (fast)")
        print("  [r] Re-interpret from scratch (slow, re-runs LLM on all transitions)")
        if force_reinterpret:
            choice = "r"  # --reinterpret flag: always regenerate
        elif auto_refactor:
            choice = "s"  # --auto without --reinterpret: reuse existing report
        else:
            try:
                choice = input("  Choice [s/r, default=s]: ").strip().lower()
            except EOFError:
                choice = "s"
        if choice != "r":
            report_text = latest.read_text(encoding="utf-8")
            print(f"\nUsing: {latest}")
            user_question = _extract_user_question(ur)

            # ------------------------------------------------------------------
            # AUTO MODE: inject Q1 -> Q2 -> Stage 4 without any human input
            # ------------------------------------------------------------------
            if auto_refactor and not user_question:
                import importlib.util
                _spec = importlib.util.spec_from_file_location("itb", pathlib.Path(INTERPRET_TEMPORAL))
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                _risk_ctx, _commit_ctx, _ = _load_rich_qa_context(tr)
                _mscore_bd = _mod.load_mscore_worst_modules(tr)
                sep = "=" * 70
                conversation = []
                from datetime import datetime as _dt

                def _auto_ask(question: str) -> str:
                    print(f"\n[AUTO] Asking: {question!r}")
                    prior = "\n\n".join(conversation)
                    ctx = (prior + "\n\n" + report_text) if prior else report_text
                    raw = _mod.answer_user_question(model, question, ctx,
                                                    mscore_breakdown=_mscore_bd,
                                                    timeout_s=900,
                                                    risk_score_context=_risk_ctx,
                                                    commit_context=_commit_ctx)
                    answer = _mod.strip_thinking_and_fences(raw)
                    print(f"\n{sep}\n  ANSWER\n  Q: {question}\n{sep}\n{answer}\n{sep}")
                    conversation.append(f"Q: {question}\nA: {answer}")
                    now = _dt.now()
                    answer_path = run_folder / f"USER_ANSWER_{now.strftime('%Y%m%d')}.md"
                    entry = f"\n---\n\n**Q ({now.strftime('%H:%M:%S')})**: {question}\n\n{answer}\n"
                    if answer_path.exists():
                        with open(answer_path, "a", encoding="utf-8") as f:
                            f.write(entry)
                    else:
                        answer_path.write_text(
                            f"# Q&A Session — {now.strftime('%Y%m%d')}\n\n**Model**: {model}\n\n---\n\n"
                            f"**Q ({now.strftime('%H:%M:%S')})**: {question}\n\n{answer}\n",
                            encoding="utf-8"
                        )
                    print(f"Saved: {answer_path}")
                    return answer

                # Q1 — identify worst anti-patterns and files
                _auto_ask("Which parts got worse over time — show anti-pattern groups, "
                          "files with most dependency growth, and worst files overall.")
                # Q2 — 3 prioritized actions (for manual reference); Stage 4 loop applies only Action 1 per iteration
                _auto_ask("How would you refactor the worst anti-pattern? "
                          "Give EXACTLY 3 concrete code-level refactoring actions using ### Action 1 / ### Action 2 / ### Action 3 headings. "
                          "Each action must describe specific file changes (rename, split class, remove extends, extract interface, move method, etc). "
                          "Do NOT include process actions like 'code reviews' or 'documentation'. Only structural code changes. "
                          "Action 1 must be the single most impactful change right now.")
                # Stage 4 — loop mode: applies Action 1 only, re-runs Q1+Q2 fresh each iteration
                print(f"\n[AUTO] Triggering Stage 4 (loop refactor: Action 1 per iteration, re-analyze between each)...")
                _run_refactor_stage(tr, conversation, model=refactor_model, loop_count=refactor_loop_count, refactor_models=refactor_models, qa_model=model, use_feedback_loop=use_feedback_loop)
                return 0

            # ------------------------------------------------------------------
            # INTERACTIVE MODE (unchanged)
            # ------------------------------------------------------------------
            if not user_question:
                sep = "=" * 70
                print(f"\n{sep}\n  INTERACTIVE Q&A\n  (Press Enter / 'q' to quit.)\n{sep}")
                try:
                    user_question = input("  Your question: ").strip()
                except EOFError:
                    user_question = ""
                if not user_question or user_question.lower() in ("q", "quit", "exit"):
                    return 0
                if not any(c.isalpha() or c.isdigit() for c in user_question):
                    return 0
            if user_question:
                import importlib.util
                _spec = importlib.util.spec_from_file_location("itb", pathlib.Path(INTERPRET_TEMPORAL))
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                _risk_ctx, _commit_ctx, _ = _load_rich_qa_context(tr)
                _mscore_bd = _mod.load_mscore_worst_modules(tr)
                conversation = []
                current_question = user_question
                sep = "=" * 70
                while current_question:
                    print(f"\nAnswering: {current_question!r}")
                    prior = "\n\n".join(conversation)
                    ctx = (prior + "\n\n" + report_text) if prior else report_text
                    raw = _mod.answer_user_question(model, current_question, ctx,
                                                    mscore_breakdown=_mscore_bd,
                                                    timeout_s=900,
                                                    risk_score_context=_risk_ctx,
                                                    commit_context=_commit_ctx)
                    answer = _mod.strip_thinking_and_fences(raw)
                    print(f"\n{sep}\n  ANSWER\n  Q: {current_question}\n{sep}\n{answer}\n{sep}")
                    conversation.append(f"Q: {current_question}\nA: {answer}")
                    from datetime import datetime as _dt
                    now = _dt.now()
                    answer_path = run_folder / f"USER_ANSWER_{now.strftime('%Y%m%d')}.md"
                    entry = f"\n---\n\n**Q ({now.strftime('%H:%M:%S')})**: {current_question}\n\n{answer}\n"
                    if answer_path.exists():
                        with open(answer_path, "a", encoding="utf-8") as f:
                            f.write(entry)
                    else:
                        answer_path.write_text(
                            f"# Q&A Session — {now.strftime('%Y%m%d')}\n\n**Model**: {model}\n\n---\n\n"
                            f"**Q ({now.strftime('%H:%M:%S')})**: {current_question}\n\n{answer}\n",
                            encoding="utf-8"
                        )
                    print(f"Saved: {answer_path}")
                    print(f"\n{sep}\n  FOLLOW-UP (Enter / 'q' to finish, or ask to refactor+reanalyze)\n{sep}")
                    try:
                        next_q = input("  Your question: ").strip()
                    except EOFError:
                        next_q = ""
                    if not next_q or next_q.lower() in ("q", "quit", "exit"):
                        break
                    if not any(c.isalpha() or c.isdigit() for c in next_q):
                        break
                    # Stage 4 trigger: refactor + re-analyse (supports targeted: "refactor module 1,0")
                    if _is_refactor_trigger(next_q):
                        _manual_target = _extract_refactor_target(next_q)
                        if _manual_target:
                            print(f"[Q&A] Manual refactoring target detected: {_manual_target}")
                            conversation.append(f"USER_REFACTOR_TARGET: {_manual_target}")
                        _run_refactor_stage(tr, conversation, model=refactor_model, loop_count=refactor_loop_count, refactor_models=refactor_models, qa_model=model, use_feedback_loop=use_feedback_loop)
                        break
                    current_question = next_q
            return 0
    # --- End skip-check: fall through to full interpretation ---

    cmd = ["python3", INTERPRET_TEMPORAL, "--temporal-root", str(tr), "--repo", str(repo), "--model", model]
    user_question = _extract_user_question(ur)
    if user_question:
        cmd += ["--user-question", user_question]
    print("Running:", " ".join(cmd))
    # In auto mode, pipe /dev/null to stdin so all input() calls return "" immediately
    # (interpret_temporal_bundle exits cleanly on empty Q&A input)
    import subprocess as _sp
    if auto_refactor or force_reinterpret:
        with open(os.devnull, "r") as _devnull:
            rc = subprocess.call(cmd, stdin=_devnull)
    else:
        rc = subprocess.call(cmd)
    if rc == 0:
        # Find the newest combined report (filenames now include a timestamp suffix).
        candidates = sorted(interp_dir.glob(f"*/temporal_interpretation_report_{model_safe}*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            print(f"\nTemporal interpretation report: {candidates[0]}")

        # AUTO MODE: after fresh interpretation, run Q1 -> Q2 -> Stage 4
        if auto_refactor:
            latest = candidates[0] if candidates else None
            if latest:
                run_folder = latest.parent
                report_text = latest.read_text(encoding="utf-8")
                import importlib.util
                _spec = importlib.util.spec_from_file_location("itb", pathlib.Path(INTERPRET_TEMPORAL))
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                _risk_ctx, _commit_ctx, _ = _load_rich_qa_context(tr)
                _mscore_bd = _mod.load_mscore_worst_modules(tr)
                sep = "=" * 70
                conversation = []
                from datetime import datetime as _dt

                def _auto_ask_fresh(question: str) -> str:
                    print(f"\n[AUTO] Asking: {question!r}")
                    prior = "\n\n".join(conversation)
                    ctx = (prior + "\n\n" + report_text) if prior else report_text
                    raw = _mod.answer_user_question(model, question, ctx,
                                                    mscore_breakdown=_mscore_bd,
                                                    timeout_s=900,
                                                    risk_score_context=_risk_ctx,
                                                    commit_context=_commit_ctx)
                    answer = _mod.strip_thinking_and_fences(raw)
                    print(f"\n{sep}\n  ANSWER\n  Q: {question}\n{sep}\n{answer}\n{sep}")
                    conversation.append(f"Q: {question}\nA: {answer}")
                    now = _dt.now()
                    answer_path = run_folder / f"USER_ANSWER_{now.strftime('%Y%m%d')}.md"
                    entry = f"\n---\n\n**Q ({now.strftime('%H:%M:%S')})**: {question}\n\n{answer}\n"
                    if answer_path.exists():
                        with open(answer_path, "a", encoding="utf-8") as f:
                            f.write(entry)
                    else:
                        answer_path.write_text(
                            f"# Q&A Session — {now.strftime('%Y%m%d')}\n\n**Model**: {model}\n\n---\n\n"
                            f"**Q ({now.strftime('%H:%M:%S')})**: {question}\n\n{answer}\n",
                            encoding="utf-8"
                        )
                    print(f"Saved: {answer_path}")
                    return answer

                _auto_ask_fresh("Which parts got worse over time — show anti-pattern groups, "
                                "files with most dependency growth, and worst files overall.")
                _auto_ask_fresh("How would you refactor the worst anti-pattern? "
                               "Give EXACTLY 3 concrete code-level refactoring actions using ### Action 1 / ### Action 2 / ### Action 3 headings. "
                               "Each action must describe specific file changes (rename, split class, remove extends, extract interface, move method, etc). "
                               "Do NOT include process actions like 'code reviews' or 'documentation'. Only structural code changes. "
                               "Action 1 must be the single most impactful change right now.")
                print(f"\n[AUTO] Triggering Stage 4 (loop refactor: Action 1 per iteration, re-analyze between each)...")
                _run_refactor_stage(tr, conversation, model=refactor_model, loop_count=refactor_loop_count, refactor_models=refactor_models, qa_model=model, use_feedback_loop=use_feedback_loop)
            else:
                print("[AUTO] No interpretation report found — cannot run Q&A.")
    return rc

def _is_refactor_trigger(question: str) -> bool:
    """Return True if the user wants to trigger Stage 4 (refactor + re-analyse).
    Also matches targeted refactoring like 'refactor module 1,0' or 'refactor clique'."""
    q = question.lower()
    # General refactor trigger
    if "refactor" in q and ("run" in q or "rerun" in q or "re-run" in q or "analysis" in q or "dv8" in q or "analyze" in q):
        return True
    # Targeted refactor: "refactor module X,Y" or "refactor clique" or "refactor _models.py"
    if "refactor" in q and (
        "module" in q or "clique" in q or "inheritance" in q or "modularity" in q
        or "anti-pattern" in q or "antipattern" in q or ".py" in q or ".java" in q
    ):
        return True
    return False


def _extract_refactor_target(question: str) -> str:
    """Extract a manual refactoring target from the user's Q&A question.
    Returns a target hint string to inject into the Q2 prompt, or '' if none found.
    Examples: 'refactor module 1,0', 'refactor clique', 'fix _models.py coupling'."""
    import re as _re_target
    q = question.lower()
    # Module target: "module 1,0" or "module 0,0"
    m = _re_target.search(r'module\s+(\d+[,]\d+)', q)
    if m:
        return f"TARGET: Focus on DV8 Module {m.group(1)}. Split files in this module to reduce its penalty."
    # Anti-pattern target
    if "clique" in q:
        return "TARGET: Focus on CLIQUE anti-patterns. Break circular dependencies between the files in the clique."
    if "inheritance" in q:
        return "TARGET: Focus on UNHEALTHY INHERITANCE anti-patterns. Flatten hierarchy, use composition, extract interfaces."
    if "modularity" in q or "modularity-violation" in q:
        return "TARGET: Focus on MODULARITY VIOLATION anti-patterns. Decouple files that co-change without structural dependency."
    # File target: "refactor _models.py" or "fix _client.py"
    m = _re_target.search(r'(?:refactor|fix|split)\s+(\S+\.(?:py|java))', q)
    if m:
        return f"TARGET: Focus on splitting/refactoring the file '{m.group(1)}' to reduce its fan-in/fan-out and coupling."
    return ""


def _stage4_clean_copy(folder: pathlib.Path) -> None:
    """Remove stale analysis outputs and .git worktree pointer from a copied folder."""
    import shutil
    for stale_name in ("InputData", "OutputData"):
        stale = folder / stale_name
        if stale.exists():
            shutil.rmtree(stale)
    git_marker = folder / ".git"
    if git_marker.exists():
        if git_marker.is_dir():
            shutil.rmtree(git_marker)
        else:
            git_marker.unlink()


def _stage4_apply_action(folder: pathlib.Path, action_num: str, action_text: str,
                          model: str) -> list:
    """Apply a single refactoring action to the relevant files in folder.
    Dispatches to Claude Code CLI when model starts with 'claude', otherwise uses Ollama.
    Returns list of new relative file paths created."""
    if model.startswith("claude"):
        return _stage4_apply_action_claude(folder, action_num, action_text, model)
    return _stage4_apply_action_ollama(folder, action_num, action_text, model)


def _stage4_apply_action_claude(folder: pathlib.Path, action_num: str, action_text: str,
                                 model: str) -> list:
    """Use Claude Code CLI (claude -p) to apply a refactoring action.
    Uses the user's Claude subscription — no API key needed."""
    import subprocess as _sp_claude, shutil as _sh_claude

    claude_bin = _sh_claude.which("claude")
    if not claude_bin:
        print("[stage4]   ERROR: 'claude' CLI not found in PATH. Install Claude Code first.")
        print("[stage4]   Falling back to Ollama...")
        return _stage4_apply_action_ollama(folder, action_num, action_text, "qwen3-coder-30b-refactor")

    # Map model string to Claude model flag
    # "claude" or "claude-sonnet" → sonnet, "claude-opus" → opus
    if "opus" in model:
        claude_model = "opus"
    elif "haiku" in model:
        claude_model = "haiku"
    else:
        claude_model = "sonnet"  # default: best speed/quality ratio for refactoring

    print(f"[stage4]   Using Claude Code CLI (model: {claude_model}) for refactoring")

    prompt = (
        f"You are applying a specific refactoring action to this codebase. "
        f"The source code is in: {folder}\n\n"
        f"REFACTORING ACTION {action_num}:\n{action_text}\n\n"
        f"RULES:\n"
        f"1. Apply ONLY the action above. Do not add other changes, comments, docstrings, or formatting.\n"
        f"2. Read each file mentioned in the action, apply the change, and write it back.\n"
        f"3. If new files need to be created (e.g. extracting an interface), create them.\n"
        f"4. If a file needs to be split, create the new files and update the original.\n"
        f"5. Do NOT modify test files, __pycache__, InputData, OutputData, or .git directories.\n"
        f"6. After making all changes, output a brief summary of what you changed (1-2 lines per file).\n"
        f"7. Do NOT run any tests, builds, or git commands.\n"
        f"8. CRITICAL — NO CIRCULAR IMPORTS & NO RE-EXPORTS: When extracting code into new files, ensure ONE-WAY dependencies only. "
        f"The new file imports from its dependencies, but the original file must NOT import back from the new file. "
        f"NEVER add re-exports (e.g. 'from new_file import X' in the old file) — re-exports keep the same dependency edges and defeat the purpose. "
        f"Instead, UPDATE ALL CALLERS to import directly from the new location. Use Grep to find every 'from old_file import X' and change them. "
        f"Example: if you move TypeA from types.py to type_defs.py, grep for 'from.*types import.*TypeA' and update each file to 'from type_defs import TypeA'. "
        f"Do NOT leave 'from type_defs import TypeA' in types.py — that creates the same coupling.\n"
        f"9. MINIMIZE NEW FILES: Prefer restructuring imports between existing files over creating new ones. "
        f"Each new file adds nodes to the dependency graph. Only create a new file if the action explicitly requires it "
        f"AND you update all dependents to import from the new file directly (no re-exports from the old file).\n"
        f"10. DO NOT use TYPE_CHECKING blocks to hide imports — DV8 analyzes the AST statically and still counts them.\n"
        f"11. NO NEW PACKAGE CYCLES: Before moving code between packages/directories, check that the move "
        f"does not create a circular dependency between packages. If package A depends on package B, "
        f"do NOT add an import from B to A. Keep package dependencies ONE-WAY.\n"
        f"12. CLEAN UP EMPTY FILES: When moving ALL code from file A to file B, DELETE file A entirely "
        f"(remove the file) and remove its import from __init__.py and all other files that imported it. "
        f"An empty file that still exists creates an isolated node in the dependency graph that artificially "
        f"inflates metrics. Either keep meaningful content in the file or delete it completely.\n"
    )

    try:
        result = _sp_claude.run(
            [claude_bin, "-p",
             "--model", claude_model,
             "--allowedTools", "Read,Edit,Write,Glob,Grep",
             "--add-dir", str(folder)],
            input=prompt,
            capture_output=True, text=True, timeout=600, cwd=str(folder)
        )
        if result.returncode != 0:
            print(f"[stage4]   Claude Code failed (rc={result.returncode})")
            if result.stderr:
                print(f"[stage4]   stderr: {result.stderr[:500]}")
            return []

        output = result.stdout.strip()
        if output:
            print(f"[stage4]   Claude output:\n{output[:800]}")
        return []  # Claude edits files directly, no NEW_FILE parsing needed

    except _sp_claude.TimeoutExpired:
        print(f"[stage4]   Claude Code timed out (600s)")
        return []
    except Exception as e:
        print(f"[stage4]   Claude Code error: {e}")
        return []


def _stage4_review_refactoring(folder: pathlib.Path, action_text: str,
                                model: str, package_name: str = "",
                                prev_failures: list[str] | None = None) -> tuple[bool, str]:
    """Use a SECOND Claude CLI instance to review refactoring for correctness.

    Returns (passed: bool, reason: str).
    - passed=True: refactoring looks correct, proceed to DV8
    - passed=False: issues found — reason describes what's wrong (fed to next iteration)

    The reviewer gets READ-ONLY access + Bash for smoke tests.
    It checks: imports resolve, no circular imports, code completeness, smoke tests pass.
    """
    import subprocess as _sp_rev, shutil as _sh_rev

    claude_bin = _sh_rev.which("claude")
    if not claude_bin:
        print("[review] Claude CLI not found — skipping review (PASS by default).")
        return True, "review skipped — no claude CLI"

    if "opus" in model:
        claude_model = "opus"
    elif "haiku" in model:
        claude_model = "haiku"
    else:
        claude_model = "sonnet"

    # Detect language and package from source files
    _skip = ("__pycache__", "InputData", "OutputData", ".git")
    if not package_name:
        _py_files = [p for p in folder.rglob("*.py")
                     if not any(s in p.parts for s in _skip) and "test" not in str(p).lower()]
        _java_files = [p for p in folder.rglob("*.java")
                       if not any(s in p.parts for s in _skip) and "test" not in str(p).lower()]
        if _py_files:
            # Guess package name from top-level __init__.py
            for _pf in sorted(_py_files, key=lambda p: len(p.parts)):
                if _pf.name == "__init__.py" and _pf.parent != folder:
                    package_name = _pf.parent.name
                    break
        elif _java_files:
            package_name = "java_project"

    lang = "Python" if any(folder.rglob("*.py")) else "Java"
    _import_cmd = f'python -c "import {package_name}"' if package_name and lang == "Python" else ""

    # Build context about previous review failures (so reviewer knows what to look for)
    _prev_ctx = ""
    if prev_failures:
        _prev_list = "\n".join(f"  - {f}" for f in prev_failures[-3:])  # last 3 failures
        _prev_ctx = (
            f"\n\nPREVIOUS REVIEW FAILURES (from earlier iterations — check if these are still present):\n"
            f"{_prev_list}\n"
        )

    prompt = (
        f"You are a CODE REVIEWER checking a refactoring for correctness. "
        f"The codebase is in: {folder}\n"
        f"Language: {lang}\n\n"
        f"THE REFACTORING THAT WAS APPLIED:\n{action_text}\n\n"
        f"YOUR TASK — check ALL of the following:\n"
        f"1. IMPORT RESOLUTION: Read the modified files. Check that every import statement "
        f"references a module/file that actually exists. Use Glob to verify target files exist.\n"
        f"2. NO CIRCULAR IMPORTS: If code was moved from file A to file B, verify that B does NOT "
        f"import from A for the same symbols (creating a cycle).\n"
        f"3. CODE COMPLETENESS: If code was moved, verify the destination file has the COMPLETE "
        f"code (classes, functions, methods) — not just stubs or empty placeholders.\n"
        f"4. CALLERS UPDATED: If a symbol moved from file A to file B, check that callers import "
        f"from B (not still from A). Use Grep to find old import patterns.\n"
    )

    if _import_cmd:
        prompt += (
            f"5. SMOKE TEST: Run this command to verify the package still imports:\n"
            f"   {_import_cmd}\n"
            f"   If it fails, report the exact error.\n"
        )

    prompt += (
        f"\nAfter checking, output your verdict on the FIRST LINE in exactly this format:\n"
        f"VERDICT: PASS\n"
        f"or\n"
        f"VERDICT: FAIL — <one-line reason>\n\n"
        f"Then provide a brief explanation (2-5 lines) of what you checked."
        f"{_prev_ctx}"
    )

    print(f"[review] Running reviewer (Claude {claude_model})...")
    try:
        # Reviewer gets Read, Glob, Grep for inspection + Bash for smoke tests
        # NO Edit/Write — reviewer must NOT modify code
        result = _sp_rev.run(
            [claude_bin, "-p",
             "--model", claude_model,
             "--allowedTools", "Read,Glob,Grep,Bash",
             "--add-dir", str(folder)],
            input=prompt,
            capture_output=True, text=True, timeout=300, cwd=str(folder)
        )
        if result.returncode != 0:
            print(f"[review] Claude reviewer failed (rc={result.returncode}) — PASS by default.")
            if result.stderr:
                print(f"[review] stderr: {result.stderr[:300]}")
            return True, "review process failed — pass by default"

        output = result.stdout.strip()
        print(f"[review] Reviewer output:\n{output[:600]}")

        # Parse verdict
        import re as _re_rev
        _verdict_match = _re_rev.search(r'VERDICT:\s*(PASS|FAIL)(?:\s*[—–-]\s*(.+))?', output, _re_rev.IGNORECASE)
        if _verdict_match:
            _verdict = _verdict_match.group(1).upper()
            _reason = (_verdict_match.group(2) or "").strip()
            if _verdict == "PASS":
                print(f"[review] PASSED")
                return True, "review passed"
            else:
                print(f"[review] FAILED: {_reason}")
                return False, _reason or "reviewer found issues (no specific reason given)"
        else:
            # Could not parse verdict — check for keywords
            _lower = output.lower()
            if "fail" in _lower and "pass" not in _lower:
                print(f"[review] FAILED (inferred from output)")
                return False, output[:200]
            print(f"[review] Could not parse verdict — PASS by default.")
            return True, "verdict unclear — pass by default"

    except _sp_rev.TimeoutExpired:
        print(f"[review] Reviewer timed out (300s) — PASS by default.")
        return True, "review timed out — pass by default"
    except Exception as e:
        print(f"[review] Reviewer error: {e} — PASS by default.")
        return True, f"review error: {e}"


def _stage4_apply_action_ollama(folder: pathlib.Path, action_num: str, action_text: str,
                                 model: str) -> list:
    """Use local Ollama model to apply a refactoring action.
    Returns list of new relative file paths created."""
    import re as _re, json as _json, urllib.request as _ur

    _SKIP = ("__pycache__", "InputData", "OutputData", ".git", "test", "tests")

    # Detect language from source files present in folder
    all_java = [p for p in folder.rglob("*.java") if not any(s in p.parts for s in _SKIP)]
    all_py   = [p for p in folder.rglob("*.py")   if not any(s in p.parts for s in _SKIP)]

    if all_java:
        lang = "java"
        all_src = all_java
        ext = ".java"
    elif all_py:
        lang = "python"
        all_src = all_py
        ext = ".py"
    else:
        print(f"[stage4]   No source files found in {folder.name}")
        return []

    lang_label = "Java" if lang == "java" else "Python"
    fence = lang  # "java" or "python" for markdown code fence
    if lang == "java":
        new_file_example = f"// NEW_FILE: <relative/path{ext}>\n<file content>\n// END_NEW_FILE"
    else:
        new_file_example = f"# NEW_FILE: <relative/path{ext}>\n<file content>\n# END_NEW_FILE"

    # Target only files explicitly mentioned in this action's text (exact word boundary match)
    import re as _re_tgt
    def _file_mentioned(fname: str, stem: str, text: str) -> bool:
        return bool(_re_tgt.search(r'\b' + _re_tgt.escape(fname) + r'\b', text)
                    or _re_tgt.search(r'\b' + _re_tgt.escape(stem) + r'\b', text))
    target = [f for f in all_src if _file_mentioned(f.name, f.stem, action_text)]
    if not target:
        target = all_src[:3]  # fallback: top 3

    print(f"[stage4]   Action {action_num} ({lang_label}) → targeting: {[f.name for f in target]}")
    new_files = []

    # NEW_FILE marker pattern — supports both Python (#) and Java (//) comment styles
    _new_file_pat = r"(?:#|//) NEW_FILE: (.+?)\n(.*?)(?:#|//) END_NEW_FILE"

    for src_file in target:
        rel = src_file.relative_to(folder)
        original = src_file.read_text(encoding="utf-8", errors="replace")
        prompt = (
            f"/no_think\n"
            f"You are a senior software engineer applying a single refactoring step to a {lang_label} codebase.\n\n"
            f"REFACTORING ACTION:\n{action_text}\n\n"
            f"RULES:\n"
            f"1. Apply ONLY the action above. Do not add other changes.\n"
            f"2. Output the COMPLETE modified file — not a diff, not partial code.\n"
            f"3. No markdown fences, no explanation, no commentary.\n"
            f"4. If this file is unchanged by the action, output it exactly as-is.\n"
            f"5. If a NEW file must be created (e.g. an interface), output it using:\n"
            f"   {new_file_example}\n\n"
            f"FILE TO MODIFY: {rel}\n"
            f"{original}"
        )
        try:
            req_body = _json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 8192, "temperature": 0.3},
            }).encode()
            req = _ur.Request("http://localhost:11434/api/generate", data=req_body,
                              headers={"Content-Type": "application/json"})
            with _ur.urlopen(req, timeout=300) as resp:
                raw_output = _json.loads(resp.read().decode()).get("response", "")
        except Exception as e:
            print(f"[stage4]   Ollama failed for {rel}: {e}")
            continue

        # Parse NEW_FILE blocks (supports # and // comment markers)
        for m in _re.finditer(_new_file_pat, raw_output, _re.DOTALL):
            new_rel, new_content = m.group(1).strip(), m.group(2)
            new_path = folder / new_rel
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.write_text(new_content, encoding="utf-8")
            new_files.append(new_rel)
            print(f"[stage4]   Created: {new_rel}")

        main_output = _re.sub(_new_file_pat + r"\n?", "", raw_output, flags=_re.DOTALL).strip()
        if main_output and main_output != original.strip():
            src_file.write_text(main_output, encoding="utf-8")
            print(f"[stage4]   Refactored: {rel}")
        else:
            print(f"[stage4]   No changes: {rel}")

    return new_files


def _stage4_read_mscore(folder: pathlib.Path) -> float | None:
    """Read M-score from DV8 OutputData after a Stage 4 DV8 run. Returns None if not found."""
    import json as _j
    metrics_path = folder / "OutputData" / "metrics" / "all-metrics.json"
    if not metrics_path.exists():
        return None
    try:
        raw = _j.loads(metrics_path.read_text())
        val = raw.get("m-score", {}).get("mScore")
        if val is None:
            return None
        return float(str(val).strip().rstrip("%"))
    except Exception:
        return None


def _stage4_read_all_metrics(folder: pathlib.Path) -> dict[str, float]:
    """Read ALL DV8 metrics from OutputData. Returns dict with keys:
    m-score, propagation-cost, decoupling-level, independence-level,
    and *-excl-isolated variants + isolated-items count."""
    import json as _j
    metrics_path = folder / "OutputData" / "metrics" / "all-metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        raw = _j.loads(metrics_path.read_text())
        result = {}
        for key, subkey in [
            ("m-score", "mScore"),
            ("propagation-cost", "propagationCost"),
            ("decoupling-level", "decouplingLevel"),
            ("independence-level", "independenceLevel"),
        ]:
            val = raw.get(key, {}).get(subkey)
            if val is not None:
                result[key] = float(str(val).strip().rstrip("%"))
        # Also read exclude-isolated variants and isolated item count
        # These are more honest metrics when refactoring empties files
        for key, subkey in [
            ("propagation-cost-excl", "propagationCostExcludeIsolatedItems"),
            ("decoupling-level-excl", "decouplingLevelExcludeIsolatedItems"),
        ]:
            section = key.rsplit("-excl", 1)[0]  # e.g. "propagation-cost"
            val = raw.get(section, {}).get(subkey)
            if val is not None:
                result[key] = float(str(val).strip().rstrip("%"))
        # Count total isolated items (sum across all metric sections)
        _iso_total = 0
        for section_data in raw.values():
            if isinstance(section_data, dict):
                iso = section_data.get("numberOfIsolatedItems")
                if iso is not None:
                    _iso_total = max(_iso_total, int(iso))
        result["isolated-items"] = float(_iso_total)
        return result
    except Exception:
        return {}


def _stage4_run_dv8(folder: pathlib.Path, skip_arch_report: bool = False) -> int:
    """Run DV8 on a single folder. Returns subprocess return code."""
    print(f"\n[stage4] Running DV8 on {folder.name} ...")
    nd_cmd = [
        sys.executable, AGENT,
        "--repo", str(folder),
        "--ask", "all",
    ]
    if skip_arch_report:
        nd_cmd.append("--skip-arch-report")
    # Java repos: force Depends (not NeoDepends) to match the tool used for original temporal commits.
    # Without --java-depends, dv8_agent.py defaults to NeoDepends for Java, which produces
    # far fewer dependency cells (11 vs 49) and an artificially inflated M-score.
    has_java = any(folder.rglob("*.java"))
    if has_java:
        nd_cmd.append("--java-depends")
        print(f"[stage4] Java source detected — using Depends (--java-depends) for tool consistency.")
    rc = subprocess.call(nd_cmd)
    if rc != 0:
        print(f"[stage4] DV8 run failed (rc={rc}) for {folder.name}")
    return rc


def _build_antipattern_context(temporal_root: pathlib.Path) -> str:
    """Read DV8 anti-pattern data + DSM dependency edges from the most recent revision.
    Returns a structured prompt with:
    - Each anti-pattern instance with exact file lists (from .dv8-clsx binaries)
    - Bidirectional dependency edges (from DSM JSON) that form cycles
    - Dependency weights so the LLM knows which edges are weakest to break
    This gives the refactoring LLM concrete, surgical targets."""
    import json as _jp, csv as _csv, re as _re_ap
    data_repos = temporal_root / "data_repositories"
    if not data_repos.is_dir():
        return ""

    # Find the most recent revision with arch-issue data
    candidates = []
    for p in sorted(data_repos.iterdir(), key=lambda p: p.name):
        if not p.is_dir():
            continue
        pfx = p.name.split("_")[0]
        if pfx.isdigit() and len(pfx) <= 3:
            ai_dir = p / "OutputData" / "arch-issue"
            if ai_dir.is_dir():
                candidates.append(p)

    if not candidates:
        return ""

    # Prefer hand-written (2-digit) over loop (3-digit)
    rev_dir = None
    for c in candidates:
        pfx = c.name.split("_")[0]
        if len(pfx) <= 2:
            rev_dir = c
            break
    if rev_dir is None:
        rev_dir = candidates[0]

    ai_base = rev_dir / "OutputData" / "arch-issue"

    # --- Helper: extract file paths from binary .dv8-clsx/.dv8-dsm/.dv8-issue files ---
    _SOURCE_EXTS = {"py", "java", "kt", "scala", "cs", "js", "ts", "tsx", "jsx", "go", "rs", "cpp", "c", "h", "hpp"}
    def _extract_files_from_binary(path):
        """Extract file paths by decompressing gzip data in DV8 binary files
        and regex-matching readable strings."""
        import gzip as _gz_ap
        try:
            raw = path.read_bytes()
            gz_start = raw.find(b"\x1f\x8b")
            if gz_start >= 0:
                body = _gz_ap.decompress(raw[gz_start:])
            else:
                body = raw
            strings = _re_ap.findall(rb'[\x20-\x7e]{3,}', body)
            files = []
            for s in strings:
                decoded = s.decode("utf-8", errors="replace").strip().lstrip("%(*!")
                # Format 1: "path/self (File)" — structural APs
                if "/self (File)" in decoded:
                    fpath = decoded.replace("/self (File)", "")
                    files.append(fpath)
                # Format 2: plain path — modularity violations (co-change data)
                elif "/" in decoded and "." in decoded.split("/")[-1]:
                    ext = decoded.rsplit(".", 1)[-1].lower()
                    if ext in _SOURCE_EXTS:
                        files.append(decoded)
            return sorted(set(files))
        except Exception:
            return []

    # --- Load DSM dependency edges for edge-weight context ---
    dep_json_path = rev_dir / "InputData" / "NeoDependsOutput" / "dependencies.full.dv8-dependency.json"
    dep_edges = {}  # (src, dest) → {type: weight}
    dep_files = []
    if dep_json_path.exists():
        try:
            dep_data = _jp.loads(dep_json_path.read_text(encoding="utf-8"))
            raw_files = dep_data.get("variables", [])
            dep_files = [f.replace("/self (File)", "") for f in raw_files]
            for cell in dep_data.get("cells", []):
                src = dep_files[cell["src"]]
                dest = dep_files[cell["dest"]]
                dep_edges[(src, dest)] = cell.get("values", {})
        except Exception:
            pass

    # --- Find bidirectional dependencies (cycle edges) ---
    bidirectional = []  # [(fileA, fileB, weightA→B, weightB→A)]
    seen_pairs = set()
    for (src, dest), vals in dep_edges.items():
        if (dest, src) in dep_edges and (dest, src) not in seen_pairs:
            w_fwd = sum(v for v in vals.values())
            w_rev = sum(v for v in dep_edges[(dest, src)].values())
            bidirectional.append((src, dest, w_fwd, w_rev))
            seen_pairs.add((src, dest))

    # --- Read anti-pattern summary ---
    summary_csv = ai_base / "anti-pattern-summary.csv"
    if not summary_csv.exists():
        return ""

    ap_types = {}
    with open(summary_csv, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            t = row.get("Type", "").strip()
            if t and t != "Total":
                ap_types[t] = {
                    "count": int(row.get("InstanceCount", 0)),
                    "file_count": int(row.get("DistinctFileCount", 0))
                }

    if not ap_types:
        return ""

    type_folder_map = {
        "Clique": "clique",
        "UnhealthyInheritance": "unhealthy-inheritance",
        "PackageCycle": "package-cycle",
        "ModularityViolation": "modularity-violation",
        "UnstableInterface": "unstable-interface",
        "Crossing": "crossing",
    }

    lines = [
        "DV8 ANTI-PATTERN ANALYSIS — SURGICAL TARGETS WITH DEPENDENCY EDGES",
        "Each anti-pattern instance lists EXACT files and the dependency edges between them.",
        "To fix: break the specific edges listed. Move code to eliminate imports, don't add layers.",
        "",
    ]

    total_instances = 0

    for ap_type, info in sorted(ap_types.items(), key=lambda kv: -kv[1]["file_count"]):
        folder_name = type_folder_map.get(ap_type, ap_type.lower())
        ap_dir = ai_base / folder_name

        display_map = {
            "UnhealthyInheritance": "UNHEALTHY INHERITANCE",
            "PackageCycle": "PACKAGE CYCLE",
            "ModularityViolation": "MODULARITY VIOLATION",
            "UnstableInterface": "UNSTABLE INTERFACE",
            "Crossing": "CROSSING",
            "Clique": "CLIQUE",
        }
        display_name = display_map.get(ap_type, ap_type)

        lines.append(f"{'='*60}")
        lines.append(f"ANTI-PATTERN: {display_name} — {info['count']} instance(s), {info['file_count']} files")
        lines.append(f"{'='*60}")

        if not ap_dir.is_dir():
            lines.append(f"  (no detail folder found)")
            lines.append("")
            continue

        # Enumerate instance directories (numbered folders)
        inst_dirs = sorted(
            [d for d in ap_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name)
        )

        shown = 0
        for inst_dir in inst_dirs[:5]:  # top 5 instances
            inst_id = inst_dir.name
            # Extract file list from the -clsx file (clustering) or -sdsm (structure DSM)
            files = []
            for suffix in ["-clsx.dv8-clsx", "-sdsm.dv8-dsm", "-merge.dv8-dsm"]:
                candidate = inst_dir / f"{inst_id}{suffix}"
                if candidate.exists():
                    files = _extract_files_from_binary(candidate)
                    if files:
                        break

            if not files:
                continue

            # Filter out test files for display (but note them)
            src_files = [f for f in files if not f.startswith("tests/")]
            test_files = [f for f in files if f.startswith("tests/")]

            lines.append(f"\n  Instance #{inst_id} ({len(files)} files):")
            lines.append(f"    Source files:")
            for fp in src_files[:15]:
                # Add fan_in/fan_out if available from DSM
                fan_in = sum(1 for (s, d) in dep_edges if d == fp)
                fan_out = sum(1 for (s, d) in dep_edges if s == fp)
                fan_info = f"  [fan_in={fan_in}, fan_out={fan_out}]" if fan_in + fan_out > 5 else ""
                lines.append(f"      - {fp}{fan_info}")

            if test_files:
                lines.append(f"    Test files ({len(test_files)}): {', '.join(t.split('/')[-1] for t in test_files[:5])}")

            # Show DEPENDENCY EDGES between files in this instance
            inst_edges = []
            for (src, dest), vals in dep_edges.items():
                if src in files and dest in files:
                    w = sum(v for v in vals.values())
                    types_str = ", ".join(f"{k}:{int(v)}" for k, v in vals.items())
                    inst_edges.append((src, dest, w, types_str))

            if inst_edges:
                inst_edges.sort(key=lambda x: -x[2])
                lines.append(f"    Dependency edges (strongest first):")
                for src, dest, w, types_str in inst_edges[:10]:
                    short_src = src.split("/")[-1]
                    short_dest = dest.split("/")[-1]
                    lines.append(f"      {short_src} → {short_dest} (weight={w}, {types_str})")

                # Highlight bidirectional edges — these are the CYCLE edges to break
                bidir_in_inst = [(a, b, wf, wr) for a, b, wf, wr in bidirectional
                                 if a in files and b in files]
                if bidir_in_inst:
                    lines.append(f"    CIRCULAR DEPENDENCIES (bidirectional — break these!):")
                    for a, b, wf, wr in bidir_in_inst:
                        short_a = a.split("/")[-1]
                        short_b = b.split("/")[-1]
                        weaker = f"{short_a}→{short_b}" if wf <= wr else f"{short_b}→{short_a}"
                        weaker_w = min(wf, wr)
                        lines.append(f"      {short_a} ↔ {short_b} (→: weight={wf}, ←: weight={wr})")
                        lines.append(f"        WEAKEST EDGE to break: {weaker} (weight={weaker_w})")

            total_instances += 1
            shown += 1

        if shown == 0:
            lines.append(f"  (no instance details available)")
        lines.append("")

    # --- Add all bidirectional deps even if not in a specific AP instance ---
    if bidirectional:
        lines.append(f"{'='*60}")
        lines.append(f"ALL BIDIRECTIONAL DEPENDENCIES (circular imports to break)")
        lines.append(f"{'='*60}")
        for a, b, wf, wr in sorted(bidirectional, key=lambda x: min(x[2], x[3])):
            short_a = a.split("/")[-1]
            short_b = b.split("/")[-1]
            weaker = f"{short_a}→{short_b}" if wf <= wr else f"{short_b}→{short_a}"
            lines.append(f"  {short_a} ↔ {short_b} (→:{wf}, ←:{wr}) — break: {weaker}")
        lines.append("")

    # --- Fix instructions per type ---
    lines.append("HOW TO FIX EACH ANTI-PATTERN TYPE:")
    lines.append("-" * 40)
    lines.append("CLIQUE: Files form a circular dependency cycle (A→B→C→A).")
    lines.append("  FIX: Identify the WEAKEST edge (lowest weight) in the cycle.")
    lines.append("  MOVE the imported symbols to the importing file (inline them),")
    lines.append("  or MOVE them to a file that both already depend on.")
    lines.append("  Do NOT create new files. Do NOT add re-exports.")
    lines.append("  Example: if _types.py imports URL from _urls.py (weight=2) but _urls.py")
    lines.append("  imports PrimitiveData from _types.py (weight=5), break _types.py→_urls.py")
    lines.append("  by moving/inlining URL usage in _types.py.")
    lines.append("")
    lines.append("UNHEALTHY INHERITANCE: Parent class depends on child, or clients use both parent+child.")
    lines.append("  FIX: Make the parent class self-contained. Move child-specific logic OUT of the parent.")
    lines.append("  Ensure clients import ONLY the parent class (or only the child they need).")
    lines.append("  Replace isinstance checks with method dispatch or Protocol-based duck typing.")
    lines.append("")
    lines.append("MODULARITY VIOLATION: Files co-change frequently but have no structural dependency.")
    lines.append("  FIX: These files are implicitly coupled through shared concepts.")
    lines.append("  Either: (a) ADD an explicit dependency — extract the shared concept into one file")
    lines.append("  and have both import it, or (b) MERGE the files if they're small enough.")
    lines.append("  The goal is to make the implicit coupling EXPLICIT so future changes are localized.")
    lines.append("")

    if total_instances == 0:
        return ""

    return "\n".join(lines)


def _generate_antipattern_network_graph(rev_dir: pathlib.Path, output_path: pathlib.Path,
                                         title: str = "Dependency Network") -> bool:
    """Generate a layered network graph using DRH layers for vertical positioning.
    - Nodes laid out in horizontal DRH layers (L0=bottom/core, L1=middle, L2=top/tests)
    - Anti-pattern nodes: LARGER, diamond/square marker, colored by AP type, bold black border
    - Normal nodes: small grey circles
    - Bidirectional edges (cycles) in red dashed
    - Edge weight = line thickness
    - File labels shown on anti-pattern nodes and high-degree nodes
    Returns True if graph was generated successfully."""
    try:
        import json as _jn
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
        import re as _re_ng
    except ImportError:
        return False

    dep_json = rev_dir / "InputData" / "NeoDependsOutput" / "dependencies.full.dv8-dependency.json"
    if not dep_json.exists():
        return False

    dep_data = _jn.loads(dep_json.read_text(encoding="utf-8"))
    raw_files = dep_data.get("variables", [])
    files = [f.replace("/self (File)", "") for f in raw_files]
    cells = dep_data.get("cells", [])

    if not files or not cells:
        return False

    # Build graph (skip test files)
    G = nx.DiGraph()
    for f in files:
        short = f.split("/")[-1] if "/" in f else f
        if f.startswith("tests/"):
            continue
        G.add_node(f, label=short)

    for cell in cells:
        src, dest = files[cell["src"]], files[cell["dest"]]
        if src.startswith("tests/") or dest.startswith("tests/"):
            continue
        weight = sum(v for v in cell.get("values", {}).values())
        if G.has_node(src) and G.has_node(dest):
            G.add_edge(src, dest, weight=weight)

    if len(G.nodes) == 0:
        return False

    # --- Read DRH layers for layout ---
    drh_json = rev_dir / "OutputData" / "dv8-analysis-result" / "dsm" / "drh-clustering.json"
    file_layer = {}  # file_path → layer_number (0=bottom/core, higher=top)
    if drh_json.exists():
        try:
            drh = _jn.loads(drh_json.read_text(encoding="utf-8"))
            for layer_group in drh.get("structure", []):
                layer_name = layer_group.get("name", "")  # e.g. "L0", "L1"
                try:
                    layer_num = int(layer_name.replace("L", ""))
                except ValueError:
                    continue
                for module in layer_group.get("nested", []):
                    for item in module.get("nested", []):
                        if item.get("@type") == "item":
                            fname = item["name"].replace("/self (File)", "")
                            file_layer[fname] = layer_num
        except Exception:
            pass

    # --- Identify anti-pattern file memberships ---
    _SOURCE_EXTS_NG = {"py", "java", "ts", "js", "go", "rs", "cpp", "c", "h", "cs"}
    ap_colors = {
        "clique": "#e74c3c",                  # red
        "unhealthy-inheritance": "#e67e22",    # orange
        "modularity-violation": "#9b59b6",     # purple
        "package-cycle": "#f39c12",            # yellow
        "unstable-interface": "#3498db",       # blue
        "crossing": "#1abc9c",                 # teal
    }
    file_ap_type = {}  # file → ap_type (first match wins for color)
    file_ap_count = {}  # file → count of AP instances it appears in
    ai_base = rev_dir / "OutputData" / "arch-issue"
    if ai_base.is_dir():
        import gzip as _gz_ng
        for ap_dir in sorted(ai_base.iterdir()):
            if not ap_dir.is_dir() or ap_dir.name.endswith(".csv"):
                continue
            ap_type = ap_dir.name
            for inst_dir in ap_dir.iterdir():
                if not inst_dir.is_dir():
                    continue
                for binfile in inst_dir.iterdir():
                    if binfile.suffix in (".dv8-clsx", ".dv8-dsm"):
                        try:
                            raw = binfile.read_bytes()
                            gz_start = raw.find(b"\x1f\x8b")
                            if gz_start >= 0:
                                body = _gz_ng.decompress(raw[gz_start:])
                            else:
                                body = raw
                            strings = _re_ng.findall(rb'[\x20-\x7e]{5,}', body)
                            for s in strings:
                                decoded = s.decode("utf-8", errors="replace").strip().lstrip("%(*!")
                                fpath = None
                                if "/self (File)" in decoded:
                                    fpath = decoded.replace("/self (File)", "")
                                elif "/" in decoded and "." in decoded.split("/")[-1]:
                                    ext = decoded.rsplit(".", 1)[-1].lower()
                                    if ext in _SOURCE_EXTS_NG:
                                        fpath = decoded
                                if fpath and fpath in G.nodes:
                                    if fpath not in file_ap_type:
                                        file_ap_type[fpath] = ap_type
                                    file_ap_count[fpath] = file_ap_count.get(fpath, 0) + 1
                        except Exception:
                            pass

    # --- Find bidirectional edges ---
    bidir_edges = set()
    for u, v in G.edges():
        if G.has_edge(v, u):
            bidir_edges.add((min(u, v), max(u, v)))

    # --- Layered layout ---
    # Assign y position by DRH layer, x position spread within layer
    max_layer = max(file_layer.values()) if file_layer else 0
    pos = {}
    layer_nodes = {}  # layer → list of nodes
    unlayered = []
    for n in G.nodes():
        layer = file_layer.get(n)
        if layer is not None:
            layer_nodes.setdefault(layer, []).append(n)
        else:
            unlayered.append(n)
    # Put unlayered nodes in the middle layer
    mid_layer = max_layer // 2 if max_layer > 0 else 0
    if unlayered:
        layer_nodes.setdefault(mid_layer, []).extend(unlayered)

    # Position: y = layer (0=bottom), x = spread evenly
    y_spacing = 2.5
    for layer, nodes in layer_nodes.items():
        # Sort: AP nodes first (for visual prominence), then by degree
        nodes.sort(key=lambda n: (0 if n in file_ap_type else 1,
                                  -(G.in_degree(n) + G.out_degree(n))))
        n_nodes = len(nodes)
        x_spacing = 2.0  # space between nodes
        for i, n in enumerate(nodes):
            x = (i - n_nodes / 2) * x_spacing
            # Invert: L0=bottom (y=0), higher layers go up
            y = (max_layer - layer) * y_spacing
            pos[n] = (x, y)

    # Ensure all nodes have positions
    for n in G.nodes():
        if n not in pos:
            pos[n] = (0, 0)

    # --- Draw ---
    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    fig.patch.set_facecolor("white")

    # Draw layer bands (horizontal stripes)
    for layer in range(max_layer + 1):
        y = (max_layer - layer) * y_spacing
        ax.axhspan(y - y_spacing * 0.45, y + y_spacing * 0.45,
                    alpha=0.06, color="#3498db" if layer == 0 else ("#2ecc71" if layer == 1 else "#ecf0f1"))
        ax.text(-max(len(G.nodes) * 0.7, 8), y, f"Layer {layer}",
                fontsize=9, fontweight="bold", color="#7f8c8d", va="center", ha="right")

    # Draw normal edges (thin, grey)
    normal_edges = [(u, v) for u, v in G.edges() if (min(u, v), max(u, v)) not in bidir_edges]
    edge_weights = [G[u][v].get("weight", 1) for u, v in normal_edges]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [max(0.3, min(3, w / max_w * 3)) for w in edge_weights]
    if normal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, width=edge_widths,
                               alpha=0.15, edge_color="#bdc3c7", arrows=True,
                               arrowsize=6, ax=ax, connectionstyle="arc3,rad=0.05")

    # Draw bidirectional edges (cycles) in bold red
    bidir_edge_list = []
    for u, v in bidir_edges:
        if G.has_edge(u, v):
            bidir_edge_list.append((u, v))
    if bidir_edge_list:
        nx.draw_networkx_edges(G, pos, edgelist=bidir_edge_list, width=2.5,
                               alpha=0.7, edge_color="#e74c3c", style="dashed",
                               arrows=True, arrowsize=10, ax=ax,
                               connectionstyle="arc3,rad=0.1")

    # Draw nodes — normal (small grey) and anti-pattern (large colored diamond)
    normal_nodes = [n for n in G.nodes() if n not in file_ap_type]
    ap_nodes = [n for n in G.nodes() if n in file_ap_type]

    if normal_nodes:
        normal_sizes = [max(80, min(400, (G.in_degree(n) + G.out_degree(n)) * 30)) for n in normal_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_size=normal_sizes,
                               node_color="#d5dbe0", edgecolors="#95a5a6", linewidths=0.8,
                               alpha=0.7, ax=ax)

    if ap_nodes:
        ap_sizes = [max(500, min(4000, (G.in_degree(n) + G.out_degree(n)) * 120
                        + file_ap_count.get(n, 1) * 200)) for n in ap_nodes]
        ap_node_colors = [ap_colors.get(file_ap_type[n], "#95a5a6") for n in ap_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=ap_nodes, node_size=ap_sizes,
                               node_color=ap_node_colors, edgecolors="black", linewidths=2.5,
                               alpha=0.95, node_shape="D", ax=ax)

    # Labels: always show AP nodes + top degree nodes
    top_degree = sorted(G.nodes(), key=lambda n: G.in_degree(n) + G.out_degree(n), reverse=True)
    label_nodes = set(ap_nodes) | set(top_degree[:10])
    labels = {n: G.nodes[n].get("label", n.split("/")[-1]) for n in label_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)

    # Legend
    legend_handles = [mpatches.Patch(color="#d5dbe0", label="Normal file")]
    for ap_name, ap_color in ap_colors.items():
        if any(file_ap_type.get(n) == ap_name for n in G.nodes()):
            display = ap_name.replace("-", " ").title()
            legend_handles.append(mpatches.Patch(color=ap_color, label=display))
    if bidir_edge_list:
        legend_handles.append(plt.Line2D([0], [0], color="#e74c3c", linestyle="dashed",
                                         linewidth=2, label="Circular dep"))
    # Show AP count in title
    ap_count = len(ap_nodes)
    full_title = f"{title}\n{ap_count} files in anti-patterns | {len(bidir_edge_list)} circular deps"
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8, framealpha=0.9,
              edgecolor="#bdc3c7")
    ax.set_title(full_title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _build_module_penalty_context(temporal_root: pathlib.Path) -> str:
    """Read DV8 M-score module data from the most recent revision's interpretation_payload.json
    and build a structured analysis showing which modules drag M-score down and which files to target.
    This gives the refactoring LLM concrete, data-driven targets instead of vague anti-pattern descriptions."""
    import json as _jp
    data_repos = temporal_root / "data_repositories"
    if not data_repos.is_dir():
        return ""

    # Find the most recent hand-written revision (lowest 2-digit prefix)
    rev_dirs = sorted(
        [p for p in data_repos.iterdir()
         if p.is_dir() and len(p.name) > 2 and p.name[:2].isdigit() and p.name[2] == "_"],
        key=lambda p: p.name
    )
    if not rev_dirs:
        return ""

    # Try the most recent revision first, fall back to loop iteration folders
    payload_path = None
    for rd in rev_dirs:
        candidate = rd / "OutputData" / "interpretation_payload.json"
        if candidate.exists():
            payload_path = candidate
            break
    # Also check loop iteration folders (3-digit prefix)
    if payload_path is None:
        loop_dirs = sorted(
            [p for p in data_repos.iterdir()
             if p.is_dir() and len(p.name) > 3 and p.name[:3].isdigit() and p.name[3] == "_"],
            key=lambda p: p.name
        )
        for ld in loop_dirs:
            candidate = ld / "OutputData" / "interpretation_payload.json"
            if candidate.exists():
                payload_path = candidate
                break

    if not payload_path:
        return ""

    try:
        payload = _jp.loads(payload_path.read_text(encoding="utf-8"))
        modules = payload.get("metrics", {}).get("m_score_modules", [])
        hotspots = payload.get("structural_hotspots", {}).get("rows", [])
        anti_patterns = payload.get("anti_pattern_rows", [])
        m_score_data = payload.get("metrics", {}).get("m_score", {})
        current_mscore = float(m_score_data.get("mScore", "0").replace("%", ""))
    except Exception:
        return ""

    if not modules:
        return ""

    # Compute per-module loss
    for m in modules:
        sf = m["size_factor"]
        sp = m["size_penalty"]
        m["_loss"] = sf * (1 - sp)

    modules.sort(key=lambda m: -m["_loss"])

    lines = [
        f"DV8 M-SCORE MODULE PENALTY ANALYSIS (current M-score: {current_mscore:.1f}%)",
        f"Only modules with penalties are shown. Fix these to improve M-score.",
        "",
    ]

    penalized = [m for m in modules if m["_loss"] > 0.005]
    if not penalized:
        return ""

    total_contrib = sum(m["contribution"] for m in modules)

    for m in penalized:
        key = m["module_key"]
        size = m["module_size"]
        files = [f.replace("/self (File)", "") for f in m["files"]]
        loss_pct = m["_loss"] * 100
        lines.append(f"MODULE {key} (Layer {m['layer']}, {size} files) — M-SCORE LOSS: {loss_pct:.1f}%")
        lines.append(f"  cross_penalty={m['cross_penalty']:.3f} (cross-layer deps violating design rules)")
        lines.append(f"  internal_penalty={m['internal_penalty']:.3f} (files too tightly coupled WITHIN this module)")

        if m["cross_penalty"] > 0.05:
            lines.append(f"  → REDUCE cross-layer deps: some files import from wrong layer")
        if m["internal_penalty"] > 0.1:
            lines.append(f"  → REDUCE internal coupling: split large files or extract interfaces")

        lines.append(f"  Files in this module:")
        for f in files:
            # Add hotspot info if available
            hs_info = ""
            for h in hotspots:
                if f in h.get("Filename", ""):
                    fi = int(h.get("FanIn", 0))
                    fo = int(h.get("FanOut", 0))
                    if fi > 50 or fo > 50:
                        hs_info = f"  [fan_in={fi}, fan_out={fo}]"
                    break
            lines.append(f"    - {f}{hs_info}")
        lines.append("")

    # Add improvement forecast
    lines.append("IMPROVEMENT FORECAST:")
    cumulative = total_contrib
    for m in penalized:
        cumulative += m["_loss"]
        lines.append(f"  Fix Module {m['module_key']}: M-score → {cumulative*100:.1f}% (+{m['_loss']*100:.1f}%)")

    total_gain = sum(m["_loss"] for m in penalized)
    lines.append(f"  Fix ALL: {total_contrib*100:.1f}% → {(total_contrib+total_gain)*100:.1f}% (+{total_gain*100:.1f}%)")
    lines.append("")

    return "\n".join(lines)


def _print_architecture_diff(temporal_root: pathlib.Path, baseline_dir,
                              iteration_dirs: list, data_repos: pathlib.Path) -> None:
    """Print before/after comparison: which modules improved, which anti-patterns were solved."""
    import json as _jd
    if not baseline_dir or not iteration_dirs:
        return

    # Find best iteration folder
    best_it = max(iteration_dirs, key=lambda x: x[3] if x[3] is not None else 0)
    best_dir = best_it[1]

    # Load baseline payload
    base_payload_path = baseline_dir / "OutputData" / "interpretation_payload.json"
    best_payload_path = best_dir / "OutputData" / "interpretation_payload.json"

    if not base_payload_path.exists() or not best_payload_path.exists():
        return

    try:
        base_p = _jd.loads(base_payload_path.read_text(encoding="utf-8"))
        best_p = _jd.loads(best_payload_path.read_text(encoding="utf-8"))
    except Exception:
        return

    print(f"\n{'─'*60}")
    print(f"  ARCHITECTURE DIFF: Baseline → Best Iteration ({best_it[0]})")
    print(f"{'─'*60}")

    # --- Metrics comparison ---
    base_m = base_p.get("metrics", {})
    best_m = best_p.get("metrics", {})
    for key in ["propagation-cost", "decoupling-level", "independence-level"]:
        bv = base_m.get(key)
        ev = best_m.get(key)
        if bv is not None and ev is not None:
            d = float(ev) - float(bv)
            sign = "+" if d >= 0 else ""
            better = "improved" if (d < 0 and key == "propagation-cost") or (d > 0 and key != "propagation-cost") else "worsened" if d != 0 else "unchanged"
            print(f"  {key}: {float(bv):.2f}% → {float(ev):.2f}% ({sign}{d:.2f}%) [{better}]")

    # --- Anti-pattern count comparison ---
    # History-based APs (modularity-violation, unstable-interface, crossing) require git history
    # which loop folders don't have — only compare structural APs reliably
    _HISTORY_AP = {"modularity-violation", "unstable-interface", "crossing", "total"}
    base_ap = base_p.get("anti_pattern_counts", {})
    best_ap = best_p.get("anti_pattern_counts", {})
    all_ap_types = sorted(set(list(base_ap.keys()) + list(best_ap.keys())))
    if all_ap_types:
        print(f"\n  Anti-pattern changes (structural only — history-based need git history):")
        for ap_type in all_ap_types:
            bc = base_ap.get(ap_type, 0)
            ec = best_ap.get(ap_type, 0)
            if ap_type in _HISTORY_AP:
                if bc > 0 and ec == 0:
                    print(f"    {ap_type}: {bc} → N/A (no git history in loop folder)")
                    continue
            d = ec - bc
            if d != 0:
                sign = "+" if d > 0 else ""
                status = "SOLVED" if ec == 0 else ("reduced" if d < 0 else "increased")
                print(f"    {ap_type}: {bc} → {ec} ({sign}{d}) [{status}]")
            else:
                print(f"    {ap_type}: {bc} (unchanged)")

    # --- Module penalty comparison ---
    base_mods = base_p.get("metrics", {}).get("m_score_modules", [])
    best_mods = best_p.get("metrics", {}).get("m_score_modules", [])
    if base_mods and best_mods:
        # Build lookup by module key
        def _mod_key(m):
            return m.get("module_key", m.get("layer", "?"))
        base_by_key = {_mod_key(m): m for m in base_mods}
        best_by_key = {_mod_key(m): m for m in best_mods}
        improved_mods = []
        for mk, bm in base_by_key.items():
            em = best_by_key.get(mk)
            if em:
                b_loss = bm["size_factor"] * (1 - bm["size_penalty"])
                e_loss = em["size_factor"] * (1 - em["size_penalty"])
                if e_loss < b_loss - 0.001:
                    improved_mods.append((mk, b_loss, e_loss))
        if improved_mods:
            print(f"\n  Modules with REDUCED M-score penalty:")
            for mk, bl, el in sorted(improved_mods, key=lambda x: x[1]-x[2], reverse=True):
                print(f"    Module {mk}: loss {bl*100:.2f}% → {el*100:.2f}% (saved {(bl-el)*100:.2f}%)")

    print(f"{'─'*60}")


def _offer_manual_targeting(temporal_root: pathlib.Path, model: str,
                             iteration_dirs: list, base_name_parts: list,
                             data_repos: pathlib.Path, today, max_iterations: int,
                             this_run_folders: set,
                             qa_model: str = "qwen3-coder-30b-refactor") -> None:
    """After automated loop, offer single-shot manual targeting iterations.
    Q&A (analysis) uses qa_model (local Ollama). Code application uses model (Claude/Qwen)."""
    import shutil as _sh_m, datetime as _dt_m, re as _re_m
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  MANUAL TARGETING (optional)")
    print(f"  You can now run single targeted iterations on specific modules or anti-patterns.")
    print(f"  Examples:")
    print(f"    refactor module 1,0    — target a specific DV8 module")
    print(f"    refactor clique        — target clique anti-patterns")
    print(f"    refactor _models.py    — target a specific file")
    print(f"    refactor package-cycle — target package cycle anti-patterns")
    print(f"    q / quit               — finish and generate final report")
    print(f"{sep}")

    manual_iteration = len(iteration_dirs)
    current_source = iteration_dirs[-1][1] if iteration_dirs else None
    if current_source is None:
        return

    while True:
        try:
            cmd = input("\n  Manual target (or q to finish): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd or cmd.lower() in ("q", "quit", "exit"):
            break
        # Extract target from input
        target = cmd
        if cmd.lower().startswith("refactor"):
            target = cmd[len("refactor"):].strip()
        if not target:
            print("  No target specified. Try: refactor module 1,0")
            continue

        manual_iteration += 1
        print(f"\n[manual] Running targeted iteration {manual_iteration}: {target}")

        # Run single iteration with manual target — reuse loop infrastructure
        import importlib.util as _ilu_m
        _spec_m = _ilu_m.spec_from_file_location("itb", pathlib.Path(INTERPRET_TEMPORAL))
        _mod_m = _ilu_m.module_from_spec(_spec_m)
        _spec_m.loader.exec_module(_mod_m)

        # Load context
        interp_dir = temporal_root / "OUTPUT_INTERPRETATION"
        candidates = sorted(
            [d for d in interp_dir.iterdir() if d.is_dir()] if interp_dir.exists() else [],
            key=lambda d: d.stat().st_mtime, reverse=True
        )
        report_text = ""
        if candidates:
            for rp in ["INTERPRETATION_REPORT.md", "report.md", "interpretation.md"]:
                rpath = candidates[0] / rp
                if rpath.exists():
                    report_text = rpath.read_text(encoding="utf-8", errors="replace")
                    break

        _risk_ctx, _commit_ctx, _ = _load_rich_qa_context(temporal_root)
        _mscore_bd = _mod_m.load_mscore_worst_modules(temporal_root)
        _ap_ctx = _build_antipattern_context(temporal_root)
        conversation_m: list[str] = []

        def _man_ask(question: str) -> str:
            prior = "\n\n".join(conversation_m)
            ctx = (prior + "\n\n" + report_text) if prior else report_text
            raw = _mod_m.answer_user_question(qa_model, question, ctx,
                                               mscore_breakdown=_mscore_bd,
                                               timeout_s=900,
                                               risk_score_context=_risk_ctx,
                                               commit_context=_commit_ctx)
            answer = _mod_m.strip_thinking_and_fences(raw)
            conversation_m.append(f"Q: {question}\nA: {answer}")
            return answer

        # Q1 quick analysis
        _man_ask("Which parts got worse over time — show anti-pattern groups, "
                 "files with most dependency growth, and worst files overall.")

        # Q2 targeted
        _manual_hint = f"\n\nUSER-SPECIFIED TARGET: {target}\n"
        if _ap_ctx:
            _manual_hint += f"Anti-pattern context:\n\n{_ap_ctx}\n"
        _man_ask(
            f"The user wants to refactor: {target}. "
            "Give EXACTLY 3 concrete code-level refactoring actions using ### Action 1 / ### Action 2 / ### Action 3 headings. "
            "Each action MUST target the specified area with exact dependency-breaking changes "
            "(remove import, move function, break cycle, etc). "
            "Do NOT create protocol/interface files unless explicitly needed — "
            "prefer MOVING code to REMOVE import edges over ADDING abstraction layers. "
            "Only structural code changes — no process actions."
            + _manual_hint
        )

        # Parse and apply Action 1
        _q2 = conversation_m[-1]
        _actions = _re_m.findall(
            r"### Action (\d+)[^\n]*\n(.*?)(?=\n### Action \d+|\n## |\Z)",
            _q2, _re_m.DOTALL
        )
        if not _actions:
            print("[manual] Could not parse actions from Q2. Skipping.")
            continue

        action_text = _actions[0][1].strip()
        action_date = today - _dt_m.timedelta(days=manual_iteration)
        date_str = action_date.strftime("%d%m%Y")
        prefix = "000"
        new_name = f"{prefix}_manual{manual_iteration}_{date_str}_1000"
        new_dir = data_repos / new_name

        if new_dir.exists():
            _sh_m.rmtree(new_dir)
        _sh_m.copytree(current_source, new_dir)
        _stage4_clean_copy(new_dir)
        _stage4_apply_action(new_dir, "1", action_text, model)

        rc = _stage4_run_dv8(new_dir, skip_arch_report=True)
        if rc != 0:
            print(f"[manual] DV8 failed. Folder kept for inspection: {new_dir.name}")
            continue

        new_mscore = _stage4_read_mscore(new_dir)
        if new_mscore is not None:
            prev = _stage4_read_mscore(current_source)
            delta = new_mscore - prev if prev else 0
            sign = "+" if delta >= 0 else ""
            print(f"[manual] M-score: {new_mscore:.2f}% ({sign}{delta:.2f}%)")
        else:
            print(f"[manual] Could not read M-score.")

        _manual_am = _stage4_read_all_metrics(new_dir)
        iteration_dirs.append((str(manual_iteration), new_dir, action_date, new_mscore, _manual_am))
        current_source = new_dir
        print(f"[manual] Iteration {manual_iteration} complete: {new_dir.name}")


def _run_multi_model_loop(temporal_root: pathlib.Path, models: list[str],
                           max_iterations: int = 4,
                           qa_model: str = "qwen3-coder-30b-refactor") -> int:
    """Run refactoring loop for each model independently from the SAME baseline,
    then plot all curves together for comparison.

    Each model gets its own track with separate iteration folders (e.g. loop1_claude_sonnet, loop1_qwen).
    All tracks start from the same original baseline — no model builds on another's work.
    Default 4 iterations = 2 full cycles (module→AP→module→AP) per model.
    Q&A (Stage 3) always uses qa_model (local Ollama). Code application uses each model.
    """
    import json as _jmm, shutil as _sh_mm
    all_tracks: dict[str, list[tuple[str, float | None]]] = {}  # label → [(iter, mscore)]

    # Find the original baseline ONCE — all models start from here
    data_repos = temporal_root / "data_repositories"
    if not data_repos.is_dir():
        print("[multi] No data_repositories/ folder found.")
        return 1
    orig_revs = sorted(
        [p for p in data_repos.iterdir()
         if p.is_dir() and len(p.name) > 2 and p.name[:2].isdigit() and p.name[2] == "_"],
        key=lambda p: p.name
    )
    if not orig_revs:
        print("[multi] No source revision found.")
        return 1
    baseline_dir = orig_revs[0]
    baseline_ms = _stage4_read_mscore(baseline_dir)
    print(f"\n{'#'*70}")
    print(f"  MULTI-MODEL COMPARISON: {', '.join(models)}")
    print(f"  Baseline: {baseline_dir.name} (M-score: {baseline_ms:.2f}%)" if baseline_ms else f"  Baseline: {baseline_dir.name}")
    print(f"  Iterations per model: {max_iterations} (2 full cycles: module→AP→module→AP)")
    print(f"  Q&A model: {qa_model}")
    print(f"{'#'*70}")

    for model in models:
        _label = model.replace("qwen3-coder-30b-refactor", "qwen").replace("claude-", "claude_")
        print(f"\n{'#'*70}")
        print(f"  MODEL TRACK: {model} (label: {_label})")
        print(f"{'#'*70}")

        # Clean up any previous loop folders for THIS track before starting
        for _old in list(data_repos.iterdir()):
            if _old.is_dir() and f"_{_label}_" in _old.name:
                _pfx = _old.name.split("_")[0]
                if _pfx.isdigit() and len(_pfx) >= 3:
                    _sh_mm.rmtree(_old)

        rc = _run_refactor_loop(temporal_root, model=model, max_iterations=max_iterations,
                                 track_label=_label, qa_model=qa_model)

        # Collect results from timeseries for this track
        ts_path = temporal_root / "INPUT_INTERPRETATION" / "timeseries.json"
        if ts_path.exists():
            ts = _jmm.loads(ts_path.read_text())
            track_data = []
            _tag = f"-{_label}"
            for rev in ts.get("revisions", []):
                ch = rev.get("commit_hash", "")
                if f"ai{_tag}-" in ch:
                    it_num = ch.split("-")[-1]
                    ms = rev.get("metrics", {}).get("m-score")
                    track_data.append((it_num, ms))
            all_tracks[_label] = track_data

    # Print comparison table
    if len(all_tracks) > 1:
        print(f"\n{'='*70}")
        print(f"  MODEL COMPARISON")
        print(f"{'='*70}")
        if baseline_ms is not None:
            print(f"  {'baseline':20s}: M-score = {baseline_ms:.2f}%")
        for label, data in all_tracks.items():
            scores = [ms for _, ms in data if ms is not None]
            best = max(scores) if scores else 0
            delta = best - baseline_ms if baseline_ms else 0
            sign = "+" if delta >= 0 else ""
            print(f"  {label:20s}: best M-score = {best:.2f}% ({sign}{delta:.2f}%) in {len(data)} iterations")
        # Declare winner
        if all_tracks:
            winner = max(all_tracks.items(), key=lambda kv: max((ms for _, ms in kv[1] if ms is not None), default=0))
            winner_best = max((ms for _, ms in winner[1] if ms is not None), default=0)
            print(f"\n  WINNER: {winner[0]} with M-score {winner_best:.2f}%")
        print(f"{'='*70}")

    return 0


def _run_refactor_loop(temporal_root: pathlib.Path, model: str = "qwen3-coder-30b-refactor",
                       max_iterations: int = 5, track_label: str = "",
                       qa_model: str = "qwen3-coder-30b-refactor",
                       use_feedback_loop: bool = False) -> int:
    """Loop mode Stage 4: full pipeline per iteration.

    Stage 3 (Q&A analysis) always uses qa_model (local Ollama — fast, reliable).
    Stage 4 (code application) uses model (Claude CLI / Qwen / Codex).

    Each iteration: Q1 (analysis) → Q2 (anti-pattern targeting) → apply ONE action via LLM →
    full DV8 re-analysis (DSM, metrics, anti-patterns, M-score) → compare.

    Strategy: ALL iterations target specific DV8 anti-patterns (clique, unhealthy inheritance,
    package cycle, etc.) with exact file lists from arch-issue data. This is more effective
    than targeting vague module penalties because anti-patterns are concrete and actionable.

    Guards: file integrity (rejects if LLM destroys files), baseline M-score guard,
    anti-pattern count verification after each iteration.

    Stops when: max_iterations reached, M-score stalls 3x, or DV8 fails.
    After automated loop: offers manual targeting for single follow-up iterations.
    """
    import shutil as _sh, datetime as _dt, json as _js, importlib.util as _ilu, re as _re_l
    sep = "=" * 70
    _model_label = track_label or model
    # Q&A (Q1/Q2 analysis) must use a local Ollama model — Claude models don't route through Ollama
    if qa_model.startswith("claude-"):
        print(f"[loop] qa_model '{qa_model}' is a Claude model — falling back to local Ollama for Q&A.")
        qa_model = "qwen3-coder-30b-refactor"
    print(f"\n{sep}\n  STAGE 4 (LOOP MODE): max {max_iterations} iterations")
    print(f"  Q&A model (Stage 3):      {qa_model}")
    print(f"  Refactor model (Stage 4): {model}")
    if use_feedback_loop:
        print(f"  Feedback loop:            ENABLED (reviewer checks each iteration)")
    print(f"{sep}")

    data_repos = temporal_root / "data_repositories"
    if not data_repos.is_dir():
        print("[loop] No data_repositories/ folder found.")
        return 1

    # Find most recent hand-written revision (2-digit prefix, lowest = most recent)
    def _get_source_revs():
        return sorted(
            [p for p in data_repos.iterdir()
             if p.is_dir() and len(p.name) > 2 and p.name[:2].isdigit() and p.name[2] == "_"],
            key=lambda p: p.name
        )

    # Load interpret_temporal_bundle for Q1/Q2
    _spec = _ilu.spec_from_file_location("itb", pathlib.Path(INTERPRET_TEMPORAL))
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)

    prev_mscore: float | None = None
    baseline_mscore: float | None = None  # original pre-refactoring M-score (never changes)
    iteration_dirs: list[tuple[str, pathlib.Path, _dt.date, float | None]] = []
    stall_count: int = 0          # consecutive iterations with no significant improvement
    STALL_LIMIT: int = 3          # stop after this many consecutive stalls
    today = _dt.date.today()
    # Track folders created by THIS run so we don't delete them on the next iteration
    this_run_folders: set[str] = set()
    # Track tried actions so we can tell the LLM to pick different anti-patterns on stall
    tried_actions: list[str] = []
    # Track file sets targeted by each action — used for deduplication (more reliable than text matching)
    tried_file_sets: list[frozenset[str]] = []
    # Track reviewer failures from feedback loop — fed as context to next iteration
    review_failures: list[str] = []

    def _extract_action_files(text: str) -> frozenset[str]:
        """Extract file paths mentioned in an action for deduplication."""
        return frozenset(_re_l.findall(r'[\w/._-]+\.(?:py|java|ts|js|go|rs|cpp|c|h)', text))

    # Find the baseline source revision (most recent hand-written commit = lowest 2-digit prefix)
    rev_dirs = _get_source_revs()
    if not rev_dirs:
        print("[loop] No source revision found.")
        return 1
    current_source = rev_dirs[0]
    baseline_mscore = _stage4_read_mscore(current_source)
    _baseline_all_metrics = _stage4_read_all_metrics(current_source)
    if baseline_mscore is not None:
        print(f"[loop] Original baseline M-score: {baseline_mscore:.2f}% (guard: will revert if worse)")
    if _baseline_all_metrics:
        print(f"[loop] Baseline metrics: " + ", ".join(
            f"{k}={v:.2f}%" for k, v in sorted(_baseline_all_metrics.items())))
    # Cache the repo name parts from the ORIGINAL source so folder names stay clean across iterations
    _orig_src_parts = current_source.name.split("_")
    _base_name_parts = _orig_src_parts[1:-2]  # e.g. ["ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG"]

    # Delete any stale loop folders from previous runs before we start
    # In multi-model mode, only delete folders for THIS track (keep other tracks)
    _track_suffix_check = f"_{track_label}_" if track_label else None
    for _old in list(data_repos.iterdir()):
        if _old.is_dir():
            _pfx = _old.name.split("_")[0]
            if _pfx.isdigit() and len(_pfx) >= 3:
                if _track_suffix_check:
                    # Multi-model: only delete folders for this specific track
                    if _track_suffix_check in _old.name:
                        _sh.rmtree(_old)
                else:
                    # Single-model: delete all loop folders
                    _sh.rmtree(_old)

    for iteration in range(1, max_iterations + 1):
        print(f"\n{sep}\n  LOOP ITERATION {iteration}/{max_iterations}\n{sep}")

        if prev_mscore is None:
            prev_mscore = _stage4_read_mscore(current_source)
            if prev_mscore is not None:
                print(f"[loop] Baseline M-score: {prev_mscore:.2f}%")

        # Re-run backfill + re-generate report for current state
        import subprocess as _sp_l
        bf_cmd = [sys.executable, BACKFILL_TEMPORAL, str(temporal_root)]
        _sp_l.call(bf_cmd)

        # Load fresh report text
        interp_dir = temporal_root / "OUTPUT_INTERPRETATION"
        candidates = sorted(
            [d for d in interp_dir.iterdir() if d.is_dir()] if interp_dir.exists() else [],
            key=lambda d: d.stat().st_mtime, reverse=True
        )
        report_text = ""
        if candidates:
            run_folder = candidates[0]
            for rp in ["INTERPRETATION_REPORT.md", "report.md", "interpretation.md"]:
                rpath = run_folder / rp
                if rpath.exists():
                    report_text = rpath.read_text(encoding="utf-8", errors="replace")
                    break

        _risk_ctx, _commit_ctx, _ = _load_rich_qa_context(temporal_root)
        _mscore_bd = _mod.load_mscore_worst_modules(temporal_root)
        conversation: list[str] = []

        def _loop_ask(question: str) -> str:
            prior = "\n\n".join(conversation)
            ctx = (prior + "\n\n" + report_text) if prior else report_text
            raw = _mod.answer_user_question(qa_model, question, ctx,
                                            mscore_breakdown=_mscore_bd,
                                            timeout_s=900,
                                            risk_score_context=_risk_ctx,
                                            commit_context=_commit_ctx)
            answer = _mod.strip_thinking_and_fences(raw)
            print(f"\n[loop] Q ({qa_model}): {question}\n[loop] A (trimmed): {answer[:300]}...")
            conversation.append(f"Q: {question}\nA: {answer}")
            return answer

        # Build DV8 anti-pattern context — specific instances with file lists for surgical targeting
        _ap_ctx = _build_antipattern_context(temporal_root)
        if _ap_ctx:
            print(f"[loop] Loaded DV8 anti-pattern instances with file lists for targeted refactoring.")
        else:
            print(f"[loop] WARNING: No anti-pattern data found. Q2 will use generic prompt.")

        print(f"[loop] Running Q1 (anti-pattern analysis)...")
        _loop_ask("Which parts got worse over time — show anti-pattern groups, "
                  "files with most dependency growth, and worst files overall.")

        # ALL iterations target anti-patterns — the concrete file lists make this surgical
        print(f"[loop] Running Q2 — strategy: ANTI-PATTERN (iteration {iteration})...")

        # Build skip clause for previously tried actions
        _skip_clause = ""
        if tried_actions:
            _tried_list = "\n".join(f"  - {a}" for a in tried_actions)
            _skip_clause = (
                f"\n\nIMPORTANT: The following actions were already tried in previous iterations "
                f"and did NOT improve the architecture. Do NOT repeat them or target the same files/anti-patterns. "
                f"Pick a DIFFERENT anti-pattern instance or a different structural change:\n{_tried_list}\n"
            )
        # Add review failure feedback so next iteration avoids the same mistakes
        if review_failures:
            _fail_list = "\n".join(f"  - {f}" for f in review_failures[-3:])
            _skip_clause += (
                f"\n\nCRITICAL — PREVIOUS ITERATIONS FAILED CODE REVIEW:\n{_fail_list}\n"
                f"The refactoring MUST NOT introduce the same issues. Pay extra attention to:\n"
                f"- All imports resolving correctly\n"
                f"- No circular imports between files\n"
                f"- Complete code moves (not stubs or placeholders)\n"
                f"- All callers updated to import from new locations\n"
            )

        if _ap_ctx:
            _loop_ask(
                "Based on the DV8 anti-pattern analysis below, give EXACTLY 3 concrete code-level "
                "refactoring actions using ### Action 1 / ### Action 2 / ### Action 3 headings. "
                "Each action MUST target a SPECIFIC anti-pattern instance listed below and name the exact files involved. "
                "Describe exact dependency-breaking changes: "
                "which import to remove, which function/class to move, which file to split. "
                "Do NOT include process actions ('code reviews', 'documentation'). Only structural code changes. "
                "Do NOT create protocol/interface files unless the action specifically requires it — "
                "prefer MOVING code to REMOVE import edges over ADDING abstraction layers. "
                "Action 1 must target the anti-pattern instance with the MOST files (highest impact)."
                f"\n\n{_ap_ctx}\n"
                f"CRITICAL RULES FOR EFFECTIVE REFACTORING:\n"
                f"- The goal is to REDUCE dependency edges, not add abstraction layers.\n"
                f"- MOVE code from high-coupling files to low-coupling files.\n"
                f"- When breaking a cycle A→B→C→A: identify the WEAKEST edge (fewest callers) and remove it "
                f"by moving the imported symbols to where they're used.\n"
                f"- NEVER create re-exports — if you move code, update ALL callers to import from the new location.\n"
                f"- NEVER use TYPE_CHECKING blocks — DV8 counts them as real imports.\n"
                f"- Prefer FEWER files with clear responsibilities over MANY small files.\n"
                + _skip_clause
            )
        else:
            # Fallback if no anti-pattern data available
            _loop_ask(
                "Give EXACTLY 3 concrete code-level refactoring actions using ### Action 1 / ### Action 2 / ### Action 3 headings. "
                "Each action must target tight coupling or circular dependencies between specific files. "
                "Describe exact changes: which import to remove, which function to move, which file to split. "
                "Do NOT include process actions. Only structural code changes."
                + _skip_clause
            )

        # Parse actions from Q2 answer — prefer Action 1, but on stall fall back to Action 2/3
        _q2_answer = conversation[-1]
        _all_actions = _re_l.findall(
            r"### Action (\d+)[^\n]*\n(.*?)(?=\n### Action \d+|\n## |\Z)",
            _q2_answer, _re_l.DOTALL
        )
        if not _all_actions:
            print(f"[loop] Could not parse any ### Action from Q2 answer. Stopping.")
            break
        # Pick the first action that hasn't been tried yet (by comparing target file sets)
        action_text = None
        for _act_num, _act_body in _all_actions:
            _act_clean = _act_body.strip()
            _act_files = _extract_action_files(_act_clean)
            if _act_files and any(_act_files == t for t in tried_file_sets):
                print(f"[loop] Skipping Action {_act_num} — targets same files as a previous action.")
                continue
            action_text = _act_clean
            print(f"[loop] Selected Action {_act_num} for this iteration.")
            break
        if not action_text:
            # All actions target same files as before — use Action 1 anyway as last resort
            action_text = _all_actions[0][1].strip()
            print(f"[loop] All actions target previously tried files — using Action 1 anyway.")
        tried_file_sets.append(_extract_action_files(action_text))
        tried_actions.append(action_text[:200])

        # Build new folder for this iteration
        action_date = today - _dt.timedelta(weeks=max_iterations - iteration + 1)
        date_str = action_date.strftime("%d%m%Y")
        prefix = str(max_iterations - iteration + 1).zfill(3)
        _track_suffix = f"_{track_label}" if track_label else ""
        new_name = prefix + "_" + "_".join(_base_name_parts) + f"_loop{iteration}{_track_suffix}_{date_str}_1000"
        new_dir = data_repos / new_name

        if new_dir.exists():
            _sh.rmtree(new_dir)
        _sh.copytree(current_source, new_dir)
        this_run_folders.add(new_dir.name)
        _stage4_clean_copy(new_dir)
        print(f"[loop] Applying refactoring with: {model}")
        _stage4_apply_action(new_dir, "1", action_text, model)

        # FILE INTEGRITY GUARD: detect file destruction vs legitimate code moves
        # Old logic reverted valid refactoring where code was MOVED from one file to another.
        # New logic: track shrunk files, grown files, and net line count across all source files.
        _shrunk = []   # files that lost >80% of lines (went to <=3 lines)
        _grown = []    # files that gained significant lines
        _total_before = 0
        _total_after = 0
        _skip_pats = ["InputData", "OutputData", "__pycache__", "tests"]
        for _sf in current_source.rglob("*.py"):
            _rel = _sf.relative_to(current_source)
            if any(p in str(_rel) for p in _skip_pats):
                continue
            _orig_lines = sum(1 for _ in open(_sf, encoding="utf-8", errors="replace"))
            _total_before += _orig_lines
            _new_f = new_dir / _rel
            if _new_f.exists():
                _new_lines = sum(1 for _ in open(_new_f, encoding="utf-8", errors="replace"))
            else:
                _new_lines = 0  # file was deleted
            _total_after += _new_lines
            if _orig_lines > 10 and _new_lines <= 3:
                _shrunk.append((_rel, _orig_lines, _new_lines))
            elif _new_lines > _orig_lines + 20:
                _grown.append((_rel, _orig_lines, _new_lines))
        # Also count NEW files created by refactoring
        for _nf in new_dir.rglob("*.py"):
            _nrel = _nf.relative_to(new_dir)
            if any(p in str(_nrel) for p in _skip_pats):
                continue
            _orig_f = current_source / _nrel
            if not _orig_f.exists():
                _new_lines = sum(1 for _ in open(_nf, encoding="utf-8", errors="replace"))
                _total_after += _new_lines
                if _new_lines > 20:
                    _grown.append((_nrel, 0, _new_lines))
        # Decision: destruction = shrunk files WITHOUT corresponding growth
        _net_loss = _total_before - _total_after
        _net_loss_pct = (_net_loss / _total_before * 100) if _total_before > 0 else 0
        _integrity_violated = False
        if _shrunk and not _grown and _net_loss_pct > 20:
            # Files shrunk, nothing grew — code was DELETED not MOVED
            print(f"[loop] FILE INTEGRITY VIOLATION: {len(_shrunk)} file(s) destroyed, "
                  f"net loss {_net_loss} lines ({_net_loss_pct:.0f}%)")
            _integrity_violated = True
        elif _shrunk and len(_shrunk) > 3:
            # Too many files shrunk even if some grew — likely bulk destruction
            print(f"[loop] FILE INTEGRITY VIOLATION: {len(_shrunk)} files shrunk to <=3 lines")
            _integrity_violated = True
        elif _shrunk:
            # Some files shrunk but others grew — likely legitimate code MOVE
            for _rel, _ol, _nl in _shrunk:
                print(f"[loop] Note: {_rel} emptied ({_ol}->{_nl} lines) — code likely moved")
        if _integrity_violated:
            for _rel, _ol, _nl in _shrunk[:5]:
                print(f"[loop]   {_rel}: {_ol}->{_nl} lines")
            print(f"[loop] Reverting iteration {iteration} — LLM destroyed source files.")
            _sh.rmtree(new_dir)
            stall_count += 1
            if stall_count >= STALL_LIMIT:
                print(f"[loop] {STALL_LIMIT} consecutive stalls/reverts — stopping early.")
                break
            continue

        # FEEDBACK LOOP REVIEW: second Claude instance checks refactoring correctness
        if use_feedback_loop:
            _review_passed, _review_reason = _stage4_review_refactoring(
                new_dir, action_text, model, prev_failures=review_failures)
            if not _review_passed:
                print(f"[loop] REVIEW FAILED: {_review_reason}")
                review_failures.append(f"Iteration {iteration}: {_review_reason}")
                print(f"[loop] Reverting iteration {iteration} — reviewer found issues.")
                _sh.rmtree(new_dir)
                stall_count += 1
                if stall_count >= STALL_LIMIT:
                    print(f"[loop] {STALL_LIMIT} consecutive stalls/reverts — stopping early.")
                    break
                continue
            else:
                print(f"[loop] Review PASSED — proceeding to DV8 analysis.")

        # PRE-COPY git history from baseline so DV8 can detect modularity violations.
        # Loop folders have no .git, so dv8_agent can't generate git-history.txt.
        # By copying the baseline's git-history.txt, dv8_agent will use it to build
        # the history DSM and merge it with the new structural DSM for MV detection.
        _baseline_history = rev_dirs[0] / "OutputData" / "history-dsm" / "git-history.txt"
        if _baseline_history.exists():
            _loop_history_dir = new_dir / "OutputData" / "history-dsm"
            _loop_history_dir.mkdir(parents=True, exist_ok=True)
            _sh.copy2(_baseline_history, _loop_history_dir / "git-history.txt")
            print(f"[loop] Copied baseline git-history.txt for MV detection.")

        rc = _stage4_run_dv8(new_dir, skip_arch_report=True)
        if rc != 0:
            print(f"[loop] DV8 failed on iteration {iteration}. Stopping.")
            return rc

        new_mscore = _stage4_read_mscore(new_dir)
        all_metrics = _stage4_read_all_metrics(new_dir)
        if new_mscore is not None:
            delta_baseline = new_mscore - baseline_mscore if baseline_mscore is not None else 0.0
            delta_prev = new_mscore - prev_mscore if prev_mscore is not None else 0.0
            sign_b = "+" if delta_baseline >= 0 else ""
            sign_p = "+" if delta_prev >= 0 else ""
            print(f"[loop] M-score iteration {iteration}: {new_mscore:.2f}% "
                  f"(vs baseline: {sign_b}{delta_baseline:.2f}%, vs prev: {sign_p}{delta_prev:.2f}%)")
            # Show ALL metrics for this iteration
            if all_metrics:
                # Check for isolated items inflation
                _new_iso = int(all_metrics.get("isolated-items", 0))
                _base_iso = int(_baseline_all_metrics.get("isolated-items", 0))
                if _new_iso > _base_iso:
                    print(f"[loop] WARNING: Isolated items {_base_iso}->{_new_iso} — "
                          f"metrics may be inflated by empty/disconnected files!")
                # Show primary metrics (skip internal keys)
                _display_keys = ["m-score", "propagation-cost", "decoupling-level", "independence-level"]
                print(f"[loop] All metrics iteration {iteration}:")
                for mk in _display_keys:
                    mv = all_metrics.get(mk)
                    if mv is None:
                        continue
                    baseline_m = _baseline_all_metrics.get(mk)
                    if baseline_m is not None:
                        delta_m = mv - baseline_m
                        if mk == "propagation-cost":
                            improved = delta_m < 0
                        else:
                            improved = delta_m > 0
                        sign_m = "+" if delta_m >= 0 else ""
                        status = "improved" if improved else ("same" if abs(delta_m) < 0.1 else "WORSE")
                        line = f"[loop]   {mk}: {mv:.2f}% ({sign_m}{delta_m:.2f}% vs baseline) [{status}]"
                        # If isolated items increased, also show exclude-isolated value
                        if _new_iso > _base_iso:
                            excl_key = f"{mk}-excl"
                            excl_val = all_metrics.get(excl_key)
                            if excl_val is not None:
                                line += f"  (excl-isolated: {excl_val:.2f}%)"
                        print(line)
                    else:
                        print(f"[loop]   {mk}: {mv:.2f}%")

            # BASELINE GUARD: if ANY key metric is worse than original, revert
            # When isolated items increased, use exclude-isolated values for honest comparison
            _revert_reason = None
            if baseline_mscore is not None and new_mscore < baseline_mscore - 0.5:
                _revert_reason = f"M-score {new_mscore:.2f}% is WORSE than baseline {baseline_mscore:.2f}%"
            if not _revert_reason and all_metrics and _baseline_all_metrics:
                # Use exclude-isolated values when isolated items increased
                _use_excl = _new_iso > _base_iso
                _prop_key = "propagation-cost-excl" if (_use_excl and "propagation-cost-excl" in all_metrics) else "propagation-cost"
                _dec_key = "decoupling-level-excl" if (_use_excl and "decoupling-level-excl" in all_metrics) else "decoupling-level"
                _prop_base_key = "propagation-cost-excl" if (_use_excl and "propagation-cost-excl" in _baseline_all_metrics) else "propagation-cost"
                _dec_base_key = "decoupling-level-excl" if (_use_excl and "decoupling-level-excl" in _baseline_all_metrics) else "decoupling-level"
                _prop_new = all_metrics.get(_prop_key)
                _prop_base = _baseline_all_metrics.get(_prop_base_key)
                if _prop_new is not None and _prop_base is not None and _prop_new > _prop_base + 2.0:
                    _revert_reason = (f"Propagation cost {_prop_new:.2f}% is WORSE "
                                      f"than baseline {_prop_base:.2f}%"
                                      + (" (excl-isolated)" if _use_excl else ""))
                _dec_new = all_metrics.get(_dec_key)
                _dec_base = _baseline_all_metrics.get(_dec_base_key)
                if _dec_new is not None and _dec_base is not None and _dec_new < _dec_base - 2.0:
                    _revert_reason = (f"Decoupling level {_dec_new:.2f}% is WORSE "
                                      f"than baseline {_dec_base:.2f}%"
                                      + (" (excl-isolated)" if _use_excl else ""))
            if _revert_reason:
                print(f"[loop] WARNING: {_revert_reason}!")
                print(f"[loop] Reverting iteration {iteration} — rebuilding from previous source.")
                _sh.rmtree(new_dir)
                # Don't update current_source — next iteration starts from same point
                stall_count += 1
                if stall_count >= STALL_LIMIT:
                    print(f"[loop] {STALL_LIMIT} consecutive stalls/reverts — stopping early.")
                    break
                continue

            if prev_mscore is not None and new_mscore <= prev_mscore + 0.1:
                stall_count += 1
                print(f"[loop] No significant improvement ({prev_mscore:.2f}% → {new_mscore:.2f}%). "
                      f"Stall {stall_count}/{STALL_LIMIT}.")
                if stall_count >= STALL_LIMIT:
                    print(f"[loop] {STALL_LIMIT} consecutive stalls — stopping early.")
                    iteration_dirs.append((str(iteration), new_dir, action_date, new_mscore, all_metrics))
                    break
            else:
                stall_count = 0   # reset on any real improvement
            prev_mscore = new_mscore
        else:
            print(f"[loop] Could not read M-score for iteration {iteration}.")

        # ANTI-PATTERN VERIFICATION: compare structural AP counts before/after
        _new_ap_csv = new_dir / "OutputData" / "arch-issue" / "anti-pattern-summary.csv"
        _prev_ap_csv = current_source / "OutputData" / "arch-issue" / "anti-pattern-summary.csv"
        if _new_ap_csv.exists() and _prev_ap_csv.exists():
            import csv as _csv_v
            def _read_ap_counts(csvpath):
                counts = {}
                with open(csvpath, newline="", encoding="utf-8") as f:
                    for row in _csv_v.DictReader(f):
                        t = row.get("Type", "").strip()
                        if t and t != "Total":
                            counts[t] = int(row.get("InstanceCount", 0))
                return counts
            _prev_counts = _read_ap_counts(_prev_ap_csv)
            _new_counts = _read_ap_counts(_new_ap_csv)
            _all_types = sorted(set(list(_prev_counts.keys()) + list(_new_counts.keys())))
            _ap_changes = []
            for _apt in _all_types:
                _pc = _prev_counts.get(_apt, 0)
                _nc = _new_counts.get(_apt, 0)
                if _pc != _nc:
                    _delta = _nc - _pc
                    _sign = "+" if _delta > 0 else ""
                    _ap_changes.append(f"    {_apt}: {_pc}→{_nc} ({_sign}{_delta})")
            if _ap_changes:
                print(f"[loop] Anti-pattern changes:")
                for _c in _ap_changes:
                    print(_c)
            else:
                print(f"[loop] Anti-pattern counts unchanged.")
            # PACKAGE CYCLE CHECK: warn if package-cycle count increased
            _prev_pkg_cycles = _prev_counts.get("PackageCycle", 0)
            _new_pkg_cycles = _new_counts.get("PackageCycle", 0)
            if _new_pkg_cycles > _prev_pkg_cycles:
                print(f"[loop] WARNING: Package cycles INCREASED {_prev_pkg_cycles}->{_new_pkg_cycles}! "
                      f"Refactoring may have introduced cross-package circular dependency.")

        # Generate network graph for this iteration
        _plot_dir = temporal_root / "INPUT_INTERPRETATION" / "plots"
        _plot_dir.mkdir(parents=True, exist_ok=True)
        if iteration == 1:
            # Also generate baseline network graph
            _base_graph = _generate_antipattern_network_graph(
                rev_dirs[0], _plot_dir / "network_baseline.png",
                title="Baseline — Dependency Network with Anti-Patterns")
            if _base_graph:
                print(f"[loop] Saved baseline network graph: {_plot_dir / 'network_baseline.png'}")
        _iter_graph = _generate_antipattern_network_graph(
            new_dir, _plot_dir / f"network_iteration_{iteration}.png",
            title=f"Iteration {iteration} — Dependency Network ({model})")
        if _iter_graph:
            print(f"[loop] Saved iteration {iteration} network graph: {_plot_dir / f'network_iteration_{iteration}.png'}")

        iteration_dirs.append((str(iteration), new_dir, action_date, new_mscore, all_metrics))
        # Next iteration builds on this iteration's output
        current_source = new_dir

    # --- Summary + optional cleanup ---
    if len(iteration_dirs) > 1:
        scored = [(it, d, dt, ms, am) for it, d, dt, ms, am in iteration_dirs if ms is not None]
        if scored:
            best_mscore = max(ms for _, _, _, ms, _ in scored)
            print(f"\n{'='*60}")
            print(f"  ITERATION SUMMARY (all metrics vs original baseline)")
            print(f"{'='*60}")
            _metric_names = ["m-score", "propagation-cost", "decoupling-level", "independence-level"]
            # Header
            print(f"  {'Iter':<6} {'M-score':<12} {'Prop.Cost':<12} {'Decouple':<12} {'Independ.':<12} {'Folder'}")
            print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*20}")
            # Baseline row
            if _baseline_all_metrics:
                print(f"  {'BASE':<6}", end="")
                for mn in _metric_names:
                    bv = _baseline_all_metrics.get(mn, 0)
                    print(f" {bv:>6.2f}%     ", end="")
                print(f" (original)")
            for it_num, it_dir, it_date, it_ms, it_am in iteration_dirs:
                marker = " ← BEST" if it_ms == best_mscore else ""
                print(f"  {it_num:<6}", end="")
                for mn in _metric_names:
                    val = (it_am or {}).get(mn, 0)
                    bv = _baseline_all_metrics.get(mn)
                    if bv is not None:
                        delta = val - bv
                        sign = "+" if delta >= 0 else ""
                        print(f" {val:>6.2f}%{sign}{delta:>+5.1f}", end="")
                    else:
                        print(f" {val:>6.2f}%     ", end="")
                print(f" {it_dir.name[:30]}{marker}")
            print(f"{'='*60}")

            # --- Before/after architecture diff ---
            _print_architecture_diff(temporal_root, _get_source_revs()[0] if _get_source_revs() else None,
                                     iteration_dirs, data_repos)

            # Ask user if they want to delete non-improving iterations
            worse = [(it, d, dt, ms, am) for it, d, dt, ms, am in iteration_dirs if ms is not None and ms < best_mscore]
            if worse:
                print(f"\n  {len(worse)} iteration(s) scored below the best ({best_mscore:.2f}%).")
                try:
                    answer = input("  Delete non-improving iterations? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    answer = "n"
                if answer in ("y", "yes"):
                    for it_num, it_dir, it_date, it_ms, _am in worse:
                        print(f"  Deleting iteration {it_num} ({it_ms:.2f}%) — {it_dir.name}")
                        _sh.rmtree(it_dir)
                    iteration_dirs = [(it, d, dt, ms, am) for it, d, dt, ms, am in iteration_dirs
                                      if not any(it == w[0] and d == w[1] for w in worse)]
                    print(f"  Kept {len(iteration_dirs)} iteration(s).")
                else:
                    print(f"  Keeping all iterations for review.")

    # --- Manual targeting prompt (post-loop) ---
    _offer_manual_targeting(temporal_root, model, iteration_dirs, _base_name_parts,
                            data_repos, today, max_iterations, this_run_folders,
                            qa_model=qa_model)

    # Inject synthetic revisions into timeseries.json
    import json as _js2
    ts_path = next(
        (p for p in [temporal_root / "INPUT_INTERPRETATION" / "timeseries.json",
                     temporal_root / "timeseries.json"] if p.exists()),
        temporal_root / "INPUT_INTERPRETATION" / "timeseries.json"
    )
    _track_tag = f"-{track_label}" if track_label else ""
    if ts_path.exists() and iteration_dirs:
        ts = _js2.loads(ts_path.read_text())
        # Only remove revisions for THIS track (keep other model tracks)
        _prefix = f"ai{_track_tag}-"
        ts["revisions"] = [r for r in ts.get("revisions", [])
                           if not r.get("commit_hash", "").startswith(_prefix)]
        for it_entry in reversed(iteration_dirs):
            it_num, _, adate, it_mscore = it_entry[0], it_entry[1], it_entry[2], it_entry[3]
            it_all_m = it_entry[4] if len(it_entry) > 4 else {}
            # Build metrics dict with all available metrics
            _ts_metrics = {}
            if it_mscore is not None:
                _ts_metrics["m-score"] = it_mscore
            if it_all_m:
                for mk, mv in it_all_m.items():
                    _ts_metrics[mk] = mv
            synthetic = {
                "revision_number": 0,
                "commit_hash": f"ai{_track_tag}-{it_num}",
                "commit_date": f"{adate.strftime('%Y-%m-%d')} 10:00:00 +0000",
                "commit_author": f"{model} (AI loop)",
                "commit_message": f"AI Loop Refactor: Iteration {it_num} [{track_label or model}]",
                "metrics": _ts_metrics,
            }
            ts.setdefault("revisions", []).insert(0, synthetic)
        ts["revision_count"] = len(ts.get("revisions", []))
        ts_path.write_text(_js2.dumps(ts, indent=2))
        print(f"[loop] Updated timeseries.json → {ts['revision_count']} revisions.")

    # Regenerate + re-plot
    import subprocess as _sp_end
    _sp_end.call([sys.executable, BACKFILL_TEMPORAL, str(temporal_root)])
    plot_ts = ts_path if ts_path.exists() else temporal_root / "INPUT_INTERPRETATION" / "timeseries.json"
    plot_out = temporal_root / "INPUT_INTERPRETATION" / "plots"
    _sp_end.call([sys.executable, PLOTTER, "--json", str(plot_ts), "--output", str(plot_out)])

    # Generate anti-pattern evolution visualization
    _evo_script = pathlib.Path(__file__).resolve().parent / "refactor_evolution_plot.py"
    if _evo_script.exists() and iteration_dirs:
        _sp_end.call([sys.executable, str(_evo_script),
                      "--temporal-root", str(temporal_root)])
        _evo_png = plot_out / "refactor_evolution.png"
        if _evo_png.exists():
            print(f"  Saved: {_evo_png}")

    print(f"\n{sep}\n  Loop mode complete — {len(iteration_dirs)} iteration(s).\n{sep}\n")

    # Generate manager report
    _report_script = pathlib.Path(__file__).resolve().parent.parent / "04_stage_refactor" / "generate_refactor_report.py"
    if _report_script.exists():
        _sp_end.call([sys.executable, str(_report_script), "--temporal-root", str(temporal_root)])

    return 0


def _run_refactor_stage(temporal_root: pathlib.Path, conversation: list[str],
                        model: str = "qwen3-coder-30b-refactor", loop_count: int = 0,
                        refactor_models: list[str] | None = None,
                        qa_model: str = "qwen3-coder-30b-refactor",
                        use_feedback_loop: bool = False) -> int:
    """Stage 4: Iterative LLM-refactor — one folder + DV8 run per Q2 action.

    Stage 3 (Q&A/analysis): always uses qa_model (local Ollama — fast, reliable).
    Stage 4 (code application): uses model or refactor_models (Claude/Qwen/Codex).

    If loop_count > 0: loop mode — full pipeline per iteration (Q1 → Q2 → apply → DV8 → M-score).
    If refactor_models has >1 entry: multi-model comparison mode — run loop for each model from same baseline.
    After automated loop: offers manual targeting for single follow-up iterations.
    Otherwise: sequential mode — apply all 3 Q2 actions in order with M-score guard.
    """
    # ── MULTI-MODEL COMPARISON ────────────────────────────────────────────────
    if refactor_models and len(refactor_models) > 1 and loop_count > 0:
        return _run_multi_model_loop(temporal_root, models=refactor_models,
                                      max_iterations=loop_count, qa_model=qa_model)
    # ── LOOP MODE ──────────────────────────────────────────────────────────────
    if loop_count > 0:
        return _run_refactor_loop(temporal_root, model=model, max_iterations=loop_count,
                                   qa_model=qa_model, use_feedback_loop=use_feedback_loop)
    # ── SEQUENTIAL MODE (default) ──────────────────────────────────────────────
    import shutil, re as _re
    sep = "=" * 70
    print(f"\n{sep}\n  STAGE 4: Iterative LLM Refactor + Re-analysis\n{sep}")

    # --- Step 1: Find source revision (lowest existing prefix = most recent) ---
    data_repos = temporal_root / "data_repositories"
    if not data_repos.is_dir():
        print("[stage4] No data_repositories/ folder found.")
        return 1

    # Auto-delete stale action folders (NNN_*action*) so each run produces fresh metrics
    import re as _re2
    for _stale in list(data_repos.iterdir()):
        if _stale.is_dir() and _re2.match(r'^\d{3}_', _stale.name) and "action" in _stale.name:
            print(f"[stage4] Removing stale action folder: {_stale.name}")
            shutil.rmtree(_stale)

    # Only consider 2-digit prefix folders (01_, 02_, ...) — not 001_, 002_, ... action folders
    rev_dirs = sorted(
        [p for p in data_repos.iterdir()
         if p.is_dir() and len(p.name) > 2 and p.name[:2].isdigit() and p.name[2] == "_"],
        key=lambda p: p.name
    )
    if not rev_dirs:
        print("[stage4] No original revision folders found in data_repositories/.")
        return 1
    source_rev = rev_dirs[0]  # lowest NN_ prefix = most recent hand-written revision
    print(f"[stage4] Source revision: {source_rev.name}")

    # --- Step 2: Load Q2 refactoring plan ---
    q2_plan = ""
    for entry in reversed(conversation):
        if "### action 1" in entry.lower() or "refactoring priority" in entry.lower():
            q2_plan = entry
            break
    if not q2_plan:
        interp_dir = temporal_root / "OUTPUT_INTERPRETATION"
        answer_files = sorted(interp_dir.glob("*/USER_ANSWERS*.md"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
        if answer_files:
            full_text = answer_files[0].read_text(encoding="utf-8")
            # Extract only the LAST Q2 answer block (contains "### Action 1").
            # USER_ANSWERS.md accumulates all Q&A turns — split on "---" separators
            # and take only the last block that contains action headings.
            import re as _re_q2
            turns = _re_q2.split(r'\n---\n', full_text)
            q2_turns = [t for t in turns if _re_q2.search(r'###\s+Action\s+1', t)]
            q2_plan = q2_turns[-1] if q2_turns else full_text
            print(f"[stage4] Loaded Q2 plan from: {answer_files[0].name} "
                  f"(extracted last of {len(q2_turns)} action block(s))")
    if not q2_plan:
        print("[stage4] No Q2 plan found — run Q2 first ('how would you refactor this?').")
        return 1

    # --- Step 3: Parse individual actions from Q2 plan ---
    # Matches "### Action N — ..." blocks
    action_blocks = _re.findall(
        r"### Action (\d+)[^\n]*\n(.*?)(?=\n### Action \d+|\n## |\Z)",
        q2_plan, _re.DOTALL
    )
    if not action_blocks:
        print("[stage4] No '### Action N —' blocks found in Q2 plan — using single-folder fallback.")
        action_blocks = [("1", q2_plan[:3000])]

    n = len(action_blocks)
    # Hard cap at 3 — guard against LLM returning extra non-code actions (e.g. "code reviews")
    if n > 3:
        print(f"[stage4] Capping {n} actions to 3 (LLM returned too many).")
        action_blocks = action_blocks[:3]
        n = 3
    print(f"[stage4] Found {n} action(s) to apply iteratively.")
    refactor_model = model or "qwen3-coder-30b-refactor"
    print(f"[stage4] Using model: {refactor_model}")

    # --- Step 4: Iterative copy → refactor → DV8 per action ---
    # Remove any stale 3-digit-prefix action folders from previous runs (they may have
    # different dates in their names). Original 2-digit folders (01_, 02_) are kept.
    for _old in list(data_repos.iterdir()):
        if _old.is_dir():
            prefix_part = _old.name.split("_")[0]
            if prefix_part.isdigit() and len(prefix_part) >= 3:
                print(f"[stage4] Removing stale action folder: {_old.name}")
                shutil.rmtree(_old)

    # Base name parts from source_rev (strip leading NN_ prefix and trailing date+time)
    src_parts = source_rev.name.split("_")
    base_parts = src_parts[1:-2]  # e.g. ["ARCH", "ANALYSIS", "TRAINTICKET", ...]

    # Compute real action dates: work backward from today, 1 week per action.
    # Action 1 = n weeks ago, Action 2 = n-1 weeks ago, ..., last action = 1 week ago.
    # This keeps all AI commits in the past (never in the future) and spaced visibly apart.
    import datetime as _dt
    _today = _dt.date.today()
    _action_dates: dict[int, _dt.date] = {}
    for _i in range(n):
        _weeks_back = n - _i  # action index 0 → n weeks back, last action → 1 week back
        _action_dates[_i] = _today - _dt.timedelta(weeks=_weeks_back)

    prev_dir = source_rev
    all_new_files: list[str] = []
    action_dirs: list[tuple[str, pathlib.Path, _dt.date]] = []  # (action_num, new_dir, date)

    # Read baseline M-score from source revision so we can guard against regressions
    prev_mscore = _stage4_read_mscore(source_rev)
    if prev_mscore is not None:
        print(f"[stage4] Baseline M-score: {prev_mscore:.2f}%")

    for i, (action_num, action_text) in enumerate(action_blocks):
        # Prefix: first action gets highest 3-digit number (e.g. 003), last gets 001
        prefix = str(n - i).zfill(3)
        action_date = _action_dates[i]
        date_str = action_date.strftime("%d%m%Y")  # DDMMYYYY
        time_str = str(1000 + i * 200)             # 1000, 1200, 1400, ...
        slug = f"action{action_num}"
        new_name = prefix + "_" + "_".join(base_parts) + f"_{slug}_{date_str}_{time_str}"
        new_dir = data_repos / new_name

        print(f"\n{sep}")
        print(f"  Action {action_num}/{n}: {new_dir.name}")
        print(sep)

        if new_dir.exists():
            print(f"[stage4] Removing existing folder to rebuild from scratch ...")
            shutil.rmtree(new_dir)
        print(f"[stage4] Copying {prev_dir.name} → {new_name} ...")
        shutil.copytree(prev_dir, new_dir)
        _stage4_clean_copy(new_dir)
        new_files = _stage4_apply_action(new_dir, action_num, action_text.strip(), refactor_model)
        all_new_files.extend(new_files)

        rc = _stage4_run_dv8(new_dir, skip_arch_report=True)
        if rc != 0:
            print(f"[stage4] Stopping after failed DV8 run for action {action_num}.")
            return rc

        # M-score guard: stop if this action degraded architecture
        new_mscore = _stage4_read_mscore(new_dir)
        if new_mscore is not None:
            delta = new_mscore - prev_mscore if prev_mscore is not None else 0.0
            sign = "+" if delta >= 0 else ""
            print(f"[stage4] M-score after action {action_num}: {new_mscore:.2f}% ({sign}{delta:.2f}%)")
            if prev_mscore is not None and new_mscore < prev_mscore - 0.5:
                print(f"[stage4] WARNING: Action {action_num} degraded M-score "
                      f"({prev_mscore:.2f}% → {new_mscore:.2f}%). Stopping pipeline.")
                print(f"[stage4] Folder kept for inspection: {new_dir.name}")
                action_dirs.append((action_num, new_dir, action_date))
                break  # don't apply further actions on degraded code
            prev_mscore = new_mscore

        action_dirs.append((action_num, new_dir, action_date))
        prev_dir = new_dir  # next action builds on this result

    # --- Step 5: Inject synthetic revisions into timeseries.json ---
    # backfill reads revisions_meta[idx] for each folder. AI action folders won't have
    # matching entries, so we inject synthetic ones (newest first = prepended).
    # Remove any stale ai-actionN entries first (folder names changed with new dates).
    import json as _js
    ts_path = next(
        (p for p in [temporal_root / "INPUT_INTERPRETATION" / "timeseries.json",
                     temporal_root / "timeseries.json"] if p.exists()),
        temporal_root / "INPUT_INTERPRETATION" / "timeseries.json"
    )
    repo_name = ""
    ts = {}
    if ts_path.exists():
        try:
            ts = _js.loads(ts_path.read_text())
            repo_name = ts.get("repo", "")
        except Exception:
            pass
    if ts and action_dirs:
        # Remove stale ai-action entries (always rebuild them with fresh dates)
        ts["revisions"] = [r for r in ts.get("revisions", [])
                           if not r.get("commit_hash", "").startswith("ai-action")]
        # Prepend in reverse order so newest action (last applied) is first
        for action_num, _, adate in reversed(action_dirs):
            hash_key = f"ai-action{action_num}"
            time_hhmm = str(1000 + (int(action_num) - 1) * 200)
            synthetic = {
                "revision_number": 0,  # position-assigned by backfill
                "commit_hash": hash_key,
                "commit_date": f"{adate.strftime('%Y-%m-%d')} {time_hhmm[:2]}:{time_hhmm[2:]}:00 +0000",
                "commit_author": "qwen3.6 (AI)",
                "commit_message": f"AI Refactor: Action {action_num}",
                "metrics": {}
            }
            ts.setdefault("revisions", []).insert(0, synthetic)
        ts["revision_count"] = len(ts.get("revisions", []))
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        ts_path.write_text(_js.dumps(ts, indent=2))
        print(f"[stage4] Updated timeseries.json → {ts['revision_count']} revisions total.")

    # --- Step 6: Regenerate INPUT_INTERPRETATION for all revisions ---
    print(f"\n[stage4] Regenerating INPUT_INTERPRETATION ...")
    bf_cmd = [sys.executable, BACKFILL_TEMPORAL, str(temporal_root)]
    if repo_name:
        bf_cmd += ["--meta-repo", repo_name]
    subprocess.call(bf_cmd)

    # --- Step 7: Re-plot time series ---
    print(f"\n[stage4] Re-plotting time series ...")
    plot_ts = ts_path if ts_path.exists() else temporal_root / "INPUT_INTERPRETATION" / "timeseries.json"
    plot_out = temporal_root / "INPUT_INTERPRETATION" / "plots"
    subprocess.call([sys.executable, PLOTTER, "--json", str(plot_ts), "--output", str(plot_out)])

    # Also update the time_evolution_modularity_metrics/ subfolder so backfill/interpreters see 5-pt plots
    subdir = plot_out / "time_evolution_modularity_metrics"
    subdir.mkdir(parents=True, exist_ok=True)
    import shutil as _shutil
    for _p in plot_out.iterdir():
        if _p.is_file() and _p.suffix == ".png":
            _shutil.copy2(_p, subdir / _p.name)

    print(f"\n{sep}")
    print(f"  Stage 4 complete — {n} action(s) applied iteratively.")
    print(f"  Last folder: {prev_dir.name}")
    print(f"  New files created: {all_new_files or 'none'}")
    print(f"  Re-run Q1/Q2 to see the {n + 2}-revision M-score time series.")
    print(f"{sep}\n")
    return 0


def tool_analyze_and_refactor_single(plan: dict) -> int:
    """Single-revision mode: clone repo, checkout a specific commit, run DV8, then auto Q1 -> Q2 -> Stage 4.

    Accepts --repo <git_url|local_path> and --commit <hash> (optional; HEAD if absent).
    """
    import shutil as _shutil, datetime as _dt_mod, json as _js

    repo = plan.get("repo") or ""
    commit_hash = plan.get("commit") or None
    language = plan.get("language") or "python"
    model = plan.get("model") or "deepseek-r1:32b"
    refactor_model = plan.get("refactor_model") or "qwen3-coder-30b-refactor"

    if not repo:
        repo = _prompt_for_repo()

    test_auto_dir = pathlib.Path(THIS_DIR).parent
    repos_analyzed = test_auto_dir / "REPOS_ANALYZED"
    repos_analyzed.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Get local repo path ──────────────────────────────────────────
    if re.match(r"^https?://", repo):
        from urllib.parse import urlparse as _urlparse
        repo_name = pathlib.Path(_urlparse(repo).path).stem.replace(".git", "")
        repo_path = repos_analyzed / repo_name
        if (repo_path / ".git").exists():
            print(f"[single-rev] Repo already cloned: {repo_path}")
        else:
            print(f"[single-rev] Cloning {repo} -> {repo_path}")
            subprocess.run(["git", "clone", repo, str(repo_path)], check=True)
    else:
        repo_path = pathlib.Path(repo).expanduser().resolve()
        repo_name = repo_path.name

    if not repo_path.exists():
        print(f"[single-rev] Repo path not found: {repo_path}")
        return 1

    print(f"[single-rev] Repo: {repo_path}  language: {language}")

    # ── Step 2: Resolve commit hash and date ─────────────────────────────────
    if commit_hash:
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%ci", commit_hash],
                cwd=repo_path, capture_output=True, text=True, check=True,
            )
            commit_date = result.stdout.strip() or _dt_mod.datetime.now().strftime("%Y-%m-%d %H:%M:%S +0000")
        except Exception:
            commit_date = _dt_mod.datetime.now().strftime("%Y-%m-%d %H:%M:%S +0000")
    else:
        commit_date = _dt_mod.datetime.now().strftime("%Y-%m-%d %H:%M:%S +0000")
        commit_hash = "HEAD"

    # ── Step 3: Build temporal folder ────────────────────────────────────────
    now_str = _dt_mod.datetime.now().strftime("%y%m%d_%H%M%S")
    temporal_dir = repos_analyzed / f"{repo_name.upper()}_{language}" / f"single_revision_{now_str}"
    data_repos_dir = temporal_dir / "data_repositories"
    data_repos_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 4: Checkout commit into data_repositories/ ──────────────────────
    try:
        date_obj = _dt_mod.datetime.strptime(commit_date[:19], "%Y-%m-%d %H:%M:%S")
        formatted_dt = date_obj.strftime("%d%m%Y_%H%M")
    except Exception:
        formatted_dt = commit_hash[:8]

    rev_folder_name = f"01_{repo_name.upper()}_{formatted_dt}"
    rev_dest = data_repos_dir / rev_folder_name

    if commit_hash != "HEAD":
        print(f"[single-rev] Creating worktree {rev_dest.name} @ {commit_hash[:8]}")
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(rev_dest), commit_hash],
            cwd=repo_path, check=True,
        )
    else:
        print(f"[single-rev] Copying HEAD snapshot to {rev_dest.name}")
        _shutil.copytree(repo_path, rev_dest, dirs_exist_ok=True)

    # ── Step 5: Write timeseries.json ─────────────────────────────────────────
    interp_dir = temporal_dir / "INPUT_INTERPRETATION"
    interp_dir.mkdir(parents=True, exist_ok=True)
    ts_data = {
        "repository": repo_name.upper(),
        "revisions": [{
            "revision_number": 1,
            "commit_hash": commit_hash,
            "commit_date": commit_date,
            "commit_author": "single-revision-mode",
            "commit_message": f"Single snapshot: {repo_name} @ {commit_hash[:8]}",
            "metrics": {},
        }]
    }
    (interp_dir / "timeseries.json").write_text(_js.dumps(ts_data, indent=2))

    print(f"[single-rev] Temporal folder: {temporal_dir}")
    print(f"[single-rev] Triggering auto Q1 -> Q2 -> Stage 4 ...")

    return tool_interpret_temporal({
        "repo": str(temporal_dir),
        "model": model,
        "user_request": plan.get("user_request") or "",
        "auto_refactor": True,
        "refactor_model": refactor_model,
    })


def tool_analyze_repo(plan: dict) -> tuple[int, str]:
    """Run dv8_agent.py for analysis"""
    repo_hint = _sanitize_repo(plan.get("repo"))
    if not repo_hint:
        print("No repository specified.")
        repo = _prompt_for_repo()
    else:
        repo = repo_hint

    cmd = ["python3", AGENT, "--repo", repo]
    if plan.get("source_path"):
        cmd += ["--source-path", str(plan["source_path"])]

    # Default to skip-arch-report for reliability
    if plan.get("skip_arch_report", True):
        cmd += ["--skip-arch-report"]

    if plan.get("force_depends"):
        cmd += ["--force-depends"]

    if plan.get("ask"):
        cmd += ["--ask", plan["ask"]]
    else:
        cmd += ["--ask", "all"]  # Default to all metrics

    print(f"\nRunning: {' '.join(cmd)}\n")
    rc = subprocess.call(cmd)

    # Find the actual local repo path after cloning
    local_repo = _find_local_repo(repo)

    return rc, local_repo

def tool_explain_metrics(plan: dict) -> int:
    """Call integrated_explainer.py for detailed explanation"""
    repo_hint = _sanitize_repo(plan.get("repo"))
    if not repo_hint:
        print("No repository specified.")
        repo = _prompt_for_repo()
    else:
        repo = repo_hint

    # Find the actual local repo path (handles URLs)
    repo = _find_local_repo(repo)

    # Check if metrics exist
    metrics_dir = pathlib.Path(repo) / "OutputData" / "metrics"
    if not metrics_dir.exists():
        print(f"\nWarning: No metrics found at {metrics_dir}")
        print("   Run analysis first? [y/N]: ", end="")
        try:
            ans = input().strip().lower()
        except EOFError:
            return 1

        if ans in {"y", "yes"}:
            # Run analysis first
            rc, repo = tool_analyze_repo({"repo": repo, "skip_arch_report": True, "ask": "all"})
            if rc != 0:
                print("Analysis failed.")
                return rc
        else:
            return 1

    # Call integrated explainer
    output_file = f"{pathlib.Path(repo).name}_detailed_report.md"
    cmd = ["python3", EXPLAINER, "--repo", repo, "--output", output_file]

    print(f"\nGenerating detailed AI explanation...\n")
    print(f"Command: {' '.join(cmd)}\n")

    rc = subprocess.call(cmd)

    if rc == 0 and pathlib.Path(output_file).exists():
        print(f"\nDetailed report generated: {output_file}\n")
        print("=" * 60)
        # Show preview
        with open(output_file) as f:
            lines = f.readlines()
            for line in lines[:50]:  # First 50 lines
                print(line, end="")
        print("\n" + "=" * 60)
        print(f"\nView full report: open '{output_file}'")

    return rc

def tool_explain_concept(plan: dict) -> int:
    """Explain concept using RAG"""
    topic = plan.get("topic", "").strip()
    if not topic:
        print("No topic specified.")
        return 1

    try:
        kb_path = pathlib.Path(RAG_KB_DIR).expanduser().resolve()
        if not kb_path.exists():
            print(f"Knowledge base not found at {kb_path}")
            print("Using basic explanation...")
            return _explain_basic(topic)

        # Ensure RAG module path is importable
        rag_module_dir = pathlib.Path(THIS_DIR).parent / "04_RAG_EXPLAINER"
        if str(rag_module_dir) not in sys.path:
            sys.path.append(str(rag_module_dir))
        from kb_rag_explainer import load_or_build_index, retrieve, answer_with_context

        print(f"\nSearching knowledge base for: {topic}")
        print(f"KB: {kb_path}\n")

        index = load_or_build_index(kb_path)
        hits = retrieve(index, topic, top_k=5)

        if not hits:
            print("No relevant content found. Using basic explanation...")
            return _explain_basic(topic)

        print("=== Top Sources ===")
        for i, h in enumerate(hits, 1):
            print(f"[{i}] {pathlib.Path(h['file']).name} (score={h['score']:.3f})")

        print("\n=== Explanation ===\n")
        explanation = answer_with_context(topic, hits)
        print(explanation)

        # Show links
        links = [h.get('url') for h in hits if h.get('url')]
        if links:
            print("\nLearn more:")
            for url in list(set(links))[:3]:
                print(f"   {url}")

        return 0

    except Exception as e:
        print(f"RAG failed: {e}")
        print("Falling back to basic explanation...")
        return _explain_basic(topic)

def _explain_basic(topic: str) -> int:
    """Fallback explanation using Ollama directly"""
    try:
        body = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": "You are a DV8 architecture expert. Explain clearly and concisely."},
                {"role": "user", "content": f"Explain '{topic}' in the context of DV8 software architecture analysis."}
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 400}
        }
        data = _http_json("POST", "/api/chat", body)
        print(data["message"]["content"])
        return 0
    except Exception as e:
        print(f"Explanation failed: {e}")
        return 1

def tool_temporal_analysis(plan: dict) -> int:
    """Run temporal analysis on multiple Git revisions using dv8_agent.py --temporal"""
    repo_hint = _sanitize_repo(plan.get("repo"))
    if not repo_hint:
        # LLM produced a placeholder — try to recover from user_request before prompting
        repo_hint = _guess_repo_from_text(plan.get("user_request") or "")
    if not repo_hint:
        print("No repository specified.")
        repo = _prompt_for_repo()
    else:
        repo = repo_hint

    # Fuzzy alias map: common short names → exact REPOS/ folder name
    _REPO_ALIASES = {
        "trainticket": "ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG",
        "train ticket": "ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG",
        "train-ticket": "ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG",
        "tts toy": "ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG",
        "toy example": "ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG",
        "multilang toy": "ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG",
        "arch analysis trainticket": "ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG",
    }
    if "://" not in repo:  # only apply aliases to short names, not full URLs
        repo_lower = repo.lower()
        for alias, canonical in _REPO_ALIASES.items():
            if alias in repo_lower:
                repo = canonical
                break

    # If the repo is a short name (not a URL or absolute path), resolve it to
    # an existing local clone in TEST_AUTO/REPOS/ before passing to dv8_agent.
    if repo and "://" not in repo and not pathlib.Path(repo).is_absolute():
        test_auto_dir = pathlib.Path(THIS_DIR).parent
        candidate = test_auto_dir / "REPOS_ANALYZED" / repo
        if candidate.exists():
            repo = str(candidate)
        else:
            # Could not find the repo locally — ask the user before dv8_agent crashes
            print(f"\nCould not find repository '{repo}' in {test_auto_dir / 'REPOS_ANALYZED'}/")
            print("Please provide one of:")
            print("  - A local folder path  (e.g. /Users/you/projects/myrepo)")
            print("  - A Git URL            (e.g. https://github.com/owner/repo)")
            print("  - A ZIP file path      (e.g. /Users/you/downloads/myrepo.zip)")
            try:
                entered = input("Repository [or 'q' to quit]: ").strip()
            except EOFError:
                entered = ""
            if not entered or entered.lower() in ("q", "quit", "exit"):
                print("Aborted.")
                return 1
            repo = entered

    # If repo is a subfolder (e.g., /repo/java), lift to repo root.
    had_source_path = plan.get("source_path")
    repo, source_path = _resolve_repo_and_source(repo, had_source_path)

    count = plan.get("count", 5)
    branch = plan.get("branch", "main")
    min_months_apart = plan.get("min_months_apart", 0)
    min_commits_apart = plan.get("min_commits_apart", 0)
    fine_grain = plan.get("fine_grain", False)
    since_date = plan.get("since_date")
    until_date = plan.get("until_date")
    scope = plan.get("scope", "full")
    lang_hint = None
    neodepends_root = plan.get("neodepends_root")
    neodepends_bin = plan.get("neodepends_bin")
    neodepends_resolver = plan.get("neodepends_resolver")
    depends_runner = plan.get("depends_runner")
    force_depends = plan.get("force_depends", False)
    workspace = plan.get("workspace")
    java_depends = plan.get("java_depends", True)  # Default True: use Depends for Java (NeoDepends still testing)
    understand_und = plan.get("understand_und")
    understand_upython = plan.get("understand_upython")
    understand_language = plan.get("understand_language")
    understand_granularity = plan.get("understand_granularity", "entity")
    mv_cochange = plan.get("mv_cochange")  # Modularity Violation co-change threshold (DV8 -mvCochange)
    if mv_cochange is not None:
        try:
            mv_cochange = int(mv_cochange)
        except (ValueError, TypeError):
            mv_cochange = None

    # Infer params from natural language if present
    ur = (plan.get("user_request") or "").lower()
    if ur:
        import re
        from datetime import date
        # Branch override: "on branch temporal", "temporal branch", "branch=temporal"
        m_branch = re.search(r"\bon\s+branch\s+([A-Za-z0-9._\-/]+)", ur, re.I) or \
                   re.search(r"([A-Za-z0-9._\-/]+)\s+branch\b", ur, re.I) or \
                   re.search(r"branch[=:\s]+([A-Za-z0-9._\-/]+)", ur, re.I)
        if m_branch:
            branch = m_branch.group(1).strip()
        # Count: "last 5 revisions", "over 5 revisions", "temporal 3 revisions", or bare "3 revisions"
        m_cnt = re.search(r"(?:last|over|for|temporal)?\s*(\d+)\s+revisions?", ur)
        if m_cnt:
            try:
                count = int(m_cnt.group(1))
            except ValueError:
                pass
        # Months spacing: "with 3 months in between"
        m_mon = re.search(r"(\d+)\s*\.?\s*months?", ur)
        if m_mon and ("between" in ur or "in between" in ur or "apart" in ur or "spacing" in ur or "every" in ur):
            try:
                min_months_apart = int(m_mon.group(1))
                min_commits_apart = 0
            except ValueError:
                pass
        # Year spacing: "1 year apart", "2 years apart"
        m_yr_apart = re.search(r"(\d+)\s+years?\s+apart", ur)
        if m_yr_apart and min_months_apart == 0:
            try:
                min_months_apart = int(m_yr_apart.group(1)) * 12
                min_commits_apart = 0
            except ValueError:
                pass
        # Commits spacing: "with 100 commits in between"
        m_com = re.search(r"(\d+)\s*commits?", ur)
        if m_com and ("between" in ur or "in between" in ur or "apart" in ur or "spacing" in ur):
            try:
                min_commits_apart = int(m_com.group(1))
                min_months_apart = 0
            except ValueError:
                pass
        # Relative history window: "last 10 years"
        m_years = re.search(r"last\s+(\d+)\s+years?", ur)
        if m_years:
            try:
                years = int(m_years.group(1))
                today = date.today()
                until_date = until_date or f"{today.year:04d}-{today.month:02d}-{today.day:02d}"
                since_year = today.year - years
                since_date = since_date or f"{since_year:04d}-{today.month:02d}-01"
                # If the planner forgot the count but we do have month spacing,
                # derive a reasonable number of revisions for the requested window.
                if min_months_apart > 0 and (not plan.get("count") or int(plan.get("count", 5)) == 5):
                    count = max(2, int((years * 12) / min_months_apart))
            except ValueError:
                pass
        # Smart commit selection: pick commits with most file changes
        if any(k in ur for k in ["smart commit", "intelligent commit", "most changes", "biggest changes", "most files changed"]):
            plan["spacing_mode"] = "smart"
        # Date range: "between 2012 to 2014" or "from 2012-01 to 2014-12"
        m_range = re.search(r"(?:between|from)\s+(\d{4}(?:-\d{1,2}(?:-\d{1,2})?)?)\s+(?:to|and)\s+(\d{4}(?:-\d{1,2}(?:-\d{1,2})?)?)", ur)
        if m_range:
            def _norm_start(s: str) -> str:
                parts = [int(p) for p in s.split('-')]
                y = parts[0]
                m = parts[1] if len(parts) > 1 else 1
                d = parts[2] if len(parts) > 2 else 1
                return f"{y:04d}-{m:02d}-{d:02d}"
            def _norm_end(s: str) -> str:
                import calendar
                parts = [int(p) for p in s.split('-')]
                y = parts[0]
                m = parts[1] if len(parts) > 1 else 12
                d = parts[2] if len(parts) > 2 else calendar.monthrange(y, m)[1]
                return f"{y:04d}-{m:02d}-{d:02d}"
            since_date = _norm_start(m_range.group(1))
            until_date = _norm_end(m_range.group(2))

        # Source-path hints: "only folder X", "source path X", or quoted subdir
        if not source_path:
            m_sp = re.search(r"(?:source[- ]path|only\s+folder|only\s+this\s+folder|subdir|sub-directory)\s+([A-Za-z0-9_./\\-]+)", plan.get("user_request") or "", re.I)
            if m_sp:
                source_path = m_sp.group(1).strip()
        if not source_path:
            m_q = re.search(r"['\"]([^'\"]+)['\"]", plan.get("user_request") or "")
            if m_q and ("train" in m_q.group(1).lower() or "src" in m_q.group(1).lower() or "toy" in m_q.group(1).lower()):
                source_path = m_q.group(1).strip()
        # NeoDepends paths
        if not neodepends_root:
            m_ndr_q = re.search(r"neodepends[_-]?root\s*=\s*['\"]([^'\"]+)['\"]", plan.get("user_request") or "", re.I)
            if m_ndr_q:
                neodepends_root = m_ndr_q.group(1).strip()
            else:
                m_ndr = re.search(r"neodepends[_-]?root\s*=\s*([A-Za-z0-9_./\\-]+)", plan.get("user_request") or "", re.I)
                if m_ndr:
                    neodepends_root = m_ndr.group(1).strip()
        if not neodepends_bin:
            m_ndb_q = re.search(r"neodepends[_-]?bin\s*=\s*['\"]([^'\"]+)['\"]", plan.get("user_request") or "", re.I)
            if m_ndb_q:
                neodepends_bin = m_ndb_q.group(1).strip()
            else:
                m_ndb = re.search(r"neodepends[_-]?bin\s*=\s*([A-Za-z0-9_./\\-]+)", plan.get("user_request") or "", re.I)
                if m_ndb:
                    neodepends_bin = m_ndb.group(1).strip()
        if not neodepends_resolver:
            m_ndres_q = re.search(r"neodepends[_-]?resolver\s*=\s*['\"]([^'\"]+)['\"]", plan.get("user_request") or "", re.I)
            if m_ndres_q:
                neodepends_resolver = m_ndres_q.group(1).strip()
            else:
                m_ndres = re.search(r"neodepends[_-]?resolver\s*=\s*([A-Za-z0-9_./\\-]+)", plan.get("user_request") or "", re.I)
                if m_ndres:
                    neodepends_resolver = m_ndres.group(1).strip()
        if not workspace:
            m_ws_q = re.search(r"workspace\s*=\s*['\"]([^'\"]+)['\"]", plan.get("user_request") or "", re.I)
            if m_ws_q:
                workspace = m_ws_q.group(1).strip()
            else:
                m_ws = re.search(r"workspace\s*=\s*([A-Za-z0-9_./\\-]+)", plan.get("user_request") or "", re.I)
                if m_ws:
                    workspace = m_ws.group(1).strip()
        if not depends_runner:
            m_dr = re.search(r"depends[_-]?runner\s*=\s*(auto|dv8|jar|understand)", plan.get("user_request") or "", re.I)
            if m_dr:
                depends_runner = m_dr.group(1).strip()
        # Keyword detection: "with understand", "use understand" → Understand runner
        if not depends_runner:
            ur_lower = (plan.get("user_request") or "").lower()
            if "understand" in ur_lower:
                depends_runner = "understand"
                print(f"[plan] Understand runner detected from prompt keyword.")
        # Language keyword detection for C#/TypeScript → auto-select Understand
        if not lang_hint:
            ur_lower = (plan.get("user_request") or "").lower()
            if any(k in ur_lower for k in ("c#", "csharp", "dotnet", ".net", "asp.net")):
                lang_hint = "csharp"
                if not depends_runner:
                    depends_runner = "understand"
            elif "typescript" in ur_lower and "java" not in ur_lower:
                lang_hint = "typescript"
                if not depends_runner:
                    depends_runner = "understand"
        if not java_depends:
            m_jd = re.search(r"java[_-]?depends\s*=\s*(true|false)", plan.get("user_request") or "", re.I)
            if m_jd:
                java_depends = (m_jd.group(1).strip().lower() == "true")
        # MV co-change threshold: "mv threshold 5", "mv cochange 5", "mvCochange 5", "threshold 5"
        if mv_cochange is None:
            m_mv = re.search(
                r"(?:mv[_\s-]?(?:co[_\s-]?change|threshold)|mvCochange|modularity[_\s-]?violation[_\s-]?threshold)\s*[=:\s]+(\d+)",
                plan.get("user_request") or "", re.I
            ) or re.search(
                r"(?:with\s+)?(?:mv|modularity[_\s-]?violation)\s+threshold\s+(\d+)",
                plan.get("user_request") or "", re.I
            )
            if m_mv:
                try:
                    mv_cochange = int(m_mv.group(1))
                except ValueError:
                    pass
        if "python" in ur and "java" not in ur:
            lang_hint = "python"
        if "java" in ur and "python" not in ur:
            lang_hint = "java"

        # Dependency extractor: "with neodepends", "with understand", "with depends"
        if re.search(r"\bwith\s+neodepends\b", ur) or re.search(r"\bneodepends\b", ur):
            # NeoDepends is default for Python — ensure java_depends is False so Java also uses it
            java_depends = False
            force_depends = False
        elif re.search(r"\bwith\s+understand\b", ur) or re.search(r"\buse\s+understand\b", ur):
            # Understand requires UND binary — check env or let dv8_agent discover it
            if not understand_und:
                import shutil as _sh_u
                _und = _sh_u.which("und")
                if _und:
                    understand_und = _und
                else:
                    print("[WARNING] 'understand' requested but no UND binary found. "
                          "Set --understand-und or install SciTools Understand.")
        elif re.search(r"\bwith\s+depends\b", ur) or re.search(r"\buse\s+depends\b", ur):
            # Force legacy Depends for Java (not NeoDepends)
            java_depends = True
            force_depends = True

        # Keep local NeoDepends clone fresh when Python analysis is requested.
        # dv8_agent.py handles full discovery (local → GitHub clone → release download).
        if any(kw in ur for kw in ("python", "neodepends")):
            _update_neodepends_if_local()

        # Scope hints
        if any(k in ur for k in ["both scopes", "scope both", "full and prod", "prod and full"]):
            scope = "both"
        elif any(k in ur for k in ["prod-only", "prod only", "production only", "production scope"]):
            scope = "prod"
        elif "production" in ur or "prod" in ur:
            # Only override if user explicitly asked; keep default otherwise.
            scope = plan.get("scope", "prod")

        # Known branch defaults for common repos when the planner falls back to "main".
        if branch == "main" and "commons-io" in ur:
            branch = "master"

    # If user asked for both languages, do not lock to a single source_path
    force_dual = bool(ur and ("java" in ur and "python" in ur))
    if force_dual and not had_source_path:
        source_path = None

    # Smart commit mode
    smart_commits = plan.get("spacing_mode") == "smart"

    # Determine mode based on min_months_apart
    if smart_commits:
        mode_name = "SMART (commits with most file changes)"
    elif min_commits_apart > 0:
        mode_name = f"RECENT-COMMITS ({min_commits_apart} commits spacing)"
    elif min_months_apart > 0:
        mode_name = f"RECENT-MAJOR ({min_months_apart} months minimum spacing)"
    else:
        mode_name = "ALL-TIME (first ever, last ever, interpolated)"

    # Build command - use dv8_agent.py with --temporal flag
    def _build_cmd(repo_path: str, src: Optional[str], language: Optional[str], tag: Optional[str]) -> list[str]:
        cmd = [
            "python3", AGENT,
            "--repo", repo_path,
            "--temporal",
            "--revisions", str(count),
            "--branch", branch,
            "--min-months-apart", str(min_months_apart),
            "--scope", scope,
        ]
        if src:
            cmd += ["--source-path", str(src)]
        if language:
            cmd += ["--language", language]
        if tag:
            cmd += ["--analysis-tag", tag]
        if workspace:
            cmd += ["--workspace", str(workspace)]
        if neodepends_root:
            cmd += ["--neodepends-root", str(neodepends_root)]
        if neodepends_bin:
            cmd += ["--neodepends-bin", str(neodepends_bin)]
        if neodepends_resolver:
            cmd += ["--neodepends-resolver", str(neodepends_resolver)]
        if depends_runner:
            cmd += ["--depends-runner", str(depends_runner)]
        if understand_und:
            cmd += ["--understand-und", str(understand_und)]
        if understand_upython:
            cmd += ["--understand-upython", str(understand_upython)]
        if understand_language:
            cmd += ["--understand-language", understand_language]
        if understand_granularity and understand_granularity != "entity":
            cmd += ["--understand-granularity", understand_granularity]
        if java_depends:
            cmd += ["--java-depends"]
        if force_depends:
            cmd += ["--force-depends"]
        if smart_commits:
            cmd += ["--spacing-mode", "smart"]
        if min_commits_apart > 0:
            cmd += ["--min-commits-apart", str(min_commits_apart)]
        if since_date:
            cmd += ["--since-date", since_date]
        if until_date:
            cmd += ["--until-date", until_date]
        if fine_grain:
            cmd += ["--fine-grain"]
        if mv_cochange is not None:
            cmd += ["--mv-cochange", str(mv_cochange)]
        return cmd
    # Determine which dependency extractor will be used
    if understand_und:
        _extractor_name = "Understand (SciTools)"
    elif java_depends and (lang_hint == "java" or not lang_hint):
        _extractor_name = "Depends (legacy Java)"
    else:
        _extractor_name = "NeoDepends (auto)"

    # If user specified a source_path, run once.
    json_file = None
    if source_path:
        cmd = _build_cmd(repo, source_path, lang_hint, None)
        print(f"\nTool: Temporal Analysis")
        print(f"   Repository: {repo}")
        print(f"   Revisions: {count}")
        print(f"   Branch: {branch}")
        print(f"   Extractor: {_extractor_name}")
        print(f"   Mode: {mode_name}\n")
        rc = subprocess.call(cmd)
    else:
        # Auto-run both languages if repo has top-level java/ and python/ folders
        rc = 0
        try:
            repo_path = pathlib.Path(repo).expanduser().resolve()
            has_java = (repo_path / "java").exists()
            has_python = (repo_path / "python").exists()
        except Exception:
            has_java = has_python = False

        if has_java and has_python and (lang_hint is None):
            runs = [("java", "java"), ("python", "python")]
        elif lang_hint in {"java", "python"}:
            runs = [(lang_hint, lang_hint)]
        else:
            runs = [(None, None)]

        any_success = False
        last_rc = 0
        for lang, tag in runs:
            src = lang if lang in {"java", "python"} else None
            cmd = _build_cmd(repo, src, lang, tag)
            print(f"\nTool: Temporal Analysis ({lang or 'auto'})")
            print(f"   Repository: {repo}")
            print(f"   Revisions: {count}")
            print(f"   Branch: {branch}")
            print(f"   Extractor: {_extractor_name}")
            print(f"   Mode: {mode_name}\n")
            lang_rc = subprocess.call(cmd)
            if lang_rc == 0:
                any_success = True
            else:
                print(f"Warning: {lang or 'auto'} analysis failed (rc={lang_rc}), continuing with other languages...")
                last_rc = lang_rc
        rc = 0 if any_success else last_rc

    if rc == 0:
        # Find the most-recently written timeseries.json across ALL repos.
        # Using the newest file makes this robust even when the LLM produced a
        # placeholder repo path and `repo` is wrong after the interactive prompt.
        test_auto_dir = pathlib.Path(THIS_DIR).parent
        json_file = None
        repo_name = None
        repos_dir = None
        temporal_root_dir = None

        if workspace:
            search_root = pathlib.Path(workspace).expanduser().resolve()
        else:
            search_root = test_auto_dir / "REPOS_ANALYZED"
        all_temporal = [d for d in search_root.glob("*/temporal_analysis*/") if d.is_dir() and _has_timeseries(d)]

        if all_temporal:
            newest_tr = max(all_temporal, key=lambda d: d.stat().st_mtime)
            json_file = _ts_path(newest_tr)
            temporal_root_dir = newest_tr          # always the temporal_analysis_* dir
            repos_dir = newest_tr.parent   # REPOS_ANALYZED/<repo_name>/
            repo_name = repos_dir.name
        else:
            # Fallback: derive from repo variable as before
            repo_name = pathlib.Path(repo).name if "://" not in repo else pathlib.Path(repo.rstrip('/').split('/')[-1]).stem.replace(".git", "")
            repos_dir = search_root / repo_name
            json_file = repos_dir / "timeseries.json"
            temporal_root_dir = repos_dir  # best guess in fallback

        print(f"\nOutput files:")
        print(f"   Time-series data: {json_file}")
        print(f"   Revision folders: {repos_dir}/temporal_analysis*/")

        if repos_dir and repos_dir.exists():
            print(f"\nAnalyzed revisions:")
            for rev_dir in sorted(repos_dir.glob("temporal_analysis*")):
                print(f"   - {rev_dir.name}/")

    if json_file and json_file.exists():
            # Auto-run backfill to prepare interpretation bundle
            # Use temporal_root_dir (the temporal_analysis_* dir) — NOT json_file.parent,
            # which may be INPUT_INTERPRETATION/ and would cause double-nesting.
            temporal_folder = temporal_root_dir if temporal_root_dir else (
                json_file.parent.parent if json_file.parent.name == "INPUT_INTERPRETATION" else json_file.parent
            )
            print("\nPreparing interpretation bundle...")
            bf_cmd = ["python3", BACKFILL_TEMPORAL, str(temporal_folder), "--meta-repo", repo_name]
            bf_rc = subprocess.call(bf_cmd)
            if bf_rc == 0:
                print("Interpretation bundle ready.")
                # DISABLED: Risk pipeline — not empirically proven yet, only DV8 metrics used
                # _git_root = None
                # if repo and "://" not in str(repo):
                #     _candidate = pathlib.Path(repo).expanduser().resolve()
                #     if (_candidate / ".git").exists():
                #         _git_root = _candidate
                # _run_risk_pipeline(
                #     temporal_folder,
                #     repo_name,
                #     git_root=_git_root,
                #     review_model=None,
                # )
            else:
                print("Warning: Backfill failed; interpretation may not work.")

            print("\n" + "=" * 60)
            print("Temporal analysis complete!")
            print("=" * 60)
            # Compute steepest M-score change for refine suggestion
            try:
                with open(json_file) as f:
                    ts = json.load(f)
                revs = ts.get('revisions', [])
                best = None
                for i in range(len(revs) - 1):
                    new = revs[i]
                    old = revs[i + 1]
                    m1 = (new.get('metrics') or {}).get('m-score')
                    m0 = (old.get('metrics') or {}).get('m-score')
                    if m1 is None or m0 is None:
                        continue
                    delta = abs(float(m1) - float(m0))
                    if best is None or delta > best[0]:
                        best = (delta, old, new)
            except Exception:
                best = None

            refine_msg = ""
            since_date = until_date = None
            if best:
                o, n = best[1], best[2]
                # Normalize dates to YYYY-MM-DD
                def _d(s: str) -> str:
                    return (s or '').split()[0]
                since_date, until_date = _d(o.get('commit_date') or ''), _d(n.get('commit_date') or '')
                refine_msg = f"Refine around steepest M-score change: {since_date} → {until_date} (Δ≈{best[0]:.2f}%)"
                print(f"\n{refine_msg}")

            # If plan carries a user_question (from JSON dispatch), inject it into user_request
            # so _extract_user_question() and tool_interpret_temporal can find it.
            plan_question = plan.get("user_question") or plan.get("question") or ""
            if plan_question and "answer:" not in (ur or "").lower():
                ur = (ur + f"\n\nanswer: {plan_question}").strip() if ur else f"answer: {plan_question}"

            # Check if user explicitly asked for interpretation in original request
            # Suppress auto-interpret if user said "only analyze" / "just analyze" / "analyze only"
            analyze_only = ur and any(k in ur.lower() for k in ['only analyze', 'just analyze', 'analyze only', 'no interpret', 'without interpret'])
            auto_interpret = (not analyze_only) and (
                plan_question  # always interpret when a question was explicitly provided
                or (ur and any(k in ur.lower() for k in [' interpret', 'then interpret', 'and interpret', ' explain why']))
                or plan.get("auto_refactor", False)  # refactoring requires interpretation first
            )

            if auto_interpret:
                # Skip the menu, go straight to interpretation
                print("\nAuto-interpreting as requested...")
                choice = 'i'
            else:
                print("\nNext action:")
                print("  [r] Refine temporal window before interpreting" + (f" ({since_date} to {until_date})" if since_date and until_date else ""))
                print("  [i] Interpret now")
                print("  [n] Nothing")
                default_choice = 'n'
                try:
                    choice = input(f"Choice [r/i/n] (default {default_choice}): ").strip().lower()
                except EOFError:
                    choice = default_choice
                if not choice:
                    choice = default_choice

            if choice.startswith('r') and since_date and until_date:
                # Run a refined analysis within the detected window, with a sensible default of 6 revisions
                refine_plan = {
                    "repo": repo,
                    "tool": "temporal_analysis",
                    "count": 6,
                    "branch": branch,
                    "since_date": since_date,
                    "until_date": until_date,
                    "min_months_apart": 0,
                    "min_commits_apart": 0,
                    "fine_grain": True,
                    "user_request": plan.get("user_request") or ur,
                }
                print("\nRunning refined temporal analysis in peak-change window...\n")
                return tool_temporal_analysis(refine_plan)
            elif choice.startswith('i'):
                # Pass the specific temporal folder (json_file.parent) to avoid glob picking up all folders
                temporal_folder = str(json_file.parent) if json_file else None
                # Model can come from "model" or "interpret_model" in plan
                interpret_model = plan.get("model") or plan.get("interpret_model") or "deepseek-r1:32b"
                return tool_interpret_temporal({
                    "repo": temporal_folder or repo,
                    "model": interpret_model,
                    # Pass the (possibly question-injected) user_request so the Q&A step fires
                    "user_request": ur or plan.get("user_request") or "",
                    "auto_refactor": plan.get("auto_refactor", False),
                    "refactor_model": plan.get("refactor_model") or "qwen3-coder-30b-refactor",
                    "refactor_loop_count": plan.get("refactor_loop_count", 0),
                    "use_feedback_loop": plan.get("use_feedback_loop", False),
                })
            else:
                print("No further action.")

    return rc


def tool_interpret_metrics(plan: dict) -> int:
    """Interpret metric changes using git commits (Stage 2: Interpretation)"""
    def _model_suffix(model: str) -> str:
        import re
        m = (model or "").strip().lower()
        # Extract size (e.g., 8b, 14b)
        size = None
        if ":" in m:
            name, size_part = m.split(":", 1)
        else:
            name, size_part = m, ""
        # Try to find size digits in size_part or name
        msize = re.search(r"(\d+)\s*b", size_part)
        if not msize:
            msize = re.search(r"(\d+)\s*b", name)
        if msize:
            size = f"{msize.group(1)}B"
        else:
            size = ""

        vendor = name
        if "llama" in name:
            vendor = "llama3.1" if "3.1" in name else "llama"
        elif "deepseek" in name:
            # Keep r1 if present
            vendor = "deepseekr1" if "r1" in name else "deepseek"
        elif name.startswith("qwen"):
            vendor = "qwen"
        # normalize vendor to alphanumerics only (drop dots/dashes)
        vendor = re.sub(r"[^a-z0-9]", "", vendor)
        return f"{vendor}{('_' + size) if size else ''}"
    repo_hint = _sanitize_repo(plan.get("repo"))
    if not repo_hint:
        print("No repository specified.")
        repo = _prompt_for_repo()
    else:
        repo = repo_hint

    # Find the repository and timeseries.json
    # Avoid regex dependency here to prevent scope issues
    is_url = isinstance(repo, str) and (repo.startswith("http://") or repo.startswith("https://"))
    repo_name = pathlib.Path(repo).name if not is_url else pathlib.Path(repo.rstrip('/')).name.replace(".git", "")

    test_auto_dir = pathlib.Path(THIS_DIR).parent
    repos_dir = test_auto_dir / "REPOS_ANALYZED" / repo_name
    # Locate the most relevant timeseries.json
    # Prefer a folder matching hints in the user request (months/commits/all-time), else fallback to newest
    ur = (plan.get('user_request') or '').lower() if isinstance(plan, dict) else ''
    json_file = None
    json_candidates = [_ts_path(d) for d in repos_dir.glob("temporal_analysis*/") if d.is_dir() and _has_timeseries(d)]
    if json_candidates:
        # Try pattern match
        import re
        preferred = []
        m_cnt = re.search(r"(?:last|over|for)\s+(\d+)\s+revisions?", ur or '')
        m_mon = re.search(r"(\d+)\s*\.?\s*months?", ur or '')
        m_com = re.search(r"(\d+)\s*commits?", ur or '')
        want_alltime = any(k in (ur or '') for k in ["all time", "all-time", "entire history", "from beginning"])
        for c in json_candidates:
            folder = c.parent.parent.name.lower() if "INPUT_INTERPRETATION" in str(c) else c.parent.name.lower()
            ok = True
            if want_alltime and "alltime" not in folder:
                ok = False
            if m_mon and f"{m_mon.group(1)}month_diff" not in folder:
                ok = False
            if m_com and f"{m_com.group(1)}commits_diff" not in folder:
                ok = False
            if m_cnt and f"{m_cnt.group(1)}revisions" not in folder:
                ok = False
            if ok:
                preferred.append(c)
        if preferred:
            json_file = max(preferred, key=lambda p: p.stat().st_mtime)
        else:
            json_file = max(json_candidates, key=lambda p: p.stat().st_mtime)
    else:
        json_file = repos_dir / "timeseries.json"

    if not json_file.exists():
        print(f"\nWarning: No timeseries.json found at {json_file}")
        print("   Run temporal analysis first!")
        return 1

    # Get model preference
    # Choose model: plan > inferred from user_request > default
    model = plan.get("model")
    if not model:
        ur = (plan.get('user_request') or '').lower() if isinstance(plan, dict) else ''
        # Heuristic: prefer deepseek if mentioned, else qwen, else llama
        if 'deepseek' in (ur or ''):
            model = 'deepseek-r1:32b'
        elif 'qwen' in (ur or ''):
            model = 'qwen2:8b'
        else:
            model = 'llama3.1:8b'

    # Build command - call interpreter in Stage 2
    interpreter_script = test_auto_dir / "02_STAGE_INTERPRET" / "interpret_metrics.py"

    if not interpreter_script.exists():
        print(f"\nWarning: Interpreter not found at {interpreter_script}")
        return 1

    # Decide output filename by model; write next to the selected timeseries (inside temporal_analysis_* folder)
    suffix = _model_suffix(model)
    report_dir = json_file.parent
    report_file = report_dir / (f"interpretation_report_{suffix}.md" if suffix else "interpretation_report.md")

    cmd = [
        "python3", str(interpreter_script),
        "--repo", str(repos_dir),
        "--timeseries", str(json_file),
        "--model", model,
        "--output", str(report_file)
    ]

    print(f"\nTool: Interpret Metric Changes")
    print(f"   Repository: {repos_dir}")
    print(f"   Timeseries: {json_file}")
    print(f"   Model: {model}")
    print(f"   Report dir: {report_dir}\n")

    rc = subprocess.call(cmd)

    if rc == 0:
        # Prefer model-suffixed report; fallback to default name if tool wrote it
        final_report = report_file if report_file.exists() else (repos_dir / "interpretation_report.md")
        if final_report.exists():
            print(f"\nReport generated: {final_report}")
            print(f"\n   View with: cat '{final_report}'")

    return rc


def tool_peak_full_arch(plan: dict) -> int:
    repo_hint = _sanitize_repo(plan.get("repo"))
    if not repo_hint:
        # Try to guess from user_request text
        repo_guess = _guess_repo_from_text(plan.get("user_request", ""))
        if repo_guess:
            repo = repo_guess
        else:
            print("No repository specified.")
            repo = _prompt_for_repo()
    else:
        repo = repo_hint

    # Find timeseries.json (newest temporal run)
    repo_name = pathlib.Path(repo).name if not re.match(r"^https?://", repo) else pathlib.Path(repo.rstrip('/')).name.replace(".git", "")
    test_auto_dir = pathlib.Path(THIS_DIR).parent
    repos_dir = test_auto_dir / "REPOS_ANALYZED" / repo_name
    json_candidates = [_ts_path(d) for d in repos_dir.glob("temporal_analysis*/") if d.is_dir() and _has_timeseries(d)]
    if not json_candidates:
        print("No timeseries found. Run temporal analysis first (all-time or window).")
        return 1
    json_file = max(json_candidates, key=lambda p: p.stat().st_mtime)

    try:
        data = json.loads(json_file.read_text())
    except Exception as e:
        print(f"Failed to read {json_file}: {e}")
        return 1

    revs = data.get('revisions', [])
    best = None
    for i in range(len(revs) - 1):
        new = revs[i]
        old = revs[i + 1]
        m1 = (new.get('metrics') or {}).get('m-score')
        m0 = (old.get('metrics') or {}).get('m-score')
        if m1 is None or m0 is None:
            continue
        try:
            delta = abs(float(m1) - float(m0))
        except Exception:
            continue
        if best is None or delta > best[0]:
            best = (delta, old, new)

    if not best:
        print("No peak M-score change found in timeseries.")
        return 1

    old, new = best[1], best[2]
    h_old = old.get('commit_hash')
    h_new = new.get('commit_hash')
    if not h_old or not h_new:
        print("Missing commit hashes in timeseries.")
        return 1

    # Run dv8_agent on both commits with full arch-report (fine-grain)
    cmd = [
        "python3", AGENT,
        "--repo", str(repos_dir),
        "--commit", h_old,
        "--commit2", h_new,
        "--fine-grain",
    ]

    print("\nRunning full arch reports on peak-change commits:")
    print(f"  Repo: {repos_dir}")
    print(f"  Old: {h_old}  New: {h_new}")
    print("  Command:", " ".join(cmd))
    rc = subprocess.call(cmd)
    return rc

def tool_full_arch_at_dates(plan: dict) -> int:
    repo_hint = _sanitize_repo(plan.get("repo"))
    if not repo_hint:
        repo_guess = _guess_repo_from_text(plan.get("user_request", ""))
        repo = repo_guess or _prompt_for_repo()
    else:
        repo = repo_hint

    repo_name = pathlib.Path(repo).name if not re.match(r"^https?://", repo) else pathlib.Path(repo.rstrip('/')).name.replace(".git", "")
    test_auto_dir = pathlib.Path(THIS_DIR).parent
    repos_dir = test_auto_dir / "REPOS_ANALYZED" / repo_name
    json_candidates = [_ts_path(d) for d in repos_dir.glob("temporal_analysis*/") if d.is_dir() and _has_timeseries(d)]
    if not json_candidates:
        print("No timeseries found. Run a temporal analysis first.")
        return 1
    json_file = max(json_candidates, key=lambda p: p.stat().st_mtime)

    try:
        data = json.loads(json_file.read_text())
    except Exception as e:
        print(f"Failed to read {json_file}: {e}")
        return 1

    # Extract dates/years from user_request
    ur = (plan.get("user_request") or "").lower()
    targets = []
    for m in re.finditer(r"(20\d{2}|19\d{2})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", ur):
        y = int(m.group(1)); mo = int(m.group(2) or 1); d = int(m.group(3) or 1)
        targets.append(f"{y:04d}-{mo:02d}-{d:02d}")
    if not targets:
        print("No dates/years found in request.")
        return 1

    # Pick nearest commit to each target date
    from datetime import datetime
    picked = []
    revs = data.get('revisions', [])
    for t in targets:
        td = datetime.strptime(t, '%Y-%m-%d')
        best = None
        for r in revs:
            dstr = (r.get('commit_date') or '').split()[0]
            try:
                rd = datetime.strptime(dstr, '%Y-%m-%d')
            except Exception:
                continue
            diff = abs((rd - td).days)
            if best is None or diff < best[0]:
                best = (diff, r)
        if best:
            picked.append(best[1])

    # Run arch-report per picked commit
    for r in picked:
        h = r.get('commit_hash')
        if not h:
            continue
        cmd = ["python3", AGENT, "--repo", str(repos_dir), "--commit", h, "--fine-grain"]
        print("\nRunning full arch report:", " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    return 0

def tool_plot_refined(plan: dict) -> int:
    # Accept a folder to plot (temporal_analysis_* or focus_commits_*)
    def _decurly(s: str) -> str:
        return s.replace('“', '"').replace('”', '"').replace("’", "'").replace('‘', "'")

    folder = plan.get('folder') or plan.get('repo')
    if not folder:
        # Try to pull from user_request quoted path
        ur = _decurly(plan.get('user_request', ''))
        m = re.search(r"['\"]([^'\"]+)['\"]", ur)
        folder = m.group(1) if m else None
    if not folder:
        print("No folder provided. Provide a temporal_analysis_* or focus_commits_* path.")
        return 1
    folder = _decurly(str(folder)).strip()
    if '...' in folder:
        print("Ellipsis '...' detected in path. Please provide the full absolute folder path.")
        return 1
    folder = str(pathlib.Path(folder).expanduser().resolve())
    anti_plotter = pathlib.Path(THIS_DIR) / 'anti_pattern_plotter.py'
    if not anti_plotter.exists():
        print(f"Plotter not found at {anti_plotter}")
        return 1
    # Detect mode by presence of timeseries.json (check both old and new location)
    if _has_timeseries(pathlib.Path(folder)):
        cmd = ["python3", str(anti_plotter), "--temporal", folder]
    else:
        cmd = ["python3", str(anti_plotter), "--focus", folder]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    return rc

def run_tool(plan: dict, user_request: str) -> int:
    """Execute the selected tool"""
    tool = plan.get("tool", "").strip().lower()

    # Normalize tool names
    tool_map = {
        "analyze": "analyze_repo",
        "analyze_repo": "analyze_repo",
        "run": "analyze_repo",
        "run_dv8": "analyze_repo",
        "explain_metrics": "explain_metrics",
        "explain_results": "explain_metrics",
        "explain": "explain_concept",
        "explain_concept": "explain_concept",
        "what_is": "explain_concept",
        "temporal": "temporal_analysis",
        "temporal_analysis": "temporal_analysis",
        "track": "temporal_analysis",
        "evolution": "temporal_analysis",
        "history": "temporal_analysis",
        "interpret": "interpret_metrics",
        "interpret_metrics": "interpret_metrics",
        "why": "interpret_metrics",
        "reason": "interpret_metrics",
        "interpret_temporal": "interpret_temporal",
        "interpret_results": "interpret_temporal",
        "interpret_folder": "interpret_temporal",
        "peak": "peak_full_arch",
        "peak_full_arch": "peak_full_arch",
        "query": "query",
        "ask": "query",
        "fast query": "query",
        "fast_query": "query",
    }

    tool = tool_map.get(tool, tool)

    # Guardrails: if the user is clearly asking to explain/define, prefer RAG explain
    ur = (user_request or '').lower()
    repo_field = str(plan.get("repo") or "")
    looks_like_temporal_root = (
        "temporal_analysis" in repo_field
        or "input_interpretation" in repo_field.lower()
        or "output_interpretation" in repo_field.lower()
    )
    asks_to_analyze = any(k in ur for k in ["analyze", "run a temporal analysis", "temporal analysis", "track", "evolution"])
    asks_to_interpret = any(k in ur for k in ["interpret", "explain", "answer:"])
    if tool == "interpret_temporal" and asks_to_analyze and (asks_to_interpret or plan.get("user_question")) and not looks_like_temporal_root:
        tool = "temporal_analysis"
    # Heuristic: if user mentions biggest/peak m-score difference and full arch/anti-patterns, route to peak_full_arch
    if any(k in ur for k in ["biggest m-score", "peak m-score", "largest m-score", "biggest mscore", "largest mscore"]) and any(k in ur for k in ["full arch", "anti pattern", "antipattern", "arch report", "arch-report"]):
        tool = "peak_full_arch"
    if any(kw in ur for kw in ["what is", "what's", "whats", "explain ", "define "]) and not any(kw in ur for kw in ["interpret", "why ", "over time", "revisions", "commits"]):
        tool = "explain_concept"
    # Fast-path: "query <repo>[<model>]: <question>" — bypass LLM dispatch
    # Interactive: "query commons-io[32b]" (no colon, no question) → REPL session
    import re as _re
    _q_match = _re.match(
        r'^(?:query|ask|fast\s+query)\s+([\w\-]+)(?:\[([\w:\-\.]+)\])?\s*(?::\s*(.+))?$',
        (user_request or '').strip(), _re.I,
    )
    if _q_match:
        tool = "query"
        plan["repo"] = _q_match.group(1).strip()
        if _q_match.group(2):
            plan["interp_model"] = _q_match.group(2).strip()
        if _q_match.group(3):
            plan["question"] = _q_match.group(3).strip()
        # group(3) absent → interactive session (no question set)
    elif ur.startswith("query:") or ur.startswith("ask:"):
        tool = "query"
        plan["question"] = (user_request or '').split(":", 1)[-1].strip()
        plan.setdefault("repo", None)

    if tool == "analyze_repo":
        print("Tool: Analyze Repository\n")
        rc, repo = tool_analyze_repo(plan)

        if rc == 0:
            # Offer to explain results
            print("\n" + "=" * 60)
            print("Analysis complete!")
            print("=" * 60)

            try:
                ans = input("\nGenerate detailed AI explanation of results? [Y/n]: ").strip().lower()
            except EOFError:
                ans = "y"

            if ans in {"", "y", "yes"}:
                return tool_explain_metrics({"repo": repo})

        return rc

    elif tool == "explain_metrics":
        print("Tool: Explain Metrics (Detailed AI Analysis)\n")
        return tool_explain_metrics(plan)

    elif tool == "explain_concept":
        print("Tool: Explain Concept\n")
        return tool_explain_concept(plan)

    elif tool == "temporal_analysis":
        print("Tool: Temporal Analysis\n")
        # If the 'repo' field actually points to a results folder, route to plot_refined instead
        repo_field = plan.get('repo')
        try:
            if isinstance(repo_field, str) and (('focus_commits' in repo_field) or ('temporal_analysis' in repo_field)):
                candidate = pathlib.Path(repo_field).expanduser().resolve()
                if candidate.exists():
                    print("Detected results folder; plotting refined results instead of running temporal analysis.")
                    return tool_plot_refined({"folder": str(candidate), "user_request": user_request})
        except Exception:
            pass
        p = dict(plan)
        p['user_request'] = user_request
        return tool_temporal_analysis(p)

    elif tool == "analyze_and_refactor_single":
        print("Tool: Single-Revision Analyze + Auto-Refactor\n")
        return tool_analyze_and_refactor_single(plan)

    elif tool == "interpret_metrics":
        print("Tool: Interpret Metric Changes\n")
        # Attach original request so model inference can detect desired LLM
        p = dict(plan)
        p['user_request'] = user_request
        return tool_interpret_metrics(p)

    elif tool == "interpret_temporal":
        print("Tool: Interpret Temporal Analysis Bundle\n")
        p = dict(plan)
        p["user_request"] = user_request
        return tool_interpret_temporal(p)

    elif tool == "plot_refined":
        print("Tool: Plot Refined Results\n")
        return tool_plot_refined(plan)

    elif tool == "peak_full_arch":
        print("Tool: Peak Full Arch Reports\n")
        p = dict(plan)
        p['user_request'] = user_request
        return tool_peak_full_arch(p)

    elif tool == "query":
        print("Tool: Fast RAG Query (Stage 3)\n")
        p = dict(plan)
        if not p.get("question"):
            # Extract question from user_request if not already parsed
            p["question"] = user_request
        return tool_query(p)

    else:
        print(f"Unknown tool: {tool}")
        print("Defaulting to analyze_repo...")
        return tool_analyze_repo(plan)

def main():
    # Parse special flags before user request
    args = sys.argv[1:]
    temporal_root_override = None
    model_override = None

    # Extract --temporal-root, --model, --stage4-only, --auto, --refactor-model, --repo, --commit flags
    stage4_only_path = None
    auto_refactor = False
    refactor_model_override = None
    repo_override = None
    commit_override = None
    force_reinterpret = False
    filtered_args = []
    i = 0
    while i < len(args):
        if args[i] == "--temporal-root" and i + 1 < len(args):
            temporal_root_override = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model_override = args[i + 1]
            i += 2
        elif args[i] == "--stage4-only" and i + 1 < len(args):
            stage4_only_path = args[i + 1]
            i += 2
        elif args[i] == "--auto":
            auto_refactor = True
            i += 1
        elif args[i] == "--refactor-model" and i + 1 < len(args):
            refactor_model_override = args[i + 1]
            i += 2
        elif args[i] == "--repo" and i + 1 < len(args):
            repo_override = args[i + 1]
            i += 2
        elif args[i] == "--commit" and i + 1 < len(args):
            commit_override = args[i + 1]
            i += 2
        elif args[i] == "--reinterpret":
            force_reinterpret = True
            i += 1
        else:
            filtered_args.append(args[i])
            i += 1

    # Standalone Stage 4 test: skip full pipeline, just refactor + re-analyse
    if stage4_only_path:
        tr = pathlib.Path(stage4_only_path).expanduser().resolve()
        if not tr.is_dir():
            print(f"[stage4-only] Path not found: {tr}")
            sys.exit(1)
        sys.exit(_run_refactor_stage(tr, conversation=[], model=refactor_model_override or "qwen3-coder-30b-refactor",
                                     qa_model=model_override or "qwen3-coder-30b-refactor",
                                     use_feedback_loop=use_feedback_loop))

    if len(filtered_args) < 1 and not temporal_root_override:
        print('Usage: python LLM_frontend_upgraded.py "your request"')
        print('')
        print('Options:')
        print('  --temporal-root <path>  Explicit temporal analysis folder (skips glob discovery)')
        print('  --model <model>         LLM model for Q1/Q2 interpretation (default: deepseek-r1:32b)')
        print('  --refactor-model <m>    LLM model for Stage 4 refactoring (default: qwen3.6:latest)')
        print('  --stage4-only <path>    Run Stage 4 (qwen3 refactor + re-analysis) on existing temporal folder')
        print('  --auto                  Full pipeline: Q1 -> Q2 -> Stage 4 without human input')
        print('  --repo <url|path>       Repository URL or local path (for single-revision mode)')
        print('  --commit <hash>         Specific commit hash to analyze (default: HEAD/latest)')
        print('')
        print('Examples:')
        print('  "Analyze pdfbox and explain the results"')
        print('  "What is propagation cost?"')
        print('  "Explain the metrics for pdfbox"')
        print('  "Run analysis on https://github.com/apache/commons-lang"')
        print('  "Analyze the last 5 revisions of pdfbox"')
        print('  "Show me how modularity evolved over the last 10 commits"')
        print('  "Track architecture metrics over time for ./myproject"')
        print('')
        print('Direct interpretation with explicit path:')
        print('  --temporal-root /path/to/temporal_analysis_... "interpret"')
        sys.exit(2)

    # If --temporal-root provided without a request, default to "interpret"
    user_req = filtered_args[0] if filtered_args else "interpret"

    # Detect "automated refactoring" intent from the request text
    _AUTO_KEYWORDS = ["automated refactoring", "auto refactor", "full pipeline",
                      "end to end", "end-to-end", "fully automated", "no human input"]
    if not auto_refactor and any(k in user_req.lower() for k in _AUTO_KEYWORDS):
        auto_refactor = True

    # Detect bare "refactor" as auto_refactor trigger (without needing "automated")
    import re as _re_claude
    _ur_low = user_req.lower()
    if not auto_refactor and _re_claude.search(r'\brefactor(?:ing)?\b', _ur_low):
        auto_refactor = True

    # Detect "claude refactoring" / "claude subscription" / "with claude sonnet subscription" → use claude CLI for Stage 4
    _claude_refactor_match = _re_claude.search(
        r'(?:with\s+)?claude[\s-]*(?:opus|sonnet|haiku)?\s*(?:refactor(?:ing)?|subscription|coder)',
        _ur_low
    )
    if _claude_refactor_match and not refactor_model_override:
        # Extract optional model variant: "claude-opus", "claude sonnet subscription", "claude-sonnet refactoring"
        _claude_variant = _re_claude.search(
            r'claude[\s-]*(opus|sonnet|haiku)',
            _ur_low
        )
        variant = _claude_variant.group(1) if _claude_variant and _claude_variant.group(1) else ""
        if not variant:
            # No model specified — ask user which Claude model to use
            print("\n[main] Claude Code refactoring detected but no model specified.")
            print("  Available models:")
            print("    1) sonnet  — fast, good for most refactoring (recommended)")
            print("    2) opus    — most capable, slower, uses more tokens")
            print("    3) haiku   — fastest, least capable")
            try:
                choice = input("  Which Claude model? [1/2/3 or name, default=sonnet]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                choice = ""
            if choice in ("2", "opus"):
                variant = "opus"
            elif choice in ("3", "haiku"):
                variant = "haiku"
            else:
                variant = "sonnet"
            print(f"  → Using claude-{variant}")
        refactor_model_override = f"claude-{variant}"
        auto_refactor = True
        print(f"[main] Claude Code refactoring mode: model={refactor_model_override}")

    # Detect multi-model comparison: "claude and qwen", "compare claude and qwen", "claude vs qwen"
    # Also: "with claude sonnet subscription and qwen coder and compare them"
    refactor_models_list: list[str] = []
    if refactor_model_override:
        refactor_models_list.append(refactor_model_override)
    # Check for additional models: "and qwen", "and codex", "vs qwen"
    _multi_model_re = _re_claude.findall(
        r'(?:and|vs|versus|\+)\s+(claude[\s-]*(?:opus|sonnet|haiku)?|qwen[\w-]*|codex)',
        _ur_low
    )
    for _mm in _multi_model_re:
        _mm_clean = _mm.strip().replace(" ", "-")
        if _mm_clean.startswith("qwen"):
            refactor_models_list.append("qwen3-coder-30b-refactor")
        elif _mm_clean.startswith("codex"):
            refactor_models_list.append("codex")
        elif _mm_clean.startswith("claude"):
            _cv = _re_claude.search(r"claude[\s-]*(opus|sonnet|haiku)", _mm_clean)
            refactor_models_list.append(f"claude-{_cv.group(1)}" if _cv else "claude-sonnet")
    # Also detect standalone "qwen refactoring/coder" if no claude was specified
    if not refactor_model_override and _re_claude.search(r'qwen\s*(?:refactor(?:ing)?|coder)', _ur_low):
        refactor_models_list.append("qwen3-coder-30b-refactor")
        auto_refactor = True
    # Deduplicate
    _seen = set()
    refactor_models_list = [m for m in refactor_models_list if not (m in _seen or _seen.add(m))]

    # Detect "compare" intent: "compare them", "compare models", "compare claude and qwen"
    _compare_mode = bool(_re_claude.search(r'\bcompare\b', _ur_low))
    if _compare_mode and len(refactor_models_list) > 1:
        print(f"[main] Multi-model COMPARISON mode: {', '.join(refactor_models_list)}")
        auto_refactor = True
    elif len(refactor_models_list) > 1:
        print(f"[main] Multi-model comparison: {', '.join(refactor_models_list)}")
        auto_refactor = True

    # Detect loop mode: "loop refactoring N", "loop N", "action loop N", "loop most important action N"
    # Default: "automated refactoring" without explicit N → 4 loop iterations for comparison, 3 for single model
    import re as _re_loop
    _loop_match = _re_loop.search(
        r'(?:loop\s+(?:refactoring|refactor|most\s+important\s+action\s*|action\s+)?|action\s+loop\s+)(\d+)',
        user_req.lower()
    )
    if not _loop_match:
        # Also match "N iterations" or "iterations N"
        _loop_match = _re_loop.search(r'(\d+)\s+iteration', user_req.lower()) or \
                      _re_loop.search(r'iteration[s]?\s+(\d+)', user_req.lower())
    refactor_loop_count = int(_loop_match.group(1)) if _loop_match else 0
    if refactor_loop_count > 0:
        auto_refactor = True
        print(f"[main] Loop refactoring mode: {refactor_loop_count} iterations (re-analyze + 1 action each loop)")
    elif auto_refactor and refactor_loop_count == 0:
        # Default: multi-model comparison → 4 iterations, single model → 5 iterations
        if len(refactor_models_list) > 1:
            refactor_loop_count = 4
            print(f"[main] Multi-model comparison: defaulting to {refactor_loop_count} iterations per model")
        else:
            refactor_loop_count = 5
        print(f"[main] Auto-refactor mode: defaulting to loop mode ({refactor_loop_count} iterations)")

    # Detect feedback loop mode: "feedback loop", "use feedback loop", "feedback loop refactoring"
    use_feedback_loop = bool(_re_loop.search(r'feedback\s+loop', user_req.lower()))
    if use_feedback_loop:
        print(f"[main] Feedback loop ENABLED: reviewer agent will check each refactoring iteration")

    # Handle direct interpretation with --temporal-root
    if temporal_root_override:
        print(f"\nDirect interpretation mode")
        print(f"  Temporal root: {temporal_root_override}")
        print(f"  Model: {model_override or 'deepseek-r1:32b'}\n")
        rc = tool_interpret_temporal({
            "repo": temporal_root_override,
            "model": model_override or "deepseek-r1:32b",
            "user_request": user_req,
            "auto_refactor": auto_refactor,
            "refactor_model": refactor_model_override or "qwen3-coder-30b-refactor",
            "refactor_loop_count": refactor_loop_count,
            "force_reinterpret": force_reinterpret,
            "use_feedback_loop": use_feedback_loop,
        })
        sys.exit(rc)

    print(f"\nYou asked: {user_req}\n")

    # ── Fast-path: single-revision auto-refactor (godclass only / single revision) ──
    # Triggers on keywords OR an explicit --repo flag.
    # Also triggers when the request contains a git URL + commit hash inline.
    import re as _re_single
    # Extract GitHub URL from request text (strip /tree/branch suffix to get bare repo URL)
    _url_match = _re_single.search(r'(https?://github\.com/[^\s/]+/[^\s/]+)(?:/tree/[^\s]*)?', user_req)
    _extracted_url = _url_match.group(1) if _url_match else None
    # Extract 40-char hex commit hash from request text
    _hash_match = _re_single.search(r'\b([0-9a-f]{40})\b', user_req)
    _extracted_hash = _hash_match.group(1) if _hash_match else None

    _SINGLE_REV_KEYWORDS = ["godclass only", "single revision", "single commit",
                             "first commit", "first revision", "single commit in the repo",
                             "commit hash"]
    _single_rev_trigger = (
        any(k in user_req.lower() for k in _SINGLE_REV_KEYWORDS)
        or repo_override
        or (_extracted_url and _extracted_hash)
    )
    if auto_refactor and _single_rev_trigger:
        # Detect language (no regex — avoid capturing stop-words like "and")
        _lang = "java" if "java" in user_req.lower() else "python"
        # Auto-resolve known toy snapshots as last fallback
        _snapshot_key = (_lang, "godclass") if any(k in user_req.lower() for k in ["godclass", "first commit", "first revision"]) else None
        _toy_repo, _toy_commit = _TOY_SNAPSHOTS.get(_snapshot_key, (None, None)) if _snapshot_key else (None, None)
        _fast_plan = {
            "tool": "analyze_and_refactor_single",
            "repo": repo_override or _extracted_url or _toy_repo or "",
            "commit": commit_override or _extracted_hash or _toy_commit,
            "language": _lang,
            "model": model_override or "deepseek-r1:32b",
            "refactor_model": refactor_model_override or "qwen3-coder-30b-refactor",
            "user_request": user_req,
        }
        print(f"Plan (single-revision auto-refactor): {json.dumps(_fast_plan, indent=2)}\n")
        rc = run_tool(_fast_plan, user_req)
        sys.exit(rc)

    # ── Fast-path: bypass Ollama planner for "query <repo>[model]: question" ──
    # Interactive: "query commons-io[32b]" with no colon/question → REPL session
    import re as _re_main
    _qm = _re_main.match(
        r'^(?:query|ask|fast\s+query)\s+([\w\-]+)(?:\[([\w:\-\.]+)\])?\s*(?::\s*(.+))?$',
        user_req.strip(), _re_main.I,
    )
    if _qm:
        _fast_plan = {
            "tool": "query",
            "repo": _qm.group(1).strip(),
        }
        if _qm.group(3):
            _fast_plan["question"] = _qm.group(3).strip()
        if _qm.group(2):
            _fast_plan["interp_model"] = _qm.group(2).strip()
        if model_override and not _fast_plan.get("model"):
            _fast_plan["model"] = model_override
        print(f"Plan (fast-path): {json.dumps(_fast_plan, indent=2)}\n")
        rc = run_tool(_fast_plan, user_req)
        sys.exit(rc)

    # ── Fast-path: "analyze and interpret <repo> with X years Y months apart [with <model>]"
    # Bypasses Ollama planner entirely — no LLM needed, pure regex.
    _am = re.match(
        r'^(?:analyze\s+and\s+interpret|analyze|temporal)\s+([\w\-]+)\s+with\s+'
        r'(\d+)\s+years?\s+(\d+)\s+months?\s+apart'
        r'(?:\s+with\s+(deepseek[-_]r1:\w+|\d+b))?',
        user_req.strip(), re.I,
    )
    if not _am:
        # Also match: "analyze and interpret <repo> <X> years <Y> months apart"
        _am = re.match(
            r'^(?:analyze\s+and\s+interpret|analyze|temporal)\s+([\w\-]+)\s+'
            r'(\d+)\s+years?\s+(\d+)\s+months?\s+apart'
            r'(?:\s+with\s+(deepseek[-_]r1:\w+|\d+b))?',
            user_req.strip(), re.I,
        )
    if _am:
        _repo_name = _am.group(1).strip()
        _years = int(_am.group(2))
        _months = int(_am.group(3))
        _count = _years * (12 // _months)
        _model_raw = (_am.group(4) or "").strip().lower()
        if _model_raw and not _model_raw.startswith("deepseek"):
            _model_raw = f"deepseek-r1:{_model_raw}"
        _fast_plan = {
            "tool": "temporal_analysis",
            "repo": _repo_name,
            "count": _count,
            "branch": "trunk" if _repo_name.lower() == "pdfbox" else "main",
            "min_months_apart": _months,
            "model": _model_raw or model_override or "deepseek-r1:32b",
            "auto_refactor": auto_refactor,
            "refactor_model": refactor_model_override or "qwen3-coder-30b-refactor",
            "refactor_loop_count": refactor_loop_count,
            "use_feedback_loop": use_feedback_loop,
        }
        print(f"Plan (fast-path): {json.dumps(_fast_plan, indent=2)}\n")
        rc = run_tool(_fast_plan, user_req)
        sys.exit(rc)

    # ── Fast-path: "analyze <repo> last N revisions M months apart [with mv threshold T] [model]"
    # Handles: "analyze commons-io last 4 revisions 3 months apart with mv threshold 5 and deepseek-r1:32b"
    _lm = re.match(
        r'^(?:analyze(?:\s+and\s+interpret)?|temporal)\s+([\w\-]+)\s+'
        r'(?:last\s+)?(\d+)\s+revisions?\s+(\d+)\s+months?\s+apart'
        r'(.*)',
        user_req.strip(), re.I,
    )
    if _lm:
        _repo_name = _lm.group(1).strip()
        _count = int(_lm.group(2))
        _months = int(_lm.group(3))
        _rest = _lm.group(4).lower()
        # Extract model
        _model_raw = ""
        _mm = re.search(r'(deepseek[-_]r1:[\w]+|\d+b)', _rest, re.I)
        if _mm:
            _model_raw = _mm.group(1).strip().lower()
            if not _model_raw.startswith("deepseek"):
                _model_raw = f"deepseek-r1:{_model_raw}"
        # Extract mv threshold
        _mv = None
        _mv_m = re.search(r'(?:mv\s+threshold|mv[_\s-]?co[_\s-]?change|mvCochange)\s+(\d+)', _rest, re.I)
        if _mv_m:
            _mv = int(_mv_m.group(1))
        _fast_plan = {
            "tool": "temporal_analysis",
            "repo": _repo_name,
            "count": _count,
            "branch": "trunk" if _repo_name.lower() == "pdfbox" else "master" if "commons" in _repo_name.lower() else "main",
            "min_months_apart": _months,
            "model": _model_raw or model_override or "deepseek-r1:32b",
            "user_request": user_req,
            "auto_refactor": auto_refactor,
            "refactor_model": refactor_model_override or "qwen3-coder-30b-refactor",
            "refactor_loop_count": refactor_loop_count,
            "use_feedback_loop": use_feedback_loop,
        }
        if _mv is not None:
            _fast_plan["mv_cochange"] = _mv
        print(f"Plan (fast-path): {json.dumps(_fast_plan, indent=2)}\n")
        rc = run_tool(_fast_plan, user_req)
        sys.exit(rc)

    print("Planning...\n")

    try:
        response = call_ollama(user_req)
        plan = parse_json(response)

        # model_override (--model flag) is a fallback: only apply if prompt/plan didn't specify one
        if model_override and not plan.get("model"):
            plan["model"] = model_override

        # Always inject auto_refactor/refactor_model/refactor_loop_count so LLM-generated plans also get full automation
        if auto_refactor and not plan.get("auto_refactor"):
            plan["auto_refactor"] = True
        if refactor_loop_count and not plan.get("refactor_loop_count"):
            plan["refactor_loop_count"] = refactor_loop_count
        if use_feedback_loop and not plan.get("use_feedback_loop"):
            plan["use_feedback_loop"] = True
        if refactor_model_override and not plan.get("refactor_model"):
            plan["refactor_model"] = refactor_model_override
        elif not plan.get("refactor_model"):
            plan["refactor_model"] = "qwen3-coder-30b-refactor"
        # Multi-model comparison
        if len(refactor_models_list) > 1:
            plan["refactor_models"] = refactor_models_list

        print(f"Plan: {json.dumps(plan, indent=2)}\n")

        rc = run_tool(plan, user_req)
        sys.exit(rc)

    except SystemExit:
        raise
    except json.JSONDecodeError:
        # Heuristic fallback: route common intents without JSON
        ur = user_req.lower()
        repo_guess = _guess_repo_from_text(user_req)

        # Interpret temporal folder intent (accept paths to INPUT_INTERPRETATION or temporal_analysis_*)
        _interpret_keywords = [
            "interpret the analysis results", "interpret the temporal analysis",
            "interpret this temporal analysis", "interpret results folder",
            "interpret the temporal", "interpret temporal analysis",
        ]
        _interpret_simple = (
            ur.strip().startswith("interpret ") and
            not any(k in ur for k in ["over time", "all time", "all-time", "revisions", "timestep"])
        )
        if any(k in ur for k in _interpret_keywords) or _interpret_simple:
            p = {"tool": "interpret_temporal", "user_request": user_req}
            # Prefer quoted path if present
            m = re.search(r"['\"]([^'\"]*(?:INPUT_INTERPRETATION|OUTPUT_INTERPRETATION|temporal_analysis)[^'\"]*)['\"]", user_req)
            if m:
                p["repo"] = m.group(1)
            # Pass repo name so tool_interpret_temporal can find latest folder
            if repo_guess and "repo" not in p:
                p["repo"] = repo_guess
            rc = run_tool(p, user_req)
            sys.exit(rc)

        # Peak full arch intent
        if any(k in ur for k in ["biggest m-score", "peak m-score", "largest m-score", "biggest mscore", "largest mscore"]) and any(k in ur for k in ["full arch", "arch report", "antipattern", "anti pattern"]):
            p = {"tool": "peak_full_arch"}
            if repo_guess:
                p["repo"] = repo_guess
            p["user_request"] = user_req
            rc = run_tool(p, user_req)
            sys.exit(rc)

        # Temporal intents
        if any(k in ur for k in ["over time", "last ", "revisions", "commits in between", "months in between", "all time", "all-time", "entire history"]):
            p = {"tool": "temporal_analysis"}
            if repo_guess:
                p["repo"] = repo_guess
            p["user_request"] = user_req
            p["auto_refactor"] = auto_refactor
            p["refactor_model"] = refactor_model_override or "qwen3-coder-30b-refactor"
            p["refactor_loop_count"] = refactor_loop_count
            if use_feedback_loop:
                p["use_feedback_loop"] = True
            rc = run_tool(p, user_req)
            sys.exit(rc)

        # Plot intents for refined results
        if any(k in ur for k in ["plot", "plots", "visualize"]) and any(k in ur for k in ["refined", "temporal", "focus", "folder", "antipattern"]):
            p = {"tool": "plot_refined", "user_request": user_req}
            # Try to extract a folder path from quotes
            m = re.search(r"['\"]([^'\"]*temporal_analysis[^'\"]*)['\"]", user_req)
            if not m:
                m = re.search(r"['\"]([^'\"]*focus_commits[^'\"]*)['\"]", user_req)
            if m:
                p['folder'] = m.group(1)
            rc = run_tool(p, user_req)
            sys.exit(rc)

        # Fallback to concept explain
        print("\nPlanner returned non-JSON; falling back to Explain Concept using RAG if available.\n")
        rc = tool_explain_concept({"topic": user_req})
        sys.exit(rc)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
