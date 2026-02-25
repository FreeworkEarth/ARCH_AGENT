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
# Use the shared RAG explainer location
EXPLAINER = os.path.join(os.path.dirname(THIS_DIR), "04_RAG_EXPLAINER", "integrated_explainer.py")
TEMPORAL = os.path.join(THIS_DIR, "temporal_analyzer.py")
PLOTTER = os.path.join(THIS_DIR, "metric_plotter.py")
BACKFILL_TEMPORAL = os.path.join(THIS_DIR, "backfill_temporal_payloads.py")
INTERPRET_TEMPORAL = os.path.join(os.path.dirname(THIS_DIR), "02_stage_interpret", "interpret_temporal_bundle.py")
BUNDLE_VERIFY = os.path.join(os.path.dirname(THIS_DIR), "02_stage_interpret", "verify_interpretation_bundle.py")

# Config
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
# Prefer explicit env, then TEST_AUTO/RAG_KnowledgeBase, else legacy KB/archdia
_PARENT = os.path.dirname(THIS_DIR)
_RAG_LOCAL = os.path.join(_PARENT, "RAG_KnowledgeBase")
_RAG_LEGACY = os.path.join(_PARENT, "KB", "archdia")
RAG_KB_DIR = os.getenv("RAG_KB_DIR") or os.getenv("ARCHDIA_KB_DIR") or (_RAG_LOCAL if os.path.isdir(_RAG_LOCAL) else _RAG_LEGACY)

# Ensure data output folder exists (created on first run)
_REPOS_ANALYZED_DIR = os.path.join(os.path.dirname(THIS_DIR), "REPOS_ANALYZED")
os.makedirs(_REPOS_ANALYZED_DIR, exist_ok=True)

# Enhanced system prompt with tool-calling
SYSTEM_PROMPT = """You are a DV8 architecture analysis assistant with the following tools:

Tools available:
1. analyze_repo - Run DV8 analysis on a repository (single revision)
2. explain_metrics - Generate detailed AI explanation of DV8 metrics
3. explain_concept - Explain DV8 concepts using knowledge base
4. temporal_analysis - Analyze multiple Git revisions and plot metrics over time
5. interpret_metrics - Interpret WHY metrics changed by analyzing git commits (requires timeseries.json)
6. interpret_temporal - Interpret a temporal analysis folder (pairwise DRH diffs + overall summary)
6. peak_full_arch - Find the two revisions with biggest M-score delta and run full arch reports on both

Output ONLY JSON:
{
  "tool": "analyze_repo|explain_metrics|explain_concept|temporal_analysis|interpret_metrics|interpret_temporal",
  "repo": "<local path or Git URL>",
  "ask": "all|m-score|propagation-cost|..." (optional for analyze_repo),
  "skip_arch_report": true|false (default: true),
  "force_depends": true|false (default: false),
  "source_path": "<subfolder path>" (optional: analyze only this subdir; relative to repo root),
  "topic": "<concept>" (for explain_concept),
  "count": <number> (for temporal_analysis, default: 5),
  "branch": "<branch>" (for temporal_analysis, default: "trunk" for PDFBox, otherwise "main"),
  "min_months_apart": <number> (for temporal_analysis: 0=all-time mode, 1-12=recent mode with N months spacing),
  "model": "<ollama_model>" (for interpret_metrics/interpret_temporal, default: "deepseek-r1:32b", recommended: "deepseek-r1:70b" for best quality)
}

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

Examples:
- "analyze 5 revisions of all time for pdfbox" → {"tool": "temporal_analysis", "repo": "pdfbox", "count": 5, "branch": "trunk", "min_months_apart": 0}
- "analyze 5 major revisions of pdfbox with 3 months in between" → {"tool": "temporal_analysis", "repo": "pdfbox", "count": 5, "branch": "trunk", "min_months_apart": 3}
- "analyze last 7 major revisions with 1 month spacing" → {"tool": "temporal_analysis", "repo": "pdfbox", "count": 7, "branch": "trunk", "min_months_apart": 1}
- "analyze jsoup all-time with 10 timesteps 3 months apart and then interpret with deepseek-r1:32b" → {"tool": "temporal_analysis", "repo": "jsoup", "count": 10, "branch": "main", "min_months_apart": 3, "model": "deepseek-r1:32b"}
- "analyze commons-io all-time in 5 timesteps and then interpret with deepseek-r1:70b" → {"tool": "temporal_analysis", "repo": "commons-io", "count": 5, "branch": "main", "min_months_apart": 0, "model": "deepseek-r1:70b"}
- "interpret the temporal analysis for jsoup with deepseek-r1:32b" → {"tool": "interpret_temporal", "repo": "jsoup", "model": "deepseek-r1:32b"}
- "interpret this temporal analysis folder '/.../temporal_analysis_alltime_...'" → {"tool": "interpret_temporal", "repo": "/.../temporal_analysis_alltime_.../INPUT_INTERPRETATION", "model": "deepseek-r1:32b"}
- "analyze ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG all-time in 2 timesteps on branch temporal and then interpret with deepseek-r1:32b" → {"tool": "temporal_analysis", "repo": "ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG", "count": 2, "branch": "temporal", "min_months_apart": 0, "model": "deepseek-r1:32b"}
- "analyze and interpret https://github.com/apache/commons-io.git all time 5 timesteps with deepseek-r1:32b and answer: why did the m-score change between 2020 and 2024?" → {"tool": "temporal_analysis", "repo": "https://github.com/apache/commons-io.git", "count": 5, "branch": "main", "min_months_apart": 0, "model": "deepseek-r1:32b"}
- "analyze https://github.com/apache/pdfbox.git all-time 5 timesteps and then interpret with deepseek-r1:32b" → {"tool": "temporal_analysis", "repo": "https://github.com/apache/pdfbox.git", "count": 5, "branch": "trunk", "min_months_apart": 0, "model": "deepseek-r1:32b"}

KNOWN REPOSITORIES (use EXACTLY these short names when no URL is given):
- "jsoup" → Java HTML parser
- "commons-io" → Apache Commons IO
- "pdfbox" → Apache PDFBox (branch: trunk)
- "ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG" → Train ticket toy example (also known as "train ticket toy", "trainticket toy", "TTS toy", "toy example", "multilang toy")

IMPORTANT: If the user provides a full GitHub/git URL (starts with https:// or git://), use the FULL URL as the "repo" value — do NOT replace it with a short name.
IMPORTANT: If no URL is given, use the exact short name from KNOWN REPOSITORIES above. If user says "train ticket toy", "TTS toy", "toy example", "multilang" → use "ARCH_ANALYSIS_TRAINTICKET_TOY_EXAMPLES_MULTILANG". Never invent placeholder paths.
IMPORTANT: Always extract the branch from the prompt. "on branch temporal" or "temporal branch" → "branch": "temporal". Default is "main" only if no branch is mentioned.
"""

def _http_json(method: str, path: str, payload: dict | None, timeout=120):
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

def _temporal_root_from_interpretation_path(p: str) -> str | None:
    """Given a path to INPUT_INTERPRETATION or its parent temporal folder, return the temporal root."""
    try:
        pp = pathlib.Path(p).expanduser().resolve()
    except Exception:
        return None
    if pp.is_file():
        pp = pp.parent
    if pp.name == "INPUT_INTERPRETATION":
        tr = pp.parent
        if (tr / "timeseries.json").exists():
            return str(tr)
        # Common user mistake: path points to a placeholder folder without timeseries.json.
        # Try siblings like "<repo>/(temporal_analysis*/timeseries.json)" and pick newest.
        try:
            sib = tr.parent
            candidates = list(sib.glob("temporal_analysis*/timeseries.json"))
            if candidates:
                newest = max(candidates, key=lambda x: x.stat().st_mtime)
                return str(newest.parent)
        except Exception:
            pass
        return None
    # If path itself is temporal root
    if (pp / "timeseries.json").exists():
        return str(pp)
    # Walk upwards a bit
    cur = pp
    for _ in range(4):
        if (cur / "timeseries.json").exists():
            return str(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    return None

def tool_interpret_temporal(plan: dict) -> int:
    """
    Interpret a temporal analysis folder (pairwise DRH diffs + overall summary).

    Accepts repo field as either:
      - <temporal_root>
      - <temporal_root>/INPUT_INTERPRETATION
    """
    ur = (plan.get("user_request") or "")

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
        m = re.search(r"['\"]([^'\"]+INPUT_INTERPRETATION[^'\"]*)['\"]", ur)
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
        print("No temporal folder provided. Pass a temporal_analysis_* path or its INPUT_INTERPRETATION subfolder.")
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
            base = pp.parent if pp.name == "INPUT_INTERPRETATION" else pp
            if base.exists():
                candidates = list(base.glob("temporal_analysis*/timeseries.json"))
                if candidates:
                    newest = max(candidates, key=lambda x: x.stat().st_mtime)
                    print(f"Hint: try temporal root: {newest.parent}")
        except Exception:
            pass
        return 1
    tr = pathlib.Path(temporal_root)
    ts_path = tr / "timeseries.json"
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

    # --- Check for existing interpretation runs ---
    model_safe = model.replace("/", "_").replace(":", "_")
    interp_dir = tr / "INPUT_INTERPRETATION"
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
        try:
            choice = input("  Choice [s/r, default=s]: ").strip().lower()
        except EOFError:
            choice = "s"
        if choice != "r":
            report_text = latest.read_text(encoding="utf-8")
            print(f"\nUsing: {latest}")
            user_question = _extract_user_question(ur)
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
                conversation = []
                current_question = user_question
                sep = "=" * 70
                while current_question:
                    print(f"\nAnswering: {current_question!r}")
                    prior = "\n\n".join(conversation)
                    ctx = (prior + "\n\n" + report_text) if prior else report_text
                    raw = _mod.answer_user_question(model, current_question, ctx, timeout_s=900)
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
                    print(f"\n{sep}\n  FOLLOW-UP (Enter / 'q' to finish)\n{sep}")
                    try:
                        next_q = input("  Your question: ").strip()
                    except EOFError:
                        next_q = ""
                    if not next_q or next_q.lower() in ("q", "quit", "exit"):
                        break
                    if not any(c.isalpha() or c.isdigit() for c in next_q):
                        break
                    current_question = next_q
            return 0
    # --- End skip-check: fall through to full interpretation ---

    cmd = ["python3", INTERPRET_TEMPORAL, "--temporal-root", str(tr), "--repo", str(repo), "--model", model]
    user_question = _extract_user_question(ur)
    if user_question:
        cmd += ["--user-question", user_question]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc == 0:
        # Find the newest combined report (filenames now include a timestamp suffix).
        candidates = sorted(interp_dir.glob(f"*/temporal_interpretation_report_{model_safe}*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            print(f"\nTemporal interpretation report: {candidates[0]}")
    return rc

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
            print(f"\nCould not find repository '{repo}' in {test_auto_dir / 'REPOS'}/")
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

    # Infer params from natural language if present
    ur = (plan.get("user_request") or "").lower()
    if ur:
        import re
        # Branch override: "on branch temporal", "temporal branch", "branch=temporal"
        m_branch = re.search(r"\bon\s+branch\s+([A-Za-z0-9._\-/]+)", ur, re.I) or \
                   re.search(r"([A-Za-z0-9._\-/]+)\s+branch\b", ur, re.I) or \
                   re.search(r"branch[=:\s]+([A-Za-z0-9._\-/]+)", ur, re.I)
        if m_branch:
            branch = m_branch.group(1).strip()
        # Count: "last 5 revisions" or "over 5 revisions"
        m_cnt = re.search(r"(?:last|over|for)\s+(\d+)\s+revisions?", ur)
        if m_cnt:
            try:
                count = int(m_cnt.group(1))
            except ValueError:
                pass
        # Months spacing: "with 3 months in between"
        m_mon = re.search(r"(\d+)\s*\.?\s*months?", ur)
        if m_mon and ("between" in ur or "in between" in ur or "apart" in ur or "spacing" in ur):
            try:
                min_months_apart = int(m_mon.group(1))
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
            m_dr = re.search(r"depends[_-]?runner\s*=\s*(auto|dv8|jar)", plan.get("user_request") or "", re.I)
            if m_dr:
                depends_runner = m_dr.group(1).strip()
        if not java_depends:
            m_jd = re.search(r"java[_-]?depends\s*=\s*(true|false)", plan.get("user_request") or "", re.I)
            if m_jd:
                java_depends = (m_jd.group(1).strip().lower() == "true")
        if "python" in ur and "java" not in ur:
            lang_hint = "python"
        if "java" in ur and "python" not in ur:
            lang_hint = "java"
        # Scope hints
        if any(k in ur for k in ["both scopes", "scope both", "full and prod", "prod and full"]):
            scope = "both"
        elif any(k in ur for k in ["prod-only", "prod only", "production only", "production scope"]):
            scope = "prod"
        elif "production" in ur or "prod" in ur:
            # Only override if user explicitly asked; keep default otherwise.
            scope = plan.get("scope", "prod")

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
        return cmd
    # If user specified a source_path, run once.
    json_file = None
    if source_path:
        cmd = _build_cmd(repo, source_path, lang_hint, None)
        print(f"\nTool: Temporal Analysis")
        print(f"   Repository: {repo}")
        print(f"   Revisions: {count}")
        print(f"   Branch: {branch}")
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

        if workspace:
            search_root = pathlib.Path(workspace).expanduser().resolve()
            all_candidates = list(search_root.glob("*/temporal_analysis*/timeseries.json"))
        else:
            search_root = test_auto_dir / "REPOS_ANALYZED"
            all_candidates = list(search_root.glob("*/temporal_analysis*/timeseries.json"))

        if all_candidates:
            json_file = max(all_candidates, key=lambda p: p.stat().st_mtime)
            repos_dir = json_file.parent.parent   # REPOS_ANALYZED/<repo_name>/
            repo_name = repos_dir.name
        else:
            # Fallback: derive from repo variable as before
            repo_name = pathlib.Path(repo).name if "://" not in repo else pathlib.Path(repo.rstrip('/').split('/')[-1]).stem.replace(".git", "")
            repos_dir = search_root / repo_name
            json_file = repos_dir / "timeseries.json"

        print(f"\nOutput files:")
        print(f"   Time-series data: {json_file}")
        print(f"   Revision folders: {repos_dir}/temporal_analysis*/")

        if repos_dir and repos_dir.exists():
            print(f"\nAnalyzed revisions:")
            for rev_dir in sorted(repos_dir.glob("temporal_analysis*")):
                print(f"   - {rev_dir.name}/")

    if json_file and json_file.exists():
            # Auto-run backfill to prepare interpretation bundle
            temporal_folder = json_file.parent
            print("\nPreparing interpretation bundle...")
            bf_cmd = ["python3", BACKFILL_TEMPORAL, str(temporal_folder), "--meta-repo", repo_name]
            bf_rc = subprocess.call(bf_cmd)
            if bf_rc == 0:
                print("Interpretation bundle ready.")
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

            # Check if user explicitly asked for interpretation in original request
            # Suppress auto-interpret if user said "only analyze" / "just analyze" / "analyze only"
            analyze_only = ur and any(k in ur.lower() for k in ['only analyze', 'just analyze', 'analyze only', 'no interpret', 'without interpret'])
            auto_interpret = (not analyze_only) and ur and any(k in ur.lower() for k in [' interpret', 'then interpret', 'and interpret', ' explain why'])

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
                    "user_request": plan.get("user_request") or ur
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
    json_candidates = list(repos_dir.glob("temporal_analysis*/timeseries.json"))
    if json_candidates:
        # Try pattern match
        import re
        preferred = []
        m_cnt = re.search(r"(?:last|over|for)\s+(\d+)\s+revisions?", ur or '')
        m_mon = re.search(r"(\d+)\s*\.?\s*months?", ur or '')
        m_com = re.search(r"(\d+)\s*commits?", ur or '')
        want_alltime = any(k in (ur or '') for k in ["all time", "all-time", "entire history", "from beginning"])
        for c in json_candidates:
            folder = c.parent.name.lower()
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
    json_candidates = list(repos_dir.glob("temporal_analysis*/timeseries.json"))
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
    json_candidates = list(repos_dir.glob("temporal_analysis*/timeseries.json"))
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
    # Detect mode by presence of timeseries.json
    if (pathlib.Path(folder) / 'timeseries.json').exists():
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
    }

    tool = tool_map.get(tool, tool)

    # Guardrails: if the user is clearly asking to explain/define, prefer RAG explain
    ur = (user_request or '').lower()
    # Heuristic: if user mentions biggest/peak m-score difference and full arch/anti-patterns, route to peak_full_arch
    if any(k in ur for k in ["biggest m-score", "peak m-score", "largest m-score", "biggest mscore", "largest mscore"]) and any(k in ur for k in ["full arch", "anti pattern", "antipattern", "arch report", "arch-report"]):
        tool = "peak_full_arch"
    if any(kw in ur for kw in ["what is", "what's", "whats", "explain ", "define "]) and not any(kw in ur for kw in ["interpret", "why ", "over time", "revisions", "commits"]):
        tool = "explain_concept"

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

    else:
        print(f"Unknown tool: {tool}")
        print("Defaulting to analyze_repo...")
        return tool_analyze_repo(plan)

def main():
    # Parse special flags before user request
    args = sys.argv[1:]
    temporal_root_override = None
    model_override = None

    # Extract --temporal-root and --model flags
    filtered_args = []
    i = 0
    while i < len(args):
        if args[i] == "--temporal-root" and i + 1 < len(args):
            temporal_root_override = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model_override = args[i + 1]
            i += 2
        else:
            filtered_args.append(args[i])
            i += 1

    if len(filtered_args) < 1 and not temporal_root_override:
        print('Usage: python LLM_frontend_upgraded.py "your request"')
        print('')
        print('Options:')
        print('  --temporal-root <path>  Explicit temporal analysis folder (skips glob discovery)')
        print('  --model <model>         LLM model for interpretation (default: deepseek-r1:32b)')
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

    # Handle direct interpretation with --temporal-root
    if temporal_root_override:
        print(f"\nDirect interpretation mode")
        print(f"  Temporal root: {temporal_root_override}")
        print(f"  Model: {model_override or 'deepseek-r1:32b'}\n")
        rc = tool_interpret_temporal({
            "repo": temporal_root_override,
            "model": model_override or "deepseek-r1:32b",
            "user_request": user_req
        })
        sys.exit(rc)

    print(f"\nYou asked: {user_req}\n")
    print("Planning...\n")

    try:
        response = call_ollama(user_req)
        plan = parse_json(response)

        # model_override (--model flag) is a fallback: only apply if prompt/plan didn't specify one
        if model_override and not plan.get("model"):
            plan["model"] = model_override

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
            m = re.search(r"['\"]([^'\"]*(?:INPUT_INTERPRETATION|temporal_analysis)[^'\"]*)['\"]", user_req)
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
