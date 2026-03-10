"""
Fast Q&A engine for ARCH_AGENT (Stage 3).

Answers architectural questions about analyzed repos using:
  - TF-IDF RAG retrieval from the unified index (rag_index.py)
  - LLMBackend for generation (llm_backend.py)

No re-running Stage 2 needed. Typical latency: <30s vs 8-10 min.

Usage:
  python3 query_engine.py --repo commons-io --question "which files should I refactor first?"
  python3 query_engine.py --repo commons-io --question "explain why M-score dropped in 2023"
  python3 query_engine.py --repo commons-io --question "what is a clique anti-pattern?"

  # Backend override (no code change needed):
  ARCH_AGENT_LLM_BACKEND=vllm ARCH_AGENT_LLM_BASE_URL=http://gpu:8000 \\
    python3 query_engine.py --repo commons-io --question "..."
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Stage 3 imports (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from llm_backend import LLMBackend
from rag_index import load_index, retrieve


# ---------------------------------------------------------------------------
# Question type detection
# ---------------------------------------------------------------------------

_FILE_REFACTOR_KEYWORDS = [
    "refactor", "which file", "specific file", "worst file", "bad file",
    "file to fix", "file to improve", "give me file", "top file", "files to",
    "file i should", "files i should", "what file", "name the file",
    "hotspot", "most problematic", "most coupled",
]

_METRIC_EXPLAIN_KEYWORDS = [
    "what is", "what does", "explain", "define", "meaning of", "how does",
    "mscore", "m-score", "propagation cost", "decoupling level", "clique",
    "drh", "dsm", "design rule", "anti-pattern", "fan-in", "fan-out",
    "independence level", "clddf", "imcf", "dependency", "coupling",
    "modular", "architecture metric",
]

_CROSS_REPO_KEYWORDS = [
    "across repos", "compare", "all repos", "other repos", "between repos",
    "which repo", "best repo", "worst repo",
]

_DATE_RANGE_KEYWORDS = [
    "from 20", "between 20", "in 20", "during 20", "since 20",
    "year", "period", "timeframe", "when did", "at what point",
]


def _detect_question_type(question: str) -> str:
    """Returns: 'file_refactor' | 'metric_explain' | 'cross_repo' | 'date_range' | 'general'"""
    q = question.lower()
    if any(k in q for k in _FILE_REFACTOR_KEYWORDS):
        return "file_refactor"
    if any(k in q for k in _CROSS_REPO_KEYWORDS):
        return "cross_repo"
    if any(k in q for k in _DATE_RANGE_KEYWORDS):
        return "date_range"
    if any(k in q for k in _METRIC_EXPLAIN_KEYWORDS):
        return "metric_explain"
    return "general"


# ---------------------------------------------------------------------------
# Layer selection by question type
# ---------------------------------------------------------------------------

_LAYER_MAP = {
    "file_refactor":  ["L3", "L2"],        # raw metrics first, then reports
    "metric_explain": ["L1", "L2"],        # KB docs first, then reports
    "cross_repo":     ["L2", "L3"],        # reports across repos
    "date_range":     ["L4", "L3", "L2"],  # commit history + metrics + reports
    "general":        ["L2", "L1", "L3"],  # reports, then KB, then metrics
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def _build_prompt(question: str, qtype: str, chunks: List[Dict[str, Any]]) -> str:
    context = _format_chunks(chunks)

    if qtype == "file_refactor":
        return f"""You are a software architect. Output ONLY a numbered list — no paragraphs, no headers, no conclusions, no thinking.

STRICT FORMAT (follow exactly, nothing else):
1. FileName.java — Layer N, contribution=X.XXXX (cross_penalty=Y.YYY, size=Z files), FanIn=A, FanOut=B; [one sentence: why this is the worst]
2. FileName.java — Layer N, contribution=X.XXXX ...
3. FileName.java — ...
4. FileName.java — ...
5. FileName.java — ...

Rules:
- Use ONLY files and numbers from the DATA below.
- Rank by cross_penalty × module_size (highest architectural damage first).
- Include FanIn/FanOut if shown in the data.
- Do NOT output anything before item 1 or after item 5.
- Do NOT output reasoning, thinking, headers, or summaries.

QUESTION: {question}

DATA (M-score modules ranked by contribution, FanIn/FanOut per file):
{context}
"""

    if qtype == "metric_explain":
        return f"""You are an expert in software architecture metrics. Answer the question clearly and concisely using the knowledge below.

Rules:
- Answer ONLY the question asked (max 250 words).
- Use the knowledge documents below as your primary source.
- Format: 1-sentence definition → 3 bullet points of detail → 1 practical example.
- Do NOT output reasoning or thinking blocks.

QUESTION: {question}

KNOWLEDGE BASE CONTEXT:
{context}
"""

    if qtype == "date_range":
        return f"""You are an expert software architect. Answer the question about architectural changes in a specific time period.

Rules:
- Answer ONLY the question asked (max 300 words).
- Use ONLY facts from the data below — do NOT invent numbers.
- For date-range questions: find the revision(s) whose dates fall in the asked range and cite their specific metric values and commit context.
- Format: bullet points with dates, metric deltas (absolute + percentage), and commit context.
- Do NOT output reasoning or thinking blocks.

QUESTION: {question}

REVISION & METRICS DATA:
{context}
"""

    # General / cross_repo
    return f"""You are an expert software architect answering a question about repository architecture.

Rules:
- Answer ONLY the question asked (max 300 words).
- Use ONLY facts from the data below — do NOT invent numbers or file names.
- Format: short header, bullet points for evidence, 1-sentence conclusion.
- Do NOT output reasoning or thinking blocks.

QUESTION: {question}

DATA:
{context}
"""


def _format_chunks(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        src = Path(c.get("source_file", "")).name
        layer = c.get("layer", "?")
        repo = c.get("repo") or "global"
        header = f"[{i}] Layer={layer} repo={repo} source={src}"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Model name resolution
# ---------------------------------------------------------------------------

def _resolve_model(base_model: str, qtype: str) -> str:
    """User-specified model is always used as-is. No automatic switching."""
    return base_model


# ---------------------------------------------------------------------------
# Main query function
# ---------------------------------------------------------------------------

def query(
    question: str,
    repo: Optional[str] = None,
    model: str = "deepseek-r1:32b",
    top_k: int = 6,
    llm: Optional[LLMBackend] = None,
    index: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    interp_model: Optional[str] = None,
    num_ctx: int = 4096,
) -> str:
    """
    Answer a question using RAG + LLM.

    Args:
        question:     Natural language question.
        repo:         Repo name to focus on (e.g. "commons-io"). None = all repos.
        model:        LLM model name. Always used exactly as specified — no auto-switching.
        top_k:        Number of RAG chunks to retrieve.
        llm:          Pre-built LLMBackend (built if None).
        index:        Pre-loaded index dict (loaded if None).
        verbose:      Print progress info.
        interp_model: Restrict L2 retrieval to chunks from this interpretation model
                      (e.g. "deepseek-r1:32b" or just "32b"). None = use all models.
        num_ctx:      Ollama context window in tokens (default 4096; Q&A needs ~2k).
    Returns:
        Answer string.
    """
    if index is None:
        index = load_index(verbose=verbose)

    qtype = _detect_question_type(question)
    layers = _LAYER_MAP.get(qtype, ["L2", "L1", "L3"])
    effective_model = _resolve_model(model, qtype)

    if verbose:
        model_tag = f", interp_model={interp_model}" if interp_model else ""
        print(f"[query_engine] question_type={qtype}, layers={layers}, model={effective_model}{model_tag}")

    chunks = retrieve(index, question, top_k=top_k, layers=layers, repo=repo, interp_model=interp_model)

    if not chunks:
        # Fallback: search all layers (keep interp_model filter if set)
        chunks = retrieve(index, question, top_k=top_k, repo=repo, interp_model=interp_model)

    if not chunks:
        return "[No relevant data found in index. Run: python3 rag_index.py to rebuild.]"

    if llm is None:
        llm = LLMBackend(model=effective_model, num_ctx=num_ctx)
    elif effective_model != llm.model:
        llm = LLMBackend(
            model=effective_model,
            backend=llm.backend,
            base_url=llm.base_url,
            api_key=llm.api_key,
            num_ctx=num_ctx,
        )

    prompt = _build_prompt(question, qtype, chunks)

    if verbose:
        print(f"[query_engine] Retrieved {len(chunks)} chunks:")
        for _i, _c in enumerate(chunks, 1):
            _src = Path(_c.get("source_file", "")).name
            _layer = _c.get("layer", "?")
            _subtype = _c.get("subtype", "")
            _mtag = f" model={_c['model']}" if _c.get("model") else ""
            _stag = f" [{_subtype}]" if _subtype else ""
            _preview = _c["text"][:100].replace("\n", " ")
            print(f"  {_i}. {_layer}{_stag}{_mtag}  src={_src}")
            print(f"     {_preview}...")
        print(f"[query_engine] Calling {llm} ...")

    answer = llm.generate(prompt)
    return answer


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def _auto_save_folder(
    repo: Optional[str],
    interp_model: Optional[str],
    repos_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Auto-discover the Stage 3 Q&A output folder under the most recent interpretation run.

    Naming convention:
      REPOS_ANALYZED/<repo>/temporal_analysis_*/OUTPUT_INTERPRETATION/<run>_Q&A_STAGE_3/

    The run folder is chosen as the most recent interpretation folder that matches
    interp_model (if given), otherwise the most recent of any model.
    Returns None if no matching folder is found.
    """
    if repo is None:
        return None

    # Locate REPOS_ANALYZED
    if repos_dir is None:
        this_dir = Path(__file__).parent
        repos_dir = this_dir.parent / "REPOS_ANALYZED"

    repo_root = repos_dir / repo
    if not repo_root.exists():
        return None

    # Find all OUTPUT_INTERPRETATION directories under this repo
    interp_dirs = sorted(repo_root.rglob("OUTPUT_INTERPRETATION"), reverse=True)
    if not interp_dirs:
        return None

    best: Optional[Path] = None
    for interp_dir in interp_dirs:
        # Iterate interpretation run folders (exclude non-dirs and special folders)
        run_folders = sorted(
            [d for d in interp_dir.iterdir()
             if d.is_dir() and d.name[:1].isdigit()  # must start with YYYYMMDD date
             and not d.name.endswith("_Q&A_STAGE_3")],  # skip already-created Q&A folders
            reverse=True,  # most recent first (folder names start with YYYYMMDD)
        )
        for rf in run_folders:
            if interp_model:
                model_lower = interp_model.lower()
                if model_lower not in rf.name.lower():
                    continue
            best = rf
            break
        if best:
            break

    if best is None:
        return None

    # Strip any existing _Q&A_STAGE_3 suffix to avoid double-appending
    base_name = best.name
    if base_name.endswith("_Q&A_STAGE_3"):
        base_name = base_name[: -len("_Q&A_STAGE_3")]
    return best.parent / f"{base_name}_Q&A_STAGE_3"


def save_answer(
    question: str,
    answer: str,
    repo: Optional[str],
    run_folder: Optional[Path] = None,
    elapsed_s: Optional[float] = None,
    interp_model: Optional[str] = None,
) -> Optional[Path]:
    """
    Append Q&A to USER_ANSWER_YYYYMMDD_HHMMSS.md in run_folder.

    - run_folder: if None, auto-discovers the Stage 3 Q&A folder under REPOS_ANALYZED
    - elapsed_s:  LLM response time in seconds (shown as first line of each entry)
    - interp_model: used for auto-folder discovery if run_folder is None
    Returns the path written to, or None.
    """
    if run_folder is None:
        run_folder = _auto_save_folder(repo, interp_model)
    if run_folder is None:
        return None

    run_folder.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H:%M:%S")
    # Include datetime in filename so multiple sessions don't collide
    out_file = run_folder / f"USER_ANSWER_{date_str}.md"

    timing_line = ""
    if elapsed_s is not None:
        timing_line = f"**Response time**: {elapsed_s:.1f}s\n\n"

    separator = "\n\n---\n\n"
    entry = f"**Q ({time_str})**: {question}\n\n{timing_line}{answer}"

    if out_file.exists():
        out_file.write_text(
            out_file.read_text(encoding="utf-8") + separator + entry,
            encoding="utf-8",
        )
    else:
        model_tag = f" | interp_model: {interp_model}" if interp_model else ""
        header = (
            f"# Q&A Session — {date_str}\n\n"
            f"**Repo**: {repo or 'all'}{model_tag}\n\n"
            f"---\n\n"
        )
        out_file.write_text(header + entry, encoding="utf-8")
    return out_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fast architectural Q&A using RAG index + LLM."
    )
    ap.add_argument("--repo", default=None,
                    help="Repo name to focus on (e.g. commons-io). Omit for cross-repo questions.")
    ap.add_argument("--question", default=None,
                    help="Question to answer. Omit to enter interactive session.")
    ap.add_argument("--model", default="deepseek-r1:32b",
                    help="Base LLM model (default: deepseek-r1:32b)")
    ap.add_argument("--top-k", type=int, default=6,
                    help="Number of RAG chunks to retrieve (default: 6)")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="Force rebuild the RAG index before answering")
    ap.add_argument("--save-to", default=None,
                    help="Folder to save USER_ANSWER_*.md (optional)")
    ap.add_argument("--interp-model", default=None,
                    help="Restrict retrieval to this interpretation model (e.g. deepseek-r1:32b or 32b)")
    ap.add_argument("--num-ctx", type=int, default=4096,
                    help="Ollama context window in tokens (default: 4096; increase for very long answers)")
    ap.add_argument("--no-interactive", action="store_true",
                    help="Exit after answering one question instead of continuing to interactive loop")
    args = ap.parse_args()

    index = load_index(force=args.rebuild_index, verbose=True)
    save_folder = Path(args.save_to) if args.save_to else None

    # Pre-build the LLM backend once so Ollama keeps the model warm across questions
    llm = LLMBackend(model=args.model, num_ctx=args.num_ctx)

    def _ask(question: str) -> None:
        t0 = _time.monotonic()
        answer = query(
            question=question,
            repo=args.repo,
            model=args.model,
            top_k=args.top_k,
            llm=llm,
            index=index,
            verbose=True,
            interp_model=args.interp_model,
            num_ctx=args.num_ctx,
        )
        elapsed_s = _time.monotonic() - t0
        print("\n" + "=" * 60)
        print(answer)
        print("=" * 60)
        print(f"\n[Response time: {elapsed_s:.1f}s]")
        out = save_answer(
            question, answer, args.repo,
            run_folder=save_folder,
            elapsed_s=elapsed_s,
            interp_model=args.interp_model,
        )
        if out:
            print(f"[Saved to {out}]")

    if args.question:
        _ask(args.question)
        if args.no_interactive:
            return  # scripted single-shot: exit now

    # Interactive loop — always entered unless --no-interactive was set
    model_tag = f" | interp: {args.interp_model}" if args.interp_model else ""
    print(f"\n{'='*60}")
    print(f"  ARCH_AGENT Q&A  |  repo: {args.repo or 'all'}  |  model: {args.model}{model_tag}")
    print(f"  Type your next question and press Enter. 'exit' or Ctrl-C to quit.")
    print(f"{'='*60}\n")
    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended]")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("[Session ended]")
            break
        _ask(question)


if __name__ == "__main__":
    main()
