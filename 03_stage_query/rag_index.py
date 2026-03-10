"""
Unified 4-layer RAG index builder for ARCH_AGENT.

Indexes all knowledge sources into a single TF-IDF index (stdlib only, no external deps):
  L1 — KB docs: DV8/M-score/DRH/coupling concept files (RAG_KnowledgeBase/)
  L2 — Repo reports: temporal_interpretation_report_*.md per analyzed repo
  L3 — Raw metrics: timeseries.json + mscore_exact_components.json (numerical facts)
  L4 — Git commits: revision metadata from timeseries.json (dates, hashes, metric deltas)

Index is stored as ARCH_AGENT/03_stage_query/.rag_index.json
Each chunk carries: text, layer, repo, source_file, chunk_id

Usage:
  python3 rag_index.py --repos-dir ../REPOS_ANALYZED --kb-dir /path/to/RAG_KnowledgeBase
  python3 rag_index.py  # auto-discovers paths relative to this file
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEX_FILE = Path(__file__).parent / ".rag_index.json"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK_LEN = 80  # discard very short chunks

_LAYER_NAMES = {
    "L1": "KB docs (DV8/M-score concepts)",
    "L2": "Repo analysis reports",
    "L3": "Raw metrics (timeseries/mscore)",
    "L4": "Git commit / revision metadata",
}


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LEN:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# TF-IDF helpers (stdlib only)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_\-\.]+", text.lower())


def _tf(tokens: List[str]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = max(len(tokens), 1)
    return {t: c / total for t, c in counts.items()}


def _build_idf(all_token_sets: List[set]) -> Dict[str, float]:
    N = len(all_token_sets)
    df: Dict[str, int] = {}
    for ts in all_token_sets:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    return {t: math.log((N + 1) / (d + 1)) + 1.0 for t, d in df.items()}


def _tfidf_vec(tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
    return {t: v * idf.get(t, 1.0) for t, v in tf.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    dot = sum(a.get(t, 0.0) * v for t, v in b.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# PDF text extraction (uses pdftotext from poppler if available)
# ---------------------------------------------------------------------------

def _extract_pdf_text(path: Path) -> str:
    """Extract plain text from a PDF using pdftotext. Returns empty string if unavailable."""
    try:
        res = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return res.stdout.strip()
    except FileNotFoundError:
        return ""  # pdftotext not installed — skip silently
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Chunk collection: L1 KB docs
# ---------------------------------------------------------------------------

def _collect_l1(kb_dir: Path) -> List[Dict[str, Any]]:
    chunks = []
    # Text-based formats
    for ext in ("*.txt", "*.md", "*.rst", "*.html"):
        for p in sorted(kb_dir.rglob(ext)):
            try:
                text = p.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                continue
            if not text:
                continue
            for i, chunk in enumerate(_chunk_text(text)):
                chunks.append({
                    "layer": "L1",
                    "repo": "",
                    "source_file": str(p),
                    "chunk_id": f"L1:{p.name}:{i}",
                    "text": chunk,
                })
    # PDF research papers (extracted via pdftotext)
    for p in sorted(kb_dir.rglob("*.pdf")):
        text = _extract_pdf_text(p)
        if not text or len(text) < MIN_CHUNK_LEN:
            continue
        for i, chunk in enumerate(_chunk_text(text)):
            chunks.append({
                "layer": "L1",
                "repo": "",
                "source_file": str(p),
                "chunk_id": f"L1:{p.name}:{i}",
                "text": chunk,
            })
    return chunks


# ---------------------------------------------------------------------------
# Chunk collection: L2 Repo reports
# ---------------------------------------------------------------------------

def _model_from_interp_path(p: Path) -> str:
    """Extract model tag from interpretation folder name.

    E.g. folder '260224_222836_deepseek-r1_32b' → 'deepseek-r1:32b'
         folder '260224_203207_deepseek-r1_14b' → 'deepseek-r1:14b'
    """
    # Walk up until we find the dated interpretation folder
    for part in reversed(p.parts):
        m = re.search(
            r'(deepseek[-_]r1|qwen[\d.]+|llama[\d.]+|mistral[\d.]+|phi[\d.]+)[_\-]([\w]+b)\b',
            part, re.I,
        )
        if m:
            return f"{m.group(1).replace('_', '-')}:{m.group(2)}"
    return ""


def _collect_l2(repos_dir: Path) -> List[Dict[str, Any]]:
    """Index all L2 interpretation artifacts:
    - temporal_interpretation_report_*.md  (overall summary per run)
    - drh_diff_report_*.md                 (per-transition diff with FanIn/FanOut/layer data)
    - USER_ANSWER_*.md                     (previous Q&A sessions)
    - interpretation_payload.md            (exact structured payload fed to LLM per revision)
    """
    chunks = []

    # 1. Temporal interpretation summary reports
    for report in sorted(repos_dir.rglob("temporal_interpretation_report_*.md")):
        repo = _repo_name_from_path(report)
        model = _model_from_interp_path(report)
        try:
            text = report.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not text:
            continue
        for i, chunk in enumerate(_chunk_text(text)):
            chunks.append({
                "layer": "L2",
                "subtype": "temporal_report",
                "repo": repo,
                "model": model,
                "source_file": str(report),
                "chunk_id": f"L2:report:{repo}:{report.name}:{i}",
                "text": chunk,
            })

    # 2. Per-transition DRH diff reports (richest source for file ranking + date-range questions)
    # Match drh_diff_report_*.md but NOT .prompt.txt and NOT .verify.md
    for report in sorted(repos_dir.rglob("drh_diff_report_*.md")):
        if report.suffix != ".md":
            continue
        repo = _repo_name_from_path(report)
        model = _model_from_interp_path(report)
        try:
            text = report.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not text:
            continue
        for i, chunk in enumerate(_chunk_text(text)):
            chunks.append({
                "layer": "L2",
                "subtype": "drh_diff",
                "repo": repo,
                "model": model,
                "source_file": str(report),
                "chunk_id": f"L2:drh:{repo}:{report.name}:{i}",
                "text": chunk,
            })

    # 3. USER_ANSWER files (previous Q&A sessions from Stage 2 runs)
    for qa_file in sorted(repos_dir.rglob("USER_ANSWER_*.md")):
        repo = _repo_name_from_path(qa_file)
        model = _model_from_interp_path(qa_file)
        try:
            text = qa_file.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not text:
            continue
        for i, chunk in enumerate(_chunk_text(text)):
            chunks.append({
                "layer": "L2",
                "subtype": "user_answer",
                "repo": repo,
                "model": model,
                "source_file": str(qa_file),
                "chunk_id": f"L2:qa:{repo}:{qa_file.name}:{i}",
                "text": chunk,
            })

    # 4. Interpretation payloads (structured LLM input per revision — hotspot data, DRH summary)
    for payload_file in sorted(repos_dir.rglob("interpretation_payload.md")):
        repo = _repo_name_from_path(payload_file)
        rev_folder = payload_file.parent.parent.name  # e.g. "01_commons-io_23022026_1058"
        try:
            text = payload_file.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not text:
            continue
        for i, chunk in enumerate(_chunk_text(text)):
            chunks.append({
                "layer": "L2",
                "subtype": "payload",
                "repo": repo,
                "model": "",
                "source_file": str(payload_file),
                "chunk_id": f"L2:payload:{repo}:{rev_folder}:{i}",
                "text": chunk,
            })

    return chunks


# ---------------------------------------------------------------------------
# Chunk collection: L3 Raw metrics
# ---------------------------------------------------------------------------

def _collect_l3(repos_dir: Path) -> List[Dict[str, Any]]:
    chunks = []

    for ts_file in sorted(repos_dir.rglob("timeseries.json")):
        repo = _repo_name_from_path(ts_file)
        try:
            ts = json.loads(ts_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        revisions = ts.get("revisions") or []
        lines = [f"TIMESERIES for {repo}:"]
        for r in revisions:
            if not isinstance(r, dict):
                continue
            num = r.get("revision_number", "?")
            date = r.get("commit_date", "?")
            m = r.get("metrics") or {}
            mscore = m.get("mscore") or m.get("m_score") or "?"
            pc = m.get("propagation_cost") or "?"
            dl = m.get("decoupling_level") or "?"
            il = m.get("independence_level") or "?"
            lines.append(
                f"  rev{num} ({date}): mscore={mscore}, propagation_cost={pc}, "
                f"decoupling_level={dl}, independence_level={il}"
            )
        text = "\n".join(lines)
        for i, chunk in enumerate(_chunk_text(text)):
            chunks.append({
                "layer": "L3",
                "repo": repo,
                "source_file": str(ts_file),
                "chunk_id": f"L3:ts:{repo}:{i}",
                "text": chunk,
            })

    for mscore_file in sorted(repos_dir.rglob("mscore_exact_components.json")):
        repo = _repo_name_from_path(mscore_file)
        try:
            data = json.loads(mscore_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        text = _mscore_to_text(repo, mscore_file, data)
        for i, chunk in enumerate(_chunk_text(text)):
            chunks.append({
                "layer": "L3",
                "repo": repo,
                "source_file": str(mscore_file),
                "chunk_id": f"L3:mscore:{repo}:{mscore_file.parent.name}:{i}",
                "text": chunk,
            })

    return chunks


def _mscore_to_text(repo: str, path: Path, data: Dict[str, Any]) -> str:
    """Serialize mscore_exact_components.json into searchable text."""
    rev_label = path.parent.name  # e.g. "01_commons-io_23022026_1058"
    lines = [f"M-SCORE COMPONENTS for {repo} revision {rev_label}:"]
    mscore = data.get("mscore") or data.get("m_score") or "?"
    lines.append(f"  overall mscore={mscore}")
    for mod in (data.get("module_details") or [])[:20]:
        if not isinstance(mod, dict):
            continue
        layer = mod.get("layer", "?")
        size = mod.get("module_size", "?")
        cp = mod.get("cross_penalty", "?")
        contrib = mod.get("contribution", "?")
        files = (mod.get("files") or [])[:4]
        fnames = ", ".join(f.split("/")[-1] for f in files if isinstance(f, str))
        lines.append(
            f"  Layer {layer}: size={size} files, cross_penalty={cp}, "
            f"contribution={contrib} | files: {fnames}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chunk collection: L4 Git commit metadata
# ---------------------------------------------------------------------------

def _collect_l4(repos_dir: Path) -> List[Dict[str, Any]]:
    chunks = []
    for ts_file in sorted(repos_dir.rglob("timeseries.json")):
        repo = _repo_name_from_path(ts_file)
        try:
            ts = json.loads(ts_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        revisions = ts.get("revisions") or []
        lines = [f"GIT REVISION HISTORY for {repo}:"]
        for r in revisions:
            if not isinstance(r, dict):
                continue
            num = r.get("revision_number", "?")
            date = r.get("commit_date", "?")
            sha = (r.get("commit_hash") or r.get("hash") or "")[:8]
            msg = (r.get("commit_message") or r.get("message") or "").strip()[:200]
            m = r.get("metrics") or {}
            mscore = m.get("mscore") or m.get("m_score") or "?"
            lines.append(f"  rev{num} | {date} | {sha} | mscore={mscore} | {msg}")
        text = "\n".join(lines)
        for i, chunk in enumerate(_chunk_text(text)):
            chunks.append({
                "layer": "L4",
                "repo": repo,
                "source_file": str(ts_file),
                "chunk_id": f"L4:{repo}:{i}",
                "text": chunk,
            })
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_name_from_path(p: Path) -> str:
    """Extract repo name from a REPOS_ANALYZED/<repo>/... path."""
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "REPOS_ANALYZED" and i + 1 < len(parts):
            return parts[i + 1]
    return p.parts[-4] if len(p.parts) >= 4 else "unknown"


# ---------------------------------------------------------------------------
# Index build / load / save
# ---------------------------------------------------------------------------

def build_index(
    repos_dir: Path,
    kb_dir: Optional[Path],
    papers_dir: Optional[Path] = None,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Build (or reload cached) unified index. Returns index dict.

    Args:
        repos_dir:  Path to REPOS_ANALYZED folder (L2/L3/L4 sources).
        kb_dir:     Path to RAG_KnowledgeBase folder with .txt/.md docs (L1).
        papers_dir: Optional path to folder with research PDFs (L1, extracted via pdftotext).
        force:      Rebuild even if cached index is fresh.
        verbose:    Print progress.
    """
    # Check if cached index is fresh enough
    if not force and INDEX_FILE.exists():
        try:
            idx = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            # Simple staleness check: if any source dir is newer than index, rebuild
            index_mtime = INDEX_FILE.stat().st_mtime
            stale = False
            if repos_dir.exists() and repos_dir.stat().st_mtime > index_mtime:
                stale = True
            if kb_dir and kb_dir.exists() and kb_dir.stat().st_mtime > index_mtime:
                stale = True
            if papers_dir and papers_dir.exists() and papers_dir.stat().st_mtime > index_mtime:
                stale = True
            if not stale:
                if verbose:
                    n = len(idx.get("chunks", []))
                    print(f"[rag_index] Loaded cached index ({n} chunks) from {INDEX_FILE}")
                return idx
        except Exception:
            pass

    if verbose:
        print("[rag_index] Building unified index...")

    chunks: List[Dict[str, Any]] = []

    # L1 — archdia KB text docs
    if kb_dir and kb_dir.exists():
        l1 = _collect_l1(kb_dir)
        if verbose:
            print(f"  L1 (KB docs):      {len(l1):4d} chunks from {kb_dir}")
        chunks.extend(l1)
    else:
        if verbose:
            print(f"  L1 (KB docs):      skipped (kb_dir not found: {kb_dir})")

    # L1 — research papers (PDFs)
    if papers_dir and papers_dir.exists():
        l1_papers = _collect_l1(papers_dir)
        if verbose:
            print(f"  L1 (papers):       {len(l1_papers):4d} chunks from {papers_dir}")
        chunks.extend(l1_papers)
    else:
        if verbose:
            print(f"  L1 (papers):       skipped (papers_dir not found: {papers_dir})")

    # L2, L3, L4
    if repos_dir.exists():
        l2 = _collect_l2(repos_dir)
        l3 = _collect_l3(repos_dir)
        l4 = _collect_l4(repos_dir)
        if verbose:
            print(f"  L2 (reports):      {len(l2):4d} chunks from {repos_dir}")
            print(f"  L3 (raw metrics):  {len(l3):4d} chunks")
            print(f"  L4 (git commits):  {len(l4):4d} chunks")
        chunks.extend(l2)
        chunks.extend(l3)
        chunks.extend(l4)
    else:
        if verbose:
            print(f"  L2/L3/L4:          skipped (repos_dir not found: {repos_dir})")

    if not chunks:
        if verbose:
            print("[rag_index] WARNING: no chunks collected — index is empty")

    # Build IDF over all chunks
    all_token_sets = [set(_tokenize(c["text"])) for c in chunks]
    idf = _build_idf(all_token_sets)

    # Compute TF-IDF vectors for each chunk (stored as flat list of [token, score] pairs
    # to keep JSON portable; we'll reconstruct the dict on load)
    for chunk, token_set in zip(chunks, all_token_sets):
        tf = _tf(_tokenize(chunk["text"]))
        vec = _tfidf_vec(tf, idf)
        chunk["_vec"] = vec  # dict: token → tfidf score

    index = {
        "built_at": time.time(),
        "chunk_count": len(chunks),
        "idf": idf,
        "chunks": chunks,
    }

    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(f"[rag_index] Index saved to {INDEX_FILE} ({len(chunks)} chunks total)")
    return index


def load_index(
    repos_dir: Optional[Path] = None,
    kb_dir: Optional[Path] = None,
    papers_dir: Optional[Path] = None,
    force: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Load index from cache or build if missing/stale."""
    _repos, _kb, _papers = _auto_discover_paths(repos_dir, kb_dir, papers_dir)
    return build_index(_repos, _kb, papers_dir=_papers, force=force, verbose=verbose)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    index: Dict[str, Any],
    query: str,
    top_k: int = 6,
    layers: Optional[List[str]] = None,
    repo: Optional[str] = None,
    interp_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top_k chunks for a query.

    Args:
        layers:       If provided, restrict to these layers (e.g. ["L1", "L3"])
        repo:         If provided, prefer chunks from this repo (still includes L1)
        interp_model: If provided, restrict L2 chunks to this interpretation model
                      (e.g. "deepseek-r1:32b"). L1 chunks are always included.
                      Case-insensitive prefix match (e.g. "32b" matches "deepseek-r1:32b").
    """
    idf = index.get("idf", {})
    chunks = index.get("chunks", [])

    # Filter by layer
    if layers:
        pool = [c for c in chunks if c.get("layer") in layers]
    else:
        pool = chunks

    # Filter L2 chunks by interpretation model if specified
    # L1 (KB docs + papers) always included regardless
    if interp_model:
        model_lower = interp_model.lower()
        pool = [
            c for c in pool
            if c.get("layer") != "L2"
            or model_lower in (c.get("model") or "").lower()
            or not c.get("model")  # keep payload chunks (model="") always
        ]

    if not pool:
        return []

    # Build query vector
    q_tokens = _tokenize(query)
    q_tf = _tf(q_tokens)
    q_vec = _tfidf_vec(q_tf, idf)
    q_token_set = set(q_tokens)

    scored = []
    for chunk in pool:
        vec = chunk.get("_vec") or {}
        score = _cosine(q_vec, vec)

        # Filename boost: if query tokens appear in the source filename
        fname = Path(chunk.get("source_file", "")).name.lower()
        boost = sum(0.15 for t in q_token_set if t in fname)

        # Repo boost: prefer chunks from the specified repo (not L1 — those are global)
        if repo and chunk.get("repo") == repo and chunk.get("layer") != "L1":
            boost += 0.1

        # Subtype boost: prefer drh_diff chunks for file-ranking queries
        # (they have the exact FanIn/FanOut/cross-penalty data)
        if chunk.get("subtype") == "drh_diff":
            boost += 0.05

        scored.append((score + boost, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# ---------------------------------------------------------------------------
# Path auto-discovery
# ---------------------------------------------------------------------------

def _auto_discover_paths(
    repos_dir: Optional[Path],
    kb_dir: Optional[Path],
    papers_dir: Optional[Path] = None,
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    this_dir = Path(__file__).parent  # ARCH_AGENT/03_stage_query/
    arch_agent = this_dir.parent      # ARCH_AGENT/
    test_auto = arch_agent.parent     # RA Software Architecture Analsysis/AGENT/

    if repos_dir is None:
        repos_dir = arch_agent / "REPOS_ANALYZED"

    if kb_dir is None:
        # Try sibling of ARCH_AGENT
        candidate = test_auto / "TEST_AUTO" / "RAG_KnowledgeBase"
        if not candidate.exists():
            # Try one level up
            candidate = test_auto.parent / "TEST_AUTO" / "RAG_KnowledgeBase"
        if not candidate.exists():
            # Fallback: look for it relative to this file more broadly
            for p in test_auto.rglob("RAG_KnowledgeBase"):
                if p.is_dir():
                    candidate = p
                    break
        kb_dir = candidate if candidate.exists() else None

    if papers_dir is None:
        # Auto-discover TEST_AUTO/DOCS/PAPER
        candidate = test_auto / "TEST_AUTO" / "DOCS" / "PAPER"
        if not candidate.exists():
            candidate = test_auto.parent / "TEST_AUTO" / "DOCS" / "PAPER"
        papers_dir = candidate if candidate.exists() else None

    return repos_dir, kb_dir, papers_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build or rebuild the ARCH_AGENT unified RAG index."
    )
    ap.add_argument(
        "--repos-dir",
        default=None,
        help="Path to REPOS_ANALYZED folder (auto-discovered if omitted)",
    )
    ap.add_argument(
        "--kb-dir",
        default=None,
        help="Path to RAG_KnowledgeBase folder (auto-discovered if omitted)",
    )
    ap.add_argument(
        "--papers-dir",
        default=None,
        help="Path to research papers folder with PDFs (auto-discovered if omitted)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index is fresh",
    )
    ap.add_argument(
        "--query",
        default=None,
        help="Test retrieval with a sample query after building",
    )
    ap.add_argument(
        "--interp-model",
        default=None,
        help="Filter L2 chunks to a specific interpretation model (e.g. deepseek-r1:32b or 32b)",
    )
    args = ap.parse_args()

    repos_dir = Path(args.repos_dir).expanduser().resolve() if args.repos_dir else None
    kb_dir = Path(args.kb_dir).expanduser().resolve() if args.kb_dir else None
    papers_dir = Path(args.papers_dir).expanduser().resolve() if args.papers_dir else None
    repos_dir, kb_dir, papers_dir = _auto_discover_paths(repos_dir, kb_dir, papers_dir)

    index = build_index(repos_dir, kb_dir, papers_dir=papers_dir, force=args.force, verbose=True)

    if args.query:
        interp_model = args.interp_model or None
        print(f"\n--- Test retrieval: {args.query!r} (interp_model={interp_model or 'any'}) ---")
        results = retrieve(index, args.query, top_k=4, interp_model=interp_model)
        for i, chunk in enumerate(results, 1):
            preview = chunk["text"][:200].replace("\n", " ")
            model_tag = f" model={chunk['model']}" if chunk.get("model") else ""
            subtype_tag = f" subtype={chunk['subtype']}" if chunk.get("subtype") else ""
            print(f"  {i}. [{chunk['layer']}]{subtype_tag}{model_tag} {Path(chunk['source_file']).name}")
            print(f"     repo={chunk.get('repo') or 'global'}")
            print(f"     {preview}...")
            print()


if __name__ == "__main__":
    main()
