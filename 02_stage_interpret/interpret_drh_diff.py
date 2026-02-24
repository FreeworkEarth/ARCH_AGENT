#!/usr/bin/env python3
"""
interpret_drh_diff.py

Generate a second-stage LLM report that explains DRH changes between 2 timesteps
using the structured interpretation payloads (and DRH clustering JSON).

The report starts with a "Comprehensive Summary" (biggest architectural threats at
the newer timestep), then explains what changed between the two DRH snapshots.

Works best on a temporal analysis folder that has:
  <temporal_root>/timeseries.json
  <temporal_root>/INPUT_INTERPRETATION/SINGLE_REVISION_ANALYSIS_DATA/<rev>/OutputData/interpretation_payload.json
  <temporal_root>/.../dv8-analysis-result/dsm/drh-clustering.json per revision

Example (Zeppelin, compare newest vs previous):
  python3 interpret_drh_diff.py \
    --temporal-root ../REPOS/zeppelin/temporal_analysis_alltime_2013-06_to_2025-11 \
    --new 1 --old 2 \
    --repo ../REPOS/zeppelin \
    --model deepseek-r1:14b

If you want to only generate the prompt (no LLM call):
  python3 interpret_drh_diff.py ... --no-llm
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ANALYZE_DIR = Path(__file__).resolve().parents[1] / "01_STAGE_ANALYZE"
if ANALYZE_DIR.exists():
    sys.path.insert(0, str(ANALYZE_DIR))
try:
    from compute_evidence_graph_diff import build_diff as build_graph_diff
    from compute_evidence_graph_diff import find_matrix_json as find_matrix_json
    from compute_evidence_graph_diff import read_json as read_matrix_json
except Exception:
    build_graph_diff = None
    find_matrix_json = None
    read_matrix_json = None

from commit_analyzer import CommitAnalyzer


_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s == "-":
        return None
    if _NUM_RE.match(s):
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if not s or s == "-":
        return None
    if _NUM_RE.match(s):
        try:
            return float(s)
        except Exception:
            return None
    return None


def find_drh_json(output_data: Path) -> Optional[Path]:
    direct = output_data / "dv8-analysis-result" / "dsm" / "drh-clustering.json"
    if direct.exists():
        return direct
    found = list(output_data.glob("**/dv8-analysis-result/dsm/drh-clustering.json"))
    return found[0] if found else None


@dataclass(frozen=True)
class DrhLocation:
    layer: str
    module_path: str  # e.g., "L3/Interpreter/Launcher"


def _walk_drh(node: Dict[str, Any], layer_name: str, path_stack: List[str], out: Dict[str, DrhLocation]) -> None:
    ntype = node.get("@type")
    name = node.get("name")
    if ntype == "item" and isinstance(name, str):
        module_path = "/".join([layer_name] + path_stack) if path_stack else layer_name
        out[name] = DrhLocation(layer=layer_name, module_path=module_path)
        return

    if ntype == "group":
        nested = node.get("nested") or []
        next_stack = path_stack
        if isinstance(name, str) and name and name != layer_name:
            next_stack = path_stack + [name]
        for child in nested:
            if isinstance(child, dict):
                _walk_drh(child, layer_name, next_stack, out)


def flatten_drh(drh: Dict[str, Any]) -> Dict[str, DrhLocation]:
    out: Dict[str, DrhLocation] = {}
    for layer in drh.get("structure") or []:
        if not isinstance(layer, dict):
            continue
        layer_name = str(layer.get("name") or "")
        if not layer_name:
            continue
        for child in layer.get("nested") or []:
            if isinstance(child, dict):
                _walk_drh(child, layer_name, [], out)
    return out


def summarize_drh(flat: Dict[str, DrhLocation]) -> Dict[str, Any]:
    layer_counts: Dict[str, int] = {}
    module_counts: Dict[str, int] = {}
    for loc in flat.values():
        layer_counts[loc.layer] = layer_counts.get(loc.layer, 0) + 1
        module_counts[loc.module_path] = module_counts.get(loc.module_path, 0) + 1

    top_modules = sorted(module_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "file_count": len(flat),
        "layer_file_counts": dict(sorted(layer_counts.items(), key=lambda kv: kv[0])),
        "top_modules_by_file_count": [{"module_path": k, "files": v} for k, v in top_modules],
    }


def diff_drh(old_flat: Dict[str, DrhLocation], new_flat: Dict[str, DrhLocation]) -> Dict[str, Any]:
    old_files = set(old_flat)
    new_files = set(new_flat)

    added = sorted(new_files - old_files)
    removed = sorted(old_files - new_files)

    moved_layer: List[Dict[str, str]] = []
    moved_module: List[Dict[str, str]] = []
    for f in sorted(old_files & new_files):
        o = old_flat[f]
        n = new_flat[f]
        if o.layer != n.layer:
            moved_layer.append(
                {"file": f, "old": o.layer, "new": n.layer, "old_module": o.module_path, "new_module": n.module_path}
            )
        elif o.module_path != n.module_path:
            moved_module.append({"file": f, "old": o.module_path, "new": n.module_path})

    return {
        "added_files_count": len(added),
        "removed_files_count": len(removed),
        "moved_layers_count": len(moved_layer),
        "moved_modules_count": len(moved_module),
        "moved_layers_sample": moved_layer[:25],
        "moved_modules_sample": moved_module[:25],
    }


def top_dangerous(dangerous_files: Dict[str, Any], n: int = 7) -> List[Dict[str, Any]]:
    rows = dangerous_files.get("rows") or []
    if not rows:
        return []

    scored = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        churn = safe_int(r.get("ChangeChurn"))
        fan_out = safe_int(r.get("FanOut"))
        fan_in = safe_int(r.get("FanIn"))
        score = (churn or 0) * 10 + (fan_out or 0) + (fan_in or 0)
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:n]]


def top_module_penalties(mscore_components: Dict[str, Any], n: int = 7) -> Dict[str, Any]:
    mods = (mscore_components or {}).get("module_details") or []
    if not isinstance(mods, list) or not mods:
        return {}

    def top_by(key: str) -> List[Dict[str, Any]]:
        vals = [m for m in mods if isinstance(m, dict) and isinstance(m.get(key), (int, float))]
        vals.sort(key=lambda m: m.get(key, 0.0), reverse=True)
        out = []
        for m in vals[:n]:
            out.append(
                {
                    "module_key": m.get("module_key"),
                    "layer": m.get("layer"),
                    "module_size": m.get("module_size"),
                    key: m.get(key),
                    "clddf": m.get("clddf"),
                    "imcf": m.get("imcf"),
                    "files_sample": (m.get("files") or [])[:6],
                }
            )
        return out

    return {
        "top_cross_penalty_modules": top_by("cross_penalty"),
        "top_internal_penalty_modules": top_by("internal_penalty"),
    }


def query_ollama(model: str, prompt: str, timeout_s: int = 900) -> str:
    cmd = ["ollama", "run", model, prompt]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    out = (res.stdout or "").strip()
    if out:
        return out
    err = (res.stderr or "").strip()
    return err


def strip_thinking_block(text: str) -> str:
    """
    Some local reasoning models emit a visible "Thinking..." prelude.
    This strips a leading block from the first "Thinking" line up to the first
    line that contains "done thinking" (case-insensitive), inclusive.
    """
    if not text:
        return text
    lines = text.splitlines()
    first = lines[0].strip().lower()
    if not first.startswith("thinking"):
        return text
    end_idx = None
    for i, ln in enumerate(lines[:200]):  # don't scan unbounded
        if "done thinking" in (ln or "").lower():
            end_idx = i
            break
    if end_idx is None:
        return text
    cleaned = "\n".join(lines[end_idx + 1 :]).lstrip()
    return cleaned if cleaned else text


def strip_code_fences(text: str) -> str:
    """
    If the model wraps the answer in ``` fences, extract the first fenced block.
    """
    if "```" not in text:
        return text
    lines = text.splitlines()
    start = None
    end = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("```"):
            start = i
            break
    if start is None:
        return text
    for j in range(start + 1, len(lines)):
        if lines[j].strip().startswith("```"):
            end = j
            break
    if end is None:
        return text
    inner = "\n".join(lines[start + 1 : end]).strip()
    return inner if inner else text


def normalize_output(text: str) -> str:
    text = strip_thinking_block(text)
    text = strip_code_fences(text)
    return text.strip() + "\n"


def extract_h2_section(text: str, heading: str) -> str:
    """
    Return a single H2 section (including its heading) from a Markdown document.
    If the heading isn't found, returns empty string.
    """
    if not text or not heading:
        return ""
    start = text.find(heading)
    if start == -1:
        return ""
    rest = text[start:]
    # End at the next H2 heading (if any).
    m2 = re.search(r"^##\s+", rest[len(heading) :], re.M)
    if not m2:
        return rest.strip() + "\n"
    end = len(heading) + m2.start()
    return rest[:end].strip() + "\n"


def sanitize_likely_drivers_section(text: str) -> str:
    """
    Enforce a single '## Likely Drivers' section with '-' bullets only:
    - drops any other headings
    - converts numbered lists to '-' bullets
    - turns plain sentences into bullets
    """
    if not text:
        return ""
    lines = [ln.rstrip() for ln in text.splitlines()]
    if not lines:
        return ""
    out: List[str] = []
    out.append("## Likely Drivers")
    for ln in lines[1:]:
        s = ln.strip()
        if not s:
            continue
        if re.match(r"^#{1,6}\s+", s):
            continue
        m = re.match(r"^\d+[\).:-]\s+(.*)$", s)
        if m:
            out.append("- " + m.group(1).strip())
            continue
        if s.startswith("- "):
            out.append("- " + s[2:].strip())
            continue
        out.append("- " + s)
    if len(out) == 1:
        out.append("- (missing)")
    # Cap to 15 bullets max (header + 15 lines)
    if len(out) > 16:
        out = out[:16]
    return "\n".join(out).strip() + "\n"


def build_narrative_hints(context: Dict[str, Any], allowed_files: List[str]) -> List[str]:
    newer = context.get("newer") or {}
    older = context.get("older") or {}
    diff = context.get("drh_diff") or {}
    graph_diff = context.get("evidence_graph_diff") or {}

    hints: List[str] = []

    old_sum = older.get("drh_summary") or {}
    new_sum = newer.get("drh_summary") or {}
    old_layers = old_sum.get("layer_file_counts") or {}
    new_layers = new_sum.get("layer_file_counts") or {}

    # Hint: DRH layer structure change with concrete counts
    if isinstance(old_layers, dict) and isinstance(new_layers, dict):
        old_nonempty = [k for k, v in old_layers.items() if v]
        new_nonempty = [k for k, v in new_layers.items() if v]
        old_desc = ", ".join(f"{k}:{v}" for k, v in sorted(old_layers.items()) if v)
        new_desc = ", ".join(f"{k}:{v}" for k, v in sorted(new_layers.items()) if v)
        if len(old_nonempty) <= 1 and len(new_nonempty) > 1:
            hints.append(
                f"DRH layering changed from {len(old_nonempty)} layer(s) ({old_desc}) "
                f"to {len(new_nonempty)} layer(s) ({new_desc}), "
                f"indicating stronger separation of concerns."
            )
        elif len(old_nonempty) != len(new_nonempty):
            hints.append(
                f"DRH layer distribution changed from ({old_desc}) to ({new_desc}), "
                f"suggesting a reorganization of responsibilities across layers."
            )

    # Hint: file count delta with numbers
    old_fc = old_sum.get("file_count")
    new_fc = new_sum.get("file_count")
    if isinstance(old_fc, int) and isinstance(new_fc, int) and new_fc != old_fc:
        delta = new_fc - old_fc
        if new_fc > old_fc:
            hints.append(
                f"DRH file count changed from {old_fc} to {new_fc} (delta={delta:+d}), "
                f"consistent with decomposing large responsibilities into more files."
            )
        else:
            hints.append(
                f"DRH file count changed from {old_fc} to {new_fc} (delta={delta:+d}), "
                f"consistent with consolidation into fewer units."
            )

    # Hint: edge weight delta with numbers
    edges = graph_diff.get("edges") or {}
    try:
        new_tw = float(((edges.get("new") or {}).get("total_weight")) or 0.0)
        old_tw = float(((edges.get("old") or {}).get("total_weight")) or 0.0)
        delta_tw = new_tw - old_tw
        if new_tw < old_tw:
            hints.append(
                f"Evidence graph total dependency weight changed from {old_tw:.0f} to {new_tw:.0f} "
                f"(delta={delta_tw:+.0f}), aligning with reduced propagation cost."
            )
        elif new_tw > old_tw:
            hints.append(
                f"Evidence graph total dependency weight changed from {old_tw:.0f} to {new_tw:.0f} "
                f"(delta={delta_tw:+.0f}), suggesting more interconnections."
            )
    except Exception:
        pass

    # Hint: SCC change with file names
    scc_new = graph_diff.get("scc_new") or {}
    scc_old = graph_diff.get("scc_old") or {}
    try:
        if isinstance(scc_old.get("scc_count"), int) and isinstance(scc_new.get("scc_count"), int):
            old_count = scc_old["scc_count"]
            new_count = scc_new["scc_count"]
            old_largest = scc_old.get("largest_scc_size", 0)
            old_top_sccs = scc_old.get("top_sccs") or []
            scc_files_str = ""
            if old_top_sccs and isinstance(old_top_sccs[0], list):
                scc_files_str = ", ".join(Path(f).name for f in old_top_sccs[0][:6])
            if new_count < old_count:
                hints.append(
                    f"SCC count changed from {old_count} (largest={old_largest} files"
                    + (f": {scc_files_str}" if scc_files_str else "")
                    + f") to {new_count}, indicating fewer dependency cycles (supports lower propagation cost)."
                )
            elif new_count > old_count:
                hints.append(
                    f"SCC count changed from {old_count} to {new_count}, "
                    f"indicating more dependency cycles."
                )
    except Exception:
        pass

    # Hint: god-class indicator from heaviest removed edge
    edges_removed = graph_diff.get("edges_removed_sample") or []
    if edges_removed and isinstance(edges_removed, list):
        try:
            heaviest = max(
                (e for e in edges_removed if isinstance(e, dict) and isinstance(e.get("weight"), (int, float))),
                key=lambda e: e["weight"],
                default=None,
            )
            if heaviest and heaviest.get("weight", 0) >= 10:
                src_name = Path(heaviest["src"]).name if heaviest.get("src") else "?"
                dest_name = Path(heaviest["dest"]).name if heaviest.get("dest") else "?"
                kind = heaviest.get("kind", "?")
                weight = heaviest["weight"]
                scc_files_list = ""
                old_top_sccs = (scc_old.get("top_sccs") or [])
                if old_top_sccs and isinstance(old_top_sccs[0], list):
                    scc_files_list = ", ".join(Path(f).name for f in old_top_sccs[0][:6])
                hint = (
                    f"The heaviest removed edge was {src_name} -> {dest_name} "
                    f"({kind} weight={weight:.0f}), indicating a high-coupling dependency was eliminated."
                )
                if scc_files_list:
                    hint += (
                        f" The older revision had {scc_old.get('scc_count', '?')} SCC(s) containing "
                        f"{scc_files_list}."
                    )
                hints.append(hint)
        except Exception:
            pass

    # Hint: heaviest ADDED edge (new coupling introduced)
    edges_added = graph_diff.get("edges_added_sample") or []
    if edges_added and isinstance(edges_added, list):
        try:
            heaviest_add = max(
                (e for e in edges_added if isinstance(e, dict) and isinstance(e.get("weight"), (int, float))),
                key=lambda e: e["weight"],
                default=None,
            )
            if heaviest_add and heaviest_add.get("weight", 0) >= 5:
                src_name = Path(heaviest_add["src"]).name if heaviest_add.get("src") else "?"
                dest_name = Path(heaviest_add["dest"]).name if heaviest_add.get("dest") else "?"
                kind = heaviest_add.get("kind", "?")
                weight = heaviest_add["weight"]
                hints.append(
                    f"The heaviest added edge was {src_name} -> {dest_name} "
                    f"({kind} weight={weight:.0f}), indicating new coupling was introduced."
                )
        except Exception:
            pass

    # Hint: anti-pattern count changes
    new_ap = (newer.get("payload_highlights", {}).get("anti_pattern_counts") or {})
    old_ap = (older.get("payload_highlights", {}).get("anti_pattern_counts") or {})
    for pat in sorted(set(list(new_ap.keys()) + list(old_ap.keys()))):
        nv = new_ap.get(pat, 0)
        ov = old_ap.get(pat, 0)
        if isinstance(nv, (int, float)) and isinstance(ov, (int, float)) and nv != ov:
            hints.append(f"Anti-pattern '{pat}' count changed from {ov} to {nv} (delta={nv - ov:+d}).")

    # Hint: metric deltas with exact numbers
    new_metrics = newer.get("metrics") or {}
    old_metrics = older.get("metrics") or {}
    for metric_key in ("m-score", "propagation-cost", "decoupling-level", "independence-level"):
        nv = new_metrics.get(metric_key)
        ov = old_metrics.get(metric_key)
        if isinstance(nv, (int, float)) and isinstance(ov, (int, float)):
            delta = round(float(nv) - float(ov), 2)
            rel = round((delta / float(ov)) * 100.0, 2) if ov != 0 else None
            rel_str = f", {rel:+.2f}% relative" if rel is not None else ""
            hints.append(
                f"{metric_key} changed from {round(float(ov), 2)} to {round(float(nv), 2)} "
                f"(delta={delta:+.2f} points{rel_str})."
            )

    # Hint: encourage mentioning concrete files
    if allowed_files:
        sample = ", ".join(allowed_files[:5])
        hints.append(f"Refer to concrete files that gained architectural importance (e.g., {sample}).")

    return hints[:15]


def build_managers_special(context: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Deterministic Comprehensive Summary to avoid hallucinated filenames/numbers.
    Returns (markdown, allowed_filenames).
    """
    newer = context.get("newer") or {}
    older = context.get("older") or {}
    diff = context.get("drh_diff") or {}
    graph_diff = context.get("evidence_graph_diff") or {}
    cc = context.get("commit_context") or {}

    new_metrics = newer.get("metrics") or {}
    old_metrics = older.get("metrics") or {}
    deltas = {}
    for k in ("propagation-cost", "m-score", "decoupling-level", "independence-level"):
        nv = new_metrics.get(k)
        ov = old_metrics.get(k)
        if isinstance(nv, (int, float)) and isinstance(ov, (int, float)):
            deltas[k] = round(nv - ov, 2)

    rel_deltas = {}
    for k, dv in deltas.items():
        ov = old_metrics.get(k)
        if isinstance(ov, (int, float)) and ov != 0:
            rel_deltas[k] = round((dv / float(ov)) * 100.0, 2)

    highlights = newer.get("payload_highlights") or {}
    dangerous_top = highlights.get("dangerous_files_top") or []
    allowed_files = []
    danger_bullets = []
    for r in dangerous_top[:5]:
        if not isinstance(r, dict):
            continue
        fn = r.get("Filename")
        if isinstance(fn, str):
            allowed_files.append(fn)
        danger_bullets.append(_fmt_kv(r, ["Filename", "FanIn", "FanOut", "ChangeCount", "ChangeChurn"]))

    issue_typed = highlights.get("issue_typed_churn") or {}
    ap_counts = highlights.get("anti_pattern_counts") or {}

    ms_pen = highlights.get("mscore_penalty_modules") or {}
    cross = (ms_pen.get("top_cross_penalty_modules") or [])[:4]
    cross_lines = []
    for m in cross:
        if not isinstance(m, dict):
            continue
        files = m.get("files_sample") or []
        for f in files:
            if isinstance(f, str):
                allowed_files.append(f)
        cross_lines.append(_fmt_kv(m, ["module_key", "layer", "module_size", "cross_penalty", "clddf"]))

    # Expand allowed file list deterministically from facts we provide (so verifier doesn't false-fail).
    for r in (diff.get("moved_layers_sample") or [])[:25]:
        if isinstance(r, dict) and isinstance(r.get("file"), str):
            allowed_files.append(r["file"])
    for r in (graph_diff.get("edges_added_sample") or [])[:25]:
        if isinstance(r, dict):
            if isinstance(r.get("src"), str):
                allowed_files.append(r["src"])
            if isinstance(r.get("dest"), str):
                allowed_files.append(r["dest"])
    for r in (graph_diff.get("edges_removed_sample") or [])[:25]:
        if isinstance(r, dict):
            if isinstance(r.get("src"), str):
                allowed_files.append(r["src"])
            if isinstance(r.get("dest"), str):
                allowed_files.append(r["dest"])
    for r in (graph_diff.get("fan_in_delta_top") or [])[:20]:
        if isinstance(r, dict) and isinstance(r.get("node"), str):
            allowed_files.append(r["node"])
    for r in (graph_diff.get("fan_out_delta_top") or [])[:20]:
        if isinstance(r, dict) and isinstance(r.get("node"), str):
            allowed_files.append(r["node"])
    for r in (cc.get("hotspot_files") or [])[:20]:
        if isinstance(r, dict) and isinstance(r.get("file"), str):
            allowed_files.append(r["file"])

    lines = ["## Comprehensive Summary"]
    if deltas:
        lines.append(f"- Metric deltas (new-old, absolute points): {deltas}")
    if rel_deltas:
        lines.append(f"- Metric deltas (new-old, relative %): {rel_deltas}")
    if ap_counts:
        lines.append(f"- Anti-pattern counts (newer): {ap_counts}")
    if danger_bullets:
        lines.append("- Most dangerous files (newer, DV8): " + " | ".join(danger_bullets[:3]))
    if issue_typed.get("commit_count") or issue_typed.get("churn_total"):
        lines.append(f"- Issue-typed churn (newer): commit_count={issue_typed.get('commit_count')}, churn_total={issue_typed.get('churn_total')}")
    if diff:
        old_fc = (older.get("drh_summary") or {}).get("file_count")
        new_fc = (newer.get("drh_summary") or {}).get("file_count")
        if isinstance(old_fc, int) and isinstance(new_fc, int):
            lines.append(f"- DRH file count: old={old_fc} → new={new_fc} (Δ={new_fc - old_fc:+d})")
        lines.append(
            f"- DRH movement (old→new): moved_layers={diff.get('moved_layers_count')}, moved_modules={diff.get('moved_modules_count')}, added_files={diff.get('added_files_count')}, removed_files={diff.get('removed_files_count')} (note: renames/moves often show up as add+remove)"
        )
    if cross_lines:
        lines.append("- Highest cross-layer penalty modules (newer): " + " | ".join(cross_lines))

    # de-dup while keeping order
    seen = set()
    dedup = []
    for f in allowed_files:
        if f not in seen:
            seen.add(f)
            dedup.append(f)
    return "\n".join(lines) + "\n", dedup


def _fmt_kv(d: Dict[str, Any], keys: List[str]) -> str:
    parts = []
    for k in keys:
        v = d.get(k)
        if v is None or v == "" or v == {} or v == []:
            continue
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def build_drh_differences_deterministic(context: Dict[str, Any]) -> str:
    newer = context.get("newer") or {}
    older = context.get("older") or {}
    diff = context.get("drh_diff") or {}

    new_sum = newer.get("drh_summary") or {}
    old_sum = older.get("drh_summary") or {}

    lines: List[str] = []
    lines.append("## DRH Differences")
    lines.append("### Structural Summary")

    old_fc = old_sum.get("file_count")
    new_fc = new_sum.get("file_count")
    if isinstance(old_fc, int) and isinstance(new_fc, int):
        lines.append(f"- DRH file count: old={old_fc} → new={new_fc} (Δ={new_fc - old_fc:+d})")

    old_layers = old_sum.get("layer_file_counts")
    new_layers = new_sum.get("layer_file_counts")
    if isinstance(old_layers, dict) and isinstance(new_layers, dict):
        lines.append(f"- Layer file counts (old): {old_layers}")
        lines.append(f"- Layer file counts (new): {new_layers}")

    old_top = (old_sum.get("top_modules_by_file_count") or [])[:4]
    new_top = (new_sum.get("top_modules_by_file_count") or [])[:4]
    if old_top:
        lines.append("- Largest modules by file count (old): " + " | ".join(_fmt_kv(r, ["module_path", "files"]) for r in old_top))
    if new_top:
        lines.append("- Largest modules by file count (new): " + " | ".join(_fmt_kv(r, ["module_path", "files"]) for r in new_top))

    lines.append("### Movement Summary")
    lines.append(
        "- DRH movement (old→new): "
        + _fmt_kv(
            diff,
            [
                "moved_layers_count",
                "moved_modules_count",
                "added_files_count",
                "removed_files_count",
            ],
        )
        + " (note: renames/moves often show up as add+remove; rely on DRH file count delta for net change)"
    )

    lines.append("### Notable Layer Moves (sample)")
    moved_layers = diff.get("moved_layers_sample") or []
    if isinstance(moved_layers, list) and moved_layers:
        for r in moved_layers[:12]:
            if not isinstance(r, dict):
                continue
            f = r.get("file")
            o = r.get("old")
            n = r.get("new")
            if f and o and n:
                lines.append(f"- {f}: {o} → {n}")
    else:
        lines.append("- (none)")

    moved_modules = diff.get("moved_modules_sample") or []
    if isinstance(moved_modules, list) and moved_modules:
        lines.append("### Notable Module Moves (sample)")
        for r in moved_modules[:12]:
            if not isinstance(r, dict):
                continue
            f = r.get("file")
            o = r.get("old")
            n = r.get("new")
            if f and o and n:
                lines.append(f"- {f}: {o} → {n}")

    return "\n".join(lines).rstrip() + "\n"


def build_metrics_and_evidence_deterministic(context: Dict[str, Any]) -> str:
    newer = context.get("newer") or {}
    older = context.get("older") or {}
    diff = context.get("drh_diff") or {}
    graph_diff = context.get("evidence_graph_diff") or {}
    graph_path = context.get("evidence_graph_diff_path")
    cc = context.get("commit_context") or {}

    new_metrics = newer.get("metrics") or {}
    old_metrics = older.get("metrics") or {}

    def metric_line(key: str) -> str | None:
        nv = new_metrics.get(key)
        ov = old_metrics.get(key)
        if not isinstance(nv, (int, float)) or not isinstance(ov, (int, float)):
            return None
        delta = round(float(nv) - float(ov), 2)
        rel = None
        if ov != 0:
            rel = round((delta / float(ov)) * 100.0, 2)
        base = f"- {key}: old={round(float(ov), 2)} → new={round(float(nv), 2)} (Δ={delta:+.2f} points"
        if rel is not None:
            base += f", {rel:+.2f}% relative"
        base += ")"
        return base

    lines: List[str] = []
    lines.append("## Metrics & Evidence")
    lines.append("### DV8 Metric Changes (older→newer)")
    for k in ("m-score", "propagation-cost", "decoupling-level", "independence-level"):
        ml = metric_line(k)
        if ml:
            lines.append(ml)
    lines.append("- note: DV8 metrics are in %; Δ points are absolute (new-old), not percentage points of a 0–1 scale.")

    if isinstance(diff, dict):
        old_fc = (older.get("drh_summary") or {}).get("file_count")
        new_fc = (newer.get("drh_summary") or {}).get("file_count")
        if isinstance(old_fc, int) and isinstance(new_fc, int):
            lines.append(f"- DRH file count (older→newer): {old_fc} → {new_fc} (Δ={new_fc - old_fc:+d})")

    lines.append("### Evidence Graph (matrix.json derived, if available)")
    if graph_path:
        lines.append(f"- evidence_graph_diff: `{graph_path}`")
    nodes = graph_diff.get("nodes")
    edges = graph_diff.get("edges")
    if isinstance(nodes, dict):
        lines.append(f"- nodes: {nodes}")
    if isinstance(edges, dict):
        lines.append(f"- edges: {edges}")
        try:
            new_tw = float(((edges.get("new") or {}).get("total_weight")) or 0.0)
            old_tw = float(((edges.get("old") or {}).get("total_weight")) or 0.0)
            lines.append(f"- edges.total_weight delta (new-old): {round(new_tw - old_tw, 2):+.2f}")
        except Exception:
            pass
    scc_new = graph_diff.get("scc_new")
    scc_old = graph_diff.get("scc_old")
    if isinstance(scc_old, dict) and isinstance(scc_new, dict):
        lines.append(f"- scc_old: {scc_old}")
        lines.append(f"- scc_new: {scc_new}")

    lines.append("### Risk Signals (newer timestep)")
    ap_counts = newer.get("payload_highlights", {}).get("anti_pattern_counts", {}) or {}
    if ap_counts:
        lines.append(f"- anti-pattern counts: {ap_counts}")

    danger = newer.get("payload_highlights", {}).get("dangerous_files_top", []) or []
    if isinstance(danger, list) and danger:
        lines.append("- dangerous files (top):")
        for r in danger[:7]:
            if not isinstance(r, dict):
                continue
            fn = r.get("Filename") or r.get("file") or r.get("path")
            fan_in = r.get("FanIn")
            fan_out = r.get("FanOut")
            churn = r.get("ChangeChurn")
            if fn:
                lines.append(f"  - {fn} (FanIn={fan_in}, FanOut={fan_out}, ChangeChurn={churn})")

    ms_pen = newer.get("payload_highlights", {}).get("mscore_penalty_modules", {}) or {}
    cross = (ms_pen.get("top_cross_penalty_modules") or [])[:4]
    if cross:
        lines.append("- highest cross-layer penalty modules (newer):")
        for r in cross:
            if isinstance(r, dict):
                lines.append("  - " + _fmt_kv(r, ["module_key", "layer", "module_size", "cross_penalty", "clddf"]))

    if isinstance(cc, dict) and (cc.get("hotspot_files") or cc.get("top_commits") or cc.get("categories")):
        lines.append("### Commit Context (older→newer window)")
        if cc.get("categories") is not None:
            lines.append(f"- categories: {cc.get('categories')}")
        top_commits = cc.get("top_commits") or []
        if isinstance(top_commits, list) and top_commits:
            lines.append("- top commits (sample):")
            for r in top_commits[:5]:
                if isinstance(r, dict):
                    lines.append("  - " + _fmt_kv(r, ["hash", "date", "message", "files_changed", "churn"]))
        hotspot_files = cc.get("hotspot_files") or []
        if isinstance(hotspot_files, list) and hotspot_files:
            lines.append("- hotspot files (sample):")
            for r in hotspot_files[:7]:
                if isinstance(r, dict):
                    lines.append("  - " + _fmt_kv(r, ["file", "churn", "commits"]))

    return "\n".join(lines).rstrip() + "\n"


def build_prompt(context: Dict[str, Any]) -> str:
    newer = context.get("newer") or {}
    older = context.get("older") or {}
    cc = context.get("commit_context") or {}

    def bullets(rows: List[Dict[str, Any]], fields: List[str], max_n: int = 7) -> str:
        out = []
        for r in (rows or [])[:max_n]:
            if not isinstance(r, dict):
                continue
            out.append("- " + _fmt_kv(r, fields))
        return "\n".join(out) if out else "- (none)"

    def moved_bullets(rows: List[Dict[str, Any]], max_n: int = 12) -> str:
        out = []
        for r in (rows or [])[:max_n]:
            if not isinstance(r, dict):
                continue
            f = r.get("file")
            o = r.get("old")
            n = r.get("new")
            if f and o and n:
                out.append(f"- {f}: {o} → {n}")
        return "\n".join(out) if out else "- (none)"

    managers_special_md, allowed_files = build_managers_special(context)
    drh_md = build_drh_differences_deterministic(context)
    evidence_md = build_metrics_and_evidence_deterministic(context)
    narrative_hints = build_narrative_hints(context, allowed_files)

    template = """You are an expert software architect. Continue the report below by writing ONLY the final section.

Hard rules (strict):
- Output MUST be Markdown text, NOT a code block. Do NOT wrap in triple backticks.
- Do NOT output hidden reasoning or "thinking".
- Do NOT modify any existing sections in REPORT SO FAR.
- Do NOT introduce new file names. Only use file names from ALLOWED FILES (if you mention any).
- You MAY quote numeric values from the FACTS and REPORT SO FAR sections. Do NOT invent new numbers. When citing a metric, use the exact value from FACTS (e.g., "m-score rose from XX.XX to YY.YY").
- Do NOT output any other headings besides the required H2 below (no extra '##' or '###' headings).
- Do NOT include recommendations, action plans, "next steps", or refactoring advice. Only explain likely drivers.

You must output exactly this section (and nothing before it):

## Likely Drivers
- (use '-' bullet list only; 10–15 bullets)
- Bullets 1-2: Describe the DRH layering change (if any) with the old and new layer counts from FACTS. If layers changed, name the largest old and new modules. If layers did not change, describe the stable structure.
- Bullets 3-5: Name specific files from ALLOWED FILES and describe their architectural role. If files were removed, mention their old fan-out or heaviest edge weight. If files were added, mention their fan-in. Reference SCC membership if relevant.
- Bullets 6-7: Explain why m-score changed (or stayed stable), citing the exact old and new values and the delta from FACTS.
- Bullets 8-9: Explain why propagation-cost changed (or stayed stable), referencing SCC changes, edge-weight changes, and the dependency weight delta from FACTS.
- Bullets 10+: Any additional drivers visible in the evidence (edge weight changes by kind, added/removed edges, anti-pattern count changes, module reorganization, pattern adoption, etc.).
- You MUST quote specific numbers from the FACTS and NARRATIVE HINTS sections to support each claim.
- Use NARRATIVE HINTS to structure your reasoning — they contain pre-computed summaries you can paraphrase.
- If all metric deltas are near zero (< 1 point absolute), write 5–8 bullets explaining why the architecture remained stable instead of forcing 10+ bullets about nonexistent changes.

EXAMPLE of well-written bullets (STYLE REFERENCE ONLY — these names and numbers are fictional, do NOT copy them; use YOUR data from FACTS):
- The DRH restructured from a flat N-layer layout (L0: X files, one module) to a K-layer hierarchy (L0: A, L1: B, L2: C), indicating the team introduced explicit separation between architectural responsibilities.
- <HubClass>.java (fan-out: W Call edges to <Target>.java) was a high-coupling hub in the older revision; its removal eliminated the N-file SCC (<FileA>.java, <FileB>.java, ...) and distributed responsibilities across newly introduced service and repository classes.
- Propagation-cost dropped from XX.XX to YY.YY (delta = -ZZ.ZZ points, -PP.PP% relative), primarily because the N-file SCC was eliminated (scc_count: M -> 0) and total edge weight decreased from E1 to E2.
    """

    new_metrics = newer.get("metrics") or {}
    old_metrics = older.get("metrics") or {}
    deltas: Dict[str, float] = {}
    for k in ("propagation-cost", "m-score", "decoupling-level", "independence-level"):
        nv = new_metrics.get(k)
        ov = old_metrics.get(k)
        if isinstance(nv, (int, float)) and isinstance(ov, (int, float)):
            deltas[k] = round(nv - ov, 2)

    newer_issue = newer.get("payload_highlights", {}).get("issue_typed_churn", {}) or {}
    older_issue = older.get("payload_highlights", {}).get("issue_typed_churn", {}) or {}
    graph_diff = context.get("evidence_graph_diff") or {}

    data_block = f"""
REPORT SO FAR (do not edit):
{managers_special_md}

{drh_md}

    {evidence_md}

    NARRATIVE HINTS (safe to reuse verbatim):
    {"" if not narrative_hints else chr(10).join([f"- {h}" for h in narrative_hints])}

    ALLOWED FILES (only these may be referenced by name):
    {chr(10).join(f"- {f}" for f in allowed_files[:120]) or "- (none)"}

FACTS (copy/paste; do not change numbers or file names)

NEWER TIMESTEP
- revision_number: {newer.get('revision_number')}
- dir: {newer.get('dir')}
- commit: {newer.get('commit_hash')} | date: {newer.get('commit_date')}
- message: {newer.get('commit_message')}
- key metrics (DV8 %, as reported): {new_metrics}

OLDER TIMESTEP
- revision_number: {older.get('revision_number')}
- dir: {older.get('dir')}
- commit: {older.get('commit_hash')} | date: {older.get('commit_date')}
- message: {older.get('commit_message')}
- key metrics (DV8 %, as reported): {old_metrics}
- metric deltas (new-old, absolute points): {deltas}

RISK SIGNALS (NEWER)
- anti-pattern counts: {newer.get('payload_highlights',{}).get('anti_pattern_counts',{})}
- dangerous files (top):\n{bullets(newer.get('payload_highlights',{}).get('dangerous_files_top',[]), ['Filename','FanIn','FanOut','ChangeCount','ChangeChurn'], 7)}
- typed issue churn (commit_count): {newer_issue.get('commit_count')}
- typed issue churn (churn_total): {newer_issue.get('churn_total')}
- M-Score penalty modules (top cross_penalty):\n{bullets((newer.get('payload_highlights',{}).get('mscore_penalty_modules',{}) or {}).get('top_cross_penalty_modules',[]), ['module_key','layer','module_size','cross_penalty','clddf','files_sample'], 6)}

DRH (STRUCTURE)
- newer DRH summary: {newer.get('drh_summary')}
- older DRH summary: {older.get('drh_summary')}

EVIDENCE GRAPH DIFF
- nodes: {graph_diff.get('nodes')}
- edges: {graph_diff.get('edges')}
- edges_added_sample: {graph_diff.get('edges_added_sample')}
- edges_removed_sample: {graph_diff.get('edges_removed_sample')}
- fan_in_delta_top: {graph_diff.get('fan_in_delta_top')}
- fan_out_delta_top: {graph_diff.get('fan_out_delta_top')}
- scc_new: {graph_diff.get('scc_new')}
- scc_old: {graph_diff.get('scc_old')}

COMMIT CONTEXT (between the two dates)
- total_commits: {cc.get('total_commits')}
- categories: {cc.get('categories')}
- top_commits: {cc.get('top_commits')}
- hotspot_files: {cc.get('hotspot_files')}
"""

    return template + "\n" + data_block


def main() -> int:
    ap = argparse.ArgumentParser(description="Explain DRH differences between two temporal timesteps using a local LLM (Ollama).")
    ap.add_argument("--temporal-root", required=True, help="Path to temporal_analysis_* folder")
    ap.add_argument("--repo", required=True, help="Path to the git repo (for commit context)")
    ap.add_argument("--new", type=int, required=True, help="Newer revision_number (from timeseries.json, usually 1 is newest)")
    ap.add_argument("--old", type=int, required=True, help="Older revision_number (from timeseries.json)")
    ap.add_argument("--model", default="deepseek-r1:14b", help="Ollama model name (default: deepseek-r1:14b)")
    ap.add_argument("--ollama-timeout-s", type=int, default=900, help="Ollama timeout in seconds (default: 900)")
    ap.add_argument("--output", default=None, help="Output markdown path (default: INPUT_INTERPRETATION/drh_diff_report_*.md)")
    ap.add_argument("--no-llm", action="store_true", help="Only write the prompt file; do not call Ollama")
    ap.add_argument("--verify", action="store_true", help="Run verifier pass on the generated report")
    ap.add_argument("--no-verify", action="store_true", help="Disable verifier pass (default: verify on)")
    args = ap.parse_args()

    temporal_root = Path(args.temporal_root).expanduser().resolve()
    repo = Path(args.repo).expanduser().resolve()
    ts_path = temporal_root / "timeseries.json"
    if not ts_path.exists():
        raise FileNotFoundError(f"timeseries.json not found: {ts_path}")

    ts = read_json(ts_path)
    revisions = ts.get("revisions") or []
    if not revisions:
        raise RuntimeError(f"No revisions found in {ts_path}")

    by_num = {int(r.get("revision_number")): r for r in revisions if isinstance(r, dict) and r.get("revision_number")}
    if args.new not in by_num or args.old not in by_num:
        raise ValueError("Requested revisions not found in timeseries.json (check --new/--old values)")

    new_rev = by_num[args.new]
    old_rev = by_num[args.old]

    interp_root = temporal_root / "INPUT_INTERPRETATION"
    single_root = interp_root / "SINGLE_REVISION_ANALYSIS_DATA"
    if not single_root.exists():
        raise FileNotFoundError(f"Missing interpretation bundle: {single_root} (run backfill_temporal_payloads.py first)")

    def rev_dir_for(n: int) -> Path:
        prefix = f"{n:02d}_"
        matches = sorted([p for p in single_root.iterdir() if p.is_dir() and p.name.startswith(prefix)])
        if not matches:
            raise FileNotFoundError(f"Revision folder not found for revision_number={n} under {single_root}")
        return matches[0]

    new_dir = rev_dir_for(args.new)
    old_dir = rev_dir_for(args.old)

    new_out = new_dir / "OutputData"
    old_out = old_dir / "OutputData"
    new_payload = read_json(new_out / "interpretation_payload.json")
    old_payload = read_json(old_out / "interpretation_payload.json")

    # Evidence graph diff (matrix.json) if available.
    evidence_dir = interp_root / "EVIDENCE_GRAPH_DIFF"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / f"evidence_graph_diff_new{args.new}_old{args.old}.json"
    evidence_summary: Dict[str, Any] = {}
    if build_graph_diff and find_matrix_json and read_matrix_json:
        # Reuse precomputed diff if available (from backfill_temporal_payloads.py).
        if evidence_path.exists():
            full = read_json(evidence_path)
            evidence_summary = {
                "nodes": full.get("nodes"),
                "edges": full.get("edges"),
                "edges_added_sample": (full.get("edges_added_sample") or [])[:10],
                "edges_removed_sample": (full.get("edges_removed_sample") or [])[:10],
                "fan_in_delta_top": (full.get("fan_in_delta_top") or [])[:10],
                "fan_out_delta_top": (full.get("fan_out_delta_top") or [])[:10],
                "scc_new": full.get("scc_new"),
                "scc_old": full.get("scc_old"),
            }
        new_matrix_path = find_matrix_json(new_out)
        old_matrix_path = find_matrix_json(old_out)
        if not evidence_summary and new_matrix_path and old_matrix_path:
            new_matrix = read_matrix_json(new_matrix_path)
            old_matrix = read_matrix_json(old_matrix_path)
            diff = build_graph_diff(new_matrix, old_matrix)
            diff["meta"] = {
                "new_output": str(new_out),
                "old_output": str(old_out),
                "new_matrix": str(new_matrix_path),
                "old_matrix": str(old_matrix_path),
            }
            evidence_path.write_text(json.dumps(diff, indent=2), encoding="utf-8")
            evidence_summary = {
                "nodes": diff.get("nodes"),
                "edges": diff.get("edges"),
                "edges_added_sample": (diff.get("edges_added_sample") or [])[:10],
                "edges_removed_sample": (diff.get("edges_removed_sample") or [])[:10],
                "fan_in_delta_top": (diff.get("fan_in_delta_top") or [])[:10],
                "fan_out_delta_top": (diff.get("fan_out_delta_top") or [])[:10],
                "scc_new": diff.get("scc_new"),
                "scc_old": diff.get("scc_old"),
            }

    new_drh_path = find_drh_json(new_out)
    old_drh_path = find_drh_json(old_out)
    new_drh = read_json(new_drh_path) if new_drh_path else {}
    old_drh = read_json(old_drh_path) if old_drh_path else {}

    new_flat = flatten_drh(new_drh) if new_drh else {}
    old_flat = flatten_drh(old_drh) if old_drh else {}

    commit_analyzer = CommitAnalyzer(str(repo))
    start_date = (old_rev.get("commit_date") or "")[:10]
    end_date = (new_rev.get("commit_date") or "")[:10]
    commit_summary = commit_analyzer.get_summary_between_revisions(start_date, end_date, limit=60) if start_date and end_date else {}

    # Pull extra M-Score module penalty detail from the raw components file (if present).
    new_mscore_components = {}
    comp_path = new_out / "metrics" / "mscore_from_dsm_drh_components.json"
    if comp_path.exists():
        new_mscore_components = read_json(comp_path)

    context = {
        "repo": ts.get("repo"),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "newer": {
            "revision_number": args.new,
            "dir": new_dir.name,
            "commit_hash": new_rev.get("commit_hash"),
            "commit_date": new_rev.get("commit_date"),
            "commit_message": new_rev.get("commit_message"),
            "metrics": new_rev.get("metrics"),
            "payload_highlights": {
                "anti_pattern_counts": new_payload.get("anti_pattern_counts") or {},
                "dangerous_files_top": top_dangerous(
                    (new_payload.get("dangerous_files") or {}) if (new_payload.get("dangerous_files") or {}).get("rows")
                    else (new_payload.get("structural_hotspots") or {}),
                    n=7
                ),
                "churn_top": (new_payload.get("churn_top") or [])[:10],
                "issue_typed_churn": new_payload.get("issue_typed_churn") or {},
                "mscore_penalty_modules": top_module_penalties(new_mscore_components, n=7),
            },
            "drh_summary": summarize_drh(new_flat),
            "drh_source": str(new_drh_path) if new_drh_path else None,
        },
        "older": {
            "revision_number": args.old,
            "dir": old_dir.name,
            "commit_hash": old_rev.get("commit_hash"),
            "commit_date": old_rev.get("commit_date"),
            "commit_message": old_rev.get("commit_message"),
            "metrics": old_rev.get("metrics"),
            "payload_highlights": {
                "anti_pattern_counts": old_payload.get("anti_pattern_counts") or {},
                "dangerous_files_top": top_dangerous(
                    (old_payload.get("dangerous_files") or {}) if (old_payload.get("dangerous_files") or {}).get("rows")
                    else (old_payload.get("structural_hotspots") or {}),
                    n=7
                ),
                "churn_top": (old_payload.get("churn_top") or [])[:10],
                "issue_typed_churn": old_payload.get("issue_typed_churn") or {},
            },
            "drh_summary": summarize_drh(old_flat),
            "drh_source": str(old_drh_path) if old_drh_path else None,
        },
        "evidence_graph_diff": evidence_summary,
        "evidence_graph_diff_path": str(evidence_path.relative_to(interp_root)) if evidence_summary else None,
        "drh_diff": diff_drh(old_flat, new_flat) if old_flat and new_flat else {"error": "Missing DRH JSON(s)"},
        "commit_context": commit_summary,
    }

    model_safe = args.model.replace("/", "_").replace(":", "_")
    old_date = (old_rev.get("commit_date") or "")[:7]   # "YYYY-MM"
    new_date = (new_rev.get("commit_date") or "")[:7]   # "YYYY-MM"
    new_hash = (new_rev.get("commit_hash") or "")[:7]   # 7-char short hash
    default_out = interp_root / f"drh_diff_report_{model_safe}_{old_date}_to_{new_date}_{new_hash}_new{args.new}_old{args.old}.md"
    out_path = Path(args.output).expanduser().resolve() if args.output else default_out
    prompt_path = out_path.with_suffix(".prompt.txt")

    prompt = build_prompt(context)
    prompt_path.write_text(prompt, encoding="utf-8")

    if args.no_llm:
        managers_special_md, _ = build_managers_special(context)
        drh_md = build_drh_differences_deterministic(context)
        evidence_md = build_metrics_and_evidence_deterministic(context)
        old_date_full = (old_rev.get("commit_date") or "")[:10]
        new_date_full = (new_rev.get("commit_date") or "")[:10]
        old_hash_short = (old_rev.get("commit_hash") or "")[:7]
        new_hash_short = (new_rev.get("commit_hash") or "")[:7]
        old_msg = (old_rev.get("commit_message") or "").split("\n")[0][:80]
        new_msg = (new_rev.get("commit_message") or "").split("\n")[0][:80]
        date_header = (
            f"# Transition Report: old=rev{args.old} → new=rev{args.new}\n"
            f"- **older** (rev{args.old}): {old_date_full} `{old_hash_short}` — {old_msg}\n"
            f"- **newer** (rev{args.new}): {new_date_full} `{new_hash_short}` — {new_msg}\n\n"
        )
        out_path.write_text(
            date_header
            + managers_special_md
            + "\n"
            + drh_md
            + "\n"
            + evidence_md
            + "\n"
            + "## Likely Drivers\n\n(prompt-only mode; see prompt file)\n"
            + f"Prompt written to: `{prompt_path}`\n",
            encoding="utf-8",
        )
        print(f"Wrote prompt: {prompt_path}")
        print(f"Wrote placeholder report: {out_path}")
        return 0

    print(f"Querying Ollama model: {args.model}")
    response = query_ollama(args.model, prompt, timeout_s=args.ollama_timeout_s)
    if not response:
        response = "No response from model (check Ollama is running and the model is available)."
    response = normalize_output(response)

    # One retry if the model ignored required heading.
    if "## Likely Drivers" not in response:
        retry_prompt = (
            prompt
            + "\n\nYour previous answer violated the required headings. "
            + "Return ONLY the required sections and headings, starting with:\n"
            + "## Likely Drivers\n"
        )
        retry = query_ollama(args.model, retry_prompt, timeout_s=args.ollama_timeout_s)
        retry = normalize_output(retry)
        if "## Likely Drivers" in retry:
            response = retry

    # Keep ONLY the Likely Drivers section (drop spillover like "## Risks", "## Next Steps", etc).
    section = extract_h2_section(response, "## Likely Drivers")
    if section:
        response = sanitize_likely_drivers_section(section)
    else:
        # Last resort: wrap whatever we got.
        response = sanitize_likely_drivers_section("## Likely Drivers\n" + response.strip())

    managers_special_md, _ = build_managers_special(context)
    drh_md = build_drh_differences_deterministic(context)
    evidence_md = build_metrics_and_evidence_deterministic(context)

    # Date header — makes every report self-describing
    old_date_full = (old_rev.get("commit_date") or "")[:10]
    new_date_full = (new_rev.get("commit_date") or "")[:10]
    old_hash_short = (old_rev.get("commit_hash") or "")[:7]
    new_hash_short = (new_rev.get("commit_hash") or "")[:7]
    old_msg = (old_rev.get("commit_message") or "").split("\n")[0][:80]
    new_msg = (new_rev.get("commit_message") or "").split("\n")[0][:80]
    date_header = (
        f"# Transition Report: old=rev{args.old} → new=rev{args.new}\n"
        f"- **older** (rev{args.old}): {old_date_full} `{old_hash_short}` — {old_msg}\n"
        f"- **newer** (rev{args.new}): {new_date_full} `{new_hash_short}` — {new_msg}\n\n"
    )

    out_path.write_text(date_header + managers_special_md + "\n\n" + drh_md + "\n" + evidence_md + "\n" + response, encoding="utf-8")
    print(f"Wrote report: {out_path}")
    print(f"Wrote prompt: {prompt_path}")
    verify = not args.no_verify or args.verify
    if verify:
        verifier = Path(__file__).with_name("verify_interpretation_report.py")
        if verifier.exists():
            vcmd = [
                sys.executable,
                str(verifier),
                "--report",
                str(out_path),
                "--prompt",
                str(prompt_path),
            ]
            subprocess.run(vcmd, check=False)
        else:
            print(f"Verifier script not found: {verifier}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
