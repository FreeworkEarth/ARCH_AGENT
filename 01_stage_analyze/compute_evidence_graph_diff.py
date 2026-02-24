#!/usr/bin/env python3
"""
compute_evidence_graph_diff.py

Create a lightweight, evidence-focused diff between two DV8 DSM matrices.

Input: two OutputData folders (newer + older)
Output: evidence_graph_diff.json with edge deltas, hub shifts, and SCC summary
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def find_matrix_json(output_dir: Path) -> Optional[Path]:
    direct = output_dir / "dsm" / "matrix.json"
    if direct.exists():
        return direct
    found = list(output_dir.glob("**/dsm/matrix.json"))
    return found[0] if found else None


@dataclass(frozen=True)
class Edge:
    src: str
    dest: str
    kind: str
    weight: float


def parse_edges(matrix: Dict[str, Any]) -> Tuple[List[str], List[Edge]]:
    variables = matrix.get("variables") or []
    cells = matrix.get("cells") or []
    edges: List[Edge] = []
    if not isinstance(variables, list) or not isinstance(cells, list):
        return [], edges
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        src_idx = cell.get("src")
        dest_idx = cell.get("dest")
        values = cell.get("values") or {}
        if not isinstance(src_idx, int) or not isinstance(dest_idx, int):
            continue
        if src_idx < 0 or dest_idx < 0:
            continue
        if src_idx >= len(variables) or dest_idx >= len(variables):
            continue
        src = str(variables[src_idx])
        dest = str(variables[dest_idx])
        for kind, val in (values or {}).items():
            try:
                weight = float(val)
            except Exception:
                continue
            if weight <= 0:
                continue
            edges.append(Edge(src=src, dest=dest, kind=str(kind), weight=weight))
    return variables, edges


def summarize_edges(edges: Iterable[Edge]) -> Dict[str, Any]:
    total = 0.0
    by_kind: Dict[str, float] = {}
    for e in edges:
        total += e.weight
        by_kind[e.kind] = by_kind.get(e.kind, 0.0) + e.weight
    return {"total_weight": round(total, 2), "by_kind": {k: round(v, 2) for k, v in by_kind.items()}}


def edge_key(e: Edge) -> Tuple[str, str, str]:
    return (e.src, e.dest, e.kind)


def index_edges(edges: Iterable[Edge]) -> Dict[Tuple[str, str, str], float]:
    out: Dict[Tuple[str, str, str], float] = {}
    for e in edges:
        out[edge_key(e)] = out.get(edge_key(e), 0.0) + e.weight
    return out


def top_edges(edge_map: Dict[Tuple[str, str, str], float], n: int = 25) -> List[Dict[str, Any]]:
    rows = sorted(edge_map.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [
        {"src": k[0], "dest": k[1], "kind": k[2], "weight": round(v, 2)}
        for k, v in rows
    ]


def fan_counts(edges: Iterable[Edge]) -> Tuple[Dict[str, float], Dict[str, float]]:
    fan_out: Dict[str, float] = {}
    fan_in: Dict[str, float] = {}
    for e in edges:
        fan_out[e.src] = fan_out.get(e.src, 0.0) + e.weight
        fan_in[e.dest] = fan_in.get(e.dest, 0.0) + e.weight
    return fan_in, fan_out


def top_delta_rows(delta: Dict[str, float], n: int = 15) -> List[Dict[str, Any]]:
    rows = sorted(delta.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [{"node": k, "delta": round(v, 2)} for k, v in rows]


def build_graph(nodes: List[str], edges: Iterable[Edge]) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for e in edges:
        if e.src not in adj:
            adj[e.src] = []
        adj[e.src].append(e.dest)
    return adj


def tarjan_scc(nodes: List[str], edges: Iterable[Edge]) -> List[List[str]]:
    adj = build_graph(nodes, edges)
    index = 0
    stack: List[str] = []
    on_stack: Dict[str, bool] = {}
    idx_map: Dict[str, int] = {}
    low: Dict[str, int] = {}
    out: List[List[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        idx_map[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True

        for w in adj.get(v, []):
            if w not in idx_map:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif on_stack.get(w):
                low[v] = min(low[v], idx_map[w])

        if low[v] == idx_map[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                out.append(comp)

    for v in nodes:
        if v not in idx_map:
            strongconnect(v)
    return out


def scc_summary(nodes: List[str], edges: Iterable[Edge]) -> Dict[str, Any]:
    comps = tarjan_scc(nodes, edges)
    sizes = sorted([len(c) for c in comps], reverse=True)
    return {
        "scc_count": len(comps),
        "largest_scc_size": sizes[0] if sizes else 0,
        "top_sccs": [sorted(c)[:20] for c in comps[:5]],
    }


def build_diff(new_matrix: Dict[str, Any], old_matrix: Dict[str, Any]) -> Dict[str, Any]:
    new_nodes, new_edges = parse_edges(new_matrix)
    old_nodes, old_edges = parse_edges(old_matrix)

    new_idx = index_edges(new_edges)
    old_idx = index_edges(old_edges)

    added = {k: v for k, v in new_idx.items() if k not in old_idx}
    removed = {k: v for k, v in old_idx.items() if k not in new_idx}

    new_fan_in, new_fan_out = fan_counts(new_edges)
    old_fan_in, old_fan_out = fan_counts(old_edges)
    fan_in_delta = {k: new_fan_in.get(k, 0.0) - old_fan_in.get(k, 0.0) for k in set(new_fan_in) | set(old_fan_in)}
    fan_out_delta = {k: new_fan_out.get(k, 0.0) - old_fan_out.get(k, 0.0) for k in set(new_fan_out) | set(old_fan_out)}

    return {
        "nodes": {
            "new": len(new_nodes),
            "old": len(old_nodes),
            "delta": len(new_nodes) - len(old_nodes),
        },
        "edges": {
            "new": summarize_edges(new_edges),
            "old": summarize_edges(old_edges),
        },
        "edges_added_sample": top_edges(added, n=25),
        "edges_removed_sample": top_edges(removed, n=25),
        "fan_in_delta_top": top_delta_rows(fan_in_delta, n=15),
        "fan_out_delta_top": top_delta_rows(fan_out_delta, n=15),
        "scc_new": scc_summary(new_nodes, new_edges),
        "scc_old": scc_summary(old_nodes, old_edges),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute evidence-level diff between two DV8 matrices.")
    ap.add_argument("--new-output", required=True, help="Path to newer OutputData folder")
    ap.add_argument("--old-output", required=True, help="Path to older OutputData folder")
    ap.add_argument("--out", required=False, help="Output JSON path (default: <new-output>/evidence_graph_diff.json)")
    args = ap.parse_args()

    new_out = Path(args.new_output).expanduser().resolve()
    old_out = Path(args.old_output).expanduser().resolve()
    new_matrix_path = find_matrix_json(new_out)
    old_matrix_path = find_matrix_json(old_out)
    if not new_matrix_path or not old_matrix_path:
        raise FileNotFoundError("matrix.json not found in one or both OutputData folders.")

    new_matrix = read_json(new_matrix_path)
    old_matrix = read_json(old_matrix_path)
    diff = build_diff(new_matrix, old_matrix)
    diff["meta"] = {
        "new_output": str(new_out),
        "old_output": str(old_out),
        "new_matrix": str(new_matrix_path),
        "old_matrix": str(old_matrix_path),
    }

    out_path = Path(args.out).expanduser().resolve() if args.out else (new_out / "evidence_graph_diff.json")
    out_path.write_text(json.dumps(diff, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
