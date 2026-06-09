#!/usr/bin/env python3
"""Compare Understand and NeoDepends dependency extraction on the same source root."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import dv8_agent


DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[1] / "REPOS_ANALYZED"


def _slug(value: str) -> str:
    out = []
    for ch in value.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {"-", "_"}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "analysis"


def _default_analysis_name(source_root: Path, lang: str) -> str:
    parts = [lang, source_root.name]
    parent_names = {p.name.lower() for p in source_root.parents}
    if "000_toy_examples" in parent_names or "arch_analysis_trainticket_toy_examples_multilang" in parent_names:
        parts.insert(0, "toy")
    return _slug("_".join(parts))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Could not read JSON {path}: {exc}") from exc


def _normalize_item_name(value: Any, source_root: Path | None = None) -> str:
    name = str(value).strip().replace("\\", "/")
    if name.endswith(" (File)"):
        name = name[: -len(" (File)")]
    if name.endswith("/self"):
        name = name[: -len("/self")]
    if source_root and name.startswith(f"{source_root.name}/"):
        name = name[len(source_root.name) + 1 :]
    return name


def _edge_key_set(
    dep_json: Path,
    source_root: Path | None = None,
) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    data = _load_json(dep_json)
    variables = data.get("variables") or []
    if not isinstance(variables, list):
        raise SystemExit(f"{dep_json} has no variables list")

    edge_pairs: set[tuple[str, str]] = set()
    typed_edges: set[tuple[str, str, str]] = set()
    for cell in data.get("cells") or data.get("matrix") or []:
        if not isinstance(cell, dict):
            continue
        src_i = cell.get("src")
        dest_i = cell.get("dest")
        try:
            src = _normalize_item_name(variables[int(src_i)], source_root)
            dest = _normalize_item_name(variables[int(dest_i)], source_root)
        except Exception:
            continue
        edge_pairs.add((src, dest))
        values = cell.get("values") or {}
        if isinstance(values, dict) and values:
            for dep_type in values:
                typed_edges.add((src, dest, str(dep_type)))
        else:
            typed_edges.add((src, dest, "Dependency"))
    return edge_pairs, typed_edges


def _write_edge_list(path: Path, edges: set[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(edge) for edge in sorted(edges)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def compare(understand_json: Path, neodepends_json: Path, out_dir: Path, source_root: Path | None = None) -> dict[str, Any]:
    u_pairs, u_typed = _edge_key_set(understand_json, source_root)
    n_pairs, n_typed = _edge_key_set(neodepends_json, source_root)

    common_pairs = u_pairs & n_pairs
    only_understand_pairs = u_pairs - n_pairs
    only_neodepends_pairs = n_pairs - u_pairs

    common_typed = u_typed & n_typed
    only_understand_typed = u_typed - n_typed
    only_neodepends_typed = n_typed - u_typed

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_edge_list(out_dir / "common_edges.tsv", common_pairs)
    _write_edge_list(out_dir / "only_understand_edges.tsv", only_understand_pairs)
    _write_edge_list(out_dir / "only_neodepends_edges.tsv", only_neodepends_pairs)
    _write_edge_list(out_dir / "common_typed_edges.tsv", common_typed)
    _write_edge_list(out_dir / "only_understand_typed_edges.tsv", only_understand_typed)
    _write_edge_list(out_dir / "only_neodepends_typed_edges.tsv", only_neodepends_typed)

    pair_union = u_pairs | n_pairs
    typed_union = u_typed | n_typed
    summary = {
        "understand_json": str(understand_json),
        "neodepends_json": str(neodepends_json),
        "normalization": {
            "source_root_name_stripped": source_root.name if source_root else None,
            "neodepends_file_suffix_stripped": "/self (File)",
        },
        "pair_edges": {
            "understand": len(u_pairs),
            "neodepends": len(n_pairs),
            "common": len(common_pairs),
            "only_understand": len(only_understand_pairs),
            "only_neodepends": len(only_neodepends_pairs),
            "jaccard": round(len(common_pairs) / len(pair_union), 6) if pair_union else 1.0,
        },
        "typed_edges": {
            "understand": len(u_typed),
            "neodepends": len(n_typed),
            "common": len(common_typed),
            "only_understand": len(only_understand_typed),
            "only_neodepends": len(only_neodepends_typed),
            "jaccard": round(len(common_typed) / len(typed_union), 6) if typed_union else 1.0,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Understand and NeoDepends, then compare file-level dependency edges.")
    parser.add_argument("--repo", required=True, help="Source repo/root to analyze.")
    parser.add_argument("--language", choices=["java", "python"], help="Language override.")
    parser.add_argument("--understand-language", help="Understand language name, e.g. Python or Java.")
    parser.add_argument(
        "--understand-granularity",
        choices=["file", "entity"],
        default="file",
        help="Understand export granularity for this comparison (default: file, to compare against NeoDepends file DSM).",
    )
    parser.add_argument("--understand-und", help="Path to Understand und.")
    parser.add_argument("--understand-upython", help="Path to Understand upython.")
    parser.add_argument("--neodepends-root", help="Path to NeoDepends root.")
    parser.add_argument("--neodepends-bin", help="Path to NeoDepends binary.")
    parser.add_argument(
        "--neodepends-resolver",
        choices=["depends", "stackgraphs"],
        default="stackgraphs",
        help="NeoDepends resolver (default: stackgraphs).",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate extractor outputs.")
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help=f"Root for generated analysis artifacts (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument("--analysis-name", help="Analysis folder name under --results-root.")
    parser.add_argument(
        "--output-dir",
        help="Full analysis output directory. Overrides --results-root and --analysis-name.",
    )
    args = parser.parse_args()

    source_root = Path(args.repo).expanduser().resolve()
    if not source_root.exists():
        raise SystemExit(f"Repo/source root not found: {source_root}")

    lang = dv8_agent.detect_language(source_root, args.language)
    if lang == "python":
        adjusted = dv8_agent.auto_adjust_python_root(source_root)
        if adjusted != source_root:
            print(f"Auto-adjusted Python source root: {adjusted}")
            source_root = adjusted

    if args.output_dir:
        analysis_dir = Path(args.output_dir).expanduser().resolve()
    else:
        results_root = Path(args.results_root).expanduser().resolve()
        analysis_name = args.analysis_name or _default_analysis_name(source_root, lang)
        analysis_dir = results_root / "dependency_extractor_comparison" / analysis_name

    extractors_dir = analysis_dir / "extractors"
    understand_dir = extractors_dir / "understand"
    neodepends_dir = extractors_dir / "neodepends"
    compare_dir = analysis_dir / "comparison"

    if args.force:
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir, ignore_errors=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    project_name = dv8_agent.guess_project_name(source_root)
    understand_json = dv8_agent.run_understand_export(
        source_root=source_root,
        output_dir=understand_dir,
        project_name=project_name,
        language=dv8_agent.understand_language_name(lang, args.understand_language),
        und_bin=args.understand_und,
        upython_bin=args.understand_upython,
        granularity=args.understand_granularity,
        force=args.force,
    )

    nd_root = dv8_agent.resolve_neodepends_root(args.neodepends_root)
    if not nd_root:
        raise SystemExit("NeoDepends root not found. Set --neodepends-root or NEODEPENDS_ROOT.")
    neodepends_json = dv8_agent.run_neodepends_python_export(
        source_root=source_root,
        output_dir=neodepends_dir,
        neodepends_root=nd_root,
        neodepends_bin=args.neodepends_bin,
        resolver=args.neodepends_resolver,
        config="default",
        langs=lang,
    )

    summary = compare(understand_json, neodepends_json, compare_dir, source_root)
    print(json.dumps(summary, indent=2))
    print(f"Analysis artifacts: {analysis_dir}")
    print(f"Comparison artifacts: {compare_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
