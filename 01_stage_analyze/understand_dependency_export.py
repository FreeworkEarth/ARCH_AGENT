#!/usr/bin/env python3
"""Export dependencies from a SciTools Understand database.

Run this with Understand's bundled `upython` so the `understand` module is
available. The output is a DV8-style dependency JSON accepted by
`dv8-console core:convert-matrix`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REF_KIND_MAP = {
    "call": "Call",
    "use": "Use",
    "set": "Use",
    "modify": "Use",
    "import": "Import",
    "include": "Import",
    "extend": "Extend",
    "inherit": "Extend",
    "implement": "Implement",
    "type": "Use",
    "typed": "Use",
    "create": "Create",
    "override": "Override",
    "throw": "Throw",
    "raise": "Throw",
    "couple": "Use",
}

SKIP_KIND_PARTS = (
    "ambiguous",
    "contain",
    "define",
    "declare",
    "end",
    "has",
    "unnamed",
    "inactive",
)


def _safe_call(obj: Any, method: str, default: Any = None) -> Any:
    try:
        return getattr(obj, method)()
    except Exception:
        return default


def _kind_name(obj: Any) -> str:
    kind = _safe_call(obj, "kindname", "")
    if kind:
        return str(kind)
    kind = _safe_call(obj, "kind", "")
    return str(kind or "")


def _canonical_ref_kind(raw: str) -> str | None:
    lower = raw.lower().strip()
    if not lower:
        return None
    if any(part in lower for part in SKIP_KIND_PARTS):
        return None
    # Reverse references such as Callby/Useby describe incoming edges. For
    # file-level DSM export we keep only the direct outgoing relation names.
    if lower.endswith("by") or lower.endswith("by implicit"):
        return None
    for marker, canonical in REF_KIND_MAP.items():
        if marker in lower:
            return canonical
    return None


def _file_name(file_ent: Any, source_root: Path | None = None) -> str | None:
    if file_ent is None:
        return None
    for method in ("relname", "longname", "name"):
        value = _safe_call(file_ent, method, "")
        if value:
            path = Path(str(value))
            if source_root and path.is_absolute():
                try:
                    return path.resolve().relative_to(source_root).as_posix()
                except Exception:
                    return None
            return str(value).replace("\\", "/")
    return None


def _definition_file(ent: Any, source_root: Path | None = None) -> str | None:
    # Prefer definition/reference locations for entities that are not files.
    for filt in ("definein", "declarein"):
        try:
            ref = ent.ref(filt)
        except Exception:
            ref = None
        if ref is not None:
            file_name = _file_name(_safe_call(ref, "file", None), source_root)
            if file_name:
                return file_name
    # File entities are their own definition site.
    if "file" in _kind_name(ent).lower():
        return _file_name(ent, source_root)
    return None


def _kind_lower(ent: Any) -> str:
    return _kind_name(ent).lower()


def _is_kind(ent: Any, *markers: str) -> bool:
    lower = _kind_lower(ent)
    return any(marker in lower for marker in markers)


def _nearest_parent(ent: Any, *markers: str) -> Any | None:
    parent = _safe_call(ent, "parent", None)
    while parent is not None:
        if _is_kind(parent, *markers):
            return parent
        parent = _safe_call(parent, "parent", None)
    return None


def _simple_name(ent: Any) -> str:
    for method in ("simplename", "name"):
        value = _safe_call(ent, method, "")
        if value:
            return str(value)
    longname = _safe_call(ent, "longname", "")
    return str(longname).split(".")[-1].split("/")[-1]


def _entity_name(ent: Any, source_root: Path | None = None) -> str | None:
    """Map an Understand entity to the DV8/NeoDepends-style entity name."""
    if ent is None:
        return None
    rel_file = _definition_file(ent, source_root)
    if not rel_file:
        return None

    kind = _kind_lower(ent)
    if "file" in kind:
        return f"{rel_file}/self (File)"

    direct_parent = _safe_call(ent, "parent", None)
    class_parent = _nearest_parent(ent, "class", "interface", "enum")
    ent_name = _simple_name(ent)

    if _is_kind(ent, "class", "interface", "enum"):
        return f"{rel_file}/{ent_name}/self (Class)"

    if _is_kind(ent, "constructor"):
        if class_parent is None:
            return None
        return f"{rel_file}/{_simple_name(class_parent)}/constructors/{ent_name} (Constructor)"

    if _is_kind(ent, "method", "function", "procedure", "subprogram", "routine"):
        if class_parent is not None:
            if ent_name in {"__init__", _simple_name(class_parent)}:
                return f"{rel_file}/{_simple_name(class_parent)}/constructors/{ent_name} (Constructor)"
            return f"{rel_file}/{_simple_name(class_parent)}/methods/{ent_name} (Method)"
        return f"{rel_file}/functions/{ent_name} (Function)"

    if _is_kind(ent, "attribute", "field") or (
        _is_kind(ent, "variable") and direct_parent is not None and _is_kind(direct_parent, "class", "interface", "enum")
    ):
        if direct_parent is None or not _is_kind(direct_parent, "class", "interface", "enum"):
            return None
        return f"{rel_file}/{_simple_name(direct_parent)}/fields/{ent_name} (Field)"

    return None


def _same_file(ref: Any, rel_file: str | None, source_root: Path | None = None) -> bool:
    if not rel_file:
        return False
    ref_file = _file_name(_safe_call(ref, "file", None), source_root)
    return ref_file == rel_file


def _normalize_dep_type(dep_type: str, target_ent: Any) -> str:
    if dep_type == "Call" and target_ent is not None and _is_kind(target_ent, "class"):
        return "Create"
    if dep_type in {"Set", "Modify", "Type", "Typed", "Couple"}:
        return "Use"
    return dep_type


def export_file_dependencies(db: Any, source_root: Path | None, project_name: str, und_db: Path) -> dict[str, Any]:
    root = source_root.resolve() if source_root else None

    file_names: set[str] = set()
    for file_ent in db.ents("file"):
        name = _file_name(file_ent, root)
        if name:
            file_names.add(name)

    edge_values: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # Iterate non-file entities as dependency targets. Each direct reference to
    # that target gives us a source file -> target definition file edge.
    for ent in db.ents():
        dest_file = _definition_file(ent, root)
        if not dest_file:
            continue
        refs = _safe_call(ent, "refs", [])
        for ref in refs or []:
            dep_type = _canonical_ref_kind(_kind_name(ref))
            if not dep_type:
                continue
            dep_type = _normalize_dep_type(dep_type, ent)
            src_file = _file_name(_safe_call(ref, "file", None), root)
            if not src_file or src_file == dest_file:
                continue
            file_names.add(src_file)
            file_names.add(dest_file)
            edge_values[(src_file, dest_file)][dep_type] += 1.0

    variables = sorted(file_names)
    index = {name: i for i, name in enumerate(variables)}
    cells = []
    for (src, dest), values in sorted(edge_values.items()):
        if src not in index or dest not in index:
            continue
        cells.append({
            "src": index[src],
            "dest": index[dest],
            "values": dict(sorted(values.items())),
        })

    return {
        "@schemaVersion": "1.0",
        "name": project_name,
        "variables": variables,
        "cells": cells,
        "metadata": {
            "extractor": "scitools-understand",
            "granularity": "file",
            "und_db": str(und_db),
        },
    }


def export_entity_dependencies(db: Any, source_root: Path | None, project_name: str, und_db: Path) -> dict[str, Any]:
    root = source_root.resolve() if source_root else None
    edge_values: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    variables: set[str] = set()

    for src_ent in db.ents():
        src_name = _entity_name(src_ent, root)
        if not src_name:
            continue
        src_file = _definition_file(src_ent, root)
        variables.add(src_name)

        for ref in _safe_call(src_ent, "refs", []) or []:
            dep_type = _canonical_ref_kind(_kind_name(ref))
            if not dep_type:
                continue
            if not _same_file(ref, src_file, root):
                continue

            dest_ent = _safe_call(ref, "ent", None)
            dest_name = _entity_name(dest_ent, root)
            if not dest_name or dest_name == src_name:
                continue

            dep_type = _normalize_dep_type(dep_type, dest_ent)
            variables.add(dest_name)
            edge_values[(src_name, dest_name)][dep_type] += 1.0

    ordered_variables = sorted(variables)
    index = {name: i for i, name in enumerate(ordered_variables)}
    cells = []
    for (src, dest), values in sorted(edge_values.items()):
        cells.append({
            "src": index[src],
            "dest": index[dest],
            "values": dict(sorted(values.items())),
        })

    return {
        "@schemaVersion": "1.0",
        "name": project_name,
        "variables": ordered_variables,
        "cells": cells,
        "metadata": {
            "extractor": "scitools-understand",
            "granularity": "entity",
            "und_db": str(und_db),
        },
    }


def export_dependencies(
    und_db: Path,
    source_root: Path | None,
    project_name: str,
    granularity: str = "entity",
) -> dict[str, Any]:
    try:
        import understand  # type: ignore
    except Exception as exc:  # pragma: no cover - only available in upython
        raise SystemExit(
            "Could not import SciTools Understand Python API. "
            "Run this script with Understand's upython executable."
        ) from exc

    db = understand.open(str(und_db))
    if granularity == "file":
        return export_file_dependencies(db, source_root, project_name, und_db)
    return export_entity_dependencies(db, source_root, project_name, und_db)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Understand dependencies as DV8-style JSON.")
    parser.add_argument("--und-db", required=True, help="Path to the .und database.")
    parser.add_argument("--source-root", help="Source root for relative paths.")
    parser.add_argument("--output", required=True, help="Output DV8-style dependency JSON.")
    parser.add_argument("--project-name", default="understand-export", help="Project name in output JSON.")
    parser.add_argument(
        "--granularity",
        choices=["entity", "file"],
        default="entity",
        help="Export entity-level dependencies, or collapse to file-level dependencies.",
    )
    args = parser.parse_args()

    und_db = Path(args.und_db).expanduser().resolve()
    source_root = Path(args.source_root).expanduser().resolve() if args.source_root else None
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = export_dependencies(und_db, source_root, args.project_name, args.granularity)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote {output} ({len(payload['variables'])} variables, "
        f"{len(payload['cells'])} edges, granularity={args.granularity})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
