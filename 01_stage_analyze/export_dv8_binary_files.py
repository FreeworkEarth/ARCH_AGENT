#!/usr/bin/env python3
"""
export_dv8_binary_files.py
==========================
Convert DV8 binary anti-pattern instance files (.dv8-clsx, .dv8-dsm)
to human-readable JSON and CSV alongside the originals.

Can be run standalone or called programmatically (export_output_dir).

Usage:
    # Convert all binary files under one revision's OutputData:
    python export_dv8_binary_files.py <OutputData_path>

    # Convert all revisions under an INPUT_INTERPRETATION folder:
    python export_dv8_binary_files.py --all <INPUT_INTERPRETATION_path>

Output (written next to each binary file):
    <name>-clsx_files.json   — file membership list from .dv8-clsx
    <name>-clsx_files.csv
    <name>-sdsm_deps.json    — file list + dep-types from .dv8-dsm
    <name>-sdsm_deps.csv

The JSON/CSV outputs are intentionally simple so they are useful
both for human inspection and downstream tool parsing.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Source file extensions accepted for file-path extraction
# ---------------------------------------------------------------------------
_SOURCE_EXTS = frozenset(
    ["java", "py", "kt", "scala", "groovy", "js", "ts", "cs", "cpp", "c", "go", "rb", "h", "hpp"]
)

# Dependency type names that DV8 uses in DSM files
_DEP_TYPE_PATTERN = re.compile(
    r"\b(Implement|Cast|Set|Call|Contain|Annotation|Use|Return|Extend|Import|Throw|Create|Parameter|Override|Mixin)\b"
)


# ---------------------------------------------------------------------------
# .dv8-clsx parser
# ---------------------------------------------------------------------------

def parse_dv8_clsx(path: Path) -> List[str]:
    """
    Extract source file paths from a DV8 .dv8-clsx file.

    Format: custom header + gzip-compressed body containing
    length-prefixed UTF-8 strings (\\x01\\x00\\xNN<string>).

    Returns list of file path strings (may be empty on parse error).
    """
    try:
        raw = path.read_bytes()
        gz_start = raw.find(b"\x1f\x8b")
        if gz_start < 0:
            return []
        body = gzip.decompress(raw[gz_start:])
    except Exception:
        return []

    paths: List[str] = []
    i = 0
    while i < len(body) - 3:
        if body[i] == 0x01 and body[i + 1] == 0x00:
            length = body[i + 2]
            end = i + 3 + length
            if end <= len(body):
                try:
                    s = body[i + 3:end].decode("utf-8", errors="replace").strip()
                    if s and "." in s and not s.startswith("\x00"):
                        ext = s.rsplit(".", 1)[-1].lower()
                        if ext in _SOURCE_EXTS:
                            paths.append(s)
                except Exception:
                    pass
                i = end
                continue
        i += 1
    return paths


# ---------------------------------------------------------------------------
# .dv8-dsm parser (best-effort string extraction)
# ---------------------------------------------------------------------------

def parse_dv8_dsm(path: Path) -> Tuple[List[str], List[str]]:
    """
    Extract file paths and dependency type names from a .dv8-dsm file.

    The DSM binary format stores a sub-DSM for one anti-pattern instance.
    We extract strings by scanning the gzip-decompressed body, looking
    for known source-file extensions and DV8 dependency type keywords.

    Returns (file_paths, dep_types) — both may be empty on error.
    """
    try:
        raw = path.read_bytes()
        gz_start = raw.find(b"\x1f\x8b")
        if gz_start < 0:
            return [], []
        body = gzip.decompress(raw[gz_start:])
    except Exception:
        return [], []

    # Decode as UTF-8 (replace errors) and extract with regex
    text = body.decode("utf-8", errors="replace")

    # Extract source file paths using the same length-prefix approach as clsx,
    # then fall back to regex scan for any remaining paths.
    seen: set = set()
    cleaned: List[str] = []

    # Method 1: length-prefix scan (same format as .dv8-clsx: \x01\x00\xNN<string>)
    i = 0
    while i < len(body) - 3:
        if body[i] == 0x01 and body[i + 1] == 0x00:
            length = body[i + 2]
            end = i + 3 + length
            if end <= len(body):
                try:
                    s = body[i + 3:end].decode("utf-8", errors="replace").strip()
                    if s and "." in s and not s.startswith("\x00"):
                        ext = s.rsplit(".", 1)[-1].lower()
                        if ext in _SOURCE_EXTS:
                            if s not in seen:
                                seen.add(s)
                                cleaned.append(s)
                except Exception:
                    pass
                i = end
                continue
        i += 1

    # Method 2: regex scan for path-like strings (catches any not caught above)
    file_pattern = re.compile(
        r"[a-zA-Z][a-zA-Z0-9_$/.\-]*/"  # must have a slash (path segment)
        r"[a-zA-Z0-9_$/.\-]*"
        r"\.(?:" + "|".join(sorted(_SOURCE_EXTS)) + r")\b"
    )
    for f in file_pattern.findall(text):
        if f not in seen:
            seen.add(f)
            cleaned.append(f)

    dep_types = sorted(set(_DEP_TYPE_PATTERN.findall(text)))

    return cleaned, dep_types


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _stem_of(path: Path) -> str:
    """Return the file stem without the .dv8-* extension."""
    name = path.name
    for suffix in (".dv8-clsx", ".dv8-dsm", ".dv8-issue"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def export_clsx(clsx_path: Path, verbose: bool = False) -> Optional[Path]:
    """
    Parse a .dv8-clsx file and write <stem>_files.json + <stem>_files.csv.
    Returns the JSON output path, or None if nothing was exported.
    """
    files = parse_dv8_clsx(clsx_path)
    stem = _stem_of(clsx_path)
    out_dir = clsx_path.parent

    json_path = out_dir / f"{stem}_files.json"
    csv_path = out_dir / f"{stem}_files.csv"

    data: Dict[str, Any] = {
        "source_file": clsx_path.name,
        "anti_pattern_type": clsx_path.parent.parent.name,
        "instance_id": clsx_path.parent.name,
        "file_count": len(files),
        "files": sorted(files),
    }
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_path"])
        for f in sorted(files):
            writer.writerow([f])

    if verbose:
        print(f"    [clsx] {clsx_path.name} → {json_path.name} ({len(files)} files)")
    return json_path


def export_dsm(dsm_path: Path, verbose: bool = False) -> Optional[Path]:
    """
    Parse a .dv8-dsm file and write <stem>_deps.json + <stem>_deps.csv.
    Returns the JSON output path, or None if nothing was exported.
    """
    files, dep_types = parse_dv8_dsm(dsm_path)
    stem = _stem_of(dsm_path)
    out_dir = dsm_path.parent

    json_path = out_dir / f"{stem}_deps.json"
    csv_path = out_dir / f"{stem}_deps.csv"

    data: Dict[str, Any] = {
        "source_file": dsm_path.name,
        "anti_pattern_type": dsm_path.parent.parent.name,
        "instance_id": dsm_path.parent.name,
        "file_count": len(files),
        "dep_types": dep_types,
        "files": sorted(files),
    }
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_path", "dep_types"])
        dt_str = "|".join(dep_types)
        for f in sorted(files):
            writer.writerow([f, dt_str])

    if verbose:
        print(f"    [dsm]  {dsm_path.name} → {json_path.name} ({len(files)} files, types: {dep_types})")
    return json_path


# ---------------------------------------------------------------------------
# Directory-level export
# ---------------------------------------------------------------------------

def export_output_dir(output_dir: Path, verbose: bool = False) -> Dict[str, int]:
    """
    Walk output_dir recursively and convert all .dv8-clsx and .dv8-dsm files.

    Searches both known layout variants:
      Layout 1: output_dir/arch-issue/<ap-type>/<id>/*.dv8-clsx
      Layout 2: output_dir/**/dv8-analysis-result/anti-pattern/anti-pattern-instances/<ap-type>/<id>/

    Returns a counts dict: {"clsx": N, "dsm": N, "errors": N}.
    """
    counts = {"clsx": 0, "dsm": 0, "errors": 0}

    # Find all .dv8-clsx and .dv8-dsm files anywhere under output_dir
    for clsx_file in output_dir.rglob("*.dv8-clsx"):
        try:
            export_clsx(clsx_file, verbose=verbose)
            counts["clsx"] += 1
        except Exception as exc:
            if verbose:
                print(f"    [ERROR] {clsx_file}: {exc}", file=sys.stderr)
            counts["errors"] += 1

    for dsm_file in output_dir.rglob("*.dv8-dsm"):
        try:
            export_dsm(dsm_file, verbose=verbose)
            counts["dsm"] += 1
        except Exception as exc:
            if verbose:
                print(f"    [ERROR] {dsm_file}: {exc}", file=sys.stderr)
            counts["errors"] += 1

    return counts


def export_all_revisions(interp_root: Path, verbose: bool = False) -> None:
    """
    Walk all SINGLE_REVISION_ANALYSIS_DATA/<rev>/OutputData/ dirs and export
    binary files in each one.
    """
    rev_data_dir = interp_root / "SINGLE_REVISION_ANALYSIS_DATA"
    if not rev_data_dir.is_dir():
        print(f"  [export] No SINGLE_REVISION_ANALYSIS_DATA found under {interp_root}", file=sys.stderr)
        return

    total = {"clsx": 0, "dsm": 0, "errors": 0}
    for rev_dir in sorted(rev_data_dir.iterdir()):
        if not rev_dir.is_dir():
            continue
        out_dir = rev_dir / "OutputData"
        if not out_dir.is_dir():
            continue
        if verbose:
            print(f"  [export] {rev_dir.name}", flush=True)
        counts = export_output_dir(out_dir, verbose=verbose)
        for k in total:
            total[k] += counts[k]

    print(
        f"  [export] Done: {total['clsx']} clsx + {total['dsm']} dsm exported"
        + (f", {total['errors']} errors" if total["errors"] else ""),
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert .dv8-clsx / .dv8-dsm binary files to JSON + CSV."
    )
    ap.add_argument(
        "path",
        help="Path to an OutputData directory OR an INPUT_INTERPRETATION directory (with --all).",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Walk all revisions under INPUT_INTERPRETATION/SINGLE_REVISION_ANALYSIS_DATA/.",
    )
    ap.add_argument("--verbose", "-v", action="store_true", help="Print per-file progress.")
    args = ap.parse_args()

    target = Path(args.path).expanduser().resolve()
    if not target.is_dir():
        print(f"ERROR: {target} is not a directory.", file=sys.stderr)
        return 1

    if args.all:
        export_all_revisions(target, verbose=args.verbose)
    else:
        counts = export_output_dir(target, verbose=args.verbose)
        print(
            f"Exported {counts['clsx']} clsx + {counts['dsm']} dsm files"
            + (f", {counts['errors']} errors" if counts["errors"] else ""),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
