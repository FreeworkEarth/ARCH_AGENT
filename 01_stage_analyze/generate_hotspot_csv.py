"""
Generate active-hotspot-roi.csv from existing DV8 outputs when arch-report
hotspot generation fails (e.g. DV8 NPE in createVersionsLanguageOverview).

Inputs (all already present after a normal DV8 run):
  - OutputData/dsm/matrix.json          — structural dependency matrix (fan-in per file)
  - OutputData/history-dsm/change-cost/all-file-change-cost.csv  — churn per file

Output:
  - OutputData/hotspot/active-hotspot-roi.csv  (same format as DV8's output)

ROI formula:  NormFi × NormChurn  (both normalised 0-1 against max in the set)
This approximates DV8's hotspot ROI score.
"""

import csv
import json
import collections
from pathlib import Path


def compute_fan_in(matrix_json: Path) -> dict:
    """Return {filename_basename: fan_in_count} from dsm/matrix.json."""
    data = json.loads(matrix_json.read_text(encoding="utf-8"))
    variables = data.get("variables", [])
    # fan_in[i] = number of cells where dest==i (i.e. files that depend on variable i)
    fan_in = collections.defaultdict(int)
    for cell in data.get("cells", []):
        dest = cell.get("dest")
        if dest is not None and cell.get("values"):
            total_w = sum(cell["values"].values())
            if total_w > 0:
                fan_in[dest] += 1
    result = {}
    for idx, var in enumerate(variables):
        # var is a full path like "org/apache/commons/io/IOUtils.java"
        result[var] = fan_in.get(idx, 0)
    return result


def compute_churn(change_cost_csv: Path) -> dict:
    """Return {filename_basename: total_churn_lines} from all-file-change-cost.csv."""
    churn = collections.defaultdict(int)
    with open(change_cost_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fname = row.get("FileName", "")
            try:
                churn[fname] += int(row.get("Churn", 0) or 0)
            except (ValueError, TypeError):
                pass
    return dict(churn)


def generate_hotspot_csv(output_dir: Path, force: bool = False) -> bool:
    """
    Generate active-hotspot-roi.csv in output_dir/hotspot/ if not already present.
    Returns True if the file was written, False otherwise.
    """
    hotspot_dir = output_dir / "hotspot"
    out_csv = hotspot_dir / "active-hotspot-roi.csv"

    if out_csv.exists() and not force:
        return False  # already present

    # Find matrix.json
    matrix_json = output_dir / "dsm" / "matrix.json"
    if not matrix_json.exists():
        # Try rglob fallback
        candidates = list(output_dir.rglob("matrix.json"))
        if not candidates:
            return False
        matrix_json = candidates[0]

    # Find all-file-change-cost.csv
    change_cost = output_dir / "history-dsm" / "change-cost" / "all-file-change-cost.csv"
    if not change_cost.exists():
        candidates = list(output_dir.rglob("all-file-change-cost.csv"))
        if not candidates:
            return False
        change_cost = candidates[0]

    fan_in_by_path = compute_fan_in(matrix_json)
    churn_by_path = compute_churn(change_cost)

    if not fan_in_by_path:
        return False

    # Aggregate churn: match by basename since change-cost uses src/ paths
    # Build a basename→full path mapping for fan_in entries
    churn_by_basename = collections.defaultdict(int)
    for fname, churn in churn_by_path.items():
        base = Path(fname).name
        churn_by_basename[base] += churn

    # Build unified rows: one per file in the structural DSM
    rows = []
    for full_path, fi in fan_in_by_path.items():
        base = Path(full_path).name
        churn = churn_by_basename.get(base, 0)
        rows.append({
            "FileName": full_path,
            "Fi": fi,
            "Churn": churn,
        })

    if not rows:
        return False

    # Normalise Fi and Churn to 0-1
    max_fi = max(r["Fi"] for r in rows) or 1
    max_churn = max(r["Churn"] for r in rows) or 1

    for r in rows:
        r["NormFi"] = round(r["Fi"] / max_fi, 4)
        r["NormChurn"] = round(r["Churn"] / max_churn, 4)
        r["NormScore"] = round(r["NormFi"] * r["NormChurn"], 6)

    # Sort by NormScore descending
    rows.sort(key=lambda r: -r["NormScore"])

    hotspot_dir.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["FileName", "Fi", "NormFi", "Churn", "NormChurn", "NormScore"])
        writer.writeheader()
        writer.writerows(rows)

    return True


def process_temporal_root(temporal_root: Path, force: bool = False) -> int:
    """Run generate_hotspot_csv on all revision OutputData folders under temporal_root."""
    data_repos = temporal_root / "data_repositories"
    search_root = data_repos if data_repos.is_dir() else temporal_root
    rev_dirs = sorted([p for p in search_root.iterdir() if p.is_dir() and p.name[:2].isdigit()])

    written = 0
    for rev_dir in rev_dirs:
        out_dir = rev_dir / "OutputData"
        if not out_dir.is_dir():
            continue
        result = generate_hotspot_csv(out_dir, force=force)
        status = "written" if result else "skipped (already exists or missing inputs)"
        print(f"  {rev_dir.name}: {status}")
        if result:
            written += 1
    return written


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate active-hotspot-roi.csv from existing DV8 outputs")
    ap.add_argument("temporal_root", help="Path to temporal_analysis_* folder")
    ap.add_argument("--force", action="store_true", help="Overwrite existing hotspot CSV")
    args = ap.parse_args()

    root = Path(args.temporal_root).resolve()
    print(f"Processing: {root}")
    n = process_temporal_root(root, force=args.force)
    print(f"Done — wrote {n} hotspot CSV(s)")
