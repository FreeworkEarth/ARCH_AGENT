#!/usr/bin/env python3
"""Build a bounded code slice for Q2-guided automated refactoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TEST_NAME_HINTS = ("Test.", "Tests.", "_test.", "test_", "TestCase.")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _basename(value: str) -> str:
    return str(value).replace("\\", "/").split("/")[-1]


def _cluster_from_raw_path(raw_path: str) -> tuple[str, str]:
    if raw_path and raw_path[0].isupper() and not raw_path[0].isdigit():
        return raw_path[0], raw_path[1:]
    return "CORE", raw_path


def _load_risk_by_basename(risk_json: Path | None) -> dict[str, dict[str, Any]]:
    if not risk_json or not risk_json.is_file():
        return {}
    data = _load_json(risk_json)
    out = {}
    for rec in data.get("files", []):
        name = _basename(rec.get("file", ""))
        if name:
            out[name] = rec
    return out


def _discover_tests(repo: Path, edit_files: list[str]) -> list[str]:
    stems = {Path(f).stem.replace("Test", "") for f in edit_files}
    tests: list[str] = []
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if not any(hint.replace(".", "") in name for hint in TEST_NAME_HINTS):
            continue
        if any(stem and stem in name for stem in stems):
            try:
                tests.append(path.relative_to(repo).as_posix())
            except Exception:
                tests.append(path.as_posix())
    return sorted(set(tests))


def build_slice(
    *,
    arch_issue_root: Path,
    ap_type: str,
    instance_id: str,
    target_cluster: str,
    repo: Path | None,
    risk_json: Path | None,
    max_files_per_cluster: int,
) -> dict[str, Any]:
    inst_dir = arch_issue_root / ap_type / str(instance_id)
    clsx_path = inst_dir / f"{instance_id}-clsx_files.json"
    merge_path = inst_dir / f"{instance_id}-merge_deps.json"

    if not merge_path.is_file():
        raise SystemExit(f"merge_deps JSON not found: {merge_path}")

    clsx = _load_json(clsx_path)
    merge = _load_json(merge_path)
    risk_by_name = _load_risk_by_basename(risk_json)

    clusters: dict[str, list[str]] = {}
    for raw_path in merge.get("files", []):
        label, clean = _cluster_from_raw_path(str(raw_path))
        clusters.setdefault(label, []).append(clean)

    # Rank each cluster by risk signals when available; otherwise preserve DV8 order.
    ranked_clusters: dict[str, list[str]] = {}
    for label, files in clusters.items():
        def score(path: str) -> tuple[int, int, str]:
            risk = risk_by_name.get(_basename(path), {})
            sig = risk.get("signals", {})
            return (
                int(sig.get("bug_churn_total", 0)),
                int(sig.get("anti_pattern_instance_load", 0)),
                _basename(path),
            )
        ranked_clusters[label] = sorted(files, key=score, reverse=True)[:max_files_per_cluster]

    if target_cluster.lower() == "auto" or target_cluster not in ranked_clusters:
        non_core = [label for label in sorted(ranked_clusters) if label != "CORE"]
        if non_core:
            target_cluster = max(non_core, key=lambda label: len(ranked_clusters.get(label, [])))
        elif ranked_clusters:
            target_cluster = sorted(ranked_clusters)[0]

    selected_labels = ["CORE"]
    if target_cluster != "CORE":
        selected_labels.append(target_cluster)
    selected_labels = [label for label in selected_labels if label in ranked_clusters]
    selected: list[str] = []
    for label in selected_labels:
        selected.extend(ranked_clusters.get(label, []))

    read_only: list[str] = []
    for label, files in ranked_clusters.items():
        if label in selected_labels:
            continue
        read_only.extend(files[: min(3, max_files_per_cluster)])

    edit_candidates = sorted(set(selected))
    tests = _discover_tests(repo, edit_candidates) if repo and repo.is_dir() else []

    dep_types = [str(x) for x in merge.get("dep_types", [])]
    return {
        "instance": {
            "ap_type": ap_type,
            "id": str(instance_id),
            "arch_issue_root": str(arch_issue_root),
            "clsx_files_json": str(clsx_path) if clsx_path.is_file() else None,
            "merge_deps_json": str(merge_path),
            "member_count": len(clsx.get("files", [])) if isinstance(clsx.get("files"), list) else None,
        },
        "target_action": f"cut dependency types {dep_types or ['unknown']} between Cluster {target_cluster} and CORE",
        "target_cluster": target_cluster,
        "clusters": ranked_clusters,
        "selected_clusters": selected_labels,
        "dependency_types": dep_types,
        "edit_candidates": edit_candidates,
        "read_only_context": sorted(set(read_only)),
        "tests": tests,
        "acceptance": {
            "tests_must_pass": True,
            "metric_gate": [
                "target anti-pattern instance shrinks or disappears",
                "propagation-cost does not increase",
                "M-score does not decrease",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Q2 automated-refactoring slice from DV8 instance data.")
    parser.add_argument("--arch-issue-root", required=True, help="OutputData/arch-issue directory.")
    parser.add_argument("--ap-type", required=True, help="Anti-pattern type, e.g. modularity-violation.")
    parser.add_argument("--instance-id", required=True, help="DV8 instance id/folder name.")
    parser.add_argument("--target-cluster", default="auto", help="Cluster to slice with CORE, e.g. A, CORE, or auto.")
    parser.add_argument("--repo", help="Repo checkout path for test discovery.")
    parser.add_argument("--risk-json", help="INPUT_INTERPRETATION/file_risk_scores.json.")
    parser.add_argument("--max-files-per-cluster", type=int, default=8)
    parser.add_argument("--output", required=True, help="Output slice JSON.")
    args = parser.parse_args()

    payload = build_slice(
        arch_issue_root=Path(args.arch_issue_root).expanduser().resolve(),
        ap_type=args.ap_type,
        instance_id=args.instance_id,
        target_cluster=args.target_cluster,
        repo=Path(args.repo).expanduser().resolve() if args.repo else None,
        risk_json=Path(args.risk_json).expanduser().resolve() if args.risk_json else None,
        max_files_per_cluster=args.max_files_per_cluster,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote slice: {output}")
    print(f"Edit candidates: {len(payload['edit_candidates'])}; tests: {len(payload['tests'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
