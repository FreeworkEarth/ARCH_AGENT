#!/usr/bin/env python3
"""
fetch_github_issues.py
======================
Fetch issues for a repository and produce an issue_map.json for the ARCH_AGENT
pipeline.  Automatically detects whether the project uses JIRA or GitHub issues
by scanning commit messages in the git repository.

Detection logic (applied to the last 300 commits):
  - If >10 % of commits reference JIRA keys (e.g. PDFBOX-1234, IO-42, ZEPPELIN-999)
    → JIRA mode: fetch from the appropriate JIRA instance.
  - Otherwise → GitHub mode: fetch from the GitHub Issues API using labels.

Output format (compatible with load_issue_type_map in prepare_interpretation_payload.py):
    {
      "meta": { "repo": "...", "source": "jira|github|none", ... },
      "issues": {
        "#123":      "bug",      ← GitHub-style key
        "IO-42":     "bug",      ← JIRA-style key
        "PDFBOX-99": "feature",
        ...
      }
    }

Usage:
    # Auto-detect from git repo + GitHub remote
    python fetch_github_issues.py --git-root /path/to/repo --out issue_map.json

    # Force GitHub mode
    python fetch_github_issues.py --repo owner/repo --out issue_map.json

    # Force JIRA mode (Apache)
    python fetch_github_issues.py --jira-project IO --out issue_map.json

    # Auto-detect then run
    python fetch_github_issues.py --git-root /path/to/repo --verbose

Options:
    --git-root PATH      Local git repository root (used for auto-detection)
    --repo OWNER/REPO    GitHub repo slug (overrides auto-detection)
    --jira-project KEY   JIRA project key, e.g. IO, PDFBOX, ZEPPELIN (overrides auto-detection)
    --jira-url URL       JIRA base URL (default: https://issues.apache.org/jira)
    --token TOKEN        GitHub personal access token (falls back to GH_TOKEN / GITHUB_TOKEN)
    --out PATH           Output path (default: ./issue_map.json)
    --label-map JSON     Override GitHub label → type mapping
    --state STATE        GitHub issue state: open|closed|all (default: all)
    --since DATE         Only issues updated since YYYY-MM-DD
    --max-pages INT      Max API pages (100 issues/page; default: 200)
    --verbose            Print progress
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_JIRA_URL = "https://issues.apache.org/jira"

# GitHub label name → canonical type (case-insensitive)
DEFAULT_LABEL_MAP: Dict[str, str] = {
    "bug": "bug", "type: bug": "bug", "type:bug": "bug",
    "kind/bug": "bug", "bug report": "bug", "defect": "bug",
    "regression": "bug", "crash": "bug", "error": "bug",
    "fix": "bug", "hotfix": "bug",
    "feature": "feature", "enhancement": "feature",
    "type: enhancement": "feature", "type:enhancement": "feature",
    "kind/feature": "feature", "new feature": "feature",
    "improvement": "feature", "feature request": "feature",
    "documentation": "documentation", "docs": "documentation",
    "type: docs": "documentation",
    "test": "test", "testing": "test", "type: test": "test",
    "refactoring": "refactoring", "refactor": "refactoring",
    "cleanup": "refactoring", "tech debt": "refactoring",
}

# JIRA issue type → canonical type
JIRA_TYPE_MAP: Dict[str, str] = {
    "bug": "bug", "bug (jira)": "bug",
    "improvement": "feature", "new feature": "feature", "wish": "feature",
    "task": "refactoring", "sub-task": "refactoring",
    "test": "test",
    "documentation": "documentation",
    "dependency upgrade": "refactoring",
}

_JIRA_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,10})-(\d+)\b")
_GITHUB_REF_RE = re.compile(r"(?:fixes?|closes?|refs?)\s+#(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def detect_issue_source(
    git_root: Optional[Path],
    verbose: bool,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Inspect git history and remote URL to decide issue source.

    Returns: (source, jira_project_key, github_slug)
      source is one of: "jira", "github", "none"
    """
    if git_root is None or not (git_root / ".git").is_dir():
        return "none", None, None

    # 1. Sample commit messages
    try:
        result = subprocess.run(
            ["git", "log", "--format=%s", "-2000"],
            cwd=str(git_root),
            capture_output=True, text=True, timeout=30,
        )
        messages = result.stdout.splitlines()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        messages = []

    total = len(messages)
    if total == 0:
        return "none", None, None

    jira_hits: Dict[str, int] = {}  # project key → count
    github_hits = 0
    for msg in messages:
        for m in _JIRA_KEY_RE.finditer(msg):
            key = m.group(1)
            jira_hits[key] = jira_hits.get(key, 0) + 1
        if _GITHUB_REF_RE.search(msg):
            github_hits += 1

    jira_total = sum(jira_hits.values())
    jira_ratio = jira_total / total

    if verbose:
        print(f"  Auto-detect: {total} commits, JIRA hits={jira_total} ({jira_ratio:.0%}), GitHub refs={github_hits}")

    # 2. Get GitHub remote slug
    github_slug: Optional[str] = None
    try:
        r2 = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(git_root), capture_output=True, text=True, timeout=10,
        )
        remote = r2.stdout.strip()
        m = re.search(r"github\.com[:/]([^/]+/[^/.]+?)(?:\.git)?$", remote)
        if m:
            github_slug = m.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # 3. Decide: JIRA if ≥2% of commits OR ≥5 absolute JIRA refs are found.
    #    Apache projects migrating to GitHub PRs may only have a few percent
    #    of commits with JIRA keys in recent history but still use JIRA as tracker.
    if (jira_ratio >= 0.02 or jira_total >= 5) and jira_hits:
        # Pick the dominant project key (most frequently referenced)
        best_key = max(jira_hits, key=jira_hits.__getitem__)
        if verbose:
            print(f"  → JIRA mode, project={best_key} ({jira_total}/{total} = {jira_ratio:.0%}) (GitHub slug={github_slug})")
        return "jira", best_key, github_slug

    # 4. GitHub issues if we have a GitHub remote
    if github_slug:
        if verbose:
            print(f"  → GitHub mode, repo={github_slug}")
        return "github", None, github_slug

    if verbose:
        print("  → No issue tracker detected, will use keyword-only fallback")
    return "none", None, None


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _get(url: str, token: Optional[str], verbose: bool) -> Tuple[Any, Dict[str, str]]:
    headers: Dict[str, str] = {"Accept": "application/json"}
    if token and "github.com" in url:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-GitHub-Api-Version"] = "2022-11-28"

    req = urllib.request.Request(url, headers=headers)
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                return body, resp_headers
        except urllib.error.HTTPError as exc:
            if exc.code == 403 and "github" in url:
                reset_ts = int(exc.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait_s = max(1, reset_ts - int(time.time()) + 2)
                if verbose:
                    print(f"  GitHub rate-limited — waiting {wait_s}s", flush=True)
                time.sleep(wait_s)
                if attempt == 1:
                    raise
            elif exc.code >= 500 and attempt == 0:
                time.sleep(5)
            else:
                raise
    raise RuntimeError("Request failed")


def _parse_link_next(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' in part:
            url_part = part.split(";")[0].strip()
            return url_part.strip("<>")
    return None


# ---------------------------------------------------------------------------
# GitHub fetcher
# ---------------------------------------------------------------------------


def fetch_github(
    slug: str,
    token: Optional[str],
    label_map: Dict[str, str],
    state: str,
    since: Optional[str],
    max_pages: int,
    verbose: bool,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns (issue_map, summaries):
      issue_map  = { "#123": "bug", ... }
      summaries  = { "#123": "Issue title / summary text", ... }
    """
    params: Dict[str, str] = {"state": state, "per_page": "100"}
    if since:
        params["since"] = f"{since}T00:00:00Z"

    url: Optional[str] = (
        f"https://api.github.com/repos/{slug}/issues?{urllib.parse.urlencode(params)}"
    )
    issue_map: Dict[str, str] = {}
    summaries: Dict[str, str] = {}
    page = 0

    while url and page < max_pages:
        page += 1
        if verbose:
            print(f"  [github] page {page}", flush=True)
        try:
            data, headers = _get(url, token, verbose)
        except urllib.error.HTTPError as exc:
            print(f"  GitHub API {exc.code} — {url}", file=sys.stderr)
            break

        if not isinstance(data, list):
            break

        for issue in data:
            if "pull_request" in issue:
                continue
            number = issue.get("number")
            if number is None:
                continue
            key = f"#{number}"
            # Store title/summary for all issues
            title = issue.get("title", "").strip()
            if title:
                summaries[key] = title
            labels = issue.get("labels", [])
            for label in labels:
                name = label.get("name", "").lower().strip()
                if name in label_map:
                    issue_map[key] = label_map[name]
                    break

        url = _parse_link_next(headers.get("link"))

        remaining = int(headers.get("x-ratelimit-remaining", 100))
        if remaining < 5:
            reset_ts = int(headers.get("x-ratelimit-reset", time.time() + 60))
            wait_s = max(1, reset_ts - int(time.time()) + 2)
            if verbose:
                print(f"  GitHub rate limit low — waiting {wait_s}s", flush=True)
            time.sleep(wait_s)

    return issue_map, summaries


# ---------------------------------------------------------------------------
# JIRA fetcher
# ---------------------------------------------------------------------------


def fetch_jira(
    project: str,
    jira_base: str,
    max_pages: int,
    verbose: bool,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Fetch all issues for a JIRA project and classify by issue type.
    Uses the JIRA REST API v2 (no auth needed for public Apache JIRA).
    Returns (issue_map, summaries):
      issue_map  = { "IO-42": "bug", "IO-99": "feature", ... }
      summaries  = { "IO-42": "BoundedInputStream does not limit reads", ... }
    """
    issue_map: Dict[str, str] = {}
    summaries: Dict[str, str] = {}
    start_at = 0
    page_size = 100
    max_results = max_pages * page_size
    fields = "summary,issuetype,status"

    while start_at < max_results:
        jql = urllib.parse.quote(f"project={project}")
        url = (
            f"{jira_base}/rest/api/2/search"
            f"?jql={jql}"
            f"&startAt={start_at}&maxResults={page_size}&fields={fields}"
        )
        if verbose:
            print(f"  [jira] startAt={start_at}", flush=True)

        try:
            data, _ = _get(url, token=None, verbose=verbose)
        except urllib.error.HTTPError as exc:
            print(f"  JIRA API {exc.code} — {url}", file=sys.stderr)
            break
        except Exception as exc:
            print(f"  JIRA request failed: {exc}", file=sys.stderr)
            break

        issues = data.get("issues", [])
        if not issues:
            break

        for issue in issues:
            key = issue.get("key", "")
            if not key:
                continue
            fields_data = issue.get("fields", {})
            itype = (
                fields_data.get("issuetype", {})
                .get("name", "")
                .lower()
                .strip()
            )
            canonical = JIRA_TYPE_MAP.get(itype)
            if canonical:
                issue_map[key] = canonical
            # Always store summary (title) regardless of type classification
            summary = fields_data.get("summary", "").strip()
            if summary:
                summaries[key] = summary

        start_at += len(issues)
        if len(issues) < page_size:
            break  # last page

    return issue_map, summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch issues (GitHub or JIRA, auto-detected) and write issue_map.json "
            "for the ARCH_AGENT pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--git-root",
        help="Local git repository root — used for auto-detection of issue tracker.",
        default=None,
    )
    ap.add_argument(
        "--repo",
        help="GitHub repo slug owner/repo (forces GitHub mode).",
        default=None,
    )
    ap.add_argument(
        "--jira-project",
        help="JIRA project key, e.g. IO, PDFBOX (forces JIRA mode).",
        default=None,
    )
    ap.add_argument(
        "--jira-url",
        help=f"JIRA base URL (default: {DEFAULT_JIRA_URL}).",
        default=DEFAULT_JIRA_URL,
    )
    ap.add_argument(
        "--token",
        help="GitHub personal access token (falls back to GH_TOKEN / GITHUB_TOKEN env var).",
        default=None,
    )
    ap.add_argument(
        "--out", default="issue_map.json",
        help="Output path (default: ./issue_map.json).",
    )
    ap.add_argument(
        "--label-map",
        help='JSON dict overriding GitHub label → type mapping.',
        default=None,
    )
    ap.add_argument(
        "--state", choices=["open", "closed", "all"], default="all",
        help="GitHub issue state (default: all).",
    )
    ap.add_argument(
        "--since",
        help="Only fetch issues updated since YYYY-MM-DD.",
        default=None,
    )
    ap.add_argument(
        "--max-pages", type=int, default=200,
        help="Max API pages (100 issues/page; default: 200 = 20,000 issues).",
    )
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    token: Optional[str] = (
        args.token
        or os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
    )

    label_map = dict(DEFAULT_LABEL_MAP)
    if args.label_map:
        try:
            overrides = json.loads(args.label_map)
            label_map.update({k.lower().strip(): v for k, v in overrides.items()})
        except json.JSONDecodeError as exc:
            print(f"ERROR: --label-map is not valid JSON: {exc}", file=sys.stderr)
            return 1

    # --- Determine mode ---
    git_root = Path(args.git_root).resolve() if args.git_root else None

    if args.jira_project:
        source, jira_project, github_slug = "jira", args.jira_project, None
        if args.repo:
            github_slug = args.repo
    elif args.repo:
        source, jira_project, github_slug = "github", None, args.repo
    else:
        if args.verbose:
            print("Auto-detecting issue tracker from git history...", flush=True)
        source, jira_project, github_slug = detect_issue_source(git_root, args.verbose)

    if args.verbose:
        print(f"Mode: {source}  jira_project={jira_project}  github={github_slug}")

    # --- Fetch ---
    issue_map: Dict[str, str] = {}
    summaries: Dict[str, str] = {}

    if source == "jira" and jira_project:
        if args.verbose:
            print(f"Fetching JIRA issues for project {jira_project} from {args.jira_url} ...")
        issue_map, summaries = fetch_jira(
            project=jira_project,
            jira_base=args.jira_url,
            max_pages=args.max_pages,
            verbose=args.verbose,
        )
        # Also fetch GitHub issues if we have a slug (PDFBOX uses both)
        if github_slug and token:
            if args.verbose:
                print(f"Also fetching GitHub labels for {github_slug} ...")
            gh_map, gh_summaries = fetch_github(
                slug=github_slug,
                token=token,
                label_map=label_map,
                state=args.state,
                since=args.since,
                max_pages=min(args.max_pages, 20),  # supplemental only
                verbose=args.verbose,
            )
            # GitHub #NNN → bug/feature enriches the map
            issue_map.update(gh_map)
            summaries.update(gh_summaries)

    elif source == "github" and github_slug:
        if not token:
            print(
                "WARNING: No GitHub token. Rate limit is 60 req/hour.\n"
                "  Set GH_TOKEN or GITHUB_TOKEN, or pass --token.",
                file=sys.stderr,
            )
        if args.verbose:
            print(f"Fetching GitHub issues for {github_slug} ...")
        issue_map, summaries = fetch_github(
            slug=github_slug,
            token=token,
            label_map=label_map,
            state=args.state,
            since=args.since,
            max_pages=args.max_pages,
            verbose=args.verbose,
        )

    else:
        print(
            "No issue tracker detected or specified.\n"
            "  The pipeline will use keyword-based commit classification as fallback.\n"
            "  Use --git-root, --repo, or --jira-project to specify.",
            file=sys.stderr,
        )

    # --- Collect commit messages from git history (if git_root available) ---
    commit_log: List[Dict[str, str]] = []
    if git_root and git_root.is_dir():
        if args.verbose:
            print("Collecting commit messages from git history...", flush=True)
        try:
            result = subprocess.run(
                ["git", "log", "--format=%H\x1f%ai\x1f%ae\x1f%s\x1f%b", "--no-merges"],
                cwd=str(git_root),
                capture_output=True,
                text=True,
                timeout=120,
            )
            for line in result.stdout.splitlines():
                parts = line.split("\x1f", 4)
                if len(parts) >= 4:
                    commit_log.append({
                        "hash": parts[0].strip(),
                        "date": parts[1].strip(),
                        "author": parts[2].strip(),
                        "subject": parts[3].strip(),
                        "body": parts[4].strip() if len(parts) > 4 else "",
                    })
            if args.verbose:
                print(f"  Collected {len(commit_log)} commit messages", flush=True)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            print(f"  WARNING: Could not collect commit messages: {exc}", file=sys.stderr)

    # --- Write output ---
    type_counts: Dict[str, int] = {}
    for t in issue_map.values():
        type_counts[t] = type_counts.get(t, 0) + 1

    output = {
        "meta": {
            "repo": github_slug or jira_project or "unknown",
            "source": source,
            "jira_project": jira_project,
            "jira_url": args.jira_url if source == "jira" else None,
            "fetched": datetime.now().isoformat(),
            "state": args.state if source == "github" else None,
            "since": args.since,
            "total_classified": len(issue_map),
            "total_summaries": len(summaries),
            "total_commits": len(commit_log),
            "by_type": type_counts,
        },
        "issues": issue_map,
        "summaries": summaries,
        "commit_log": commit_log,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"Written {out_path}: {len(issue_map)} classified issues, {len(summaries)} summaries, {len(commit_log)} commits (source={source})")
    for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {t}: {c}")

    if not issue_map:
        print(
            "\nNo issues classified. For GitHub repos try --label-map to match your labels.\n"
            "For Apache projects JIRA is auto-detected from commit messages.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
