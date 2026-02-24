#!/usr/bin/env python3
"""
commit_analyzer.py - Extract and analyze git commit data

This module extracts commit messages, diffs, and metadata from git repos
to help explain metric changes between revisions.
"""

import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import re


class CommitAnalyzer:
    """Extract and analyze git commits between two dates/revisions"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        if not self.repo_path.exists():
            raise ValueError(f"Repository not found: {repo_path}")

    def get_commits_between_dates(
        self,
        start_date: str,
        end_date: str,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get commits between two dates.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of commits to return

        Returns:
            List of commit dictionaries
        """
        cmd = [
            "git", "-C", str(self.repo_path),
            "log",
            f"--since={start_date}",
            f"--until={end_date}",
            f"--max-count={limit}",
            "--pretty=format:%H|%an|%ae|%ad|%s",
            "--date=iso"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            commits = []

            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = line.split('|', 4)
                if len(parts) == 5:
                    commits.append({
                        'hash': parts[0],
                        'author': parts[1],
                        'email': parts[2],
                        'date': parts[3],
                        'message': parts[4],
                        'message_short': parts[4][:100]  # First 100 chars
                    })

            return commits

        except subprocess.CalledProcessError as e:
            print(f"Error getting commits: {e}")
            return []

    def get_file_changes(self, commit_hash: str) -> Dict:
        """
        Get detailed file changes for a commit.

        Args:
            commit_hash: Git commit hash

        Returns:
            Dictionary with file changes info
        """
        # Get changed files
        cmd_files = [
            "git", "-C", str(self.repo_path),
            "diff-tree", "--no-commit-id", "--name-status", "-r", commit_hash
        ]

        # Get commit stats
        cmd_stats = [
            "git", "-C", str(self.repo_path),
            "show", "--stat", "--format=", commit_hash
        ]

        try:
            # Get file changes
            result_files = subprocess.run(cmd_files, capture_output=True, text=True, check=True)
            files = []
            for line in result_files.stdout.strip().split('\n'):
                if not line:
                    continue
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    status, filename = parts
                    files.append({
                        'status': status,  # A=Added, M=Modified, D=Deleted
                        'filename': filename
                    })

            # Get stats
            result_stats = subprocess.run(cmd_stats, capture_output=True, text=True, check=True)
            stats_lines = result_stats.stdout.strip().split('\n')

            return {
                'commit': commit_hash,
                'files': files,
                'files_count': len(files),
                'stats_summary': stats_lines[-1] if stats_lines else "No stats"
            }

        except subprocess.CalledProcessError as e:
            print(f"Error getting file changes: {e}")
            return {'commit': commit_hash, 'files': [], 'files_count': 0}

    def categorize_commits(self, commits: List[Dict]) -> Dict:
        """
        Categorize commits by type (refactoring, bugfix, feature, etc).

        Args:
            commits: List of commit dictionaries

        Returns:
            Dictionary with categorized commits
        """
        categories = {
            'refactoring': [],
            'bugfix': [],
            'feature': [],
            'documentation': [],
            'test': [],
            'other': []
        }

        # Keywords for categorization
        keywords = {
            'refactoring': ['refactor', 'restructure', 'reorganize', 'extract', 'move', 'rename'],
            'bugfix': ['fix', 'bug', 'issue', 'patch', 'hotfix', 'correct'],
            'feature': ['add', 'new', 'implement', 'feature', 'enhance'],
            'documentation': ['doc', 'readme', 'comment', 'javadoc'],
            'test': ['test', 'junit', 'spec', 'coverage']
        }

        for commit in commits:
            msg_lower = commit['message'].lower()
            categorized = False

            for category, words in keywords.items():
                if any(word in msg_lower for word in words):
                    categories[category].append(commit)
                    categorized = True
                    break

            if not categorized:
                categories['other'].append(commit)

        return categories

    def get_summary_between_revisions(
        self,
        start_date: str,
        end_date: str,
        limit: int = 50
    ) -> Dict:
        """
        Get comprehensive summary of changes between two revisions.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum commits to analyze

        Returns:
            Summary dictionary
        """
        commits = self.get_commits_between_dates(start_date, end_date, limit)

        if not commits:
            return {
                'period': f"{start_date} to {end_date}",
                'total_commits': 0,
                'message': 'No commits found in this period'
            }

        # Categorize commits
        categories = self.categorize_commits(commits)

        # Get top authors
        authors = {}
        for commit in commits:
            author = commit['author']
            authors[author] = authors.get(author, 0) + 1

        top_authors = sorted(authors.items(), key=lambda x: x[1], reverse=True)[:5]

        # Get file change details for top commits (first 5)
        top_commit_details = []
        for commit in commits[:5]:
            details = self.get_file_changes(commit['hash'])
            top_commit_details.append({
                'hash': commit['hash'][:8],
                'message': commit['message'][:80],
                'date': commit['date'][:10],
                'files_changed': details['files_count']
            })

        # Compute simple hotspot files (most frequently changed) within this period
        file_counts: Dict[str, int] = {}
        try:
            list_hashes = [c['hash'] for c in commits]
            # Use git show --name-only per commit for efficiency
            for h in list_hashes:
                res = subprocess.run([
                    "git", "-C", str(self.repo_path),
                    "show", "--pretty=format:", "--name-only", h
                ], capture_output=True, text=True, check=False)
                for line in res.stdout.splitlines():
                    f = line.strip()
                    if f and not f.startswith(" ") and "/" in f or "." in f:
                        file_counts[f] = file_counts.get(f, 0) + 1
        except Exception:
            pass
        hotspot_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'period': f"{start_date} to {end_date}",
            'total_commits': len(commits),
            'categories': {k: len(v) for k, v in categories.items()},
            'top_commits': top_commit_details,
            'top_authors': [{'name': name, 'commits': count} for name, count in top_authors],
            'commit_samples': commits[:10],  # First 10 commits
            'hotspot_files': [{'file': f, 'changes': n} for f, n in hotspot_files]
        }


if __name__ == "__main__":
    # Test the analyzer
    import sys

    if len(sys.argv) < 2:
        print("Usage: python commit_analyzer.py <repo_path> [start_date] [end_date]")
        print("Example: python commit_analyzer.py ../REPOS/pdfbox 2025-07-01 2025-10-01")
        sys.exit(1)

    repo = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else "2025-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2025-12-31"

    analyzer = CommitAnalyzer(repo)
    summary = analyzer.get_summary_between_revisions(start, end)

    print(json.dumps(summary, indent=2))
