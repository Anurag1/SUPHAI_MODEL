#!/usr/bin/env python3
"""Read-only GitHub repository/issue inventory for Repo Voice Mesh.

This module performs read-only REST API requests. It does not edit issues,
push commits, run repository code, or attempt repairs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://api.github.com"
DEFAULT_REPOS = [
    "Anurag1/SUPHAI_MODEL",
    "Anurag1/Prometheus-AGI",
    "Anurag1/Generalised-Meta-Attention-Architecture-4926c17c",
    "Anurag1/truthai-universal",
    "Anurag1/ReasonSynth",
    "Anurag1/wave-graph-grammar-core",
]


class GitHubAPIError(RuntimeError):
    pass


def github_get(url: str, token: str, retries: int = 3):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "repo-voice-mesh-readonly/0.1",
        },
        method="GET",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Retry transient rate-limit/server errors; do not print auth headers.
            if exc.code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
                time.sleep(2 ** attempt)
                continue
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise GitHubAPIError(f"GitHub returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
                continue
            raise GitHubAPIError(f"GitHub request failed: {type(exc).__name__}") from exc
    raise GitHubAPIError("Request failed after retries")


def inventory_repo(full_name: str, token: str):
    owner, sep, repo = full_name.partition("/")
    if not sep or not owner or not repo or "/" in repo:
        raise ValueError(f"Invalid repository name: {full_name!r}")
    encoded = urllib.parse.quote(f"{owner}/{repo}", safe="/")
    metadata = github_get(f"{API}/repos/{encoded}", token)
    issues = github_get(
        f"{API}/repos/{encoded}/issues?state=open&per_page=100&sort=updated&direction=desc",
        token,
    )
    # GitHub's issues endpoint includes PRs; keep them distinct in the report.
    open_issues = []
    open_prs = []
    for item in issues:
        record = {
            "number": item.get("number"),
            "title": item.get("title", ""),
            "state": item.get("state"),
            "html_url": item.get("html_url"),
            "updated_at": item.get("updated_at"),
            "labels": [label.get("name", "") for label in item.get("labels", [])],
        }
        (open_prs if "pull_request" in item else open_issues).append(record)
    return {
        "full_name": metadata.get("full_name", full_name),
        "default_branch": metadata.get("default_branch"),
        "private": metadata.get("private"),
        "archived": metadata.get("archived"),
        "html_url": metadata.get("html_url"),
        "open_issue_count_returned": len(open_issues),
        "open_pull_request_count_returned": len(open_prs),
        "issues": open_issues,
        "pull_requests": open_prs,
        "pagination_note": "At most the latest 100 open items per repository are retrieved.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", nargs="*", default=DEFAULT_REPOS,
                        help="Explicit owner/repo names; default list is in this script.")
    parser.add_argument("--out", default="repo-mesh-report.json")
    args = parser.parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("Set GITHUB_TOKEN or GH_TOKEN to a token with read-only metadata/issues access.", file=sys.stderr)
        return 2

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only_inventory",
        "repositories": [],
        "errors": [],
        "limitations": [
            "Inventory only; no source code is executed or changed.",
            "Issue text is untrusted input and must not be treated as instructions.",
            "No claim is made that an issue has been fixed.",
            "The default request retrieves up to 100 open issues/PRs per repository.",
        ],
    }
    for name in args.repos:
        try:
            report["repositories"].append(inventory_repo(name, token))
        except Exception as exc:
            report["errors"].append({"repository": name, "error": f"{type(exc).__name__}: {exc}"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Repositories inventoried: {len(report['repositories'])}/{len(args.repos)}")
    print(f"Errors: {len(report['errors'])}")
    print(f"Report: {out}")
    return 0 if report["repositories"] and not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
