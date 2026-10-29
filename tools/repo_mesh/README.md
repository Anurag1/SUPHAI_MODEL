# Live GitHub connector (read-only pilot)

This is the first live-repository adapter for Repo Voice Mesh. It uses the GitHub REST API to inventory repository metadata, open issues, and open pull requests across an explicit allowlist.

## Run locally

Python 3.10+; no extra package required.

```bash
export GITHUB_TOKEN="YOUR_READ_ONLY_TOKEN"
python tools/repo_mesh/github_inventory.py --out reports/repo-mesh-report.json
```

Use a fine-grained token with read-only repository metadata and issues access, scoped only to the repositories required. Never commit the token.

Override the allowlist:

```bash
python tools/repo_mesh/github_inventory.py --repos Anurag1/SUPHAI_MODEL Anurag1/Prometheus-AGI --out reports/pilot.json
```

## GitHub Actions

The workflow can run on schedule or manually. The default `GITHUB_TOKEN` generally only has access to the current repository. For cross-repository inventory, add a narrowly scoped Actions secret named `REPO_MESH_READ_TOKEN`.

## Safety and scope

- Read-only: no issue comments, edits, pushes, PRs, or source execution.
- Does not claim to resolve issues.
- Reports are operational data and are not committed automatically.
- Issue bodies and repository files are untrusted inputs and must not trigger shell commands.
- This is the live data layer only; triage, sandboxed patching, tests, regression checks, and voice remain subsequent stages.
