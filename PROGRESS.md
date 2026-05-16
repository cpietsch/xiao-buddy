# Project Progress

## Current State

- Branch: `main`, ahead of `origin/main` by 76 commits.
- HEAD: `b9be92b` (`Record quality report provenance`).
- Full HEAD: `b9be92b32b5ea1d1d2014f34498927625f12071c`.
- Tailscale app URL: http://100.103.106.102:7861/.

## Completed Verification Hardening

- Reranker quality report now includes metadata.
- Reranker quality report passed `62/62` at git head `b9be92b32b5ea1d1d2014f34498927625f12071c`.
- Quality report provenance has been recorded in the current HEAD.

## Live Endpoint / Report Status

- Live app endpoint is available at http://100.103.106.102:7861/.
- Reranker quality report status: complete, metadata present, `62/62` passing.
- Answer quality report status: complete, metadata present, `33/33` passing.
- Local export verification status: ready to rerun with both quality reports pinned to `b9be92b32b5ea1d1d2014f34498927625f12071c`.

## Current In-Progress Work

- Refreshing the local export manifest and verifying the bundle/patch handoff after regenerated quality reports.

## Remaining Next Steps

- Run `make export-local verify-local-export`.
- Update this file with the final export bundle path and verification outcome.

## Key Commands / Artifacts

- Current commit check: `git rev-parse HEAD`
- Branch status check: `git status --short --branch`
- Answer quality regeneration: `REQUEST_TIMEOUT_SECONDS=90 make answer-eval`
- Export verification: `make export-local verify-local-export`
- Reranker quality result: `62/62` passing with metadata at `b9be92b32b5ea1d1d2014f34498927625f12071c`
- Answer quality result: `33/33` passing with metadata at `b9be92b32b5ea1d1d2014f34498927625f12071c`
