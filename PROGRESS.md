# Project Progress

## Current State

- Branch: `main`, ahead of `origin/main`.
- Current commit: check with `git rev-parse HEAD`.
- Current export bundle: check `dist/local-export/manifest.json` after running `make export-local verify-local-export`.
- Tailscale app URL: http://100.103.106.102:7861/.

## Completed Verification Hardening

- Reranker quality reports include metadata.
- Answer quality reports include metadata.
- Local export verification rejects missing, partial, failing, or stale quality reports.
- Browser smoke verifies visible partial streamed answer text before the final ready state.

## Live Endpoint / Report Status

- Live app endpoint is available at http://100.103.106.102:7861/.
- Reranker quality report status: complete, metadata present, `62/62` passing.
- Answer quality report status: complete, metadata present, `33/33` passing.
- Local export verification status: rerun after reports are regenerated at the current commit.

## Current In-Progress Work

- Refreshing report metadata and local export evidence after progress-log updates.

## Remaining Next Steps

- Regenerate answer and reranker quality reports after any new commit.
- Run `make export-local verify-local-export`.
- Check `dist/local-export/manifest.json` for final bundle path and verification evidence.

## Key Commands / Artifacts

- Current commit check: `git rev-parse HEAD`
- Branch status check: `git status --short --branch`
- Answer quality regeneration: `REQUEST_TIMEOUT_SECONDS=90 make answer-eval`
- Export verification: `make export-local verify-local-export`
- Reranker quality result target: `62/62` passing with metadata matching current `HEAD`.
- Answer quality result target: `33/33` passing with metadata matching current `HEAD`.
