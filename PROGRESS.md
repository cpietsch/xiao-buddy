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
- Reranker quality report status: must be regenerated after the latency-control commit.
- Answer quality report status: must be regenerated after the latency-control commit.
- Local export verification status: must be rerun after regenerated quality reports are present.

## Current In-Progress Work

- Reducing hosted agent response latency without dropping required answer facts.
- Added `AGENT_MAX_TOKENS=350` and tightened the agent prompt to put direct settings first.
- Targeted live YAML answer case passed at the new default: `350` streamed chunks, `964` chars, first token `204.0 ms`, total `32430.3 ms`.

## Remaining Next Steps

- Commit the latency-control change and this progress update.
- Regenerate answer and reranker quality reports after any new commit.
- Run `make export-local verify-local-export` after regenerated reports are present.
- Check `dist/local-export/manifest.json` for final bundle path and verification evidence.

## Key Commands / Artifacts

- Current commit check: `git rev-parse HEAD`
- Branch status check: `git status --short --branch`
- Answer quality regeneration: `REQUEST_TIMEOUT_SECONDS=90 make answer-eval`
- Export verification: `make export-local verify-local-export`
- Reranker quality result target: `62/62` passing with metadata matching current `HEAD`.
- Answer quality result target: `33/33` passing with metadata matching current `HEAD`.
