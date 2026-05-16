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
- Reranker quality report status: must be regenerated after the draft-latency metrics commit.
- Answer quality report status: must be regenerated after the draft-latency metrics commit.
- Local export verification status: must be rerun after regenerated quality reports are present.

## Current In-Progress Work

- Measuring perceived answer latency while keeping final answers agent-generated and cited.
- Added `AGENT_MAX_TOKENS=350` and tightened the agent prompt to put direct settings first.
- Added a deterministic exact-term safety pass so compact agent answers keep cited hardware terms such as `2.4G`, `USB-UART`, `firmware.uf2`, and `frequency plan`.
- Added an immediate source-backed draft after retrieval so users see relevant cited source notes before the hosted agent produces its first token.
- Browser smoke now verifies both the source-backed draft and visible partial hosted-agent stream.
- Answer reports now separate source-draft visible latency, hosted-agent first-token latency, and hosted-agent visible-from-request latency.

## Remaining Next Steps

- Commit the draft-latency metrics change.
- Regenerate answer and reranker quality reports after the commit.
- Run `make export-local verify-local-export` after regenerated reports are present.

## Key Commands / Artifacts

- Current commit check: `git rev-parse HEAD`
- Branch status check: `git status --short --branch`
- Answer quality regeneration: `REQUEST_TIMEOUT_SECONDS=90 make answer-eval`
- Export verification: `make export-local verify-local-export`
- Reranker quality result target: `62/62` passing with metadata matching current `HEAD`.
- Answer quality result target: `33/33` passing with metadata matching current `HEAD`.
