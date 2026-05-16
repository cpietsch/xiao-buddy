# Project Progress

## Current State

- Branch: `main`, ahead of `origin/main`.
- Current commit: `de3c9b8f1e89ce4b5b6056bb748f9ec310626fc4` (`Report source draft latency`).
- Current export bundle: `dist/local-export/xiao-buddy-de3c9b8.bundle`.
- Current export manifest: `dist/local-export/manifest.json`.
- Tailscale app URL: http://100.103.106.102:7861/.
- App process: detached `app.py` server was responding on the Tailscale URL during the latest check.

## Completed Verification Hardening

- Reranker quality reports include metadata.
- Answer quality reports include metadata.
- Local export verification rejects missing, partial, failing, or stale quality reports.
- Browser smoke verifies visible partial streamed answer text before the final ready state.
- Answer reports separate source-draft visible latency, hosted-agent generation first-token latency, hosted-agent visible-from-request latency, and final answer latency.

## Live Endpoint / Report Status

- Live app endpoint is available at http://100.103.106.102:7861/.
- Endpoint functional smoke passed:
  - embedding functional: `dim=2048`, `192 ms`.
  - reranker functional: native scores, `215 ms`.
  - agent functional: `3365 ms`.
  - agent stream functional: first token `277 ms`, total `323 ms`.
- Answer quality report is current for `de3c9b8`: `33/33` passing.
  - fact, citation, inline-citation, agent, and stream rates are all `100%`.
  - source-draft visible latency: p50 `13609.6 ms`, p95 `15366.0 ms`, max `15516.5 ms`.
  - hosted-agent generation first-token latency: p50 `7608.1 ms`, p95 `11051.5 ms`, max `12401.6 ms`.
  - hosted-agent visible-from-request latency: p50 `21067.5 ms`, p95 `26441.0 ms`, max `27903.8 ms`.
  - final answer latency: p50 `29111.9 ms`, p95 `37626.7 ms`, max `41672.3 ms`.
- Reranker quality report is current for `de3c9b8`: `62/62` passing.
  - native scoring mode.
  - margin: min `0.0089`, p50 `0.1260`.
  - latency: p50 `1637.5 ms`, p95 `2272.8 ms`, max `2750.5 ms`.
  - close margins remain concentrated in similar-board and similar-topic pages.
- Local export verification passed for `dist/local-export/xiao-buddy-de3c9b8.bundle`.

## Current In-Progress Work

- Measuring perceived answer latency while keeping final answers agent-generated and cited.
- Added `AGENT_MAX_TOKENS=350` and tightened the agent prompt to put direct settings first.
- Added a deterministic exact-term safety pass so compact agent answers keep cited hardware terms such as `2.4G`, `USB-UART`, `firmware.uf2`, and `frequency plan`.
- Added an immediate source-backed draft after retrieval so users see relevant cited source notes before the hosted agent produces its first token.
- Browser smoke now verifies both the source-backed draft and visible partial hosted-agent stream.
- Current bottleneck: the first useful source-backed draft is still too slow for a chat UX at about `13.6 s` p50 and `15.4 s` p95.
- Current interpretation: the hosted agent is slow enough to matter, but the pre-agent retrieval/draft phase is now the highest priority because users wait for it before seeing useful content.
- Added stage-level retrieval timings for knowledge-base load, lexical prefilter, query embedding, HNSW search, score merge, candidate selection, reranker preparation, reranker call, source-context expansion, and retrieval total.
- Added startup retrieval-cache warming in `app.py`.
- Latest focused timing probe:
  - cold retrieval source-draft path was about `14.1 s`, dominated by lexical prefilter at `7.6 s` and hosted reranker at `5.0 s`.
  - explicit startup warmup took about `8.9 s` once at process start.
  - warmed retrieval dropped to about `4.9 s`, with lexical prefilter at `0.33 s`, HNSW at `0.05 s`, and hosted reranker still dominant at `4.35 s`.

## Remaining Next Steps

- Re-run the full answer and reranker reports after committing the stage timing and startup warming change.
- Restart the Tailscale app so the running server includes startup cache warming.
- Use the new warm timings to reduce reranker latency next, likely by benchmarking fewer rerank candidates or adding an earlier pre-rerank source preview.
- Harden close-margin reranker eval cases where positive and negative pages are near duplicates or similar-board pages.
- Re-run browser smoke after any UI/progress changes and regenerate reports before the next export.

## Key Commands / Artifacts

- Current commit check: `git rev-parse HEAD`
- Branch status check: `git status --short --branch`
- Answer quality regeneration: `REQUEST_TIMEOUT_SECONDS=90 make answer-eval`
- Reranker quality regeneration: `REQUEST_TIMEOUT_SECONDS=90 make rerank-quality-all`
- Export verification: `make export-local verify-local-export`
- Reranker quality result target: `62/62` passing with metadata matching current `HEAD`.
- Answer quality result target: `33/33` passing with metadata matching current `HEAD`.
- Tailscale app URL: http://100.103.106.102:7861/
