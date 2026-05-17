# Project Progress

## Current State

- Branch: `main`, ahead of `origin/main`.
- Latest implementation commit: `b9efd8e` (`Gate repeated reranker cache smoke`).
- Latest fully exported commit: `6929264abff460d18c4c46a1228e229e0a84f277` (`Show reranker cache hits in progress`).
- Current export bundle: `dist/local-export/xiao-buddy-6929264.bundle`.
- Current export manifest: `dist/local-export/manifest.json`.
- Tailscale app URL: http://100.103.106.102:7861/.
- App process: detached `app.py` server is responding on the Tailscale URL after startup cache warming; current PID is `1832272`.

## Completed Verification Hardening

- Reranker quality reports include metadata.
- Answer quality reports include metadata.
- Local export verification rejects missing, partial, failing, or stale quality reports.
- Browser smoke verifies visible source drafts, first-token wait heartbeats when the hosted agent is quiet long enough, and partial streamed answer text before the final ready state.
- App smoke verifies visible source-draft events, streamed first-token events, and first-token wait heartbeats when the hosted agent is quiet long enough.
- App cache smoke now sends the same live `/ask` query twice and fails if the repeated request does not show visible cached-rerank progress or still emits hosted reranker wait events.
- Answer reports separate source-draft visible latency, hosted-agent generation first-token latency, hosted-agent visible-from-request latency, and final answer latency.
- Answer reports now include source-draft build time and nested retrieval stage timings.
- Answer reports now include final answer output-size percentiles so generation-length tuning can be tracked from the summary.
- Agent answer budget is now `AGENT_MAX_TOKENS=260`; the earlier Wio Terminal `LIS3DHTR` miss is covered by contextual exact-term repair and regression coverage.
- App startup warms the lexical retrieval caches, loads the configured HNSW vector index, and sends one bounded streamed warmup request to the hosted agent before serving traffic.
- Retrieval now emits a pre-rerank source-backed preview before the hosted reranker finishes, then refreshes the draft after final reranked sources are ready.
- Retrieval now emits reranker-wait progress events while the hosted reranker is running, so the source draft can keep updating during the remaining rerank delay.
- Retrieval now caches successful reranker score results for repeated identical query/candidate batches, so manual QA and repeated live questions can skip the hosted reranker wait without changing cold-query quality.
- Progress UI and app smoke now explicitly show reranker cache hits as cached rerank work instead of silently reporting a generic rerank.
- The hosted agent stream now emits one-second progress heartbeats while waiting for the first token, so the source-backed draft remains visible with an active wait indicator instead of a quiet UI.
- The answer panel now shows an immediate hosted-agent first-token wait note after the source draft, closing the browser-visible handoff gap when server heartbeats are coalesced or the hosted model returns quickly.
- The final agent prompt now uses `AGENT_CONTEXT_CHARS=2000` with query-relevant source excerpts, while retrieval and reranking still use richer source text.
- The fixed agent instructions are compressed so the model sees the same source budget with about 1.1k fewer prompt characters.
- Answer quality reports include agent prompt size percentiles.
- Exact-term repair now uses contextual wording for UART parity, SHT40 library details, and ESPHome YAML settings instead of mechanical `Source detail:` appendices.
- Citation repair reuses an existing `Sources:` line instead of adding duplicate source sections.
- Answer quality now gates clean source formatting: duplicate `Sources:` sections, legacy `Source detail:` appendices, and generic `Relevant exact source terms:` appendices fail the report.
- Missing exact-term repair now uses contextual cited sentences for battery circuits, WiFi band details, LoRaWAN app values, network-server options, SX1262 kit applications, RS485 enable pins, firmware flows, Reachy services, OpenClaw roles, Round Display storage/controller/bus names, OLED addresses, PDM microphone labels/GPIOs, and Wio Terminal hardware terms; irrelevant forced terms are dropped instead of appended.
- Reranker window tuning is documented and gated; `RERANK_TEXT_CHARS=2800` is the current default after lower windows failed strict close-margin quality checks.

## Live Endpoint / Report Status

- Live app endpoint is available at http://100.103.106.102:7861/.
- Endpoint functional smoke passed:
  - embedding functional: `dim=2048`, `189 ms`.
  - reranker functional: native scores, `236 ms`.
  - agent functional: `564 ms`.
  - agent stream functional: first token `287 ms`, total `333 ms`.
- Startup warmup log shows retrieval warmup total `9291.4 ms` and agent warmup first token `288.5 ms`, total `335.9 ms`.
- App smoke passed against the detached Tailscale app process with `78` stream events, `11` source-draft events, `3` reranker-wait events, `0` reranker-cache events, `6` first-token wait events, first token `5281.4 ms`, and total `12139.8 ms`.
- Repeated app smoke against the same detached process hit the reranker result cache visibly: `70` stream events, `3` source-draft events, `0` reranker-wait events, `67` reranker-cache events, `1` first-token wait event, first token `174.9 ms`, and total `3668.2 ms`.
- App cache smoke for `b9efd8e` passed against the same detached process:
  - first pass: `70` stream events, `0` reranker-wait events, `67` reranker-cache events, first token `170.3 ms`, total `3385.6 ms`.
  - required repeated pass: `70` stream events, `0` reranker-wait events, `67` reranker-cache events, first token `150.5 ms`, total `3357.3 ms`.
- Browser-agent smoke passed against the Tailscale app with heartbeat observation enabled and saved current screenshots in `dist/browser-smoke/`.
- Direct hosted stream probe observed `5` `waiting_first_token` heartbeat updates at roughly one-second intervals before the agent stream began, then finished with `283` streaming updates and status `ok`.
- Answer quality report is current for `6929264`: `33/33` passing.
  - fact, citation, inline-citation, agent, stream, and answer-format rates are all `100%`.
  - pre-rerank source-preview latency: p50 `543.3 ms`, p95 `728.6 ms`, max `8961.7 ms`.
  - source-draft visible latency: p50 `543.4 ms`, p95 `728.7 ms`, max `8961.8 ms`.
  - source-draft build latency: p50 `0.0 ms`, p95 `0.1 ms`, max `0.2 ms`.
  - hosted-agent generation first-token latency: p50 `5087.7 ms`, p95 `5417.2 ms`, max `5862.0 ms`.
  - hosted-agent visible-from-request latency: p50 `8816.0 ms`, p95 `10975.3 ms`, max `12840.0 ms`.
  - final answer latency: p50 `15170.6 ms`, p95 `20561.2 ms`, max `21185.5 ms`.
  - agent prompt size: p50 `4376 chars`, p95 `4685 chars`, max `4749 chars`.
  - answer output size: p50 `460 chars`, p95 `854 chars`, max `953 chars`.
  - retrieval stage p50s: lexical prefilter `357.3 ms`, query embedding `139.8 ms`, HNSW search `2.5 ms`, candidate selection `10.2 ms`, hosted reranker `3403.3 ms`, retrieval total `3978.5 ms`.
- Reranker quality report is current for `b9efd8e`: `62/62` passing.
  - native scoring mode.
  - margin: min `0.0124`, p50 `0.1328`.
  - latency: p50 `1600.5 ms`, p95 `2177.5 ms`, max `2581.5 ms`.
  - close margins remain concentrated in similar-board and similar-topic pages.
- Local export verification passed for `dist/local-export/xiao-buddy-6929264.bundle`.

## Current In-Progress Work

- Measuring perceived answer latency while keeping final answers agent-generated and cited.
- Lowered `AGENT_MAX_TOKENS` from `350` to `300`, then to `260`, after the full answer-quality gate stayed at `33/33`.
- Added a deterministic exact-term safety pass so compact agent answers keep cited hardware terms such as `2.4G`, `USB-UART`, `firmware.uf2`, and `frequency plan`.
- Added an immediate source-backed draft after retrieval so users see relevant cited source notes before the hosted agent produces its first token.
- Added a pre-rerank source-backed preview so first useful cited text appears before the hosted reranker completes.
- Added reranker-wait progress events so the pre-rerank source draft visibly reports hosted reranker work instead of sitting static during the rerank call.
- Added a bounded in-memory reranker result cache:
  - cache keys include reranker endpoint/model, normalized query, rerank text window, metadata flags, candidate IDs/sources, and SHA-256 hashes of the exact rerank text.
  - only successful scored reranker responses are stored, and the cache is LRU-pruned at `128` entries.
  - regression coverage verifies exact-input keying and that `retrieve_progressive` skips the hosted reranker path on a repeated identical request.
  - repeated app smoke on the detached process dropped from `3` reranker-wait events and `12139.8 ms` total to `0` reranker-wait events and `3668.2 ms` total.
  - the progress panel now labels the repeated path as `cached native rerank` and timing as `rerank cache`, and app smoke counts visible `reranker_cache_events`.
- Added `make app-cache-smoke` and `APP_SMOKE_REQUIRE_RERANK_CACHE` / `APP_SMOKE_REQUIRE_NO_RERANK_WAIT` so the repeated-query cache path is a failing live gate instead of only a reported observation.
- Added an agent-stream heartbeat wrapper so the progress panel updates about once per second while the hosted agent is quiet before its first token.
- Added query-aware compact source excerpts for the final agent prompt with `AGENT_CONTEXT_CHARS=2000`.
- Added bounded startup hosted-agent warmup with `AGENT_STARTUP_WARMUP_SECONDS=8`:
  - the app logs whether warmup ran and how long first token/total warmup took.
  - the current Tailscale restart warmed the agent stream in `335.9 ms`.
  - the subsequent full answer eval passed `33/33` with first-token p50 `5087.7 ms`, p95 `5417.2 ms`, and max `5862.0 ms`.
- Compressed the fixed system/user prompt instructions:
  - focused 20-case hosted run passed `20/20`, with prompt p50 `4378` chars and first-token p50 `5225.3 ms`.
  - full hosted answer eval passed `33/33`; prompt p50 dropped from `5538` to `4376` chars, answer p50 from `545` to `460` chars, first-token p50 from `5740.0 ms` to `5191.1 ms`, and total p50 from `17461.2 ms` to `15369.9 ms`.
  - p95/max latency worsened in the full run because two hosted agent calls spiked, so this is a median/prompt-size win rather than a proven tail-latency fix.
- Expanded exact-term preservation for compact prompts so board IDs, chip names, GPIOs, library names, robotics service names, and LoRaWAN network-server names remain cited.
- Replaced noisy exact-term `Source detail:` appendices with contextual repairs:
  - UART answers now normalize "no parity" to parity `None`.
  - SHT40 answers add a natural library/package sentence when needed.
  - ESPHome YAML answers use the cited Arduino-framework block when the model chooses a conflicting ESP-IDF block.
  - Round Display answers preserve `SD card slot`, `GC9A01`, and `CHSC6X`.
  - Round Display bus answers also preserve `SPI` and `I2C` when compact context causes the agent to omit them.
  - OLED answers normalize bare `3c` to `0x3C`.
  - S3 Sense microphone answers preserve `Clock`, `Data`, `GPIO41`, and `GPIO42` labels for the PDM pins.
  - Wio Terminal hardware answers preserve `ATSAMD51P19`, `Realtek RTL8720DN`, `120MHz`, `4MB`, `192KB`, and `LIS3DHTR`.
- Fixed duplicate `Sources:` sections by appending missing source IDs to the existing source line.
- Added an answer-format eval gate and removed the remaining generic exact-term appendix path:
  - focused hosted repair eval passed `16/16` on the cases that previously failed only formatting.
  - full hosted answer eval passed `33/33`, including `33/33` answer-format rate.
- Tuned the reranker text window:
  - corrected the S3 Sense PDM microphone retrieval eval to check source wording `PDM Microphone CLK` and `PDM Microphone DATA`.
  - `RERANK_TEXT_CHARS=900` passed full retrieval benchmark after that correction but failed strict reranker quality at `56/62`.
  - `RERANK_TEXT_CHARS=2400` failed strict reranker quality at `61/62`; `2600` failed a close nRF54 Zigbee target case; a targeted `3000` check also failed that close case.
  - `RERANK_TEXT_CHARS=2800` passed strict reranker quality `62/62` and full answer quality `33/33`, so it is now the default.
- Browser smoke now verifies both the source-backed draft and visible partial hosted-agent stream.
- Browser smoke now records the first-token wait heartbeat and fails if a long draft-to-stream gap has no progress heartbeat.
- Browser smoke now also accepts the answer-panel hosted-agent wait note as visible progress, after it caught a roughly `3.9 s` browser-side quiet gap between the source draft and first rendered streamed token.
- App smoke now records source-draft, reranker-wait, and first-token heartbeat events, and fails if a long hosted-agent first-token wait has no heartbeat.
- Current bottleneck: the hosted reranker still costs about `3.4 s` p50 inside full answer evaluation, but it no longer blocks first useful source text.
- Current interpretation: local ANN, source draft construction, and candidate selection are not the main problem; final perceived latency now mostly comes from hosted rerank plus hosted agent prefill/generation.
- Added stage-level retrieval timings for knowledge-base load, lexical prefilter, query embedding, HNSW search, score merge, candidate selection, reranker preparation, reranker call, source-context expansion, and retrieval total.
- Added answer output-size percentiles to the answer-quality summary; the current passing set has p50 `518` chars, p95 `936` chars, and max `1191` chars.
- With `AGENT_MAX_TOKENS=300`, answer output-size percentiles were p50 `518` chars, p95 `887` chars, and max `1058` chars.
- With `AGENT_MAX_TOKENS=260`, answer output-size percentiles are p50 `538` chars, p95 `875` chars, and max `957` chars; this mostly tightens the worst-case answer length and p95/max total latency rather than improving first-token latency.
- Added startup retrieval-cache warming in `app.py`.
- Latest focused timing probe before the full report:
  - cold retrieval source-draft path was about `14.1 s`, dominated by lexical prefilter at `7.6 s` and hosted reranker at `5.0 s`.
  - explicit startup warmup took about `8.9 s` once at process start.
  - warmed retrieval dropped to about `4.9 s`, with lexical prefilter at `0.33 s`, HNSW at `0.05 s`, and hosted reranker still dominant at `4.35 s`.
- Candidate-count tuning:
  - `CANDIDATE_K=8` was faster but failed the MG24 deep-sleep answer case by omitting the required `erase` term.
  - `CANDIDATE_K=12` passed the full answer quality suite and is now the committed default.
- Deployed model speed interpretation:
  - Embedding endpoint looks healthy for this workload.
  - Reranker endpoint is accurate and acceptable for quality gating, but it is too slow to block first visible text in the full RAG path.
  - Agent endpoint is healthy on short prompts and acceptable for final synthesis, but long-context prefill makes it too slow to be the first user-visible response.
- Agent-context tuning:
  - `AGENT_CONTEXT_CHARS=3500` passed the full `33/33` answer-quality gate.
  - Compared with `cfb9ea3`, agent first-token p50 improved from `7384.1 ms` to `6785.9 ms`, agent-visible p50 from `11686.8 ms` to `10493.9 ms`, and total p50 from `19763.3 ms` to `18390.6 ms`.
  - `AGENT_CONTEXT_CHARS=2600` initially missed a few exact source terms on hard cases; contextual repairs now preserve `Built-in LoRaWAN Network Server`, `2.5km`, `LoRaWAN Node`, and RS485 enable pin `D2`.
  - The committed `2600` default passed full hosted answer quality `33/33`.
  - Versus the previous `3500` report, prompt p50 dropped from `6705` to `6038` chars, first-token p50 improved from `6796.8 ms` to `5984.1 ms`, agent-visible p50 from `10302.6 ms` to `9883.5 ms`, and total p50 from `18236.6 ms` to `17270.6 ms`.
  - `AGENT_CONTEXT_CHARS=2200` passed a 20-case hard subset; `2000` initially failed exact wording on Round Display, OLED, and PDM microphone cases, then passed the full hosted answer-quality gate after contextual exact-term repairs.
  - The committed `2000` default passed full hosted answer quality `33/33`; versus `2600`, prompt p50 dropped from `6038` to `5539` chars and first-token p50 improved from `5955.9 ms` to `5682.6 ms`.
  - `AGENT_CONTEXT_CHARS=1600` failed a 20-case hard subset at `18/20`: PDM microphone omitted `GPIO41`/`GPIO42`, and Round Display bus omitted `SPI`. Prompt p50 only dropped to `5316` chars and first-token p50 did not improve, so the default stays `2000`.
  - The two compact-context failures now pass after exact-term repairs for PDM GPIOs and Round Display `SPI`/`I2C`.
  - The committed `260` answer-token cap passed a focused `16/16` hard-case run and the full hosted answer-quality gate `33/33`; versus the previous `300` report, max answer size dropped from `1083` to `957` chars and final-answer p95 improved from `22783.7 ms` to `21669.6 ms`.
  - Deployed model speed answer: the endpoints are usable but not fast. Embeddings are healthy, the reranker is accurate but adds about `3.4 s` p50 inside full RAG, and the Gemma agent remains the primary UX bottleneck with about `5.7 s` p50 to first token and `17.5 s` p50 full-answer latency.
  - Short endpoint probes are fast, so the deployed services are not generally unhealthy; the latency appears when the real RAG path sends many rerank candidates and a long synthesis prompt.

## Remaining Next Steps

- Continue optimizing final answer latency without weakening the current `33/33` answer-quality gate.
- Investigate reducing hosted agent prefill/generation latency; this is now the main remaining visible delay after source previews.
- Consider reranker request batching or service-side tuning if it can improve the full RAG p95 without reducing the current `62/62` reranker gate.
- Consider a faster or smaller hosted agent model for first-pass synthesis, while keeping the current agent as a high-quality final path.
- Harden close-margin reranker eval cases where positive and negative pages are near duplicates or similar-board pages.
- For `b9efd8e`, regenerate answer quality and local export before the next handoff bundle; those artifacts are still current for `6929264`.

## Key Commands / Artifacts

- Current commit check: `git rev-parse HEAD`
- Branch status check: `git status --short --branch`
- Answer quality regeneration: `REQUEST_TIMEOUT_SECONDS=90 make answer-eval`
- Reranker quality regeneration: `REQUEST_TIMEOUT_SECONDS=90 make rerank-quality-all`
- Repeated-cache live smoke: `REQUEST_TIMEOUT_SECONDS=90 APP_SMOKE_TIMEOUT_SECONDS=180 make app-cache-smoke`
- Export verification: `make export-local verify-local-export`
- Reranker quality result target: `62/62` passing with metadata matching current `HEAD`.
- Answer quality result target: `33/33` passing with metadata matching current `HEAD`.
- Tailscale app URL: http://100.103.106.102:7861/
