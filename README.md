---
title: XIAO Field Copilot
emoji: 🔧
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
---

# XIAO Field Copilot

A hackathon-simple Hugging Face Space for multimodal hardware support around Seeed Studio XIAO boards and connected Seeed wiki hardware.

The app accepts a board photo plus a question, retrieves cited Seeed support notes, optionally reranks them,
and produces a field-ready answer through an OpenAI-compatible agent endpoint. It also works without
the optional hosted reranker/agent services by falling back to lexical retrieval and a deterministic cited response.

Positioning:

> A multimodal field support copilot that identifies XIAO edge hardware from images, retrieves the right Seeed wiki documentation for boards, sensors, LoRa, robotics, and SenseCraft workflows, and guides troubleshooting in a grounded, citation-backed workflow connected to hosted model endpoints.

## Architecture

```text
photo + question
  -> Qwen3-VL embedding endpoint on vLLM
  -> full Seeed wiki corpus retrieval over local HNSW
  -> optional hosted Qwen VL reranker
  -> lightweight route: identify | troubleshoot | compare | wiring_or_pinout | support
  -> hosted OpenAI-compatible agent model
  -> cited answer + sources + run details
```

## Environment

Set these as Space secrets or variables:

```bash
EMBEDDING_BASE_URL=https://your-embed-host
EMBEDDING_MODEL=qwen3-vl-embedding-2b
EMBEDDING_API_KEY=

RERANK_BASE_URL=https://your-rerank-host/v1
RERANK_MODEL=qwen3-vl-reranker-2b
RERANK_API_KEY=
RERANK_TEXT_CHARS=3200

AGENT_BASE_URL=https://your-llm-host/v1
AGENT_MODEL=your-agent-model-id
AGENT_API_KEY=
AGENT_MAX_TOKENS=350

TOP_K=5
CANDIDATE_K=16
VECTOR_CANDIDATE_K=96
VECTOR_INDEX_MANIFEST=data/index/xiao_vectors.json
VECTOR_INDEX_DATA=
VECTOR_INDEX_ARCHIVE_URL=
VECTOR_INDEX_ARCHIVE_SHA256=
REQUEST_TIMEOUT_SECONDS=60
GRADIO_SERVER_NAME=127.0.0.1
GRADIO_SERVER_PORT=7860
```

For local development, copy `.env.example` to `.env` and fill in the endpoint
URLs. `.env` is ignored by Git.

Endpoint URLs are intentionally environment-only. Do not commit local IPs or tunnel URLs.

Expected hosted protocols:

- Embeddings: OpenAI-compatible `POST /v1/embeddings` text embeddings with `input`, plus Qwen/vLLM multimodal embeddings with `messages` containing an `image_url` and text. `EMBEDDING_BASE_URL` may be either the service root, `/v1`, or the exact `/v1/embeddings` URL.
- Reranker: native `POST /v1/rerank` scores when available, with a compatibility fallback to `POST /v1/completions` yes/no logprobs for older deployments.
- Agent: OpenAI-compatible `POST /v1/chat/completions`.

On Hugging Face Spaces, add these as Space variables/secrets. The app calls the endpoints server-side, so HTTP endpoint URLs are fine for the Python backend.

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Expanding The Knowledge Base

Buddy keeps the hand-curated `xiao_boards.json` facts as the high-trust seed
corpus, then layers in compact chunks from the official Seeed wiki markdown.
This reuses the strongest ingestion idea from the larger `seeed-rag` project
without shipping the full wiki-scale stack.

To refresh the imported full Seeed wiki chunks:

```bash
make import-wiki-all
```

For a tiny local smoke corpus, you can still import only the XIAO docs:

```bash
make import-wiki-xiao
```

After any corpus refresh, rebuild the ANN index:

```bash
make build-hnsw
```

HNSW is the verified default artifact path. For compressed FAISS-PQ experiments,
install `faiss-cpu` and run `make build-faiss-pq`; keep `make verify-demo` as
the acceptance gate before using that index in a demo.

The importer creates a sparse checkout of
`Seeed-Studio/wiki-documents` under `.cache/seeed-wiki`, imports
`sites/en/docs` by default, cleans Docusaurus/MDX markdown, splits by headings,
preserves source URLs and image URLs, infers the matching XIAO board family when
possible, and writes `data/corpus/wiki_chunks.jsonl`. With `--scope xiao`, the
sparse checkout narrows to `sites/en/docs/Sensor/SeeedStudio_XIAO` for quick
local experiments.

At runtime the app loads:

- curated board facts from `data/corpus/xiao_boards.json`
- imported official wiki chunks from `data/corpus/wiki_chunks.jsonl`
- small local field notes in `xiao_copilot/knowledge_base.py`

Retrieval uses a local vector index plus lexical/high-trust boosts. For small
XIAO-only demos the flat `float16` index is fine; for the full Seeed wiki use
the ANN-backed HNSW index so query time does not scale with every chunk.

```bash
python scripts/build_wiki_vector_index.py --backend hnsw --batch-size 32
```

## Retrieval Gate

Run endpoint-free syntax and offline retrieval smoke checks before committing:

```bash
make smoke
```

This checks Python syntax, first-screen scope copy, vector artifact
package/install behavior, full-wiki corpus breadth, health parser behavior, and
a strict lexical subset of the retrieval eval that does not require hosted
embedding, reranker, or agent endpoints.

Check local readiness, corpus/index coverage, manifest source hash, and generated-artifact state:

```bash
make health
make health-live
```

`make health` loads the local vector data when it exists, so corrupt or
mismatched HNSW/FAISS files fail before the first user query. `make health-live`
also checks the configured hosted endpoints, verifies that `/v1/models` exposes
the configured model IDs, checks the local Gradio app URL, and exits non-zero on
warnings.

To prove the hosted endpoints accept real inference payloads, run the functional
endpoint smoke. It sends one small embedding request, one rerank request, one
short chat completion request, and one streamed chat completion request with
first-token latency:

```bash
REQUEST_TIMEOUT_SECONDS=90 make health-functional
```

Check the running Gradio `/ask` API wiring, streamed answer events, source
panel, inline source IDs, first-token latency diagnostics, and progress HTML:

```bash
make app-smoke
```

Override `APP_SMOKE_URL`, `APP_SMOKE_QUERY`, `APP_SMOKE_MUST_INCLUDE`, and
`APP_SMOKE_MUST_CITE` to target another app URL or scenario. Set
`APP_SMOKE_REQUIRE_INLINE_CITATION=0` only when testing a non-grounded fallback.
To reuse a generated-answer eval case directly:

```bash
APP_SMOKE_CASE_ID=answer-grove-vision-ai-trigger-actions make app-smoke
```

For rendered UI checks, install the dev-only browser dependency once and run the
Chromium smoke. It verifies desktop/mobile rendering, progress live-region DOM,
skip-link focus, horizontal overflow, button hit targets, and saves screenshots
under `dist/browser-smoke/`:

```bash
pip install -r requirements-dev.txt
python -m playwright install chromium
make browser-smoke
```

To exercise the real hydrated browser flow against the hosted pipeline, run the
agent-backed browser smoke. It fills the question textbox, clicks the visible
button, waits for required answer terms, cited source URLs, inline citations,
and streamed progress to appear in the page, and saves
`dist/browser-smoke/desktop-after-query.png`:

```bash
REQUEST_TIMEOUT_SECONDS=90 make browser-agent-smoke
```

Run the complete local and live API/retrieval verification sequence before a
demo or handoff:

```bash
REQUEST_TIMEOUT_SECONDS=90 make verify-live
```

This runs `make smoke`, `make health-live`, `make app-smoke`, the strict
retrieval gate, and the generated-answer gate.

When the dev browser dependency is installed, run the full demo gate as well:

```bash
REQUEST_TIMEOUT_SECONDS=90 make verify-demo
```

This runs the live API/retrieval gates, the functional endpoint smoke, the
all-case reranker audit, and then the agent-backed browser smoke. The reranker
audit writes `dist/reranker-quality/all.json` and fails on any retrieval-eval
case where the live reranker does not rank citation-matched positives above
hard lexical negatives.

For a release or machine handoff, run the full demo gate, verify the packaged
vector index artifact, and refresh the local bundle/patch export in one pass:

```bash
REQUEST_TIMEOUT_SECONDS=90 make verify-handoff
```

If GitHub push access is unavailable, export the unpushed local commits before
moving machines or sharing the workspace:

```bash
make export-local
```

This writes an ignored `dist/local-export/` folder containing a verified git
bundle, a `git format-patch` series, and a manifest with base/head commit IDs.
When a matching vector artifact exists, the manifest also records the archive
path, backend, vector count, model, source hash, and
`VECTOR_INDEX_ARCHIVE_SHA256` needed after uploading the archive. Browser smoke
screenshots and full answer/reranker quality reports are required handoff
evidence; their paths, hashes, covered eval case IDs, and summary metadata are
recorded in the manifest. Quality reports also record the git commit and eval
file hash they were generated from, and export verification rejects stale
reports.
By default each export keeps only the current `xiao-buddy-*.bundle`; set
`EXPORT_LOCAL_KEEP_BUNDLES` or pass `--keep-bundles` to retain more local bundle
history.
Verify the finished export manifest before sharing it:

```bash
make verify-local-export
```

Run the strict live retrieval gate before demos, imports, or ranking changes:

```bash
make eval-gate
```

The target runs `scripts/eval_gate.sh`, which defaults to:

```bash
OFFLINE_EVAL=0 STRICT_EVAL=1 .venv/bin/python scripts/eval_smoke.py
```

The gate requires the embedding endpoint and local vector index to be available.
It checks board/target match, required content, and required citation for every
case in `data/corpus/eval_queries.jsonl`.

Run a focused retrieval case while tuning:

```bash
EVAL_CASE_IDS=eval-c5-dual-band make eval-gate
```

Run the smaller generated-answer gate when changing prompts, agent endpoints, or
streaming behavior:

```bash
make answer-eval
```

This calls the full RAG pipeline and checks that final answers include required
facts, required source URLs, matching inline `[source-id]` citations, live agent
output, and streamed token updates for the cases in
`data/corpus/answer_eval_queries.jsonl`. It also writes a compact structured
report to `dist/answer-quality/all.json`, including per-case pass flags, source
IDs, stream counts, first-token latency, and total latency percentiles.

Run one generated-answer case while debugging a prompt or endpoint:

```bash
ANSWER_EVAL_CASE_IDS=answer-c5-dual-band make answer-eval
```

To tune reranker context length against latency, compare windows over the strict
retrieval eval set:

```bash
make rerank-benchmark
```

The production default is `RERANK_TEXT_CHARS=3200`, which keeps richer evidence
available to the reranker while still bounding each candidate payload.

To verify the live reranker is helping on representative Seeed wiki topics, run
the focused reranker quality gate:

```bash
REQUEST_TIMEOUT_SECONDS=90 make rerank-quality
```

It builds one citation-matched positive and hard lexical negatives for each
query, then fails if the live reranker does not score the intended source above
the best negative. The default set covers XIAO boards, Grove Vision AI,
Raspberry Pi, Jetson, robotics, and SenseCAP documentation.

For a broader deployment audit across all retrieval evals, run:

```bash
REQUEST_TIMEOUT_SECONDS=90 make rerank-quality-all
```

That writes `dist/reranker-quality/all.json`, scores up to three
citation-matched positive chunks against hard lexical negatives, reports pass
rates by topic category, and fails if the all-case rate drops below the
configured threshold.
Override `RERANK_QUALITY_ALL_MIN_PASS_RATE` or
`RERANK_QUALITY_ALL_MAX_FAILURES` when validating a known experimental
reranker.

This writes:

- `data/index/xiao_vectors.json` - vector index manifest, chunk IDs, source hash
- `data/index/xiao_vectors.hnsw` - local ANN graph over normalized chunk embeddings

The `.f16`, HNSW, and FAISS index data files are generated artifacts and are
git-ignored so they do not exceed normal Git hosting limits. Rebuild them after
cloning or after changing the corpus. The manifest is kept in the repo so the
app can report the intended backend and expected chunk IDs, but the local data
file must exist for true ANN search.

For deployments where rebuilding the index on the app host is inconvenient,
package the local vector data file, upload the archive to object storage or a
release asset, then set `VECTOR_INDEX_ARCHIVE_URL` and
`VECTOR_INDEX_ARCHIVE_SHA256`:

```bash
make vector-artifact
```

The package step writes deterministic tarballs: if the index bytes and manifest
source hash are unchanged, rerunning the command produces the same archive SHA.
The metadata file is portable: it stores the archive as a co-located filename
and the manifest as a repo-relative path, not a machine-local absolute path.
By default it keeps only the current `*.tar.gz` artifact plus metadata in
`dist/vector-index`; set `VECTOR_ARTIFACT_KEEP` or pass `--keep-artifacts` to
retain more local archive history.

At runtime, if the manifest exists but the local vector data file is missing,
the app downloads the artifact, verifies the archive checksum when configured,
extracts the manifest-named data file, and then loads HNSW/FAISS normally. To
install the same artifact explicitly during deployment:

```bash
make install-vector-artifact
```

To verify an artifact-backed deployment without installing the index into the
repo, run health with artifact verification enabled:

```bash
python scripts/health_check.py --verify-artifacts --strict-warnings
```

Before publishing a handoff, verify that the current archive matches the
checked-in manifest and can restore into a clean index directory:

```bash
make verify-vector-artifact-restore
```

At query time the app embeds only the user/photo query, searches this local
vector index, merges vector candidates with lexical/high-trust curated matches,
reranks the evidence, and sends only the selected chunks to the final agent.

For local debugging or tiny indexes, the exact-scan backend is still available:

```bash
python scripts/build_wiki_vector_index.py --backend flat --batch-size 32
```

For memory-constrained deployments, an optional FAISS Product Quantization
backend is wired in without making FAISS a default Space dependency:

```bash
pip install faiss-cpu
python scripts/build_wiki_vector_index.py --backend faiss-pq --pq-m 64 --pq-nbits 8
```

Keep the full factual chunks in the corpus. HNSW/PQ compress and accelerate the
search structure; they should not replace board, sensor, robotics, or pinout
chunks with family-level summaries.

## Hosted Model Deployment

This Space is a thin UI and orchestration layer. It needs an embedding endpoint
for vector search, can use an optional reranker endpoint for better ordering, and
can call any OpenAI-compatible chat-completions endpoint for the final answer.
Use `AGENT_MAX_TOKENS` to cap generated answer length when a larger hosted model
streams quickly but takes too long to finish full responses.

One tested all-vLLM setup self-hosts the Qwen services on a single AMD Instinct
MI300X GPU:

| Service | Model | Port | App variable |
|---|---|---:|---|
| Multimodal embedder | `Qwen/Qwen3-VL-Embedding-2B` | 8000 | `EMBEDDING_BASE_URL` |
| Reranker | `Qwen/Qwen3-VL-Reranker-2B` | 8001 | `RERANK_BASE_URL` |
| Agent example | `Qwen/Qwen3.6-35B-A3B` | 8002 | `AGENT_BASE_URL` |

The tested setup uses a DigitalOcean AMD GPU Droplet with an MI300X, ROCm, and
vLLM. The 35B MoE model starts first with a larger memory reservation, then the
2B embedding and reranking models share the remaining GPU memory. The agent does
not have to be this Qwen model; any OpenAI-compatible chat endpoint works,
including llama.cpp or vLLM services exposing `/v1/chat/completions`.

See the full walkthrough in [`amd-droplet.md`](amd-droplet.md) for:

- opening the required droplet ports
- using Jupyter terminals inside the ROCm container
- serving the Qwen endpoint stack with `vllm serve`
- testing `/v1/models`, `/v1/embeddings`, `/v1/completions`, and `/v1/chat/completions`
- security notes for public HTTP endpoints

For a public Space, set the deployed endpoint URLs as Hugging Face Space
variables rather than committing them to the repo.

## Project Layout

- `app.py` - Gradio Blocks UI.
- `amd-droplet.md` - AMD MI300X / ROCm / vLLM deployment walkthrough for one hosted Qwen endpoint stack.
- `data/corpus/xiao_boards.json` - curated XIAO-only board facts, pin maps, gotchas, citations, and image URLs.
- `data/corpus/wiki_chunks.jsonl` - imported chunks from the official Seeed wiki markdown.
- `data/index/xiao_vectors.*` - optional local vector index built from the corpus for true query-time RAG.
- `data/corpus/support_examples.jsonl` - seed support examples for demo planning.
- `data/corpus/eval_queries.jsonl` - strict retrieval benchmark covering XIAO boards plus selected full-wiki sensor, robotics, LoRa, and AI workflows.
- `data/corpus/answer_eval_queries.jsonl` - generated-answer benchmark for final facts, citations, agent use, and streaming.
- `xiao_copilot/config.py` - environment-driven endpoint settings.
- `xiao_copilot/clients.py` - thin HTTP clients for embeddings, reranking, and chat completions.
- `xiao_copilot/knowledge_base.py` - corpus loader that turns the curated JSON into retrieval chunks.
- `xiao_copilot/retrieval.py` - embedding retrieval with lexical fallback and optional reranking.
- `xiao_copilot/image_utils.py` - uploaded image metadata summary.
- `xiao_copilot/pipeline.py` - question answering orchestration.

## Demo Scenarios

1. Upload a XIAO ESP32S3 Sense photo and ask: "What board is this, and is it good for image classification?"
2. Ask: "My XIAO ESP32C3 analog reading on A3 is noisy. Should I move pins?"
3. Ask: "How do I switch my XIAO ESP32C6 to the external antenna?"
4. Ask: "Which supported XIAO board should I pick for an image and audio TinyML demo?"
5. Ask: "What are the key radio specs and MCU interface for the Wio-SX1262 module?"
6. Ask: "What trigger actions can Grove Vision AI V2 perform from SenseCraft AI model output settings?"
7. Ask: "Which Arduino libraries and function are used to read Grove SHT40 data on Wio Terminal?"
8. Ask: "In the XIAO RS485 Expansion Board ESP32C3 example, which pins are used for RS485 UART RX/TX and enable?"

## Submission Notes

- Runtime target: hosted embedding, reranking, and OpenAI-compatible agent endpoints. The AMD MI300X/vLLM guide is one known-good deployment path, not a hard requirement.
- Public demo target: Hugging Face Space running this Gradio app and connecting to the hosted endpoints.
- Benchmark hooks: `make eval-gate` checks retrieval over `data/corpus/eval_queries.jsonl`; `make answer-eval` checks generated answers over `data/corpus/answer_eval_queries.jsonl`; `make verify-live` runs the full pre-demo live gate.
