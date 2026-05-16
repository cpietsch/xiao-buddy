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
EMBEDDING_BASE_URL=https://your-embed-host/v1/embeddings
EMBEDDING_MODEL=qwen3-vl-embedding-2b
EMBEDDING_API_KEY=

RERANK_BASE_URL=https://your-rerank-host/v1
RERANK_MODEL=qwen3-vl-reranker-2b
RERANK_API_KEY=
RERANK_TEXT_CHARS=3200

AGENT_BASE_URL=https://your-llm-host/v1
AGENT_MODEL=your-agent-model-id
AGENT_API_KEY=

TOP_K=5
CANDIDATE_K=16
VECTOR_CANDIDATE_K=96
VECTOR_INDEX_MANIFEST=data/index/xiao_vectors.json
VECTOR_INDEX_DATA=
VECTOR_INDEX_ARCHIVE_URL=
VECTOR_INDEX_ARCHIVE_SHA256=
REQUEST_TIMEOUT_SECONDS=60
```

Endpoint URLs are intentionally environment-only. Do not commit local IPs or tunnel URLs.

Expected hosted protocols:

- Embeddings: OpenAI-compatible text embeddings with `input`, plus Qwen/vLLM multimodal embeddings with `messages` containing an `image_url` and text.
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

To refresh the imported XIAO wiki chunks:

```bash
python scripts/import_seeed_wiki_xiao.py --refresh
```

To move from the XIAO-only demo corpus to the full Seeed wiki, import the whole
docs tree and then rebuild the ANN index:

```bash
python scripts/import_seeed_wiki_xiao.py --scope all --refresh
python scripts/build_wiki_vector_index.py --backend hnsw --batch-size 32
```

The importer creates a sparse checkout of
`Seeed-Studio/wiki-documents` under `.cache/seeed-wiki`, filters to
`sites/en/docs/Sensor/SeeedStudio_XIAO` by default, cleans Docusaurus/MDX
markdown, splits by headings, preserves source URLs and image URLs, infers the
matching XIAO board family when possible, and writes
`data/corpus/wiki_chunks.jsonl`. With `--scope all`, the sparse checkout expands
to `sites/en/docs` and imports boards, sensors, robotics, and other Seeed docs
into the same chunk file.

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
package/install behavior, health parser behavior, and a strict lexical subset of
the retrieval eval that does not require hosted embedding, reranker, or agent
endpoints.

Check local readiness, corpus/index coverage, manifest source hash, and generated-artifact state:

```bash
make health
make health-live
```

`make health-live` also checks the configured hosted endpoints, verifies that
`/v1/models` exposes the configured model IDs, checks the local Gradio app URL,
and exits non-zero on warnings.

To prove the hosted endpoints accept real inference payloads, run the functional
endpoint smoke. It sends one small embedding request, one rerank request, and one
short chat completion request:

```bash
REQUEST_TIMEOUT_SECONDS=90 make health-functional
```

Check the running Gradio `/ask` API wiring, streamed answer events, source
panel, inline source IDs, diagnostics, and progress HTML:

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

This runs the live API/retrieval gates, the functional endpoint smoke, and then
the agent-backed browser smoke.

If GitHub push access is unavailable, export the unpushed local commits before
moving machines or sharing the workspace:

```bash
make export-local
```

This writes an ignored `dist/local-export/` folder containing a verified git
bundle, a `git format-patch` series, and a manifest with base/head commit IDs.

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
`data/corpus/answer_eval_queries.jsonl`.

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

This writes:

- `data/index/xiao_vectors.json` - vector index manifest, chunk IDs, source hash
- `data/index/xiao_vectors.hnsw` - local ANN graph over normalized chunk embeddings

The HNSW/FAISS index data files are generated artifacts and are git-ignored so
they do not exceed normal Git hosting limits. Rebuild them after cloning or
after changing the corpus. The manifest is kept in the repo so the app can
report the intended backend and expected chunk IDs, but the local data file must
exist for true ANN search.

For deployments where rebuilding the index on the app host is inconvenient,
package the local vector data file, upload the archive to object storage or a
release asset, then set `VECTOR_INDEX_ARCHIVE_URL` and
`VECTOR_INDEX_ARCHIVE_SHA256`:

```bash
make vector-artifact
```

At runtime, if the manifest exists but the local vector data file is missing,
the app downloads the artifact, verifies the archive checksum when configured,
extracts the manifest-named data file, and then loads HNSW/FAISS normally. To
install the same artifact explicitly during deployment:

```bash
make install-vector-artifact
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
