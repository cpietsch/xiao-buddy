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

A hackathon-simple Hugging Face Space for multimodal hardware support around Seeed Studio XIAO boards.

The app accepts a board photo plus a question, retrieves cited XIAO support notes, optionally reranks them,
and produces a field-ready answer through an OpenAI-compatible Qwen agent endpoint. It also works without
the optional hosted reranker/agent services by falling back to lexical retrieval and a deterministic cited response.

Positioning:

> A multimodal field support copilot that identifies XIAO edge hardware from images, retrieves the right documentation, and guides troubleshooting in a grounded, citation-backed workflow connected to hosted vLLM deployments.

## Architecture

```text
photo + question
  -> Qwen3-VL embedding endpoint on vLLM
  -> full Seeed wiki corpus retrieval over local HNSW
  -> optional hosted Qwen VL reranker
  -> lightweight route: identify | troubleshoot | compare | wiring_or_pinout | support
  -> hosted small Qwen3.5-style agent model
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
AGENT_MODEL=qwen3p6-35b-a3b
AGENT_API_KEY=

TOP_K=5
CANDIDATE_K=8
VECTOR_CANDIDATE_K=96
VECTOR_INDEX_MANIFEST=data/index/xiao_vectors.json
VECTOR_INDEX_DATA=
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

Run the smaller generated-answer gate when changing prompts, agent endpoints, or
streaming behavior:

```bash
make answer-eval
```

This calls the full RAG pipeline and checks that final answers include required
facts, source URLs, live agent output, and streamed token updates for the cases
in `data/corpus/answer_eval_queries.jsonl`.

To tune reranker context length against latency, compare windows over the strict
retrieval eval set:

```bash
make rerank-benchmark
```

The production default is `RERANK_TEXT_CHARS=3200`, which keeps richer evidence
available to the reranker while still bounding each candidate payload.

Proposed cases that are useful but not yet reliable live in
`data/corpus/rejected_eval_candidates.jsonl`. Treat them as a retrieval tuning
backlog, not as a passing gate.

This writes:

- `data/index/xiao_vectors.json` - vector index manifest, chunk IDs, source hash
- `data/index/xiao_vectors.hnsw` - local ANN graph over normalized chunk embeddings

The HNSW/FAISS index data files are generated artifacts and are git-ignored so
they do not exceed normal Git hosting limits. Rebuild them after cloning or
after changing the corpus. The manifest is kept in the repo so the app can
report the intended backend and expected chunk IDs, but the local data file must
exist for true ANN search.

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

## AMD MI300X vLLM Deployment

This Space is a thin UI and orchestration layer. The three Qwen services can be
self-hosted on a single AMD Instinct MI300X GPU with vLLM:

| Service | Model | Port | App variable |
|---|---|---:|---|
| Multimodal embedder | `Qwen/Qwen3-VL-Embedding-2B` | 8000 | `EMBEDDING_BASE_URL` |
| Reranker | `Qwen/Qwen3-VL-Reranker-2B` | 8001 | `RERANK_BASE_URL` |
| Generator | `Qwen/Qwen3.6-35B-A3B` | 8002 | `AGENT_BASE_URL` |

The tested setup uses a DigitalOcean AMD GPU Droplet with an MI300X, ROCm, and
vLLM. The 35B MoE model starts first with a larger memory reservation, then the
2B embedding and reranking models share the remaining GPU memory. See the full
walkthrough in [`amd-droplet.md`](amd-droplet.md) for:

- opening the required droplet ports
- using Jupyter terminals inside the ROCm container
- serving all three models with `vllm serve`
- testing `/v1/models`, `/v1/embeddings`, `/v1/completions`, and `/v1/chat/completions`
- security notes for public HTTP endpoints

For a public Space, set the deployed endpoint URLs as Hugging Face Space
variables rather than committing them to the repo.

## Project Layout

- `app.py` - Gradio Blocks UI.
- `amd-droplet.md` - AMD MI300X / ROCm / vLLM deployment walkthrough for the hosted Qwen endpoints.
- `data/corpus/xiao_boards.json` - curated XIAO-only board facts, pin maps, gotchas, citations, and image URLs.
- `data/corpus/wiki_chunks.jsonl` - imported chunks from the official Seeed wiki markdown.
- `data/index/xiao_vectors.*` - optional local vector index built from the corpus for true query-time RAG.
- `data/corpus/support_examples.jsonl` - seed support examples for demo planning.
- `data/corpus/eval_queries.jsonl` - strict retrieval benchmark covering XIAO boards plus selected full-wiki sensor, robotics, LoRa, and AI workflows.
- `data/corpus/answer_eval_queries.jsonl` - generated-answer benchmark for final facts, citations, agent use, and streaming.
- `data/corpus/rejected_eval_candidates.jsonl` - retrieval hardening backlog for candidates that do not pass the live gate yet.
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

## Submission Notes

- Runtime target: AMD Developer Cloud / MI300X hosted vLLM endpoints for embedding, reranking, and agent generation.
- Public demo target: Hugging Face Space running this Gradio app and connecting to the hosted endpoints.
- Benchmark hooks: `make eval-gate` checks retrieval over `data/corpus/eval_queries.jsonl`; `make answer-eval` checks generated answers over `data/corpus/answer_eval_queries.jsonl`.
