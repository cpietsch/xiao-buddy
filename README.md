---
title: XIAO Field Copilot
emoji: 🔧
colorFrom: teal
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
  -> XIAO-only corpus retrieval
  -> optional hosted Qwen VL reranker
  -> lightweight route: identify | troubleshoot | compare | wiring_or_pinout | support
  -> hosted small Qwen3.5-style agent model
  -> cited answer + sources + run details
```

## Environment

Set these as Space secrets or variables:

```bash
EMBEDDING_BASE_URL=http://129.212.184.41:8000/v1/embeddings
EMBEDDING_MODEL=qwen3-vl-embedding-2b
EMBEDDING_API_KEY=

RERANK_BASE_URL=
RERANK_MODEL=qwen3-vl-reranker-2b
RERANK_API_KEY=

AGENT_BASE_URL=
AGENT_MODEL=qwen3.5-small
AGENT_API_KEY=

TOP_K=5
CANDIDATE_K=8
REQUEST_TIMEOUT_SECONDS=20
```

`EMBEDDING_BASE_URL` defaults to `http://129.212.184.41:8000/v1/embeddings`.

Expected hosted protocols:

- Embeddings: OpenAI-compatible text embeddings with `input`, plus Qwen/vLLM multimodal embeddings with `messages` containing an `image_url` and text.
- Reranker: placeholder support for `POST /rerank`, with fallback to `POST /v1/rerank`.
- Agent: OpenAI-compatible `POST /v1/chat/completions`.

On Hugging Face Spaces, add these as Space variables/secrets. The app calls the endpoints server-side, so HTTP endpoint URLs are fine for the Python backend.

## Local Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## Project Layout

- `app.py` - Gradio Blocks UI.
- `data/corpus/xiao_boards.json` - curated XIAO-only board facts, pin maps, gotchas, citations, and image URLs.
- `data/corpus/support_examples.jsonl` - seed support examples for demo planning.
- `data/corpus/eval_queries.jsonl` - tiny benchmark set for the submission.
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
- Benchmark hook: `data/corpus/eval_queries.jsonl` contains answer checks for retrieval, board identification, pin mapping, and troubleshooting.
