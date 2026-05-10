# Self-Hosting Qwen3-VL Embedding & Reranker + Qwen3.6-35B-A3B on AMD MI300X with vLLM

A working setup for serving multimodal embedding, reranking, and generative models on a single AMD MI300X GPU using vLLM, exposed over Cloudflare Tunnel. Built as an alternative to Fireworks AI, whose `/v1/embeddings` API turned out to be text-only even for multimodal models — verified via cosine-similarity testing where different images with the same text produced identical embeddings (cos = 1.0000). On self-hosted vLLM, the same test produced cos = 0.59, confirming the vision encoder is actually engaged.

## Hardware

- **GPU**: AMD Instinct MI300X (192 GB HBM)
- **Provider**: DigitalOcean GPU Droplet — `vllm-0-17-1` marketplace image (`0.17.1-gpu-mi300x1-192gb-devcloud-atl1`), Ubuntu 24.04, Atlanta region (ATL1)
- **vLLM**: 0.17.1 (preinstalled in the marketplace image)

The MI300X's 192 GB is well-matched for running all three models simultaneously: a 35B MoE generator plus the two 2B VL models, with headroom for batching.

## Models

| Model | Role | Size (BF16) | Local port |
|---|---|---|---|
| `Qwen/Qwen3-VL-Embedding-2B` | Multimodal embedder | ~5 GB | 8000 |
| `Qwen/Qwen3-VL-Reranker-2B` | Multimodal reranker | ~5 GB | 8001 |
| `Qwen/Qwen3.6-35B-A3B` | Generative LLM (MoE, ~3B active) | ~70 GB | 8002 |

The two VL models are based on `Qwen3VLForConditionalGeneration` (model_type `qwen3_vl`). The 35B is a Mixture-of-Experts model — 35B total parameters, ~3B active per token — so generation throughput is closer to a dense 3B than a dense 35B.

## GPU memory budget

When co-locating multiple models on one MI300X, set `--gpu-memory-utilization` explicitly per process — vLLM defaults to 0.9, which means the first process to start grabs the whole card.

| Process | Util | ~GB on 192GB |
|---|---|---|
| 35B-A3B | 0.55 | ~106 GB |
| Embedder (2B) | 0.15 | ~29 GB |
| Reranker (2B) | 0.15 | ~29 GB |
| **Total** | **0.85** | **~164 GB** |
| Headroom | 0.15 | ~28 GB |

**Always start the biggest model first.** vLLM's allocator grabs contiguous memory at startup. Small-first can leave the 35B unable to allocate a contiguous block even when total free space is sufficient.

## Serving the 35B (port 8002) — start first

```bash
HIP_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.6-35B-A3B \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --served-model-name qwen3p6-35b-a3b \
  --gpu-memory-utilization 0.55 \
  --port 8002
```

Notes:
- First start takes 5–10 minutes (large weights, MoE kernel compilation). Don't kill it thinking it hung — watch for `Application startup complete`.
- MoE on ROCm is newer than dense models. If startup errors with anything about `fused_moe`, consider trying `--enforce-eager` to skip CUDA-graph capture, or fall back to a dense model.
- Try `--quantization fp8` if you want to halve weight memory; FP8 on MI300X works for many architectures but verify quality on your task before committing.

## Serving the embedder (port 8000)

```bash
HIP_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-Embedding-2B \
  --runner pooling \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --served-model-name qwen3-vl-embedding-2b \
  --gpu-memory-utilization 0.15 \
  --port 8000
```

Key flags:
- `--runner pooling` — embedding/pooling mode (returns hidden-state vectors instead of generating)
- `--trust-remote-code` — Qwen3-VL ships custom modeling code
- `--dtype bfloat16` — matches the on-disk weights, no upcast cost
- `HIP_VISIBLE_DEVICES=0` — AMD's equivalent of `CUDA_VISIBLE_DEVICES`

Loaded direct from the HuggingFace hub. No prep needed.

## Serving the reranker (port 8001)

The reranker has a quirky chat-template config that crashes when loaded straight from the HF hub on transformers' processor loader. Workaround: download locally, then serve from the local path.

```bash
# Download once
hf download Qwen/Qwen3-VL-Reranker-2B --local-dir ./qwen3-vl-reranker-2b

# Serve
HIP_VISIBLE_DEVICES=0 vllm serve ./qwen3-vl-reranker-2b \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --served-model-name qwen3-vl-reranker-2b \
  --gpu-memory-utilization 0.15 \
  --port 8001
```

Notes:
- **No `--runner pooling`** — this model is a generative scorer, not a pooler. It scores by emitting "yes" or "no" as the next token; you read the logprobs.
- vLLM logs a `TypeError: unhashable type: 'dict'` during chat-template warmup — this is a vLLM bug, not a model issue. The server starts and serves requests anyway.

## Exposing via Cloudflare Tunnel

Direct port exposure on the GPU droplet wasn't reachable from outside (likely DigitalOcean firewall/NAT on this image — couldn't get inbound traffic working through standard means). Cloudflare Tunnel solves this by making an outbound connection from the droplet to Cloudflare's edge — no inbound ports need to be open on the droplet at all. As a bonus you get HTTPS with a real cert and DDoS protection for free.

### Prerequisites

- A domain on Cloudflare (free plan is fine).
- Cloudflare account.

### One-time setup

```bash
# Install cloudflared on the droplet
curl -L --output cloudflared.deb \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# Login (opens a browser URL — copy/paste into your local browser, pick the domain)
cloudflared tunnel login

# Create the tunnel (generates credentials)
cloudflared tunnel create vllm-mi300x
# Output: Created tunnel vllm-mi300x with id <UUID>
# Credentials saved to /root/.cloudflared/<UUID>.json

# Route DNS for each subdomain to this tunnel
cloudflared tunnel route dns vllm-mi300x embed.example.com
cloudflared tunnel route dns vllm-mi300x rerank.example.com
cloudflared tunnel route dns vllm-mi300x llm.example.com
```

(Replace `example.com` with your domain.)

### Tunnel config

Create `/root/.cloudflared/config.yml`:

```yaml
tunnel: vllm-mi300x
credentials-file: /root/.cloudflared/<UUID>.json

ingress:
  - hostname: embed.example.com
    service: http://localhost:8000
  - hostname: rerank.example.com
    service: http://localhost:8001
  - hostname: llm.example.com
    service: http://localhost:8002
  - service: http_status:404
```

### Run the tunnel

For dev:

```bash
cloudflared tunnel run vllm-mi300x
```

For prod, install as a systemd service so it survives reboots:

```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared
```

### Lock it down with auth

By default, the three subdomains are now publicly reachable on the internet. Anyone who finds the URLs can use your GPU. Two reasonable options for auth — pick one:

**Option A: Cloudflare Access (zero-trust, recommended).** Cloudflare dashboard → Zero Trust → Access → Applications → Add. Apply to `*.example.com`. Choose an auth method (email PIN, Google SSO, GitHub, etc.). Cloudflare handles the auth at their edge before traffic hits your tunnel. No code changes; vLLM never sees unauthenticated requests.

**Option B: Service token / bearer token.** Use Cloudflare Access with a service token, then your client sends the token as a header on every request. More work to set up but works well for headless / programmatic clients.

For dev only, you can leave it unauthenticated, but treat the URLs as semi-secret and don't share them.

## Endpoints summary

| Model | Public URL | Endpoint | Served name |
|---|---|---|---|
| Embedder | `https://embed.example.com` | `POST /v1/embeddings` | `qwen3-vl-embedding-2b` |
| Reranker | `https://rerank.example.com` | `POST /v1/chat/completions` | `qwen3-vl-reranker-2b` |
| 35B | `https://llm.example.com` | `POST /v1/chat/completions` | `qwen3p6-35b-a3b` |

To discover what each vLLM process actually exposes:

```bash
curl https://embed.example.com/openapi.json | jq '.paths | keys'
curl https://rerank.example.com/openapi.json | jq '.paths | keys'
curl https://llm.example.com/openapi.json | jq '.paths | keys'
```

## Querying the embedder

**Text:**

```bash
curl https://embed.example.com/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-vl-embedding-2b", "input": "A woman with her dog on a beach"}'
```

Returns a 2048-dim float vector. Batch by passing an array: `"input": ["text1", "text2", ...]`.

**Multimodal (the actual point of using this model):**

Use the `messages` chat-style shape, *not* the OpenAI content-parts `input` shape. The latter returns an empty embedding (the model card's vLLM example uses `messages`):

```bash
curl https://embed.example.com/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-vl-embedding-2b",
    "messages": [
      {"role": "system", "content": "Represent the user'"'"'s input."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}},
        {"type": "text", "text": "a cat on a couch"}
      ]}
    ]
  }'
```

For batch multimodal, fire requests in parallel with `asyncio.gather`. vLLM's continuous batching merges them server-side automatically — within a few percent of native-batch throughput.

## Querying the reranker

Use the chat-completions API with logprobs. The Qwen3 reranker family scores using a templated prompt and the next-token probability of "yes" vs "no":

```python
from transformers import AutoTokenizer
import json, math
from urllib.request import Request, urlopen

tok = AutoTokenizer.from_pretrained("./qwen3-vl-reranker-2b")
YES_ID, NO_ID = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")

INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

def rerank(query, document, api="https://rerank.example.com/v1/chat/completions"):
    prompt = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {document}"
    payload = {
        "model": "qwen3-vl-reranker-2b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "logprobs": True,
        "top_logprobs": 20,
        "temperature": 0.0,
    }
    req = Request(api, method="POST",
                  data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
    body = json.loads(urlopen(req).read())
    top = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    yes_lp = next((e["logprob"] for e in top if e.get("token", "").strip().lower() == "yes"), -1e9)
    no_lp  = next((e["logprob"] for e in top if e.get("token", "").strip().lower() == "no"),  -1e9)
    return math.exp(yes_lp) / (math.exp(yes_lp) + math.exp(no_lp))
```

## Querying the 35B

Standard OpenAI chat-completions — works with the OpenAI Python SDK by setting `base_url="https://llm.example.com/v1"` and any string as the API key:

```bash
curl https://llm.example.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3p6-35b-a3b",
    "messages": [{"role": "user", "content": "Explain MoE in one paragraph."}],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

## Cloudflare Tunnel — operational notes

- **Cold-start aggregation.** First request after long idle goes: client → Cloudflare edge → tunnel → localhost. ~50–200 ms overhead per request vs direct, mostly TLS termination at the edge. Negligible for inference workloads.
- **Streaming works fine.** Cloudflare proxies chunked responses, so SSE / streaming chat completions from the 35B endpoint work without special config.
- **Large request bodies.** Multimodal embedding with base64 images can produce multi-MB request bodies. Cloudflare's free plan caps request body at 100 MB, which is plenty for typical images. For very large images, prefer image URLs (the model fetches them server-side).
- **Tunnel monitoring.** `cloudflared` exposes Prometheus metrics on `:20241/metrics` if you want to track edge latency, error rates, etc.
- **Upgrade path.** If you outgrow the tunnel approach (latency-sensitive workloads, very high QPS, or you want regional pinning), switch to a proper VPS with the GPU droplet on the same private network — but for now, the tunnel removes the entire networking-config problem.

## What didn't work

- **Fireworks AI** for multimodal embedding: `Supports Image Input: true` on the model record turned out to be aspirational. `/v1/embeddings` runs all input through the text tokenizer regardless of `prompt_template` shape, URL/data-URI/base64 input format, or chat-template tokens. Image content is silently dropped. Confirmed by cosine similarity = 1.0000 between completely different images with identical text. Fireworks runs a proprietary inference engine (FireAttention) rather than vLLM, so they have to port multimodal API surfaces themselves and haven't yet for Qwen3-VL.
- **Fireworks AMD MI300X**: also blocked — new accounts default to NVIDIA H100 only, and MI300X requires quota approval.
- **Direct port exposure on the DigitalOcean GPU droplet**: couldn't get inbound traffic working — Cloudflare Tunnel was the path that actually worked. Bonus: real HTTPS, no firewall config, no public IP exposed, optional Cloudflare Access for auth.
- **Single GPU + native multimodal batching**: vLLM's `messages` field doesn't accept a list-of-conversations array. Use parallel client-side requests instead — vLLM's continuous batching handles the rest.

Self-hosting on vLLM gave us the multimodal API the model card actually documents, on hardware we control, with all three models cohabiting on one card and exposed cleanly over Cloudflare Tunnel.