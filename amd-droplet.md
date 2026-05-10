# Self-Hosting Qwen3-VL Embedding & Reranker + Qwen3.6-35B-A3B on AMD MI300X with vLLM

A working setup for serving multimodal embedding, reranking, and generative models on a single AMD MI300X GPU using vLLM. Built as an alternative to Fireworks AI, whose `/v1/embeddings` API turned out to be text-only even for multimodal models — verified via cosine-similarity testing where different images with the same text produced identical embeddings (cos = 1.0000). On self-hosted vLLM, the same test produced cos = 0.59, confirming the vision encoder is actually engaged.

## Hardware

- **GPU**: AMD Instinct MI300X (192 GB HBM)
- **Provider**: DigitalOcean GPU Droplet — **vLLM Droplet 1-Click** image, Ubuntu 24.04
- **Container**: AMD-published `rocm` image (pre-pulled and pre-started by the 1-Click), with vLLM and ROCm 6.x preinstalled, plus a Jupyter Lab server on port 8888
- **vLLM**: shipped inside the `rocm` image

The MI300X's 192 GB is well-matched for running all three models simultaneously: a 35B MoE generator plus the two 2B VL models, with headroom for batching.

## Setup overview

The 1-Click bootstraps a Docker container named `rocm` containing the ROCm runtime, vLLM, and a Jupyter Lab server bound to `0.0.0.0:8888`. We launch all three vLLM servers from terminals **inside Jupyter Lab**. Each Jupyter terminal acts as a long-lived shell session inside the container — closing the browser tab does not kill the process, which is exactly what we want.

The host is essentially a thin wrapper: Ubuntu + Docker + UFW.

## Opening ports on the droplet

The 1-Click ships with UFW enabled and only `22 (SSH)`, `80 (HTTP)`, and `443 (HTTPS)` open. Jupyter is reachable on port 80 (via a redirect the 1-Click sets up), but the three vLLM ports — `8000`, `8001`, `8002` — are blocked by UFW until you open them.

```bash
ufw status                       # confirm current rules
ufw allow 8000/tcp
ufw allow 8001/tcp
ufw allow 8002/tcp
ufw status                       # verify
```

> **Security note.** Once these ports are open, the vLLM endpoints are reachable from anywhere on the internet, served over **plain HTTP, with no authentication**. Anyone who finds the droplet's IP can use the GPU. For anything beyond solo dev work, restrict source IPs explicitly:
>
> ```bash
> # Replace the example IPs with your client IPs
> ufw delete allow 8000/tcp
> ufw delete allow 8001/tcp
> ufw delete allow 8002/tcp
> ufw allow from 203.0.113.42 to any port 8000 proto tcp
> ufw allow from 203.0.113.42 to any port 8001 proto tcp
> ufw allow from 203.0.113.42 to any port 8002 proto tcp
> ```
>
> Or front the ports with a reverse proxy that terminates TLS and adds auth (nginx + basic auth, Caddy + auth_request, etc.).

The Docker daemon's port mappings (`-p 8000:8000` etc. on the `rocm` container) are independent of UFW — both layers need to allow the traffic. If `docker port rocm` doesn't list a port you need, see "Adding port 8001" below.

## Accessing Jupyter

The droplet's MOTD prints the Jupyter URL and token on SSH login:

```
Access the Jupyter Server:
  * http://<droplet-ip>
  * Token: <token>
```

Two ways to get to it:

1. **Via port 80** (works out of the box, but plain HTTP — the token travels in the clear).
2. **SSH tunnel** (recommended for the token):

   ```bash
   ssh -L 8888:localhost:8888 root@<droplet-ip>
   ```

   Then open `http://localhost:8888` locally and paste the token.

Once inside Jupyter Lab: **File → New → Terminal** opens a shell already inside the container, at `/app`, with the full ROCm + vLLM environment. No `docker exec` needed.

## Models

| Model | Role | Size (BF16) | Port |
|---|---|---|---|
| `Qwen/Qwen3-VL-Embedding-2B` | Multimodal embedder | ~5 GB | 8000 |
| `Qwen/Qwen3-VL-Reranker-2B` | Multimodal reranker | ~5 GB | 8001 |
| `Qwen/Qwen3.6-35B-A3B` | Generative LLM (MoE, ~3B active) | ~70 GB | 8002 |

The two VL models are based on `Qwen3VLForConditionalGeneration` (model_type `qwen3_vl`). The 35B is a Mixture-of-Experts model — 35B total parameters, ~3B active per token — so generation throughput is closer to a dense 3B than a dense 35B.

## GPU memory budget

When co-locating multiple models on one MI300X, set `--gpu-memory-utilization` explicitly per process — vLLM defaults to 0.9, which means the first process to start grabs the whole card.

| Process | Util | ~GB on 192 GB |
|---|---|---|
| 35B-A3B | 0.55 | ~106 GB |
| Embedder (2B) | 0.15 | ~29 GB |
| Reranker (2B) | 0.15 | ~29 GB |
| **Total** | **0.85** | **~164 GB** |
| Headroom | 0.15 | ~28 GB |

**Always start the biggest model first.** vLLM's allocator grabs contiguous memory at startup. Small-first can leave the 35B unable to allocate a contiguous block even when total free space is sufficient.

## Workflow: three Jupyter terminals, one per model

Open **three Jupyter terminals** (File → New → Terminal, three times). Each terminal is independent and survives browser closes. Run one `vllm serve` per terminal. Use the Jupyter tab title to keep track — right-click the tab to rename it to `llm`, `embed`, `rerank`.

### Terminal 1 — 35B (port 8002), start first

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
- MoE on ROCm is newer than dense models. If startup errors with anything about `fused_moe`, try `--enforce-eager` to skip CUDA-graph capture, or fall back to a dense model.
- Try `--quantization fp8` to halve weight memory; FP8 on MI300X works for many architectures but verify quality on your task first.
- The HuggingFace cache lives at `/root/.cache/huggingface` *inside the container* and is lost on container rebuild. Set `HF_HOME=/shared-docker/hf` (or bind-mount the cache dir) if you want weights to persist.

Wait until you see `Application startup complete` before launching the next two — otherwise they'll race for memory during the slow weight-load phase.

### Terminal 2 — embedder (port 8000)

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
- `HIP_VISIBLE_DEVICES=0` — AMD's equivalent of `CUDA_VISIBLE_DEVICES`. All three processes share GPU 0 (there's only one MI300X).

Loaded direct from the HuggingFace hub. No prep needed.

### Terminal 3 — reranker (port 8001)

The reranker has a quirky chat-template config that crashes when loaded straight from the HF hub on transformers' processor loader. Workaround: download locally inside the container first, then serve from the local path. Use `/shared-docker` so the download survives container rebuilds.

```bash
cd /shared-docker
hf download Qwen/Qwen3-VL-Reranker-2B --local-dir ./qwen3-vl-reranker-2b

HIP_VISIBLE_DEVICES=0 vllm serve /shared-docker/qwen3-vl-reranker-2b \
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

## Verifying from the host

From an SSH session on the droplet, ports inside the container map to the same ports on the host:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models
curl http://localhost:8002/v1/models
```

From inside any Jupyter terminal, `rocm-smi` shows live GPU memory and utilization — useful to confirm all three processes loaded and the budget worked out.

Once UFW is open and vLLM is listening on `0.0.0.0`, the same `curl`s also work from your laptop with the droplet's public IP:

```bash
curl http://<droplet-ip>:8000/v1/models
```

## Adding port 8001 (if needed)

The default 1-Click container exposes `8000`, `8888`, and `30000`. If `docker port rocm` doesn't list `8001`, you need to recreate the container with the extra mapping — Docker port bindings are fixed at container creation and cannot be added to a running container.

```bash
# Save the current Jupyter token first
docker inspect rocm --format '{{range .Config.Env}}{{println .}}{{end}}' | grep JUPYTER_TOKEN

# Stop and recreate
docker rm -f rocm

docker run --name=rocm --hostname=rocm \
  -v /shared-docker:/shared-docker \
  -e SHELL=/bin/bash \
  -e JUPYTER_TOKEN=<paste the token from above> \
  --cap-add=CAP_SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt apparmor=unconfined \
  --group-add video --shm-size=16g \
  --network=bridge --workdir=/app \
  -p 8888:8888 \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -p 30000:30000 \
  --device /dev/kfd:/dev/kfd --device /dev/dri:/dev/dri \
  --runtime=runc -dt \
  rocm \
  /bin/sh -c 'jupyter lab --no-browser --notebook-dir=/shared-docker --ServerApp.allow_remote_access=true --IdentityProvider.token=${JUPYTER_TOKEN} --allow-root --ip=0.0.0.0'
```

Replace `<paste the token from above>` literally — the angle-bracket placeholder is not valid shell syntax (bash will try to read from a file called `paste`). After this, Jupyter is back up at the same URL and token, and `8001` is reachable.

## Endpoints summary

Replace `<droplet-ip>` with the droplet's public IPv4 throughout.

| Model | URL | Endpoint | Served name |
|---|---|---|---|
| Embedder | `http://<droplet-ip>:8000` | `POST /v1/embeddings` | `qwen3-vl-embedding-2b` |
| Reranker | `http://<droplet-ip>:8001` | `POST /v1/chat/completions` | `qwen3-vl-reranker-2b` |
| 35B | `http://<droplet-ip>:8002` | `POST /v1/chat/completions` | `qwen3p6-35b-a3b` |

To discover what each vLLM process actually exposes:

```bash
curl http://<droplet-ip>:8000/openapi.json | jq '.paths | keys'
curl http://<droplet-ip>:8001/openapi.json | jq '.paths | keys'
curl http://<droplet-ip>:8002/openapi.json | jq '.paths | keys'
```

## Querying the embedder

**Text:**

```bash
curl http://<droplet-ip>:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-vl-embedding-2b", "input": "A woman with her dog on a beach"}'
```

Returns a 2048-dim float vector. Batch by passing an array: `"input": ["text1", "text2", ...]`.

**Multimodal (the actual point of using this model):**

Use the `messages` chat-style shape, *not* the OpenAI content-parts `input` shape. The latter returns an empty embedding (the model card's vLLM example uses `messages`):

```bash
curl http://<droplet-ip>:8000/v1/embeddings \
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

tok = AutoTokenizer.from_pretrained("/shared-docker/qwen3-vl-reranker-2b")
YES_ID, NO_ID = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")

INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

def rerank(query, document, api="http://<droplet-ip>:8001/v1/chat/completions"):
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

Standard OpenAI chat-completions — works with the OpenAI Python SDK by setting `base_url="http://<droplet-ip>:8002/v1"` and any string as the API key:

```bash
curl http://<droplet-ip>:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3p6-35b-a3b",
    "messages": [{"role": "user", "content": "Explain MoE in one paragraph."}],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

## Operational notes

### Jupyter terminals and process lifetime

Jupyter terminals are server-side — they run inside the container, and the Jupyter Lab server is what holds them. Closing the browser tab, losing your laptop's wifi, or even rebooting your laptop does **not** kill the vLLM processes. Reconnect to Jupyter, click the terminal in the left-side **Running** panel, and you're back at the same scrollback.

What *does* kill them:
- `docker restart rocm` or `docker stop rocm` — the entire container goes down.
- The host rebooting.
- The Jupyter server itself crashing (rare, but the terminals are its children).

### After a host reboot

The container has `RestartPolicy: no`, so after a reboot:

```bash
docker start rocm                       # bring the container back
# then open Jupyter, open three terminals, relaunch each vllm serve
```

UFW rules persist across reboots, so you don't need to re-run the `ufw allow` commands. To survive reboots automatically at the container level, recreate the container with `--restart unless-stopped` added. The vLLM processes still need to be relaunched — they're not part of the container's CMD (Jupyter is). If you want true zero-touch restart for the vLLM processes themselves, that's the point where you'd reach for `docker compose` with one container per model — out of scope here.

### Persisting weights across container rebuilds

The HuggingFace cache is inside the container's overlay filesystem and disappears when you `docker rm` it. Either:

- Set `HF_HOME=/shared-docker/hf` in each terminal before launching `vllm serve`, or
- Add `-e HF_HOME=/shared-docker/hf` to the `docker run` line when rebuilding the container.

Saves an hour of re-downloading on every config change. The reranker is already stored under `/shared-docker/qwen3-vl-reranker-2b` so it's fine.

## What didn't work

- **Fireworks AI** for multimodal embedding: `Supports Image Input: true` on the model record turned out to be aspirational. `/v1/embeddings` runs all input through the text tokenizer regardless of `prompt_template` shape, URL/data-URI/base64 input format, or chat-template tokens. Image content is silently dropped. Confirmed by cosine similarity = 1.0000 between completely different images with identical text. Fireworks runs a proprietary inference engine (FireAttention) rather than vLLM, so they have to port multimodal API surfaces themselves and haven't yet for Qwen3-VL.
- **Fireworks AMD MI300X**: also blocked — new accounts default to NVIDIA H100 only, and MI300X requires quota approval.
- **Single GPU + native multimodal batching**: vLLM's `messages` field doesn't accept a list-of-conversations array. Use parallel client-side requests instead — vLLM's continuous batching handles the rest.

Self-hosting on vLLM gave us the multimodal API the model card actually documents, on hardware we control, with all three models cohabiting on one card.