# XIAO Field Copilot Corpus

This is a hackathon-simple Seeed hardware corpus for a multimodal support copilot. It combines compact curated XIAO board facts with imported official Seeed wiki chunks across boards, sensors, robotics, LoRa, SenseCAP, and edge workflows.

## Strategy

1. Keep the curated XIAO board/accessory facts as high-trust seed chunks: ESP32S3, ESP32C3, ESP32C5, ESP32C6, RP2040, RP2350, nRF52840, nRF54L15, SAMD21, RA4M1, MG24, and the XIAO W5500 Ethernet Adapter.
2. Layer the full official Seeed wiki on top for sensors, robotics, LoRa, SenseCAP, Raspberry Pi HATs, Jetson workflows, and other hardware support docs.
3. Prefer official Seeed wiki pages and official resource links as citations. Avoid forum lore unless it becomes a known failure case that needs a separate community-evidence bucket.
4. Keep each curated board doc small enough for direct retrieval: identity, aliases, MCU, wireless/sensor capabilities, pin map, bootloader notes, power caveats, and known gotchas.
5. Treat images as retrieval hints, not required truth. Store product and pinout image URLs where known, and let the app accept user-uploaded photos against board aliases, visual features, and pin labels.
6. Split support examples from facts. Examples model the desired field-copilot behavior: ask for the board variant when needed, cite sources, and give short hardware-safe next steps.
7. Use evals that test board identification, pin mapping, bootloader recovery, power/ADC caveats, capability selection, and imported full-wiki workflows.

## Files

- `xiao_boards.json`: curated board/accessory facts, citations, pin maps, field gotchas, and support notes.
- `wiki_chunks.jsonl`: generated chunks from official Seeed wiki markdown. Rebuild the full wiki with `make import-wiki-all`, or a tiny XIAO-only smoke corpus with `make import-wiki-xiao`.
- `support_examples.jsonl`: few-shot style question/answer examples with expected citations.
- `eval_queries.jsonl`: strict retrieval evaluation set for board, citation, and full-wiki content checks.
- `answer_eval_queries.jsonl`: generated-answer evaluation set for final answer facts, source URLs, inline citation IDs, agent usage, and streaming.

To refresh the full Seeed wiki corpus:

```bash
make import-wiki-all
```

## Vector Index

After refreshing `wiki_chunks.jsonl`, build the local vector index:

```bash
make build-hnsw
```

The app will use `data/index/xiao_vectors.json` and
the data file named in the manifest, usually `data/index/xiao_vectors.hnsw`.
If the index is missing, retrieval falls back to embedding a smaller lexical
candidate pool per request.

The HNSW/FAISS data files are local generated artifacts and are ignored by git.
Run the build command after cloning or after changing `wiki_chunks.jsonl`.
For deployment hosts that should not rebuild the index, run
`make vector-artifact`, upload the archive, and set
`VECTOR_INDEX_ARCHIVE_URL` plus `VECTOR_INDEX_ARCHIVE_SHA256`. The app can
restore the missing manifest-named data file from that artifact at runtime, or
you can install it explicitly with `make install-vector-artifact`. The packaged
archive is deterministic for unchanged index bytes, so the SHA is stable across
repeated packaging runs.

To prove an artifact-backed deployment can restore the index, run:

```bash
python scripts/health_check.py --verify-artifacts --strict-warnings
```

For tiny/local indexes you can still build an exact flat file:

```bash
python scripts/build_wiki_vector_index.py --backend flat --batch-size 32
```

For very large deployments where RAM/disk matter more than exact vector scores,
install FAISS separately and build a Product Quantization index:

```bash
pip install faiss-cpu
make build-faiss-pq
```

## Notes For The App Layer

- Every answer should return citation URLs from the retrieved docs.
- If a user asks for code or wiring that touches power, battery, RF switches, boot pins, or ADC limits, answer conservatively and surface the relevant caveat.
- For multimodal queries, first classify the board from image cues, silkscreen labels, USB-C side layout, antenna/camera/mic modules, and pinout labels, then retrieve the matching board doc.
- Reranking sends a bounded text window per candidate. Tune `RERANK_TEXT_CHARS`
  with `python scripts/benchmark_rerank_window.py --windows 900,1600,2400,3200`
  before changing the production default.
- The live reranker quality gate uses representative cases from
  `eval_queries.jsonl`, one citation-matched positive chunk, and hard lexical
  negatives. Run it with `make rerank-quality` after deployment changes.
- Run `make rerank-quality-all` for an all-case audit with category summaries
  and a JSON report under `dist/reranker-quality/`.
