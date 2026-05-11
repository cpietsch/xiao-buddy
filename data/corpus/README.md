# XIAO Field Copilot Corpus

This is a hackathon-simple, XIAO-only corpus for a multimodal hardware support copilot. It combines compact curated board facts with imported official Seeed wiki chunks.

## Strategy

1. Scope only Seeed Studio XIAO boards and tightly related XIAO accessories: ESP32S3, ESP32C3, ESP32C5, ESP32C6, RP2040, RP2350, nRF52840, nRF54L15, SAMD21, RA4M1, MG24, and the XIAO W5500 Ethernet Adapter.
2. Prefer official Seeed wiki pages and official resource links as citations. Avoid forum lore unless it becomes a known failure case that needs a separate community-evidence bucket.
3. Keep each curated board doc small enough for direct retrieval: identity, aliases, MCU, wireless/sensor capabilities, pin map, bootloader notes, power caveats, and known gotchas.
4. Treat images as retrieval hints, not required truth. Store product and pinout image URLs where known, and let the app accept user-uploaded photos against board aliases, visual features, and pin labels.
5. Split support examples from facts. Examples model the desired field-copilot behavior: ask for the board variant when needed, cite sources, and give short hardware-safe next steps.
6. Use evals that test board identification, pin mapping, bootloader recovery, power/ADC caveats, capability selection, and imported wiki-only workflows.

## Files

- `xiao_boards.json`: curated board/accessory facts, citations, pin maps, field gotchas, and support notes.
- `wiki_chunks.jsonl`: generated chunks from official Seeed wiki XIAO markdown. Rebuild with `python scripts/import_seeed_wiki_xiao.py --refresh`.
- `support_examples.jsonl`: few-shot style question/answer examples with expected citations.
- `eval_queries.jsonl`: small evaluation set for retrieval and answer checks.

## Vector Index

After refreshing `wiki_chunks.jsonl`, build the local vector index:

```bash
python scripts/build_wiki_vector_index.py --batch-size 32
```

The app will use `data/index/xiao_vectors.json` and
`data/index/xiao_vectors.f32` when present. If the index is missing, retrieval
falls back to embedding a smaller lexical candidate pool per request.

## Notes For The App Layer

- Every answer should return citation URLs from the retrieved docs.
- If a user asks for code or wiring that touches power, battery, RF switches, boot pins, or ADC limits, answer conservatively and surface the relevant caveat.
- For multimodal queries, first classify the board from image cues, silkscreen labels, USB-C side layout, antenna/camera/mic modules, and pinout labels, then retrieve the matching board doc.
