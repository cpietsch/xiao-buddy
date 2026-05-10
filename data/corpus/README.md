# XIAO Field Copilot Corpus

This is a hackathon-simple, XIAO-only seed corpus for a multimodal hardware support copilot. It is intentionally compact: one curated board facts file, one support-example file, and one eval-query file.

## Strategy

1. Scope only Seeed Studio XIAO boards and tightly related XIAO accessories: ESP32S3, ESP32C3, ESP32C5, ESP32C6, RP2040, RP2350, nRF52840, nRF54L15, SAMD21, RA4M1, MG24, and the XIAO W5500 Ethernet Adapter.
2. Prefer official Seeed wiki pages and official resource links as citations. Avoid forum lore unless it becomes a known failure case that needs a separate community-evidence bucket.
3. Keep each board doc small enough for direct retrieval: identity, aliases, MCU, wireless/sensor capabilities, pin map, bootloader notes, power caveats, and known gotchas.
4. Treat images as retrieval hints, not required truth. Store product and pinout image URLs where known, and let the app accept user-uploaded photos against board aliases, visual features, and pin labels.
5. Split support examples from facts. Examples model the desired field-copilot behavior: ask for the board variant when needed, cite sources, and give short hardware-safe next steps.
6. Use evals that test board identification, pin mapping, bootloader recovery, power/ADC caveats, and capability selection.

## Files

- `xiao_boards.json`: curated board/accessory facts, citations, pin maps, field gotchas, and support notes.
- `support_examples.jsonl`: few-shot style question/answer examples with expected citations.
- `eval_queries.jsonl`: small evaluation set for retrieval and answer checks.

## Notes For The App Layer

- Every answer should return citation URLs from the retrieved docs.
- If a user asks for code or wiring that touches power, battery, RF switches, boot pins, or ADC limits, answer conservatively and surface the relevant caveat.
- For multimodal queries, first classify the board from image cues, silkscreen labels, USB-C side layout, antenna/camera/mic modules, and pinout labels, then retrieve the matching board doc.
