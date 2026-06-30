from __future__ import annotations

import base64
import html
import io
import json
import re
from collections.abc import Iterator
from time import perf_counter

import gradio as gr
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from xiao_copilot.clients import chat_completion_stream
from xiao_copilot.config import load_settings
from xiao_copilot.pipeline import answer_question_stream as answer_question_stream_raw
from xiao_copilot.pipeline import format_progress
from xiao_copilot.retrieval import warm_retrieval_caches


EXAMPLES = [
    [
        None,
        "My XIAO ESP32S3 Sense is not detected over USB. What should I check first?",
    ],
    [
        None,
        "Which pins are I2C on the XIAO ESP32C6?",
    ],
    [
        None,
        "Which XIAO should I choose for 5 GHz WiFi?",
    ],
    [
        None,
        "My XIAO MG24 stopped accepting uploads after deep sleep. What should I try?",
    ],
    [
        None,
        "What SPI pins does the XIAO W5500 Ethernet Adapter use?",
    ],
    [
        None,
        "What board target string should I select for XIAO nRF54L15 Sense Zigbee in nRF Connect SDK?",
    ],
    [
        None,
        "At what I2C address is the SSD1306 OLED configured for XIAO ESP32C3 with Zephyr?",
    ],
    [
        None,
        "How do I switch the XIAO ESP32C6 to the external antenna?",
    ],
    [
        None,
        "What is the safe battery-voltage read sequence on XIAO nRF52840 Sense?",
    ],
    [
        None,
        "Which editor setup does the XIAO ESP32C6 CircuitPython guide recommend?",
    ],
    [
        None,
        "How do I configure WiFi for a XIAO ESP32C3 ESPHome node in Home Assistant?",
    ],
    [
        None,
        "My XIAO RP2040 serial port disappeared after flashing. How do I recover it?",
    ],
    [
        None,
        "What visible markings should I photograph so you can identify a XIAO board?",
    ],
    [
        None,
        "What are the key radio specs and MCU interface for the Wio-SX1262 module?",
    ],
    [
        None,
        "What trigger actions can Grove Vision AI V2 perform from SenseCraft AI model output settings?",
    ],
    [
        None,
        "Which Arduino libraries and function are used to read Grove SHT40 data on Wio Terminal?",
    ],
    [
        None,
        "In the XIAO RS485 Expansion Board ESP32C3 example, which pins are used for RS485 UART RX/TX and enable?",
    ],
    [
        None,
        "What voltage, MCU, ADC, Grove ports, bus, and I2C address does Grove Base Hat for Raspberry Pi Zero use?",
    ],
    [
        None,
        "For the Reachy Mini fleet dance demo, which Jetson-side and laptop-side services run, and what ports are used?",
    ],
    [
        None,
        "In the Jetson Thor OpenClaw SO-Arm guide, what roles do OpenClaw and LeRobot play?",
    ],
]

INITIAL_ANSWER = (
    "Ask about a XIAO board, Seeed sensor, robotics kit, LoRa module, pinout, upload issue, or power symptom."
)
API_HISTORY_MAX_MESSAGES = 8
API_HISTORY_MAX_CHARS = 3200
API_HISTORY_ITEM_MAX_CHARS = 900


THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="zinc",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    radius_size="sm",
).set(
    button_primary_background_fill="#0f8f6b",
    button_primary_background_fill_hover="#0b7558",
    block_border_width="1px",
    block_shadow="0 1px 2px rgb(15 23 42 / 0.08)",
)

CSS = """
html {
    color-scheme: light dark;
}
.gradio-container {
    max-width: 1220px !important;
    background:
        linear-gradient(180deg, #f8fafc 0%, #f5f9fb 48%, #f8fafc 100%);
    padding-left: max(1rem, env(safe-area-inset-left)) !important;
    padding-right: max(1rem, env(safe-area-inset-right)) !important;
}
.dark .gradio-container {
    background: linear-gradient(180deg, #020617 0%, #07111f 52%, #020617 100%);
}
.skip-link {
    background: #0f8f6b;
    border-radius: 8px;
    color: #ffffff;
    font-weight: 720;
    left: max(0.5rem, env(safe-area-inset-left));
    padding: 0.65rem 0.85rem;
    position: fixed;
    top: max(0.5rem, env(safe-area-inset-top));
    transition: none !important;
    transform: translateY(-240%);
    z-index: 20;
}
.skip-link:focus,
.skip-link:focus-visible {
    outline: 3px solid #34d399 !important;
    outline-offset: 2px !important;
    transform: translateY(0) !important;
}
.hero {
    padding: 1.35rem 0 0.7rem;
    border-bottom: 1px solid #dbe7e1;
    margin-bottom: 1rem;
}
.dark .hero {
    border-bottom: 1px solid #1e293b;
}
.hero h1 {
    color: #0f172a;
    font-size: 2.45rem;
    line-height: 1.1;
    margin: 0 0 0.35rem;
    font-weight: 760;
    letter-spacing: 0;
    text-wrap: balance;
}
.dark .hero h1 {
    color: #f8fafc;
}
.hero p {
    color: #334155;
    font-size: 1.02rem;
    margin: 0;
    max-width: 780px;
    text-wrap: balance;
}
.dark .hero p {
    color: #cbd5e1;
}
.chips {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.85rem;
}
.chip {
    background: #ffffff;
    border: 1px solid #cbded5;
    border-radius: 999px;
    color: #235245;
    font-size: 0.84rem;
    font-weight: 650;
    line-height: 1;
    padding: 0.42rem 0.62rem;
    white-space: nowrap;
}
.dark .chip {
    background: #0f172a;
    border: 1px solid #334155;
    color: #34d399;
}
.chip-board {
    border-color: #b7dccd;
    color: #0f513f;
}
.chip-radio {
    border-color: #bfdbfe;
    color: #1d4ed8;
}
.chip-vision {
    border-color: #ddd6fe;
    color: #6d28d9;
}
.chip-platform {
    border-color: #fed7aa;
    color: #9a3412;
}
.chip-robotics {
    border-color: #fecdd3;
    color: #be123c;
}
.dark .chip-board {
    border-color: #14532d;
    color: #86efac;
}
.dark .chip-radio {
    border-color: #1e3a8a;
    color: #93c5fd;
}
.dark .chip-vision {
    border-color: #4c1d95;
    color: #c4b5fd;
}
.dark .chip-platform {
    border-color: #7c2d12;
    color: #fdba74;
}
.dark .chip-robotics {
    border-color: #881337;
    color: #fda4af;
}
.bench-strip {
    display: grid;
    gap: 0.55rem;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin: 0 0 1rem;
}
.bench-card {
    background: #ffffff;
    border: 1px solid #dbe7e1;
    border-radius: 8px;
    color: #334155;
    font-size: 0.9rem;
    min-height: 74px;
    padding: 0.75rem 0.85rem;
    overflow-wrap: anywhere;
}
.dark .bench-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    color: #cbd5e1;
}
.bench-card strong {
    color: #0f513f;
    display: block;
    font-size: 0.74rem;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
}
.dark .bench-card strong {
    color: #10b981;
}
.bench-label {
    color: #0f513f;
    font-size: 0.78rem;
    font-weight: 760;
    letter-spacing: 0.08em;
    margin: 0 0 0.5rem;
    text-transform: uppercase;
}
.dark .bench-label {
    color: #10b981;
}
.panel-copy {
    color: #475569;
    font-size: 0.92rem;
    margin: -0.25rem 0 0.75rem;
}
.dark .panel-copy {
    color: #94a3b8;
}
.progress-box {
    margin-bottom: 0.75rem;
}
.progress-panel {
    background: #ffffff;
    border: 1px solid #dbe7e1;
    border-radius: 8px;
    padding: 0.85rem 1rem;
}
.dark .progress-panel {
    background: #0f172a;
    border-color: #334155;
}
.progress-title {
    color: #0f513f;
    font-size: 0.76rem;
    font-weight: 760;
    letter-spacing: 0.08em;
    margin: 0 0 0.65rem;
    text-transform: uppercase;
}
.dark .progress-title {
    color: #10b981;
}
.progress-list {
    display: grid;
    gap: 0.45rem;
    list-style: none;
    margin: 0;
    padding: 0;
}
.progress-step {
    align-items: center;
    border-left: 3px solid transparent;
    border-radius: 6px;
    color: #64748b;
    display: grid;
    gap: 0.6rem;
    grid-template-columns: 1.6rem minmax(0, 1fr);
    min-height: 2.3rem;
    padding: 0.1rem 0 0.1rem 0.35rem;
}
.progress-number {
    align-items: center;
    border: 1px solid #cbd5e1;
    border-radius: 999px;
    display: inline-flex;
    font-size: 0.78rem;
    font-variant-numeric: tabular-nums;
    font-weight: 760;
    height: 1.6rem;
    justify-content: center;
    width: 1.6rem;
}
.progress-step strong {
    color: #334155;
    display: block;
    font-size: 0.9rem;
    line-height: 1.15;
}
.progress-step small {
    color: #64748b;
    display: block;
    font-size: 0.78rem;
    line-height: 1.2;
    margin-top: 0.12rem;
    overflow-wrap: anywhere;
}
.progress-step-active {
    background: #f0fdfa;
    border-left-color: #0f8f6b;
}
.progress-step-active .progress-number {
    background: #0f8f6b;
    border-color: #0f8f6b;
    color: #ffffff;
}
.progress-step-active strong {
    color: #0f513f;
}
.progress-step-done .progress-number {
    background: #ecfdf5;
    border-color: #34d399;
    color: #0f513f;
}
.progress-step-done strong {
    color: #0f172a;
}
.dark .progress-step {
    color: #94a3b8;
}
.dark .progress-step-active {
    background: #042f2e;
    border-left-color: #10b981;
}
.dark .progress-number {
    border-color: #475569;
}
.dark .progress-step strong {
    color: #cbd5e1;
}
.dark .progress-step small {
    color: #94a3b8;
}
.dark .progress-step-active .progress-number {
    background: #10b981;
    border-color: #10b981;
    color: #052e16;
}
.dark .progress-step-active strong {
    color: #34d399;
}
.dark .progress-step-done .progress-number {
    background: #052e16;
    border-color: #10b981;
    color: #86efac;
}
.dark .progress-step-done strong {
    color: #f8fafc;
}
.block.result-box {
    background-color: #ffffff;
    border: 1px solid #dbe7e1;
    border-radius: 8px;
    min-height: 260px;
    padding: 1rem 1.1rem;
}
.block.result-box .prose.result-box {
    background: transparent;
    border: 0;
    border-radius: 0;
    min-height: 0;
    padding: 0;
}
.block.result-box .md,
.block.sources-box .md {
    overflow-wrap: anywhere;
}
.block.result-box .md p,
.block.result-box .md li {
    line-height: 1.55;
}
.block.result-box .md ul,
.block.result-box .md ol {
    padding-left: 1.15rem;
}
.block.result-box .md code,
.block.sources-box .md code {
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    color: #0f172a;
    padding: 0.08rem 0.24rem;
}
.dark .block.result-box {
    background-color: #0f172a;
    border-color: #334155;
    color: #e2e8f0;
}
.block.sources-box {
    background-color: #ffffff;
    border: 1px solid #dbe7e1;
    border-radius: 8px;
    max-height: min(58vh, 680px);
    overflow: auto;
    padding: 0.85rem 1rem;
    scroll-behavior: smooth;
}
.block.sources-box:has(.md:empty) {
    display: none;
}
.block.sources-box .prose.sources-box {
    background: transparent;
    border: 0;
    border-radius: 0;
    padding: 0;
}
.block.sources-box .md > p:first-child {
    color: #0f513f;
    font-size: 0.76rem;
    font-weight: 760;
    letter-spacing: 0.08em;
    margin-bottom: 0.45rem;
}
.source-snippets-title {
    color: #0f513f;
    font-size: 0.76rem;
    font-weight: 760;
    letter-spacing: 0.08em;
    margin: 0 0 0.55rem;
}
.block.sources-box .md ul {
    list-style: none;
    margin: 0;
    padding: 0;
}
.block.sources-box .md li {
    border-top: 1px solid #e2e8f0;
    line-height: 1.35;
    padding: 0.55rem 0;
}
.block.sources-box .md li:first-child {
    border-top: 0;
}
.block.sources-box .md a {
    font-weight: 650;
}
.source-snippet {
    border-top: 1px solid #e2e8f0;
    padding: 0.72rem 0;
    scroll-margin-top: max(1rem, env(safe-area-inset-top));
}
.source-snippet:first-of-type {
    border-top: 0;
    padding-top: 0.1rem;
}
.source-snippet:focus,
.source-snippet:target {
    outline: 3px solid #34d399;
    outline-offset: 3px;
}
.source-snippet-kicker {
    color: #0f513f;
    font-size: 0.72rem;
    font-weight: 760;
    letter-spacing: 0.08em;
    margin: 0 0 0.25rem;
}
.source-snippet-title {
    color: #0f172a;
    font-size: 0.95rem;
    font-weight: 700;
    line-height: 1.25;
    margin: 0;
}
.source-snippet-url {
    color: #64748b;
    font-size: 0.76rem;
    line-height: 1.25;
    margin: 0.2rem 0 0;
    overflow-wrap: anywhere;
}
.source-snippet-text {
    color: #334155;
    display: block;
    font-size: 0.86rem;
    line-height: 1.45;
    margin: 0.45rem 0 0;
    overflow-wrap: anywhere;
}
.dark .block.sources-box {
    background-color: #0f172a;
    border-color: #334155;
    color: #e2e8f0;
}
.trace-box {
    background-color: #ffffff;
    border: 1px solid #dbe7e1;
    border-radius: 8px;
    padding: 0.85rem 1rem;
}
.dark .trace-box {
    background-color: #0f172a;
    border-color: #334155;
    color: #e2e8f0;
}
.dark .result-box *,
.dark .sources-box *,
.dark .trace-box * {
    color: #e2e8f0 !important;
}
.dark .result-box a,
.dark .sources-box a {
    color: #5eead4 !important;
}
.dark .result-box code,
.dark .sources-box code,
.dark .trace-box code,
.dark .trace-box pre {
    background: #020617 !important;
    color: #f8fafc !important;
}
.dark .block.sources-box .md > p:first-child {
    color: #34d399;
}
.dark .block.sources-box .md li {
    border-top-color: #1e293b;
}
.dark .source-snippet {
    border-top-color: #1e293b;
}
.dark .source-snippets-title {
    color: #34d399 !important;
}
.dark .source-snippet-kicker {
    color: #34d399 !important;
}
.dark .source-snippet-title {
    color: #f8fafc !important;
}
.dark .source-snippet-url {
    color: #94a3b8 !important;
}
.dark .source-snippet-text {
    color: #cbd5e1 !important;
}
button {
    border-radius: 8px !important;
    min-height: 2.75rem;
    min-width: 2.75rem;
    touch-action: manipulation;
}
.gr-button-primary {
    font-weight: 720 !important;
}
textarea,
input,
select {
    font-size: 16px !important;
}
button:focus-visible,
textarea:focus-visible,
input:focus-visible,
select:focus-visible,
a:focus-visible {
    outline: 3px solid #34d399 !important;
    outline-offset: 2px !important;
}
@media (min-width: 980px) {
    .workspace-row {
        align-items: flex-start !important;
    }
    .answer-column {
        position: sticky;
        top: max(1rem, env(safe-area-inset-top));
        align-self: flex-start;
    }
    .evidence-row {
        align-items: flex-start !important;
    }
    .progress-list {
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.55rem;
    }
    .progress-step {
        align-items: flex-start;
        grid-template-columns: 1.6rem minmax(0, 1fr);
        min-height: 4.35rem;
        padding: 0.55rem 0.55rem 0.55rem 0.45rem;
    }
}
@media (max-width: 760px) {
    .hero h1 {
        font-size: 2rem;
    }
    .bench-strip {
        grid-template-columns: 1fr;
    }
    .chip {
        white-space: normal;
    }
}
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
    }
    .skip-link {
        transition: none !important;
    }
}
footer {
    visibility: hidden;
}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="XIAO Buddy") as demo:
        gr.HTML(
            """
<a class="skip-link" href="#support-bench">Skip to support bench</a>
<section class="hero">
  <h1>XIAO Buddy</h1>
  <p>Photo-aware support for Seeed XIAO boards plus connected sensors, robotics kits, LoRa modules, SenseCraft workflows, and field recovery.</p>
  <div class="chips">
    <span class="chip chip-board">ESP32S3 Sense</span>
    <span class="chip chip-board">ESP32C6</span>
    <span class="chip chip-board">ESP32C5</span>
    <span class="chip chip-board">nRF54L15</span>
    <span class="chip chip-radio">Wio-SX1262</span>
    <span class="chip chip-vision">Grove Vision AI V2</span>
    <span class="chip chip-platform">Raspberry Pi HATs</span>
    <span class="chip chip-robotics">Jetson robotics</span>
    <span class="chip chip-platform">RS485 expansion</span>
    <span class="chip chip-vision">SenseCraft AI</span>
  </div>
</section>
""",
        )
        gr.HTML(
            """
<section class="bench-strip">
  <div class="bench-card"><strong>Corpus</strong>Curated XIAO facts plus full Seeed wiki chunks for sensors, robotics, LoRa, and AI workflows.</div>
  <div class="bench-card"><strong>Routes</strong>Identify, troubleshoot, compare, or answer wiring, sensor, and robotics questions with cited source chunks.</div>
  <div class="bench-card"><strong>Runtime</strong>Hosted embedding, reranking, and OpenAI-compatible agent endpoints with live progress feedback.</div>
</section>
""",
        )

        with gr.Row(equal_height=False, elem_classes=["workspace-row"]):
            with gr.Column(scale=5, min_width=330, elem_classes=["input-column"]):
                gr.HTML('<p class="bench-label" id="support-bench">Support bench</p>')
                image = gr.Image(
                    label="Board or wiring photo",
                    type="pil",
                    sources=["upload", "clipboard", "webcam"],
                    height=315,
                )
                question = gr.Textbox(
                    label="Hardware question",
                    placeholder="Example: Which pins does the XIAO RS485 Expansion Board use for UART and enable? …",
                    lines=4,
                    max_lines=8,
                )
                with gr.Row():
                    submit = gr.Button("Ask Buddy", variant="primary")
                    clear = gr.ClearButton([image, question], value="Reset")

            with gr.Column(scale=8, min_width=360, elem_classes=["answer-column"]):
                gr.HTML('<p class="bench-label">Field answer</p>')
                progress = gr.HTML(
                    value=format_progress(),
                    elem_classes=["progress-box"],
                )
                with gr.Row(equal_height=False, elem_classes=["evidence-row"]):
                    with gr.Column(scale=7, min_width=340):
                        answer = gr.Markdown(
                            value=INITIAL_ANSWER,
                            label="Answer",
                            elem_classes=["result-box"],
                        )
                    with gr.Column(scale=4, min_width=285, elem_classes=["source-snippets-column"]):
                        citations = gr.Markdown(
                            label="Source snippets",
                            elem_classes=["sources-box"],
                            sanitize_html=False,
                        )
                with gr.Accordion("Retrieval trace", open=False):
                    diagnostics = gr.JSON(label="Run details", elem_classes=["trace-box"], open=False)

        gr.Examples(
            examples=EXAMPLES,
            inputs=[image, question],
            label="Bench scenarios",
        )

        submit.click(
            fn=answer_question_stream_ui,
            inputs=[image, question],
            outputs=[answer, citations, diagnostics, progress],
            api_name="ask",
        )
        question.submit(
            fn=answer_question_stream_ui,
            inputs=[image, question],
            outputs=[answer, citations, diagnostics, progress],
            api_name=None,
            api_visibility="private",
        )
        clear.click(
            fn=_reset_outputs,
            outputs=[answer, citations, diagnostics, progress],
            api_name=None,
            api_visibility="private",
            queue=False,
            show_progress="hidden",
        )

    return demo.queue()


def build_server_app(demo: gr.Blocks | None = None) -> FastAPI:
    settings = load_settings()
    server = FastAPI(title="XIAO Buddy")
    server.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|100\.103\.106\.102)(:\d+)?",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @server.post("/api/ask")
    async def ask_api(request: Request) -> StreamingResponse:
        payload = await request.json()
        question = str(payload.get("question") or "")
        history = _api_history_from_payload(payload.get("history"))
        contextual_question = _question_with_history(question, history)
        image = _api_image_from_payload(payload.get("image"))
        return StreamingResponse(
            _answer_question_sse(image, contextual_question),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return gr.mount_gradio_app(
        server,
        demo or build_demo(),
        path="/",
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        theme=THEME,
        css=CSS,
    )


def _answer_question_sse(image: Image.Image | None, question: str) -> Iterator[str]:
    try:
        for event in answer_question_stream_ui(image, question):
            yield _sse("message", event)
        yield _sse("complete", None)
    except Exception as exc:  # noqa: BLE001 - stream API should return a structured SSE error.
        yield _sse("error", {"error": str(exc) or exc.__class__.__name__})


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _api_image_from_payload(value: object) -> Image.Image | None:
    if not value:
        return None
    if not isinstance(value, str) or not value.startswith("data:image/"):
        return None
    try:
        _header, encoded = value.split(",", 1)
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            return image.copy()
    except Exception:  # noqa: BLE001 - invalid optional images are ignored.
        return None


def _api_history_from_payload(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    history: list[dict[str, str]] = []
    for item in value[-API_HISTORY_MAX_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _compact_history_text(str(item.get("content") or ""))
        if not content:
            continue
        history.append(
            {
                "role": role,
                "content": content[:API_HISTORY_ITEM_MAX_CHARS],
            }
        )
    return history


def _question_with_history(question: str, history: list[dict[str, str]]) -> str:
    question = question.strip()
    if not question or not history:
        return question

    context_lines: list[str] = []
    context_chars = 0
    for item in reversed(history):
        label = "User" if item["role"] == "user" else "Assistant"
        line = f"{label}: {item['content']}"
        if context_chars + len(line) > API_HISTORY_MAX_CHARS:
            break
        context_lines.append(line)
        context_chars += len(line)

    if not context_lines:
        return question

    context_lines.reverse()
    return (
        "Current follow-up question:\n"
        f"{question}\n\n"
        "Use this recent conversation only to resolve pronouns, omitted subjects, "
        "and follow-up intent:\n"
        + "\n".join(context_lines)
    )


def _compact_history_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def answer_question_stream_ui(
    image: object | None,
    question: str,
) -> Iterator[tuple[str, str, dict[str, object], str]]:
    for answer, _citations, diagnostics, progress in answer_question_stream_raw(image, question):
        yield (
            _render_user_answer(answer, diagnostics),
            _render_source_snippets(diagnostics),
            _display_diagnostics(diagnostics),
            progress,
        )


def _reset_outputs() -> tuple[str, str, dict[str, object], str]:
    return INITIAL_ANSWER, "", {}, format_progress()


def _render_user_answer(answer: str, diagnostics: dict[str, object]) -> str:
    refs = _source_refs(diagnostics)
    if not refs:
        return answer

    rendered = _replace_source_reference_groups(answer, refs)
    rendered = _hide_incomplete_source_id(rendered, refs)
    return _SOURCE_ID_RE.sub("source", rendered)


def _replace_source_reference_groups(answer: str, refs: list[dict[str, str]]) -> str:
    ref_links = {
        ref["id"]: f"[{ref['label']}](#{ref['anchor']})"
        for ref in refs
    }

    def replace_group(match: re.Match[str]) -> str:
        links: list[str] = []
        for source_id in _SOURCE_ID_RE.findall(match.group(1)):
            link = ref_links.get(source_id, "[source](#source-snippets)")
            if link not in links:
                links.append(link)
        return ", ".join(links) if links else "[source](#source-snippets)"

    return _SOURCE_ID_GROUP_RE.sub(replace_group, answer)


def _render_source_snippets(diagnostics: dict[str, object]) -> str:
    refs = _source_refs(diagnostics)
    if not refs:
        return ""

    blocks = ['<p id="source-snippets" class="source-snippets-title">Source snippets</p>']
    for ref in refs:
        url_html = (
            f'<p class="source-snippet-url">{html.escape(ref["source"])}</p>'
            if ref["source"]
            else ""
        )
        if _is_http_url(ref["source"]):
            title_html = (
                f'<a href="{html.escape(ref["source"], quote=True)}" '
                f'target="_blank" rel="noopener noreferrer">{html.escape(ref["title"])}</a>'
            )
        else:
            title_html = html.escape(ref["title"])
        blocks.append(
            "\n".join(
                [
                    (
                        f'<section class="source-snippet" id="{ref["anchor"]}" '
                        f'tabindex="-1" aria-label="{html.escape(ref["label"].title(), quote=True)} snippet">'
                    ),
                    f'<p class="source-snippet-kicker">{html.escape(ref["label"].title())}</p>',
                    f'<p class="source-snippet-title">{title_html}</p>',
                    url_html,
                    f'<p class="source-snippet-text">{html.escape(ref["snippet"])}</p>',
                    "</section>",
                ]
            )
        )
    return "\n\n".join(blocks)


def _display_diagnostics(diagnostics: dict[str, object]) -> dict[str, object]:
    refs = _source_refs(diagnostics)
    if not refs:
        return diagnostics
    id_map = {ref["id"]: ref["label"] for ref in refs}
    return _replace_source_ids(diagnostics, id_map)


def _replace_source_ids(value: object, id_map: dict[str, str]) -> object:
    if isinstance(value, str):
        return id_map.get(value, value)
    if isinstance(value, list):
        return [_replace_source_ids(item, id_map) for item in value]
    if isinstance(value, dict):
        return {key: _replace_source_ids(item, id_map) for key, item in value.items()}
    return value


def _source_refs(diagnostics: dict[str, object]) -> list[dict[str, str]]:
    raw_sources = diagnostics.get("top_sources")
    if not isinstance(raw_sources, list):
        return []

    refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in raw_sources:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("id") or "").strip()
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        index = len(refs) + 1
        title = str(source.get("title") or "Retrieved source").strip()
        refs.append(
            {
                "id": source_id,
                "label": f"source {index}",
                "anchor": f"source-snippet-{index}",
                "title": title or f"Source {index}",
                "source": str(source.get("source") or "").strip(),
                "snippet": str(source.get("snippet") or "").strip() or "No snippet available.",
            }
        )
    return refs


def _hide_incomplete_source_id(answer: str, refs: list[dict[str, str]]) -> str:
    match = re.search(r"\[([A-Za-z0-9_.:-]{1,96})$", answer)
    if not match:
        return answer
    partial = match.group(1)
    if any(ref["id"].startswith(partial) for ref in refs):
        return answer[: match.start()]
    return answer


def _is_http_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


_SOURCE_ID_PATTERN = r"(?:wiki-[A-Za-z0-9]+|field-[A-Za-z0-9_.:-]+|xiao-[A-Za-z0-9_.:-]+)"
_SOURCE_ID_RE = re.compile(rf"\b{_SOURCE_ID_PATTERN}\b")
_SOURCE_ID_GROUP_RE = re.compile(rf"\[([^\]\n]*(?:{_SOURCE_ID_PATTERN})[^\]\n]*)\]")


def warm_agent_endpoint(settings) -> dict[str, object]:
    if not settings.agent_base_url:
        return {"enabled": False, "reason": "agent endpoint is not configured"}
    if settings.agent_startup_warmup_seconds <= 0:
        return {"enabled": False, "reason": "AGENT_STARTUP_WARMUP_SECONDS <= 0"}

    started_at = perf_counter()
    chunks = 0
    chars = 0
    first_token_ms: float | None = None
    messages = [
        {"role": "system", "content": "Reply with exactly one word: ok."},
        {"role": "user", "content": "Warm the endpoint."},
    ]
    for result in chat_completion_stream(
        base_url=settings.agent_base_url,
        model=settings.agent_model,
        messages=messages,
        api_key=settings.agent_api_key,
        timeout=settings.agent_startup_warmup_seconds,
        max_tokens=2,
    ):
        if not result.ok:
            return {
                "enabled": True,
                "ok": False,
                "error": result.error,
                "total_ms": round((perf_counter() - started_at) * 1000, 1),
            }
        if first_token_ms is None:
            first_token_ms = round((perf_counter() - started_at) * 1000, 1)
        text = str(result.data or "")
        chunks += 1
        chars += len(text)

    return {
        "enabled": True,
        "ok": chunks > 0,
        "chunks": chunks,
        "chars": chars,
        "first_token_ms": first_token_ms,
        "total_ms": round((perf_counter() - started_at) * 1000, 1),
    }


demo = build_demo()
server_app = build_server_app(demo)


if __name__ == "__main__":
    settings = load_settings()
    warm_diagnostics = warm_retrieval_caches(settings)
    print(f"warmed retrieval caches: {warm_diagnostics}", flush=True)
    agent_warm_diagnostics = warm_agent_endpoint(settings)
    print(f"warmed agent endpoint: {agent_warm_diagnostics}", flush=True)
    uvicorn.run(
        server_app,
        host=settings.gradio_server_name,
        port=settings.gradio_server_port,
    )
