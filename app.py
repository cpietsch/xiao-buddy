from __future__ import annotations

import gradio as gr

from xiao_copilot.config import load_settings
from xiao_copilot.pipeline import answer_question


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
]


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
.gradio-container {
    max-width: 1220px !important;
    background: linear-gradient(180deg, #f8fafc 0%, #eefdf6 42%, #f8fafc 100%);
}
.dark .gradio-container {
    background: linear-gradient(180deg, #020617 0%, #052e16 42%, #020617 100%);
}
.hero {
    padding: 1.5rem 0 0.65rem;
    border-bottom: 1px solid #dbe7e1;
    margin-bottom: 1rem;
}
.dark .hero {
    border-bottom: 1px solid #1e293b;
}
.hero h1 {
    color: #0f172a;
    font-size: 2.55rem;
    line-height: 1.1;
    margin: 0 0 0.35rem;
    font-weight: 760;
    letter-spacing: 0;
}
.dark .hero h1 {
    color: #f8fafc;
}
.hero p {
    color: #334155;
    font-size: 1.02rem;
    margin: 0;
    max-width: 780px;
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
}
.dark .chip {
    background: #0f172a;
    border: 1px solid #334155;
    color: #34d399;
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
.result-box {
    background-color: #ffffff;
    border: 1px solid #dbe7e1;
    border-radius: 8px;
    min-height: 360px;
    padding: 1rem 1.1rem;
}
.dark .result-box {
    background-color: #0f172a;
    border-color: #334155;
    color: #e2e8f0;
}
.sources-box {
    background-color: #ffffff;
    border: 1px solid #dbe7e1;
    border-radius: 8px;
    padding: 0.85rem 1rem;
}
.dark .sources-box {
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
button {
    border-radius: 8px !important;
}
.gr-button-primary {
    font-weight: 720 !important;
}
@media (max-width: 760px) {
    .hero h1 {
        font-size: 2rem;
    }
    .bench-strip {
        grid-template-columns: 1fr;
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
<section class="hero">
  <h1>XIAO Buddy</h1>
  <p>Photo-aware support for Seeed XIAO boards, pinouts, wireless bring-up, power checks, and field recovery.</p>
  <div class="chips">
    <span class="chip">ESP32S3 Sense</span>
    <span class="chip">ESP32C6</span>
    <span class="chip">ESP32C5</span>
    <span class="chip">nRF54L15</span>
    <span class="chip">MG24</span>
    <span class="chip">RA4M1</span>
    <span class="chip">RP2350</span>
    <span class="chip">W5500 adapter</span>
  </div>
</section>
""",
        )
        gr.HTML(
            """
<section class="bench-strip">
  <div class="bench-card"><strong>Corpus</strong>Curated Seeed XIAO board notes, pinouts, boot modes, power limits, and accessories.</div>
  <div class="bench-card"><strong>Routes</strong>Identify, troubleshoot, compare, or answer wiring questions with cited source chunks.</div>
  <div class="bench-card"><strong>Runtime</strong>Hosted Qwen embedding, reranking, and agent calls served from an AMD MI300X stack.</div>
</section>
""",
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=330):
                gr.HTML('<p class="bench-label">Support bench</p>')
                image = gr.Image(
                    label="Board or wiring photo",
                    type="pil",
                    sources=["upload", "clipboard", "webcam"],
                    height=315,
                )
                question = gr.Textbox(
                    label="Hardware question",
                    placeholder="Example: My XIAO RP2350 will not enter BOOT mode. What should I check?",
                    lines=4,
                    max_lines=8,
                )
                with gr.Row():
                    submit = gr.Button("Diagnose XIAO", variant="primary")
                    clear = gr.ClearButton([image, question], value="Reset")

            with gr.Column(scale=7, min_width=360):
                gr.HTML('<p class="bench-label">Field answer</p>')
                answer = gr.Markdown(
                    value="Ask about a XIAO board, accessory, pinout, upload issue, power symptom, or wireless requirement.",
                    label="Answer",
                    elem_classes=["result-box"],
                )
                citations = gr.Markdown(label="Source trail", elem_classes=["sources-box"])
                with gr.Accordion("Retrieval trace", open=False):
                    diagnostics = gr.JSON(label="Run details", elem_classes=["trace-box"], open=False)

        gr.Examples(
            examples=EXAMPLES,
            inputs=[image, question],
            label="Bench scenarios",
        )

        submit.click(
            fn=answer_question,
            inputs=[image, question],
            outputs=[answer, citations, diagnostics],
            api_name="ask",
        )
        question.submit(
            fn=answer_question,
            inputs=[image, question],
            outputs=[answer, citations, diagnostics],
            api_name=False,
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    settings = load_settings()
    demo.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
        theme=THEME,
        css=CSS,
    )
