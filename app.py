from __future__ import annotations

import gradio as gr

from xiao_copilot.config import load_settings
from xiao_copilot.pipeline import answer_question


EXAMPLES = [
    [
        None,
        "My XIAO ESP32S3 Sense is not detected over USB. What should I try first?",
    ],
    [
        None,
        "Which pins should I avoid when adding an I2C sensor to a XIAO board?",
    ],
    [
        None,
        "I uploaded a board photo. Help me inspect it and suggest the safest next checks.",
    ],
]


THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="stone",
    neutral_hue="slate",
    radius_size="sm",
).set(
    button_primary_background_fill="#0f766e",
    button_primary_background_fill_hover="#115e59",
    block_border_width="1px",
)


CSS = """
.gradio-container {
    max-width: 1180px !important;
}
.app-title h1 {
    font-size: 2.2rem;
    line-height: 1.05;
    margin-bottom: 0.35rem;
}
.app-title p {
    color: #475569;
    font-size: 1rem;
    margin: 0;
}
.result-box textarea {
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
footer {
    visibility: hidden;
}
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="XIAO Field Copilot") as demo:
        gr.Markdown(
            """
# XIAO Field Copilot
Upload a board photo, ask a hardware-support question, and get a cited field answer.
""",
            elem_classes=["app-title"],
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=330):
                image = gr.Image(
                    label="Board photo",
                    type="pil",
                    sources=["upload", "clipboard", "webcam"],
                    height=330,
                )
                question = gr.Textbox(
                    label="Question",
                    placeholder="Example: My XIAO RP2040 resets when a sensor starts. What should I check?",
                    lines=4,
                    max_lines=8,
                )
                with gr.Row():
                    submit = gr.Button("Ask copilot", variant="primary")
                    clear = gr.ClearButton([image, question], value="Clear")

            with gr.Column(scale=7, min_width=360):
                answer = gr.Markdown(label="Answer", elem_classes=["result-box"])
                citations = gr.Markdown(label="Citations")
                diagnostics = gr.JSON(label="Run details", open=False)

        gr.Examples(
            examples=EXAMPLES,
            inputs=[image, question],
            label="Try these",
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
