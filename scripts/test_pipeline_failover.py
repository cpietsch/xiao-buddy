from __future__ import annotations

import sys
from pathlib import Path
from time import sleep
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xiao_copilot.pipeline as pipeline
from xiao_copilot.clients import EndpointResult
from xiao_copilot.config import Settings
from xiao_copilot.knowledge_base import KnowledgeChunk
from xiao_copilot.retrieval import RetrievalStage


def main() -> None:
    _assert_text_repair()
    _assert_system_prompt_preserves_acronyms()
    _assert_exact_term_hint_preserves_hardware_terms()
    _assert_exact_term_append_cites_missing_terms()
    _assert_contextual_exact_term_repair_avoids_source_detail_noise()
    _assert_inline_citation_repair_reuses_sources_line()
    _assert_agent_context_budget_keeps_relevant_excerpt()
    _assert_agent_wait_heartbeat_keeps_draft_visible()
    _assert_success_stream_reports_first_token_latency()

    original_load_settings = pipeline.load_settings
    original_retrieve_progressive = pipeline.retrieve_progressive
    original_generate_stream = pipeline._generate_with_agent_stream
    try:
        pipeline.load_settings = _fake_settings  # type: ignore[assignment]
        pipeline.retrieve_progressive = _fake_retrieve_progressive  # type: ignore[assignment]
        pipeline._generate_with_agent_stream = _broken_agent_stream  # type: ignore[assignment]

        events = list(pipeline.answer_question_stream(None, "Which pin should I check?"))
        _assert(events, "pipeline should yield events")
        answer, citations, diagnostics, progress_html = events[-1]

        _assert("Partial agent text" not in answer, "partial streamed text must not be trusted after stream error")
        _assert("deterministic fallback" in answer, "stream errors should use deterministic fallback answer")
        _assert("[test-source]" in answer, "fallback answer should preserve source citations")
        _assert("- [test-source]" in citations, "citation trail should still be returned")
        _assert(diagnostics.get("agent_used") is False, "agent should be marked unused after stream error")
        _assert(diagnostics.get("agent_error") == "simulated stream failure", "agent error should be diagnostic")
        _assert(diagnostics.get("agent", {}).get("status") == "fallback", "agent status should be fallback")
        _assert("fallback answer" in progress_html, "progress should report fallback final mode")
    finally:
        pipeline.load_settings = original_load_settings  # type: ignore[assignment]
        pipeline.retrieve_progressive = original_retrieve_progressive  # type: ignore[assignment]
        pipeline._generate_with_agent_stream = original_generate_stream  # type: ignore[assignment]

    print("PASS pipeline failover regression")


def _assert_success_stream_reports_first_token_latency() -> None:
    original_load_settings = pipeline.load_settings
    original_retrieve_progressive = pipeline.retrieve_progressive
    original_generate_stream = pipeline._generate_with_agent_stream
    original_perf_counter = pipeline.perf_counter
    clock = {"now": 0.0}

    def fake_perf_counter() -> float:
        clock["now"] += 0.05
        return clock["now"]

    try:
        pipeline.load_settings = _fake_settings  # type: ignore[assignment]
        pipeline.retrieve_progressive = _fake_retrieve_progressive  # type: ignore[assignment]
        pipeline._generate_with_agent_stream = _good_agent_stream  # type: ignore[assignment]
        pipeline.perf_counter = fake_perf_counter  # type: ignore[assignment]

        events = list(pipeline.answer_question_stream(None, "Which pin should I check?"))
        draft_events = [
            event
            for event in events
            if event[2].get("draft", {}).get("visible")
        ]
        _assert(draft_events, "pipeline should yield a source-backed draft before the agent stream")
        _assert("Source-backed draft" in draft_events[0][0], "draft event should show source-backed answer text")
        _assert(
            draft_events[0][2].get("agent", {}).get("first_token_ms") is None,
            "draft event must not masquerade as the hosted agent first token",
        )

        stream_events = [
            event
            for event in events
            if event[2].get("agent", {}).get("first_token_ms") is not None
        ]
        _assert(stream_events, "streaming diagnostics should include first-token latency")
        _assert(
            events.index(draft_events[0]) < events.index(stream_events[0]),
            "source-backed draft should appear before the first hosted agent token",
        )
        _assert(
            draft_events[0][2].get("retrieval", {}).get("preliminary") is True,
            "first source-backed draft should use preliminary pre-rerank sources",
        )
        _assert(
            any(not event[2].get("retrieval", {}).get("preliminary") for event in draft_events[1:]),
            "pipeline should refresh the source-backed draft after rerank finishes",
        )
        first_token_ms = stream_events[0][2]["agent"]["first_token_ms"]
        _assert(isinstance(first_token_ms, float), "first-token latency should be numeric")
        _assert(first_token_ms > 0, "first-token latency should be positive once streaming starts")

        _answer, _citations, diagnostics, progress_html = events[-1]
        agent = diagnostics.get("agent", {})
        _assert(agent.get("first_token_ms") == first_token_ms, "final diagnostics should keep first-token latency")
        _assert(
            diagnostics.get("agent_first_token_ms") == first_token_ms,
            "final diagnostics should expose top-level first-token latency",
        )
        _assert("first token" in progress_html, "final progress should show first-token latency")
        _assert("embed 12 ms" in progress_html, "final progress should expose embedding latency")
        _assert("rerank 34 ms" in progress_html, "final progress should expose reranker latency")
        _assert(
            diagnostics.get("agent_prompt", {}).get("chars"),
            "final diagnostics should expose agent prompt size",
        )
    finally:
        pipeline.load_settings = original_load_settings  # type: ignore[assignment]
        pipeline.retrieve_progressive = original_retrieve_progressive  # type: ignore[assignment]
        pipeline._generate_with_agent_stream = original_generate_stream  # type: ignore[assignment]
        pipeline.perf_counter = original_perf_counter  # type: ignore[assignment]


def _assert_agent_wait_heartbeat_keeps_draft_visible() -> None:
    original_load_settings = pipeline.load_settings
    original_retrieve_progressive = pipeline.retrieve_progressive
    original_generate_stream = pipeline._generate_with_agent_stream
    original_heartbeat = pipeline.AGENT_PROGRESS_HEARTBEAT_SECONDS
    try:
        pipeline.load_settings = _fake_settings  # type: ignore[assignment]
        pipeline.retrieve_progressive = _fake_retrieve_progressive  # type: ignore[assignment]
        pipeline._generate_with_agent_stream = _slow_agent_stream  # type: ignore[assignment]
        pipeline.AGENT_PROGRESS_HEARTBEAT_SECONDS = 0.001

        events = list(pipeline.answer_question_stream(None, "Which pin should I check?"))
        waiting_events = [
            event
            for event in events
            if event[2].get("agent", {}).get("status") == "waiting_first_token"
        ]
        _assert(waiting_events, "pipeline should yield progress heartbeats while waiting for the first token")
        answer, _citations, diagnostics, progress_html = waiting_events[0]
        _assert("Source-backed draft" in answer, "heartbeat should keep the source-backed draft visible")
        _assert("waiting" in progress_html, "heartbeat progress should expose active waiting")
        _assert("first token" in progress_html, "heartbeat progress should name the first-token wait")
        _assert(
            isinstance(diagnostics.get("agent_wait_ms"), float),
            "heartbeat diagnostics should expose agent wait latency",
        )
    finally:
        pipeline.load_settings = original_load_settings  # type: ignore[assignment]
        pipeline.retrieve_progressive = original_retrieve_progressive  # type: ignore[assignment]
        pipeline._generate_with_agent_stream = original_generate_stream  # type: ignore[assignment]
        pipeline.AGENT_PROGRESS_HEARTBEAT_SECONDS = original_heartbeat


def _assert_system_prompt_preserves_acronyms() -> None:
    _assert(
        "full service name and acronym" in pipeline.SYSTEM_PROMPT,
        "agent prompt should expand service acronyms for clear cited answers",
    )


def _assert_text_repair() -> None:
    repaired = pipeline._repair_generated_text("Basics\u00e2\u0084\u00a2 Station and LoRaWAN\u00c2\u00ae coverage")
    _assert("Basics™ Station" in repaired, "agent text repair should handle trademark mojibake")
    _assert("LoRaWAN® coverage" in repaired, "agent text repair should handle registered-symbol mojibake")


def _assert_exact_term_hint_preserves_hardware_terms() -> None:
    chunk = KnowledgeChunk(
        id="term-source",
        title="Firmware setup",
        source="https://wiki.seeedstudio.com/test/",
        text=(
            "BL702 is the USB-UART chip. Select the Frequenct Plan, then copy "
            "**firmware.uf2** to **GROVEAI**."
        ),
        kind="wiki",
    )
    terms = pipeline._exact_terms_hint("What values do I need from the app?", [chunk])
    for expected in ("BL702", "USB-UART", "frequency plan", "firmware.uf2", "GROVEAI"):
        _assert(expected in terms, f"exact term hint should include {expected}")


def _assert_exact_term_append_cites_missing_terms() -> None:
    chunk = KnowledgeChunk(
        id="wifi-source",
        title="Gateway network",
        source="https://wiki.seeedstudio.com/test/",
        text="The wifi you expect to use should be 2.4G.",
        kind="wiki",
    )
    answer = pipeline._ensure_answer_exact_terms(
        "Use the XIAO ESP32S3 & Wio-SX1262 Kit [wifi-source].",
        "What hardware needs LoRa and WiFi?",
        [chunk],
    )
    _assert("`2.4G` [wifi-source]" in answer, "missing exact terms should be appended with citations")
    _assert("Source detail:" not in answer, "generic exact-term repair should avoid mechanical source-detail label")
    _assert(
        "Relevant exact source terms:" not in answer,
        "generic exact-term repair should avoid a catch-all exact-term appendix",
    )

    unrelated = pipeline._ensure_answer_exact_terms(
        "The OLED address is `0x3C` [oled-source].",
        "At what I2C address is the SSD1306 OLED configured?",
        [
            KnowledgeChunk(
                id="oled-source",
                title="OLED sample",
                source="https://wiki.seeedstudio.com/test/",
                text="The same page also reads temperature and humidity in another sample.",
                kind="wiki",
            )
        ],
    )
    _assert(
        "temperature" not in unrelated and "humidity" not in unrelated,
        "irrelevant forced exact terms should not be appended to a complete answer",
    )


def _assert_contextual_exact_term_repair_avoids_source_detail_noise() -> None:
    uart = pipeline._ensure_answer_exact_terms(
        "Use 921600 baud, 8 data bits, no parity, and 1 stop bit [uart-source].",
        "Which UART serial settings are used?",
        [
            KnowledgeChunk(
                id="uart-source",
                title="UART settings",
                source="https://wiki.seeedstudio.com/test/",
                text="Use 921600 baud, SERIAL_8N1, None parity, and 1 stop bit.",
                kind="wiki",
            )
        ],
    )
    _assert("parity `None`" in uart, "UART repair should express no parity as the exact None value")
    _assert("Source detail:" not in uart, "UART repair should not append a mechanical source-detail line")

    esphome = pipeline._ensure_answer_exact_terms(
        "Use an ESP-IDF settings block [yaml-source].",
        "What ESPHome YAML board settings should I use for Seeed Studio XIAO ESP32C3?",
        [
            KnowledgeChunk(
                id="yaml-source",
                title="ESPHome YAML",
                source="https://wiki.seeedstudio.com/test/",
                text=(
                    "esp32:\n"
                    "  board: seeed_xiao_esp32c3\n"
                    "  variant: esp32c3\n"
                    "  framework:\n"
                    "    type: arduino\n"
                    "    version: 2.0.5\n"
                    "    platform_version: 5.2.0\n"
                ),
                kind="wiki",
            )
        ],
    )
    for expected in ("seeed_xiao_esp32c3", "arduino", "2.0.5", "5.2.0", "platform_version"):
        _assert(expected in esphome, f"ESPHome contextual repair should include {expected}")
    _assert("ESP-IDF" not in esphome, "ESPHome repair should replace a conflicting generated block")

    sht40 = pipeline._ensure_answer_exact_terms(
        "Use `SensirionI2CSht4x` and `measureHighPrecision()` [sht-source].",
        "Which Arduino libraries and function read Grove Temp&Humi Sensor SHT40 data?",
        [
            KnowledgeChunk(
                id="sht-source",
                title="SHT40 libraries",
                source="https://wiki.seeedstudio.com/test/",
                text="Install arduino-i2c-sht4x and Sensirion Arduino Core to read temperature and humidity.",
                kind="wiki",
            )
        ],
    )
    for expected in ("arduino-i2c-sht4x", "Sensirion Arduino Core", "temperature", "humidity"):
        _assert(expected in sht40, f"SHT40 contextual repair should include {expected}")
    _assert("Source detail:" not in sht40, "SHT40 repair should use a natural sentence")

    sensecap = pipeline._ensure_answer_exact_terms(
        "The gateway supports AWS, TTN, ChirpStack, Packet Forwarder, and Basics Station [lns-source].",
        "What network-server options and range does SenseCAP M2 Multi-Platform LoRaWAN Gateway support?",
        [
            KnowledgeChunk(
                id="lns-source",
                title="SenseCAP network servers",
                source="https://wiki.seeedstudio.com/test/",
                text="It supports AWS, TTN, ChirpStack, Packet Forwarder, Basics Station, and Built-in LoRaWAN Network Server.",
                kind="wiki",
            )
        ],
    )
    _assert(
        "`Built-in LoRaWAN Network Server` [lns-source]" in sensecap,
        "SenseCAP network-server repair should preserve the built-in server option",
    )

    sx1262 = pipeline._ensure_answer_exact_terms(
        "The kit can be used as a single channel LoRaWAN gateway and for Meshtastic [kit-source].",
        "What can the XIAO ESP32S3 & Wio-SX1262 kit with 3D case be used for?",
        [
            KnowledgeChunk(
                id="kit-source",
                title="Kit applications",
                source="https://wiki.seeedstudio.com/test/",
                text="Applications include a 2.5km single channel LoRaWAN gateway, Meshtastic, and a LoRaWAN Node.",
                kind="wiki",
            )
        ],
    )
    for expected in ("`2.5km` [kit-source]", "`LoRaWAN Node` [kit-source]"):
        _assert(expected in sx1262, f"kit application repair should include {expected}")

    rs485 = pipeline._ensure_answer_exact_terms(
        "RS485 RX is D4, TX is D5, and the enable pin is GPIO4 [rs485-source].",
        "Which pins are used for RS485 UART RX/TX and which pin controls enable?",
        [
            KnowledgeChunk(
                id="rs485-source",
                title="RS485 pins",
                source="https://wiki.seeedstudio.com/test/",
                text="Receiver code uses D4 for RX, D5 for TX, and D2 as the RS485 enable pin.",
                kind="wiki",
            )
        ],
    )
    _assert("`D2` [rs485-source]" in rs485, "RS485 repair should preserve the D2 enable-pin label")


def _assert_inline_citation_repair_reuses_sources_line() -> None:
    chunks = [
        KnowledgeChunk(id="source-a", title="A", source="https://wiki.seeedstudio.com/a/", text="A", kind="wiki"),
        KnowledgeChunk(id="source-b", title="B", source="https://wiki.seeedstudio.com/b/", text="B", kind="wiki"),
    ]
    answer = pipeline._ensure_inline_citations("Use source A [source-a].\n\nSources: [source-a]", chunks)
    _assert(answer.count("Sources:") == 1, "citation repair should not create duplicate Sources lines")
    _assert("[source-b]" in answer, "citation repair should append missing citations to the existing Sources line")


def _assert_agent_context_budget_keeps_relevant_excerpt() -> None:
    chunk = KnowledgeChunk(
        id="yaml-source",
        title="ESPHome YAML",
        source="https://wiki.seeedstudio.com/test/",
        text=(
            "Introductory material that is not useful for the specific question. " * 20
            + "esp32:\n"
            + "  board: seeed_xiao_esp32c3\n"
            + "  variant: esp32c3\n"
            + "  framework:\n"
            + "    type: arduino\n"
            + "    version: 2.0.5\n"
            + "    platform_version: 5.2.0\n"
            + "Trailing material that can be trimmed. " * 20
        ),
        kind="wiki",
    )
    messages = pipeline._build_agent_messages(
        question="What ESPHome YAML board settings should I use for Seeed Studio XIAO ESP32C3?",
        image_summary={},
        image_data_url=None,
        intent="text",
        chunks=[chunk],
        context_chars=700,
    )
    prompt = str(messages[1]["content"])
    _assert(len(prompt) < len(chunk.text) + 900, "context budget should shorten long source text")
    for expected in ("seeed_xiao_esp32c3", "variant", "platform_version", "5.2.0"):
        _assert(expected in prompt, f"context excerpt should keep {expected}")


def _fake_settings() -> Settings:
    return Settings(
        embedding_base_url="",
        rerank_base_url="",
        agent_base_url="https://example.test/v1",
    )


def _fake_retrieve_progressive(
    _question: str,
    _settings: Settings,
    image_data_url: str | None = None,
) -> Iterator[RetrievalStage]:
    chunk = KnowledgeChunk(
        id="test-source",
        title="Test source",
        source="https://wiki.seeedstudio.com/test/",
        text="Use the cited source instead of incomplete streamed agent text.",
        kind="wiki",
    )
    preview_diagnostics = {
        "vector_index_backend": "lexical",
        "image_used": bool(image_data_url),
        "preliminary": True,
        "reranker_pending": True,
        "timings_ms": {
            "query_embedding": 12.0,
            "total": 16.0,
        },
    }
    yield RetrievalStage(stage="pre_rerank", chunks=[chunk], diagnostics=preview_diagnostics)
    final_diagnostics = {
        "vector_index_backend": "lexical",
        "image_used": bool(image_data_url),
        "reranker_used": True,
        "reranker_mode": "native",
        "reranker_ms": 34.0,
        "timings_ms": {
            "query_embedding": 12.0,
            "reranker": 34.0,
            "total": 50.0,
        },
    }
    yield RetrievalStage(stage="final", chunks=[chunk], diagnostics=final_diagnostics)


def _broken_agent_stream(*_args, **_kwargs) -> Iterator[EndpointResult]:
    yield EndpointResult(ok=True, data="Partial agent text that should be discarded.")
    yield EndpointResult(ok=False, error="simulated stream failure")


def _good_agent_stream(*_args, **_kwargs) -> Iterator[EndpointResult]:
    yield EndpointResult(ok=True, data="Use the cited source ")
    yield EndpointResult(ok=True, data="for the pin check [test-source].")


def _slow_agent_stream(*_args, **_kwargs) -> Iterator[EndpointResult]:
    sleep(0.02)
    yield from _good_agent_stream()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
