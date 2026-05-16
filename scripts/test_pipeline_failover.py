from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xiao_copilot.pipeline as pipeline
from xiao_copilot.clients import EndpointResult
from xiao_copilot.config import Settings
from xiao_copilot.knowledge_base import KnowledgeChunk


def main() -> None:
    _assert_text_repair()
    _assert_system_prompt_preserves_acronyms()
    _assert_exact_term_hint_preserves_hardware_terms()
    _assert_exact_term_append_cites_missing_terms()
    _assert_success_stream_reports_first_token_latency()

    original_load_settings = pipeline.load_settings
    original_retrieve = pipeline.retrieve
    original_generate_stream = pipeline._generate_with_agent_stream
    try:
        pipeline.load_settings = _fake_settings  # type: ignore[assignment]
        pipeline.retrieve = _fake_retrieve  # type: ignore[assignment]
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
        pipeline.retrieve = original_retrieve  # type: ignore[assignment]
        pipeline._generate_with_agent_stream = original_generate_stream  # type: ignore[assignment]

    print("PASS pipeline failover regression")


def _assert_success_stream_reports_first_token_latency() -> None:
    original_load_settings = pipeline.load_settings
    original_retrieve = pipeline.retrieve
    original_generate_stream = pipeline._generate_with_agent_stream
    original_perf_counter = pipeline.perf_counter
    clock = {"now": 0.0}

    def fake_perf_counter() -> float:
        clock["now"] += 0.05
        return clock["now"]

    try:
        pipeline.load_settings = _fake_settings  # type: ignore[assignment]
        pipeline.retrieve = _fake_retrieve  # type: ignore[assignment]
        pipeline._generate_with_agent_stream = _good_agent_stream  # type: ignore[assignment]
        pipeline.perf_counter = fake_perf_counter  # type: ignore[assignment]

        events = list(pipeline.answer_question_stream(None, "Which pin should I check?"))
        stream_events = [
            event
            for event in events
            if event[2].get("agent", {}).get("first_token_ms") is not None
        ]
        _assert(stream_events, "streaming diagnostics should include first-token latency")
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
    finally:
        pipeline.load_settings = original_load_settings  # type: ignore[assignment]
        pipeline.retrieve = original_retrieve  # type: ignore[assignment]
        pipeline._generate_with_agent_stream = original_generate_stream  # type: ignore[assignment]
        pipeline.perf_counter = original_perf_counter  # type: ignore[assignment]


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


def _fake_settings() -> Settings:
    return Settings(
        embedding_base_url="",
        rerank_base_url="",
        agent_base_url="https://example.test/v1",
    )


def _fake_retrieve(_question: str, _settings: Settings, image_data_url: str | None = None):
    chunk = KnowledgeChunk(
        id="test-source",
        title="Test source",
        source="https://wiki.seeedstudio.com/test/",
        text="Use the cited source instead of incomplete streamed agent text.",
        kind="wiki",
    )
    return [chunk], {"vector_index_backend": "lexical", "image_used": bool(image_data_url)}


def _broken_agent_stream(*_args, **_kwargs) -> Iterator[EndpointResult]:
    yield EndpointResult(ok=True, data="Partial agent text that should be discarded.")
    yield EndpointResult(ok=False, error="simulated stream failure")


def _good_agent_stream(*_args, **_kwargs) -> Iterator[EndpointResult]:
    yield EndpointResult(ok=True, data="Use the cited source ")
    yield EndpointResult(ok=True, data="for the pin check [test-source].")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
