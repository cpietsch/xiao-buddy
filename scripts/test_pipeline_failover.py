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


def _assert_system_prompt_preserves_acronyms() -> None:
    _assert(
        "full service name and acronym" in pipeline.SYSTEM_PROMPT,
        "agent prompt should expand service acronyms for clear cited answers",
    )


def _assert_text_repair() -> None:
    repaired = pipeline._repair_generated_text("Basics\u00e2\u0084\u00a2 Station and LoRaWAN\u00c2\u00ae coverage")
    _assert("Basics™ Station" in repaired, "agent text repair should handle trademark mojibake")
    _assert("LoRaWAN® coverage" in repaired, "agent text repair should handle registered-symbol mojibake")


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


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
