from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.app_smoke import (
    _agent_wait_heartbeat_events,
    _first_token_ms,
    _format_ms,
    _require_first_token_latency,
    _require_inline_citation,
    _require_reranker_cache_behavior,
    _reranker_cache_hit_events,
    _reranker_wait_events,
    _source_label_count,
    _source_draft_events,
    _stream_events_with_first_token,
)


def main() -> None:
    _assert_first_token_helpers()
    _assert_stream_first_token_events()
    _assert_source_draft_events()
    _assert_source_link_helpers()
    _assert_reranker_wait_events()
    _assert_reranker_cache_hit_events()
    _assert_reranker_cache_requirements()
    _assert_agent_wait_heartbeat_events()
    print("PASS app smoke parser regression")


def _assert_first_token_helpers() -> None:
    nested = {"agent": {"first_token_ms": 123.4}}
    top_level = {"agent_first_token_ms": 234.5}
    missing = {"agent": {"streamed": True}}

    _assert(_first_token_ms(nested) == 123.4, "nested first-token latency should be extracted")
    _assert(_first_token_ms(top_level) == 234.5, "top-level first-token latency should be extracted")
    _assert(_first_token_ms(missing) is None, "missing first-token latency should return None")
    _assert(_format_ms(12.34) == "12.3", "millisecond formatting should use one decimal")
    _assert(_format_ms(None) == "n/a", "missing millisecond formatting should be explicit")

    _require_first_token_latency(123.4, "<small>agent answer; first token 123 ms</small>")
    _assert_raises_system_exit(
        lambda: _require_first_token_latency(None, "<small>agent answer</small>"),
        "missing first-token latency should fail app smoke",
    )
    _assert_raises_system_exit(
        lambda: _require_first_token_latency(123.4, "<small>agent answer</small>"),
        "missing first-token progress text should fail app smoke",
    )


def _assert_stream_first_token_events() -> None:
    good_event = [
        "partial answer",
        "",
        {"agent": {"streaming": True, "streamed": True, "first_token_ms": 123.4}},
        "<small>streaming 20 chars; first token 123 ms</small>",
    ]
    final_only_event = [
        "final answer",
        "",
        {"agent": {"streaming": False, "streamed": True, "first_token_ms": 123.4}},
        "<small>agent answer; first token 123 ms</small>",
    ]
    missing_progress_event = [
        "partial answer",
        "",
        {"agent": {"streaming": True, "streamed": True, "first_token_ms": 123.4}},
        "<small>streaming 20 chars</small>",
    ]

    _assert(
        _stream_events_with_first_token([good_event, final_only_event]) == [good_event],
        "only running stream events with visible first-token progress should match",
    )
    _assert(
        _stream_events_with_first_token([final_only_event, missing_progress_event]) == [],
        "final-only first-token evidence should not satisfy streamed-progress evidence",
    )


def _assert_source_draft_events() -> None:
    draft_event = [
        "Source-backed draft\n\nUse the cited source.",
        "",
        {"draft": {"visible": True, "chars": 48}},
        "<small>source draft 48 chars in 500 ms; preparing agent</small>",
    ]
    missing_answer_event = [
        "Generating a cited answer with the agent...",
        "",
        {"draft": {"visible": True, "chars": 48}},
        "<small>source draft 48 chars in 500 ms; preparing agent</small>",
    ]
    missing_progress_event = [
        "Source-backed draft\n\nUse the cited source.",
        "",
        {"draft": {"visible": True, "chars": 48}},
        "<small>preparing agent</small>",
    ]

    _assert(
        _source_draft_events([draft_event, missing_answer_event, missing_progress_event]) == [draft_event],
        "only visible source-backed draft events with matching progress should count",
    )


def _assert_source_link_helpers() -> None:
    answer = "Choose ESP32-C5 [source 1](#source-snippet-1)."
    source_text = """
<span id="source-snippets"></span>
<section class="source-snippet" id="source-snippet-1">
<p class="source-snippet-kicker">Source 1</p>
<p class="source-snippet-title">XIAO ESP32-C5 WiFi Usage</p>
<p class="source-snippet-url">https://wiki.seeedstudio.com/xiao_esp32c5_wifi_usage/</p>
<p class="source-snippet-text">Use the XIAO ESP32-C5 when 5 GHz WiFi is required.</p>
</section>
"""
    _require_inline_citation(
        answer,
        source_text,
        ("https://wiki.seeedstudio.com/xiao_esp32c5_wifi_usage/",),
    )
    _assert(_source_label_count(source_text) == 1, "source labels should be counted from snippet panel HTML")
    _assert_raises_system_exit(
        lambda: _require_inline_citation(
            "Choose ESP32-C5 without a source link.",
            source_text,
            ("https://wiki.seeedstudio.com/xiao_esp32c5_wifi_usage/",),
        ),
        "missing source link should fail app smoke",
    )


def _assert_agent_wait_heartbeat_events() -> None:
    heartbeat_event = [
        "Source-backed draft\n\nUse the cited source.",
        "",
        {"agent": {"status": "waiting_first_token", "wait_ms": 1001.0}},
        "<small>source draft 48 chars in 500 ms; waiting 1001 ms for first token</small>",
    ]
    missing_status_event = [
        "Source-backed draft\n\nUse the cited source.",
        "",
        {"agent": {"status": "streaming", "wait_ms": 1001.0}},
        "<small>source draft 48 chars in 500 ms; waiting 1001 ms for first token</small>",
    ]
    missing_progress_event = [
        "Source-backed draft\n\nUse the cited source.",
        "",
        {"agent": {"status": "waiting_first_token", "wait_ms": 1001.0}},
        "<small>source draft 48 chars in 500 ms; preparing agent</small>",
    ]

    _assert(
        _agent_wait_heartbeat_events([heartbeat_event, missing_status_event, missing_progress_event])
        == [heartbeat_event],
        "only waiting-first-token events with visible heartbeat progress should count",
    )


def _assert_reranker_wait_events() -> None:
    wait_event = [
        "Source-backed draft\n\nUse the cited source.\n\n_Refining source order with hosted reranker: 1001 ms…_",
        "",
        {"retrieval": {"reranker_wait_ms": 1001.0}},
        "<small>source draft 48 chars in 500 ms; reranker running 1001 ms</small>",
    ]
    missing_answer_event = [
        "Source-backed draft\n\nUse the cited source.",
        "",
        {"retrieval": {"reranker_wait_ms": 1001.0}},
        "<small>source draft 48 chars in 500 ms; reranker running 1001 ms</small>",
    ]
    missing_progress_event = [
        "Source-backed draft\n\nUse the cited source.\n\n_Refining source order with hosted reranker: 1001 ms…_",
        "",
        {"retrieval": {"reranker_wait_ms": 1001.0}},
        "<small>source draft 48 chars in 500 ms</small>",
    ]

    _assert(
        _reranker_wait_events([wait_event, missing_answer_event, missing_progress_event]) == [wait_event],
        "only reranker wait events with visible answer-panel progress should count",
    )


def _assert_reranker_cache_hit_events() -> None:
    cache_event = [
        "Source-backed draft\n\nUse the cited source.",
        "",
        {"retrieval": {"reranker_cache_hit": True}},
        "<small>5 sources via hnsw; cached native rerank; embed 141 ms; hnsw 2 ms; rerank cache 0 ms</small>",
    ]
    missing_progress_event = [
        "Source-backed draft\n\nUse the cited source.",
        "",
        {"retrieval": {"reranker_cache_hit": True}},
        "<small>5 sources via hnsw; native rerank; embed 141 ms; hnsw 2 ms; rerank 0 ms</small>",
    ]
    missing_diagnostics_event = [
        "Source-backed draft\n\nUse the cited source.",
        "",
        {"retrieval": {"reranker_cache_hit": False}},
        "<small>5 sources via hnsw; cached native rerank; embed 141 ms; hnsw 2 ms; rerank cache 0 ms</small>",
    ]

    _assert(
        _reranker_cache_hit_events([cache_event, missing_progress_event, missing_diagnostics_event])
        == [cache_event],
        "only reranker cache hits with visible cached-rerank progress should count",
    )


def _assert_reranker_cache_requirements() -> None:
    wait_event = [
        "Source-backed draft\n\nUse the cited source.\n\n_Refining source order with hosted reranker: 1001 ms…_",
        "",
        {"retrieval": {"reranker_wait_ms": 1001.0}},
        "<small>source draft 48 chars in 500 ms; reranker running 1001 ms</small>",
    ]
    cache_event = [
        "Source-backed draft\n\nUse the cited source.",
        "",
        {"retrieval": {"reranker_cache_hit": True}},
        "<small>5 sources via hnsw; cached native rerank; rerank cache 0 ms</small>",
    ]

    _require_reranker_cache_behavior(
        reranker_wait_events=[],
        reranker_cache_events=[cache_event],
        require_cache=True,
        require_no_wait=True,
    )
    _assert_raises_system_exit(
        lambda: _require_reranker_cache_behavior(
            reranker_wait_events=[],
            reranker_cache_events=[],
            require_cache=True,
            require_no_wait=False,
        ),
        "missing visible cache-hit progress should fail cache-required app smoke",
    )
    _assert_raises_system_exit(
        lambda: _require_reranker_cache_behavior(
            reranker_wait_events=[wait_event],
            reranker_cache_events=[cache_event],
            require_cache=False,
            require_no_wait=True,
        ),
        "reranker wait progress should fail no-wait app smoke",
    )


def _assert_raises_system_exit(fn, message: str) -> None:
    try:
        fn()
    except SystemExit:
        return
    raise AssertionError(message)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
