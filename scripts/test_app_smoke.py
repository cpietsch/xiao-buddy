from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.app_smoke import (
    _first_token_ms,
    _format_ms,
    _require_first_token_latency,
    _stream_events_with_first_token,
)


def main() -> None:
    _assert_first_token_helpers()
    _assert_stream_first_token_events()
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
