from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.endpoint_smoke import _collect_stream_stats
from xiao_copilot.clients import EndpointResult


def main() -> None:
    _assert_stream_stats_collect_first_token_latency()
    _assert_stream_stats_reject_provider_errors()
    _assert_stream_stats_reject_empty_streams()
    _assert_stream_stats_reject_zero_first_token_latency()
    print("PASS endpoint smoke regression")


def _assert_stream_stats_collect_first_token_latency() -> None:
    stats = _collect_stream_stats(
        [
            EndpointResult(ok=True, data=""),
            EndpointResult(ok=True, data="ok"),
            EndpointResult(ok=True, data=" done"),
        ],
        started_at=100.0,
        clock=_Clock([100.042, 100.125]),
    )

    _assert(stats.chunks == 2, f"expected two text chunks, got {stats.chunks}")
    _assert(stats.chars == 7, f"expected seven streamed chars, got {stats.chars}")
    _assert(stats.first_token_ms == 42, f"unexpected first-token latency: {stats.first_token_ms}")
    _assert(stats.total_ms == 125, f"unexpected total latency: {stats.total_ms}")


def _assert_stream_stats_reject_provider_errors() -> None:
    _assert_raises(
        lambda: _collect_stream_stats(
            [EndpointResult(ok=False, error="Provider error: quota exceeded")],
            started_at=100.0,
            clock=_Clock([100.1]),
        ),
        "quota exceeded",
    )


def _assert_stream_stats_reject_empty_streams() -> None:
    _assert_raises(
        lambda: _collect_stream_stats([], started_at=100.0, clock=_Clock([100.1])),
        "at least one text chunk",
    )


def _assert_stream_stats_reject_zero_first_token_latency() -> None:
    _assert_raises(
        lambda: _collect_stream_stats(
            [EndpointResult(ok=True, data="ok")],
            started_at=100.0,
            clock=_Clock([100.0, 100.0]),
        ),
        "positive first-token latency",
    )


class _Clock:
    def __init__(self, values: list[float]) -> None:
        self._values = values
        self._index = 0

    def __call__(self) -> float:
        if self._index >= len(self._values):
            raise AssertionError("test clock exhausted")
        value = self._values[self._index]
        self._index += 1
        return value


def _assert_raises(fn, expected_message: str) -> None:
    try:
        fn()
    except AssertionError as exc:
        _assert(expected_message in str(exc), f"expected {expected_message!r} in {exc!r}")
        return
    raise AssertionError(f"expected AssertionError containing {expected_message!r}")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
