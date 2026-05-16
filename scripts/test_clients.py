from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xiao_copilot.clients as clients
from xiao_copilot.clients import (
    _chat_delta_content,
    _chat_message_content,
    _content_to_text,
    _parse_native_rerank_scores,
    _parse_yes_no_score,
    _provider_error_message,
    chat_completion_stream,
)


def main() -> None:
    _assert_chat_content_variants()
    _assert_stream_content_variants()
    _assert_provider_error_variants()
    _assert_stream_provider_errors_are_yielded()
    _assert_rerank_parsers()
    print("PASS client parser regression")


def _assert_chat_content_variants() -> None:
    _assert(
        _chat_message_content({"choices": [{"message": {"content": "plain answer"}}]}) == "plain answer",
        "chat parser should read message.content strings",
    )
    _assert(
        _chat_message_content({"choices": [{"text": "legacy text answer"}]}) == "legacy text answer",
        "chat parser should read choices[].text fallback",
    )
    body = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "part one "},
                        {"type": "text", "text": {"content": "part two"}},
                    ]
                }
            }
        ]
    }
    _assert(
        _chat_message_content(body) == "part one part two",
        "chat parser should flatten text content parts",
    )
    _assert(_content_to_text(None) == "", "None content should become empty text")


def _assert_stream_content_variants() -> None:
    _assert(
        _chat_delta_content({"choices": [{"delta": {"content": "stream delta"}}]}) == "stream delta",
        "stream parser should read delta.content",
    )
    _assert(
        _chat_delta_content({"choices": [{"text": "stream text"}]}) == "stream text",
        "stream parser should read choices[].text fallback",
    )
    body = {"choices": [{"delta": {"content": [{"text": "a"}, {"content": "b"}]}}]}
    _assert(_chat_delta_content(body) == "ab", "stream parser should flatten content parts")


def _assert_provider_error_variants() -> None:
    _assert(
        _provider_error_message({"error": {"message": "quota exceeded"}}) == "Provider error: quota exceeded",
        "provider error parser should read error.message",
    )
    _assert(
        _provider_error_message({"error": "model unavailable"}) == "Provider error: model unavailable",
        "provider error parser should read string errors",
    )
    _assert(_provider_error_message({"choices": []}) == "", "non-error payloads should not produce errors")


def _assert_stream_provider_errors_are_yielded() -> None:
    original_post = clients.requests.post
    try:
        clients.requests.post = _fake_stream_post  # type: ignore[assignment]
        results = list(
            chat_completion_stream(
                base_url="https://example.test/v1",
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
            )
        )
    finally:
        clients.requests.post = original_post  # type: ignore[assignment]

    _assert(len(results) == 2, f"expected token then error, got {len(results)} results")
    _assert(results[0].ok and results[0].data == "partial", "stream should yield content before error")
    _assert(not results[1].ok, "stream provider error should yield failed EndpointResult")
    _assert("quota exceeded" in results[1].error, "stream provider error should include provider detail")


def _assert_rerank_parsers() -> None:
    scores = _parse_native_rerank_scores(
        {"results": [{"index": 1, "relevance_score": 0.25}, {"index": 0, "score": 0.75}]},
        expected_count=2,
    )
    _assert(scores == [(1, 0.25), (0, 0.75)], "native rerank parser should preserve indexes and scores")

    yes_score = _parse_yes_no_score({"choices": [{"text": "yes"}]})
    no_score = _parse_yes_no_score({"choices": [{"message": {"content": "no"}}]})
    _assert(yes_score == 1.0, "yes fallback should score as relevant")
    _assert(no_score == 0.0, "no fallback should score as irrelevant")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _FakeStreamResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode: bool = False):
        lines = [
            'data: {"choices":[{"delta":{"content":"partial"}}]}',
            'data: {"error":{"message":"quota exceeded"}}',
            "data: [DONE]",
        ]
        yield from lines if decode_unicode else [line.encode() for line in lines]


def _fake_stream_post(*_args, **_kwargs):
    return _FakeStreamResponse()


if __name__ == "__main__":
    main()
