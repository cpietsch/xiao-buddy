from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.clients import (
    _chat_delta_content,
    _chat_message_content,
    _content_to_text,
    _parse_native_rerank_scores,
    _parse_yes_no_score,
)


def main() -> None:
    _assert_chat_content_variants()
    _assert_stream_content_variants()
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


if __name__ == "__main__":
    main()
