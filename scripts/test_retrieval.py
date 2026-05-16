from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.knowledge_base import KnowledgeChunk
from xiao_copilot.retrieval import (
    RERANK_METADATA_TAG_LIMIT,
    _include_rerank_board_metadata,
    _rerank_text,
)


def main() -> None:
    _assert_rerank_text_includes_useful_metadata()
    _assert_rerank_text_adds_board_metadata_only_when_requested()
    _assert_rerank_text_stays_compact_without_metadata()
    _assert_board_metadata_choice_heuristic()
    print("PASS retrieval formatting regression")


def _assert_rerank_text_includes_useful_metadata() -> None:
    tags = [f"tag-{index}" for index in range(RERANK_METADATA_TAG_LIMIT + 3)]
    chunk = KnowledgeChunk(
        id="chunk-1",
        title="XIAO board setup",
        source="https://example.test/xiao",
        text="abcdefghijklmnopqrstuvwxyz",
        board_id="xiao-esp32c3",
        metadata={
            "aliases": ["XIAO ESP32C3"],
            "tags": tags,
        },
    )
    text = _rerank_text(chunk, max_chars=10)
    _assert("Metadata: tags=tag-0, tag-1" in text, "tags should be included")
    _assert("tag-15" in text and "tag-16" not in text, "reranker tags should be bounded")
    _assert("board=xiao-esp32c3" not in text, "board metadata should be opt-in")
    _assert("XIAO ESP32C3" not in text, "aliases should not be added to reranker metadata")
    _assert(text.endswith("abcdefghij"), "content should be truncated after metadata is assembled")


def _assert_rerank_text_adds_board_metadata_only_when_requested() -> None:
    chunk = KnowledgeChunk(
        id="chunk-board",
        title="XIAO board setup",
        source="https://example.test/xiao",
        text="setup details",
        board_id="xiao-esp32c3",
        metadata={"tags": ["xiao", "esp32c3"]},
    )
    text = _rerank_text(chunk, max_chars=100, include_board_metadata=True)
    _assert(
        "Metadata: board=xiao-esp32c3; tags=xiao, esp32c3" in text,
        "board metadata should be included when requested",
    )


def _assert_rerank_text_stays_compact_without_metadata() -> None:
    chunk = KnowledgeChunk(
        id="chunk-2",
        title="General Grove note",
        source="https://example.test/grove",
        text=" short note ",
    )
    text = _rerank_text(chunk, max_chars=100)
    _assert("Metadata:" not in text, "empty metadata should not add a metadata line")
    _assert(text == "General Grove note\nhttps://example.test/grove\nshort note", "basic reranker text changed")


def _assert_board_metadata_choice_heuristic() -> None:
    c3 = KnowledgeChunk(id="c3", title="", source="", text="", board_id="xiao-esp32c3")
    s3 = KnowledgeChunk(id="s3", title="", source="", text="", board_id="xiao-esp32s3")
    _assert(
        _include_rerank_board_metadata("Which XIAO board supports channel sounding?", [c3, s3]),
        "board-choice queries with distinct boards should include board metadata",
    )
    _assert(
        not _include_rerank_board_metadata("Which XIAO board supports channel sounding?", [c3, c3]),
        "same-board candidates should not include board metadata",
    )
    _assert(
        not _include_rerank_board_metadata("What I2C address is used on XIAO ESP32C3?", [c3, s3]),
        "non-choice board-specific queries should not include board metadata",
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
