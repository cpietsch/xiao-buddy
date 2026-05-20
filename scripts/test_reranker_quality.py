from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_reranker_quality import _select_negatives, _summarize_results
from xiao_copilot.knowledge_base import KnowledgeChunk


def main() -> None:
    _assert_answer_equivalent_chunks_are_not_hard_negatives()
    _assert_summary_reports_close_margins()
    print("PASS reranker quality regression")


def _assert_answer_equivalent_chunks_are_not_hard_negatives() -> None:
    positive = KnowledgeChunk(
        id="positive",
        title="Expected ESP32C6 source",
        source="https://wiki.seeedstudio.com/xiao_esp32c6_getting_started/",
        text="XIAO ESP32C6 is Matter native and supports Zigbee and Thread over 802.15.4.",
        board_id="xiao-esp32c6",
        metadata={"source_file": "expected.md"},
    )
    answer_equivalent = KnowledgeChunk(
        id="answer-equivalent",
        title="Alternate ESP32C6 source",
        source="https://wiki.seeedstudio.com/xiao-esp32-swift/",
        text="XIAO ESP32C6 is Matter native and supports Zigbee and Thread over 802.15.4.",
        board_id="xiao-esp32c6",
        metadata={"source_file": "alternate.md"},
    )
    true_negative = KnowledgeChunk(
        id="true-negative",
        title="Different wireless board",
        source="https://wiki.seeedstudio.com/wio_e5/",
        text="Wio-E5 LoRa module setup notes.",
        board_id="",
        metadata={"source_file": "negative.md"},
    )
    negatives = _select_negatives(
        query="Which XIAO board is Matter native and supports Zigbee and Thread over 802.15.4?",
        corpus=[positive, answer_equivalent, true_negative],
        citations=["https://wiki.seeedstudio.com/xiao_esp32c6_getting_started/"],
        terms=["XIAO ESP32C6", "Matter native", "Zigbee", "Thread", "802.15.4"],
        expected_board_id="xiao-esp32c6",
        positive=positive,
        count=1,
    )
    _assert([chunk.id for chunk in negatives] == ["true-negative"], "answer-equivalent chunks are not negatives")


def _assert_summary_reports_close_margins() -> None:
    summary = _summarize_results(
        [
            {
                "id": "close-case",
                "category": "xiao-core",
                "ok": True,
                "margin": 0.012,
                "ms": 10.0,
                "positive_id": "positive",
                "best_negative_id": "negative",
                "positive_title": "Positive title",
                "best_negative_title": "Negative title",
            },
            {
                "id": "wide-case",
                "category": "xiao-core",
                "ok": True,
                "margin": 0.25,
                "ms": 12.0,
                "positive_id": "positive-2",
                "best_negative_id": "negative-2",
                "positive_title": "Positive title 2",
                "best_negative_title": "Negative title 2",
            },
        ],
        close_margin=0.05,
    )
    _assert(summary["passes"] == 2, "summary should count passing cases")
    _assert([case["id"] for case in summary["close_cases"]] == ["close-case"], "close cases should be reported")
    _assert(summary["close_cases"][0]["best_negative_id"] == "negative", "close case should include negative id")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
