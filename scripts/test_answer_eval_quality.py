from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.eval_answer_quality as answer_eval  # noqa: E402


def main() -> None:
    _assert_matching_helpers_are_normalized_and_source_specific()
    _assert_failure_diagnostic_includes_actionable_context()
    _assert_summary_reports_latency_and_rates()
    _assert_main_prints_failure_detail_on_miss()
    print("PASS answer eval quality regression")


def _assert_matching_helpers_are_normalized_and_source_specific() -> None:
    answer = "Use the XIAO ESP32C6 for Matter-native Thread and Zigbee projects [esp32c6-guide]."
    citations = "\n".join(
        [
            "- [esp32c6-guide] XIAO ESP32C6 Getting Started https://wiki.seeedstudio.com/xiao_esp32c6_getting_started/",
            "- [grove-sensor] Grove sensor reference https://wiki.seeedstudio.com/grove_sensor/",
        ]
    )

    _assert(
        answer_eval._missing_terms(answer, ["XIAO ESP32C6", "Matter native", "Thread", "Zigbee"]) == [],
        "term matching should ignore case and punctuation",
    )
    _assert(
        answer_eval._missing_terms(answer, ["802.15.4"]) == ["802.15.4"],
        "missing terms should preserve the configured term text",
    )
    _assert(
        answer_eval._contains_any_citation(
            citations,
            ["https://wiki.seeedstudio.com/xiao_esp32c6_getting_started/"],
        ),
        "citation matching should accept required source URLs",
    )
    _assert(
        answer_eval._inline_cites_required_source(answer, citations, ["xiao_esp32c6_getting_started"]),
        "inline citation should pass when the answer cites the required source id",
    )
    _assert(
        not answer_eval._inline_cites_required_source(
            "Use a different source instead [grove-sensor].",
            citations,
            ["xiao_esp32c6_getting_started"],
        ),
        "inline citation should fail when the answer cites only a non-required source",
    )


def _assert_failure_diagnostic_includes_actionable_context() -> None:
    detail = answer_eval._format_failure_detail(
        case={
            "id": "esp32c6-wireless",
            "query": "Which XIAO board supports Zigbee, Thread, and Matter over 802.15.4?",
        },
        missing_terms=["802.15.4", "Matter native"],
        required_citations=["https://wiki.seeedstudio.com/xiao_esp32c6_getting_started/"],
        answer="This answer cites a different source [grove-sensor].\n\nIt omits the radio detail.",
        citations_text="\n".join(
            [
                "- [grove-sensor] Grove sensor reference https://wiki.seeedstudio.com/grove_sensor/",
                "- [esp32c6-guide] XIAO ESP32C6 Getting Started https://wiki.seeedstudio.com/xiao_esp32c6_getting_started/",
            ]
        ),
    )

    for expected in [
        "failure detail:",
        "query: Which XIAO board supports Zigbee, Thread, and Matter over 802.15.4?",
        "missing_terms: 802.15.4, Matter native",
        "required_citations: https://wiki.seeedstudio.com/xiao_esp32c6_getting_started/",
        "required_source_ids: esp32c6-guide",
        "all_source_ids: grove-sensor, esp32c6-guide",
        "answer_excerpt: This answer cites a different source [grove-sensor]. It omits the radio detail.",
        "citation_excerpt: - [grove-sensor] Grove sensor reference https://wiki.seeedstudio.com/grove_sensor/ | - [esp32c6-guide] XIAO ESP32C6 Getting Started https://wiki.seeedstudio.com/xiao_esp32c6_getting_started/",
    ]:
        _assert(expected in detail, f"failure diagnostic should include {expected!r}")


def _assert_summary_reports_latency_and_rates() -> None:
    summary = answer_eval._summarize_results(
        [
            {
                "id": "pass-case",
                "ok": True,
                "fact_ok": True,
                "citation_ok": True,
                "inline_citation_ok": True,
                "agent_ok": True,
                "stream_ok": True,
                "draft_visible_ms": 2.0,
                "source_draft_build_ms": 0.2,
                "first_token_ms": 4.0,
                "agent_first_visible_ms": 6.0,
                "retrieval_timings_ms": {
                    "knowledge_load": 0.1,
                    "query_embedding": 1.0,
                    "vector_search": 2.0,
                    "reranker": 3.0,
                    "total": 7.0,
                },
                "total_ms": 10.0,
            },
            {
                "id": "miss-case",
                "ok": False,
                "fact_ok": False,
                "citation_ok": True,
                "inline_citation_ok": False,
                "agent_ok": True,
                "stream_ok": True,
                "draft_visible_ms": 6.0,
                "source_draft_build_ms": 0.6,
                "first_token_ms": 20.0,
                "agent_first_visible_ms": 26.0,
                "retrieval_timings_ms": {
                    "knowledge_load": 0.3,
                    "query_embedding": 5.0,
                    "vector_search": 6.0,
                    "reranker": 9.0,
                    "total": 21.0,
                },
                "total_ms": 30.0,
            },
        ]
    )
    _assert(summary["cases"] == 2, "summary should count answer cases")
    _assert(summary["passes"] == 1, "summary should count passing answer cases")
    _assert(summary["failures"] == ["miss-case"], "summary should list failing answer cases")
    _assert(summary["fact_rate"] == 0.5, "summary should compute fact hit rate")
    _assert(summary["citation_rate"] == 1.0, "summary should compute citation hit rate")
    _assert(summary["inline_citation_rate"] == 0.5, "summary should compute inline citation rate")
    _assert(summary["agent_rate"] == 1.0, "summary should compute agent hit rate")
    _assert(summary["stream_rate"] == 1.0, "summary should compute stream hit rate")
    _assert(summary["p50_draft_visible_ms"] == 4.0, "summary should compute p50 draft latency")
    _assert(abs(float(summary["p95_draft_visible_ms"]) - 5.8) < 0.0001, "summary should compute p95 draft latency")
    _assert(summary["max_draft_visible_ms"] == 6.0, "summary should compute max draft latency")
    _assert(summary["p50_source_draft_build_ms"] == 0.4, "summary should compute p50 draft-build latency")
    _assert(
        abs(float(summary["p95_source_draft_build_ms"]) - 0.58) < 0.0001,
        "summary should compute p95 draft-build latency",
    )
    _assert(summary["max_source_draft_build_ms"] == 0.6, "summary should compute max draft-build latency")
    _assert(summary["p50_first_token_ms"] == 12.0, "summary should compute p50 first-token latency")
    _assert(summary["p95_first_token_ms"] == 19.2, "summary should compute p95 first-token latency")
    _assert(summary["max_first_token_ms"] == 20.0, "summary should compute max first-token latency")
    _assert(summary["p50_agent_first_visible_ms"] == 16.0, "summary should compute p50 agent-visible latency")
    _assert(
        abs(float(summary["p95_agent_first_visible_ms"]) - 25.0) < 0.0001,
        "summary should compute p95 agent-visible latency",
    )
    _assert(summary["max_agent_first_visible_ms"] == 26.0, "summary should compute max agent-visible latency")
    _assert(summary["p50_ms"] == 20.0, "summary should compute p50 latency")
    _assert(summary["p95_ms"] == 29.0, "summary should compute p95 latency")
    _assert(summary["max_ms"] == 30.0, "summary should compute max latency")
    retrieval_stage_ms = summary["retrieval_stage_ms"]
    _assert(retrieval_stage_ms["query_embedding"]["p50_ms"] == 3.0, "summary should compute retrieval p50")
    _assert(retrieval_stage_ms["query_embedding"]["p95_ms"] == 4.8, "summary should compute retrieval p95")
    _assert(retrieval_stage_ms["query_embedding"]["max_ms"] == 5.0, "summary should compute retrieval max")
    _assert(retrieval_stage_ms["reranker"]["p50_ms"] == 6.0, "summary should include reranker stage")


def _assert_main_prints_failure_detail_on_miss() -> None:
    original_answer_question = answer_eval.answer_question
    env_keys = [
        "ANSWER_EVAL_PATH",
        "ANSWER_EVAL_CASE_IDS",
        "ANSWER_EVAL_LIMIT",
        "ANSWER_EVAL_REQUIRE_AGENT",
        "ANSWER_EVAL_REQUIRE_STREAM",
        "ANSWER_EVAL_JSON_OUTPUT",
    ]
    original_env = {key: os.environ.get(key) for key in env_keys}
    report_payload: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as tmp:
        eval_path = Path(tmp) / "answer_eval.jsonl"
        report_path = Path(tmp) / "answer-quality.json"
        eval_path.write_text(
            json.dumps(
                {
                    "id": "synthetic-miss",
                    "query": "Which protocol detail is required?",
                    "must_include": ["802.15.4"],
                    "must_cite": ["https://wiki.seeedstudio.com/xiao_esp32c6_getting_started/"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        def fake_answer_question(_image: object, _query: str) -> tuple[str, str, dict[str, object]]:
            answer = "Use XIAO ESP32C6 for Thread and Zigbee [esp32c6-guide]."
            citations = (
                "- [esp32c6-guide] XIAO ESP32C6 Getting Started "
                "https://wiki.seeedstudio.com/xiao_esp32c6_getting_started/"
            )
            diagnostics = {
                "agent_used": True,
                "agent_streamed": True,
                "agent": {"stream_chunks": 1, "stream_chars": len(answer), "first_token_ms": 3.5},
                "agent_first_visible_ms": 6.0,
                "draft": {"visible": True, "chars": 120, "first_visible_ms": 2.5},
                "retrieval": {
                    "timings_ms": {
                        "knowledge_load": 0.1,
                        "query_embedding": 1.2,
                        "vector_search": 0.8,
                        "reranker": 2.1,
                        "total": 5.0,
                    }
                },
                "timings_ms": {"source_draft": 0.4, "total": 7},
            }
            return answer, citations, diagnostics

        try:
            answer_eval.answer_question = fake_answer_question
            os.environ["ANSWER_EVAL_PATH"] = str(eval_path)
            os.environ["ANSWER_EVAL_CASE_IDS"] = ""
            os.environ["ANSWER_EVAL_LIMIT"] = "0"
            os.environ["ANSWER_EVAL_REQUIRE_AGENT"] = "1"
            os.environ["ANSWER_EVAL_REQUIRE_STREAM"] = "1"
            os.environ["ANSWER_EVAL_JSON_OUTPUT"] = str(report_path)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                try:
                    answer_eval.main()
                except SystemExit as exc:
                    _assert(exc.code == 1, "answer eval should exit non-zero on a miss")
                else:
                    raise AssertionError("answer eval should fail for a missing required term")
            _assert(report_path.exists(), "answer eval should write JSON report before exiting")
            report_payload.update(json.loads(report_path.read_text(encoding="utf-8")))
        finally:
            answer_eval.answer_question = original_answer_question
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    captured = output.getvalue()
    for expected in [
        "MISS synthetic-miss",
        "missing=802.15.4",
        "failure detail:",
        "query: Which protocol detail is required?",
        "missing_terms: 802.15.4",
        "required_citations: https://wiki.seeedstudio.com/xiao_esp32c6_getting_started/",
        "required_source_ids: esp32c6-guide",
        "answer_excerpt: Use XIAO ESP32C6 for Thread and Zigbee [esp32c6-guide].",
        "first_token_ms=3.5",
        "answer source-draft latency: p50_ms=2.5 p95_ms=2.5 max_ms=2.5",
        "answer source-draft build latency: p50_ms=0.4 p95_ms=0.4 max_ms=0.4",
        "answer first-token latency: p50_ms=3.5 p95_ms=3.5 max_ms=3.5",
        "answer agent-visible latency: p50_ms=6.0 p95_ms=6.0 max_ms=6.0",
        "answer latency: p50_ms=7.0 p95_ms=7.0 max_ms=7.0",
        "answer retrieval stage latency:",
        "query_embedding: p50_ms=1.2 p95_ms=1.2 max_ms=1.2",
        "reranker: p50_ms=2.1 p95_ms=2.1 max_ms=2.1",
        "wrote ",
    ]:
        _assert(expected in captured, f"main failure output should include {expected!r}")
    summary = report_payload["summary"]
    results = report_payload["results"]
    metadata = report_payload["metadata"]
    _assert(isinstance(metadata, dict), "JSON report should include metadata object")
    _assert(metadata["report_type"] == "answer_quality", "JSON report should identify answer quality report type")
    _assert(metadata["selected_case_ids"] == ["synthetic-miss"], "JSON report metadata should include selected case ids")
    _assert(isinstance(summary, dict), "JSON report should include summary object")
    _assert(isinstance(results, list) and len(results) == 1, "JSON report should include one result")
    _assert(summary["failures"] == ["synthetic-miss"], "JSON report should include failing case id")
    _assert(results[0]["missing_terms"] == ["802.15.4"], "JSON report should include missing terms")
    _assert(results[0]["required_source_ids"] == ["esp32c6-guide"], "JSON report should include required source ids")
    _assert(results[0]["stream_chunks"] == 1, "JSON report should include stream chunk count")
    _assert(results[0]["draft_visible_ms"] == 2.5, "JSON report should include draft-visible latency")
    _assert(results[0]["source_draft_build_ms"] == 0.4, "JSON report should include draft-build latency")
    _assert(results[0]["first_token_ms"] == 3.5, "JSON report should include first-token latency")
    _assert(results[0]["agent_first_visible_ms"] == 6.0, "JSON report should include agent-visible latency")
    _assert(
        results[0]["retrieval_timings_ms"]["query_embedding"] == 1.2,
        "JSON report should include retrieval stage timing",
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
