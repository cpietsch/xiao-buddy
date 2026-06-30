from __future__ import annotations

import json
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xiao_copilot.pipeline as pipeline
from xiao_copilot.clients import EndpointResult
from xiao_copilot.config import Settings
from xiao_copilot.knowledge_base import KnowledgeChunk
from xiao_copilot.retrieval import RetrievalStage


def main() -> None:
    _assert_pipeline_writes_answer_run_log()
    print("PASS answer logger regression")


def _assert_pipeline_writes_answer_run_log() -> None:
    original_load_settings = pipeline.load_settings
    original_retrieve_progressive = pipeline.retrieve_progressive
    original_generate_stream = pipeline._generate_with_agent_stream

    with tempfile.TemporaryDirectory(prefix="xiao-answer-log-") as temp_dir:
        log_path = Path(temp_dir) / "answer-runs.jsonl"

        def fake_settings() -> Settings:
            return Settings(
                embedding_base_url="",
                rerank_base_url="",
                agent_base_url="https://example.test/v1",
                answer_log_enabled=True,
                answer_log_path=str(log_path),
                answer_log_answer_chars=24,
            )

        try:
            pipeline.load_settings = fake_settings  # type: ignore[assignment]
            pipeline.retrieve_progressive = _fake_retrieve_progressive  # type: ignore[assignment]
            pipeline._generate_with_agent_stream = _fake_agent_stream  # type: ignore[assignment]

            events = list(pipeline.answer_question_stream(None, "Which log source should I inspect?"))
        finally:
            pipeline.load_settings = original_load_settings  # type: ignore[assignment]
            pipeline.retrieve_progressive = original_retrieve_progressive  # type: ignore[assignment]
            pipeline._generate_with_agent_stream = original_generate_stream  # type: ignore[assignment]

        _assert(events, "pipeline should yield events")
        final_answer, _citations, final_diagnostics, _progress = events[-1]
        run_id = final_diagnostics.get("run_id")
        _assert(isinstance(run_id, str) and run_id, "final diagnostics should include run_id")
        _assert(log_path.exists(), "answer log file should be written")

        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        _assert(len(records) == 1, "one completed answer should write one log record")
        record = records[0]
        _assert(record["run_id"] == run_id, "log run_id should match diagnostics")
        _assert(record["status"] == "ok", "successful answer should log ok status")
        _assert(record["question"] == "Which log source should I inspect?", "question should be logged")
        _assert(record["answer_chars"] == len(final_answer), "full answer length should be logged")
        _assert(record["answer_truncated"] is True, "long answer should be marked truncated")
        _assert(len(record["answer"]) <= 24, "logged answer should respect configured char limit")
        _assert(record["citations"][0]["id"] == "log-source", "citations should be structured")
        _assert("Sources used" in record["citations_text"], "citation markdown should be retained")
        _assert(record["diagnostics"]["top_sources"][0]["id"] == "log-source", "top source should be logged")
        _assert("fallback_answer" not in record["failure_flags"], "agent answer should not be marked fallback")
        _assert(record["event_count"] == len(events), "event count should be logged")


def _fake_retrieve_progressive(
    _question: str,
    _settings: Settings,
    image_data_url: str | None = None,
) -> Iterator[RetrievalStage]:
    chunk = KnowledgeChunk(
        id="log-source",
        title="Log source",
        source="https://wiki.seeedstudio.com/log/",
        text="Use this source when testing answer run logging.",
        kind="wiki",
    )
    diagnostics = {
        "vector_index_backend": "lexical",
        "embedding_used": False,
        "reranker_used": False,
        "citations": [chunk.id],
        "timings_ms": {"query_embedding": 0.0, "total": 1.0},
        "image_used": bool(image_data_url),
    }
    yield RetrievalStage(stage="final", chunks=[chunk], diagnostics=diagnostics)


def _fake_agent_stream(*_args, **_kwargs) -> Iterator[EndpointResult]:
    yield EndpointResult(ok=True, data="Inspect the logged source record [log-source].")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
