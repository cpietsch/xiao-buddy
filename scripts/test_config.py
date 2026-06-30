from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.config import load_settings


def main() -> None:
    original = load_settings()
    with _temporary_env(
        AGENT_MODEL="dynamic-agent-a",
        AGENT_MAX_TOKENS="321",
        AGENT_CONTEXT_CHARS="4321",
        AGENT_STARTUP_WARMUP_SECONDS="6.5",
        REQUEST_TIMEOUT_SECONDS="12.5",
        TOP_K="3",
        ADAPTIVE_TOP_K_ENABLED="1",
        FOCUSED_TOP_K="2",
        FOCUSED_CANDIDATE_K="4",
        GRAPH_RETRIEVAL_ENABLED="1",
        GRAPH_CANDIDATE_SLOTS="2",
        GRAPH_ARTIFACT_PATH="data/index/test_graph.json",
        ANSWER_LOG_ENABLED="1",
        ANSWER_LOG_PATH="dist/test-answer-runs.jsonl",
        ANSWER_LOG_ANSWER_CHARS="3456",
        GRADIO_SERVER_PORT="8877",
    ):
        settings = load_settings()
        _assert(settings.agent_model == "dynamic-agent-a", "agent model should be read at load time")
        _assert(settings.agent_max_tokens == 321, "agent max tokens should be read at load time")
        _assert(settings.agent_context_chars == 4321, "agent context budget should be read at load time")
        _assert(settings.agent_startup_warmup_seconds == 6.5, "agent warmup timeout should be read at load time")
        _assert(settings.request_timeout_seconds == 12.5, "request timeout should be read at load time")
        _assert(settings.top_k == 3, "top_k should be read at load time")
        _assert(settings.adaptive_top_k_enabled is True, "adaptive top_k flag should be read at load time")
        _assert(settings.focused_top_k == 2, "focused_top_k should be read at load time")
        _assert(settings.focused_candidate_k == 4, "focused_candidate_k should be read at load time")
        _assert(settings.graph_retrieval_enabled is True, "graph retrieval flag should be read at load time")
        _assert(settings.graph_candidate_slots == 2, "graph candidate slots should be read at load time")
        _assert(settings.graph_artifact_path == "data/index/test_graph.json", "graph artifact path should be read at load time")
        _assert(settings.answer_log_enabled is True, "answer log flag should be read at load time")
        _assert(settings.answer_log_path == "dist/test-answer-runs.jsonl", "answer log path should be read at load time")
        _assert(settings.answer_log_answer_chars == 3456, "answer log answer limit should be read at load time")
        _assert(settings.gradio_server_port == 8877, "Gradio port should be read at load time")

    with _temporary_env(GRAPH_RETRIEVAL_ENABLED="0", ADAPTIVE_TOP_K_ENABLED="0"):
        settings = load_settings()
        _assert(settings.graph_retrieval_enabled is False, "graph retrieval flag should parse false values")
        _assert(settings.adaptive_top_k_enabled is False, "adaptive top_k flag should parse false values")

    with _temporary_env(AGENT_MODEL="dynamic-agent-b"):
        settings = load_settings()
        _assert(settings.agent_model == "dynamic-agent-b", "subsequent loads should see env changes")

    offline = type(original)(
        embedding_base_url="",
        rerank_base_url="",
        agent_base_url="",
        top_k=original.top_k,
        candidate_k=original.candidate_k,
        adaptive_top_k_enabled=original.adaptive_top_k_enabled,
        focused_top_k=original.focused_top_k,
        focused_candidate_k=original.focused_candidate_k,
    )
    _assert(offline.embedding_base_url == "", "partial Settings override should still work")
    _assert(offline.candidate_k == original.candidate_k, "partial Settings override should preserve candidate_k")

    print("PASS config dynamic settings regression")


@contextmanager
def _temporary_env(**updates: str):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
