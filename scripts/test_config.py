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
        REQUEST_TIMEOUT_SECONDS="12.5",
        TOP_K="3",
        GRADIO_SERVER_PORT="8877",
    ):
        settings = load_settings()
        _assert(settings.agent_model == "dynamic-agent-a", "agent model should be read at load time")
        _assert(settings.agent_max_tokens == 321, "agent max tokens should be read at load time")
        _assert(settings.request_timeout_seconds == 12.5, "request timeout should be read at load time")
        _assert(settings.top_k == 3, "top_k should be read at load time")
        _assert(settings.gradio_server_port == 8877, "Gradio port should be read at load time")

    with _temporary_env(AGENT_MODEL="dynamic-agent-b"):
        settings = load_settings()
        _assert(settings.agent_model == "dynamic-agent-b", "subsequent loads should see env changes")

    offline = type(original)(
        embedding_base_url="",
        rerank_base_url="",
        agent_base_url="",
        top_k=original.top_k,
        candidate_k=original.candidate_k,
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
