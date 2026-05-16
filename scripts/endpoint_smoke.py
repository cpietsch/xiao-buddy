from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.clients import chat_completion, embed_texts, rerank
from xiao_copilot.config import load_settings


def main() -> None:
    settings = load_settings()
    checks = [
        _check_embedding(settings),
        _check_reranker(settings),
        _check_agent(settings),
    ]
    for line in checks:
        print(line)
    print("PASS endpoint functional smoke")


def _check_embedding(settings) -> str:
    _require(settings.embedding_base_url, "EMBEDDING_BASE_URL is required")
    started_at = perf_counter()
    result = embed_texts(
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        texts=["Seeed Studio XIAO ESP32-C5 supports dual-band Wi-Fi 6."],
        api_key=settings.embedding_api_key,
        timeout=settings.request_timeout_seconds,
    )
    _assert(result.ok, f"embedding request failed: {result.error}")
    vectors = result.data or []
    _assert(len(vectors) == 1, f"expected one embedding vector, got {len(vectors)}")
    vector = vectors[0]
    _assert(isinstance(vector, list) and len(vector) > 100, "embedding vector should be a non-trivial list")
    _assert(all(isinstance(value, (int, float)) for value in vector[:16]), "embedding values should be numeric")
    return f"OK   embedding functional: dim={len(vector)} ms={_elapsed_ms(started_at)}"


def _check_reranker(settings) -> str:
    _require(settings.rerank_base_url, "RERANK_BASE_URL is required")
    started_at = perf_counter()
    result = rerank(
        base_url=settings.rerank_base_url,
        model=settings.rerank_model,
        query="Which board should I choose for 5 GHz WiFi?",
        documents=[
            "The Seeed Studio XIAO ESP32-C5 supports dual-band 2.4 GHz and 5 GHz Wi-Fi 6.",
            "The XIAO RP2040 is a small RP2040 microcontroller board without Wi-Fi.",
        ],
        api_key=settings.rerank_api_key,
        timeout=settings.request_timeout_seconds,
    )
    _assert(result.ok, f"reranker request failed: {result.error}")
    scores = result.data or []
    _assert(len(scores) == 2, f"expected two rerank scores, got {len(scores)}")
    indexes = {index for index, _score in scores}
    _assert(indexes == {0, 1}, f"reranker returned unexpected indexes: {sorted(indexes)}")
    for _index, score in scores:
        _assert(isinstance(score, (int, float)), "reranker scores should be numeric")
    score_by_index = dict(scores)
    _assert(
        score_by_index[0] > score_by_index[1],
        f"reranker should score the relevant Wi-Fi 6 document higher: {_format_scores(scores)}",
    )
    mode = (result.meta or {}).get("mode", "unknown")
    return f"OK   reranker functional: mode={mode} scores={_format_scores(scores)} ms={_elapsed_ms(started_at)}"


def _check_agent(settings) -> str:
    _require(settings.agent_base_url, "AGENT_BASE_URL is required")
    started_at = perf_counter()
    result = chat_completion(
        base_url=settings.agent_base_url,
        model=settings.agent_model,
        messages=[
            {"role": "system", "content": "Reply with one short sentence."},
            {"role": "user", "content": "Say that endpoint smoke is ok."},
        ],
        api_key=settings.agent_api_key,
        timeout=settings.request_timeout_seconds,
    )
    _assert(result.ok, f"agent request failed: {result.error}")
    text = str(result.data or "").strip()
    _assert(text, "agent response should not be empty")
    return f"OK   agent functional: chars={len(text)} ms={_elapsed_ms(started_at)}"


def _format_scores(scores: list[tuple[int, float]]) -> str:
    return ",".join(f"{index}:{score:.4g}" for index, score in scores)


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _require(value: str, message: str) -> None:
    if not value:
        raise SystemExit(message)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
