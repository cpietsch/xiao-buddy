from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_EMBEDDING_BASE_URL = "http://129.212.184.41:8000/v1/embeddings"


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    embedding_base_url: str = _env("EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
    embedding_model: str = _env("EMBEDDING_MODEL", "qwen3-vl-embedding-2b")
    embedding_api_key: str = _env("EMBEDDING_API_KEY")

    rerank_base_url: str = _env("RERANK_BASE_URL")
    rerank_model: str = _env("RERANK_MODEL", "qwen3-vl-reranker-2b")
    rerank_api_key: str = _env("RERANK_API_KEY")

    agent_base_url: str = _env("AGENT_BASE_URL")
    agent_model: str = _env("AGENT_MODEL", "qwen3.5-small")
    agent_api_key: str = _env("AGENT_API_KEY")

    request_timeout_seconds: float = float(_env("REQUEST_TIMEOUT_SECONDS", "20"))
    top_k: int = int(_env("TOP_K", "5"))
    candidate_k: int = int(_env("CANDIDATE_K", "8"))


def load_settings() -> Settings:
    return Settings()
