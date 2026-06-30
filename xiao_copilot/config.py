from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar


DEFAULT_EMBEDDING_BASE_URL = ""
T = TypeVar("T")


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_field(name: str, default: str = ""):
    return field(default_factory=lambda: _env(name, default))


def _typed_env_field(converter: Callable[[str], T], name: str, default: str):
    return field(default_factory=lambda: converter(_env(name, default)))


def _bool_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    embedding_base_url: str = _env_field("EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
    embedding_model: str = _env_field("EMBEDDING_MODEL", "qwen3-vl-embedding-2b")
    embedding_api_key: str = _env_field("EMBEDDING_API_KEY")

    rerank_base_url: str = _env_field("RERANK_BASE_URL")
    rerank_model: str = _env_field("RERANK_MODEL", "qwen3-vl-reranker-2b")
    rerank_api_key: str = _env_field("RERANK_API_KEY")
    rerank_text_chars: int = _typed_env_field(int, "RERANK_TEXT_CHARS", "2800")

    agent_base_url: str = _env_field("AGENT_BASE_URL")
    agent_model: str = _env_field("AGENT_MODEL", "openai-compatible-agent")
    agent_api_key: str = _env_field("AGENT_API_KEY")
    agent_max_tokens: int = _typed_env_field(int, "AGENT_MAX_TOKENS", "2000")
    agent_context_chars: int = _typed_env_field(int, "AGENT_CONTEXT_CHARS", "2000")
    agent_startup_warmup_seconds: float = _typed_env_field(float, "AGENT_STARTUP_WARMUP_SECONDS", "8")

    request_timeout_seconds: float = _typed_env_field(float, "REQUEST_TIMEOUT_SECONDS", "20")
    top_k: int = _typed_env_field(int, "TOP_K", "10")
    candidate_k: int = _typed_env_field(int, "CANDIDATE_K", "12")
    adaptive_top_k_enabled: bool = _typed_env_field(_bool_env, "ADAPTIVE_TOP_K_ENABLED", "1")
    focused_top_k: int = _typed_env_field(int, "FOCUSED_TOP_K", "6")
    focused_candidate_k: int = _typed_env_field(int, "FOCUSED_CANDIDATE_K", "8")
    vector_index_manifest: str = _env_field("VECTOR_INDEX_MANIFEST", "data/index/xiao_vectors.json")
    vector_index_data: str = _env_field("VECTOR_INDEX_DATA")
    vector_index_archive_url: str = _env_field("VECTOR_INDEX_ARCHIVE_URL")
    vector_index_archive_sha256: str = _env_field("VECTOR_INDEX_ARCHIVE_SHA256")
    vector_candidate_k: int = _typed_env_field(int, "VECTOR_CANDIDATE_K", "96")
    graph_retrieval_enabled: bool = _typed_env_field(_bool_env, "GRAPH_RETRIEVAL_ENABLED", "0")
    graph_candidate_slots: int = _typed_env_field(int, "GRAPH_CANDIDATE_SLOTS", "4")
    graph_artifact_path: str = _env_field("GRAPH_ARTIFACT_PATH", "data/index/knowledge_graph.json")

    answer_log_enabled: bool = _typed_env_field(_bool_env, "ANSWER_LOG_ENABLED", "0")
    answer_log_path: str = _env_field("ANSWER_LOG_PATH", "dist/answer-runs.jsonl")
    answer_log_answer_chars: int = _typed_env_field(int, "ANSWER_LOG_ANSWER_CHARS", "12000")
    gradio_server_name: str = _env_field("GRADIO_SERVER_NAME", "127.0.0.1")
    gradio_server_port: int = _typed_env_field(int, "GRADIO_SERVER_PORT", "7860")


def load_settings() -> Settings:
    return Settings()
