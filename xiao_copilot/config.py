from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EMBEDDING_BASE_URL = ""


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


@dataclass(frozen=True)
class Settings:
    embedding_base_url: str = _env("EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL)
    embedding_model: str = _env("EMBEDDING_MODEL", "qwen3-vl-embedding-2b")
    embedding_api_key: str = _env("EMBEDDING_API_KEY")

    rerank_base_url: str = _env("RERANK_BASE_URL")
    rerank_model: str = _env("RERANK_MODEL", "qwen3-vl-reranker-2b")
    rerank_api_key: str = _env("RERANK_API_KEY")
    rerank_text_chars: int = int(_env("RERANK_TEXT_CHARS", "3200"))

    agent_base_url: str = _env("AGENT_BASE_URL")
    agent_model: str = _env("AGENT_MODEL", "qwen3.5-small")
    agent_api_key: str = _env("AGENT_API_KEY")

    request_timeout_seconds: float = float(_env("REQUEST_TIMEOUT_SECONDS", "20"))
    top_k: int = int(_env("TOP_K", "5"))
    candidate_k: int = int(_env("CANDIDATE_K", "8"))
    vector_index_manifest: str = _env("VECTOR_INDEX_MANIFEST", "data/index/xiao_vectors.json")
    vector_index_data: str = _env("VECTOR_INDEX_DATA")
    vector_index_archive_url: str = _env("VECTOR_INDEX_ARCHIVE_URL")
    vector_index_archive_sha256: str = _env("VECTOR_INDEX_ARCHIVE_SHA256")
    vector_candidate_k: int = int(_env("VECTOR_CANDIDATE_K", "96"))
    gradio_server_name: str = _env("GRADIO_SERVER_NAME", "127.0.0.1")
    gradio_server_port: int = int(_env("GRADIO_SERVER_PORT", "7860"))


def load_settings() -> Settings:
    return Settings()
