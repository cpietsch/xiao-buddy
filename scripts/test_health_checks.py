from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.health_check import _extract_model_names, _models_url


def main() -> None:
    openai_body = {
        "object": "list",
        "data": [
            {"id": "qwen3-vl-embedding-2b", "object": "model"},
        ],
    }
    llama_cpp_body = {
        "models": [
            {"name": "unsloth/gemma-4-31B-it-GGUF", "model": "unsloth/gemma-4-31B-it-GGUF"},
        ],
        "object": "list",
        "data": [
            {
                "id": "unsloth/gemma-4-31B-it-GGUF",
                "aliases": ["gemma-local"],
                "object": "model",
            },
        ],
    }

    names = _extract_model_names(openai_body)
    _assert("qwen3-vl-embedding-2b" in names, "OpenAI data ids should be extracted")

    names = _extract_model_names(llama_cpp_body)
    _assert("unsloth/gemma-4-31B-it-GGUF" in names, "llama.cpp model name should be extracted")
    _assert("gemma-local" in names, "llama.cpp aliases should be extracted")

    _assert(
        _models_url("https://example.test/v1/embeddings") == "https://example.test/v1/models",
        "embedding URL should resolve to /v1/models",
    )
    _assert(
        _models_url("https://example.test/") == "https://example.test/v1/models",
        "bare endpoint should resolve to /v1/models",
    )

    print("PASS health-check parser regression")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
