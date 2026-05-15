from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.health_check import _extract_model_names, _models_url

ROOT = Path(__file__).resolve().parents[1]


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
    _assert_strict_warnings_exit_nonzero()

    print("PASS health-check parser regression")


def _assert_strict_warnings_exit_nonzero() -> None:
    with tempfile.TemporaryDirectory(prefix="xiao-health-") as temp_dir:
        missing_manifest = Path(temp_dir) / "missing-vector-index.json"
        env = {
            **os.environ,
            "VECTOR_INDEX_MANIFEST": str(missing_manifest),
        }
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "health_check.py"),
                "--strict-warnings",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    _assert(result.returncode == 2, f"strict warnings should exit 2, got {result.returncode}")
    _assert("summary:" in result.stdout and "warn" in result.stdout, "strict warning run should print warning summary")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
