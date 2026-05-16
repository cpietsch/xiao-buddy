from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.index_corpus import hash_chunks, indexable_chunks
from xiao_copilot.knowledge_base import load_knowledge_base
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
    _assert_missing_vector_data_warns_without_artifact()
    _assert_artifact_backed_strict_health_ok()
    _assert_corrupt_local_vector_data_fails_health()

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


def _assert_missing_vector_data_warns_without_artifact() -> None:
    with tempfile.TemporaryDirectory(prefix="xiao-health-missing-data-") as temp_dir:
        work = Path(temp_dir)
        manifest_path = work / "index" / "test_vectors.json"
        data_path = work / "index" / "test_vectors.f16"
        _write_artifact_backed_manifest(manifest_path, data_path)
        data_path.unlink()

        env = {
            **os.environ,
            "VECTOR_INDEX_MANIFEST": str(manifest_path),
            "VECTOR_INDEX_DATA": "",
            "VECTOR_INDEX_ARCHIVE_URL": "",
            "VECTOR_INDEX_ARCHIVE_SHA256": "",
        }
        command = [
            sys.executable,
            str(ROOT / "scripts" / "health_check.py"),
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        strict_result = subprocess.run(
            [*command, "--strict-warnings"],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    _assert(
        result.returncode == 0,
        f"missing vector data without strict warnings should pass health, got {result.returncode}",
    )
    _assert(
        "WARN vector data:" in result.stdout and "run scripts/build_wiki_vector_index.py" in result.stdout,
        "health should tell fresh clones to build the local vector index",
    )
    _assert(
        "runtime artifact restore is configured" not in result.stdout,
        "health should not report artifact restore readiness without artifact env vars",
    )
    _assert(
        strict_result.returncode == 2,
        f"strict missing-vector-data warning should exit 2, got {strict_result.returncode}",
    )


def _assert_artifact_backed_strict_health_ok() -> None:
    with tempfile.TemporaryDirectory(prefix="xiao-health-artifact-") as temp_dir:
        work = Path(temp_dir)
        manifest_path = work / "index" / "test_vectors.json"
        data_path = work / "index" / "test_vectors.f16"
        output_dir = work / "out"
        _write_artifact_backed_manifest(manifest_path, data_path)
        archive_path = output_dir / "test-vectors.tar.gz"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "package_vector_artifact.py"),
                "--manifest",
                str(manifest_path),
                "--data",
                str(data_path),
                "--output-dir",
                str(output_dir),
                "--name",
                archive_path.name,
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        metadata_path = archive_path.with_suffix(archive_path.suffix + ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        data_path.unlink()

        env = {
            **os.environ,
            "EMBEDDING_BASE_URL": "https://example.test/v1/embeddings",
            "RERANK_BASE_URL": "https://example.test/v1",
            "AGENT_BASE_URL": "https://example.test/v1",
            "VECTOR_INDEX_MANIFEST": str(manifest_path),
            "VECTOR_INDEX_ARCHIVE_URL": str(archive_path),
            "VECTOR_INDEX_ARCHIVE_SHA256": str(metadata["archive_sha256"]),
        }
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "health_check.py"),
                "--strict-warnings",
                "--verify-artifacts",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    _assert(
        result.returncode == 0,
        f"artifact-backed strict health should pass, got {result.returncode}\n{result.stdout}\n{result.stderr}",
    )
    _assert(
        "vector data:" in result.stdout and "runtime artifact restore is configured" in result.stdout,
        "health should report artifact-backed vector data readiness",
    )
    _assert(
        "vector artifact restore:" in result.stdout and "verified" in result.stdout,
        "health should verify artifact restore when requested",
    )
    _assert(
        "vector data load:" in result.stdout and "loaded flat index" in result.stdout,
        "health should load the restored vector artifact when requested",
    )


def _assert_corrupt_local_vector_data_fails_health() -> None:
    with tempfile.TemporaryDirectory(prefix="xiao-health-corrupt-") as temp_dir:
        work = Path(temp_dir)
        manifest_path = work / "index" / "test_vectors.json"
        data_path = work / "index" / "test_vectors.f16"
        _write_artifact_backed_manifest(manifest_path, data_path)
        data_path.write_bytes(b"\x00")

        env = {
            **os.environ,
            "VECTOR_INDEX_MANIFEST": str(manifest_path),
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
    _assert(result.returncode == 1, f"corrupt local vector data should fail health, got {result.returncode}")
    _assert("FAIL vector data load:" in result.stdout, "health should report vector data load failure")


def _write_artifact_backed_manifest(manifest_path: Path, data_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = indexable_chunks(load_knowledge_base(), include_field_notes=False)
    data_path.write_bytes(b"\x00\x00" * len(chunks))
    manifest = {
        "schema_version": 1,
        "model": "test-model",
        "dim": 1,
        "backend": "flat",
        "dtype": "float16",
        "count": len(chunks),
        "ids": [chunk.id for chunk in chunks],
        "data_file": data_path.name,
        "source_hash": hash_chunks(chunks),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
