from __future__ import annotations

import array
import json
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from xiao_copilot.knowledge_base import KnowledgeChunk


_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VECTOR_MANIFEST = _ROOT / "data" / "index" / "xiao_vectors.json"
DEFAULT_VECTOR_DATA = _ROOT / "data" / "index" / "xiao_vectors.f16"


@dataclass(frozen=True)
class VectorMatch:
    chunk: KnowledgeChunk
    score: float


@dataclass(frozen=True)
class VectorSearchResult:
    ok: bool
    matches: list[VectorMatch]
    total_vectors: int = 0
    dim: int = 0
    error: str = ""


@dataclass(frozen=True)
class VectorIndex:
    ids: list[str]
    dim: int
    model: str
    source_hash: str
    vectors: array.array

    @property
    def count(self) -> int:
        return len(self.ids)


def configured_index_paths(
    manifest_path: str = "",
    data_path: str = "",
) -> tuple[Path, Path]:
    manifest = Path(manifest_path).expanduser() if manifest_path else DEFAULT_VECTOR_MANIFEST
    data = Path(data_path).expanduser() if data_path else DEFAULT_VECTOR_DATA
    if not manifest.is_absolute():
        manifest = _ROOT / manifest
    if not data.is_absolute():
        data = _ROOT / data
    return manifest, data


def search_vector_index(
    query_vector: list[float],
    chunks_by_id: dict[str, KnowledgeChunk],
    *,
    manifest_path: str = "",
    data_path: str = "",
    limit: int = 96,
) -> VectorSearchResult:
    manifest, data = configured_index_paths(manifest_path, data_path)
    try:
        index = load_vector_index(str(manifest), str(data))
    except Exception as exc:  # noqa: BLE001 - returned as retrieval diagnostics.
        return VectorSearchResult(ok=False, matches=[], error=str(exc))

    if index.dim <= 0 or len(query_vector) != index.dim:
        return VectorSearchResult(
            ok=False,
            matches=[],
            total_vectors=index.count,
            dim=index.dim,
            error=f"Query dim {len(query_vector)} does not match index dim {index.dim}.",
        )

    normalized_query = _normalize(query_vector)
    scored: list[tuple[str, float]] = []
    for row, chunk_id in enumerate(index.ids):
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        score = _dot_at(index.vectors, row * index.dim, normalized_query)
        scored.append((chunk_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    matches = [
        VectorMatch(chunk=chunks_by_id[chunk_id], score=score)
        for chunk_id, score in scored[:limit]
    ]
    return VectorSearchResult(
        ok=True,
        matches=matches,
        total_vectors=index.count,
        dim=index.dim,
    )


@lru_cache(maxsize=2)
def load_vector_index(manifest_path: str, data_path: str) -> VectorIndex:
    manifest = Path(manifest_path)
    data = Path(data_path)
    if not manifest.exists() or not data.exists():
        raise FileNotFoundError(
            f"Vector index not found. Expected {manifest} and {data}. "
            "Run scripts/build_wiki_vector_index.py first."
        )

    meta = json.loads(manifest.read_text(encoding="utf-8"))
    ids = list(meta["ids"])
    dim = int(meta["dim"])
    vectors = _read_vectors(data, len(ids) * dim, str(meta.get("dtype", "float16")))

    if len(vectors) != len(ids) * dim:
        raise ValueError(
            f"Vector file has {len(vectors)} floats, expected {len(ids) * dim}."
        )

    return VectorIndex(
        ids=ids,
        dim=dim,
        model=str(meta.get("model", "")),
        source_hash=str(meta.get("source_hash", "")),
        vectors=vectors,
    )


def _read_vectors(data_path: Path, expected_floats: int, dtype: str) -> array.array:
    vectors = array.array("f")
    if dtype == "float32":
        with data_path.open("rb") as handle:
            vectors.fromfile(handle, expected_floats)
        return vectors
    if dtype != "float16":
        raise ValueError(f"Unsupported vector index dtype: {dtype}")

    expected_bytes = expected_floats * 2
    if data_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Vector file has {data_path.stat().st_size} bytes, expected {expected_bytes}."
        )

    with data_path.open("rb") as handle:
        while block := handle.read(1_048_576):
            vectors.extend(value for (value,) in struct.iter_unpack("<e", block))
    return vectors


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _value in vector]
    return [float(value) / norm for value in vector]


def _dot_at(vectors: array.array, start: int, query: list[float]) -> float:
    total = 0.0
    for offset, value in enumerate(query):
        total += vectors[start + offset] * value
    return total


def manifest_summary(manifest_path: str = "", data_path: str = "") -> dict[str, Any]:
    manifest, data = configured_index_paths(manifest_path, data_path)
    if not manifest.exists() or not data.exists():
        return {"available": False}
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "available": True,
        "count": len(meta.get("ids", [])),
        "dim": meta.get("dim"),
        "model": meta.get("model"),
        "source_hash": meta.get("source_hash"),
    }
