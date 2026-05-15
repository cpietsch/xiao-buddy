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
DEFAULT_HNSW_DATA = _ROOT / "data" / "index" / "xiao_vectors.hnsw"
DEFAULT_FAISS_DATA = _ROOT / "data" / "index" / "xiao_vectors.faiss"


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
    backend: str = ""
    error: str = ""


@dataclass(frozen=True)
class VectorIndex:
    ids: list[str]
    dim: int
    model: str
    source_hash: str
    backend: str = "flat"
    vectors: array.array | None = None
    hnsw: Any | None = None
    faiss: Any | None = None
    ef_search: int = 96

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
        index = load_vector_index(str(manifest), str(data), bool(data_path))
    except Exception as exc:  # noqa: BLE001 - returned as retrieval diagnostics.
        return VectorSearchResult(ok=False, matches=[], error=str(exc))

    if index.dim <= 0 or len(query_vector) != index.dim:
        return VectorSearchResult(
            ok=False,
            matches=[],
            total_vectors=index.count,
            dim=index.dim,
            backend=index.backend,
            error=f"Query dim {len(query_vector)} does not match index dim {index.dim}.",
        )

    normalized_query = _normalize(query_vector)
    if index.backend == "hnsw":
        return _search_hnsw_index(index, normalized_query, chunks_by_id, limit)

    if index.backend == "faiss_pq":
        return _search_faiss_index(index, normalized_query, chunks_by_id, limit)

    if index.vectors is None:
        return VectorSearchResult(
            ok=False,
            matches=[],
            total_vectors=index.count,
            dim=index.dim,
            backend=index.backend,
            error=f"Vector backend '{index.backend}' did not load vectors.",
        )

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
        backend=index.backend,
    )


@lru_cache(maxsize=2)
def load_vector_index(
    manifest_path: str,
    data_path: str,
    data_path_explicit: bool = False,
) -> VectorIndex:
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(
            f"Vector index manifest not found. Expected {manifest}. "
            "Run scripts/build_wiki_vector_index.py first."
        )

    meta = json.loads(manifest.read_text(encoding="utf-8"))
    ids = list(meta["ids"])
    dim = int(meta["dim"])
    backend = _index_backend(meta)
    data = _resolve_data_path(meta, manifest, Path(data_path), data_path_explicit)
    if not data.exists():
        raise FileNotFoundError(
            f"Vector index data not found. Expected {data}. "
            "Run scripts/build_wiki_vector_index.py first."
        )

    if backend == "hnsw":
        hnsw_index = _load_hnsw(data, dim, str(meta.get("space", "cosine")))
        ef_search = int(meta.get("ef_search", max(96, min(512, len(ids)))))
        hnsw_index.set_ef(min(max(ef_search, 1), max(len(ids), 1)))
        return VectorIndex(
            ids=ids,
            dim=dim,
            model=str(meta.get("model", "")),
            source_hash=str(meta.get("source_hash", "")),
            backend=backend,
            hnsw=hnsw_index,
            ef_search=ef_search,
        )

    if backend == "faiss_pq":
        faiss_index = _load_faiss(data)
        return VectorIndex(
            ids=ids,
            dim=dim,
            model=str(meta.get("model", "")),
            source_hash=str(meta.get("source_hash", "")),
            backend=backend,
            faiss=faiss_index,
        )

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
        backend=backend,
        vectors=vectors,
    )


def _index_backend(meta: dict[str, Any]) -> str:
    backend = meta.get("backend") or meta.get("index_backend") or "flat"
    if isinstance(backend, dict):
        backend = backend.get("name", "flat")
    backend = str(backend).replace("-", "_")
    if backend not in {"flat", "hnsw", "faiss_pq"}:
        raise ValueError(f"Unsupported vector index backend: {backend}")
    return backend


def _resolve_data_path(
    meta: dict[str, Any],
    manifest_path: Path,
    data_path: Path,
    data_path_explicit: bool,
) -> Path:
    if data_path_explicit:
        return data_path
    data_file = meta.get("data_file")
    if data_file:
        path = Path(str(data_file))
        return path if path.is_absolute() else manifest_path.parent / path
    return data_path


def default_data_path_for_backend(backend: str) -> Path:
    backend = backend.replace("-", "_")
    if backend == "hnsw":
        return DEFAULT_HNSW_DATA
    if backend == "faiss_pq":
        return DEFAULT_FAISS_DATA
    return DEFAULT_VECTOR_DATA


def _load_hnsw(data_path: Path, dim: int, space: str):
    try:
        import hnswlib  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise ImportError(
            "hnswlib is required for HNSW vector indexes. "
            "Install it with `pip install hnswlib` or rebuild with `--backend flat`."
        ) from exc

    index = hnswlib.Index(space=space, dim=dim)
    index.load_index(str(data_path))
    return index


def _load_faiss(data_path: Path):
    try:
        import faiss  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise ImportError(
            "faiss-cpu is required for FAISS-PQ vector indexes. "
            "Install it with `pip install faiss-cpu` or rebuild with `--backend hnsw`."
        ) from exc

    return faiss.read_index(str(data_path))


def _search_hnsw_index(
    index: VectorIndex,
    normalized_query: list[float],
    chunks_by_id: dict[str, KnowledgeChunk],
    limit: int,
) -> VectorSearchResult:
    if index.hnsw is None:
        return VectorSearchResult(
            ok=False,
            matches=[],
            total_vectors=index.count,
            dim=index.dim,
            backend=index.backend,
            error="HNSW index was not loaded.",
        )

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - hnswlib needs numpy too.
        return VectorSearchResult(
            ok=False,
            matches=[],
            total_vectors=index.count,
            dim=index.dim,
            backend=index.backend,
            error=f"numpy is required for HNSW search: {exc}",
        )

    k = min(max(limit, 1), index.count)
    labels, distances = index.hnsw.knn_query(
        np.asarray([normalized_query], dtype=np.float32),
        k=k,
    )
    matches: list[VectorMatch] = []
    for row, distance in zip(labels[0], distances[0], strict=True):
        chunk_id = index.ids[int(row)]
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        # hnswlib's cosine space returns 1 - cosine similarity.
        matches.append(VectorMatch(chunk=chunk, score=1.0 - float(distance)))
    return VectorSearchResult(
        ok=True,
        matches=matches,
        total_vectors=index.count,
        dim=index.dim,
        backend=index.backend,
    )


def _search_faiss_index(
    index: VectorIndex,
    normalized_query: list[float],
    chunks_by_id: dict[str, KnowledgeChunk],
    limit: int,
) -> VectorSearchResult:
    if index.faiss is None:
        return VectorSearchResult(
            ok=False,
            matches=[],
            total_vectors=index.count,
            dim=index.dim,
            backend=index.backend,
            error="FAISS index was not loaded.",
        )

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - FAISS search needs numpy arrays.
        return VectorSearchResult(
            ok=False,
            matches=[],
            total_vectors=index.count,
            dim=index.dim,
            backend=index.backend,
            error=f"numpy is required for FAISS search: {exc}",
        )

    k = min(max(limit, 1), index.count)
    scores, rows = index.faiss.search(np.asarray([normalized_query], dtype=np.float32), k)
    matches: list[VectorMatch] = []
    for row, score in zip(rows[0], scores[0], strict=True):
        if row < 0:
            continue
        chunk_id = index.ids[int(row)]
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        matches.append(VectorMatch(chunk=chunk, score=float(score)))
    return VectorSearchResult(
        ok=True,
        matches=matches,
        total_vectors=index.count,
        dim=index.dim,
        backend=index.backend,
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
        if not manifest.exists():
            return {"available": False}
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    backend = _index_backend(meta)
    resolved_data = _resolve_data_path(meta, manifest, data, bool(data_path))
    if not resolved_data.exists():
        return {"available": False, "backend": backend}
    return {
        "available": True,
        "backend": backend,
        "count": len(meta.get("ids", [])),
        "dim": meta.get("dim"),
        "model": meta.get("model"),
        "source_hash": meta.get("source_hash"),
        "data_file": str(resolved_data),
    }
