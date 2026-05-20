from __future__ import annotations

import argparse
import array
import json
import math
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.clients import embed_texts
from xiao_copilot.config import load_settings
from xiao_copilot.index_corpus import hash_chunks, indexable_chunks
from xiao_copilot.knowledge_base import load_knowledge_base
from xiao_copilot.vector_index import configured_index_paths


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local vector index for the Seeed wiki RAG corpus."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Limit chunks for a quick smoke build.")
    parser.add_argument(
        "--backend",
        choices=["flat", "hnsw", "faiss-pq"],
        default="hnsw",
        help="Index backend. HNSW is the default for larger wiki-scale retrieval.",
    )
    parser.add_argument("--hnsw-m", type=int, default=32, help="HNSW graph degree.")
    parser.add_argument(
        "--hnsw-ef-construction",
        type=int,
        default=200,
        help="HNSW build-time recall/speed setting.",
    )
    parser.add_argument(
        "--hnsw-ef-search",
        type=int,
        default=96,
        help="HNSW query-time recall/speed setting stored in the manifest.",
    )
    parser.add_argument(
        "--pq-m",
        type=int,
        default=64,
        help="FAISS-PQ subquantizers. Must divide the embedding dimension.",
    )
    parser.add_argument("--pq-nbits", type=int, default=8, help="FAISS-PQ bits per subquantizer.")
    parser.add_argument(
        "--include-field-notes",
        action="store_true",
        help="Also embed local field notes. Curated board facts and wiki chunks are always included.",
    )
    parser.add_argument("--manifest", default="")
    parser.add_argument("--data", default="")
    args = parser.parse_args()

    settings = load_settings()
    if not settings.embedding_base_url:
        raise SystemExit("EMBEDDING_BASE_URL is required to build the vector index.")

    backend = args.backend.replace("-", "_")
    manifest_path, configured_data_path = configured_index_paths(args.manifest, args.data)
    data_path = configured_data_path if args.data else default_data_path(manifest_path, backend)
    chunks = indexable_chunks(load_knowledge_base(), include_field_notes=args.include_field_notes)
    if args.limit:
        chunks = chunks[: args.limit]
    if not chunks:
        raise SystemExit("No chunks available to index.")

    source_hash = hash_chunks(chunks)
    log(f"Building vector index for {len(chunks)} chunks")
    log(f"Embedding model: {settings.embedding_model}")
    log(f"Index backend: {backend}")
    log(f"Source hash: {source_hash}")

    vectors = array.array("f")
    ids: list[str] = []
    dim = 0
    started = time.time()

    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start : start + args.batch_size]
        result = embed_texts(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            texts=[chunk.search_text for chunk in batch],
            api_key=settings.embedding_api_key,
            timeout=settings.request_timeout_seconds,
        )
        if not result.ok:
            raise SystemExit(f"Embedding batch {start} failed: {result.error}")

        for chunk, vector in zip(batch, result.data, strict=True):
            normalized = normalize(vector)
            if dim == 0:
                dim = len(normalized)
            elif len(normalized) != dim:
                raise SystemExit(
                    f"Embedding dimension changed from {dim} to {len(normalized)} at {chunk.id}."
                )
            ids.append(chunk.id)
            vectors.extend(normalized)

        done = min(start + len(batch), len(chunks))
        elapsed = time.time() - started
        rate = done / elapsed if elapsed else 0
        log(f"  embedded {done}/{len(chunks)} chunks ({rate:.1f} chunks/s)")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_data = data_path.with_suffix(data_path.suffix + ".tmp")
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    if tmp_data.exists():
        tmp_data.unlink()

    log(f"Writing {backend} index data to {data_path}")
    backend_meta = write_index_data(
        backend=backend,
        data_path=tmp_data,
        vectors=vectors,
        count=len(ids),
        dim=dim,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
        pq_m=args.pq_m,
        pq_nbits=args.pq_nbits,
    )

    manifest = {
        "schema_version": 1,
        "model": settings.embedding_model,
        "dim": dim,
        "backend": backend,
        "count": len(ids),
        "ids": ids,
        "data_file": data_path.name,
        "source_hash": source_hash,
        "include_field_notes": bool(args.include_field_notes),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **backend_meta,
    }
    tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_data.replace(data_path)
    tmp_manifest.replace(manifest_path)

    mb = data_path.stat().st_size / (1024 * 1024)
    log(f"Saved {len(ids)} x {dim} {backend} index to {data_path} ({mb:.1f} MiB)")
    log(f"Saved manifest to {manifest_path}")


def default_data_path(manifest_path: Path, backend: str) -> Path:
    if backend == "hnsw":
        return manifest_path.with_suffix(".hnsw")
    if backend == "faiss_pq":
        return manifest_path.with_suffix(".faiss")
    return manifest_path.with_suffix(".f16")


def write_index_data(
    *,
    backend: str,
    data_path: Path,
    vectors: array.array,
    count: int,
    dim: int,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
    pq_m: int,
    pq_nbits: int,
) -> dict[str, object]:
    if backend == "flat":
        write_flat_vectors(data_path, vectors)
        return {"dtype": "float16"}

    if backend == "hnsw":
        return write_hnsw_index(
            data_path=data_path,
            vectors=vectors,
            count=count,
            dim=dim,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_ef_search=hnsw_ef_search,
        )

    if backend == "faiss_pq":
        return write_faiss_pq_index(
            data_path=data_path,
            vectors=vectors,
            count=count,
            dim=dim,
            pq_m=pq_m,
            pq_nbits=pq_nbits,
        )

    raise SystemExit(f"Unsupported backend: {backend}")


def write_flat_vectors(data_path: Path, vectors: array.array) -> None:
    with data_path.open("wb") as handle:
        for start in range(0, len(vectors), 65_536):
            block = vectors[start : start + 65_536]
            handle.write(struct.pack(f"<{len(block)}e", *block))


def write_hnsw_index(
    *,
    data_path: Path,
    vectors: array.array,
    count: int,
    dim: int,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
) -> dict[str, object]:
    try:
        import hnswlib  # type: ignore
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "The HNSW backend requires hnswlib and numpy. "
            "Install with `pip install hnswlib numpy`, or run with `--backend flat`."
        ) from exc

    matrix = np.asarray(vectors, dtype=np.float32).reshape(count, dim)
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(
        max_elements=count,
        ef_construction=hnsw_ef_construction,
        M=hnsw_m,
    )
    index.add_items(matrix, np.arange(count))
    index.set_ef(min(max(hnsw_ef_search, 1), max(count, 1)))
    index.save_index(str(data_path))
    return {
        "space": "cosine",
        "hnsw_m": hnsw_m,
        "hnsw_ef_construction": hnsw_ef_construction,
        "ef_search": hnsw_ef_search,
    }


def write_faiss_pq_index(
    *,
    data_path: Path,
    vectors: array.array,
    count: int,
    dim: int,
    pq_m: int,
    pq_nbits: int,
) -> dict[str, object]:
    try:
        import faiss  # type: ignore
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "The FAISS-PQ backend requires faiss-cpu and numpy. "
            "Install with `pip install faiss-cpu numpy`, or run with `--backend hnsw`."
        ) from exc

    if dim % pq_m != 0:
        raise SystemExit(f"--pq-m={pq_m} must divide embedding dimension {dim}.")
    if count < 2**pq_nbits:
        raise SystemExit(
            f"FAISS-PQ with {pq_nbits} bits needs at least {2**pq_nbits} vectors to train."
        )

    matrix = np.asarray(vectors, dtype=np.float32).reshape(count, dim)
    index = faiss.IndexPQ(dim, pq_m, pq_nbits, faiss.METRIC_INNER_PRODUCT)
    index.train(matrix)
    index.add(matrix)
    faiss.write_index(index, str(data_path))
    return {
        "space": "inner_product",
        "pq_m": pq_m,
        "pq_nbits": pq_nbits,
    }


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _value in vector]
    return [float(value) / norm for value in vector]


if __name__ == "__main__":
    main()
