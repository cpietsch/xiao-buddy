from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.clients import embed_texts
from xiao_copilot.config import load_settings
from xiao_copilot.knowledge_base import KnowledgeChunk, load_knowledge_base
from xiao_copilot.vector_index import configured_index_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local vector index for the XIAO wiki RAG corpus."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Limit chunks for a quick smoke build.")
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

    manifest_path, data_path = configured_index_paths(args.manifest, args.data)
    chunks = indexable_chunks(load_knowledge_base(), include_field_notes=args.include_field_notes)
    if args.limit:
        chunks = chunks[: args.limit]
    if not chunks:
        raise SystemExit("No chunks available to index.")

    source_hash = hash_chunks(chunks)
    print(f"Building vector index for {len(chunks)} chunks")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Source hash: {source_hash}")

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
        print(f"  embedded {done}/{len(chunks)} chunks ({rate:.1f} chunks/s)")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_data = data_path.with_suffix(data_path.suffix + ".tmp")
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    with tmp_data.open("wb") as handle:
        for start in range(0, len(vectors), 65_536):
            block = vectors[start : start + 65_536]
            handle.write(struct.pack(f"<{len(block)}e", *block))

    manifest = {
        "schema_version": 1,
        "model": settings.embedding_model,
        "dim": dim,
        "dtype": "float16",
        "count": len(ids),
        "ids": ids,
        "source_hash": source_hash,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp_data.replace(data_path)
    tmp_manifest.replace(manifest_path)

    mb = data_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(ids)} x {dim} float16 vectors to {data_path} ({mb:.1f} MiB)")
    print(f"Saved manifest to {manifest_path}")


def indexable_chunks(
    chunks: list[KnowledgeChunk],
    *,
    include_field_notes: bool,
) -> list[KnowledgeChunk]:
    allowed = {"identity", "pinout", "gotchas", "support", "wiki"}
    if include_field_notes:
        allowed.add("note")
    return [chunk for chunk in chunks if chunk.kind in allowed]


def hash_chunks(chunks: list[KnowledgeChunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.id.encode())
        digest.update(b"\0")
        digest.update(chunk.search_text.encode())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return [0.0 for _value in vector]
    return [float(value) / norm for value in vector]


if __name__ == "__main__":
    main()
