from __future__ import annotations

import hashlib

from xiao_copilot.knowledge_base import KnowledgeChunk


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
