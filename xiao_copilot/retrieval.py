from __future__ import annotations

import math
import re
from collections import Counter

from xiao_copilot.clients import embed_query, embed_texts, rerank
from xiao_copilot.config import Settings
from xiao_copilot.knowledge_base import KnowledgeChunk, load_knowledge_base


TOKEN_RE = re.compile(r"[a-zA-Z0-9_+.-]+")


def retrieve(
    query: str,
    settings: Settings,
    image_data_url: str | None = None,
) -> tuple[list[KnowledgeChunk], dict[str, object]]:
    chunks = load_knowledge_base()
    diagnostics: dict[str, object] = {
        "embedding_endpoint": settings.embedding_base_url,
        "embedding_used": False,
        "multimodal_query": bool(image_data_url),
        "reranker_used": False,
    }

    query_embedding = embed_query(
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        text=query,
        image_data_url=image_data_url,
        api_key=settings.embedding_api_key,
        timeout=settings.request_timeout_seconds,
    )
    document_embeddings = embed_texts(
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        texts=[chunk.search_text for chunk in chunks],
        api_key=settings.embedding_api_key,
        timeout=settings.request_timeout_seconds,
    )

    if query_embedding.ok and document_embeddings.ok:
        diagnostics["embedding_used"] = True
        scored = [
            (chunk, _cosine(query_embedding.data, vector) + _board_hint_score(query, chunk) * 0.15)
            for chunk, vector in zip(chunks, document_embeddings.data, strict=True)
        ]
    else:
        diagnostics["embedding_error"] = query_embedding.error or document_embeddings.error
        scored = [
            (chunk, _lexical_score(query, chunk.search_text) + _board_hint_score(query, chunk))
            for chunk in chunks
        ]

    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    diagnostics["initial_top"] = [
        {"id": chunk.id, "score": round(float(score), 4), "source": chunk.source}
        for chunk, score in ranked[: settings.candidate_k]
    ]
    top = [chunk for chunk, _score in ranked[: settings.candidate_k]]

    rerank_result = rerank(
        base_url=settings.rerank_base_url,
        model=settings.rerank_model,
        query=query,
        documents=[chunk.search_text for chunk in top],
        api_key=settings.rerank_api_key,
        timeout=settings.request_timeout_seconds,
    )
    if rerank_result.ok and rerank_result.data:
        diagnostics["reranker_used"] = True
        ordered = [top[index] for index, _score in sorted(rerank_result.data, key=lambda item: item[1], reverse=True)]
        top = ordered[: settings.top_k]
    else:
        diagnostics["reranker_error"] = rerank_result.error
        top = top[: settings.top_k]

    diagnostics["citations"] = [chunk.id for chunk in top]
    return top, diagnostics


def _cosine(a: list[float], b: list[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)


def _lexical_score(query: str, text: str) -> float:
    query_terms = Counter(_tokens(query))
    text_terms = Counter(_tokens(text))
    if not query_terms:
        return 0.0
    return sum(min(count, text_terms.get(term, 0)) for term, count in query_terms.items()) / len(query_terms)


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _board_hint_score(query: str, chunk: KnowledgeChunk) -> float:
    normalized_query = _normalize_id(query)
    hints = [chunk.board_id, *chunk.metadata.get("aliases", [])]
    for hint in hints:
        normalized_hint = _normalize_id(hint)
        if normalized_hint and normalized_hint in normalized_query:
            return 1.0
    return 0.0


def _normalize_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
