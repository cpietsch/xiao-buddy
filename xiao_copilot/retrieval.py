from __future__ import annotations

import math
import re
from collections import Counter

from xiao_copilot.clients import embed_query, embed_texts, rerank
from xiao_copilot.config import Settings
from xiao_copilot.knowledge_base import KnowledgeChunk, load_knowledge_base
from xiao_copilot.vector_index import search_vector_index


TOKEN_RE = re.compile(r"[a-zA-Z0-9_+.-]+")


def retrieve(
    query: str,
    settings: Settings,
    image_data_url: str | None = None,
) -> tuple[list[KnowledgeChunk], dict[str, object]]:
    chunks = load_knowledge_base()
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    lexical_ranked = _lexical_prefilter(query, chunks, settings)
    embedding_pool_size = max(settings.vector_candidate_k, settings.candidate_k * 12, 72)
    embedding_pool = [chunk for chunk, _score in lexical_ranked[:embedding_pool_size]]
    diagnostics: dict[str, object] = {
        "embedding_used": False,
        "multimodal_query": bool(image_data_url),
        "corpus_chunks": len(chunks),
        "embedding_pool_chunks": len(embedding_pool),
        "vector_index_used": False,
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

    if query_embedding.ok:
        vector_result = search_vector_index(
            query_embedding.data,
            chunks_by_id,
            manifest_path=settings.vector_index_manifest,
            data_path=settings.vector_index_data,
            limit=embedding_pool_size,
        )
    else:
        vector_result = None

    if query_embedding.ok and vector_result and vector_result.ok:
        diagnostics["embedding_used"] = True
        diagnostics["vector_index_used"] = True
        diagnostics["vector_index_chunks"] = vector_result.total_vectors
        diagnostics["vector_index_dim"] = vector_result.dim
        scored = _merge_vector_and_lexical_scores(query, vector_result.matches, lexical_ranked)
    else:
        if not query_embedding.ok:
            diagnostics["embedding_error"] = query_embedding.error
        elif vector_result and vector_result.error:
            diagnostics["vector_index_error"] = vector_result.error
        document_embeddings = embed_texts(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            texts=[chunk.search_text for chunk in embedding_pool],
            api_key=settings.embedding_api_key,
            timeout=settings.request_timeout_seconds,
        )

        if query_embedding.ok and document_embeddings.ok:
            diagnostics["embedding_used"] = True
            diagnostics["embedding_fallback"] = "per_query_candidate_embeddings"
            scored = [
                (
                    chunk,
                    _cosine(query_embedding.data, vector)
                    + _board_hint_score(query, chunk) * 0.15
                    + _kind_hint_score(query, chunk) * 0.05,
                )
                for chunk, vector in zip(embedding_pool, document_embeddings.data, strict=True)
            ]
        else:
            diagnostics["embedding_error"] = query_embedding.error or document_embeddings.error
            scored = lexical_ranked

    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    diagnostics["initial_top"] = [
        {"id": chunk.id, "score": round(float(score), 4)}
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


def _merge_vector_and_lexical_scores(
    query: str,
    vector_matches,
    lexical_ranked: list[tuple[KnowledgeChunk, float]],
) -> list[tuple[KnowledgeChunk, float]]:
    lexical_by_id = {chunk.id: score for chunk, score in lexical_ranked}
    candidates: dict[str, KnowledgeChunk] = {}
    vector_by_id: dict[str, float] = {}

    for match in vector_matches:
        candidates[match.chunk.id] = match.chunk
        vector_by_id[match.chunk.id] = match.score

    for chunk, _score in lexical_ranked[: len(vector_matches)]:
        candidates.setdefault(chunk.id, chunk)

    scored: list[tuple[KnowledgeChunk, float]] = []
    for chunk in candidates.values():
        vector_score = vector_by_id.get(chunk.id)
        lexical_score = lexical_by_id.get(chunk.id, 0.0)
        if vector_score is None:
            score = min(lexical_score, 2.5) * 0.25
        else:
            score = vector_score + min(lexical_score, 2.5) * 0.12
        score += _board_hint_score(query, chunk) * 0.15
        score += _kind_hint_score(query, chunk) * 0.05
        scored.append((chunk, score))

    return scored


def _lexical_prefilter(
    query: str,
    chunks: list[KnowledgeChunk],
    settings: Settings,
) -> list[tuple[KnowledgeChunk, float]]:
    minimum = max(settings.candidate_k * 12, settings.top_k * 12, 72)
    scored = [
        (
            chunk,
            _lexical_score(query, chunk.search_text)
            + _board_hint_score(query, chunk)
            + _kind_hint_score(query, chunk),
        )
        for chunk in chunks
    ]
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    if len(ranked) <= minimum:
        return ranked

    positive = [item for item in ranked if item[1] > 0]
    if len(positive) >= minimum:
        return positive

    seen_ids = {chunk.id for chunk, _score in positive}
    padded = positive[:]
    for chunk, score in ranked:
        if chunk.id in seen_ids:
            continue
        padded.append((chunk, score))
        if len(padded) >= minimum:
            break
    return padded


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


def _kind_hint_score(query: str, chunk: KnowledgeChunk) -> float:
    q = query.lower()
    score = 0.0

    # Keep the hand-curated board facts prominent. The imported wiki gives
    # breadth, while these chunks are the compact facts we trust most for demos.
    if chunk.kind != "wiki":
        score += 0.2

    if any(term in q for term in ("pin", "pins", "i2c", "spi", "uart", "gpio", "adc", "dac")):
        if chunk.kind == "pinout":
            score += 0.8
    if any(term in q for term in ("compare", "which", "choose", "pick", "wifi", "wireless", "camera")):
        if chunk.kind == "identity":
            score += 0.35
    if any(term in q for term in ("not detected", "not responding", "fails", "error", "reset", "brownout", "boot")):
        if chunk.kind in {"gotchas", "support", "note"}:
            score += 0.35

    return score


def _normalize_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
