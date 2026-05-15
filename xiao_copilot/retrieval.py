from __future__ import annotations

import math
import re
from collections import Counter
from time import perf_counter

from xiao_copilot.clients import embed_query, embed_texts, rerank
from xiao_copilot.config import Settings
from xiao_copilot.knowledge_base import KnowledgeChunk, load_knowledge_base
from xiao_copilot.vector_index import search_vector_index


TOKEN_RE = re.compile(r"[a-zA-Z0-9_+.-]+")
DEFAULT_RERANK_TEXT_CHARS = 3200
RERANK_WEIGHT = 1.0
INITIAL_RETRIEVAL_WEIGHT = 0.25


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
        diagnostics["vector_index_backend"] = vector_result.backend
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
    top_scored = ranked[: settings.candidate_k]
    top = [chunk for chunk, _score in top_scored]

    rerank_text_chars = max(200, settings.rerank_text_chars or DEFAULT_RERANK_TEXT_CHARS)
    rerank_documents = [_rerank_text(chunk, rerank_text_chars) for chunk in top]
    diagnostics["reranker_candidate_count"] = len(rerank_documents)
    diagnostics["reranker_text_chars"] = rerank_text_chars
    diagnostics["reranker_input_chars"] = {
        "max": max((len(document) for document in rerank_documents), default=0),
        "total": sum(len(document) for document in rerank_documents),
    }
    rerank_started_at = perf_counter()
    rerank_result = rerank(
        base_url=settings.rerank_base_url,
        model=settings.rerank_model,
        query=query,
        documents=rerank_documents,
        api_key=settings.rerank_api_key,
        timeout=settings.request_timeout_seconds,
    )
    diagnostics["reranker_ms"] = _elapsed_ms(rerank_started_at)
    if rerank_result.ok and rerank_result.data:
        diagnostics["reranker_used"] = True
        diagnostics["reranker_mode"] = (rerank_result.meta or {}).get("mode", "unknown")
        if rerank_result.meta and rerank_result.meta.get("native_error"):
            diagnostics["reranker_native_error"] = rerank_result.meta["native_error"]
        hybrid_scores = _hybrid_rerank_scores(rerank_result.data, top_scored)
        diagnostics["reranker_scores"] = [
            {
                "id": top[index].id,
                "score": round(float(score), 4),
                "combined_score": round(float(combined_score), 4),
            }
            for index, score, combined_score in hybrid_scores
            if 0 <= index < len(top)
        ]
        ordered = [
            top[index]
            for index, _score, _combined_score in hybrid_scores
            if 0 <= index < len(top)
        ]
        top = ordered[: settings.top_k]
    else:
        if rerank_result.meta:
            diagnostics["reranker_mode"] = rerank_result.meta.get("mode", "unknown")
        diagnostics["reranker_error"] = rerank_result.error
        top = top[: settings.top_k]

    diagnostics["citations"] = [chunk.id for chunk in top]
    return top, diagnostics


def _rerank_text(chunk: KnowledgeChunk, max_chars: int) -> str:
    text = chunk.text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return f"{chunk.title}\n{chunk.source}\n{text}".strip()


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 1)


def _hybrid_rerank_scores(
    rerank_scores: list[tuple[int, float]],
    initial_scores: list[tuple[KnowledgeChunk, float]],
) -> list[tuple[int, float, float]]:
    raw_initial = [float(score) for _chunk, score in initial_scores]
    if not raw_initial:
        return []
    min_initial = min(raw_initial)
    max_initial = max(raw_initial)
    span = max_initial - min_initial
    combined: list[tuple[int, float, float]] = []
    for index, rerank_score in rerank_scores:
        if index < 0 or index >= len(raw_initial):
            continue
        initial_score = raw_initial[index]
        normalized_initial = (initial_score - min_initial) / span if span > 0 else 0.0
        combined_score = RERANK_WEIGHT * float(rerank_score) + INITIAL_RETRIEVAL_WEIGHT * normalized_initial
        combined.append((index, float(rerank_score), combined_score))
    return sorted(combined, key=lambda item: item[2], reverse=True)


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
            # Curated facts can be absent from the ANN top-N when a query is
            # phrased as a broad capability choice. Keep strong lexical matches
            # competitive so exact board facts are not buried by generic wiki pages.
            score = min(lexical_score, 2.5) * 0.45
        else:
            score = vector_score + min(lexical_score, 2.5) * 0.12
        score += _board_hint_score(query, chunk) * 0.15
        score += _kind_hint_score(query, chunk) * 0.05
        score += _capability_hint_score(query, chunk) * 0.45
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


def _capability_hint_score(query: str, chunk: KnowledgeChunk) -> float:
    q = query.lower()
    text = chunk.search_text.lower()
    score = 0.0

    image_query = any(term in q for term in ("image", "vision", "camera", "photo"))
    audio_query = any(term in q for term in ("audio", "microphone", "mic", "speech", "keyword"))
    query_tokens = set(_tokens(q))
    tinyml_query = (
        any(term in q for term in ("tinyml", "machine learning", "edge impulse", "classification"))
        or "ml" in query_tokens
    )
    choice_query = any(term in q for term in ("which", "choose", "pick", "best", "should"))
    board_choice_query = any(
        term in q
        for term in ("which supported", "which xiao", "which board", "choose", "pick", "best board", "board should")
    )
    if not (image_query or audio_query or tinyml_query or choice_query):
        return 0.0

    image_match = any(term in text for term in ("image", "vision", "camera", "photo"))
    audio_match = any(term in text for term in ("audio", "microphone", "mic", "pdm", "speech", "keyword"))
    tinyml_match = any(
        term in text
        for term in ("tinyml", "machine learning", "embedded ml", "edge impulse", "classification")
    )

    if image_query and image_match:
        score += 0.4
    if audio_query and audio_match:
        score += 0.4
    if tinyml_query and tinyml_match:
        score += 0.25
    if "imu" in query_tokens and "imu" in text:
        score += 0.3
    if any(term in q for term in ("package", "library")) and (
        "mbed-enabled" in text or "board package" in text
    ):
        score += 0.7
    if image_query and audio_query and image_match and audio_match:
        score += 0.8
        if board_choice_query:
            if chunk.board_id == "xiao-esp32s3":
                score += 0.6
            if "xiao esp32s3 sense" in text:
                score += 0.4
            if tinyml_query and "speech recognition" in text and "image processing" in text:
                score += 0.3
    if chunk.kind == "vision":
        score += 0.5
    elif chunk.kind == "identity" and choice_query:
        score += 0.1

    return score


def _normalize_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
