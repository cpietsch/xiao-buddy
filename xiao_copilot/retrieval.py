from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

from xiao_copilot.clients import embed_query, embed_texts, rerank
from xiao_copilot.config import Settings
from xiao_copilot.knowledge_base import KnowledgeChunk, load_knowledge_base
from xiao_copilot.vector_index import configured_index_paths, load_vector_index, search_vector_index


TOKEN_RE = re.compile(r"[a-zA-Z0-9_+.-]+")
DEFAULT_RERANK_TEXT_CHARS = 3200
RERANK_METADATA_TAG_LIMIT = 16
RERANK_WEIGHT = 1.0
INITIAL_RETRIEVAL_WEIGHT = 0.25
SOURCE_CONTEXT_SIBLINGS = 4
LEXICAL_RECALL_SCAN = 1000
LEXICAL_RECALL_SLOTS = 6
COMMON_QUERY_TERMS = {
    "about",
    "after",
    "and",
    "are",
    "between",
    "can",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "lower",
    "mode",
    "need",
    "not",
    "the",
    "through",
    "to",
    "used",
    "what",
    "when",
    "which",
    "with",
}
TECHNICAL_QUERY_TERMS = {
    "adc",
    "ble",
    "bluetooth",
    "boot",
    "camera",
    "comparison",
    "current",
    "deploy",
    "firmware",
    "gpio",
    "gpios",
    "i2c",
    "lora",
    "matter",
    "micropython",
    "model",
    "mqtt",
    "power",
    "sensecraft",
    "slot",
    "spi",
    "table",
    "thread",
    "uart",
    "uf2",
    "wifi",
    "zigbee",
}


@dataclass(frozen=True)
class _QueryProfile:
    text: str
    lower: str
    normalized: str
    token_counts: Counter[str]
    token_set: frozenset[str]
    distinctive_terms: tuple[str, ...]
    technical_terms: tuple[str, ...]


_CHUNK_SEARCH_TEXT_CACHE: dict[tuple[object, ...], str] = {}
_CHUNK_SEARCH_LOWER_CACHE: dict[tuple[object, ...], str] = {}
_CHUNK_TEXT_LOWER_CACHE: dict[tuple[object, ...], str] = {}
_CHUNK_TOKEN_COUNTS_CACHE: dict[tuple[object, ...], Counter[str]] = {}
_CHUNK_TOKEN_SET_CACHE: dict[tuple[object, ...], frozenset[str]] = {}


def warm_retrieval_caches(settings: Settings | None = None) -> dict[str, object]:
    started_at = perf_counter()
    chunks = load_knowledge_base()
    lexical_started_at = perf_counter()
    for chunk in chunks:
        _chunk_search_text(chunk)
        _chunk_search_lower(chunk)
        _chunk_title_text_lower(chunk)
        _chunk_token_counts(chunk)
        _chunk_token_set(chunk)
    diagnostics: dict[str, object] = {
        "chunks": len(chunks),
        "lexical_cache_ms": _elapsed_ms(lexical_started_at),
    }
    if settings is not None:
        vector_started_at = perf_counter()
        try:
            manifest, data = configured_index_paths(
                settings.vector_index_manifest,
                settings.vector_index_data,
            )
            index = load_vector_index(str(manifest), str(data), bool(settings.vector_index_data))
            diagnostics["vector_index_backend"] = index.backend
            diagnostics["vector_index_chunks"] = index.count
            diagnostics["vector_index_dim"] = index.dim
        except Exception as exc:  # noqa: BLE001 - surfaced as startup diagnostics only.
            diagnostics["vector_index_error"] = str(exc)
        diagnostics["vector_index_ms"] = _elapsed_ms(vector_started_at)
    diagnostics["total_ms"] = _elapsed_ms(started_at)
    return diagnostics


def retrieve(
    query: str,
    settings: Settings,
    image_data_url: str | None = None,
) -> tuple[list[KnowledgeChunk], dict[str, object]]:
    retrieval_started_at = perf_counter()
    timing_ms: dict[str, float] = {}
    load_started_at = perf_counter()
    chunks = load_knowledge_base()
    timing_ms["knowledge_load"] = _elapsed_ms(load_started_at)
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    lexical_started_at = perf_counter()
    lexical_ranked = _lexical_prefilter(query, chunks, settings)
    timing_ms["lexical_prefilter"] = _elapsed_ms(lexical_started_at)
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

    query_embedding_started_at = perf_counter()
    query_embedding = embed_query(
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        text=query,
        image_data_url=image_data_url,
        api_key=settings.embedding_api_key,
        timeout=settings.request_timeout_seconds,
    )
    timing_ms["query_embedding"] = _elapsed_ms(query_embedding_started_at)

    if query_embedding.ok:
        vector_started_at = perf_counter()
        vector_result = search_vector_index(
            query_embedding.data,
            chunks_by_id,
            manifest_path=settings.vector_index_manifest,
            data_path=settings.vector_index_data,
            archive_url=settings.vector_index_archive_url,
            archive_sha256=settings.vector_index_archive_sha256,
            archive_timeout=settings.request_timeout_seconds,
            limit=embedding_pool_size,
        )
        timing_ms["vector_search"] = _elapsed_ms(vector_started_at)
    else:
        vector_result = None

    scoring_started_at = perf_counter()
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
        candidate_embedding_started_at = perf_counter()
        document_embeddings = embed_texts(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            texts=[chunk.search_text for chunk in embedding_pool],
            api_key=settings.embedding_api_key,
            timeout=settings.request_timeout_seconds,
        )
        timing_ms["candidate_embeddings"] = _elapsed_ms(candidate_embedding_started_at)

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
    timing_ms["score_merge"] = _elapsed_ms(scoring_started_at)

    candidate_started_at = perf_counter()
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    diagnostics["initial_top"] = [
        {"id": chunk.id, "score": round(float(score), 4)}
        for chunk, score in ranked[: settings.candidate_k]
    ]
    top_scored = _select_rerank_candidates(query, ranked, lexical_ranked, settings.candidate_k)
    top = [chunk for chunk, _score in top_scored]
    diagnostics["selected_reranker_candidates"] = [
        {"id": chunk.id, "score": round(float(score), 4)}
        for chunk, score in top_scored
    ]
    timing_ms["candidate_selection"] = _elapsed_ms(candidate_started_at)

    rerank_prepare_started_at = perf_counter()
    rerank_text_chars = max(200, settings.rerank_text_chars or DEFAULT_RERANK_TEXT_CHARS)
    include_board_metadata = _include_rerank_board_metadata(query, top)
    include_source_topic_metadata = _include_rerank_source_topic_metadata(query)
    rerank_documents = [
        _rerank_text(
            chunk,
            rerank_text_chars,
            include_board_metadata=include_board_metadata,
            include_source_topic_metadata=include_source_topic_metadata,
        )
        for chunk in top
    ]
    diagnostics["reranker_candidate_count"] = len(rerank_documents)
    diagnostics["reranker_text_chars"] = rerank_text_chars
    diagnostics["reranker_board_metadata"] = include_board_metadata
    diagnostics["reranker_source_topic_metadata"] = include_source_topic_metadata
    diagnostics["reranker_input_chars"] = {
        "max": max((len(document) for document in rerank_documents), default=0),
        "total": sum(len(document) for document in rerank_documents),
    }
    timing_ms["reranker_prepare"] = _elapsed_ms(rerank_prepare_started_at)
    rerank_started_at = perf_counter()
    rerank_result = rerank(
        base_url=settings.rerank_base_url,
        model=settings.rerank_model,
        query=query,
        documents=rerank_documents,
        api_key=settings.rerank_api_key,
        timeout=settings.request_timeout_seconds,
    )
    reranker_ms = _elapsed_ms(rerank_started_at)
    diagnostics["reranker_ms"] = reranker_ms
    timing_ms["reranker"] = reranker_ms
    context_started_at = perf_counter()
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
        top = _expand_source_context(query, ordered, chunks, settings.top_k)
        diagnostics["source_context_expanded"] = [chunk.id for chunk in top] != [
            chunk.id for chunk in ordered[: settings.top_k]
        ]
    else:
        if rerank_result.meta:
            diagnostics["reranker_mode"] = rerank_result.meta.get("mode", "unknown")
        diagnostics["reranker_error"] = rerank_result.error
        top = top[: settings.top_k]
    timing_ms["source_context"] = _elapsed_ms(context_started_at)

    diagnostics["citations"] = [chunk.id for chunk in top]
    timing_ms["total"] = _elapsed_ms(retrieval_started_at)
    diagnostics["timings_ms"] = timing_ms
    return top, diagnostics


def _rerank_text(
    chunk: KnowledgeChunk,
    max_chars: int,
    include_board_metadata: bool = False,
    include_source_topic_metadata: bool = False,
) -> str:
    text = chunk.text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    parts = [chunk.title, chunk.source]
    metadata = _rerank_metadata(
        chunk,
        include_board_metadata=include_board_metadata,
        include_source_topic_metadata=include_source_topic_metadata,
    )
    if metadata:
        parts.append(f"Metadata: {'; '.join(metadata)}")
    parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def _rerank_metadata(
    chunk: KnowledgeChunk,
    include_board_metadata: bool = False,
    include_source_topic_metadata: bool = False,
) -> list[str]:
    metadata: list[str] = []
    if include_board_metadata and chunk.board_id:
        metadata.append(f"board={chunk.board_id}")
    tags = [
        str(tag).strip()
        for tag in chunk.metadata.get("tags", [])
        if str(tag).strip()
    ][:RERANK_METADATA_TAG_LIMIT]
    if tags:
        metadata.append(f"tags={', '.join(tags)}")
    if include_source_topic_metadata:
        source_topic = _source_topic_metadata(chunk)
        if source_topic:
            metadata.append(f"source_topic={source_topic}")
    return metadata


def _include_rerank_board_metadata(query: str, chunks: list[KnowledgeChunk]) -> bool:
    q = query.lower()
    asks_for_board_choice = (
        "which supported xiao board" in q
        or "which xiao board" in q
        or "which board supports" in q
        or ("which xiao" in q and any(term in q for term in ("choose", "pick", "supports", "should")))
    )
    if not asks_for_board_choice:
        return False
    board_ids = {chunk.board_id for chunk in chunks if chunk.board_id}
    return len(board_ids) > 1


def _include_rerank_source_topic_metadata(query: str) -> bool:
    q = query.lower()
    return any(
        term in q
        for term in (
            "3d case",
            "3d enclosure",
            "3d-printed",
            "3d printed enclosure",
            "ai sensor",
            "model output",
            "sensecraft",
            "sscmacore",
        )
    )


def _source_topic_metadata(chunk: KnowledgeChunk) -> str:
    source_file = str(chunk.metadata.get("source_file", "")).strip()
    if not source_file:
        return ""
    topic = source_file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    topic = re.sub(r"[_-]+", " ", topic)
    topic = re.sub(r"\s+", " ", topic).strip()
    return topic[:160]


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


def _select_rerank_candidates(
    query: str,
    ranked: list[tuple[KnowledgeChunk, float]],
    lexical_ranked: list[tuple[KnowledgeChunk, float]],
    candidate_k: int,
) -> list[tuple[KnowledgeChunk, float]]:
    if candidate_k <= 0:
        return []

    profile = _query_profile(query)
    scored_by_id = {chunk.id: score for chunk, score in ranked}
    selected: list[tuple[KnowledgeChunk, float]] = []
    seen: set[str] = set()

    def add(chunk: KnowledgeChunk, score: float) -> None:
        if chunk.id in seen or len(selected) >= candidate_k:
            return
        seen.add(chunk.id)
        selected.append((chunk, score))

    recall_slots = min(LEXICAL_RECALL_SLOTS, max(2, candidate_k // 2), candidate_k)
    primary_slots = max(candidate_k - recall_slots, 1)
    for chunk, score in ranked[:primary_slots]:
        add(chunk, score)

    lexical_candidates: list[tuple[KnowledgeChunk, float, float]] = []
    for chunk, lexical_score in lexical_ranked[: max(LEXICAL_RECALL_SCAN, candidate_k * 8)]:
        if chunk.id in seen:
            continue
        recall_score = _recall_match_score_for_profile(profile, chunk)
        if recall_score <= 0 and lexical_score < 0.55:
            continue
        score = scored_by_id.get(chunk.id, min(lexical_score, 2.5) * 0.45)
        priority = recall_score + min(lexical_score, 1.0) * 0.1
        lexical_candidates.append((chunk, score, priority))

    lexical_candidates.sort(key=lambda item: item[2], reverse=True)
    for chunk, score, _priority in lexical_candidates:
        add(chunk, score)

    for chunk, score in ranked:
        add(chunk, score)
        if len(selected) >= candidate_k:
            break

    return selected


def _expand_source_context(
    query: str,
    ordered: list[KnowledgeChunk],
    corpus: list[KnowledgeChunk],
    limit: int,
) -> list[KnowledgeChunk]:
    if limit <= 0 or not ordered:
        return []
    if not _should_expand_source_context(query):
        return ordered[:limit]

    expansion_window = ordered[: min(len(ordered), max(limit, limit + 3))]
    source_counts = Counter(
        key for key in (_source_context_key(chunk) for chunk in expansion_window) if key
    )
    expandable_sources = {key for key, count in source_counts.items() if count >= 2}
    if not expandable_sources:
        return ordered[:limit]

    by_source: dict[str, list[KnowledgeChunk]] = {}
    for chunk in corpus:
        key = _source_context_key(chunk)
        if key:
            by_source.setdefault(key, []).append(chunk)

    selected = list(ordered[:limit])
    seen = {chunk.id for chunk in selected}
    sibling_candidates: list[tuple[KnowledgeChunk, float]] = []
    for chunk in selected:
        key = _source_context_key(chunk)
        if key not in expandable_sources:
            continue
        siblings = by_source.get(key, []) if key else []
        for sibling in _rank_source_siblings(query, chunk, siblings):
            if sibling.id in seen:
                continue
            sibling_candidates.append((sibling, _context_utility(query, sibling)))

    sibling_candidates.sort(key=lambda item: item[1], reverse=True)
    for sibling, sibling_utility in sibling_candidates:
        if sibling.id in seen:
            continue
        victim_index = _source_context_replacement_index(query, selected, sibling, sibling_utility)
        if victim_index is None:
            continue
        seen.discard(selected[victim_index].id)
        selected[victim_index] = sibling
        seen.add(sibling.id)

    return selected


def _rank_source_siblings(
    query: str,
    anchor: KnowledgeChunk,
    siblings: list[KnowledgeChunk],
) -> list[KnowledgeChunk]:
    if len(siblings) <= 1:
        return []

    try:
        anchor_index = next(index for index, chunk in enumerate(siblings) if chunk.id == anchor.id)
    except StopIteration:
        anchor_index = -1

    profile = _query_profile(query)
    ranked: list[tuple[KnowledgeChunk, float]] = []
    for index, sibling in enumerate(siblings):
        if sibling.id == anchor.id:
            continue
        lexical_score = _lexical_score_for_profile(profile, sibling)
        recall_score = _recall_match_score_for_profile(profile, sibling)
        procedure_score = _procedure_stage_score_for_profile(profile, sibling)
        detail_score = _configuration_detail_score_for_profile(profile, sibling)
        if lexical_score <= 0 and recall_score <= 0:
            continue
        proximity = 0.0
        if anchor_index >= 0:
            proximity = 1.0 / (abs(index - anchor_index) + 1)
        ranked.append(
            (
                sibling,
                lexical_score + recall_score + procedure_score + detail_score + proximity * 0.12,
            )
        )

    ranked.sort(key=lambda item: item[1], reverse=True)
    return [chunk for chunk, _score in ranked[:SOURCE_CONTEXT_SIBLINGS]]


def _should_expand_source_context(query: str) -> bool:
    q = query.lower()
    return any(
        term in q
        for term in (
            "camera slot",
            "compile",
            "deploy",
            "firmware",
            "mqtt gateway",
            "trigger action",
        )
    )


def _source_context_replacement_index(
    query: str,
    selected: list[KnowledgeChunk],
    sibling: KnowledgeChunk,
    sibling_utility: float,
) -> int | None:
    if not selected:
        return None
    utilities = [_context_utility(query, chunk) for chunk in selected]
    sibling_key = _source_context_key(sibling)
    off_source_indices = [
        index
        for index, chunk in enumerate(selected)
        if _source_context_key(chunk) != sibling_key
    ]
    if off_source_indices:
        weakest_index = min(
            off_source_indices,
            key=lambda index: utilities[index]
            - (0.75 if _is_weekly_wiki_source(selected[index]) else 0.0),
        )
        if _procedure_stage_score(query, sibling) > 0 or sibling_utility > utilities[weakest_index] - 0.25:
            return weakest_index
        return None

    weakest_index = min(range(len(utilities)), key=lambda index: utilities[index])
    if sibling_utility <= utilities[weakest_index] + 0.15:
        return None
    return weakest_index


def _context_utility(query: str, chunk: KnowledgeChunk) -> float:
    profile = _query_profile(query)
    return _context_utility_for_profile(profile, chunk)


def _context_utility_for_profile(profile: _QueryProfile, chunk: KnowledgeChunk) -> float:
    return (
        _lexical_score_for_profile(profile, chunk)
        + _recall_match_score_for_profile(profile, chunk)
        + _procedure_stage_score_for_profile(profile, chunk)
        + _configuration_detail_score_for_profile(profile, chunk)
        + _comparison_table_hint_score_for_profile(profile, chunk)
    )


def _source_context_key(chunk: KnowledgeChunk) -> str:
    source_file = str(chunk.metadata.get("source_file", "")).strip()
    if _is_weekly_wiki_source(chunk):
        return ""
    if source_file:
        return f"file:{source_file}"
    if chunk.kind == "wiki" and chunk.source:
        return f"source:{chunk.source}"
    return ""


def _procedure_stage_score(query: str, chunk: KnowledgeChunk) -> float:
    return _procedure_stage_score_for_profile(_query_profile(query), chunk)


def _procedure_stage_score_for_profile(profile: _QueryProfile, chunk: KnowledgeChunk) -> float:
    q = profile.lower
    text = _chunk_title_text_lower(chunk)
    if not any(
        term in q
        for term in (
            "compile",
            "connect",
            "deploy",
            "firmware",
            "preview",
            "setup",
            "steps",
            "update",
            "verify",
        )
    ):
        return 0.0

    score = 0.0
    if "step" in chunk.title.lower():
        score += 0.2
    if "deploy" in q and any(term in text for term in ("deploy", "upload", "connect device")):
        score += 0.75
    if "verify" in q and any(term in text for term in ("preview", "real-time", "feedback", "bounding box")):
        score += 0.45
    if "firmware" in q and any(term in text for term in ("firmware", "uf2", "bootloader")):
        score += 0.4
    if "compile" in q and any(term in text for term in ("build", "compile", "make ")):
        score += 0.35
    if "connect" in q and any(term in text for term in ("register", "configure", "otaa", "eui", "app key")):
        score += 0.3
    return score


def _configuration_detail_score(query: str, chunk: KnowledgeChunk) -> float:
    return _configuration_detail_score_for_profile(_query_profile(query), chunk)


def _configuration_detail_score_for_profile(profile: _QueryProfile, chunk: KnowledgeChunk) -> float:
    q = profile.lower
    text = _chunk_title_text_lower(chunk)
    score = 0.0

    if (
        any(term in q for term in ("yaml", "settings", "configuration", "config"))
        and "esphome" in q
    ):
        if any(term in text for term in ("platform_version", "variant:", "version:", "seeed_xiao_esp32c3")):
            score += 1.0

    if "camera slot" in q and any(term in q for term in ("cam_scl", "cam_sda", "gpio", "gpios")):
        if "occupies 14 gpios" in text:
            score += 1.2
        if all(term in text for term in ("cam_scl", "cam_sda", "gpio39", "gpio40")):
            score += 1.0

    if "trigger" in q and "action" in q:
        if any(term in text for term in ("light up the led", "save image to the sd card", "microsd card")):
            score += 1.2

    return score


def _is_weekly_wiki_source(chunk: KnowledgeChunk) -> bool:
    return bool(re.search(r"/wiki\d+/?$", chunk.source)) or "Weekly Wiki" in chunk.title


def _merge_vector_and_lexical_scores(
    query: str,
    vector_matches,
    lexical_ranked: list[tuple[KnowledgeChunk, float]],
) -> list[tuple[KnowledgeChunk, float]]:
    profile = _query_profile(query)
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
        score += _board_hint_score_for_profile(profile, chunk) * 0.15
        score += _kind_hint_score_for_profile(profile, chunk) * 0.05
        score += _capability_hint_score_for_profile(profile, chunk) * 0.45
        score += _configuration_detail_score_for_profile(profile, chunk)
        score += _comparison_table_hint_score_for_profile(profile, chunk)
        scored.append((chunk, score))

    return scored


def _lexical_prefilter(
    query: str,
    chunks: list[KnowledgeChunk],
    settings: Settings,
) -> list[tuple[KnowledgeChunk, float]]:
    profile = _query_profile(query)
    minimum = max(settings.candidate_k * 12, settings.top_k * 12, 72)
    scored = [
        (
            chunk,
            _lexical_score_for_profile(profile, chunk)
            + _board_hint_score_for_profile(profile, chunk)
            + _kind_hint_score_for_profile(profile, chunk)
            + _configuration_detail_score_for_profile(profile, chunk)
            + _comparison_table_hint_score_for_profile(profile, chunk),
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
    return _lexical_score_counts(query_terms, text_terms)


def _lexical_score_for_profile(profile: _QueryProfile, chunk: KnowledgeChunk) -> float:
    return _lexical_score_counts(profile.token_counts, _chunk_token_counts(chunk))


def _lexical_score_counts(query_terms: Counter[str], text_terms: Counter[str]) -> float:
    if not query_terms:
        return 0.0
    return sum(min(count, text_terms.get(term, 0)) for term, count in query_terms.items()) / len(query_terms)


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0).lower().strip(".+-")
        if not token:
            continue
        tokens.append(token)
        collapsed = re.sub(r"[.+-]+", "", token)
        if collapsed and collapsed != token:
            tokens.append(collapsed)
        for part in re.split(r"[.+-]+", token):
            if part and part != token:
                tokens.append(part)
    return tokens


def _query_profile(query: str) -> _QueryProfile:
    lower = query.lower()
    tokens = tuple(_tokens(query))
    return _QueryProfile(
        text=query,
        lower=lower,
        normalized=_normalize_id(query),
        token_counts=Counter(tokens),
        token_set=frozenset(tokens),
        distinctive_terms=_distinctive_terms_from_tokens(tokens),
        technical_terms=_technical_terms_from_tokens(tokens),
    )


def _chunk_search_text(chunk: KnowledgeChunk) -> str:
    key = _chunk_cache_key(chunk)
    cached = _CHUNK_SEARCH_TEXT_CACHE.get(key)
    if cached is None:
        cached = chunk.search_text
        _CHUNK_SEARCH_TEXT_CACHE[key] = cached
    return cached


def _chunk_search_lower(chunk: KnowledgeChunk) -> str:
    key = _chunk_cache_key(chunk)
    cached = _CHUNK_SEARCH_LOWER_CACHE.get(key)
    if cached is None:
        cached = _chunk_search_text(chunk).lower()
        _CHUNK_SEARCH_LOWER_CACHE[key] = cached
    return cached


def _chunk_title_text_lower(chunk: KnowledgeChunk) -> str:
    key = _chunk_cache_key(chunk)
    cached = _CHUNK_TEXT_LOWER_CACHE.get(key)
    if cached is None:
        cached = f"{chunk.title}\n{chunk.text}".lower()
        _CHUNK_TEXT_LOWER_CACHE[key] = cached
    return cached


def _chunk_token_counts(chunk: KnowledgeChunk) -> Counter[str]:
    key = _chunk_cache_key(chunk)
    cached = _CHUNK_TOKEN_COUNTS_CACHE.get(key)
    if cached is None:
        cached = Counter(_tokens(_chunk_search_text(chunk)))
        _CHUNK_TOKEN_COUNTS_CACHE[key] = cached
    return cached


def _chunk_token_set(chunk: KnowledgeChunk) -> frozenset[str]:
    key = _chunk_cache_key(chunk)
    cached = _CHUNK_TOKEN_SET_CACHE.get(key)
    if cached is None:
        cached = frozenset(_chunk_token_counts(chunk))
        _CHUNK_TOKEN_SET_CACHE[key] = cached
    return cached


def _chunk_cache_key(chunk: KnowledgeChunk) -> tuple[object, ...]:
    aliases = chunk.metadata.get("aliases", [])
    tags = chunk.metadata.get("tags", [])
    return (
        chunk.id,
        len(chunk.title),
        len(chunk.source),
        len(chunk.text),
        chunk.board_id,
        chunk.kind,
        len(aliases) if isinstance(aliases, list) else 0,
        len(tags) if isinstance(tags, list) else 0,
    )


def _distinctive_match_score(query: str, text: str) -> float:
    terms = _distinctive_query_terms(query)
    if not terms:
        return 0.0
    text_terms = set(_tokens(text))
    matched = sum(1 for term in terms if term in text_terms)
    return matched / len(terms)


def _distinctive_match_score_for_profile(profile: _QueryProfile, chunk: KnowledgeChunk) -> float:
    terms = profile.distinctive_terms
    if not terms:
        return 0.0
    text_terms = _chunk_token_set(chunk)
    matched = sum(1 for term in terms if term in text_terms)
    return matched / len(terms)


def _recall_match_score(query: str, text: str) -> float:
    technical_terms = _technical_query_terms(query)
    distinctive_score = _distinctive_match_score(query, text)
    if not technical_terms:
        return distinctive_score * 0.4
    text_terms = set(_tokens(text))
    technical_matches = sum(1 for term in technical_terms if term in text_terms)
    return (technical_matches / len(technical_terms)) * 1.5 + distinctive_score * 0.35


def _recall_match_score_for_profile(profile: _QueryProfile, chunk: KnowledgeChunk) -> float:
    technical_terms = profile.technical_terms
    distinctive_score = _distinctive_match_score_for_profile(profile, chunk)
    if not technical_terms:
        return distinctive_score * 0.4
    text_terms = _chunk_token_set(chunk)
    technical_matches = sum(1 for term in technical_terms if term in text_terms)
    return (technical_matches / len(technical_terms)) * 1.5 + distinctive_score * 0.35


def _distinctive_query_terms(query: str) -> list[str]:
    return list(_query_profile(query).distinctive_terms)


def _distinctive_terms_from_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen or token in COMMON_QUERY_TERMS:
            continue
        if _is_distinctive_token(token):
            seen.add(token)
            terms.append(token)
    return tuple(terms)


def _technical_query_terms(query: str) -> list[str]:
    return list(_query_profile(query).technical_terms)


def _technical_terms_from_tokens(tokens: tuple[str, ...]) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen or token in COMMON_QUERY_TERMS:
            continue
        if _is_technical_token(token):
            seen.add(token)
            terms.append(token)
    return tuple(terms)


def _is_distinctive_token(token: str) -> bool:
    if len(token) <= 1:
        return False
    if "_" in token or any(char.isdigit() for char in token):
        return True
    if token in {"adc", "ble", "boot", "i2c", "lora", "mqtt", "spi", "uart", "uf2", "wifi"}:
        return True
    return len(token) >= 5


def _is_technical_token(token: str) -> bool:
    return (
        "_" in token
        or any(char.isdigit() for char in token)
        or token in TECHNICAL_QUERY_TERMS
    )


def _board_hint_score(query: str, chunk: KnowledgeChunk) -> float:
    return _board_hint_score_for_profile(_query_profile(query), chunk)


def _board_hint_score_for_profile(profile: _QueryProfile, chunk: KnowledgeChunk) -> float:
    normalized_query = profile.normalized
    hints = [chunk.board_id, *chunk.metadata.get("aliases", [])]
    for hint in hints:
        normalized_hint = _normalize_id(hint)
        if normalized_hint and normalized_hint in normalized_query:
            return 1.0
    return 0.0


def _kind_hint_score(query: str, chunk: KnowledgeChunk) -> float:
    return _kind_hint_score_for_profile(_query_profile(query), chunk)


def _kind_hint_score_for_profile(profile: _QueryProfile, chunk: KnowledgeChunk) -> float:
    q = profile.lower
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


def _comparison_table_hint_score(query: str, chunk: KnowledgeChunk) -> float:
    return _comparison_table_hint_score_for_profile(_query_profile(query), chunk)


def _comparison_table_hint_score_for_profile(profile: _QueryProfile, chunk: KnowledgeChunk) -> float:
    q = profile.lower
    if "xiao" not in q:
        return 0.0
    if not any(term in q for term in ("between", "compare", "comparison", "table")):
        return 0.0
    title = chunk.title.lower()
    text = _chunk_title_text_lower(chunk)
    if "comparison table" not in title:
        return 0.0

    score = 0.9
    if any(term in q for term in ("current", "low power", "power mode", "battery")) and (
        "low power mode" in text or "power consumption" in text
    ):
        score += 0.5
    return score


def _capability_hint_score(query: str, chunk: KnowledgeChunk) -> float:
    return _capability_hint_score_for_profile(_query_profile(query), chunk)


def _capability_hint_score_for_profile(profile: _QueryProfile, chunk: KnowledgeChunk) -> float:
    q = profile.lower
    text = _chunk_search_lower(chunk)
    score = 0.0

    image_query = any(term in q for term in ("image", "vision", "camera", "photo"))
    audio_query = any(term in q for term in ("audio", "microphone", "mic", "speech", "keyword"))
    query_tokens = profile.token_set
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
