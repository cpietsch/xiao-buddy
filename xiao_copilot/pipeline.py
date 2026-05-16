from __future__ import annotations

from collections.abc import Iterator
from time import perf_counter

from PIL import Image

from xiao_copilot.clients import EndpointResult, chat_completion, chat_completion_stream
from xiao_copilot.config import load_settings
from xiao_copilot.image_utils import image_to_data_url, summarize_image
from xiao_copilot.knowledge_base import KnowledgeChunk
from xiao_copilot.retrieval import retrieve


SYSTEM_PROMPT = """You are XIAO Field Copilot, a concise hardware support assistant.
Use the provided context first. Give safe, practical next steps for Seeed Studio XIAO boards.
When uncertain, ask for the exact board variant or say what to measure instead of guessing.
Preserve exact part numbers, pin labels, constants, library names, function names, units, and numeric settings from the context.
Cite relevant sources as [id]."""


def answer_question(image: Image.Image | None, question: str) -> tuple[str, str, dict[str, object]]:
    latest: tuple[str, str, dict[str, object]] | None = None
    for answer, citations, diagnostics, _progress in answer_question_stream(image, question):
        latest = (answer, citations, diagnostics)
    if latest is None:
        return (
            "Ask a XIAO hardware question to start.",
            "",
            {"status": "waiting_for_question"},
        )
    return latest


def answer_question_stream(
    image: Image.Image | None,
    question: str,
) -> Iterator[tuple[str, str, dict[str, object], str]]:
    run_started_at = perf_counter()
    question = (question or "").strip()
    if not question:
        yield (
            "Ask a XIAO hardware question to start.",
            "",
            {"status": "waiting_for_question"},
            format_progress(stage="waiting"),
        )
        return

    settings = load_settings()
    prepare_started_at = perf_counter()
    timings_ms: dict[str, float] = {}
    diagnostics = _run_diagnostics(
        status="running",
        stage="prepare",
        timings_ms=timings_ms,
        run_started_at=run_started_at,
    )
    yield (
        "Preparing the request...",
        "",
        diagnostics,
        format_progress(stage="prepare"),
    )

    image_summary = summarize_image(image)
    image_data_url = image_to_data_url(image)
    intent = _route_intent(question, bool(image_data_url))
    timings_ms["prepare"] = _elapsed_ms(prepare_started_at)
    retrieve_started_at = perf_counter()
    diagnostics = _run_diagnostics(
        status="running",
        stage="retrieve",
        intent=intent,
        image_summary=image_summary,
        agent_multimodal=bool(image_data_url),
        timings_ms=timings_ms,
        run_started_at=run_started_at,
    )
    yield (
        "Retrieving relevant Seeed wiki sources...",
        "",
        diagnostics,
        format_progress(stage="retrieve", intent=intent),
    )

    chunks, retrieval_diagnostics = retrieve(question, settings, image_data_url=image_data_url)
    timings_ms["retrieve"] = _elapsed_ms(retrieve_started_at)
    citations = _format_citations(chunks)
    backend = str(retrieval_diagnostics.get("vector_index_backend") or "lexical")
    retrieval_detail = _retrieval_progress_detail(retrieval_diagnostics, backend, len(chunks))
    generate_started_at = perf_counter()
    diagnostics = _run_diagnostics(
        status="running",
        stage="generate",
        intent=intent,
        image_summary=image_summary,
        agent_multimodal=bool(image_data_url),
        retrieval_diagnostics=retrieval_diagnostics,
        chunks=chunks,
        timings_ms=timings_ms,
        run_started_at=run_started_at,
        agent_status="starting",
        agent_streaming=True,
    )
    yield (
        "Generating a cited answer with the agent...",
        citations,
        diagnostics,
        format_progress(
            stage="generate",
            backend=backend,
            source_count=len(chunks),
            retrieval_detail=retrieval_detail,
        ),
    )

    answer_parts: list[str] = []
    agent_error = ""
    agent_chunks = 0
    for result in _generate_with_agent_stream(
        question=question,
        image_summary=image_summary,
        image_data_url=image_data_url,
        intent=intent,
        chunks=chunks,
        settings=settings,
    ):
        if not result.ok:
            agent_error = result.error
            break
        answer_parts.append(str(result.data or ""))
        agent_chunks += 1
        streamed_answer = _repair_generated_text("".join(answer_parts))
        if streamed_answer.strip():
            timings_ms["generate"] = _elapsed_ms(generate_started_at)
            yield (
                streamed_answer,
                citations,
                _run_diagnostics(
                    status="running",
                    stage="generate",
                    intent=intent,
                    image_summary=image_summary,
                    agent_multimodal=bool(image_data_url),
                    retrieval_diagnostics=retrieval_diagnostics,
                    chunks=chunks,
                    timings_ms=timings_ms,
                    run_started_at=run_started_at,
                    agent_status="streaming",
                    agent_streaming=True,
                    agent_streamed=True,
                    agent_chunks=agent_chunks,
                    agent_chars=len(streamed_answer),
                ),
                format_progress(
                    stage="generate",
                    backend=backend,
                    source_count=len(chunks),
                    retrieval_detail=retrieval_detail,
                    agent_chunks=agent_chunks,
                    stream_chars=len(streamed_answer),
                ),
            )

    answer = _ensure_primary_inline_citation(
        _repair_generated_text("".join(answer_parts)).strip(),
        chunks,
    )
    timings_ms["generate"] = _elapsed_ms(generate_started_at)

    if answer:
        agent_used = True
        agent_status = "done"
    else:
        agent_used = False
        agent_status = "fallback"
        answer = _fallback_answer(question, image_summary, chunks)
    diagnostics = _run_diagnostics(
        status="ok",
        stage="done",
        intent=intent,
        image_summary=image_summary,
        agent_multimodal=bool(image_data_url),
        retrieval_diagnostics=retrieval_diagnostics,
        chunks=chunks,
        timings_ms=timings_ms,
        run_started_at=run_started_at,
        agent_status=agent_status,
        agent_streaming=False,
        agent_streamed=agent_used,
        agent_used=agent_used,
        agent_chunks=agent_chunks,
        agent_chars=len(answer),
        agent_error=agent_error,
    )

    yield (
        answer,
        citations,
        diagnostics,
        format_progress(
            stage="done",
            backend=backend,
            source_count=len(chunks),
            retrieval_detail=retrieval_detail,
            agent_used=bool(diagnostics["agent_used"]),
            agent_chunks=agent_chunks,
            stream_chars=len(answer),
            elapsed_ms=diagnostics["timings_ms"]["total"],
        ),
    )


def format_progress(
    *,
    stage: str = "waiting",
    intent: str = "",
    backend: str = "",
    source_count: int = 0,
    agent_used: bool | None = None,
    agent_chunks: int = 0,
    stream_chars: int = 0,
    elapsed_ms: float | None = None,
    retrieval_detail: str = "",
) -> str:
    labels = [
        ("prepare", "Prepare request"),
        ("retrieve", "Retrieve sources"),
        ("generate", "Generate answer"),
        ("done", "Ready"),
    ]
    rank = {key: index for index, (key, _label) in enumerate(labels)}
    active_rank = rank.get(stage, -1)
    items: list[str] = []
    for index, (key, label) in enumerate(labels):
        if stage == "waiting":
            state = "next"
            detail = "Waiting"
        elif index < active_rank or stage == "done":
            state = "done"
            detail = _progress_detail(
                key,
                intent,
                backend,
                source_count,
                agent_used,
                agent_chunks,
                stream_chars,
                elapsed_ms,
                retrieval_detail,
            )
        elif index == active_rank:
            state = "active"
            detail = _progress_detail(
                key,
                intent,
                backend,
                source_count,
                agent_used,
                agent_chunks,
                stream_chars,
                elapsed_ms,
                retrieval_detail,
            )
        else:
            state = "next"
            detail = "Queued"
        items.append(
            "<li class=\"progress-step progress-step-"
            f"{state}\"><span class=\"progress-number\">{index + 1}</span>"
            f"<span><strong>{label}</strong><small>{detail}</small></span></li>"
        )
    return (
        "<div class=\"progress-panel\" role=\"status\" aria-live=\"polite\" aria-atomic=\"true\">"
        "<p class=\"progress-title\">Run progress</p>"
        f"<ol class=\"progress-list\">{''.join(items)}</ol>"
        "</div>"
    )


def _progress_detail(
    stage: str,
    intent: str,
    backend: str,
    source_count: int,
    agent_used: bool | None,
    agent_chunks: int,
    stream_chars: int,
    elapsed_ms: float | None,
    retrieval_detail: str,
) -> str:
    if stage == "prepare":
        return "Reading input"
    if stage == "retrieve":
        return f"Route: {intent}" if intent else "Embedding and HNSW lookup"
    if stage == "generate":
        if stream_chars:
            prefix = retrieval_detail or f"{source_count} sources via {backend}"
            stream_label = "streamed" if agent_used is not None else "streaming"
            return f"{prefix}; {stream_label} {stream_chars} chars"
        if source_count:
            prefix = retrieval_detail or f"{source_count} sources via {backend}"
            return f"{prefix}; agent running"
        return "Agent running"
    if stage == "done":
        mode = "agent" if agent_used else "fallback"
        suffix = f" in {elapsed_ms:.0f} ms" if elapsed_ms is not None else ""
        if retrieval_detail:
            return f"{retrieval_detail}; {mode} answer{suffix}"
        if source_count and backend:
            return f"{source_count} sources via {backend}; {mode} answer{suffix}"
        return f"{mode} answer{suffix}"
    return ""


def _retrieval_progress_detail(
    retrieval_diagnostics: dict[str, object],
    backend: str,
    source_count: int,
) -> str:
    base = f"{source_count} sources via {backend}"
    reranker_ms = retrieval_diagnostics.get("reranker_ms")
    reranker_suffix = f" ({reranker_ms:.0f} ms)" if isinstance(reranker_ms, (int, float)) else ""
    if retrieval_diagnostics.get("reranker_used"):
        mode = str(retrieval_diagnostics.get("reranker_mode") or "rerank")
        return f"{base} + {mode} rerank{reranker_suffix}"
    if retrieval_diagnostics.get("reranker_error"):
        return f"{base}; rerank skipped{reranker_suffix}"
    return base


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 1)


def _run_diagnostics(
    *,
    status: str,
    stage: str,
    timings_ms: dict[str, float],
    run_started_at: float,
    intent: str = "",
    image_summary: dict[str, object] | None = None,
    agent_multimodal: bool = False,
    retrieval_diagnostics: dict[str, object] | None = None,
    chunks: list[KnowledgeChunk] | None = None,
    agent_status: str = "",
    agent_streaming: bool = False,
    agent_streamed: bool = False,
    agent_used: bool | None = None,
    agent_chunks: int = 0,
    agent_chars: int = 0,
    agent_error: str = "",
) -> dict[str, object]:
    timing_snapshot = dict(timings_ms)
    timing_snapshot["total"] = _elapsed_ms(run_started_at)

    diagnostics: dict[str, object] = {
        "status": status,
        "stage": stage,
        "timings_ms": timing_snapshot,
    }
    if intent:
        diagnostics["intent"] = intent
    if image_summary is not None:
        diagnostics["image"] = image_summary
    diagnostics["agent_multimodal"] = agent_multimodal

    if retrieval_diagnostics is not None:
        diagnostics["retrieval"] = retrieval_diagnostics
    if chunks is not None:
        diagnostics["top_sources"] = _source_summaries(chunks)

    agent: dict[str, object] = {
        "status": agent_status or ("streaming" if agent_streaming else "waiting"),
        "streaming": agent_streaming,
        "streamed": agent_streamed,
        "stream_chunks": agent_chunks,
        "stream_chars": agent_chars,
    }
    if agent_used is not None:
        agent["used"] = agent_used
        diagnostics["agent_used"] = agent_used
    if agent_error:
        agent["error"] = agent_error
        diagnostics["agent_error"] = agent_error
    diagnostics["agent"] = agent
    diagnostics["agent_streaming"] = agent_streaming
    diagnostics["agent_streamed"] = agent_streamed
    return diagnostics


def _source_summaries(chunks: list[KnowledgeChunk]) -> list[dict[str, object]]:
    return [
        {
            "id": chunk.id,
            "title": chunk.title,
            "source": chunk.source,
            "board_id": chunk.board_id,
            "kind": chunk.kind,
        }
        for chunk in chunks
    ]


def _generate_with_agent(
    question: str,
    image_summary: dict[str, object],
    image_data_url: str | None,
    intent: str,
    chunks: list[KnowledgeChunk],
    settings,
) -> str | None:
    messages = _build_agent_messages(
        question=question,
        image_summary=image_summary,
        image_data_url=image_data_url,
        intent=intent,
        chunks=chunks,
    )
    result = chat_completion(
        base_url=settings.agent_base_url,
        model=settings.agent_model,
        messages=messages,
        api_key=settings.agent_api_key,
        timeout=settings.request_timeout_seconds,
    )
    if not result.ok:
        return None
    return _repair_generated_text(str(result.data)).strip()


def _generate_with_agent_stream(
    question: str,
    image_summary: dict[str, object],
    image_data_url: str | None,
    intent: str,
    chunks: list[KnowledgeChunk],
    settings,
) -> Iterator[EndpointResult]:
    messages = _build_agent_messages(
        question=question,
        image_summary=image_summary,
        image_data_url=image_data_url,
        intent=intent,
        chunks=chunks,
    )
    yield from chat_completion_stream(
        base_url=settings.agent_base_url,
        model=settings.agent_model,
        messages=messages,
        api_key=settings.agent_api_key,
        timeout=settings.request_timeout_seconds,
    )


def _build_agent_messages(
    question: str,
    image_summary: dict[str, object],
    image_data_url: str | None,
    intent: str,
    chunks: list[KnowledgeChunk],
) -> list[dict[str, object]]:
    context = "\n\n".join(
        f"[{chunk.id}] {chunk.title}\nSource: {chunk.source}\n{chunk.text}"
        for chunk in chunks
    )
    prompt = (
        f"Agent route: {intent}\n"
        f"Question: {question}\n\n"
        f"Image metadata: {image_summary}\n\n"
        f"Context:\n{context}\n\n"
        "If an image is provided, inspect visible board markings, MCU labels, connectors, "
        "antenna parts, sensor modules, camera/microphone hardware, and pin labels. "
        "Do not claim a visual detail unless it is visible. "
        "Return: likely product or board family when relevant, short next checks, and citations."
    )
    user_content: str | list[dict[str, object]]
    if image_data_url:
        user_content = [
            {"type": "image_url", "image_url": {"url": image_data_url}},
            {"type": "text", "text": prompt},
        ]
    else:
        user_content = prompt

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages


def _fallback_answer(
    question: str,
    image_summary: dict[str, object],
    chunks: list[KnowledgeChunk],
) -> str:
    lead = (
        "The agent endpoint did not return a usable answer, so this is a deterministic "
        "fallback based on the retrieved Seeed support notes."
    )
    image_line = ""
    if image_summary.get("provided"):
        image_line = (
            "\n\n**Photo received:** I can see the upload metadata, but true visual inspection needs "
            "a multimodal agent endpoint. Use the photo to check solder bridges, shifted headers, "
            "reversed connectors, and damaged pads before applying power [xiao-soldering]."
        )

    steps = "\n".join(
        f"- **{chunk.title}:** {chunk.text} [{chunk.id}]"
        for chunk in chunks[:3]
    )
    return f"**Answer**\n\n{lead}{image_line}\n\n{steps}\n\n**Question tracked:** {question}"


def _repair_generated_text(text: str) -> str:
    replacements = {
        "Î¼": "μ",
        "Î©": "Ω",
        "Âµ": "µ",
        "Â°": "°",
        "â€“": "-",
        "â€”": "-",
        "â€™": "'",
        "â€œ": '"',
        "â€\u009d": '"',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _ensure_primary_inline_citation(text: str, chunks: list[KnowledgeChunk]) -> str:
    if not text or not chunks:
        return text
    primary_citation = f"[{chunks[0].id}]"
    if primary_citation in text:
        return text
    return f"{text.rstrip()} {primary_citation}"


def _format_citations(chunks: list[KnowledgeChunk]) -> str:
    if not chunks:
        return ""
    lines = ["**Sources used**"]
    lines.extend(f"- [{chunk.id}] {chunk.title} — {chunk.source}" for chunk in chunks)
    return "\n".join(lines)


def _route_intent(question: str, has_image: bool) -> str:
    q = question.lower()
    if any(term in q for term in ("compare", " vs ", "versus", "which board", "pick")):
        return "compare"
    if any(term in q for term in ("pin", "wire", "i2c", "spi", "uart", "gpio", "adc", "dac")):
        return "wiring_or_pinout"
    if any(term in q for term in ("not detected", "not responding", "fails", "error", "reset", "brownout", "boot")):
        return "troubleshoot"
    if has_image or any(term in q for term in ("what board", "identify", "which xiao", "what is this")):
        return "identify"
    return "support"
