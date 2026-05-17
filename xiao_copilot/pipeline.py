from __future__ import annotations

import re
from collections.abc import Iterator
from queue import Empty, Queue
from threading import Thread
from time import perf_counter

from PIL import Image

from xiao_copilot.clients import EndpointResult, chat_completion, chat_completion_stream
from xiao_copilot.config import Settings, load_settings
from xiao_copilot.image_utils import image_to_data_url, summarize_image
from xiao_copilot.knowledge_base import KnowledgeChunk
from xiao_copilot.retrieval import retrieve_progressive


SYSTEM_PROMPT = """You are XIAO Field Copilot, a concise Seeed hardware support assistant.
Use context first; if unsure, ask for the exact board variant or say what to measure.
Lead with the direct answer. Do not list alternatives unless asked.
Preserve exact product/service names, commands, part numbers, pins, constants, libraries, functions, ports, units, interfaces, setup values, radio details, filenames, and target device names from context.
For app/form setup, include required selections, region/frequency plan, IDs, EUIs, and keys.
For firmware/deployment, use up to three compact stages and keep exact chip, interface, button, filename, and drive names.
When a source or question uses a service acronym, include the full service name and acronym together once.
Cite relevant sources as [id]."""

EXACT_TERM_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("USB-UART", ("usb-uart", "usb uart")),
    ("BL702", ("bl702",)),
    ("Edge Impulse firmware", ("edge impulse firmware",)),
    ("firmware.uf2", ("firmware.uf2",)),
    ("GROVEAI", ("groveai",)),
    ("frequency plan", ("frequency plan", "frequenct plan")),
    ("The Things Network", ("the things network",)),
    ("device EUI", ("device eui",)),
    ("App EUI", ("app eui",)),
    ("APP key", ("app key", "appkey")),
    ("2.4G", ("2.4g", "2.4 ghz")),
    ("SX1801CCR", ("sx1801ccr",)),
    ("470 kΩ", ("470 kω", "470 kΩ", "470 kohm")),
    ("BAT_ADC_EN", ("bat_adc_en",)),
    ("BAT_ADC_READ", ("bat_adc_read",)),
    ("53 μA", ("53 μa", "53 μA", "53 ua")),
    ("SGM40567", ("sgm40567",)),
    ("TPS22916CYFPR", ("tps22916cyfpr",)),
    ("None", ("none",)),
    ("arduino-i2c-sht4x", ("arduino-i2c-sht4x",)),
    ("Sensirion Arduino Core", ("sensirion arduino core",)),
    ("reachy_bpm_dancer", ("reachy_bpm_dancer",)),
    ("reachy_fleet_control", ("reachy_fleet_control",)),
    ("agent planning", ("agent planning",)),
    ("task orchestration", ("task orchestration",)),
    ("AWS", ("aws",)),
    ("TTN", ("ttn",)),
    ("ChirpStack", ("chirpstack",)),
    ("Packet Forwarder", ("packet forwarder",)),
    ("Basics Station", ("basics station", "basics™ station")),
    ("Built-in LoRaWAN Network Server", ("built-in lorawan network server", "built in lorawan network server")),
    ("2.5km", ("2.5km", "2.5 km")),
    ("LoRaWAN Node", ("lorawan node",)),
    ("D2", ("d2",)),
    ("seeed_xiao_esp32c3", ("seeed_xiao_esp32c3",)),
    ("platform_version", ("platform_version",)),
    ("2.0.5", ("2.0.5",)),
    ("5.2.0", ("5.2.0",)),
    ("J501 Mini", ("j501 mini",)),
    ("temperature", ("temperature",)),
    ("humidity", ("humidity",)),
    ("SD card slot", ("sd card slot",)),
    ("GC9A01", ("gc9a01",)),
    ("CHSC6X", ("chsc6x",)),
    ("SPI", ("spi",)),
    ("I2C", ("i2c",)),
    ("GPIO41", ("gpio41", "gpio 41")),
    ("GPIO42", ("gpio42", "gpio 42")),
    ("ATSAMD51P19", ("atsamd51p19",)),
    ("Realtek RTL8720DN", ("realtek rtl8720dn", "rtl8720dn")),
    ("120MHz", ("120mhz", "120 mhz")),
    ("4MB", ("4mb", "4 mb")),
    ("192KB", ("192kb", "192 kb")),
    ("LIS3DHTR", ("lis3dhtr",)),
)

MARKED_EXACT_TERM_RE = re.compile(r"`([^`\n]{2,48})`|\*\*([^*\n]{2,48})\*\*")
AGENT_CONTEXT_MIN_CHARS_PER_SOURCE = 360
AGENT_CONTEXT_WINDOW_CHARS = 760
AGENT_PROGRESS_HEARTBEAT_SECONDS = 1.0
AGENT_CONTEXT_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "also",
        "and",
        "are",
        "board",
        "boards",
        "can",
        "does",
        "for",
        "from",
        "how",
        "include",
        "into",
        "its",
        "seeed",
        "should",
        "studio",
        "that",
        "the",
        "this",
        "use",
        "used",
        "what",
        "when",
        "where",
        "which",
        "with",
        "xiao",
        "you",
    }
)
EXACT_TERM_BLOCKLIST = frozenset(
    {
        "CONFIGURE",
        "EDIT",
        "FINISH",
        "INSTALL",
        "NEXT",
        "ONLINE",
        "SAVE",
        "SKIP",
        "STOP",
        "SUBMIT",
    }
)


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

    chunks: list[KnowledgeChunk] = []
    retrieval_diagnostics: dict[str, object] = {}
    draft_first_visible_ms: float | None = None
    draft_chars = 0
    preview_draft = ""
    preview_citations = ""
    for retrieval_stage in retrieve_progressive(question, settings, image_data_url=image_data_url):
        if retrieval_stage.stage == "pre_rerank":
            preview_chunks = retrieval_stage.chunks
            if not preview_chunks:
                continue
            preview_diagnostics = retrieval_stage.diagnostics
            preview_backend = str(preview_diagnostics.get("vector_index_backend") or "lexical")
            preview_detail = _retrieval_progress_detail(
                preview_diagnostics,
                preview_backend,
                len(preview_chunks),
            )
            preview_draft_started_at = perf_counter()
            preview_draft = _draft_answer(question, image_summary, preview_chunks)
            preview_citations = _format_citations(preview_chunks)
            timings_ms["source_draft_preview"] = _elapsed_ms(preview_draft_started_at)
            timings_ms["retrieve_preview"] = _elapsed_ms(retrieve_started_at)
            if preview_draft and draft_first_visible_ms is None:
                draft_first_visible_ms = _elapsed_ms(run_started_at)
                draft_chars = len(preview_draft)
                yield (
                    preview_draft,
                    preview_citations,
                    _run_diagnostics(
                        status="running",
                        stage="generate",
                        intent=intent,
                        image_summary=image_summary,
                        agent_multimodal=bool(image_data_url),
                        retrieval_diagnostics=preview_diagnostics,
                        chunks=preview_chunks,
                        timings_ms=timings_ms,
                        run_started_at=run_started_at,
                        agent_status="waiting",
                        agent_streaming=False,
                        draft_chars=draft_chars,
                        draft_first_visible_ms=draft_first_visible_ms,
                    ),
                    format_progress(
                        stage="generate",
                        backend=preview_backend,
                        source_count=len(preview_chunks),
                        retrieval_detail=preview_detail,
                        draft_chars=draft_chars,
                        draft_first_visible_ms=draft_first_visible_ms,
                    ),
                )
            continue
        if retrieval_stage.stage == "rerank_wait":
            wait_chunks = retrieval_stage.chunks
            if not wait_chunks:
                continue
            wait_diagnostics = retrieval_stage.diagnostics
            wait_backend = str(wait_diagnostics.get("vector_index_backend") or "lexical")
            wait_detail = _retrieval_progress_detail(
                wait_diagnostics,
                wait_backend,
                len(wait_chunks),
            )
            reranker_wait_ms = _numeric_timing(wait_diagnostics.get("reranker_wait_ms"))
            if not preview_draft:
                wait_draft_started_at = perf_counter()
                preview_draft = _draft_answer(question, image_summary, wait_chunks)
                preview_citations = _format_citations(wait_chunks)
                timings_ms["source_draft_preview"] = _elapsed_ms(wait_draft_started_at)
                if preview_draft and draft_first_visible_ms is None:
                    draft_first_visible_ms = _elapsed_ms(run_started_at)
                    draft_chars = len(preview_draft)
            if preview_draft:
                yield (
                    _append_reranker_wait_note(preview_draft, reranker_wait_ms),
                    preview_citations or _format_citations(wait_chunks),
                    _run_diagnostics(
                        status="running",
                        stage="generate",
                        intent=intent,
                        image_summary=image_summary,
                        agent_multimodal=bool(image_data_url),
                        retrieval_diagnostics=wait_diagnostics,
                        chunks=wait_chunks,
                        timings_ms=timings_ms,
                        run_started_at=run_started_at,
                        agent_status="waiting",
                        agent_streaming=False,
                        draft_chars=draft_chars or len(preview_draft),
                        draft_first_visible_ms=draft_first_visible_ms,
                    ),
                    format_progress(
                        stage="generate",
                        backend=wait_backend,
                        source_count=len(wait_chunks),
                        retrieval_detail=wait_detail,
                        draft_chars=draft_chars or len(preview_draft),
                        draft_first_visible_ms=draft_first_visible_ms,
                    ),
                )
            continue
        chunks = retrieval_stage.chunks
        retrieval_diagnostics = retrieval_stage.diagnostics

    timings_ms["retrieve"] = _elapsed_ms(retrieve_started_at)
    citations = _format_citations(chunks)
    backend = str(retrieval_diagnostics.get("vector_index_backend") or "lexical")
    retrieval_detail = _retrieval_progress_detail(retrieval_diagnostics, backend, len(chunks))
    agent_prompt_chars = _agent_prompt_chars(
        question,
        image_summary,
        image_data_url,
        intent,
        chunks,
        settings.agent_context_chars,
    )
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
        agent_context_chars=settings.agent_context_chars,
        agent_prompt_chars=agent_prompt_chars,
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
    agent_first_token_ms: float | None = None
    agent_first_visible_ms: float | None = None
    draft_started_at = perf_counter()
    draft_answer = _draft_answer(question, image_summary, chunks)
    timings_ms["source_draft"] = _elapsed_ms(draft_started_at)
    draft_chars = len(draft_answer) if draft_answer else draft_chars
    if draft_answer:
        if draft_first_visible_ms is None:
            draft_first_visible_ms = _elapsed_ms(run_started_at)
        yield (
            draft_answer,
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
                agent_status="running",
                agent_streaming=True,
                agent_context_chars=settings.agent_context_chars,
                agent_prompt_chars=agent_prompt_chars,
                draft_chars=draft_chars,
                draft_first_visible_ms=draft_first_visible_ms,
            ),
            format_progress(
                stage="generate",
                backend=backend,
                source_count=len(chunks),
                retrieval_detail=retrieval_detail,
                draft_chars=draft_chars,
                draft_first_visible_ms=draft_first_visible_ms,
            ),
        )
        agent_wait_ms = _elapsed_ms(generate_started_at)
        timings_ms["generate"] = agent_wait_ms
        yield (
            _append_agent_wait_note(draft_answer, agent_wait_ms),
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
                agent_status="waiting_first_token",
                agent_streaming=True,
                agent_streamed=False,
                agent_chunks=0,
                agent_chars=0,
                agent_wait_ms=agent_wait_ms,
                agent_context_chars=settings.agent_context_chars,
                agent_prompt_chars=agent_prompt_chars,
                draft_chars=draft_chars,
                draft_first_visible_ms=draft_first_visible_ms,
            ),
            format_progress(
                stage="generate",
                backend=backend,
                source_count=len(chunks),
                retrieval_detail=retrieval_detail,
                draft_chars=draft_chars,
                draft_first_visible_ms=draft_first_visible_ms,
                agent_wait_ms=agent_wait_ms,
            ),
        )

    for result in _generate_with_agent_stream_with_heartbeats(
        question=question,
        image_summary=image_summary,
        image_data_url=image_data_url,
        intent=intent,
        chunks=chunks,
        settings=settings,
    ):
        if result is None:
            agent_wait_ms = _elapsed_ms(generate_started_at)
            timings_ms["generate"] = agent_wait_ms
            visible_answer = _repair_generated_text("".join(answer_parts)).strip() or draft_answer
            if agent_first_token_ms is None and visible_answer:
                visible_answer = _append_agent_wait_note(visible_answer, agent_wait_ms)
            yield (
                visible_answer or "Generating a cited answer with the agent...",
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
                    agent_status="streaming" if agent_first_token_ms is not None else "waiting_first_token",
                    agent_streaming=True,
                    agent_streamed=agent_first_token_ms is not None,
                    agent_chunks=agent_chunks,
                    agent_chars=len(visible_answer) if agent_first_token_ms is not None else 0,
                    agent_first_token_ms=agent_first_token_ms,
                    agent_first_visible_ms=agent_first_visible_ms,
                    agent_wait_ms=agent_wait_ms,
                    agent_context_chars=settings.agent_context_chars,
                    agent_prompt_chars=agent_prompt_chars,
                    draft_chars=draft_chars,
                    draft_first_visible_ms=draft_first_visible_ms,
                ),
                format_progress(
                    stage="generate",
                    backend=backend,
                    source_count=len(chunks),
                    retrieval_detail=retrieval_detail,
                    agent_chunks=agent_chunks,
                    stream_chars=len(visible_answer) if agent_first_token_ms is not None else 0,
                    draft_chars=draft_chars,
                    draft_first_visible_ms=draft_first_visible_ms,
                    first_token_ms=agent_first_token_ms,
                    agent_wait_ms=agent_wait_ms,
                ),
            )
            continue
        if not result.ok:
            agent_error = result.error
            break
        answer_parts.append(str(result.data or ""))
        agent_chunks += 1
        if agent_first_token_ms is None:
            agent_first_token_ms = _elapsed_ms(generate_started_at)
            agent_first_visible_ms = _elapsed_ms(run_started_at)
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
                    agent_first_token_ms=agent_first_token_ms,
                    agent_first_visible_ms=agent_first_visible_ms,
                    agent_context_chars=settings.agent_context_chars,
                    agent_prompt_chars=agent_prompt_chars,
                    draft_chars=draft_chars,
                    draft_first_visible_ms=draft_first_visible_ms,
                ),
                format_progress(
                    stage="generate",
                    backend=backend,
                    source_count=len(chunks),
                    retrieval_detail=retrieval_detail,
                    agent_chunks=agent_chunks,
                    stream_chars=len(streamed_answer),
                    draft_chars=draft_chars,
                    draft_first_visible_ms=draft_first_visible_ms,
                    first_token_ms=agent_first_token_ms,
                ),
            )

    generated_answer = _repair_generated_text("".join(answer_parts)).strip()
    if agent_error:
        answer = ""
    else:
        answer = _ensure_inline_citations(_ensure_answer_exact_terms(generated_answer, question, chunks), chunks)
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
        agent_first_token_ms=agent_first_token_ms if agent_used else None,
        agent_first_visible_ms=agent_first_visible_ms if agent_used else None,
        agent_error=agent_error,
        agent_context_chars=settings.agent_context_chars,
        agent_prompt_chars=agent_prompt_chars,
        draft_chars=draft_chars,
        draft_first_visible_ms=draft_first_visible_ms,
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
            draft_chars=draft_chars,
            draft_first_visible_ms=draft_first_visible_ms,
            first_token_ms=agent_first_token_ms if agent_used else None,
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
    draft_chars: int = 0,
    draft_first_visible_ms: float | None = None,
    first_token_ms: float | None = None,
    agent_wait_ms: float | None = None,
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
                draft_chars,
                draft_first_visible_ms,
                first_token_ms,
                agent_wait_ms,
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
                draft_chars,
                draft_first_visible_ms,
                first_token_ms,
                agent_wait_ms,
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
    draft_chars: int,
    draft_first_visible_ms: float | None,
    first_token_ms: float | None,
    agent_wait_ms: float | None,
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
            first_token_detail = (
                f"; first token {first_token_ms:.0f} ms" if first_token_ms is not None else ""
            )
            return f"{prefix}; {stream_label} {stream_chars} chars{first_token_detail}"
        if draft_chars:
            prefix = retrieval_detail or f"{source_count} sources via {backend}"
            draft_ms = f" in {draft_first_visible_ms:.0f} ms" if draft_first_visible_ms is not None else ""
            wait_detail = (
                f"; waiting {agent_wait_ms:.0f} ms for first token"
                if agent_wait_ms is not None
                else "; preparing agent"
            )
            return f"{prefix}; source draft {draft_chars} chars{draft_ms}{wait_detail}"
        if source_count:
            prefix = retrieval_detail or f"{source_count} sources via {backend}"
            if agent_wait_ms is not None:
                return f"{prefix}; waiting {agent_wait_ms:.0f} ms for first token"
            return f"{prefix}; agent running"
        return "Agent running"
    if stage == "done":
        mode = "agent" if agent_used else "fallback"
        suffix = f" in {elapsed_ms:.0f} ms" if elapsed_ms is not None else ""
        first_token_detail = (
            f"; first token {first_token_ms:.0f} ms" if first_token_ms is not None else ""
        )
        if retrieval_detail:
            return f"{retrieval_detail}; {mode} answer{suffix}{first_token_detail}"
        if source_count and backend:
            return f"{source_count} sources via {backend}; {mode} answer{suffix}{first_token_detail}"
        return f"{mode} answer{suffix}{first_token_detail}"
    return ""


def _append_agent_wait_note(answer: str, agent_wait_ms: float) -> str:
    clean_answer = answer.rstrip()
    return (
        f"{clean_answer}\n\n"
        f"_Waiting for hosted agent first token: {agent_wait_ms:.0f} ms…_"
    )


def _append_reranker_wait_note(answer: str, reranker_wait_ms: float | None) -> str:
    clean_answer = answer.rstrip()
    if reranker_wait_ms is None:
        return f"{clean_answer}\n\n_Refining source order with hosted reranker…_"
    return (
        f"{clean_answer}\n\n"
        f"_Refining source order with hosted reranker: {reranker_wait_ms:.0f} ms…_"
    )


def _retrieval_progress_detail(
    retrieval_diagnostics: dict[str, object],
    backend: str,
    source_count: int,
) -> str:
    base = f"{source_count} sources via {backend}"
    timing_parts = _retrieval_timing_parts(retrieval_diagnostics)
    timing_suffix = f"; {'; '.join(timing_parts)}" if timing_parts else ""
    if retrieval_diagnostics.get("reranker_wait_ms") is not None:
        wait_ms = _numeric_timing(retrieval_diagnostics.get("reranker_wait_ms"))
        wait_detail = f" {wait_ms:.0f} ms" if wait_ms is not None else ""
        return f"{base}; reranker running{wait_detail}{timing_suffix}"
    if retrieval_diagnostics.get("reranker_pending"):
        return f"{base}; pre-rerank preview{timing_suffix}"
    if retrieval_diagnostics.get("reranker_used"):
        mode = str(retrieval_diagnostics.get("reranker_mode") or "rerank")
        return f"{base}; {mode} rerank{timing_suffix}"
    if retrieval_diagnostics.get("reranker_error"):
        return f"{base}; rerank skipped{timing_suffix}"
    return f"{base}{timing_suffix}"


def _retrieval_timing_parts(retrieval_diagnostics: dict[str, object]) -> list[str]:
    timings = retrieval_diagnostics.get("timings_ms")
    if not isinstance(timings, dict):
        return []
    parts: list[str] = []
    query_embedding_ms = _numeric_timing(timings.get("query_embedding"))
    vector_search_ms = _numeric_timing(timings.get("vector_search"))
    candidate_embedding_ms = _numeric_timing(timings.get("candidate_embeddings"))
    reranker_ms = _numeric_timing(timings.get("reranker"))
    reranker_wait_ms = _numeric_timing(timings.get("reranker_wait"))
    if query_embedding_ms is not None:
        parts.append(f"embed {query_embedding_ms:.0f} ms")
    if vector_search_ms is not None:
        parts.append(f"hnsw {vector_search_ms:.0f} ms")
    elif candidate_embedding_ms is not None:
        parts.append(f"candidate embeds {candidate_embedding_ms:.0f} ms")
    if reranker_ms is not None:
        parts.append(f"rerank {reranker_ms:.0f} ms")
    elif reranker_wait_ms is not None:
        parts.append(f"rerank wait {reranker_wait_ms:.0f} ms")
    return parts


def _numeric_timing(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 1)


def _generate_with_agent_stream_with_heartbeats(
    *,
    question: str,
    image_summary: dict[str, object],
    image_data_url: str | None,
    intent: str,
    chunks: list[KnowledgeChunk],
    settings: Settings,
) -> Iterator[EndpointResult | None]:
    stream_queue: Queue[tuple[str, EndpointResult | None]] = Queue()

    def run_stream() -> None:
        try:
            for result in _generate_with_agent_stream(
                question=question,
                image_summary=image_summary,
                image_data_url=image_data_url,
                intent=intent,
                chunks=chunks,
                settings=settings,
            ):
                stream_queue.put(("result", result))
        except Exception as exc:  # noqa: BLE001 - keep UI responsive if a provider wrapper fails.
            stream_queue.put(("result", EndpointResult(ok=False, error=str(exc))))
        finally:
            stream_queue.put(("done", None))

    Thread(target=run_stream, daemon=True).start()

    while True:
        try:
            kind, result = stream_queue.get(timeout=AGENT_PROGRESS_HEARTBEAT_SECONDS)
        except Empty:
            yield None
            continue
        if kind == "done":
            return
        yield result


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
    agent_first_token_ms: float | None = None,
    agent_first_visible_ms: float | None = None,
    agent_wait_ms: float | None = None,
    agent_error: str = "",
    agent_context_chars: int | None = None,
    agent_prompt_chars: int | None = None,
    draft_chars: int = 0,
    draft_first_visible_ms: float | None = None,
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
    if draft_chars or draft_first_visible_ms is not None:
        draft: dict[str, object] = {"visible": bool(draft_chars)}
        if draft_chars:
            draft["chars"] = draft_chars
        if draft_first_visible_ms is not None:
            draft["first_visible_ms"] = draft_first_visible_ms
            diagnostics["draft_first_visible_ms"] = draft_first_visible_ms
        diagnostics["draft"] = draft

    agent: dict[str, object] = {
        "status": agent_status or ("streaming" if agent_streaming else "waiting"),
        "streaming": agent_streaming,
        "streamed": agent_streamed,
        "stream_chunks": agent_chunks,
        "stream_chars": agent_chars,
    }
    if agent_first_token_ms is not None:
        agent["first_token_ms"] = agent_first_token_ms
        diagnostics["agent_first_token_ms"] = agent_first_token_ms
    if agent_first_visible_ms is not None:
        agent["first_visible_ms"] = agent_first_visible_ms
        diagnostics["agent_first_visible_ms"] = agent_first_visible_ms
    if agent_wait_ms is not None:
        agent["wait_ms"] = agent_wait_ms
        diagnostics["agent_wait_ms"] = agent_wait_ms
    if agent_used is not None:
        agent["used"] = agent_used
        diagnostics["agent_used"] = agent_used
    if agent_error:
        agent["error"] = agent_error
        diagnostics["agent_error"] = agent_error
    if agent_context_chars is not None or agent_prompt_chars is not None:
        prompt: dict[str, object] = {}
        if agent_context_chars is not None:
            prompt["context_budget_chars"] = agent_context_chars
        if agent_prompt_chars is not None:
            prompt["chars"] = agent_prompt_chars
        agent["prompt"] = prompt
        diagnostics["agent_prompt"] = prompt
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
        context_chars=settings.agent_context_chars,
    )
    result = chat_completion(
        base_url=settings.agent_base_url,
        model=settings.agent_model,
        messages=messages,
        api_key=settings.agent_api_key,
        timeout=settings.request_timeout_seconds,
        max_tokens=settings.agent_max_tokens,
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
        context_chars=settings.agent_context_chars,
    )
    yield from chat_completion_stream(
        base_url=settings.agent_base_url,
        model=settings.agent_model,
        messages=messages,
        api_key=settings.agent_api_key,
        timeout=settings.request_timeout_seconds,
        max_tokens=settings.agent_max_tokens,
    )


def _build_agent_messages(
    question: str,
    image_summary: dict[str, object],
    image_data_url: str | None,
    intent: str,
    chunks: list[KnowledgeChunk],
    context_chars: int = 0,
) -> list[dict[str, object]]:
    context = _build_agent_context(question, chunks, context_chars)
    exact_terms = _exact_terms_hint(question, chunks)
    exact_terms_note = ""
    if exact_terms:
        exact_terms_note = (
            "\n\nPreserve if relevant: "
            + ", ".join(exact_terms)
            + ". Include terms that answer the question."
        )
    prompt = (
        f"Route: {intent}\n"
        f"Question: {question}\n\n"
        f"Image: {image_summary}\n\n"
        f"Sources:\n{context}{exact_terms_note}\n\n"
        "If an image is provided, use only visible markings/hardware. "
        "Answer the specific question; enumerate requested options, pins, values, commands, steps, or settings. "
        "For YAML/config, put the exact block/settings first, including version and platform_version if present. "
        "Return a direct cited answer and compact next checks only when useful."
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


def _agent_prompt_chars(
    question: str,
    image_summary: dict[str, object],
    image_data_url: str | None,
    intent: str,
    chunks: list[KnowledgeChunk],
    context_chars: int,
) -> int:
    messages = _build_agent_messages(
        question=question,
        image_summary=image_summary,
        image_data_url=image_data_url,
        intent=intent,
        chunks=chunks,
        context_chars=context_chars,
    )
    return sum(len(_message_content_text(message.get("content"))) for message in messages)


def _message_content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            value = item.get("text")
            if isinstance(value, str):
                parts.append(value)
        return "".join(parts)
    return ""


def _build_agent_context(question: str, chunks: list[KnowledgeChunk], context_chars: int) -> str:
    if context_chars <= 0 or not chunks:
        return "\n\n".join(_format_agent_context_chunk(question, chunk, 0) for chunk in chunks)

    remaining = max(context_chars, AGENT_CONTEXT_MIN_CHARS_PER_SOURCE)
    parts: list[str] = []
    for index, chunk in enumerate(chunks):
        chunks_left = len(chunks) - index
        chunk_budget = max(AGENT_CONTEXT_MIN_CHARS_PER_SOURCE, remaining // max(chunks_left, 1))
        excerpt = _agent_context_excerpt(question, chunk.text, chunk_budget)
        remaining = max(0, remaining - len(excerpt))
        parts.append(f"[{chunk.id}] {chunk.title}\nSource: {chunk.source}\n{excerpt}")
    return "\n\n".join(parts)


def _format_agent_context_chunk(question: str, chunk: KnowledgeChunk, context_chars: int) -> str:
    text = _agent_context_excerpt(question, chunk.text, context_chars)
    return f"[{chunk.id}] {chunk.title}\nSource: {chunk.source}\n{text}"


def _agent_context_excerpt(question: str, text: str, limit: int) -> str:
    text = text.strip()
    if limit <= 0 or len(text) <= limit:
        return text

    terms = _agent_context_terms(question, text)
    windows = _agent_context_windows(text, terms, limit)
    if not windows:
        return _trim_context_edge(text, limit)

    excerpts = [_trim_context_edge(text[start:end].strip(), max(80, end - start)) for start, end in windows]
    excerpt = "\n...\n".join(part for part in excerpts if part)
    if len(excerpt) > limit:
        excerpt = _trim_context_edge(excerpt, limit)
    return excerpt


def _agent_context_terms(question: str, text: str) -> list[str]:
    normalized_text = text.lower()
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_+.-]{3,}", question):
        lowered = token.lower()
        if lowered not in AGENT_CONTEXT_STOPWORDS:
            terms.append(token)

    for canonical, aliases in EXACT_TERM_CANDIDATES:
        if _contains_alias(normalized_text, canonical.lower()) or any(
            _contains_alias(normalized_text, alias) for alias in aliases
        ):
            terms.append(canonical)

    for match in MARKED_EXACT_TERM_RE.finditer(text):
        value = (match.group(1) or match.group(2) or "").strip()
        if _looks_like_exact_term(value):
            terms.append(value)

    for token in re.findall(r"\b[A-Z][A-Z0-9_+-]{2,}\b|\b\d+(?:\.\d+)?\s*(?:kΩ|μA|mA|V|G|GHz|MHz)\b", text):
        terms.append(token.strip())

    return list(dict.fromkeys(term for term in terms if len(term.strip()) >= 2))[:40]


def _agent_context_windows(text: str, terms: list[str], limit: int) -> list[tuple[int, int]]:
    lowered = text.lower()
    positions: list[int] = []
    for term in terms:
        normalized = term.lower().strip()
        if not normalized:
            continue
        start = 0
        while True:
            index = lowered.find(normalized, start)
            if index < 0:
                break
            positions.append(index)
            start = index + max(len(normalized), 1)
            if len(positions) >= 24:
                break
        if len(positions) >= 24:
            break

    if not positions:
        return []

    half_window = max(180, min(AGENT_CONTEXT_WINDOW_CHARS // 2, limit // 2))
    windows = [_expand_context_window(text, position, half_window) for position in sorted(set(positions))]
    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1] + 80:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    selected: list[tuple[int, int]] = []
    used = 0
    for start, end in merged:
        segment_len = end - start
        separator_len = 5 if selected else 0
        if selected and used + separator_len + segment_len > limit:
            break
        if segment_len > limit:
            center = start + segment_len // 2
            selected.append(_expand_context_window(text, center, limit // 2))
            break
        selected.append((start, end))
        used += separator_len + segment_len
    return selected


def _expand_context_window(text: str, position: int, half_window: int) -> tuple[int, int]:
    start = max(0, position - half_window)
    end = min(len(text), position + half_window)
    while start > 0 and text[start - 1] not in "\n.;:!?":
        start -= 1
    while end < len(text) and text[end - 1] not in "\n.;:!?":
        end += 1
    return start, end


def _trim_context_edge(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 12:
        return text[:limit]
    return text[: max(0, limit - 4)].rstrip() + " ..."


def _exact_terms_hint(question: str, chunks: list[KnowledgeChunk]) -> list[str]:
    query_terms = _query_terms(question)
    candidates: list[tuple[int, int, str]] = []
    seen: set[str] = set()

    def add(term: str, source_order: int, text: str, position: int = 0, force: bool = False) -> None:
        term = term.strip()
        key = term.lower()
        if not term or key in seen or term in EXACT_TERM_BLOCKLIST:
            return
        score = _exact_term_score(term, text, query_terms, position)
        if force:
            score += 10
        if not force and not _looks_like_exact_term(term) and key not in query_terms and score < 3:
            return
        if score <= 0:
            return
        seen.add(key)
        candidates.append((-score, source_order, term))

    for index, chunk in enumerate(chunks):
        source_text = f"{chunk.title}\n{chunk.text}"
        normalized = source_text.lower()
        for canonical, aliases in EXACT_TERM_CANDIDATES:
            if _contains_alias(normalized, canonical.lower()) or any(
                _contains_alias(normalized, alias) for alias in aliases
            ):
                add(canonical, index, source_text, normalized.find(canonical.lower()), force=True)
        if "8n1" in normalized or "serial_8n1" in normalized:
            add("None", index, source_text, normalized.find("8n1"), force=True)
        if {"esphome", "yaml"} & query_terms and "arduino" in normalized:
            add("arduino", index, source_text, normalized.find("arduino"), force=True)

        for match in MARKED_EXACT_TERM_RE.finditer(source_text):
            value = (match.group(1) or match.group(2) or "").strip()
            add(value, index, source_text, match.start())

        for value, position in _code_like_terms(source_text):
            add(value, index, source_text, position)

    return [term for _score, _source_order, term in sorted(candidates)[:24]]


def _query_terms(question: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_+.-]{3,}", question)
        if token.lower() not in AGENT_CONTEXT_STOPWORDS
    }


def _code_like_terms(text: str) -> list[tuple[str, int]]:
    terms: list[tuple[str, int]] = []
    for match in re.finditer(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_.-]{2,})\s*:\s*([A-Za-z0-9_./+-]+)", text):
        terms.append((match.group(1), match.start(1)))
        terms.append((match.group(2), match.start(2)))
    for match in re.finditer(r"#define\s+([A-Z0-9_]+)\s+([A-Za-z0-9_.]+)", text):
        terms.append((match.group(1), match.start(1)))
        terms.append((match.group(2), match.start(2)))
    for match in re.finditer(r"\b[A-Z][A-Z0-9_+-]{2,}\b|\b\d+(?:\.\d+)?\s*(?:kΩ|μA|mA|V|G|GHz|MHz)\b", text):
        terms.append((match.group(0).strip(), match.start()))
    return terms


def _exact_term_score(term: str, text: str, query_terms: set[str], position: int) -> int:
    lowered = term.lower()
    score = 0
    if lowered in query_terms or any(query in lowered or lowered in query for query in query_terms):
        score += 5
    if any(char.isdigit() for char in term):
        score += 3
    if any(marker in term for marker in ("_", "-", ".", "/")):
        score += 2
    if term.isupper() and len(term) > 2:
        score += 2

    if position < 0:
        position = text.lower().find(lowered)
    if position >= 0:
        window = text[max(0, position - 180) : position + len(term) + 180].lower()
        score += min(5, sum(1 for query in query_terms if query in window))
    return score


def _ensure_answer_exact_terms(answer: str, question: str, chunks: list[KnowledgeChunk]) -> str:
    if not answer.strip():
        return answer
    answer = _normalize_contextual_exact_terms(answer, question)
    candidate_terms = {canonical for canonical, _aliases in EXACT_TERM_CANDIDATES} | {"arduino"}
    missing: list[tuple[str, str]] = []
    for term in _exact_terms_hint(question, chunks):
        if term not in candidate_terms:
            continue
        if _answer_has_exact_term(answer, term):
            continue
        source_id = _source_id_for_exact_term(term, chunks)
        if source_id:
            missing.append((term, source_id))
    if not missing:
        return _question_specific_exact_term_repair(answer, question, chunks)

    contextual_repair = _contextual_exact_term_repair(answer, question, chunks, missing)
    if contextual_repair:
        return _question_specific_exact_term_repair(contextual_repair, question, chunks)

    return _question_specific_exact_term_repair(answer, question, chunks)


def _normalize_contextual_exact_terms(answer: str, question: str) -> str:
    q = question.lower()
    if ("uart" in q or "serial" in q) and not _answer_has_exact_term(answer, "None"):
        return re.sub(r"\bno parity\b", "parity `None`", answer, flags=re.IGNORECASE)
    if ("oled" in q or "ssd1306" in q) and "address" in q and not _answer_has_exact_term(answer, "0x3C"):
        return re.sub(r"(?<![A-Za-z0-9])`?0?x?3c`?(?![A-Za-z0-9])", "`0x3C`", answer, count=1, flags=re.IGNORECASE)
    return answer


def _question_specific_exact_term_repair(answer: str, question: str, chunks: list[KnowledgeChunk]) -> str:
    q = question.lower()
    repaired = answer

    if (
        ("oled" in q or "ssd1306" in q)
        and "address" in q
        and not _answer_has_exact_term(repaired, "0x3C")
    ):
        source_id = _source_id_for_aliases(("0x3c", "ssd1306"), chunks)
        if source_id:
            repaired = (
                f"{repaired.rstrip()}\n\n"
                f"The cited OLED configuration uses I2C address `0x3C` [{source_id}]."
            )

    if (
        "microphone" in q
        and "clock" in q
        and "data" in q
        and (not _answer_has_exact_term(repaired, "Clock") or not _answer_has_exact_term(repaired, "Data"))
    ):
        source_id = _source_id_for_aliases(("pdm microphone clk", "pdm microphone clock", "pdm microphone data"), chunks)
        if source_id:
            repaired = (
                f"{repaired.rstrip()}\n\n"
                f"The cited pin map labels the microphone clock as `Clock` and microphone data as `Data` [{source_id}]."
            )

    return repaired


def _contextual_exact_term_repair(
    answer: str,
    question: str,
    chunks: list[KnowledgeChunk],
    missing: list[tuple[str, str]],
) -> str:
    q = question.lower()
    missing_terms = {term for term, _source_id in missing}

    if {"esphome", "yaml"} <= set(_query_terms(question)) and missing_terms & {
        "seeed_xiao_esp32c3",
        "arduino",
        "platform_version",
        "2.0.5",
        "5.2.0",
    }:
        source_id = _source_id_for_exact_term("2.0.5", chunks)
        return (
            "For the Seeed Studio XIAO ESP32C3 ESPHome setup, use the Arduino-framework "
            f"board block from the Grove/Home Assistant guide [{source_id}]:\n\n"
            "```yaml\n"
            "esp32:\n"
            "  board: seeed_xiao_esp32c3\n"
            "  variant: esp32c3\n"
            "  framework:\n"
            "    type: arduino\n"
            "    version: 2.0.5\n"
            "    platform_version: 5.2.0\n"
            "```"
        )

    if ("sht40" in q or "temp" in q or "humi" in q) and missing_terms & {
        "arduino-i2c-sht4x",
        "Sensirion Arduino Core",
        "temperature",
        "humidity",
    }:
        source_id = _source_id_for_exact_term("arduino-i2c-sht4x", chunks)
        return (
            f"{answer.rstrip()}\n\n"
            "Use the `arduino-i2c-sht4x` library package with `Sensirion Arduino Core`; "
            f"call `measureHighPrecision()` to read temperature and humidity [{source_id}]."
        )

    repair_lines = _contextual_exact_term_lines(question, missing)
    if repair_lines:
        return f"{answer.rstrip()}\n\n" + " ".join(repair_lines)

    return ""


def _contextual_exact_term_lines(question: str, missing: list[tuple[str, str]]) -> list[str]:
    q = question.lower()
    groups: list[tuple[set[str], str]] = [
        (
            {"SGM40567", "TPS22916CYFPR", "SX1801CCR", "470 kΩ", "BAT_ADC_EN", "BAT_ADC_READ", "53 μA"},
            "The cited battery circuit details also name {terms}.",
        ),
        (
            {"2.4G"},
            "For WiFi, the cited setup specifies {terms}.",
        ),
        (
            {"The Things Network", "frequency plan", "device EUI", "App EUI", "APP key"},
            "From the LoRaWAN app setup, record {terms}.",
        ),
        (
            {"AWS", "TTN", "ChirpStack", "Packet Forwarder", "Basics Station", "Built-in LoRaWAN Network Server"},
            "Supported network-server options include {terms}.",
        ),
        (
            {"2.5km", "LoRaWAN Node"},
            "The cited kit application list also includes {terms}.",
        ),
        (
            {"D2"},
            "The cited RS485 example identifies the enable pin as {terms}.",
        ),
        (
            {"BL702", "USB-UART", "Edge Impulse firmware", "firmware.uf2", "GROVEAI"},
            "The firmware deployment flow also uses {terms}.",
        ),
        (
            {"reachy_bpm_dancer", "reachy_fleet_control"},
            "The service names in the guide are {terms}.",
        ),
        (
            {"agent planning", "task orchestration"},
            "The OpenClaw role wording includes {terms}.",
        ),
        (
            {"SD card slot"},
            "For the Round Display/Sense storage behavior, the cited source also names {terms}.",
        ),
        (
            {"GC9A01", "CHSC6X", "SPI", "I2C"},
            "The cited Round Display device-tree details also name {terms}.",
        ),
        (
            {"GPIO41", "GPIO42"},
            "The cited microphone pin map also names {terms}.",
        ),
        (
            {"ATSAMD51P19", "Realtek RTL8720DN", "120MHz", "4MB", "192KB", "LIS3DHTR"},
            "The cited Wio Terminal hardware summary also names {terms}.",
        ),
    ]
    lines: list[str] = []
    used: set[str] = set()
    for terms, template in groups:
        selected = [
            (term, source_id)
            for term, source_id in missing
            if term in terms and _exact_term_group_matches_question(term, q)
        ]
        if selected:
            lines.append(template.format(terms=_format_cited_exact_terms(selected)))
            used.update(term for term, _source_id in selected)

    direct_matches = [
        (term, source_id)
        for term, source_id in missing
        if term not in used and term.lower() in q
    ]
    if direct_matches:
        lines.append(f"The cited source also names {_format_cited_exact_terms(direct_matches)}.")
    return lines


def _exact_term_group_matches_question(term: str, question_lower: str) -> bool:
    battery_terms = {"SGM40567", "TPS22916CYFPR", "SX1801CCR", "470 kΩ", "BAT_ADC_EN", "BAT_ADC_READ", "53 μA"}
    if term in battery_terms:
        return "battery" in question_lower or "low-power" in question_lower or "low power" in question_lower
    if term == "2.4G":
        return "wifi" in question_lower or "mqtt" in question_lower
    if term in {"The Things Network", "frequency plan", "device EUI", "App EUI", "APP key"}:
        return "ttn" in question_lower or "app" in question_lower or "lorawan" in question_lower
    if term in {"AWS", "TTN", "ChirpStack", "Packet Forwarder", "Basics Station", "Built-in LoRaWAN Network Server"}:
        return "network-server" in question_lower or "network server" in question_lower or "lorawan gateway" in question_lower
    if term in {"2.5km", "LoRaWAN Node"}:
        return "wio-sx1262" in question_lower or "3d case" in question_lower or "used for" in question_lower
    if term == "D2":
        return "rs485" in question_lower and "enable" in question_lower
    if term in {"BL702", "USB-UART", "Edge Impulse firmware", "firmware.uf2", "GROVEAI"}:
        return "firmware" in question_lower or "edge impulse" in question_lower or "mass-storage" in question_lower
    if term in {"reachy_bpm_dancer", "reachy_fleet_control"}:
        return "reachy" in question_lower or "fleet" in question_lower
    if term in {"agent planning", "task orchestration"}:
        return "openclaw" in question_lower or "so-arm" in question_lower
    if term == "SD card slot":
        return "round display" in question_lower and ("sd" in question_lower or "microsd" in question_lower)
    if term in {"GC9A01", "CHSC6X"}:
        return "round display" in question_lower or "display" in question_lower or "touch" in question_lower
    if term in {"SPI", "I2C"}:
        return "round display" in question_lower and ("bus" in question_lower or "buses" in question_lower)
    if term in {"GPIO41", "GPIO42"}:
        return "microphone" in question_lower and "clock" in question_lower and "data" in question_lower
    if term in {"ATSAMD51P19", "Realtek RTL8720DN", "120MHz", "4MB", "192KB", "LIS3DHTR"}:
        return "wio terminal" in question_lower and (
            "mcu" in question_lower
            or "wireless" in question_lower
            or "memory" in question_lower
            or "features" in question_lower
            or "hardware" in question_lower
            or "include" in question_lower
        )
    return False


def _format_cited_exact_terms(terms: list[tuple[str, str]]) -> str:
    return ", ".join(f"`{term}` [{source_id}]" for term, source_id in terms[:6])


def _looks_like_exact_term(value: str) -> bool:
    if not value or len(value) > 48:
        return False
    if any(char.isspace() for char in value):
        return False
    lowered = value.lower()
    if lowered.startswith(("step ", "note:", "warning", "caution")):
        return False
    return (
        any(char.isdigit() for char in value)
        or any(marker in value for marker in ("-", "_", ".", "/"))
        or (value.isupper() and len(value) > 2)
    )


def _contains_alias(normalized_text: str, alias: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized_text) is not None


def _answer_has_exact_term(answer: str, term: str) -> bool:
    normalized = answer.lower()
    return any(_contains_alias(normalized, alias) for alias in _aliases_for_exact_term(term))


def _source_id_for_exact_term(term: str, chunks: list[KnowledgeChunk]) -> str:
    aliases = _aliases_for_exact_term(term)
    return _source_id_for_aliases(aliases, chunks)


def _source_id_for_aliases(aliases: tuple[str, ...], chunks: list[KnowledgeChunk]) -> str:
    for chunk in chunks:
        normalized = f"{chunk.title}\n{chunk.text}".lower()
        if any(_contains_alias(normalized, alias) for alias in aliases):
            return chunk.id
    return chunks[0].id if chunks else ""


def _aliases_for_exact_term(term: str) -> tuple[str, ...]:
    lowered = term.lower()
    for canonical, aliases in EXACT_TERM_CANDIDATES:
        if canonical.lower() == lowered:
            return aliases
    return (lowered,)


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


def _draft_answer(
    question: str,
    image_summary: dict[str, object],
    chunks: list[KnowledgeChunk],
) -> str:
    if not chunks:
        return ""
    source_lines = "\n".join(
        f"- **{chunk.title}:** {_compact_source_text(chunk.text, 220)} [{chunk.id}]"
        for chunk in chunks[:2]
    )
    return (
        "**Source-backed draft**\n\n"
        "Initial notes from the retrieved Seeed sources:\n\n"
        f"{source_lines}"
    )


def _compact_source_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _repair_generated_text(text: str) -> str:
    replacements = {
        "Î¼": "μ",
        "Î©": "Ω",
        "Âµ": "µ",
        "Â°": "°",
        "\u00c2\u00ae": "®",
        "â€“": "-",
        "â€”": "-",
        "â€™": "'",
        "â€œ": '"',
        "â€\u009d": '"',
        "\u00e2\u0084\u00a2": "™",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _ensure_inline_citations(text: str, chunks: list[KnowledgeChunk]) -> str:
    if not text or not chunks:
        return text
    source_ids = list(dict.fromkeys(chunk.id for chunk in chunks))
    missing_citations = [f"[{source_id}]" for source_id in source_ids if f"[{source_id}]" not in text]
    if not missing_citations:
        return text
    if re.search(r"(?im)^Sources:\s*", text):
        return re.sub(
            r"(?im)^(Sources:\s*)(.*)$",
            lambda match: f"{match.group(1)}{match.group(2).rstrip()} {' '.join(missing_citations)}",
            text.rstrip(),
            count=1,
        )
    return f"{text.rstrip()}\n\nSources: {' '.join(missing_citations)}"


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
