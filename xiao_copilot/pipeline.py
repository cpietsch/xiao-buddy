from __future__ import annotations

from PIL import Image

from xiao_copilot.clients import chat_completion
from xiao_copilot.config import load_settings
from xiao_copilot.image_utils import image_to_data_url, summarize_image
from xiao_copilot.knowledge_base import KnowledgeChunk
from xiao_copilot.retrieval import retrieve


SYSTEM_PROMPT = """You are XIAO Field Copilot, a concise hardware support assistant.
Use the provided context first. Give safe, practical next steps for Seeed Studio XIAO boards.
When uncertain, ask for the exact board variant or say what to measure instead of guessing.
Cite relevant sources as [id]."""


def answer_question(image: Image.Image | None, question: str) -> tuple[str, str, dict[str, object]]:
    question = (question or "").strip()
    if not question:
        return (
            "Ask a XIAO hardware question to start.",
            "",
            {"status": "waiting_for_question"},
        )

    settings = load_settings()
    image_summary = summarize_image(image)
    image_data_url = image_to_data_url(image)
    intent = _route_intent(question, bool(image_data_url))
    chunks, retrieval_diagnostics = retrieve(question, settings, image_data_url=image_data_url)

    answer = _generate_with_agent(
        question=question,
        image_summary=image_summary,
        image_data_url=image_data_url,
        intent=intent,
        chunks=chunks,
        settings=settings,
    )

    diagnostics: dict[str, object] = {
        "status": "ok",
        "intent": intent,
        "image": image_summary,
        "agent_multimodal": bool(image_data_url),
        "retrieval": retrieval_diagnostics,
    }

    if answer is None:
        diagnostics["agent_used"] = False
        answer = _fallback_answer(question, image_summary, chunks)
    else:
        diagnostics["agent_used"] = True

    return answer, _format_citations(chunks), diagnostics


def _generate_with_agent(
    question: str,
    image_summary: dict[str, object],
    image_data_url: str | None,
    intent: str,
    chunks: list[KnowledgeChunk],
    settings,
) -> str | None:
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
    result = chat_completion(
        base_url=settings.agent_base_url,
        model=settings.agent_model,
        messages=messages,
        api_key=settings.agent_api_key,
        timeout=settings.request_timeout_seconds,
    )
    if not result.ok:
        return None
    return str(result.data).strip()


def _fallback_answer(
    question: str,
    image_summary: dict[str, object],
    chunks: list[KnowledgeChunk],
) -> str:
    lead = (
        "Here is a safe field triage path based on the local XIAO support notes. "
        "The generation endpoint is not configured yet, so this is a deterministic fallback."
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
