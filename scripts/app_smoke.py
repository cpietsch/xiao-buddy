from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.config import load_settings


DEFAULT_QUERY = "Which XIAO should I choose for 5 GHz WiFi?"
DEFAULT_TERMS = ("XIAO ESP32-C5", "5 GHz", "Wi-Fi 6")
DEFAULT_CITATIONS = ("https://wiki.seeedstudio.com/xiao_esp32c5_wifi_usage/",)
DEFAULT_EVAL_PATH = Path("data/corpus/answer_eval_queries.jsonl")
DEFAULT_HEARTBEAT_THRESHOLD_MS = 1500.0


def main() -> None:
    settings = load_settings()
    base_url = os.environ.get("APP_SMOKE_URL") or _app_url(settings)
    query, terms, citations, case_id = _load_case_from_env()
    timeout = float(os.environ.get("APP_SMOKE_TIMEOUT_SECONDS", settings.request_timeout_seconds or 90))
    require_stream = os.environ.get("APP_SMOKE_REQUIRE_STREAM", "1") == "1"
    require_first_token = os.environ.get("APP_SMOKE_REQUIRE_FIRST_TOKEN", "1" if require_stream else "0") == "1"
    require_source_draft = os.environ.get("APP_SMOKE_REQUIRE_SOURCE_DRAFT", "1" if require_stream else "0") == "1"
    require_heartbeat = os.environ.get("APP_SMOKE_REQUIRE_AGENT_HEARTBEAT", "1" if require_stream else "0") == "1"
    heartbeat_threshold_ms = float(os.environ.get("APP_SMOKE_HEARTBEAT_THRESHOLD_MS", DEFAULT_HEARTBEAT_THRESHOLD_MS))
    require_inline_citation = os.environ.get("APP_SMOKE_REQUIRE_INLINE_CITATION", "1") == "1"
    require_reranker_cache = os.environ.get("APP_SMOKE_REQUIRE_RERANK_CACHE", "0") == "1"
    require_no_reranker_wait = os.environ.get("APP_SMOKE_REQUIRE_NO_RERANK_WAIT", "0") == "1"

    event_id = _start_call(base_url, query, timeout=timeout)
    events = list(_stream_call(base_url, event_id, timeout=timeout))
    if not events:
        raise SystemExit("App smoke did not receive any stream events.")

    final = events[-1]
    answer = str(final[0] if len(final) > 0 else "")
    source_text = str(final[1] if len(final) > 1 else "")
    diagnostics = final[2] if len(final) > 2 and isinstance(final[2], dict) else {}
    progress_html = str(final[3] if len(final) > 3 else "")

    _require_terms("answer", answer, terms)
    _require_terms("source trail", source_text, citations)
    if require_inline_citation:
        _require_inline_citation(answer, source_text, citations)
    if require_stream:
        draft_events = _source_draft_events(events)
        if require_source_draft and not draft_events:
            raise SystemExit("Expected a visible source-backed draft event before the hosted agent stream.")
        stream_events = [
            event
            for event in events
            if len(event) > 2
            and isinstance(event[2], dict)
            and (event[2].get("agent_streamed") or event[2].get("agent", {}).get("streamed"))
        ]
        if len(stream_events) < 2:
            raise SystemExit(f"Expected multiple streamed answer events, got {len(stream_events)}.")
        if require_first_token and not _stream_events_with_first_token(stream_events):
            raise SystemExit("Expected first-token diagnostics during streamed answer events.")
        if require_heartbeat:
            heartbeat_events = _agent_wait_heartbeat_events(events)
            if first_token := _first_token_ms(final[2] if len(final) > 2 and isinstance(final[2], dict) else {}):
                if first_token >= heartbeat_threshold_ms and not heartbeat_events:
                    raise SystemExit(
                        "Expected progress heartbeat events while waiting for the hosted agent first token; "
                        f"first_token_ms={_format_ms(first_token)} threshold_ms={heartbeat_threshold_ms:.0f}."
                    )
    first_token_ms = _first_token_ms(diagnostics)
    if require_first_token:
        _require_first_token_latency(first_token_ms, progress_html)
    if diagnostics.get("status") != "ok":
        raise SystemExit(f"Expected final diagnostics status=ok, got {diagnostics.get('status')!r}.")
    if not diagnostics.get("agent_used"):
        raise SystemExit("Expected final diagnostics to report agent_used=true.")
    if "Ready" not in progress_html:
        raise SystemExit("Expected final progress HTML to include Ready.")
    reranker_wait_events = _reranker_wait_events(events)
    reranker_cache_events = _reranker_cache_hit_events(events)
    _require_reranker_cache_behavior(
        reranker_wait_events=reranker_wait_events,
        reranker_cache_events=reranker_cache_events,
        require_cache=require_reranker_cache,
        require_no_wait=require_no_reranker_wait,
    )

    print(
        "PASS app smoke: "
        f"events={len(events)} "
        f"case={case_id or 'custom'} "
        f"chars={len(answer)} "
        f"sources={source_text.count('- [')} "
        f"inline_citation={require_inline_citation} "
        f"source_draft_events={len(_source_draft_events(events))} "
        f"reranker_wait_events={len(reranker_wait_events)} "
        f"reranker_cache_events={len(reranker_cache_events)} "
        f"heartbeat_events={len(_agent_wait_heartbeat_events(events))} "
        f"first_token_ms={_format_ms(first_token_ms)} "
        f"total_ms={diagnostics.get('timings_ms', {}).get('total', 0)}"
    )


def _load_case_from_env() -> tuple[str, tuple[str, ...], tuple[str, ...], str]:
    case_id = os.environ.get("APP_SMOKE_CASE_ID", "").strip()
    if case_id:
        case = _load_answer_eval_case(
            path=Path(os.environ.get("APP_SMOKE_EVAL_PATH", str(DEFAULT_EVAL_PATH))),
            case_id=case_id,
        )
        return (
            str(case["query"]),
            tuple(str(term) for term in case.get("must_include", [])),
            tuple(str(citation) for citation in case.get("must_cite", [])),
            case_id,
        )
    return (
        os.environ.get("APP_SMOKE_QUERY", DEFAULT_QUERY),
        _csv_env("APP_SMOKE_MUST_INCLUDE", DEFAULT_TERMS),
        _csv_env("APP_SMOKE_MUST_CITE", DEFAULT_CITATIONS),
        "",
    )


def _load_answer_eval_case(path: Path, case_id: str) -> dict[str, object]:
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if case.get("id") == case_id:
            return case
    raise SystemExit(f"App smoke case {case_id!r} was not found in {path}.")


def _start_call(base_url: str, query: str, *, timeout: float) -> str:
    response = requests.post(
        urljoin(base_url.rstrip("/") + "/", "gradio_api/call/v2/ask"),
        json={"image": None, "question": query},
        timeout=timeout,
    )
    response.raise_for_status()
    event_id = response.json().get("event_id")
    if not event_id:
        raise SystemExit(f"App smoke did not receive an event_id: {response.text[:500]}")
    return str(event_id)


def _stream_call(base_url: str, event_id: str, *, timeout: float):
    url = urljoin(base_url.rstrip("/") + "/", f"gradio_api/call/ask/{event_id}")
    current_event = ""
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith("event:"):
                current_event = line.removeprefix("event:").strip()
                if current_event == "error":
                    raise SystemExit("App smoke received an error event.")
                continue
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if current_event == "complete" and data in {"", "null"}:
                return
            try:
                value = json.loads(data)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"App smoke received invalid JSON data: {exc}") from exc
            if isinstance(value, list):
                yield value


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _require_terms(label: str, text: str, terms: tuple[str, ...]) -> None:
    normalized = _normalize(text)
    missing = [term for term in terms if _normalize(term) not in normalized]
    if missing:
        raise SystemExit(f"{label} missing required terms: {', '.join(missing)}")


def _require_inline_citation(answer: str, source_text: str, citations: tuple[str, ...]) -> None:
    source_ids = _source_ids_for_required_citations(source_text, citations)
    if not source_ids:
        source_ids = _source_ids_from_source_text(source_text)
    if not any(f"[{source_id}]" in answer for source_id in source_ids):
        expected = ", ".join(f"[{source_id}]" for source_id in source_ids) or "a retrieved source id"
        raise SystemExit(f"answer missing inline citation for {expected}")


def _first_token_ms(diagnostics: dict[str, object]) -> float | None:
    agent = diagnostics.get("agent", {})
    value = agent.get("first_token_ms") if isinstance(agent, dict) else None
    if value is None:
        value = diagnostics.get("agent_first_token_ms")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _require_first_token_latency(first_token_ms: float | None, progress_html: str) -> None:
    if first_token_ms is None or first_token_ms <= 0:
        raise SystemExit(f"Expected final diagnostics to include positive first_token_ms, got {first_token_ms!r}.")
    if "first token" not in progress_html.lower():
        raise SystemExit("Expected final progress HTML to include first-token latency.")


def _stream_events_with_first_token(events: list[list[object]]) -> list[list[object]]:
    matches: list[list[object]] = []
    for event in events:
        if len(event) <= 3 or not isinstance(event[2], dict):
            continue
        diagnostics = event[2]
        agent = diagnostics.get("agent", {})
        agent_streaming = (
            agent.get("streaming") if isinstance(agent, dict) else diagnostics.get("agent_streaming")
        )
        progress_html = str(event[3])
        if (
            agent_streaming
            and _first_token_ms(diagnostics) is not None
            and "first token" in progress_html.lower()
        ):
            matches.append(event)
    return matches


def _source_draft_events(events: list[list[object]]) -> list[list[object]]:
    matches: list[list[object]] = []
    for event in events:
        if len(event) <= 3 or not isinstance(event[2], dict):
            continue
        diagnostics = event[2]
        draft = diagnostics.get("draft", {})
        progress_html = str(event[3])
        answer = str(event[0] if event else "")
        if (
            isinstance(draft, dict)
            and draft.get("visible")
            and "source draft" in progress_html.lower()
            and "Source-backed draft" in answer
        ):
            matches.append(event)
    return matches


def _agent_wait_heartbeat_events(events: list[list[object]]) -> list[list[object]]:
    matches: list[list[object]] = []
    for event in events:
        if len(event) <= 3 or not isinstance(event[2], dict):
            continue
        diagnostics = event[2]
        agent = diagnostics.get("agent", {})
        status = agent.get("status") if isinstance(agent, dict) else diagnostics.get("agent_status")
        wait_ms = agent.get("wait_ms") if isinstance(agent, dict) else diagnostics.get("agent_wait_ms")
        progress_html = str(event[3]).lower()
        if (
            status == "waiting_first_token"
            and isinstance(wait_ms, (int, float))
            and "waiting" in progress_html
            and "first token" in progress_html
        ):
            matches.append(event)
    return matches


def _reranker_wait_events(events: list[list[object]]) -> list[list[object]]:
    matches: list[list[object]] = []
    for event in events:
        if len(event) <= 3 or not isinstance(event[2], dict):
            continue
        diagnostics = event[2]
        retrieval = diagnostics.get("retrieval", {})
        reranker_wait_ms = (
            retrieval.get("reranker_wait_ms")
            if isinstance(retrieval, dict)
            else diagnostics.get("reranker_wait_ms")
        )
        answer = str(event[0] if event else "")
        progress_html = str(event[3]).lower()
        if (
            isinstance(reranker_wait_ms, (int, float))
            and "reranker running" in progress_html
            and "Refining source order" in answer
        ):
            matches.append(event)
    return matches


def _reranker_cache_hit_events(events: list[list[object]]) -> list[list[object]]:
    matches: list[list[object]] = []
    for event in events:
        if len(event) <= 3 or not isinstance(event[2], dict):
            continue
        diagnostics = event[2]
        retrieval = diagnostics.get("retrieval", {})
        cache_hit = (
            retrieval.get("reranker_cache_hit")
            if isinstance(retrieval, dict)
            else diagnostics.get("reranker_cache_hit")
        )
        progress_html = str(event[3]).lower()
        if cache_hit is True and "cached" in progress_html and "rerank" in progress_html:
            matches.append(event)
    return matches


def _require_reranker_cache_behavior(
    *,
    reranker_wait_events: list[list[object]],
    reranker_cache_events: list[list[object]],
    require_cache: bool,
    require_no_wait: bool,
) -> None:
    if require_cache and not reranker_cache_events:
        raise SystemExit("Expected visible reranker cache-hit progress events on the repeated app-smoke request.")
    if require_no_wait and reranker_wait_events:
        raise SystemExit(
            "Expected repeated app-smoke request to skip hosted reranker wait events, "
            f"got {len(reranker_wait_events)}."
        )


def _format_ms(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


def _source_ids_for_required_citations(source_text: str, citations: tuple[str, ...]) -> list[str]:
    if not citations:
        return []
    required = [_normalize(citation) for citation in citations]
    source_ids: list[str] = []
    for line in source_text.splitlines():
        normalized_line = _normalize(line)
        if any(citation in normalized_line for citation in required):
            source_id = _source_id_from_line(line)
            if source_id:
                source_ids.append(source_id)
    return source_ids


def _source_ids_from_source_text(source_text: str) -> list[str]:
    return [
        source_id
        for line in source_text.splitlines()
        if (source_id := _source_id_from_line(line))
    ]


def _source_id_from_line(line: str) -> str:
    line = line.strip()
    if not line.startswith("- ["):
        return ""
    end = line.find("]")
    if end <= 3:
        return ""
    return line[3:end]


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _app_url(settings) -> str:
    host = settings.gradio_server_name
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{settings.gradio_server_port}/"


if __name__ == "__main__":
    main()
