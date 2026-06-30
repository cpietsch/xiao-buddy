from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.app_smoke import (
    _agent_wait_heartbeat_events,
    _first_token_ms,
    _format_ms,
    _load_case_from_env,
    _require_first_token_latency,
    _require_inline_citation,
    _require_reranker_cache_behavior,
    _require_terms,
    _reranker_cache_hit_events,
    _reranker_wait_events,
    _source_draft_events,
    _source_label_count,
    _stream_events_with_first_token,
)
from xiao_copilot.config import load_settings


def main() -> None:
    settings = load_settings()
    base_url = os.environ.get("API_SMOKE_URL") or _app_url(settings)
    query, terms, citations, case_id = _load_case_from_env()
    timeout = float(os.environ.get("API_SMOKE_TIMEOUT_SECONDS", settings.request_timeout_seconds or 90))
    require_reranker_cache = os.environ.get("API_SMOKE_REQUIRE_RERANK_CACHE", "0") == "1"
    require_no_reranker_wait = os.environ.get("API_SMOKE_REQUIRE_NO_RERANK_WAIT", "0") == "1"

    events = list(_stream_direct_api(base_url, query, timeout=timeout))
    if not events:
        raise SystemExit("API smoke did not receive any stream events.")

    final = events[-1]
    answer = str(final[0] if len(final) > 0 else "")
    source_text = str(final[1] if len(final) > 1 else "")
    diagnostics = final[2] if len(final) > 2 and isinstance(final[2], dict) else {}
    progress_html = str(final[3] if len(final) > 3 else "")

    _require_terms("answer", answer, terms)
    _require_terms("source snippets", source_text, citations)
    _reject_orphan_source_lines(answer)
    _require_inline_citation(answer, source_text, citations)

    draft_events = _source_draft_events(events)
    if not draft_events:
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
    if not _stream_events_with_first_token(stream_events):
        raise SystemExit("Expected first-token diagnostics during streamed answer events.")

    first_token_ms = _first_token_ms(diagnostics)
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
        "PASS api smoke: "
        f"events={len(events)} "
        f"case={case_id or 'custom'} "
        f"chars={len(answer)} "
        f"sources={_source_label_count(source_text)} "
        f"source_draft_events={len(draft_events)} "
        f"reranker_wait_events={len(reranker_wait_events)} "
        f"reranker_cache_events={len(reranker_cache_events)} "
        f"heartbeat_events={len(_agent_wait_heartbeat_events(events))} "
        f"first_token_ms={_format_ms(first_token_ms)} "
        f"total_ms={diagnostics.get('timings_ms', {}).get('total', 0)}"
    )


def _stream_direct_api(base_url: str, query: str, *, timeout: float):
    url = urljoin(base_url.rstrip("/") + "/", "api/ask")
    current_event = ""
    with requests.post(
        url,
        json={"image": None, "question": query},
        stream=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith("event:"):
                current_event = line.removeprefix("event:").strip()
                continue
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if current_event == "complete":
                return
            if current_event == "error":
                raise SystemExit(f"API smoke received an error event: {data[:500]}")
            value = json.loads(data)
            if isinstance(value, list):
                yield value


def _reject_orphan_source_lines(answer: str) -> None:
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("sources:"):
            continue
        if stripped.lower().startswith("source ") and "source " in stripped.lower()[7:]:
            raise SystemExit(f"answer has an orphan source-only line: {stripped}")


def _app_url(settings) -> str:
    host = settings.gradio_server_name
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{settings.gradio_server_port}/"


if __name__ == "__main__":
    main()
