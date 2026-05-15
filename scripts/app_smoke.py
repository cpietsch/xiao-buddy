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


def main() -> None:
    settings = load_settings()
    base_url = os.environ.get("APP_SMOKE_URL") or _app_url(settings)
    query = os.environ.get("APP_SMOKE_QUERY", DEFAULT_QUERY)
    terms = _csv_env("APP_SMOKE_MUST_INCLUDE", DEFAULT_TERMS)
    citations = _csv_env("APP_SMOKE_MUST_CITE", DEFAULT_CITATIONS)
    timeout = float(os.environ.get("APP_SMOKE_TIMEOUT_SECONDS", settings.request_timeout_seconds or 90))
    require_stream = os.environ.get("APP_SMOKE_REQUIRE_STREAM", "1") == "1"

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
    if require_stream:
        stream_events = [
            event
            for event in events
            if len(event) > 2
            and isinstance(event[2], dict)
            and (event[2].get("agent_streamed") or event[2].get("agent", {}).get("streamed"))
        ]
        if len(stream_events) < 2:
            raise SystemExit(f"Expected multiple streamed answer events, got {len(stream_events)}.")
    if diagnostics.get("status") != "ok":
        raise SystemExit(f"Expected final diagnostics status=ok, got {diagnostics.get('status')!r}.")
    if not diagnostics.get("agent_used"):
        raise SystemExit("Expected final diagnostics to report agent_used=true.")
    if "Ready" not in progress_html:
        raise SystemExit("Expected final progress HTML to include Ready.")

    print(
        "PASS app smoke: "
        f"events={len(events)} "
        f"chars={len(answer)} "
        f"sources={source_text.count('- [')} "
        f"total_ms={diagnostics.get('timings_ms', {}).get('total', 0)}"
    )


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


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _app_url(settings) -> str:
    host = settings.gradio_server_name
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{settings.gradio_server_port}/"


if __name__ == "__main__":
    main()
