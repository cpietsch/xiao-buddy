from __future__ import annotations

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from xiao_copilot.config import load_settings
from xiao_copilot.pipeline import format_progress


class ProgressParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.attrs: dict[str, str] = {}
        self.text: list[str] = []
        self.progress_steps = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        attr_map = {key: value or "" for key, value in attrs}
        if tag == "div" and "progress-panel" in attr_map.get("class", ""):
            self.attrs = attr_map
        if tag == "li" and "progress-step" in attr_map.get("class", ""):
            self.progress_steps += 1

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self.text.append(stripped)


def main() -> None:
    args = _parse_args()
    _check_visible_scope()
    _check_accessibility_contract()
    _check_progress_contract()
    if args.live:
        _check_live_app(args.timeout)
    print("PASS ui contract smoke")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check XIAO Buddy UI copy and accessibility contract.")
    parser.add_argument("--live", action="store_true", help="Fetch the configured Gradio app root.")
    parser.add_argument("--timeout", type=float, default=8.0, help="Network timeout in seconds.")
    return parser.parse_args()


def _check_visible_scope() -> None:
    examples = "\n".join(str(example[1]) for example in app.EXAMPLES)
    ui_text = "\n".join([app.INITIAL_ANSWER, examples, app.CSS])
    required = {
        "XIAO": ui_text,
        "Seeed sensor": app.INITIAL_ANSWER,
        "robotics kit": app.INITIAL_ANSWER,
        "LoRa module": app.INITIAL_ANSWER,
        "Wio Terminal": examples,
        "Grove Vision AI V2": examples,
        "Raspberry Pi Zero": examples,
        "Reachy Mini": examples,
        "SO-Arm": examples,
    }
    for term, haystack in required.items():
        _assert(term in haystack, f"UI scope should mention {term}")

    _assert(len(app.EXAMPLES) >= 20, "bench scenarios should cover the expanded corpus")


def _check_accessibility_contract() -> None:
    css = app.CSS
    required_css = [
        ".skip-link",
        ":focus-visible",
        "touch-action: manipulation",
        "env(safe-area-inset-left)",
        "env(safe-area-inset-right)",
        "font-size: 16px",
        "font-variant-numeric: tabular-nums",
    ]
    for token in required_css:
        _assert(token in css, f"CSS accessibility hook missing: {token}")


def _check_progress_contract() -> None:
    for stage in ("waiting", "prepare", "retrieve", "generate", "done"):
        parser = ProgressParser()
        parser.feed(format_progress(stage=stage, backend="hnsw", source_count=5, agent_used=True, stream_chars=123))
        _assert(parser.attrs.get("role") == "status", f"{stage} progress should use role=status")
        _assert(parser.attrs.get("aria-live") == "polite", f"{stage} progress should be polite live region")
        _assert(parser.attrs.get("aria-atomic") == "true", f"{stage} progress should be atomic")
        _assert("ol" in parser.tags, f"{stage} progress should use an ordered list")
        _assert(parser.progress_steps == 4, f"{stage} progress should expose four steps")
    done_html = format_progress(stage="done", backend="hnsw", source_count=5, agent_used=True, stream_chars=123)
    _assert("Ready" in done_html, "done progress should show Ready")
    _assert("agent answer" in done_html, "done progress should identify the answer path")


def _check_live_app(timeout: float) -> None:
    settings = load_settings()
    host = "127.0.0.1" if settings.gradio_server_name == "0.0.0.0" else settings.gradio_server_name
    url = f"http://{host}:{settings.gradio_server_port}/"
    response = requests.get(url, timeout=timeout)
    _assert(200 <= response.status_code < 400, f"live app returned {response.status_code}: {url}")
    _assert("XIAO Buddy" in response.text, "live app should include the page title")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
