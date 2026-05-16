from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.config import load_settings


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERY = "Which XIAO should I choose for 5 GHz WiFi?"
DEFAULT_TERMS = ("XIAO ESP32-C5", "5 GHz", "Wi-Fi 6")
DEFAULT_CITATIONS = ("https://wiki.seeedstudio.com/xiao_esp32c5_wifi_usage/",)
DEFAULT_EVAL_PATH = Path("data/corpus/answer_eval_queries.jsonl")


def main() -> None:
    args = _parse_args()
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required for browser smoke. Install dev dependencies with "
            "`pip install -r requirements-dev.txt` and then run "
            "`python -m playwright install chromium`."
        ) from exc

    settings = load_settings()
    url = args.url or _app_url(settings)
    screenshot_dir = _resolve_output_dir(args.screenshot_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    query, terms, citations, case_id = _load_query(args)

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - surface browser-install guidance.
            raise SystemExit(
                "Chromium is required for browser smoke. Run "
                "`python -m playwright install chromium` and retry."
            ) from exc
        with browser:
            results = []
            for name, width, height in (
                ("desktop", 1440, 980),
                ("mobile", 390, 844),
            ):
                page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
                try:
                    results.append(_check_page(page, url, screenshot_dir, name, args.timeout_ms))
                    if args.run_query and name == "desktop":
                        results.append(
                            _check_query_interaction(
                                page,
                                screenshot_dir,
                                query=query,
                                terms=terms,
                                citations=citations,
                                case_id=case_id,
                                timeout_ms=args.timeout_ms,
                            )
                        )
                except PlaywrightTimeoutError as exc:
                    raise SystemExit(f"{name} browser smoke timed out: {exc}") from exc
                finally:
                    page.close()

    for result in results:
        print(result)
    print("PASS browser smoke")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the live Gradio app in Chromium and check UI basics.")
    parser.add_argument("--url", default="", help="App URL. Defaults to the configured Gradio URL.")
    parser.add_argument("--screenshot-dir", default="dist/browser-smoke", help="Directory for screenshots.")
    parser.add_argument("--timeout-ms", type=int, default=30000, help="Playwright timeout in milliseconds.")
    parser.add_argument("--run-query", action="store_true", help="Submit a real browser query on the desktop viewport.")
    parser.add_argument("--query", default="", help="Query to use with --run-query.")
    parser.add_argument(
        "--must-include",
        action="append",
        default=[],
        help="Required answer/body term for --run-query. May be passed more than once.",
    )
    parser.add_argument(
        "--must-cite",
        action="append",
        default=[],
        help="Required rendered citation URL for --run-query. May be passed more than once.",
    )
    parser.add_argument("--case-id", default="", help="Load query and required terms from answer eval JSONL.")
    parser.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH), help="Answer eval JSONL path for --case-id.")
    return parser.parse_args()


def _check_page(page, url: str, screenshot_dir: Path, name: str, timeout_ms: int) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_selector("h1", timeout=timeout_ms)
    page.wait_for_function(
        "() => document.body && document.body.innerText.toLowerCase().includes('support bench')",
        timeout=timeout_ms,
    )

    title = page.title()
    _assert("XIAO Buddy" in title, f"{name}: page title should include XIAO Buddy, got {title!r}")
    _assert(page.locator("h1", has_text="XIAO Buddy").count() == 1, f"{name}: expected one XIAO Buddy h1")
    _assert(page.locator(".skip-link").count() == 1, f"{name}: skip link missing")
    _assert(page.locator('[role="status"][aria-live="polite"][aria-atomic="true"]').count() >= 1, f"{name}: progress live region missing")
    _assert(page.locator(".progress-step").count() == 4, f"{name}: expected four progress steps")
    _assert(page.locator(".chip", has_text="Raspberry Pi HATs").count() == 1, f"{name}: Raspberry Pi chip missing")
    _assert(page.locator(".chip", has_text="Jetson robotics").count() == 1, f"{name}: Jetson chip missing")
    _assert(page.locator("textarea").count() >= 1, f"{name}: question textarea missing")

    page.keyboard.press("Tab")
    focused = page.evaluate("() => document.activeElement && document.activeElement.classList.contains('skip-link')")
    _assert(bool(focused), f"{name}: first Tab should focus skip link")

    metrics = page.evaluate(
        """
        () => {
          const root = document.documentElement;
          const bodyText = document.body.innerText || "";
          const textarea = document.querySelector("textarea");
          const buttons = Array.from(document.querySelectorAll("button"))
            .map((button) => {
              const rect = button.getBoundingClientRect();
              return { text: button.innerText, width: rect.width, height: rect.height };
            });
          return {
            bodyText,
            overflowX: Math.max(root.scrollWidth, document.body.scrollWidth) - window.innerWidth,
            textareaFontSize: textarea ? parseFloat(getComputedStyle(textarea).fontSize) : 0,
            buttons,
          };
        }
        """
    )
    for term in ("Seeed", "robotics", "LoRa", "Raspberry Pi", "Jetson"):
        _assert(term in metrics["bodyText"], f"{name}: rendered body should mention {term}")
    _assert(metrics["overflowX"] <= 2, f"{name}: horizontal overflow is {metrics['overflowX']}px")
    _assert(metrics["textareaFontSize"] >= 16, f"{name}: textarea font should be at least 16px")
    for button in metrics["buttons"]:
        label = str(button.get("text") or "").strip()
        if label in {"Ask Buddy", "Reset"}:
            _assert(button["width"] >= 44 and button["height"] >= 44, f"{name}: {label} target is too small")

    screenshot_path = screenshot_dir / f"{name}.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    _assert(_is_nonblank_image(screenshot_path), f"{name}: screenshot appears blank")
    return f"OK   {name}: {screenshot_path.relative_to(ROOT)}"


def _check_query_interaction(
    page,
    screenshot_dir: Path,
    *,
    query: str,
    terms: tuple[str, ...],
    citations: tuple[str, ...],
    case_id: str,
    timeout_ms: int,
) -> str:
    page.get_by_label("Hardware question").fill(query)
    page.get_by_role("button", name="Ask Buddy").click()
    page.wait_for_function(
        """
        () => {
          const text = document.body.innerText || "";
          return text.includes("Preparing the request")
            || text.includes("Retrieving relevant Seeed wiki sources")
            || text.includes("Generating a cited answer");
        }
        """,
        timeout=timeout_ms,
    )
    page.wait_for_function(
        """
        (terms) => {
          const normalize = (value) => (value || "").toLowerCase().replace(/[^a-z0-9]/g, "");
          const text = document.body.innerText || "";
          const normalized = normalize(text);
          return terms.every((term) => normalized.includes(normalize(term)))
            && text.includes("streamed")
            && text.includes("agent answer")
            && text.includes("Sources used")
            && text.includes("Ready");
        }
        """,
        arg=list(terms),
        timeout=timeout_ms,
    )

    body_text = page.evaluate("() => document.body.innerText || ''")
    _require_terms("browser answer", body_text, terms)
    _require_terms("browser citations", body_text, citations)
    _require_inline_citation(body_text, citations)
    _assert("RUN PROGRESS" in body_text, "browser answer should keep progress visible")
    _assert("FIELD ANSWER" in body_text, "browser answer should keep answer panel visible")
    _assert("Sources used" in body_text, "browser answer should keep source trail visible")
    _assert("streamed" in body_text, "browser answer should expose streamed progress")
    _assert("agent answer" in body_text, "browser answer should expose final agent status")

    screenshot_path = screenshot_dir / "desktop-after-query.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    _assert(_is_nonblank_image(screenshot_path), "desktop-after-query: screenshot appears blank")
    case_label = case_id or "custom"
    return f"OK   desktop query {case_label}: {screenshot_path.relative_to(ROOT)}"


def _load_query(args: argparse.Namespace) -> tuple[str, tuple[str, ...], tuple[str, ...], str]:
    case_id = args.case_id or os.environ.get("BROWSER_SMOKE_CASE_ID", "").strip()
    if case_id:
        case = _load_answer_eval_case(Path(args.eval_path), case_id)
        return (
            str(case["query"]),
            tuple(str(term) for term in case.get("must_include", [])),
            tuple(str(citation) for citation in case.get("must_cite", [])),
            case_id,
        )
    query = args.query or os.environ.get("BROWSER_SMOKE_QUERY", DEFAULT_QUERY)
    terms = tuple(args.must_include) if args.must_include else _csv_env("BROWSER_SMOKE_MUST_INCLUDE", DEFAULT_TERMS)
    citations = tuple(args.must_cite) if args.must_cite else _csv_env("BROWSER_SMOKE_MUST_CITE", DEFAULT_CITATIONS)
    return query, terms, citations, ""


def _load_answer_eval_case(path: Path, case_id: str) -> dict[str, object]:
    if not path.is_absolute():
        path = ROOT / path
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if case.get("id") == case_id:
            return case
    raise SystemExit(f"Browser smoke case {case_id!r} was not found in {path}.")


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _require_terms(label: str, text: str, terms: tuple[str, ...]) -> None:
    normalized = _normalize(text)
    missing = [term for term in terms if _normalize(term) not in normalized]
    if missing:
        raise AssertionError(f"{label} missing required terms: {', '.join(missing)}")


def _require_inline_citation(body_text: str, citations: tuple[str, ...]) -> None:
    source_ids = _source_ids_for_required_citations(body_text, citations)
    if not source_ids:
        source_ids = _source_ids_from_source_text(body_text)
    answer_text = body_text.split("Sources used", 1)[0]
    if not any(f"[{source_id}]" in answer_text for source_id in source_ids):
        expected = ", ".join(f"[{source_id}]" for source_id in source_ids) or "a rendered source id"
        raise AssertionError(f"browser answer missing inline citation for {expected}")


def _source_ids_for_required_citations(body_text: str, citations: tuple[str, ...]) -> list[str]:
    if not citations:
        return []
    required = [_normalize(citation) for citation in citations]
    source_ids: list[str] = []
    for line in body_text.splitlines():
        normalized_line = _normalize(line)
        if any(citation in normalized_line for citation in required):
            source_id = _source_id_from_line(line)
            if source_id:
                source_ids.append(source_id)
    return source_ids


def _source_ids_from_source_text(body_text: str) -> list[str]:
    return [
        source_id
        for line in body_text.splitlines()
        if (source_id := _source_id_from_line(line))
    ]


def _source_id_from_line(line: str) -> str:
    line = line.strip()
    if line.startswith("- ["):
        line = line[2:].strip()
    if not line.startswith("["):
        return ""
    end = line.find("]")
    if end <= 1:
        return ""
    return line[1:end]


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _resolve_output_dir(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _is_nonblank_image(path: Path) -> bool:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((64, 64))
        colors = image.getcolors(maxcolors=4096)
        if not colors:
            return True
        return len(colors) > 8


def _app_url(settings) -> str:
    host = "127.0.0.1" if settings.gradio_server_name == "0.0.0.0" else settings.gradio_server_name
    return f"http://{host}:{settings.gradio_server_port}/"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
