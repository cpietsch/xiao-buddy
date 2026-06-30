from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://100.103.106.102:5173/"
DEFAULT_QUERY = "What SPI pins does the XIAO W5500 Ethernet Adapter use?"


def main() -> None:
    args = _parse_args()
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required for frontend smoke. Install dev dependencies with "
            "`pip install -r requirements-dev.txt` and then run "
            "`python -m playwright install chromium`."
        ) from exc

    url = args.url or os.environ.get("FRONTEND_SMOKE_URL", DEFAULT_URL)
    screenshot_dir = _resolve_output_dir(args.screenshot_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    query = args.query or os.environ.get("FRONTEND_SMOKE_QUERY", DEFAULT_QUERY)

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - surface browser-install guidance.
            raise SystemExit(
                "Chromium is required for frontend smoke. Run "
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
                                timeout_ms=args.timeout_ms,
                            )
                        )
                except PlaywrightTimeoutError as exc:
                    raise SystemExit(f"{name} frontend smoke timed out: {exc}") from exc
                finally:
                    page.close()

    for result in results:
        print(result)
    print("PASS frontend smoke")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the live React frontend in Chromium and check UI basics.")
    parser.add_argument("--url", default="", help="Frontend URL. Defaults to FRONTEND_SMOKE_URL or the Tailscale dev URL.")
    parser.add_argument("--screenshot-dir", default="dist/frontend-smoke", help="Directory for screenshots.")
    parser.add_argument("--timeout-ms", type=int, default=120000, help="Playwright timeout in milliseconds.")
    parser.add_argument("--run-query", action="store_true", help="Submit a real browser query on the desktop viewport.")
    parser.add_argument("--query", default="", help="Query to use with --run-query.")
    return parser.parse_args()


def _check_page(page, url: str, screenshot_dir: Path, name: str, timeout_ms: int) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_selector("h1", timeout=timeout_ms)
    page.wait_for_function(
        "() => document.body && document.body.innerText.toLowerCase().includes('project brief')",
        timeout=timeout_ms,
    )

    title = page.title()
    _assert(title == "Seeed Project Workbench", f"{name}: page title should be Seeed Project Workbench, got {title!r}")
    _assert(page.locator("h1", has_text="Seeed Project Workbench").count() == 1, f"{name}: expected one Seeed Project Workbench h1")
    _assert(page.locator(".skip-link").count() == 1, f"{name}: skip link missing")
    _assert(page.locator('section[aria-label="Run progress"]').count() == 1, f"{name}: progress region missing")
    _assert(page.locator(".progress-step").count() == 4, f"{name}: expected four progress steps")
    _assert(page.get_by_label("Answer").count() == 1, f"{name}: answer region missing")
    _assert(page.get_by_label("Workbench prompt").count() == 1, f"{name}: question textarea missing")
    _assert(page.locator(".project-rail").count() == 1, f"{name}: project rail missing")
    _assert(page.locator(".domain-strip").count() == 0, f"{name}: header domain pills should be removed")

    page.keyboard.press("Tab")
    _assert_focused_skip_link_visible(page, name)

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
    for term in ("Seeed", "robotics", "LoRa", "firmware recovery", "field troubleshooting"):
        _assert(term in metrics["bodyText"], f"{name}: rendered body should mention {term}")
    _assert("Run progress" not in metrics["bodyText"], f"{name}: Run progress should not be visible text")
    _assert("Field answer" not in metrics["bodyText"], f"{name}: Field answer should not be visible text")
    _assert(metrics["overflowX"] <= 2, f"{name}: horizontal overflow is {metrics['overflowX']}px")
    _assert(metrics["textareaFontSize"] >= 16, f"{name}: textarea font should be at least 16px")
    for button in metrics["buttons"]:
        label = str(button.get("text") or "").strip()
        if label in {"Ask Workbench", "Stop", "Reset", "Theme"}:
            _assert(button["width"] >= 44 and button["height"] >= 44, f"{name}: {label} target is too small")

    screenshot_path = screenshot_dir / f"{name}.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    _assert(_is_nonblank_image(screenshot_path), f"{name}: screenshot appears blank")
    return f"OK   {name}: {screenshot_path.relative_to(ROOT)}"


def _check_query_interaction(page, screenshot_dir: Path, *, query: str, timeout_ms: int) -> str:
    request_urls: list[str] = []
    page.on("request", lambda request: request_urls.append(request.url))
    page.get_by_label("Workbench prompt").fill(query)
    page.get_by_role("button", name="Ask Workbench").click()
    page.wait_for_function(
        """
        () => {
          const text = document.body.innerText || "";
          return text.includes("Agent answer") && text.includes("Source") && text.includes("total");
        }
        """,
        timeout=timeout_ms,
    )
    page.wait_for_selector(".progress-chips .run-chip", timeout=timeout_ms)

    body_text = page.evaluate("() => document.body.innerText || ''")
    chips = [text.strip() for text in page.locator(".progress-chips .run-chip").all_inner_texts()]
    source_title = page.locator(".source-snippets-title").inner_text(timeout=timeout_ms)
    first_source_title = page.locator(".source-snippet-title").first.inner_text(timeout=timeout_ms)
    source_count = page.locator(".source-snippet").count()

    _assert(any(chip in {"Focused evidence", "Full evidence", "Fixed evidence"} for chip in chips), f"missing evidence chip: {chips}")
    _assert(any(re.search(r"\d+(?:/\d+)? sources", chip) for chip in chips), f"missing source-count chip: {chips}")
    _assert(source_title.startswith(("Focused sources", "Full sources", "Sources")), f"unexpected source panel title: {source_title}")
    _assert(source_count >= 1, "source snippets should render after query")
    if query == DEFAULT_QUERY:
        _assert("pin map" in first_source_title.lower(), f"default pinout query should lead with pin map source, got {first_source_title!r}")
    _assert(page.locator(".source-snippet-markdown").count() >= 1, "source snippets should use Markdown rendering")
    _assert(page.locator(".chat-message-user").count() >= 1, "submitted question should render as a user chat message")
    _assert(page.locator(".chat-message-assistant").count() >= 1, "answer should render as an assistant chat message")
    _assert(page.locator('.answer-markdown a[href^="#source-snippet-"]').count() >= 1, "answer should link to source panel")
    _assert(any("/api/ask" in url for url in request_urls), "frontend should call the direct /api/ask stream")
    _assert(not any("/gradio_api" in url for url in request_urls), "frontend should not call the legacy Gradio API")
    _assert("Run progress" not in body_text, "Run progress should not be visible text after query")
    _assert("Field answer" not in body_text, "Field answer should not be visible text after query")
    _reject_visible_source_ids(body_text)
    _reject_standalone_source_label_lines(body_text)

    overflow = page.evaluate("() => Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - window.innerWidth")
    _assert(overflow <= 2, f"desktop-after-query: horizontal overflow is {overflow}px")
    screenshot_path = screenshot_dir / "desktop-after-query.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    _assert(_is_nonblank_image(screenshot_path), "desktop-after-query: screenshot appears blank")
    return f"OK   desktop query: {screenshot_path.relative_to(ROOT)}; chips={chips}; {source_title}"


def _assert_focused_skip_link_visible(page, name: str) -> None:
    state = page.evaluate(
        """
        () => {
          const skip = document.querySelector(".skip-link");
          const rect = skip ? skip.getBoundingClientRect() : null;
          const style = skip ? getComputedStyle(skip) : null;
          return {
            focused: document.activeElement === skip,
            visible: Boolean(rect && rect.bottom > 0 && rect.top < window.innerHeight),
            outlineWidth: style ? parseFloat(style.outlineWidth) : 0,
          };
        }
        """
    )
    _assert(state["focused"], f"{name}: first Tab should focus skip link")
    _assert(state["visible"], f"{name}: focused skip link should be visible immediately")
    _assert(state["outlineWidth"] >= 2, f"{name}: focused skip link should have a visible focus ring")


def _reject_visible_source_ids(body_text: str) -> None:
    match = re.search(
        r"\b(?:wiki-[0-9a-f]{8,}|field-[A-Za-z0-9_.:-]+|xiao-[A-Za-z0-9_.:-]+-(?:identity|pinout|gotchas))\b",
        body_text,
        flags=re.IGNORECASE,
    )
    if match:
        raise AssertionError(f"browser output still exposes raw source id {match.group(0)}")


def _reject_standalone_source_label_lines(body_text: str) -> None:
    for line in body_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("sources:"):
            continue
        if re.fullmatch(r"source \d+(?:\s*(?:,|;|and)\s*source \d+)+", stripped, flags=re.IGNORECASE):
            raise AssertionError(f"browser output has an orphan source-only line: {stripped}")


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


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
