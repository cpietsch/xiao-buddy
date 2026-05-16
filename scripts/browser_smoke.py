from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.config import load_settings


ROOT = Path(__file__).resolve().parents[1]


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
