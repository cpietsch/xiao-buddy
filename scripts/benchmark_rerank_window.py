from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.config import load_settings
from xiao_copilot.retrieval import retrieve


DEFAULT_WINDOWS = (900, 1600, 2400, 3200)


def main() -> None:
    args = _parse_args()
    cases = _load_cases(args.eval_path)
    if args.case:
        selected = set(args.case)
        cases = [case for case in cases if case["id"] in selected]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No benchmark cases selected.")

    settings = load_settings()
    print(f"cases={len(cases)} windows={','.join(str(window) for window in args.windows)}", flush=True)
    print("window cases board content citation rerank p50_ms p95_ms max_ms total_s failures", flush=True)

    for window in args.windows:
        print(f"running window={window}", flush=True)
        window_started_at = perf_counter()
        window_settings = replace(settings, rerank_text_chars=window)
        results = [_run_case(case, window_settings) for case in cases]
        rerank_ms = [result["rerank_ms"] for result in results if result["reranker_used"]]
        failures = [result["id"] for result in results if not result["ok"]]
        board_hits = sum(1 for result in results if result["board_ok"])
        content_hits = sum(1 for result in results if result["content_ok"])
        citation_hits = sum(1 for result in results if result["citation_ok"])
        rerank_hits = sum(1 for result in results if result["reranker_used"])
        print(
            f"{window} "
            f"{len(results)} "
            f"{board_hits}/{len(results)} "
            f"{content_hits}/{len(results)} "
            f"{citation_hits}/{len(results)} "
            f"{rerank_hits}/{len(results)} "
            f"{_percentile(rerank_ms, 50):.1f} "
            f"{_percentile(rerank_ms, 95):.1f} "
            f"{(max(rerank_ms) if rerank_ms else 0.0):.1f} "
            f"{perf_counter() - window_started_at:.1f} "
            f"{','.join(failures) if failures else '-'}",
            flush=True,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark retrieval quality and reranker latency by text window.")
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=Path("data/corpus/eval_queries.jsonl"),
        help="JSONL retrieval eval file.",
    )
    parser.add_argument(
        "--windows",
        type=_parse_windows,
        default=DEFAULT_WINDOWS,
        help="Comma-separated rerank text window sizes.",
    )
    parser.add_argument("--case", action="append", default=[], help="Only run a specific eval case id. Repeatable.")
    parser.add_argument("--limit", type=int, default=0, help="Limit cases after filtering.")
    return parser.parse_args()


def _parse_windows(value: str) -> tuple[int, ...]:
    windows = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not windows:
        raise argparse.ArgumentTypeError("At least one window must be provided.")
    if any(window < 200 for window in windows):
        raise argparse.ArgumentTypeError("Window sizes must be at least 200 characters.")
    return windows


def _load_cases(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run_case(case: dict[str, object], settings) -> dict[str, object]:
    chunks, diagnostics = retrieve(str(case["query"]), settings)
    retrieved = {chunk.board_id for chunk in chunks}
    expected_board_id = str(case.get("expected_board_id", ""))
    board_ok = not expected_board_id or expected_board_id in retrieved
    content_ok = _contains_all_terms(chunks, list(case.get("must_include", [])))
    citation_ok = _contains_any_citation(chunks, list(case.get("must_cite", [])))
    return {
        "id": case["id"],
        "ok": board_ok and content_ok and citation_ok,
        "board_ok": board_ok,
        "content_ok": content_ok,
        "citation_ok": citation_ok,
        "reranker_used": bool(diagnostics.get("reranker_used")),
        "rerank_ms": float(diagnostics.get("reranker_ms") or 0.0),
    }


def _contains_all_terms(chunks, terms: list[str]) -> bool:
    if not terms:
        return True
    text = _normalize_match_text(
        "\n".join(f"{chunk.title}\n{chunk.source}\n{chunk.text}" for chunk in chunks)
    )
    return all(_normalize_match_text(term) in text for term in terms)


def _contains_any_citation(chunks, citations: list[str]) -> bool:
    if not citations:
        return True
    haystack: list[str] = []
    for chunk in chunks:
        haystack.append(chunk.source)
        for citation in chunk.metadata.get("citations", []):
            if isinstance(citation, dict):
                haystack.append(str(citation.get("url", "")))
    source_text = _normalize_match_text("\n".join(haystack))
    return any(_normalize_match_text(citation) in source_text for citation in citations)


def _normalize_match_text(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[percentile - 1]


if __name__ == "__main__":
    main()
