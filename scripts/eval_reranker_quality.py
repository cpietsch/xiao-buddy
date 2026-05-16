from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.clients import rerank
from xiao_copilot.config import load_settings
from xiao_copilot.knowledge_base import KnowledgeChunk, load_knowledge_base
from xiao_copilot.retrieval import (
    _include_rerank_board_metadata,
    _include_rerank_source_topic_metadata,
    _is_weekly_wiki_source,
    _lexical_score,
    _rerank_text,
)


DEFAULT_CASE_IDS = (
    "eval-pin-i2c-c3",
    "eval-c5-dual-band",
    "eval-robotics-bus-servo-adapter-c3",
    "eval-grove-vision-ai-v2-trigger-actions",
    "eval-rpi-grove-base-hat-zero-specs",
    "eval-jetson-grove-pihat-sensors",
    "eval-robotics-reachy-fleet-ports",
    "eval-sensecap-device-management-node-fields",
)

CATEGORY_PREFIXES = (
    ("robotics", "eval-robotics-"),
    ("grove", "eval-grove-"),
    ("raspberry-pi", "eval-rpi-"),
    ("jetson", "eval-jetson-"),
    ("sensecap", "eval-sensecap-"),
    ("wio", "eval-wio-"),
    ("edge-ai", "eval-edge-"),
    ("xiao-advanced", "eval-adv-"),
    ("xiao-wiki", "eval-wiki-"),
)


def main() -> None:
    args = _parse_args()
    settings = load_settings()
    if not settings.rerank_base_url:
        raise SystemExit("RERANK_BASE_URL is required for reranker quality evaluation.")

    cases = _load_cases(args.eval_path, args.case_ids, use_all=args.all_cases)
    if args.limit:
        cases = cases[: args.limit]
    corpus = load_knowledge_base()
    print(
        f"cases={len(cases)} positives={args.positives} negatives={args.negatives} "
        f"min_margin={args.min_margin:.4f} min_pass_rate={args.min_pass_rate:.0%} "
        f"max_failures={args.max_failures}",
        flush=True,
    )

    results = [
        _run_case(case, corpus, settings, args.positives, args.negatives, args.min_margin)
        for case in cases
    ]
    summary = _summarize_results(results, close_margin=args.close_margin)
    _print_summary(summary)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps({"summary": summary, "results": results}, indent=2) + "\n")
        print(f"wrote {args.json_output}", flush=True)

    failures = [str(result["id"]) for result in results if not result["ok"]]
    threshold_failed = (
        summary["pass_rate"] < args.min_pass_rate or len(failures) > args.max_failures
    )
    if threshold_failed:
        if failures:
            print(f"\nfailures: {', '.join(failures)}", flush=True)
        if args.warn_only:
            print("warning: reranker quality thresholds were not met", flush=True)
            return
        raise SystemExit(1)


def _summarize_results(results: list[dict[str, Any]], close_margin: float = 0.05) -> dict[str, Any]:
    passes = [result for result in results if result["ok"]]
    margins = [float(result["margin"]) for result in results]
    latencies = [float(result["ms"]) for result in results]
    by_category: dict[str, dict[str, Any]] = {}
    for result in results:
        category = str(result["category"])
        category_summary = by_category.setdefault(
            category,
            {"cases": 0, "passes": 0, "failures": [], "min_margin": None},
        )
        category_summary["cases"] += 1
        category_summary["passes"] += int(bool(result["ok"]))
        if not result["ok"]:
            category_summary["failures"].append(result["id"])
        min_margin = category_summary["min_margin"]
        margin = float(result["margin"])
        category_summary["min_margin"] = margin if min_margin is None else min(float(min_margin), margin)

    for category_summary in by_category.values():
        cases = int(category_summary["cases"])
        passes_count = int(category_summary["passes"])
        category_summary["pass_rate"] = passes_count / cases if cases else 0.0
        if category_summary["min_margin"] is None:
            category_summary["min_margin"] = 0.0
    close_cases = [
        {
            "id": str(result["id"]),
            "category": str(result["category"]),
            "margin": float(result["margin"]),
            "positive_id": str(result.get("positive_id", "")),
            "best_negative_id": str(result.get("best_negative_id", "")),
            "positive_title": str(result.get("positive_title", "")),
            "best_negative_title": str(result.get("best_negative_title", "")),
        }
        for result in sorted(results, key=lambda item: float(item["margin"]))
        if bool(result["ok"]) and 0 <= float(result["margin"]) < close_margin
    ]

    return {
        "cases": len(results),
        "passes": len(passes),
        "failures": [str(result["id"]) for result in results if not result["ok"]],
        "pass_rate": len(passes) / len(results) if results else 0.0,
        "close_margin": close_margin,
        "close_cases": close_cases,
        "min_margin": min(margins) if margins else 0.0,
        "p50_margin": _percentile(margins, 50),
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "max_ms": max(latencies) if latencies else 0.0,
        "by_category": by_category,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        "\nreranker top-positive rate: "
        f"{summary['passes']}/{summary['cases']} = {summary['pass_rate']:.0%}",
        flush=True,
    )
    print(
        f"reranker margin: min={summary['min_margin']:.4f} p50={summary['p50_margin']:.4f}",
        flush=True,
    )
    print(
        f"reranker latency: p50_ms={summary['p50_ms']:.1f} "
        f"p95_ms={summary['p95_ms']:.1f} max_ms={summary['max_ms']:.1f}",
        flush=True,
    )
    print("reranker categories:", flush=True)
    for category, category_summary in sorted(summary["by_category"].items()):
        print(
            f"  {category}: {category_summary['passes']}/{category_summary['cases']} "
            f"min_margin={float(category_summary['min_margin']):.4f}",
            flush=True,
        )
    close_cases = list(summary.get("close_cases", []))
    if close_cases:
        print(f"reranker close margins (<{float(summary['close_margin']):.4f}):", flush=True)
        for case in close_cases[:10]:
            print(
                f"  {case['id']}: margin={float(case['margin']):.4f} "
                f"positive={case['positive_id']} negative={case['best_negative_id']}",
                flush=True,
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate live reranker quality on real Seeed wiki positives and hard lexical negatives."
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=Path("data/corpus/eval_queries.jsonl"),
        help="JSONL retrieval eval file used to pick query/citation positives.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Only run a specific eval case id. Repeatable. Defaults to a mixed representative set.",
    )
    parser.add_argument("--all", dest="all_cases", action="store_true", help="Run every eval case.")
    parser.add_argument("--limit", type=int, default=0, help="Limit cases after filtering.")
    parser.add_argument("--positives", type=int, default=3, help="Citation-matched positives per query.")
    parser.add_argument("--negatives", type=int, default=2, help="Hard lexical negatives per query.")
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="Required positive-score margin over the best negative.",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=1.0,
        help="Minimum acceptable top-positive pass rate.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="Maximum acceptable failed cases.",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Print threshold failures without exiting non-zero.",
    )
    parser.add_argument(
        "--close-margin",
        type=float,
        default=0.05,
        help="Report passing cases below this positive-score margin as close calls.",
    )
    parser.add_argument("--json-output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    if args.all_cases and args.case:
        raise SystemExit("--all cannot be combined with --case.")
    if args.positives < 1:
        raise SystemExit("--positives must be at least 1.")
    if args.negatives < 1:
        raise SystemExit("--negatives must be at least 1.")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative.")
    if not 0 <= args.min_pass_rate <= 1:
        raise SystemExit("--min-pass-rate must be between 0 and 1.")
    if args.max_failures < 0:
        raise SystemExit("--max-failures must be non-negative.")
    if args.close_margin < 0:
        raise SystemExit("--close-margin must be non-negative.")
    selected = tuple(args.case or DEFAULT_CASE_IDS)
    args.case_ids = selected
    return args


def _load_cases(path: Path, case_ids: tuple[str, ...], use_all: bool = False) -> list[dict[str, object]]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if use_all:
        return cases
    by_id = {str(case["id"]): case for case in cases}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise SystemExit(f"Unknown eval case ids: {', '.join(missing)}")
    return [by_id[case_id] for case_id in case_ids]


def _run_case(
    case: dict[str, object],
    corpus: list[KnowledgeChunk],
    settings,
    positive_count: int,
    negative_count: int,
    min_margin: float,
) -> dict[str, object]:
    case_id = str(case["id"])
    category = _case_category(case_id)
    query = str(case["query"])
    citations = [str(citation) for citation in case.get("must_cite", [])]
    terms = [str(term) for term in case.get("must_include", [])]
    print(f"RUN  {case_id}", flush=True)
    positives = _select_positives(case, corpus, citations, terms, positive_count)
    negatives = _select_negatives(
        query,
        corpus,
        citations,
        terms,
        str(case.get("expected_board_id", "") or ""),
        positives[0],
        negative_count,
    )
    candidates = [*positives, *negatives]
    include_board_metadata = _include_rerank_board_metadata(query, candidates)
    include_source_topic_metadata = _include_rerank_source_topic_metadata(query)
    documents = [
        _rerank_text(
            chunk,
            settings.rerank_text_chars,
            include_board_metadata=include_board_metadata,
            include_source_topic_metadata=include_source_topic_metadata,
        )
        for chunk in candidates
    ]

    started_at = perf_counter()
    result = rerank(
        base_url=settings.rerank_base_url,
        model=settings.rerank_model,
        query=query,
        documents=documents,
        api_key=settings.rerank_api_key,
        timeout=settings.request_timeout_seconds,
    )
    elapsed_ms = round((perf_counter() - started_at) * 1000, 1)
    if not result.ok:
        print(f"FAIL {case_id}: reranker request failed: {result.error}", flush=True)
        return {"id": case_id, "category": category, "ok": False, "margin": -1.0, "ms": elapsed_ms}

    scores = {index: float(score) for index, score in result.data or []}
    missing = [index for index in range(len(candidates)) if index not in scores]
    if missing:
        print(f"FAIL {case_id}: reranker did not score indexes {missing}", flush=True)
        return {"id": case_id, "category": category, "ok": False, "margin": -1.0, "ms": elapsed_ms}

    positive_indexes = range(len(positives))
    negative_indexes = range(len(positives), len(candidates))
    best_positive_index = max(positive_indexes, key=lambda index: scores[index])
    best_negative_index = max(negative_indexes, key=lambda index: scores[index])
    positive_score = scores[best_positive_index]
    best_negative_score = scores[best_negative_index]
    margin = positive_score - best_negative_score
    ok = margin >= min_margin
    status = "PASS" if ok else "FAIL"
    mode = (result.meta or {}).get("mode", "unknown")
    print(
        f"{status} {case_id}: pos={positive_score:.4f} "
        f"best_neg={best_negative_score:.4f} margin={margin:+.4f} "
        f"mode={mode} ms={elapsed_ms:.1f}",
        flush=True,
    )
    print(f"     positive: {_short_title(candidates[best_positive_index])}", flush=True)
    print(f"     negative: {_short_title(candidates[best_negative_index])}", flush=True)
    return {
        "id": case_id,
        "category": category,
        "ok": ok,
        "margin": margin,
        "ms": elapsed_ms,
        "mode": mode,
        "positive_score": positive_score,
        "best_negative_score": best_negative_score,
        "positive_id": candidates[best_positive_index].id,
        "positive_title": _short_title(candidates[best_positive_index]),
        "positive_ids": [chunk.id for chunk in positives],
        "best_negative_id": candidates[best_negative_index].id,
        "best_negative_title": _short_title(candidates[best_negative_index]),
        "board_metadata": include_board_metadata,
        "source_topic_metadata": include_source_topic_metadata,
    }


def _select_positive(
    case: dict[str, object],
    corpus: list[KnowledgeChunk],
    citations: list[str],
    terms: list[str],
) -> KnowledgeChunk:
    return _select_positives(case, corpus, citations, terms, count=1)[0]


def _select_positives(
    case: dict[str, object],
    corpus: list[KnowledgeChunk],
    citations: list[str],
    terms: list[str],
    count: int,
) -> list[KnowledgeChunk]:
    if not citations:
        raise SystemExit(f"{case['id']} has no must_cite entries for positive selection.")
    query = str(case["query"])
    positives = [chunk for chunk in corpus if _matches_any_citation(chunk, citations)]
    if not positives:
        raise SystemExit(f"{case['id']} has no corpus chunks matching must_cite.")
    non_weekly = [chunk for chunk in positives if not _is_weekly_wiki_source(chunk)]
    positives = non_weekly or positives
    positives.sort(
        key=lambda chunk: (
            _term_hits(chunk, terms),
            _lexical_score(query, chunk.search_text),
            len(chunk.text),
        ),
        reverse=True,
    )
    return positives[:count]


def _select_negatives(
    query: str,
    corpus: list[KnowledgeChunk],
    citations: list[str],
    terms: list[str],
    expected_board_id: str,
    positive: KnowledgeChunk,
    count: int,
) -> list[KnowledgeChunk]:
    positive_source = _source_key(positive)
    negatives = [
        chunk
        for chunk in corpus
        if chunk.id != positive.id
        and not _matches_any_citation(chunk, citations)
        and not _is_answer_equivalent_negative(chunk, terms, expected_board_id)
        and not _is_weekly_wiki_source(chunk)
        and _source_key(chunk) != positive_source
    ]
    negatives.sort(key=lambda chunk: _lexical_score(query, chunk.search_text), reverse=True)
    selected = negatives[:count]
    if len(selected) < count:
        raise SystemExit(f"Could not select {count} hard negatives for query: {query}")
    return selected


def _is_answer_equivalent_negative(chunk: KnowledgeChunk, terms: list[str], expected_board_id: str = "") -> bool:
    if not terms:
        return False
    if expected_board_id and chunk.board_id and chunk.board_id != expected_board_id:
        return False
    return _term_hits(chunk, terms) == len(terms)


def _matches_any_citation(chunk: KnowledgeChunk, citations: list[str]) -> bool:
    haystack = [_normalize_urlish(chunk.source)]
    for citation in chunk.metadata.get("citations", []):
        if isinstance(citation, dict):
            haystack.append(_normalize_urlish(str(citation.get("url", ""))))
    return any(
        needle and any(needle in value for value in haystack)
        for needle in (_normalize_urlish(citation) for citation in citations)
    )


def _source_key(chunk: KnowledgeChunk) -> str:
    source_file = str(chunk.metadata.get("source_file", "")).strip()
    if source_file:
        return f"file:{source_file}"
    return f"source:{chunk.source}"


def _case_category(case_id: str) -> str:
    for category, prefix in CATEGORY_PREFIXES:
        if case_id.startswith(prefix):
            return category
    return "xiao-core"


def _term_hits(chunk: KnowledgeChunk, terms: list[str]) -> int:
    text = _normalize_match_text(f"{chunk.title}\n{chunk.source}\n{chunk.text}")
    return sum(1 for term in terms if _normalize_match_text(term) in text)


def _normalize_urlish(text: str) -> str:
    return text.lower().strip().rstrip("/")


def _normalize_match_text(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _short_title(chunk: KnowledgeChunk) -> str:
    title = " ".join(chunk.title.split())
    if len(title) > 100:
        return f"{title[:97]}..."
    return title


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[percentile - 1]


if __name__ == "__main__":
    main()
