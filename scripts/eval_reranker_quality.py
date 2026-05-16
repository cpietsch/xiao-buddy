from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.clients import rerank
from xiao_copilot.config import load_settings
from xiao_copilot.knowledge_base import KnowledgeChunk, load_knowledge_base
from xiao_copilot.retrieval import _is_weekly_wiki_source, _lexical_score, _rerank_text


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


def main() -> None:
    args = _parse_args()
    settings = load_settings()
    if not settings.rerank_base_url:
        raise SystemExit("RERANK_BASE_URL is required for reranker quality evaluation.")

    cases = _load_cases(args.eval_path, args.case_ids)
    corpus = load_knowledge_base()
    print(
        f"cases={len(cases)} negatives={args.negatives} "
        f"min_margin={args.min_margin:.4f}",
        flush=True,
    )

    results = [_run_case(case, corpus, settings, args.negatives, args.min_margin) for case in cases]
    passes = [result for result in results if result["ok"]]
    margins = [float(result["margin"]) for result in results]
    latencies = [float(result["ms"]) for result in results]
    failures = [str(result["id"]) for result in results if not result["ok"]]
    print(
        "\nreranker top-positive rate: "
        f"{len(passes)}/{len(results)} = {len(passes) / len(results):.0%}",
        flush=True,
    )
    print(
        f"reranker margin: min={min(margins):.4f} p50={_percentile(margins, 50):.4f}",
        flush=True,
    )
    print(
        f"reranker latency: p50_ms={_percentile(latencies, 50):.1f} "
        f"p95_ms={_percentile(latencies, 95):.1f} max_ms={max(latencies):.1f}",
        flush=True,
    )
    if failures:
        print(f"\nfailures: {', '.join(failures)}", flush=True)
        raise SystemExit(1)


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
    parser.add_argument("--limit", type=int, default=0, help="Limit cases after filtering.")
    parser.add_argument("--negatives", type=int, default=2, help="Hard lexical negatives per query.")
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        help="Required positive-score margin over the best negative.",
    )
    args = parser.parse_args()
    if args.negatives < 1:
        raise SystemExit("--negatives must be at least 1.")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative.")
    selected = tuple(args.case or DEFAULT_CASE_IDS)
    args.case_ids = selected[: args.limit] if args.limit else selected
    return args


def _load_cases(path: Path, case_ids: tuple[str, ...]) -> list[dict[str, object]]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_id = {str(case["id"]): case for case in cases}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise SystemExit(f"Unknown eval case ids: {', '.join(missing)}")
    return [by_id[case_id] for case_id in case_ids]


def _run_case(
    case: dict[str, object],
    corpus: list[KnowledgeChunk],
    settings,
    negative_count: int,
    min_margin: float,
) -> dict[str, object]:
    case_id = str(case["id"])
    query = str(case["query"])
    citations = [str(citation) for citation in case.get("must_cite", [])]
    terms = [str(term) for term in case.get("must_include", [])]
    print(f"RUN  {case_id}", flush=True)
    positive = _select_positive(case, corpus, citations, terms)
    negatives = _select_negatives(query, corpus, citations, positive, negative_count)
    candidates = [positive, *negatives]
    documents = [_rerank_text(chunk, settings.rerank_text_chars) for chunk in candidates]

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
        return {"id": case_id, "ok": False, "margin": -1.0, "ms": elapsed_ms}

    scores = {index: float(score) for index, score in result.data or []}
    missing = [index for index in range(len(candidates)) if index not in scores]
    if missing:
        print(f"FAIL {case_id}: reranker did not score indexes {missing}", flush=True)
        return {"id": case_id, "ok": False, "margin": -1.0, "ms": elapsed_ms}

    positive_score = scores[0]
    best_negative_index = max(range(1, len(candidates)), key=lambda index: scores[index])
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
    print(f"     positive: {_short_title(positive)}", flush=True)
    print(f"     negative: {_short_title(candidates[best_negative_index])}", flush=True)
    return {"id": case_id, "ok": ok, "margin": margin, "ms": elapsed_ms}


def _select_positive(
    case: dict[str, object],
    corpus: list[KnowledgeChunk],
    citations: list[str],
    terms: list[str],
) -> KnowledgeChunk:
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
    return positives[0]


def _select_negatives(
    query: str,
    corpus: list[KnowledgeChunk],
    citations: list[str],
    positive: KnowledgeChunk,
    count: int,
) -> list[KnowledgeChunk]:
    positive_source = _source_key(positive)
    negatives = [
        chunk
        for chunk in corpus
        if chunk.id != positive.id
        and not _matches_any_citation(chunk, citations)
        and not _is_weekly_wiki_source(chunk)
        and _source_key(chunk) != positive_source
    ]
    negatives.sort(key=lambda chunk: _lexical_score(query, chunk.search_text), reverse=True)
    selected = negatives[:count]
    if len(selected) < count:
        raise SystemExit(f"Could not select {count} hard negatives for query: {query}")
    return selected


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
