from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.pipeline import answer_question
from scripts.report_metadata import quality_report_metadata


def main() -> None:
    eval_path = Path(os.environ.get("ANSWER_EVAL_PATH", "data/corpus/answer_eval_queries.jsonl"))
    json_output = Path(output) if (output := os.environ.get("ANSWER_EVAL_JSON_OUTPUT", "").strip()) else None
    require_agent = os.environ.get("ANSWER_EVAL_REQUIRE_AGENT", "1") == "1"
    require_stream = os.environ.get("ANSWER_EVAL_REQUIRE_STREAM", "1") == "1"
    limit = int(os.environ.get("ANSWER_EVAL_LIMIT", "0"))
    case_ids = {
        case_id.strip()
        for case_id in os.environ.get("ANSWER_EVAL_CASE_IDS", "").split(",")
        if case_id.strip()
    }

    cases = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]
    if case_ids:
        cases = [case for case in cases if case["id"] in case_ids]
    if limit:
        cases = cases[:limit]
    if not cases:
        raise SystemExit("No answer eval cases selected.")

    total = 0
    fact_hits = 0
    citation_hits = 0
    inline_citation_hits = 0
    agent_hits = 0
    stream_hits = 0
    failures: list[str] = []
    results: list[dict[str, object]] = []

    for case in cases:
        print(f"RUN  {case['id']}", flush=True)
        query = str(case["query"])
        expected_terms = [str(term) for term in case.get("must_include", [])]
        required_citations = [str(citation) for citation in case.get("must_cite", [])]
        answer, citations, diagnostics = answer_question(None, query)
        missing_terms = _missing_terms(answer, expected_terms)
        fact_ok = not missing_terms
        citation_ok = _contains_any_citation(citations, required_citations)
        inline_citation_ok = _inline_cites_required_source(answer, citations, required_citations)
        agent_ok = not require_agent or bool(diagnostics.get("agent_used"))
        stream_ok = not require_stream or bool(
            diagnostics.get("agent_streamed") or diagnostics.get("agent", {}).get("streamed")
        )
        ok = fact_ok and citation_ok and inline_citation_ok and agent_ok and stream_ok

        total += 1
        fact_hits += int(fact_ok)
        citation_hits += int(citation_ok)
        inline_citation_hits += int(inline_citation_ok)
        agent_hits += int(agent_ok)
        stream_hits += int(stream_ok)
        if not ok:
            failures.append(case["id"])

        agent = diagnostics.get("agent", {})
        timings = diagnostics.get("timings_ms", {})
        first_token_ms = _first_token_ms(diagnostics)
        required_source_ids = _source_ids_for_required_citations(citations, required_citations)
        all_source_ids = _source_ids_from_citations(citations)
        result = {
            "id": str(case["id"]),
            "ok": ok,
            "fact_ok": fact_ok,
            "citation_ok": citation_ok,
            "inline_citation_ok": inline_citation_ok,
            "agent_ok": agent_ok,
            "stream_ok": stream_ok,
            "missing_terms": missing_terms,
            "required_citations": required_citations,
            "required_source_ids": required_source_ids,
            "all_source_ids": all_source_ids,
            "stream_chunks": int(agent.get("stream_chunks", 0) or 0),
            "stream_chars": int(agent.get("stream_chars", len(answer)) or 0),
            "first_token_ms": first_token_ms,
            "answer_chars": len(answer),
            "total_ms": float(timings.get("total", 0) or 0),
        }
        results.append(result)
        print(
            f"{'PASS' if ok else 'MISS'} {case['id']}: "
            f"facts={'ok' if fact_ok else 'miss'} "
            f"cite={'ok' if citation_ok else 'miss'} "
            f"inline_cite={'ok' if inline_citation_ok else 'miss'} "
            f"agent={'ok' if agent_ok else 'miss'} "
            f"stream={'ok' if stream_ok else 'miss'} "
            f"chunks={agent.get('stream_chunks', 0)} "
            f"chars={agent.get('stream_chars', len(answer))} "
            f"first_token_ms={_format_ms(first_token_ms)} "
            f"total_ms={timings.get('total', 0)}"
            f"{' missing=' + ', '.join(missing_terms) if missing_terms else ''}",
            flush=True,
        )
        if not ok:
            print(
                _format_failure_detail(case, answer, citations, missing_terms, required_citations),
                flush=True,
            )

    summary = _summarize_results(results)
    print(f"\nanswer fact-hit rate: {fact_hits}/{total} = {summary['fact_rate']:.0%}", flush=True)
    print(f"answer citation-hit rate: {citation_hits}/{total} = {summary['citation_rate']:.0%}", flush=True)
    print(
        f"answer inline-citation-hit rate: "
        f"{inline_citation_hits}/{total} = {summary['inline_citation_rate']:.0%}",
        flush=True,
    )
    print(f"answer agent-hit rate: {agent_hits}/{total} = {summary['agent_rate']:.0%}", flush=True)
    print(f"answer stream-hit rate: {stream_hits}/{total} = {summary['stream_rate']:.0%}", flush=True)
    print(
        f"answer first-token latency: p50_ms={summary['p50_first_token_ms']:.1f} "
        f"p95_ms={summary['p95_first_token_ms']:.1f} max_ms={summary['max_first_token_ms']:.1f}",
        flush=True,
    )
    print(
        f"answer latency: p50_ms={summary['p50_ms']:.1f} "
        f"p95_ms={summary['p95_ms']:.1f} max_ms={summary['max_ms']:.1f}",
        flush=True,
    )
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        metadata = quality_report_metadata(
            report_type="answer_quality",
            eval_path=eval_path,
            selected_cases=cases,
            extra={
                "require_agent": require_agent,
                "require_stream": require_stream,
                "case_filter": sorted(case_ids),
                "limit": limit,
            },
        )
        json_output.write_text(
            json.dumps({"metadata": metadata, "summary": summary, "results": results}, indent=2) + "\n"
        )
        print(f"wrote {json_output}", flush=True)
    if failures:
        print(f"\nanswer eval failures: {', '.join(failures)}", flush=True)
        raise SystemExit(1)


def _summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    total = len(results)
    totals_ms = [float(result["total_ms"]) for result in results]
    first_token_ms = [
        float(value)
        for result in results
        if isinstance(value := result.get("first_token_ms"), (int, float))
    ]
    return {
        "cases": total,
        "passes": sum(1 for result in results if bool(result["ok"])),
        "failures": [str(result["id"]) for result in results if not bool(result["ok"])],
        "fact_rate": _rate(results, "fact_ok"),
        "citation_rate": _rate(results, "citation_ok"),
        "inline_citation_rate": _rate(results, "inline_citation_ok"),
        "agent_rate": _rate(results, "agent_ok"),
        "stream_rate": _rate(results, "stream_ok"),
        "p50_first_token_ms": _percentile(first_token_ms, 50),
        "p95_first_token_ms": _percentile(first_token_ms, 95),
        "max_first_token_ms": max(first_token_ms) if first_token_ms else 0.0,
        "p50_ms": _percentile(totals_ms, 50),
        "p95_ms": _percentile(totals_ms, 95),
        "max_ms": max(totals_ms) if totals_ms else 0.0,
    }


def _rate(results: list[dict[str, object]], key: str) -> float:
    return sum(1 for result in results if bool(result[key])) / len(results) if results else 0.0


def _first_token_ms(diagnostics: dict[str, object]) -> float | None:
    agent = diagnostics.get("agent", {})
    value = agent.get("first_token_ms") if isinstance(agent, dict) else None
    if value is None:
        value = diagnostics.get("agent_first_token_ms")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_ms(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _contains_all_terms(text: str, terms: list[str]) -> bool:
    return not _missing_terms(text, terms)


def _format_failure_detail(
    case: dict[str, object],
    answer: str,
    citations_text: str,
    missing_terms: list[str],
    required_citations: list[str],
) -> str:
    required_source_ids = _source_ids_for_required_citations(citations_text, required_citations)
    all_source_ids = _source_ids_from_citations(citations_text)
    citation_lines = [line.strip() for line in citations_text.splitlines() if line.strip().startswith("- [")]
    query_excerpt = _compact_text(str(case.get("query", "")), 240) or "<none>"
    answer_excerpt = _compact_text(answer, 500) or "<none>"
    citation_excerpt = _compact_text(" | ".join(citation_lines[:5]), 500) or "<none>"
    detail = [
        "  failure detail:",
        f"    query: {query_excerpt}",
        f"    missing_terms: {_format_list(missing_terms)}",
        f"    required_citations: {_format_list(required_citations)}",
        f"    required_source_ids: {_format_list(required_source_ids)}",
        f"    all_source_ids: {_format_list(all_source_ids)}",
        f"    answer_excerpt: {answer_excerpt}",
        f"    citation_excerpt: {citation_excerpt}",
    ]
    return "\n".join(detail)


def _missing_terms(text: str, terms: list[str]) -> list[str]:
    normalized = _normalize_match_text(text)
    return [term for term in terms if _normalize_match_text(term) not in normalized]


def _contains_any_citation(text: str, citations: list[str]) -> bool:
    if not citations:
        return True
    normalized = _normalize_match_text(text)
    return any(_normalize_match_text(citation) in normalized for citation in citations)


def _inline_cites_required_source(answer: str, citations_text: str, required_citations: list[str]) -> bool:
    source_ids = _source_ids_for_required_citations(citations_text, required_citations)
    if not source_ids:
        source_ids = _source_ids_from_citations(citations_text)
    return any(f"[{source_id}]" in answer for source_id in source_ids)


def _source_ids_for_required_citations(citations_text: str, required_citations: list[str]) -> list[str]:
    if not required_citations:
        return []
    required = [_normalize_match_text(citation) for citation in required_citations]
    source_ids: list[str] = []
    for line in citations_text.splitlines():
        normalized_line = _normalize_match_text(line)
        if any(citation in normalized_line for citation in required):
            source_id = _source_id_from_line(line)
            if source_id:
                source_ids.append(source_id)
    return source_ids


def _source_ids_from_citations(citations_text: str) -> list[str]:
    return [
        source_id
        for line in citations_text.splitlines()
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


def _normalize_match_text(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _compact_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    if limit <= 3:
        return compact[:limit]
    return compact[: limit - 3].rstrip() + "..."


def _format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "<none>"


if __name__ == "__main__":
    main()
