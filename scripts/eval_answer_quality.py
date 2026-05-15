from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.pipeline import answer_question


def main() -> None:
    eval_path = Path(os.environ.get("ANSWER_EVAL_PATH", "data/corpus/answer_eval_queries.jsonl"))
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

    for case in cases:
        print(f"RUN  {case['id']}", flush=True)
        answer, citations, diagnostics = answer_question(None, case["query"])
        fact_ok = _contains_all_terms(answer, case.get("must_include", []))
        required_citations = case.get("must_cite", [])
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
        print(
            f"{'PASS' if ok else 'MISS'} {case['id']}: "
            f"facts={'ok' if fact_ok else 'miss'} "
            f"cite={'ok' if citation_ok else 'miss'} "
            f"inline_cite={'ok' if inline_citation_ok else 'miss'} "
            f"agent={'ok' if agent_ok else 'miss'} "
            f"stream={'ok' if stream_ok else 'miss'} "
            f"chunks={agent.get('stream_chunks', 0)} "
            f"chars={agent.get('stream_chars', len(answer))} "
            f"total_ms={timings.get('total', 0)}",
            flush=True,
        )

    print(f"\nanswer fact-hit rate: {fact_hits}/{total} = {fact_hits / total:.0%}", flush=True)
    print(f"answer citation-hit rate: {citation_hits}/{total} = {citation_hits / total:.0%}", flush=True)
    print(
        f"answer inline-citation-hit rate: {inline_citation_hits}/{total} = {inline_citation_hits / total:.0%}",
        flush=True,
    )
    print(f"answer agent-hit rate: {agent_hits}/{total} = {agent_hits / total:.0%}", flush=True)
    print(f"answer stream-hit rate: {stream_hits}/{total} = {stream_hits / total:.0%}", flush=True)
    if failures:
        print(f"\nanswer eval failures: {', '.join(failures)}", flush=True)
        raise SystemExit(1)


def _contains_all_terms(text: str, terms: list[str]) -> bool:
    normalized = _normalize_match_text(text)
    return all(_normalize_match_text(term) in normalized for term in terms)


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


if __name__ == "__main__":
    main()
