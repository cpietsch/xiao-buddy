from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.config import load_settings
from xiao_copilot.retrieval import retrieve


def main() -> None:
    settings = load_settings()
    offline = os.environ.get("OFFLINE_EVAL", "1") == "1"
    strict = os.environ.get("STRICT_EVAL", "0") == "1"
    limit = int(os.environ.get("EVAL_LIMIT", "0"))
    case_ids = {
        case_id.strip()
        for case_id in os.environ.get("EVAL_CASE_IDS", "").split(",")
        if case_id.strip()
    }
    if offline:
        settings = type(settings)(
            embedding_base_url="",
            rerank_base_url="",
            agent_base_url="",
            top_k=settings.top_k,
            candidate_k=settings.candidate_k,
        )

    eval_path = Path(__file__).resolve().parents[1] / "data" / "corpus" / "eval_queries.jsonl"
    cases = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]
    if case_ids:
        cases = [case for case in cases if case["id"] in case_ids]
    if limit:
        cases = cases[:limit]
    if not cases:
        raise SystemExit("No eval cases selected.")

    total = 0
    board_hits = 0
    content_hits = 0
    citation_hits = 0
    failures: list[str] = []
    for case in cases:
        chunks, diag = retrieve(case["query"], settings)
        retrieved = {chunk.board_id for chunk in chunks}
        expected_board_id = case.get("expected_board_id", "")
        board_ok = not expected_board_id or expected_board_id in retrieved
        content_ok = _contains_all_terms(chunks, case.get("must_include", []))
        citation_ok = _contains_any_citation(chunks, case.get("must_cite", []))
        total += 1
        board_hits += int(board_ok)
        content_hits += int(content_ok)
        citation_hits += int(citation_ok)
        ok = board_ok and (not strict or (content_ok and citation_ok))
        if not ok:
            failures.append(case["id"])
        status = "PASS" if ok else "MISS"
        checks = (
            f"board={'ok' if board_ok else 'miss'} "
            f"content={'ok' if content_ok else 'miss'} "
            f"cite={'ok' if citation_ok else 'miss'}"
        )
        backend = diag.get("vector_index_backend") or "lexical"
        expected_label = expected_board_id or "content/citation target"
        print(
            f"{status} {case['id']}: {expected_label} in "
            f"{sorted(retrieved)} [{checks}; backend={backend}]"
        )

    print(f"\nretrieval board-hit rate: {board_hits}/{total} = {board_hits / total:.0%}")
    print(f"retrieval content-hit rate: {content_hits}/{total} = {content_hits / total:.0%}")
    print(f"retrieval citation-hit rate: {citation_hits}/{total} = {citation_hits / total:.0%}")
    if strict and failures:
        print(f"\nstrict failures: {', '.join(failures)}")
        raise SystemExit(1)


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


if __name__ == "__main__":
    main()
