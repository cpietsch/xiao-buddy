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
    if os.environ.get("OFFLINE_EVAL", "1") == "1":
        settings = type(settings)(
            embedding_base_url="",
            rerank_base_url="",
            agent_base_url="",
            top_k=settings.top_k,
            candidate_k=settings.candidate_k,
        )

    eval_path = Path(__file__).resolve().parents[1] / "data" / "corpus" / "eval_queries.jsonl"
    total = 0
    hits = 0
    for line in eval_path.read_text().splitlines():
        case = json.loads(line)
        chunks, _diag = retrieve(case["query"], settings)
        retrieved = {chunk.board_id for chunk in chunks}
        ok = case["expected_board_id"] in retrieved
        total += 1
        hits += int(ok)
        status = "PASS" if ok else "MISS"
        print(f"{status} {case['id']}: {case['expected_board_id']} in {sorted(retrieved)}")

    print(f"\nretrieval board-hit rate: {hits}/{total} = {hits / total:.0%}")


if __name__ == "__main__":
    main()
