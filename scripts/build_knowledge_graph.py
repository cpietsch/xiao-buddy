from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.config import load_settings
from xiao_copilot.knowledge_base import load_knowledge_base
from xiao_copilot.knowledge_graph import configured_graph_artifact_path, get_knowledge_graph, graph_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or validate the persisted knowledge graph artifact.")
    parser.add_argument("--output", default="", help="Graph artifact path. Defaults to GRAPH_ARTIFACT_PATH.")
    parser.add_argument("--force", action="store_true", help="Remove the existing artifact before building.")
    args = parser.parse_args()

    settings = load_settings()
    artifact_path = configured_graph_artifact_path(args.output or settings.graph_artifact_path)
    if args.force and artifact_path.exists():
        artifact_path.unlink()

    chunks = load_knowledge_base()
    started_at = perf_counter()
    graph = get_knowledge_graph(chunks, artifact_path=str(artifact_path))
    elapsed_ms = (perf_counter() - started_at) * 1000
    summary = graph_summary(graph)
    print(
        "knowledge_graph "
        f"status={graph.metadata.get('artifact_status', '')} "
        f"path={artifact_path} "
        f"chunks={len(chunks)} "
        f"entities={summary['entities']} "
        f"chunk_links={summary['chunk_links']} "
        f"ms={elapsed_ms:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
