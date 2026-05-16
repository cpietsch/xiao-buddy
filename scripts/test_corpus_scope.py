from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WIKI_CHUNKS = ROOT / "data" / "corpus" / "wiki_chunks.jsonl"

MIN_TOTAL_CHUNKS = 15_000
MIN_UNIQUE_DOCS = 1_500
REQUIRED_PREFIXES = {
    "Sensor/SeeedStudio_XIAO": (100, 1_000),
    "Sensor/Grove": (100, 500),
    "Sensor/SenseCAP": (50, 500),
    "Edge/NVIDIA_Jetson": (50, 500),
    "Edge/Raspberry_Pi_Devices": (50, 500),
    "Robotics/Robot_Kits": (25, 250),
    "Network/Meshtastic_Network": (10, 100),
}


def main() -> None:
    _assert(WIKI_CHUNKS.exists(), f"missing {WIKI_CHUNKS.relative_to(ROOT)}")
    chunks = _load_chunks()
    source_files = [
        str(chunk.get("metadata", {}).get("source_file", ""))
        for chunk in chunks
    ]
    unique_docs = set(source_files)

    _assert(
        len(chunks) >= MIN_TOTAL_CHUNKS,
        f"expected at least {MIN_TOTAL_CHUNKS} wiki chunks, got {len(chunks)}",
    )
    _assert(
        len(unique_docs) >= MIN_UNIQUE_DOCS,
        f"expected at least {MIN_UNIQUE_DOCS} source docs, got {len(unique_docs)}",
    )
    _assert(len({chunk["id"] for chunk in chunks}) == len(chunks), "wiki chunk ids must be unique")

    missing_fields = [
        chunk.get("id", "<unknown>")
        for chunk in chunks
        if not chunk.get("title") or not chunk.get("source") or not chunk.get("text")
    ]
    _assert(not missing_fields, f"chunks missing title/source/text: {', '.join(missing_fields[:5])}")

    by_prefix = _profile_prefixes(source_files)
    failures = []
    for prefix, (min_docs, min_chunks) in REQUIRED_PREFIXES.items():
        docs, rows = by_prefix[prefix]
        if docs < min_docs or rows < min_chunks:
            failures.append(f"{prefix}: docs={docs}/{min_docs} chunks={rows}/{min_chunks}")
    _assert(not failures, "full-wiki coverage too narrow: " + "; ".join(failures))

    top = Counter(_top_prefix(source) for source in source_files).most_common(6)
    top_summary = ", ".join(f"{prefix}={count}" for prefix, count in top)
    print(
        "PASS corpus scope: "
        f"{len(chunks)} chunks, {len(unique_docs)} docs, required full-wiki families present ({top_summary})"
    )


def _load_chunks() -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    lines = WIKI_CHUNKS.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        _assert(
            isinstance(item.get("metadata"), dict),
            f"line {line_no} missing metadata object",
        )
        chunks.append(item)
    return chunks


def _profile_prefixes(source_files: list[str]) -> dict[str, tuple[int, int]]:
    doc_sets: dict[str, set[str]] = defaultdict(set)
    chunk_counts: Counter[str] = Counter()
    for source in source_files:
        for prefix in REQUIRED_PREFIXES:
            if source == prefix or source.startswith(prefix + "/"):
                doc_sets[prefix].add(source)
                chunk_counts[prefix] += 1
    return {prefix: (len(doc_sets[prefix]), chunk_counts[prefix]) for prefix in REQUIRED_PREFIXES}


def _top_prefix(source_file: str) -> str:
    parts = [part for part in source_file.split("/") if part]
    return "/".join(parts[:2]) if len(parts) >= 2 else source_file or "<missing>"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
