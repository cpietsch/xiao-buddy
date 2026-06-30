from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_VECTOR_DATA_PATTERNS = {
    "data/index/*.f16",
    "data/index/*.hnsw",
    "data/index/*.faiss",
    "data/index/knowledge_graph.json",
}
TRACKED_INDEX_ALLOWLIST = {
    "data/index/xiao_vectors.json",
}


def main() -> None:
    ignore_text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in sorted(IGNORED_VECTOR_DATA_PATTERNS):
        _assert(pattern in ignore_text.splitlines(), f".gitignore should include {pattern}")

    tracked = _git_lines("ls-files", "data/index")
    unexpected = sorted(path for path in tracked if path not in TRACKED_INDEX_ALLOWLIST)
    _assert(
        not unexpected,
        "generated vector data files should not be tracked: " + ", ".join(unexpected),
    )
    _assert("data/index/xiao_vectors.json" in tracked, "vector manifest should remain tracked")

    print("PASS generated artifact tracking regression")


def _git_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
