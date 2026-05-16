from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_local_changes import _matching_vector_artifact, _prune_old_bundles
from scripts.verify_local_export import _patch_series_shas, _verify_patch_series_applies


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="xiao-export-") as temp_dir:
        work = Path(temp_dir)
        current = _write_bundle(work / "xiao-buddy-current.bundle", mtime=30)
        oldest = _write_bundle(work / "xiao-buddy-oldest.bundle", mtime=10)
        newer = _write_bundle(work / "xiao-buddy-newer.bundle", mtime=20)
        unrelated = _write_bundle(work / "other.bundle", mtime=1)

        pruned = _prune_old_bundles(work, current, keep_bundles=1)
        _assert(pruned == [newer.name, oldest.name], f"unexpected prune order: {pruned}")
        _assert(current.exists(), "current bundle should be retained")
        _assert(not oldest.exists(), "oldest bundle should be pruned")
        _assert(not newer.exists(), "newer old bundle should be pruned")
        _assert(unrelated.exists(), "unrelated bundles should not be touched")

        newest = _write_bundle(work / "xiao-buddy-newest.bundle", mtime=40)
        older = _write_bundle(work / "xiao-buddy-older.bundle", mtime=35)
        pruned = _prune_old_bundles(work, current, keep_bundles=2)
        _assert(pruned == [older.name], f"keep_bundles=2 should prune only the oldest generated bundle: {pruned}")
        _assert(current.exists(), "current bundle should still be retained")
        _assert(newest.exists(), "newest previous bundle should be retained when keep_bundles=2")
        _assert(not older.exists(), "oldest previous bundle should be pruned when keep_bundles=2")

        _assert_matching_vector_artifact_manifest(work)
        _assert_patch_series_sha_extraction(work)
        _assert_patch_series_tree_verification(work)

    print("PASS export-local manifest regression")


def _assert_matching_vector_artifact_manifest(work: Path) -> None:
    manifest_dir = work / "data" / "index"
    artifact_dir = work / "dist" / "vector-index"
    manifest_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)

    manifest_meta = {
        "backend": "hnsw",
        "count": 42,
        "dim": 2048,
        "model": "qwen3-vl-embedding-2b",
        "source_hash": "current-source",
    }
    (manifest_dir / "xiao_vectors.json").write_text(json.dumps(manifest_meta), encoding="utf-8")

    stale_archive = artifact_dir / "xiao-vectors-hnsw-stale.tar.gz"
    stale_archive.write_bytes(b"stale")
    stale_metadata = {
        **manifest_meta,
        "source_hash": "stale-source",
        "archive": str(stale_archive),
        "archive_sha256": "stale-sha",
        "data_file": "xiao_vectors.hnsw",
        "data_bytes": 5,
        "data_sha256": "stale-data-sha",
    }
    (artifact_dir / "xiao-vectors-hnsw-stale.tar.gz.json").write_text(json.dumps(stale_metadata), encoding="utf-8")

    current_archive = artifact_dir / "xiao-vectors-hnsw-current.tar.gz"
    current_archive.write_bytes(b"current")
    current_metadata = {
        **manifest_meta,
        "archive": str(current_archive),
        "archive_sha256": "current-sha",
        "data_file": "xiao_vectors.hnsw",
        "data_bytes": 7,
        "data_sha256": "current-data-sha",
    }
    (artifact_dir / "xiao-vectors-hnsw-current.tar.gz.json").write_text(
        json.dumps(current_metadata),
        encoding="utf-8",
    )

    artifact = _matching_vector_artifact(work)
    _assert(artifact is not None, "matching vector artifact should be included in export manifest")
    _assert(
        artifact["archive"] == "dist/vector-index/xiao-vectors-hnsw-current.tar.gz",
        f"unexpected artifact archive: {artifact}",
    )
    _assert(artifact["archive_sha256"] == "current-sha", "export manifest should include archive sha")
    _assert(artifact["data_sha256"] == "current-data-sha", "export manifest should include data sha")
    _assert(artifact["count"] == 42, "export manifest should include vector count")
    _assert(artifact["env"]["VECTOR_INDEX_ARCHIVE_SHA256"] == "current-sha", "env sha should match artifact sha")


def _assert_patch_series_sha_extraction(work: Path) -> None:
    first = work / "0001-one.patch"
    second = work / "0002-two.patch"
    first.write_text(
        "From 1111111111111111111111111111111111111111 Mon Sep 17 00:00:00 2001\n"
        "Subject: [PATCH 1/2] one\n",
        encoding="utf-8",
    )
    second.write_text(
        "From 2222222222222222222222222222222222222222 Mon Sep 17 00:00:00 2001\n"
        "Subject: [PATCH 2/2] two\n",
        encoding="utf-8",
    )
    _assert(
        _patch_series_shas([first, second])
        == [
            "1111111111111111111111111111111111111111",
            "2222222222222222222222222222222222222222",
        ],
        "patch sha extraction should preserve patch order",
    )


def _assert_patch_series_tree_verification(work: Path) -> None:
    repo = work / "patch-repo"
    patches_dir = work / "patches"
    repo.mkdir()
    patches_dir.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test User")

    (repo / "note.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "note.txt")
    _git(repo, "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")

    (repo / "note.txt").write_text("one\ntwo\n", encoding="utf-8")
    (repo / "extra.txt").write_text("extra\n", encoding="utf-8")
    _git(repo, "add", "note.txt", "extra.txt")
    _git(repo, "commit", "-q", "-m", "change")
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "format-patch", "-q", "-o", str(patches_dir), f"{base}..{head}")

    _verify_patch_series_applies(base, head, sorted(patches_dir.glob("*.patch")), repo)


def _write_bundle(path: Path, *, mtime: int) -> Path:
    path.write_text(path.name, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
