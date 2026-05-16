from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.vector_artifacts import ensure_vector_data, install_vector_artifact, sha256_file
from scripts.verify_vector_artifact import _metadata_matches_manifest


ROOT = Path(__file__).resolve().parents[1]
TEST_BYTES = b"fake-hnsw-data"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="xiao-artifacts-") as temp_dir:
        work = Path(temp_dir)
        manifest_path = work / "index" / "test_vectors.json"
        data_path = work / "index" / "test_vectors.hnsw"
        output_dir = work / "out"
        _write_test_index(manifest_path, data_path)

        archive_path = _package_test_index(manifest_path, data_path, output_dir, "test-index-a.tar.gz")
        second_archive_path = _package_test_index(manifest_path, data_path, output_dir, "test-index-b.tar.gz")
        _assert(
            sha256_file(archive_path) == sha256_file(second_archive_path),
            "packaging the same vector index twice should produce identical archive bytes",
        )
        metadata_path = archive_path.with_suffix(archive_path.suffix + ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        archive_sha = str(metadata["archive_sha256"])

        data_path.unlink()
        result = install_vector_artifact(
            artifact_url=str(archive_path),
            target_data_path=data_path,
            artifact_sha256=f"sha256:{archive_sha}",
            member_name=data_path.name,
        )
        _assert(result.ok, result.detail)
        _assert(result.installed, "expected install to report installed=True")
        _assert(data_path.read_bytes() == TEST_BYTES, "installed data did not match original bytes")
        existing_result = ensure_vector_data(
            target_data_path=data_path,
            artifact_url=str(archive_path),
            artifact_sha256=archive_sha,
            member_name=data_path.name,
        )
        _assert(existing_result.ok, existing_result.detail)
        _assert(not existing_result.installed, "existing data should not be reinstalled")

        data_path.unlink()
        restored_result = ensure_vector_data(
            target_data_path=data_path,
            artifact_url=str(archive_path),
            artifact_sha256=archive_sha,
            member_name=data_path.name,
        )
        _assert(restored_result.ok, restored_result.detail)
        _assert(restored_result.installed, "missing data should be restored")
        _assert(data_path.read_bytes() == TEST_BYTES, "restored data did not match original bytes")

        data_path.unlink()
        bad_result = install_vector_artifact(
            artifact_url=str(archive_path),
            target_data_path=data_path,
            artifact_sha256="0" * 64,
            member_name=data_path.name,
        )
        _assert(not bad_result.ok, "checksum mismatch should fail")
        _assert(not data_path.exists(), "failed checksum should not leave installed data")

        raw_copy = work / "raw-copy.hnsw"
        raw_result = install_vector_artifact(
            artifact_url=str(data_path.with_suffix(".missing")),
            target_data_path=raw_copy,
        )
        _assert(not raw_result.ok, "missing raw artifact should fail")

        _assert_metadata_manifest_matching(work)
        _assert_artifact_pruning(work)

    print("PASS vector artifact package/install regression")


def _assert_metadata_manifest_matching(work: Path) -> None:
    manifest_meta = {
        "backend": "hnsw",
        "count": 42,
        "dim": 2048,
        "model": "qwen3-vl-embedding-2b",
        "source_hash": "current-source",
    }
    good_metadata = work / "matching-artifact.json"
    stale_metadata = work / "stale-artifact.json"
    good_metadata.write_text(json.dumps(manifest_meta), encoding="utf-8")
    stale_payload = {**manifest_meta, "source_hash": "stale-source"}
    stale_metadata.write_text(json.dumps(stale_payload), encoding="utf-8")

    _assert(_metadata_matches_manifest(good_metadata, manifest_meta), "matching artifact metadata should match")
    _assert(
        not _metadata_matches_manifest(stale_metadata, manifest_meta),
        "stale artifact metadata should not match the current manifest",
    )


def _assert_artifact_pruning(work: Path) -> None:
    manifest_path = work / "prune" / "index" / "test_vectors.json"
    data_path = work / "prune" / "index" / "test_vectors.hnsw"
    output_dir = work / "prune" / "out"
    _write_test_index(manifest_path, data_path)
    output_dir.mkdir(parents=True)
    stale_archive = output_dir / "stale.tar.gz"
    stale_metadata = output_dir / "stale.tar.gz.json"
    stale_archive.write_bytes(b"stale")
    stale_metadata.write_text("{}", encoding="utf-8")

    archive_path = _package_test_index(
        manifest_path,
        data_path,
        output_dir,
        "current.tar.gz",
        keep_artifacts=1,
    )
    metadata = json.loads(archive_path.with_suffix(archive_path.suffix + ".json").read_text(encoding="utf-8"))
    _assert(archive_path.exists(), "current artifact archive should remain")
    _assert(not stale_archive.exists(), "stale artifact archive should be pruned")
    _assert(not stale_metadata.exists(), "stale artifact metadata should be pruned")
    _assert(
        metadata["artifact_retention"]["pruned"] == ["stale.tar.gz", "stale.tar.gz.json"],
        "artifact retention metadata should list pruned files",
    )


def _write_test_index(manifest_path: Path, data_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_bytes(TEST_BYTES)
    manifest = {
        "schema_version": 1,
        "model": "test-model",
        "dim": 3,
        "backend": "hnsw",
        "count": 1,
        "ids": ["chunk-1"],
        "data_file": data_path.name,
        "source_hash": "testhash",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _package_test_index(
    manifest_path: Path,
    data_path: Path,
    output_dir: Path,
    name: str,
    *,
    keep_artifacts: int = 2,
) -> Path:
    archive_path = output_dir / name
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_vector_artifact.py"),
            "--manifest",
            str(manifest_path),
            "--data",
            str(data_path),
            "--output-dir",
            str(output_dir),
            "--name",
            archive_path.name,
            "--keep-artifacts",
            str(keep_artifacts),
        ],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    _assert(archive_path.exists(), f"archive was not written: {archive_path}")
    _assert(archive_path.with_suffix(archive_path.suffix + ".json").exists(), "metadata was not written")
    _assert(len(sha256_file(archive_path)) == 64, "archive sha256 is malformed")
    return archive_path


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
