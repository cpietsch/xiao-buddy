from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_vector_artifact import _resolve_archive_path, _resolve_metadata_path
from xiao_copilot.vector_artifacts import ensure_vector_data, sha256_file
from xiao_copilot.vector_index import configured_index_paths, load_vector_index


def main() -> None:
    args = _parse_args()
    manifest_path, _fallback_data_path = configured_index_paths(args.manifest, "")
    _assert(manifest_path.exists(), f"manifest missing: {manifest_path}")

    metadata_path = _resolve_metadata_path(args.metadata, str(manifest_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    archive_path = _resolve_archive_path(metadata, metadata_path)

    archive_url = args.url or str(archive_path)
    archive_sha256 = args.sha256 or str(metadata.get("archive_sha256") or "")
    member_name = str(metadata.get("data_file") or "")
    expected_data_sha = str(metadata.get("data_sha256") or "")
    expected_count = int(metadata.get("count") or 0)
    expected_dim = int(metadata.get("dim") or 0)
    expected_backend = str(metadata.get("backend") or "").replace("-", "_")
    expected_source_hash = str(metadata.get("source_hash") or "")

    _assert(member_name, "metadata missing data_file")
    _assert(archive_sha256, "metadata missing archive_sha256")
    _assert(expected_data_sha, "metadata missing data_sha256")

    with tempfile.TemporaryDirectory(prefix="xiao-vector-restore-") as temp_dir:
        temp_index_dir = Path(temp_dir) / "index"
        temp_index_dir.mkdir(parents=True)
        temp_manifest = temp_index_dir / manifest_path.name
        shutil.copyfile(manifest_path, temp_manifest)

        target_data = temp_index_dir / member_name
        result = ensure_vector_data(
            target_data_path=target_data,
            artifact_url=archive_url,
            artifact_sha256=archive_sha256,
            member_name=member_name,
            timeout=args.timeout,
        )
        _assert(result.ok, result.detail)
        _assert(result.installed, "restore verifier expected to install into a clean temp directory")
        _assert(target_data.exists(), f"restored data missing: {target_data}")
        _assert(sha256_file(target_data) == expected_data_sha, "restored data sha256 mismatch")

        load_vector_index.cache_clear()
        index = load_vector_index(str(temp_manifest), "", False)
        _assert(index.count == expected_count, f"loaded count={index.count}, expected={expected_count}")
        _assert(index.dim == expected_dim, f"loaded dim={index.dim}, expected={expected_dim}")
        _assert(index.backend == expected_backend, f"loaded backend={index.backend}, expected={expected_backend}")
        _assert(
            index.source_hash == expected_source_hash,
            f"loaded source_hash={index.source_hash}, expected={expected_source_hash}",
        )

    print(
        "PASS vector artifact restore: "
        f"{Path(member_name).name} loaded as {expected_backend} "
        f"({expected_count} vectors, dim={expected_dim})"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a packaged vector artifact restores and loads in a clean index directory."
    )
    parser.add_argument(
        "--metadata",
        default="",
        help="Metadata JSON path. Defaults to the dist/vector-index artifact matching the current manifest.",
    )
    parser.add_argument("--manifest", default="data/index/xiao_vectors.json", help="Vector manifest path.")
    parser.add_argument("--url", default="", help="Artifact URL override. Defaults to the metadata archive path.")
    parser.add_argument("--sha256", default="", help="Artifact sha256 override. Defaults to metadata archive_sha256.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Artifact read/download timeout in seconds.")
    return parser.parse_args()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
