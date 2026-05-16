from __future__ import annotations

import argparse
import json
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.vector_artifacts import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = _parse_args()
    metadata_path = _resolve_metadata_path(args.metadata, args.manifest)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    archive_path = _resolve_archive_path(metadata, metadata_path)

    _assert(archive_path.exists(), f"archive missing: {archive_path}")
    expected_archive_sha = str(metadata.get("archive_sha256") or "")
    _assert(expected_archive_sha, "metadata missing archive_sha256")
    actual_archive_sha = sha256_file(archive_path)
    _assert(actual_archive_sha == expected_archive_sha, f"archive sha256 mismatch: {actual_archive_sha}")

    member_name = str(metadata.get("data_file") or "")
    expected_data_sha = str(metadata.get("data_sha256") or "")
    expected_data_bytes = int(metadata.get("data_bytes") or 0)
    _assert(member_name, "metadata missing data_file")
    _assert(expected_data_sha, "metadata missing data_sha256")
    _assert(expected_data_bytes > 0, "metadata data_bytes should be positive")

    with tempfile.TemporaryDirectory(prefix="xiao-vector-verify-") as temp_dir:
        extracted = Path(temp_dir) / member_name
        _extract_member(archive_path, member_name, extracted)
        _assert(extracted.stat().st_size == expected_data_bytes, "data byte count mismatch")
        _assert(sha256_file(extracted) == expected_data_sha, "data sha256 mismatch")

    print(f"PASS vector artifact: {archive_path.name} ({_format_bytes(archive_path.stat().st_size)})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a packaged vector artifact and metadata file.")
    parser.add_argument(
        "--metadata",
        default="",
        help="Metadata JSON path. Defaults to the dist/vector-index artifact matching the current manifest.",
    )
    parser.add_argument(
        "--manifest",
        default="data/index/xiao_vectors.json",
        help="Vector manifest used to choose a matching artifact when --metadata is omitted.",
    )
    return parser.parse_args()


def _resolve_metadata_path(value: str, manifest: str = "data/index/xiao_vectors.json") -> Path:
    if value:
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path
    candidates = sorted(
        (ROOT / "dist" / "vector-index").glob("*.tar.gz.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    _assert(bool(candidates), "no vector artifact metadata found in dist/vector-index")
    manifest_path = Path(manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if manifest_path.exists():
        manifest_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
        matching = [
            candidate
            for candidate in candidates
            if _metadata_matches_manifest(candidate, manifest_meta)
        ]
        _assert(
            bool(matching),
            f"no vector artifact metadata matches current manifest {manifest_path}",
        )
        return matching[0]
    return candidates[0]


def _metadata_matches_manifest(metadata_path: Path, manifest_meta: dict[str, object]) -> bool:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - unreadable metadata is not a match.
        return False
    fields = ("backend", "count", "dim", "model", "source_hash")
    for field in fields:
        manifest_value = manifest_meta.get(field)
        if field == "backend":
            manifest_value = manifest_meta.get("backend") or manifest_meta.get("index_backend")
        if manifest_value is not None and metadata.get(field) != manifest_value:
            return False
    return True


def _resolve_archive_path(metadata: dict[str, object], metadata_path: Path) -> Path:
    archive = str(metadata.get("archive") or "")
    if archive:
        path = Path(archive).expanduser()
        return path if path.is_absolute() else ROOT / path
    return metadata_path.with_suffix("")


def _extract_member(archive_path: Path, member_name: str, target_path: Path) -> None:
    _assert(tarfile.is_tarfile(archive_path), f"archive is not a tar file: {archive_path}")
    with tarfile.open(archive_path) as archive:
        matches = [
            member
            for member in archive.getmembers()
            if member.isfile() and Path(member.name).name == member_name
        ]
        _assert(bool(matches), f"archive does not contain {member_name}")
        _assert(len(matches) == 1, f"archive contains multiple {member_name} entries")
        source = archive.extractfile(matches[0])
        _assert(source is not None, f"could not read {member_name}")
        with source, target_path.open("wb") as output:
            output.write(source.read())


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
