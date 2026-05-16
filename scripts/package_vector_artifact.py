from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.vector_artifacts import sha256_file
from xiao_copilot.vector_index import configured_index_paths


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = _parse_args()
    manifest_path, fallback_data_path = configured_index_paths(args.manifest, args.data)
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = _resolve_data_path(meta, manifest_path, fallback_data_path, bool(args.data))
    if not data_path.exists():
        raise SystemExit(f"Vector data file is missing: {data_path}")

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_name = args.name or _default_archive_name(meta, data_path)
    archive_path = output_dir / archive_name
    metadata_path = archive_path.with_suffix(archive_path.suffix + ".json")

    write_deterministic_tar_gz(archive_path, data_path)

    metadata = {
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "data_file": data_path.name,
        "data_bytes": data_path.stat().st_size,
        "data_sha256": sha256_file(data_path),
        "manifest": str(manifest_path),
        "backend": meta.get("backend") or meta.get("index_backend") or "flat",
        "count": meta.get("count", len(meta.get("ids", []))),
        "dim": meta.get("dim"),
        "model": meta.get("model"),
        "source_hash": meta.get("source_hash"),
    }
    pruned_artifacts = _prune_old_artifacts(output_dir, archive_path, args.keep_artifacts)
    metadata["artifact_retention"] = {
        "keep_artifacts": max(1, args.keep_artifacts),
        "pruned": pruned_artifacts,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote archive: {archive_path}")
    print(f"wrote metadata: {metadata_path}")
    print(f"VECTOR_INDEX_ARCHIVE_SHA256={metadata['archive_sha256']}")
    if pruned_artifacts:
        print(f"pruned old vector artifacts: {len(pruned_artifacts)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the local vector data file for deployment.")
    parser.add_argument("--manifest", default="", help="Vector manifest path.")
    parser.add_argument("--data", default="", help="Vector data path override.")
    parser.add_argument("--output-dir", default="dist/vector-index", help="Output directory.")
    parser.add_argument("--name", default="", help="Archive filename. Defaults to backend/source hash.")
    parser.add_argument(
        "--keep-artifacts",
        type=int,
        default=int(os.environ.get("VECTOR_ARTIFACT_KEEP", "1")),
        help="Number of vector artifact archives to retain in the output directory, including the current archive.",
    )
    return parser.parse_args()


def _resolve_data_path(
    meta: dict[str, object],
    manifest_path: Path,
    fallback_data_path: Path,
    data_path_explicit: bool,
) -> Path:
    if data_path_explicit:
        return fallback_data_path
    data_file = meta.get("data_file")
    if data_file:
        path = Path(str(data_file))
        return path if path.is_absolute() else manifest_path.parent / path
    return fallback_data_path


def write_deterministic_tar_gz(archive_path: Path, data_path: Path) -> None:
    with archive_path.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as gzip_output:
            with tarfile.open(fileobj=gzip_output, mode="w") as archive:
                info = archive.gettarinfo(str(data_path), arcname=data_path.name)
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mode = 0o644
                with data_path.open("rb") as source:
                    archive.addfile(info, source)


def _default_archive_name(meta: dict[str, object], data_path: Path) -> str:
    stem = data_path.stem.replace("_", "-")
    backend = str(meta.get("backend") or meta.get("index_backend") or data_path.suffix.lstrip(".") or "vector")
    source_hash = str(meta.get("source_hash") or "unhashed")
    return f"{stem}-{backend}-{source_hash}.tar.gz"


def _prune_old_artifacts(output_dir: Path, current_archive: Path, keep_artifacts: int) -> list[str]:
    keep_artifacts = max(1, keep_artifacts)
    current_archive = current_archive.resolve()
    archives = sorted(
        output_dir.glob("*.tar.gz"),
        key=lambda path: (path.resolve() == current_archive, path.stat().st_mtime_ns),
        reverse=True,
    )
    pruned: list[str] = []
    for archive in archives[keep_artifacts:]:
        if archive.resolve() == current_archive:
            continue
        metadata = archive.with_suffix(archive.suffix + ".json")
        archive.unlink()
        pruned.append(archive.name)
        if metadata.exists():
            metadata.unlink()
            pruned.append(metadata.name)

    for metadata in output_dir.glob("*.tar.gz.json"):
        archive = metadata.with_suffix("")
        if archive.exists() or archive.resolve() == current_archive:
            continue
        metadata.unlink()
        pruned.append(metadata.name)
    return pruned


if __name__ == "__main__":
    main()
