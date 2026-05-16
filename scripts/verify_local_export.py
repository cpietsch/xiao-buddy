from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.vector_artifacts import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    args = _parse_args()
    manifest_path = _resolve_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    _verify_git_export(manifest)
    _verify_patch_series(manifest)
    _verify_vector_artifact(manifest)

    print(f"PASS local export: {manifest_path.relative_to(ROOT)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the local handoff export manifest, bundle, patches, and vector artifact reference.")
    parser.add_argument("--manifest", default="dist/local-export/manifest.json", help="Local export manifest path.")
    return parser.parse_args()


def _verify_git_export(manifest: dict[str, Any]) -> None:
    head = str(manifest.get("head") or "")
    base_ref = str(manifest.get("base_ref") or "")
    base = str(manifest.get("base") or "")
    bundle = _resolve_path(str(manifest.get("bundle") or ""))

    _assert(head, "manifest missing head")
    _assert(base_ref, "manifest missing base_ref")
    _assert(base, "manifest missing base")
    _assert(bundle.exists(), f"bundle missing: {bundle}")

    current_head = _git_one("rev-parse", "HEAD")
    current_base = _git_one("rev-parse", base_ref)
    _assert(head == current_head, f"manifest head={head} but current HEAD={current_head}")
    _assert(base == current_base, f"manifest base={base} but {base_ref}={current_base}")

    _run_git("bundle", "verify", str(bundle))


def _verify_patch_series(manifest: dict[str, Any]) -> None:
    patches_dir = _resolve_path(str(manifest.get("patches_dir") or ""))
    commits = manifest.get("commits")
    commit_count = int(manifest.get("commit_count") or 0)
    _assert(patches_dir.exists() and patches_dir.is_dir(), f"patches dir missing: {patches_dir}")
    _assert(isinstance(commits, list), "manifest commits must be a list")
    _assert(commit_count == len(commits), f"commit_count={commit_count} but commits={len(commits)}")

    patches = sorted(patches_dir.glob("*.patch"))
    _assert(len(patches) == commit_count, f"patch count={len(patches)} but commit_count={commit_count}")


def _verify_vector_artifact(manifest: dict[str, Any]) -> None:
    artifact = manifest.get("vector_artifact")
    if artifact is None:
        return
    _assert(isinstance(artifact, dict), "vector_artifact must be an object or null")

    archive = _resolve_path(str(artifact.get("archive") or ""))
    metadata_path = _resolve_path(str(artifact.get("metadata") or ""))
    _assert(archive.exists(), f"vector artifact archive missing: {archive}")
    _assert(metadata_path.exists(), f"vector artifact metadata missing: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    archive_sha = str(artifact.get("archive_sha256") or "")
    _assert(archive_sha, "vector_artifact missing archive_sha256")
    _assert(sha256_file(archive) == archive_sha, "vector artifact archive sha256 mismatch")

    for field in (
        "archive_sha256",
        "data_file",
        "data_bytes",
        "data_sha256",
        "backend",
        "count",
        "dim",
        "model",
        "source_hash",
    ):
        _assert(
            artifact.get(field) == metadata.get(field),
            f"vector_artifact {field}={artifact.get(field)!r} but metadata has {metadata.get(field)!r}",
        )

    vector_manifest = ROOT / "data" / "index" / "xiao_vectors.json"
    _assert(vector_manifest.exists(), f"vector manifest missing: {vector_manifest}")
    vector_meta = json.loads(vector_manifest.read_text(encoding="utf-8"))
    _assert(_artifact_matches_vector_manifest(artifact, vector_meta), "vector_artifact does not match current vector manifest")

    env = artifact.get("env")
    _assert(isinstance(env, dict), "vector_artifact env must be an object")
    _assert(
        env.get("VECTOR_INDEX_ARCHIVE_SHA256") == archive_sha,
        "vector artifact env sha does not match archive sha",
    )


def _artifact_matches_vector_manifest(
    artifact: dict[str, Any],
    vector_meta: dict[str, Any],
) -> bool:
    for field in ("backend", "count", "dim", "model", "source_hash"):
        vector_value = vector_meta.get(field)
        if field == "backend":
            vector_value = vector_meta.get("backend") or vector_meta.get("index_backend")
        if vector_value is not None and artifact.get(field) != vector_value:
            return False
    return True


def _resolve_path(value: str) -> Path:
    _assert(value, "path value is empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def _git_one(*args: str) -> str:
    result = _run_git(*args)
    return result.stdout.strip()


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
