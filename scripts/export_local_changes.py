from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "dist" / "local-export"


def main() -> None:
    args = _parse_args()
    base_ref = args.base_ref
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    if not args.allow_dirty:
        _require_clean_worktree()

    commits = _git_lines("log", "--reverse", "--format=%H%x09%s", f"{base_ref}..HEAD")
    if not commits:
        raise SystemExit(f"No commits to export in {base_ref}..HEAD.")

    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    for old_patch in patches_dir.glob("*.patch"):
        old_patch.unlink()

    head = _git_one("rev-parse", "HEAD")
    base = _git_one("rev-parse", base_ref)
    short_head = _git_one("rev-parse", "--short", "HEAD")
    bundle_path = output_dir / f"xiao-buddy-{short_head}.bundle"
    manifest_path = output_dir / "manifest.json"

    _run_git("format-patch", "-o", str(patches_dir), f"{base_ref}..HEAD")
    if bundle_path.exists():
        bundle_path.unlink()
    _run_git("bundle", "create", str(bundle_path), "HEAD", f"^{base_ref}")
    _run_git("bundle", "verify", str(bundle_path))
    pruned_bundles = _prune_old_bundles(output_dir, bundle_path, args.keep_bundles)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_ref": base_ref,
        "base": base,
        "head": head,
        "commit_count": len(commits),
        "bundle": str(bundle_path.relative_to(ROOT)),
        "patches_dir": str(patches_dir.relative_to(ROOT)),
        "bundle_retention": {
            "keep_bundles": max(1, args.keep_bundles),
            "pruned": pruned_bundles,
        },
        "vector_artifact": _matching_vector_artifact(ROOT),
        "remote": _git_one("remote", "get-url", "origin", allow_failure=True),
        "commits": [
            {"sha": row.split("\t", 1)[0], "subject": row.split("\t", 1)[1]}
            for row in commits
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Exported {len(commits)} commits from {base_ref}..HEAD")
    print(f"Bundle: {bundle_path}")
    print(f"Patches: {patches_dir}")
    print(f"Manifest: {manifest_path}")
    if pruned_bundles:
        print(f"Pruned old bundles: {len(pruned_bundles)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export unpushed local commits as patches and a git bundle.")
    parser.add_argument("--base-ref", default="origin/main", help="Base ref to export from.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT)), help="Output directory.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow exporting with uncommitted work present.")
    parser.add_argument(
        "--keep-bundles",
        type=int,
        default=int(os.environ.get("EXPORT_LOCAL_KEEP_BUNDLES", "1")),
        help="Number of xiao-buddy-*.bundle files to retain in the output directory, including the current export.",
    )
    return parser.parse_args()


def _require_clean_worktree() -> None:
    status = _git_lines("status", "--porcelain")
    if status:
        raise SystemExit("Refusing to export with uncommitted changes. Commit/stash first or use --allow-dirty.")


def _git_one(*args: str, allow_failure: bool = False) -> str:
    result = _run_git(*args, check=not allow_failure)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _git_lines(*args: str) -> list[str]:
    output = _git_one(*args)
    return [line for line in output.splitlines() if line.strip()]


def _prune_old_bundles(output_dir: Path, current_bundle: Path, keep_bundles: int) -> list[str]:
    keep_bundles = max(1, keep_bundles)
    current_bundle = current_bundle.resolve()
    bundles = sorted(
        output_dir.glob("xiao-buddy-*.bundle"),
        key=lambda path: (path.resolve() == current_bundle, path.stat().st_mtime_ns),
        reverse=True,
    )
    pruned: list[str] = []
    for bundle in bundles[keep_bundles:]:
        if bundle.resolve() == current_bundle:
            continue
        bundle.unlink()
        pruned.append(bundle.name)
    return pruned


def _matching_vector_artifact(root: Path = ROOT) -> dict[str, object] | None:
    manifest_path = root / "data" / "index" / "xiao_vectors.json"
    artifact_dir = root / "dist" / "vector-index"
    if not manifest_path.exists() or not artifact_dir.exists():
        return None
    try:
        manifest_meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - export should not fail when optional artifact metadata is unavailable.
        return None

    candidates = sorted(
        artifact_dir.glob("*.tar.gz.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for metadata_path in candidates:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if not _vector_metadata_matches_manifest(metadata, manifest_meta):
            continue
        archive_path = _resolve_vector_archive_path(metadata, metadata_path, root)
        if not archive_path.exists():
            continue
        return {
            "archive": _relative_to_root(archive_path, root),
            "metadata": _relative_to_root(metadata_path, root),
            "archive_sha256": str(metadata.get("archive_sha256") or ""),
            "data_file": str(metadata.get("data_file") or ""),
            "data_bytes": int(metadata.get("data_bytes") or 0),
            "data_sha256": str(metadata.get("data_sha256") or ""),
            "backend": str(metadata.get("backend") or ""),
            "count": int(metadata.get("count") or 0),
            "dim": int(metadata.get("dim") or 0),
            "model": str(metadata.get("model") or ""),
            "source_hash": str(metadata.get("source_hash") or ""),
            "env": {
                "VECTOR_INDEX_ARCHIVE_URL": "<upload the archive and set its URL here>",
                "VECTOR_INDEX_ARCHIVE_SHA256": str(metadata.get("archive_sha256") or ""),
            },
        }
    return None


def _vector_metadata_matches_manifest(
    metadata: dict[str, object],
    manifest_meta: dict[str, object],
) -> bool:
    fields = ("backend", "count", "dim", "model", "source_hash")
    for field in fields:
        manifest_value = manifest_meta.get(field)
        if field == "backend":
            manifest_value = manifest_meta.get("backend") or manifest_meta.get("index_backend")
        if manifest_value is not None and metadata.get(field) != manifest_value:
            return False
    return True


def _resolve_vector_archive_path(metadata: dict[str, object], metadata_path: Path, root: Path) -> Path:
    archive = str(metadata.get("archive") or "")
    if archive:
        path = Path(archive).expanduser()
        if not path.is_absolute():
            path = root / path
        return path
    return metadata_path.with_suffix("")


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _run_git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


if __name__ == "__main__":
    main()
