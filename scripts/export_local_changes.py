from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xiao_copilot.vector_artifacts import sha256_file

DEFAULT_OUTPUT_DIR = ROOT / "dist" / "local-export"
QUALITY_REPORTS = {
    "answer_quality": Path("dist/answer-quality/all.json"),
    "reranker_quality": Path("dist/reranker-quality/all.json"),
}
BROWSER_SCREENSHOTS = {
    "desktop": Path("dist/browser-smoke/desktop.png"),
    "mobile": Path("dist/browser-smoke/mobile.png"),
    "desktop_after_query": Path("dist/browser-smoke/desktop-after-query.png"),
}


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
        "browser_screenshots": _browser_screenshots(ROOT),
        "quality_reports": _quality_reports(ROOT),
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


def _quality_reports(root: Path = ROOT) -> dict[str, object]:
    reports: dict[str, object] = {}
    for name, default_path in QUALITY_REPORTS.items():
        path = root / default_path
        if not path.exists():
            continue
        reports[name] = _quality_report_entry(path, root)
    return reports


def _browser_screenshots(root: Path = ROOT) -> dict[str, object]:
    screenshots: dict[str, object] = {}
    for name, default_path in BROWSER_SCREENSHOTS.items():
        path = root / default_path
        if not path.exists():
            continue
        screenshots[name] = _browser_screenshot_entry(path, root)
    return screenshots


def _browser_screenshot_entry(path: Path, root: Path) -> dict[str, object]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
            color_count = _sample_color_count(image)
    except Exception as exc:  # noqa: BLE001 - malformed local evidence should block handoff export.
        raise SystemExit(f"Browser screenshot is not a valid image: {_relative_to_root(path, root)}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise SystemExit(f"Browser screenshot has invalid dimensions: {_relative_to_root(path, root)}")
    if color_count <= 8:
        raise SystemExit(f"Browser screenshot appears blank: {_relative_to_root(path, root)}")
    return {
        "path": _relative_to_root(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "width": int(width),
        "height": int(height),
        "sample_color_count": color_count,
    }


def _sample_color_count(image) -> int:
    sample = image.convert("RGB").resize((64, 64))
    colors = sample.getcolors(maxcolors=4096)
    return 4097 if colors is None else len(colors)


def _quality_report_entry(path: Path, root: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - malformed local evidence should block handoff export.
        raise SystemExit(f"Quality report is not valid JSON: {_relative_to_root(path, root)}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Quality report must be a JSON object: {_relative_to_root(path, root)}")
    summary = payload.get("summary")
    results = payload.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        raise SystemExit(
            f"Quality report must include summary object and results list: {_relative_to_root(path, root)}"
        )
    cases_value = summary.get("cases", len(results))
    try:
        cases = int(cases_value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Quality report summary.cases must be numeric: {_relative_to_root(path, root)}") from exc
    failures = summary.get("failures", [])
    if not isinstance(failures, list):
        raise SystemExit(f"Quality report summary.failures must be a list: {_relative_to_root(path, root)}")
    return {
        "path": _relative_to_root(path, root),
        "sha256": sha256_file(path),
        "cases": cases,
        "failures": [str(item) for item in failures],
        "summary": summary,
    }


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
        if path.is_absolute():
            return path
        metadata_relative = metadata_path.parent / path
        if metadata_relative.exists():
            return metadata_relative
        return root / path
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
