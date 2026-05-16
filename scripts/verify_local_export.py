from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
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
    _verify_quality_reports(manifest)
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
    bundle_heads = _git_lines("bundle", "list-heads", str(bundle))
    _assert(
        f"{head} HEAD" in bundle_heads,
        f"bundle heads do not include manifest HEAD {head}: {bundle_heads}",
    )

    expected_commits = _git_commit_rows(base_ref)
    manifest_commits = _manifest_commit_rows(manifest)
    _assert(
        manifest_commits == expected_commits,
        "manifest commits do not match current git log for base_ref..HEAD",
    )


def _verify_patch_series(manifest: dict[str, Any]) -> None:
    patches_dir = _resolve_path(str(manifest.get("patches_dir") or ""))
    commits = manifest.get("commits")
    commit_count = int(manifest.get("commit_count") or 0)
    _assert(patches_dir.exists() and patches_dir.is_dir(), f"patches dir missing: {patches_dir}")
    _assert(isinstance(commits, list), "manifest commits must be a list")
    _assert(commit_count == len(commits), f"commit_count={commit_count} but commits={len(commits)}")

    patches = sorted(patches_dir.glob("*.patch"))
    _assert(len(patches) == commit_count, f"patch count={len(patches)} but commit_count={commit_count}")
    patch_shas = _patch_series_shas(patches)
    manifest_shas = [sha for sha, _subject in _manifest_commit_rows(manifest)]
    _assert(patch_shas == manifest_shas, "patch series commit SHAs do not match manifest commit order")
    _verify_patch_series_applies(
        str(manifest.get("base") or ""),
        str(manifest.get("head") or ""),
        patches,
        ROOT,
    )


def _verify_quality_reports(manifest: dict[str, Any], root: Path = ROOT) -> None:
    reports = manifest.get("quality_reports", {})
    _assert(isinstance(reports, dict), "quality_reports must be an object")
    for name, report in reports.items():
        _assert(isinstance(report, dict), f"quality report {name} must be an object")
        path = _resolve_path_at(str(report.get("path") or ""), root)
        _assert(path.exists(), f"quality report missing: {path}")
        expected_sha = str(report.get("sha256") or "")
        _assert(expected_sha, f"quality report {name} missing sha256")
        _assert(sha256_file(path) == expected_sha, f"quality report {name} sha256 mismatch")
        payload = json.loads(path.read_text(encoding="utf-8"))
        _assert(isinstance(payload, dict), f"quality report {name} must be a JSON object")
        summary = payload.get("summary")
        results = payload.get("results")
        _assert(isinstance(summary, dict), f"quality report {name} missing summary object")
        _assert(isinstance(results, list), f"quality report {name} missing results list")
        _assert(
            int(report.get("cases") or 0) == _quality_report_case_count(summary, results, name),
            f"quality report {name} cases mismatch",
        )
        _assert(
            report.get("failures") == _quality_report_failures(summary, name),
            f"quality report {name} failures mismatch",
        )


def _quality_report_case_count(summary: dict[str, Any], results: list[Any], name: str) -> int:
    cases_value = summary.get("cases", len(results))
    try:
        return int(cases_value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"quality report {name} summary.cases must be numeric") from exc


def _quality_report_failures(summary: dict[str, Any], name: str) -> list[str]:
    failures = summary.get("failures", [])
    _assert(isinstance(failures, list), f"quality report {name} summary.failures must be a list")
    return [str(item) for item in failures]


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


def _git_commit_rows(base_ref: str) -> list[tuple[str, str]]:
    rows = _git_lines("log", "--reverse", "--format=%H%x09%s", f"{base_ref}..HEAD")
    return [_split_commit_row(row) for row in rows]


def _manifest_commit_rows(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    commits = manifest.get("commits")
    _assert(isinstance(commits, list), "manifest commits must be a list")
    rows: list[tuple[str, str]] = []
    for index, row in enumerate(commits, start=1):
        _assert(isinstance(row, dict), f"manifest commit {index} must be an object")
        sha = str(row.get("sha") or "")
        subject = str(row.get("subject") or "")
        _assert(sha, f"manifest commit {index} missing sha")
        _assert(subject, f"manifest commit {index} missing subject")
        rows.append((sha, subject))
    return rows


def _patch_series_shas(patches: list[Path]) -> list[str]:
    shas: list[str] = []
    for patch in patches:
        first_line = patch.read_text(encoding="utf-8").splitlines()[0].strip()
        prefix = "From "
        suffix = " Mon Sep 17 00:00:00 2001"
        _assert(first_line.startswith(prefix), f"{patch.name} first line is not a format-patch From header")
        sha = first_line[len(prefix) :]
        if sha.endswith(suffix):
            sha = sha[: -len(suffix)]
        _assert(len(sha) == 40, f"{patch.name} has malformed commit sha {sha!r}")
        shas.append(sha)
    return shas


def _verify_patch_series_applies(
    base: str,
    head: str,
    patches: list[Path],
    repo_root: Path = ROOT,
) -> None:
    _assert(base, "manifest missing base")
    _assert(head, "manifest missing head")
    with tempfile.TemporaryDirectory(prefix="xiao-export-apply-") as temp_dir:
        worktree = Path(temp_dir) / "worktree"
        _run_git_at(repo_root, "worktree", "add", "--detach", "--quiet", str(worktree), base)
        try:
            if patches:
                _run_git_at(
                    worktree,
                    "apply",
                    "--index",
                    "--binary",
                    "--whitespace=nowarn",
                    *[str(patch) for patch in patches],
                )
            applied_tree = _git_one_at(worktree, "write-tree")
            expected_tree = _git_one_at(repo_root, "rev-parse", f"{head}^{{tree}}")
            _assert(
                applied_tree == expected_tree,
                f"patch series tree={applied_tree} but manifest HEAD tree={expected_tree}",
            )
        finally:
            _run_git_at(repo_root, "worktree", "remove", "--force", str(worktree), check=False)


def _split_commit_row(row: str) -> tuple[str, str]:
    sha, separator, subject = row.partition("\t")
    _assert(separator == "\t" and sha and subject, f"malformed git log row: {row!r}")
    return sha, subject


def _resolve_path(value: str) -> Path:
    return _resolve_path_at(value, ROOT)


def _resolve_path_at(value: str, root: Path) -> Path:
    _assert(value, "path value is empty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def _git_one(*args: str) -> str:
    return _git_one_at(ROOT, *args)


def _git_lines(*args: str) -> list[str]:
    output = _git_one(*args)
    return [line for line in output.splitlines() if line.strip()]


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return _run_git_at(ROOT, *args)


def _git_one_at(repo_root: Path, *args: str) -> str:
    result = _run_git_at(repo_root, *args)
    return result.stdout.strip()


def _run_git_at(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
