from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_local_changes import _browser_screenshots, _matching_vector_artifact, _prune_old_bundles, _quality_reports
from scripts.verify_local_export import (
    _patch_series_shas,
    _verify_browser_screenshots,
    _verify_patch_series_applies,
    _verify_quality_reports,
)
from xiao_copilot.vector_artifacts import sha256_file


def main() -> None:
    _assert_direct_export_cli_import()

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
        _assert_browser_screenshots_manifest(work)
        _assert_quality_reports_manifest(work)
        _assert_patch_series_sha_extraction(work)
        _assert_patch_series_tree_verification(work)

    print("PASS export-local manifest regression")


def _assert_direct_export_cli_import() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "scripts/export_local_changes.py", "--help"],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _assert(
        result.returncode == 0,
        "direct export script invocation should load local modules without PYTHONPATH\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
    )
    _assert("Export unpushed local commits" in result.stdout, "export script help text should be printed")


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


def _assert_browser_screenshots_manifest(work: Path) -> None:
    screenshot_dir = work / "dist" / "browser-smoke"
    screenshot_dir.mkdir(parents=True)
    _write_png(screenshot_dir / "desktop.png", size=(144, 96), color=(20, 80, 160))
    _write_png(screenshot_dir / "mobile.png", size=(39, 84), color=(80, 160, 20))
    _write_png(screenshot_dir / "desktop-after-query.png", size=(144, 120), color=(160, 20, 80))

    screenshots = _browser_screenshots(work)
    expected = {"desktop", "mobile", "desktop_after_query"}
    _assert(set(screenshots) == expected, f"unexpected browser screenshots: {screenshots}")
    desktop = screenshots["desktop"]
    _assert(desktop["path"] == "dist/browser-smoke/desktop.png", "desktop screenshot path should be relative")
    _assert(desktop["sha256"] == sha256_file(screenshot_dir / "desktop.png"), "desktop screenshot sha should match")
    _assert(desktop["width"] == 144 and desktop["height"] == 96, "desktop screenshot dimensions should match")
    _assert(desktop["bytes"] == (screenshot_dir / "desktop.png").stat().st_size, "desktop screenshot size should match")
    _assert(desktop["sample_color_count"] > 8, "desktop screenshot should be nonblank")

    _verify_browser_screenshots({"browser_screenshots": screenshots}, work)


def _assert_quality_reports_manifest(work: Path) -> None:
    answer_report = work / "dist" / "answer-quality" / "all.json"
    reranker_report = work / "dist" / "reranker-quality" / "all.json"
    answer_report.parent.mkdir(parents=True)
    reranker_report.parent.mkdir(parents=True)
    _write_jsonl(
        work / "data" / "corpus" / "answer_eval_queries.jsonl",
        [{"id": "answer-one"}, {"id": "answer-two"}],
    )
    _write_jsonl(work / "data" / "corpus" / "eval_queries.jsonl", [{"id": "rerank-one"}])
    answer_metadata = _report_metadata(
        work,
        "answer_quality",
        Path("data/corpus/answer_eval_queries.jsonl"),
        ["answer-one", "answer-two"],
    )
    reranker_metadata = _report_metadata(
        work,
        "reranker_quality",
        Path("data/corpus/eval_queries.jsonl"),
        ["rerank-one"],
    )
    answer_payload = {
        "metadata": answer_metadata,
        "summary": {"cases": 2, "passes": 2, "failures": [], "p50_ms": 123.4},
        "results": [{"id": "answer-one", "ok": True}, {"id": "answer-two", "ok": True}],
    }
    reranker_payload = {
        "metadata": reranker_metadata,
        "summary": {"cases": 1, "passes": 1, "failures": [], "min_margin": 0.4},
        "results": [{"id": "rerank-one", "ok": True}],
    }
    answer_report.write_text(json.dumps(answer_payload), encoding="utf-8")
    reranker_report.write_text(json.dumps(reranker_payload), encoding="utf-8")

    reports = _quality_reports(work)
    _assert(set(reports) == {"answer_quality", "reranker_quality"}, f"unexpected quality reports: {reports}")
    _assert(reports["answer_quality"]["path"] == "dist/answer-quality/all.json", "answer report path should be relative")
    _assert(
        reports["answer_quality"]["eval_path"] == "data/corpus/answer_eval_queries.jsonl",
        "answer report eval path should be recorded",
    )
    _assert(reports["answer_quality"]["sha256"] == sha256_file(answer_report), "answer report sha should match file")
    _assert(reports["answer_quality"]["cases"] == 2, "answer report case count should come from summary")
    _assert(reports["answer_quality"]["expected_cases"] == 2, "answer report expected case count should match eval file")
    _assert(
        reports["answer_quality"]["case_ids"] == ["answer-one", "answer-two"],
        "answer report should record covered case ids",
    )
    _assert(reports["answer_quality"]["metadata"] == answer_metadata, "answer report metadata should be recorded")
    _assert(reports["answer_quality"]["failures"] == [], "answer report failures should come from summary")
    _assert(reports["reranker_quality"]["failures"] == [], "reranker report failures should come from summary")

    _verify_quality_reports({"quality_reports": reports}, work)

    stale_payload = {
        **answer_payload,
        "metadata": {**answer_metadata, "git_head": "0" * 40},
    }
    answer_report.write_text(json.dumps(stale_payload), encoding="utf-8")
    _assert_raises_system_exit(lambda: _quality_reports(work), "stale quality report should block export")

    partial_payload = {
        "metadata": _report_metadata(
            work,
            "answer_quality",
            Path("data/corpus/answer_eval_queries.jsonl"),
            ["answer-one"],
        ),
        "summary": {"cases": 1, "passes": 1, "failures": [], "p50_ms": 12.3},
        "results": [{"id": "answer-one", "ok": True}],
    }
    answer_report.write_text(json.dumps(partial_payload), encoding="utf-8")
    _assert_raises_system_exit(lambda: _quality_reports(work), "partial quality report should block export")
    partial_reports = {
        "answer_quality": {
            "path": "dist/answer-quality/all.json",
            "eval_path": "data/corpus/answer_eval_queries.jsonl",
            "sha256": sha256_file(answer_report),
            "cases": 1,
            "expected_cases": 2,
            "case_ids": ["answer-one"],
            "metadata": partial_payload["metadata"],
            "failures": [],
            "summary": partial_payload["summary"],
        },
        "reranker_quality": reports["reranker_quality"],
    }
    _assert_raises_assertion(lambda: _verify_quality_reports({"quality_reports": partial_reports}, work), "covers")

    answer_report.write_text(json.dumps(answer_payload), encoding="utf-8")
    failing_payload = {
        "metadata": reranker_metadata,
        "summary": {"cases": 1, "passes": 0, "failures": ["rerank-one"], "min_margin": -0.1},
        "results": [{"id": "rerank-one", "ok": False}],
    }
    reranker_report.write_text(json.dumps(failing_payload), encoding="utf-8")
    _assert_raises_system_exit(lambda: _quality_reports(work), "failing quality report should block export")
    failed_reports = {
        "answer_quality": {
            "path": "dist/answer-quality/all.json",
            "eval_path": "data/corpus/answer_eval_queries.jsonl",
            "sha256": sha256_file(answer_report),
            "cases": 2,
            "expected_cases": 2,
            "case_ids": ["answer-one", "answer-two"],
            "metadata": answer_metadata,
            "failures": [],
            "summary": answer_payload["summary"],
        },
        "reranker_quality": {
            "path": "dist/reranker-quality/all.json",
            "eval_path": "data/corpus/eval_queries.jsonl",
            "sha256": sha256_file(reranker_report),
            "cases": 1,
            "expected_cases": 1,
            "case_ids": ["rerank-one"],
            "metadata": reranker_metadata,
            "failures": ["rerank-one"],
            "summary": failing_payload["summary"],
        }
    }
    _assert_raises_assertion(lambda: _verify_quality_reports({"quality_reports": failed_reports}, work), "failures")


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


def _write_png(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    from PIL import Image

    image = Image.new("RGB", size, color)
    pixels = image.load()
    for x in range(size[0]):
        for y in range(size[1]):
            pixels[x, y] = (
                (color[0] + x * 3 + y) % 256,
                (color[1] + x + y * 5) % 256,
                (color[2] + x * 2 + y * 7) % 256,
            )
    image.save(path)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _report_metadata(work: Path, report_type: str, eval_path: Path, selected_ids: list[str]) -> dict[str, object]:
    eval_file = work / eval_path
    all_ids = [
        str(json.loads(line)["id"])
        for line in eval_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "schema_version": 1,
        "report_type": report_type,
        "generated_at": "2026-05-16T00:00:00+00:00",
        "git_head": _git(Path(__file__).resolve().parents[1], "rev-parse", "HEAD"),
        "eval_path": str(eval_path),
        "eval_sha256": sha256_file(eval_file),
        "total_eval_cases": len(all_ids),
        "selected_case_count": len(selected_ids),
        "selected_case_ids": selected_ids,
    }


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


def _assert_raises_system_exit(fn, message: str) -> None:
    try:
        fn()
    except SystemExit:
        return
    raise AssertionError(message)


def _assert_raises_assertion(fn, expected_message: str) -> None:
    try:
        fn()
    except AssertionError as exc:
        _assert(expected_message in str(exc), f"expected {expected_message!r} in {exc!r}")
        return
    raise AssertionError(f"expected AssertionError containing {expected_message!r}")


if __name__ == "__main__":
    main()
