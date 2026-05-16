from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_local_changes import _prune_old_bundles


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

    print("PASS export-local bundle pruning regression")


def _write_bundle(path: Path, *, mtime: int) -> Path:
    path.write_text(path.name, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
