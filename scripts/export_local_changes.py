from __future__ import annotations

import argparse
import json
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

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_ref": base_ref,
        "base": base,
        "head": head,
        "commit_count": len(commits),
        "bundle": str(bundle_path.relative_to(ROOT)),
        "patches_dir": str(patches_dir.relative_to(ROOT)),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export unpushed local commits as patches and a git bundle.")
    parser.add_argument("--base-ref", default="origin/main", help="Base ref to export from.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT)), help="Output directory.")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow exporting with uncommitted work present.")
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
