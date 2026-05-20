from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from xiao_copilot.vector_artifacts import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1


def quality_report_metadata(
    *,
    report_type: str,
    eval_path: Path,
    selected_cases: list[dict[str, object]],
    root: Path = ROOT,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eval_file = _resolve_path(eval_path, root)
    all_case_ids = _jsonl_case_ids(eval_file)
    selected_case_ids = [str(case["id"]) for case in selected_cases]
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report_type": report_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(root),
        "eval_path": _display_path(eval_file, root),
        "eval_sha256": sha256_file(eval_file),
        "total_eval_cases": len(all_case_ids),
        "selected_case_count": len(selected_case_ids),
        "selected_case_ids": selected_case_ids,
    }
    if extra:
        metadata.update(extra)
    return metadata


def _jsonl_case_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("id"):
            raise ValueError(f"{path} line {line_number} must include an id.")
        ids.append(str(row["id"]))
    if not ids:
        raise ValueError(f"{path} has no eval cases.")
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"{path} has duplicate ids: {', '.join(duplicates)}.")
    return ids


def _git_head(root: Path = ROOT) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()


def _resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
