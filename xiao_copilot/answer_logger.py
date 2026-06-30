from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from xiao_copilot.config import Settings


_ROOT = Path(__file__).resolve().parents[1]
_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class AnswerRunLogger:
    enabled: bool
    path: Path
    run_id: str
    answer_char_limit: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "AnswerRunLogger":
        path = configured_answer_log_path(settings.answer_log_path)
        return cls(
            enabled=bool(settings.answer_log_enabled),
            path=path,
            run_id=uuid4().hex,
            answer_char_limit=max(0, int(settings.answer_log_answer_chars)),
        )

    def log_success(
        self,
        *,
        question: str,
        answer: str,
        citations: str,
        diagnostics: dict[str, object],
        progress_html: str,
        event_count: int,
        stages: list[str],
    ) -> None:
        self._append(
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "created_at": _now_iso(),
                "status": str(diagnostics.get("status") or "ok"),
                "question": _trim_text(question, 4000),
                "question_chars": len(question),
                "answer": _trim_text(answer, self.answer_char_limit),
                "answer_chars": len(answer),
                "answer_truncated": self.answer_char_limit > 0 and len(answer) > self.answer_char_limit,
                "citations": _citation_records(citations, diagnostics),
                "citations_text": _trim_text(citations, 8000),
                "event_count": event_count,
                "stages": stages,
                "failure_flags": _failure_flags(answer, diagnostics),
                "diagnostics": diagnostics,
                "progress_text": _strip_html(progress_html),
            }
        )

    def log_exception(
        self,
        *,
        question: str,
        event_count: int,
        stages: list[str],
        last_diagnostics: dict[str, object] | None,
        error: BaseException,
    ) -> None:
        diagnostics = dict(last_diagnostics or {})
        diagnostics["exception_type"] = type(error).__name__
        diagnostics["exception"] = str(error)
        self._append(
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "created_at": _now_iso(),
                "status": "exception",
                "question": _trim_text(question, 4000),
                "question_chars": len(question),
                "event_count": event_count,
                "stages": stages,
                "failure_flags": sorted({"exception", *_failure_flags("", diagnostics)}),
                "diagnostics": diagnostics,
                "traceback": traceback.format_exception_only(type(error), error),
            }
        )

    def _append(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(_json_safe(record), ensure_ascii=False, separators=(",", ":"))
            with _WRITE_LOCK:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception:
            return


def configured_answer_log_path(path: str) -> Path:
    resolved = Path(path).expanduser() if path else _ROOT / "dist" / "answer-runs.jsonl"
    if not resolved.is_absolute():
        resolved = _ROOT / resolved
    return resolved


def _failure_flags(answer: str, diagnostics: dict[str, object]) -> list[str]:
    flags: set[str] = set()
    if not answer.strip():
        flags.add("empty_answer")
    if diagnostics.get("agent_used") is False:
        flags.add("fallback_answer")
    if diagnostics.get("agent_error"):
        flags.add("agent_error")
    if diagnostics.get("agent_truncated"):
        flags.add("truncated_answer")
    retrieval = diagnostics.get("retrieval")
    if isinstance(retrieval, dict):
        if retrieval.get("reranker_error"):
            flags.add("reranker_error")
        if retrieval.get("embedding_error"):
            flags.add("embedding_error")
        if retrieval.get("vector_index_error"):
            flags.add("vector_index_error")
    sources = diagnostics.get("top_sources")
    if isinstance(sources, list) and not sources:
        flags.add("no_sources")
    return sorted(flags)


def _citation_records(citations: str, diagnostics: dict[str, object]) -> list[dict[str, object]]:
    sources = diagnostics.get("top_sources")
    if isinstance(sources, list):
        records = [_source_record(source) for source in sources]
        records = [record for record in records if record.get("id") or record.get("source")]
        if records:
            return records

    records: list[dict[str, object]] = []
    for line in citations.splitlines():
        line = line.strip()
        if not line.startswith("- [") or "] " not in line:
            continue
        source_id, rest = line[3:].split("] ", 1)
        title, _, source_url = rest.partition(" — ")
        records.append({"id": source_id.strip(), "title": title.strip(), "source": source_url.strip()})
    return records


def _source_record(source: object) -> dict[str, object]:
    if not isinstance(source, dict):
        return {}
    record: dict[str, object] = {}
    for key in ("id", "title", "source", "board_id", "kind", "snippet"):
        value = source.get(key)
        if value is not None:
            record[key] = value
    return record


def _trim_text(value: str, limit: int) -> str:
    if limit <= 0 or len(value) <= limit:
        return value
    return value[: max(0, limit - 4)].rstrip() + " ..."


def _strip_html(value: str) -> str:
    text = value.replace("<", " <")
    parts: list[str] = []
    in_tag = False
    for char in text:
        if char == "<":
            in_tag = True
            parts.append(" ")
            continue
        if char == ">":
            in_tag = False
            parts.append(" ")
            continue
        if not in_tag:
            parts.append(char)
    return " ".join("".join(parts).split())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")
