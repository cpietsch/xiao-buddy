from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.config import load_settings
from xiao_copilot.index_corpus import hash_chunks, indexable_chunks
from xiao_copilot.knowledge_base import KnowledgeChunk, load_knowledge_base
from xiao_copilot.vector_index import configured_index_paths


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    status: str
    name: str
    detail: str


def main() -> None:
    args = _parse_args()
    settings = load_settings()
    checks: list[Check] = []
    corpus_checks, chunks = _check_corpus()
    checks.extend(corpus_checks)
    checks.extend(
        _check_vector_index(
            settings.vector_index_manifest,
            settings.vector_index_data,
            chunks,
            artifact_url=settings.vector_index_archive_url,
        )
    )
    checks.extend(_check_settings(settings))
    if args.live:
        checks.extend(_check_live_endpoints(settings, timeout=args.timeout))
    if args.app:
        checks.append(_check_app(_app_url(settings), timeout=args.timeout))

    for check in checks:
        print(f"{check.status.upper():4} {check.name}: {check.detail}")

    failures = [check for check in checks if check.status == "fail"]
    warnings = [check for check in checks if check.status == "warn"]
    print(f"\nsummary: {len(failures)} fail, {len(warnings)} warn, {len(checks)} checks")
    if failures:
        raise SystemExit(1)
    if args.strict_warnings and warnings:
        raise SystemExit(2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local XIAO Buddy readiness.")
    parser.add_argument("--live", action="store_true", help="Check configured hosted endpoints.")
    parser.add_argument("--app", action="store_true", help="Check the configured Gradio app URL.")
    parser.add_argument("--strict-warnings", action="store_true", help="Exit non-zero when warnings are present.")
    parser.add_argument("--timeout", type=float, default=8.0, help="Network timeout in seconds.")
    return parser.parse_args()


def _check_corpus() -> tuple[list[Check], list[KnowledgeChunk]]:
    checks: list[Check] = []
    chunks: list[KnowledgeChunk] = []
    board_path = ROOT / "data" / "corpus" / "xiao_boards.json"
    wiki_path = ROOT / "data" / "corpus" / "wiki_chunks.jsonl"
    eval_path = ROOT / "data" / "corpus" / "eval_queries.jsonl"
    answer_eval_path = ROOT / "data" / "corpus" / "answer_eval_queries.jsonl"

    checks.append(_file_check("curated corpus", board_path))
    checks.append(_file_check("wiki chunks", wiki_path))
    checks.append(_file_check("retrieval evals", eval_path))
    checks.append(_file_check("answer evals", answer_eval_path))

    try:
        chunks = load_knowledge_base()
        checks.append(Check("ok", "knowledge base", f"{len(chunks)} chunks loaded"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("fail", "knowledge base", str(exc)))

    for name, path in (("wiki chunk rows", wiki_path), ("retrieval eval rows", eval_path), ("answer eval rows", answer_eval_path)):
        if path.exists():
            rows = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            checks.append(Check("ok", name, f"{rows} rows"))
    return checks, chunks


def _check_vector_index(
    manifest_path: str,
    data_path: str,
    chunks: list[KnowledgeChunk],
    *,
    artifact_url: str,
) -> list[Check]:
    checks: list[Check] = []
    manifest, fallback_data = configured_index_paths(manifest_path, data_path)
    if not manifest.exists():
        return [Check("warn", "vector manifest", f"missing {manifest}; retrieval will use slower fallback")]

    try:
        meta = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return [Check("fail", "vector manifest", f"could not parse {manifest}: {exc}")]

    ids = list(meta.get("ids", []))
    count = int(meta.get("count", len(ids)))
    backend = str(meta.get("backend") or meta.get("index_backend") or "flat").replace("-", "_")
    data_file = meta.get("data_file")
    data = Path(str(data_file)) if data_file else fallback_data
    if not data.is_absolute():
        data = manifest.parent / data

    if len(ids) != count:
        checks.append(Check("fail", "vector manifest", f"count={count} but ids={len(ids)}"))
    else:
        checks.append(Check("ok", "vector manifest", f"{count} ids, backend={backend}"))

    duplicate_count = len(ids) - len(set(ids))
    if duplicate_count:
        checks.append(Check("fail", "vector manifest ids", f"{duplicate_count} duplicate ids"))

    if chunks:
        include_field_notes = bool(meta.get("include_field_notes", False))
        indexable = indexable_chunks(chunks, include_field_notes=include_field_notes)
        expected_ids = {chunk.id for chunk in indexable}
        actual_ids = set(ids)
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        if missing or extra:
            detail = f"{len(missing)} missing, {len(extra)} extra"
            sample = (missing or extra)[:3]
            if sample:
                detail = f"{detail}; sample={', '.join(sample)}"
            checks.append(Check("fail", "vector corpus coverage", detail))
        else:
            checks.append(Check("ok", "vector corpus coverage", f"{len(indexable)} indexable chunks"))

        expected_hash = hash_chunks(indexable)
        manifest_hash = str(meta.get("source_hash") or "")
        if not manifest_hash:
            checks.append(Check("warn", "vector source hash", "manifest has no source_hash"))
        elif manifest_hash != expected_hash:
            checks.append(
                Check(
                    "fail",
                    "vector source hash",
                    f"manifest={manifest_hash} expected={expected_hash}; rebuild vector index",
                )
            )
        else:
            checks.append(Check("ok", "vector source hash", manifest_hash))

    if data.exists():
        checks.append(Check("ok", "vector data", f"{data.name} exists ({_format_bytes(data.stat().st_size)})"))
    else:
        detail = f"missing {data}; run scripts/build_wiki_vector_index.py"
        if artifact_url:
            detail = f"missing {data}; runtime artifact restore is configured"
        checks.append(Check("warn", "vector data", detail))

    if artifact_url:
        checks.append(Check("ok", "vector artifact URL", "configured"))

    if backend == "hnsw":
        try:
            import hnswlib  # noqa: F401

            checks.append(Check("ok", "hnswlib", "installed"))
        except ImportError:
            checks.append(Check("fail", "hnswlib", "required for HNSW indexes"))
    return checks


def _check_settings(settings) -> list[Check]:
    checks = [
        _configured_check("embedding endpoint", settings.embedding_base_url),
        _configured_check("reranker endpoint", settings.rerank_base_url),
        _configured_check("agent endpoint", settings.agent_base_url),
    ]
    checks.append(Check("ok", "rerank window", f"{settings.rerank_text_chars} chars"))
    checks.append(Check("ok", "gradio bind", f"{settings.gradio_server_name}:{settings.gradio_server_port}"))
    return checks


def _check_live_endpoints(settings, timeout: float) -> list[Check]:
    checks: list[Check] = []
    endpoints = [
        ("embedding models", _models_url(settings.embedding_base_url), settings.embedding_model),
        ("reranker models", _models_url(settings.rerank_base_url), settings.rerank_model),
        ("agent models", _models_url(settings.agent_base_url), settings.agent_model),
    ]
    for name, url, expected_model in endpoints:
        if not url:
            checks.append(Check("warn", name, "not configured"))
            continue
        checks.append(_http_model_check(name, url, expected_model, timeout=timeout))
    return checks


def _check_app(url: str, timeout: float) -> Check:
    return _http_check("gradio app", url, timeout=timeout)


def _file_check(name: str, path: Path) -> Check:
    if path.exists():
        return Check("ok", name, f"{path.relative_to(ROOT)} exists ({_format_bytes(path.stat().st_size)})")
    return Check("fail", name, f"missing {path.relative_to(ROOT)}")


def _configured_check(name: str, value: str) -> Check:
    if value:
        return Check("ok", name, "configured")
    return Check("warn", name, "not configured")


def _http_check(name: str, url: str, timeout: float) -> Check:
    try:
        response = requests.get(url, timeout=timeout)
        if 200 <= response.status_code < 400:
            return Check("ok", name, f"{response.status_code} {url}")
        return Check("warn", name, f"{response.status_code} {url}")
    except Exception as exc:  # noqa: BLE001
        return Check("warn", name, f"{url}: {exc}")


def _http_model_check(name: str, url: str, expected_model: str, timeout: float) -> Check:
    try:
        response = requests.get(url, timeout=timeout)
        if not 200 <= response.status_code < 400:
            return Check("warn", name, f"{response.status_code} {url}")
        if not expected_model:
            return Check("ok", name, f"{response.status_code} {url}")

        model_names = _extract_model_names(response.json())
        if expected_model in model_names:
            return Check("ok", name, f"{response.status_code} {url}; model={expected_model}")

        sample = ", ".join(sorted(model_names)[:5]) or "none"
        return Check("fail", name, f"{response.status_code} {url}; missing model={expected_model}; found={sample}")
    except Exception as exc:  # noqa: BLE001
        return Check("warn", name, f"{url}: {exc}")


def _extract_model_names(body: object) -> set[str]:
    names: set[str] = set()
    if isinstance(body, list):
        for item in body:
            names.update(_extract_model_names(item))
        return names
    if not isinstance(body, dict):
        return names

    for key in ("id", "name", "model"):
        value = body.get(key)
        if isinstance(value, str) and value:
            names.add(value)
    aliases = body.get("aliases")
    if isinstance(aliases, list):
        names.update(alias for alias in aliases if isinstance(alias, str) and alias)

    for key in ("data", "models"):
        value = body.get(key)
        if isinstance(value, list):
            for item in value:
                names.update(_extract_model_names(item))
    return names


def _models_url(base_url: str) -> str:
    if not base_url:
        return ""
    parsed = urlparse(base_url.rstrip("/") + "/")
    path = parsed.path.rstrip("/")
    if path.endswith("/embeddings") or path.endswith("/chat/completions") or path.endswith("/completions") or path.endswith("/rerank"):
        path = path.rsplit("/", 1)[0]
    if not path.endswith("/v1"):
        path = f"{path.rstrip('/')}/v1"
    parsed = parsed._replace(path=f"{path}/models", params="", query="", fragment="")
    return urlunparse(parsed)


def _app_url(settings) -> str:
    host = settings.gradio_server_name
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return urljoin(f"http://{host}:{settings.gradio_server_port}", "/")


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


if __name__ == "__main__":
    main()
