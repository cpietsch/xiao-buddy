from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class EndpointResult:
    ok: bool
    data: Any | None = None
    error: str = ""


def _headers(api_key: str = "") -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _join_url(base_url: str, path: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith(path):
        return base_url
    if path.startswith("/v1/") and base_url.endswith("/v1"):
        return f"{base_url}{path[3:]}"
    return f"{base_url}{path}"


def embed_texts(
    base_url: str,
    model: str,
    texts: list[str],
    api_key: str = "",
    timeout: float = 20,
) -> EndpointResult:
    if not base_url:
        return EndpointResult(ok=False, error="Embedding endpoint is not configured.")

    payload: dict[str, Any] = {"input": texts}
    if model:
        payload["model"] = model

    try:
        response = requests.post(
            base_url,
            headers=_headers(api_key),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        vectors = [row["embedding"] for row in body.get("data", [])]
        if len(vectors) != len(texts):
            return EndpointResult(
                ok=False,
                error=f"Expected {len(texts)} embeddings, got {len(vectors)}.",
            )
        return EndpointResult(ok=True, data=vectors)
    except Exception as exc:  # noqa: BLE001 - surfaced in diagnostics for hackathon use.
        return EndpointResult(ok=False, error=str(exc))


def embed_query(
    base_url: str,
    model: str,
    text: str,
    image_data_url: str | None = None,
    api_key: str = "",
    timeout: float = 20,
) -> EndpointResult:
    if not base_url:
        return EndpointResult(ok=False, error="Embedding endpoint is not configured.")

    if image_data_url:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Represent the user's hardware support query."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": text},
                    ],
                },
            ],
        }
    else:
        payload = {"input": text}
        if model:
            payload["model"] = model

    try:
        response = requests.post(
            base_url,
            headers=_headers(api_key),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        rows = body.get("data", [])
        if not rows:
            return EndpointResult(ok=False, error="Embedding endpoint returned no data.")
        return EndpointResult(ok=True, data=rows[0]["embedding"])
    except Exception as exc:  # noqa: BLE001
        return EndpointResult(ok=False, error=str(exc))


def rerank(
    base_url: str,
    model: str,
    query: str,
    documents: list[str],
    api_key: str = "",
    timeout: float = 20,
) -> EndpointResult:
    if not base_url:
        return EndpointResult(ok=False, error="Reranker endpoint is not configured.")

    payload: dict[str, Any] = {"query": query, "documents": documents}
    if model:
        payload["model"] = model

    try:
        response = requests.post(
            _join_url(base_url, "/rerank"),
            headers=_headers(api_key),
            json=payload,
            timeout=timeout,
        )
        if response.status_code == 404:
            response = requests.post(
                _join_url(base_url, "/v1/rerank"),
                headers=_headers(api_key),
                json=payload,
                timeout=timeout,
            )
        response.raise_for_status()
        body = response.json()
        scores = _parse_rerank_scores(body, len(documents))
        return EndpointResult(ok=True, data=scores)
    except Exception as exc:  # noqa: BLE001
        return EndpointResult(ok=False, error=str(exc))


def _parse_rerank_scores(body: dict[str, Any], count: int) -> list[tuple[int, float]]:
    results = body.get("results", body.get("data", []))
    parsed: list[tuple[int, float]] = []
    for position, item in enumerate(results):
        index = int(item.get("index", position))
        score = float(item.get("relevance_score", item.get("score", 0)))
        if 0 <= index < count:
            parsed.append((index, score))
    return parsed


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str = "",
    timeout: float = 20,
) -> EndpointResult:
    if not base_url:
        return EndpointResult(ok=False, error="Agent endpoint is not configured.")

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 700,
    }

    try:
        response = requests.post(
            _join_url(base_url, "/v1/chat/completions"),
            headers=_headers(api_key),
            data=json.dumps(payload),
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        return EndpointResult(ok=True, data=content)
    except Exception as exc:  # noqa: BLE001
        return EndpointResult(ok=False, error=str(exc))
