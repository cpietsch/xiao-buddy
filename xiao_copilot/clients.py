from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class EndpointResult:
    ok: bool
    data: Any | None = None
    error: str = ""
    meta: dict[str, Any] | None = None


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


def _response_detail(response: requests.Response | None) -> str:
    if response is None:
        return ""
    text = response.text.strip()
    if not text:
        return ""
    return f" — {text[:500]}"


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
    except requests.HTTPError as exc:
        return EndpointResult(ok=False, error=f"{exc}{_response_detail(exc.response)}")
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
    except requests.HTTPError as exc:
        return EndpointResult(ok=False, error=f"{exc}{_response_detail(exc.response)}")
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

    native_result = _rerank_native(base_url, model, query, documents, api_key, timeout)
    if native_result.ok:
        return native_result
    if not _should_try_completion_rerank_fallback(native_result):
        return native_result

    try:
        scores = [
            (index, _rerank_one(base_url, model, query, document, api_key, timeout))
            for index, document in enumerate(documents)
        ]
        return EndpointResult(
            ok=True,
            data=scores,
            meta={
                "mode": "completion_logprob",
                "native_error": native_result.error,
            },
        )
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        if native_result.error:
            error = f"Native rerank failed: {native_result.error}; fallback failed: {error}"
        return EndpointResult(ok=False, error=error)


def _rerank_native(
    base_url: str,
    model: str,
    query: str,
    documents: list[str],
    api_key: str,
    timeout: float,
) -> EndpointResult:
    payload: dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": len(documents),
    }
    try:
        response = requests.post(
            _join_url(base_url, "/v1/rerank"),
            headers=_headers(api_key),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return EndpointResult(
            ok=True,
            data=_parse_native_rerank_scores(response.json(), len(documents)),
            meta={"mode": "native"},
        )
    except requests.HTTPError as exc:
        return EndpointResult(
            ok=False,
            error=f"{exc}{_response_detail(exc.response)}",
            meta={"mode": "native", "status_code": exc.response.status_code if exc.response else None},
        )
    except Exception as exc:  # noqa: BLE001
        return EndpointResult(ok=False, error=str(exc), meta={"mode": "native"})


def _should_try_completion_rerank_fallback(result: EndpointResult) -> bool:
    status_code = (result.meta or {}).get("status_code")
    if status_code in {404, 501}:
        return True
    error = result.error.lower()
    return "does not support reranking" in error


def _parse_native_rerank_scores(body: Any, expected_count: int) -> list[tuple[int, float]]:
    results = body
    if isinstance(body, dict):
        results = body.get("results", body.get("data", []))
    if not isinstance(results, list):
        raise ValueError("Native rerank response did not include a results list.")

    scores: list[tuple[int, float]] = []
    for position, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        index = int(row.get("index", position))
        score = row.get("relevance_score", row.get("score"))
        if score is None:
            continue
        if index < 0 or index >= expected_count:
            raise ValueError(f"Native rerank returned out-of-range index {index}.")
        scores.append((index, float(score)))

    if not scores:
        raise ValueError("Native rerank response did not include numeric scores.")
    return scores


def _rerank_one(
    base_url: str,
    model: str,
    query: str,
    document: str,
    api_key: str,
    timeout: float,
) -> float:
    prompt = (
        "Given a hardware support query, decide whether the document is relevant. "
        "Answer with exactly one word: yes or no.\n\n"
        f"Query: {query}\n"
        f"Document: {document}\n"
        "Relevant:"
    )
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "logprobs": 20,
        "temperature": 0.0,
    }
    response = requests.post(
        _join_url(base_url, "/v1/completions"),
        headers=_headers(api_key),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return _parse_yes_no_score(response.json())


def _parse_yes_no_score(body: dict[str, Any]) -> float:
    choice = body["choices"][0]
    logprobs = choice.get("logprobs") or {}
    top = (logprobs.get("top_logprobs") or [{}])[0]
    yes_lp = _find_token_logprob(top, "yes")
    no_lp = _find_token_logprob(top, "no")
    if yes_lp is None or no_lp is None:
        generated = (choice.get("text") or (choice.get("message") or {}).get("content", "")).strip().lower()
        return 1.0 if generated.startswith("yes") else 0.0
    yes = math.exp(yes_lp)
    no = math.exp(no_lp)
    return yes / (yes + no) if yes + no else 0.0


def _find_token_logprob(top: Any, token: str) -> float | None:
    if isinstance(top, dict):
        for key, value in top.items():
            if key.strip().lower() == token:
                return float(value)
    if isinstance(top, list):
        for entry in top:
            if entry.get("token", "").strip().lower() == token:
                return float(entry["logprob"])
    return None


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str = "",
    timeout: float = 20,
) -> EndpointResult:
    if not base_url:
        return EndpointResult(ok=False, error="Agent endpoint is not configured.")

    payload = _chat_payload(model, messages)

    try:
        response = requests.post(
            _join_url(base_url, "/v1/chat/completions"),
            headers=_headers(api_key),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        error = _provider_error_message(body)
        if error:
            return EndpointResult(ok=False, error=error)
        content = _chat_message_content(body)
        return EndpointResult(ok=True, data=content)
    except requests.HTTPError as exc:
        return EndpointResult(ok=False, error=f"{exc}{_response_detail(exc.response)}")
    except Exception as exc:  # noqa: BLE001
        return EndpointResult(ok=False, error=str(exc))


def chat_completion_stream(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str = "",
    timeout: float = 20,
) -> Iterator[EndpointResult]:
    if not base_url:
        yield EndpointResult(ok=False, error="Agent endpoint is not configured.")
        return

    payload = _chat_payload(model, messages)
    payload["stream"] = True

    try:
        with requests.post(
            _join_url(base_url, "/v1/chat/completions"),
            headers=_headers(api_key),
            json=payload,
            stream=True,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if line.startswith("data:"):
                    line = line.removeprefix("data:").strip()
                if not line or line == "[DONE]":
                    continue
                body = json.loads(line)
                error = _provider_error_message(body)
                if error:
                    yield EndpointResult(ok=False, error=error)
                    return
                content = _chat_delta_content(body)
                if content:
                    yield EndpointResult(ok=True, data=content)
    except requests.HTTPError as exc:
        yield EndpointResult(ok=False, error=f"{exc}{_response_detail(exc.response)}")
    except Exception as exc:  # noqa: BLE001
        yield EndpointResult(ok=False, error=str(exc))


def _chat_payload(model: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 700,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _chat_message_content(body: dict[str, Any]) -> str:
    choice = body["choices"][0]
    message = choice.get("message") or {}
    if "content" in message:
        return _content_to_text(message.get("content"))
    return _content_to_text(choice.get("text"))


def _chat_delta_content(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    delta = choice.get("delta") or {}
    if "content" in delta:
        return _content_to_text(delta.get("content"))
    return _content_to_text(choice.get("text"))


def _provider_error_message(body: Any) -> str:
    if not isinstance(body, dict) or "error" not in body:
        return ""
    error = body.get("error")
    if isinstance(error, str):
        detail = error
    elif isinstance(error, dict):
        detail = _content_to_text(error.get("message") or error.get("detail") or error.get("type"))
    else:
        detail = _content_to_text(error)
    return f"Provider error: {detail or 'unknown error'}"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_to_text(part) for part in content)
    if isinstance(content, dict):
        if "text" in content:
            return _content_to_text(content.get("text"))
        if "content" in content:
            return _content_to_text(content.get("content"))
        return ""
    return str(content)
