from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_EVALS = ROOT / "data" / "corpus" / "eval_queries.jsonl"
ANSWER_EVALS = ROOT / "data" / "corpus" / "answer_eval_queries.jsonl"

MIN_RETRIEVAL_CASES = 60
MIN_ANSWER_CASES = 30
REQUIRED_CLOSE_MARGIN_ANSWER_IDS = {
    "answer-c3-esphome-yaml",
    "answer-c3-round-display-zephyr-buses",
    "answer-c3-zephyr-oled-address",
    "answer-c5-battery-voltage",
    "answer-nrf54-zigbee-target",
    "answer-s3-sense-pdm-mic-pins",
    "answer-s3-sensecraft-i2c",
    "answer-s3-sx1262-kit-applications",
    "answer-sensecap-m2-lns",
    "answer-sensecraft-grove-v2-pretrained",
}


def main() -> None:
    retrieval_cases = _load_jsonl(RETRIEVAL_EVALS)
    answer_cases = _load_jsonl(ANSWER_EVALS)
    _assert(len(retrieval_cases) >= MIN_RETRIEVAL_CASES, "retrieval eval coverage is unexpectedly narrow")
    _assert(len(answer_cases) >= MIN_ANSWER_CASES, "answer eval coverage is unexpectedly narrow")
    _assert_unique_ids(retrieval_cases, "retrieval eval")
    _assert_unique_ids(answer_cases, "answer eval")
    for case in retrieval_cases:
        _assert_eval_case_shape(case, "eval-", require_board_id=True)
    for case in answer_cases:
        _assert_eval_case_shape(case, "answer-", require_board_id=False)
    answer_ids = {str(case["id"]) for case in answer_cases}
    missing_close_margin = sorted(REQUIRED_CLOSE_MARGIN_ANSWER_IDS - answer_ids)
    _assert(
        not missing_close_margin,
        "answer eval missing close-margin coverage cases: " + ", ".join(missing_close_margin),
    )
    print(
        "PASS eval contracts: "
        f"{len(retrieval_cases)} retrieval cases, {len(answer_cases)} answer cases"
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    _assert(path.exists(), f"missing {path.relative_to(ROOT)}")
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{path.relative_to(ROOT)} line {line_no} is not valid JSON: {exc}") from exc
        _assert(isinstance(item, dict), f"{path.relative_to(ROOT)} line {line_no} must be a JSON object")
        cases.append(item)
    _assert(cases, f"{path.relative_to(ROOT)} should contain at least one case")
    return cases


def _assert_unique_ids(cases: list[dict[str, Any]], label: str) -> None:
    ids = [str(case.get("id", "")) for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    _assert(not duplicates, f"{label} ids must be unique: {', '.join(duplicates)}")


def _assert_eval_case_shape(case: dict[str, Any], id_prefix: str, require_board_id: bool) -> None:
    case_id = str(case.get("id", ""))
    _assert(case_id.startswith(id_prefix), f"case id should start with {id_prefix}: {case_id or '<missing>'}")
    _assert(str(case.get("query", "")).strip(), f"{case_id} query must be non-empty")
    _assert_string_list(case, "must_include", case_id)
    _assert_string_list(case, "must_cite", case_id)
    if require_board_id:
        _assert("expected_board_id" in case, f"{case_id} must declare expected_board_id")
        _assert(isinstance(case.get("expected_board_id"), str), f"{case_id} expected_board_id must be a string")
    for citation in case["must_cite"]:
        _assert(
            str(citation).startswith("https://wiki.seeedstudio.com/"),
            f"{case_id} citation should be a Seeed wiki URL: {citation}",
        )


def _assert_string_list(case: dict[str, Any], key: str, case_id: str) -> None:
    values = case.get(key)
    _assert(isinstance(values, list) and values, f"{case_id} {key} must be a non-empty list")
    invalid = [value for value in values if not isinstance(value, str) or not value.strip()]
    _assert(not invalid, f"{case_id} {key} entries must be non-empty strings")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
