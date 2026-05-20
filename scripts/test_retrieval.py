from __future__ import annotations

import sys
from pathlib import Path
from time import sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xiao_copilot.retrieval as retrieval
from xiao_copilot.config import Settings
from xiao_copilot.knowledge_base import KnowledgeChunk
from xiao_copilot.retrieval import (
    RERANK_METADATA_TAG_LIMIT,
    _configuration_detail_score,
    _include_rerank_board_metadata,
    _include_rerank_source_topic_metadata,
    _rerank_text,
)


def main() -> None:
    _assert_rerank_text_includes_useful_metadata()
    _assert_rerank_text_adds_board_metadata_only_when_requested()
    _assert_rerank_text_adds_source_topic_only_when_requested()
    _assert_rerank_text_stays_compact_without_metadata()
    _assert_board_metadata_choice_heuristic()
    _assert_source_topic_metadata_heuristic()
    _assert_detail_scores_promote_exact_configuration_chunks()
    _assert_rerank_heartbeats_surface_wait_progress()
    _assert_rerank_result_cache_is_keyed_to_exact_inputs()
    _assert_retrieve_progressive_reuses_cached_rerank_result()
    print("PASS retrieval formatting regression")


def _assert_rerank_text_includes_useful_metadata() -> None:
    tags = [f"tag-{index}" for index in range(RERANK_METADATA_TAG_LIMIT + 3)]
    chunk = KnowledgeChunk(
        id="chunk-1",
        title="XIAO board setup",
        source="https://example.test/xiao",
        text="abcdefghijklmnopqrstuvwxyz",
        board_id="xiao-esp32c3",
        metadata={
            "aliases": ["XIAO ESP32C3"],
            "tags": tags,
        },
    )
    text = _rerank_text(chunk, max_chars=10)
    _assert("Metadata: tags=tag-0, tag-1" in text, "tags should be included")
    _assert("tag-15" in text and "tag-16" not in text, "reranker tags should be bounded")
    _assert("board=xiao-esp32c3" not in text, "board metadata should be opt-in")
    _assert("XIAO ESP32C3" not in text, "aliases should not be added to reranker metadata")
    _assert(text.endswith("abcdefghij"), "content should be truncated after metadata is assembled")


def _assert_rerank_text_adds_board_metadata_only_when_requested() -> None:
    chunk = KnowledgeChunk(
        id="chunk-board",
        title="XIAO board setup",
        source="https://example.test/xiao",
        text="setup details",
        board_id="xiao-esp32c3",
        metadata={"tags": ["xiao", "esp32c3"]},
    )
    text = _rerank_text(chunk, max_chars=100, include_board_metadata=True)
    _assert(
        "Metadata: board=xiao-esp32c3; tags=xiao, esp32c3" in text,
        "board metadata should be included when requested",
    )


def _assert_rerank_text_adds_source_topic_only_when_requested() -> None:
    chunk = KnowledgeChunk(
        id="chunk-topic",
        title="Wio-SX1262 kit",
        source="https://example.test/wio",
        text="kit details",
        metadata={
            "source_file": "Network/LoRa_Wio_Series/Wio_SX1262/Wio_SX1262_and_XIAO_ESP32S3_kit_with_3DPrinted_Enclosure_introduction_and_assembly_guide.md",
            "tags": ["lora"],
        },
    )
    text = _rerank_text(chunk, max_chars=100, include_source_topic_metadata=True)
    _assert(
        "source_topic=Wio SX1262 and XIAO ESP32S3 kit with 3DPrinted Enclosure introduction and assembly guide" in text,
        "source topic should be included when requested",
    )
    text_without_topic = _rerank_text(chunk, max_chars=100)
    _assert("source_topic=" not in text_without_topic, "source topic metadata should be opt-in")


def _assert_rerank_text_stays_compact_without_metadata() -> None:
    chunk = KnowledgeChunk(
        id="chunk-2",
        title="General Grove note",
        source="https://example.test/grove",
        text=" short note ",
    )
    text = _rerank_text(chunk, max_chars=100)
    _assert("Metadata:" not in text, "empty metadata should not add a metadata line")
    _assert(text == "General Grove note\nhttps://example.test/grove\nshort note", "basic reranker text changed")


def _assert_board_metadata_choice_heuristic() -> None:
    c3 = KnowledgeChunk(id="c3", title="", source="", text="", board_id="xiao-esp32c3")
    s3 = KnowledgeChunk(id="s3", title="", source="", text="", board_id="xiao-esp32s3")
    _assert(
        _include_rerank_board_metadata("Which XIAO board supports channel sounding?", [c3, s3]),
        "board-choice queries with distinct boards should include board metadata",
    )
    _assert(
        not _include_rerank_board_metadata("Which XIAO board supports channel sounding?", [c3, c3]),
        "same-board candidates should not include board metadata",
    )
    _assert(
        not _include_rerank_board_metadata("What I2C address is used on XIAO ESP32C3?", [c3, s3]),
        "non-choice board-specific queries should not include board metadata",
    )


def _assert_source_topic_metadata_heuristic() -> None:
    _assert(
        _include_rerank_source_topic_metadata("What can the XIAO ESP32S3 & Wio-SX1262 kit with 3D case be used for?"),
        "3D case queries should include source-topic metadata",
    )
    _assert(
        _include_rerank_source_topic_metadata("For XIAO ESP32S3 Sense model output over I2C, what pins are used?"),
        "model-output protocol queries should include source-topic metadata",
    )
    _assert(
        not _include_rerank_source_topic_metadata("What can the XIAO ESP32S3 Wio-SX1262 kit be used for?"),
        "generic kit queries should not include source-topic metadata",
    )


def _assert_detail_scores_promote_exact_configuration_chunks() -> None:
    esphome_chunk = KnowledgeChunk(
        id="esphome",
        title="Add Seeed Studio XIAO ESP32C3 to ESPHome",
        source="https://example.test/esphome",
        text="esp32:\n  board: seeed_xiao_esp32c3\n  variant: esp32c3\n  framework:\n    type: arduino\n    version: 2.0.5\n    platform_version: 5.2.0",
        board_id="xiao-esp32c3",
    )
    _assert(
        _configuration_detail_score(
            "What ESPHome YAML board settings should I use for Seeed Studio XIAO ESP32C3?",
            esphome_chunk,
        )
        >= 1.0,
        "ESPHome YAML questions should promote exact board setting chunks",
    )

    camera_chunk = KnowledgeChunk(
        id="camera",
        title="Camera slot circuit design for expansion boards",
        source="https://example.test/camera",
        text="The XIAO ESP32S3 Sense card slot occupies 14 GPIOs. GPIO39 | CAM_SCL | GPIO40 | CAM_SDA",
        board_id="xiao-esp32s3",
    )
    _assert(
        _configuration_detail_score(
            "How many GPIOs does the camera slot occupy, and which GPIOs are CAM_SCL and CAM_SDA?",
            camera_chunk,
        )
        >= 2.0,
        "camera-slot pin questions should promote the exact occupancy table",
    )


def _assert_rerank_heartbeats_surface_wait_progress() -> None:
    original_rerank = retrieval.rerank
    original_heartbeat = retrieval.RERANK_PROGRESS_HEARTBEAT_SECONDS
    try:
        retrieval.RERANK_PROGRESS_HEARTBEAT_SECONDS = 0.001

        def slow_rerank(**_kwargs):
            sleep(0.01)
            return retrieval.EndpointResult(ok=True, data=[(0, 0.9)])

        retrieval.rerank = slow_rerank  # type: ignore[assignment]
        events = list(
            retrieval._rerank_with_heartbeats(
                base_url="http://reranker.test",
                model="rerank-test",
                query="Which board?",
                documents=["doc"],
                api_key="",
                timeout=1,
            )
        )
        _assert(any(event is None for event in events), "slow rerank should emit wait heartbeat events")
        _assert(
            isinstance(events[-1], retrieval.EndpointResult) and events[-1].ok,
            "rerank heartbeat wrapper should yield the final rerank result",
        )
    finally:
        retrieval.rerank = original_rerank  # type: ignore[assignment]
        retrieval.RERANK_PROGRESS_HEARTBEAT_SECONDS = original_heartbeat


def _assert_rerank_result_cache_is_keyed_to_exact_inputs() -> None:
    retrieval._clear_rerank_result_cache()
    chunk = KnowledgeChunk(
        id="chunk-cache",
        title="Cacheable Rerank Chunk",
        source="https://example.test/cache",
        text="LoRa setup details",
    )
    key = retrieval._rerank_cache_key(
        base_url="https://reranker.test/v1/",
        model="rerank-test",
        query=" Which   board supports LoRa? ",
        chunks=[chunk],
        documents=["LoRa setup details"],
        rerank_text_chars=2800,
        include_board_metadata=False,
        include_source_topic_metadata=False,
    )
    equivalent_key = retrieval._rerank_cache_key(
        base_url="https://reranker.test/v1",
        model="rerank-test",
        query="which board supports lora?",
        chunks=[chunk],
        documents=["LoRa setup details"],
        rerank_text_chars=2800,
        include_board_metadata=False,
        include_source_topic_metadata=False,
    )
    changed_key = retrieval._rerank_cache_key(
        base_url="https://reranker.test/v1",
        model="rerank-test",
        query="which board supports lora?",
        chunks=[chunk],
        documents=["Different setup details"],
        rerank_text_chars=2800,
        include_board_metadata=False,
        include_source_topic_metadata=False,
    )
    _assert(key == equivalent_key, "reranker cache key should normalize endpoint slash and query whitespace")
    _assert(key != changed_key, "reranker cache key should change when rerank text changes")

    result = retrieval.EndpointResult(ok=True, data=[(0, 0.9)], meta={"mode": "native"})
    retrieval._store_cached_rerank_result(key, result)
    cached = retrieval._get_cached_rerank_result(equivalent_key)
    _assert(cached is not None and cached.ok, "stored reranker result should be reusable")
    _assert(cached is not result, "cached reranker result should be cloned before reuse")
    _assert(cached.data == [(0, 0.9)], "cached reranker scores changed")
    _assert((cached.meta or {}).get("cached") is True, "cache hits should be marked in result metadata")
    _assert(result.meta == {"mode": "native"}, "cache hit metadata should not mutate the original result")
    _assert(retrieval._get_cached_rerank_result(changed_key) is None, "changed rerank text should miss the cache")
    retrieval._clear_rerank_result_cache()


def _assert_retrieve_progressive_reuses_cached_rerank_result() -> None:
    original_load_knowledge_base = retrieval.load_knowledge_base
    original_rerank_with_heartbeats = retrieval._rerank_with_heartbeats
    retrieval._clear_rerank_result_cache()
    calls: list[dict[str, object]] = []
    chunks = [
        KnowledgeChunk(
            id="lora",
            title="XIAO LoRa setup",
            source="https://example.test/lora",
            text="The LoRa expansion supports long-range radio projects.",
        ),
        KnowledgeChunk(
            id="wifi",
            title="XIAO Wi-Fi setup",
            source="https://example.test/wifi",
            text="Wi-Fi setup details for a wireless sensor project.",
        ),
    ]
    settings = Settings(
        embedding_base_url="",
        rerank_base_url="https://reranker.test/v1",
        rerank_model="rerank-test",
        request_timeout_seconds=1,
        top_k=1,
        candidate_k=2,
        vector_candidate_k=2,
    )

    def fake_load_knowledge_base() -> list[KnowledgeChunk]:
        return chunks

    def fake_rerank_with_heartbeats(**kwargs):
        calls.append(kwargs)
        yield retrieval.EndpointResult(ok=True, data=[(0, 0.9), (1, 0.1)], meta={"mode": "native"})

    try:
        retrieval.load_knowledge_base = fake_load_knowledge_base  # type: ignore[assignment]
        retrieval._rerank_with_heartbeats = fake_rerank_with_heartbeats  # type: ignore[assignment]
        first_stages = list(retrieval.retrieve_progressive("Which XIAO board supports LoRa?", settings))
        first_final = first_stages[-1]
        second_stages = list(retrieval.retrieve_progressive("Which XIAO board supports LoRa?", settings))
        second_final = second_stages[-1]
        _assert(len(calls) == 1, "second identical retrieval should reuse cached rerank scores")
        _assert(first_final.diagnostics["reranker_cache_hit"] is False, "first retrieval should miss rerank cache")
        _assert(second_final.diagnostics["reranker_cache_hit"] is True, "second retrieval should hit rerank cache")
        _assert(
            not any(stage.stage == "rerank_wait" for stage in second_stages),
            "cached rerank should not emit hosted reranker wait events",
        )
        _assert(
            first_final.diagnostics["citations"] == second_final.diagnostics["citations"],
            "cached rerank should preserve final citation ordering",
        )
    finally:
        retrieval.load_knowledge_base = original_load_knowledge_base  # type: ignore[assignment]
        retrieval._rerank_with_heartbeats = original_rerank_with_heartbeats  # type: ignore[assignment]
        retrieval._clear_rerank_result_cache()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
