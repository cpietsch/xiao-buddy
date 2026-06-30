from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xiao_copilot.knowledge_base import KnowledgeChunk
from xiao_copilot.knowledge_graph import (
    build_knowledge_graph,
    get_knowledge_graph,
    graph_source_hash,
    graph_summary,
    load_knowledge_graph_artifact,
)


def main() -> None:
    _assert_source_title_aliases_recover_product_pages()
    _assert_entity_graph_retrieves_related_robotics_chunks()
    _assert_graph_artifact_roundtrip()
    print("PASS knowledge graph regression")


def _assert_source_title_aliases_recover_product_pages() -> None:
    chunks = [
        KnowledgeChunk(
            id="base-hat-specs",
            title="Grove Base Hat for Raspberry Pi Zero > Specifications",
            source="https://wiki.seeedstudio.com/Grove_Base_Hat_for_Raspberry_Pi_Zero/",
            text="Operating voltage is 3.3V. It has a 12-bit ADC, 6 channel ADC, and I2C address 0x04.",
            metadata={
                "citations": [
                    {
                        "title": "Grove Base Hat for Raspberry Pi Zero",
                        "url": "https://wiki.seeedstudio.com/Grove_Base_Hat_for_Raspberry_Pi_Zero/",
                    }
                ],
                "source_file": "Top_Brand/Raspberry_Pi/Pi_HAT/Grove_Base_Hat_for_Raspberry_Pi_Zero.md",
                "tags": ["wiki", "raspberrypi", "grove", "i2c"],
            },
        ),
        KnowledgeChunk(
            id="unrelated",
            title="XIAO ESP32C3 I2C",
            source="https://example.test/xiao",
            text="D4 is SDA and D5 is SCL.",
            board_id="xiao-esp32c3",
        ),
    ]
    graph = build_knowledge_graph(chunks)
    matches = graph.rank_chunks("What voltage and I2C address does Grove Base Hat for Raspberry Pi Zero use?")
    _assert(matches, "source-title graph should return matches")
    _assert(matches[0].chunk_id == "base-hat-specs", "exact product title should rank its source chunk first")
    _assert(graph_summary(graph)["source_entities"] >= 1, "graph summary should include source entities")


def _assert_entity_graph_retrieves_related_robotics_chunks() -> None:
    chunks = [
        KnowledgeChunk(
            id="bus-servo",
            title="Getting Started with XIAO Bus Servo Adapter",
            source="https://wiki.seeedstudio.com/xiao_bus_servo_adapter/",
            text="XIAO Bus Servo Adapter can directly control bus servos from a XIAO ESP32-C3 in a 3D-printed case.",
            board_id="xiao-esp32c3",
            metadata={
                "aliases": ["XIAO ESP32C3", "XIAO ESP32-C3", "xiao-esp32c3"],
                "citations": [
                    {
                        "title": "Getting Started with XIAO Bus Servo Adapter",
                        "url": "https://wiki.seeedstudio.com/xiao_bus_servo_adapter/",
                    }
                ],
                "tags": ["wiki", "robotics", "bus", "servo", "adapter"],
            },
        ),
        KnowledgeChunk(
            id="battery",
            title="Power budget and brownout checks",
            source="local field note",
            text="Measure 5V and 3V3 under load before adding peripherals.",
        ),
    ]
    graph = build_knowledge_graph(chunks)
    matches = graph.rank_chunks("Which XIAO-based board should I use for a ready-to-use bus servo robotics controller?")
    _assert(matches, "robotics query should return graph matches")
    _assert(matches[0].chunk_id == "bus-servo", "bus servo product should outrank unrelated field notes")


def _assert_graph_artifact_roundtrip() -> None:
    chunks = [
        KnowledgeChunk(
            id="roundtrip",
            title="Grove Base Hat for Raspberry Pi Zero",
            source="https://wiki.seeedstudio.com/Grove_Base_Hat_for_Raspberry_Pi_Zero/",
            text="The Grove Base Hat uses I2C and exposes Grove ports.",
            metadata={
                "citations": [
                    {
                        "title": "Grove Base Hat for Raspberry Pi Zero",
                        "url": "https://wiki.seeedstudio.com/Grove_Base_Hat_for_Raspberry_Pi_Zero/",
                    }
                ],
                "tags": ["grove", "i2c"],
            },
        )
    ]
    with tempfile.TemporaryDirectory(prefix="xiao-graph-test-") as temp_dir:
        artifact_path = Path(temp_dir) / "knowledge_graph.json"
        built = get_knowledge_graph(chunks, artifact_path=str(artifact_path))
        _assert(artifact_path.exists(), "graph artifact should be written after build")
        _assert(
            built.metadata.get("artifact_status") == "built_and_saved",
            "first graph load should build and save the artifact",
        )

        loaded = load_knowledge_graph_artifact(
            artifact_path,
            expected_source_hash=graph_source_hash(chunks),
        )
        _assert(loaded is not None, "fresh graph artifact should load")
        _assert(
            loaded.rank_chunks("Which Grove Base Hat uses I2C?")[0].chunk_id == "roundtrip",
            "loaded graph artifact should preserve ranking behavior",
        )
        stale = load_knowledge_graph_artifact(artifact_path, expected_source_hash="stale")
        _assert(stale is None, "stale source hash should reject graph artifact")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
