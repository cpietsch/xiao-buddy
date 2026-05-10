from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class KnowledgeChunk:
    id: str
    title: str
    source: str
    text: str
    board_id: str = ""
    kind: str = "note"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def search_text(self) -> str:
        aliases = " ".join(self.metadata.get("aliases", []))
        tags = " ".join(self.metadata.get("tags", []))
        return f"{self.title}\n{aliases}\n{tags}\n{self.text}"


_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_PATH = _ROOT / "data" / "corpus" / "xiao_boards.json"


FIELD_NOTES: list[KnowledgeChunk] = [
    KnowledgeChunk(
        id="field-usb-boot",
        title="USB detection and bootloader recovery",
        source="local field note",
        text=(
            "If a XIAO board is not detected over USB, first verify the USB cable supports data, "
            "try another host port, disconnect peripherals, and watch for a bootloader device. "
            "Then use the board-specific bootloader entry method and flash a minimal blink sketch."
        ),
    ),
    KnowledgeChunk(
        id="field-power-brownout",
        title="Power budget and brownout checks",
        source="local field note",
        text=(
            "Resets during sensor startup usually point to voltage sag, inrush current, a weak "
            "USB supply, or a peripheral overloading the 3V3 rail. Measure 5V and 3V3 under load, "
            "then add one peripheral at a time."
        ),
    ),
    KnowledgeChunk(
        id="field-visual-inspection",
        title="Visual inspection before power",
        source="local field note",
        text=(
            "Before powering an unknown XIAO setup, inspect both sides for solder bridges, reversed "
            "connectors, damaged pads, stray wire strands, overheated parts, and headers shifted by "
            "one pin."
        ),
    ),
]


@lru_cache(maxsize=1)
def load_knowledge_base() -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    if _CORPUS_PATH.exists():
        data = json.loads(_CORPUS_PATH.read_text())
        for doc in data.get("documents", []):
            chunks.extend(_chunks_for_board(doc))
    return chunks + FIELD_NOTES


def _chunks_for_board(doc: dict[str, Any]) -> list[KnowledgeChunk]:
    board_id = doc["id"]
    aliases = doc.get("aliases", [])
    tags = doc.get("retrieval_tags", [])
    citation_url = _primary_citation(doc)
    common = {
        "aliases": aliases,
        "tags": tags,
        "citations": doc.get("citations", []),
        "image_urls": doc.get("image_urls", {}),
    }

    identity_parts = [
        f"Aliases: {', '.join(aliases)}.",
        f"Summary: {doc.get('summary', '')}",
        f"MCU: {_format_mcu(doc.get('mcu', {}))}.",
        f"Connectivity: {', '.join(doc.get('connectivity', []) or ['none'])}.",
        f"Onboard features: {', '.join(doc.get('onboard_features', []))}.",
    ]
    if doc.get("sense_features"):
        identity_parts.append(f"Sense features: {', '.join(doc['sense_features'])}.")

    chunks = [
        KnowledgeChunk(
            id=f"{board_id}-identity",
            title=f"{doc['title']} identity and capabilities",
            source=citation_url,
            text=" ".join(identity_parts),
            board_id=board_id,
            kind="identity",
            metadata=common,
        )
    ]

    pin_rows = [
        f"{pin['xiao_pin']} maps to {pin.get('chip_pin', 'unknown')} for {', '.join(pin.get('functions', []))}"
        for pin in doc.get("pin_map", [])
    ]
    if pin_rows:
        chunks.append(
            KnowledgeChunk(
                id=f"{board_id}-pinout",
                title=f"{doc['title']} pin map",
                source=citation_url,
                text="; ".join(pin_rows) + ".",
                board_id=board_id,
                kind="pinout",
                metadata=common,
            )
        )

    gotchas = doc.get("field_gotchas", [])
    if gotchas:
        chunks.append(
            KnowledgeChunk(
                id=f"{board_id}-gotchas",
                title=f"{doc['title']} field gotchas",
                source=citation_url,
                text=" ".join(f"{item}" for item in gotchas),
                board_id=board_id,
                kind="gotchas",
                metadata=common,
            )
        )

    return chunks


def _primary_citation(doc: dict[str, Any]) -> str:
    citations = doc.get("citations") or []
    if not citations:
        return "local corpus"
    return citations[0].get("url", "local corpus")


def _format_mcu(mcu: dict[str, Any]) -> str:
    return " ".join(str(mcu.get(key, "")).strip() for key in ("vendor", "part", "architecture")).strip()
