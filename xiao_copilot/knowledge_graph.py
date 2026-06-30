from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from xiao_copilot.knowledge_base import KnowledgeChunk


_ROOT = Path(__file__).resolve().parents[1]
GRAPH_ARTIFACT_SCHEMA_VERSION = 1
DEFAULT_GRAPH_ARTIFACT = _ROOT / "data" / "index" / "knowledge_graph.json"


@dataclass(frozen=True)
class GraphEntity:
    key: str
    label: str
    kind: str
    weight: float


@dataclass(frozen=True)
class GraphMatch:
    chunk_id: str
    score: float
    matched_entities: tuple[str, ...]


@dataclass
class KnowledgeGraph:
    entities: dict[str, GraphEntity]
    aliases: dict[str, str]
    compact_aliases: dict[str, str]
    chunk_entities: dict[str, frozenset[str]]
    entity_chunks: dict[str, tuple[str, ...]]
    metadata: dict[str, object] = field(default_factory=dict)

    def entities_for_text(self, text: str) -> frozenset[str]:
        normalized = _normalize_phrase(text)
        compact = _compact_phrase(text)
        matched: set[str] = set()
        if not normalized:
            return frozenset()

        padded = f" {normalized} "
        for alias, key in self.aliases.items():
            if f" {alias} " in padded:
                matched.add(key)

        for alias, key in self.compact_aliases.items():
            if alias in compact:
                matched.add(key)

        for token in _identifier_tokens(text):
            key = self.aliases.get(_normalize_phrase(token))
            if key:
                matched.add(key)

        return frozenset(matched)

    def rank_chunks(self, query: str, limit: int = 64) -> list[GraphMatch]:
        query_entities = self.entities_for_text(query)
        if not query_entities:
            return []

        entity_counts = {key: len(self.entity_chunks.get(key, ())) for key in query_entities}
        broad_cutoff = max(64, int(max(len(self.chunk_entities), 1) * 0.08))
        query_has_specific_entity = any(
            self.entities[key].kind in {"board", "identifier", "source", "product"}
            for key in query_entities
            if key in self.entities
        )
        scores: Counter[str] = Counter()
        matches: dict[str, set[str]] = defaultdict(set)
        for key in query_entities:
            entity = self.entities.get(key)
            if entity is None:
                continue
            chunk_ids = self.entity_chunks.get(key, ())
            if len(chunk_ids) > broad_cutoff and query_has_specific_entity and entity.kind not in {
                "board",
                "identifier",
                "source",
                "product",
            }:
                continue
            fanout_penalty = _fanout_penalty(entity_counts[key])
            for chunk_id in chunk_ids:
                scores[chunk_id] += entity.weight * fanout_penalty
                matches[chunk_id].add(key)

        ranked: list[GraphMatch] = []
        for chunk_id, score in scores.items():
            matched = tuple(sorted(matches[chunk_id]))
            if len(matched) >= 2:
                score += 0.35 * (len(matched) - 1)
            ranked.append(GraphMatch(chunk_id=chunk_id, score=float(score), matched_entities=matched))
        ranked.sort(key=lambda match: (match.score, len(match.matched_entities)), reverse=True)
        return ranked[:limit]


_GRAPH_CACHE_KEY: tuple[str, str] | None = None
_GRAPH_CACHE: KnowledgeGraph | None = None


def get_knowledge_graph(
    chunks: list[KnowledgeChunk],
    artifact_path: str = "",
) -> KnowledgeGraph:
    global _GRAPH_CACHE, _GRAPH_CACHE_KEY
    source_hash = graph_source_hash(chunks)
    path = configured_graph_artifact_path(artifact_path)
    cache_key = (source_hash, str(path))
    if _GRAPH_CACHE is not None and _GRAPH_CACHE_KEY == cache_key:
        return _GRAPH_CACHE

    graph = load_knowledge_graph_artifact(path, expected_source_hash=source_hash)
    if graph is not None:
        _GRAPH_CACHE = graph
        _GRAPH_CACHE_KEY = cache_key
        return graph

    graph = build_knowledge_graph(chunks)
    graph.metadata = {
        "artifact_path": str(path),
        "artifact_status": "built",
        "source_hash": source_hash,
    }
    try:
        save_knowledge_graph_artifact(graph, path, source_hash=source_hash)
        graph.metadata = {
            **graph.metadata,
            "artifact_status": "built_and_saved",
        }
    except Exception as exc:  # noqa: BLE001 - retrieval should continue without persistent cache.
        graph.metadata = {
            **graph.metadata,
            "artifact_status": "built_save_failed",
            "artifact_error": str(exc),
        }
    _GRAPH_CACHE = graph
    _GRAPH_CACHE_KEY = cache_key
    return graph


def configured_graph_artifact_path(artifact_path: str = "") -> Path:
    path = Path(artifact_path).expanduser() if artifact_path else DEFAULT_GRAPH_ARTIFACT
    if not path.is_absolute():
        path = _ROOT / path
    return path


def build_knowledge_graph(chunks: list[KnowledgeChunk]) -> KnowledgeGraph:
    builder = _GraphBuilder()
    for chunk in chunks:
        builder.add_chunk(chunk)
    return builder.build()


def graph_source_hash(chunks: list[KnowledgeChunk]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.title.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.source.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.board_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.kind.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_graph_metadata_signature(chunk).encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def load_knowledge_graph_artifact(
    artifact_path: Path,
    *,
    expected_source_hash: str = "",
) -> KnowledgeGraph | None:
    if not artifact_path.exists():
        return None
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != GRAPH_ARTIFACT_SCHEMA_VERSION:
        return None
    source_hash = str(payload.get("source_hash", ""))
    if expected_source_hash and source_hash != expected_source_hash:
        return None

    entities_payload = payload.get("entities", {})
    if not isinstance(entities_payload, dict):
        return None
    aliases = payload.get("aliases", {})
    compact_aliases = payload.get("compact_aliases", {})
    chunk_entities = payload.get("chunk_entities", {})
    entity_chunks = payload.get("entity_chunks", {})
    if not all(
        isinstance(value, dict)
        for value in (aliases, compact_aliases, chunk_entities, entity_chunks)
    ):
        return None

    entities: dict[str, GraphEntity] = {}
    for key, item in entities_payload.items():
        if not isinstance(item, dict):
            return None
        entity_key = str(item.get("key", key))
        entities[entity_key] = GraphEntity(
            key=entity_key,
            label=str(item.get("label", "")),
            kind=str(item.get("kind", "")),
            weight=float(item.get("weight", 0.0)),
        )

    graph = KnowledgeGraph(
        entities=entities,
        aliases={str(key): str(value) for key, value in aliases.items()},
        compact_aliases={str(key): str(value) for key, value in compact_aliases.items()},
        chunk_entities={
            str(chunk_id): frozenset(str(entity_id) for entity_id in entity_ids)
            for chunk_id, entity_ids in chunk_entities.items()
            if isinstance(entity_ids, list)
        },
        entity_chunks={
            str(entity_id): tuple(str(chunk_id) for chunk_id in chunk_ids)
            for entity_id, chunk_ids in entity_chunks.items()
            if isinstance(chunk_ids, list)
        },
        metadata={
            "artifact_path": str(artifact_path),
            "artifact_status": "loaded",
            "source_hash": source_hash,
        },
    )
    return graph


def save_knowledge_graph_artifact(
    graph: KnowledgeGraph,
    artifact_path: Path,
    *,
    source_hash: str,
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": GRAPH_ARTIFACT_SCHEMA_VERSION,
        "source_hash": source_hash,
        "summary": graph_summary(graph),
        "entities": {
            key: {
                "key": entity.key,
                "label": entity.label,
                "kind": entity.kind,
                "weight": entity.weight,
            }
            for key, entity in sorted(graph.entities.items())
        },
        "aliases": dict(sorted(graph.aliases.items())),
        "compact_aliases": dict(sorted(graph.compact_aliases.items())),
        "chunk_entities": {
            chunk_id: sorted(entity_ids)
            for chunk_id, entity_ids in sorted(graph.chunk_entities.items())
        },
        "entity_chunks": {
            entity_id: list(chunk_ids)
            for entity_id, chunk_ids in sorted(graph.entity_chunks.items())
        },
    }
    tmp_path = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    tmp_path.replace(artifact_path)


class _GraphBuilder:
    def __init__(self) -> None:
        self.entities: dict[str, GraphEntity] = {}
        self.aliases: dict[str, str] = {}
        self.compact_aliases: dict[str, str] = {}
        self.chunk_entities: dict[str, set[str]] = defaultdict(set)
        self.entity_chunks: dict[str, set[str]] = defaultdict(set)

    def add_chunk(self, chunk: KnowledgeChunk) -> None:
        if chunk.board_id:
            board_label = _board_label(chunk)
            board_key = self._add_entity("board", board_label, 3.2)
            self._link(chunk.id, board_key)
            self._add_alias(chunk.board_id, board_key, compact=True)
            self._add_alias(board_label, board_key, compact=True)
            for alias in _string_items(chunk.metadata.get("aliases", [])):
                self._add_alias(alias, board_key, compact=True)

        source_key = self._source_entity(chunk)
        if source_key:
            self._link(chunk.id, source_key)

        for tag in _string_items(chunk.metadata.get("tags", [])):
            if _useful_tag(tag):
                tag_key = self._add_entity("tag", _normalize_label(tag), 0.65)
                self._link(chunk.id, tag_key)
                self._add_alias(tag, tag_key, compact=False)

        search_text = _graph_search_text(chunk)
        search_lower = _search_lower(search_text)
        for phrase, kind, weight, phrase_lower in _KNOWN_PHRASE_ENTRIES:
            if f" {phrase_lower} " in search_lower:
                key = self._add_entity(kind, phrase, weight)
                self._link(chunk.id, key)
                self._add_alias(phrase, key, compact=True)

        for identifier in _identifier_tokens(_identifier_search_text(chunk)):
            if not _useful_identifier(identifier):
                continue
            key = self._add_entity("identifier", identifier, 2.25)
            self._link(chunk.id, key)
            self._add_alias(identifier, key, compact=True)

    def build(self) -> KnowledgeGraph:
        return KnowledgeGraph(
            entities=dict(self.entities),
            aliases=dict(self.aliases),
            compact_aliases=dict(self.compact_aliases),
            chunk_entities={
                chunk_id: frozenset(entity_ids)
                for chunk_id, entity_ids in self.chunk_entities.items()
            },
            entity_chunks={
                entity_id: tuple(sorted(chunk_ids))
                for entity_id, chunk_ids in self.entity_chunks.items()
            },
        )

    def _source_entity(self, chunk: KnowledgeChunk) -> str:
        labels = _source_aliases(chunk)
        if not labels:
            return ""
        key = self._add_entity("source", labels[0], 3.0)
        for label in labels:
            self._add_alias(label, key, compact=True)
        return key

    def _add_entity(self, kind: str, label: str, weight: float) -> str:
        normalized = _normalize_phrase(label)
        if not normalized:
            return ""
        key = f"{kind}:{normalized}"
        self.entities.setdefault(
            key,
            GraphEntity(key=key, label=_normalize_label(label), kind=kind, weight=weight),
        )
        return key

    def _add_alias(self, alias: str, key: str, *, compact: bool) -> None:
        normalized = _normalize_phrase(alias)
        if not key or not _useful_alias(normalized):
            return
        self.aliases.setdefault(normalized, key)
        if compact:
            compact_alias = _compact_phrase(alias)
            if _useful_compact_alias(compact_alias):
                self.compact_aliases.setdefault(compact_alias, key)

    def _link(self, chunk_id: str, entity_key: str) -> None:
        if not entity_key:
            return
        self.chunk_entities[chunk_id].add(entity_key)
        self.entity_chunks[entity_key].add(chunk_id)


def graph_summary(graph: KnowledgeGraph) -> dict[str, int]:
    entity_kinds = Counter(entity.kind for entity in graph.entities.values())
    return {
        "entities": len(graph.entities),
        "aliases": len(graph.aliases) + len(graph.compact_aliases),
        "chunk_links": sum(len(entities) for entities in graph.chunk_entities.values()),
        **{f"{kind}_entities": count for kind, count in sorted(entity_kinds.items())},
    }


def _source_aliases(chunk: KnowledgeChunk) -> list[str]:
    labels: list[str] = []
    for citation in chunk.metadata.get("citations", []):
        if isinstance(citation, dict):
            labels.extend(_split_source_title(str(citation.get("title", ""))))
    source_file = str(chunk.metadata.get("source_file", "")).strip()
    if source_file:
        labels.extend(_split_source_title(source_file.rsplit("/", 1)[-1].rsplit(".", 1)[0]))
    if chunk.source:
        slug = chunk.source.rstrip("/").rsplit("/", 1)[-1]
        labels.extend(_split_source_title(slug))
    return _unique_aliases(labels)


def _graph_metadata_signature(chunk: KnowledgeChunk) -> str:
    metadata: dict[str, Any] = {
        "aliases": _string_items(chunk.metadata.get("aliases", [])),
        "tags": _string_items(chunk.metadata.get("tags", [])),
        "source_file": str(chunk.metadata.get("source_file", "")),
        "citations": [],
    }
    citations: list[dict[str, str]] = []
    for citation in chunk.metadata.get("citations", []):
        if not isinstance(citation, dict):
            continue
        citations.append(
            {
                "title": str(citation.get("title", "")),
                "url": str(citation.get("url", "")),
            }
        )
    metadata["citations"] = citations
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _split_source_title(title: str) -> list[str]:
    cleaned = re.sub(r"[_-]+", " ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []
    aliases = [cleaned]
    trimmed = re.sub(
        r"\s+(overview|getting started|introduction|guide|tutorial)$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if len(trimmed) >= 8 and _normalize_phrase(trimmed) != _normalize_phrase(cleaned):
        aliases.append(trimmed)
    for separator in (" with ", " for ", " on ", " by "):
        if separator in cleaned.lower():
            first = re.split(separator, cleaned, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            if len(first) >= 8:
                aliases.append(first)
    return aliases


def _unique_aliases(aliases: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for alias in aliases:
        normalized = _normalize_phrase(alias)
        if normalized in seen or not _useful_alias(normalized):
            continue
        seen.add(normalized)
        unique.append(alias)
    return unique


def _board_label(chunk: KnowledgeChunk) -> str:
    for alias in _string_items(chunk.metadata.get("aliases", [])):
        if "xiao" in alias.lower():
            return alias
    return chunk.board_id.replace("-", " ").upper()


def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ").replace("-", " ")).strip()


def _normalize_phrase(value: str) -> str:
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-zA-Z0-9.+]+", " ", value.lower())
    return re.sub(r"\s+", " ", value).strip()


def _compact_phrase(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _search_lower(value: str) -> str:
    value = value.lower()
    value = value.replace("_", " ").replace("-", " ").replace("/", " ")
    collapsed = re.sub(r"\s+", " ", value)
    return f" {collapsed} "


def _graph_search_text(chunk: KnowledgeChunk) -> str:
    aliases = " ".join(_string_items(chunk.metadata.get("aliases", [])))
    tags = " ".join(_string_items(chunk.metadata.get("tags", [])))
    return f"{chunk.title}\n{aliases}\n{tags}\n{chunk.text[:6000]}"


def _identifier_search_text(chunk: KnowledgeChunk) -> str:
    aliases = " ".join(_string_items(chunk.metadata.get("aliases", [])))
    tags = " ".join(_string_items(chunk.metadata.get("tags", [])))
    source_file = str(chunk.metadata.get("source_file", ""))
    return f"{chunk.title}\n{aliases}\n{tags}\n{source_file}\n{chunk.text[:1800]}"


def _identifier_tokens(text: str) -> set[str]:
    return {
        token.strip(".,:;()[]{}")
        for token in _IDENTIFIER_RE.findall(text)
        if token.strip(".,:;()[]{}")
    }


def _string_items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _fanout_penalty(count: int) -> float:
    if count <= 12:
        return 1.0
    if count <= 64:
        return 0.75
    if count <= 256:
        return 0.45
    return 0.25


def _useful_tag(tag: str) -> bool:
    normalized = _normalize_phrase(tag)
    if normalized in _GENERIC_ALIASES:
        return False
    if len(normalized) < 3:
        return False
    return bool(re.search(r"[a-z]", normalized))


def _useful_alias(alias: str) -> bool:
    if alias in _GENERIC_ALIASES:
        return False
    if len(alias) < 4:
        return False
    return bool(re.search(r"[a-z0-9]", alias))


def _useful_compact_alias(alias: str) -> bool:
    if len(alias) < 6:
        return False
    return alias not in _GENERIC_COMPACT_ALIASES


def _useful_identifier(identifier: str) -> bool:
    normalized = _normalize_phrase(identifier)
    if normalized in _GENERIC_ALIASES:
        return False
    if len(_compact_phrase(identifier)) < 2:
        return False
    if identifier.isdigit():
        return False
    return True


_IDENTIFIER_RE = re.compile(
    r"\b(?:"
    r"GPIO\d{1,3}|"
    r"P\d\.\d{1,2}|P\d{3}|P[A-Z]?\d{1,3}|"
    r"[A-Z]{1,3}\d{1,4}[A-Z0-9_-]*|"
    r"[A-Z][A-Z0-9]+_[A-Z0-9_]*\d[A-Z0-9_]*|"
    r"[a-z]+_[a-z0-9_/]*\d[a-z0-9_/]*"
    r")\b"
)


_KNOWN_PHRASES: tuple[tuple[str, str, float], ...] = (
    ("5 GHz", "protocol", 1.5),
    ("802.15.4", "protocol", 1.5),
    ("ADC", "bus", 1.1),
    ("Arduino", "platform", 1.05),
    ("battery", "feature", 1.15),
    ("BLE", "protocol", 1.25),
    ("Bluetooth", "protocol", 1.25),
    ("bootloader", "workflow", 1.2),
    ("bus servo", "product", 2.2),
    ("camera", "feature", 1.2),
    ("CAN bus", "bus", 1.25),
    ("DAC", "bus", 1.2),
    ("digital microphone", "feature", 1.7),
    ("Edge Impulse", "platform", 1.35),
    ("ESPHome", "platform", 1.45),
    ("Grove", "product", 0.9),
    ("I2C", "bus", 1.2),
    ("IIC", "bus", 1.0),
    ("Jetson", "platform", 1.35),
    ("LeRobot", "platform", 1.45),
    ("LoRa", "protocol", 1.3),
    ("LoRaWAN", "protocol", 1.45),
    ("Matter", "protocol", 1.45),
    ("Meshtastic", "platform", 1.45),
    ("microphone", "feature", 1.2),
    ("MicroPython", "platform", 1.35),
    ("microSD", "feature", 1.2),
    ("MQTT", "protocol", 1.35),
    ("OpenClaw", "platform", 1.55),
    ("PDM", "bus", 1.25),
    ("Raspberry Pi", "platform", 1.35),
    ("Reachy", "product", 1.6),
    ("robotics", "domain", 1.0),
    ("RS485", "bus", 1.45),
    ("SenseCAP", "product", 1.2),
    ("SenseCraft", "platform", 1.2),
    ("SPI", "bus", 1.15),
    ("The Things Network", "platform", 1.45),
    ("Thread", "protocol", 1.45),
    ("TinyML", "platform", 1.2),
    ("UART", "bus", 1.2),
    ("Wio Terminal", "product", 1.8),
    ("WiFi", "protocol", 1.15),
    ("Wi-Fi", "protocol", 1.15),
    ("Zigbee", "protocol", 1.45),
)
_KNOWN_PHRASE_ENTRIES: tuple[tuple[str, str, float, str], ...] = tuple(
    (phrase, kind, weight, _search_lower(phrase).strip())
    for phrase, kind, weight in _KNOWN_PHRASES
)


_GENERIC_ALIASES = {
    "about",
    "adapter",
    "application",
    "applications",
    "base",
    "board",
    "boards",
    "cloud",
    "device",
    "devices",
    "for",
    "getting started",
    "guide",
    "hardware",
    "introduction",
    "kit",
    "kits",
    "overview",
    "product",
    "seeed",
    "seeed studio",
    "sensor",
    "sensors",
    "series",
    "software",
    "studio",
    "tutorial",
    "wiki",
    "xiao",
}
_GENERIC_COMPACT_ALIASES = {_compact_phrase(alias) for alias in _GENERIC_ALIASES}
