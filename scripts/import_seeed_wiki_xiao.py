from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKOUT = ROOT / ".cache" / "seeed-wiki"
DEFAULT_OUTPUT = ROOT / "data" / "corpus" / "wiki_chunks.jsonl"
WIKI_REPO = "https://github.com/Seeed-Studio/wiki-documents.git"
WIKI_BRANCH = "docusaurus-version"
XIAO_SPARSE_PATH = "sites/en/docs/Sensor/SeeedStudio_XIAO"
RAW_BASE_URL = (
    "https://raw.githubusercontent.com/Seeed-Studio/wiki-documents/"
    f"{WIKI_BRANCH}/"
)
MAX_CHARS = 3200
MIN_CHARS = 180


@dataclass(frozen=True)
class BoardHint:
    board_id: str
    aliases: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import XIAO-only Seeed wiki markdown into compact retrieval chunks."
    )
    parser.add_argument(
        "--wiki-root",
        type=Path,
        help="Existing wiki checkout root or sites/en/docs directory. If omitted, a sparse checkout is created in .cache/seeed-wiki.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh", action="store_true", help="Fetch the latest sparse wiki checkout before importing.")
    parser.add_argument("--limit", type=int, default=0, help="Limit docs for a quick smoke run.")
    args = parser.parse_args()

    wiki_root = args.wiki_root or ensure_wiki_checkout(refresh=args.refresh)
    docs_root = resolve_docs_root(wiki_root)
    board_hints = load_board_hints()
    docs = discover_xiao_docs(docs_root)
    if args.limit:
        docs = docs[: args.limit]

    chunks: list[dict[str, object]] = []
    for path in docs:
        chunks.extend(chunks_for_doc(path, docs_root, board_hints))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"Imported {len(chunks)} chunks from {len(docs)} XIAO wiki docs -> {args.output}")


def ensure_wiki_checkout(refresh: bool = False) -> Path:
    if not DEFAULT_CHECKOUT.exists():
        DEFAULT_CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                WIKI_BRANCH,
                WIKI_REPO,
                str(DEFAULT_CHECKOUT),
            ]
        )
        run(["git", "-C", str(DEFAULT_CHECKOUT), "sparse-checkout", "set", XIAO_SPARSE_PATH])
    elif refresh:
        run(["git", "-C", str(DEFAULT_CHECKOUT), "pull", "--ff-only"])
    return DEFAULT_CHECKOUT


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def resolve_docs_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "sites" / "en" / "docs").exists():
        return path / "sites" / "en" / "docs"
    if path.name == "docs" and path.exists():
        return path
    raise FileNotFoundError(
        f"Could not find wiki docs root under {path}. Expected a repo root or sites/en/docs."
    )


def load_board_hints() -> list[BoardHint]:
    data = json.loads((ROOT / "data" / "corpus" / "xiao_boards.json").read_text(encoding="utf-8"))
    hints = []
    for doc in data.get("documents", []):
        aliases = [doc.get("title", ""), *doc.get("aliases", []), doc.get("id", "")]
        hints.append(BoardHint(board_id=doc["id"], aliases=[a for a in aliases if a]))
    return hints


def discover_xiao_docs(docs_root: Path) -> list[Path]:
    candidates = sorted(
        path
        for path in docs_root.rglob("*")
        if path.suffix.lower() in {".md", ".mdx"} and not path.name.startswith("_")
    )
    docs = []
    for path in candidates:
        rel = path.relative_to(docs_root).as_posix()
        if "SeeedStudio_XIAO" in rel or "XIAO" in path.stem.upper():
            docs.append(path)
    return docs


def chunks_for_doc(path: Path, docs_root: Path, board_hints: list[BoardHint]) -> list[dict[str, object]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    frontmatter, body = split_frontmatter(raw)
    rel = path.relative_to(docs_root).as_posix()
    title = (
        get_frontmatter_value(frontmatter, "title")
        or get_frontmatter_value(frontmatter, "sidebar_label")
        or path.stem.replace("_", " ").replace("-", " ")
    )
    wiki_url = build_wiki_url(frontmatter, path)
    image_urls = extract_image_urls(body, path, docs_root)
    cleaned = clean_markdown(body)
    sections = split_sections(cleaned)

    doc_chunks = []
    for index, (heading, text) in enumerate(sections):
        for part_index, part in enumerate(split_long_text(text)):
            if len(part) < MIN_CHARS and len(sections) > 1:
                continue
            chunk_title = title if not heading else f"{title} > {heading}"
            board_id, aliases = infer_board(chunk_title + "\n" + part + "\n" + rel, board_hints)
            chunk_id = stable_id(rel, index, part_index)
            doc_chunks.append(
                {
                    "id": f"wiki-{chunk_id}",
                    "title": chunk_title,
                    "source": wiki_url,
                    "text": part,
                    "board_id": board_id,
                    "kind": "wiki",
                    "metadata": {
                        "aliases": aliases,
                        "tags": ["wiki", "xiao", *keywords_for_text(chunk_title + " " + part)],
                        "citations": [{"title": title, "url": wiki_url}],
                        "source_file": rel,
                        "heading_path": heading,
                        "image_urls": image_urls[:6],
                    },
                }
            )
    return doc_chunks


def split_frontmatter(text: str) -> tuple[str, str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        return "", text
    return match.group(1), text[match.end() :]


def get_frontmatter_value(frontmatter: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*[\"']?(.+?)[\"']?\s*$", frontmatter, re.MULTILINE)
    return match.group(1).strip() if match else ""


def build_wiki_url(frontmatter: str, path: Path) -> str:
    for key in ("wiki_url", "wikiurl", "url"):
        value = get_frontmatter_value(frontmatter, key)
        if value.startswith("http"):
            return value
    slug = get_frontmatter_value(frontmatter, "slug")
    if slug:
        return f"https://wiki.seeedstudio.com/{slug.strip('/')}/"
    return f"https://wiki.seeedstudio.com/{path.stem}/"


def extract_image_urls(text: str, path: Path, docs_root: Path) -> list[str]:
    urls: list[str] = []
    patterns = [
        r"!\[[^\]]*\]\(([^\)]+)\)",
        r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"'][^>]*>",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            url = match.group(1).strip()
            if not is_image_url(url):
                continue
            urls.append(normalize_image_url(url, path, docs_root))
    return list(dict.fromkeys(urls))


def is_image_url(url: str) -> bool:
    lowered = url.split("?", 1)[0].split("#", 1)[0].lower()
    return lowered.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif"))


def normalize_image_url(url: str, path: Path, docs_root: Path) -> str:
    if url.startswith(("http://", "https://")):
        return url
    target = (path.parent / url).resolve()
    try:
        rel = target.relative_to(docs_root.parents[2]).as_posix()
        return urljoin(RAW_BASE_URL, rel)
    except ValueError:
        return url


def clean_markdown(text: str) -> str:
    text = re.sub(r"^import\s+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^export\s+.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = convert_html_tables(text)
    text = re.sub(r"^:::[ \t]*(\w+)[^\n]*$", r"**\1:**", text, flags=re.MULTILINE)
    text = re.sub(r"^:::$", "", text, flags=re.MULTILINE)
    text = re.sub(r"</?(?:Tabs|TabItem|details|summary|Details|Summary)[^>]*>", "", text)
    text = re.sub(r"<\w+[^>]*/>", "", text)
    text = re.sub(r"<sup>(.*?)</sup>", r"\1", text)
    text = re.sub(r"<sub>(.*?)</sub>", r"\1", text)
    text = re.sub(r"<(?!-)[^>]+>", "", text)
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", image_alt_text, text)
    text = re.sub(r"\{props\.\w+\}", "", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def convert_html_tables(text: str) -> str:
    def table_to_rows(match: re.Match[str]) -> str:
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", match.group(0), re.DOTALL)
        rendered_rows = []
        for row in rows:
            cells = re.findall(r"<(?:td|th)[^>]*>(.*?)</(?:td|th)>", row, re.DOTALL)
            clean_cells = [re.sub(r"<[^>]+>", "", cell).strip() for cell in cells]
            clean_cells = [cell for cell in clean_cells if cell]
            if clean_cells:
                rendered_rows.append(" | ".join(clean_cells))
        return "\n".join(rendered_rows)

    return re.sub(r"<table[^>]*>.*?</table>", table_to_rows, text, flags=re.DOTALL)


def image_alt_text(match: re.Match[str]) -> str:
    alt = match.group(1).strip()
    if not alt or alt.lower() in {"image", "img", "photo", "picture", "png", "jpg"}:
        return ""
    return alt


def split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current: list[str] = []
    headings: dict[int, str] = {}

    for line in text.splitlines():
        match = re.match(r"^(#{2,3})\s+(.+)$", line)
        if match:
            if "\n".join(current).strip():
                sections.append((current_heading, "\n".join(current).strip()))
            level = len(match.group(1))
            headings[level] = match.group(2).strip()
            for old_level in list(headings):
                if old_level > level:
                    del headings[old_level]
            current_heading = " > ".join(headings[key] for key in sorted(headings))
            current = []
        else:
            current.append(line)

    if "\n".join(current).strip():
        sections.append((current_heading, "\n".join(current).strip()))
    return sections or [("", text)]


def split_long_text(text: str) -> list[str]:
    blocks = re.split(r"\n\n+", text)
    parts: list[str] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        extra = len(block) + 2
        if current and current_len + extra > MAX_CHARS:
            parts.append("\n\n".join(current).strip())
            current = [block]
            current_len = len(block)
        else:
            current.append(block)
            current_len += extra
    if current:
        parts.append("\n\n".join(current).strip())
    return parts


def infer_board(text: str, board_hints: list[BoardHint]) -> tuple[str, list[str]]:
    normalized = normalize(text)
    best: tuple[int, BoardHint] | None = None
    for hint in board_hints:
        score = 0
        for alias in hint.aliases:
            alias_norm = normalize(alias)
            if alias_norm and alias_norm in normalized:
                score = max(score, len(alias_norm))
        if score and (best is None or score > best[0]):
            best = (score, hint)
    if best is None:
        return "", []
    return best[1].board_id, best[1].aliases


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def keywords_for_text(text: str) -> list[str]:
    keywords = []
    checks = {
        "arduino": "arduino",
        "micropython": "micropython",
        "circuitpython": "circuitpython",
        "zephyr": "zephyr",
        "i2c": "i2c",
        "spi": "spi",
        "uart": "uart",
        "wifi": "wifi",
        "bluetooth": "bluetooth",
        "zigbee": "zigbee",
        "thread": "thread",
        "matter": "matter",
        "camera": "camera",
        "microphone": "microphone",
        "imu": "imu",
        "sleep": "sleep",
        "boot": "boot",
        "flash": "flash",
    }
    lowered = text.lower()
    for needle, keyword in checks.items():
        if needle in lowered:
            keywords.append(keyword)
    return keywords


def stable_id(source_file: str, index: int, part_index: int) -> str:
    digest = hashlib.sha1(f"{source_file}:{index}:{part_index}".encode()).hexdigest()
    return digest[:14]


if __name__ == "__main__":
    main()
