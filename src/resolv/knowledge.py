"""Local text loading and chunking without models, databases or network access."""

import json
import re
import tomllib
from collections.abc import Iterator, Mapping
from hashlib import sha256
from pathlib import Path

from resolv.schemas.knowledge import (
    Chunk,
    Document,
    KnowledgeChunk,
    KnowledgeFileConfig,
    KnowledgeIndexEntry,
    KnowledgeIndexProfile,
)

DEFAULT_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_SEPARATORS = (
    re.compile(r"\n[ \t]*\n"),
    re.compile(r"\n"),
    re.compile(r"[。！？]|[.!?](?=\s|$)"),
)


def _source_root(source_root: str | Path) -> Path:
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise ValueError("source_root must be a directory.")
    return root


def _document_source(path: str | Path, root: Path) -> tuple[Path, str]:
    """Resolve a supported source without reading it or requiring it to exist."""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = root / file_path
    file_path = file_path.resolve()
    try:
        source = file_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Document path must be inside source_root.") from exc
    if file_path.suffix.lower() not in {".md", ".markdown", ".txt"}:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    return file_path, source


def load_document(
    path: str | Path, *, source_root: str | Path = DEFAULT_SOURCE_ROOT
) -> Document:
    """Read UTF-8 text; derive identity from its resolved root-relative source."""
    file_path, source = _document_source(path, _source_root(source_root))
    text = file_path.read_text(encoding="utf-8-sig")
    return Document(
        document_id=sha256(source.encode("utf-8")).hexdigest(),
        source=source,
        text=text,
    )


def load_knowledge_file_configs(
    path: str | Path, *, source_root: str | Path = DEFAULT_SOURCE_ROOT
) -> dict[str, KnowledgeFileConfig]:
    """Read explicit TOML file mappings keyed by normalized Document.source."""
    root = _source_root(source_root)
    with Path(path).open("rb") as handle:
        data = tomllib.load(handle)
    if set(data) != {"files"} or not isinstance(data["files"], dict):
        raise ValueError("Knowledge config requires only a files table.")
    configs = {}
    physical_sources: set[tuple[int, int]] = set()
    for path_key, payload in data["files"].items():
        file_path, source = _document_source(path_key, root)
        if source in configs:
            raise ValueError(f"Duplicate configured source: {source}")
        try:
            stat = file_path.stat()
        except FileNotFoundError:
            pass  # A mapping may be prepared before its source file exists.
        else:
            identity = (stat.st_dev, stat.st_ino)
            if identity in physical_sources:
                raise ValueError(f"Duplicate configured source file: {source}")
            physical_sources.add(identity)
        configs[source] = KnowledgeFileConfig.model_validate(payload)
    return configs


def get_knowledge_file_config(
    configs: Mapping[str, KnowledgeFileConfig], source: str
) -> KnowledgeFileConfig:
    """Require a business type for the Loader's normalized source identity."""
    try:
        return configs[source]
    except KeyError as exc:
        raise ValueError(f"Missing file configuration for source: {source}") from exc


def _markdown_sections(text: str) -> Iterator[tuple[tuple[str, ...], str]]:
    """Keep titles structured until category/questions have been derived."""
    headings: list[tuple[int, str]] = []
    body: list[str] = []
    fence: tuple[str, int] | None = None
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if fence is not None:
            body.append(line)
            character, length = fence
            if re.fullmatch(
                rf" {{0,3}}{re.escape(character)}{{{length},}}[ \t]*", content
            ):
                fence = None
            continue
        opening = _FENCE.match(content)
        if opening and (opening.group(1)[0] == "~" or "`" not in opening.group(2)):
            marker = opening.group(1)
            fence = (marker[0], len(marker))
            body.append(line)
            continue
        heading = _ATX_HEADING.match(content)
        if heading:
            yield tuple(title for _, title in headings), "".join(body)
            body = []
            level = len(heading.group(1))
            title = re.sub(r"(?:^|[ \t]+)#+[ \t]*$", "", heading.group(2) or "").strip()
            while headings and headings[-1][0] >= level:
                headings.pop()
            if title:
                headings.append((level, title))
        else:
            body.append(line)
    yield tuple(title for _, title in headings), "".join(body)


def _split_boundary(
    text: str, start: int, end: int, overlap: int, level: int = 0
) -> int:
    """Fall back through separator priorities, then to the hard window end."""
    if level == len(_SEPARATORS):
        return end
    boundary = None
    for match in _SEPARATORS[level].finditer(text, start, end + 1):
        if start + overlap < match.end() <= end:
            boundary = match.end()
    if boundary is not None:
        return boundary
    return _split_boundary(text, start, end, overlap, level + 1)


def _body_windows(text: str, budget: int, overlap: int) -> Iterator[str]:
    start = 0
    while start < len(text):
        end = min(start + budget, len(text))
        if end < len(text):
            end = _split_boundary(text, start, end, overlap)
        window = text[start:end].strip()
        if window:
            yield window
        if end == len(text):
            break
        start = end - overlap


def build_embedding_text(category: str, questions: str, answer: str) -> str:
    """Join ancestor context, the current title and body without duplicating it."""
    header = " > ".join(part for part in (category, questions) if part)
    return f"{header}\n\n{answer}" if header else answer


def _validate_chunk_rules(max_chars: int, overlap: int) -> None:
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer.")
    if type(overlap) is not int or not 0 <= overlap < max_chars:
        raise ValueError(
            "overlap must be a nonnegative integer smaller than max_chars."
        )


def _chunk_windows(
    document: Document, max_chars: int, overlap: int
) -> Iterator[tuple[str, str, str]]:
    """Share chapter context and bounded body windows with both entry points."""
    _validate_chunk_rules(max_chars, overlap)
    for _, titles, body in iter_document_sections(document):
        body = body.strip()
        if not body:
            continue
        category = " > ".join(titles[:-1])
        questions = titles[-1] if titles else ""
        prefix = build_embedding_text(category, questions, "")
        budget = max_chars - len(prefix)
        if budget <= overlap:
            raise ValueError("Heading leaves insufficient body budget for overlap.")
        for window in _body_windows(body, budget, overlap):
            yield category, questions, window


def iter_document_sections(document: Document) -> Iterator[tuple[int, tuple[str, ...], str]]:
    """Enumerate actual title occurrences, including empty sections and prelude."""
    sections = (
        _markdown_sections(document.text)
        if Path(document.source).suffix.lower() in {".md", ".markdown"}
        else iter([((), document.text)])
    )
    for ordinal, (titles, body) in enumerate(sections):
        yield ordinal, titles, body


def _chunk_id(document_id: str, index: int) -> str:
    return sha256(f"{document_id}:{index}".encode()).hexdigest()


def chunk_document(
    document: Document, *, max_chars: int = 800, overlap: int = 100
) -> list[Chunk]:
    """Preserve the v0.1 text-only Chunk contract and deterministic identities."""
    return [
        Chunk(
            chunk_id=_chunk_id(document.document_id, index),
            document_id=document.document_id,
            text=build_embedding_text(category, questions, answer),
            source=document.source,
            chunk_index=index,
        )
        for index, (category, questions, answer) in enumerate(
            _chunk_windows(document, max_chars, overlap)
        )
    ]


def chunk_knowledge_document(
    document: Document, file_config: KnowledgeFileConfig,
    *, max_chars: int = 800, overlap: int = 100,
) -> list[KnowledgeChunk]:
    """Produce metadata-rich chunks before persistence or model inference."""
    return [
        KnowledgeChunk(
            chunk_id=_chunk_id(document.document_id, index),
            document_id=document.document_id,
            text=build_embedding_text(category, questions, answer),
            source=document.source,
            chunk_index=index,
            category=category,
            questions=questions,
            answer=answer,
            section_path=" > ".join(part for part in (category, questions) if part),
            content_type=file_config.content_type,
            is_key_clause=int(any(word in answer for word in file_config.key_clause_keywords)),
        )
        for index, (category, questions, answer) in enumerate(
            _chunk_windows(document, max_chars, overlap)
        )
    ]


def _knowledge_hash(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def knowledge_ingest_hash(
    document: Document, file_config: KnowledgeFileConfig,
    *, max_chars: int = 800, overlap: int = 100,
) -> str:
    """Fingerprint decoded source, its selected config and the chunking rule."""
    _validate_chunk_rules(max_chars, overlap)
    return _knowledge_hash({
        "algorithm": "section-chunker-v1", "document": document.model_dump(),
        "file_config": file_config.model_dump(mode="json"),
        "chunking": {"max_chars": max_chars, "overlap": overlap},
    })


def knowledge_index_hash(chunk: KnowledgeChunk, profile: KnowledgeIndexProfile) -> str:
    """Fingerprint the dense target and its explicit storage projection input."""
    projection = {
        name: getattr(chunk, name) for name in (
            "chunk_id", "document_id", "chunk_index", "source", "text",
            "section_path", "content_type", "is_key_clause",
        )
    }
    return _knowledge_hash({"profile": profile.model_dump(), "chunk": projection})


def knowledge_index_entry(chunk: KnowledgeChunk, profile: KnowledgeIndexProfile) -> KnowledgeIndexEntry:
    """Project scalar storage inputs explicitly, preserving the existing hash."""
    return KnowledgeIndexEntry(
        **{name: getattr(chunk, name) for name in KnowledgeIndexEntry.model_fields if name != "index_hash"},
        index_hash=knowledge_index_hash(chunk, profile),
    )


def knowledge_chunk_from_record(record: object) -> KnowledgeChunk:
    """Copy only knowledge payload fields while the caller's SQL Session is open."""
    return KnowledgeChunk.model_validate({name: getattr(record, name) for name in KnowledgeChunk.model_fields})
