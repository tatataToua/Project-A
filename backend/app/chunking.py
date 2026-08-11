import re
from dataclasses import dataclass

_HEADING_PATTERN = re.compile(r"^(#{2,3}\s.*)$", re.MULTILINE)


@dataclass
class Chunk:
    source_file: str
    chunk_index: int
    text: str


def _heading_level(heading_line: str) -> int:
    return len(heading_line) - len(heading_line.lstrip("#"))


def chunk_markdown(text: str, source_file: str) -> list[Chunk]:
    """Split markdown into chunks on H2/H3 heading boundaries. Text before the
    first H2/H3 (e.g. an H1 title) is dropped -- it's not a retrievable unit.

    An H3 chunk is prefixed with its enclosing H2 heading so the section topic
    (e.g. "## Starters" above "### Two Owls Chowder") travels with the chunk into
    retrieval. Headings with no body text beneath them -- an H2 that only
    introduces the H3s below it -- are skipped rather than stored as content-free
    rows; `chunk_index` stays contiguous across the chunks actually emitted."""
    matches = list(_HEADING_PATTERN.finditer(text))
    chunks = []
    current_h2: str | None = None
    chunk_index = 0
    for i, match in enumerate(matches):
        heading_line = match.group(1).strip()
        is_h2 = _heading_level(heading_line) == 2
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()

        if is_h2:
            current_h2 = heading_line

        _, _, body = section.partition("\n")
        if not body.strip():
            continue

        if is_h2 or current_h2 is None:
            chunk_text = section
        else:
            chunk_text = f"{current_h2}\n{section}"
        chunks.append(Chunk(source_file=source_file, chunk_index=chunk_index, text=chunk_text))
        chunk_index += 1
    return chunks
