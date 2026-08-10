import re
from dataclasses import dataclass

_HEADING_PATTERN = re.compile(r"^(#{2,3}\s.*)$", re.MULTILINE)


@dataclass
class Chunk:
    source_file: str
    chunk_index: int
    text: str


def chunk_markdown(text: str, source_file: str) -> list[Chunk]:
    """Split markdown into chunks on H2/H3 heading boundaries. Text before the
    first H2/H3 (e.g. an H1 title) is dropped -- it's not a retrievable unit."""
    matches = list(_HEADING_PATTERN.finditer(text))
    chunks = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(source_file=source_file, chunk_index=i, text=chunk_text))
    return chunks
