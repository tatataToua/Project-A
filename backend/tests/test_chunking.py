from app.chunking import chunk_markdown

SAMPLE = """# Title

Intro text before any heading (should be dropped).

## Section One
Some text here.

## Section Two
More text.

### Subsection
Nested text.
"""


def test_splits_on_h2_and_h3_headings():
    chunks = chunk_markdown(SAMPLE, source_file="sample.md")
    assert len(chunks) == 3
    assert chunks[0].text.startswith("## Section One")
    assert "Some text here." in chunks[0].text
    assert chunks[1].text.startswith("## Section Two")
    assert "More text." in chunks[1].text
    assert "### Subsection" not in chunks[1].text
    # An H3 chunk carries its enclosing H2 heading, then its own heading.
    assert chunks[2].text.startswith("## Section Two\n### Subsection")
    assert "Nested text." in chunks[2].text


def test_chunk_index_and_source_file_are_set():
    chunks = chunk_markdown(SAMPLE, source_file="sample.md")
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.source_file == "sample.md" for c in chunks)


def test_no_headings_returns_no_chunks():
    assert chunk_markdown("Just plain text, no headings.", source_file="x.md") == []


ORPHAN_H2 = """# Menu

## Starters

### Chowder — $13
Clam and smoked bacon chowder.

### Shishito Peppers — $11
Blistered peppers, lemon, pecorino.
"""


def test_h3_chunk_includes_its_enclosing_h2_heading():
    chunks = chunk_markdown(ORPHAN_H2, source_file="menu.md")
    assert [c.text.splitlines()[0] for c in chunks] == ["## Starters", "## Starters"]
    assert chunks[0].text.startswith("## Starters\n### Chowder — $13")
    assert "Clam and smoked bacon chowder." in chunks[0].text
    assert "## Starters" in chunks[1].text
    assert "### Shishito Peppers — $11" in chunks[1].text


def test_heading_with_no_body_produces_no_chunk():
    chunks = chunk_markdown(ORPHAN_H2, source_file="menu.md")
    # "## Starters" has only H3s beneath it -- it is never a chunk of its own,
    # and chunk_index stays contiguous over the chunks that are emitted.
    assert len(chunks) == 2
    assert [c.chunk_index for c in chunks] == [0, 1]
    assert not any(c.text.strip() == "## Starters" for c in chunks)


def test_trailing_heading_with_no_body_is_dropped():
    chunks = chunk_markdown("## Real\nBody text.\n\n## Empty\n", source_file="x.md")
    assert len(chunks) == 1
    assert chunks[0].text.startswith("## Real")
