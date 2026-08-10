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
    assert chunks[2].text.startswith("### Subsection")
    assert "Nested text." in chunks[2].text


def test_chunk_index_and_source_file_are_set():
    chunks = chunk_markdown(SAMPLE, source_file="sample.md")
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert all(c.source_file == "sample.md" for c in chunks)


def test_no_headings_returns_no_chunks():
    assert chunk_markdown("Just plain text, no headings.", source_file="x.md") == []
