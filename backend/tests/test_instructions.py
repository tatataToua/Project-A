# backend/tests/test_instructions.py
from app import instructions


def test_returns_empty_string_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(instructions, "INSTRUCTIONS_FILE", tmp_path / "instructions.txt")

    assert instructions.get_custom_instructions() == ""


def test_returns_stripped_file_contents(tmp_path, monkeypatch):
    path = tmp_path / "instructions.txt"
    path.write_text("  Keep answers short.  \n", encoding="utf-8")
    monkeypatch.setattr(instructions, "INSTRUCTIONS_FILE", path)

    assert instructions.get_custom_instructions() == "Keep answers short."


def test_returns_empty_string_when_file_is_whitespace_only(tmp_path, monkeypatch):
    path = tmp_path / "instructions.txt"
    path.write_text("   \n\n  ", encoding="utf-8")
    monkeypatch.setattr(instructions, "INSTRUCTIONS_FILE", path)

    assert instructions.get_custom_instructions() == ""
