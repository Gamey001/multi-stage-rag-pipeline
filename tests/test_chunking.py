import pytest

from rag_pipeline.pipelines.chunking import chunk_document, split_sentences
from rag_pipeline.pipelines.loader import load_documents
from rag_pipeline.schemas import Document


def test_split_sentences_keeps_decimal_numbers_intact():
    text = "Python 3.12 is supported. FastAPI runs on it! Does it scale? Yes."
    assert split_sentences(text) == [
        "Python 3.12 is supported.",
        "FastAPI runs on it!",
        "Does it scale?",
        "Yes.",
    ]


def test_headings_become_section_metadata_not_chunks():
    doc = Document("guide.md", "# Install\n\nRun pip install.\n\n## Usage\n\nCall the API.")
    chunks = chunk_document(doc, max_chars=200)

    assert [(c.section, c.text) for c in chunks] == [
        ("Install", "Run pip install."),
        ("Usage", "Call the API."),
    ]
    assert chunks[1].searchable_text == "Usage\nCall the API."


def test_short_paragraphs_in_same_section_are_packed():
    doc = Document("a.md", "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
    chunks = chunk_document(doc, max_chars=40)

    assert [c.text for c in chunks] == [
        "First paragraph.\n\nSecond paragraph.",
        "Third paragraph.",
    ]


def test_long_paragraph_splits_on_sentence_boundaries_with_overlap():
    sentences = [f"Sentence number {i} has some words." for i in range(6)]
    doc = Document("long.md", " ".join(sentences))
    chunks = chunk_document(doc, max_chars=80, overlap_sentences=1)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.endswith(".")
        assert all(part in sentences for part in split_sentences(chunk.text))
    for previous, current in zip(chunks, chunks[1:]):
        assert split_sentences(previous.text)[-1] == split_sentences(current.text)[0]


def test_oversized_sentence_is_kept_whole():
    sentence = "word " * 50 + "end."
    chunks = chunk_document(Document("x.md", sentence), max_chars=20)
    assert len(chunks) == 1
    assert chunks[0].text == " ".join(sentence.split())


def test_chunk_ids_are_stable_and_unique():
    doc = Document("a.md", "One.\n\n# H\n\nTwo.\n\nThree.")
    ids = [c.id for c in chunk_document(doc, max_chars=5)]
    assert ids == ["a.md::0", "a.md::1", "a.md::2"]


def test_loader_reads_supported_files_and_skips_others(tmp_path):
    (tmp_path / "keep.md").write_text("hello", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "notes.txt").write_text("world", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "empty.md").write_text("   ", encoding="utf-8")

    docs = load_documents(tmp_path)

    assert [d.source for d in docs] == ["keep.md", "nested/notes.txt"]


def test_loader_raises_on_missing_or_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_documents(tmp_path)
