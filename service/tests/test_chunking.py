import pytest

from tamu_ai.chunking import card_chunks, word_windows
from tamu_ai.corpus import Item


def test_short_text_is_one_window():
    assert word_windows("one two three", max_words=5, overlap=1) == ["one two three"]
    assert word_windows("", max_words=5, overlap=1) == []


def test_windows_overlap_and_cover_every_word():
    words = [f"w{i}" for i in range(12)]
    windows = word_windows(" ".join(words), max_words=5, overlap=2)
    assert windows == ["w0 w1 w2 w3 w4", "w3 w4 w5 w6 w7", "w6 w7 w8 w9 w10", "w9 w10 w11"]


def test_overlap_must_be_smaller_than_window():
    with pytest.raises(ValueError):
        word_windows("a b c", max_words=3, overlap=3)


def _item(description: str) -> Item:
    return Item(uuid="u-1", nid=7, title="Bonfire 1936", object_types=["Image"], collections=["Archives"],
                metadata={"type": "Image", "date_created": "1936", "description": [description]})


def test_short_item_is_one_card_with_filters():
    [chunk] = card_chunks(_item("Cadets stacking logs."))
    assert chunk.id == "u-1:card:0"
    assert chunk.text.startswith("Title: Bonfire 1936\nType: Image")
    assert (chunk.year_min, chunk.year_max) == (1936, 1936)
    doc = chunk.to_solr([0.1, 0.2])
    assert doc["object_type"] == ["Image"] and doc["vector"] == [0.1, 0.2]


def test_long_item_splits_and_repeats_the_header():
    chunks = card_chunks(_item(" ".join(["word"] * 900)))
    assert len(chunks) > 1
    assert all(c.text.startswith("Title: Bonfire 1936") for c in chunks)
    assert [c.id for c in chunks] == [f"u-1:card:{n}" for n in range(len(chunks))]
