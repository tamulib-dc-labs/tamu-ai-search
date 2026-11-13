import json

from tamu_ai.cards import card_chunks, flatten, split_words, year_range


def test_flatten_handles_every_value_shape():
    assert flatten("<p>Hello &amp; welcome</p>") == ["Hello & welcome"]
    assert flatten(1898) == ["1898"]
    assert flatten([{"value": "Smith, John", "role": "photographer"}, "", None]) == ["Smith, John (photographer)"]
    assert flatten(True) == []


def test_year_range_reads_edtf_dates():
    assert year_range({"date_created": "1935-XX-XX~/1936-XX-XX~"}) == (1935, 1936)
    assert year_range({"date_issued": "193X"}) == (1930, 1939)
    assert year_range({"date_created": "undated"}) == (None, None)


def test_split_words_overlaps_pieces():
    pieces = split_words(" ".join(str(n) for n in range(10)), size=4, overlap=1)
    assert pieces == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]
    assert split_words("", size=4, overlap=1) == [""]


def make_doc(metadata):
    return {"ss_node_uuid": "u1", "its_nid": 7, "ss_title_string": "Map of Cuba",
            "sm_digital_object_type": ["Map"], "sm_sbf_entity_reference_title": ["Historic Maps"],
            "zs_raw_json": json.dumps(metadata)}


def test_short_item_gives_one_readable_chunk():
    chunks = card_chunks(make_doc({"language": ["spa"], "subject": ["Cuba"], "description": "A colored map.",
                                   "date_created": "1898"}))
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["id"] == "u1:card:0"
    assert chunk["text"].startswith("Title: Map of Cuba\n")
    assert "Language: Spanish" in chunk["text"]
    assert "Collection: Historic Maps" in chunk["text"]
    assert "Description: A colored map." in chunk["text"]
    assert (chunk["year_min"], chunk["year_max"]) == (1898, 1898)
    assert chunk["subject"] == ["Cuba"]


def test_long_description_is_split_and_every_piece_keeps_the_header():
    chunks = card_chunks(make_doc({"description": "word " * 700}))
    assert len(chunks) > 1
    assert all(chunk["text"].startswith("Title: Map of Cuba") for chunk in chunks)
