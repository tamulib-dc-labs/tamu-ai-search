from tamu_ai.cards import build_card, clean, subjects, values, years


def test_values_flattens_every_shape():
    assert values("  Kyle   Field ") == ["Kyle Field"]
    assert values(["A", ["B", ""], None]) == ["A", "B"]
    assert values({"value": "Braun, Georg", "role": "cartographer"}) == ["Braun, Georg (cartographer)"]
    assert values({"value": "Texas", "uri": "http://id.loc.gov/x"}) == ["Texas"]
    assert values(1936) == ["1936"]
    assert values(True) == []


def test_clean_strips_html_and_entities():
    assert clean("<p>Aggie&nbsp;<b>Bonfire</b></p>") == "Aggie Bonfire"


def test_years_reads_plain_and_edtf_dates():
    assert years({"date_created": "1936-11-25"}) == (1936, 1936)
    assert years({"date_issued": "1935-XX-XX~/1936-XX-XX~"}) == (1935, 1936)
    assert years({"date_created": "193X"}) == (1930, 1939)
    assert years({"date_created": "", "date_issued": None}) == (None, None)


def test_card_labels_languages_and_skips_empty_fields():
    card = build_card("Letter to McInnis", {
        "type": "Manuscript", "creator": "", "language": ["eng", "spa"],
        "agent": [{"value": "Gathright, Thomas S.", "role": "author"}],
        "description": ["A letter about the curriculum."],
    }, ["Thomas S. Gathright Letters"])
    assert card.header == [
        "Title: Letter to McInnis",
        "Type: Manuscript",
        "Creators and contributors: Gathright, Thomas S. (author)",
        "Language: English; Spanish",
        "Collection: Thomas S. Gathright Letters",
    ]
    assert card.body == ["Description: A letter about the curriculum."]


def test_subjects_merge_plain_and_linked_without_duplicates():
    metadata = {"subject": ["Bonfire"], "subject_linked_data": [{"value": "bonfire", "uri": "u"}],
                "geographic_subject": "College Station (Tex.)"}
    assert subjects(metadata) == ["Bonfire", "College Station (Tex.)"]
