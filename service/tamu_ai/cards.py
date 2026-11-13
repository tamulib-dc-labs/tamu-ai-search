# Turn Archipelago items into "metadata cards": short, readable text about each item.
#
# The card is what we embed (turn into a vector) and, later, what the LLM reads to write an answer.
# So it uses labels a person understands ("Subjects: ...") instead of raw JSON keys.
#
# Example card:
#   Title: Map of the Island of Cuba
#   Type: Map
#   Date created: 1898
#   Places: Cuba
#   Collection: Historic Maps
#   Description: A colored map showing ...

import html
import json
import logging
import re

from . import config, solr

# Filters for Archipelago's index that give the same items /search shows:
# published objects and collections, but not the individual pages of a book.
PUBLIC_ITEMS = [
    "ss_search_api_datasource:entity\\:node",
    "ss_type:(digital_object OR digital_object_collection)",
    "bs_status:true",
    "-sm_digital_object_type:(Page OR Leaf)",
]
# The Solr fields we need. zs_raw_json holds the item's full metadata as a JSON string.
SOLR_FIELDS = "its_nid,ss_node_uuid,ss_title_string,sm_digital_object_type,sm_sbf_entity_reference_title,zs_raw_json"

# Short fields go at the top of every chunk of a card. label -> metadata keys to read.
# Labels come from archipelago-metadata-mappings (main.yml).
SHORT_FIELDS = {
    "Alternative title": ["title_alternative"],
    "Type": ["type"],
    "Genre": ["genre_form", "genre_form_linked_data"],
    "Creators and contributors": ["creator", "creator_linked_data", "contributor", "contributor_linked_data",
                                  "agent", "agent_linked_data"],
    "Date created": ["date_created"],
    "Date issued": ["date_issued"],
    "Date built": ["date_built"],
    "Date razed": ["date_razed"],
    "Subjects": ["subject", "subject_linked_data"],
    "Places": ["geographic_subject", "geographic_subject_linked_data"],
    "Time periods": ["temporal_subject", "temporal_subject_linked_data"],
    "Language": ["language"],
    "Extent": ["extent"],
    "Medium": ["medium"],
    "Original publisher": ["original_publisher"],
    "Place of origin": ["place_of_origin"],
    "Collecting area": ["collecting_area"],
    "Physical collection": ["physical_collection"],
}
# Long free text. If it is too long for one chunk it gets split across several.
LONG_FIELDS = {
    "Description": ["description"],
    "Abstract": ["abstract"],
    "Note": ["note"],
    "Table of contents": ["table_of_contents"],
    "Provenance": ["provenance"],
}
DATE_FIELDS = ["date_created", "date_issued", "date_built", "date_razed"]
SUBJECT_FIELDS = ["subject", "subject_linked_data", "geographic_subject", "geographic_subject_linked_data",
                  "temporal_subject", "temporal_subject_linked_data"]
# Our metadata stores language codes ("spa"); spell them out so a search for "Spanish" matches.
LANGUAGES = {"eng": "English", "spa": "Spanish", "fre": "French", "fra": "French", "ger": "German",
             "deu": "German", "ita": "Italian", "lat": "Latin", "por": "Portuguese", "chi": "Chinese",
             "zho": "Chinese", "jpn": "Japanese", "rus": "Russian", "ara": "Arabic", "cze": "Czech",
             "pol": "Polish", "dut": "Dutch", "nld": "Dutch", "gre": "Greek", "heb": "Hebrew",
             "zxx": "No linguistic content"}

# Embedding models read about 512 tokens at most. ~300 words stays safely under that.
MAX_WORDS = 300
# When long text is split, neighbouring pieces share 50 words, so a sentence cut at the edge
# of one piece still appears whole in the next.
OVERLAP_WORDS = 50

log = logging.getLogger(__name__)


def clean(text):
    # Remove HTML tags, decode entities like &amp;, and squash runs of whitespace into one space.
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(html.unescape(text).split())


def flatten(value):
    # Metadata values come in many shapes: "text", 1898, ["a", "b"],
    # or {"value": "Smith, John", "role": "photographer"}. Turn any of them into a list of strings.
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, list):
        return [text for entry in value for text in flatten(entry)]
    if isinstance(value, dict):
        text = clean(str(value.get("value") or value.get("label") or value.get("name") or ""))
        if text and value.get("role"):
            text += f" ({clean(str(value['role']))})"
        return [text] if text else []
    text = clean(str(value))
    return [text] if text else []


def dedupe(values):
    # Drop repeats (ignoring upper/lower case) but keep the original order.
    seen = set()
    result = []
    for value in values:
        if value.lower() not in seen:
            seen.add(value.lower())
            result.append(value)
    return result


def get_values(metadata, keys):
    # All the display strings stored under any of these keys.
    found = []
    for key in keys:
        found += flatten(metadata.get(key))
    return dedupe(found)


def year_range(metadata):
    # Earliest and latest year mentioned in the date fields, used for date filters.
    # Dates are in EDTF format, where X means "unknown digit": "193X" means 1930-1939.
    years = []
    for text in get_values(metadata, DATE_FIELDS):
        for year in re.findall(r"(?<![\dX])(\d{2}[\dX]{2})(?![\dX])", text):
            years.append(int(year.replace("X", "0")))
            years.append(int(year.replace("X", "9")))
    if not years:
        return None, None
    return min(years), max(years)


def split_words(text, size, overlap):
    # Split text into pieces of `size` words, each overlapping the previous by `overlap` words.
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    return [" ".join(words[start:start + size]) for start in range(0, len(words) - overlap, step)]


def card_chunks(doc):
    # Build the chunks for one item (one Solr document from Archipelago's index).
    # Most items give one chunk. Items with a long description give several, and each one
    # repeats the short fields at the top, so every chunk makes sense on its own.
    metadata = json.loads(doc.get("zs_raw_json") or "{}")
    uuid = doc["ss_node_uuid"]
    title = doc.get("ss_title_string") or metadata.get("label") or ""
    collections = dedupe(doc.get("sm_sbf_entity_reference_title") or [])

    # The header: title plus every short field the item has.
    header = [f"Title: {clean(title)}"]
    for label, keys in SHORT_FIELDS.items():
        values = get_values(metadata, keys)
        if label == "Language":
            values = dedupe([LANGUAGES.get(code.lower(), code) for code in values])
        if values:
            header.append(f"{label}: {'; '.join(values)}")
    if collections:
        header.append(f"Collection: {'; '.join(collections)}")
    header = "\n".join(header)

    # The body: the long text fields, split to fit in whatever room the header leaves.
    body = "\n".join(f"{label}: {text}"
                     for label, keys in LONG_FIELDS.items()
                     for text in get_values(metadata, keys))
    room = max(MAX_WORDS - len(header.split()), 100)
    pieces = split_words(body, room, min(OVERLAP_WORDS, room // 3))

    # Everything except "text" is stored next to the chunk so search can filter on it
    # (by type, collection, years) and link back to the item.
    year_min, year_max = year_range(metadata)
    chunks = []
    for n, piece in enumerate(pieces):
        chunks.append({
            "id": f"{uuid}:card:{n}",
            "item_uuid": uuid,
            "nid": int(doc["its_nid"]),
            "title": title,
            "text": f"{header}\n{piece}".strip(),
            "object_type": doc.get("sm_digital_object_type") or [],
            "collection": collections,
            "subject": get_values(metadata, SUBJECT_FIELDS),
            "year_min": year_min,
            "year_max": year_max,
            "rights": str(metadata.get("rights") or ""),
        })
    return chunks


def all_chunks(limit=None):
    # Read every public item from Archipelago's index and yield its chunks.
    # `limit` stops after that many items (handy for quick tests).
    docs = solr.select_all(config.SOURCE_CORE, "*:*", PUBLIC_ITEMS, SOLR_FIELDS)
    for count, doc in enumerate(docs):
        if limit and count >= limit:
            break
        try:
            yield from card_chunks(doc)
        except (KeyError, ValueError) as error:  # missing uuid/nid, or metadata JSON that won't parse
            log.warning("skipping nid %s: %s", doc.get("its_nid"), error)


def item_chunks(nid):
    # The chunks for one item, looked up by its Drupal node id. Used by `tamu-ai card <nid>`.
    data = solr.select(config.SOURCE_CORE, {"q": f"its_nid:{nid}", "fq": PUBLIC_ITEMS, "fl": SOLR_FIELDS})
    return [chunk for doc in data["response"]["docs"] for chunk in card_chunks(doc)]
