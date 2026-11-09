"""Turn an item's metadata JSON into a readable "metadata card".

The card is what gets embedded and what the LLM later reads, so it uses the labels a person would understand
(from archipelago-metadata-mappings/mappings/main.yml) and leaves out technical keys (files, checksums, tasks).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any

# (label, keys) in the order a cataloger would read them. Keys missing from an item are skipped.
HEADER_FIELDS: list[tuple[str, list[str]]] = [
    ("Alternative title", ["title_alternative"]),
    ("Type", ["type"]),
    ("Genre", ["genre_form", "genre_form_linked_data"]),
    ("Creators and contributors", ["creator", "creator_linked_data", "contributor", "contributor_linked_data",
                                   "agent", "agent_linked_data"]),
    ("Date created", ["date_created"]),
    ("Date issued", ["date_issued"]),
    ("Date built", ["date_built"]),
    ("Date razed", ["date_razed"]),
    ("Subjects", ["subject", "subject_linked_data"]),
    ("Places", ["geographic_subject", "geographic_subject_linked_data"]),
    ("Time periods", ["temporal_subject", "temporal_subject_linked_data"]),
    ("Language", ["language"]),
    ("Extent", ["extent"]),
    ("Medium", ["medium"]),
    ("Original publisher", ["original_publisher"]),
    ("Place of origin", ["place_of_origin"]),
    ("Collecting area", ["collecting_area"]),
    ("Physical collection", ["physical_collection"]),
]
BODY_FIELDS: list[tuple[str, list[str]]] = [
    ("Description", ["description"]),
    ("Abstract", ["abstract"]),
    ("Note", ["note"]),
    ("Table of contents", ["table_of_contents"]),
    ("Provenance", ["provenance"]),
]
DATE_KEYS = ["date_created", "date_issued", "date_built", "date_razed"]
# ISO 639-2 codes used in our metadata, spelled out so "Spanish" matches "spa".
LANGUAGES = {"eng": "English", "spa": "Spanish", "fre": "French", "fra": "French", "ger": "German",
             "deu": "German", "ita": "Italian", "lat": "Latin", "por": "Portuguese", "chi": "Chinese",
             "zho": "Chinese", "jpn": "Japanese", "rus": "Russian", "ara": "Arabic", "cze": "Czech",
             "pol": "Polish", "dut": "Dutch", "nld": "Dutch", "gre": "Greek", "heb": "Hebrew", "zxx": "No linguistic content"}
SUBJECT_KEYS = ["subject", "subject_linked_data", "geographic_subject", "geographic_subject_linked_data",
                "temporal_subject", "temporal_subject_linked_data"]

_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"\s+")
# A year, or an EDTF year with unspecified digits such as 19XX or 193X.
_YEAR = re.compile(r"(?<![\dX])(\d{2}[\dX]{2})(?![\dX])")


def clean(text: str) -> str:
    return _SPACE.sub(" ", html.unescape(_TAG.sub(" ", text))).strip()


def values(raw: Any) -> list[str]:
    """Flatten a metadata value of any shape (string, list, {value, role, uri}) into display strings."""
    if raw is None:
        return []
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return [str(raw)]
    if isinstance(raw, str):
        text = clean(raw)
        return [text] if text else []
    if isinstance(raw, list):
        out: list[str] = []
        for entry in raw:
            out.extend(values(entry))
        return out
    if isinstance(raw, dict):
        label = raw.get("value") or raw.get("label") or raw.get("name") or ""
        text = clean(str(label)) if label else ""
        role = raw.get("role")
        if text and role:
            text = f"{text} ({clean(str(role))})"
        return [text] if text else []
    return []


def unique(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for value in seq:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def collect(metadata: dict[str, Any], keys: list[str]) -> list[str]:
    found: list[str] = []
    for key in keys:
        found.extend(values(metadata.get(key)))
    return unique(found)


def years(metadata: dict[str, Any]) -> tuple[int | None, int | None]:
    """Earliest and latest year in the date fields; EDTF "193X" counts as 1930-1939."""
    lows, highs = [], []
    for key in DATE_KEYS:
        for text in values(metadata.get(key)):
            for token in _YEAR.findall(text):
                lows.append(int(token.replace("X", "0")))
                highs.append(int(token.replace("X", "9")))
    if not lows:
        return None, None
    return min(lows), max(highs)


@dataclass
class Card:
    header: list[str]  # short lines: title, type, creators, dates, collection, subjects...
    body: list[str]  # long free text: description, abstract, notes


def build_card(title: str, metadata: dict[str, Any], collections: list[str]) -> Card:
    header = [f"Title: {clean(title)}"]
    for label, keys in HEADER_FIELDS:
        found = collect(metadata, keys)
        if label == "Language":
            found = unique([LANGUAGES.get(code.lower(), code) for code in found])
        if found:
            header.append(f"{label}: {'; '.join(found)}")
    if collections:
        header.append(f"Collection: {'; '.join(unique(collections))}")
    body = []
    for label, keys in BODY_FIELDS:
        for text in collect(metadata, keys):
            body.append(f"{label}: {text}")
    return Card(header=header, body=body)


def subjects(metadata: dict[str, Any]) -> list[str]:
    return collect(metadata, SUBJECT_KEYS)
