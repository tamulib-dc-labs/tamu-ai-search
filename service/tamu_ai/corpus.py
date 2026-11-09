"""Read public items from Archipelago's own Solr index.

Every item document in `default_solr_index` stores the full ADO JSON (`zs_raw_json`), so for milestone 1 we read
straight from Solr. Milestone 2 swaps this for the tamu_ai_search Drupal module's export endpoint.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

from .config import Settings
from .solr import Solr

log = logging.getLogger(__name__)

# The same items /search shows: published objects and collections, without child pages.
PUBLIC_FILTERS = [
    "ss_search_api_datasource:entity\\:node",
    "ss_type:(digital_object OR digital_object_collection)",
    "bs_status:true",
    "-sm_digital_object_type:(Page OR Leaf)",
]
FIELDS = "id,its_nid,ss_node_uuid,ss_title_string,sm_digital_object_type,sm_sbf_entity_reference_title,zs_raw_json"


@dataclass
class Item:
    uuid: str
    nid: int
    title: str
    object_types: list[str]
    collections: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


def item_from_doc(doc: dict[str, Any]) -> Item | None:
    try:
        metadata = json.loads(doc.get("zs_raw_json") or "{}")
    except json.JSONDecodeError:
        log.warning("Skipping nid %s: raw JSON does not parse", doc.get("its_nid"))
        return None
    if not doc.get("ss_node_uuid"):
        return None
    return Item(
        uuid=doc["ss_node_uuid"],
        nid=int(doc["its_nid"]),
        title=doc.get("ss_title_string") or metadata.get("label") or "",
        object_types=list(doc.get("sm_digital_object_type") or []),
        collections=list(doc.get("sm_sbf_entity_reference_title") or []),
        metadata=metadata,
    )


def iter_items(settings: Settings, limit: int | None = None) -> Iterator[Item]:
    solr = Solr(settings.solr_url, settings.source_core)
    count = 0
    for doc in solr.iter_docs("*:*", PUBLIC_FILTERS, FIELDS):
        item = item_from_doc(doc)
        if item is None:
            continue
        yield item
        count += 1
        if limit is not None and count >= limit:
            return
