# Search the ai_chunks core in four ways:
#
#   keyword        - classic word matching (what Solr has always done)
#   vector         - meaning matching: compare the question's vector with every chunk's vector
#   hybrid         - run both and merge the two result lists
#   hybrid+rerank  - hybrid, then let the reranker model re-order the top results

from . import config, models, solr

MODES = ["keyword", "vector", "hybrid", "hybrid+rerank"]
FIELDS = "id,item_uuid,nid,title,text,object_type,collection"


def quote(value):
    # Put a value in double quotes for a Solr query, escaping any quotes or backslashes inside it.
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def filter_queries(object_types=(), collections=(), year_from=None, year_to=None):
    # Build Solr filter queries ("fq") that narrow results without changing their order.
    fq = []
    if object_types:
        fq.append("object_type:(" + " OR ".join(quote(t) for t in object_types) + ")")
    if collections:
        fq.append("collection:(" + " OR ".join(quote(c) for c in collections) + ")")
    if year_from or year_to:
        # Keep items whose date range overlaps the requested years. Example: an item dated 1935-1945
        # matches "1940 to 1950" because it ends after 1940 and starts before 1950.
        fq.append(f"year_max:[{year_from or '*'} TO *] AND year_min:[* TO {year_to or '*'}]")
    return fq


def keyword_search(query, fq=(), rows=50):
    # edismax is Solr's query parser for user-typed text. "title^2" makes a match in the title
    # count twice as much as a match in the card text.
    params = {"q": query, "defType": "edismax", "qf": "title^2 text", "fq": list(fq), "fl": FIELDS, "rows": rows}
    return solr.select(config.CHUNK_CORE, params)["response"]["docs"]


def vector_search(query, fq=(), rows=50):
    # Turn the question into a vector, then ask Solr for the `rows` chunks with the closest vectors.
    vector = models.embed([query], config.EMBED_MODEL, query=True)[0]
    knn = "{!knn f=vector topK=%d}[%s]" % (rows, ",".join(f"{x:.6f}" for x in vector))
    return solr.select(config.CHUNK_CORE, {"q": knn, "fq": list(fq), "fl": FIELDS, "rows": rows})["response"]["docs"]


def rrf(rankings, weights, k=60):
    # Reciprocal rank fusion: merge several ranked lists of ids into one.
    # In each list, the id at rank r earns weight / (k + r) points; points are added up across lists.
    # So an id near the top of either list does well, and one high in both lists does best.
    # k=60 is the usual value; it keeps rank 1 from counting hugely more than rank 2.
    points = {}
    for ids, weight in zip(rankings, weights):
        for rank, id_ in enumerate(ids, start=1):
            points[id_] = points.get(id_, 0) + weight / (k + rank)
    return sorted(points, key=points.get, reverse=True)


def max_per_item(docs, limit=2):
    # An item with a long description has several chunks. Keep at most `limit` per item,
    # so one item can't fill the whole results page.
    count = {}
    kept = []
    for doc in docs:
        uuid = doc["item_uuid"]
        if count.get(uuid, 0) < limit:
            kept.append(doc)
            count[uuid] = count.get(uuid, 0) + 1
    return kept


def search(query, mode="hybrid+rerank", fq=(), limit=10, candidates=50):
    # Run one search and return the best `limit` chunks (Solr documents, as dicts).
    # `candidates` is how many results each method fetches before merging and reranking.
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    if mode == "keyword":
        docs = keyword_search(query, fq, candidates)
    elif mode == "vector":
        docs = vector_search(query, fq, candidates)
    else:
        keyword_docs = keyword_search(query, fq, candidates)
        vector_docs = vector_search(query, fq, candidates)
        by_id = {doc["id"]: doc for doc in keyword_docs + vector_docs}
        merged_ids = rrf([[d["id"] for d in keyword_docs], [d["id"] for d in vector_docs]],
                         weights=[config.KEYWORD_WEIGHT, 1.0])
        docs = [by_id[id_] for id_ in merged_ids[:candidates]]

        if mode == "hybrid+rerank":
            scores = models.rerank(query, [doc["text"] for doc in docs], config.RERANK_MODEL)
            ranked = sorted(zip(scores, range(len(docs))), reverse=True)
            docs = [docs[i] for _, i in ranked]

    return max_per_item(docs)[:limit]
