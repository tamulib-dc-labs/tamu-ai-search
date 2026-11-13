# Small helpers for talking to Solr over HTTP.
#
# Solr is a search server. Each "core" is like a separate database with its own fields.
# We send it JSON or form data and get JSON back. That is all this file does.

import json

import httpx

from . import config

http = httpx.Client(timeout=60)


def _check(response):
    # Raise an error that includes Solr's own message (much more useful than just "400 Bad Request").
    if response.is_error:
        raise RuntimeError(f"Solr {response.status_code}: {response.text[:300]}")
    return response.json()


def get(core, path):
    return _check(http.get(f"{config.SOLR_URL}/{core}/{path}"))


def post(core, path, **kwargs):
    return _check(http.post(f"{config.SOLR_URL}/{core}/{path}", **kwargs))


def select(core, params):
    # Run a search. We POST instead of GET because a vector query is 768+ numbers,
    # which is too long to put in a URL.
    return post(core, "select", data=params)


def select_all(core, query, filters, fields, rows=500):
    # Yield every document that matches, page by page.
    # Solr's "cursorMark" is a bookmark: each page tells us where the next one starts.
    # When the bookmark stops changing, we have reached the end.
    cursor = "*"
    while True:
        data = select(core, {"q": query, "fq": filters, "fl": fields, "rows": rows,
                             "sort": "id asc", "cursorMark": cursor})
        yield from data["response"]["docs"]
        if data["nextCursorMark"] == cursor:
            break
        cursor = data["nextCursorMark"]


def add(core, docs):
    # Add (or overwrite, if the id already exists) a list of documents.
    post(core, "update", content=json.dumps(docs), headers={"Content-Type": "application/json"})


def delete_all(core):
    post(core, "update", json={"delete": {"query": "*:*"}})


def commit(core):
    # Changes only become searchable after a commit.
    post(core, "update", json={"commit": {}})
