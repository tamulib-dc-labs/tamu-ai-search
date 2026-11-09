"""Score retrieval on test questions: did the right items come back, and how high?"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from .retrieve import MODES, Retriever


@dataclass
class Question:
    id: str
    question: str
    relevant: list[str]  # item uuids a good answer should find
    note: str = ""
    style: str = ""  # keyword | meaning


def load_questions(path: Path) -> list[Question]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [Question(id=str(q["id"]), question=q["question"], relevant=list(q["relevant"]),
                     note=q.get("note", ""), style=q.get("style", "")) for q in raw]


def ranked_items(item_uuids: list[str]) -> list[str]:
    """Several chunks of one item count once, at the rank of its best chunk."""
    seen: set[str] = set()
    return [u for u in item_uuids if not (u in seen or seen.add(u))]


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at(ranked: list[str], relevant: set[str], k: int) -> float:
    """1.0 when all relevant items fill the top ranks; less the lower they sit or if any are missing."""
    dcg = sum(1.0 / math.log2(rank + 1) for rank, u in enumerate(ranked[:k], start=1) if u in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def mrr(ranked: list[str], relevant: set[str]) -> float:
    for rank, u in enumerate(ranked, start=1):
        if u in relevant:
            return 1.0 / rank
    return 0.0


def score(questions: list[Question], rank: Callable[[Question], list[str]], k: int = 10) -> tuple[dict, dict]:
    """Run `rank` (question -> item uuids, best first) on every question; return (averages, per-question rows)."""
    totals = {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0, "seconds": 0.0}
    rows = {}
    for q in questions:
        started = time.monotonic()
        ranked = ranked_items(rank(q))[:k]
        elapsed = time.monotonic() - started
        relevant = set(q.relevant)
        row = {"recall": recall_at(ranked, relevant, k), "ndcg": ndcg_at(ranked, relevant, k),
               "mrr": mrr(ranked, relevant), "seconds": elapsed}
        for key in totals:
            totals[key] += row[key]
        rows[q.id] = {**row, "top": ranked[:3]}
    return {key: value / len(questions) for key, value in totals.items()}, rows


def run(retriever: Retriever, questions: list[Question], modes: list[str] | None = None, k: int = 10) -> dict:
    """Score the live Solr retriever in each mode."""
    report: dict = {"k": k, "modes": {}, "questions": {}}
    for mode in modes or list(MODES):
        averages, rows = score(
            questions, lambda q: [h.item_uuid for h in retriever.search(q.question, mode=mode, limit=k * 3)], k)
        report["modes"][mode] = averages
        for qid, row in rows.items():
            report["questions"].setdefault(qid, {})[mode] = row
    return report


def print_table(report: dict, label: str = "mode") -> None:
    width = max([len(label)] + [len(name) for name in report["modes"]]) + 2
    print(f"{label:<{width}}{'recall@10':>10}{'nDCG@10':>10}{'MRR':>8}{'sec/q':>8}")
    for name, m in report["modes"].items():
        print(f"{name:<{width}}{m['recall']:>10.3f}{m['ndcg']:>10.3f}{m['mrr']:>8.3f}{m['seconds']:>8.2f}")


def save(report: dict, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"retrieval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
