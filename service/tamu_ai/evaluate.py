# Measure how good search is, using test questions with known right answers.
#
# eval/questions.yaml lists questions and, for each one, the items ("relevant") a good search should find.
# For each question we look at the top 10 items returned and compute three scores between 0 and 1:
#
#   recall@10  - what share of the relevant items appear in the top 10?
#   nDCG@10    - like recall, but finding them near the top counts more than near the bottom.
#   MRR        - 1 / rank of the first relevant item (1.0 if it's first, 0.5 if second, ...).
#
# Each question also has a "style": keyword (uses the same words as the record) or meaning
# (says it in different words). We report nDCG for each style, because vector search should
# shine on "meaning" questions.

import math

import yaml


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def recall(ranked, relevant, k=10):
    return len(set(ranked[:k]) & relevant) / len(relevant)


def ndcg(ranked, relevant, k=10):
    # A hit at rank r is worth 1 / log2(r + 1): rank 1 is worth 1.0, rank 2 is 0.63, rank 10 is 0.29.
    # Divide by the best possible score (all relevant items at the very top) to get 0..1.
    found = sum(1 / math.log2(rank + 1) for rank, uuid in enumerate(ranked[:k], start=1) if uuid in relevant)
    best = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return found / best


def mrr(ranked, relevant, k=10):
    for rank, uuid in enumerate(ranked[:k], start=1):
        if uuid in relevant:
            return 1 / rank
    return 0.0


def average(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def score(questions, rankings, k=10):
    # rankings: {question id: [item uuids, best first]}.
    # An item can appear several times (one per chunk); it counts once, at its best position.
    rows = []
    for q in questions:
        ranked = list(dict.fromkeys(rankings[q["id"]]))
        relevant = set(q["relevant"])
        rows.append({"style": q.get("style", ""), "recall": recall(ranked, relevant, k),
                     "ndcg": ndcg(ranked, relevant, k), "mrr": mrr(ranked, relevant, k)})

    result = {name: average(row[name] for row in rows) for name in ("recall", "ndcg", "mrr")}
    for style in ("keyword", "meaning"):
        result[f"ndcg_{style}"] = average(row["ndcg"] for row in rows if row["style"] == style)
    return result


def print_table(results):
    # results: {variant name: score(...) dict}. Prints one row per variant.
    columns = ["recall", "ndcg", "mrr", "ndcg_keyword", "ndcg_meaning"]
    width = max(len(name) for name in results) + 2
    print(" " * width + "".join(f"{c:>14}" for c in columns))
    for name, result in results.items():
        print(f"{name:<{width}}" + "".join(f"{result[c]:>14.3f}" for c in columns))
