# The `tamu-ai` command. Run `tamu-ai --help` to see all commands, or `tamu-ai <command> --help`.
#
# Typical order:
#   tamu-ai setup                     add our fields to the ai_chunks core (once)
#   tamu-ai export data               items -> data/chunks.jsonl + data/questions.json
#   tamu-ai embed data                chunks -> data/vectors-<model>.npy   (Grace, or laptop for small tests)
#   tamu-ai load data                 chunks + vectors -> ai_chunks core
#   tamu-ai search "old maps of Cuba"
#   tamu-ai eval                      score search on the test questions
#   tamu-ai compare data --models ... compare models without Solr (Grace)

import argparse
import logging
import time
from pathlib import Path

from . import config

QUESTIONS = Path(__file__).resolve().parent.parent / "eval" / "questions.yaml"


def cmd_setup(args):
    from .pipeline import setup_schema
    added = setup_schema()
    print("added:", ", ".join(added) if added else "nothing, the schema was already complete")


def cmd_card(args):
    # Show the card text for one item: a quick way to check what the models will "see".
    from .cards import item_chunks
    for chunk in item_chunks(args.nid):
        print(f"--- {chunk['id']}  years {chunk['year_min']}-{chunk['year_max']}\n{chunk['text']}\n")


def cmd_export(args):
    from .pipeline import export
    count = export(args.folder, args.questions, args.limit)
    print(f"wrote {count} chunks and the test questions to {args.folder}")


def cmd_embed(args):
    from .pipeline import embed
    path, shape = embed(args.folder, args.model)
    print(f"wrote {shape[0]} vectors of {shape[1]} numbers to {path}")


def cmd_load(args):
    from .pipeline import load
    count, with_vectors = load(args.folder, args.model)
    print(f"loaded {count} chunks into {config.CHUNK_CORE}" + ("" if with_vectors else " (text only, no vectors)"))


def cmd_search(args):
    from .search import search
    for rank, doc in enumerate(search(args.query, args.mode, limit=args.limit), start=1):
        print(f"{rank:>2}. nid {doc['nid']:<6} {doc['title'][:70]}  [{', '.join(doc.get('object_type', []))}]")


def cmd_eval(args):
    # Run every test question through live Solr search in each mode and print the scores.
    from .evaluate import load_questions, print_table, score
    from .search import MODES, search
    questions = load_questions(args.questions)
    results = {}
    for mode in args.modes or MODES:
        started = time.time()
        rankings = {q["id"]: [doc["item_uuid"] for doc in search(q["question"], mode, limit=30)]
                    for q in questions}
        results[mode] = score(questions, rankings)
        print(f"{mode}: {(time.time() - started) / len(questions):.2f} s per question")
    print(f"\n{len(questions)} questions")
    print_table(results)


def cmd_compare(args):
    from .compare import compare
    from .evaluate import print_table
    results = compare(args.folder, args.models, args.rerankers, args.keyword_weights, args.depth)
    print()
    print_table(results)


def main():
    parser = argparse.ArgumentParser(prog="tamu-ai")
    commands = parser.add_subparsers(required=True)

    p = commands.add_parser("setup", help="add the vector type and fields to the ai_chunks core")
    p.set_defaults(run=cmd_setup)

    p = commands.add_parser("card", help="print the card text built for one item")
    p.add_argument("nid", type=int, help="Drupal node id")
    p.set_defaults(run=cmd_card)

    p = commands.add_parser("export", help="write chunks.jsonl and questions.json into a folder")
    p.add_argument("folder", type=Path)
    p.add_argument("--limit", type=int, help="only this many items (for quick tests)")
    p.add_argument("--questions", type=Path, default=QUESTIONS)
    p.set_defaults(run=cmd_export)

    p = commands.add_parser("embed", help="embed the chunks in a folder")
    p.add_argument("folder", type=Path)
    p.add_argument("--model", default=config.EMBED_MODEL)
    p.set_defaults(run=cmd_embed)

    p = commands.add_parser("load", help="replace the ai_chunks core with the chunks (and vectors) in a folder")
    p.add_argument("folder", type=Path)
    p.add_argument("--model", default=config.EMBED_MODEL, help="which vectors file to load")
    p.set_defaults(run=cmd_load)

    p = commands.add_parser("search", help="run one search and print the results")
    p.add_argument("query")
    p.add_argument("--mode", default="hybrid+rerank")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(run=cmd_search)

    p = commands.add_parser("eval", help="score live search on the test questions")
    p.add_argument("--questions", type=Path, default=QUESTIONS)
    p.add_argument("--modes", nargs="*", help="default: all four modes")
    p.set_defaults(run=cmd_eval)

    p = commands.add_parser("compare", help="compare models on the test questions, without Solr")
    p.add_argument("folder", type=Path)
    p.add_argument("--models", nargs="+", required=True, help="embedding models")
    p.add_argument("--rerankers", nargs="*", default=[], help="reranker models (optional)")
    p.add_argument("--keyword-weights", nargs="+", type=float, default=[1.0, 0.5])
    p.add_argument("--depth", type=int, default=50, help="how many results to merge and rerank")
    p.set_defaults(run=cmd_compare)

    args = parser.parse_args()
    # Only show warnings (e.g. "skipping nid ..."); libraries are very chatty at the INFO level.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    args.run(args)


if __name__ == "__main__":
    main()
