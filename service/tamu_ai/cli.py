"""Command line for the AI search service. Run `tamu-ai --help` for the list of commands."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import Settings

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
QUESTIONS = EVAL_DIR / "questions.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tamu-ai")
    sub = parser.add_subparsers(dest="command", required=True)

    # Next to Solr (laptop testbed now, the cluster later)
    sub.add_parser("setup", help="add the vector type and fields to the ai_chunks core")
    p = sub.add_parser("ingest", help="chunk, embed and index items in one go (small local runs)")
    p.add_argument("--limit", type=int, help="stop after this many items")
    p.add_argument("--fresh", action="store_true", help="empty the ai_chunks core first")
    p = sub.add_parser("export", help="write every public item's chunks to a JSONL file (no model needed)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p = sub.add_parser("export-eval", help="write the test questions plus Solr's keyword ranking for each")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--questions", type=Path, default=QUESTIONS)
    p = sub.add_parser("load", help="write chunks (and vectors from `embed`) into the ai_chunks core")
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--vectors", type=Path, help="vectors-<model>.npz; leave out to load text only")
    p = sub.add_parser("search", help="run one query against Solr and print the results")
    p.add_argument("query")
    p.add_argument("--mode", default="hybrid+rerank")
    p.add_argument("--limit", type=int, default=10)
    p = sub.add_parser("eval", help="score the live Solr search on the test questions")
    p.add_argument("--questions", type=Path, default=QUESTIONS)
    p.add_argument("--modes", nargs="*")
    p = sub.add_parser("card", help="print the chunks built for one item (by nid)")
    p.add_argument("nid", type=int)

    # On Grace (GPU), files only
    p = sub.add_parser("download-models", help="fetch models into the Hugging Face cache (Grace login node)")
    p.add_argument("models", nargs="+")
    p = sub.add_parser("embed", help="embed a chunks.jsonl file into a vectors .npz file")
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", help="defaults to TAMU_AI_EMBED_MODEL")
    p.add_argument("--batch-size", type=int, default=64)
    p = sub.add_parser("compare", help="score embedding models, merge weights and rerankers in memory")
    p.add_argument("--chunks", type=Path, required=True)
    p.add_argument("--evalpack", type=Path, required=True)
    p.add_argument("--models", nargs="+", required=True, help="embedding models to compare")
    p.add_argument("--rerankers", nargs="*", default=[], help="cross-encoder models; leave out to skip reranking")
    p.add_argument("--keyword-weights", nargs="+", type=float, default=[1.0, 0.5])
    p.add_argument("--rerank-depths", nargs="+", type=int, default=[20, 50])
    p.add_argument("--cache-dir", type=Path, default=Path("vectors-cache"))
    p.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = Settings.from_env()
    command = args.command

    if command == "setup":
        from .schema import apply
        added = apply(settings)
        print("added:", ", ".join(added) if added else "nothing, schema already complete")

    elif command == "card":
        from .chunking import card_chunks
        from .corpus import FIELDS, item_from_doc
        from .solr import Solr
        docs = Solr(settings.solr_url, settings.source_core).select(
            {"q": f"its_nid:{args.nid}", "fq": "ss_search_api_datasource:entity\\:node", "fl": FIELDS})
        for doc in docs["response"]["docs"]:
            item = item_from_doc(doc)
            for chunk in card_chunks(item) if item else []:
                print(f"--- {chunk.id}  years={chunk.year_min}-{chunk.year_max}\n{chunk.text}\n")

    elif command == "export":
        from .files import export_chunks
        print(f"wrote {export_chunks(settings, args.out, limit=args.limit)} chunks to {args.out}")

    elif command == "export-eval":
        from .evaluate import load_questions
        from .files import export_evalpack
        count = export_evalpack(settings, load_questions(args.questions), args.out)
        print(f"wrote {count} questions with keyword rankings to {args.out}")

    elif command == "load":
        from .files import load_into_solr
        print(f"loaded {load_into_solr(settings, args.chunks, args.vectors)} chunks into {settings.chunk_core}")

    elif command == "download-models":
        from huggingface_hub import snapshot_download
        for name in args.models:
            print(name, "->", snapshot_download(name))

    elif command == "embed":
        from .files import embed_file
        count, dim = embed_file(args.chunks, args.out, args.model or settings.embed_model, args.batch_size)
        print(f"embedded {count} chunks ({dim} numbers each) into {args.out}")

    elif command == "compare":
        from .compare import run_compare
        from .evaluate import print_table
        report = run_compare(args.chunks, args.evalpack, args.models, args.rerankers, args.keyword_weights,
                             args.rerank_depths, args.cache_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"{report['questions']} questions over {report['chunks']} chunks, k={report['k']}")
        print_table(report, label="variant")
        print("saved", args.out)

    else:  # ingest, search, eval need the models next to Solr
        from .embed import Models
        models = Models(settings)
        if command == "ingest":
            from .ingest import ingest
            print(ingest(settings, models, limit=args.limit, fresh=args.fresh))
            return
        from .retrieve import Retriever
        retriever = Retriever(settings, models)
        if command == "search":
            for rank, hit in enumerate(retriever.search(args.query, mode=args.mode, limit=args.limit), start=1):
                print(f"{rank:>2}. {hit.score:8.4f}  nid {hit.nid:<6} {hit.title[:70]}  "
                      f"[{', '.join(hit.object_types)}]")
        elif command == "eval":
            from .evaluate import load_questions, print_table, run, save
            questions = load_questions(args.questions)
            report = run(retriever, questions, modes=args.modes)
            print(f"{len(questions)} questions, k={report['k']}")
            print_table(report)
            print("saved", save(report, EVAL_DIR / "results"))


if __name__ == "__main__":
    main()
