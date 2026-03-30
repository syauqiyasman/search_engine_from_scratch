#!/usr/bin/env python3
"""
irengine.py — Unified command-line interface for the IR Engine.

Sub-commands
────────────
  index      Build an inverted index (BSBI or SPIMI) with a chosen compression.
  search     Run ad-hoc queries against an existing index (TF-IDF, BM25, WAND, LSI).
  lsi        Build or rebuild the LSI / FAISS vector index.
  evaluate   Score the system against a qrel file (all metrics, all methods).
  compress   Show compression statistics for each encoding scheme.

Run  python irengine.py <sub-command> --help  for full option lists.
"""

import argparse
import os
import sys
import time
import math


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_encoding(name):
    """Return the postings-encoding class for a string name."""
    from compression import StandardPostings, VBEPostings, EliasGammaPostings
    mapping = {
        "standard": StandardPostings,
        "vbe":      VBEPostings,
        "elias":    EliasGammaPostings,
    }
    if name not in mapping:
        print(f"[ERROR] Unknown compression '{name}'. "
              f"Choices: {', '.join(mapping)}", file=sys.stderr)
        sys.exit(1)
    return mapping[name]


def _make_indexer(mode, data_dir, output_dir, encoding, index_name,
                  block_size=None):
    """Instantiate BSBIIndex or SPIMIIndex based on *mode*."""
    from bsbi import BSBIIndex, SPIMIIndex
    enc = _get_encoding(encoding)
    if mode == "bsbi":
        return BSBIIndex(data_dir=data_dir, output_dir=output_dir,
                         postings_encoding=enc, index_name=index_name)
    elif mode == "spimi":
        bs = block_size or 1_000_000
        return SPIMIIndex(data_dir=data_dir, output_dir=output_dir,
                          postings_encoding=enc, index_name=index_name,
                          block_size=bs)
    else:
        print(f"[ERROR] Unknown indexing mode '{mode}'. Choices: bsbi, spimi",
              file=sys.stderr)
        sys.exit(1)


def _ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def _load_indexer(output_dir, encoding, index_name):
    """Load an already-built index (just maps, no re-indexing)."""
    from bsbi import BSBIIndex
    enc = _get_encoding(encoding)
    idx = BSBIIndex(data_dir="", output_dir=output_dir,
                    postings_encoding=enc, index_name=index_name)
    idx.load()
    return idx


# ── sub-command: index ────────────────────────────────────────────────────────

def cmd_index(args):
    """
    Build an inverted index from a document collection.

    The collection directory must contain sub-directories (blocks), each
    holding plain-text (.txt) files.  One sub-directory = one indexing block.

    Example layout
    ──────────────
    collection/
      0/  doc1.txt  doc2.txt  …
      1/  doc3.txt  …
    """
    _ensure_output_dir(args.output_dir)

    print(f"[index] mode={args.mode.upper()}  compression={args.compression.upper()}"
          f"  data={args.data_dir}  output={args.output_dir}")
    if args.mode == "spimi":
        print(f"[index] block_size={args.block_size:,} tokens")

    indexer = _make_indexer(
        mode        = args.mode,
        data_dir    = args.data_dir,
        output_dir  = args.output_dir,
        encoding    = args.compression,
        index_name  = args.index_name,
        block_size  = args.block_size,
    )

    t0 = time.time()
    indexer.index()
    elapsed = time.time() - t0

    # Report index size
    idx_file = os.path.join(args.output_dir, args.index_name + ".index")
    size_mb  = os.path.getsize(idx_file) / 1024 / 1024 if os.path.exists(idx_file) else 0

    print(f"\n[index] Done in {elapsed:.1f}s")
    print(f"[index] Vocabulary  : {len(indexer.term_id_map):,} terms")
    print(f"[index] Documents   : {len(indexer.doc_id_map):,} docs")
    print(f"[index] Index file  : {idx_file} ({size_mb:.2f} MB)")


# ── sub-command: search ───────────────────────────────────────────────────────

def cmd_search(args):
    """
    Run one or more queries against a pre-built index.

    Queries may be supplied directly on the command line (--query) or read
    from a file (--query-file, one query per line, optionally prefixed with
    a query-ID: "Q1 some query text").

    Retrieval methods
    ─────────────────
    tfidf   Classical TF-IDF (TaaT)
    bm25    BM25 (TaaT)
    wand    BM25 with WAND top-K pruning (faster for large k)
    lsi     Latent Semantic Indexing via FAISS (requires prior `lsi build`)
    """
    if not args.query and not args.query_file:
        print("[ERROR] Provide --query TEXT or --query-file FILE", file=sys.stderr)
        sys.exit(1)

    # Collect queries
    queries = []
    if args.query:
        queries.append(("*", args.query))
    if args.query_file:
        with open(args.query_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[0].startswith("Q"):
                    queries.append((parts[0], parts[1]))
                else:
                    queries.append(("*", line))

    method = args.method.lower()

    if method == "lsi":
        from lsi import LSIIndexBuilder
        enc = _get_encoding(args.compression)
        retriever = LSIIndexBuilder.load_lsi(
            args.index_name, args.output_dir, enc
        )
        retrieve_fn = lambda q, k: retriever.retrieve(q, k=k)
    else:
        indexer = _load_indexer(args.output_dir, args.compression, args.index_name)
        if method == "tfidf":
            retrieve_fn = lambda q, k: indexer.retrieve_tfidf(q, k=k)
        elif method == "bm25":
            retrieve_fn = lambda q, k: indexer.retrieve_bm25(
                q, k=k, k1=args.bm25_k1, b=args.bm25_b)
        elif method == "wand":
            retrieve_fn = lambda q, k: indexer.retrieve_bm25_wand(
                q, k=k, k1=args.bm25_k1, b=args.bm25_b)
        else:
            print(f"[ERROR] Unknown method '{args.method}'. "
                  "Choices: tfidf, bm25, wand, lsi", file=sys.stderr)
            sys.exit(1)

    for qid, query in queries:
        tag = f"[{qid}]" if qid != "*" else ""
        print(f"\nQuery {tag}: {query}")
        print(f"Method: {method.upper()}"
              + (f"  k1={args.bm25_k1} b={args.bm25_b}"
                 if method in ("bm25", "wand") else ""))
        print(f"{'Rank':<5}  {'Score':>8}  Document")
        print("-" * 70)

        t0 = time.time()
        results = retrieve_fn(query, args.k)
        elapsed = time.time() - t0

        if not results:
            print("  (no results)")
        for rank, (score, doc) in enumerate(results, 1):
            print(f"  {rank:<4}  {score:>8.4f}  {doc}")

        print(f"  — {len(results)} result(s) in {elapsed*1000:.1f} ms")


# ── sub-command: lsi ──────────────────────────────────────────────────────────

def cmd_lsi(args):
    """
    Build (or rebuild) the LSI + FAISS vector index.

    Must be run AFTER `index` so that the inverted index already exists.
    The resulting model is saved inside --output-dir and is loaded
    automatically by `search --method lsi` and `evaluate --method lsi`.

    SVD algorithms
    ──────────────
    randomized   Halko-Martinsson-Tropp randomized SVD. Fast; recommended
                 for large vocabularies and n-components ≥ 50.
    arpack       Iterative Arnoldi method. More accurate for very small
                 n-components (< 30) but slower on large matrices.
    """
    from bsbi import BSBIIndex
    from lsi import LSIIndexBuilder

    enc = _get_encoding(args.compression)
    bsbi = BSBIIndex(data_dir=args.data_dir, output_dir=args.output_dir,
                     postings_encoding=enc, index_name=args.index_name)
    bsbi.load()

    print(f"[lsi] Building LSI index:")
    print(f"      n_components={args.n_components}  "
          f"algorithm={args.svd_algorithm}  "
          f"n_probe={args.n_probe}")

    builder = LSIIndexBuilder(
        bsbi,
        n_components  = args.n_components,
        svd_algorithm = args.svd_algorithm,
        n_probe       = args.n_probe,
    )

    t0  = time.time()
    lsi = builder.build()
    elapsed = time.time() - t0

    print(f"[lsi] Done in {elapsed:.1f}s — "
          f"{lsi._faiss_index.ntotal} docs indexed at dim={lsi.n_components}")


# ── sub-command: evaluate ─────────────────────────────────────────────────────

def cmd_evaluate(args):
    """
    Evaluate retrieval quality against a qrel file.

    Computes four metrics for every query in --query-file, then reports
    the mean over all queries:

      RBP   Rank-Biased Precision  (p = 0.8)
      DCG   Discounted Cumulative Gain
      NDCG  Normalized DCG
      AP    Average Precision  (mean = MAP)

    Use --method all to compare TF-IDF, BM25, and LSI side-by-side.
    LSI evaluation requires a pre-built LSI model (run `lsi` first, or
    pass --lsi-build to build it on-the-fly).
    """
    from evaluation import (load_qrels, eval_tfidf, eval_bm25,
                             eval_lsi, compare_all)

    qrels = load_qrels(
        qrel_file  = args.qrel_file,
        max_q_id   = args.max_q_id,
        max_doc_id = args.max_doc_id,
    )

    method = args.method.lower()

    # Monkey-patch data_dir / output_dir into evaluation helpers by
    # temporarily swapping the module-level defaults.
    import evaluation as _eval_mod
    from bsbi import BSBIIndex
    from compression import StandardPostings, VBEPostings, EliasGammaPostings

    enc = _get_encoding(args.compression)

    # Override the BSBIIndex construction inside eval helpers via a thin
    # wrapper so that custom paths are respected.
    def _tfidf_fn(q, k):
        idx = _load_indexer(args.output_dir, args.compression, args.index_name)
        return idx.retrieve_tfidf(q, k=k)

    def _bm25_fn(q, k):
        idx = _load_indexer(args.output_dir, args.compression, args.index_name)
        return idx.retrieve_bm25(q, k=k, k1=args.bm25_k1, b=args.bm25_b)

    def _lsi_fn(q, k):
        from lsi import LSIIndexBuilder
        lsi = LSIIndexBuilder.load_lsi(args.index_name, args.output_dir, enc)
        return lsi.retrieve(q, k=k)

    if method == "tfidf":
        _eval_mod.eval_retrieval(
            _tfidf_fn, qrels,
            query_file=args.query_file, k=args.k,
            method_name="TF-IDF",
        )

    elif method == "bm25":
        _eval_mod.eval_retrieval(
            _bm25_fn, qrels,
            query_file=args.query_file, k=args.k,
            method_name=f"BM25 k1={args.bm25_k1} b={args.bm25_b}",
        )

    elif method == "lsi":
        if args.lsi_build:
            eval_lsi(
                qrels, query_file=args.query_file, k=args.k,
                n_components=args.lsi_components,
                svd_algorithm=args.lsi_algorithm,
                build=True,
            )
        else:
            _eval_mod.eval_retrieval(
                _lsi_fn, qrels,
                query_file=args.query_file, k=args.k,
                method_name=f"LSI k={args.lsi_components}",
            )

    elif method == "all":
        # For 'all' we need a lsi retriever; build if requested
        if args.lsi_build:
            from bsbi import BSBIIndex
            bsbi = BSBIIndex(data_dir=args.data_dir,
                             output_dir=args.output_dir,
                             postings_encoding=enc,
                             index_name=args.index_name)
            bsbi.load()
            from lsi import LSIIndexBuilder
            builder = LSIIndexBuilder(bsbi,
                                      n_components=args.lsi_components,
                                      svd_algorithm=args.lsi_algorithm)
            builder.build()
            print(f"[evaluate] LSI model built.")

        SEP = "=" * 60
        print(SEP)
        scores_tfidf = _eval_mod.eval_retrieval(
            _tfidf_fn, qrels, query_file=args.query_file, k=args.k,
            method_name="TF-IDF")
        print(SEP)
        scores_bm25 = _eval_mod.eval_retrieval(
            _bm25_fn, qrels, query_file=args.query_file, k=args.k,
            method_name=f"BM25 k1={args.bm25_k1} b={args.bm25_b}")
        print(SEP)
        scores_lsi = _eval_mod.eval_retrieval(
            _lsi_fn, qrels, query_file=args.query_file, k=args.k,
            method_name=f"LSI k={args.lsi_components}")

        # Comparison table
        all_scores = {
            "TF-IDF":                              scores_tfidf,
            f"BM25 (k1={args.bm25_k1} b={args.bm25_b})": scores_bm25,
            f"LSI  (k={args.lsi_components})":    scores_lsi,
        }
        metrics = ["rbp", "dcg", "ndcg", "ap"]
        col_w   = 12
        print(f"\n{'':32}", end="")
        for m in metrics:
            print(f"{m.upper():>{col_w}}", end="")
        print()
        print("-" * (32 + col_w * len(metrics)))
        for method_name, sc in all_scores.items():
            print(f"{method_name:32}", end="")
            for m in metrics:
                print(f"{sc[m]:>{col_w}.4f}", end="")
            print()
        print("-" * (32 + col_w * len(metrics)))
        best_row = []
        for m in metrics:
            best = max(sc[m] for sc in all_scores.values())
            winner = next(name for name, sc in all_scores.items()
                          if abs(sc[m] - best) < 1e-9)
            best_row.append(f"*{winner[:9]}")
        print(f"{'  * best per metric':32}", end="")
        for cell in best_row:
            print(f"{cell:>{col_w}}", end="")
        print()

    else:
        print(f"[ERROR] Unknown method '{args.method}'. "
              "Choices: tfidf, bm25, lsi, all", file=sys.stderr)
        sys.exit(1)


# ── sub-command: compress ─────────────────────────────────────────────────────

def cmd_compress(args):
    """
    Display compression statistics for all three encoding schemes.

    Uses the postings and TF lists already stored in the index to measure
    the size in bytes produced by each encoder, without re-reading documents.
    Also shows the ratio relative to StandardPostings (the baseline).

    Three encoders are compared
    ───────────────────────────
    standard    Raw 4-byte unsigned integers (no compression)
    vbe         Variable-Byte Encoding with gap-based postings
    elias       Elias-Gamma Encoding with gap-based postings
    """
    from index import InvertedIndexReader
    from compression import StandardPostings, VBEPostings, EliasGammaPostings

    enc_class = _get_encoding(args.compression)
    index_path = os.path.join(args.output_dir, args.index_name + ".index")
    if not os.path.exists(index_path):
        print(f"[ERROR] Index not found: {index_path}\n"
              "Run `irengine.py index` first.", file=sys.stderr)
        sys.exit(1)

    encoders = [
        ("standard", StandardPostings),
        ("vbe",      VBEPostings),
        ("elias",    EliasGammaPostings),
    ]

    # Accumulators: bytes for postings and TF per encoder
    totals = {name: {"postings": 0, "tf": 0, "terms": 0}
              for name, _ in encoders}

    with InvertedIndexReader(args.index_name, enc_class,
                             directory=args.output_dir) as idx:
        for term_id, postings, tf_list in idx:
            for name, enc in encoders:
                totals[name]["postings"] += len(enc.encode(postings))
                totals[name]["tf"]       += len(enc.encode_tf(tf_list))
                totals[name]["terms"]    += 1

    baseline_total = (totals["standard"]["postings"]
                      + totals["standard"]["tf"])

    n_terms = totals["standard"]["terms"]
    print(f"\nCompression statistics  ({n_terms:,} terms)\n")
    print(f"{'Encoding':<12}  {'Postings':>12}  {'TF list':>12}  "
          f"{'Total':>12}  {'Ratio':>8}  {'Saving':>8}")
    print("-" * 72)
    for name, enc in encoders:
        p   = totals[name]["postings"]
        tf  = totals[name]["tf"]
        tot = p + tf
        ratio  = tot / baseline_total if baseline_total else 1.0
        saving = (1 - ratio) * 100
        print(f"{name:<12}  {p:>12,}  {tf:>12,}  "
              f"{tot:>12,}  {ratio:>7.3f}x  {saving:>7.1f}%")
    print("-" * 72)
    print(f"(baseline = standard, {baseline_total:,} bytes total)\n")


# ── argument parser ───────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="irengine",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── shared options (reused across sub-commands) ────────────────────────
    def add_shared(p, need_data=False):
        if need_data:
            p.add_argument("--data-dir", default="collection",
                           metavar="DIR",
                           help="Root directory of the document collection "
                                "(default: collection)")
        p.add_argument("--output-dir", default="index",
                       metavar="DIR",
                       help="Directory for index files (default: index)")
        p.add_argument("--index-name", default="main_index",
                       metavar="NAME",
                       help="Base name of the index files (default: main_index)")
        p.add_argument("--compression",
                       choices=["standard", "vbe", "elias"],
                       default="vbe",
                       help="Postings-list compression scheme "
                            "(default: vbe)")

    def add_bm25_opts(p):
        p.add_argument("--bm25-k1", type=float, default=1.2, metavar="K1",
                       dest="bm25_k1",
                       help="BM25 term-saturation parameter (default: 1.2)")
        p.add_argument("--bm25-b",  type=float, default=0.75, metavar="B",
                       dest="bm25_b",
                       help="BM25 length-normalisation parameter (default: 0.75)")

    def add_lsi_opts(p):
        p.add_argument("--lsi-components", type=int, default=100,
                       metavar="K", dest="lsi_components",
                       help="LSI latent dimensions (default: 100)")
        p.add_argument("--lsi-algorithm",
                       choices=["randomized", "arpack"],
                       default="randomized", dest="lsi_algorithm",
                       help="SVD algorithm for LSI (default: randomized)")
        p.add_argument("--lsi-build", action="store_true", dest="lsi_build",
                       help="(Re)build the LSI model before evaluating")

    # ── index ──────────────────────────────────────────────────────────────
    p_idx = sub.add_parser(
        "index",
        help="Build an inverted index from a document collection",
        description=cmd_index.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_shared(p_idx, need_data=True)
    p_idx.add_argument("--mode", choices=["bsbi", "spimi"], default="bsbi",
                       help="Indexing algorithm (default: bsbi)")
    p_idx.add_argument("--block-size", type=int, default=1_000_000,
                       metavar="N", dest="block_size",
                       help="SPIMI: max tokens per in-memory block "
                            "(default: 1000000)")
    p_idx.set_defaults(func=cmd_index)

    # ── search ─────────────────────────────────────────────────────────────
    p_srch = sub.add_parser(
        "search",
        help="Query an existing index",
        description=cmd_search.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_shared(p_srch)
    p_srch.add_argument("--method",
                        choices=["tfidf", "bm25", "wand", "lsi"],
                        default="bm25",
                        help="Retrieval method (default: bm25)")
    p_srch.add_argument("--query", metavar="TEXT",
                        help="A single query string")
    p_srch.add_argument("--query-file", metavar="FILE", dest="query_file",
                        help="File with one query per line "
                             "(optional QID prefix: 'Q1 some query')")
    p_srch.add_argument("-k", type=int, default=10,
                        help="Number of results to return (default: 10)")
    add_bm25_opts(p_srch)
    p_srch.set_defaults(func=cmd_search)

    # ── lsi ────────────────────────────────────────────────────────────────
    p_lsi = sub.add_parser(
        "lsi",
        help="Build the LSI + FAISS vector index",
        description=cmd_lsi.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_shared(p_lsi, need_data=True)
    p_lsi.add_argument("--n-components", type=int, default=100,
                       metavar="K", dest="n_components",
                       help="Number of LSI dimensions (default: 100)")
    p_lsi.add_argument("--svd-algorithm",
                       choices=["randomized", "arpack"],
                       default="randomized", dest="svd_algorithm",
                       help="SVD algorithm (default: randomized)")
    p_lsi.add_argument("--n-probe", type=int, default=10,
                       metavar="N", dest="n_probe",
                       help="FAISS IVF n_probe — cells searched per query "
                            "(default: 10, only used for large collections)")
    p_lsi.set_defaults(func=cmd_lsi)

    # ── evaluate ───────────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate retrieval quality against a qrel file",
        description=cmd_evaluate.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_shared(p_eval, need_data=True)
    p_eval.add_argument("--method",
                        choices=["tfidf", "bm25", "lsi", "all"],
                        default="all",
                        help="Which retrieval method(s) to evaluate "
                             "(default: all)")
    p_eval.add_argument("--qrel-file", default="qrels.txt",
                        metavar="FILE", dest="qrel_file",
                        help="Path to the qrel file (default: qrels.txt)")
    p_eval.add_argument("--query-file", default="queries.txt",
                        metavar="FILE", dest="query_file",
                        help="Path to queries file (default: queries.txt)")
    p_eval.add_argument("-k", type=int, default=1000,
                        help="Retrieval depth per query (default: 1000)")
    p_eval.add_argument("--max-q-id", type=int, default=30,
                        metavar="N", dest="max_q_id",
                        help="Highest query ID in the qrel file (default: 30)")
    p_eval.add_argument("--max-doc-id", type=int, default=1033,
                        metavar="N", dest="max_doc_id",
                        help="Highest document ID in the qrel file "
                             "(default: 1033)")
    add_bm25_opts(p_eval)
    add_lsi_opts(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    # ── compress ───────────────────────────────────────────────────────────
    p_cmp = sub.add_parser(
        "compress",
        help="Show compression statistics for all encoding schemes",
        description=cmd_compress.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_shared(p_cmp)
    p_cmp.set_defaults(func=cmd_compress)

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
