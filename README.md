# IR Engine — Documentation

A modular Information Retrieval engine with multiple indexing algorithms,
compression schemes, retrieval models, and evaluation metrics, accessible
through a single command-line interface (`irengine.py`).

---

## Table of Contents

1. [Requirements](#1-requirements)
2. [Project Layout](#2-project-layout)
3. [Collection Format](#3-collection-format)
4. [Quick Start](#4-quick-start)
5. [Command Reference](#5-command-reference)
   - [index](#51-index)
   - [search](#52-search)
   - [lsi](#53-lsi)
   - [evaluate](#54-evaluate)
   - [compress](#55-compress)
6. [Indexing Algorithms](#6-indexing-algorithms)
7. [Compression Schemes](#7-compression-schemes)
8. [Retrieval Models](#8-retrieval-models)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Typical Workflows](#10-typical-workflows)
11. [Module Overview](#11-module-overview)

---

## 1. Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.9 | Runtime |
| numpy | any | Array operations |
| scipy | any | Sparse matrices, ARPACK SVD |
| scikit-learn | any | Randomized Truncated SVD |
| tqdm | any | Progress bars during indexing |
| faiss-cpu | any | Fast vector search for LSI *(optional)* |

Install all at once:

```bash
pip install numpy scipy scikit-learn tqdm
pip install faiss-cpu          # optional but recommended for LSI
```

or

```bash
pip install -r requirements.txt
```

> **No FAISS?** The engine falls back to an exact numpy nearest-neighbour
> search automatically. Results are identical; performance degrades for very
> large collections.

---

## 2. Project Layout

```
.
├── irengine.py       ← unified CLI (this is the entry point)
├── bsbi.py           ← BSBIIndex, SPIMIIndex, retrieval methods
├── index.py          ← InvertedIndexReader / InvertedIndexWriter
├── compression.py    ← StandardPostings, VBEPostings, EliasGammaPostings
├── util.py           ← TrieIdMap (IdMap alias), sorted_merge_posts_and_tfs
├── lsi.py            ← LSIIndex, LSIIndexBuilder
├── evaluation.py     ← metrics (RBP, DCG, NDCG, AP) + eval helpers
│
├── collection/       ← document collection (user-provided)
│   ├── 1/
│   │   ├── 1.txt
│   │   └── 2.txt
│   └── 2/
│       └── 100.txt
│
├── index/            ← generated index files (created by `index`)
│   ├── main_index.index
│   ├── main_index.dict
│   ├── terms.dict
│   ├── docs.dict
│   ├── lsi_model.pkl         ← created by `lsi`
│   └── lsi_faiss.index       ← created by `lsi`
│
├── queries.txt       ← query file for search / evaluate
└── qrels.txt         ← relevance judgements for evaluate
```

---

## 3. Collection Format

The collection must be a directory containing **sub-directories** (blocks),
each holding plain-text `.txt` files. One sub-directory = one indexing block.

```
collection/
  0/   1.txt  2.txt  …
  1/   100.txt  …
  2/   …
```

For BSBI the block boundary follows the directory structure.
For SPIMI the block boundary is token-count based (`--block-size`).

---

## 4. Quick Start

```bash
# 1. Build an index with default settings (BSBI, VBE compression)
python irengine.py index --data-dir collection --output-dir index

# 2. Run a query
python irengine.py search --query "lipid metabolism pregnancy" --method bm25

# 3. Build the LSI vector index (run once after step 1)
python irengine.py lsi

# 4. Search with LSI
python irengine.py search --query "lipid metabolism pregnancy" --method lsi

# 5. Evaluate all methods
python irengine.py evaluate --method all --qrel-file qrels.txt --lsi-build
```

---

## 5. Command Reference

All commands share three optional path arguments:

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir DIR` | `index` | Directory for all index files |
| `--index-name NAME` | `main_index` | Base name for index files |
| `--compression {standard,vbe,elias}` | `vbe` | Postings encoding |

---

### 5.1 `index`

Build an inverted index from a document collection.

```
python irengine.py index [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir DIR` | `collection` | Root of the document collection |
| `--mode {bsbi,spimi}` | `bsbi` | Indexing algorithm |
| `--block-size N` | `1000000` | SPIMI: max tokens per in-memory block |
| `--compression` | `vbe` | Postings-list encoding |

**Examples**

```bash
# Default: BSBI with Variable-Byte Encoding
python irengine.py index

# SPIMI with Elias-Gamma, flushing every 500 000 tokens
python irengine.py index --mode spimi --compression elias --block-size 500000

# Custom paths and index name
python irengine.py index \
    --data-dir /data/cranfield \
    --output-dir /data/idx \
    --index-name cranfield \
    --compression vbe

# Standard (uncompressed) encoding for debugging
python irengine.py index --compression standard
```

**Output printed**

```
[index] mode=BSBI  compression=VBE  data=collection  output=index

[index] Done in 4.2s
[index] Vocabulary  : 18 327 terms
[index] Documents   : 1 033 docs
[index] Index file  : index/main_index.index (1.03 MB)
```

---

### 5.2 `search`

Run ad-hoc queries against an existing index.

```
python irengine.py search [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--method {tfidf,bm25,wand,lsi}` | `bm25` | Retrieval model |
| `--query TEXT` | — | A single query string |
| `--query-file FILE` | — | File with one query per line |
| `-k N` | `10` | Number of results to return |
| `--bm25-k1 K1` | `1.2` | BM25 term-saturation parameter |
| `--bm25-b B` | `0.75` | BM25 length-normalisation parameter |

At least one of `--query` or `--query-file` is required.

**Query file format**

Plain queries (one per line), or with an optional query-ID prefix:

```
Q1 alkylated with radioactive iodoacetate
Q2 psychodrama for disturbed children
lipid metabolism in toxemia
```

**Examples**

```bash
# Single query, BM25, top-5
python irengine.py search --query "psychodrama disturbed children" -k 5

# TF-IDF, batch queries from file
python irengine.py search --method tfidf --query-file queries.txt -k 20

# BM25 with custom parameters
python irengine.py search \
    --method bm25 --bm25-k1 1.5 --bm25-b 0.6 \
    --query "lipid metabolism"

# WAND top-K (BM25 with early termination, faster for large k)
python irengine.py search --method wand -k 1000 --query "blood pressure"

# LSI semantic search (requires prior `lsi` build)
python irengine.py search --method lsi --query "cardiovascular disease treatment"

# Query a different index
python irengine.py search \
    --output-dir /data/idx --index-name cranfield \
    --method bm25 --query "radiation effects"
```

**Output example**

```
Query [Q2]: psychodrama for disturbed children
Method: BM25  k1=1.2 b=0.75
Rank      Score  Document
----------------------------------------------------------------------
  1       4.8821  ./collection/3/486.txt
  2       4.1033  ./collection/1/107.txt
  3       3.9247  ./collection/5/623.txt
  — 3 result(s) in 1.2 ms
```

---

### 5.3 `lsi`

Build (or rebuild) the LSI + FAISS vector index.

Must be run **after** `index`, because it reads the inverted index from disk.
The model is saved inside `--output-dir` and loaded automatically by
`search --method lsi` and `evaluate --method lsi`.

```
python irengine.py lsi [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir DIR` | `collection` | Collection directory (needed to load maps) |
| `--n-components K` | `100` | Number of LSI latent dimensions |
| `--svd-algorithm {randomized,arpack}` | `randomized` | SVD backend |
| `--n-probe N` | `10` | FAISS IVF cells searched per query *(large collections only)* |

**SVD algorithm choice**

| Algorithm | When to use |
|-----------|-------------|
| `randomized` | Default. Fast; recommended for `--n-components ≥ 50` or large vocabularies (Halko et al. 2011 randomized range finder). |
| `arpack` | More accurate for very small `--n-components` (< 30) at the cost of speed. |

**Examples**

```bash
# Default (100 dimensions, randomized SVD)
python irengine.py lsi

# Higher-dimensional semantic space
python irengine.py lsi --n-components 200

# ARPACK for small k
python irengine.py lsi --n-components 20 --svd-algorithm arpack

# Custom index
python irengine.py lsi \
    --data-dir /data/cranfield \
    --output-dir /data/idx \
    --index-name cranfield \
    --n-components 150
```

**Output example**

```
[lsi] Building LSI index:
      n_components=100  algorithm=randomized  n_probe=10
[lsi] Done in 12.3s — 1033 docs indexed at dim=100
```

---

### 5.4 `evaluate`

Score the system against a relevance judgement (qrel) file.

Computes four metrics per query, then reports means across all queries:

| Metric | Full name |
|--------|-----------|
| RBP | Rank-Biased Precision (p = 0.8) |
| DCG | Discounted Cumulative Gain |
| NDCG | Normalized DCG |
| AP | Average Precision (mean = MAP) |

```
python irengine.py evaluate [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--method {tfidf,bm25,lsi,all}` | `all` | Method(s) to evaluate |
| `--qrel-file FILE` | `qrels.txt` | Path to qrel file |
| `--query-file FILE` | `queries.txt` | Path to queries file |
| `-k N` | `1000` | Retrieval depth per query |
| `--max-q-id N` | `30` | Highest query ID in qrel file |
| `--max-doc-id N` | `1033` | Highest document ID in qrel file |
| `--bm25-k1 K1` | `1.2` | BM25 k1 |
| `--bm25-b B` | `0.75` | BM25 b |
| `--lsi-components K` | `100` | LSI dimensions |
| `--lsi-algorithm` | `randomized` | SVD algorithm |
| `--lsi-build` | *(flag)* | Rebuild LSI model before evaluating |

**Qrel file format**

One relevance judgement per line: `<query-id> <doc-id>` (relevant docs only).

```
Q1 166
Q1 258
Q2 104
Q3 55
```

**Query file format** (same as for `search`)

```
Q1 alkylated with radioactive iodoacetate
Q2 psychodrama for disturbed children
```

**Examples**

```bash
# Evaluate all three methods side-by-side
python irengine.py evaluate --method all

# Evaluate only BM25 with custom parameters
python irengine.py evaluate --method bm25 --bm25-k1 1.5 --bm25-b 0.6

# Evaluate LSI, building the model first
python irengine.py evaluate --method lsi --lsi-components 150 --lsi-build

# Evaluate all, building LSI on-the-fly, custom index
python irengine.py evaluate \
    --method all \
    --data-dir /data/cranfield \
    --output-dir /data/idx \
    --index-name cranfield \
    --lsi-build --lsi-components 200 \
    -k 1000
```

**Output example (`--method all`)**

```
============================================================
Hasil evaluasi [TF-IDF] terhadap 30 queries (k=1000)
  RBP  = 0.2341
  DCG  = 4.1203
  NDCG = 0.4820
  AP   = 0.3102  (→ MAP jika di-rata-rata)
============================================================
Hasil evaluasi [BM25 k1=1.2 b=0.75] terhadap 30 queries (k=1000)
  RBP  = 0.2598
  DCG  = 4.3871
  NDCG = 0.5214
  AP   = 0.3489  (→ MAP jika di-rata-rata)
============================================================
Hasil evaluasi [LSI k=100] terhadap 30 queries (k=1000)
  RBP  = 0.2107
  DCG  = 3.9542
  NDCG = 0.4531
  AP   = 0.2876  (→ MAP jika di-rata-rata)

                                   RBP         DCG        NDCG          AP
--------------------------------------------------------------------------
TF-IDF                          0.2341      4.1203      0.4820      0.3102
BM25 (k1=1.2 b=0.75)            0.2598      4.3871      0.5214      0.3489
LSI  (k=100)                    0.2107      3.9542      0.4531      0.2876
--------------------------------------------------------------------------
  * best per metric           *BM25       *BM25       *BM25       *BM25
```

---

### 5.5 `compress`

Display compression statistics for all three encoding schemes on the
existing index.

```
python irengine.py compress [OPTIONS]
```

No extra options beyond the shared path and compression flags.
The `--compression` flag here only selects which encoder was used to
**read** the index (must match what was used to build it).

**Example**

```bash
python irengine.py compress --compression vbe
```

**Output example**

```
Compression statistics  (18,327 terms)

Encoding        Postings       TF list         Total     Ratio    Saving
------------------------------------------------------------------------
standard         4,132,216     4,132,216     8,264,432    1.000x      0.0%
vbe                516,527       618,304     1,134,831    0.137x     86.3%
elias              389,042       461,903       850,945    0.103x     89.7%
------------------------------------------------------------------------
(baseline = standard, 8,264,432 bytes total)
```

---

## 6. Indexing Algorithms

### BSBI — Block Sort-Based Indexing

The default algorithm. Documents are grouped by sub-directory; within each
block, all `(termID, docID)` pairs are collected, sorted, and written to an
intermediate index. After all blocks are processed, a single **external
merge sort** produces the final merged index.

**Characteristics**

- Block boundary = one sub-directory of the collection.
- In-memory data structure: list of `(termID, docID)` pairs (sorted before flush).
- TermIDs are assigned globally during parsing, before sorting.
- Merge is done with `heapq.merge` over all intermediate readers.

```bash
python irengine.py index --mode bsbi
```

### SPIMI — Single-Pass In-Memory Indexing

Streams the entire collection in one pass. Tokens are accumulated in a
hash-table `{token_str → {doc_id: tf}}` without converting to termIDs first.
When the hash-table reaches `--block-size` tokens it is flushed to disk.

**Characteristics**

- Block boundary = number of tokens (configurable via `--block-size`).
- In-memory data structure: dict of dicts (no global sort until flush).
- TermIDs are assigned at flush time (after alphabetic sort of token strings);
  entries are then re-sorted by termID before writing, ensuring correct
  `heapq.merge` behaviour during the final merge.
- More memory-efficient than BSBI for very large collections.

```bash
python irengine.py index --mode spimi --block-size 2000000
```

### Trie-based Vocabulary

Both algorithms use a **Trie** (`TrieIdMap`) rather than a plain Python
dictionary for the term-to-ID mapping. This provides:

- **O(L)** guaranteed lookup with no hash collisions (L = token length).
- **Shared prefix storage**: tokens that share a common prefix (e.g.
  `interest`, `interesting`, `interests`) reuse the same path in the trie.
- **Native lexicographic iteration** used by SPIMI during flush.

---

## 7. Compression Schemes

Select with `--compression {standard,vbe,elias}`.

### Standard

Raw 4-byte unsigned integers (`array.array('L')`). No compression.
Used as the baseline for size comparisons.

### VBE — Variable-Byte Encoding *(default)*

Postings lists are stored as **gap sequences** (each entry = difference from
the previous doc ID). Gaps are then encoded with Variable-Byte Encoding:
small numbers occupy fewer bytes. TF lists are encoded directly (no gap).

Typical saving vs. standard: **≈ 85 %**.

### Elias-Gamma

Also gap-based. Each gap value `g` is represented as:
- Unary code of `⌊log₂(g+1)⌋` zeros followed by a `1`
- Followed by the binary representation of `g+1` minus its leading bit

Values are packed into a bit-stream, padded to the nearest byte.
A +1 shift is applied before encoding (and reversed on decode) to handle
gaps of 0 (which occur when doc_id = 0).

Typical saving vs. standard: **≈ 90 %** (slightly better than VBE for
highly skewed distributions).

---

## 8. Retrieval Models

Select with `--method {tfidf,bm25,wand,lsi}`.

### TF-IDF

Classical vector-space model, Term-at-a-Time (TaaT).

```
w(t, D) = 1 + log tf(t, D)     if tf > 0, else 0
w(t, Q) = log(N / df(t))
score(D, Q) = Σ_t  w(t, Q) · w(t, D)
```

### BM25

Okapi BM25 with sublinear TF saturation and document-length normalisation.

```
score(D, Q) = Σ_t  IDF(t) · tf(t,D)·(k1+1)
                              ─────────────────────────────────
                              tf(t,D) + k1·(1 − b + b·|D|/avgdl)

IDF(t) = log((N − df(t) + 0.5) / (df(t) + 0.5) + 1)
```

Tune with `--bm25-k1` (default 1.2) and `--bm25-b` (default 0.75).

### WAND

Weak AND — BM25 scoring with **top-K pruning**.  Uses per-term upper bounds
pre-computed at index time (`max_tf` stored as the 5th element of each
`postings_dict` entry) to skip document candidates that cannot enter the
top-K heap. Results are identical to BM25; speed improvement grows with k.

```bash
# Faster for large k
python irengine.py search --method wand -k 1000 --query "blood pressure"
```

### LSI — Latent Semantic Indexing

Builds a low-rank approximation of the TF-IDF weighted Term-Document Matrix
via **Truncated SVD**, then indexes all document vectors with **FAISS**.

**Pipeline**

```
Inverted index (disk)
  ↓  sparse CSC matrix (no dense conversion)
TF-IDF weighting  (IDF diagonal × TDM)
  ↓  still sparse
Truncated SVD  →  doc_vecs ∈ ℝ^(|D|×k)
  ↓  L2-normalized
FAISS IndexFlatIP  (|D| ≤ 50 000)
FAISS IndexIVFFlat (|D| > 50 000, approximate)
```

Query transform at retrieval time:

```
q_lsi = q_tfidf @ V   (project onto LSI term axes)
q_lsi = q_lsi / ‖q_lsi‖₂
nearest neighbours via FAISS inner product
```

**Benefits over bag-of-words models**

- Handles synonymy (similar meanings, different words).
- Handles polysemy (same word, different meanings) partially.
- Robust to exact-term mismatch.

**Limitations**

- Requires a separate `lsi` build step.
- Quality depends on `--n-components` (tuning required).
- Slower to build than a plain inverted index.

---

## 9. Evaluation Metrics

All four metrics are computed per query and averaged.

### RBP — Rank-Biased Precision (p = 0.8)

```
RBP = (1 − p) · Σ_{i=1}^{n}  rel_i · p^{i−1}
```

Models a user who reads the ranked list with probability p of continuing
past each result. p = 0.8 represents a fairly persistent user.

### DCG — Discounted Cumulative Gain

```
DCG@k = Σ_{i=1}^{k}  rel_i / log₂(i + 1)
```

Rewards relevant documents placed early in the ranking by discounting
contributions at lower ranks logarithmically.

### NDCG — Normalized DCG

```
NDCG@k = DCG@k / IDCG@k
```

IDCG is the DCG of the ideal ranking (all relevant documents first).
NDCG = 1.0 means the system's ranking is perfect.

### AP — Average Precision (→ MAP)

```
AP = (1/R) · Σ_{i: rel_i=1}  P@i
```

Precision at every rank position where a relevant document was retrieved,
averaged over all R relevant documents. The mean of AP across all queries
is **MAP** (Mean Average Precision).

---

## 10. Typical Workflows

### Workflow A — Standard inverted-index search

```bash
# Step 1: index
python irengine.py index --mode bsbi --compression vbe

# Step 2: search
python irengine.py search --method bm25 --query "YOUR QUERY" -k 10

# Step 3: evaluate
python irengine.py evaluate --method bm25
```

### Workflow B — LSI semantic search

```bash
# Step 1: build inverted index (required by LSI)
python irengine.py index

# Step 2: build LSI model
python irengine.py lsi --n-components 150

# Step 3: search
python irengine.py search --method lsi --query "YOUR QUERY" -k 10

# Step 4: evaluate
python irengine.py evaluate --method lsi
```

### Workflow C — Full comparison

```bash
# Step 1: index
python irengine.py index --mode bsbi --compression vbe

# Step 2: build LSI
python irengine.py lsi --n-components 150

# Step 3: compression stats
python irengine.py compress

# Step 4: evaluate all methods and compare
python irengine.py evaluate --method all -k 1000
```

### Workflow D — SPIMI with Elias-Gamma, custom paths

```bash
python irengine.py index \
    --mode spimi \
    --compression elias \
    --block-size 500000 \
    --data-dir /datasets/medline \
    --output-dir /output/medline_idx \
    --index-name medline

python irengine.py search \
    --output-dir /output/medline_idx \
    --index-name medline \
    --compression elias \
    --method wand -k 100 \
    --query "gene expression cancer"

python irengine.py lsi \
    --data-dir /datasets/medline \
    --output-dir /output/medline_idx \
    --index-name medline \
    --compression elias \
    --n-components 200

python irengine.py evaluate \
    --data-dir /datasets/medline \
    --output-dir /output/medline_idx \
    --index-name medline \
    --compression elias \
    --qrel-file /datasets/medline/qrels.txt \
    --query-file /datasets/medline/queries.txt \
    --method all --lsi-build --lsi-components 200
```

---

## 11. Module Overview

| Module | Responsibility |
|--------|---------------|
| `irengine.py` | Unified CLI; thin wrappers over library modules |
| `bsbi.py` | `BSBIIndex`, `SPIMIIndex`; all retrieval methods |
| `index.py` | `InvertedIndexReader`, `InvertedIndexWriter`; file I/O |
| `compression.py` | `StandardPostings`, `VBEPostings`, `EliasGammaPostings` |
| `util.py` | `TrieNode`, `TrieIdMap` (= `IdMap`), merge utilities |
| `lsi.py` | `LSIIndex`, `LSIIndexBuilder`, `NumpyFlatIndex` fallback |
| `evaluation.py` | `rbp`, `dcg`, `ndcg`, `ap`; `eval_tfidf`, `eval_bm25`, `eval_lsi`, `compare_all` |
