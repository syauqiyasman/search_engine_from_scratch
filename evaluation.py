import re
import math
from bsbi import BSBIIndex
from compression import VBEPostings


######## >>>>> IR Metrics

def rbp(ranking, p=0.8):
    """
    Menghitung Rank Biased Precision (RBP).

    Parameters
    ----------
    ranking : List[int]
        Vektor biner relevansi, misal [1, 0, 1, 1, 0].
        Indeks ke-i mewakili dokumen di peringkat i+1.
    p : float
        Persistence parameter (default: 0.8).

    Returns
    -------
    float
        Skor RBP.
    """
    score = 0.0
    for i in range(1, len(ranking) + 1):
        score += ranking[i - 1] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking, k=None):
    """
    Menghitung Discounted Cumulative Gain (DCG) hingga posisi ke-k.

    Formula:
        DCG@k = Σ_{i=1}^{k} rel_i / log2(i + 1)

    Catatan: formula ini menggunakan versi standar (bukan versi alternatif
    dengan 2^rel - 1), yang cocok untuk relevance biner (0/1).

    Parameters
    ----------
    ranking : List[int]
        Vektor relevansi terurut berdasarkan peringkat sistem,
        misal [1, 0, 1, 1, 0]. Nilai bisa biner (0/1) atau
        graded (0, 1, 2, ...).
    k : int or None
        Cutoff kedalaman ranking. Jika None, gunakan seluruh ranking.

    Returns
    -------
    float
        Skor DCG@k.
    """
    if k is None:
        k = len(ranking)
    score = 0.0
    for i in range(1, min(k, len(ranking)) + 1):
        score += ranking[i - 1] / math.log2(i + 1)
    return score


def ndcg(ranking, k=None):
    """
    Menghitung Normalized Discounted Cumulative Gain (NDCG) hingga posisi ke-k.

    NDCG@k = DCG@k / IDCG@k

    dimana IDCG@k (Ideal DCG) adalah DCG dari ranking ideal: semua dokumen
    relevan diletakkan di posisi teratas. Jika tidak ada dokumen relevan
    sama sekali (IDCG = 0), maka NDCG didefinisikan sebagai 0.

    Parameters
    ----------
    ranking : List[int]
        Vektor relevansi terurut berdasarkan peringkat sistem,
        misal [1, 0, 1, 1, 0].
    k : int or None
        Cutoff kedalaman ranking. Jika None, gunakan seluruh ranking.

    Returns
    -------
    float
        Skor NDCG@k, bernilai antara 0.0 dan 1.0.
    """
    if k is None:
        k = len(ranking)

    actual_dcg = dcg(ranking, k)

    # Ideal ranking: semua dokumen relevan di posisi paling atas
    ideal_ranking = sorted(ranking, reverse=True)
    ideal_dcg = dcg(ideal_ranking, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def ap(ranking):
    """
    Menghitung Average Precision (AP).

    AP = (1 / R) * Σ_{i=1}^{n} P@i * rel_i

    dimana:
        R     = total jumlah dokumen relevan di seluruh koleksi
                (bukan hanya yang berhasil di-retrieve)
        P@i   = precision di posisi i
        rel_i = 1 jika dokumen di posisi i relevan, 0 jika tidak

    Jika R = 0 (tidak ada dokumen relevan untuk query ini), AP = 0.

    Catatan penting: R di sini adalah jumlah dokumen relevan yang DITEMUKAN
    di dalam ranking (karena qrels hanya mencatat dokumen yang dinilai),
    sesuai dengan konvensi evaluasi TREC standar.

    Parameters
    ----------
    ranking : List[int]
        Vektor relevansi terurut berdasarkan peringkat sistem,
        misal [1, 0, 1, 1, 0].

    Returns
    -------
    float
        Skor AP.
    """
    total_relevant = sum(ranking)
    if total_relevant == 0:
        return 0.0

    score = 0.0
    hits = 0
    for i, rel in enumerate(ranking, start=1):
        if rel == 1:
            hits += 1
            score += hits / i  # P@i dikali rel_i (rel_i=1, jadi cukup hits/i)

    return score / total_relevant


######## >>>>> Memuat qrels

def load_qrels(qrel_file="qrels.txt", max_q_id=30, max_doc_id=1033):
    """
    Memuat query relevance judgment (qrels) dalam format dictionary of dictionary:
        qrels[query_id][doc_id] = relevansi (0 atau 1)

    Contoh: qrels["Q3"][12] = 1 artinya Doc 12 relevan dengan Q3.
    """
    qrels = {
        "Q" + str(i): {j: 0 for j in range(1, max_doc_id + 1)}
        for i in range(1, max_q_id + 1)
    }
    with open(qrel_file) as f:
        for line in f:
            parts = line.strip().split()
            qid  = parts[0]
            did  = int(parts[1])
            qrels[qid][did] = 1
    return qrels


######## >>>>> Helper: bangun ranking vector dari hasil retrieval

def _build_ranking(retrieved_docs, qrels, qid):
    """
    Mengubah list hasil retrieval [(score, doc_path), ...] menjadi
    vektor biner relevansi [1, 0, 1, ...] berdasarkan qrels.
    """
    ranking = []
    for _score, doc in retrieved_docs:
        did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
        ranking.append(qrels[qid][did])
    return ranking


######## >>>>> Fungsi evaluasi utama

def eval_retrieval(retrieve_fn, qrels, query_file="queries.txt", k=1000,
                   method_name=""):
    """
    Fungsi evaluasi generik: menerima fungsi retrieval apapun (TF-IDF atau BM25),
    lalu menghitung dan mencetak semua metrik: RBP, DCG, NDCG, AP.

    Parameters
    ----------
    retrieve_fn : callable
        Fungsi dengan signature retrieve_fn(query, k) -> List[(score, doc_path)].
    qrels : dict
        Struktur qrels hasil load_qrels().
    query_file : str
        Path ke file queries.
    k : int
        Jumlah dokumen yang di-retrieve per query (default: 1000).
    method_name : str
        Label metode untuk ditampilkan di output (misal "TF-IDF" atau "BM25").
    """
    rbp_scores, dcg_scores, ndcg_scores, ap_scores = [], [], [], []

    with open(query_file) as f:
        for qline in f:
            parts  = qline.strip().split()
            qid    = parts[0]
            query  = " ".join(parts[1:])

            retrieved = retrieve_fn(query, k)
            ranking   = _build_ranking(retrieved, qrels, qid)

            rbp_scores.append(rbp(ranking))
            dcg_scores.append(dcg(ranking))
            ndcg_scores.append(ndcg(ranking))
            ap_scores.append(ap(ranking))

    n = len(rbp_scores)
    label = f"[{method_name}]" if method_name else ""
    print(f"Hasil evaluasi {label} terhadap {n} queries (k={k})")
    print(f"  RBP  = {sum(rbp_scores)  / n:.4f}")
    print(f"  DCG  = {sum(dcg_scores)  / n:.4f}")
    print(f"  NDCG = {sum(ndcg_scores) / n:.4f}")
    print(f"  AP   = {sum(ap_scores)   / n:.4f}  (→ MAP jika di-rata-rata)")

    return {
        "rbp":  sum(rbp_scores)  / n,
        "dcg":  sum(dcg_scores)  / n,
        "ndcg": sum(ndcg_scores) / n,
        "ap":   sum(ap_scores)   / n,
    }


def eval_tfidf(qrels, query_file="queries.txt", k=1000):
    """Evaluasi dengan metode TF-IDF."""
    BSBI = BSBIIndex(data_dir='collection',
                     postings_encoding=VBEPostings,
                     output_dir='index')
    return eval_retrieval(BSBI.retrieve_tfidf, qrels,
                          query_file=query_file, k=k, method_name="TF-IDF")


def eval_bm25(qrels, query_file="queries.txt", k=1000, k1=1.2, b=0.75):
    """
    Evaluasi dengan metode BM25.

    Parameters
    ----------
    k1 : float
        Parameter saturasi TF (default: 1.2).
    b : float
        Parameter normalisasi panjang dokumen (default: 0.75).
    """
    BSBI = BSBIIndex(data_dir='collection',
                     postings_encoding=VBEPostings,
                     output_dir='index')
    retrieve_fn = lambda query, k: BSBI.retrieve_bm25(query, k=k, k1=k1, b=b)
    return eval_retrieval(retrieve_fn, qrels,
                          query_file=query_file, k=k,
                          method_name=f"BM25 k1={k1} b={b}")


# Alias untuk backward-compatibility
def eval(qrels, query_file="queries.txt", k=1000):
    return eval_tfidf(qrels, query_file=query_file, k=k)


def eval_lsi(qrels, query_file="queries.txt", k=1000,
             n_components=100, svd_algorithm='randomized',
             build=False):
    """
    Evaluasi retrieval menggunakan Latent Semantic Indexing (LSI) + FAISS.

    LSI memproyeksikan query dan dokumen ke ruang semantik berdimensi rendah
    (n_components) menggunakan Truncated SVD atas Term-Document Matrix yang
    sudah di-weight dengan TF-IDF. Pencarian dilakukan via FAISS nearest-
    neighbor (cosine similarity) di ruang proyeksi tersebut.

    Catatan penting tentang k dan LSI
    ──────────────────────────────────
    LSI hanya dapat mengembalikan maksimal sebanyak jumlah dokumen di koleksi.
    Tidak seperti TF-IDF/BM25 yang bisa dengan mudah skip dokumen skor nol,
    LSI selalu memiliki skor cosine untuk setiap dokumen (positif atau negatif).
    Nilai k yang sangat besar (misal 1000) valid selama k ≤ |D|.

    Parameters
    ----------
    qrels : dict
        Struktur qrels hasil load_qrels().
    query_file : str
        Path ke file queries (default: 'queries.txt').
    k : int
        Jumlah dokumen teratas yang di-retrieve per query (default: 1000).
    n_components : int
        Dimensi ruang LSI / jumlah singular value yang dipertahankan
        (default: 100). Nilai lebih besar = lebih akurat tapi lebih lambat
        dan butuh lebih banyak memori saat build.
    svd_algorithm : str
        'randomized' (default, cepat untuk koleksi besar) atau
        'arpack' (lebih akurat untuk n_components kecil).
    build : bool
        Jika True, bangun (atau rebuild) model LSI dari awal sebelum evaluasi.
        Jika False (default), muat model LSI yang sudah ada dari disk.
        Set True saat pertama kali atau setelah re-indexing koleksi.

    Returns
    -------
    dict
        {'rbp': float, 'dcg': float, 'ndcg': float, 'ap': float}
        Mean score atas semua query untuk setiap metrik.
    """
    from lsi import LSIIndexBuilder

    if build:
        # Build dari scratch: perlu BSBIIndex untuk akses term_id_map/doc_id_map
        bsbi = BSBIIndex(data_dir='collection',
                         postings_encoding=VBEPostings,
                         output_dir='index')
        bsbi.load()
        builder = LSIIndexBuilder(bsbi,
                                  n_components=n_components,
                                  svd_algorithm=svd_algorithm)
        lsi = builder.build()
        print(f"[LSI] Model dibangun: {lsi._faiss_index.ntotal} docs, "
              f"dim={lsi.n_components}, algo='{svd_algorithm}'")
    else:
        # Muat model yang sudah ada — tidak perlu BSBIIndex
        lsi = LSIIndexBuilder.load_lsi('main_index', 'index', VBEPostings)
        print(f"[LSI] Model dimuat dari disk: {lsi._faiss_index.ntotal} docs, "
              f"dim={lsi.n_components}")

    return eval_retrieval(lsi.retrieve, qrels,
                          query_file=query_file, k=k,
                          method_name=f"LSI k={lsi.n_components} {svd_algorithm}")


######## >>>>> Perbandingan semua metode sekaligus

def compare_all(qrels, query_file="queries.txt", k=1000,
                bm25_k1=1.2, bm25_b=0.75,
                lsi_components=100, lsi_algorithm='randomized',
                lsi_build=False):
    """
    Jalankan evaluasi semua metode (TF-IDF, BM25, LSI) dan cetak tabel
    perbandingan metrik secara berdampingan.

    Parameters
    ----------
    qrels : dict
        Struktur qrels hasil load_qrels().
    query_file : str
        Path ke file queries.
    k : int
        Kedalaman ranking yang dievaluasi.
    bm25_k1 : float
        Parameter k1 untuk BM25.
    bm25_b : float
        Parameter b untuk BM25.
    lsi_components : int
        Dimensi ruang LSI.
    lsi_algorithm : str
        Algoritma SVD untuk LSI ('randomized' atau 'arpack').
    lsi_build : bool
        Rebuild model LSI dari awal jika True.

    Returns
    -------
    dict[str, dict]
        {'tfidf': {...}, 'bm25': {...}, 'lsi': {...}}
        Masing-masing berisi {'rbp', 'dcg', 'ndcg', 'ap'}.
    """
    SEP = "=" * 60

    print(SEP)
    scores_tfidf = eval_tfidf(qrels, query_file=query_file, k=k)

    print(SEP)
    scores_bm25 = eval_bm25(qrels, query_file=query_file, k=k,
                             k1=bm25_k1, b=bm25_b)

    print(SEP)
    scores_lsi = eval_lsi(qrels, query_file=query_file, k=k,
                           n_components=lsi_components,
                           svd_algorithm=lsi_algorithm,
                           build=lsi_build)

    # ── Tabel perbandingan ───────────────────────────────────────────────────
    all_scores = {
        'TF-IDF':                         scores_tfidf,
        f'BM25 (k1={bm25_k1} b={bm25_b})': scores_bm25,
        f'LSI  (k={lsi_components})':       scores_lsi,
    }

    metrics = ['rbp', 'dcg', 'ndcg', 'ap']
    col_w   = 12

    print(f"\n{'':30s}", end="")
    for m in metrics:
        print(f"{m.upper():>{col_w}}", end="")
    print()

    print("-" * (30 + col_w * len(metrics)))
    for method, scores in all_scores.items():
        print(f"{method:30s}", end="")
        for m in metrics:
            print(f"{scores[m]:>{col_w}.4f}", end="")
        print()
    print("-" * (30 + col_w * len(metrics)))

    # Tandai nilai terbaik per metrik dengan tanda bintang
    print(f"{'* = nilai terbaik':30s}", end="")
    for m in metrics:
        best = max(all_scores[method][m] for method in all_scores)
        best_method = [name for name, sc in all_scores.items()
                       if abs(sc[m] - best) < 1e-9][0]
        print(f"{'* ' + best_method[:8]:>{col_w}}", end="")
    print("\n")

    return {'tfidf': scores_tfidf, 'bm25': scores_bm25, 'lsi': scores_lsi}


######## >>>>> Unit tests metrik

def _test_metrics():
    """Verifikasi kebenaran implementasi metrik dengan kasus sederhana."""
    ranking = [1, 0, 1, 0, 1]   # relevan di rank 1, 3, 5

    # DCG: 1/log2(2) + 1/log2(4) + 1/log2(6) = 1 + 0.5 + ~0.387 ≈ 1.887
    assert abs(dcg(ranking) - (1/math.log2(2) + 1/math.log2(4) + 1/math.log2(6))) < 1e-9, \
        "DCG salah"

    # NDCG: ideal = [1,1,1,0,0], IDCG = 1/log2(2)+1/log2(3)+1/log2(4)
    ideal = [1, 1, 1, 0, 0]
    idcg = dcg(ideal)
    assert abs(ndcg(ranking) - dcg(ranking) / idcg) < 1e-9, "NDCG salah"

    # AP: hits di rank 1,3,5 → P@1=1/1, P@3=2/3, P@5=3/5 → AP=(1+2/3+3/5)/3
    expected_ap = (1/1 + 2/3 + 3/5) / 3
    assert abs(ap(ranking) - expected_ap) < 1e-9, "AP salah"

    # NDCG = 1.0 untuk ranking sempurna
    assert ndcg([1, 1, 0]) == 1.0, "NDCG ranking sempurna salah"

    # AP = 0 dan NDCG = 0 jika tidak ada relevan
    assert ap([0, 0, 0]) == 0.0, "AP tanpa relevan salah"
    assert ndcg([0, 0, 0]) == 0.0, "NDCG tanpa relevan salah"

    print("Semua unit test metrik PASSED ✓")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluasi metode IR: TF-IDF, BM25, LSI"
    )
    parser.add_argument('--method', choices=['tfidf', 'bm25', 'lsi', 'all'],
                        default='all',
                        help="Metode yang dievaluasi (default: all)")
    parser.add_argument('--k', type=int, default=1000,
                        help="Kedalaman ranking (default: 1000)")
    parser.add_argument('--bm25-k1', type=float, default=1.2,
                        dest='bm25_k1')
    parser.add_argument('--bm25-b', type=float, default=0.75,
                        dest='bm25_b')
    parser.add_argument('--lsi-k', type=int, default=100,
                        dest='lsi_k',
                        help="Dimensi ruang LSI (default: 100)")
    parser.add_argument('--lsi-algo', choices=['randomized', 'arpack'],
                        default='randomized', dest='lsi_algo')
    parser.add_argument('--lsi-build', action='store_true',
                        dest='lsi_build',
                        help="Rebuild model LSI dari awal")
    args = parser.parse_args()

    _test_metrics()

    qrels = load_qrels()
    assert qrels["Q1"][166] == 1, "qrels salah"
    assert qrels["Q1"][300] == 0, "qrels salah"

    if args.method == 'tfidf':
        eval_tfidf(qrels, k=args.k)

    elif args.method == 'bm25':
        eval_bm25(qrels, k=args.k, k1=args.bm25_k1, b=args.bm25_b)

    elif args.method == 'lsi':
        eval_lsi(qrels, k=args.k, n_components=args.lsi_k,
                 svd_algorithm=args.lsi_algo, build=args.lsi_build)

    else:  # 'all'
        compare_all(qrels, k=args.k,
                    bm25_k1=args.bm25_k1, bm25_b=args.bm25_b,
                    lsi_components=args.lsi_k, lsi_algorithm=args.lsi_algo,
                    lsi_build=args.lsi_build)
