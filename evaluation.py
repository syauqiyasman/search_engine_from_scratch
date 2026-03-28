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
    _test_metrics()

    qrels = load_qrels()

    assert qrels["Q1"][166] == 1, "qrels salah"
    assert qrels["Q1"][300] == 0, "qrels salah"

    print("=" * 55)
    eval_tfidf(qrels)

    print("=" * 55)
    eval_bm25(qrels)