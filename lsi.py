"""
lsi.py — Latent Semantic Indexing (LSI) dengan FAISS Vector Index

Arsitektur
──────────
1. Bangun Term-Document Matrix (TDM) *sparse* langsung dari merged inverted
   index yang sudah ada di disk — tidak perlu re-parsing koleksi.
2. Terapkan Truncated SVD (TruncatedSVD dari sklearn / svds dari scipy) pada
   sparse TDM. Hanya k singular value terbesar yang disimpan sehingga
   kompleksitas O(N·M·k) bukan O(N·M·min(N,M)) seperti SVD penuh.
3. Proyeksikan setiap vektor dokumen ke ruang LSI k-dimensi.
4. Bangun FAISS vector index di atas vektor dokumen tersebut untuk pencarian
   nearest-neighbor yang sangat cepat.
5. Saat query: transformasikan vektor query ke ruang LSI yang sama, lalu
   cari dokumen terdekat via FAISS.

Efisiensi untuk TDM yang sangat besar
──────────────────────────────────────
Masalah utama: TDM berukuran (|V| × |D|) bisa ratusan ribu × ratusan ribu
elemen jika di-store sebagai dense matrix → tidak muat di RAM.

Solusi yang diimplementasikan:

  A. Sparse matrix (scipy.sparse.csc_matrix)
     TDM hampir selalu sangat sparse: rata-rata term hanya muncul di sebagian
     kecil dokumen. Dengan CSC format, hanya nilai non-zero dan indeksnya yang
     disimpan. Untuk koleksi 100k docs × 200k terms dengan rata-rata 50 term
     unik per dokumen, densitas ≈ 0.025% → sparse matrix ~50× lebih hemat RAM
     dibanding dense float32.

  B. Truncated SVD (randomized algorithm)
     sklearn TruncatedSVD dengan algorithm='randomized' menggunakan algoritma
     Halko-Martinsson-Tropp (2011): mengestimasi k singular vector terbesar
     tanpa menghitung semua singular value. Kompleksitas O(N·M·k) dan bekerja
     langsung pada sparse input tanpa konversi ke dense.

     Alternatif: scipy.sparse.linalg.svds (ARPACK) — lebih akurat untuk
     k kecil tapi lebih lambat untuk koleksi sangat besar.

  C. FAISS IndexIVFFlat (approximate search untuk koleksi besar)
     Untuk |D| > LARGE_DOC_THRESHOLD, gunakan IndexIVFFlat yang membagi ruang
     vektor menjadi n_lists sel Voronoi. Pencarian hanya dilakukan di n_probe
     sel terdekat → trade-off akurasi vs kecepatan yang bisa dikonfigurasi.
     Untuk |D| kecil, gunakan IndexFlatIP (exact search).

TF-IDF Weighting
────────────────
w(t, D) = (1 + log tf(t, D))  × log(N / df(t))
           ── sublinear TF ──    ─── IDF ────────

Sublinear TF mengurangi dominasi term dengan frekuensi sangat tinggi,
standar untuk LSI (Manning et al., Introduction to IR, Ch. 18).

Kebergantungan (requirements)
──────────────────────────────
  numpy, scipy, sklearn  — selalu diperlukan
  faiss-cpu              — opsional; fallback ke numpy jika tidak tersedia
                           Install: pip install faiss-cpu
"""

import os
import math
import pickle
import logging
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from index import InvertedIndexReader

logger = logging.getLogger(__name__)

# ── Coba import FAISS; fallback ke numpy jika tidak tersedia ────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning(
        "faiss tidak ditemukan. Menggunakan numpy exact-search sebagai fallback. "
        "Install faiss untuk performa lebih baik: pip install faiss-cpu"
    )

# Threshold jumlah dokumen untuk beralih dari exact (FlatIP) ke approximate
# (IVFFlat) FAISS index.
LARGE_DOC_THRESHOLD = 50_000


# ══════════════════════════════════════════════════════════════════════════════
#  NumpyFlatIndex  — fallback exact search jika FAISS tidak tersedia
# ══════════════════════════════════════════════════════════════════════════════

class NumpyFlatIndex:
    """
    Brute-force cosine similarity search menggunakan numpy.
    Drop-in replacement untuk faiss.IndexFlatIP sehingga kode LSIIndex
    tidak perlu cabang if/else di mana-mana.

    Kompleksitas: O(|D|·k) per query — cukup untuk koleksi kecil/menengah.
    Untuk produksi dengan |D| besar, gunakan FAISS.
    """

    def __init__(self, dim):
        self.dim     = dim
        self._vecs   = None   # (n_docs, dim) float32, L2-normalized

    def add(self, vectors):
        """Tambah vektor (numpy array float32, shape (n, dim))."""
        self._vecs = vectors.astype(np.float32)

    def search(self, query_vecs, k):
        """
        Cari k tetangga terdekat berdasarkan inner product (≡ cosine karena
        vektor sudah L2-normalized).

        Returns
        -------
        scores : np.ndarray shape (n_queries, k)
        indices : np.ndarray shape (n_queries, k)
        """
        q  = query_vecs.astype(np.float32)               # (n_q, dim)
        S  = q @ self._vecs.T                             # (n_q, n_docs)
        k_ = min(k, self._vecs.shape[0])
        # argpartition untuk top-k tanpa full sort — O(n_docs·k)
        idx = np.argpartition(S, -k_, axis=1)[:, -k_:]
        # sort hasil kecil itu saja
        order = np.argsort(-S[np.arange(len(S))[:, None], idx], axis=1)
        idx    = idx[np.arange(len(S))[:, None], order]
        scores = S[np.arange(len(S))[:, None], idx]
        return scores, idx

    @property
    def ntotal(self):
        return 0 if self._vecs is None else self._vecs.shape[0]


# ══════════════════════════════════════════════════════════════════════════════
#  LSIIndex
# ══════════════════════════════════════════════════════════════════════════════

class LSIIndex:
    """
    Latent Semantic Indexing dengan FAISS vector index.

    Attributes
    ----------
    index_name : str
        Nama merged inverted index (default: 'main_index').
    output_dir : str
        Direktori yang berisi file index (.index, .dict) dan tempat
        menyimpan model LSI yang sudah dilatih.
    postings_encoding : class
        Kelas encoding (VBEPostings, StandardPostings, dsb.).
    n_components : int
        Dimensi ruang LSI / jumlah singular value yang dipertahankan.
        Nilai umum: 100–300. Lebih besar = lebih akurat tapi lebih lambat.
    svd_algorithm : str
        'randomized' (default, cepat, cocok untuk |V| dan |D| besar) atau
        'arpack' (lebih akurat untuk k kecil, lebih lambat).
    n_lists : int
        Jumlah sel Voronoi untuk FAISS IVFFlat (hanya dipakai jika
        |D| > LARGE_DOC_THRESHOLD). Rule of thumb: sqrt(|D|).
    n_probe : int
        Jumlah sel yang diperiksa saat query pada IVFFlat. Naikkan untuk
        akurasi lebih tinggi dengan biaya kecepatan.
    """

    # Nama file untuk menyimpan model LSI yang sudah dilatih
    _MODEL_FILE = 'lsi_model.pkl'
    _FAISS_FILE = 'lsi_faiss.index'

    def __init__(self, index_name, output_dir, postings_encoding,
                 n_components=100, svd_algorithm='randomized',
                 n_lists=None, n_probe=10):
        self.index_name       = index_name
        self.output_dir       = output_dir
        self.postings_encoding = postings_encoding
        self.n_components     = n_components
        self.svd_algorithm    = svd_algorithm
        self.n_probe          = n_probe

        # n_lists di-set otomatis saat build() jika tidak dispesifikasi
        self._n_lists_override = n_lists

        # Atribut yang diisi setelah build() atau load()
        self.svd_components_  = None   # V^T shape (k, n_terms): term vectors in LSI space
        self.svd_singular_    = None   # Sigma shape (k,)
        self.doc_id_map_list  = None   # list: indeks posisi -> doc_id_int
        self.term_id_to_row   = None   # dict: termID -> baris di TDM
        self.idf_             = None   # np.ndarray shape (n_terms,): IDF per term
        self._faiss_index     = None   # FAISS atau NumpyFlatIndex

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self):
        """
        Bangun LSI model dari merged inverted index yang sudah ada di disk.

        Langkah:
        1. Baca TDM sebagai sparse matrix dari InvertedIndexReader (satu pass).
        2. Terapkan TF-IDF weighting pada sparse matrix.
        3. Jalankan Truncated SVD untuk mendapatkan ruang LSI k-dimensi.
        4. Proyeksikan semua dokumen ke ruang LSI.
        5. L2-normalize vektor dokumen (untuk cosine similarity).
        6. Bangun FAISS index.
        7. Simpan model ke disk.
        """
        logger.info("LSI build: membaca inverted index dari disk...")
        tdm, doc_ids_ordered, term_ids_ordered = self._build_sparse_tdm()

        n_terms, n_docs = tdm.shape
        logger.info(f"  TDM shape: {n_terms} terms × {n_docs} docs, "
                    f"nnz={tdm.nnz}, density={tdm.nnz/(n_terms*n_docs):.5%}")

        logger.info(f"LSI build: TF-IDF weighting...")
        tdm_tfidf, idf = self._apply_tfidf(tdm, n_docs)

        k = min(self.n_components, n_terms - 1, n_docs - 1)
        logger.info(f"LSI build: Truncated SVD k={k}, algorithm='{self.svd_algorithm}'...")
        doc_vecs, components, singular = self._truncated_svd(tdm_tfidf, k)

        # L2-normalize agar inner product ≡ cosine similarity
        doc_vecs = normalize(doc_vecs, norm='l2').astype(np.float32)

        logger.info(f"LSI build: membangun FAISS index ({n_docs} vecs, dim={k})...")
        faiss_idx = self._build_faiss_index(doc_vecs, n_docs)

        # Simpan semua state
        self.svd_components_  = components    # (k, n_terms) — term vectors
        self.svd_singular_    = singular      # (k,)
        self.doc_id_map_list  = doc_ids_ordered
        self.term_id_to_row   = {tid: i for i, tid in enumerate(term_ids_ordered)}
        self.idf_             = idf
        self._faiss_index     = faiss_idx
        # Mapping string (token -> row, doc_id int -> path) diisi oleh
        # LSIIndexBuilder.build() setelah method ini selesai.
        self._token_to_row   = {}
        self._int_to_docpath = {}

        logger.info("LSI build selesai (mapping string akan diisi oleh builder).")

    def _build_sparse_tdm(self):
        """
        Bangun Term-Document Matrix sebagai scipy.sparse.csc_matrix
        langsung dari merged inverted index (satu pass, tidak perlu
        re-parsing koleksi).

        Menggunakan format COO (koordinat) saat konstruksi karena
        penambahan elemen lebih efisien, lalu konversi ke CSC untuk SVD.

        Returns
        -------
        tdm : csc_matrix shape (n_terms, n_docs)
            Raw TF matrix (belum di-weight).
        doc_ids_ordered : list[int]
            Urutan doc_id integer (kolom TDM).
        term_ids_ordered : list[int]
            Urutan termID (baris TDM).
        """
        rows, cols, data = [], [], []

        # Mapping doc_id -> kolom TDM dibangun secara lazy saat pertama kali
        # doc_id muncul di postings list mana pun.
        doc_col  = {}   # doc_id_int -> kolom index di TDM
        term_ids = []   # termID per baris (sama urutan dengan iterasi index)

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as idx:
            for row_idx, (term_id, postings, tf_list) in enumerate(idx):
                term_ids.append(term_id)
                for doc_id, tf in zip(postings, tf_list):
                    if doc_id not in doc_col:
                        doc_col[doc_id] = len(doc_col)
                    col_idx = doc_col[doc_id]

                    # Sublinear TF: 1 + log(tf) — mengurangi dominasi term
                    # yang sangat sering muncul dalam satu dokumen.
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(1.0 + math.log(tf) if tf > 0 else 0.0)

        n_terms = len(term_ids)
        n_docs  = len(doc_col)

        tdm = sp.csc_matrix(
            (np.array(data, dtype=np.float32),
             (np.array(rows, dtype=np.int32),
              np.array(cols, dtype=np.int32))),
            shape=(n_terms, n_docs)
        )

        # Urutan kolom (doc_id) perlu disimpan untuk reverse lookup
        doc_ids_ordered = [None] * n_docs
        for doc_id, col in doc_col.items():
            doc_ids_ordered[col] = doc_id

        return tdm, doc_ids_ordered, term_ids

    def _apply_tfidf(self, tdm, n_docs):
        """
        Terapkan pembobotan IDF pada TDM secara in-place (column-scale).

        IDF(t) = log(N / df(t))

        df(t) dihitung dari jumlah kolom non-zero per baris TDM,
        tanpa membaca ulang dari disk.

        Returns
        -------
        tdm_tfidf : csc_matrix
            TDM setelah dikali IDF (masih sparse).
        idf : np.ndarray shape (n_terms,)
            Nilai IDF per term untuk digunakan saat query transform.
        """
        # df(t) = jumlah dokumen yang mengandung term t
        # Untuk CSC matrix, binarisasi lalu sum per baris
        binary = tdm.copy()
        binary.data[:] = 1.0
        df = np.asarray(binary.sum(axis=1)).ravel()   # (n_terms,)
        df = np.maximum(df, 1)                         # hindari log(0)

        idf = np.log(n_docs / df).astype(np.float32)  # (n_terms,)

        # Scale baris TDM dengan IDF menggunakan sparse diagonal multiplication
        # sp.diags(idf) @ tdm — O(nnz), tidak ada konversi ke dense
        idf_diag  = sp.diags(idf)
        tdm_tfidf = idf_diag @ tdm       # masih sparse

        return tdm_tfidf, idf

    def _truncated_svd(self, matrix, k):
        """
        Truncated SVD: matrix ≈ U · diag(Σ) · V^T, simpan hanya k terbesar.

        Pilihan algorithm:
        - 'randomized': sklearn TruncatedSVD dengan randomized range finder
          (Halko et al. 2011). Kompleksitas O(nnz·k + (n+m)·k²).
          Cocok untuk matrix sangat besar dan k kecil-menengah (< 500).
        - 'arpack': scipy svds dengan iterative Arnoldi. Lebih akurat untuk
          k sangat kecil tapi lebih lambat untuk k besar.

        Proyeksi dokumen:
          D_lsi = Σ^{-1} · U^T · TDM^T  → bentuk (k, n_docs)^T = (n_docs, k)
          atau ekuivalen (lebih numerik stabil):
          D_lsi = V · Σ   (dari dekomposisi TDM = U·Σ·V^T, kolom V = doc vecs)

        Di sini kita pakai:
          TDM (n_terms × n_docs) = U · Σ · V^T
          doc_vecs = V · Σ  → shape (n_docs, k)
          Sehingga V^T (= components_) adalah "term vectors" di ruang LSI.

        Returns
        -------
        doc_vecs : np.ndarray (n_docs, k)
        components : np.ndarray (k, n_terms)   — V^T
        singular : np.ndarray (k,)             — Σ diagonal
        """
        if self.svd_algorithm == 'arpack':
            # scipy svds mengembalikan singular values dalam urutan MENAIK;
            # kita balik agar indeks 0 = singular value terbesar.
            U, s, Vt = svds(matrix.T, k=k)   # input: docs × terms (transpose)
            order    = np.argsort(-s)
            s, U, Vt = s[order], U[:, order], Vt[order, :]
            # U di sini = V dalam notasi TDM = U·Σ·V^T
            # doc_vecs = U * s
            doc_vecs   = U * s                # (n_docs, k)
            components = Vt                   # (k, n_terms)
            singular   = s
        else:
            # sklearn TruncatedSVD: input matrix (n_terms × n_docs)
            # fit_transform menghasilkan "transformed X" yaitu U·Σ (n_terms,k)
            # kita butuh V·Σ untuk doc_vecs
            svd = TruncatedSVD(n_components=k, algorithm='randomized',
                               random_state=42)
            # Fit pada matrix^T supaya doc = baris output
            # matrix^T: (n_docs × n_terms)
            doc_vecs   = svd.fit_transform(matrix.T)    # (n_docs, k) = V·Σ
            components = svd.components_                # (k, n_terms) = U^T
            singular   = svd.singular_values_           # (k,)

        return doc_vecs.astype(np.float32), \
               components.astype(np.float32), \
               singular.astype(np.float32)

    def _build_faiss_index(self, doc_vecs, n_docs):
        """
        Bangun FAISS (atau numpy fallback) vector index.

        Strategi:
        - n_docs ≤ LARGE_DOC_THRESHOLD → IndexFlatIP (exact cosine search)
        - n_docs > LARGE_DOC_THRESHOLD → IndexIVFFlat (approximate, lebih cepat)

        Untuk IndexIVFFlat:
          n_lists = n_lists_override atau ceil(sqrt(n_docs))
          n_lists harus ≥ k (query top-k) dan ≤ n_docs.
          n_probe mengontrol berapa sel yang dicek saat query.

        Semua vektor di-L2-normalize sebelum masuk index, sehingga
        inner product = cosine similarity.
        """
        dim = doc_vecs.shape[1]

        if not FAISS_AVAILABLE:
            logger.info("  FAISS tidak tersedia; menggunakan NumpyFlatIndex.")
            idx = NumpyFlatIndex(dim)
            idx.add(doc_vecs)
            return idx

        if n_docs <= LARGE_DOC_THRESHOLD:
            logger.info(f"  FAISS IndexFlatIP (exact), dim={dim}")
            idx = faiss.IndexFlatIP(dim)
            idx.add(doc_vecs)
        else:
            n_lists = self._n_lists_override or max(int(math.ceil(math.sqrt(n_docs))), 1)
            n_lists = min(n_lists, n_docs)
            logger.info(f"  FAISS IndexIVFFlat (approximate), dim={dim}, "
                        f"n_lists={n_lists}, n_probe={self.n_probe}")
            quantizer = faiss.IndexFlatIP(dim)
            idx       = faiss.IndexIVFFlat(quantizer, dim, n_lists,
                                           faiss.METRIC_INNER_PRODUCT)
            idx.train(doc_vecs)
            idx.add(doc_vecs)
            idx.nprobe = self.n_probe

        return idx

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query, k=10):
        """
        Ranked retrieval menggunakan LSI + FAISS nearest-neighbor search.

        Pipeline:
        1. Tokenisasi query dan bangun vektor TF-IDF sparse (|V|,).
        2. Proyeksikan ke ruang LSI k-dimensi:
              q_lsi = (q_tfidf @ V) / Σ
              (di mana V·Σ = doc space; membagi Σ agar skala setara)
        3. L2-normalize q_lsi.
        4. Cari k nearest neighbor di FAISS index.
        5. Kembalikan top-k (score, doc_path).

        Parameters
        ----------
        query : str
            Token query dipisahkan spasi.
        k : int
            Jumlah dokumen teratas yang dikembalikan.

        Returns
        -------
        List[(float, str)]
            Top-K pasangan (cosine_score, doc_path), terurut menurun.
        """
        if self._faiss_index is None:
            raise RuntimeError("LSIIndex belum dibangun. Panggil build() atau load() dulu.")

        # ── Bangun vektor query sparse (1 × n_terms) ────────────────────────
        n_terms  = len(self.term_id_to_row)
        q_data, q_rows, q_cols = [], [], []

        tf_map = {}
        for token in query.split():
            tf_map[token] = tf_map.get(token, 0) + 1

        for token, tf in tf_map.items():
            # term_id_to_row hanya berisi token yang ada di vocabulary index;
            # token query yang tidak dikenal di-skip (tidak error)
            if token not in self._token_to_row:
                continue
            row = self._token_to_row[token]
            w   = (1.0 + math.log(tf)) * self.idf_[row]
            q_data.append(w)
            q_rows.append(0)
            q_cols.append(row)

        if not q_data:
            return []   # semua query token tidak ada di vocabulary

        q_sparse = sp.csr_matrix(
            (np.array(q_data, dtype=np.float32),
             (np.array(q_rows, dtype=np.int32),
              np.array(q_cols, dtype=np.int32))),
            shape=(1, n_terms)
        )

        # ── Proyeksikan ke ruang LSI ─────────────────────────────────────────
        # components_ shape: (k, n_terms) = U^T (dalam notasi TDM = U·Σ·V^T)
        # q_lsi = q_sparse @ components_.T / sigma
        # Catatan: doc_vecs dibangun sebagai V·Σ, lalu dinormalisasi.
        # Untuk cosine similarity yang konsisten, query harus di-project
        # dengan cara yang sama: q @ V / ||q @ V||
        q_lsi = q_sparse @ self.svd_components_.T   # (1, k)
        q_lsi = np.asarray(q_lsi).astype(np.float32)

        # L2-normalize query vector
        norm = np.linalg.norm(q_lsi)
        if norm < 1e-10:
            return []
        q_lsi /= norm

        # ── FAISS search ─────────────────────────────────────────────────────
        scores, indices = self._faiss_index.search(q_lsi, k)
        scores  = scores[0]    # (k,)
        indices = indices[0]   # (k,)

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:         # FAISS kadang kembalikan -1 jika k > n_docs
                continue
            doc_id   = self.doc_id_map_list[idx]
            doc_path = self._int_to_docpath[doc_id]
            results.append((float(score), doc_path))

        return sorted(results, key=lambda x: x[0], reverse=True)

    # ── Persist ────────────────────────────────────────────────────────────────

    def _save(self):
        """
        Simpan model LSI ke disk.
        - Model metadata (SVD components, IDF, mappings) → pickle
        - FAISS index → file terpisah via faiss.write_index (lebih efisien
          untuk index besar; numpy fallback via pickle)
        """
        model_path = os.path.join(self.output_dir, self._MODEL_FILE)
        faiss_path = os.path.join(self.output_dir, self._FAISS_FILE)

        # Bangun token→row mapping (dari term_id_to_row + doc_id_map)
        # Disimpan terpisah agar retrieve() tidak perlu BSBIIndex saat query.
        # (harus di-set dari luar setelah build, lihat LSIIndexBuilder)
        model = {
            'svd_components': self.svd_components_,
            'svd_singular':   self.svd_singular_,
            'idf':            self.idf_,
            'doc_id_map_list': self.doc_id_map_list,
            'term_id_to_row':  self.term_id_to_row,
            'token_to_row':    self._token_to_row,
            'int_to_docpath':  self._int_to_docpath,
            'n_components':    self.n_components,
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"  Model LSI disimpan ke {model_path}")

        if FAISS_AVAILABLE and not isinstance(self._faiss_index, NumpyFlatIndex):
            faiss.write_index(self._faiss_index, faiss_path)
            logger.info(f"  FAISS index disimpan ke {faiss_path}")
        else:
            # Simpan NumpyFlatIndex via pickle
            with open(faiss_path + '.npy.pkl', 'wb') as f:
                pickle.dump(self._faiss_index, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"  Numpy index disimpan ke {faiss_path}.npy.pkl")

    def load(self):
        """
        Muat model LSI yang sudah dibangun sebelumnya dari disk.
        Panggil ini di awal program agar tidak perlu build() ulang.
        """
        model_path = os.path.join(self.output_dir, self._MODEL_FILE)
        faiss_path = os.path.join(self.output_dir, self._FAISS_FILE)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.svd_components_  = model['svd_components']
        self.svd_singular_    = model['svd_singular']
        self.idf_             = model['idf']
        self.doc_id_map_list  = model['doc_id_map_list']
        self.term_id_to_row   = model['term_id_to_row']
        self._token_to_row    = model['token_to_row']
        self._int_to_docpath  = model['int_to_docpath']
        self.n_components     = model['n_components']

        npy_path = faiss_path + '.npy.pkl'
        if os.path.exists(npy_path):
            with open(npy_path, 'rb') as f:
                self._faiss_index = pickle.load(f)
        elif FAISS_AVAILABLE and os.path.exists(faiss_path):
            self._faiss_index = faiss.read_index(faiss_path)
            if hasattr(self._faiss_index, 'nprobe'):
                self._faiss_index.nprobe = self.n_probe
        else:
            raise FileNotFoundError(
                f"File FAISS index tidak ditemukan di {faiss_path}. "
                "Jalankan build() terlebih dahulu."
            )
        logger.info(f"LSI model dimuat dari {self.output_dir} "
                    f"({self._faiss_index.ntotal} docs, dim={self.n_components})")


# ══════════════════════════════════════════════════════════════════════════════
#  LSIIndexBuilder  — jembatan antara BSBIIndex dan LSIIndex
# ══════════════════════════════════════════════════════════════════════════════

class LSIIndexBuilder:
    """
    Helper yang menghubungkan BSBIIndex (yang menyimpan term_id_map dan
    doc_id_map) dengan LSIIndex (yang hanya tahu integer ID).

    Dipisahkan dari LSIIndex agar LSIIndex tetap ringan saat dipakai
    hanya untuk retrieval (tanpa perlu load BSBIIndex).

    Usage
    ─────
    builder = LSIIndexBuilder(bsbi_instance, n_components=150)
    lsi     = builder.build()       # bangun dan simpan model
    # ...
    lsi2    = LSIIndexBuilder.load_lsi(index_name, output_dir, postings_encoding)
    results = lsi2.retrieve("query terms", k=10)
    """

    def __init__(self, bsbi_index, n_components=100, svd_algorithm='randomized',
                 n_lists=None, n_probe=10):
        """
        Parameters
        ----------
        bsbi_index : BSBIIndex (atau SPIMIIndex)
            Instance index yang sudah di-load (term_id_map dan doc_id_map
            sudah terisi). Gunakan bsbi.load() sebelum memanggil builder ini.
        n_components : int
            Dimensi ruang LSI (default 100).
        svd_algorithm : str
            'randomized' (default) atau 'arpack'.
        n_lists : int or None
            Jumlah sel IVF untuk FAISS approximate index. None = otomatis.
        n_probe : int
            Jumlah sel yang dicek saat query IVF (default 10).
        """
        if len(bsbi_index.term_id_map) == 0 or len(bsbi_index.doc_id_map) == 0:
            bsbi_index.load()
        self.bsbi = bsbi_index

        self._lsi = LSIIndex(
            index_name        = bsbi_index.index_name,
            output_dir        = bsbi_index.output_dir,
            postings_encoding = bsbi_index.postings_encoding,
            n_components      = n_components,
            svd_algorithm     = svd_algorithm,
            n_lists           = n_lists,
            n_probe           = n_probe,
        )

    def build(self):
        """
        Bangun LSI model dan kembalikan instance LSIIndex yang siap dipakai.

        Selain memanggil LSIIndex.build(), method ini juga menyuntikkan dua
        mapping tambahan yang dibutuhkan retrieve():
          _token_to_row   : token string -> indeks baris TDM
          _int_to_docpath : doc_id int   -> path string dokumen
        """
        # Jalankan build utama (bangun TDM, SVD, FAISS)
        self._lsi.build()

        # Suntikkan mapping string yang hanya tersedia di BSBIIndex
        # token string → row TDM
        # term_id_to_row: {termID: row} — dibangun di build()
        # Kita perlu: token_string → row
        term_str_map = {v: k for k, v in
                        self.bsbi.term_id_map.str_to_id.items()}
        token_to_row = {}
        for term_id, row in self._lsi.term_id_to_row.items():
            if term_id in term_str_map:
                token_to_row[term_str_map[term_id]] = row
        self._lsi._token_to_row = token_to_row

        # doc_id int → path string
        int_to_docpath = {}
        for doc_path, doc_id in self.bsbi.doc_id_map.str_to_id.items():
            int_to_docpath[doc_id] = doc_path
        self._lsi._int_to_docpath = int_to_docpath

        # Simpan ulang model agar mapping baru ikut tersimpan
        self._lsi._save()

        return self._lsi

    @staticmethod
    def load_lsi(index_name, output_dir, postings_encoding, n_probe=10):
        """
        Muat LSIIndex yang sudah dibangun tanpa perlu instance BSBIIndex.

        Parameters
        ----------
        index_name : str
        output_dir : str
        postings_encoding : class
        n_probe : int

        Returns
        -------
        LSIIndex
        """
        lsi = LSIIndex(index_name, output_dir, postings_encoding,
                       n_probe=n_probe)
        lsi.load()
        return lsi


# ══════════════════════════════════════════════════════════════════════════════
#  Demo / __main__
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # Contoh penggunaan bersama BSBIIndex yang sudah di-index
    sys.path.insert(0, os.path.dirname(__file__))
    from bsbi import BSBIIndex
    from compression import VBEPostings

    bsbi = BSBIIndex(data_dir='collection',
                     postings_encoding=VBEPostings,
                     output_dir='index')
    bsbi.load()

    builder = LSIIndexBuilder(bsbi, n_components=100, svd_algorithm='randomized')
    lsi     = builder.build()

    queries = [
        "alkylated with radioactive iodoacetate",
        "psychodrama for disturbed children",
        "lipid metabolism in toxemia and normal pregnancy",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        for score, doc in lsi.retrieve(q, k=5):
            print(f"  {doc:50s}  {score:.4f}")
