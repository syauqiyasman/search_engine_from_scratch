import os
import pickle
import contextlib
import heapq
import time
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings
from tqdm import tqdm


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                for token in f.read().split():
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi orted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan:
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Ranked Retrieval TaaT (Term-at-a-Time) dengan scoring BM25.
        Setiap dokumen yang mengandung setidaknya satu query term dievaluasi
        secara penuh. Gunakan retrieve_bm25_wand untuk versi yang lebih efisien.

        Formula BM25:
            Score(D, Q) = Σ_t  IDF(t) · tf(t,D)·(k1+1)
                                         ─────────────────────────────────
                                         tf(t,D) + k1·(1 − b + b·|D|/avgdl)

            IDF(t) = log((N − df(t) + 0.5) / (df(t) + 0.5) + 1)   [Robertson]

        Parameters
        ----------
        query : str
            Token query dipisahkan spasi.
        k : int
            Jumlah dokumen teratas yang dikembalikan.
        k1 : float
            Parameter saturasi TF (default 1.2).
        b : float
            Parameter normalisasi panjang dokumen (default 0.75).

        Returns
        -------
        List[(float, str)]
            Top-K pasangan (score, path_dokumen), terurut menurun berdasarkan skor.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            avgdl = sum(merged_index.doc_length.values()) / N if N > 0 else 1

            scores = {}
            for term in terms:
                if term not in merged_index.postings_dict:
                    continue
                df = merged_index.postings_dict[term][1]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                postings, tf_list = merged_index.get_postings_list(term)
                for doc_id, tf in zip(postings, tf_list):
                    dl = merged_index.doc_length.get(doc_id, 0)
                    weight = idf * tf * (k1 + 1) / \
                             (tf + k1 * (1 - b + b * dl / avgdl))
                    scores[doc_id] = scores.get(doc_id, 0.0) + weight

        docs = [(score, self.doc_id_map[doc_id])
                for doc_id, score in scores.items()]
        return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_bm25_wand(self, query, k=10, k1=1.2, b=0.75):
        """
        WAND (Weak AND) Top-K Retrieval dengan BM25 scoring.

        Berbeda dengan retrieve_bm25 yang menghitung skor semua dokumen yang
        mengandung query term, WAND secara agresif melewati (skip) kandidat
        dokumen yang tidak mungkin masuk top-K menggunakan upper bound skor
        per term.

        ─── Ide utama WAND ───────────────────────────────────────────────────
        Setiap term t memiliki upper bound UB(t): nilai maksimum kontribusi
        BM25 term t ke skor dokumen manapun. UB(t) dihitung saat indexing
        (disimpan sebagai max_tf di postings_dict elemen ke-4) dan dievaluasi
        saat retrieval:

            UB(t) = IDF(t) · max_tf(t)·(k1+1)
                              ─────────────────────────────
                              max_tf(t) + k1·(1 − b)

        Formula ini memaksimalkan bobot BM25 term dengan mengasumsikan:
          - tf = max_tf   (TF terbesar yang pernah muncul untuk term ini)
          - dl → 0        (dokumen "sepanjang" 0 token → b·dl/avgdl → 0)
        Asumsi dl → 0 valid karena dl aktual ≥ 1, sehingga penyebut aktual
        selalu ≥ penyebut yang diasumsikan → UB terbukti sebagai upper bound.

        ─── Algoritma WAND ───────────────────────────────────────────────────
        1. Muat postings setiap query term ke memori; tiap term punya pointer
           (ptr) ke posisi dokumen saat ini.
        2. Urutkan term berdasarkan doc_id saat ini (ascending).
        3. Seleksi pivot:
           a. Kumpulkan UB term dari kiri hingga jumlah kumulatif > threshold θ.
           b. Term pertama yang membuat jumlah melampaui θ disebut "pivot".
           c. pivot_doc = doc_id saat ini dari pivot term.
        4. Jika term[0].doc_id < pivot_doc (tidak semua term sebelum pivot
           berada di pivot_doc):
           → Skip setiap term sebelum pivot ke ≥ pivot_doc via binary search.
           → Ulangi dari langkah 2.
        5. Jika term[0].doc_id == pivot_doc (semua term 0..pivot berada di
           pivot_doc):
           → Hitung skor BM25 penuh untuk pivot_doc (termasuk term setelah
             pivot yang kebetulan juga berada di pivot_doc).
           → Perbarui heap top-K dan threshold θ.
           → Majukan semua term yang berada di pivot_doc ke posisi berikutnya.
           → Ulangi dari langkah 2.
        6. Terminasi: jika total UB semua term ≤ θ, tidak ada dokumen yang
           bisa mengalahkan threshold → selesai.

        Kebenaran: sebelum melewati dokumen d < pivot_doc, terbukti bahwa
        sum(UB[0..pivot-1]) ≤ θ, artinya d tidak bisa masuk top-K.

        Parameters
        ----------
        query : str
            Token query dipisahkan spasi.
        k : int
            Jumlah dokumen teratas yang dikembalikan.
        k1 : float
            Parameter saturasi TF (default 1.2).
        b : float
            Parameter normalisasi panjang dokumen (default 0.75).

        Returns
        -------
        List[(float, str)]
            Top-K pasangan (score, path_dokumen), terurut menurun berdasarkan skor.
            Hasil identik dengan retrieve_bm25 namun jauh lebih efisien.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        # Deduplikasi term; term tidak dikenal di-skip via cek postings_dict.
        seen, query_terms = set(), []
        for word in query.split():
            tid = self.term_id_map[word]
            if tid not in seen:
                seen.add(tid)
                query_terms.append(tid)

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:

            N = len(merged_index.doc_length)
            if N == 0:
                return []
            avgdl = sum(merged_index.doc_length.values()) / N

            # ── Bangun struktur data per term ────────────────────────────
            # Setiap entry menyimpan postings, tf, pointer saat ini,
            # IDF, dan upper bound BM25 term ini.
            term_data = []
            for term in query_terms:
                if term not in merged_index.postings_dict:
                    continue
                meta = merged_index.postings_dict[term]
                df = meta[1]
                max_tf = meta[4]  # pre-computed saat indexing

                postings, tf_list = merged_index.get_postings_list(term)
                if not postings:
                    continue

                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

                # Upper bound BM25 term t:
                # Dimaksimalkan saat tf = max_tf dan dl → 0 (b·dl/avgdl → 0)
                #   UB(t) = IDF · max_tf·(k1+1) / (max_tf + k1·(1−b))
                ub = idf * max_tf * (k1 + 1) / (max_tf + k1 * (1 - b))

                term_data.append({
                    'postings': postings,
                    'tf_list': tf_list,
                    'ptr': 0,
                    'idf': idf,
                    'ub': ub,
                })

            if not term_data:
                return []

            # ── Helper functions ─────────────────────────────────────────

            def cur_doc(td):
                """Doc_id saat ini; float('inf') jika postings habis."""
                return td['postings'][td['ptr']] \
                    if td['ptr'] < len(td['postings']) else float('inf')

            def advance_to(td, target):
                """
                Binary-search: majukan ptr ke posisi pertama dengan
                doc_id ≥ target. O(log n) — lebih cepat dari scan linear.
                Ini yang memungkinkan WAND melewati banyak dokumen sekaligus.
                """
                lo, hi = td['ptr'], len(td['postings'])
                while lo < hi:
                    mid = (lo + hi) // 2
                    if td['postings'][mid] < target:
                        lo = mid + 1
                    else:
                        hi = mid
                td['ptr'] = lo

            # ── Main WAND loop ───────────────────────────────────────────
            # Min-heap menyimpan top-K (score, doc_id).
            # threshold θ = skor minimum di heap (heap[0][0]) saat heap penuh.
            heap = []  # min-heap
            threshold = 0.0

            while True:
                # Buang term yang sudah habis.
                term_data = [td for td in term_data
                             if td['ptr'] < len(td['postings'])]
                if not term_data:
                    break

                # Urutkan term berdasarkan doc_id saat ini (ascending).
                term_data.sort(key=cur_doc)

                # ── Seleksi pivot ────────────────────────────────────────
                # Kumpulkan UB dari kiri sampai jumlah kumulatif > threshold.
                # pivot_idx adalah indeks term yang pertama kali membuat
                # jumlah melampaui threshold.
                cum_ub = 0.0
                pivot_idx = -1
                for i, td in enumerate(term_data):
                    cum_ub += td['ub']
                    if cum_ub > threshold:
                        pivot_idx = i
                        break

                if pivot_idx == -1:
                    # Seluruh UB gabungan ≤ threshold → tidak ada kandidat
                    # yang bisa mengalahkan threshold → terminasi.
                    break

                pivot_doc = cur_doc(term_data[pivot_idx])

                # ── Advance atau evaluate ────────────────────────────────
                if cur_doc(term_data[0]) < pivot_doc:
                    # Tidak semua term sebelum pivot sudah di pivot_doc.
                    # Skip (via binary search) setiap term sebelum pivot
                    # ke setidaknya pivot_doc.
                    #
                    # Ini aman karena sum(UB[0..pivot_idx-1]) ≤ threshold,
                    # sehingga dokumen manapun < pivot_doc tidak bisa top-K.
                    for td in term_data[:pivot_idx]:
                        if cur_doc(td) < pivot_doc:
                            advance_to(td, pivot_doc)
                else:
                    # cur_doc(term_data[0]) == pivot_doc.
                    # Karena list terurut: semua term 0..pivot_idx berada
                    # tepat di pivot_doc → hitung skor BM25 penuh.
                    dl = merged_index.doc_length.get(pivot_doc, 0)
                    score = 0.0
                    for td in term_data:
                        if cur_doc(td) != pivot_doc:
                            break  # terurut → term sesudah ini pasti > pivot_doc
                        tf = td['tf_list'][td['ptr']]
                        score += td['idf'] * tf * (k1 + 1) / \
                                 (tf + k1 * (1 - b + b * dl / avgdl))

                    # Perbarui heap top-K.
                    if len(heap) < k:
                        heapq.heappush(heap, (score, pivot_doc))
                        if len(heap) == k:
                            threshold = heap[0][0]
                    elif score > threshold:
                        heapq.heapreplace(heap, (score, pivot_doc))
                        threshold = heap[0][0]

                    # Majukan semua term yang berada di pivot_doc.
                    for td in term_data:
                        if cur_doc(td) == pivot_doc:
                            td['ptr'] += 1

        results = [(score, self.doc_id_map[doc_id])
                   for score, doc_id in heap]
        return sorted(results, key=lambda x: x[0], reverse=True)

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(
                    InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":
    BSBI_instance = BSBIIndex(data_dir='collection', \
                              postings_encoding=VBEPostings, \
                              output_dir='index')
    BSBI_instance.index()  # memulai indexing!