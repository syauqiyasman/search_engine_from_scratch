class TrieNode:
    """
    Satu simpul dalam struktur data Trie.

    Attributes
    ----------
    children : dict[str, TrieNode]
        Pemetaan satu karakter ke simpul anak. Menggunakan dict (bukan array
        fixed-size) agar hemat memori untuk karakter arbitrer (token teks,
        path dokumen, dsb.) dan tetap O(1) akses per karakter.
    term_id : int or None
        Jika simpul ini merupakan akhir dari sebuah string yang sudah
        di-insert, term_id berisi ID integer-nya. None jika bukan akhir term.
    """

    __slots__ = ('children', 'term_id')   # hemat memori: hindari __dict__ per node

    def __init__(self):
        self.children = {}    # char -> TrieNode
        self.term_id  = None  # None = bukan akhir term


class TrieIdMap:
    """
    Mapping dua arah antara string (term atau path dokumen) dan integer ID,
    diimplementasikan menggunakan Trie (Prefix Tree).

    Keunggulan Trie vs Python dict (implementasi lama di kelas IdMap)
    ─────────────────────────────────────────────────────────────────
    Kecepatan   dict: O(L) rata-rata (hitung hash lalu bandingkan string),
                      O(L·n) kasus terburuk (hash collision).
                Trie: O(L) *selalu*, karakter per karakter, tanpa hashing,
                      tanpa collision.

    Memori      dict: menyimpan salinan penuh setiap string sebagai key.
                      10.000 term dengan prefix "collection/" menyimpan
                      "collection/" sebanyak 10.000 kali.
                Trie: prefix yang sama dibagi (shared) antar-term sebagai
                      jalur simpul. "interest", "interesting", "interests"
                      hanya memerlukan 7 simpul bersama untuk prefiks
                      "interest", plus 3 simpul akhiran berbeda.

    Keterurutan Trie secara alami mendukung enumerasi term dalam urutan
                leksikografis melalui DFS in-order, tanpa sort tambahan.
                Berguna untuk SPIMI yang mem-flush term secara terurut.

    Interface
    ─────────
    map["token"] -> int : lookup/insert string, kembalikan ID-nya
    map[42]      -> str : reverse lookup O(1), kembalikan string untuk ID 42
    len(map)     -> int : jumlah string yang tersimpan
    map.str_to_id       : view kompatibilitas (dibangun dari Trie secara lazy)
    map.id_to_str       : list reverse mapping, akses O(1) by integer ID

    Kelas ini sepenuhnya pickle-able sehingga kompatibel dengan mekanisme
    save/load yang ada di BSBIIndex.
    """

    def __init__(self):
        self._root     = TrieNode()
        self.id_to_str = []          # index = term_id, value = string asli

    # ── Operasi utama ────────────────────────────────────────────────────────

    def __len__(self):
        """Jumlah string yang terdaftar di TrieIdMap."""
        return len(self.id_to_str)

    def __get_id(self, s):
        """
        Cari ID untuk string s di Trie. Jika belum ada, insert s ke Trie
        dan assign ID baru secara otomatis.

        Kompleksitas: O(L) dimana L = len(s).
        Tidak ada hashing, tidak ada collision.

        Cara kerja:
        1. Mulai dari root, telusuri Trie karakter per karakter.
        2. Jika karakter belum punya simpul anak, buat TrieNode baru.
        3. Di simpul akhir (setelah karakter terakhir s), cek term_id:
           - Sudah ada -> kembalikan term_id yang ada.
           - Belum ada -> assign len(id_to_str) sebagai ID baru, simpan
             s ke id_to_str untuk mendukung reverse lookup O(1).
        """
        node = self._root
        for ch in s:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]

        if node.term_id is None:
            node.term_id = len(self.id_to_str)
            self.id_to_str.append(s)

        return node.term_id

    def __get_str(self, i):
        """
        Reverse lookup: kembalikan string untuk ID i.
        Kompleksitas: O(1) via list id_to_str.
        """
        return self.id_to_str[i]

    def __getitem__(self, key):
        """
        Akses dua arah dengan syntax kurung siku []:
          map["token"] -> int   (str -> ID, auto-insert jika belum ada)
          map[42]      -> str   (ID -> str, reverse lookup O(1))
        """
        if type(key) is int:
            return self.__get_str(key)
        elif type(key) is str:
            return self.__get_id(key)
        else:
            raise TypeError(f"Key harus str atau int, bukan {type(key).__name__}")

    # ── Kompatibilitas & utilitas ────────────────────────────────────────────

    @property
    def str_to_id(self):
        """
        Bangun dan kembalikan dict {string: id} dari seluruh isi Trie.

        Property ini dipertahankan untuk backward-compatibility dengan kode
        yang mengakses .str_to_id secara langsung. Kompleksitas O(N·L) untuk
        membangun dict dari N string dengan rata-rata panjang L.

        Untuk lookup satu term, gunakan map["token"] yang O(L).
        """
        result = {}
        self._collect(self._root, [], result)
        return result

    def _collect(self, node, prefix_chars, result):
        """
        DFS traversal Trie untuk mengumpulkan semua (string, term_id) pairs.
        Menghasilkan string dalam urutan leksikografis secara alami.

        Parameters
        ----------
        node         : TrieNode saat ini dalam traversal
        prefix_chars : list of char yang membentuk prefix menuju node ini
        result       : dict output { string: term_id } yang diisi in-place
        """
        if node.term_id is not None:
            result[''.join(prefix_chars)] = node.term_id
        for ch in sorted(node.children):       # sorted -> output leksikografis
            prefix_chars.append(ch)
            self._collect(node.children[ch], prefix_chars, result)
            prefix_chars.pop()

    def starts_with(self, prefix):
        """
        Cek apakah ada string yang terdaftar dan berawalan `prefix`.
        Kompleksitas: O(|prefix|).

        Parameters
        ----------
        prefix : str

        Returns
        -------
        bool
            True jika ada setidaknya satu string terdaftar dengan awalan prefix.
        """
        node = self._root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

    def keys_with_prefix(self, prefix):
        """
        Kembalikan semua string yang terdaftar dan berawalan `prefix`,
        dalam urutan leksikografis.

        Kompleksitas: O(|prefix| + K·L) dimana K = jumlah hasil,
        L = panjang rata-rata string hasil.

        Parameters
        ----------
        prefix : str

        Returns
        -------
        List[str]
            Daftar string yang berawalan prefix, terurut leksikografis.
            List kosong jika tidak ada yang cocok.
        """
        node = self._root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        result = {}
        self._collect(node, list(prefix), result)
        return list(result.keys())


# Alias backward-compatible: kode lama yang meng-instantiate IdMap() tetap berjalan
# tanpa perubahan apapun. BSBIIndex dan kode lain cukup import IdMap seperti biasa.
IdMap = TrieIdMap


def sorted_merge_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabung (merge) dua lists of tuples (doc id, tf) dan mengembalikan
    hasil penggabungan keduanya (TF perlu diakumulasikan untuk semua tuple
    dengn doc id yang sama), dengan aturan berikut:

    contoh: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
            posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

            return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                   = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Dua buah sorted list of tuples yang akan di-merge.

    Returns
    -------
    List[(Comparablem, int)]
        Penggabungan yang sudah terurut
    """
    i, j = 0, 0
    merge = []
    while (i < len(posts_tfs1)) and (j < len(posts_tfs2)):
        if posts_tfs1[i][0] == posts_tfs2[j][0]:
            freq = posts_tfs1[i][1] + posts_tfs2[j][1]
            merge.append((posts_tfs1[i][0], freq))
            i += 1
            j += 1
        elif posts_tfs1[i][0] < posts_tfs2[j][0]:
            merge.append(posts_tfs1[i])
            i += 1
        else:
            merge.append(posts_tfs2[j])
            j += 1
    while i < len(posts_tfs1):
        merge.append(posts_tfs1[i])
        i += 1
    while j < len(posts_tfs2):
        merge.append(posts_tfs2[j])
        j += 1
    return merge


def test(output, expected):
    """ simple function for testing """
    return "PASSED" if output == expected else "FAILED"


if __name__ == '__main__':

    # ── Test interface yang sama persis dengan IdMap lama ─────────────────────
    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = TrieIdMap()
    assert [term_id_map[term] for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
    assert term_id_map[1] == "semua",   "reverse lookup salah"
    assert term_id_map[0] == "halo",    "reverse lookup salah"
    assert term_id_map["selamat"] == 2, "str lookup salah"
    assert term_id_map["pagi"]    == 3, "str lookup salah"
    assert len(term_id_map) == 4,       "len salah"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = TrieIdMap()
    assert [doc_id_map[d] for d in docs] == [0, 1, 2], "docs_id salah"

    # ── Test fitur khusus Trie ────────────────────────────────────────────────
    t = TrieIdMap()
    for word in ["interest", "interesting", "interests", "inter", "internal"]:
        t[word]

    assert t.starts_with("inter"),   "starts_with salah"
    assert not t.starts_with("xyz"), "starts_with salah"

    kp = t.keys_with_prefix("interest")
    assert set(kp) == {"interest", "interesting", "interests"}, \
        f"keys_with_prefix salah: {kp}"

    # ── Test str_to_id property ───────────────────────────────────────────────
    s2i = term_id_map.str_to_id
    assert s2i["halo"]    == 0, "str_to_id salah"
    assert s2i["semua"]   == 1, "str_to_id salah"
    assert s2i["selamat"] == 2, "str_to_id salah"
    assert s2i["pagi"]    == 3, "str_to_id salah"

    # ── Test alias IdMap ──────────────────────────────────────────────────────
    m = IdMap()
    m["hello"]
    assert type(m) is TrieIdMap, "IdMap alias salah"

    # ── Test pickle ───────────────────────────────────────────────────────────
    import pickle
    blob   = pickle.dumps(term_id_map)
    loaded = pickle.loads(blob)
    assert loaded["halo"]  == 0,       "pickle lookup salah"
    assert loaded[1]       == "semua", "pickle reverse lookup salah"
    assert len(loaded)     == 4,       "pickle len salah"

    # ── Test sorted_merge_posts_and_tfs ───────────────────────────────────────
    assert sorted_merge_posts_and_tfs([(1, 34), (3, 2), (4, 23)],
                                      [(1, 11), (2, 4), (4, 3), (6, 13)]) \
           == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], \
           "sorted_merge_posts_and_tfs salah"

    print("Semua test TrieIdMap PASSED ✓")
