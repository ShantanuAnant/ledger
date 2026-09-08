"""Candidate-side signals for the WHO score: recent trend, and 2-hop
co-occurrence.

Motivation (RESEARCH.md 2026-08-23). LEDGER is a global two-tower retriever:
score(q, c) = <Wq h, Wc S[c]> plus the D3 interaction terms. RelBench's
ID-GNN, which dominates the recommendation leaderboard, is something quite
different -- it only scores destinations that appear in the source node's
sampled 2-hop neighbourhood:

    scores = torch.zeros(batch_size, task.num_dst_nodes)
    scores[batch[dst].batch, batch[dst].n_id] = sigmoid(out)   # rest stay 0

That restriction is an enormous prior where the answer is structurally near
(rel-stack 15.17, rel-trial 17.43) and fatal where it is not (rel-amazon 0.13,
last place, beaten by global popularity). We keep global recall and add the
prior as features instead of as a hard restriction.

Two families live here:

TemporalIndex -- "how active was candidate c shortly BEFORE time t". Answers
    count/recency queries for millions of (candidate, time) pairs with a
    single vectorised searchsorted, by exploiting the fact that per-entity
    histories are already grouped and time-sorted in the corpus. This also
    lets the popularity feature be computed strictly before the query time,
    which removes the mild "corpus-aggregate" leak the v1 global popularity
    feature had during training.

CoocTable -- top-K item-item co-occurrence, i.e. how many source entities
    linked to BOTH items. That is exactly the src -> dst -> src -> dst path
    ID-GNN's second hop walks. Computed once in chunks and cached.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np

# Pack (entity, unix_seconds) into one sorted int64 key. Times are < 2^31, so
# a 2^32 stride keeps entity ordering dominant and leaves the low bits to
# time. 40M entities * 2^32 ~= 1.7e17, comfortably inside int64.
TSCALE = 1 << 32

DAY = 86400.0
CACHE = pathlib.Path.home() / ".cache" / "ledger"


class TemporalIndex:
    """Time-aware activity statistics for one entity table.

    `count_before(ents, ts)` returns, for parallel arrays of entity ids and
    query times, how many corpus events that entity had strictly before that
    time -- vectorised over the whole batch at once.
    """

    def __init__(self, corpus, table: str):
        off = corpus.hist_offset[table]
        idx = corpus.hist_index[table]
        self.offset = off.astype(np.int64)
        self.times = corpus.time[idx].astype(np.int64)
        ent = np.repeat(np.arange(len(off) - 1, dtype=np.int64), np.diff(off))
        # Sorted by construction: entities are grouped and each group is
        # time-ordered, because hist_index was built from time-ordered ids.
        self.keys = ent * TSCALE + self.times
        self.n_entities = len(off) - 1

    def _pos_multi(self, key: np.ndarray, offsets) -> list:
        """searchsorted for `key - off` for each off, with sorted needles.

        Binary search into a 120 MB array with unsorted needles is
        cache-miss bound: measured 508 ms for 490K lookups vs 28 ms once the
        needles are sorted -- 18x. And because every query here is the same
        key shifted by a CONSTANT, one argsort serves all of them: subtracting
        a constant is monotone, so the sort order is identical.
        """
        order = np.argsort(key, kind="stable")
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        ks = key[order]
        return [np.searchsorted(self.keys, ks - off, side="left")[inv]
                for off in offsets]

    def stats(self, ents: np.ndarray, ts: np.ndarray, window_days: float,
              prev_windows: float, tau_days: float) -> dict:
        """All candidate-side temporal features, vectorised.

        ents, ts are flat parallel arrays. Everything is computed strictly
        BEFORE ts, so it is valid at any query cutoff.
        """
        ents = ents.astype(np.int64)
        ts = ts.astype(np.int64)
        base = self.offset[ents]

        w = int(window_days * DAY)
        pos_now, pos_recent, pos_prev = self._pos_multi(
            ents * TSCALE + ts, (0, w, int(w * (1 + prev_windows))))

        n_total = pos_now - base                    # all activity before t
        n_recent = pos_now - pos_recent             # last `window_days`
        n_prev = pos_recent - pos_prev              # the `prev_windows` before

        # Time since the candidate was last active at all (not w.r.t. this
        # query entity -- that is the D3 `recency` feature).
        has_prev = pos_now > base
        last_t = np.where(has_prev, self.times[np.clip(pos_now - 1, 0,
                                                       len(self.times) - 1)],
                          0)
        age_days = np.where(has_prev, (ts - last_t) / DAY, np.inf)
        staleness = np.where(has_prev, np.exp(-age_days / tau_days), 0.0)

        # Acceleration: is this candidate hotter now than it just was?
        trend = (np.log1p(n_recent)
                 - np.log1p(n_prev / max(prev_windows, 1e-6)))
        return {
            "log_total": np.log1p(n_total).astype(np.float32),
            "log_recent": np.log1p(n_recent).astype(np.float32),
            "trend": trend.astype(np.float32),
            "staleness": staleness.astype(np.float32),
        }


class CoocTable:
    """Top-K destination-destination co-occurrence over a source table.

    cooc[i, j] = number of source entities that linked to both i and j. For
    (customer -> article) this is "how many customers bought both", the
    classic item-item collaborative signal, and structurally it is the second
    hop of ID-GNN's neighbourhood.

    Stored as fixed-width neighbour/weight arrays (`nbr`, `wt`) so lookups
    vectorise. Built in column chunks because the full product is O(n_dst^2)
    dense and has ~10^8 nonzeros sparse on rel-hm.
    """

    def __init__(self, nbr: np.ndarray, wt: np.ndarray):
        self.nbr = nbr          # int32 [n_dst, K], -1 padded
        self.wt = wt            # float32 [n_dst, K], 0 padded
        self.topk = nbr.shape[1]

    @staticmethod
    def _pairs(corpus, src_table: str, dst_table: str):
        """All (src_row, dst_row) links, from every fact table carrying both."""
        S, D = [], []
        for name, spec in corpus.schema.fact_tables.items():
            tgts = list(spec.fkeys.values())
            si = [j for j, t in enumerate(tgts) if t == src_table]
            di = [j for j, t in enumerate(tgts) if t == dst_table]
            if not si or not di:
                continue
            ev = np.flatnonzero(corpus.row_of[name] >= 0)
            rows = corpus.row_of[name][ev]
            for a in si:
                for b in di:
                    s = corpus.links[name][rows, a]
                    d = corpus.links[name][rows, b]
                    ok = (s >= 0) & (d >= 0)
                    S.append(s[ok]); D.append(d[ok])
        if not S:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        return np.concatenate(S), np.concatenate(D)

    @classmethod
    def build(cls, corpus, src_table: str, dst_table: str, topk: int = 32,
              max_src_items: int = 500, chunk: int = 4096,
              cache_key: str | None = None, verbose: bool = True):
        if cache_key:
            CACHE.mkdir(parents=True, exist_ok=True)
            h = hashlib.md5(
                f"{cache_key}|{src_table}|{dst_table}|{topk}|{max_src_items}"
                f"|{corpus.num_events}".encode()).hexdigest()[:16]
            path = CACHE / f"cooc-{h}.npz"
            if path.exists():
                z = np.load(path)
                if verbose:
                    print(f"  cooc: loaded {path.name}", flush=True)
                return cls(z["nbr"], z["wt"])

        from scipy.sparse import csr_matrix

        n_src = corpus.schema.entity_counts[src_table]
        n_dst = corpus.schema.entity_counts[dst_table]
        s, d = cls._pairs(corpus, src_table, dst_table)

        # Drop pathological source entities: a user with 50k links contributes
        # 2.5e9 pairs on its own and adds nothing but noise to co-occurrence.
        if len(s):
            cnt = np.bincount(s, minlength=n_src)
            keep = cnt[s] <= max_src_items
            s, d = s[keep], d[keep]
        if verbose:
            print(f"  cooc: {len(s):,} links, {n_dst:,} destinations",
                  flush=True)

        A = csr_matrix((np.ones(len(s), np.float32), (s, d)),
                       shape=(n_src, n_dst))
        A.sum_duplicates()
        A.data[:] = 1.0                      # presence, not multiplicity
        At = A.T.tocsr()

        nbr = np.full((n_dst, topk), -1, dtype=np.int32)
        wt = np.zeros((n_dst, topk), dtype=np.float32)
        for lo in range(0, n_dst, chunk):
            hi = min(lo + chunk, n_dst)
            block = (At[lo:hi] @ A).tocsr()   # [chunk, n_dst] co-counts
            for r in range(hi - lo):
                a, b = block.indptr[r], block.indptr[r + 1]
                if a == b:
                    continue
                cols = block.indices[a:b]
                vals = block.data[a:b]
                self_mask = cols != (lo + r)  # drop the diagonal
                cols, vals = cols[self_mask], vals[self_mask]
                if not len(cols):
                    continue
                k = min(topk, len(cols))
                sel = np.argpartition(-vals, k - 1)[:k]
                nbr[lo + r, :k] = cols[sel]
                wt[lo + r, :k] = vals[sel]
            if verbose and (lo // chunk) % 8 == 0:
                print(f"  cooc: {hi:,}/{n_dst:,}", flush=True)

        if cache_key:
            np.savez_compressed(path, nbr=nbr, wt=wt)
            if verbose:
                print(f"  cooc: cached -> {path.name}", flush=True)
        return cls(nbr, wt)

    def score_sparse(self, prior_items: np.ndarray):
        """-> (row, col, val) of the nonzero co-occurrence entries.

        At most M*topk entries per row, so this is the right shape for the
        eval path: a query's co-occurrence touches ~256 of 105K candidates,
        and materialising [B, n_dst] to hold that would be 13M mostly-zero
        entries per batch.
        """
        T, M = prior_items.shape
        valid = prior_items >= 0
        if not valid.any():
            return (np.empty(0, np.int64),) * 2 + (np.empty(0, np.float32),)
        t_idx = np.repeat(np.arange(T), M).reshape(T, M)[valid]
        nb = self.nbr[prior_items[valid]]
        wv = self.wt[prior_items[valid]]
        ok = nb >= 0
        if not ok.any():
            return (np.empty(0, np.int64),) * 2 + (np.empty(0, np.float32),)
        rows = np.repeat(t_idx, self.topk).reshape(-1, self.topk)[ok]
        cols = nb[ok].astype(np.int64)
        vals = wv[ok].astype(np.float32)

        key = rows.astype(np.int64) * (int(cols.max()) + 2) + cols
        order = np.argsort(key, kind="stable")
        key, rows, cols, vals = (key[order], rows[order], cols[order],
                                 vals[order])
        uniq, start = np.unique(key, return_index=True)
        return rows[start], cols[start], np.add.reduceat(vals, start)

    def score(self, prior_items: np.ndarray, cands: np.ndarray,
              n_dst: int) -> np.ndarray:
        """-> float32 [T, C]: co-occurrence of each candidate with the query's
        recent items.

        prior_items [T, M] (-1 padded), cands [T, C]. Done by packing
        (target, item) into one key so the whole batch is a sort plus a
        searchsorted, rather than T dict lookups.
        """
        T, M = prior_items.shape
        C = cands.shape[1]
        out = np.zeros((T, C), dtype=np.float32)
        valid = prior_items >= 0
        if not valid.any():
            return out

        t_idx = np.repeat(np.arange(T), M).reshape(T, M)[valid]     # [P]
        rows = prior_items[valid]                                    # [P]
        nb = self.nbr[rows]                                          # [P, K]
        wv = self.wt[rows]                                           # [P, K]
        ok = nb >= 0
        if not ok.any():
            return out
        keys = (np.repeat(t_idx, self.topk).reshape(-1, self.topk)[ok]
                .astype(np.int64) * n_dst + nb[ok].astype(np.int64))
        vals = wv[ok].astype(np.float32)

        order = np.argsort(keys, kind="stable")
        keys, vals = keys[order], vals[order]
        uniq, start = np.unique(keys, return_index=True)
        summed = np.add.reduceat(vals, start)

        q = (np.arange(T, dtype=np.int64)[:, None] * n_dst
             + cands.astype(np.int64))
        p = np.searchsorted(uniq, q.ravel())
        p_cl = np.clip(p, 0, len(uniq) - 1)
        hit = uniq[p_cl] == q.ravel()
        out.ravel()[hit] = summed[p_cl[hit]]
        return out
