"""Candidate generation with exact proposal probabilities.

One object serves two roles that are usually written twice and quietly drift
apart:

  * TRAINING negatives, with the proposal probability Q(c | query) needed for
    the logQ correction;
  * INFERENCE stage-1 candidates, the shortlist the reranker sees.

Using the same generator for both is the point. A reranker trained on uniform
negatives and then applied to a structurally-selected shortlist is being asked
a question it never saw; here the training negatives ARE draws from the
shortlist distribution.

The mixture has three components:

    Q(c|q) = a / n_dst                      uniform    -- keeps global recall
           + b * pop[c] / Z                 popularity -- the head-of-catalogue
           + g * cooc[c|q] / Zq             2-hop      -- plausible-but-wrong

The 2-hop component is what makes the negatives HARD: items bought by people
who bought what this customer bought are exactly the confusable ones, and a
uniform negative from a 100K catalogue is trivially separable after a few
thousand steps.

COMBINING TWO PROPOSALS. When per-query negatives (N draws from Q_local) and
a shared pool (S draws from Q_shared) are used together, correcting each set
with its own `log(N_i * Q_i)` DOUBLE COUNTS the catalogue: each set is
separately an unbiased estimate of the full denominator, so their sum
estimates twice it. The correct treatment is one mixture proposal -- a
candidate drawn by either route has effective weight

    log_q(c) = log( N * Q_local(c)  +  S * Q_shared(c) )

which makes the combined sum an unbiased estimate of the true denominator
exactly once. This is why `q_local_dense` exists: the correction for a SHARED
negative still needs the probability the LOCAL sampler would have drawn it.

WHY logQ. Sampled softmax with a non-uniform proposal is a biased estimator of
the full softmax: a candidate drawn more often is penalised more often. The
standard fix is to score negatives at `s_c - log(N * Q(c))`, which makes the
sampled denominator an unbiased importance-weighted estimate of the true one.
Without it, popularity-proportional negatives systematically push down popular
items -- and popular items are most of what the metric rewards. We were
sampling half our negatives by popularity with no correction at all.
"""

from __future__ import annotations

import numpy as np


class CandidateGenerator:
    """Mixture sampler over one destination table, with exact Q."""

    def __init__(self, corpus, dst_table: str, rng: np.random.Generator,
                 cooc=None, p_uniform: float = 0.4, p_pop: float = 0.3,
                 p_cooc: float = 0.3):
        self.c = corpus
        self.dst = dst_table
        self.rng = rng
        self.cooc = cooc
        self.n_dst = corpus.schema.entity_counts[dst_table]

        if cooc is None:                       # renormalise without the 2-hop
            tot = p_uniform + p_pop
            p_uniform, p_pop, p_cooc = p_uniform / tot, p_pop / tot, 0.0
        self.p = (p_uniform, p_pop, p_cooc)

        pop = corpus.link_counts(dst_table).astype(np.float64) + 1.0
        self.pop_p = pop / pop.sum()
        self.pop_cdf = np.cumsum(self.pop_p)

    # -- per-query 2-hop distribution -------------------------------------
    def cooc_dist(self, prior_items: np.ndarray):
        """-> (row, col, prob) of the per-query 2-hop proposal.

        prior_items [T, M]. Probabilities are normalised within each row, so
        this is a proper conditional distribution and can enter Q directly.
        """
        if self.cooc is None:
            return None
        r, c, v = self.cooc.score_sparse(prior_items)
        if not len(r):
            return None
        tot = np.zeros(prior_items.shape[0], dtype=np.float64)
        np.add.at(tot, r, v)
        keep = tot[r] > 0
        r, c, v = r[keep], c[keep], v[keep]
        return r, c, v / tot[r]

    def sample(self, n_targets: int, n_neg: int, cd) -> tuple:
        """-> (cands [T, n_neg], logQ [T, n_neg]) for NEGATIVES only.

        `cd` is the output of `cooc_dist` (or None).
        """
        pu, pp, _ = self.p
        T = n_targets
        n_u = int(round(n_neg * pu))
        n_p = int(round(n_neg * pp))
        n_c = n_neg - n_u - n_p

        parts = [self.rng.integers(0, self.n_dst, size=(T, n_u)),
                 np.clip(np.searchsorted(
                     self.pop_cdf, self.rng.random((T, n_p))),
                     0, self.n_dst - 1)]

        # 2-hop draws, per row, from that row's own neighbour distribution
        cooc_draw = np.full((T, n_c), -1, dtype=np.int64)
        if n_c and cd is not None:
            r, c, p = cd
            order = np.argsort(r, kind="stable")
            r, c, p = r[order], c[order], p[order]
            bounds = np.searchsorted(r, np.arange(T + 1))
            u = self.rng.random((T, n_c))
            for t in range(T):
                a, b = bounds[t], bounds[t + 1]
                if b <= a:
                    continue
                cdf = np.cumsum(p[a:b])
                cdf /= cdf[-1]
                cooc_draw[t] = c[a + np.clip(
                    np.searchsorted(cdf, u[t]), 0, b - a - 1)]
        # rows with no 2-hop mass fall back to uniform so shapes stay fixed
        miss = cooc_draw < 0
        if miss.any():
            cooc_draw[miss] = self.rng.integers(0, self.n_dst, size=miss.sum())
        parts.append(cooc_draw)

        cands = np.concatenate(parts, axis=1)
        return cands, np.log(np.maximum(self.q_of(cands, cd), 1e-12))

    def q_of(self, cands: np.ndarray, cd) -> np.ndarray:
        """Exact Q(c | query) for arbitrary candidates. [T, C] -> [T, C]."""
        pu, pp, pc = self.p
        q = pu / self.n_dst + pp * self.pop_p[cands]
        if pc and cd is not None:
            r, c, p = cd
            key = r.astype(np.int64) * self.n_dst + c
            order = np.argsort(key, kind="stable")
            key, p = key[order], p[order]
            T, C = cands.shape
            want = (np.arange(T, dtype=np.int64)[:, None] * self.n_dst
                    + cands).ravel()
            pos = np.searchsorted(key, want)
            pos_cl = np.clip(pos, 0, len(key) - 1)
            hit = key[pos_cl] == want
            add = np.zeros(len(want))
            add[hit] = p[pos_cl[hit]]
            q = q + pc * add.reshape(T, C)
        return q

    # -- shared (in-batch) negative pool ----------------------------------
    def shared_pool(self, n: int):
        """A query-INDEPENDENT negative pool, drawn once and scored by all.

        Why this exists. Per-target negatives cost a [T, N, d] state gather:
        on rel-amazon at 8k tokens that is 15.8 GB for N=256, which is what
        caps the batch size and why we sample 0.05% of a 506,012 catalogue
        when ARCHITECTURE 7.3 asks for 8,192. A pool shared across the batch
        is ONE [S, d] projection and one matmul, so S=8,192 costs 17 MB.

        The pool deliberately carries only the query-independent components of
        the proposal (uniform + popularity). The 2-hop hard negatives stay
        per-query, because "items bought by people like YOU" is not something
        a shared pool can express -- and they are the negatives that actually
        teach.

        Returns (ids [S], q [S]) with q the EXACT per-draw probability, so the
        logQ correction stays exact. Drawn with replacement: that is what makes
        q a clean per-draw probability, and duplicates are harmless because the
        estimator sums over draws, not over distinct items.
        """
        pu, pp, _ = self.p
        tot = pu + pp
        if tot <= 0:                      # cooc-only proposal: fall back
            pu, pp, tot = 0.5, 0.5, 1.0
        pu, pp = pu / tot, pp / tot       # renormalise: no cooc in the pool
        n_u = int(round(n * pu))
        ids = np.concatenate([
            self.rng.integers(0, self.n_dst, size=n_u),
            np.clip(np.searchsorted(self.pop_cdf,
                                    self.rng.random(n - n_u)),
                    0, self.n_dst - 1),
        ])
        q = pu / self.n_dst + pp * self.pop_p[ids]
        return ids, q

    def q_local_dense(self, cands: np.ndarray) -> np.ndarray:
        """Query-independent part of the LOCAL proposal, for any candidates.

        Needed by the mixture correction below: a shared negative could also
        have been drawn by the local sampler, and the estimator has to account
        for both routes or it is biased.
        """
        pu, pp, _ = self.p
        return pu / self.n_dst + pp * self.pop_p[cands]

    # -- inference-time shortlist -----------------------------------------
    def shortlist(self, prior_items: np.ndarray, n_pop: int,
                  n_cooc: int) -> list:
        """Structural stage-1 candidates per query: history + 2-hop + popular.

        Returned as a list of int arrays (ragged). queries.py unions these
        with the DENSE RETRIEVAL top-K -- that union is the whole design.
        ID-GNN restricts scoring to the sampled neighbourhood and therefore
        cannot recover an item that is structurally far away, which is why it
        places last on all three rel-amazon tasks. Keeping the dense arm means
        the structural prior can only add.
        """
        T = prior_items.shape[0]
        top_pop = np.argsort(-self.pop_p)[:n_pop]
        out = []
        cd = self.cooc_dist(prior_items) if self.cooc is not None else None
        if cd is not None:
            r, c, p = cd
            order = np.lexsort((-p, r))
            r, c = r[order], c[order]
            bounds = np.searchsorted(r, np.arange(T + 1))
        for t in range(T):
            parts = [prior_items[t][prior_items[t] >= 0], top_pop]
            if cd is not None:
                a, b = bounds[t], bounds[t + 1]
                parts.append(c[a:min(b, a + n_cooc)])
            out.append(np.unique(np.concatenate(parts)))
        return out
