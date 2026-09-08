"""Sequence packing: entity histories -> fixed-budget training batches.

Each batch row is a concatenation of several entities' event sequences up to
`max_len` tokens (no padding waste); `seq_id` fences attention between them.
For every token k that has a successor in the same entity's history, the
batch carries the aligned next-event target (gap, table, links, values); the
final token of each entity before the corpus cutoff yields a right-censored
WHEN target (gap = cutoff - t_last, censored=True).

WHO negatives: mixture of uniform and popularity sampling (popularity =
empirical link frequency in the corpus), per ARCHITECTURE.md 7.3. One
foreign-key slot is designated the "who slot" per fact table; multi-slot
supervision is a planned extension.

WHO targets are grouped BY DESTINATION TABLE (`target["who"]`). They must be:
different fact tables in the same database generally point their who-slot at
different entity tables (on rel-f1 with entity=drivers, `races` -> circuits
but `results`/`qualifying`/`standings` -> races), and candidate ids are row
indices that are only meaningful relative to their own table. v0 carried a
single `who_table` string for the whole batch and read every candidate out of
that one state buffer -- in range, no error raised, and silently wrong on
every database with more than one fact table. rel-hm has exactly one, which is
why it survived. See RESEARCH.md 2026-08-23.
"""

from __future__ import annotations

import numpy as np
import torch

from .corpus import EventCorpus, NO_ENTITY

from .candidates import CandidateGenerator
from .features import CoocTable, TemporalIndex

# who_slot marker: the destination is the sequence entity's own table, reached
# through whichever foreign-key slot is NOT this event's sequence entity. Must
# be resolved per event, so it is a marker rather than a slot index. Chosen
# well outside any real slot range so an accidental use as an index raises
# instead of silently selecting the last slot (which -1 would do).
SELF_LINK = -99

# Per-category query-token counters (see PackedBatcher.query_feats). A column
# wider than QFEAT_CAT_MAX is a poor conditioning variable and an expensive
# one -- the block costs one float per value per query -- and the total is
# capped so a wide schema (rel-trial has eleven fact tables) cannot blow the
# query token up.
QFEAT_CAT_MAX = 16
QFEAT_CAT_SLOTS = 64

# Neighbour-aggregate query features (see PackedBatcher.nbr_feats). One block
# per (fact table, foreign-key slot) pair, capped so a wide schema cannot blow
# the query token up. rel-trial has eleven fact tables and rel-f1's `results`
# alone has four slots, so the cap binds on both.
QFEAT_NBR_SLOTS = 12
#: distinct neighbours pooled per slot, most recent first. Each costs one
#: `corpus.history` call (a slice plus one binary search), so this is the
#: knob that decides whether the block is microseconds or milliseconds.
QFEAT_NBR_K = 4


def query_nbr_layout(schema) -> list:
    """-> [(table, table_idx, slot position, target entity table)]

    WHY THIS BLOCK EXISTS. Our architecture reaches a linked entity only
    through the stale bf16 EMA vector the tokenizer adds per event, and
    RESEARCH.md's post-mortem on rel-f1 driver-dnf named that as the one
    architectural gap the null results pointed at: "whether a driver finishes
    depends mostly on CAR AND CONSTRUCTOR reliability -- a property of a
    linked entity that our architecture can only see through a stale bf16 EMA
    vector, while the GNN baselines aggregate the 2-hop neighbourhood
    directly and score 84.6 to our 68.5."

    This is the sequence-native form of that aggregation. Rather than message
    passing over a graph, it summarises, at the query token, what the
    entity's OWN neighbours have been doing before `t_q`: how many distinct
    ones there are, how concentrated the attachment is, and how active and
    how recent those neighbours are in their own right. For driver-dnf the
    neighbour is the constructor and "how many `results` events has this
    constructor accumulated, and what did their status column look like" is
    exactly the missing variable.

    Self-referential slots are skipped: a slot pointing back at the entity
    table whose history we are summarising re-reads the entity itself, which
    the per-table counters already cover.

    Deterministic from the schema alone and cached on it, exactly like
    `query_cat_layout`, so the training batcher and every eval path agree on
    the layout without passing it around.
    """
    lay = getattr(schema, "_qnbr_layout", None)
    if lay is not None:
        return lay
    per_table = []
    for name, spec in schema.fact_tables.items():
        slots = [(j, tgt) for j, tgt in enumerate(spec.fkeys.values())
                 if tgt in schema.entity_counts]
        if slots:
            per_table.append((name, spec.table_idx, slots))
    lay = []
    # Round-robin over tables for the same reason query_cat_layout does it:
    # taking slots in table order would spend the whole budget on rel-trial's
    # first two tables.
    for depth in range(max((len(s) for _, _, s in per_table), default=0)):
        for name, tidx, slots in per_table:
            if depth >= len(slots) or len(lay) >= QFEAT_NBR_SLOTS:
                continue
            j, tgt = slots[depth]
            lay.append((name, tidx, j, tgt))
    schema._qnbr_layout = lay
    return lay


def query_cat_layout(schema) -> list:
    """-> [(table, table_idx, position among the table's categoricals, card)]

    Deterministic and derived from the schema alone, so the training batcher
    and every eval path agree on the layout without passing it around. The
    budget is spent ROUND-ROBIN across fact tables: taking columns in table
    order would give rel-trial's first two tables the whole allowance and
    leave `outcome_analyses` -- the one that matters -- with none.

    Cached on the schema object, which is per-process (each worker unpickles
    its own), so this is a lookup after the first call.
    """
    lay = getattr(schema, "_qcat_layout", None)
    if lay is not None:
        return lay
    per_table = []
    for name, spec in schema.fact_tables.items():
        cats = [c for c in spec.columns if c.kind == "categorical"]
        keep = [(j, c.cardinality) for j, c in enumerate(cats)
                if 2 <= c.cardinality <= QFEAT_CAT_MAX]
        if keep:
            per_table.append((name, spec.table_idx, keep))
    lay, used = [], 0
    for depth in range(max((len(k) for _, _, k in per_table), default=0)):
        for name, tidx, keep in per_table:
            if depth >= len(keep):
                continue
            j, card = keep[depth]
            if used + card > QFEAT_CAT_SLOTS:
                continue
            lay.append((name, tidx, j, card))
            used += card
    try:
        schema._qcat_layout = lay
    except Exception:
        pass
    return lay


class PackedBatcher:
    """Packs entity histories into training batches.

    `entity_tables` may name SEVERAL sequence entities (D1). Each call to
    `batch()` draws one of them and builds that side's sequences.

    Why both sides matter. `update_states` writes an event's hidden state into
    every entity it links to, so with customer-only sequences S[article] is an
    EMA of CUSTOMER-space hidden states -- "the average customer who bought
    this", which is popularity-confounded and never trained as an article
    representation. Training over article-centric sequences too gives each
    article a state produced from its own history (and with reduce="last",
    its most recent one). See RESEARCH.md 2026-08-23 on the ~0.75 MAP ceiling
    that WHO-loss improvements could not move.
    """

    def __init__(self, corpus: EventCorpus, entity_tables,
                 max_len: int = 512, batch_rows: int = 8,
                 n_neg: int = 256, min_hist: int = 2, seed: int = 0,
                 table_probs=None, use_cooc: bool = False,
                 cooc_topk: int = 32, cooc_cache_key: str | None = None,
                 window: bool = False, logq: bool = False,
                 hard_negs: bool = False, p_uniform: float = 0.5,
                 p_pop: float = 0.5, p_cooc: float = 0.34,
                 window_days: float = 0.0, window_sigma: float = 0.6,
                 retarget: dict | None = None,
                 window_head: bool = False, n_query: int = 4,
                 win_max_events: int = 256, q_tail: float = 0.0,
                 next_event: bool = True, dense_window: bool = False,
                 n_dense: int = 64, query_dyn: bool = False,
                 query_nbr: bool = False):
        self.c = corpus
        # WINDOW-PRIMARY MODE (`next_event=False`). The next-event targets are
        # not merely down-weighted, they are not BUILT: `_build_who` (candidate
        # sampling plus the D3 feature block) is the dominant cost of a batch,
        # and the WHAT gather touches two corpus arrays per target. Zeroing a
        # loss weight would still pay both. See ARCHITECTURE 8.2a for why the
        # window objective is the one that matches the 21 entity tasks.
        self.next_event = next_event
        # -- WindowHead supervision (ledger/model/window.py) -------------------
        # `window_head` inserts QUERY TOKENS into the packed stream: a
        # synthetic token at a sampled time t_q carrying (elapsed, calendar
        # t_q) and nothing else, whose hidden state is asked for the
        # aggregates of (t_q, t_q + D]. Without it the backbone never sees the
        # query time at all -- `pack_histories` uses the cutoff only to
        # truncate, so h is bit-identical whether the query is one day or one
        # year after the entity's last event (RESEARCH.md 2026-08-25).
        self.window_head = window_head
        self.n_query = n_query
        # -- DENSE WINDOW SUPERVISION (`dense_window`) ----------------------
        #
        # The defect this addresses is a supervision-DENSITY one, and it is
        # the reason `--objective window` failed on 2026-08-25 rather than
        # anything about the objective itself. The next-event heads put a
        # target on EVERY token: `_append_target` is called once per event.
        # The window head puts one on `n_query` inserted query tokens per
        # packed sequence -- with the measured settings (max_len 512, two
        # sequences per row, n_query 6) that is ~12 supervised positions out
        # of 512, i.e. 2.3%. Turning the next-event terms off therefore did
        # not test "window objective vs next-event objective"; it tested
        # 2.3% supervision against 100%, with the objective change
        # confounded in. rel-stack duly peaked at step 500 and decayed for
        # the remaining 11,500 (RESEARCH.md 2026-08-26).
        #
        # So: attach a window target to ORDINARY EVENT TOKENS as well. For a
        # token at time t_k we draw an offset d >= 0 and a horizon D and ask
        # for the aggregates of (t_k + d, t_k + d + D]. No new tokens are
        # inserted and the token stream is bit-identical -- this adds
        # supervision without changing what the backbone sees, which is what
        # makes it a clean ablation against the sparse arm.
        #
        # Only the RATE and COUNT-QUANTILE outputs are supervised densely.
        # The per-column detail (category rates, numeric buckets, column
        # sums) needs the actual event rows of each window, which is the
        # expensive part of `_append_query` and the long tail of the task
        # list; it stays on the sparse query tokens. That split falls out of
        # the same grouping `--win_balance weighted` uses.
        #
        # KNOWN ASYMMETRY, stated rather than buried: a dense target is read
        # off an EVENT token, while every eval query is read off an inserted
        # QUERY token (`pack_histories(query_token=True)`). The head is
        # conditioned on (elapsed, horizon) in both cases and the sparse
        # query tokens keep the eval regime in the training mix, but the two
        # populations are not identical. If dense supervision helps the loss
        # and not the metric, this is the first thing to suspect.
        self.dense_window = dense_window
        self.n_dense = n_dense
        # Dense sampling draws from a SEPARATE stream. Sharing `self.rng`
        # would advance it and change which entities the packer picks next,
        # so a dense arm and its control would see different data -- turning
        # a single-variable ablation into two. Derived from the same seed, so
        # a run stays reproducible.
        self.rng_dense = np.random.default_rng(seed + 9973)
        self.win_max_events = win_max_events
        self.q_tail = q_tail
        self.query_dyn = query_dyn
        self.query_nbr = query_nbr
        # Path retargeting (see _seq_destinations). CSR over the intermediate
        # entity: off[v]:off[v+1] indexes val[] with that intermediate's final
        # destinations. Built from PRE-CUTOFF rows by the caller.
        self.retarget = retarget
        # Horizon prior for window mode. 0.0 keeps the v1 behaviour
        # (log-uniform over 1-365 days); a positive value concentrates draws
        # log-normally around it. See _build_who for why that matters.
        self.window_days = window_days
        self.window_sigma = window_sigma
        # Supervision accounting: window mode DROPS targets whose horizon runs
        # past the corpus cutoff, and a starved arm looks exactly like an
        # architectural failure (RESEARCH.md 2026-08-23, D2). Counted so it is
        # observable instead of inferred.
        self.n_targets_seen = 0
        self.n_targets_dropped = 0
        self.use_cooc = use_cooc
        self.cooc_topk = cooc_topk
        self.cooc_cache_key = cooc_cache_key
        self.window = window
        self.logq = logq
        self.hard_negs = hard_negs
        # With hard negatives on, the three-way split; otherwise the v1
        # uniform/popularity mix, so the no-flag path stays comparable to
        # every number measured before today.
        if hard_negs:
            rest = 1.0 - p_cooc
            self.p_uniform, self.p_pop, self.p_cooc = (
                p_uniform * rest, p_pop * rest, p_cooc)
        else:
            self.p_uniform, self.p_pop, self.p_cooc = p_uniform, p_pop, 0.0
        if isinstance(entity_tables, str):
            entity_tables = [entity_tables]
        self.entity_tables = list(entity_tables)
        self.ent = self.entity_tables[0]     # back-compat: the primary side
        self.ent_current = self.ent          # side of the batch being built
        self.max_len = max_len
        self.batch_rows = batch_rows
        self.n_neg = n_neg
        self.rng = np.random.default_rng(seed)

        self.entities = {}
        for t in self.entity_tables:
            lens = np.diff(corpus.hist_offset[t])
            self.entities[t] = np.flatnonzero(lens >= min_hist)
            if len(self.entities[t]) == 0:
                raise ValueError(f"no {t} entity has >= {min_hist} events")
        if table_probs is None:
            table_probs = [1.0] * len(self.entity_tables)
        p = np.asarray(table_probs, dtype=np.float64)
        self.table_probs = p / p.sum()

        # Which foreign-key slot does WHO predict, per (sequence entity, fact
        # table)? It depends on the sequence entity, so with several sides
        # this is a dict of dicts.
        #
        # v0 used fkeys[0] unconditionally. On rel-hm that is `customer` --
        # the SAME table the sequences are built from -- so WHO was asked to
        # predict the customer of the next event in a customer's own history,
        # i.e. the identity function. It learned that easily (perplexity
        # 257 -> 11) and was consequently useless for ranking articles:
        # zero-shot MAP 0.0022. See RESEARCH.md 2026-08-22.
        #
        # Prefer the first slot pointing AWAY from the sequence entity; that
        # is the link carrying actual predictive content.
        #
        # If EVERY slot points back at the sequence entity, the table is
        # skipped entirely (no WHO supervision) rather than falling back to
        # slot 0. Slot 0 there is the identity function -- "the next event in
        # this user's history belongs to this user" -- which is exactly the
        # 2026-08-22 bug that produced a beautiful WHO curve and MAP 0.0022.
        # On rel-stack `badges` is such a table and was contributing 25% of
        # all WHO targets; it teaches nothing and its gradient still reaches
        # the shared q/c projections.
        #
        # SELF-LINK is the exception to that skip rule. `postLinks` points
        # BOTH `PostId` and `RelatedPostId` at `posts`, so for ent=posts every
        # slot points "back" and the table was skipped -- which is why
        # rel-stack post-post-related could not be trained at all. But a table
        # with TWO slots into the sequence entity's own table is not the
        # identity: the other endpoint is a genuinely different post. Which
        # slot is "the other one" depends on the event (the corpus puts a
        # postLinks row in BOTH endpoints' histories), so it cannot be a fixed
        # slot index -- it is resolved per event in `_seq_destinations`.
        self.who_slot_by_ent = {}
        for ent in self.entity_tables:
            m = {}
            for name, spec in corpus.schema.fact_tables.items():
                targets = list(spec.fkeys.values())
                if not targets:
                    continue
                away = [j for j, t in enumerate(targets) if t != ent]
                if away:
                    m[name] = away[0]
                elif len(targets) >= 2:
                    m[name] = SELF_LINK
            self.who_slot_by_ent[ent] = m
        self.who_slot = self.who_slot_by_ent[self.ent]
        # Popularity for negative sampling, stored as a CUMULATIVE distribution.
        #
        # Do not use rng.choice(..., p=probs): that is O(num_entities) per call
        # because it rebuilds the cumulative distribution every time. Called
        # once per target event it dominated everything -- measured at 12.4
        # s/step on rel-hm (1.37M customers), vs 0.06 s/step on rel-f1 (857
        # drivers). Inverse-transform sampling against a prebuilt CDF is
        # O(log num_entities) per draw.
        self.pop_cdf = {}
        for name in corpus.schema.entity_tables:
            o = corpus.hist_offset[name]
            p = np.diff(o).astype(np.float64) + 1.0
            self.pop_cdf[name] = np.cumsum(p / p.sum())

    def _sample_negatives(self, table: str, n: int) -> np.ndarray:
        half = n // 2
        n_ent = self.c.schema.entity_counts[table]
        uni = self.rng.integers(0, n_ent, size=half)
        cdf = self.pop_cdf[table]
        pop = np.searchsorted(cdf, self.rng.random(n - half))
        np.clip(pop, 0, n_ent - 1, out=pop)
        return np.concatenate([uni, pop])

    def batch(self, device="cpu", entity_table: str | None = None) -> dict:
        cutoff_s = int(self.c.cutoff.timestamp())
        # D1: one side per batch. Mixing sides within a batch would be fine
        # for the backbone but makes `seq_entity` -- which decides the state
        # write rule -- ambiguous.
        ent = entity_table or (
            self.entity_tables[0] if len(self.entity_tables) == 1
            else self.rng.choice(self.entity_tables, p=self.table_probs))
        who_slot = self.who_slot_by_ent[ent]
        self.ent_current = ent

        rows_t, rows_gap, rows_sid = [], [], []
        per_table: dict = {}
        target = dict(b=[], l=[], gap=[], censored=[], table_idx=[],
                      who_tbl=[], who_cands=[], what={},
                      seq_idx=[], k_in_seq=[], q_time=[])
        seqs: list = []      # per-sequence destination history, for D3 feats
        win = dict(b=[], l=[], elapsed=[], horizon=[], feats=[], counts=[],
                   per_table={}, n_capped=0)
        # dense window targets on ordinary event tokens (rate + qcount only)
        wd = dict(b=[], l=[], elapsed=[], horizon=[], feats=[], counts=[])

        for b in range(self.batch_rows):
            t_row, gap_row, sid_row = [], [], []
            sid = 0
            while len(t_row) < self.max_len:
                room = self.max_len - len(t_row)
                # Query tokens occupy slots in the SAME budget. Reserve them
                # up front: overshooting max_len would leave target positions
                # pointing past the row after truncation, which indexes the
                # wrong hidden state rather than raising.
                n_q = (self.n_query if self.window_head
                       and room > self.n_query + 1 else 0)
                budget = room - n_q
                e = self.rng.choice(self.entities[ent])
                ids_all = self.c.history(ent, e)
                ids = ids_all[-min(budget, self.max_len // 2):]
                ts = self.c.time[ids]
                gaps = np.diff(ts, prepend=ts[0])
                s_idx = len(seqs)
                seqs.append(self._seq_destinations(ids, ts, who_slot,
                                                   ent, int(e)))
                queries = (self._sample_queries(ts, cutoff_s, n_q)
                           if n_q else {})
                pos_of_k = np.empty(len(ids), dtype=np.int64)
                for k, ev in enumerate(ids):
                    name = self._table_of(ev)
                    p = len(t_row)
                    pos_of_k[k] = p
                    self._append_event(per_table, name, ev, b, p)
                    t_row.append(ts[k]); gap_row.append(gaps[k])
                    sid_row.append(sid)
                    # target for position k = event k+1 (or censored end)
                    if k + 1 < len(ids):
                        nxt = ids[k + 1]
                        self._append_target(target, b, p,
                                            self.c.time[nxt] - ts[k],
                                            False, nxt, who_slot,
                                            s_idx, k, int(ts[k]))
                    else:
                        self._append_target(target, b, p,
                                            cutoff_s - ts[k], True, None,
                                            who_slot, s_idx, k, int(ts[k]))
                    # query tokens whose query time falls in [t_k, t_{k+1})
                    for t_q, hz in queries.get(k, ()):
                        q = len(t_row)
                        t_row.append(t_q); gap_row.append(t_q - ts[k])
                        sid_row.append(sid)
                        self._append_query(win, b, q, ids, ts, k, t_q, hz,
                                           ids_all)
                if self.dense_window:
                    self._append_dense(wd, b, pos_of_k, ids_all, ts,
                                       cutoff_s)
                sid += 1
            rows_t.append(t_row[:self.max_len])
            rows_gap.append(gap_row[:self.max_len])
            rows_sid.append(sid_row[:self.max_len])

        return self._to_tensors(rows_t, rows_gap, rows_sid, per_table,
                                target, seqs, ent, device, win, wd)

    # -- WindowHead targets ---------------------------------------------

    def _draw_horizon(self, rng=None) -> float:
        """One prediction horizon in SECONDS.

        Log-uniform over [WINDOW_MIN_DAYS, WINDOW_MAX_DAYS] by default, which
        brackets every RelBench timedelta (4d rel-avito .. 365d rel-trial), or
        log-normal around `window_days` when the run is aimed at one task. The
        head is conditioned on the draw either way, so one model answers every
        horizon instead of being implicitly fitted to a single one.
        """
        rng = self.rng if rng is None else rng
        if self.window_days > 0:
            d = float(np.exp(rng.normal(np.log(self.window_days),
                                        self.window_sigma)))
            d = min(max(d, self.WINDOW_MIN_DAYS), self.WINDOW_MAX_DAYS)
        else:
            lo, hi = np.log(self.WINDOW_MIN_DAYS), np.log(self.WINDOW_MAX_DAYS)
            d = float(np.exp(rng.uniform(lo, hi)))
        return d * 86400.0

    def _sample_queries(self, ts, cutoff_s: int, n: int) -> dict:
        """-> {k: [(t_q, horizon_seconds), ...]}, k = last event at or before t_q.

        THE SAMPLING ORDER MATTERS. Drawing t_q uniformly inside the gap
        (t_k, t_{k+1}] would make t_q itself a readout of the next gap -- the
        model could infer "a late t_q means the next event is imminent" and
        the window head would be scored on a leak. So the horizon is drawn
        first, then t_q uniformly over the sequence's whole observed span, and
        k is looked up. That distribution is independent of the gaps given the
        span, and it naturally over-samples states with long silences, which
        is exactly the population a task cutoff grid queries.

        `t_q + D <= corpus cutoff` is required, so a window is never partially
        observed; that is the leakage rule, and it is why some draws are
        rejected rather than clipped.
        """
        out: dict = {}
        t0 = int(ts[0])
        for _ in range(n):
            hz = self._draw_horizon()
            hi = int(cutoff_s - hz)     # truncate BEFORE the comparison: a
            if hi <= t0:                # float `hi` a fraction of a second
                continue                # above t0 gives an empty integer range
            if self.q_tail > 0 and self.rng.random() < self.q_tail:
                # Every EVAL query sits at the split boundary: rel-stack's and
                # rel-event's val/test tables are a single timestamp equal to
                # it. A t_q drawn uniformly over an entity's whole span puts
                # most of its mass years earlier, on a platform with a
                # different activity level, and the head then under-predicts
                # at the only time it is ever asked about. This draws a share
                # of query times log-uniformly back from the latest admissible
                # one instead. It uses no labels -- only the corpus cutoff,
                # which the leakage rule already fixes.
                age = float(np.exp(self.rng.uniform(
                    np.log(86400.0), np.log(max(hi - t0, 86400.0) + 1.0))))
                t_q = int(max(t0, hi - age))
            else:
                t_q = int(self.rng.integers(t0, hi))
            k = int(np.searchsorted(ts, t_q, side="right")) - 1
            if k < 0:
                continue
            out.setdefault(k, []).append((t_q, hz))
        # Two queries can share a k -- every t_q past the entity's last event
        # lands on the final index -- and they are inserted in sampling order,
        # so without this sort the token stream stops being time-ordered and a
        # query token attends to a LATER query token. Nothing about that is
        # fatal (query tokens carry no event content, so what leaks is a
        # random draw rather than data) but it feeds the TimeEncoder a
        # backwards gap and breaks the invariant the tests check.
        for v in out.values():
            v.sort(key=lambda x: x[0])
        return out

    N_QUERY_FEATS_PER_TABLE = 3

    # -- the classification/regression feature block (2026-08-26) -----------
    #
    # WHY A SECOND BLOCK. The features above answer "how much has this entity
    # done, and how long ago". Reading all 34 classification tasks in
    # `relbench_datasets_tasks.xlsx` shows the entity-level ones ask three
    # questions that those cannot express:
    #
    #   1. "will it go quiet"   -- rel-hm/rel-amazon/rel-ratebeer user-churn,
    #      item-churn, beer-churn, brewer-dormant, rel-stack user-engagement,
    #      rel-arxiv paper-citation. What separates a lapsed entity from a
    #      merely slow one is elapsed silence RELATIVE to its own usual gap,
    #      not either quantity alone. A customer who buys yearly and last
    #      bought 11 months ago is healthy; a daily buyer silent for a month
    #      is gone. The model gets `elapsed` and `count` and has to discover
    #      the ratio through a nonlinearity; features 5-6 hand it over.
    #
    #   2. "more than k in the window" -- rel-avito user-visits and
    #      user-clicks (>1 ad in 4 days), rel-event user-ignore (>2 invites
    #      in 7 days). A rate alone does not determine a tail probability:
    #      the same mean with bursty arrivals gives a much larger P(N > k).
    #      Features 0-4 are a nonparametric estimate of the count
    #      distribution over past windows OF THE TASK'S OWN LENGTH -- the
    #      empirical answer to the question the label asks, computed from
    #      pre-query events only. `frac >= 2` in particular IS the historical
    #      base rate of the rel-avito labels.
    #
    #   3. "will the next one be of a particular kind" -- rel-stack
    #      user-badge, rel-event user-repeat, rel-f1 driver-dnf and
    #      driver-top3. The existing per-category counters are ALL-TIME, so a
    #      linear layer can form a lifetime yes-fraction but not a recent
    #      one, and these labels are about current form. Hence the recent
    #      per-category block, which is the same layout restricted to the
    #      trailing window.
    #
    # These are the same move as the D3 interaction features that tripled
    # recommendation MAP (RESEARCH.md 2026-08-23): give the head the cheap
    # exact facts instead of asking a compressed hidden state to have
    # retained them. The 2026-08-25 probe measured exactly that failure for
    # classification -- 33 hand-rolled count features beat the backbone state
    # 84.8 to 74.2 AUROC, and adding the state made them WORSE.
    #
    # Every one is computed from events strictly before `t_q`, so the block
    # uses no labels and cannot leak.
    N_DYN_FEATS_PER_TABLE = 8
    N_DYN_GLOBAL = 3
    #: past windows of length `hz` used for the empirical count distribution.
    #: Each costs two binary searches, so this is a handful of microseconds.
    DYN_ANCHORS = 8
    #: gap statistics use at most this many recent gaps. Bounds the cost on
    #: entities with enormous histories, and recent cadence is the better
    #: predictor of near-future cadence anyway.
    DYN_GAP_TAIL = 512

    #: The 6 per-slot neighbour features. See `nbr_feats`.
    N_NBR_FEATS_PER_SLOT = 6

    @staticmethod
    def n_query_feats(schema, dyn: bool = False, nbr: bool = False) -> int:
        """Width of the query-token feature vector for this schema.

        The blocks are APPEND-ONLY and in flag order (base, per-category,
        dynamics, neighbour). A checkpoint trained without a later block has a
        narrower `query_feat_proj` and must keep loading, which is why a new
        block may never be inserted in the middle -- HANDOFF 10.1b, the rule
        the per-category block had to be taught the hard way.
        """
        n = (PackedBatcher.N_QUERY_FEATS_PER_TABLE
             * len(schema.fact_tables) + 2
             + sum(card for _, _, _, card in query_cat_layout(schema)))
        if dyn:
            n += (PackedBatcher.N_DYN_FEATS_PER_TABLE
                  * len(schema.fact_tables)
                  + PackedBatcher.N_DYN_GLOBAL
                  + sum(card for _, _, _, card in query_cat_layout(schema)))
        if nbr:
            n += (PackedBatcher.N_NBR_FEATS_PER_SLOT
                  * len(query_nbr_layout(schema)))
        return n

    @staticmethod
    def nbr_feats(corpus, ids_p, ti, t_q, out, off) -> None:
        """Neighbour aggregates for one query. Writes 6 floats per slot.

        `ids_p` are the entity's event ids strictly before `t_q` and `ti`
        their table indices, i.e. exactly what `query_feats` already has in
        hand; nothing here reads an event at or after `t_q`, so the block is
        leak-free by the same construction as the counters around it.

        Per (fact table T, foreign-key slot s) of `query_nbr_layout`, over the
        entity's own pre-`t_q` events of T:

            0 log1p(number of DISTINCT neighbours attached through s)
            1 top-1 concentration -- the share of the entity's T-events that
              went to its single most frequent neighbour. Separates "one
              long-standing partner" from "many one-off ones", which is the
              difference between a works driver and a journeyman.
            2 mean over the pooled neighbours of log1p(their TOTAL event
              count before t_q)      -- how established the neighbour is
            3 mean over the pooled neighbours of log1p(their count of
              T-events before t_q)   -- the 2-hop aggregate proper: for
              driver-dnf this is the constructor's own race-results volume
            4 mean over the pooled neighbours of log1p(days since their last
              event before t_q)      -- is the neighbour still active
            5 log1p(events the entity sent to the MOST RECENT neighbour),
              the "how invested in the current partner" counterpart to 1

        Only the `QFEAT_NBR_K` most recently attached distinct neighbours are
        pooled. An entity with thousands of neighbours would otherwise cost a
        history lookup for each, and recent attachment is the better
        predictor of the near future anyway -- the same bound, and the same
        argument, as `DYN_GAP_TAIL`.
        """
        n_per = PackedBatcher.N_NBR_FEATS_PER_SLOT
        day = 86400.0
        for name, tidx, slot, tgt in query_nbr_layout(corpus.schema):
            o, off = off, off + n_per
            m = np.flatnonzero(ti == tidx)
            if not len(m):
                continue
            rows = corpus.row_of[name][ids_p[m]]
            ent = corpus.links[name][rows, slot]
            ent = ent[ent >= 0]                    # NO_ENTITY
            if not len(ent):
                continue
            uniq, counts = np.unique(ent, return_counts=True)
            out[o] = np.log1p(len(uniq))
            out[o + 1] = counts.max() / len(ent)
            # most recently attached distinct neighbours: `ent` follows the
            # entity's own event order, which is ascending in time
            seen, recent = set(), []
            for e in ent[::-1]:
                e = int(e)
                if e not in seen:
                    seen.add(e)
                    recent.append(e)
                    if len(recent) >= QFEAT_NBR_K:
                        break
            out[o + 5] = np.log1p(int(counts[uniq == recent[0]][0]))
            tot = np.zeros(3, dtype=np.float64)
            for e in recent:
                h = corpus.history(tgt, e, before=int(t_q))
                if not len(h):
                    continue
                tot[0] += np.log1p(len(h))
                tot[1] += np.log1p(int((corpus.table_idx[h] == tidx).sum()))
                tot[2] += np.log1p(max(t_q - int(corpus.time[h[-1]]), 0) / day)
            out[o + 2:o + 5] = tot / len(recent)

    @staticmethod
    def _dyn_table_feats(t_m, t_q, hz, out):
        """The 8 per-table dynamics features. `t_m` is that table's
        pre-query timestamps, ascending. Writes into `out` (length 8).

        0 log1p(max count in a past window of length hz)
        1 fraction of past windows with >= 1 event
        2 fraction of past windows with >= 2 events
        3 log1p(mean count per window)
        4 dispersion of the window counts (std/mean; 1.0 under Poisson)
        5 log1p(median gap in days)
        6 log1p(elapsed / median gap)   -- "how overdue", the churn signal
        7 trend: log1p(count in the last window) - log1p(the one before)
        """
        n = len(t_m)
        if n == 0 or hz <= 0:
            return
        day = 86400.0
        k = PackedBatcher.DYN_ANCHORS
        # Windows (a_i - hz, a_i] for a_i = t_q - i*hz, i = 0 .. n_win-1.
        # Window 0 is the immediate past, the exact mirror of the future
        # window the label asks about.
        #
        # `n_win` is bounded by how long the entity has EXISTED, so a
        # two-week-old entity is not credited with six empty months of "no
        # activity" -- young entities are most of several eval populations
        # (80.3% of rel-avito ads have under two events), and that bias would
        # land squarely on them. The bound is a ceiling rather than a floor:
        # the oldest window is then partially observed and slightly
        # undercounts, which is the cheaper error. Flooring it drops to a
        # single window for any entity whose history is under two horizons
        # long, and with one window there is no previous window to compare
        # against, so the TREND feature silently disappears for exactly the
        # entities whose trend is most informative.
        span = int(t_q - t_m[0])
        hzi = max(int(hz), 1)
        n_win = int(min(k, max(1, -(-span // hzi))))
        hi = t_q - np.arange(n_win, dtype=np.int64) * int(hz)
        lo = hi - int(hz)
        cnt = (np.searchsorted(t_m, hi, side="right")
               - np.searchsorted(t_m, lo, side="right")).astype(np.float64)
        out[0] = np.log1p(cnt.max())
        out[1] = float((cnt >= 1).mean())
        out[2] = float((cnt >= 2).mean())
        mean = cnt.mean()
        out[3] = np.log1p(mean)
        out[4] = float(cnt.std() / mean) if mean > 0 else 0.0
        if n >= 2:
            g = np.diff(t_m[-PackedBatcher.DYN_GAP_TAIL:]).astype(np.float64)
            med = float(np.median(g)) if len(g) else 0.0
            out[5] = np.log1p(med / day)
            # `+ day` keeps a burst of same-second events (median gap 0) from
            # producing an infinite ratio; the floor is one day, which is
            # below every task horizon in the benchmark.
            out[6] = np.log1p((t_q - t_m[-1]) / (med + day))
        else:
            out[6] = np.log1p((t_q - t_m[-1]) / day)
        if n_win >= 2:
            out[7] = np.log1p(cnt[0]) - np.log1p(cnt[1])

    def query_feats(self, ids_all, t_q, hz) -> np.ndarray:
        """Per-table history counters at the query time. Leak-free by
        construction: everything is computed from events strictly before t_q.

        Why the model is handed these rather than left to infer them. A frozen
        probe on 2026-08-25 found the backbone state carried LESS
        task-relevant signal on rel-stack user-badge than two scalars
        (n_events, elapsed), and less than 33 hand-rolled count/recency
        features (74.2 vs 81.5 vs 84.8 test AUROC). The top features were
        per-type recency and per-type totals. Attention over 512 tokens can in
        principle compute a running count, but nothing in a next-event
        objective rewards keeping one, and the window objective only started
        rewarding it today. Giving the query token the counters directly is
        the same move as giving a GNN its degree features, it is schema
        generic (one block per fact table, no per-dataset code), and it uses
        no labels.

        Layout, per fact table T in schema order:
            log1p(count of T before t_q)
            log1p(days since the last T event)      -- 0 if never
            log1p(count of T in the trailing window of length hz)
        then two globals: log1p(total count), log1p(days since the FIRST event
        of any kind), i.e. the entity's age. Then, appended, the per-CATEGORY
        block of `query_cat_layout`.

        WHY THE PER-CATEGORY BLOCK (2026-08-25). Per-table counts answer "how
        active", never "what kind". rel-event user-repeat is the label whose
        whole signal is the second question: the historical yes-fraction alone
        scores 70.31 AUROC while the event COUNT scores 41.43, i.e. the
        counters the query token already had are ANTI-correlated with it. The
        fraction is `log1p(n_yes) - log1p(n_total)`, a difference of two
        entries of this vector, so a linear layer can form it -- which it
        cannot do from per-table counts at all. Same shape for user-badge
        (badge class) and user-ignore (invitation status).

        Zero-initialised in the model like the rest of the block, so an arm
        with the wider vector starts numerically identical to one without.
        """
        n_tbl = len(self.c.schema.fact_tables)
        lay = query_cat_layout(self.c.schema)
        dyn = bool(getattr(self, "query_dyn", False))
        nbr = bool(getattr(self, "query_nbr", False))
        base = self.N_QUERY_FEATS_PER_TABLE * n_tbl + 2
        out = np.zeros(self.n_query_feats(self.c.schema, dyn, nbr),
                       dtype=np.float32)
        # start of the neighbour block: after everything the dyn flag adds,
        # so the two flags are independent suffixes
        nbr_o = self.n_query_feats(self.c.schema, dyn)
        # start of the dynamics block; the layout before it is unchanged, so
        # a run with --query_dyn differs from its control only by a suffix
        dyn_o = base + sum(c for _, _, _, c in lay)
        if len(ids_all) == 0:
            return out
        t_all = self.c.time[ids_all]
        cut = np.searchsorted(t_all, t_q, side="right")
        if cut == 0:
            return out
        ids_p = ids_all[:cut]
        t_p = t_all[:cut]
        ti = self.c.table_idx[ids_p]
        cnt = np.bincount(ti, minlength=n_tbl)
        back = np.searchsorted(t_p, t_q - hz, side="right")
        cnt_w = np.bincount(ti[back:], minlength=n_tbl)
        day = 86400.0
        for j in range(n_tbl):
            m = np.flatnonzero(ti == j)
            o = self.N_QUERY_FEATS_PER_TABLE * j
            out[o] = np.log1p(cnt[j])
            out[o + 1] = (np.log1p((t_q - t_p[m[-1]]) / day)
                          if len(m) else 0.0)
            out[o + 2] = np.log1p(cnt_w[j])
        out[base - 2] = np.log1p(cut)
        out[base - 1] = np.log1p(max(t_q - t_p[0], 0) / day)
        o = base
        for name, tidx, jcol, card in lay:
            m = np.flatnonzero(ti == tidx)
            if len(m):
                rows = self.c.row_of[name][ids_p[m]]
                codes = np.clip(self.c.feat_cat[name][rows, jcol], 0, card - 1)
                out[o:o + card] = np.log1p(
                    np.bincount(codes, minlength=card))
            o += card
        if nbr:
            self.nbr_feats(self.c, ids_p, ti, t_q, out, nbr_o)
        if not dyn:
            return out

        # -- dynamics block -------------------------------------------------
        day = 86400.0
        o = dyn_o
        for j in range(n_tbl):
            m = np.flatnonzero(ti == j)
            self._dyn_table_feats(t_p[m], t_q, hz,
                                  out[o:o + self.N_DYN_FEATS_PER_TABLE])
            o += self.N_DYN_FEATS_PER_TABLE
        # globals: cadence regularity, weekday match, and the all-table
        # overdue ratio (the churn signal for a single-fact-table database,
        # where the per-table version above is the only other place it
        # appears).
        if cut >= 2:
            g = np.diff(t_p[-self.DYN_GAP_TAIL:]).astype(np.float64)
            lg = np.log1p(np.maximum(g, 0.0) / day)
            out[o] = float(lg.std())
            med = float(np.median(g))
            out[o + 2] = np.log1p((t_q - t_p[-1]) / (med + day))
        # 1970-01-01 was a Thursday; the same arithmetic the TimeEncoder uses.
        # A weekly shopper queried on "their" weekday is a different
        # proposition from one queried on any other, and rel-hm user-churn
        # (7 days) and rel-avito user-visits (4 days) both live at that scale.
        dow_q = ((t_q // 86400) + 4) % 7
        out[o + 1] = float((((t_p // 86400) + 4) % 7 == dow_q).mean())
        o += self.N_DYN_GLOBAL
        # -- recent per-category counters, same layout as the all-time block
        back_c = np.searchsorted(t_p, t_q - hz, side="right")
        ti_r, ids_r = ti[back_c:], ids_p[back_c:]
        for name, tidx, jcol, card in lay:
            m = np.flatnonzero(ti_r == tidx)
            if len(m):
                rows = self.c.row_of[name][ids_r[m]]
                codes = np.clip(self.c.feat_cat[name][rows, jcol], 0, card - 1)
                out[o:o + card] = np.log1p(
                    np.bincount(codes, minlength=card))
            o += card
        return out

    def _sample_dense(self, ts, cutoff_s: int, n: int):
        """Dense query offsets for event tokens. -> (k [Q], t_q [Q], hz [Q]).

        `k` indexes `ts`, i.e. the token whose hidden state answers the query.
        Positions are drawn WITHOUT replacement so one token is not asked the
        same kind of question twice in a batch.

        The offset `d = t_q - t_k` is drawn LOG-UNIFORMLY over the admissible
        range rather than set to zero. Zero would be the maximal-information
        choice, but it would also train the head almost exclusively at
        `elapsed = 0` while every eval query has `elapsed > 0` -- a train/eval
        mismatch on the head's own conditioning variable. Log-uniform keeps
        the mass on fresh states while covering long silences, which is the
        same argument `q_tail` makes for the sparse query times.

        `t_q + D <= cutoff` is enforced, so a dense window is never partially
        observed; draws that cannot satisfy it are dropped and counted.
        """
        n_tok = len(ts)
        if n_tok == 0 or n <= 0:
            z = np.zeros(0, dtype=np.int64)
            return z, z, np.zeros(0, dtype=np.float64)
        rng = self.rng_dense
        k = (rng.permutation(n_tok)[:n] if n < n_tok
             else rng.permutation(n_tok))
        hz = np.array([self._draw_horizon(rng) for _ in range(len(k))])
        t_k = ts[k].astype(np.int64)
        hi = (cutoff_s - hz).astype(np.int64)      # latest admissible t_q
        ok = hi > t_k
        self.n_targets_seen += len(k)
        self.n_targets_dropped += int((~ok).sum())
        k, hz, t_k, hi = k[ok], hz[ok], t_k[ok], hi[ok]
        if not len(k):
            z = np.zeros(0, dtype=np.int64)
            return z, z, np.zeros(0, dtype=np.float64)
        span = (hi - t_k).astype(np.float64)
        age = np.exp(rng.uniform(np.log(3600.0),
                                 np.log(span + 3600.0)))
        t_q = np.minimum(t_k + (age - 3600.0).astype(np.int64), hi)
        return k, t_q, hz

    def _dense_stats(self, ids_all, t_q, hz):
        """Vectorised `query_feats` + exact window counts for many t_q at once.

        -> (feats [Q, n_query_feats], counts [Q, n_fact_tables])

        WHY THIS IS NOT A LOOP OVER `query_feats`. That function is O(len
        history) per call -- it bincounts the whole prefix. Calling it once
        per dense token would make batch construction quadratic in the
        sequence length, and dense supervision exists precisely to raise the
        number of tokens asked.

        Instead: every query needs running statistics at three cut points
        (t_q - D, t_q, t_q + D). Collect all of them, sort, and sweep the
        history ONCE, accumulating between consecutive marks with numpy and
        snapshotting the running totals. Cost is O(len history + Q * marks)
        with no [len history, width] array ever materialised, so memory is
        proportional to the OUTPUT rather than to the entity's activity --
        which matters, because the heaviest entities are exactly the ones
        with the longest histories (HANDOFF section 9.2: host RAM is the
        binding constraint, not GPU memory).

        `test_dense_feats_match_query_feats_exactly` pins this against the
        scalar implementation; the two must not be allowed to drift, because
        a divergence would train the GLM branch of the wide-and-deep head on
        one feature definition and evaluate it on another.
        """
        c = self.c
        n_tbl = len(c.schema.fact_tables)
        lay = query_cat_layout(c.schema)
        W = sum(card for _, _, _, card in lay)
        base = self.N_QUERY_FEATS_PER_TABLE * n_tbl + 2
        Q = len(t_q)
        feats = np.zeros((Q, base + W), dtype=np.float32)
        counts = np.zeros((Q, n_tbl), dtype=np.float32)
        if Q == 0 or len(ids_all) == 0:
            return feats, counts

        t_all = c.time[ids_all]
        ti_all = c.table_idx[ids_all]
        # per-event slot in the flattened per-category block, -1 = not covered
        cat_slot = np.full(len(ids_all), -1, dtype=np.int64)
        off = 0
        for name, tidx, jcol, card in lay:
            m = np.flatnonzero(ti_all == tidx)
            if len(m):
                rows = c.row_of[name][ids_all[m]]
                codes = np.clip(c.feat_cat[name][rows, jcol], 0, card - 1)
                cat_slot[m] = off + codes
            off += card

        i_cut = np.searchsorted(t_all, t_q, side="right")
        i_back = np.searchsorted(t_all, t_q - hz.astype(np.int64),
                                 side="right")
        i_end = np.searchsorted(t_all, t_q + hz.astype(np.int64),
                                side="right")
        marks = np.unique(np.concatenate([i_cut, i_back, i_end]))

        cnt_at = np.zeros((len(marks), n_tbl), dtype=np.int64)
        # -inf, NOT -1: rel-f1 races start in 1950, so a legitimate event
        # time is NEGATIVE unix seconds and any finite sentinel is a value the
        # data can actually take. A -1 sentinel here reported "never seen" for
        # every pre-1970 event and silently zeroed the recency features on the
        # one dataset whose history reaches back that far.
        last_at = np.full((len(marks), n_tbl), -np.inf, dtype=np.float64)
        cat_at = np.zeros((len(marks), W), dtype=np.int64) if W else None
        run_cnt = np.zeros(n_tbl, dtype=np.int64)
        run_last = np.full(n_tbl, -np.inf, dtype=np.float64)
        run_cat = np.zeros(W, dtype=np.int64) if W else None
        prev = 0
        for mi, m in enumerate(marks):
            m = int(m)
            if m > prev:
                seg_ti = ti_all[prev:m]
                seg_t = t_all[prev:m]
                run_cnt += np.bincount(seg_ti, minlength=n_tbl)
                for j in np.unique(seg_ti):
                    run_last[j] = seg_t[seg_ti == j][-1]
                if W:
                    cs = cat_slot[prev:m]
                    cs = cs[cs >= 0]
                    if len(cs):
                        run_cat += np.bincount(cs, minlength=W)
                prev = m
            cnt_at[mi] = run_cnt
            last_at[mi] = run_last
            if W:
                cat_at[mi] = run_cat

        j_cut = np.searchsorted(marks, i_cut)
        j_back = np.searchsorted(marks, i_back)
        j_end = np.searchsorted(marks, i_end)

        counts[:] = (cnt_at[j_end] - cnt_at[j_cut]).astype(np.float32)

        day = 86400.0
        c_cut, c_back = cnt_at[j_cut], cnt_at[j_back]
        l_cut = last_at[j_cut]
        tq = t_q.astype(np.float64)
        for j in range(n_tbl):
            o = self.N_QUERY_FEATS_PER_TABLE * j
            feats[:, o] = np.log1p(c_cut[:, j])
            seen = np.isfinite(l_cut[:, j])
            feats[seen, o + 1] = np.log1p(
                (tq[seen] - l_cut[seen, j]) / day)
            feats[:, o + 2] = np.log1p(c_cut[:, j] - c_back[:, j])
        feats[:, base - 2] = np.log1p(i_cut)
        feats[:, base - 1] = np.log1p(
            np.maximum(tq - float(t_all[0]), 0.0) / day)
        if W:
            feats[:, base:] = np.log1p(cat_at[j_cut])
        # `query_feats` returns an all-zero vector when no event precedes t_q;
        # match that exactly rather than emitting log1p(0) noise in the
        # globals block.
        feats[i_cut == 0] = 0.0
        return feats, counts

    def _append_query(self, win, b, l, ids, ts, k, t_q, hz, ids_all=None):
        qi = len(win["b"])
        win["b"].append(b); win["l"].append(l)
        win["elapsed"].append(float(t_q - ts[k]))
        win["horizon"].append(float(hz))
        win["feats"].append(
            self.query_feats(ids if ids_all is None else ids_all, t_q, hz))
        n_tbl = len(self.c.schema.fact_tables)
        lo = int(np.searchsorted(ts, t_q, side="right"))
        hi = int(np.searchsorted(ts, t_q + hz, side="right"))
        sel = ids[lo:hi]

        # The EXACT per-table counts, before any capping. These are the target
        # for the total rate and the count quantiles, i.e. for every
        # classification and count-regression readout.
        #
        # BUG FIXED 2026-08-25: this used to be derived from the CAPPED event
        # list, so an article with 3,000 transactions in a 365-day window
        # taught the head a rate of 256. It surfaced as a window loss of
        # -13.67 on rel-hm -- the Poisson NLL's minimum is `t - t log t`, so
        # large counts drive it far negative and the number looked like a
        # divergence rather than what it was. Every heavy-tailed dataset was
        # affected: rel-hm, rel-amazon, rel-avito.
        exact = np.bincount(self.c.table_idx[sel], minlength=n_tbl) if len(sel) \
            else np.zeros(n_tbl, dtype=np.int64)
        win["counts"].append(exact.astype(np.float32))

        if len(sel) > self.win_max_events:
            # Per-column DETAIL (category and numeric-bucket rates, column
            # sums) is capped for memory. A RANDOM subsample keeps the scaled
            # detail an unbiased estimate of the full window; taking the first
            # k would bias it toward the start of the window.
            win["n_capped"] += 1
            keep = self.rng.choice(len(sel), self.win_max_events,
                                   replace=False)
            sel = sel[np.sort(keep)]
        if len(sel) == 0:
            return
        tis = self.c.table_idx[sel]
        kept = np.bincount(tis, minlength=n_tbl)
        for ti in np.unique(tis):
            name = self._code_to_table(int(ti))
            m = np.flatnonzero(tis == ti)
            d = win["per_table"].setdefault(name, dict(qi=[], rows=[], w=[]))
            d["qi"].append(np.full(len(m), qi, dtype=np.int64))
            d["rows"].append(self.c.row_of[name][sel[m]])
            # weight restoring the detail to the exact count
            d["w"].append(np.full(len(m),
                                  exact[ti] / max(kept[ti], 1),
                                  dtype=np.float32))

    def _append_dense(self, wd, b, pos_of_k, ids_all, ts, cutoff_s):
        """Dense window targets for one packed sequence.

        `pos_of_k[k]` is the ROW position of the sequence's k-th event, which
        is what the loss must index -- `k` alone is meaningless once several
        sequences and their query tokens share a row.

        Positions at or past `max_len` are dropped: `batch()` truncates each
        row to that width afterwards, and a target pointing past the end
        would silently read a DIFFERENT entity's hidden state rather than
        raise (the same class of defect as the reserved-query-slot fix above).
        """
        k, t_q, hz = self._sample_dense(ts, cutoff_s, self.n_dense)
        if not len(k):
            return
        pos = pos_of_k[k]
        keep = pos < self.max_len
        if not keep.all():
            k, t_q, hz, pos = k[keep], t_q[keep], hz[keep], pos[keep]
            if not len(k):
                return
        feats, counts = self._dense_stats(ids_all, t_q, hz)
        wd["b"].append(np.full(len(k), b, dtype=np.int64))
        wd["l"].append(pos)
        wd["elapsed"].append((t_q - ts[k].astype(np.int64)).astype(np.float32))
        wd["horizon"].append(hz.astype(np.float32))
        wd["feats"].append(feats)
        wd["counts"].append(counts)

    def _code_to_table(self, ti: int) -> str:
        cache = self.__dict__.setdefault("_name_of_idx", None)
        if cache is None:
            cache = {s.table_idx: n
                     for n, s in self.c.schema.fact_tables.items()}
            self._name_of_idx = cache
        return cache[ti]

    def _seq_destinations(self, ids, ts, who_slot, ent, ent_id) -> dict:
        """The who-destination(s) of each event in one sequence.

        Feeds the D3 repeat/recency features: "has this entity already linked
        to candidate c, how often, how recently". Stored per sequence so the
        features can be computed for ALL candidates (true and negative)
        identically -- computing them only for the true candidate would let
        the model read the answer off the feature.

        Arrays are indexed by (event, destination) PAIR, not by event. With
        path retargeting one event yields several destinations (a study has
        ~1.6 sponsors), so `t` repeats the event's timestamp and `k` records
        which event a pair came from. Without retargeting there is exactly one
        pair per event and `k` is the identity, so every consumer behaves
        as before.
        """
        # dst_code is an INTEGER code per position, not a list of table names.
        # The name list forced a Python comparison per (target, position) in
        # the feature loops; on rel-hm that alone took batching from 131 ms to
        # 1757 ms. Integer codes keep the selection in numpy.
        codes = self.__dict__.setdefault("_tbl_code", {})
        rt = self.retarget
        d_dst, d_code, d_t, d_k = [], [], [], []
        for k, ev in enumerate(ids):
            name = self._table_of(ev)
            spec = self.c.schema.fact_tables[name]
            targets = list(spec.fkeys.values())
            if not targets or name not in who_slot:
                continue                # no informative who-slot: skip
            j = who_slot[name]
            row = self.c.row_of[name][ev]
            if j == SELF_LINK:
                # Every slot targets `ent`; the destination is the endpoint
                # that is not this sequence's own entity. A genuine self-loop
                # (both slots equal ent_id) carries no information and is left
                # out rather than supervising the identity function.
                vals = self.c.links[name][row]
                other = [v for v in vals if v != ent_id and v != NO_ENTITY]
                if not other:
                    continue
                d_dst.append(int(other[0]))
                d_code.append(codes.setdefault(ent, len(codes)))
                d_t.append(ts[k]); d_k.append(k)
                continue
            tgt = targets[j]
            v = int(self.c.links[name][row, j])
            if v < 0:
                continue
            if rt is not None and tgt == rt["inter"]:
                # PATH RETARGETING. The WHO target becomes the entity at the
                # end of the declared foreign-key path, not the direct
                # neighbour. The events themselves are untouched -- only the
                # target is remapped -- so WHEN/WHERE/WHAT still see the real
                # event stream. This is what puts the retrieval problem in the
                # space where repeats exist: a condition never links to the
                # same STUDY twice (0.00% corpus repeat) while 28% of the task
                # answers are SPONSORS it has already worked with.
                lo, hi = rt["off"][v], rt["off"][v + 1]
                if hi <= lo:
                    continue
                fin = rt["val"][lo:hi]
                d_dst.extend(int(x) for x in fin)
                c_f = codes.setdefault(rt["final"], len(codes))
                d_code.extend([c_f] * len(fin))
                d_t.extend([ts[k]] * len(fin)); d_k.extend([k] * len(fin))
                continue
            d_dst.append(v)
            d_code.append(codes.setdefault(tgt, len(codes)))
            d_t.append(ts[k]); d_k.append(k)
        n = len(ids)
        return {"dst_code": np.asarray(d_code, dtype=np.int16),
                "dst": np.asarray(d_dst, dtype=np.int64),
                "t": np.asarray(d_t, dtype=np.int64),
                "k": np.asarray(d_k, dtype=np.int64),
                "n_ev": n}

    # -- helpers ----------------------------------------------------------
    def _table_of(self, ev: int) -> str:
        ti = self.c.table_idx[ev]
        for name, spec in self.c.schema.fact_tables.items():
            if spec.table_idx == ti:
                return name
        raise KeyError(ti)

    def _who_table(self, ev: int) -> str | None:
        name = self._table_of(ev)
        spec = self.c.schema.fact_tables[name]
        targets = list(spec.fkeys.values())
        if not targets or name not in self.who_slot:
            return None
        j = self.who_slot[name]
        return self.ent if j == SELF_LINK else targets[j]

    def _append_event(self, per_table, name, ev, b, l):
        d = per_table.setdefault(name, dict(b=[], l=[], rows=[]))
        d["b"].append(b); d["l"].append(l)
        d["rows"].append(self.c.row_of[name][ev])

    def _append_target(self, target, b, l, gap, censored, nxt,
                       who_slot, seq_idx, k_in_seq, q_time):
        target["b"].append(b); target["l"].append(l)
        target["gap"].append(max(float(gap), 1.0))
        target["censored"].append(censored)
        target["seq_idx"].append(seq_idx)
        target["k_in_seq"].append(k_in_seq)
        target["q_time"].append(q_time)
        if censored:
            target["table_idx"].append(0)
            target["who_tbl"].append(None)
            target["who_cands"].append(None)
        else:
            target["table_idx"].append(int(self.c.table_idx[nxt]))
            name = self._table_of(nxt)
            spec = self.c.schema.fact_tables[name]
            row = self.c.row_of[name][nxt]
            # WHO targets are no longer built here: they are window-based and
            # multi-positive, so they are derived from the whole sequence in
            # `_build_who` after the batch is assembled. WHEN/WHERE/WHAT stay
            # next-event.
            if self.next_event:
                w = target["what"].setdefault(
                    name, dict(pos=[], feat_num=[], feat_cat=[]))
                w["pos"].append(len(target["b"]) - 1)
                w["feat_num"].append(self.c.feat_num[name][row])
                w["feat_cat"].append(self.c.feat_cat[name][row])

    # -- D3: interaction features between a query and a candidate ---------
    #
    # The WHO score is a pure content inner product, so it has no way to say
    # "this customer already bought that article" -- which is the whole of the
    # signal in the `user_hist` heuristic that outscores LEDGER 2.6x. These four
    # features give the head that vocabulary. Order is fixed and shared with
    # queries.py:
    #
    #   0  is_repeat        candidate appears earlier in this entity's history
    #   1  log1p(count)     how many times
    #   2  recency          exp(-days_since_last / TAU_DAYS), 0 if never
    #   3  log1p(total)     candidate's events strictly before the query time
    #   4  log1p(recent)    ... within the last TREND_WINDOW_DAYS
    #   5  trend            log1p(recent) - log1p(prior windows), "heating up"
    #   6  staleness        exp(-days_since_candidate_last_active / TREND_TAU)
    #   7  log1p(cooc)      co-occurrence with this entity's recent links
    #
    # ALL of them are computed strictly BEFORE the query position/time, for
    # true and negative candidates alike. Feature 3 replaces v1's global
    # popularity: that was a whole-corpus aggregate, which during training
    # could see counts from later in the corpus than the target. The
    # time-indexed version is both leak-free and more informative.
    #
    # 4-6 are the "recent trend" family: what a candidate's popularity is
    # doing NOW rather than over all time. On rel-stack global popularity is
    # worse than useless (GlobPop 0.03) while recency-of-activity plausibly
    # dominates -- a user comments on live posts.
    #
    # 7 is the 2-hop term: how many other source entities linked to both the
    # candidate and something this entity recently linked to. This is the
    # src->dst->src->dst path that ID-GNN's second hop walks, expressed as a
    # feature so we keep full-catalogue recall instead of restricting to a
    # sampled neighbourhood.

    N_WHO_FEATS = 8
    # Prediction horizons seen during training, log-uniform in days. Chosen to
    # bracket every RelBench link task: rel-hm 7d, rel-stack 91d, rel-trial
    # 365d. The head is conditioned on the draw, so one model covers all of
    # them rather than being implicitly fitted to "until the next event".
    WINDOW_MIN_DAYS = 1.0
    WINDOW_MAX_DAYS = 365.0
    HIST_R = 32                 # history tokens the reranker attends over
    TAU_DAYS = 7.0
    TREND_WINDOW_DAYS = 7.0
    TREND_PREV_WINDOWS = 3.0
    TREND_TAU_DAYS = 14.0
    COOC_RECENT_ITEMS = 8      # how many of the entity's last links feed cooc

    P_MAX = 8          # positives kept per target (window sets can be long)

    def _code_name(self, code: int) -> str:
        for k, v in self.__dict__.get("_tbl_code", {}).items():
            if v == code:
                return k
        raise KeyError(code)

    def _build_who(self, seqs, target) -> dict:
        """Window-based, multi-positive WHO targets, grouped by dst table.

        For a query at position k and time t we supervise on EVERY distinct
        destination the entity links to in (t, t+W]. W is drawn per target
        (log-uniform over WINDOW_MIN/MAX_DAYS) and handed to the head, so the
        model learns horizon-conditioned retrieval rather than a single
        implicit "until the next event" horizon -- which matches no RelBench
        task, whose timedeltas span 7 to 365 days.

        Targets whose window extends past the corpus cutoff are DROPPED: their
        positive set is truncated by the corpus boundary, not by the data, and
        keeping them would teach the model that late histories go quiet.
        """
        n = len(target["b"])
        seq_idx = np.asarray(target["seq_idx"])
        k_in = np.asarray(target["k_in_seq"])
        q_t = np.asarray(target["q_time"])
        cens = np.asarray(target["censored"], dtype=bool)
        cutoff_s = int(self.c.cutoff.timestamp())

        if self.window:
            if self.window_days > 0:
                # Concentrate the horizon prior on the horizon the metric
                # actually asks about. v1 drew log-uniform over 1-365 days,
                # which on rel-hm (task horizon 7 days) spent most of its
                # capacity on horizons the metric never queries, AND -- worse
                # -- made most windows run past the corpus cutoff, so most
                # targets were DROPPED and the arm was supervision-starved.
                # That is the confound RESEARCH.md 2026-08-24 flagged before
                # judging window mode.
                #
                # Log-normal around the task timedelta keeps the head genuinely
                # horizon-CONDITIONED (it still sees a spread, so one model
                # serves several horizons) while putting the mass where it is
                # scored. The timedelta is public task metadata, not a label,
                # so this is not leakage; set window_days=0 for a fully
                # task-agnostic prior.
                wd = self.window_days * np.exp(
                    self.rng.normal(0.0, self.window_sigma, n))
                wd = np.clip(wd, self.WINDOW_MIN_DAYS, self.WINDOW_MAX_DAYS)
            else:
                wd = np.exp(self.rng.uniform(np.log(self.WINDOW_MIN_DAYS),
                                             np.log(self.WINDOW_MAX_DAYS), n))
        else:
            wd = np.zeros(n)

        raw: dict = {}
        for i in range(n):
            if cens[i]:
                continue
            s = seqs[seq_idx[i]]
            if self.window:
                self.n_targets_seen += 1
                if q_t[i] + wd[i] * 86400 > cutoff_s:
                    self.n_targets_dropped += 1
                    continue
                j = np.flatnonzero((s["t"] > q_t[i])
                                   & (s["t"] <= q_t[i] + wd[i] * 86400)
                                   & (s["dst"] >= 0))
            else:
                # next-event target = every PAIR belonging to event nk (one
                # pair normally, several under path retargeting)
                nk = k_in[i] + 1
                j = (np.flatnonzero(s["k"] == nk) if nk < s["n_ev"]
                     else np.empty(0, dtype=np.int64))
            if not len(j):
                continue
            for code in np.unique(s["dst_code"][j]):
                if code < 0:
                    continue
                sel = j[s["dst_code"][j] == code]
                pos = np.unique(s["dst"][sel])
                if len(pos) > self.P_MAX:      # keep the earliest few
                    pos = pos[:self.P_MAX]
                g = raw.setdefault(self._code_name(int(code)),
                                   dict(i=[], pos=[]))
                g["i"].append(i)
                g["pos"].append(pos)

        out: dict = {}
        for tbl, g in raw.items():
            idx = np.asarray(g["i"])
            T = len(idx)
            P, N = self.P_MAX, self.n_neg
            # EXACT softmax for small catalogues. Sampled softmax is an
            # approximation to the full one; when the catalogue is no bigger
            # than the negative sample the approximation is not merely
            # unnecessary but degenerate -- drawing 256 negatives from 77
            # circuits yields mostly duplicates, makes the logQ correction
            # meaningless and lets accidental-hit masking delete most of the
            # row. Scoring every entity is then both cheaper and exact.
            n_dst = self.c.schema.entity_counts[tbl]
            if n_dst <= P + N:
                cands = np.tile(np.arange(n_dst, dtype=np.int64), (T, 1))
                pos_mask = np.zeros((T, n_dst), dtype=bool)
                for r, pp in enumerate(g["pos"]):
                    pos_mask[r, pp] = True
                valid = np.ones((T, n_dst), dtype=bool)
                log_q = np.zeros((T, n_dst), dtype=np.float32)  # exact: no correction
                feats = self._who_features(seqs, seq_idx[idx], k_in[idx],
                                           q_t[idx], tbl, cands)
                out[tbl] = dict(
                    pos=idx, cands=cands, pos_mask=pos_mask, valid=valid,
                    log_q=log_q, feats=feats,
                    win_days=wd[idx].astype(np.float32),
                    hist=self._hist_slice(target, idx),
                )
                continue
            cands = np.zeros((T, P + N), dtype=np.int64)
            pos_mask = np.zeros((T, P + N), dtype=bool)
            valid = np.ones((T, P + N), dtype=bool)
            for r, p in enumerate(g["pos"]):
                cands[r, :len(p)] = p
                pos_mask[r, :len(p)] = True
                valid[r, len(p):P] = False     # padded positive slots

            prior = self._prior_items(seqs, seq_idx[idx], k_in[idx],
                                      tbl, times=q_t[idx])
            gen = self._generator(tbl)
            cd = gen.cooc_dist(prior) if gen.cooc is not None else None
            negs, log_q_neg = gen.sample(T, N, cd)
            cands[:, P:] = negs

            log_q = np.zeros((T, P + N), dtype=np.float32)
            if self.logq:
                # correction applies to SAMPLED candidates only; the positives
                # are included by construction, not drawn from Q
                log_q[:, P:] = log_q_neg + np.log(N)

            # Accidental hits: a sampled negative that is in fact one of this
            # query's positives must not be used as a negative. Standard
            # sampled-softmax hygiene, and it matters more here because the
            # 2-hop proposal deliberately samples plausible items.
            #
            # Done as one broadcast instead of a per-row `np.isin`: at most
            # P=8 positives per row, so [T, P, N] is small, and the row loop
            # was ~T numpy calls per destination table per batch.
            pos_block = np.where(pos_mask[:, :P], cands[:, :P], -1)
            valid[:, P:] &= ~(cands[:, None, P:] == pos_block[:, :, None]
                              ).any(axis=1)

            # `prior` is exactly what feature 7 needs; computing it here and
            # passing it down removes a second identical pass over every
            # sequence (it was built once for the 2-hop PROPOSAL and again for
            # the 2-hop FEATURE).
            feats = self._who_features(seqs, seq_idx[idx], k_in[idx],
                                       q_t[idx], tbl, cands, prior=prior)
            out[tbl] = dict(
                pos=idx, cands=cands, pos_mask=pos_mask, valid=valid,
                log_q=log_q, feats=feats,
                win_days=wd[idx].astype(np.float32),
                hist=self._hist_slice(target, idx),
            )
        return out

    def _prior_items(self, seqs, si, kk, dst_table, times=None):
        """[T, COOC_RECENT_ITEMS] of each query's most recent prior links.

        Selection is by TIME, not by array index: with path retargeting the
        arrays are indexed by (event, destination) pair, so an index is no
        longer a position in the sequence. Sequences are time-ordered, so the
        two agree whenever one pair per event.
        """
        code = self.__dict__.get("_tbl_code", {}).get(dst_table, -1)
        M = self.COOC_RECENT_ITEMS
        prior = np.full((len(si), M), -1, dtype=np.int64)
        for r in range(len(si)):
            s = seqs[si[r]]
            m = s["dst_code"] == code
            if times is not None:
                m &= s["t"] <= times[r]
            else:
                m &= s["k"] <= kk[r]
            sel = np.flatnonzero(m)[-M:]
            if len(sel):
                prior[r, :len(sel)] = s["dst"][sel]
        return prior

    def _hist_slice(self, target, idx):
        """Token positions the reranker attends over: the last HIST_R tokens
        of the query's own packed row, at or before its position. -1 = pad."""
        R = self.HIST_R
        b = np.asarray(target["b"])[idx]
        l = np.asarray(target["l"])[idx]
        k = np.asarray(target["k_in_seq"])[idx]
        base = l - k                        # first token of this sequence
        offs = np.arange(-R + 1, 1)
        pos = l[:, None] + offs[None, :]
        pos[pos < base[:, None]] = -1
        return np.stack([np.where(pos >= 0, b[:, None], 0), pos], axis=0)

    def _generator(self, dst_table: str):
        key = (self.ent_current, dst_table)
        cache = self.__dict__.setdefault("_gen", {})
        if key not in cache:
            cache[key] = CandidateGenerator(
                self.c, dst_table, self.rng,
                cooc=(self._cooc_table(dst_table) if self.hard_negs else None),
                p_uniform=self.p_uniform, p_pop=self.p_pop,
                p_cooc=self.p_cooc if self.hard_negs else 0.0)
        return cache[key]

    def _temporal_index(self, table: str):
        cache = self.__dict__.setdefault("_ti", {})
        if table not in cache:
            cache[table] = TemporalIndex(self.c, table)
        return cache[table]

    def _cooc_table(self, dst_table: str):
        """Top-K co-occurrence over the CURRENT sequence entity, lazily built.

        Keyed by (sequence entity, destination): "customers who bought both"
        is a different table from "posts commented by the same users".
        """
        if not (self.use_cooc or self.hard_negs):
            return None
        key = (self.ent_current, dst_table)
        cache = self.__dict__.setdefault("_cooc", {})
        if key not in cache:
            if self.ent_current == dst_table:
                cache[key] = None          # self-pairs carry no 2-hop signal
            else:
                cache[key] = CoocTable.build(
                    self.c, self.ent_current, dst_table, topk=self.cooc_topk,
                    cache_key=self.cooc_cache_key)
        return cache[key]

    def _prefix_index(self, ids_arr, pos_arr):
        """Group positions by entity id, each group left in position order."""
        order = np.argsort(ids_arr, kind="stable")
        sid, spos = ids_arr[order], pos_arr[order]
        uniq, start = np.unique(sid, return_index=True)
        end = np.append(start[1:], len(sid))
        return uniq, start, end, spos

    def _who_features(self, seqs, tgt_seq, tgt_k, tgt_time, dst_table, cands,
                      prior=None):
        """-> float32 [n_targets, 1+n_neg, N_WHO_FEATS] for ONE dst table."""
        n, m = cands.shape
        feats = np.zeros((n, m, self.N_WHO_FEATS), dtype=np.float32)

        # -- candidate-side temporal (3-6), one vectorised pass -------------
        ti = self._temporal_index(dst_table)
        flat_t = np.repeat(tgt_time, m)
        st = ti.stats(cands.ravel(), flat_t, self.TREND_WINDOW_DAYS,
                      self.TREND_PREV_WINDOWS, self.TREND_TAU_DAYS)
        feats[:, :, 3] = st["log_total"].reshape(n, m)
        feats[:, :, 4] = st["log_recent"].reshape(n, m)
        feats[:, :, 5] = st["trend"].reshape(n, m)
        feats[:, :, 6] = st["staleness"].reshape(n, m)

        # Positions in each sequence that link to THIS destination table,
        # computed once per sequence and shared by both feature loops.
        code = self.__dict__.get("_tbl_code", {}).get(dst_table, -1)
        sel_cache: dict = {}

        def _sel(si):
            if si not in sel_cache:
                s = seqs[si]
                sel_cache[si] = np.flatnonzero((s["dst"] >= 0)
                                               & (s["dst_code"] == code))
            return sel_cache[si]

        # -- 2-hop co-occurrence (7) ---------------------------------------
        cooc = self._cooc_table(dst_table)
        if cooc is not None:
            if prior is None:                  # standalone call (tests, eval)
                prior = self._prior_items(seqs, tgt_seq, tgt_k,
                                          dst_table, times=tgt_time)
            feats[:, :, 7] = np.log1p(cooc.score(
                prior, cands, self.c.schema.entity_counts[dst_table]))

        # Candidate hits against a short history are rare (a random negative
        # is in a <=256-event history with probability ~0.2%), so find the few
        # hits with a vectorised set test and only then do the exact
        # prefix-count lookup.
        cache: dict = {}
        for i in range(n):
            s = seqs[tgt_seq[i]]
            key = tgt_seq[i]
            if key not in cache:
                sel = _sel(key)
                if len(sel):
                    # keyed by TIME, not array index -- under path retargeting
                    # one event contributes several pairs, so an index is not
                    # a sequence position. Sequences are time-ordered, so for
                    # the unexpanded case this is the same selection.
                    cache[key] = self._prefix_index(s["dst"][sel],
                                                    s["t"][sel])
                else:
                    cache[key] = None
            idx = cache[key]
            if idx is None:
                continue
            uniq, start, end, spos = idx
            c = cands[i]
            j = np.searchsorted(uniq, c)
            j_cl = np.clip(j, 0, len(uniq) - 1)
            hit = uniq[j_cl] == c
            for h in np.flatnonzero(hit):
                g = j_cl[h]
                times_g = spos[start[g]:end[g]]
                # at or before the query time (the old position-based form
                # included the query's own event too, so this matches)
                cnt = int(np.searchsorted(times_g, tgt_time[i], side="right"))
                if cnt == 0:
                    continue
                feats[i, h, 0] = 1.0
                feats[i, h, 1] = np.log1p(cnt)
                dt = (tgt_time[i] - times_g[cnt - 1]) / 86400.0
                feats[i, h, 2] = np.exp(-max(dt, 0.0) / self.TAU_DAYS)
        return feats

    # -- inference-time packing -------------------------------------------
    @staticmethod
    def pack_histories(corpus, entity_table, rows, cuts, max_len=256,
                       device="cpu", query_token=False, horizon_s=0.0,
                       query_feats=False, query_dyn=False,
                       query_nbr=False):
        """One entity per batch row: histories strictly before each cutoff.

        Used by queries.py to obtain the query state for a (entity, cutoff)
        pair. Returns (batch, keep) where batch is the same structure encode()
        consumes -- minus targets, which inference does not need -- and
        keep["last_l"] indexes the final real token of each row.

        The `before=cut` truncation is the per-row leakage boundary: a
        prediction at time T must not see events at or after T, even though
        the corpus as a whole extends to the corpus cutoff.

        `query_token=True` appends the synthetic query token of the WindowHead
        at time `cut`, and points `last_l` at IT rather than at the entity's
        final event. That is the difference between "the state after the last
        thing that happened" and "the state now": without it the two are the
        same tensor no matter how long the silence has been, which is the
        defect measured in RESEARCH.md 2026-08-25. An entity with no history
        still gets a query token, so a cold row carries the query time rather
        than an all-zero state.
        """
        B = len(rows)
        hists = [corpus.history(entity_table, int(r), before=int(c))[-max_len:]
                 for r, c in zip(rows, cuts)]
        extra = 1 if query_token else 0
        qfeat = []
        _qf = None
        if query_feats:
            # a throwaway batcher would rebuild popularity tables, so bind the
            # unbound method to a shim carrying only what query_feats reads
            _qdyn, _qnbr = query_dyn, query_nbr
            #                        a class body does NOT close over the
            #                        enclosing function, and `x = x` inside one
            #                        makes x class-local, so the RHS misses the
            #                        closure entirely -> NameError. Rebind first.
            class _Shim:
                c = corpus
                N_QUERY_FEATS_PER_TABLE = PackedBatcher.N_QUERY_FEATS_PER_TABLE
                N_DYN_FEATS_PER_TABLE = PackedBatcher.N_DYN_FEATS_PER_TABLE
                N_DYN_GLOBAL = PackedBatcher.N_DYN_GLOBAL
                DYN_ANCHORS = PackedBatcher.DYN_ANCHORS
                DYN_GAP_TAIL = PackedBatcher.DYN_GAP_TAIL
                # staticmethod(): rebinding a plain function into a class body
                # makes it an INSTANCE method, so _Shim() would pass itself as
                # the first argument.
                _dyn_table_feats = staticmethod(PackedBatcher._dyn_table_feats)
                n_query_feats = staticmethod(PackedBatcher.n_query_feats)
                nbr_feats = staticmethod(PackedBatcher.nbr_feats)
                N_NBR_FEATS_PER_SLOT = PackedBatcher.N_NBR_FEATS_PER_SLOT
                query_dyn = _qdyn
                query_nbr = _qnbr
            _qf = PackedBatcher.query_feats.__get__(_Shim(), _Shim)
        L = max(1, max(len(h) for h in hists) + extra)

        t = np.zeros((B, L), dtype=np.int64)
        gap = np.zeros((B, L), dtype=np.float32)
        sid = np.zeros((B, L), dtype=np.int64)
        last_l = np.zeros(B, dtype=np.int64)
        elapsed = np.zeros(B, dtype=np.float64)
        qb, ql = [], []
        per_rows: dict = {}

        name_of = {spec.table_idx: name
                   for name, spec in corpus.schema.fact_tables.items()}
        for b, ids in enumerate(hists):
            n = len(ids)
            cut = int(cuts[b])
            if query_token:
                t_last = int(corpus.time[ids[-1]]) if n else cut
                t[b, n] = cut
                gap[b, n] = max(cut - t_last, 0.0)
                elapsed[b] = max(cut - t_last, 1.0)
                last_l[b] = n
                qb.append(b); ql.append(n)
                if _qf is not None:
                    full = corpus.history(entity_table, int(rows[b]),
                                          before=cut)
                    qfeat.append(_qf(full, cut, horizon_s))
            if n == 0:
                continue           # all-zero token row; state stays uninformed
            ts = corpus.time[ids]
            t[b, :n] = ts
            gap[b, :n] = np.diff(ts, prepend=ts[0])
            if not query_token:
                last_l[b] = n - 1
                elapsed[b] = max(cut - int(ts[-1]), 1.0)
            tis = corpus.table_idx[ids]
            for ti in np.unique(tis):
                name = name_of[int(ti)]
                sel = np.flatnonzero(tis == ti)
                d = per_rows.setdefault(name, dict(b=[], l=[], rows=[]))
                d["b"].append(np.full(len(sel), b, dtype=np.int64))
                d["l"].append(sel.astype(np.int64))
                d["rows"].append(corpus.row_of[name][ids[sel]])

        T = lambda x, dt=torch.long: torch.as_tensor(
            np.asarray(x), dtype=dt, device=device)
        pt = {}
        for name, d in per_rows.items():
            r = np.concatenate(d["rows"])
            spec = corpus.schema.fact_tables[name]
            pt[name] = dict(
                b=T(np.concatenate(d["b"])), l=T(np.concatenate(d["l"])),
                table_idx=T(np.full(len(r), spec.table_idx)),
                feat_num=T(corpus.feat_num[name][r], torch.float32),
                feat_cat=T(corpus.feat_cat[name][r]),
                links=T(corpus.links[name][r]),
            )
        batch = dict(t=T(t), gap=T(gap, torch.float32), seq_id=T(sid),
                     per_table=pt, seq_entity=entity_table)
        if query_token:
            batch["query_pos"] = dict(b=T(qb), l=T(ql))
            if qfeat:
                batch["query_pos"]["feats"] = T(np.stack(qfeat),
                                                torch.float32)
        keep = dict(b=T(np.arange(B)), last_l=T(last_l),
                    n_events=np.array([len(h) for h in hists]),
                    elapsed=elapsed)
        return batch, keep

    def _to_tensors(self, t, gap, sid, per_table, target, seqs, ent, device,
                    win=None, wd=None):
        T = lambda x, dt=torch.long: torch.as_tensor(
            np.asarray(x), dtype=dt, device=device)
        pt = {}
        for name, d in per_table.items():
            rows = np.asarray(d["rows"])
            spec = self.c.schema.fact_tables[name]
            pt[name] = dict(
                b=T(d["b"]), l=T(d["l"]),
                table_idx=T(np.full(len(rows), spec.table_idx)),
                feat_num=T(self.c.feat_num[name][rows], torch.float32),
                feat_cat=T(self.c.feat_cat[name][rows]),
                links=T(self.c.links[name][rows]),
            )
        who = {tbl: {k: (T(v, torch.float32)
                         if v.dtype.kind == "f" else
                         T(v, torch.bool) if v.dtype == bool else T(v))
                     for k, v in g.items()}
               for tbl, g in (self._build_who(seqs, target).items()
                              if self.next_event else ())}

        tgt = dict(
            b=T(target["b"]), l=T(target["l"]),
            gap=T(target["gap"], torch.float32),
            censored=T(target["censored"], torch.bool),
            table_idx=T(target["table_idx"]),
            q_time=T(target["q_time"]),
            who=who,
            what={
                name: dict(pos=T(w["pos"]),
                           feat_num=T(np.stack(w["feat_num"]),
                                      torch.float32),
                           feat_cat=T(np.stack(w["feat_cat"])))
                for name, w in target["what"].items()
            },
        )
        out = dict(t=T(t), gap=T(gap, torch.float32), seq_id=T(sid),
                   per_table=pt, target=tgt, seq_entity=ent)
        if win is not None and len(win["b"]):
            wpt = {}
            for name, d in win["per_table"].items():
                r = np.concatenate(d["rows"])
                wpt[name] = dict(
                    qi=T(np.concatenate(d["qi"])),
                    w=T(np.concatenate(d["w"]), torch.float32),
                    feat_num=T(self.c.feat_num[name][r], torch.float32),
                    feat_cat=T(self.c.feat_cat[name][r]),
                )
            out["window"] = dict(
                b=T(win["b"]), l=T(win["l"]),
                elapsed=T(win["elapsed"], torch.float32),
                horizon=T(win["horizon"], torch.float32),
                feats=T(np.stack(win["feats"]), torch.float32),
                counts=T(np.stack(win["counts"]), torch.float32),
                n_query=len(win["b"]), per_table=wpt,
                n_capped=win["n_capped"])
            out["query_pos"] = dict(b=out["window"]["b"],
                                    l=out["window"]["l"],
                                    feats=out["window"]["feats"])
        if wd is not None and len(wd["b"]):
            cat = lambda k: np.concatenate(wd[k])
            out["window_dense"] = dict(
                b=T(cat("b")), l=T(cat("l")),
                elapsed=T(cat("elapsed"), torch.float32),
                horizon=T(cat("horizon"), torch.float32),
                feats=T(cat("feats"), torch.float32),
                counts=T(cat("counts"), torch.float32),
                n_query=len(cat("b")))
        return out
