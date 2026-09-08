"""WindowHead: horizon-conditioned window aggregates.

WHY THIS EXISTS
---------------
The four heads of ARCHITECTURE 7 predict the NEXT EVENT. Every RelBench
entity task is a WINDOW AGGREGATE: a count, a sum, or a ratio over the events
an entity emits in `(t, t+D]`. Those are different objects, and the gap
between them was measured on 2026-08-25 (RESEARCH.md): a frozen probe of the
next-event hidden state `h` reaches 74.2 test AUROC on rel-stack user-badge,
while two scalars (n_events, elapsed) reach 81.5 and 33 hand-rolled count
features reach 84.8. Nothing in the next-event objective pressures `h` to
retain "how many badges so far" once that stops predicting the immediate next
token, so it does not.

This head changes the pretraining objective rather than the readout. It is
still fully self-supervised and uses no task labels: sample a query time
`t_q` inside an entity's pre-cutoff history and a horizon `D`, and predict
the SUFFICIENT STATISTICS of the window `(t_q, t_q+D]`:

    per fact table T          E[N_T]                        total arrivals
    per categorical column    E[N_T(col = v)]  for each v   categorical filters
    per numeric column        E[N_T(col in bucket q)]       numeric filters
    per numeric column        quantiles of log1p(sum v)     sum regression
    per fact table            quantiles of log1p(N_T)       count regression

That set is chosen to be COMPLETE for the benchmark: reading every
`make_table` in `relbench/tasks/*.py` shows all 21 entity tasks are one of
{count, count of a filtered subset, sum of a column, ratio of two such}. So

    churn / engagement / badge  = P(N = 0)                 = exp(-rate)
    "more than k"               = P(N > k)                 Poisson tail
    count regression            = median(N)                quantile head
    sum regression (LTV, sales) = median(sum v)            quantile head
    `position <= 3`             = P(N(col in low buckets) > 0)
    `p_value < 0.05`            = same, on that column's buckets
    ad-ctr, driver-position,
    site-success                = ratio of two rates

The five tasks ARCHITECTURE 8.1 records as "not expressible" become
expressible, because the head emits a bucketed DISTRIBUTION over column
values rather than the WHAT head's point estimate.

WHAT IS AND IS NOT APPROXIMATED
-------------------------------
The rates are exact expectations of the target quantities -- no thinning
approximation, no `S_q = S^q`, none of the Poisson-arrival assumptions
ARCHITECTURE 8.1 has to make, because the head is trained directly on the
window count rather than deriving it from a next-gap density. What remains
approximate is the map from a rate to a probability: `P(N = 0) = exp(-rate)`
and the Poisson tail assume the arrivals are Poisson given `h`. The quantile
outputs sidestep even that for the regression families, which is why they are
there: NMAE is minimised by the conditional median, and a pinball loss
estimates the median directly rather than through a distributional
assumption.

LEAKAGE
-------
Bucket edges are quantiles of the corpus's own standardized feature arrays,
which contain pre-cutoff rows only (EventCorpus.build enforces this). Query
times and horizons are drawn so that `t_q + D <= corpus cutoff`, so a window
is never partially observed. Both are asserted in tests.
"""

from __future__ import annotations

import math

from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Categorical columns wider than this get no per-category rate outputs. The
# cost is |vocab| floats of output per column per table, and a 1024-way hashed
# column is both expensive and useless as a task predicate.
CAT_RATE_MAX = 64

# Quantile levels for the sum/count regression outputs. 0.5 is the one the
# NMAE readout uses; the others are kept because a pinball loss trained at a
# single level is noticeably less stable, and because a spread is diagnostic.
TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)


def slog(x):
    """Signed log1p. Column sums can be negative (deltas, scores, ratings),
    so a bare log1p would silently produce NaNs on exactly the columns whose
    sums a regression task might ask for. Monotone, so quantiles survive it."""
    return torch.sign(x) * torch.log1p(x.abs())


def sexp(y):
    """Inverse of `slog`."""
    return torch.sign(y) * torch.expm1(y.abs())


class WinState(NamedTuple):
    """What `WindowHead.state` returns: the trunk output AND the raw query
    features that produced the token it was read from.

    Why a pair rather than a bare tensor. The wide-and-deep rate path (see
    `WindowHead`) adds a GLM on the raw features directly to the head's
    output, so every readout needs both halves. Returning a tensor and
    accepting an optional `feats=None` on each accessor would mean a caller
    that forgot it still got a number back -- a number computed with the GLM
    branch silently zeroed, i.e. a model evaluated as a different model. That
    is exactly the class of failure HANDOFF section 10.1a catalogues. A pair
    turns it into an AttributeError at the first accessor instead.

    Carries `.device` and `.shape` so the readout code that only ever asked
    `z` for those keeps working unchanged.
    """
    z: torch.Tensor
    feats: torch.Tensor | None = None

    @property
    def device(self):
        return self.z.device

    @property
    def shape(self):
        return self.z.shape

    def __len__(self):
        return len(self.z)

    def index(self, idx):
        return WinState(self.z[idx],
                        None if self.feats is None else self.feats[idx])


def cat_states(states):
    """Concatenate WinStates along the query axis (batched evaluation)."""
    states = list(states)
    if not states:
        raise ValueError("cat_states: nothing to concatenate")
    z = torch.cat([s.z for s in states])
    if any(s.feats is None for s in states):
        if not all(s.feats is None for s in states):
            raise ValueError("cat_states: some states carry features and "
                             "some do not; that would silently zero the GLM "
                             "branch for part of the evaluation set")
        return WinState(z, None)
    return WinState(z, torch.cat([s.feats for s in states]))


class HorizonEncoder(nn.Module):
    """Fourier features of (elapsed since last event, horizon), both in days.

    Both span orders of magnitude -- elapsed runs from minutes to years, the
    benchmark's horizons from 4 to 365 days -- so the encoding is on the log,
    exactly as the TimeEncoder does for gaps.

    Conditioning the HEAD on the horizon rather than baking it into the token
    stream means one forward pass answers every horizon, which is what makes
    a task-agnostic pretraining objective cheap to evaluate.
    """

    def __init__(self, dim: int, n_freq: int = 8):
        super().__init__()
        self.freq = nn.Parameter(torch.logspace(-1, 1, n_freq))
        self.proj = nn.Linear(4 * n_freq, dim)

    def forward(self, elapsed_s, horizon_s):
        d = 86400.0
        a = torch.log1p((elapsed_s / d).clamp(min=0)).unsqueeze(-1) * self.freq
        b = torch.log1p((horizon_s / d).clamp(min=0)).unsqueeze(-1) * self.freq
        return self.proj(torch.cat([a.sin(), a.cos(), b.sin(), b.cos()], -1))


def _snap_to_separators(col: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Move each quantile onto a distinct separating value, strictly forward.

    `col` is one column of observations, `edges` its interior quantiles. The
    separators are the midpoints between consecutive DISTINCT values, i.e.
    every cut that actually splits the data. Each edge takes the nearest
    unused separator at or after the one the previous edge took, so the result
    is strictly increasing whenever the column has enough distinct values, and
    saturates at the top (producing empty high buckets, which `centers`
    already handles) when it does not.
    """
    u = np.unique(col[np.isfinite(col)])
    if len(u) < 2:
        return np.asarray(edges, dtype=np.float64)
    mid = (u[:-1] + u[1:]) / 2.0
    prev = -1
    out = np.empty(len(edges), dtype=np.float64)
    for i, x in enumerate(edges):
        k = int(np.searchsorted(mid, x))
        # nearest of the two neighbouring separators
        if k > 0 and (k >= len(mid) or abs(mid[k - 1] - x) <= abs(mid[k] - x)):
            k -= 1
        k = min(max(k, prev + 1), len(mid) - 1)
        out[i] = mid[k]
        prev = k
    return out


def bucket_edges(corpus, n_buckets: int) -> dict:
    """Interior quantile edges and bucket means per (fact table, numeric col).

    Returns {table: {"edges": [n_num, B-1] standardized,
                     "centers": [n_num, B] standardized}}. Read off
    `corpus.feat_num`, which holds standardized values of PRE-CUTOFF rows
    only, so these are legitimate training-time statistics.

    `centers` is the MEAN of the values that fall in each bucket, not the
    midpoint: the outer buckets are unbounded and a midpoint is undefined
    there, and the mean is what a bucketed expectation
    `E[v] = sum_b rate_b * center_b / sum_b rate_b` actually needs.

    A column that is constant yields duplicate edges; the bucket assignment
    then concentrates in one bucket, which is correct rather than degenerate.

    SKEWED COLUMNS ARE SNAPPED TO REAL SEPARATORS (2026-08-25). Bare quantiles
    plus an epsilon are wrong for a zero-inflated count -- and child-aggregate
    denormalization (data/denorm.py) produces exactly those: 99% of
    `SearchInfo` rows have zero clicked ads, so every interior quantile is the
    same value and the seven edges land within 6e-6 of it, BELOW the whole
    distribution. Every bucket weight for `> 0.5 clicks` then evaluates to 1
    and the filtered rate silently equals the unfiltered one -- the same class
    of silent-degradation bug as HANDOFF 10.1a.

    Each quantile is instead snapped to the nearest midpoint between two
    distinct observed values, then forced strictly forward along that grid.
    On a count column that recovers the natural 0 | 1 | 2 | ... split; on a
    continuous column the snap moves an edge by at most half a gap between
    neighbouring observations, i.e. by nothing that matters.
    """
    qs = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    out = {}
    for name, spec in corpus.schema.fact_tables.items():
        n_num = sum(c.kind == "numeric" for c in spec.columns)
        if not n_num:
            continue
        arr = corpus.feat_num[name]
        if len(arr) == 0:
            out[name] = dict(
                edges=np.zeros((n_num, n_buckets - 1), dtype=np.float32),
                centers=np.zeros((n_num, n_buckets), dtype=np.float32))
            continue
        a = arr.astype(np.float64)
        e = np.quantile(a, qs, axis=0).T                        # [n_num, B-1]
        for j in range(n_num):
            e[j] = _snap_to_separators(a[:, j], e[j])
        # strictly increasing, so searchsorted cannot collapse two buckets.
        # After snapping this only separates edges that SATURATED at the top
        # separator (a column with fewer distinct values than buckets); the
        # nudge lands them in territory that holds no data, which is the
        # honest representation of an empty bucket.
        e = np.maximum.accumulate(e, axis=1)
        e = e + np.arange(e.shape[1], dtype=np.float64) * 1e-6
        cen = np.zeros((n_num, n_buckets), dtype=np.float64)
        for j in range(n_num):
            b = np.searchsorted(e[j], a[:, j])
            s = np.bincount(b, weights=a[:, j], minlength=n_buckets)
            c = np.bincount(b, minlength=n_buckets)
            cen[j] = np.where(c > 0, s / np.maximum(c, 1), 0.0)
            # empty buckets fall back to the edge they sit against, so an
            # expectation over them stays monotone in the bucket index
            for k in range(n_buckets):
                if c[k] == 0:
                    cen[j, k] = e[j, min(k, n_buckets - 2)]
        out[name] = dict(
            edges=np.ascontiguousarray(e, dtype=np.float32),
            centers=np.ascontiguousarray(cen, dtype=np.float32))
    return out


class WindowHead(nn.Module):
    """Predicts the sufficient statistics of `(t_q, t_q + D]` from `h`.

    One trunk shared across tables; per-table linear outputs. Rates are
    emitted in log space and trained with a Poisson negative log-likelihood,
    which is the right loss for a count with a lot of zeros and needs no
    zero-inflation machinery: `exp(-rate)` IS the zero probability, and it is
    exactly the churn readout.
    """

    def __init__(self, dim: int, schema, edges: dict | None = None,
                 n_buckets: int = 8, taus=TAUS, balance: str = "legacy",
                 w_rate: float = 1.0, w_qcount: float = 1.0,
                 w_detail: float = 1.0, n_qfeat: int = 0,
                 feat_path: bool = False, query_head: bool = False,
                 rate_balance: str = "none", rate_zero_scale: float = 1.0):
        super().__init__()
        self.schema = schema
        self.n_buckets = n_buckets
        # TERM BALANCE (see `loss`). "legacy" is the v1 unweighted mean over
        # every term, kept so that every checkpoint measured before
        # 2026-08-26 remains reproducible; "weighted" groups the terms by what
        # a readout actually reads. Not a buffer -- it changes no parameter
        # shape, so a checkpoint loads under either setting.
        if balance not in ("legacy", "weighted"):
            raise ValueError(f"unknown balance: {balance}")
        self.balance = balance
        # See `_pois`. Not a buffer: it changes no parameter shape, so a
        # checkpoint loads under either setting.
        if rate_balance not in ("none", "cb"):
            raise ValueError(f"unknown rate_balance: {rate_balance}")
        self.rate_balance = rate_balance
        self.rate_zero_scale = float(rate_zero_scale)
        self.w_rate = w_rate
        self.w_qcount = w_qcount
        self.w_detail = w_detail
        self.taus = tuple(taus)
        self.register_buffer("tau_t", torch.tensor(self.taus,
                                                   dtype=torch.float32))
        self.horizon = HorizonEncoder(dim)
        self.trunk = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU())

        # QUERY-CONDITIONED HEAD (see ledger/qsample.py). Trained on RANDOM
        # window aggregations drawn from the schema, never from the task
        # registry, so it learns to answer a query rather than to answer one
        # task. Optional: a checkpoint trained without it loads unchanged.
        self.query_head = None
        if query_head:
            from ..qsample import AGGS, MAX_CAT, max_cols
            mc = max_cols(schema)
            n_tbl = len(schema.fact_tables)
            e = max(dim // 8, 16)
            self.q_tbl = nn.Embedding(max(n_tbl, 1), e)
            self.q_agg = nn.Embedding(len(AGGS), e)
            self.q_kind = nn.Embedding(4, e)
            self.q_col = nn.Embedding(max(n_tbl * 2 * mc, 1), e)
            self.q_cat = nn.Embedding(MAX_CAT + 1, e)   # +1 = pad
            self.q_val = nn.Embedding(max(n_tbl * mc, 1), e)
            self.query_head = nn.Sequential(
                nn.Linear(dim + 6 * e + 1, dim), nn.GELU(),
                nn.Linear(dim, dim), nn.GELU(),
                nn.Linear(dim, len(self.taus)))
            # OUTPUT RANGE, per aggregation. The head predicts in `slog`
            # space and `sexp` inverts it exponentially, so an error of +2
            # there becomes a factor of ~7 in raw units: on rel-f1 one eval
            # landed at NMAE 4.43 against a 0.55 neighbourhood, and the whole
            # first half of that run oscillated. These track the range of
            # targets actually seen in training and clamp predictions to it,
            # which bounds the blow-up without touching the loss.
            self.register_buffer("q_lo", torch.full((len(AGGS),), float("inf")))
            self.register_buffer("q_hi", torch.full((len(AGGS),), -float("inf")))

        self.rate = nn.ModuleDict()        # table -> Linear(dim, 1)
        self.qcount = nn.ModuleDict()      # table -> Linear(dim, n_taus)
        self.cat_rate = nn.ModuleDict()    # table -> ModuleList over cat cols
        self.num_rate = nn.ModuleDict()    # table -> Linear(dim, n_num*B)
        self.num_qsum = nn.ModuleDict()    # table -> Linear(dim, n_num*n_taus)
        self.cat_cols: dict = {}           # table -> [(col idx, cardinality)]
        self.num_cols: dict = {}           # table -> n numeric columns

        for name, spec in schema.fact_tables.items():
            self.rate[name] = nn.Linear(dim, 1)
            self.qcount[name] = nn.Linear(dim, len(self.taus))
            cats = [c for c in spec.columns if c.kind == "categorical"]
            keep = [(j, c.cardinality) for j, c in enumerate(cats)
                    if c.cardinality <= CAT_RATE_MAX]
            self.cat_cols[name] = keep
            if keep:
                self.cat_rate[name] = nn.ModuleList(
                    nn.Linear(dim, card) for _, card in keep)
            n_num = sum(c.kind == "numeric" for c in spec.columns)
            self.num_cols[name] = n_num
            if n_num:
                self.num_rate[name] = nn.Linear(dim, n_num * n_buckets)
                self.num_qsum[name] = nn.Linear(dim, n_num * len(self.taus))

        # -- WIDE-AND-DEEP RATE PATH (`feat_path`) -------------------------
        #
        # The measurement this exists for (RESEARCH.md 2026-08-25, rel-stack
        # user-badge, frozen probes): the backbone hidden state reaches 74.2
        # test AUROC, two scalars (n_events, elapsed) reach 81.5, 33 count
        # features reach 84.8 -- and `h` CONCATENATED with those 33 scores
        # WORSE than the 33 alone.
        #
        # The last clause is the informative one. It says the features are
        # sufficient and the hidden state is, at the margin, noise. The
        # previous response was `LEDGER.query_feat_proj`, which adds the
        # features to the query TOKEN -- so they enter at the bottom of a
        # 12-layer pre-norm trunk and have to survive a LayerNorm in every
        # block before this head sees them. A log-count IS its magnitude, and
        # LayerNorm is precisely the operation that discards magnitude. We
        # handed the model the answer along the one path guaranteed to erase
        # it.
        #
        # So: a direct linear map from the RAW feature vector to the head's
        # output, bypassing the trunk entirely, added to the deep branch.
        #
        #     log_rate = deep(z) + glm(feats)
        #
        # This is the standard wide-and-deep / residual-GLM structure, and it
        # changes what the transformer is asked to do: not to re-derive "how
        # many badges so far" from attention, but to predict the RESIDUAL that
        # the count features leave behind. It also makes the "h + features is
        # worse" result unreachable by construction -- with the deep branch
        # zero-initialised the model STARTS at the GLM and can only be moved
        # off it by data.
        #
        # Which outputs get the branch, and why not all of them:
        #   rate     yes -- counts predict counts; every classification task
        #   qcount   yes -- same target, different loss
        #   cat_rate yes -- the per-category counters ARE in the feature
        #            vector (`query_cat_layout`), and rel-event user-repeat is
        #            a task whose entire signal is the historical yes-FRACTION,
        #            which is a difference of two entries of that vector and
        #            so is linear in it (RESEARCH.md 2026-08-25: the fraction
        #            alone scores 70.31 while the count scores 41.43).
        #   num_rate/q_sum  NO -- the query features carry no per-numeric-
        #            column history at all, so a GLM on them would be a bias
        #            term wearing a weight matrix.
        #
        # The features are log1p counts and recencies, already compressed to
        # roughly [0, 15]; they are fed RAW. Normalising them here would undo
        # the one property that makes them useful.
        self.feat_path = bool(feat_path and n_qfeat)
        self.n_qfeat = n_qfeat
        self.feat_rate = None
        self.feat_qcount = None
        self.feat_cat_rate = None
        if self.feat_path:
            self.feat_rate = nn.ModuleDict()
            self.feat_qcount = nn.ModuleDict()
            self.feat_cat_rate = nn.ModuleDict()
            for name in schema.fact_tables:
                self.feat_rate[name] = nn.Linear(n_qfeat, 1)
                self.feat_qcount[name] = nn.Linear(n_qfeat, len(self.taus))
                keep = self.cat_cols.get(name, [])
                if keep:
                    self.feat_cat_rate[name] = nn.ModuleList(
                        nn.Linear(n_qfeat, card) for _, card in keep)
                # The deep branch starts at exactly zero so the head starts at
                # exactly the GLM. Biases are left alone -- they are the base
                # rate, and the GLM has its own.
                nn.init.zeros_(self.rate[name].weight)
                nn.init.zeros_(self.qcount[name].weight)
                if name in self.cat_rate:
                    for lin in self.cat_rate[name]:
                        nn.init.zeros_(lin.weight)

        # Bucket edges ride in the state_dict as buffers, so a checkpoint
        # carries the discretisation it was trained with and an eval-time
        # reconstruction cannot silently use different ones (HANDOFF 10.1a is
        # the same class of bug for categorical vocabularies).
        for name, spec in schema.fact_tables.items():
            n_num = self.num_cols[name]
            if not n_num:
                continue
            d = (edges or {}).get(name) or {}
            e = d.get("edges")
            cen = d.get("centers")
            if e is None:
                e = np.zeros((n_num, n_buckets - 1), dtype=np.float32)
            if cen is None:
                cen = np.zeros((n_num, n_buckets), dtype=np.float32)
            self.register_buffer(f"edges__{name}",
                                 torch.as_tensor(e, dtype=torch.float32))
            self.register_buffer(f"centers__{name}",
                                 torch.as_tensor(cen, dtype=torch.float32))
            # Standardization constants, so the head can talk about RAW column
            # values: the corpus stores (v - mean) / std, but a task asks for
            # the sum of prices and for `position <= 3` in raw units.
            nums = [c for c in spec.columns if c.kind == "numeric"]
            self.register_buffer(
                f"nmean__{name}",
                torch.tensor([c.mean for c in nums], dtype=torch.float32))
            self.register_buffer(
                f"nstd__{name}",
                torch.tensor([c.std for c in nums], dtype=torch.float32))

    def edges_of(self, name: str):
        return getattr(self, f"edges__{name}", None)

    def centers_of(self, name: str, raw: bool = False):
        """Bucket means. `raw=True` de-standardizes them to column units."""
        c = getattr(self, f"centers__{name}", None)
        if c is None or not raw:
            return c
        return (c * getattr(self, f"nstd__{name}").unsqueeze(-1)
                + getattr(self, f"nmean__{name}").unsqueeze(-1))

    def numeric_index(self, name: str, column: str):
        """Position of `column` among the fact table's numeric columns."""
        nums = [c for c in self.schema.fact_tables[name].columns
                if c.kind == "numeric"]
        for j, c in enumerate(nums):
            if c.name == column:
                return j
        return None

    def categorical_index(self, name: str, column: str):
        """(position among categoricals, ColumnSpec) or (None, None)."""
        cats = [c for c in self.schema.fact_tables[name].columns
                if c.kind == "categorical"]
        for j, c in enumerate(cats):
            if c.name == column:
                return j, c
        return None, None

    def integral_shift(self, name: str, col_idx: int) -> float:
        """Half a step, standardized, when column `col_idx` is INTEGER-valued.
        0.0 otherwise. -> float

        WHY THIS EXISTS (2026-09-01). `_bucket_weights` gives the bucket a
        threshold falls inside the fraction of its WIDTH below that
        threshold, which assumes the values are locally uniform. That is
        false for an integer column, and the error is not a harmless scaling.
        Measured on rel-f1 `qualifying.position` (values 1..24, so bucket 1
        spans raw (2.5, 5.5] = positions 3, 4, 5):

            P(position <= 3)   linear 0.1122   truth 0.1346

        -- the bucket that holds position 3 gets weight 0.167 instead of
        0.333, so the filtered rate is under-read by 17%. `driver-top3` is
        exactly `position <= 3`, and because the deficit falls on ONE bucket
        it re-ranks rather than rescales: a driver who habitually qualifies
        3rd is scored against one who qualifies 1st-2nd at half the weight
        the data supports.

        The fix is to evaluate the predicate at the SEPARATOR, `T + 0.5`,
        which is where the discrete predicate actually changes -- the same
        move `_snap_to_separators` makes for the edges themselves. With it
        the readout reproduces the empirical fraction exactly (0.1346).

        Integrality is read off the persisted edges rather than recorded
        separately, so this needs no corpus rebuild and fixes checkpoints
        already on disk. That inference is sound because `_snap_to_separators`
        puts every edge on a midpoint between two DISTINCT observed values:
        on an integer column those midpoints are half-integers, and on a
        continuous one all seven landing within tolerance of a half-integer
        does not happen. The `+ arange * 1e-6` nudge that keeps the edges
        strictly increasing is why the tolerance is not exact equality.
        """
        e = self.edges_of(name)
        if e is None or col_idx >= e.shape[0]:
            return 0.0
        s = getattr(self, f"nstd__{name}")[col_idx].item()
        m = getattr(self, f"nmean__{name}")[col_idx].item()
        if not s:
            return 0.0
        raw = e[col_idx].detach().cpu().numpy().astype(float) * s + m
        if not len(raw):
            return 0.0
        frac = np.abs(raw - np.floor(raw) - 0.5)
        if not np.all(frac < 1e-3):
            return 0.0
        return 0.5 / s

    def standardize(self, name: str, column: str, value: float):
        """A raw threshold in the standardized units the buckets live in."""
        j = self.numeric_index(name, column)
        if j is None:
            return None
        m = getattr(self, f"nmean__{name}")[j].item()
        s = getattr(self, f"nstd__{name}")[j].item()
        return (value - m) / (s if s else 1.0)

    def destandardize(self, name: str, z_sum, count):
        """sum of RAW values from the standardized sum and the arrival count.

        `sum v = std * sum z + mean * N`, exactly.
        """
        return (getattr(self, f"nstd__{name}") * z_sum
                + getattr(self, f"nmean__{name}") * count.unsqueeze(-1))

    # -- forward ---------------------------------------------------------

    def state(self, h, elapsed, horizon, feats=None) -> "WinState":
        """Query state conditioned on (elapsed, horizon). -> WinState.

        `feats` is the raw query-feature vector of the token `h` was read from
        (`batch["window"]["feats"]`, or `batch["query_pos"]["feats"]` on the
        eval path). It is REQUIRED when the head was built with `feat_path`,
        and passing it to a head without one is an error rather than a silent
        no-op -- a mismatch here means the model is being evaluated as a
        different model than it was trained as.
        """
        if self.feat_path and feats is None:
            raise ValueError(
                "this WindowHead has a wide-and-deep feature path, so "
                "state() needs the query features. Pass "
                "batch['window']['feats'] (training) or "
                "batch['query_pos']['feats'] (eval); building the eval batch "
                "with query_feats=True is what produces them.")
        if feats is not None and not self.feat_path:
            raise ValueError(
                "state() was given query features but this WindowHead has no "
                "feature path, so they would be ignored. The checkpoint and "
                "the caller disagree about the architecture.")
        return WinState(self.trunk(h + self.horizon(elapsed, horizon)), feats)

    def _glm(self, branch, s: "WinState", name: str, slot=None):
        """The wide half. Returns 0 when the head has no feature path."""
        if not self.feat_path or branch is None or name not in branch:
            return None
        lin = branch[name] if slot is None else branch[name][slot]
        return lin(s.feats)

    def log_rate(self, s: "WinState", name: str):
        """log E[N_T] for one fact table. -> [Q]"""
        out = self.rate[name](s.z)
        g = self._glm(self.feat_rate, s, name)
        if g is not None:
            out = out + g
        return out.squeeze(-1)

    def log_rate_cat(self, s: "WinState", name: str, col_pos: int):
        """log E[N_T(col = v)] for every v. -> [Q, cardinality] or None."""
        keep = self.cat_cols.get(name, [])
        for slot, (j, _) in enumerate(keep):
            if j == col_pos:
                out = self.cat_rate[name][slot](s.z)
                g = self._glm(self.feat_cat_rate, s, name, slot)
                return out if g is None else out + g
        return None

    def log_rate_num(self, s: "WinState", name: str):
        """log E[N_T(col in bucket b)]. -> [Q, n_num, n_buckets] or None."""
        if name not in self.num_rate:
            return None
        q = s.z.shape[0]
        return self.num_rate[name](s.z).view(q, self.num_cols[name],
                                             self.n_buckets)

    def q_sum(self, s: "WinState", name: str):
        """Quantiles of log1p(sum v). -> [Q, n_num, n_taus] or None."""
        if name not in self.num_qsum:
            return None
        q = s.z.shape[0]
        return self.num_qsum[name](s.z).view(q, self.num_cols[name],
                                             len(self.taus))

    def query_predict(self, s: "WinState", enc: dict, clamp: bool = True):
        """Quantiles of one query's answer, for every query token.

        `enc` comes from `qsample.encode` and describes ONE query; the same
        query is asked of all Q tokens, so the embedding is broadcast.
        -> [Q, n_taus] in the TRANSFORMED space (`slog`, except ratios).
        """
        if self.query_head is None:
            raise RuntimeError("this checkpoint has no query head")
        z = s.z
        nq = z.shape[0]
        # a categorical filter names a SET; the padded slots are masked out
        # so a one-value filter is not diluted by three zero embeddings
        ce = self.q_cat(enc["cat"])                       # [1, MAXV, e]
        m = enc["cat_mask"].unsqueeze(-1)                 # [1, MAXV, 1]
        ce = (ce * m).sum(1) / m.sum(1).clamp(min=1.0)    # [1, e]
        e = torch.cat([self.q_tbl(enc["t"]), self.q_agg(enc["a"]),
                       self.q_kind(enc["k"]), self.q_col(enc["c"]), ce,
                       self.q_val(enc["v"]), enc["thr"].unsqueeze(-1)],
                      dim=-1).to(z.dtype)                 # [1, 6e+1]
        out = self.query_head(torch.cat([z, e.expand(nq, -1)], dim=-1))
        if clamp:
            a = int(enc["a"][0])
            lo, hi = float(self.q_lo[a]), float(self.q_hi[a])
            if lo <= hi:      # both finite: training saw this aggregation
                out = out.clamp(min=lo, max=hi)
        return out

    def query_loss(self, s: "WinState", win, specs, schema):
        """Pinball loss of the sampled queries' true answers.

        The target is transformed with `slog` for counts, sums and means --
        the same monotone squash `q_sum` uses, so a heavy tail cannot
        dominate -- and left raw for ratios, which are already in [0, 1].
        Rows whose answer is undefined (an empty denominator) are dropped
        rather than taught as 0.
        """
        from ..qsample import AGGS, answer, encode
        if self.query_head is None or not specs:
            return s.z.new_zeros(()), 0
        nq = s.z.shape[0]
        tot, n = s.z.new_zeros(()), 0
        for spec in specs:
            tgt, valid = answer(spec, win, schema, nq, s.z.device)
            if not bool(valid.any()):
                continue
            y = tgt if spec.agg == "ratio" else slog(tgt)
            with torch.no_grad():
                ai = AGGS.index(spec.agg)
                yv = y[valid].float()
                self.q_lo[ai] = torch.minimum(self.q_lo[ai], yv.min())
                self.q_hi[ai] = torch.maximum(self.q_hi[ai], yv.max())
            # clamped only at READ time: clamping the training path would
            # zero the gradient exactly where the head is most wrong
            pred = self.query_predict(s, encode([spec], schema, s.z.device),
                                      clamp=False)
            tot = tot + self._pinball(pred[valid], y[valid].to(pred.dtype),
                                      self.tau_t)
            n += 1
        return (tot / max(n, 1)), n

    def q_count(self, s: "WinState", name: str):
        """Quantiles of log1p(N_T). -> [Q, n_taus]"""
        out = self.qcount[name](s.z)
        g = self._glm(self.feat_qcount, s, name)
        return out if g is None else out + g

    # -- loss ------------------------------------------------------------

    def _pois(self, log_rate, tgt):
        """Poisson NLL, optionally CLASS-BALANCED over zero / non-zero cells.

        WHY. The rate heads are trained on windows that are overwhelmingly
        empty, so the plain mean is dominated by cells whose target is 0 and
        the head converges onto the global base rate. That is the measured
        failure on the sparse tasks: rel-trial study-adverse predicted a
        median arrival rate of 0.003 and scored numerically identically to
        predicting zero, and rel-event user-attendance predicted 0.033 and
        sat on the constant. Both are winnable -- KumoRFM-2 reaches 0.1277 on
        study-adverse against a 0.1697 constant -- so the information is
        there and the objective is not asking for it.

        `cb` reweights so the non-empty cells carry the same total weight as
        the empty ones. The zero cells are NOT dropped: `exp(-rate)` IS the
        zero probability and the readout depends on it, so they are
        downweighted rather than removed.

        Degenerate batches (all-zero or all-positive) fall back to the plain
        mean, which is what the weighting converges to anyway.
        """
        pl = F.poisson_nll_loss(log_rate, tgt, log_input=True, full=False,
                                reduction="none")
        if self.rate_balance != "cb":
            return pl.mean()
        pos = (tgt > 0).to(pl.dtype)
        n_pos = pos.sum()
        n_zero = pos.numel() - n_pos
        if n_pos.item() == 0 or n_zero.item() == 0:
            return pl.mean()
        w = pos + (1.0 - pos) * (n_pos / n_zero) * self.rate_zero_scale
        return (pl * w).sum() / w.sum().clamp(min=1e-9)

    @staticmethod
    def _pinball(pred, target, taus):
        """Mean pinball loss. pred [..., n_tau], target [...], taus [n_tau]."""
        d = target.unsqueeze(-1) - pred
        return torch.maximum(taus * d, (taus - 1.0) * d).mean()

    def _bucket_of(self, name: str, vals):
        """Standardized values -> bucket index. vals [N, n_num] -> [N, n_num]"""
        e = self.edges_of(name)                       # [n_num, B-1]
        # searchsorted per column; boundary convention matches np.digitize
        # with right=False, i.e. bucket b holds [edge_{b-1}, edge_b).
        return torch.searchsorted(
            e.contiguous(), vals.transpose(0, 1).contiguous()).transpose(0, 1)

    def loss_dense(self, h, wd) -> torch.Tensor:
        """Rate and count-quantile terms for DENSE targets (batching.py).

        Same two heads and the same weights as the `rate` / `qcount` groups of
        `loss`, evaluated on window targets attached to ordinary event tokens.
        The per-column detail terms are deliberately absent: their targets
        need the event rows of each window, which is what makes them
        expensive, and they are not what the dense signal is for.

        Deliberately NOT folded into `loss` as extra query rows. The detail
        terms build their targets as zeros over ALL queries and scatter the
        observed events in, so a query carrying no enumerated detail is
        indistinguishable from one whose window is genuinely empty -- dense
        rows would silently teach every categorical rate to be zero. Two
        blocks, two losses, no shared target tensor.
        """
        if wd is None or not wd["n_query"]:
            return h.new_zeros(())
        z = self.state(h, wd["elapsed"], wd["horizon"],
                       wd["feats"] if self.feat_path else None)
        zt = z.z
        total = zt.new_zeros(())
        n = 0
        for name, spec in self.schema.fact_tables.items():
            cnt = wd["counts"][:, spec.table_idx].to(zt.dtype)
            total = total + self.w_rate * self._pois(
                self.log_rate(z, name), cnt)
            total = total + self.w_qcount * self._pinball(
                self.q_count(z, name), torch.log1p(cnt), self.tau_t)
            n += 1
        return total / max(n, 1)

    def loss(self, h, win) -> torch.Tensor:
        """`win` is the batcher's window-target block (see batching.py).

        Every term is a mean over queries, so tables that appear rarely do not
        dominate.

        TERM BALANCE (`self.balance`)
        -----------------------------
        "legacy" is v1: an unweighted mean over every term, with the
        categorical and numeric-bucket terms scaled by their output width so
        each is a per-query SUM over its outputs.

        That balance is badly wrong for what the readouts read. Counting the
        terms on rel-stack gives 52 of them across seven fact tables -- and
        `user-badge`, `user-engagement`, `post-votes` and every other entity
        task on that database read exactly ONE of the 52, `log_rate(z,
        "badges")`. The `* card` scaling makes it worse than 1/52 in
        magnitude: a card-64 categorical column contributes a sum over 64
        outputs against the rate term's single scalar. So ~98% of the window
        gradient was spent fitting marginals no readout ever queries, which is
        a candidate mechanical explanation for the window objective saturating
        within 500 steps on rel-stack (RESEARCH.md 2026-08-26).

        "weighted" groups the terms by what reads them and normalises each
        group to a per-output mean:

            w_rate    log E[N_T]                 -- every classification task
            w_qcount  quantiles of log1p(N_T)    -- every count regression
            w_detail  per-category rates, per-bucket rates, sum quantiles
                      -- filtered and sum tasks

        The first two are averaged over fact tables, the third over every
        detail output group, so the rate term is 1/3 of the objective instead
        of 1/52 and no table's column count can dilute it.

        The `* card` / `* n_buckets` scalings are DROPPED in weighted mode on
        purpose: a per-output mean is the normalisation that makes a wide
        categorical column and a narrow one contribute comparably, which is
        the whole point of putting them in a shared budget.
        """
        if win is None or win["n_query"] == 0:
            return h.new_zeros(())
        z = self.state(h, win["elapsed"], win["horizon"],
                       win["feats"] if self.feat_path else None)
        zt = z.z          # the trunk tensor, for shape/dtype/device only
        nq = win["n_query"]
        taus = self.tau_t
        weighted = self.balance == "weighted"
        total = zt.new_zeros(())
        n_terms = 0
        # weighted-mode accumulators
        rate_sum = zt.new_zeros(())
        qcount_sum = zt.new_zeros(())
        detail_sum = zt.new_zeros(())
        n_tbl = 0
        n_detail = 0

        for name, spec in self.schema.fact_tables.items():
            g = win["per_table"].get(name)
            qi = g["qi"] if g is not None else None
            # -- total arrivals. `win["counts"]` is the EXACT per-table count
            # of the window, computed before the per-column detail is capped.
            # Deriving it from the capped event list instead taught heavy
            # entities a rate of `win_max_events` (fixed 2026-08-25). Queries
            # with zero events are included: counting only queries that HAVE
            # events would train the head on a truncated distribution and
            # destroy exactly the P(N = 0) readout.
            cnt = win["counts"][:, spec.table_idx].to(zt.dtype)
            lr = self.log_rate(z, name)
            t_rate = self._pois(lr, cnt)
            t_qcount = self._pinball(self.q_count(z, name),
                                     torch.log1p(cnt), taus)
            if weighted:
                rate_sum = rate_sum + t_rate
                qcount_sum = qcount_sum + t_qcount
                n_tbl += 1
            else:
                total = total + t_rate + t_qcount
                n_terms += 2

            if qi is None or not qi.numel():
                continue
            # per-event weight restoring a capped detail sample to the exact
            # count; 1.0 for every uncapped window
            w = g.get("w")
            if w is None:
                w = torch.ones_like(qi, dtype=zt.dtype)
            # -- per-category rates
            for slot, (j, card) in enumerate(self.cat_cols.get(name, [])):
                tgt = torch.zeros(nq, card, device=z.device)
                idx = g["feat_cat"][:, j].long().clamp(0, card - 1)
                tgt.index_put_((qi, idx), w, accumulate=True)
                # the per-CATEGORY rate is what a filtered readout reads
                # (rel-event user-attendance is `status in {yes, maybe}`), so
                # it needs the balancing as much as the table rate does
                t_cat = self._pois(self.log_rate_cat(z, name, j), tgt)
                if weighted:
                    detail_sum = detail_sum + t_cat
                    n_detail += 1
                else:
                    total = total + t_cat * card
                    n_terms += 1
            # -- per-numeric-bucket rates and sum quantiles
            n_num = self.num_cols[name]
            if n_num:
                vals = g["feat_num"]                        # [N, n_num]
                bk = self._bucket_of(name, vals)            # [N, n_num]
                tgt = torch.zeros(nq, n_num, self.n_buckets, device=z.device)
                col = torch.arange(n_num, device=z.device).expand_as(bk)
                tgt.index_put_(
                    (qi.unsqueeze(-1).expand_as(bk), col, bk),
                    w.unsqueeze(-1).expand_as(vals), accumulate=True)
                t_num = self._pois(self.log_rate_num(z, name), tgt)

                zs = torch.zeros(nq, n_num, device=z.device)
                zs.index_add_(0, qi, vals * w.unsqueeze(-1))  # standardized
                raw = self.destandardize(name, zs, cnt)     # RAW sum
                t_qsum = self._pinball(self.q_sum(z, name), slog(raw), taus)
                if weighted:
                    detail_sum = detail_sum + t_num + t_qsum
                    n_detail += 2
                else:
                    total = total + t_num * self.n_buckets + t_qsum
                    n_terms += 2

        if not weighted:
            return total / max(n_terms, 1)
        # Each group is a mean over its own members, so adding a fact table
        # with many columns cannot dilute the rate term.
        out = zt.new_zeros(())
        if n_tbl:
            out = out + (self.w_rate * rate_sum + self.w_qcount * qcount_sum) \
                / n_tbl
        if n_detail:
            out = out + self.w_detail * detail_sum / n_detail
        return out
