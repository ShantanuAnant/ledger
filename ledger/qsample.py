"""Random window-aggregation queries, and their exact answers.

WHY THIS EXISTS. The window head predicts a fixed set of MARGINALS -- a rate
per table, a rate per category, a rate per numeric bucket, quantiles of counts
and of column sums -- and `win_readout` then composes them into whatever
functional a task asks for. Two things go wrong with that, both measured:

  * COMPOSITION. Filters compose as independent marginals, so on
    rel-trial site-success the `Primary` factor cancels exactly between the
    numerator and the denominator of a ratio and the filter does nothing.
  * CALIBRATION. Nothing in the objective ever sees the scalar a task is
    scored on, so the composed answer is free to collapse. On rel-f1
    driver-position the predictions had a spread of 0.667 against a true
    spread of 5.6 (Pearson r = 0.099), and on rel-trial study-adverse the
    output was numerically indistinguishable from predicting zero.

This module supervises the ANSWER instead. A random query is drawn from the
schema, its true value over the window is computed from the events the
batcher already materialises, and a query-conditioned head is trained to emit
that value's median under a pinball loss -- the estimator NMAE actually
rewards.

ZERO-SHOT. Queries are sampled from the schema, never from the task registry,
so the capability learned is "answer a window aggregation" rather than
"answer this task". The evaluation task's own signature is additionally
EXCLUDED from sampling (the trainer derives it from `--eval_entity`),
which makes the held-out query
held out in the strict sense: the model is never trained on it, only on
others of its kind. That is the leave-one-out reading of zero-shot, and it is
the reading this file is written to support. Sampling from `win_readout.QUERIES`
would be teaching to the test and is deliberately not done.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Aggregations, in the order their ids are embedded.
AGGS = ("count", "sum", "mean", "ratio")
# Filter kinds.
F_NONE, F_CAT, F_LE, F_GT = 0, 1, 2, 3
MAX_CAT = 1025          # matches schema.MAX_CATEGORY_VOCAB + 1
PAD_CAT = MAX_CAT       # padding id for a filter naming fewer than MAXV values
# A categorical filter may name a SET, not one value: rel-event
# user-attendance is `status in {yes, maybe}`. Encoding only singletons would
# force that task (and every other set-valued one) back onto the composed
# marginals, which is the path this head exists to replace.
MAXV = 4
# A categorical column wider than this has no per-category rate head, so a
# filter on it could not be answered from the marginals either.
CAT_RATE_MAX = 256


@dataclass(frozen=True)
class QSpec:
    """One window aggregation, in the same vocabulary `win_readout.Q` uses."""
    table: str
    agg: str
    f_kind: int = F_NONE
    f_col: int = -1          # index within the table's cat or num columns
    f_cats: tuple = ()       # vocabulary indices, for F_CAT (a SET)
    f_thr: float = 0.0       # STANDARDIZED threshold, for F_LE / F_GT
    v_col: int = -1          # numeric column index, for sum / mean

    def signature(self):
        """What `--query_exclude` compares on: the shape of the question,
        not the exact threshold (a task's 0.05 and a sampled 0.049 ask the
        same thing)."""
        return (self.table, self.agg, self.f_kind, self.f_col,
                tuple(sorted(self.f_cats)) if self.f_kind == F_CAT else (),
                self.v_col)


def _cols(schema, table):
    spec = schema.fact_tables[table]
    nums = [c for c in spec.columns if c.kind == "numeric"]
    cats = [c for c in spec.columns if c.kind == "categorical"]
    return nums, cats


def max_cols(schema) -> int:
    m = 1
    for t in schema.fact_tables:
        nums, cats = _cols(schema, t)
        m = max(m, len(nums), len(cats))
    return m


def sample(schema, rng: np.random.Generator, n: int,
           exclude: set | None = None, tries: int = 40) -> list:
    """`n` random queries. Tables with no columns still yield `count`."""
    exclude = exclude or set()
    tables = [t for t in schema.fact_tables]
    if not tables:
        return []
    out = []
    for _ in range(n):
        for _ in range(tries):
            t = tables[rng.integers(len(tables))]
            nums, cats = _cols(schema, t)
            choices = ["count", "ratio"] + (["sum", "mean"] if nums else [])
            agg = choices[rng.integers(len(choices))]
            v_col = int(rng.integers(len(nums))) if agg in ("sum", "mean") else -1

            # `ratio` is only meaningful against a filter: unfiltered it is
            # identically 1 and teaches nothing.
            kinds = [F_CAT] if agg == "ratio" and not nums else []
            if not kinds:
                kinds = [F_NONE, F_CAT, F_LE, F_GT]
                if agg == "ratio":
                    kinds = [F_CAT, F_LE, F_GT]
                if not cats:
                    kinds = [k for k in kinds if k != F_CAT]
                if not nums:
                    kinds = [k for k in kinds if k not in (F_LE, F_GT)]
            if not kinds:
                continue
            f_kind = int(kinds[rng.integers(len(kinds))])
            f_col, f_cats, f_thr = -1, (), 0.0
            if f_kind == F_CAT:
                usable = [j for j, c in enumerate(cats)
                          if c.cardinality <= CAT_RATE_MAX]
                if not usable:
                    continue
                f_col = int(usable[rng.integers(len(usable))])
                card = max(cats[f_col].cardinality, 1)
                k = 1 + int(rng.integers(min(MAXV, card)))
                f_cats = tuple(sorted(int(x) for x in rng.choice(
                    card, size=min(k, card), replace=False)))
            elif f_kind in (F_LE, F_GT):
                f_col = int(rng.integers(len(nums)))
                # thresholds live where the data is: standardized columns are
                # ~N(0, 1), so this covers the informative range instead of
                # sampling cuts that select everything or nothing.
                f_thr = float(rng.normal(0.0, 1.0))
            q = QSpec(t, agg, f_kind, f_col, f_cats, f_thr, v_col)
            if q.signature() in exclude:
                continue
            out.append(q)
            break
    return out


def encode(specs: list, schema, device) -> dict:
    """Integer ids + the threshold, for the query-conditioned head."""
    mc = max_cols(schema)
    tidx = {t: s.table_idx for t, s in schema.fact_tables.items()}
    t_id, a_id, k_id, c_id, cat_id, cat_m, v_id, thr = (
        [], [], [], [], [], [], [], [])
    for q in specs:
        ti = tidx[q.table]
        t_id.append(ti)
        a_id.append(AGGS.index(q.agg))
        k_id.append(q.f_kind)
        # numeric and categorical column slots must not collide
        base = ti * 2 * mc + (0 if q.f_kind in (F_LE, F_GT) else mc)
        c_id.append(base + q.f_col if q.f_col >= 0 else 0)
        vals = [min(max(int(x), 0), MAX_CAT - 1) for x in q.f_cats][:MAXV]
        cat_id.append(vals + [PAD_CAT] * (MAXV - len(vals)))
        cat_m.append([1.0] * len(vals) + [0.0] * (MAXV - len(vals)))
        v_id.append(ti * mc + q.v_col if q.v_col >= 0 else 0)
        thr.append(q.f_thr)
    L = lambda x: torch.as_tensor(x, dtype=torch.long, device=device)
    F = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    return dict(t=L(t_id), a=L(a_id), k=L(k_id), c=L(c_id), cat=L(cat_id),
                cat_mask=F(cat_m), v=L(v_id), thr=F(thr))


def _mask(spec: QSpec, feat_num, feat_cat) -> torch.Tensor:
    """Per-event membership of the filter. -> float [N]"""
    if spec.f_kind == F_NONE:
        return torch.ones(feat_num.shape[0], device=feat_num.device)
    if spec.f_kind == F_CAT:
        col = feat_cat[:, spec.f_col].long()
        want = torch.as_tensor(list(spec.f_cats), device=col.device,
                               dtype=col.dtype)
        return torch.isin(col, want).float()
    v = feat_num[:, spec.f_col]
    return (v <= spec.f_thr).float() if spec.f_kind == F_LE \
        else (v > spec.f_thr).float()


@torch.no_grad()
def answer(spec: QSpec, win: dict, schema, nq: int, device):
    """True value of `spec` over each query token's window.

    Returns `(target [nq], valid [nq] bool)`. `valid` is False where the
    answer is undefined -- a mean or a ratio with an empty denominator -- so
    those rows are not supervised rather than being taught a fabricated 0.

    ACCURACY. Per-column detail is capped at `--win_max_events` and carries
    weights restoring the exact count, so filtered counts, sums, means and
    ratios are unbiased ESTIMATES of the true window value. Unfiltered counts
    use `win["counts"]`, which is exact. This is the same approximation every
    existing detail term makes.
    """
    tspec = schema.fact_tables[spec.table]
    exact = win["counts"][:, tspec.table_idx].to(torch.float32)
    g = win["per_table"].get(spec.table)
    zeros = torch.zeros(nq, device=device)
    if g is None or not g["qi"].numel():
        if spec.agg == "count":
            return exact, torch.ones(nq, dtype=torch.bool, device=device)
        return zeros, torch.zeros(nq, dtype=torch.bool, device=device)

    qi = g["qi"]
    w = g.get("w")
    w = torch.ones_like(qi, dtype=torch.float32) if w is None else w.float()
    fn, fc = g["feat_num"].float(), g["feat_cat"]
    m = _mask(spec, fn, fc)
    wm = w * m
    cnt = zeros.clone().index_add_(0, qi, wm)
    tot = zeros.clone().index_add_(0, qi, w)

    if spec.agg == "count":
        if spec.f_kind == F_NONE:
            return exact, torch.ones(nq, dtype=torch.bool, device=device)
        return cnt, torch.ones(nq, dtype=torch.bool, device=device)
    if spec.agg == "ratio":
        return (cnt / tot.clamp(min=1e-9)).clamp(0.0, 1.0), tot > 0

    nums, _ = _cols(schema, spec.table)
    c = nums[spec.v_col]
    raw = fn[:, spec.v_col] * float(c.std) + float(c.mean)
    s = zeros.clone().index_add_(0, qi, wm * raw)
    if spec.agg == "sum":
        return s, torch.ones(nq, dtype=torch.bool, device=device)
    return s / cnt.clamp(min=1e-9), cnt > 0        # mean


def from_task_q(q, schema):
    """A `win_readout.Q` as a `QSpec`, or None when it cannot be expressed.

    Not everything maps. `agg="occur"` is a classification label, a filter
    naming several categories is a set membership this encoding has no slot
    for, and a query carrying BOTH a categorical and a numeric filter is a
    conjunction the head is not trained on. Returning None in those cases is
    what lets the caller fall back to the marginal readout rather than answer
    a different question -- the failure mode `_cat_rate`'s silent fallback
    caused on rel-avito user-clicks.
    """
    if q.agg not in AGGS or not q.tables:
        return None
    table = q.tables[0]
    if table not in schema.fact_tables:
        return None
    nums, cats = _cols(schema, table)
    filts = [f for f in (q.cat_in, q.cat_not, q.num_le, q.num_gt)
             if f is not None]
    if len(filts) > 1 or q.cat_not is not None:
        return None

    f_kind, f_col, f_cats, f_thr = F_NONE, -1, (), 0.0
    if q.cat_in is not None:
        _, col, allowed = q.cat_in
        if not allowed or len(allowed) > MAXV:
            return None
        j = next((i for i, c in enumerate(cats) if c.name == col), None)
        if j is None or cats[j].cardinality > CAT_RATE_MAX:
            return None
        vocab = cats[j].vocab
        idxs = []
        for v in allowed:
            cands = [str(v)]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                cands += [str(float(v)), str(int(v))]
            hit = next((vocab[c] for c in cands if c in vocab), None)
            if hit is None:
                return None          # a value the corpus never saw
            idxs.append(int(hit))
        f_kind, f_col, f_cats = F_CAT, j, tuple(sorted(idxs))
    elif q.num_le is not None or q.num_gt is not None:
        f = q.num_le if q.num_le is not None else q.num_gt
        _, col, thr = f
        j = next((i for i, c in enumerate(nums) if c.name == col), None)
        if j is None:
            return None
        # task thresholds are in RAW units; the head sees standardized ones
        f_kind = F_LE if q.num_le is not None else F_GT
        f_col = j
        f_thr = (float(thr) - float(nums[j].mean)) / (float(nums[j].std) or 1.0)

    v_col = -1
    if q.agg in ("sum", "mean"):
        if q.value is None:
            return None
        _, vcol = q.value
        v_col = next((i for i, c in enumerate(nums) if c.name == vcol), None)
        if v_col is None:
            return None
    return QSpec(table, q.agg, f_kind, f_col, f_cats, f_thr, v_col)


def signatures_of(q, schema) -> set:
    """Every sampler signature this task query could match, for exclusion.

    A task filter naming SEVERAL categories is one question to the task and
    several to the sampler, so each is excluded: training on any one of them
    would leak part of the evaluated query.
    """
    out = set()
    base = from_task_q(q, schema)
    if base is not None:
        out.add(base.signature())
    # `occur` is "did at least one such event happen" -- a thresholded COUNT,
    # and the sampler asks that same question under the name `count`. It is
    # not in AGGS, so without this normalisation the block below was skipped
    # and `signatures_of` returned the EMPTY SET for all 12 `occur` tasks --
    # every classification task in the registry. The exclusion then silently
    # did nothing and the evaluated query was sampled and trained on like any
    # other, forfeiting the zero-shot claim the docstring makes. `from_task_q`
    # still returns None for `occur` on purpose: it cannot be ANSWERED as a
    # count, only recognised as the same question.
    agg = "count" if q.agg == "occur" else q.agg
    if agg in AGGS and q.tables and q.tables[0] in schema.fact_tables:
        table = q.tables[0]
        nums, cats = _cols(schema, table)
        if q.cat_in is not None:
            _, col, allowed = q.cat_in
            j = next((i for i, c in enumerate(cats) if c.name == col), None)
            if j is not None:
                vocab = cats[j].vocab
                vc = base.v_col if base else -1
                for v in allowed:
                    for cand in {str(v), str(v).rstrip("0").rstrip(".")}:
                        if cand in vocab:
                            # holding out only the full SET would let the
                            # sampler train on {yes} and {maybe} separately,
                            # which is most of the evaluated question
                            out.add((table, agg, F_CAT, j,
                                     (int(vocab[cand]),), vc))
        # the unfiltered shape of the same aggregation is also the same
        # question when the filter matches everything, so hold it out too
        vc = base.v_col if base else -1
        out.add((table, agg, F_NONE, -1, (), vc))
    return out
