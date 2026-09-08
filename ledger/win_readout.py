"""Reading RelBench entity tasks off the WindowHead.

This replaces the qualified-hazard construction of `readout.py` for models
trained with `--window_head`. The difference is not cosmetic:

    readout.py    derives a window count from the WHEN head's next-gap
                  density, thinned by the WHERE head's next-table
                  distribution. Exact only for a Poisson process, and the
                  WHAT head emits a point estimate, so numeric predicates
                  (`position <= 3`, `p_value < 0.05`) and ratios have no
                  expression at all -- 5 of 21 entity tasks.

    this module   reads the window count DIRECTLY, from a head trained on
                  window counts. Numeric predicates are a sum over the
                  buckets that satisfy them; ratios are a quotient of two
                  rates; the NMAE median comes from a pinball-trained
                  quantile rather than from a Poisson approximation.

Everything here is still ZERO-SHOT with respect to the task: the only task
information used is `(tables, filter, aggregate, k, D)`, which is the
QUESTION, read off `make_table` in `relbench/tasks/*.py`. No labels from any
split are touched, and the training objective that produced the head is
self-supervised over pre-cutoff events.

WHAT IS EXACT AND WHAT IS NOT
-----------------------------
Exact given a calibrated head: E[N] for any table subset, categorical filter,
or numeric bucket union; and the quantiles of N and of column sums.

Approximate:
  * `P(N = 0) = exp(-rate)` and the Poisson tail for `P(N > k)` assume
    arrivals are conditionally Poisson. AUROC needs only a ranking, and any
    monotone function of `rate` gives the same ranking when k = 0 -- so for
    the churn/occurrence family this assumption costs nothing at all. It bites
    only for k >= 1 (rel-avito user-visits/clicks, rel-event user-ignore).
  * a FILTERED sum (rel-trial study-adverse: subjects_affected of `serious`
    and `deaths` rows only) is estimated as the unfiltered sum scaled by the
    filter's share of the rate. The marginals the head emits do not carry the
    joint of a categorical filter with a numeric column.
  * a threshold that falls inside a bucket is interpolated linearly within
    that bucket, which assumes values are locally uniform in it.

Each is named at the point of use rather than buried.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .model.window import sexp


# ---------------------------------------------------------------------------
# Query description
# ---------------------------------------------------------------------------

@dataclass
class Q:
    """One RelBench entity task, as a query over window aggregates.

    tables    fact tables whose rows can qualify
    cat_in    (table, column, {values})    keep rows whose value is in the set
    cat_not   (table, column, {values})    ... is NOT in the set
    num_le    (table, column, threshold)   keep rows with value <= threshold
    num_gt    (table, column, threshold)   ... > threshold
    agg       "occur" | "count" | "sum" | "mean" | "ratio"
    k         "more than k" threshold for occurrence labels
    absence   label 1 means the window is EMPTY (churn)
    value     column summed/averaged, for agg in {"sum", "mean"}
    denom     a second Q whose rate is the denominator, for agg == "ratio"
    needs     corpus flags the query's columns depend on ("denorm_fk",
              "child_aggs"). A query naming a derived column is unanswerable
              against a corpus built without it, and the failure mode is a
              filter that silently degrades to the unfiltered rate -- the
              exact bug that made user-clicks read 48.4 AUROC. Recorded here
              so the eval scripts and the registry test can refuse instead.
    """
    tables: list
    cat_in: tuple | None = None
    cat_not: tuple | None = None
    num_le: tuple | None = None
    num_gt: tuple | None = None
    agg: str = "occur"
    k: int = 0
    absence: bool = False
    value: tuple | None = None          # (table, column)
    denom: "Q | None" = None
    form: str = "auto"                  # arrival model; see `classify`
    needs: tuple = ()
    note: str = ""


# ---------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------

def _bucket_weights(win, name, col_idx, thr_std, op):
    """Per-bucket weight in [0, 1] for `value <op> threshold`. -> [B] tensor.

    A bucket entirely on the satisfying side gets 1, entirely on the other
    side gets 0, and the bucket the threshold falls inside gets the fraction
    of its width below the threshold. Linear interpolation inside a bucket
    assumes the values are locally uniform; with 8 quantile buckets that is a
    small assumption and it is the only alternative to snapping the threshold
    to a bucket boundary, which on a skewed column can move it a long way.
    """
    e = win.edges_of(name)[col_idx]                       # [B-1] standardized
    cen = win.centers_of(name)[col_idx]                   # [B]   standardized
    nb = e.numel() + 1
    lo = torch.cat([e.new_tensor([-float("inf")]), e])    # bucket lower edges
    hi = torch.cat([e, e.new_tensor([float("inf")])])
    # The threshold is computed in float64 from the column's mean/std while
    # the edges are a float32 buffer, so an edge that IS the threshold (very
    # common: `statusId <= 1` on a column where 1 is the modal value, so the
    # 1/8 quantile is exactly 1) misses an exact-equality test by ~1e-7. That
    # sent the modal bucket down the "straddled" branch and gave it weight
    # 0.5 -- half the finished races counted as DNFs, on the one task where
    # that bucket holds most of the mass.
    tol = 1e-5 * max(1.0, abs(thr_std))
    # An INTEGER column changes truth-value at T + 0.5, not at T, and the
    # within-bucket linear interpolation below is evaluated at whatever point
    # it is handed. Evaluating at T under-reads the straddled bucket by half
    # a step -- 17% of the filtered rate on rel-f1 driver-top3, concentrated
    # in one bucket, so it RE-RANKS. See WindowHead.integral_shift.
    thr_std = thr_std + win.integral_shift(name, col_idx)
    w = torch.zeros(nb, device=e.device, dtype=torch.float64)
    for b in range(nb):
        a, z, c = lo[b].item(), hi[b].item(), cen[b].item()
        if thr_std >= z - tol:
            frac = 1.0
        elif thr_std <= a + tol:
            frac = 0.0
        elif np.isfinite(a) and np.isfinite(z) and z > a:
            frac = (thr_std - a) / (z - a)
        else:
            # unbounded outer bucket genuinely straddled: decide by where its
            # MEAN sits, which is the only summary of it the head carries
            frac = 1.0 if c <= thr_std else 0.0
        w[b] = frac
    return w if op == "le" else (1.0 - w)


@torch.no_grad()
def rate(model, z, q: Q) -> torch.Tensor:
    """E[number of qualifying events in the window]. -> float64 [B]

    A single filter is EXACT: it is one of the marginals the head is trained
    on. Two filters on the same table compose as INDEPENDENT marginals,
    `total * P(cat) * P(num)`, because the head emits marginals and not their
    joint. rel-trial study-outcome is the one task that needs it -- "a Primary
    outcome analysis with p_value <= 0.05" -- and the alternative is worse in a
    measurable way: dropping either half counts every outcome type (a 4x
    dilution, and the reason the task scored 54.16 against 94.6).

    A first version of this function chained the filters with `elif`, so the
    second one was silently ignored. Composition is now explicit and a caller
    can see the assumption.
    """
    win = model.win
    out = None
    for name in q.tables:
        if name not in win.schema.fact_tables:
            continue
        total = win.log_rate(z, name).double().exp()      # [B]
        parts = []
        for filt, kind in ((q.cat_in, "cat_in"), (q.cat_not, "cat_not"),
                           (q.num_le, "le"), (q.num_gt, "gt")):
            if filt is None or filt[0] != name:
                continue
            if kind == "cat_in":
                parts.append(_cat_rate(win, z, name, filt, False, total))
            elif kind == "cat_not":
                parts.append(_cat_rate(win, z, name, filt, True, total))
            else:
                parts.append(_num_rate(win, z, name, filt, kind, total))
        if not parts:
            r = total
        elif len(parts) == 1:
            # exactly the marginal the head was trained on -- taken as-is, so
            # every query that existed before composition was added is
            # bit-identical to what it was
            r = parts[0]
        else:
            denom = total.clamp(min=1e-12)
            r = total
            for part in parts:
                r = r * (part / denom).clamp(0.0, 1.0)
        out = r if out is None else out + r
    if out is None:
        raise KeyError(f"no fact table of {q.tables} is in the schema")
    return out.clamp(min=1e-9)


def _cat_rate(win, z, name, filt, negate, total):
    _, col, allowed = filt
    j, spec = win.categorical_index(name, col)
    lg = win.log_rate_cat(z, name, j) if j is not None else None
    if lg is None:
        # column absent, or wider than CAT_RATE_MAX so it has no rate head.
        # Returning the unfiltered rate is the honest degradation: the query
        # becomes "any event of this table", which is what readout.py's
        # thinning also falls back to. It is visible in the eval output.
        return total
    # Vocabulary keys are `str(value)` of the RAW column, so a numerically
    # typed column stores "1.0" while a task predicate naturally says 1.
    # Looking up "1" then misses, `_cat_rate` falls back to the unfiltered
    # rate, and the readout answers a DIFFERENT QUESTION with no error --
    # rel-avito user-clicks read 48.4 AUROC that way, i.e. "did this user
    # search at all" instead of "did they click". Try the several spellings a
    # value can have rather than assuming one.
    idx = set()
    miss = []
    for v in allowed:
        cands = [str(v)]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            cands += [str(float(v)), str(int(v))]
        elif isinstance(v, bool):
            cands += [str(int(v)), str(float(int(v)))]
        hit = next((spec.vocab[cc] for cc in cands if cc in spec.vocab), None)
        if hit is None:
            miss.append(v)
        else:
            idx.add(hit)
    if miss:
        raise KeyError(
            f"{name}.{col}: no vocabulary entry for {miss}; the filter would "
            f"silently degrade to the unfiltered rate. Vocabulary sample: "
            f"{list(spec.vocab)[:8]}")
    idx = torch.as_tensor(sorted(i for i in idx if i < lg.shape[-1]),
                          device=z.device)
    if idx.numel() == 0:
        return total * 0 + 1e-9
    part = lg.double().exp()[:, idx].sum(-1)
    return (total - part).clamp(min=1e-9) if negate else part


def _num_rate(win, z, name, filt, op, total):
    _, col, thr = filt
    j = win.numeric_index(name, col)
    lg = win.log_rate_num(z, name)
    if j is None or lg is None:
        return total
    thr_std = win.standardize(name, col, float(thr))
    w = _bucket_weights(win, name, j, thr_std, op)         # [n_buckets]
    return (lg[:, j, :].double().exp() * w).sum(-1)


# ---------------------------------------------------------------------------
# Readouts
# ---------------------------------------------------------------------------

def p_more_than(rate, k: int):
    """P(N > k) under a Poisson with this mean. k = 0 is monotone in `rate`,
    so for occurrence labels the Poisson assumption does not affect AUROC."""
    terms = torch.ones_like(rate)
    acc = torch.ones_like(rate)
    for j in range(1, k + 1):
        acc = acc * rate / j
        terms = terms + acc
    return (1.0 - torch.exp(-rate) * terms).clamp(0.0, 1.0)


def _binom_tail(n, p, k: int):
    """P(Binomial(n, p) > k), with a real-valued n. -> [B]

    Why this exists. A FILTERED occurrence label ("did any of this driver's
    results fail to finish") has two factors: how many events happen, and what
    fraction of them qualify. Poisson thinning collapses them --
    `P(N_q = 0) = exp(-lambda_q)` -- and that identity is exact only when the
    arrival count is itself Poisson. It is not, for a scheduled process: every
    active F1 driver enters almost exactly the same number of races in 30
    days, so N is nearly a constant and the informative quantity is the
    per-event failure probability, which Poisson thinning smears back into the
    rate. Measured on rel-f1 driver-dnf: Poisson 51.8 AUROC, this 61.8, the
    bare per-event probability 63.0.

    Generalized binomial coefficients keep n continuous; k never exceeds 2 in
    the benchmark.
    """
    p = p.clamp(0.0, 1.0)
    n = n.clamp(min=0.0)
    q1 = (1.0 - p).clamp(min=1e-12)
    terms = q1 ** n                                    # j = 0
    acc = torch.ones_like(p)
    for j in range(1, k + 1):
        acc = acc * (n - (j - 1)) / j
        terms = terms + acc.clamp(min=0.0) * p ** j * q1 ** (n - j)
    return (1.0 - terms).clamp(0.0, 1.0)


def _is_filtered(q: Q) -> bool:
    return any(x is not None for x in (q.cat_in, q.cat_not,
                                       q.num_le, q.num_gt))


@torch.no_grad()
def classify(model, z, q: Q, form: str = "auto") -> torch.Tensor:
    """Probability of the positive label. Higher = label 1 more likely.

    `form` selects the arrival model, and which one is right is a property of
    the DATA, not of the task:

      poisson    P(N_q > k) with N_q ~ Poisson(rate_q). Correct when arrivals
                 are Poisson; the only option for an unfiltered query, where
                 the per-event probability is 1 by construction.
      binomial   P(Binom(rate_total, rate_q/rate_total) > k). Correct when the
                 arrival COUNT is roughly known and each event qualifies
                 independently. Reduces to poisson as p -> 0.
      fraction   the per-event probability alone. The n -> constant limit of
                 binomial; use it when the head's count estimate is noisier
                 than it is informative.
      auto       binomial for a filtered query, poisson otherwise.
    """
    rq = rate(model, z, q)
    if form == "auto":
        form = q.form
    if form == "auto":
        # Measured rule, not a guess. A "more than k" label (k >= 1) asks for
        # an absolute COUNT, and the Poisson tail on the qualifying rate is
        # the right object -- rel-event user-ignore reads 84.25 that way
        # against 77.40 for the binomial. A k = 0 occurrence label on a
        # FILTERED subset asks whether any one event qualifies, and there the
        # per-event probability is the right object because the count is
        # near-constant and often ANTI-correlated with the label: on rel-event
        # user-repeat the historical yes-fraction scores 70.3 alone while the
        # event count scores 41.4. Individual tasks override via `Q.form`.
        form = "poisson" if q.k >= 1 else (
            "fraction" if _is_filtered(q) else "poisson")
    if form == "poisson" or not _is_filtered(q):
        s = p_more_than(rq, q.k)
    else:
        tot = rate(model, z, Q(tables=q.tables))
        p = (rq / tot.clamp(min=1e-9)).clamp(0.0, 1.0)
        if form == "fraction" and q.k == 0:
            s = p
        else:
            s = _binom_tail(tot, p, q.k)
    return (1.0 - s) if q.absence else s


def resolvable(q: "Q", schema) -> tuple:
    """-> (query with unresolvable filters removed, list of what was dropped).

    A filter naming a column the corpus does not have would fall through
    `_cat_rate`/`_num_rate`'s "column absent" branch and answer the
    UNFILTERED question with no error. This makes the degradation explicit, so
    a caller can either refuse (`scripts/eval_window.py`, which reports
    headline numbers) or proceed loudly (the in-training eval of a control arm
    deliberately built without the denormalized columns).
    """
    import dataclasses
    dropped, patch = [], {}
    for field in ("cat_in", "cat_not", "num_le", "num_gt"):
        filt = getattr(q, field)
        if filt is None:
            continue
        tname, col = filt[0], filt[1]
        spec = schema.fact_tables.get(tname)
        want = "categorical" if field.startswith("cat") else "numeric"
        if spec is None or not any(c.name == col and c.kind == want
                                   for c in spec.columns):
            dropped.append(f"{tname}.{col}")
            patch[field] = None
    if q.value is not None:
        tname, col = q.value
        spec = schema.fact_tables.get(tname)
        if spec is None or not any(c.name == col and c.kind == "numeric"
                                   for c in spec.columns):
            return None, [f"{tname}.{col} (the summed column)"]
    if not dropped:
        return q, []
    return dataclasses.replace(q, **patch), dropped


def _tau_index(win, tau=0.5):
    return int(np.argmin(np.abs(np.asarray(win.taus) - tau)))


@torch.no_grad()
def count_median(model, z, q: Q) -> torch.Tensor:
    """Conditional MEDIAN of the qualifying count -- the NMAE-optimal point
    estimate. Uses the pinball-trained quantile head when the query is a whole
    fact table (which is what the head was trained on) and falls back to the
    Poisson median of the filtered rate otherwise."""
    win = model.win
    unfiltered = (q.cat_in is None and q.cat_not is None
                  and q.num_le is None and q.num_gt is None
                  and len(q.tables) == 1)
    if unfiltered and q.tables[0] in win.schema.fact_tables:
        qc = win.q_count(z, q.tables[0])[:, _tau_index(win)].double()
        return torch.expm1(qc.clamp(min=0.0))
    r = rate(model, z, q)
    # median of Poisson(r); floor form is exact enough below r ~ 10 and this
    # is a count of events in a window
    return torch.floor(r + 1.0 / 3.0 - 0.02 / r.clamp(min=1e-9)).clamp(min=0.0)


@torch.no_grad()
def sum_median(model, z, q: Q) -> torch.Tensor:
    """Conditional median of `sum of q.value` over qualifying events."""
    win = model.win
    name, col = q.value
    j = win.numeric_index(name, col)
    if j is None:
        raise KeyError(f"{name}.{col} is not a numeric column of the corpus")
    qs = win.q_sum(z, name)[:, j, _tau_index(win)].double()
    med = sexp(qs)
    if not _is_filtered(q):
        return med.clamp(min=0.0)
    # Filtered sum: the head emits the marginal sum over the whole table and
    # the marginal rate per category, not their joint. Scaling the sum by the
    # filter's share of the rate assumes the mean value per event is the same
    # inside and outside the filter. Stated, not hidden.
    full = Q(tables=[name])
    share = (rate(model, z, q) / rate(model, z, full)).clamp(0.0, 1.0)
    return (med * share).clamp(min=0.0)


@torch.no_grad()
def mean_value(model, z, q: Q) -> torch.Tensor:
    """E[value | a qualifying event happens], from the bucketed distribution.

    `sum_b rate_b * center_b / sum_b rate_b` -- the head's own discretisation
    of the column, in raw units. This is what rel-f1 driver-position asks for
    (mean finishing position over the window).
    """
    win = model.win
    name, col = q.value
    j = win.numeric_index(name, col)
    lg = win.log_rate_num(z, name)
    if j is None or lg is None:
        raise KeyError(f"{name}.{col} has no bucket head")
    r = lg[:, j, :].double().exp()                        # [B, n_buckets]
    c = win.centers_of(name, raw=True)[j].double()        # [n_buckets]
    return (r * c).sum(-1) / r.sum(-1).clamp(min=1e-9)


@torch.no_grad()
def ratio(model, z, q: Q) -> torch.Tensor:
    """rate(q) / rate(q.denom). Both are exact expectations, so the quotient
    is a ratio of estimates rather than an estimate of the ratio -- a
    first-order approximation, and the one every CTR model makes."""
    return (rate(model, z, q)
            / rate(model, z, q.denom).clamp(min=1e-9)).clamp(0.0, 1.0)


@torch.no_grad()
def predict_query(model, z, q: Q):
    """Read the QUERY-CONDITIONED head (ledger/qsample.py) instead of composing
    marginals. -> [B] or None when the checkpoint has no such head or the
    query cannot be expressed in the sampler's vocabulary.

    None is a refusal, not a fallback to something similar: answering a
    conjunction the head was never trained on would be a different question
    silently returned as if it were this one.
    """
    win = model.win
    if getattr(win, "query_head", None) is None:
        return None
    from . import qsample as qs
    spec = qs.from_task_q(q, win.schema)
    if spec is None:
        return None
    pred = win.query_predict(z, qs.encode([spec], win.schema, z.z.device))
    med = pred[:, _tau_index(win)].double()
    if spec.agg == "ratio":
        return med.clamp(0.0, 1.0)
    out = sexp(med)
    return out if spec.agg == "mean" else out.clamp(min=0.0)


@torch.no_grad()
def predict(model, z, q: Q, form: str = "auto") -> torch.Tensor:
    if q.agg == "occur":
        return classify(model, z, q, form)
    if q.agg == "count":
        return count_median(model, z, q)
    if q.agg == "sum":
        return sum_median(model, z, q)
    if q.agg == "mean":
        return mean_value(model, z, q)
    if q.agg == "ratio":
        return ratio(model, z, q)
    raise ValueError(q.agg)


# ---------------------------------------------------------------------------
# The task registry, read off `make_table` in relbench/tasks/*.py.
# Only the QUESTION is encoded: table subset, column filter, aggregate,
# horizon. No labels, no split statistics.
# ---------------------------------------------------------------------------

QUERIES: dict[tuple[str, str], Q] = {
    # -- classification -------------------------------------------------
    ("rel-amazon", "user-churn"): Q(["review"], absence=True,
                                    note="no review in 91d"),
    ("rel-amazon", "item-churn"): Q(["review"], absence=True),
    ("rel-hm", "user-churn"): Q(["transactions"], absence=True),
    ("rel-stack", "user-engagement"): Q(["votes", "posts", "comments"]),
    ("rel-stack", "user-badge"): Q(["badges"]),
    ("rel-avito", "user-visits"): Q(["VisitStream"], k=1),
    # WAS NOT REACHABLE, and scored EXACTLY 50.00 for a structural reason:
    # `SearchStream` has no foreign key to `UserInfo`, so a user reaches it
    # only through `UserInfo -> SearchInfo -> SearchStream` and those events
    # never entered a user's history at all.
    #
    # CHILD-AGGREGATE DENORMALIZATION closes it without moving any event.
    # `SearchInfo` -- which IS in the user's history -- now carries
    # `SearchStream@SearchID.IsClick=1.0.n`, the number of clicked ads in that
    # search, as one of its own numeric columns. The query is then a numeric
    # predicate on a table the head already models: how many of this user's
    # searches in the window have at least one click. The label counts CLICKS
    # rather than clicking SEARCHES, and those differ only when one search
    # yields two clicks (261 searches of 1.96M), so the approximation is
    # negligible and is stated rather than hidden. Requires --child_aggs.
    ("rel-avito", "user-clicks"): Q(
        ["SearchInfo"], k=1, form="poisson",
        num_gt=("SearchInfo", "SearchStream@SearchID.IsClick=1.0.n", 0.5),
        needs=("child_aggs",),
        note="clicks reached through SearchInfo's child aggregate"),
    ("rel-event", "user-repeat"): Q(
        ["event_attendees"],
        cat_in=("event_attendees", "status", {"yes", "maybe"})),
    ("rel-event", "user-ignore"): Q(
        ["event_attendees"],
        cat_in=("event_attendees", "status", {"invited"}), k=2),
    # `statusId != 1` is a NUMERIC column (results has 9 numeric columns and
    # statusId is one of them), so readout.py's categorical filter for this
    # task silently matched nothing and left the predicate at "any result".
    ("rel-f1", "driver-dnf"): Q(
        ["results"], num_gt=("results", "statusId", 1.0),
        note="a race not finished"),
    # MIN(position) <= 3 over the window == at least one qualifying event with
    # position <= 3. Exactly a bucket-union rate; not expressible at all from
    # a point-estimate WHAT head.
    ("rel-f1", "driver-top3"): Q(
        ["qualifying"], num_le=("qualifying", "position", 3.0)),
    # MIN(p_value) <= 0.05 over PRIMARY outcome analyses in the window.
    #
    # `outcome_type` is a column of `outcomes`, not of the event, so until
    # FK-target denormalization this predicate counted analyses of EVERY
    # outcome type -- and only 25.6% of them are Primary (65,247 of 254,420),
    # a 4x dilution that left the readout at 54.16 against a raw-analysis-count
    # baseline of 55.09. `outcome_id.outcome_type` is now a categorical column
    # of `outcome_analyses` itself.
    #
    # The two filters compose as INDEPENDENT marginals (see `rate`); the head
    # carries the marginal rate per outcome type and the marginal rate per
    # p_value bucket, not their joint. The task's third condition
    # (`p_value_modifier != '>'`) is deliberately left out: it moves little
    # mass and every extra marginal compounds the independence assumption.
    # Requires --denorm_fk.
    ("rel-trial", "study-outcome"): Q(
        ["outcome_analyses"], form="binomial",
        cat_in=("outcome_analyses", "outcome_id.outcome_type", {"Primary"}),
        num_le=("outcome_analyses", "p_value", 0.05),
        needs=("denorm_fk",),
        note="Primary outcome analyses with p_value <= 0.05"),

    # -- regression -----------------------------------------------------
    ("rel-stack", "post-votes"): Q(
        ["votes"], cat_in=("votes", "VoteTypeId", {2}), agg="count"),
    ("rel-event", "user-attendance"): Q(
        ["event_attendees"],
        cat_in=("event_attendees", "status", {"yes", "maybe"}), agg="count"),
    ("rel-hm", "item-sales"): Q(
        ["transactions"], agg="sum", value=("transactions", "price")),
    ("rel-trial", "study-adverse"): Q(
        ["reported_event_totals"],
        cat_in=("reported_event_totals", "event_type",
                {"serious", "deaths"}),
        agg="sum", value=("reported_event_totals", "subjects_affected")),
    ("rel-f1", "driver-position"): Q(
        ["results"], agg="mean", value=("results", "positionOrder")),
    ("rel-avito", "ad-ctr"): Q(
        ["SearchStream"], cat_in=("SearchStream", "IsClick", {1}),
        agg="ratio", denom=Q(["SearchStream"])),
    # LTV = sum of `product.price` over the reviews in the window. `price` is
    # an attribute of a DIMENSION table, so `review` had no numeric column at
    # all and neither task was merely weak -- it was inexpressible. FK-target
    # denormalization puts the price on the review event itself, and the
    # WindowHead's pinball-trained sum quantile then answers both. The two
    # tasks differ only in which entity's history is read. Requires
    # --denorm_fk.
    ("rel-amazon", "user-ltv"): Q(
        ["review"], agg="sum", value=("review", "product_id.price"),
        needs=("denorm_fk",)),
    ("rel-amazon", "item-ltv"): Q(
        ["review"], agg="sum", value=("review", "product_id.price"),
        needs=("denorm_fk",)),

    # SUCCESS RATE of a trial site over the next year. Was UNEXPRESSIBLE
    # until target-side path expansion (corpus._expand_path): the qualifying
    # `outcome_analyses` key on `nct_id -> studies` and so never entered a
    # facility's history at all -- measured 0% of facility histories held
    # one, against 90.5% holding `facilities_studies` and nothing else. With
    # `facilities:facilities_studies:outcome_analyses` they reach 26.6%.
    #
    # TWO APPROXIMATIONS, both stated rather than hidden.
    #
    # 1. The label is `SUM(is_successful)/COUNT(...)` over TRIALS, where a
    #    trial counts as successful when the MIN p_value of its Primary
    #    analyses is < 0.05. That is a group-wise MIN then an average over
    #    groups; the Q language has no group-by, so this estimates the
    #    analysis-level rate `P(p_value <= 0.05)` instead. The two agree when
    #    a trial contributes one Primary analysis and diverge as it
    #    contributes more.
    # 2. `cat_in` and `num_le` compose as INDEPENDENT marginals (see `rate`),
    #    so `total * P(Primary) * P(p<=.05)` over `total * P(Primary)`
    #    CANCELS the Primary factor exactly: the estimate is the marginal
    #    significant-analysis rate over every outcome type, not over Primary
    #    ones. The filter is kept because it names the intended question and
    #    becomes real the moment `rate` carries a joint, but it is doing no
    #    work today and must not be read as if it were.
    #
    # The task's third condition (`p_value_modifier != '>'`) is left out for
    # the reason study-outcome leaves it out: it moves little mass and every
    # extra marginal compounds the independence assumption.
    ("rel-trial", "site-success"): Q(
        ["outcome_analyses"], agg="ratio",
        cat_in=("outcome_analyses", "outcome_id.outcome_type", {"Primary"}),
        num_le=("outcome_analyses", "p_value", 0.05),
        denom=Q(["outcome_analyses"],
                cat_in=("outcome_analyses", "outcome_id.outcome_type",
                        {"Primary"})),
        needs=("denorm_fk", "path_expand"),
        note="significant Primary analyses / Primary analyses, over the "
             "studies the facility ran"),
}

# Deliberately absent, with the reason. All three entries have now been
# removed by construction rather than by hope: two on 2026-08-25 by the
# denormalization changes, the last on 2026-08-29.
#
#   rel-trial site-success -- RESOLVED 2026-08-29 by target-side path
#     expansion (`corpus._expand_path`), the analogue of `--retarget` this
#     entry used to ask for and record as "Not built". A facility's history
#     held `facilities_studies` rows and nothing else (measured: 90.5% of
#     histories, and 0% holding a single `outcome_analyses` row), so the
#     readout was computing a rate over events the entity had never seen.
#     Filing the analyses of the studies a facility ran into that facility's
#     history costs 4,363,569 references on this dataset -- the estimate here
#     was ~1.7M, so it was low by 2.5x -- and lifts `outcome_analyses`
#     coverage to 26.6% of facility histories. The reference is kept only
#     when the bridge row predates the event, so the leakage rule that made
#     CHILD-AGGREGATE denormalization unusable here does not apply: the event
#     carries its own timestamp and `history(before=cut)` still filters it.
#
#     WHAT IS STILL TRUE, AND IS THE RISK. The original measurement stands:
#     analyses arrive a median of THREE YEARS after `facilities_studies.date`
#     and only 7.6% land inside the task's 365-day horizon. Path expansion
#     fixes what the model can SEE (past analyses are now in history); it
#     does not make the FUTURE window dense. Expect a low predicted arrival
#     rate, which is the regime where the window readout has been measured to
#     collapse onto the constant (rel-trial study-adverse at rate 0.003,
#     rel-event user-attendance at 0.033). Being expressible is necessary
#     here, not sufficient.
UNEXPRESSIBLE: dict = {}
