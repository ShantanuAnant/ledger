"""Qualified-hazard readout: entity tasks as functionals of the event process.

ARCHITECTURE 8.1/8.2 said classification and regression are queries against
the generative model. This is that query layer, and it is one object rather
than three.

THE CONSTRUCTION
----------------
Every RelBench entity task aggregates the events an entity emits in a forward
window `(t, t+D]`, keeping only events that satisfy a predicate (a table
subset, optionally a column filter). Write:

    Lambda = H(elapsed + D) - H(elapsed)          conditional cumulative hazard
                                                  of ANY next event, where
                                                  H(w) = -log S(w) comes
                                                  straight from the WHEN head

    q      = sum_T P_WHERE(T | h) * P_WHAT(filter | T, h)     thinning prob.

    mu     = q * Lambda                    hazard of the QUALIFYING process

Under random thinning of the arrival process, keeping each event independently
with probability q scales the cumulative hazard by q. Then a single mu answers
all three benchmark families:

    P(no qualifying event) = exp(-mu)               churn / engagement / badge
    P(N > k)               = 1 - PoissonCDF(k; mu)  "more than k visits"
    E[N]                   = mu                     count regression
    E[sum v]               = mu * E_WHAT[v | T]     sum regression (LTV, sales)

WHY THIS AND NOT A ROLLOUT
--------------------------
ARCHITECTURE 8.4 proposes sampling trajectories. Over a 91- or 365-day window
that compounds error across four heads and needs a re-encode per simulated
event, to estimate a quantity the metric only needs a RANKING of. The hazard
form is closed, deterministic, and costs one forward pass.

THE APPROXIMATION, STATED PLAINLY
---------------------------------
`S_q = S^q` is exact when the arrival process is Poisson: then
P(no qualifying event) = G_N(1-q) = exp(-Lambda*q). For a general renewal
process it is first-order, in the same spirit as the first-event
approximation ARCHITECTURE 8.1 already makes. Two further approximations:
q is evaluated at the query state h and held fixed over the window, and marks
are treated as independent of gaps. All three are testable against the
measured metric rather than assumed -- and note the whole thing collapses to
the exact, already-validated survival readout when a database has ONE fact
table, because then q = 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Predicate:
    """Which future events count, for one task.

    `tables`  -- fact tables whose rows can qualify.
    `col_in`  -- optional (table, column, {allowed values}) categorical filter.
    `col_not` -- same, negated. rel-f1 driver-dnf is `statusId != 1`.

    This is public task metadata of the same kind as `dst_entity_table` and
    `timedelta`: it describes the QUESTION, never the answer, and uses no
    labels from any split.
    """
    tables: list[str]
    col_in: tuple | None = None
    col_not: tuple | None = None
    note: str = ""


def _table_index(schema, name: str) -> int:
    return schema.fact_tables[name].table_idx


@torch.no_grad()
def thinning_prob(model, schema, h, pred: Predicate) -> torch.Tensor:
    """q = P(the next event satisfies the predicate | h).   -> [B]"""
    logp = F.log_softmax(model.where.proj(h).float(), dim=-1)      # [B, T]
    total = torch.zeros(h.shape[0], device=h.device, dtype=torch.float64)
    for name in pred.tables:
        if name not in schema.fact_tables:
            continue
        p_tbl = logp[:, _table_index(schema, name)].exp().double()
        p_col = torch.ones_like(p_tbl)
        for filt, negate in ((pred.col_in, False), (pred.col_not, True)):
            if filt is None or filt[0] != name:
                continue
            _, col_name, allowed = filt
            spec = schema.fact_tables[name]
            cats = [c for c in spec.columns if c.kind == "categorical"]
            j = next((k for k, c in enumerate(cats) if c.name == col_name), None)
            if j is None or name not in model.what.cat_out:
                continue                     # column dropped: leave p_col = 1
            z = model.what.trunk(h)
            lp = F.log_softmax(model.what.cat_out[name][j](z).float(), -1)
            idx = [cats[j].vocab.get(str(v), 0) for v in allowed]
            idx = torch.as_tensor(sorted(set(idx)), device=h.device)
            m = lp[:, idx].exp().sum(-1).double().clamp(0, 1)
            p_col = (1.0 - m) if negate else m
        total = total + p_tbl * p_col
    return total.clamp(1e-9, 1.0)


def empirical_q(corpus, entity_table, rows, cuts, pred: Predicate,
                prior_w: float = 5.0, max_len: int = 512) -> np.ndarray:
    """Composition of the entity's OWN recent history: fraction of its events
    that satisfy the predicate, smoothed toward the corpus-wide rate.

    Why this exists. `thinning_prob` asks the WHERE head what the NEXT event
    will be, then holds that fixed across a window of up to a year. Those are
    different questions whenever composition drifts with activity, and
    rel-stack user-badge shows the failure sharply: a user whose recorded
    history is badge-dominated is an INACTIVE one, while an active user's next
    event is a vote or a comment -- so next-event q_badge is LOW exactly for
    the users who go on to earn badges, and the readout scores below random.

    An entity's own historical mix is a direct estimate of the window
    composition and does not depend on the calibration of a head that was
    trained at 0.1 loss weight. It is still leakage-free: pre-cutoff rows only.
    """
    want = {schema_idx for schema_idx in
            (corpus.schema.fact_tables[t].table_idx for t in pred.tables
             if t in corpus.schema.fact_tables)}
    # corpus-wide qualifying rate, used as the smoothing prior
    base = float(np.isin(corpus.table_idx, list(want)).mean()) if want else 0.0
    out = np.full(len(rows), base, dtype=np.float64)
    for i, (r, cut) in enumerate(zip(rows, cuts)):
        ids = corpus.history(entity_table, int(r), before=int(cut))[-max_len:]
        if len(ids) == 0:
            continue
        hit = float(np.isin(corpus.table_idx[ids], list(want)).sum())
        out[i] = (hit + prior_w * base) / (len(ids) + prior_w)
    return np.clip(out, 1e-9, 1.0)


@torch.no_grad()
def qualified_hazard(model, schema, h, elapsed, delta, pred: Predicate,
                     q_override=None, q_blend: float = 0.0):
    """-> (mu, Lambda, q). mu is the qualifying process's cumulative hazard."""
    e = torch.as_tensor(elapsed, device=h.device, dtype=torch.float32)
    d = torch.as_tensor(float(delta), device=h.device, dtype=torch.float32)
    # H(e+D) - H(e); >= 0 because survival is non-increasing
    lam = (model.when.log_survival(h, e)
           - model.when.log_survival(h, e + d)).double().clamp(min=0.0)
    q = thinning_prob(model, schema, h, pred)
    if q_override is not None:
        qo = torch.as_tensor(q_override, device=h.device, dtype=torch.float64)
        # q_blend = 0 -> empirical only; 1 -> WHERE head only
        q = qo ** (1.0 - q_blend) * q ** q_blend if q_blend > 0 else qo
    return q * lam, lam, q


def p_no_event(mu):
    """P(N = 0) = exp(-mu). Churn score: higher = more likely to churn."""
    return torch.exp(-mu)


def p_more_than(mu, k: int):
    """P(N > k) for the thinned process. rel-avito user-visits is k=1."""
    terms = torch.ones_like(mu)
    acc = torch.ones_like(mu)
    for j in range(1, k + 1):
        acc = acc * mu / j
        terms = terms + acc
    return (1.0 - torch.exp(-mu) * terms).clamp(0.0, 1.0)


def expected_count(mu):
    """E[N] = mu. Count regression (rel-event user-attendance, post-votes)."""
    return mu


def poisson_median(mu):
    """Median of Poisson(mu). THE metric is NMAE = MAE / train-std, and MAE is
    minimised by the conditional MEDIAN, not the mean -- the leaderboard's own
    baselines show it (Entity Median 0.4278 beats Entity Mean 0.4551). Using
    E[N] here would be optimising the wrong functional, which is what
    ARCHITECTURE 8.2 currently specifies.

    Wilson-Hilferty style approximation, floored at 0 and exact enough for a
    count whose mean is usually below 10.
    """
    import numpy as _np
    m = _np.asarray(mu, dtype=_np.float64)
    return _np.maximum(_np.floor(m + 1.0 / 3.0 - 0.02 / _np.maximum(m, 1e-9)),
                       0.0)


# Regression tasks expressible as a functional of the qualifying arrivals.
# `value` names the column summed over those events; None means the target is
# the COUNT itself. Ratio targets (rel-avito ad-ctr, rel-f1 driver-position,
# rel-trial site-success) are NOT here -- a ratio is not a functional of the
# arrival count and needs its own readout.
REG_PREDICATES: dict[tuple[str, str], tuple[Predicate, str | None]] = {
    ("rel-stack", "post-votes"): (Predicate(["votes"], note="upvotes"), None),
    ("rel-event", "user-attendance"): (
        Predicate(["event_attendees"],
                  col_in=("event_attendees", "status", {"yes", "maybe"}),
                  note="events attended"), None),
    ("rel-amazon", "user-ltv"): (Predicate(["review"]), "price"),
    ("rel-amazon", "item-ltv"): (Predicate(["review"]), "price"),
    ("rel-hm", "item-sales"): (Predicate(["transactions"]), "price"),
    ("rel-trial", "study-adverse"): (
        Predicate(["reported_event_totals"]), "subjects_affected"),
}


@torch.no_grad()
def expected_value(model, schema, h, table: str, column: str):
    """E[column | next event is in `table`], de-standardised. -> [B]"""
    spec = schema.fact_tables[table]
    nums = [c for c in spec.columns if c.kind == "numeric"]
    j = next((k for k, c in enumerate(nums) if c.name == column), None)
    if j is None or table not in model.num_out_tables():
        return None
    z = model.what.trunk(h)
    v = model.what.num_out[table](z)[:, j].double()
    return v * nums[j].std + nums[j].mean


# ---------------------------------------------------------------------------
# Task predicates, read off the `make_table` SQL in relbench/tasks/*.py.
# Only the QUESTION is encoded here -- table subset and column filter -- never
# a label. Tasks whose target is an attribute of a future row rather than a
# count of events (rel-trial study-outcome) or a RATIO (rel-avito ad-ctr,
# rel-f1 driver-position, rel-trial site-success) are deliberately absent:
# they are not functionals of the arrival count and need a different readout.
# ---------------------------------------------------------------------------
PREDICATES: dict[tuple[str, str], Predicate] = {
    ("rel-amazon", "user-churn"): Predicate(["review"], note="no review in 91d"),
    ("rel-amazon", "item-churn"): Predicate(["review"], note="no review in 91d"),
    ("rel-hm", "user-churn"): Predicate(["transactions"], note="no txn in 7d"),
    ("rel-stack", "user-engagement"): Predicate(
        ["votes", "posts", "comments"], note="any vote/post/comment"),
    ("rel-stack", "user-badge"): Predicate(["badges"], note="any badge"),
    ("rel-avito", "user-visits"): Predicate(
        ["VisitStream"], note="more than one ad visited"),
    ("rel-avito", "user-clicks"): Predicate(
        ["SearchStream"], col_in=("SearchStream", "IsClick", {1, "1", True}),
        note="more than one ad clicked"),
    ("rel-event", "user-repeat"): Predicate(
        ["event_attendees"],
        col_in=("event_attendees", "status", {"yes", "maybe"}),
        note="attends an event"),
    ("rel-event", "user-ignore"): Predicate(
        ["event_attendees"], col_in=("event_attendees", "status", {"invited"}),
        note="ignores more than 2 invitations"),
    ("rel-f1", "driver-dnf"): Predicate(
        ["results"], col_not=("results", "statusId", {1, "1"}),
        note="a race not finished"),
    ("rel-trial", "study-outcome"): Predicate(
        ["outcome_analyses"],
        note="APPROXIMATE: any outcome analysis reported. The true label is "
             "p_value < 0.05, a NUMERIC filter the WHAT head cannot give a "
             "tail probability for (it emits a point estimate, not a "
             "distribution). This measures whether results get reported, not "
             "whether they succeed -- read the number with that caveat."),
    ("rel-f1", "driver-top3"): Predicate(
        ["qualifying"], note="qualifies (position filter unavailable)"),
}

# k for "more than k" tasks; absent means the label is "any event" (k = 0).
THRESHOLDS: dict[tuple[str, str], int] = {
    ("rel-avito", "user-visits"): 1,
    ("rel-avito", "user-clicks"): 1,
    ("rel-event", "user-ignore"): 2,
}

# Does label 1 mean the qualifying event HAPPENS, or that it does NOT?
# Churn is the odd one out: `churn = 1` iff the window is empty. Everything
# else flags an occurrence. Getting this backwards is not a subtle error --
# it drives AUROC below 50, which is how it was caught -- but it IS silent in
# the sense that the readout still returns a plausible probability.
ABSENCE: set[tuple[str, str]] = {
    ("rel-amazon", "user-churn"),
    ("rel-amazon", "item-churn"),
    ("rel-hm", "user-churn"),
}


def task_score(mu, key):
    """Probability of the positive label, for the registered task."""
    k = THRESHOLDS.get(key, 0)
    if key in ABSENCE:
        return p_no_event(mu)          # label 1 = window is empty
    return p_more_than(mu, k)          # label 1 = more than k qualifying
