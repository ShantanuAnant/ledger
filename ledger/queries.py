"""Answering RelBench tasks as queries against the trained world model.

This is the piece heads.py referred to but that did not exist: turning the
WHO head's retrieval scores into a ranked top-k list that RelBench's own
evaluator can score, so LEDGER produces numbers directly comparable to the
leaderboard.

Recommendation (LINK_PREDICTION):
    For each (src entity, cutoff) row of the task table, take that entity's
    event history strictly before the cutoff, run the backbone over it, take
    the final hidden state, and score it against the states of EVERY candidate
    destination entity in one matmul. Return the top-k.

LEAKAGE DISCIPLINE -- the two ways to get this silently wrong:
  1. History must be truncated at the row's OWN timestamp, not merely at the
     corpus cutoff. `EventCorpus.history(..., before=)` does this and is used
     here unconditionally.
  2. Candidates must be the full destination catalogue. Filtering the
     candidate set using anything derived from the labels (e.g. "only items
     purchased during the eval window") inflates MAP dramatically and is the
     classic way recommendation numbers become unreproducible.

Entities whose state has never been written are scored anyway; they carry a
zero state and therefore a constant score, so they rank arbitrarily among
themselves. `states.coverage()` reports how large that group is -- read it
before believing a low MAP is the model's fault.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .data.batching import PackedBatcher
from .data.features import CoocTable, TemporalIndex


@torch.no_grad()
def refresh_states(model, corpus, entity_table: str, max_len: int = 256,
                   batch_size: int = 256, device: str = "cuda",
                   reset: bool = True, verbose: bool = True,
                   mode: str = "mean") -> dict:
    """Recompute the entity-state table with the CURRENT weights, in one pass.

    Why this exists. During training, `update_states` writes states as a side
    effect of whatever batch happened to be sampled. So at the end of a 30k-
    step run, S[article] is an EMA whose oldest terms were produced by a model
    30k steps stale, and an article not sampled recently holds a state from
    what is effectively a different network. `queries.py` then ranks the whole
    catalogue against that table. The candidate matrix is not self-consistent
    with the query encoder it is scored against.

    A refresh is one no-grad forward over every entity's history with the
    final weights. It is O(corpus) -- minutes -- and makes the two sides of
    the inner product come from the same model. Nothing about it is a
    training change; it is the evaluation the architecture always implied.

    `mode="mean"` accumulates an exact, order-independent mean per entity;
    `mode="ema"` reproduces the training-time EMA write and is kept only so
    the two can be compared (it is order-dependent -- see states.py).

    Returns the post-refresh coverage dict.
    """
    if getattr(model.entity_states, "kind", None) == "learned":
        return {}                      # nothing to recompute
    model.eval()
    if mode == "mean":
        # Two-phase, and the phasing matters. The tokenizer READS entity
        # states to build each event token (`link_proj(state)`), so zeroing
        # the table before the pass would encode every history with all-zero
        # link summaries -- an input distribution the backbone never saw in
        # training. Instead: read the old table throughout, accumulate into
        # separate buffers, swap in at the end.
        model.entity_states.begin_accumulate()
    elif reset:
        model.entity_states.reset()

    off = corpus.hist_offset[entity_table]
    lens = np.diff(off)
    rows = np.flatnonzero(lens > 0).astype(np.int64)
    # Sort by history length. `pack_histories` pads each batch to its longest
    # row, so mixing a 3-event customer with a 256-event one costs the full
    # 256 for both. rel-hm's median customer has 15 events, so length-sorted
    # batching is ~10x less compute for an identical result.
    rows = rows[np.argsort(np.minimum(lens[rows], max_len), kind="stable")]
    # Whole history: the corpus is already cut at its own cutoff, so `before`
    # only has to sit strictly above the last event.
    cut = int(corpus.cutoff.timestamp()) + 1

    for start in range(0, len(rows), batch_size):
        r = rows[start:start + batch_size]
        batch, _ = PackedBatcher.pack_histories(
            corpus, entity_table, r, np.full(len(r), cut, dtype=np.int64),
            max_len=max_len, device=device)
        h = model.encode(batch)
        model.update_states(batch, h)
        if verbose and (start // batch_size) % 200 == 0:
            print(f"  refresh {start:,}/{len(rows):,}", flush=True)

    if mode == "mean":
        model.entity_states.end_accumulate()

    cov = (model.entity_states.coverage()
           if hasattr(model.entity_states, "coverage") else {})
    if verbose:
        print("  refreshed coverage:",
              {k: f"{100*v:.1f}%" for k, v in cov.items()}, flush=True)
    return cov


def popularity_ranking(corpus, dst_table: str, k: int,
                       window_days: int | None = 7) -> np.ndarray:
    """Top-k destination rows by link count, optionally over a recent window.

    Used as the cold-start fallback (see `evaluate_recommendation`). This is
    the `pop_recent` baseline from RESEARCH.md 2026-08-22, computed from the
    corpus rather than the task tables.
    """
    counts = np.zeros(corpus.schema.entity_counts[dst_table], dtype=np.int64)
    lo = (corpus.time.max() - window_days * 86400
          if window_days is not None else None)
    for name, spec in corpus.schema.fact_tables.items():
        slots = [j for j, t in enumerate(spec.fkeys.values()) if t == dst_table]
        if not slots:
            continue
        ev = np.flatnonzero(corpus.row_of[name] >= 0)
        if lo is not None:
            ev = ev[corpus.time[ev] >= lo]
        tr = corpus.row_of[name][ev]
        for j in slots:
            e = corpus.links[name][tr, j]
            e = e[e >= 0]
            if len(e):
                counts += np.bincount(e, minlength=len(counts))
    return np.argsort(-counts)[:k].astype(np.int64)


@torch.no_grad()
def rank_destinations(model, corpus, src_table: str, src_rows: np.ndarray,
                      cutoffs: np.ndarray, dst_table: str, k: int,
                      max_len: int = 256, batch_size: int = 256,
                      device: str = "cuda", cooc=None,
                      win_days: float | None = None,
                      n_stage1: int = 2048, project=None, retarget=None):
    """-> (ranked [n, k] destination ROW indices, n_events [n]).

    src_rows / cutoffs are parallel arrays: entity row index in `src_table`
    and the unix-second cutoff for that query. `n_events` is how many
    pre-cutoff events each query entity actually had; rows with 0 are ranked
    from an all-zero state and the caller should override them.
    """
    model.eval()

    # Candidate matrix: every destination entity, projected once.
    n_dst = corpus.schema.entity_counts[dst_table]
    all_ids = torch.arange(n_dst, device=device)
    cand_states_raw = model.entity_states.read(dst_table, all_ids)
    cand = model.who.c(cand_states_raw) * model.who.scale      # [n_dst, d]

    # D3 interaction features. The popularity term is dense over the whole
    # catalogue and constant across queries, so it is added once. The
    # repeat/recency terms are nonzero only for the (few) candidates in a
    # given query entity's own history, so they go in as a sparse per-row
    # correction rather than a [B, n_dst] dense build.
    # D3 interaction features, mirroring PackedBatcher._who_features.
    #
    # Split by how they vary:
    #   3-6 candidate-side temporal -- same for every query at a given time,
    #       and RelBench task splits carry ONE timestamp, so they are computed
    #       once as a dense [n_dst] vector rather than per row.
    #   0-2 repeat/recency and 7 co-occurrence -- nonzero only for the handful
    #       of candidates tied to a given query's own history, so they go in
    #       as a sparse per-row correction.
    w = None
    if getattr(model.who, "n_feats", 0):
        w = model.who.feat.weight.flatten()                     # [F]
        ti = TemporalIndex(corpus, dst_table)
        all_np = np.arange(n_dst, dtype=np.int64)
        dense = {}
        for cut in np.unique(cutoffs):
            st = ti.stats(all_np, np.full(n_dst, cut, dtype=np.int64),
                          PackedBatcher.TREND_WINDOW_DAYS,
                          PackedBatcher.TREND_PREV_WINDOWS,
                          PackedBatcher.TREND_TAU_DAYS)
            v = (w[3].item() * st["log_total"] + w[4].item() * st["log_recent"]
                 + w[5].item() * st["trend"] + w[6].item() * st["staleness"])
            dense[int(cut)] = torch.as_tensor(v, device=device).float()

    out = np.zeros((len(src_rows), k), dtype=np.int64)
    n_events = np.zeros(len(src_rows), dtype=np.int64)
    for start in range(0, len(src_rows), batch_size):
        rows = src_rows[start:start + batch_size]
        cuts = cutoffs[start:start + batch_size]
        hq, ne, hstates, hmask = _query_states(
            model, corpus, src_table, rows, cuts, max_len, device)
        if win_days is not None:
            hq = model.who.condition(
                hq, torch.full((len(hq),), float(win_days), device=device))
        q = model.who.q(hq)                                     # [B, d]
        scores = q @ cand.T                                     # [B, n_dst]
        if w is not None:
            uc = np.unique(cuts)
            if len(uc) == 1:
                scores = scores + dense[int(uc[0])].unsqueeze(0)
            else:
                for c0 in uc:
                    m = torch.as_tensor(cuts == c0, device=device)
                    scores[m] = scores[m] + dense[int(c0)].unsqueeze(0)
            _add_history_features(scores, w, corpus, src_table, dst_table,
                                  rows, cuts, max_len, cooc, retarget)

        if model.rerank is not None:
            scores = _rerank(model, scores, cand_states_raw, hstates, hmask,
                             n_stage1)
        if project is not None:
            scores = _project_scores(scores, project)
        out[start:start + len(rows)] = (
            torch.topk(scores, k, dim=-1).indices.cpu().numpy())
        n_events[start:start + len(rows)] = ne
    return out, n_events



def build_path_map(db, corpus, inter_table: str, final_table: str,
                   cutoff) -> dict:
    """Incidence between an intermediate entity table and a 2-hop destination.

    The WHO head predicts DIRECT foreign-key links, so a task whose
    destination is two hops away (rel-f1 drivers -> races -> circuits;
    rel-trial conditions -> studies -> sponsors) is not representable by it at
    all. Rather than change the head, marginalize its distribution over the
    path: the driver competes at circuit c if they race in ANY race held at c,
    so score(c) = logsumexp over those races. The soft-OR (rather than a max)
    is deliberate -- a destination reachable by several likely intermediates
    should outrank one reachable by a single equally likely intermediate,
    which is what the metric rewards.

    Built from PRE-CUTOFF rows only. A destination reachable solely through a
    post-cutoff link is therefore unreachable, which is correct: that link is
    exactly the future information the benchmark forbids.
    """
    pk_i = corpus.schema.pkey_index[inter_table]
    pk_f = corpus.schema.pkey_index[final_table]

    def _pairs(name):
        t = db.table_dict[name]
        df = t.df if t.time_col is None else t.df[t.df[t.time_col] <= cutoff]
        cols = t.fkey_col_to_pkey_table
        i_col = ([c for c, tgt in cols.items() if tgt == inter_table]
                 or ([t.pkey_col] if name == inter_table else []))
        f_col = [c for c, tgt in cols.items() if tgt == final_table]
        if not i_col or not f_col:
            return None
        i = df[i_col[0]].map(pk_i).to_numpy(dtype="float64")
        f = df[f_col[0]].map(pk_f).to_numpy(dtype="float64")
        ok = ~(np.isnan(i) | np.isnan(f))
        return i[ok].astype(np.int64), f[ok].astype(np.int64)

    # The intermediate table's own row may carry the foreign key (rel-f1:
    # races.circuitId); otherwise a join table carries both (rel-trial:
    # sponsors_studies).
    for name in [inter_table] + list(db.table_dict):
        got = _pairs(name)
        if got is not None:
            i, f = got
            uniq = np.unique(np.stack([i, f]), axis=1)
            return {"via": name, "inter": uniq[0], "final": uniq[1],
                    "n_final": corpus.schema.entity_counts[final_table]}
    raise ValueError(f"no table links {inter_table} to {final_table}")


def _project_scores(scores, project):
    """[B, n_inter] -> [B, n_final] by logsumexp over the path incidence."""
    i = project["i"]; f = project["f"]; n_final = project["n_final"]
    s = scores[:, i]                                    # [B, M]
    B = s.shape[0]
    m = torch.full((B, n_final), float("-inf"), device=s.device)
    m.scatter_reduce_(1, f.expand(B, -1), s, reduce="amax",
                      include_self=True)
    gathered = m.gather(1, f.expand(B, -1))
    acc = torch.zeros((B, n_final), device=s.device)
    acc.scatter_reduce_(1, f.expand(B, -1), (s - gathered).exp(),
                        reduce="sum", include_self=True)
    out = m + acc.clamp_min(1e-30).log()
    # Destinations with no pre-cutoff path are unreachable, not merely
    # unlikely; -inf keeps them out of the top-k instead of at an arbitrary 0.
    return torch.where(torch.isfinite(out), out,
                       torch.full_like(out, float("-inf")))


@torch.no_grad()
def _rerank(model, scores, cand_states_raw, hist, hmask, n_stage1):
    """Stage 2: cross-interaction rescoring of the stage-1 shortlist.

    The shortlist is the TOP-K OF THE DENSE SCORES, which already carry the
    structural priors (repeat, 2-hop, trend) as additive features. So the
    shortlist is "structurally plausible OR globally similar", never one at
    the expense of the other -- unlike a hard neighbourhood restriction,
    which cannot recover an item the sampler missed.
    """
    B, n_dst = scores.shape
    kk = min(n_stage1, n_dst)
    top = torch.topk(scores, kk, dim=-1)
    cs = cand_states_raw[top.indices]                     # [B, kk, d]
    delta = model.rerank(cs, hist, hmask)                 # [B, kk]
    out = torch.full_like(scores, float("-inf"))
    out.scatter_(1, top.indices, top.values + delta)
    return out


@torch.no_grad()
def _add_history_features(scores, w, corpus, src_table, dst_table, rows,
                          cuts, max_len, cooc=None, retarget=None):
    """In-place sparse add of features 0-2 (repeat/recency) and 7 (co-occ).

    Must mirror `PackedBatcher._who_features` exactly, including TAU_DAYS,
    COOC_RECENT_ITEMS and the strictly-before-cutoff truncation -- a
    train/eval mismatch here would be invisible and would poison the metric.
    """
    tau = PackedBatcher.TAU_DAYS
    M = PackedBatcher.COOC_RECENT_ITEMS
    n_dst = corpus.schema.entity_counts[dst_table]
    prior_rows, prior_b = [], []
    bi, ci, val = [], [], []
    for b, (r, cut) in enumerate(zip(rows, cuts)):
        ids = corpus.history(src_table, int(r), before=int(cut))[-max_len:]
        if len(ids) == 0:
            continue
        dst, times = [], []
        # Under path retargeting the source's events do NOT link to the
        # destination directly (a facility links to studies, not sponsors), so
        # collecting only slots that target `dst_table` would leave the repeat
        # feature identically zero -- which is exactly the failure this change
        # exists to fix. Expand the intermediate through the path instead.
        want = dst_table if retarget is None else retarget["inter"]
        for name, spec in corpus.schema.fact_tables.items():
            slots = [j for j, t in enumerate(spec.fkeys.values())
                     if t == want]
            if not slots:
                continue
            sel = ids[corpus.row_of[name][ids] >= 0]
            if len(sel) == 0:
                continue
            tr = corpus.row_of[name][sel]
            for j in slots:
                e = corpus.links[name][tr, j]
                ok = e >= 0
                dst.append(e[ok]); times.append(corpus.time[sel[ok]])
        if not dst:
            continue
        dst = np.concatenate(dst); times = np.concatenate(times)
        if retarget is not None:
            # NB: `val` is this function's output accumulator -- do not shadow
            rt_off, rt_val = retarget["off"], retarget["val"]
            cnt = rt_off[dst + 1] - rt_off[dst]
            keep = cnt > 0
            if not keep.any():
                continue
            dst_k, times_k, cnt_k = dst[keep], times[keep], cnt[keep]
            starts = rt_off[dst_k]
            idx = np.concatenate([np.arange(a, a + c)
                                  for a, c in zip(starts, cnt_k)])
            dst = rt_val[idx]
            times = np.repeat(times_k, cnt_k)
        if cooc is not None:
            order = np.argsort(times, kind="stable")[-M:]
            p = np.full(M, -1, dtype=np.int64)
            p[:len(order)] = dst[order]
            prior_rows.append(p); prior_b.append(b)
        uniq, inv = np.unique(dst, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(uniq))
        last = np.zeros(len(uniq), dtype=np.int64)
        np.maximum.at(last, inv, times)
        rec = np.exp(-np.maximum((int(cut) - last) / 86400.0, 0.0) / tau)
        v = (w[0].item()
             + w[1].item() * np.log1p(cnt)
             + w[2].item() * rec)
        bi.append(np.full(len(uniq), b)); ci.append(uniq); val.append(v)
    dev = scores.device
    if bi:
        scores[torch.as_tensor(np.concatenate(bi), device=dev),
               torch.as_tensor(np.concatenate(ci), device=dev)] += \
            torch.as_tensor(np.concatenate(val), device=dev,
                            dtype=scores.dtype)

    # Feature 7: co-occurrence is dense over the catalogue for the few rows
    # that have history, so score it as a [rows_with_history, n_dst] block.
    if cooc is not None and prior_rows:
        r, c, v = cooc.score_sparse(np.stack(prior_rows))
        if len(r):
            b_of = np.asarray(prior_b, dtype=np.int64)[r]
            scores[torch.as_tensor(b_of, device=dev),
                   torch.as_tensor(c, device=dev)] += \
                (w[7].item() * torch.as_tensor(np.log1p(v), device=dev,
                                               dtype=scores.dtype))


@torch.no_grad()
def _query_states(model, corpus, src_table, rows, cuts, max_len, device,
                  branch: str = "retrieval"):
    """Backbone final hidden state for each (entity, cutoff) history.

    `branch` must be the one the CONSUMING head was trained on
    (Backbone.BRANCHES): "retrieval" for WHO and the reranker, "temporal" for
    the WHEN-head readouts in scripts/eval_entity.py and eval_regression.py.
    Those two share this function with the recommendation path, and with
    `--branch_layers 0` the two branches are the same tensor -- so the wrong
    default here is invisible on every checkpoint trained so far and silently
    wrong on every branched one.
    """
    batch, keep = PackedBatcher.pack_histories(
        corpus, src_table, rows, cuts, max_len=max_len, device=device)
    h = model.encode(batch, branch=branch)        # [B, L, d]
    # final real token of each packed sequence
    q = h[keep["b"], keep["last_l"]]

    # The last HIST_R token states, for the reranker's cross-attention. Must
    # mirror PackedBatcher._hist_slice: same R, same "at or before the query
    # position", same padding convention.
    R = PackedBatcher.HIST_R
    last = keep["last_l"]
    ne = torch.as_tensor(keep["n_events"], device=h.device)
    offs = torch.arange(-R + 1, 1, device=h.device)
    pos = last[:, None] + offs[None, :]
    pad = (pos < 0) | (pos > last[:, None]) | (ne[:, None] <= 0)
    hs = h[keep["b"][:, None], pos.clamp(min=0)] * (~pad).unsqueeze(-1)
    return q, keep["n_events"], hs, pad


def evaluate_recommendation(model, corpus, task, split: str = "test",
                            max_len: int = 256, device: str = "cuda",
                            batch_size: int = 256,
                            cold_fallback: bool = True,
                            cold_window_days: int | None = 7,
                            cooc_cache_key: str | None = None,
                            n_stage1: int = 2048,
                            project_through: str | None = None,
                            retarget: dict | None = None,
                            db=None, return_pred: bool = False) -> dict:
    """Score LEDGER on a RelBench LINK_PREDICTION task with its own evaluator.

    Returns the metric dict from `task.evaluate`, plus diagnostics.
    """
    # The head is horizon-conditioned; at inference the horizon IS the task's
    # timedelta. This is what lets one pretrained model serve a 7-day rel-hm
    # task and a 365-day rel-trial task without retraining.
    win_days = (task.timedelta.total_seconds() / 86400.0
                if getattr(model.who, "window", False) else None)
    target = task.get_table(split, mask_input_cols=False)
    df = target.df
    src_table, dst_table = task.src_entity_table, task.dst_entity_table
    # 2-hop destination: the model ranks the INTERMEDIATE table it was
    # actually trained to link to, and the scores are marginalized onto the
    # task's destination afterwards. Everything before the projection --
    # candidate matrix, D3 features, co-occurrence -- therefore stays in the
    # intermediate's space, which is the space the checkpoint was trained in.
    rank_table = project_through or dst_table
    project = None
    if project_through is not None:
        # `db` may be the Database or a zero-arg loader (train.py hands the
        # loader so a run without projection never pays the load).
        pm = build_path_map(db() if callable(db) else db, corpus,
                            project_through, dst_table, corpus.cutoff)
        print(f"path map: {src_table} -> {project_through} -> {dst_table} "
              f"via `{pm['via']}`, {len(pm['inter']):,} pre-cutoff pairs "
              f"covering {len(np.unique(pm['final'])):,}/{pm['n_final']:,} "
              f"{dst_table}")
        project = {"i": torch.as_tensor(pm["inter"], device=device),
                   "f": torch.as_tensor(pm["final"], device=device),
                   "n_final": pm["n_final"]}
    src_col, dst_col = task.src_entity_col, task.dst_entity_col

    # task-table primary keys -> corpus row indices
    src_map = _pkey_to_row(corpus, src_table)
    dst_rows_to_pkey = _row_to_pkey(corpus, dst_table)

    src_rows = df[src_col].map(src_map).fillna(-1).to_numpy(dtype=np.int64)
    cuts = (pd.to_datetime(df[task.time_col]).astype("int64")
            // 10 ** 9).to_numpy()

    # The 2-hop table is only needed when the model actually learned a weight
    # for it (8-feature models); building it is minutes on a large catalogue.
    cooc = None
    if getattr(model.who, "n_feats", 0) >= 8 and src_table != rank_table:
        cooc = CoocTable.build(corpus, src_table, rank_table,
                               cache_key=cooc_cache_key)

    known = src_rows >= 0
    ranked = np.zeros((len(df), task.eval_k), dtype=np.int64)
    n_events = np.zeros(len(df), dtype=np.int64)
    if known.any():
        ranked[known], n_events[known] = rank_destinations(
            model, corpus, src_table, src_rows[known], cuts[known],
            rank_table, task.eval_k, max_len=max_len,
            batch_size=batch_size, device=device, cooc=cooc,
            win_days=win_days, n_stage1=n_stage1, project=project,
            retarget=retarget)

    # Cold start. An entity with no pre-cutoff events is encoded from an
    # all-zero token, so its "ranking" is a fixed function of nothing -- pure
    # noise scored against the whole catalogue. On rel-hm that is ~10.8% of
    # eval customers (RESEARCH.md 2026-08-22). Recent popularity is the
    # correct thing to say when the model has no information, and it is
    # legitimate: it uses only pre-cutoff rows every method may use.
    cold = ~known | (n_events == 0)
    if cold_fallback and cold.any():
        ranked[cold] = popularity_ranking(corpus, dst_table, task.eval_k,
                                          window_days=cold_window_days)

    pred = dst_rows_to_pkey[ranked]
    metrics = task.evaluate(pred, target)
    diag = {
        "rows": len(df),
        "rank_table": rank_table,
        "src_not_in_corpus_frac": float((~known).mean()),
        "cold_frac": float(cold.mean()),
        "cold_fallback": cold_fallback,
    }
    if hasattr(model.entity_states, "coverage"):
        diag["state_coverage"] = model.entity_states.coverage()
    out = {**metrics, "_diagnostics": diag}
    if return_pred:
        # Opt-in: callers iterate this dict and format every value as a
        # number, so an array in it breaks them (a test caught exactly that).
        out["_pred"] = pred
    return out


def _pkey_to_row(corpus, table):
    return corpus.schema.pkey_index[table]


def _row_to_pkey(corpus, table):
    return corpus.schema.pkey_values[table]
