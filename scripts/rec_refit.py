"""Refit the WHO head's feature combination PER TASK, on that task's labels.

`rank_destinations` scores a (query, candidate) pair as

    q.c  +  sum_j w[j] * f_j(query, candidate)

where f_0..f_7 are the D3 features (in-history, repeat count, recency,
popularity, recent popularity, trend, staleness, co-occurrence) and `w` is
`model.who.feat.weight` -- learned during PRETRAINING and then frozen. One
recipe serves every task, so a task where popularity is decisive and a task
where it is irrelevant are combined identically.

This refits the 9 coefficients (the dot product plus the 8 features, plus a
reranker delta where the checkpoint has one) on the task's own train split
and rescores. Nothing in the network changes: the backbone, the entity
states and both heads are frozen, and only the linear combination on top of
scores the model already produces is relearned. The window head -- which
produces every classification and regression number -- is not touched at all.

Why this can work where scripts/icl_readout.py's probe cannot: the probe's
387 statistics describe the query ENTITY, so they take the same value for
every candidate and cannot induce an order. These features are per-PAIR.

The features are read out by calling the SHIPPING feature code with a
one-hot weight vector rather than by reimplementing it. `_add_history_features`
warns that a train/eval mismatch in it would be invisible and would poison the
metric; a private copy here would be exactly that mismatch.

Causality: every feature is computed with the query row's OWN timestamp as
the cutoff, so a train row never sees its own future. The candidate state
table is the one built at the corpus cutoff, which is what the published
eval uses too.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from ledger.data.batching import PackedBatcher
from ledger.data.cache import load_corpus
from ledger.model.load import build_model, corpus_kwargs, load_checkpoint
from ledger.data.features import CoocTable, TemporalIndex
from ledger.queries import (_add_history_features, _query_states, _rerank,
                          build_path_map, popularity_ranking)

FEAT_NAMES = ["dot", "in_hist", "repeat", "recency", "pop_total",
              "pop_recent", "trend", "staleness", "cooc", "rerank"]
SPARSE_J = (0, 1, 2, 7)      # the features _add_history_features owns


class Scorer:
    """Per-(row, candidate) feature blocks for one checkpoint and task."""

    def __init__(self, model, corpus, task, max_len, device, cooc, retarget,
                 n_stage1):
        self.m, self.c, self.task = model, corpus, task
        self.max_len, self.dev = max_len, device
        self.cooc, self.retarget, self.n_stage1 = cooc, retarget, n_stage1
        self.src = task.src_entity_table
        self.dst = task.dst_entity_table
        self.n_dst = corpus.schema.entity_counts[self.dst]
        self.n_feats = int(getattr(model.who, "n_feats", 0))
        self.w0 = (model.who.feat.weight.flatten().detach().cpu().numpy()
                   if self.n_feats else np.zeros(8))
        all_ids = torch.arange(self.n_dst, device=device)
        self.cand_raw = model.entity_states.read(self.dst, all_ids)
        self.cand = model.who.c(self.cand_raw) * model.who.scale
        self.win_days = (task.timedelta.total_seconds() / 86400.0
                         if getattr(model.who, "window", False) else None)
        self._ti = TemporalIndex(corpus, self.dst) if self.n_feats else None
        self._dense_cache = {}

    def _dense(self, cut):
        """Features 3-6: candidate-side, identical for every query at `cut`."""
        if cut not in self._dense_cache:
            a = np.arange(self.n_dst, dtype=np.int64)
            st = self._ti.stats(a, np.full(self.n_dst, cut, dtype=np.int64),
                                PackedBatcher.TREND_WINDOW_DAYS,
                                PackedBatcher.TREND_PREV_WINDOWS,
                                PackedBatcher.TREND_TAU_DAYS)
            self._dense_cache[cut] = np.stack(
                [st["log_total"], st["log_recent"], st["trend"],
                 st["staleness"]]).astype(np.float32)
        return self._dense_cache[cut]

    @torch.no_grad()
    def blocks(self, rows, cuts):
        """-> [n_feat, B, n_dst] float32 on device, and n_events [B]."""
        B = len(rows)
        hq, ne, hstates, hmask = _query_states(
            self.m, self.c, self.src, rows, cuts, self.max_len, self.dev)
        if self.win_days is not None:
            hq = self.m.who.condition(
                hq, torch.full((B,), float(self.win_days), device=self.dev))
        dot = self.m.who.q(hq) @ self.cand.T                  # [B, n_dst]

        F = torch.zeros((len(FEAT_NAMES), B, self.n_dst),
                        device=self.dev, dtype=torch.float32)
        F[0] = dot
        if self.n_feats:
            for b, cut in enumerate(cuts):
                d = self._dense(int(cut))
                F[4:8, b] = torch.as_tensor(d, device=self.dev)
            # 0-2 and 7 come out of the shipping implementation, one at a
            # time, by zeroing every weight but one.
            for j in SPARSE_J:
                oh = torch.zeros(8, device=self.dev)
                oh[j] = 1.0
                buf = torch.zeros((B, self.n_dst), device=self.dev,
                                  dtype=torch.float32)
                _add_history_features(buf, oh, self.c, self.src, self.dst,
                                      rows, cuts, self.max_len, self.cooc,
                                      self.retarget)
                F[1 + j if j < 3 else 8] = buf

        # The reranker's delta is defined only on the shortlist it rescores,
        # and that shortlist is the top-n_stage1 of the PUBLISHED score. Fix
        # it there rather than letting it move with the coefficients being
        # fitted, which would make the feature circular.
        if self.m.rerank is not None:
            base = dot.clone()
            if self.n_feats:
                for j in range(8):
                    base += float(self.w0[j]) * F[1 + j]
            re = _rerank(self.m, base, self.cand_raw, hstates, hmask,
                         self.n_stage1)
            F[9] = torch.where(torch.isfinite(re), re - base,
                               torch.zeros_like(base))
        return F, ne

    def combine(self, F, coef):
        c = torch.as_tensor(coef, device=F.device, dtype=F.dtype)
        return (F * c[:, None, None]).sum(0)

    @property
    def published_coef(self):
        c = np.zeros(len(FEAT_NAMES), dtype=np.float64)
        c[0] = 1.0
        if self.n_feats:
            c[1:9] = self.w0
        if self.m.rerank is not None:
            c[9] = 1.0
        return c


def iter_rows(sc, df, task, batch_size, src_map):
    src_rows = df[task.src_entity_col].map(src_map).fillna(-1) \
                                      .to_numpy(dtype=np.int64)
    cuts = (pd.to_datetime(df[task.time_col]).astype("int64")
            // 10 ** 9).to_numpy()
    for s in range(0, len(df), batch_size):
        sl = slice(s, s + batch_size)
        keep = src_rows[sl] >= 0
        if not keep.any():
            continue
        yield s, np.nonzero(keep)[0], src_rows[sl][keep], cuts[sl][keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--fit-rows", type=int, default=3000,
                    help="labelled TRAIN rows the coefficients are fitted on")
    ap.add_argument("--eval-rows", type=int, default=0,
                    help="0 = the whole split")
    ap.add_argument("--eval-split", default="val",
                    choices=["val", "holdout"],
                    help="`holdout` fits on part of the TRAIN split and "
                         "scores the rest of it. driver-circuit-compete has "
                         "27 val rows, which cannot separate a real gain "
                         "from noise; its train split has 2,649. Not a "
                         "substitute for val -- the holdout rows share the "
                         "train period -- but it does say whether the "
                         "coefficients generalise past the rows they saw.")
    ap.add_argument("--n-neg", type=int, default=64,
                    help="negatives per fit row, from the top of the "
                         "published ranking (the ones we must beat)")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=0)
    ap.add_argument("--n_stage1", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = load_checkpoint(args.ckpt)
    targs = ck["args"]
    max_len = args.max_len or max(1, int(targs.get("max_len", 512)) // 2)

    ds = get_dataset(args.dataset, download=True)
    corpus = load_corpus(args.dataset, pd.Timestamp(ds.val_timestamp),
                         db=ds.get_db, **corpus_kwargs(ck))
    model = build_model(ck, corpus, dev)
    task = get_task(args.dataset, args.task, download=True)

    retarget = None
    if targs.get("retarget"):
        inter, final = [x.strip() for x in str(targs["retarget"]).split(":")]
        pm = build_path_map(ds.get_db(), corpus, inter, final, corpus.cutoff)
        iv, fv = pm["inter"], pm["final"]
        o = np.argsort(iv, kind="stable"); iv, fv = iv[o], fv[o]
        n_i = corpus.schema.entity_counts[inter]
        off = np.zeros(n_i + 1, dtype=np.int64)
        np.cumsum(np.bincount(iv, minlength=n_i), out=off[1:])
        retarget = {"inter": inter, "final": final, "off": off, "val": fv}

    cooc = None
    if getattr(model.who, "n_feats", 0) >= 8 \
            and task.src_entity_table != task.dst_entity_table:
        cooc = CoocTable.build(corpus, task.src_entity_table,
                               task.dst_entity_table, cache_key=args.dataset)

    sc = Scorer(model, corpus, task, max_len, dev, cooc, retarget,
                args.n_stage1)
    src_map = corpus.schema.pkey_index[task.src_entity_table]
    dst_map = corpus.schema.pkey_index[task.dst_entity_table]
    rng = np.random.default_rng(args.seed)

    # ---- fit set -----------------------------------------------------
    tr_tab = task.get_table("train", mask_input_cols=False)
    tr = tr_tab.df
    ho = None
    if args.eval_split == "holdout":
        perm = rng.permutation(len(tr))
        n_fit = min(args.fit_rows, len(tr) // 2)
        tr, ho = tr.iloc[perm[:n_fit]], tr.iloc[perm[n_fit:]]
    elif len(tr) > args.fit_rows:
        tr = tr.iloc[rng.choice(len(tr), args.fit_rows, replace=False)]
    print(f"task {args.dataset}/{args.task}  eval_k={task.eval_k}  "
          f"catalogue={sc.n_dst:,}  fit rows={len(tr):,}")

    X, Y, G = [], [], []
    pub = sc.published_coef
    for s, keep, rows, cuts in iter_rows(sc, tr, task, args.batch_size,
                                         src_map):
        F, ne = sc.blocks(rows, cuts)
        base = sc.combine(F, pub)
        kk = min(args.n_neg, sc.n_dst)
        top = torch.topk(base, kk, dim=-1).indices.cpu().numpy()
        labs = tr[task.dst_entity_col].to_numpy()[s:s + args.batch_size]
        for bi, gi in enumerate(keep):
            if ne[bi] == 0:
                continue                      # cold: popularity fallback
            pos = np.array([dst_map[p] for p in np.atleast_1d(labs[gi])
                            if p in dst_map], dtype=np.int64)
            if not len(pos):
                continue
            neg = np.setdiff1d(top[bi], pos, assume_unique=False)
            cand = np.concatenate([pos, neg])
            X.append(F[:, bi, torch.as_tensor(cand, device=dev)]
                     .T.cpu().numpy())
            Y.append(np.concatenate([np.ones(len(pos)),
                                     np.zeros(len(neg))]))
            G.append(len(cand))
    X = np.concatenate(X).astype(np.float64)
    Y = np.concatenate(Y)
    print(f"fit pairs: {len(Y):,}  positives {int(Y.sum()):,}  "
          f"queries {len(G):,}")

    # Standardise: the dot product and log-count features differ by orders of
    # magnitude and an unscaled L2 penalty would be a pure function of that.
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    from sklearn.linear_model import LogisticRegression
    best, best_c = None, None
    for C in (0.01, 0.1, 1.0, 10.0):
        lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        lr.fit((X - mu) / sd, Y)
        s = lr.score((X - mu) / sd, Y)
        if best is None or s > best:
            best, best_c, coef_, int_ = s, C, lr.coef_[0], lr.intercept_[0]
    # Fold the standardisation back so the coefficients apply to raw
    # features, and drop the intercept: it is constant across candidates
    # within a query and cannot change any ranking.
    new = coef_ / sd
    print(f"chosen C={best_c}")
    print(f"\n{'feature':<12}{'published':>12}{'refit':>12}")
    for i, n in enumerate(FEAT_NAMES):
        print(f"{n:<12}{pub[i]:>12.4f}{new[i]:>12.4f}")

    # ---- evaluate ----------------------------------------------------
    if ho is not None:
        ev, edf = tr_tab, ho
    else:
        ev = task.get_table(args.eval_split, mask_input_cols=False)
        edf = ev.df
    if args.eval_rows and len(edf) > args.eval_rows:
        edf = edf.iloc[rng.choice(len(edf), args.eval_rows, replace=False)]
    if len(edf) != len(ev.df):
        ev = type(ev)(df=edf, fkey_col_to_pkey_table=ev.fkey_col_to_pkey_table,
                      pkey_col=ev.pkey_col, time_col=ev.time_col)
    k = task.eval_k
    out = {}
    for name, coef in (("published", pub), ("refit", new)):
        ranked = np.zeros((len(edf), k), dtype=np.int64)
        cold = np.ones(len(edf), dtype=bool)
        for s, keep, rows, cuts in iter_rows(sc, edf, task, args.batch_size,
                                             src_map):
            F, ne = sc.blocks(rows, cuts)
            sco = sc.combine(F, coef)
            tk = torch.topk(sco, k, dim=-1).indices.cpu().numpy()
            gi = s + keep
            ranked[gi] = tk
            cold[gi] = ne == 0
        if cold.any():
            ranked[cold] = popularity_ranking(corpus, sc.dst, k,
                                              window_days=7)
        pk = corpus.schema.pkey_values[sc.dst]
        out[name] = task.evaluate(pk[ranked], ev)
        print(f"\n=== {name} ({args.eval_split}, {len(edf):,} rows)")
        for kk, v in out[name].items():
            print(f"  {kk:<32}{100 * v:9.4f}")

    metric = "link_prediction_map"
    d = 100 * (out["refit"][metric] - out["published"][metric])
    print(f"\nMAP {args.eval_split}: {100*out['published'][metric]:.4f} -> "
          f"{100*out['refit'][metric]:.4f}   ({d:+.4f})")
    if args.json_out:
        with open(args.json_out, "a") as fh:
            fh.write(json.dumps({
                "dataset": args.dataset, "task": args.task,
                "split": args.eval_split, "ckpt": args.ckpt,
                "fit_rows": len(tr), "C": best_c,
                "coef": {n: float(new[i]) for i, n in enumerate(FEAT_NAMES)},
                "published": {k_: 100 * v for k_, v in out["published"].items()},
                "refit": {k_: 100 * v for k_, v in out["refit"].items()},
                "delta_map": d}) + "\n")


if __name__ == "__main__":
    main()
