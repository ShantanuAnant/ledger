"""Zero-shot regression readout: E[sum of values over the qualifying window].

The metric is NMAE = MAE / train-split std, so the target functional is the
conditional MEDIAN, not the mean (ARCHITECTURE 8.2 specifies the mean, which
is the wrong estimator -- the leaderboard's own Entity Median 0.4278 beats
Entity Mean 0.4551). Both are reported here so the difference is visible.

Baselines from the same harness:
  entity-mean/median -- the entity's own historical per-window value. This is
                        the bar the published Entity Mean / Entity Median rows
                        represent, and it is a strong one.
  global-median      -- one constant for every row.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from ledger.data.cache import load_corpus
from ledger.model.load import build_model, corpus_kwargs, load_checkpoint
from ledger.queries import _query_states
from ledger import readout as ro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--rebuild-at-test", action="store_true")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = load_checkpoint(args.ckpt)
    t = ck["args"]
    max_len = max(1, int(t.get("max_len", 512)) // 2)
    ds = get_dataset(args.dataset, download=True)
    cutoff = pd.Timestamp(ds.test_timestamp if args.rebuild_at_test
                          else ds.val_timestamp)
    corpus = load_corpus(args.dataset, cutoff, db=ds.get_db,
                         **corpus_kwargs(ck))
    model = build_model(ck, corpus, dev)

    task = get_task(args.dataset, args.task, download=True)
    key = (args.dataset, args.task)
    if key not in ro.REG_PREDICATES:
        raise SystemExit(f"no regression predicate registered for {key}")
    pred, value_col = ro.REG_PREDICATES[key]

    # NMAE normaliser comes from the TRAIN split, per the leaderboard metric
    tr = task.get_table("train", mask_input_cols=False).df
    norm = float(tr[task.target_col].std()) or 1.0

    tbl = task.get_table(args.split, mask_input_cols=False)
    df = tbl.df
    if args.rows and len(df) > args.rows:
        idx = np.random.default_rng(0).choice(len(df), args.rows, replace=False)
        df = df.iloc[np.sort(idx)].reset_index(drop=True)
    ent = task.entity_table
    pk = corpus.schema.pkey_index[ent]
    rows = df[task.entity_col].map(pk).fillna(-1).to_numpy(dtype=np.int64)
    cuts = (pd.to_datetime(df[task.time_col]).astype("int64") // 10**9).to_numpy()
    y = df[task.target_col].to_numpy(dtype=np.float64)
    delta_s = float(task.timedelta.total_seconds())
    known = rows >= 0

    print(f"task {args.dataset}/{args.task} split={args.split} rows={len(df):,} "
          f"delta={task.timedelta} target={task.target_col} "
          f"value_col={value_col} train_std={norm:.4f}")

    mu = np.zeros(len(df)); ev = np.ones(len(df))
    for s in range(0, int(known.sum()), args.batch_size):
        r = rows[known][s:s + args.batch_size]; c = cuts[known][s:s + args.batch_size]
        h, ne, _, _ = _query_states(model, corpus, ent, r, c, max_len,
                                    dev, branch="temporal")
        t_last = np.array([corpus.time[corpus.history(ent, int(x), before=int(cc))[-1]]
                           if len(corpus.history(ent, int(x), before=int(cc))) else cc
                           for x, cc in zip(r, c)])
        elapsed = np.maximum(c - t_last, 1.0)
        m_, _, _ = ro.qualified_hazard(model, corpus.schema, h, elapsed,
                                       delta_s, pred)
        sl = slice(s, s + len(r))
        mu[np.flatnonzero(known)[sl]] = m_.cpu().numpy()
        if value_col:
            v = ro.expected_value(model, corpus.schema, h, pred.tables[0],
                                  value_col)
            if v is not None:
                ev[np.flatnonzero(known)[sl]] = v.cpu().numpy()

    # entity-history baseline: the entity's own past value per window
    hist_rate = np.zeros(len(df))
    want = [corpus.schema.fact_tables[x].table_idx for x in pred.tables
            if x in corpus.schema.fact_tables]
    for i, (r, c) in enumerate(zip(rows, cuts)):
        if r < 0: continue
        ids = corpus.history(ent, int(r), before=int(c))
        if not len(ids): continue
        n_q = int(np.isin(corpus.table_idx[ids], want).sum())
        span = max(float(c - corpus.time[ids[0]]), delta_s)
        hist_rate[i] = n_q * delta_s / span

    preds = {
        "LEDGER  mu (mean)":       mu * ev,
        "LEDGER  Poisson median":  ro.poisson_median(mu) * ev,
        "baseline entity-rate":  hist_rate * (ev if value_col else 1.0),
        "baseline global-median": np.full(len(df), float(np.median(tr[task.target_col]))),
        "baseline zero":         np.zeros(len(df)),
    }
    print(f"\n  {'readout':26s} {'MAE':>10s} {'NMAE':>8s}")
    for name, p in preds.items():
        mae = float(np.mean(np.abs(p - y)))
        print(f"  {name:26s} {mae:10.4f} {mae/norm:8.4f}")


if __name__ == "__main__":
    main()
