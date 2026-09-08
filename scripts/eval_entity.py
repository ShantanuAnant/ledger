"""Zero-shot readout of a RelBench ENTITY task from the WHEN head.

The point of the experiment (ARCHITECTURE 8.1). A churn label is
`no qualifying event in (t, t+delta]`, which is exactly the survival function
of the temporal point process the WHEN head already fits. If the head is
calibrated, classification needs no new parameters at all.

THE CONDITIONING IS THE WHOLE TRICK. The head models the gap from the
entity's LAST event, not from the query time. At query time t the entity has
already been silent for `elapsed = t - t_last`, so the quantity wanted is the
CONDITIONAL survival

    P(churn) = P(dt > elapsed + delta | dt > elapsed)
             = S(elapsed + delta) / S(elapsed)

Reading S(delta) directly would ignore the elapsed time and score a customer
who bought yesterday the same as one silent for a year -- which is most of the
signal in the task. Computed in log space; AUROC only needs the ranking, so
the log-ratio is used as the score.

Baselines from the same harness, so the comparison is apples to apples:
  recency  -- score = elapsed. "Silent for longer => more likely churned."
              Trivial, and on churn tasks it is genuinely strong.
  count    -- score = -n_events. "Fewer past events => more likely churned."
  random   -- sanity floor, should give AUROC ~50.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from ledger.data.cache import load_corpus
from ledger.data.corpus import EventCorpus
from ledger.data.schema import build_schema
from ledger.model.load import build_model, load_checkpoint
from ledger.queries import _query_states
from ledger import readout as ro


@torch.no_grad()
def survival_scores(model, corpus, entity_table, rows, cuts, delta_s,
                    max_len, device, batch_size=256, pred=None, k=0,
                    key_=None, q_src="where", q_blend=0.0, corpus_for_q=None):
    """-> (score, n_events, elapsed, mu, q). Higher score = label more likely.

    With `pred` the score is the QUALIFIED-hazard readout (ledger/readout.py);
    without it, the plain any-event survival. On a single-fact-table database
    the two coincide because q = 1, which is the correctness check on the
    generalisation.
    """
    out = np.zeros(len(rows), dtype=np.float64)
    nev = np.zeros(len(rows), dtype=np.int64)
    elapsed_all = np.zeros(len(rows), dtype=np.float64)
    mu_all = np.zeros(len(rows), dtype=np.float64)
    q_all = np.zeros(len(rows), dtype=np.float64)
    for s in range(0, len(rows), batch_size):
        r, c = rows[s:s + batch_size], cuts[s:s + batch_size]
        h, ne, _, _ = _query_states(model, corpus, entity_table, r, c,
                                    max_len, device, branch="temporal")
        # time of each entity's last pre-cutoff event
        t_last = np.zeros(len(r), dtype=np.int64)
        for i, (row, cut) in enumerate(zip(r, c)):
            ids = corpus.history(entity_table, int(row), before=int(cut))
            t_last[i] = corpus.time[ids[-1]] if len(ids) else cut
        elapsed = np.maximum(c - t_last, 1.0)
        e = torch.as_tensor(elapsed, device=device, dtype=torch.float32)
        d = torch.as_tensor(delta_s, device=device, dtype=torch.float32)
        if pred is not None:
            qo = None
            if q_src in ("empirical", "blend"):
                qo = ro.empirical_q(corpus, entity_table, r, c, pred)
            mu, lam, q = ro.qualified_hazard(
                model, corpus.schema, h, elapsed, delta_s, pred,
                q_override=qo, q_blend=(q_blend if q_src == "blend" else 0.0))
            val = ro.task_score(mu, key_)
            out[s:s + len(r)] = val.double().cpu().numpy()
            mu_all[s:s + len(r)] = mu.cpu().numpy()
            q_all[s:s + len(r)] = q.cpu().numpy()
        else:
            log_s_after = model.when.log_survival(h, e + d)
            log_s_now = model.when.log_survival(h, e)
            out[s:s + len(r)] = (log_s_after - log_s_now).float().cpu().numpy()
        nev[s:s + len(r)] = ne
        elapsed_all[s:s + len(r)] = elapsed
    return out, nev, elapsed_all, mu_all, q_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--rebuild-at-test", action="store_true")
    ap.add_argument("--max_len", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--rows", type=int, default=0, help="subsample for speed")
    ap.add_argument("--q-source", default="where",
                    choices=["where", "empirical", "blend"],
                    help="thinning probability from the WHERE head, from the "
                         "entity's own history composition, or a geometric "
                         "blend of the two")
    ap.add_argument("--q-blend", type=float, default=0.5)
    ap.add_argument("--plain-survival", action="store_true",
                    help="ignore the task predicate and read the any-event "
                         "survival (the pre-2026-08-25 readout)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = load_checkpoint(args.ckpt)
    t = ck["args"]
    max_len = args.max_len or max(1, int(t.get("max_len", 512)) // 2)

    ds = get_dataset(args.dataset, download=True)
    cutoff = pd.Timestamp(ds.test_timestamp if args.rebuild_at_test
                          else ds.val_timestamp)
    corpus_kw = dict(self_rows=bool(t.get("self_rows")),
                     denorm_fk=bool(t.get("denorm_fk")),
                     child_aggs=bool(t.get("child_aggs")))
    corpus = load_corpus(args.dataset, cutoff, db=ds.get_db, **corpus_kw)
    # Categorical vocabularies are index-shifted across cutoffs (HANDOFF
    # 10.1a): rel-stack badges.Name goes 320 -> 328 with 158 values remapped,
    # which fails the state_dict load outright. Same pinning as eval_rec.py.
    if args.rebuild_at_test:
        train_schema = load_corpus(args.dataset, pd.Timestamp(ds.val_timestamp),
                                   db=ds.get_db, verbose=False,
                                   **corpus_kw).schema
        drift = [f"{t_}.{c.name}" for t_, sp in train_schema.fact_tables.items()
                 for c in sp.columns if c.kind == "categorical" and not c.hashed
                 and any(d.name == c.name and d.vocab != c.vocab
                         for d in corpus.schema.fact_tables[t_].columns)]
        if drift:
            print(f"schema drift on {drift}; re-encoding at training vocab")
            sch = build_schema(ds.get_db(), cutoff,
                               denorm_fk=corpus_kw["denorm_fk"],
                               child_aggs=corpus_kw["child_aggs"])
            for nm, sp in sch.fact_tables.items():
                if nm in train_schema.fact_tables:
                    sp.columns = train_schema.fact_tables[nm].columns
            corpus = EventCorpus.build(ds.get_db(), cutoff, schema=sch,
                                       **corpus_kw)

    model = build_model(ck, corpus, dev)

    task = get_task(args.dataset, args.task, download=True)
    tbl = task.get_table(args.split, mask_input_cols=False)
    df = tbl.df
    if args.rows and len(df) > args.rows:
        idx = np.random.default_rng(0).choice(len(df), args.rows,
                                              replace=False)
        df = df.iloc[np.sort(idx)].reset_index(drop=True)
        tbl = type(tbl)(df=df, fkey_col_to_pkey_table=tbl.fkey_col_to_pkey_table,
                        pkey_col=tbl.pkey_col, time_col=tbl.time_col)

    ent = task.entity_table
    pk = corpus.schema.pkey_index[ent]
    rows = df[task.entity_col].map(pk).fillna(-1).to_numpy(dtype=np.int64)
    cuts = (pd.to_datetime(df[task.time_col]).astype("int64")
            // 10 ** 9).to_numpy()
    known = rows >= 0
    delta_s = float(task.timedelta.total_seconds())

    print(f"task {args.dataset}/{args.task} split={args.split} "
          f"rows={len(df):,} delta={task.timedelta} "
          f"entity={ent} in-corpus={100*known.mean():.1f}%")

    key = (args.dataset, args.task)
    pred = None if args.plain_survival else ro.PREDICATES.get(key)
    k = ro.THRESHOLDS.get(key, 0)
    if pred is not None:
        n_tbl = len(corpus.schema.fact_tables)
        print(f"predicate: tables={pred.tables} k={k} "
              f"(of {n_tbl} fact tables) -- {pred.note}")
    elif not args.plain_survival:
        print(f"no predicate registered for {key}; using any-event survival")

    score = np.zeros(len(df)); nev = np.zeros(len(df), dtype=np.int64)
    elapsed = np.full(len(df), delta_s)
    mu = np.zeros(len(df)); qq = np.zeros(len(df))
    if known.any():
        sc, ne, el, m_, q_ = survival_scores(
            model, corpus, ent, rows[known], cuts[known], delta_s, max_len,
            dev, args.batch_size, pred=pred, k=k, key_=key,
            q_src=args.q_source, q_blend=args.q_blend)
        score[known], nev[known], elapsed[known] = sc, ne, el
        mu[known], qq[known] = m_, q_
    if pred is not None and known.any():
        print(f"thinning q: mean={qq[known].mean():.4f} "
              f"median={np.median(qq[known]):.4f}   "
              f"mu: mean={mu[known].mean():.4f} "
              f"median={np.median(mu[known]):.4f}")
    # An entity with no pre-cutoff history has no evidence against churn.
    cold = ~known | (nev == 0)
    score[cold] = np.max(score[~cold]) if (~cold).any() else 0.0

    rng = np.random.default_rng(0)
    readouts = {
        ("LEDGER qualified-hazard" if pred is not None
         else "LEDGER survival (zero-shot)"): score,
        "baseline: recency": (elapsed if key in ro.ABSENCE
                              else -elapsed).astype(np.float64),
        "baseline: count": (-nev if key in ro.ABSENCE
                            else nev).astype(np.float64),
        "baseline: random": rng.random(len(df)),
    }
    print(f"cold (no pre-cutoff history): {100*cold.mean():.1f}%\n")
    res = {}
    for name, v in readouts.items():
        m = task.evaluate(v, tbl)
        res[name] = m
        print(f"  {name:28s} " + "  ".join(
            f"{k}={100*val:.2f}" for k, val in m.items()
            if k in ("roc_auc", "average_precision")))
    print("\n" + json.dumps({k: {kk: float(vv) for kk, vv in v.items()}
                             for k, v in res.items()}, indent=2))


if __name__ == "__main__":
    main()
