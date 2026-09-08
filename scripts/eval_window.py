"""Score a RelBench ENTITY task (classification or regression) from a
WindowHead checkpoint.

    PYTHONPATH=. .venv/bin/python scripts/eval_window.py \
        --ckpt runs/rel-f1-drivers-nonparam-0-winf1-best.pt \
        --dataset rel-f1 --task driver-top3 --split val

One script covers both boards because the head answers both from the same
object: an occurrence label is `P(N > k)` and a regression target is a
quantile of `N` or of a column sum over the same window. See
`ledger/win_readout.py` for what is exact and what is approximated.

Baselines are printed from the same harness so the comparison is honest:
`recency` and `count` are the trivial readouts, and for regression the
train-split global median is the estimator the leaderboard's own "Entity
Median" row is a per-entity version of.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from relbench.base import TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from ledger.data.batching import PackedBatcher
from ledger.data.cache import load_corpus
from ledger.data.corpus import EventCorpus
from ledger.data.schema import build_schema
from ledger.model.load import build_model, load_checkpoint
from ledger.model.window import cat_states
from ledger import win_readout as wr


def load_model(ckpt, corpus, device):
    ck = load_checkpoint(ckpt)
    t = ck["args"]
    if not t.get("window_head"):
        raise SystemExit(
            f"{ckpt} was not trained with --window_head; use "
            f"scripts/eval_entity.py for the hazard readout instead")
    return build_model(ck, corpus, device), t


@torch.no_grad()
def query_states(model, corpus, entity_table, rows, cuts, horizon_s,
                 max_len, device, batch_size=512, query_feats=False,
                 query_dyn=False, query_nbr=False,
                 return_feats=False):
    """WindowHead query state for each (entity, cutoff). -> ([N, d], elapsed)

    The query token is placed AT the cutoff, so the backbone sees how long the
    entity has been silent. Without it `h` is the state after the last event
    and is identical for one day and one year of silence.
    """
    zs, els, fts = [], [], []
    for s in range(0, len(rows), batch_size):
        r, c = rows[s:s + batch_size], cuts[s:s + batch_size]
        batch, keep = PackedBatcher.pack_histories(
            corpus, entity_table, r, c, max_len=max_len, device=device,
            query_token=True, horizon_s=horizon_s,
            query_feats=query_feats, query_dyn=query_dyn,
            query_nbr=query_nbr)
        # the WINDOW head lives on the temporal branch (Backbone.BRANCHES);
        # inert when the checkpoint was trained with --branch_layers 0
        h = model.encode(batch, branch="temporal")[keep["b"], keep["last_l"]]
        el = torch.as_tensor(keep["elapsed"], device=device,
                             dtype=torch.float32)
        hz = torch.full_like(el, float(horizon_s))
        zs.append(model.win.state(
            h, el, hz,
            batch["query_pos"]["feats"] if model.win.feat_path else None))
        els.append(keep["elapsed"])
        if return_feats:
            # The RAW query-token features (per-table counts, recency,
            # per-category counters). They reach the model only through
            # `state()` when the head has a wide-and-deep path, so a head
            # without one leaves them unused -- but an in-context readout
            # wants them, and they are the very features a frozen probe
            # found beat the backbone state (RESEARCH 2026-08-25).
            f = batch["query_pos"].get("feats")
            fts.append(None if f is None else f.detach().cpu().numpy())
    z, el = cat_states(zs), np.concatenate(els)
    if not return_feats:
        return z, el
    F = (None if not fts or any(f is None for f in fts)
         else np.concatenate(fts, axis=0))
    return z, el, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--rebuild-at-test", action="store_true")
    ap.add_argument("--corpus-at", default="cutoff",
                    choices=["cutoff", "rowmax"],
                    help="`cutoff` stops the corpus at the split boundary "
                         "(the pre-2026-08-25 behaviour). `rowmax` extends it "
                         "to the latest eval row's own timestamp, which the "
                         "benchmark permits -- leakage is enforced per row by "
                         "history(before=cut) either way -- and matters on the "
                         "datasets whose eval table spans years past the "
                         "boundary (rel-f1, rel-trial).")
    ap.add_argument("--max_len", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--form", default="auto",
                    choices=["auto", "poisson", "binomial", "fraction"],
                    help="arrival model for an occurrence label; see "
                         "ledger/win_readout.classify")
    ap.add_argument("--all-forms", action="store_true",
                    help="score every arrival model, not just --form")
    ap.add_argument("--json", default="", help="append one result line here")
    ap.add_argument("--device", default="",
                    help="force 'cpu' or 'cuda'; default auto-detect")
    args = ap.parse_args()

    key = (args.dataset, args.task)
    if key in wr.UNEXPRESSIBLE:
        raise SystemExit(f"{key} is not expressible: {wr.UNEXPRESSIBLE[key]}")
    q = wr.QUERIES.get(key)
    if q is None:
        raise SystemExit(f"no query registered for {key}")

    # `torch.cuda.is_available()` returning True is NOT proof the device
    # works: a GPU whose contexts were leaked by a SIGKILLed job reports
    # available and then fails on the first allocation (HANDOFF 10.5e2). The
    # override also makes a CPU dry-run of the whole readout possible on a
    # small dataset, which is how this path is tested without a GPU.
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = get_dataset(args.dataset, download=True)
    task = get_task(args.dataset, args.task, download=True)
    tbl = task.get_table(args.split, mask_input_cols=False)

    cutoff = pd.Timestamp(ds.test_timestamp if args.rebuild_at_test
                          else ds.val_timestamp)
    if args.corpus_at == "rowmax":
        # An eval row at time T may legally use every event before T -- that
        # is the benchmark's rule and what the GNN baselines do with a
        # time-masked neighbour loader. Capping the corpus at the SPLIT
        # boundary instead is stricter than the rule and, on the datasets
        # whose eval tables span years past that boundary, throws most of the
        # available history away: rel-f1's val rows run to 2008-03 against a
        # 2005-01 corpus, and 30.8% of them then have no history at all.
        # Leakage is still enforced per row by `history(before=cut)`, and the
        # column statistics stay pinned to the TRAINING cutoff below, so
        # nothing about the model's inputs comes from after the row's time.
        cutoff = max(cutoff, pd.Timestamp(tbl.df[task.time_col].max()))
    train_cutoff = pd.Timestamp(ds.val_timestamp)
    # the corpus must be built the same way the checkpoint's was, or the
    # entity histories differ between training and inference
    _ta = load_checkpoint(args.ckpt)["args"]
    corpus_kw = dict(self_rows=bool(_ta.get("self_rows")),
                     denorm_fk=bool(_ta.get("denorm_fk")),
                     child_aggs=bool(_ta.get("child_aggs")),
                     path_expand=str(_ta.get("path_expand") or ""))
    if cutoff == train_cutoff:
        corpus = load_corpus(args.dataset, cutoff, db=ds.get_db, **corpus_kw)
    else:
        # Categorical vocabularies are index-shifted across cutoffs (HANDOFF
        # 10.1a): a value first seen after the training cutoff does not extend
        # the vocabulary, it shifts the index of everything sorting after it.
        # That both fails the state_dict load and silently remaps the category
        # ids a task filter names. Pin every column spec -- vocabularies,
        # standardization constants, and hence the bucket edges the numeric
        # predicates are read on -- to the training cutoff.
        train_schema = load_corpus(args.dataset, train_cutoff,
                                   db=ds.get_db, verbose=False,
                                   **corpus_kw).schema
        sch = build_schema(ds.get_db(), cutoff,
                           denorm_fk=corpus_kw["denorm_fk"],
                           child_aggs=corpus_kw["child_aggs"])
        for nm, sp in sch.fact_tables.items():
            if nm in train_schema.fact_tables:
                sp.columns = train_schema.fact_tables[nm].columns
        print(f"building corpus at {cutoff} with training-cutoff column specs")
        corpus = EventCorpus.build(ds.get_db(), cutoff, schema=sch,
                                   **corpus_kw)

    if _ta.get("no_query_cat"):
        corpus.schema._qcat_layout = []      # match the training layout

    if "path_expand" in q.needs:
        # `needs` only proves the FLAG was set, not that it named the right
        # path. A wrong spec builds a corpus in which the query's tables are
        # still unreachable from the entity, the rate is 0 for every row, and
        # the readout answers "no qualifying events, ever" with no error --
        # the same class of silent wrongness `_cat_rate`'s fallback caused on
        # rel-avito user-clicks. Measure the reachability instead.
        _ent = task.entity_table
        _want = [sp.table_idx for nm, sp in corpus.schema.fact_tables.items()
                 if nm in q.tables]
        _off, _idx = corpus.hist_offset[_ent], corpus.hist_index[_ent]
        _n = len(_off) - 1
        _s = np.random.default_rng(0).choice(_n, min(4000, _n), replace=False)
        _hit = sum(1 for r in _s if np.isin(
            corpus.table_idx[_idx[_off[r]:_off[r + 1]]], _want).any())
        _frac = _hit / max(len(_s), 1)
        print(f"path_expand: {q.tables} reachable from {_ent} in "
              f"{100 * _frac:.1f}% of sampled histories")
        if _frac == 0.0:
            raise SystemExit(
                f"path_expand={corpus_kw['path_expand']!r} leaves {q.tables} "
                f"unreachable from {_ent}: every rate would be 0. Check the "
                f"<entity>:<bridge>:<events> spec.")

    missing = [f for f in q.needs if not corpus_kw.get(f)]
    if missing:
        # A query naming a derived column against a corpus built without it
        # would fall through `_cat_rate`'s "column absent" branch and answer a
        # DIFFERENT question with no error. Refuse instead.
        raise SystemExit(
            f"{key} needs {missing}; this checkpoint was trained without "
            f"them (corpus flags {corpus_kw}). Retrain with "
            + " ".join("--" + f for f in missing))

    model, targs = load_model(args.ckpt, corpus, dev)
    max_len = args.max_len or max(1, int(targs.get("max_len", 512)) // 2)

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
    horizon_s = float(task.timedelta.total_seconds())
    is_cls = task.task_type == TaskType.BINARY_CLASSIFICATION

    print(f"task {args.dataset}/{args.task} split={args.split} "
          f"rows={len(df):,} delta={task.timedelta} entity={ent} "
          f"in-corpus={100*known.mean():.1f}% agg={q.agg} k={q.k}"
          + (f" -- {q.note}" if q.note else ""))

    pred = np.zeros(len(df), dtype=np.float64)
    nev = np.zeros(len(df), dtype=np.int64)
    elapsed = np.full(len(df), horizon_s, dtype=np.float64)
    rate = np.zeros(len(df), dtype=np.float64)
    if known.any():
        z, el = query_states(model, corpus, ent, rows[known], cuts[known],
                             horizon_s, max_len, dev, args.batch_size,
                             query_feats=bool(targs.get("query_feats")),
                             query_dyn=bool(targs.get("query_dyn")),
                             query_nbr=bool(targs.get("query_nbr")))
        _pq = wr.predict_query(model, z, q)
        if _pq is not None:
            print("readout: query-conditioned head (ledger/qsample.py)")
            pred[known] = _pq.cpu().numpy()
        else:
            print("readout: composed marginals (win_readout)")
            pred[known] = wr.predict(model, z, q, args.form).cpu().numpy()
        rate[known] = wr.rate(model, z, q).cpu().numpy()
        elapsed[known] = el
        nev[known] = [len(corpus.history(ent, int(r), before=int(c)))
                      for r, c in zip(rows[known], cuts[known])]
    print(f"rate: mean={rate[known].mean():.4f} "
          f"median={np.median(rate[known]):.4f} "
          f"p(zero-rate)={float((rate[known] < 1e-6).mean()):.3f}")

    # Entities absent from the corpus have no evidence at all. For an
    # occurrence label the neutral fill is the population base rate direction;
    # we use the cold group's own predicted value where it exists and the
    # median otherwise, so cold rows neither help nor hurt the ranking.
    cold = ~known | (nev == 0)
    if cold.any() and (~cold).any():
        pred[cold] = np.median(pred[~cold])
    print(f"cold (no pre-cutoff history): {100*cold.mean():.1f}%\n")

    rng = np.random.default_rng(0)
    if is_cls:
        sign = 1.0 if q.absence else -1.0
        cands = {
            "LEDGER window-head": pred,}
        if args.all_forms and known.any():
            for f in ("poisson", "binomial", "fraction"):
                v = np.zeros(len(df))
                v[known] = wr.classify(model, z, q, f).cpu().numpy()
                if cold.any() and (~cold).any():
                    v[cold] = np.median(v[~cold])
                cands[f"  form={f}"] = v
        cands.update({
            "baseline: recency": sign * elapsed,
            "baseline: count": -sign * nev.astype(np.float64),
            "baseline: random": rng.random(len(df)),
        })
    else:
        med = float(task.get_table("train").df[task.target_col].median())
        cands = {
            "LEDGER window-head": pred,
            "baseline: zero": np.zeros(len(df)),
            "baseline: train median": np.full(len(df), med),
        }

    res = {}
    for name, v in cands.items():
        m = task.evaluate(v, tbl)
        res[name] = {k: float(val) for k, val in m.items()}
        keep = ("roc_auc", "average_precision") if is_cls else ("mae", "r2")
        print(f"  {name:24s} " + "  ".join(
            f"{k}={m[k]:.4f}" for k in keep if k in m))

    if not is_cls:
        std = float(task.get_table("train").df[task.target_col].std())
        for name in res:
            res[name]["nmae"] = res[name]["mae"] / std
        print("\n  NMAE = MAE / train-split std (the leaderboard metric):")
        for name, m in res.items():
            print(f"  {name:24s} nmae={m['nmae']:.4f}")

    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps({"dataset": args.dataset, "task": args.task,
                                "split": args.split, "ckpt": args.ckpt,
                                "results": res}) + "\n")
    print("\n" + json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
