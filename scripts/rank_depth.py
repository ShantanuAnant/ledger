"""Where does the TRUE destination actually land in LEDGER's full ranking?

`rank_destinations` scores the whole catalogue densely (`q @ cand.T`), so a
link-prediction miss is never a shortlist miss -- the candidate was always
scored, it just lost. This asks by how much.

The distinction decides which lever is worth building:

  truth lands shallow (say rank < 200) but outside eval_k
      -> the SCORE ORDER is wrong near the top and a reranker / a
         learned combination of the existing per-pair features can
         recover it. Cheap.
  truth lands deep (rank in the tens of thousands)
      -> `q @ c` does not put the right item anywhere near the top and no
         reordering of a top-k slice reaches it. That is retrieval
         capacity, and it costs a retrain.

Val split only, and deliberately: the corpus is cut at `val_timestamp`, which
is exactly the regime the checkpoint trained in, so a deep rank here cannot be
blamed on distribution shift or on the categorical-vocabulary drift that
`eval_rec.py --rebuild-at-test` has to repair.
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
from ledger.model.load import build_model, corpus_kwargs, load_checkpoint
from ledger.queries import build_path_map, rank_destinations

DEPTHS = (1, 5, 10, 12, 20, 50, 100, 500, 1000, 5000, 10000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--max-rows", type=int, default=2000,
                    help="rows sampled from the val target table. The full "
                         "table times a 10k-deep ranking does not fit in "
                         "memory on the larger catalogues.")
    ap.add_argument("--depth", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=0)
    ap.add_argument("--project-through", default=None)
    ap.add_argument("--n_stage1", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle", type=int, nargs="*", default=None,
                    help="also report the CEILING on reranking: for each of "
                         "these shortlist sizes N, reorder our own top-N by "
                         "putting the true destinations first and score that. "
                         "No reranker, learned or otherwise, can beat it, so "
                         "if the ceiling at N is below ID-GNN then no amount "
                         "of work on the readout closes that task and the "
                         "deficit is retrieval.")
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

    task = get_task(args.dataset, args.task, download=True)
    df = task.get_table("val", mask_input_cols=False).df
    n_dst = corpus.schema.entity_counts[task.dst_entity_table]
    depth = min(args.depth, n_dst)

    rng = np.random.default_rng(args.seed)
    if len(df) > args.max_rows:
        df = df.iloc[rng.choice(len(df), args.max_rows, replace=False)]
    print(f"task: {args.dataset}/{args.task}  eval_k={task.eval_k}  "
          f"catalogue={n_dst:,}  rows={len(df):,}  depth={depth:,}")

    src_map = corpus.schema.pkey_index[task.src_entity_table]
    dst_map = corpus.schema.pkey_index[task.dst_entity_table]
    src_rows = df[task.src_entity_col].map(src_map).fillna(-1) \
                                      .to_numpy(dtype=np.int64)
    cuts = (pd.to_datetime(df[task.time_col]).astype("int64")
            // 10 ** 9).to_numpy()

    project = None
    rank_table = args.project_through or task.dst_entity_table
    if args.project_through is not None:
        pm = build_path_map(ds.get_db(), corpus, args.project_through,
                            task.dst_entity_table, corpus.cutoff)
        project = {"i": torch.as_tensor(pm["inter"], device=dev),
                   "f": torch.as_tensor(pm["final"], device=dev),
                   "n_final": pm["n_final"]}

    cooc = None
    if getattr(model.who, "n_feats", 0) >= 8 \
            and task.src_entity_table != rank_table:
        from ledger.queries import CoocTable
        cooc = CoocTable.build(corpus, task.src_entity_table, rank_table,
                               cache_key=args.dataset)

    known = src_rows >= 0
    ranked = np.full((len(df), depth), -1, dtype=np.int64)
    n_events = np.zeros(len(df), dtype=np.int64)
    if known.any():
        ranked[known], n_events[known] = rank_destinations(
            model, corpus, task.src_entity_table, src_rows[known],
            cuts[known], rank_table, depth, max_len=max_len,
            batch_size=args.batch_size, device=dev, cooc=cooc,
            win_days=(task.timedelta.total_seconds() / 86400.0
                      if getattr(model.who, "window", False) else None),
            n_stage1=args.n_stage1, project=project, retarget=retarget)

    # Position of each ground-truth destination in that ranking. A row's
    # label is an ARRAY of pkeys; `best` is the shallowest of them, which is
    # the one that decides whether the row contributes to precision at all.
    pos = {d: 0 for d in DEPTHS if d <= depth}
    best_ranks, n_truth, n_found, n_scored = [], 0, 0, 0
    for i, lab in enumerate(df[task.dst_entity_col].to_numpy()):
        if not known[i] or n_events[i] == 0:
            continue          # cold start: eval_rec substitutes popularity
        n_scored += 1
        tr = np.array([dst_map[p] for p in np.atleast_1d(lab)
                       if p in dst_map], dtype=np.int64)
        if not len(tr):
            continue
        n_truth += 1
        where = np.nonzero(np.isin(ranked[i], tr))[0]
        if len(where):
            n_found += 1
            best_ranks.append(int(where[0]) + 1)
            for d in pos:
                if where[0] < d:
                    pos[d] += 1

    br = np.array(best_ranks) if best_ranks else np.array([0])
    out = {
        "dataset": args.dataset, "task": args.task, "ckpt": args.ckpt,
        "eval_k": int(task.eval_k), "catalogue": int(n_dst),
        "rows_scored": n_scored, "rows_with_truth": n_truth,
        "found_within_depth": n_found, "depth": int(depth),
        "hit_rate": {str(d): (v / n_truth if n_truth else 0.0)
                     for d, v in pos.items()},
        "median_rank_when_found": float(np.median(br)),
        "p90_rank_when_found": float(np.percentile(br, 90)),
    }
    print(f"\nrows scored (warm) {n_scored:,}   with a mappable label "
          f"{n_truth:,}   truth inside depth {n_found:,} "
          f"({100 * n_found / max(n_truth, 1):.1f}%)")
    print("\n  depth   hit-rate")
    for d in sorted(pos):
        mark = "  <- eval_k" if d == task.eval_k else ""
        print(f"  {d:>6}   {100 * pos[d] / max(n_truth, 1):7.3f}%{mark}")
    print(f"\nmedian rank of truth (when inside depth): "
          f"{out['median_rank_when_found']:.0f}")
    print(f"p90    rank of truth (when inside depth): "
          f"{out['p90_rank_when_found']:.0f}")

    if args.oracle is not None:
        pk = corpus.schema.pkey_values[task.dst_entity_table]
        k = task.eval_k
        tab = task.get_table("val", mask_input_cols=False)
        if len(df) != len(tab.df):
            tab = type(tab)(df=df,
                            fkey_col_to_pkey_table=tab.fkey_col_to_pkey_table,
                            pkey_col=tab.pkey_col, time_col=tab.time_col)
        base = task.evaluate(pk[ranked[:, :k]], tab)
        print(f"\n  as-ranked        MAP {100 * base['link_prediction_map']:8.4f}")
        out["oracle"] = {"as_ranked": 100 * base["link_prediction_map"]}
        labs = df[task.dst_entity_col].to_numpy()
        for N in sorted(args.oracle):
            N = min(N, depth)
            re = ranked[:, :k].copy()
            for i in range(len(df)):
                if not known[i] or n_events[i] == 0:
                    continue
                tr = np.array([dst_map[p] for p in np.atleast_1d(labs[i])
                               if p in dst_map], dtype=np.int64)
                head = ranked[i, :N]
                hit = head[np.isin(head, tr)]
                rest = head[~np.isin(head, tr)]
                re[i] = np.concatenate([hit, rest])[:k]
            m = task.evaluate(pk[re], tab)
            print(f"  oracle@{N:<9} MAP {100 * m['link_prediction_map']:8.4f}"
                  f"   (perfect reordering of our own top {N})")
            out["oracle"][f"oracle@{N}"] = 100 * m["link_prediction_map"]

    if args.json_out:
        with open(args.json_out, "a") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
