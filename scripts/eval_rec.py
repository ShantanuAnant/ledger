"""Score a trained LEDGER checkpoint on a RelBench link-prediction task.

Zero-shot: the model was trained only on next-event prediction over the raw
event stream. It has never seen this task, its labels, or its train split.

Default split is `val`, because the training corpus is cut at
`dataset.val_timestamp` -- so a val query at time T has exactly the history the
model was trained with. Scoring `test` requires a corpus rebuilt at
`test_timestamp` (still leakage-free: those events precede the test window),
which `--rebuild-at-test` does.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from ledger.data.cache import load_corpus
from ledger.data.corpus import EventCorpus
from ledger.data.schema import build_schema
from ledger.model.load import build_model, corpus_kwargs, load_checkpoint
from ledger.queries import evaluate_recommendation, refresh_states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="rel-hm")
    ap.add_argument("--task", default="user-item-purchase")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--rebuild-at-test", action="store_true",
                    help="build the corpus at test_timestamp instead of "
                         "val_timestamp (needed to score the test split)")
    ap.add_argument("--no-pin-schema", action="store_true",
                    help="do NOT re-encode the test corpus with the training "
                         "cutoff's categorical vocabularies (debug only: the "
                         "vocabularies are index-shifted across cutoffs)")
    ap.add_argument("--max_len", type=int, default=0,
                    help="events per entity. 0 = match the checkpoint's "
                         "training context, which is its `max_len // 2` (a "
                         "training row is a PACKED budget shared between "
                         "entities). Evaluating a long-context model at short "
                         "context understates it.")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--refresh", action="store_true",
                    help="recompute the entity-state table with the loaded "
                         "weights before scoring (A4)")
    ap.add_argument("--refresh-entity", default=None,
                    help="entity table to sweep during refresh "
                         "(default: the one the checkpoint trained on)")
    ap.add_argument("--refresh-mode", default="mean",
                    choices=["mean", "ema"],
                    help="exact order-independent mean, or the training-time "
                         "EMA write (order-dependent; for comparison only)")
    ap.add_argument("--project-through", default=None,
                    help="intermediate entity table for a 2-hop destination "
                         "(rel-f1: races; rel-trial: studies). The checkpoint "
                         "ranks this table and the scores are marginalized "
                         "onto the task's destination.")
    ap.add_argument("--n_stage1", type=int, default=2048,
                    help="shortlist size the reranker rescores")
    ap.add_argument("--pred-out", default="",
                    help="directory to save the ranked destination-id matrix "
                         "[n_rows, eval_k] as .npy, aligned to the eval "
                         "split's own row order, for "
                         "scripts/make_submission.py.")
    ap.add_argument("--json-out", default="",
                    help="append one JSON line {task, metrics, ckpt, split} "
                         "here, so a sweep can collect cells without "
                         "re-parsing stdout")
    ap.add_argument("--no-cold-fallback", action="store_true",
                    help="rank cold-start entities from their zero state "
                         "instead of falling back to recent popularity")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = load_checkpoint(args.ckpt)
    targs = ck["args"]
    print("checkpoint trained with:", {k: targs[k] for k in
          ("dataset", "entity", "steps", "dim", "layers", "states")})

    max_len = args.max_len or max(1, int(targs.get("max_len", 512)) // 2)
    print(f"eval context: {max_len} events/entity "
          f"(training max_len {targs.get('max_len')})")

    ds = get_dataset(args.dataset, download=True)
    cutoff = pd.Timestamp(ds.test_timestamp if args.rebuild_at_test
                          else ds.val_timestamp)
    # Cached + memory-mapped: `ds.get_db` is only called on a cache miss, and
    # on rel-amazon that call alone reads a 7 GB parquet.
    # The corpus must be built exactly the way the checkpoint's was: these
    # three flags change which events an entity's history holds and which
    # columns every event carries, so a mismatch is a shape error at best and
    # a silently different model at worst. `self_rows` was already recorded in
    # the checkpoint and was NOT read back here.
    corpus_kw = corpus_kwargs(ck)
    corpus = load_corpus(args.dataset, cutoff, db=ds.get_db, **corpus_kw)

    # Categorical vocabularies are built from PRE-CUTOFF rows and indexed by
    # `sorted(unique values)`, so a value first seen between the val and test
    # cutoffs does not merely extend the vocabulary -- it shifts the index of
    # every value that sorts after it. On rel-stack `badges.Name` goes 320 ->
    # 328 categories with 158 existing values remapped. Encoding test events
    # with the test vocabulary and reading them through embeddings trained on
    # the val vocabulary is silently wrong (rel-amazon and rel-hm happen to be
    # unaffected -- their vocabularies are identical across cutoffs -- which is
    # exactly why this went unnoticed). Pin the column specs to the training
    # cutoff and re-encode.
    if args.rebuild_at_test and not args.no_pin_schema:
        train_schema = load_corpus(args.dataset,
                                   pd.Timestamp(ds.val_timestamp),
                                   db=ds.get_db, verbose=False,
                                   **corpus_kw).schema
        drift = [f"{t}.{c.name}" for t, s in train_schema.fact_tables.items()
                 for c in s.columns if c.kind == "categorical" and not c.hashed
                 and any(d.name == c.name and d.vocab != c.vocab
                         for d in corpus.schema.fact_tables[t].columns)]
        if drift:
            print(f"schema drift across cutoffs on {drift}; re-encoding the "
                  f"test corpus with the training-cutoff column specs")
            schema = build_schema(ds.get_db(), cutoff,
                                  denorm_fk=corpus_kw["denorm_fk"],
                                  child_aggs=corpus_kw["child_aggs"])
            for name, spec in schema.fact_tables.items():
                if name in train_schema.fact_tables:
                    spec.columns = train_schema.fact_tables[name].columns
            corpus = EventCorpus.build(ds.get_db(), cutoff, schema=schema,
                                       **corpus_kw)
        else:
            print("schema check: categorical vocabularies identical across "
                  "cutoffs, cached test corpus is usable as-is")

    # One shared reconstruction (ledger/model/load.py). This script used to
    # keep its own kwargs list, which omitted window_head/query_feats/
    # feat_path -- so a checkpoint trained with the window objective could not
    # be scored on a recommendation task at all.
    model = build_model(ck, corpus, dev)

    if args.refresh:
        ent = args.refresh_entity or targs["entity"]
        print(f"refreshing entity states from `{ent}` histories ...")
        refresh_states(model, corpus, ent, max_len=max_len,
                       batch_size=args.batch_size, device=dev,
                       mode=args.refresh_mode)

    # Path retargeting is a property of the CHECKPOINT, not a CLI choice: a
    # model trained with --retarget ranks the final entity directly and its
    # repeat features must be expanded the same way, or train and eval
    # silently disagree about what the features mean.
    retarget = None
    if targs.get("retarget"):
        from ledger.queries import build_path_map
        import numpy as _np
        inter, final = [x.strip() for x in str(targs["retarget"]).split(":")]
        pm = build_path_map(ds.get_db(), corpus, inter, final, corpus.cutoff)
        iv, fv = pm["inter"], pm["final"]
        o = _np.argsort(iv, kind="stable"); iv, fv = iv[o], fv[o]
        n_i = corpus.schema.entity_counts[inter]
        off = _np.zeros(n_i + 1, dtype=_np.int64)
        _np.cumsum(_np.bincount(iv, minlength=n_i), out=off[1:])
        retarget = {"inter": inter, "final": final, "off": off, "val": fv}
        print(f"retarget: {inter} -> {final} via `{pm['via']}`, "
              f"{len(fv):,} pre-cutoff pairs")

    task = get_task(args.dataset, args.task, download=True)
    print(f"task: {args.dataset}/{args.task}  split={args.split}  "
          f"k={task.eval_k}  dst catalogue="
          f"{corpus.schema.entity_counts[task.dst_entity_table]:,}")

    res = evaluate_recommendation(model, corpus, task, split=args.split,
                                  max_len=max_len, device=dev,
                                  batch_size=args.batch_size,
                                  cold_fallback=not args.no_cold_fallback,
                                  cooc_cache_key=args.dataset,
                                  n_stage1=args.n_stage1,
                                  project_through=args.project_through,
                                  retarget=retarget,
                                  db=ds.get_db(),
                                  return_pred=bool(args.pred_out))
    diag = res.pop("_diagnostics")
    if args.pred_out:
        import pathlib as _pl, numpy as _np
        d = _pl.Path(args.pred_out); d.mkdir(parents=True, exist_ok=True)
        stem = f"{args.dataset}__{args.task}__{args.split}"
        # KEYED, not positional. relbench 2.1.2 and 3.x return the same test
        # rows in different order (v3 sorts by timestamp), so a bare array
        # written into v3's table is silently permuted -- the same trap that
        # turned rel-f1 driver-position's 0.6331 into 0.7051 on the entity
        # side. Save the keys and let the consumer join on them.
        _p = res.pop("_pred")
        _tt = task.get_table(args.split, mask_input_cols=False).df
        keyed = _tt[[task.src_entity_col, task.time_col]].copy()
        keyed["pred_probe"] = list(_np.asarray(_p))
        keyed.to_parquet(d / f"{stem}.parquet", index=False)
        print(f"  saved {len(keyed):,} keyed rankings -> {d}/{stem}.parquet")
    else:
        res.pop("_pred", None)
    print("\n=== LEDGER (zero-shot) ===")
    for k, v in res.items():
        print(f"  {k:<28} {100*v:8.4f}")
    print("\ndiagnostics:", json.dumps(diag, indent=2, default=str))

    if args.json_out:
        # percent, matching how the leaderboard publishes MAP
        with open(args.json_out, "a") as fh:
            fh.write(json.dumps({
                "dataset": args.dataset, "task": args.task,
                "split": args.split, "ckpt": args.ckpt,
                "metrics": {k: 100 * v for k, v in res.items()}}) + "\n")


if __name__ == "__main__":
    main()
