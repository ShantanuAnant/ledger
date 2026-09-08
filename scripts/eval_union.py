"""Score a UNION checkpoint (`--datasets`) on one database's entity task.

    PYTHONPATH=. .venv/bin/python scripts/eval_union.py \
        --ckpt runs/...-xdb_joint-bestwin.pt --dataset rel-f1 \
        --task driver-top3 --split val

WHY A SEPARATE SCRIPT. `eval_window.py` builds a single-database corpus and a
model from that schema. A union checkpoint's tokenizer, WHAT/WINDOW heads and
`table_emb` are sized for EVERY database it trained on, so loading it against
one database's schema fails the state_dict -- loudly, which is the good case.
Rebuilding the identical union and then reading one database's entities out of
it is the only way the two arms of the cross-database A/B are comparable at
all: the control's number must come from the same readout code, differing only
in which weights it loads.

The union is rebuilt from `ck["args"]["datasets"]`, so the ordering and the
namespacing match training exactly -- `build_union` sorts by dataset name and
assigns table indices in that order, and a different order would silently
permute `table_emb` rows.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")
from eval_window import query_states                      # noqa: E402
from ledger import win_readout as wr                        # noqa: E402
from ledger.data.cache import load_corpus                   # noqa: E402
from ledger.data.schema import Schema                       # noqa: E402
from ledger.data.union import build_union, qualify          # noqa: E402
from ledger.model.load import build_model, load_checkpoint  # noqa: E402
from relbench.datasets import get_dataset                 # noqa: E402
from relbench.tasks import get_task                       # noqa: E402


def rebuild_union(ck, cutoff_kind="val", rebuild_at_test=False, target=None):
    a = ck["args"]
    specs = [x.strip() for x in a["datasets"].split(",") if x.strip()]
    raw, ents = {}, {}
    for sp in specs:
        dsn, _, es = sp.partition(":")
        d = get_dataset(dsn, download=True)
        # Only the TARGET database moves to the test cutoff; the others are
        # context and their cutoff is irrelevant to the scored rows.
        kind = "test" if (rebuild_at_test and dsn == target) else cutoff_kind
        raw[dsn] = load_corpus(
            dsn, pd.Timestamp(getattr(d, f"{kind}_timestamp")), db=d.get_db,
            self_rows=bool(a.get("self_rows")),
            denorm_fk=bool(a.get("denorm_fk")),
            child_aggs=bool(a.get("child_aggs")),
            path_expand=str(a.get("path_expand") or ""), verbose=False)
        ents[dsn] = [e for e in es.split("+") if e]
    cor, sch = build_union(raw)
    del raw
    for c in cor.values():
        own = c.schema
        c.schema = Schema(fact_tables=sch.fact_tables,
                          entity_tables=own.entity_tables,
                          entity_counts=own.entity_counts,
                          pkey_values=own.pkey_values,
                          pkey_index=own.pkey_index)
    import copy
    view = copy.copy(cor[sorted(cor)[0]])
    view.schema = sch
    for f in ("feat_num", "feat_cat", "links", "row_of",
              "hist_index", "hist_offset"):
        m = {}
        for c in cor.values():
            m.update(getattr(c, f))
        setattr(view, f, m)
    return cor, sch, view, ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    ck = load_checkpoint(a.ckpt)
    if not ck["args"].get("datasets"):
        raise SystemExit("not a union checkpoint; use scripts/eval_window.py")

    key = (a.dataset, a.task)
    q = wr.QUERIES.get(key)
    if q is None:
        raise SystemExit(f"no query registered for {key}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    task = get_task(a.dataset, a.task, download=True)
    tbl = task.get_table(a.split, mask_input_cols=False)

    cor, sch, view, _ = rebuild_union(
        ck, rebuild_at_test=(a.split == "test"), target=a.dataset)
    # the query names UNQUALIFIED tables; the union speaks qualified ones
    qq = type(q)(**{**q.__dict__,
                    "tables": [qualify(a.dataset, t) for t in q.tables]})
    for fld in ("cat_in", "cat_not", "num_le", "num_gt"):
        f = getattr(qq, fld)
        if f is not None:
            object.__setattr__(qq, fld, (qualify(a.dataset, f[0]),) + tuple(f[1:]))

    model = build_model(ck, view, dev)
    model.eval()

    ent = qualify(a.dataset, task.entity_table)
    c = cor[a.dataset]
    pk = c.schema.pkey_index[ent]
    rows = pd.Series(tbl.df[task.entity_col]).map(pk).to_numpy()
    known = pd.notna(rows)
    rows = np.where(known, rows, -1).astype(np.int64)
    cuts = (tbl.df[task.time_col].astype("int64") // 10**9).to_numpy()
    horizon_s = float(task.timedelta.total_seconds())
    max_len = max(1, int(ck["args"].get("max_len", 512)) // 2)

    pred = np.zeros(len(tbl.df), dtype=np.float64)
    if known.any():
        z, _ = query_states(model, c, ent, rows[known], cuts[known],
                            horizon_s, max_len, dev, a.batch_size,
                            query_feats=bool(ck["args"].get("query_feats")),
                            query_dyn=bool(ck["args"].get("query_dyn")),
                            query_nbr=bool(ck["args"].get("query_nbr")))
        pred[known] = wr.predict(model, z, qq, "auto").cpu().numpy()
    if (~known).any() and known.any():
        pred[~known] = np.median(pred[known])

    m = {k: float(v) for k, v in task.evaluate(pred, tbl).items()}
    print(f"{a.dataset}/{a.task} [{a.split}] union={sch.num_fact_tables} tables "
          + "  ".join(f"{k}={v:.4f}" for k, v in m.items()
                      if k in ("roc_auc", "average_precision", "mae")))
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps({"dataset": a.dataset, "task": a.task,
                                "split": a.split, "ckpt": a.ckpt,
                                "results": {"LEDGER window-head": m}}) + "\n")


if __name__ == "__main__":
    main()
