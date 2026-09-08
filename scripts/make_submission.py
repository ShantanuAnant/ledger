"""Turn keyed LEDGER predictions into official RelBench submission CSVs.

Run with the SUBMISSION venv (relbench 3.x), not the training venv:

    .venv-submit/bin/python scripts/make_submission.py preds_raw preds

WHY A MERGE AND NOT A POSITIONAL WRITE. The training environment pins
relbench 2.1.2 and the submission tooling needs 3.x. Both return the same
test ROWS, but in DIFFERENT ORDER -- v3 sorts by timestamp, 2.1.2 does not.
Writing a positionally-aligned vector into v3's table therefore scrambles it
silently: rel-f1 driver-position scored 0.6331 in our harness and 0.7051
through `relbench.submit` before this was found. The producer saves the key
columns alongside each prediction and this joins on them.

`--column` picks which readout to submit per task: `probe`, `zero`, or
`sel` (default) which reads the per-task val decision from a JSON mapping
`"<dataset> <task>" -> "probe"|"zero"`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

import relbench
from relbench.submit import LEADERBOARD_TASKS, write_prediction_table

HUB = "stanford-star/relbench-v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--column", default="sel", choices=["probe", "zero", "sel"])
    ap.add_argument("--sel-map", default="",
                    help="JSON: {'<dataset> <task>': 'probe'|'zero'}")
    args = ap.parse_args()

    raw = pathlib.Path(args.raw_dir)
    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    sel = json.load(open(args.sel_map)) if args.sel_map else {}

    wanted = [t for fam in LEADERBOARD_TASKS.values() for t in fam]
    done, missing, bad = [], [], []
    for full in wanted:
        dsname, tname = full.split("/")
        f = raw / f"{dsname}__{tname}__{args.split}.parquet"
        if not f.exists():
            missing.append(full); continue
        ds = relbench.load_dataset(f"{HUB}/{dsname}")
        task = ds.load_task(tname)
        official = task.get_table(args.split, mask_input_cols=True).df
        keys = [task.entity_col, task.time_col] \
            if hasattr(task, "entity_col") else \
            [task.src_entity_col, task.time_col]

        ours = pd.read_parquet(f)
        col = ({"probe": "pred_probe", "zero": "pred_zero"}[args.column]
               if args.column != "sel"
               else {"probe": "pred_probe", "zero": "pred_zero"}[
                   sel.get(f"{dsname} {tname}", "probe")])
        if col not in ours.columns:
            col = "pred_probe"
        # normalise dtypes so the join cannot silently miss
        for k in keys:
            if np.issubdtype(official[k].dtype, np.datetime64):
                ours[k] = pd.to_datetime(ours[k])
            else:
                ours[k] = ours[k].astype(official[k].dtype)
        merged = official[keys].merge(ours[keys + [col]], on=keys, how="left")
        if len(merged) != len(official) or merged[col].isna().any():
            bad.append((full, f"{int(merged[col].isna().sum())} unmatched of "
                              f"{len(official)}"))
            continue
        pred = merged[col].to_numpy()
        if pred.dtype == object:                      # recommendation
            pred = np.stack([np.asarray(x) for x in pred])
        write_prediction_table(task, pred, out / f"{dsname}__{tname}.csv",
                               split=args.split)
        done.append(full)

    print(f"wrote {len(done)} CSVs -> {out}")
    if missing:
        print(f"\nMISSING predictions for {len(missing)}:")
        for m in missing: print("   ", m)
    if bad:
        print(f"\nJOIN FAILED for {len(bad)}:")
        for m, why in bad: print("   ", m, "--", why)
    return 1 if (missing or bad) else 0


if __name__ == "__main__":
    sys.exit(main())
