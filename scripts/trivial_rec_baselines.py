"""Trivial baselines for EVERY RelBench link-prediction (recommendation) task.

Question this answers: how much of the recommendation leaderboard is
reproducible without learning anything, using only recency and popularity?

Method (schema-agnostic -- works on any link-prediction task):
  History is built from the task's OWN train/val tables, which contain past
  positive (src -> [dst]) links with timestamps. This is information every
  method is entitled to use, and it avoids hardcoding each database's
  interaction table.

  For split=val  : history = train rows with timestamp < val cutoff
  For split=test : history = train + val rows with timestamp < test cutoff

  An assertion enforces the cutoff, so no evaluation-window link can leak in.

Baselines, all ranked to eval_k:
  pop_all     most-linked dst over all history
  pop_recent  most-linked dst in the trailing window before the cutoff
  user_hist   the src's own most-recently-linked dst, back-filled with pop_recent
  user_freq   the src's own most-frequently-linked dst, back-filled with pop_recent

Also reports the repeat rate: what fraction of true evaluation links are
(src, dst) pairs that already appeared in history. This is the ceiling for any
pure re-link heuristic, and distinguishes "re-purchase prediction" tasks from
genuine cold-start discovery.
"""

from __future__ import annotations

import argparse
import traceback

import numpy as np
import pandas as pd

from relbench.tasks import get_task, get_task_names
from relbench.base import TaskType

# published RelBench-paper test MAP for the RDL best model, for reference
PUBLISHED = {
    ("rel-amazon", "user-item-purchase"): 0.74,
    ("rel-amazon", "user-item-rate"): 0.87,
    ("rel-amazon", "user-item-review"): 0.47,
    ("rel-avito", "user-ad-visit"): 3.66,
    ("rel-hm", "user-item-purchase"): 2.81,
    ("rel-stack", "user-post-comment"): 12.72,
    ("rel-stack", "post-post-related"): 10.83,
    ("rel-trial", "condition-sponsor-run"): 11.36,
    ("rel-trial", "site-sponsor-run"): 19.00,
}


def topk_pad(seq, fill, k):
    out, seen = [], set()
    for x in seq:
        if x in seen:
            continue
        out.append(x); seen.add(x)
        if len(out) == k:
            return out
    for x in fill:
        if x in seen:
            continue
        out.append(x); seen.add(x)
        if len(out) == k:
            return out
    while len(out) < k:
        out.append(fill[0] if len(fill) else 0)
    return out


def explode_history(task, splits, cutoff):
    """-> DataFrame [src, dst, time] of positive links strictly before cutoff."""
    frames = []
    for sp in splits:
        df = task.get_table(sp, mask_input_cols=False).df
        df = df[df[task.time_col] < cutoff]
        if len(df) == 0:
            continue
        frames.append(df[[task.src_entity_col, task.dst_entity_col,
                          task.time_col]])
    if not frames:
        return pd.DataFrame(columns=[task.src_entity_col, task.dst_entity_col,
                                     task.time_col])
    h = pd.concat(frames, ignore_index=True)
    h = h.explode(task.dst_entity_col).dropna(subset=[task.dst_entity_col])
    return h


def run_task(dataset, task_name, split, recent_frac=0.15):
    task = get_task(dataset, task_name, download=True)
    if task.task_type != TaskType.LINK_PREDICTION:
        return None
    k = task.eval_k
    src, dst, tcol = task.src_entity_col, task.dst_entity_col, task.time_col

    target = task.get_table(split, mask_input_cols=False)
    cutoff = pd.Timestamp(target.df[tcol].min())
    splits = ["train"] if split == "val" else ["train", "val"]
    hist = explode_history(task, splits, cutoff)

    if len(hist) == 0:
        print(f"  {dataset}/{task_name}: no history before cutoff, skipped")
        return None
    assert hist[tcol].max() < cutoff, "LEAK: history reaches the cutoff"

    # popularity pools
    pop_all = hist[dst].value_counts().index.to_numpy()
    span = cutoff - hist[tcol].min()
    recent = hist[hist[tcol] >= cutoff - span * recent_frac]
    pop_recent = recent[dst].value_counts().index.to_numpy()
    if len(pop_recent) < k:
        pop_recent = pop_all
    fill = list(pop_recent[: max(k, 200)])

    hs = hist.sort_values(tcol, ascending=False)
    by_recency = hs.groupby(src)[dst].apply(list).to_dict()
    by_freq = (hist.groupby([src, dst]).size().sort_values(ascending=False)
                   .reset_index(level=1).groupby(level=0)[dst]
                   .apply(list).to_dict())
    hist_sets = {u: set(v) for u, v in by_recency.items()}

    # repeat rate
    n_true = n_rep = n_cold = 0
    for u, truth in zip(target.df[src], target.df[dst]):
        truth = np.asarray(truth)
        n_true += len(truth)
        s = hist_sets.get(u)
        if s is None:
            n_cold += 1
            continue
        n_rep += int(np.isin(truth, list(s)).sum())

    users = target.df[src].to_numpy()

    def build(mode):
        if mode == "pop_all":
            row = list(pop_all[:k]); return np.asarray([row] * len(users))
        if mode == "pop_recent":
            row = list(pop_recent[:k]); return np.asarray([row] * len(users))
        src_map = by_recency if mode == "user_hist" else by_freq
        return np.asarray([topk_pad(src_map.get(u, []), fill, k) for u in users])

    out = {
        "dataset": dataset, "task": task_name, "split": split,
        "eval_rows": len(target.df), "eval_k": k,
        "true_pairs": n_true,
        "cold_frac": 100 * n_cold / max(len(target.df), 1),
        "repeat_rate": 100 * n_rep / max(n_true, 1),
    }
    for mode in ("pop_all", "pop_recent", "user_hist", "user_freq"):
        m = task.evaluate(build(mode), target)
        out[mode] = 100 * m["link_prediction_map"]
    out["best_trivial"] = max(out[m] for m in
                              ("pop_all", "pop_recent", "user_hist", "user_freq"))
    out["published"] = PUBLISHED.get((dataset, task_name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["rel-hm", "rel-stack", "rel-trial"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", default="runs/trivial_rec_baselines.csv")
    args = ap.parse_args()

    rows = []
    for d in args.datasets:
        for tn in get_task_names(d):
            try:
                r = run_task(d, tn, args.split)
            except Exception:
                print(f"  FAIL {d}/{tn}"); traceback.print_exc(); continue
            if r is None:
                continue
            rows.append(r)
            # write after EVERY task: a late OOM must not lose the sweep
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"  done {d}/{tn}: best_trivial={r['best_trivial']:.2f} "
                  f"published={r['published']}  [saved]", flush=True)

    if not rows:
        print("no link-prediction tasks found"); return
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*104}\nTRIVIAL BASELINES vs PUBLISHED  (MAP@k, split={args.split})\n{'='*104}")
    hdr = (f"{'dataset':<12}{'task':<24}{'rep%':>6}{'cold%':>7}"
           f"{'pop_all':>9}{'pop_rec':>9}{'u_hist':>8}{'u_freq':>8}"
           f"{'BEST':>8}{'pub':>7}{'%of pub':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        pub = r["published"]
        frac = f"{100*r['best_trivial']/pub:>8.0f}%" if pub else "       -"
        print(f"{r['dataset']:<12}{r['task']:<24}"
              f"{r['repeat_rate']:>6.1f}{r['cold_frac']:>7.1f}"
              f"{r['pop_all']:>9.3f}{r['pop_recent']:>9.3f}"
              f"{r['user_hist']:>8.3f}{r['user_freq']:>8.3f}"
              f"{r['best_trivial']:>8.3f}"
              f"{(f'{pub:.2f}' if pub else '-'):>7}{frac}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
