"""Training loop. Config via argparse; logs CSV per run.

Usage:
    python -m ledger.train --dataset rel-amazon --entity customer \
        --who_feats --logq --hard_negs --steps 30000 --workers 48

What changed for the B200 (RESEARCH.md 2026-08-24), and why each mattered:

  * `--workers`: batch construction is single-threaded numpy and was measured
    at 962 ms/step on rel-amazon against a 28 ms GPU step -- 97% of wall time
    with the accelerator idle. Batches are now built by a pool of worker
    processes over the memory-mapped corpus (see prefetch.py). 64 workers take
    the batcher to 17 ms, i.e. off the critical path.
  * cached corpora (`data/cache.py`): every process used to rebuild the corpus
    from parquet. Concurrent runs now mmap one shared copy.
  * `--amp bf16` + FlexAttention + fused AdamW + TF32: the compute path itself.
  * `--eval_every`: RESEARCH.md's standing lesson is that WHO loss is NOT a
    proxy for MAP -- three separate occasions where retrieval loss improved
    and the graded metric did not move, or moved the other way. A loss curve
    is therefore not evidence of progress, so the ranking metric is computed
    DURING training and written to the same CSV.
  * `--cutoff`: was hardcoded to val_timestamp, which made test-split numbers
    unobtainable without editing the file (the standing "A3" item).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time as wallclock

import numpy as np
import pandas as pd
import torch

from .data.cache import load_corpus
from .data.batching import PackedBatcher
from .data.prefetch import PrefetchLoader
from .model.ledger import LEDGER


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rel-f1")
    ap.add_argument("--datasets", default="",
                    help="JOINT CROSS-DATABASE training. Comma-separated "
                         "`<dataset>:<entity>[+<entity>]` specs, e.g. "
                         "`rel-f1:drivers,rel-event:users`. Each corpus is "
                         "namespaced (`rel-f1::results`) and merged into ONE "
                         "schema, so the per-table ModuleDicts gain entries "
                         "while the TRUNK, the time encoding and the WHEN "
                         "head are shared -- that sharing is the hypothesis. "
                         "One batch is drawn from one database; the mixing "
                         "is across steps. This is JOINT training, NOT "
                         "leave-one-database-out zero-shot: a held-out "
                         "database's tables would have no projections and no "
                         "table embedding, so the model could not read them "
                         "at all. Overrides --dataset/--entity when set.")
    ap.add_argument("--entity", default="drivers",
                    help="entity table(s) whose histories form sequences; "
                         "comma-separated trains both sides of a link (D1), "
                         "e.g. --entity customer,article")
    ap.add_argument("--who_feats", action="store_true",
                    help="add interaction features to the WHO score (D3): "
                         "repeat/recency, candidate recent-trend, 2-hop "
                         "co-occurrence")
    ap.add_argument("--window", action="store_true",
                    help="window-conditioned multi-positive WHO: supervise on "
                         "every destination in (t, t+W] for a sampled horizon "
                         "W, and condition the head on W (Tier2 #5)")
    ap.add_argument("--window_days", type=float, default=0.0,
                    help="with --window, concentrate sampled horizons "
                         "log-normally around this many days instead of "
                         "log-uniform over 1-365. Set it to the task's "
                         "timedelta (rel-hm 7, rel-amazon/rel-stack 91, "
                         "rel-trial 365). 0 keeps the task-agnostic prior.")
    ap.add_argument("--window_sigma", type=float, default=0.6,
                    help="log-space spread of that prior; keeps the head "
                         "horizon-conditioned rather than fitted to one W")
    ap.add_argument("--logq", action="store_true",
                    help="logQ correction on sampled negatives (Tier2 #6)")
    ap.add_argument("--hard_negs", action="store_true",
                    help="draw a share of negatives from the 2-hop "
                         "co-occurrence neighbourhood (Tier2 #6)")
    ap.add_argument("--rerank", action="store_true",
                    help="train the cross-interaction reranker (Tier2 #4)")
    ap.add_argument("--w_rerank", type=float, default=1.0)
    ap.add_argument("--no_cooc", action="store_true",
                    help="with --who_feats, leave the 2-hop co-occurrence "
                         "feature at zero (ablation)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_rows", type=int, default=8)
    ap.add_argument("--objective", choices=["both", "window", "next_event"],
                    default="both",
                    help="which pretraining objective shapes the backbone. "
                         "'both' (default) is the historical multi-task "
                         "setup. 'window' trains SOLELY on the "
                         "window-aggregate objective -- the next-event heads "
                         "are not run and their targets are not built -- "
                         "which is the query family all 21 entity tasks "
                         "reduce to (ARCHITECTURE 8.2a). 'window' implies "
                         "--window_head.")
    ap.add_argument("--n_neg", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--min_lr_frac", type=float, default=0.1,
                    help="cosine floor as a fraction of peak lr "
                         "(ARCHITECTURE 9.2); 1.0 disables decay")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--states", default="learned",
                    choices=["learned", "nonparam", "hybrid"],
                    help="entity-state table: v0 nn.Embedding, the detached "
                         "bf16 EMA (HANDOFF 4.1), or learned-for-small + "
                         "EMA-for-large")
    ap.add_argument("--no_ema_write", action="store_true",
                    help="never write the EMA state table. With --states "
                         "hybrid this leaves learned destination embeddings "
                         "plus all-zero source states -- the configuration "
                         "that produced rel-amazon MAP 2.2882, reachable "
                         "explicitly now that the write gate is fixed.")
    ap.add_argument("--state_momentum", type=float, default=0.1)
    ap.add_argument("--state_normalize", action="store_true",
                    help="EMA on unit vectors (variant 1)")
    ap.add_argument("--state_reduce", default="mean", choices=["mean", "last"],
                    help="how duplicate ids collapse in one state write; "
                         "'last' keeps the most recent token for the sequence "
                         "entity, matching what queries.py reads at inference")
    ap.add_argument("--state_center", action="store_true",
                    help="subtract running global mean at read (variant 2)")
    ap.add_argument("--uniformity", type=float, default=0.0,
                    help="weight on the candidate-uniformity term (variant 3)")
    ap.add_argument("--learned_tables", default="",
                    help="comma-separated entity tables to give TRAINED "
                         "embeddings, overriding --learned_max. Use this to "
                         "name the DESTINATION catalogue: destination is not "
                         "a function of size (posts > users on rel-stack, "
                         "AdsInfo is the largest table on rel-avito), so the "
                         "size gate cannot select it there at any threshold.")
    ap.add_argument("--learned_max", type=int, default=200_000,
                    help="hybrid: tables at or below this size are learned")
    ap.add_argument("--window_head", action="store_true",
                    help="train the WindowHead: insert query tokens at "
                         "sampled times t_q and supervise the AGGREGATES of "
                         "(t_q, t_q+D] -- per-table arrival rates, per-column "
                         "rates by category and by numeric bucket, and "
                         "quantiles of counts and sums. This is the "
                         "self-supervised objective the classification and "
                         "regression readouts actually need; see "
                         "ledger/model/window.py.")
    ap.add_argument("--self_rows", action="store_true",
                    help="file a fact table's own row in its own entity's "
                         "history. A table that is both a fact table and an "
                         "entity table otherwise never appears in its own "
                         "history: on rel-stack that hides account age from "
                         "every user, and on rel-trial it hides enrollment, "
                         "phase and study type from every study. See "
                         "EventCorpus.build.")
    ap.add_argument("--denorm_fk", action="store_true",
                    help="copy an FK target's low-cardinality categorical and "
                         "numeric columns onto the event's own feature "
                         "vector. Unblocks a filter whose column lives in a "
                         "DIFFERENT table (rel-trial `outcomes.outcome_type`) "
                         "and a sum whose column lives in a dimension table "
                         "(rel-amazon `product.price`). See data/denorm.py.")
    ap.add_argument("--rate_balance", default="none",
                    choices=["none", "cb"],
                    help="CLASS-BALANCE the Poisson rate terms over zero vs "
                         "non-zero cells. The rate heads see windows that "
                         "are overwhelmingly empty, so the plain mean is "
                         "dominated by zeros and the head settles on the "
                         "global base rate: measured as a predicted rate of "
                         "0.003 on rel-trial study-adverse and 0.033 on "
                         "rel-event user-attendance, both scoring at the "
                         "constant. `cb` gives the non-empty cells the same "
                         "total weight as the empty ones. Applies to the "
                         "table, per-category and per-bucket rates -- a "
                         "filtered readout reads the per-category one.")
    ap.add_argument("--rate_zero_scale", type=float, default=1.0,
                    help="extra multiplier on the ZERO cells' weight under "
                         "--rate_balance cb. 1.0 balances the two groups "
                         "exactly; <1 leans further toward the rare "
                         "non-empty windows.")
    ap.add_argument("--query_head", action="store_true",
                    help="train a QUERY-CONDITIONED head on RANDOM window "
                         "aggregations sampled from the schema (see "
                         "ledger/qsample.py). Supervises the SCALAR a task is "
                         "scored on, with a pinball loss at the median, "
                         "instead of leaving the readout to compose "
                         "marginals it was never calibrated against. "
                         "Queries never come from the task registry, and "
                         "--eval_entity's own signature is excluded from "
                         "sampling, so the evaluated query is held out.")
    ap.add_argument("--n_rand_q", type=int, default=8,
                    help="random queries sampled per step for --query_head")
    ap.add_argument("--w_query", type=float, default=1.0,
                    help="weight of the random-query term")
    ap.add_argument("--path_expand", default="",
                    help="`<entity>:<bridge>:<events>` (comma-separated) -- "
                         "file events reachable only through a bridge table "
                         "into the entity's history. rel-trial site-success "
                         "needs "
                         "`facilities:facilities_studies:outcome_analyses`; "
                         "without it a facility's history holds "
                         "`facilities_studies` and nothing else and the "
                         "readout rates events the entity never saw. Changes "
                         "which events a history contains, so it is part of "
                         "the corpus cache identity (see "
                         "EventCorpus._expand_path).")
    ap.add_argument("--child_aggs", action="store_true",
                    help="add count / sum / per-category counts of a fact "
                         "table's CHILDREN to its own columns, so a two-hop "
                         "quantity becomes a column of an event the entity's "
                         "history already contains (rel-avito SearchInfo "
                         "gains the number of clicked ads in that search).")
    ap.add_argument("--min_hist", type=int, default=2,
                    help="minimum events for an entity to enter the TRAINING "
                         "pool. The default of 2 silently excludes 42.7%% of "
                         "rel-stack users while 52.5%% of user-badge eval rows "
                         "have fewer than 2 pre-cutoff events -- a training/"
                         "eval population mismatch on the worst task. With "
                         "--self_rows every entity has at least its own row, "
                         "so 1 admits everyone.")
    ap.add_argument("--n_query", type=int, default=4,
                    help="query tokens inserted per packed sequence")
    ap.add_argument("--query_feats", action="store_true",
                    help="feed per-fact-table history counters (count, "
                         "days-since-last, count-in-window) into the query "
                         "token. See PackedBatcher.query_feats for why the "
                         "model is handed these rather than left to infer "
                         "them.")
    ap.add_argument("--query_dyn", action="store_true",
                    help="add the classification/regression dynamics block to "
                         "the query token: per-table empirical window-count "
                         "distribution, cadence/overdue ratios, weekday "
                         "match, and RECENT per-category counters. Targets "
                         "the churn, 'more than k', and rare-sub-type task "
                         "shapes. Requires --query_feats.")
    ap.add_argument("--query_nbr", action="store_true",
                    help="add the NEIGHBOUR-AGGREGATE block to the query "
                         "token: per (fact table, foreign-key slot), how "
                         "many distinct neighbours the entity has attached "
                         "to, how concentrated that attachment is, and how "
                         "established / active / recent those neighbours are "
                         "in their OWN histories. This is the sequence-native "
                         "form of the 2-hop aggregation the GNN baselines "
                         "get for free, and it targets the tasks whose label "
                         "is a property of a LINKED entity -- rel-f1 "
                         "driver-dnf (constructor reliability) above all. "
                         "See data/batching.query_nbr_layout. Requires "
                         "--query_feats.")
    ap.add_argument("--no_query_cat", action="store_true",
                    help="with --query_feats, leave the per-(table, category) "
                         "counter block OUT of the query token (ablation). "
                         "The control arm for that block: everything else "
                         "about the run is unchanged, so the difference is "
                         "attributable.")
    ap.add_argument("--q_tail", type=float, default=0.0,
                    help="fraction of query times drawn log-uniformly back "
                         "from the LATEST admissible one instead of uniformly "
                         "over the entity's span. Every eval query sits at the "
                         "split boundary, so a uniform draw trains the head "
                         "mostly on a period whose activity level differs from "
                         "the one it is asked about.")
    ap.add_argument("--win_buckets", type=int, default=8,
                    help="quantile buckets per numeric column; a numeric "
                         "predicate (`position <= 3`) is read as the rate "
                         "over the buckets below its threshold")
    ap.add_argument("--win_max_events", type=int, default=256,
                    help="cap on per-column detail per window; the total "
                         "count target stays exact above it")
    ap.add_argument("--dense_window", action="store_true",
                    help="attach window targets to ORDINARY EVENT TOKENS as "
                         "well as to the inserted query tokens. The sparse "
                         "objective supervises ~2.3%% of tokens (n_query 6, "
                         "two sequences per 512-token row) against the "
                         "next-event heads' 100%%, which confounded the "
                         "2026-08-25 --objective window experiment. Rate and "
                         "count-quantile terms only; the per-column detail "
                         "stays sparse. Requires --window_head.")
    ap.add_argument("--n_dense", type=int, default=64,
                    help="dense window targets per packed sequence")
    ap.add_argument("--feat_path", action="store_true",
                    help="wide-and-deep window head: add a GLM on the RAW "
                         "query features directly to the rate / count-"
                         "quantile / per-category outputs, bypassing the "
                         "trunk, with the deep branch zero-initialised so "
                         "training starts AT the GLM. Addresses the probe "
                         "result that 33 count features beat the hidden "
                         "state 84.8 to 74.2 and that concatenating the two "
                         "is worse than the features alone. Requires "
                         "--query_feats.")
    ap.add_argument("--win_balance", choices=["legacy", "weighted"],
                    default="legacy",
                    help="how WindowHead.loss combines its terms. `legacy` is "
                         "an unweighted mean over all of them, which on "
                         "rel-stack is 52 terms of which every entity task "
                         "reads exactly ONE (the per-table rate) -- ~98%% of "
                         "the window gradient goes to marginals no readout "
                         "queries. `weighted` groups them into rate / count-"
                         "quantile / detail and normalises each group to a "
                         "per-output mean, making the rate term 1/3 of the "
                         "objective. See ledger/model/window.py.")
    for k, d in (("rate", 1.0), ("qcount", 1.0), ("detail", 1.0),
                 ("windense", 1.0)):
        ap.add_argument(f"--w_{k}", type=float, default=d,
                        help=f"weight on the {k} group of the window loss; "
                             f"only used with --win_balance weighted")
    for k, d in (("when", 1.0), ("where", 1.0), ("who", 1.0), ("what", 1.0),
                 ("win", 1.0)):
        ap.add_argument(f"--w_{k}", type=float, default=d,
                        help=f"weight on L_{k} in the total loss")
    ap.add_argument("--learn_loss_weights", action="store_true",
                    help="learn the per-head loss weights by uncertainty "
                         "weighting instead of fixing them. The --w_* values "
                         "become the INITIALISATION. Addresses ARCHITECTURE "
                         "9.6: a constant w_when trades classification "
                         "against recommendation, and the right constant "
                         "differs by dataset and by step.")
    ap.add_argument("--lw_clamp", type=float, default=3.0,
                    help="bound on |log sigma^2| with --learn_loss_weights, "
                         "i.e. the learned weight stays in "
                         "[exp(-c), exp(c)]. Stops a head being switched off "
                         "entirely -- w_when=0 diverges the WHEN head and "
                         "the classification readout reads out of it.")

    # -- branched backbone (ARCHITECTURE 9.6) -------------------------------
    ap.add_argument("--branch_layers", type=int, default=0,
                    help="per-head-group layers ON TOP of the shared trunk. "
                         "0 reproduces the single-trunk model exactly. "
                         "WHO/reranker read the 'retrieval' branch; "
                         "WHEN/WHERE/WHAT/WINDOW read the 'temporal' one.")
    ap.add_argument("--branch_kind", default="transformer",
                    choices=["transformer", "mlp"],
                    help="branch block type. 'mlp' is position-wise and much "
                         "cheaper but cannot re-mix across tokens.")

    # -- B200 execution -----------------------------------------------------
    ap.add_argument("--workers", type=int, default=0,
                    help="batcher worker processes. 0 = build inline (the "
                         "pre-B200 behaviour). ~48-64 saturates one training "
                         "process; budget across concurrent jobs so the total "
                         "stays under the core count")
    ap.add_argument("--prefetch_factor", type=int, default=2,
                    help="batches queued per worker. HOST RAM is the binding "
                         "constraint on this machine (cgroup memory.max = 64 "
                         "GiB, not the 2 TB the host reports), and in-flight "
                         "batches are workers x prefetch_factor x batch_bytes "
                         "-- on rel-amazon a 16x512 batch is 94 MB, so 48 "
                         "workers at prefetch 4 is 18 GB for ONE job. 2 is "
                         "enough to hide the handoff.")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp32"],
                    help="mixed precision for tokenizer+backbone; heads always "
                         "run in fp32 (see LEDGER.encode)")
    ap.add_argument("--attn", default="auto",
                    choices=["auto", "flex", "sdpa"],
                    help="flex = FlexAttention block-sparse packing mask; "
                         "sdpa = dense bool mask")
    ap.add_argument("--cutoff", default="val", choices=["val", "test"],
                    help="corpus boundary. `test` builds at test_timestamp so "
                         "test-split numbers can be scored; still leakage-free")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="steps between checkpoints (0 = only at the end)")
    ap.add_argument("--resume", default="",
                    help="checkpoint to resume from")
    ap.add_argument("--log_every", type=int, default=10)

    # -- in-training evaluation --------------------------------------------
    ap.add_argument("--retarget", default="",
                    help="`<intermediate>:<final>` -- retarget the WHO head "
                         "along a foreign-key path, e.g. `studies:sponsors`. "
                         "The EVENTS are untouched (WHEN/WHERE/WHAT still see "
                         "the real stream); only the retrieval target and its "
                         "features move into the final entity's space, where "
                         "the destination has a trainable embedding and where "
                         "repeat/recency actually exist. Use INSTEAD of "
                         "--eval_project_through, not with it.")
    ap.add_argument("--eval_project_through", default="",
                    help="intermediate entity table when the eval task's "
                         "destination is 2 hops away (rel-f1: races; "
                         "rel-trial: studies). Without it the in-training "
                         "eval ranks a table the WHO head never links to and "
                         "reports a meaningless MAP, so -best.pt would be "
                         "selected on noise.")
    ap.add_argument("--eval_task", default="",
                    help="RelBench link task to score during training, e.g. "
                         "user-item-purchase. Empty disables it.")
    ap.add_argument("--eval_entity", default="",
                    help="RelBench ENTITY task (classification or regression) "
                         "to score through the WindowHead during training, "
                         "e.g. user-badge. Selects `-bestwin.pt`, which is "
                         "NOT the same checkpoint as `-best.pt`: the two "
                         "categories peak at opposite ends of a run "
                         "(RESEARCH.md 9.6).")
    ap.add_argument("--eval_entity_rows", type=int, default=20000,
                    help="subsample for the in-training entity eval")
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--eval_split", default="val", choices=["val", "test"])
    ap.add_argument("--eval_rows", type=int, default=20000,
                    help="subsample the eval table to this many rows; MAP on a "
                         "random subsample is an unbiased estimate of MAP on "
                         "the whole and 20k rows is plenty to rank configs. "
                         "0 = score every row (use for final numbers)")
    ap.add_argument("--eval_max_len", type=int, default=0,
                    help="events per entity at eval. 0 = match training, "
                         "which is max_len//2, NOT max_len: a training row is "
                         "a PACKED budget and `batch()` gives any single "
                         "entity at most half of it. Getting this wrong "
                         "silently evaluates a long-context model at short "
                         "context and hides exactly the effect a context "
                         "sweep is trying to measure.")
    ap.add_argument("--eval_batch", type=int, default=512)

    ap.add_argument("--tag", default="", help="suffix for the run log name")
    ap.add_argument("--out", default="runs")
    return ap


def _lr_lambda(args):
    """Linear warm-up then cosine decay to `min_lr_frac` (ARCHITECTURE 9.2).

    v0 warmed up and then held peak lr forever. That is fine for a 6k-step
    probe and wrong for the 30k+ runs this hardware makes routine.
    """
    def f(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        if args.min_lr_frac >= 1.0:
            return 1.0
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        p = min(1.0, max(0.0, p))
        cos = 0.5 * (1.0 + np.cos(np.pi * p))
        return args.min_lr_frac + (1.0 - args.min_lr_frac) * cos
    return f


def _make_entity_eval(args, corpus, device):
    """In-training scoring of an ENTITY task through the WindowHead.

    Why this is not optional. RESEARCH.md 2026-08-25 (9.6) records that
    classification and retrieval peak at OPPOSITE ends of a run, and the
    window head reproduces it: on rel-stack user-badge the step-10,000
    checkpoint scores 63.5 and the step-20,000 one 57.8, while the window loss
    keeps falling throughout. A loss curve is therefore not a selector, and
    `-best.pt` selects on WHO MAP, so without this the run saves the worst
    classification checkpoint it produced. The selected checkpoint is written
    to `-bestwin.pt`, separately from `-best.pt`.

    -> callable(model) -> metric dict, or None.
    """
    if not args.eval_entity or args.eval_every <= 0:
        return None
    import numpy as _np
    from relbench.base import TaskType
    from relbench.tasks import get_task
    from . import win_readout as wr

    key = (args.dataset, args.eval_entity)
    q = wr.QUERIES.get(key)
    if q is None:
        print(f"no window query registered for {key}; entity eval disabled")
        return None
    # A query naming a column this corpus does not have would silently answer
    # the unfiltered question. Degrade LOUDLY instead -- a control arm built
    # without the denormalized columns is exactly the pre-denorm query, and
    # that is the comparison it exists to provide.
    q, dropped = wr.resolvable(q, corpus.schema)
    if q is None:
        print(f"entity eval disabled: {key} needs {dropped}; rebuild the "
              f"corpus with --denorm_fk / --child_aggs")
        return None
    if dropped:
        print(f"entity eval: {key} DROPPING unresolvable filter(s) {dropped} "
              f"-- this is the pre-denormalization query, not the task")
    task = get_task(args.dataset, args.eval_entity, download=True)
    tbl = task.get_table(args.eval_split, mask_input_cols=False)
    df = tbl.df
    n = args.eval_entity_rows
    if n and len(df) > n:
        idx = _np.random.default_rng(args.seed).choice(len(df), n,
                                                       replace=False)
        df = df.iloc[_np.sort(idx)].reset_index(drop=True)
        # `evaluate` compares against the table it is HANDED, so the
        # subsampled predictions must be scored against the subsampled table
        tbl = type(tbl)(df=df,
                        fkey_col_to_pkey_table=tbl.fkey_col_to_pkey_table,
                        pkey_col=tbl.pkey_col, time_col=tbl.time_col)
    pk = corpus.schema.pkey_index[task.entity_table]
    rows = df[task.entity_col].map(pk).fillna(-1).to_numpy(dtype=np.int64)
    cuts = (pd.to_datetime(df[task.time_col]).astype("int64")
            // 10 ** 9).to_numpy()
    keep = rows >= 0
    horizon = float(task.timedelta.total_seconds())
    is_cls = task.task_type == TaskType.BINARY_CLASSIFICATION
    std = float(task.get_table("train").df[task.target_col].std()) or 1.0
    eval_len = args.eval_max_len or max(1, args.max_len // 2)
    from .data.batching import PackedBatcher

    def run(model):
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                vals = []
                r, c = rows[keep], cuts[keep]
                for s in range(0, len(r), args.eval_batch):
                    batch, k2 = PackedBatcher.pack_histories(
                        corpus, task.entity_table, r[s:s + args.eval_batch],
                        c[s:s + args.eval_batch], max_len=eval_len,
                        device=device, query_token=True, horizon_s=horizon,
                        query_feats=args.query_feats,
                        query_dyn=args.query_dyn,
                        query_nbr=args.query_nbr)
                    # entity tasks read the WINDOW head, so they must read the
                    # branch WINDOW was trained on
                    h = model.encode(batch, branch="temporal")[
                        k2["b"], k2["last_l"]]
                    el = torch.as_tensor(k2["elapsed"], device=device,
                                         dtype=torch.float32)
                    z = model.win.state(
                        h, el, torch.full_like(el, horizon),
                        batch["query_pos"]["feats"]
                        if model.win.feat_path else None)
                    # the query-conditioned head answers the scalar
                    # directly; fall back to composing marginals when the
                    # checkpoint has no such head or the query is outside
                    # its vocabulary
                    pq = wr.predict_query(model, z, q)
                    vals.append((pq if pq is not None
                                 else wr.predict(model, z, q)).cpu().numpy())
            pred = _np.zeros(len(df))
            pred[keep] = _np.concatenate(vals) if vals else 0.0
            if (~keep).any() and keep.any():
                pred[~keep] = _np.median(pred[keep])
            m = task.evaluate(pred, tbl)
            out = {"entity_auroc": m["roc_auc"]} if is_cls else {
                "entity_nmae": m["mae"] / std}
            return out
        finally:
            model.train(was_training)

    return run


def _make_eval(args, corpus, device, retarget=None):
    """-> callable(model) -> metric dict, or None."""
    if not args.eval_task or args.eval_every <= 0:
        return None
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    from .queries import evaluate_recommendation

    task = get_task(args.dataset, args.eval_task, download=True)
    rng = np.random.default_rng(args.seed)
    eval_len = args.eval_max_len or max(1, args.max_len // 2)

    # Subsampling is done by monkey-patching nothing: evaluate_recommendation
    # reads task.get_table(split), so we wrap the task object in a shim that
    # returns a subsampled table. Keeps the eval path byte-identical to the
    # one scripts/eval_rec.py uses for headline numbers.
    class _Sub:
        def __init__(self, t):
            self._t = t
            self._cache = {}

        def __getattr__(self, k):
            return getattr(self._t, k)

        def get_table(self, split, **kw):
            # key on the kwargs too: a cache keyed on `split` alone would
            # hand back a table masked the wrong way if the eval path ever
            # asks for both variants, and the failure would be a silently
            # wrong metric rather than an error
            key = (split, tuple(sorted(kw.items())))
            if key not in self._cache:
                tbl = self._t.get_table(split, **kw)
                if args.eval_rows and len(tbl.df) > args.eval_rows:
                    idx = rng.choice(len(tbl.df), args.eval_rows,
                                     replace=False)
                    tbl = type(tbl)(
                        df=tbl.df.iloc[np.sort(idx)].reset_index(drop=True),
                        fkey_col_to_pkey_table=tbl.fkey_col_to_pkey_table,
                        pkey_col=tbl.pkey_col, time_col=tbl.time_col)
                self._cache[key] = tbl
            return self._cache[key]

        def evaluate(self, pred, target_table=None):
            return self._t.evaluate(
                pred, target_table or self.get_table(args.eval_split,
                                                     mask_input_cols=False))

    sub = _Sub(task)

    def run(model):
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                res = evaluate_recommendation(
                    model, corpus, sub, split=args.eval_split,
                    max_len=eval_len, device=device,
                    batch_size=args.eval_batch,
                    cooc_cache_key=args.dataset,
                    project_through=args.eval_project_through or None,
                    retarget=retarget,
                    db=(get_dataset(args.dataset, download=False).get_db
                        if args.eval_project_through else None))
            res.pop("_diagnostics", None)
            return res
        finally:
            model.train(was_training)

    return run


def main():
    args = build_argparser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # TF32 for the fp32 matmuls that remain (heads, WHO scoring). Free on
    # Blackwell and irrelevant to the bf16 backbone.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from relbench.datasets import get_dataset
    ds = get_dataset(args.dataset, download=True)
    cutoff = pd.Timestamp(getattr(ds, f"{args.cutoff}_timestamp"))
    corpus = load_corpus(args.dataset, cutoff, db=ds.get_db,
                         self_rows=args.self_rows,
                         denorm_fk=args.denorm_fk,
                         child_aggs=args.child_aggs,
                         path_expand=args.path_expand)
    n_derived = sum(1 for sp in corpus.schema.fact_tables.values()
                    for c in sp.columns if c.source is not None)
    if n_derived:
        print(f"denorm: {n_derived} derived columns across "
              f"{len(corpus.schema.fact_tables)} fact tables")

    if args.no_query_cat:
        # `query_cat_layout` memoises on the schema object, so seeding it with
        # an empty layout is the whole ablation -- and it travels to the
        # batcher workers with the pickled schema, which a module-level global
        # would not.
        corpus.schema._qcat_layout = []

    ents = [e.strip() for e in args.entity.split(",") if e.strip()]

    union_corpora = None
    if args.datasets:
        # Each database is loaded at ITS OWN cutoff -- RelBench cutoffs differ
        # by years across datasets, and one shared cutoff would either leak
        # future events into the early databases or throw away most of the
        # late ones.
        from .data.union import build_union, qualify
        specs = [x.strip() for x in args.datasets.split(",") if x.strip()]
        raw, want = {}, {}
        for sp in specs:
            dsn, _, es = sp.partition(":")
            d2 = get_dataset(dsn, download=True)
            raw[dsn] = load_corpus(
                dsn, pd.Timestamp(getattr(d2, f"{args.cutoff}_timestamp")),
                db=d2.get_db, self_rows=args.self_rows,
                denorm_fk=args.denorm_fk, child_aggs=args.child_aggs,
                path_expand=args.path_expand)
            want[dsn] = [e for e in es.split("+") if e]
        union_corpora, union_sch = build_union(raw)
        del raw
        # A batcher's corpus needs the UNION's fact tables -- the token ids it
        # emits are global -- but only its OWN entities: `PackedBatcher`
        # indexes `corpus.hist_offset[name]` for every entity table in the
        # schema, and a database has no histories for another database's
        # entities. Handing it the merged entity list raises KeyError on the
        # first foreign name, which is the good outcome; the bad one would
        # have been an empty history silently standing in for a real one.
        import copy as _copy
        from .data.schema import Schema as _Schema
        for c in union_corpora.values():
            own = c.schema
            c.schema = _Schema(fact_tables=union_sch.fact_tables,
                               entity_tables=own.entity_tables,
                               entity_counts=own.entity_counts,
                               pkey_values=own.pkey_values,
                               pkey_index=own.pkey_index)
        # The MODEL, by contrast, needs every entity: `states` allocates one
        # table per entity name and each database writes into its own.
        # `corpus` from here on is a UNION VIEW: the schema spans every
        # database, and every by-table-name dict is the merge of all of them,
        # because the names are namespaced and therefore disjoint. Anything
        # that reads the corpus BY TABLE NAME -- `bucket_edges` over
        # `feat_num`, the query-feature layout -- then sees the whole union.
        # The flat per-event arrays (`time`, `table_idx`) stay one database's
        # and are NOT meaningful on this view; only the batchers, which each
        # hold their own corpus, read those.
        corpus = _copy.copy(union_corpora[sorted(union_corpora)[0]])
        corpus.schema = union_sch
        for _f in ("feat_num", "feat_cat", "links", "row_of",
                   "hist_index", "hist_offset"):
            merged = {}
            for _c in union_corpora.values():
                merged.update(getattr(_c, _f))
            setattr(corpus, _f, merged)
        ents = [qualify(d, e) for d in sorted(want) for e in want[d]]
        print(f"union: {len(specs)} databases, "
              f"{union_sch.num_fact_tables} fact tables, "
              f"{sum(c.num_events for c in union_corpora.values()):,} events")
        for d in sorted(union_corpora):
            print(f"  {d:11s} {union_corpora[d].num_events:>12,} events  "
                  f"entities={want[d]}")

    retarget = None
    if args.retarget:
        from .queries import build_path_map
        inter, final = [x.strip() for x in args.retarget.split(":")]
        pm = build_path_map(ds.get_db(), corpus, inter, final, corpus.cutoff)
        iv, fv = pm["inter"], pm["final"]
        order = np.argsort(iv, kind="stable")
        iv, fv = iv[order], fv[order]
        n_inter = corpus.schema.entity_counts[inter]
        off = np.zeros(n_inter + 1, dtype=np.int64)
        np.cumsum(np.bincount(iv, minlength=n_inter), out=off[1:])
        retarget = {"inter": inter, "final": final, "off": off, "val": fv}
        print(f"retarget: {inter} -> {final} via `{pm['via']}`, "
              f"{len(fv):,} pre-cutoff pairs, "
              f"{np.mean(np.diff(off)[np.diff(off) > 0]):.2f} {final} per "
              f"{inter} (of those linked)")

    if args.query_nbr and not args.query_feats:
        raise SystemExit(
            "--query_nbr is a suffix block of the query-token feature "
            "vector, so it does nothing without --query_feats.")
    if args.query_nbr and args.dense_window:
        # `_dense_stats` reproduces `query_feats` with one vectorised sweep
        # and emits only the base + per-category width; the neighbour block
        # is O(history-lookups) per query and cannot join that sweep without
        # reintroducing the quadratic the dense path exists to avoid. The
        # widths would then disagree with `query_feat_proj`. Refuse LOUDLY
        # rather than shape-error deep in the first forward pass.
        raise SystemExit(
            "--query_nbr and --dense_window are incompatible: the dense "
            "sweep cannot produce the neighbour block. Drop one.")
    if args.objective == "window":
        # The window head IS the objective here; without it there is no loss
        # at all. Turn it on rather than failing 200 lines later.
        args.window_head = True
        # Query tokens are now the ONLY supervised positions in the row, so
        # the auxiliary-era default of 4 per sequence makes most of the
        # forward pass produce no gradient. Raise it unless asked otherwise.
        if not any(a.startswith("--n_query") for a in sys.argv):
            args.n_query = 16
        if args.eval_task:
            # -best.pt would be selected on a WHO head that receives no
            # gradient in this mode: a ranking of noise, saved as "best".
            raise SystemExit(
                "--objective window trains no WHO head, so --eval_task "
                "would select -best.pt on noise. Use --eval_entity "
                "<classification-or-regression task>, which selects "
                "-bestwin.pt through the window head.")
        if not args.eval_entity:
            print("WARNING: --objective window without --eval_entity: no "
                  "checkpoint selection, and 9.6 says the window metric "
                  "peaks well before the loss stops falling.")
        print(f"objective=window: next-event heads OFF, "
              f"n_query={args.n_query} supervised positions per sequence")
    elif args.objective == "next_event":
        args.w_win = 0.0

    def _mk_batcher(_corpus, _ents, _cooc_key):
        return PackedBatcher(_corpus, _ents, max_len=args.max_len,
                                batch_rows=args.batch_rows, seed=args.seed,
                                min_hist=args.min_hist,
                                n_neg=args.n_neg,
                                use_cooc=(args.who_feats and not args.no_cooc),
                                cooc_cache_key=_cooc_key,
                                window=args.window, logq=args.logq,
                                hard_negs=args.hard_negs,
                                window_days=args.window_days,
                                window_sigma=args.window_sigma,
                                retarget=retarget,
                                window_head=args.window_head,
                                n_query=args.n_query,
                                win_max_events=args.win_max_events,
                                q_tail=args.q_tail,
                                dense_window=args.dense_window,
                                n_dense=args.n_dense,
                                next_event=(args.objective != "window"),
                                query_dyn=args.query_dyn,
                                query_nbr=args.query_nbr)

    if union_corpora is not None:
        # One batcher per database over ITS OWN entity tables. The cooc
        # cache key must stay per-database: sharing it would hand two
        # databases one co-occurrence matrix built from whichever ran
        # first, silently, with no shape error to catch it.
        from .data.union import MultiBatcher
        per = {}
        for _d, _c in union_corpora.items():
            _e = [x for x in ents if x.startswith(_d + '::')]
            if _e:
                per[_d] = _mk_batcher(_c, _e, _d)
        batcher = MultiBatcher(per, seed=args.seed)
        print(f'batcher: round-robin over {len(per)} databases '
              f"({', '.join(sorted(per))})")
    else:
        batcher = _mk_batcher(corpus, ents, args.dataset)
    n_qfeat = PackedBatcher.n_query_feats(corpus.schema, args.query_dyn,
                                          args.query_nbr)
    win_edges = None
    if args.window_head:
        from .model.window import bucket_edges
        win_edges = bucket_edges(corpus, args.win_buckets)
    model = LEDGER(corpus.schema, dim=args.dim, layers=args.layers,
                 heads=args.heads,
                 states=args.states, state_momentum=args.state_momentum,
                 state_normalize=args.state_normalize,
                 state_center=args.state_center,
                 state_reduce=args.state_reduce,
                 learned_max=args.learned_max,
                 learned_tables=([x.strip() for x in
                                  args.learned_tables.split(",") if x.strip()]
                                 or None),
                 uniformity=args.uniformity,
                 loss_weights={"when": args.w_when, "where": args.w_where,
                               "who": args.w_who, "what": args.w_what,
                               "rerank": args.w_rerank, "win": args.w_win,
                               "query": args.w_query,
                               "windense": args.w_windense},
                 next_event=(args.objective != "window"),
                 window_head=args.window_head, win_edges=win_edges,
                 win_buckets=args.win_buckets,
                 win_balance=args.win_balance, w_rate=args.w_rate,
                 w_qcount=args.w_qcount, w_detail=args.w_detail,
                 feat_path=args.feat_path,
                 query_feats=(n_qfeat if args.query_feats else 0),
                 who_feats=(PackedBatcher.N_WHO_FEATS
                            if args.who_feats else 0),
                 window=args.window, rerank=args.rerank,
                 no_ema_write=args.no_ema_write,
                 amp_dtype=(torch.bfloat16 if args.amp == "bf16" else None),
                 attn=args.attn,
                 branch_layers=args.branch_layers,
                 branch_kind=args.branch_kind,
                 learn_loss_weights=args.learn_loss_weights,
                 lw_clamp=args.lw_clamp,
                 query_head=args.query_head,
                 rate_balance=args.rate_balance,
                 rate_zero_scale=args.rate_zero_scale,
                 ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_ent = sum(corpus.schema.entity_counts.values())
    state_gb = n_ent * model.entity_states.bytes_per_entity() / 1e9
    print(f"model: {n_params/1e6:.1f}M parameters on {device}, "
          f"amp={args.amp}, attn={args.attn}")
    print(f"states: {args.states}, {n_ent:,} entities, "
          f"{state_gb:.3f} GB for the state table")
    print(f"batch: {args.batch_rows} x {args.max_len} = "
          f"{args.batch_rows * args.max_len:,} tokens/step, "
          f"{args.workers} batcher workers")

    # The learned log-variances are not weights and must NOT be decayed:
    # weight decay pulls log sigma^2 toward 0, i.e. pulls every learned loss
    # weight back toward 1, which is precisely the balance the parameter
    # exists to move away from. Same reasoning excludes them from the
    # `--w_*` semantics -- see LEDGER.log_var.
    lw_params = [p for n, p in model.named_parameters()
                 if n.startswith("log_var.")]
    lw_ids = {id(p) for p in lw_params}
    groups = [{"params": [p for p in model.parameters()
                          if id(p) not in lw_ids], "weight_decay": 0.1}]
    if lw_params:
        groups.append({"params": lw_params, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.1,
                            fused=(device == "cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(args))

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            start_step = ck["step"] + 1
        print(f"resumed from {args.resume} at step {start_step}")

    run_dir = pathlib.Path(args.out); run_dir.mkdir(exist_ok=True, parents=True)
    stem = f"{args.dataset}-{'+'.join(ents)}-{args.states}-{args.seed}"
    if args.tag:
        stem += f"-{args.tag}"
    ckpt_path = run_dir / f"{stem}.pt"

    best = {"map": -1.0}

    def save(step, path=None, extra=None):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "args": vars(args), **(extra or {})},
                   path or ckpt_path)

    def save_periodic(step):
        """Step-TAGGED, not overwritten.

        The first rel-amazon sweep wrote every periodic checkpoint to one
        path, so when MAP turned out to fall monotonically across the run
        there was no way to compare the step-5k and step-15k weights and see
        what had moved. Diagnosing that cost a rerun. Keeping the tag is a few
        GB and buys the whole trajectory.
        """
        save(step, ckpt_path.with_name(f"{ckpt_path.stem}-s{step}.pt"))

    evaluate = _make_eval(args, corpus, device, retarget)
    evaluate_entity = _make_entity_eval(args, corpus, device)
    best_win = {"score": None}
    metric_keys: list = []
    loader = PrefetchLoader(batcher, workers=args.workers, seed=args.seed,
                            device=device,
                            prefetch_factor=args.prefetch_factor)

    # RANDOM-QUERY SAMPLER. The excluded signature is the EVAL task's own
    # query: training on it would be teaching to the test and would forfeit
    # the zero-shot claim, so the evaluated question is held out and only
    # others of its kind are learned.
    qsampler = None
    if args.query_head:
        from . import qsample as _qs
        from . import win_readout as _wr
        _excl = set()
        if args.eval_entity:
            _q = _wr.QUERIES.get((args.dataset, args.eval_entity))
            if _q is not None:
                for _sig in _qs.signatures_of(_q, corpus.schema):
                    _excl.add(_sig)
        print(f"query head: sampling {args.n_rand_q}/step, "
              f"excluding {len(_excl)} signature(s) for "
              f"{args.eval_entity or '-'}", flush=True)
        _rng = np.random.default_rng(args.seed + 7717)

        def qsampler():
            return _qs.sample(corpus.schema, _rng, args.n_rand_q,
                              exclude=_excl)

    log_path = run_dir / f"{stem}.csv"
    # Append on resume so a preempted run keeps one continuous curve.
    mode = "a" if (args.resume and log_path.exists()) else "w"
    with open(log_path, mode, newline="") as f:
        log = csv.writer(f)
        if mode == "w":
            log.writerow(["step", "total", "when", "where", "who", "what",
                          "rerank", "win", "windense", "query", "sec",
                          "tok_per_s", "lr",
                          "metrics"])
        t0 = wallclock.time()
        last_t, last_step = t0, start_step
        try:
            for step in range(start_step, args.steps):
                batch = next(loader)
                if qsampler is not None:
                    batch["qspecs"] = qsampler()
                losses = model.loss(batch)
                opt.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()

                do_eval = ((evaluate is not None or evaluate_entity is not None)
                           and args.eval_every > 0
                           and (step % args.eval_every == 0
                                or step == args.steps - 1)
                           and step > start_step)
                if step % args.log_every == 0 or step == args.steps - 1 \
                        or do_eval:
                    now = wallclock.time()
                    dt = max(now - last_t, 1e-9)
                    tps = ((step - last_step) * args.batch_rows
                           * args.max_len / dt)
                    last_t, last_step = now, step
                    mtxt = ""
                    if do_eval:
                        m = evaluate(model) if evaluate is not None else {}
                        if evaluate_entity is not None:
                            me = evaluate_entity(model)
                            m.update(me)
                            # AUROC up, NMAE down -- one comparison either way
                            k2 = next(iter(me))
                            v = me[k2] if k2.endswith("auroc") else -me[k2]
                            if best_win["score"] is None or v > best_win["score"]:
                                best_win.update(score=v, step=step)
                                save(step, ckpt_path.with_name(
                                    f"{ckpt_path.stem}-bestwin.pt"),
                                    extra={"metric": m})
                        metric_keys = list(m)
                        # AUROC/MAP are reported as percentages; NMAE is a
                        # ratio and must NOT be scaled. Scaling it made
                        # post-votes read 11.48 when it was 0.1148, which was
                        # then mistaken for a modelling catastrophe and blamed
                        # on the query-time distribution.
                        mtxt = json.dumps(
                            {k: round(v if "nmae" in k else 100 * v, 4)
                             for k, v in m.items()})
                        # Select on the METRIC, not on step count. On
                        # rel-amazon MAP peaks early and then falls for the
                        # rest of the run (RESEARCH.md 2026-08-24), so taking
                        # the final checkpoint ships the worst model the run
                        # produced. Cheap insurance on every dataset.
                        sel = m.get("link_prediction_map")
                        if sel is not None and sel > best["map"]:
                            best.update(map=sel, step=step)
                            save(step, ckpt_path.with_name(
                                f"{ckpt_path.stem}-best.pt"),
                                extra={"metric": m})
                    row = [step] + [
                        f"{losses.get(k, torch.tensor(float('nan'))).item():.4f}"
                        for k in ("total", "when", "where", "who",
                                  "what", "rerank", "win", "windense",
                                  "query")
                    ] + [f"{now - t0:.1f}", f"{tps:.0f}",
                         f"{sched.get_last_lr()[0]:.2e}", mtxt]
                    log.writerow(row); f.flush()
                    wtxt = ""
                    if model.log_var is not None:
                        wtxt = " | w " + " ".join(
                            f"{k}={losses[f'w_{k}'].item():.3f}"
                            for k in ("when", "who", "win")
                            if f"w_{k}" in losses)
                    print(f"step {step:>7} total {row[1]} when {row[2]} "
                          f"who {row[4]} win {row[7]} "
                          f"wind {row[8]} q {row[9]} | {tps:>9,.0f} tok/s"
                          + wtxt + (f" | {mtxt}" if mtxt else ""), flush=True)
                if args.ckpt_every and step and step % args.ckpt_every == 0:
                    save_periodic(step)
        finally:
            loader.close()

    if hasattr(model.entity_states, "coverage"):
        cov = model.entity_states.coverage()
        print("state coverage:",
              {k: f"{100*v:.1f}%" for k, v in cov.items()})

    save(args.steps - 1)
    if best_win["score"] is not None:
        print(f"best {args.eval_entity} at step {best_win['step']} "
              f"-> {ckpt_path.with_name(ckpt_path.stem + '-bestwin.pt')}")
    if best["map"] >= 0:
        print(f"best {args.eval_task} MAP {100*best['map']:.4f} "
              f"at step {best['step']} "
              f"-> {ckpt_path.with_name(ckpt_path.stem + '-best.pt')}")
    print("saved", log_path, "and", ckpt_path)


if __name__ == "__main__":
    main()
