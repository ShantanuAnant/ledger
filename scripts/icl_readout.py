"""In-context readout: FIT the composition instead of deriving it.

    PYTHONPATH=. .venv/bin/python scripts/icl_readout.py \
        --ckpt runs/<...>-bestwin.pt --dataset rel-event --task user-ignore \
        --split val --support 1024

WHAT THIS CHANGES, AND WHY IT IS THE NATURAL NEXT STEP
------------------------------------------------------
The WindowHead emits, per (entity, query time), a compact vector of
SUFFICIENT STATISTICS: per-table log rates, per-category rates, per-numeric-
bucket rates, and quantiles of counts and of column sums. `win_readout` then
composes them into an answer with a FIXED ANALYTIC FORMULA that has zero free
parameters -- `P(N=0) = exp(-rate)`, `total * P(cat) * P(num)`, and so on.

That formula is a hypothesis about how the statistics combine, and 2026-09-01
measured two places where it is simply wrong:

  * rel-f1 driver-dnf ranks BACKWARDS (0.36-0.51 AUROC). So does the trivial
    `count` baseline (0.417). Anything below 0.5 is a sign error the
    composition cannot see and a fitted combination learns from ~50 labels.
  * rel-trial study-outcome asks for the rate of `outcome_analyses`, which
    appears in 0 of 960 val histories. The analytic path must use that rate;
    a fitted one can put ~zero weight on it and pick up the attribute
    features (phase, enrollment, design) already in the query token.

So: keep the statistics self-supervised, and FIT the composition on a small
labelled support set drawn from the TRAIN split. The head is untouched and no
GPU training happens here.

WHY THIS IS SMALL FOR US AND LARGE FOR EVERYONE ELSE. TabPFN / TabICL /
RDB-PFN attend over raw table rows because their backbones produce nothing
like a sufficient-statistic vector. Ours does, so the in-context learner is a
logistic regression over a few hundred numbers rather than a transformer over
a database.

CALIBRATION WOULD BUY NOTHING. AUROC is invariant to any monotone transform
of the score, so Platt/isotonic on the support set scores +0.00. The gain has
to come from RE-RANKING, i.e. from changing how the statistics are combined.
That is why this fits a full linear model over the vector and not a squashing
function over the composed scalar.

LEAKAGE. Support rows come from the TRAIN split and each row's features are
built from `history(before=its own timestamp)`, the same per-row rule the
zero-label path uses. No support row may be at or after an eval row's cutoff
by construction: the train table ends before the val cutoff.

HONEST REPORTING. This is a DIFFERENT SETTING from the zero-label readout and
must be reported as its own row (`LEDGER+ICL`), never merged into the zero-shot
one. `--support 1024` is the default because that is the budget the
RDB-PFN / TabPFN-2.5 / TabICL leaderboard rows use; KumoRFM-2 uses up to
10,000.
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

from ledger import win_readout as wr

from ledger.data.cache import load_corpus
from ledger.data.corpus import EventCorpus
from ledger.data.schema import build_schema

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_window import load_model, query_states


def build_corpus(ck, ds, task, dataset, split, corpus_at):
    """The eval corpus, built the way the checkpoint's was.

    Mirrors `eval_window.main`'s construction, and for the same reasons:

    * corpus flags come from `ck["args"]`, or the entity histories differ
      between training and inference;
    * when the cutoff moves past the training one, every COLUMN SPEC is
      pinned to the training cutoff. Categorical vocabularies are index-
      shifted across cutoffs (HANDOFF 10.1a) -- a value first seen later does
      not extend the vocabulary, it shifts the index of everything sorting
      after it -- and the standardization constants set the bucket edges the
      numeric predicates are read on.

    The cutoff must cover the SUPPORT rows as well as the eval rows. Support
    rows come from the train table, which ends before the val cutoff, so the
    eval-driven cutoff already contains them; each row is still truncated to
    `history(before=its own timestamp)`.
    """
    tbl = task.get_table(split, mask_input_cols=False)
    cutoff = pd.Timestamp(ds.test_timestamp if split == "test"
                          else ds.val_timestamp)
    if corpus_at == "rowmax":
        cutoff = max(cutoff, pd.Timestamp(tbl.df[task.time_col].max()))
    train_cutoff = pd.Timestamp(ds.val_timestamp)
    ta = ck["args"]
    kw = dict(self_rows=bool(ta.get("self_rows")),
              denorm_fk=bool(ta.get("denorm_fk")),
              child_aggs=bool(ta.get("child_aggs")),
              path_expand=str(ta.get("path_expand") or ""))
    if cutoff == train_cutoff:
        corpus = load_corpus(dataset, cutoff, db=ds.get_db, **kw)
    else:
        train_schema = load_corpus(dataset, train_cutoff, db=ds.get_db,
                                   verbose=False, **kw).schema
        sch = build_schema(ds.get_db(), cutoff,
                           denorm_fk=kw["denorm_fk"],
                           child_aggs=kw["child_aggs"])
        for nm, sp in sch.fact_tables.items():
            if nm in train_schema.fact_tables:
                sp.columns = train_schema.fact_tables[nm].columns
        print(f"building corpus at {cutoff} with training-cutoff column specs")
        corpus = EventCorpus.build(ds.get_db(), cutoff, schema=sch, **kw)
    if ta.get("no_query_cat"):
        corpus.schema._qcat_layout = []      # match the training layout
    return corpus


def feature_names_and_values(model, z, horizon_s):
    """Every sufficient statistic the head emits, as a flat matrix.

    -> (names [F], X [B, F])

    This is deliberately the WHOLE vector rather than the subset a particular
    task's Q happens to name: which statistics matter is exactly the question
    the support set is being asked, and a task-specific selection would put
    the analytic hypothesis back in through the side door.
    """
    win = model.win
    names, cols = [], []

    def add(name, v):
        v = v.detach().double().cpu().numpy()
        if v.ndim == 1:
            names.append(name)
            cols.append(v[:, None])
        else:
            names.extend(f"{name}[{i}]" for i in range(v.shape[1]))
            cols.append(v)

    for tname, spec in win.schema.fact_tables.items():
        add(f"lograte.{tname}", win.log_rate(z, tname))
        qc = win.q_count(z, tname)
        if qc is not None:
            add(f"qcount.{tname}", qc)
        cats = [c for c in spec.columns if c.kind == "categorical"]
        for j, c in enumerate(cats):
            r = win.log_rate_cat(z, tname, j)
            if r is not None:
                add(f"cat.{tname}.{c.name}", r)
        rn = win.log_rate_num(z, tname)
        if rn is not None:
            add(f"num.{tname}", rn.reshape(rn.shape[0], -1))
        qs = win.q_sum(z, tname)
        if qs is not None:
            add(f"qsum.{tname}", qs.reshape(qs.shape[0], -1))

    # The raw query-token features are appended by the caller, which can get
    # them from the batcher whether or not the head has a feature path.
    return names, np.concatenate(cols, axis=1)


def rows_for(task, corpus, split, rng, cap=0):
    tbl = task.get_table(split, mask_input_cols=False)
    df = tbl.df
    if cap and len(df) > cap:
        idx = rng.choice(len(df), cap, replace=False)
        df = df.iloc[np.sort(idx)].reset_index(drop=True)
    pk = corpus.schema.pkey_index[task.entity_table]
    rows = df[task.entity_col].map(pk).fillna(-1).to_numpy(dtype=np.int64)
    cuts = (pd.to_datetime(df[task.time_col]).astype("int64")
            // 10 ** 9).to_numpy()
    y = df[df.columns[-1]].to_numpy(dtype=np.float64)
    return df, rows, cuts, y


def features_for(model, corpus, task, rows, cuts, horizon_s, max_len, dev,
                 targs, batch_size):
    known = rows >= 0
    z, _, qf = query_states(
        model, corpus, task.entity_table, rows[known], cuts[known],
        horizon_s, max_len, dev, batch_size,
        query_feats=bool(targs.get("query_feats")),
        query_dyn=bool(targs.get("query_dyn")),
        query_nbr=bool(targs.get("query_nbr")), return_feats=True)
    names, Xk = feature_names_and_values(model, z, horizon_s)
    if qf is not None:
        # The head only consumes these when it has a wide-and-deep path, so
        # for every checkpoint we have they are UNUSED signal sitting next to
        # the state. They are also the features a frozen probe found beat the
        # backbone (81.5 / 84.8 vs 74.2 on user-badge, RESEARCH 2026-08-25),
        # and the `count` baseline that outranks the analytic readout on
        # rel-f1 driver-top3 is a linear function of them.
        names = names + [f"qfeat[{i}]" for i in range(qf.shape[1])]
        Xk = np.concatenate([Xk, qf.astype(np.float64)], axis=1)
    X = np.zeros((len(rows), Xk.shape[1]), dtype=np.float64)
    X[known] = Xk
    # `z` is returned so the zero-label comparison can reuse it. Recomputing
    # it was a THIRD full forward pass over the eval rows for a number we
    # already had the state for -- a third of the GPU time, wasted.
    return names, X, known, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--support", type=int, default=1024,
                    help="labelled TRAIN rows fitted at inference; 1024 is "
                         "the RDB-PFN / TabPFN-2.5 / TabICL budget")
    ap.add_argument("--corpus-at", default="rowmax",
                    choices=["cutoff", "rowmax"])
    ap.add_argument("--rebuild-at-test", action="store_true")
    ap.add_argument("--max_len", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--C", type=float, default=0.0,
                    help="inverse L2 strength. 0 (default) SELECTS it by "
                         "cross-validation INSIDE the support set. Never tune "
                         "it on the eval split: with ~1500 features over 1024 "
                         "rows the choice is worth ~2 AUROC (user-ignore: "
                         "82.30 at C=1.0, 84.16 at C=0.003), so hand-picking "
                         "it against the reported number is test-tuning.")
    ap.add_argument("--readout", default="linear",
                    choices=["linear", "auto"],
                    help="`linear` is the published probe: one regularised "
                         "linear model over the statistics. `auto` adds "
                         "NONLINEAR candidates (gradient boosting, extremely "
                         "randomised trees) and picks among ALL of them, "
                         "linear included, by cross-validation INSIDE the "
                         "support set. Because the linear family stays in the "
                         "pool, `auto` can only match or beat `linear` up to "
                         "CV noise -- it is a strictly wider search, not a "
                         "different method. The label budget is unchanged: "
                         "the same 1,024 train rows fit the same features, "
                         "and the eval split is never consulted.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pred-out", default="",
                    help="directory to save the PROBE and ZERO-LABEL "
                         "prediction vectors as .npy, aligned to the eval "
                         "split's own row order. Only meaningful with the "
                         "default --rows 0, which keeps every row: a capped "
                         "run subsamples and the vector would not line up "
                         "with the official test table. Consumed by "
                         "scripts/make_submission.py.")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from sklearn.linear_model import (LogisticRegression,
                                      LogisticRegressionCV, SGDRegressor)
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.ensemble import (ExtraTreesClassifier,
                                  HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = get_dataset(args.dataset, download=True)
    task = get_task(args.dataset, args.task, download=True)
    is_cls = task.task_type == TaskType.BINARY_CLASSIFICATION
    rng = np.random.default_rng(args.seed)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False,
                    mmap=True)
    corpus = build_corpus(ck, ds, task, args.dataset, args.split,
                          args.corpus_at)
    model, targs = load_model(args.ckpt, corpus, dev)
    max_len = args.max_len or max(1, int(targs.get("max_len", 512)) // 2)
    horizon_s = float(task.timedelta.total_seconds())

    _, tr_rows, tr_cuts, tr_y = rows_for(task, corpus, "train", rng,
                                         args.support)
    # NMAE NORMALISER. The published board defines "NMAE = MAE / train-split
    # std" over the WHOLE train split. `tr_y` is only the sampled support
    # (default 1,024 rows), and on the heavy-tailed targets a sample that size
    # badly underestimates the population std -- rel-amazon item-ltv is 265.6
    # against the true 590.9. Dividing by the support's std therefore INFLATED
    # our own NMAE by up to 2.2x and made us look worse than we are, while
    # also making numbers incomparable across `--support` values. Take the
    # normaliser from the full train split, once.
    nmae_std = float(pd.to_numeric(
        task.get_table("train", mask_input_cols=False).df[task.target_col],
        errors="coerce").to_numpy(dtype=float).std()) or 1.0
    print(f"NMAE normaliser: train-split std = {nmae_std:.4f} "
          f"(support std {float(tr_y.std()):.4f})")
    ev_df, ev_rows, ev_cuts, ev_y = rows_for(task, corpus, args.split, rng,
                                            args.rows)
    print(f"task {args.dataset}/{args.task}  support={len(tr_rows):,} "
          f"(train)  eval={len(ev_rows):,} ({args.split})  "
          f"{'classification' if is_cls else 'regression'}")

    names, Xtr, ktr, _ = features_for(model, corpus, task, tr_rows, tr_cuts,
                                      horizon_s, max_len, dev, targs,
                                      args.batch_size)
    _, Xev, kev, z_ev = features_for(model, corpus, task, ev_rows, ev_cuts,
                                     horizon_s, max_len, dev, targs,
                                     args.batch_size)
    print(f"features: {Xtr.shape[1]} sufficient statistics per entity")

    Xtr, Xev = np.nan_to_num(Xtr), np.nan_to_num(Xev)
    sc = StandardScaler().fit(Xtr[ktr])
    A, B = sc.transform(Xtr), sc.transform(Xev)

    out = {}
    GRID = np.logspace(-4, 1, 12)

    def pick_by_cv(cands, y, scoring, cv):
        """Select among candidates by CV INSIDE the support set.

        The eval split is never touched here -- that is the whole discipline
        of this script, and it is why the number it prints is reportable. The
        linear family is always in `cands`, so a wider pool cannot lose to the
        narrower one by more than CV noise.

        Ties go to the EARLIER candidate, and the list is ordered
        simplest-first, so a nonlinear model has to actually win to be chosen
        rather than merely draw.
        """
        best, best_s, table = None, -np.inf, []
        for nm, est in cands:
            try:
                sc_ = float(np.mean(cross_val_score(
                    est, A[ktr], y, cv=cv, scoring=scoring, n_jobs=-1)))
            except Exception as e:                      # one bad candidate
                table.append((nm, float("nan")))        # must not kill the run
                print(f"    candidate {nm} failed: {type(e).__name__}")
                continue
            table.append((nm, sc_))
            if sc_ > best_s + 1e-12:
                best, best_s = (nm, est), sc_
        for nm, sc_ in table:
            mark = " <-- chosen" if best and nm == best[0] else ""
            print(f"    cv {sc_:8.4f}  {nm}{mark}")
        return best, best_s

    if is_cls and args.readout == "auto":
        # Simplest first: linear at the CV-chosen C, then two nonlinear
        # families. Depths are small because the support set is 1,024 rows
        # over ~1,500 features -- an unconstrained booster memorises it.
        cv = StratifiedKFold(5, shuffle=True, random_state=args.seed)
        cands = [
            ("linear (LogisticRegressionCV)",
             LogisticRegressionCV(Cs=GRID, cv=5, scoring="roc_auc",
                                  max_iter=2000, n_jobs=-1)),
            ("hist-gbdt d3 lr0.06",
             HistGradientBoostingClassifier(
                 max_depth=3, learning_rate=0.06, max_iter=400,
                 l2_regularization=1.0, early_stopping=True,
                 validation_fraction=0.2, random_state=args.seed)),
            ("hist-gbdt d6 lr0.06",
             HistGradientBoostingClassifier(
                 max_depth=6, learning_rate=0.06, max_iter=400,
                 l2_regularization=1.0, early_stopping=True,
                 validation_fraction=0.2, random_state=args.seed)),
            ("extra-trees 500",
             ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2,
                                  n_jobs=-1, random_state=args.seed)),
        ]
        print("  readout=auto, selecting on 5-fold CV inside the support:")
        best, best_cv = pick_by_cv(cands, tr_y[ktr], "roc_auc", cv)
        if best is None:
            raise SystemExit("every candidate failed")
        name, m = best
        m.fit(A[ktr], tr_y[ktr])
        out["readout"] = name
        out["readout_cv"] = best_cv
        out["C"] = float(np.ravel(getattr(m, "C_", [0.0]))[0])
        p = (m.predict_proba(B)[:, 1] if hasattr(m, "predict_proba")
             else m.decision_function(B))
        out["roc_auc"] = float(roc_auc_score(ev_y, p))
        out["average_precision"] = float(average_precision_score(ev_y, p))
        print(f"\n  LEDGER+ICL ({name})  roc_auc={out['roc_auc']:.4f}"
              f"  average_precision={out['average_precision']:.4f}"
              f"  [support CV {best_cv:.4f}]")
    elif is_cls:
        if args.C > 0:
            m = LogisticRegression(C=args.C, max_iter=2000)
        else:
            # 5-fold over the SUPPORT set only. `scoring="roc_auc"` matches
            # what is reported, and cv is stratified for LogisticRegressionCV.
            m = LogisticRegressionCV(Cs=GRID, cv=5, scoring="roc_auc",
                                     max_iter=2000, n_jobs=-1)
        m.fit(A[ktr], tr_y[ktr])
        out["C"] = float(np.ravel(getattr(m, "C_", [args.C]))[0])
        p = m.predict_proba(B)[:, 1]
        out["roc_auc"] = float(roc_auc_score(ev_y, p))
        out["average_precision"] = float(average_precision_score(ev_y, p))
        print(f"\n  LEDGER+ICL (fitted composition)  roc_auc={out['roc_auc']:.4f}"
              f"  average_precision={out['average_precision']:.4f}"
              f"  [C={out['C']:.4g}, chosen by 5-fold CV inside the support]")
    elif args.readout == "auto":
        # Same median-fitting discipline as the linear branch below: the
        # metric is NMAE, whose optimal predictor is the conditional MEDIAN,
        # so every candidate here optimises L1 and the target is
        # median-centred / robustly scaled before fitting.
        #
        # The CONSTANT MEDIAN is an explicit candidate rather than a
        # post-hoc guard. The linear branch bolts on a "did CV beat the
        # constant?" check after selection; making it a candidate is the same
        # protection expressed once, in the place that already compares
        # models, and it means a task where nothing beats the constant
        # reports the constant by SELECTION rather than by rescue.
        from sklearn.dummy import DummyRegressor
        c0 = float(np.median(tr_y[ktr]))
        s0 = float(np.median(np.abs(tr_y[ktr] - c0))) * 1.4826
        if not np.isfinite(s0) or s0 <= 0:
            s0 = float(tr_y[ktr].std()) or 1.0
        ys = (tr_y[ktr] - c0) / s0
        lin = GridSearchCV(
            SGDRegressor(loss="epsilon_insensitive", epsilon=0.0,
                         penalty="l2", max_iter=20000, tol=1e-5,
                         average=True, random_state=args.seed),
            {"alpha": 1.0 / GRID}, cv=5,
            scoring="neg_mean_absolute_error", n_jobs=-1)
        cands = [
            ("constant median", DummyRegressor(strategy="median")),
            ("linear L1 (SGD, alpha by CV)", lin),
            ("hist-gbdt L1 d3 lr0.06",
             HistGradientBoostingRegressor(
                 loss="absolute_error", max_depth=3, learning_rate=0.06,
                 max_iter=400, l2_regularization=1.0, early_stopping=True,
                 validation_fraction=0.2, random_state=args.seed)),
            ("hist-gbdt L1 d6 lr0.06",
             HistGradientBoostingRegressor(
                 loss="absolute_error", max_depth=6, learning_rate=0.06,
                 max_iter=400, l2_regularization=1.0, early_stopping=True,
                 validation_fraction=0.2, random_state=args.seed)),
        ]
        print("  readout=auto, selecting on 5-fold CV inside the support:")
        best, best_cv = pick_by_cv(cands, ys, "neg_mean_absolute_error", 5)
        if best is None:
            raise SystemExit("every candidate failed")
        name, m = best
        m.fit(A[ktr], ys)
        out["readout"] = name
        out["readout_cv"] = best_cv
        out["fell_back_to_median"] = bool(name == "constant median")
        p = m.predict(B) * s0 + c0
        mae = float(np.abs(p - ev_y).mean())
        std = nmae_std
        out["mae"], out["nmae"] = mae, mae / std
        print(f"\n  LEDGER+ICL ({name})  mae={mae:.4f}  nmae={out['nmae']:.4f}"
              f"  [support CV {best_cv:.4f}]")
    else:
        # FIT THE MEDIAN, NOT THE MEAN. The metric is NMAE, whose optimal
        # predictor is the conditional MEDIAN; least squares estimates the
        # conditional MEAN, and on these skewed targets the two diverge
        # badly. Measured 2026-09-01 with RidgeCV: ad-ctr 0.4196 -> 4.2670
        # (10x worse), item-ltv 0.1911 -> 0.7254. The zero-label path never
        # had this bug -- `win_readout` reads a PINBALL-trained quantile
        # head, a median estimator by construction.
        #
        # THE TARGET MUST BE SCALED. SGD's step size is not scale-free, so a
        # target living on [0, 1] (ad-ctr, a click RATIO) and one living on
        # [0, 10^4] (item-ltv) cannot share a learning rate: unscaled, ad-ctr
        # still came out at NMAE 2.14 against a 0.42 baseline even AFTER the
        # loss was fixed. The median is equivariant under an affine map, so
        # centring on the median and dividing by a robust scale is exact
        # rather than an approximation, and it is inverted before scoring.
        c0 = float(np.median(tr_y[ktr]))
        s0 = float(np.median(np.abs(tr_y[ktr] - c0))) * 1.4826
        if not np.isfinite(s0) or s0 <= 0:
            s0 = float(tr_y[ktr].std()) or 1.0
        ys = (tr_y[ktr] - c0) / s0
        base = SGDRegressor(loss="epsilon_insensitive", epsilon=0.0,
                            penalty="l2", max_iter=20000, tol=1e-5,
                            average=True, random_state=args.seed)
        if args.C > 0:
            m = base.set_params(alpha=1.0 / args.C)
            m.fit(A[ktr], ys)
            cv_best = None
        else:
            gs = GridSearchCV(base, {"alpha": 1.0 / GRID}, cv=5,
                              scoring="neg_mean_absolute_error", n_jobs=-1)
            gs.fit(A[ktr], ys)
            m, cv_best = gs.best_estimator_, -gs.best_score_
        out["alpha"] = float(m.get_params()["alpha"])
        # GUARD. If the fit cannot beat the constant median ON THE SUPPORT
        # SET's own cross-validation, it has diverged and must not be
        # reported -- that is what produced ad-ctr's 10x. Falling back to the
        # constant is the honest answer, and it is decided WITHOUT touching
        # the eval split.
        const_cv = float(np.abs(ys - np.median(ys)).mean())
        out["fell_back_to_median"] = bool(cv_best is not None
                                          and cv_best >= const_cv)
        if out.get("fell_back_to_median"):
            p = np.full(len(ev_y), c0, dtype=np.float64)
        else:
            p = m.predict(B) * s0 + c0
        mae = float(np.abs(p - ev_y).mean())
        std = nmae_std
        out["mae"] = mae
        out["nmae"] = mae / std
        print(f"\n  LEDGER+ICL (fitted composition)  mae={mae:.4f}"
              f"  nmae={out['nmae']:.4f}"
              f"  [L1/median, alpha={out['alpha']:.4g} by 5-fold CV"
              + (", FELL BACK to constant median" if
                 out.get("fell_back_to_median") else "") + "]")

    # The zero-label readout on the SAME checkpoint and the SAME eval rows,
    # so the two settings are comparable and the delta is attributable.
    q = wr.QUERIES.get((args.dataset, args.task))
    if q is not None:
        z = z_ev
        base = np.zeros(len(ev_rows))
        base[kev] = wr.predict(model, z, q, "auto").cpu().numpy()
        if is_cls:
            out["zero_label_roc_auc"] = float(roc_auc_score(ev_y, base))
            print(f"  LEDGER (zero-label, composed)    "
                  f"roc_auc={out['zero_label_roc_auc']:.4f}")
            print(f"  delta {100*(out['roc_auc']-out['zero_label_roc_auc']):+.2f}")
        else:
            mae0 = float(np.abs(base - ev_y).mean())
            out["zero_label_nmae"] = mae0 / nmae_std
            print(f"  LEDGER (zero-label, composed)    "
                  f"nmae={out['zero_label_nmae']:.4f}")
            print(f"  delta {out['nmae']-out['zero_label_nmae']:+.4f}")

    if args.pred_out:
        if args.rows:
            raise SystemExit("--pred-out requires --rows 0 (full split); "
                             "a capped run does not align with the official "
                             "prediction table")
        d = pathlib.Path(args.pred_out); d.mkdir(parents=True, exist_ok=True)
        stem = f"{args.dataset}__{args.task}__{args.split}"
        # Save the KEY COLUMNS with the predictions, not a bare vector. The
        # installed relbench (2.1.2) and the submission-side relbench (3.x)
        # return the same test rows in DIFFERENT ORDER -- v3 is sorted by
        # timestamp, 2.1.2 is not -- so writing a positional vector into v3's
        # table silently scrambles it (rel-f1 driver-position: NMAE 0.6331
        # became 0.7051). The consumer merges on these keys instead.
        keyed = ev_df[[task.entity_col, task.time_col]].copy()
        keyed["pred_probe"] = np.asarray(p, dtype=np.float64)
        if "base" in dir():
            keyed["pred_zero"] = np.asarray(base, dtype=np.float64)
        keyed.to_parquet(d / f"{stem}.parquet", index=False)
        print(f"  saved {len(keyed):,} keyed predictions -> {d}/{stem}.parquet")

    if args.json:
        with open(args.json, "a") as f:
            f.write(json.dumps(dict(dataset=args.dataset, task=args.task,
                                    split=args.split, support=len(tr_rows),
                                    results=out)) + "\n")


if __name__ == "__main__":
    main()
