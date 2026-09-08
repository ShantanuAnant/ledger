"""Denormalization: pulling related-table information onto an event's own
feature vector.

WHY THIS EXISTS
---------------
An event token carries its own row's columns, plus the learned STATE of every
entity its foreign keys point at. It never carries a related row's column
VALUES. Three benchmark tasks are blocked by exactly that gap, and reading
their `make_table` SQL shows the gap is the whole of the difficulty
(RESEARCH.md 2026-08-25):

  rel-trial study-outcome   filters `outcomes.outcome_type = 'Primary'`, a
                            column of a DIFFERENT table from the event
                            (`outcome_analyses`). Counting every outcome type
                            is a 4x dilution: only 25.6% of analyses are
                            Primary, and the unfiltered analysis count alone
                            scores 55.09 AUROC against our 54.16.

  rel-amazon user-ltv       sums `product.price`. `review` has NO numeric
  rel-amazon item-ltv       column at all, so the two tasks were not merely
                            weak, they were inexpressible.

  rel-avito user-clicks     counts `SearchStream` rows with `IsClick = 1`,
                            reachable only as UserInfo -> SearchInfo ->
                            SearchStream. Those events never enter a user's
                            history, so the task scored EXACTLY 50.0.

Two mechanisms, both schema-generic and both applied at corpus-build time so
every consumer (tokenizer, WHAT head, WindowHead rates/buckets/sums, the
readouts) picks them up with no further code:

  FK-TARGET COLUMNS (`denorm_fk`)
      For each foreign key of a fact table, copy the TARGET row's
      low-cardinality categorical and numeric columns onto the event.
      `outcome_analyses` gains `outcome_id.outcome_type`; `review` gains
      `product_id.price`.

  CHILD AGGREGATES (`child_aggs`)
      For a fact table M with children F (any table holding a foreign key
      into M), add to M's rows: `count(F)`, the sum of each numeric column of
      F, and the count of each value of each low-cardinality categorical
      column of F. `SearchInfo` gains
      `SearchStream@SearchID.IsClick=1.0.n` -- the number of clicked ads in
      that search -- and the user-clicks label is then a window aggregate of a
      column the user's own history already contains.

      Deliberately NOT "file the child events into the grandparent's history":
      that would spread rel-avito's 9.25M SearchStream rows across user
      histories and change what an event IS. An aggregate is one float per
      parent row.

LEAKAGE
-------
Both mechanisms respect the same rule as everything else: a value attached to
an event at time `t` must be knowable at `t`.

  * A DIMENSION target (no timestamp) is a static attribute record, visible
    whenever the entity is -- the same treatment ARCHITECTURE 3.1 already
    gives dimension rows, and the same information the RDL baselines get.
  * A FACT target with a timestamp is blanked wherever the target row is
    LATER than the event that points at it.
  * A child contributes to its parent's aggregate only when the child's own
    timestamp is at or before the parent's. On rel-avito that keeps
    everything (SearchStream shares SearchInfo's date exactly); on rel-trial
    it correctly drops the outcome analyses that a study has not produced
    yet.
  * Every value is read from the PRE-CUTOFF slice of both tables, so column
    statistics and vocabularies stay pre-cutoff statistics.

There is no flag that relaxes any of these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A denormalized categorical wider than this gets no column: it would add a
# large embedding to every event of the table and it is useless as a task
# predicate. 64 matches window.CAT_RATE_MAX, above which the WindowHead emits
# no per-category rate either -- so a wider column could not be filtered on
# even if it were carried.
FK_CAT_MAX = 64
# Per fact table, across all of its foreign keys. rel-trial's `studies` has 28
# columns and ten tables point at it; without a cap one schema decision would
# quietly triple the token width.
FK_MAX_COLS = 16

# Categorical child columns wider than this are not expanded into per-value
# counters (each value costs one float per parent row).
CHILD_CAT_MAX = 16
CHILD_MAX_COLS = 32
# Longest value that still reads as a label rather than a sentence.
CHILD_CAT_LABEL_CHARS = 64


@dataclass
class Extra:
    """One derived column, aligned to the PRE-CUTOFF rows of its fact table."""
    values: pd.Series
    source: tuple            # provenance, recorded on the ColumnSpec
    force_numeric: bool = False
    max_vocab: int = FK_CAT_MAX


def _unix(s: pd.Series) -> np.ndarray:
    return s.astype("int64").to_numpy() // 10 ** 9


def _positions(keys: pd.Series, index_values: np.ndarray) -> np.ndarray:
    """Row positions of `keys` in a table keyed by `index_values`; -1 = miss."""
    m = pd.Series(np.arange(len(index_values), dtype=np.int64),
                  index=index_values)
    # duplicate primary keys would make `map` raise; RelBench pkeys are unique,
    # and if one ever is not, failing loudly here beats a silent join fan-out
    return keys.map(m).fillna(-1).to_numpy(dtype=np.int64)


def _hashable(s: pd.Series) -> bool:
    """rel-amazon `product.category` holds a numpy ARRAY per row, so `unique()`
    raises `unhashable type`. Such a column is not a category and not a
    number; it is skipped rather than crashing the build."""
    v = s.dropna()
    if not len(v):
        return False
    try:
        hash(v.iloc[0])
    except TypeError:
        return False
    return True


def fk_target_columns(db, name: str, cutoff: pd.Timestamp,
                      pre_mask: np.ndarray, want: set | None = None) -> dict:
    """Columns copied from each foreign-key TARGET onto `name`'s rows.

    `pre_mask` selects the fact table's pre-cutoff rows, in table order --
    the same rows `EventCorpus.build` keeps, so the returned values are
    positionally aligned with the corpus arrays.

    The column budget is spent ROUND-ROBIN across the foreign keys, on
    columns that will survive typing. Taking the first `FK_MAX_COLS` in slot
    order instead is what a first version did, and on rel-trial
    `outcome_analyses` the whole budget went to `nct_id -> studies` -- five of
    those columns were free text that the typing layer then dropped anyway,
    and `outcome_id -> outcomes` never got a turn. `outcome_type` is the one
    column the task needs.

    `want` names the columns to produce, bypassing both the viability test and
    the budget. It exists for the TEST split: `scripts/eval_window.py` pins
    the column specs to the TRAINING cutoff (HANDOFF 10.1a) and then rebuilds
    the corpus at the later one, so the selection must be reproduced by NAME
    rather than re-derived -- a column whose cardinality crossed a threshold
    between cutoffs would otherwise silently drop out of the schema the
    checkpoint expects.
    """
    from .schema import column_spec

    table = db.table_dict[name]
    df = table.df.loc[pre_mask]
    etime = _unix(df[table.time_col])
    per_slot: list = []
    for slot, target in table.fkey_col_to_pkey_table.items():
        tt = db.table_dict.get(target)
        if tt is None or tt.pkey_col is None:
            continue
        tdf = tt.df
        if tt.time_col is not None:
            tdf = tdf[tdf[tt.time_col] <= cutoff]
        if not len(tdf):
            continue
        # Viability is decided on the TARGET's own column, not on the joined
        # array: same verdict (a join creates no new values), ~20x cheaper on
        # rel-amazon, and it is the honest question -- "is `product.title` an
        # identifier?" is a property of `product`.
        if want is not None:
            keep = [c for c in tt.df.columns if f"{slot}.{c}" in want]
        else:
            keep = [c for c in tt.df.columns
                    if c not in {tt.time_col, tt.pkey_col,
                                 *tt.fkey_col_to_pkey_table}
                    and (pd.api.types.is_numeric_dtype(tt.df[c])
                         or _hashable(tt.df[c]))
                    and column_spec(tdf[c], c, max_vocab=FK_CAT_MAX)
                    is not None]
        if keep:
            per_slot.append((slot, target, tt, keep))

    out: dict = {}
    cache: dict = {}          # slot -> (target row index, visible mask)
    for depth in range(max((len(k) for _, _, _, k in per_slot), default=0)):
        for slot, target, tt, keep in per_slot:
            if depth >= len(keep) or (want is None
                                      and len(out) >= FK_MAX_COLS):
                continue
            col = keep[depth]
            if slot not in cache:
                pos = _positions(df[slot], tt.df[tt.pkey_col].to_numpy())
                ok = pos >= 0
                safe = np.where(ok, pos, 0)
                if tt.time_col is not None:
                    # a fact-table target may be LATER than the event pointing
                    # at it; such a row is not knowable at the event's time,
                    # and a post-cutoff row is never knowable at all
                    ttime = _unix(tt.df[tt.time_col])
                    ok = (ok & (ttime[safe] <= etime)
                          & (ttime[safe] <= int(pd.Timestamp(cutoff)
                                                .timestamp())))
                cache[slot] = (safe, ok)
            safe, ok = cache[slot]
            src = tt.df[col]
            vals = pd.Series(src.to_numpy()[safe])
            vals[~ok] = np.nan if pd.api.types.is_numeric_dtype(src) else None
            out[f"{slot}.{col}"] = Extra(
                values=vals, source=("fk", slot, target, col),
                max_vocab=FK_CAT_MAX)
    return out


def _child_numeric_and_categorical(s: pd.Series):
    """-> ("numeric" | "categorical" | None, uniques). Deliberately the same
    20-distinct-value split `schema._column_specs` uses, so a column is
    aggregated the way it would be tokenized."""
    if not _hashable(s) and not pd.api.types.is_numeric_dtype(s):
        return None, None
    if pd.api.types.is_numeric_dtype(s):
        try:
            if s.nunique() > 20:
                return "numeric", None
        except TypeError:
            return None, None
    if not _hashable(s):
        return None, None
    u = s.dropna().unique()
    if len(u) > CHILD_CAT_MAX or len(u) == 0:
        return None, None
    # A column with a handful of 200-character sentences passes the
    # distinct-value test and is still free text, not a label. rel-trial
    # `outcome_analyses.ci_upper_limit_na_comment` is one, and expanding it
    # spent eight of the parent's thirty-two column slots on prose.
    if max((len(str(v)) for v in u), default=0) > CHILD_CAT_LABEL_CHARS:
        return None, None
    return "categorical", u


def child_aggregate_columns(db, name: str, cutoff: pd.Timestamp,
                            pre_mask: np.ndarray,
                            want: set | None = None) -> dict:
    """count / sum / per-category-count of every child table of `name`.

    Emitted in three rounds -- counts for every child, then numeric sums, then
    categorical expansions -- so a parent with many children (rel-trial
    `studies` has ten) spends its column budget across all of them instead of
    exhausting it on the first.
    """
    table = db.table_dict[name]
    if table.pkey_col is None:
        return {}
    df = table.df.loc[pre_mask]
    n_par = len(df)
    ptime = _unix(df[table.time_col])
    keys = df[table.pkey_col].to_numpy()
    cut = int(pd.Timestamp(cutoff).timestamp())

    # -- resolve every child once: which parent row each child row belongs to
    children = []
    for cname, ct in db.table_dict.items():
        for slot, tgt in ct.fkey_col_to_pkey_table.items():
            if tgt != name:
                continue
            cdf = ct.df
            if ct.time_col is not None:
                cdf = cdf[cdf[ct.time_col] <= cutoff]
            if not len(cdf):
                continue
            pos = _positions(cdf[slot], keys)
            ok = pos >= 0
            if ct.time_col is not None:
                ctime = _unix(cdf[ct.time_col])
                safe = np.where(ok, pos, 0)
                ok = ok & (ctime <= ptime[safe]) & (ctime <= cut)
            if not ok.any():
                continue
            children.append((cname, slot, cdf, pos[ok], ok))

    out: dict = {}

    def _emit(key, arr, source):
        # `want` (see fk_target_columns) reproduces a pinned schema by name
        if want is not None:
            if key in want:
                out[key] = Extra(values=pd.Series(arr), force_numeric=True,
                                 source=source)
        elif len(out) < CHILD_MAX_COLS:
            out[key] = Extra(values=pd.Series(arr), force_numeric=True,
                             source=source)

    for cname, slot, cdf, pos, ok in children:
        _emit(f"{cname}@{slot}.n",
              np.bincount(pos, minlength=n_par).astype(np.float64),
              ("child_count", cname, slot))

    skip_of = {(c, s): {db.table_dict[c].time_col, db.table_dict[c].pkey_col,
                        *db.table_dict[c].fkey_col_to_pkey_table}
               for c, s, _, _, _ in children}
    kinds: dict = {}
    for cname, slot, cdf, pos, ok in children:
        for col in cdf.columns:
            if col in skip_of[(cname, slot)]:
                continue
            kinds[(cname, slot, col)] = _child_numeric_and_categorical(
                pd.Series(cdf[col].to_numpy()[ok]))

    for cname, slot, cdf, pos, ok in children:
        for col in cdf.columns:
            kind, _ = kinds.get((cname, slot, col), (None, None))
            if kind != "numeric":
                continue
            v = np.nan_to_num(
                cdf[col].to_numpy(dtype=np.float64)[ok], nan=0.0)
            _emit(f"{cname}@{slot}.{col}.sum",
                  np.bincount(pos, weights=v, minlength=n_par),
                  ("child_sum", cname, slot, col))

    for cname, slot, cdf, pos, ok in children:
        for col in cdf.columns:
            kind, u = kinds.get((cname, slot, col), (None, None))
            if kind != "categorical":
                continue
            vals = pd.Series(cdf[col].to_numpy()[ok]).astype(str).to_numpy()
            for v in sorted(map(str, u)):
                _emit(f"{cname}@{slot}.{col}={v}.n",
                      np.bincount(pos[vals == v], minlength=n_par
                                  ).astype(np.float64),
                      ("child_cat_count", cname, slot, col, v))
    return out


def extra_columns(db, name: str, cutoff: pd.Timestamp, pre_mask: np.ndarray,
                  denorm_fk: bool = False, child_aggs: bool = False,
                  want: set | None = None) -> dict:
    """All derived columns for one fact table, in a deterministic order.

    `want`, when given, is the exact set of derived column names to produce --
    the names a pinned schema already carries. See `fk_target_columns`.
    """
    out: dict = {}
    if child_aggs:
        out.update(child_aggregate_columns(db, name, cutoff, pre_mask, want))
    if denorm_fk:
        out.update(fk_target_columns(db, name, cutoff, pre_mask, want))
    return out
