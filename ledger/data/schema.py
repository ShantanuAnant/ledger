"""Schema description extracted from a RelBench database.

The schema is the static vocabulary of the world model: which tables exist,
which of them are *fact tables* (have a timestamp -> their rows are events),
which are *dimension tables* (no timestamp -> they define entity identities),
and for each fact table, its foreign-key slots and feature columns.

Everything downstream (tokenizer, heads) is sized from this object, so a
single model can in principle be re-instantiated on a new database by
swapping the schema (schema-agnostic mode replaces the learned id embeddings
with name/type text embeddings; see ARCHITECTURE.md section 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# Categorical columns with more distinct values than this are hashed into
# this many buckets instead of getting a dedicated vocabulary. Keeps the
# embedding tables bounded on high-cardinality columns (zip codes, ...).
MAX_CATEGORY_VOCAB = 1024


@dataclass
class ColumnSpec:
    name: str
    kind: str  # "numeric" | "categorical"
    # numeric: standardization stats from the pre-cutoff corpus only
    mean: float = 0.0
    std: float = 1.0
    # categorical: value -> index (index 0 reserved for unknown/missing)
    vocab: dict = field(default_factory=dict)
    hashed: bool = False
    # Provenance for a DERIVED column (see data/denorm.py): None for a column
    # the fact table physically has, otherwise a tuple naming the mechanism
    # and the table/column it was pulled from. A plain default (not a
    # default_factory) so unpickling a corpus cached before this field existed
    # falls back to the class attribute instead of raising.
    source: tuple | None = None

    @property
    def cardinality(self) -> int:
        return (MAX_CATEGORY_VOCAB if self.hashed else len(self.vocab)) + 1


@dataclass
class FactTableSpec:
    name: str
    table_idx: int
    # foreign-key slot name -> target (dimension or fact) table name
    fkeys: dict = field(default_factory=dict)
    columns: list = field(default_factory=list)  # list[ColumnSpec]


@dataclass
class Schema:
    fact_tables: dict  # name -> FactTableSpec
    entity_tables: list  # every table that can be pointed at by a foreign key
    entity_counts: dict  # entity table name -> number of ids (contiguous 0..n-1)
    # Primary-key <-> contiguous row-index translation, needed to move between
    # RelBench task tables (which speak primary keys) and the corpus (which
    # speaks row indices). Built once here so corpus.py and queries.py agree.
    pkey_values: dict = field(default_factory=dict)  # table -> np[pkey] by row
    pkey_index: dict = field(default_factory=dict)   # table -> Series pkey->row

    @property
    def num_fact_tables(self) -> int:
        return len(self.fact_tables)


# A column whose values are nearly all distinct is free text or a stray
# identifier, not a category. Hashing it into MAX_CATEGORY_VOCAB buckets --
# what v0 did -- is strictly worse than dropping it: the buckets carry no
# signal, they feed noise into the tokenizer, AND the WHAT head is then asked
# to predict a 1025-way uniform distribution it can never reduce. rel-stack
# post bodies and comment texts are the case that forced this.
# (The real fix for text is the frozen sentence-embedding route of
# ARCHITECTURE.md section 4; dropping is the honest interim.)
NEAR_UNIQUE_FRAC = 0.2


def column_spec(s: pd.Series, name: str,
                max_vocab: int = MAX_CATEGORY_VOCAB,
                force_numeric: bool = False,
                source: tuple | None = None) -> "ColumnSpec | None":
    """One ColumnSpec from one column of PRE-CUTOFF rows, or None to drop it.

    `force_numeric` is for derived count/sum columns (data/denorm.py): a child
    count takes values 0, 1, 2, ... and the 20-distinct-value rule would type
    it as a CATEGORY, which loses the ordering the whole point of the column
    depends on -- the user-clicks label is a SUM of one of them.
    """
    if force_numeric:
        v = pd.to_numeric(s, errors="coerce")
        return ColumnSpec(name, "numeric", mean=float(v.mean()),
                          std=float(v.std()) or 1.0, source=source)
    if pd.api.types.is_numeric_dtype(s):
        try:
            many = s.nunique() > 20
        except TypeError:
            return None
        if many:
            return ColumnSpec(name, "numeric", mean=float(s.mean()),
                              std=float(s.std()) or 1.0, source=source)
    try:
        uniques = s.dropna().unique()
    except TypeError:
        # a column holding an unhashable payload per row (rel-amazon
        # `product.category` is a numpy array of tags) is neither a number nor
        # a category. Dimension tables were never typed before FK
        # denormalization, so this only became reachable today.
        return None
    n_valid = max(int(s.notna().sum()), 1)
    if len(uniques) > max_vocab and len(uniques) > NEAR_UNIQUE_FRAC * n_valid:
        return None                       # free text / identifier: drop
    if len(uniques) > max_vocab:
        if max_vocab < MAX_CATEGORY_VOCAB:
            return None      # a denormalized column is capped, not hashed
        return ColumnSpec(name, "categorical", hashed=True, source=source)
    vocab = {v: i + 1 for i, v in enumerate(sorted(map(str, uniques)))}
    return ColumnSpec(name, "categorical", vocab=vocab, source=source)


def _column_specs(df: pd.DataFrame, skip: set, cutoff_mask: np.ndarray,
                  extra: dict | None = None) -> list:
    """Build column specs using statistics from pre-cutoff rows ONLY.

    Using full-corpus statistics would leak aggregate information about the
    future (e.g. the mean of a price column shifting over time). Cheap to do
    correctly, so we do it correctly.

    `extra` holds derived columns (data/denorm.py), already restricted to the
    same pre-cutoff rows and in the same order.
    """
    specs = []
    sub = df.loc[cutoff_mask]
    for col in df.columns:
        if col in skip:
            continue
        sp = column_spec(sub[col], col)
        if sp is not None:
            specs.append(sp)
    for name, ex in (extra or {}).items():
        sp = column_spec(ex.values, name, max_vocab=ex.max_vocab,
                         force_numeric=ex.force_numeric, source=ex.source)
        if sp is not None:
            specs.append(sp)
    return specs


def build_schema(db, cutoff: pd.Timestamp, denorm_fk: bool = False,
                 child_aggs: bool = False) -> Schema:
    """Extract the Schema from a relbench Database.

    `cutoff` is mandatory: column statistics must not see post-cutoff rows.

    `denorm_fk` / `child_aggs` add the derived columns of data/denorm.py to
    each fact table's spec. They are schema-level rather than model-level
    because everything downstream -- tokenizer, WHAT head, WindowHead rate and
    bucket outputs, the task readouts -- is sized and named from the spec, so
    a derived column becomes usable everywhere without another line of code.
    """
    from .denorm import extra_columns
    # An "entity" is a table some foreign key actually POINTS AT. A primary
    # key alone is not enough: on rel-stack all seven tables have one, but
    # nothing references comments / votes / badges / postHistory / postLinks,
    # so treating them as entities allocated state buffers for ~3.6M ids that
    # can never be read or ranked. Restricting to real FK targets takes
    # rel-stack from 4.2M entities to 589K.
    fk_targets = {t for tbl in db.table_dict.values()
                  for t in tbl.fkey_col_to_pkey_table.values()}

    fact_tables: dict = {}
    entity_counts: dict = {}
    pkey_values: dict = {}
    pkey_index: dict = {}
    idx = 0
    for name, table in db.table_dict.items():
        if table.pkey_col is not None:
            # pkey maps stay available for every keyed table (queries.py needs
            # them to translate task tables); only entity_counts is narrowed.
            if name in fk_targets:
                entity_counts[name] = int(len(table.df))
            pk = table.df[table.pkey_col].to_numpy()
            pkey_values[name] = pk
            pkey_index[name] = pd.Series(
                np.arange(len(pk), dtype=np.int64), index=pk)
        if table.time_col is None:
            continue
        df = table.df
        mask = (df[table.time_col] <= cutoff).to_numpy()
        skip = {table.time_col, table.pkey_col, *table.fkey_col_to_pkey_table}
        extra = extra_columns(db, name, cutoff, mask,
                              denorm_fk=denorm_fk, child_aggs=child_aggs)
        fact_tables[name] = FactTableSpec(
            name=name,
            table_idx=idx,
            fkeys=dict(table.fkey_col_to_pkey_table),
            columns=_column_specs(df, skip, mask, extra),
        )
        idx += 1
    return Schema(
        fact_tables=fact_tables,
        entity_tables=sorted(entity_counts),
        entity_counts=entity_counts,
        pkey_values=pkey_values,
        pkey_index=pkey_index,
    )
