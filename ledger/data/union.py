"""Merge several single-database corpora into ONE schema, for joint training.

WHY THIS WORKS AT ALL. Every per-table parameter in the model is an
`nn.ModuleDict` keyed by table NAME -- `EventTokenizer.num_proj` / `cat_emb` /
`link_proj`, `WhatHead.num_out` / `cat_out`, `WindowHead.rate` and friends. So
a union schema does not need any of them to change shape: it adds entries.
What is genuinely SHARED is the transformer trunk, the time encoding, the WHEN
head, and `table_emb`, and that sharing is the whole hypothesis under test.

WHAT THIS IS NOT. It is not leave-one-database-out zero-shot transfer, and it
cannot be made into it. A held-out database's tables have no `num_proj`, no
`cat_emb` and no row in `table_emb`, so the model has no way to read them --
`RT (zero-shot, leave-one-DB-out)` on the leaderboard is a schema-agnostic
architecture and ours is not. What is testable here is JOINT training: does a
trunk that has seen seven databases beat a trunk that has seen one, on a task
from a database both have seen?

NAME COLLISIONS ARE REAL. `users` is a fact table in BOTH rel-event and
rel-stack, and merging without namespacing would silently make one database's
`users` projection consume the other's features -- same failure class as the
categorical index shift in HANDOFF 10.1a: no error, plausible numbers, wrong
model. Every table and entity name is therefore prefixed with its dataset.
"""
from __future__ import annotations

import copy

import numpy as np

from .schema import Schema


def qualify(dataset: str, name: str) -> str:
    return f"{dataset}::{name}"


def namespaced(corpus, dataset: str, base_idx: int):
    """A copy of `corpus` whose tables are `dataset::name` and whose
    `table_idx` values start at `base_idx`.

    The event arrays are shared by reference where they are not rewritten --
    only `table_idx` is a new array, because it is the one thing whose VALUES
    change. rel-amazon's corpus is 19 GB; copying it per union would not fit.
    """
    sch = corpus.schema
    ft, off = {}, {}
    for i, (name, spec) in enumerate(sorted(sch.fact_tables.items(),
                                            key=lambda kv: kv[1].table_idx)):
        sp = copy.copy(spec)
        sp.table_idx = base_idx + i
        # `fkeys` names its TARGET tables, and those names index
        # `entity_counts` and the per-entity state tables. Left unqualified
        # they would resolve against whichever database happened to define
        # that name -- rel-f1's `races` and rel-stack's `users` are both live
        # cases -- so a foreign key would silently point into another
        # database's entity space.
        sp.fkeys = {col: qualify(dataset, tgt)
                    for col, tgt in spec.fkeys.items()}
        off[spec.table_idx] = sp.table_idx
        ft[qualify(dataset, name)] = sp

    remap = np.full(int(max(off) + 1) if off else 1, -1, dtype=np.int64)
    for a, b in off.items():
        remap[a] = b

    new = copy.copy(corpus)
    new.schema = Schema(
        fact_tables=ft,
        entity_tables=[qualify(dataset, e) for e in sch.entity_tables],
        entity_counts={qualify(dataset, k): v
                       for k, v in sch.entity_counts.items()},
        pkey_values={qualify(dataset, k): v
                     for k, v in sch.pkey_values.items()},
        pkey_index={qualify(dataset, k): v
                    for k, v in sch.pkey_index.items()},
    )
    new.table_idx = remap[corpus.table_idx.astype(np.int64)].astype(np.int16)
    new.feat_num = {qualify(dataset, k): v for k, v in corpus.feat_num.items()}
    new.feat_cat = {qualify(dataset, k): v for k, v in corpus.feat_cat.items()}
    new.links = {qualify(dataset, k): v for k, v in corpus.links.items()}
    new.row_of = {qualify(dataset, k): v for k, v in corpus.row_of.items()}
    new.hist_index = {qualify(dataset, k): v
                      for k, v in corpus.hist_index.items()}
    new.hist_offset = {qualify(dataset, k): v
                       for k, v in corpus.hist_offset.items()}
    return new, len(ft)


def union_schema(corpora: dict) -> Schema:
    """One Schema spanning every namespaced corpus. Table indices are already
    globally unique, so this is a merge and not a renumbering."""
    ft, ent, cnt, pv, pi = {}, [], {}, {}, {}
    for c in corpora.values():
        s = c.schema
        ft.update(s.fact_tables)
        ent += list(s.entity_tables)
        cnt.update(s.entity_counts)
        pv.update(s.pkey_values)
        pi.update(s.pkey_index)
    idx = sorted(sp.table_idx for sp in ft.values())
    assert idx == list(range(len(ft))), \
        f"table_idx must be contiguous 0..{len(ft) - 1}, got {idx[:5]}..."
    return Schema(fact_tables=ft, entity_tables=ent, entity_counts=cnt,
                  pkey_values=pv, pkey_index=pi)


def build_union(loaded: dict):
    """`{dataset: corpus}` -> `({dataset: namespaced corpus}, union Schema)`."""
    out, base = {}, 0
    for ds in sorted(loaded):
        c, n = namespaced(loaded[ds], ds, base)
        out[ds] = c
        base += n
    return out, union_schema(out)


class MultiBatcher:
    """Round-robin over one `PackedBatcher` per database.

    One batch is drawn from ONE database. Mixing databases inside a batch
    would be possible -- the union schema makes the token ids compatible --
    but the state-write rule keys off `seq_entity`, which is per batch, and
    the WHO negatives are sampled within a database's own entity space. Both
    become ambiguous the moment two databases share a batch, so the mixing
    happens ACROSS steps and the shared trunk is what carries signal between
    them. That is the same granularity `PackedBatcher` already uses for its
    two entity sides (D1).

    Databases are sampled in proportion to `probs` (default uniform, NOT by
    corpus size: rel-event has 154x the events of rel-f1 and size-weighting
    would make the small databases invisible).
    """

    def __init__(self, batchers: dict, probs=None, seed: int = 0):
        self.names = sorted(batchers)
        self.b = batchers
        self.rng = np.random.default_rng(seed + 991)
        if probs is None:
            probs = np.full(len(self.names), 1.0 / len(self.names))
        self.probs = np.asarray(probs, dtype=np.float64)
        self.probs /= self.probs.sum()

    def batch(self, device="cpu", entity_table: str | None = None) -> dict:
        ds = self.names[int(self.rng.choice(len(self.names), p=self.probs))]
        out = self.b[ds].batch(device=device, entity_table=entity_table)
        out["dataset"] = ds
        return out

    def __getattr__(self, k):
        # PrefetchLoader and the trainer read a few batcher attributes
        # (`c`, `schema`, ...). Delegate to the first database's batcher;
        # anything genuinely per-database must be read off the batch instead.
        return getattr(self.b[self.names[0]], k)
