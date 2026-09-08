"""EventCorpus: the database as flat, columnar event arrays.

Design (ARCHITECTURE.md section 3): events are stored ONCE in flat numpy
arrays sorted by time; per-entity histories are index lists into them. An
event appearing in several entities' histories costs one integer per entity,
not a copy.

THE LEAKAGE INVARIANT lives here. `cutoff` is a required positional argument
of `build`, the corpus physically contains no event after it, and
`test_leakage.py` asserts this on every build. There is no flag to disable
it. See ARCHITECTURE.md section 3.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import Schema, build_schema

# Sentinel for "no linked entity in this slot" (NaN foreign keys).
NO_ENTITY = -1


def parse_path_expand(spec: str) -> list:
    """`"<entity>:<bridge>:<events>[,...]"` -> [(entity, bridge, events)]."""
    out = []
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        bits = [x.strip() for x in part.split(":")]
        if len(bits) != 3 or not all(bits):
            raise ValueError(
                f"--path_expand entry {part!r} must be "
                f"<entity>:<bridge_table>:<event_table>")
        out.append(tuple(bits))
    return out


def _expand_path(schema, ent, bridge, evt, time, table_idx, row_of, links):
    """Events of `evt` filed into `ent`'s history through the `bridge` table.

    TARGET-SIDE PATH EXPANSION (ARCHITECTURE 7.3.1, the target-side analogue
    of `--retarget`). Histories are otherwise built purely by inverting
    foreign keys, so an entity only ever sees events that point AT it. That
    makes rel-trial site-success inexpressible: the label aggregates
    `outcome_analyses` of the studies a facility ran, but those analyses key
    on `nct_id -> studies`, never on `facility_id`, so a facility's history
    contains `facilities_studies` rows and nothing else. Measured: 90.5% of
    facility histories hold `facilities_studies` only and 0% hold a single
    `outcome_analyses` row, so the readout was computing a rate over events
    the entity had never observed.

    `ent` and `evt` are joined over the entity BOTH sides point at (`studies`
    here), which `bridge` must reference along with `ent`.

    LEAKAGE. A reference is kept only when the bridge row is dated at or
    before the event: `history(before=cut)` filters on the EVENT's time, so
    without this a cutoff falling between the two would let a facility see an
    analysis before it was known to have run that study. The event keeps its
    own timestamp, which is what makes this sound where child-aggregate
    denormalization is not -- there a child dated after its parent is folded
    into the parent's token, and no per-row filter can undo it.

    Pairs are de-duplicated: a facility that appears in several
    `facilities_studies` rows for one study must not have that study's
    analyses counted more than once, which would inflate every rate the
    window head reads.
    """
    b_spec = schema.fact_tables.get(bridge)
    e_spec = schema.fact_tables.get(evt)
    if b_spec is None or e_spec is None:
        raise KeyError(f"path_expand {ent}:{bridge}:{evt}: "
                       f"{bridge if b_spec is None else evt} is not a fact "
                       f"table")
    b_tgt, e_tgt = list(b_spec.fkeys.values()), list(e_spec.fkeys.values())
    if ent not in b_tgt:
        raise KeyError(f"path_expand {ent}:{bridge}:{evt}: {bridge} has no "
                       f"foreign key to {ent} (targets {b_tgt})")
    shared = sorted((set(b_tgt) & set(e_tgt)) - {ent})
    if len(shared) != 1:
        raise KeyError(f"path_expand {ent}:{bridge}:{evt}: need exactly one "
                       f"shared entity between {bridge} and {evt}, got "
                       f"{shared or 'none'}")
    via = shared[0]
    b_slot_ent, b_slot_via = b_tgt.index(ent), b_tgt.index(via)
    e_slot_via = e_tgt.index(via)

    b_ev = np.flatnonzero(table_idx == b_spec.table_idx)
    b_rows = row_of[bridge][b_ev]
    b_ent = links[bridge][b_rows, b_slot_ent]
    b_via = links[bridge][b_rows, b_slot_via]
    b_t = time[b_ev]
    ok = (b_ent != NO_ENTITY) & (b_via != NO_ENTITY)
    b_ent, b_via, b_t = b_ent[ok], b_via[ok], b_t[ok]
    if not len(b_ent):
        return None

    order = np.argsort(b_via, kind="stable")
    b_ent, b_via, b_t = b_ent[order], b_via[order], b_t[order]
    n_via = schema.entity_counts[via]
    per_via = np.bincount(b_via, minlength=n_via)
    off_via = np.concatenate([[0], np.cumsum(per_via)]).astype(np.int64)

    e_ev = np.flatnonzero(table_idx == e_spec.table_idx)
    e_rows = row_of[evt][e_ev]
    e_via = links[evt][e_rows, e_slot_via]
    e_t = time[e_ev]
    ok = e_via != NO_ENTITY
    e_ev, e_via, e_t = e_ev[ok], e_via[ok], e_t[ok]
    reps = per_via[e_via] if len(e_via) else np.zeros(0, dtype=np.int64)
    keep = reps > 0
    e_ev, e_via, e_t, reps = e_ev[keep], e_via[keep], e_t[keep], reps[keep]
    total = int(reps.sum())
    if total == 0:
        return None

    # CSR gather: for every event, the bridge rows sharing its `via` entity.
    starts = np.repeat(off_via[e_via], reps)
    within = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(reps) - reps, reps)
    idx = starts + within
    ref = b_ent[idx]
    ev = np.repeat(e_ev, reps)
    known = b_t[idx] <= np.repeat(e_t, reps)        # the leakage rule
    ref, ev = ref[known], ev[known]
    if not len(ref):
        return None
    key = ref * (int(table_idx.shape[0]) + 1) + ev
    _, uniq = np.unique(key, return_index=True)
    return ref[uniq], ev[uniq]


@dataclass
class EventCorpus:
    schema: Schema
    cutoff: pd.Timestamp

    # Flat event arrays, all sorted by (time, stable original order):
    time: np.ndarray          # int64 unix seconds          [num_events]
    table_idx: np.ndarray     # int16 fact-table index      [num_events]
    # Per fact table (ragged across tables, aligned within a table):
    #   feat_num[t]  float32 [rows_t, num_numeric_cols_t]   standardized
    #   feat_cat[t]  int32   [rows_t, num_categorical_cols_t]
    #   links[t]     int64   [rows_t, num_fkey_slots_t]     entity row index
    #   row_of[t]    int32   [num_events] -> row in table t's arrays (or -1)
    feat_num: dict
    feat_cat: dict
    links: dict
    row_of: dict

    # entity table name -> {entity histories}
    #   hist_index[name]: int64 [total_refs] event ids, grouped by entity
    #   hist_offset[name]: int64 [num_entities + 1] CSR-style offsets
    hist_index: dict
    hist_offset: dict

    @property
    def num_events(self) -> int:
        return len(self.time)

    def history(self, entity_table: str, entity_row: int,
                before: int | None = None) -> np.ndarray:
        """Event ids for one entity, optionally truncated to time < before.

        `before` is unix seconds; used by the query layer to enforce per-task
        cutoffs tighter than the corpus cutoff.
        """
        o = self.hist_offset[entity_table]
        ids = self.hist_index[entity_table][o[entity_row]:o[entity_row + 1]]
        if before is not None:
            ids = ids[self.time[ids] < before]
        return ids

    def link_counts(self, entity_table: str,
                    since: int | None = None) -> np.ndarray:
        """How many corpus events link to each entity of `entity_table`.

        The popularity prior used both by the WHO interaction features and by
        the cold-start fallback. Cached, because it is read once per batch.
        `since` (unix seconds) restricts to a recent window.
        """
        key = (entity_table, since)
        cache = self.__dict__.setdefault("_link_count_cache", {})
        if key in cache:
            return cache[key]
        counts = np.zeros(self.schema.entity_counts[entity_table],
                          dtype=np.int64)
        for name, spec in self.schema.fact_tables.items():
            slots = [j for j, t in enumerate(spec.fkeys.values())
                     if t == entity_table]
            if not slots:
                continue
            ev = np.flatnonzero(self.row_of[name] >= 0)
            if since is not None:
                ev = ev[self.time[ev] >= since]
            rows = self.row_of[name][ev]
            for j in slots:
                e = self.links[name][rows, j]
                e = e[e >= 0]
                if len(e):
                    counts += np.bincount(e, minlength=len(counts))
        cache[key] = counts
        return counts

    @classmethod
    def build(cls, db, cutoff: pd.Timestamp, schema: Schema | None = None,
              self_rows: bool = False, denorm_fk: bool = False,
              child_aggs: bool = False, path_expand: str = ""):
        """`self_rows` files a fact table's row in the history of the entity
        it IS, not only in the histories of the entities it POINTS AT.

        `path_expand` (`"<entity>:<bridge>:<events>"`, comma-separated) files
        events reachable only through a bridge table into the entity's
        history -- see `_expand_path`. It changes which events a history
        contains, so like `self_rows` it is part of the cache identity.

        `denorm_fk` and `child_aggs` add the derived columns of
        `data/denorm.py` to every fact table -- an FK target's own column
        values, and count/sum/per-category aggregates of a table's children.
        Both are leakage-checked at the point of construction (see that
        module) and both change the schema, so they are part of the cache
        identity in `data/cache.py`.

        Why (measured 2026-08-25). Histories were built purely by inverting
        foreign keys, so a table that is BOTH a fact table and an entity table
        never put its own row into its own history. On rel-stack a user's
        registration row -- the only carrier of account age, the second most
        important feature in the supervised probe -- was invisible to that
        user's model, and 40.5% of user-badge eval rows had no history at all.
        On rel-trial the `studies` row holds enrollment, phase, study type and
        sponsor source, i.e. essentially everything study-outcome depends on,
        and it too was invisible. rel-stack `posts` is the same case one step
        removed: `ParentId -> posts` files a post under its PARENT, never
        under itself.

        Affects rel-stack, rel-event, rel-f1, rel-trial and rel-avito;
        rel-hm and rel-amazon have no table that is both, which is why this
        went unnoticed on the two datasets most of the tuning happened on.

        Defaults to OFF so every number measured before 2026-08-25 stays
        reproducible; `--self_rows` turns it on and the flag is recorded in
        the checkpoint so eval reconstructs the same corpus.
        """
        pinned = schema is not None
        schema = schema or build_schema(db, cutoff, denorm_fk=denorm_fk,
                                        child_aggs=child_aggs)
        from .denorm import extra_columns

        # Validated up front: a typo'd entity would otherwise match no
        # history and silently expand nothing, which is the failure mode
        # this whole mechanism exists to remove.
        path_specs = parse_path_expand(path_expand)
        for pe_ent, _, _ in path_specs:
            if pe_ent not in schema.entity_tables:
                raise KeyError(
                    f"--path_expand names entity {pe_ent!r}, which is not an "
                    f"entity table (have {schema.entity_tables})")

        # Map each entity table's primary keys to contiguous row indices.
        pkey_maps = {}
        for name in schema.entity_tables:
            t = db.table_dict[name]
            pkey_maps[name] = pd.Series(
                np.arange(len(t.df), dtype=np.int64),
                index=t.df[t.pkey_col],
            )

        times, tbl_ids, rows_in_tbl = [], [], []
        feat_num, feat_cat, links = {}, {}, {}
        own_row: dict = {}      # fact table -> entity row index of each row
        for name, spec in schema.fact_tables.items():
            t = db.table_dict[name]
            pre = (t.df[t.time_col] <= cutoff).to_numpy()   # <-- the invariant
            df = t.df.loc[pre]
            n = len(df)
            # Derived columns are aligned to exactly these pre-cutoff rows.
            # They are recomputed rather than carried over from build_schema
            # because a caller may pass a schema built elsewhere; the specs
            # name what to produce, this produces it.
            # When the caller supplied a schema (the test-split path pins
            # column specs to the TRAINING cutoff), produce exactly the
            # derived columns that schema names. Re-deriving the selection at
            # a later cutoff can pick a different set, and the spec would then
            # name a column that does not exist.
            want = {c.name for c in spec.columns if c.source is not None}
            extra = ({} if not (denorm_fk or child_aggs) else
                     extra_columns(db, name, cutoff, pre,
                                   denorm_fk=denorm_fk,
                                   child_aggs=child_aggs,
                                   want=(want if pinned else None)))

            def _col(c):
                """Values for one ColumnSpec: the table's own column, or the
                derived one of the same name."""
                ex = extra.get(c.name)
                return df[c.name] if ex is None else ex.values
            if self_rows and name in pkey_maps and name in schema.entity_counts:
                own_row[name] = (df[t.pkey_col].map(pkey_maps[name])
                                 .fillna(NO_ENTITY).to_numpy(dtype=np.int64))
            times.append(df[t.time_col].astype("int64").to_numpy() // 10**9)
            tbl_ids.append(np.full(n, spec.table_idx, dtype=np.int16))
            rows_in_tbl.append(np.arange(n, dtype=np.int32))

            nums = [c for c in spec.columns if c.kind == "numeric"]
            cats = [c for c in spec.columns if c.kind == "categorical"]
            fn = np.zeros((n, len(nums)), dtype=np.float32)
            for j, c in enumerate(nums):
                raw = pd.to_numeric(_col(c), errors="coerce").to_numpy(
                    dtype=np.float64)
                fn[:, j] = np.nan_to_num((raw - c.mean) / c.std, nan=0.0)
            fc = np.zeros((n, len(cats)), dtype=np.int32)
            for j, c in enumerate(cats):
                s = _col(c).astype(str)
                if c.hashed:
                    fc[:, j] = (
                        s.map(hash).to_numpy() % (c.cardinality - 1)
                    ) + 1
                else:
                    fc[:, j] = s.map(c.vocab).fillna(0).to_numpy(dtype=np.int32)
            lk = np.full((n, len(spec.fkeys)), NO_ENTITY, dtype=np.int64)
            for j, (slot, target) in enumerate(spec.fkeys.items()):
                lk[:, j] = (
                    df[slot].map(pkey_maps[target]).fillna(NO_ENTITY)
                    .to_numpy(dtype=np.int64)
                )
            feat_num[name], feat_cat[name], links[name] = fn, fc, lk

        time = np.concatenate(times)
        order = np.argsort(time, kind="stable")
        time = time[order]
        table_idx = np.concatenate(tbl_ids)[order]
        row_flat = np.concatenate(rows_in_tbl)[order]
        row_of = {
            name: np.where(table_idx == spec.table_idx, row_flat, -1)
            for name, spec in schema.fact_tables.items()
        }

        # Invert links -> per-entity histories (CSR layout), already sorted
        # by time because event ids are time-ordered.
        hist_index, hist_offset = {}, {}
        for ent in schema.entity_tables:
            refs, evs = [], []
            for name, spec in schema.fact_tables.items():
                slots = [j for j, tgt in enumerate(spec.fkeys.values())
                         if tgt == ent]
                if not slots:
                    continue
                # event ids of this table's rows, in flat order
                ev_ids = np.flatnonzero(table_idx == spec.table_idx)
                tbl_rows = row_of[name][ev_ids]
                for j in slots:
                    e = links[name][tbl_rows, j]
                    ok = e != NO_ENTITY
                    refs.append(e[ok])
                    evs.append(ev_ids[ok])
            # events reachable only through a bridge table (site-success)
            for pe_ent, pe_bridge, pe_evt in path_specs:
                if pe_ent != ent:
                    continue
                got = _expand_path(schema, ent, pe_bridge, pe_evt,
                                   time, table_idx, row_of, links)
                if got is not None:
                    refs.append(got[0])
                    evs.append(got[1])
            # the entity's OWN row, when this entity table is also a fact table
            if ent in own_row:
                ev_ids = np.flatnonzero(
                    table_idx == schema.fact_tables[ent].table_idx)
                e = own_row[ent][row_of[ent][ev_ids]]
                ok = e != NO_ENTITY
                refs.append(e[ok])
                evs.append(ev_ids[ok])
            n_ent = schema.entity_counts[ent]
            if refs:
                refs = np.concatenate(refs)
                evs = np.concatenate(evs)
                order2 = np.lexsort((evs, refs))
                refs, evs = refs[order2], evs[order2]
                counts = np.bincount(refs, minlength=n_ent)
            else:
                evs = np.empty(0, dtype=np.int64)
                counts = np.zeros(n_ent, dtype=np.int64)
            hist_index[ent] = evs
            hist_offset[ent] = np.concatenate(
                [[0], np.cumsum(counts)]).astype(np.int64)

        return cls(schema, cutoff, time, table_idx,
                   feat_num, feat_cat, links, row_of,
                   hist_index, hist_offset)
