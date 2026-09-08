"""Build-once, memory-map-many corpus cache.

`EventCorpus.build` reads the whole database through pandas: on rel-amazon
that is a 7 GB parquet whose two text columns the schema layer then throws
away, and it costs minutes plus tens of gigabytes of peak RSS. Every training
process paid that independently, so running K experiments concurrently -- the
entire point of the B200 -- multiplied a fixed cost by K.

Here the corpus is built once into a directory of raw `.npy` files and every
later process opens it with `mmap_mode="r"`. The arrays then live in the page
cache exactly once no matter how many jobs are running, and a job that only
touches part of the corpus never faults in the rest.

Cache identity is (dataset name, cutoff timestamp, self_rows, denorm_fk,
child_aggs). The cutoff is part of the key rather than an assumption: the
val-cutoff and test-cutoff corpora are different objects and silently sharing
them would be a leakage bug. `self_rows` is part of it for the same reason --
it changes which events an entity's history contains, so two corpora that
differ in it are different objects, and keeping both addressable is what lets
the change be measured rather than assumed. The two denormalization flags
change the COLUMNS of every fact table, so a checkpoint trained under one of
them cannot be loaded against a corpus built without it -- keying on them
turns that into a cache miss instead of a shape error, or worse, a silently
different column ordering.

The mmapped arrays are read-only. Nothing in the training path writes to a
corpus array -- it is immutable by construction -- and `_link_count_cache`
holds derived arrays in normal memory, so read-only is the honest mode. If a
caller ever needs to write, it must copy first, and the read-only flag turns
that mistake into an exception instead of silent per-process divergence.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import pickle
import time

import numpy as np
import pandas as pd

from .corpus import EventCorpus, parse_path_expand

# Fields of EventCorpus that are plain arrays, and those that are dicts of
# arrays keyed by table name. Kept explicit so adding a field to the corpus
# fails loudly here rather than being silently dropped from the cache.
_FLAT = ("time", "table_idx")
_DICTS = ("feat_num", "feat_cat", "links", "row_of",
          "hist_index", "hist_offset")

CACHE_ROOT = pathlib.Path(
    os.environ.get("LEDGER_CACHE", pathlib.Path.home() / ".cache" / "ledger"))


def cache_dir(dataset: str, cutoff: pd.Timestamp,
              self_rows: bool = False, denorm_fk: bool = False,
              child_aggs: bool = False,
              path_expand: str = "") -> pathlib.Path:
    stamp = pd.Timestamp(cutoff).strftime("%Y%m%dT%H%M%S")
    # `path_expand` changes which events a history contains, exactly as
    # `self_rows` does, so two corpora differing in it are different objects
    # and both must stay addressable. The spec is a free-form string, so it
    # is hashed rather than spelled into the path; the specs themselves are
    # recorded in the checkpoint.
    px = ""
    if path_expand:
        norm = ",".join(sorted(
            ":".join(x) for x in parse_path_expand(path_expand)))
        px = "+px" + hashlib.sha1(norm.encode()).hexdigest()[:8]
    suffix = ("" if not self_rows else "+self") \
        + ("" if not denorm_fk else "+fk") \
        + ("" if not child_aggs else "+agg") + px
    return CACHE_ROOT / f"{dataset}@{stamp}{suffix}"


def _save(path: pathlib.Path, corpus: EventCorpus) -> None:
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    manifest = {"flat": {}, "dicts": {}}
    for f in _FLAT:
        a = np.ascontiguousarray(getattr(corpus, f))
        np.save(tmp / f"{f}.npy", a)
        manifest["flat"][f] = [list(a.shape), a.dtype.str]
    for f in _DICTS:
        manifest["dicts"][f] = {}
        for k, a in getattr(corpus, f).items():
            a = np.ascontiguousarray(a)
            # table names are arbitrary strings; index them so the filename
            # is always a safe path component
            fn = f"{f}__{len(manifest['dicts'][f])}.npy"
            np.save(tmp / fn, a)
            manifest["dicts"][f][k] = fn

    with open(tmp / "schema.pkl", "wb") as fh:
        pickle.dump({"schema": corpus.schema, "cutoff": corpus.cutoff}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    with open(tmp / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)

    # Atomic publish: a half-written cache must never be openable, or a
    # concurrent job will train on a truncated corpus and we would never
    # know.
    #
    # `--rebuild` over an EXISTING cache used to raise FileExistsError here
    # (renaming a directory onto a non-empty one), after paying the whole
    # build. The old copy is swapped out first and deleted only once the new
    # one is in place, so an interrupted rebuild leaves either the old cache
    # or the old cache under `.old`, never nothing.
    import shutil
    old = path.with_name(path.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    had_old = path.exists()
    if had_old:
        path.rename(old)
    try:
        tmp.rename(path)
    except BaseException:
        if had_old:
            old.rename(path)
        raise
    if had_old:
        shutil.rmtree(old)


def _load(path: pathlib.Path, mmap: bool = True) -> EventCorpus:
    mode = "r" if mmap else None
    with open(path / "manifest.json") as fh:
        manifest = json.load(fh)
    with open(path / "schema.pkl", "rb") as fh:
        meta = pickle.load(fh)

    kw = {f: np.load(path / f"{f}.npy", mmap_mode=mode) for f in _FLAT}
    for f in _DICTS:
        kw[f] = {k: np.load(path / fn, mmap_mode=mode)
                 for k, fn in manifest["dicts"][f].items()}
    return EventCorpus(schema=meta["schema"], cutoff=meta["cutoff"], **kw)


def load_corpus(dataset: str, cutoff: pd.Timestamp, db=None,
                mmap: bool = True, rebuild: bool = False,
                verbose: bool = True,
                self_rows: bool = False, denorm_fk: bool = False,
                child_aggs: bool = False,
                path_expand: str = "") -> EventCorpus:
    """Get the corpus for (dataset, cutoff), building and caching on miss.

    `db` is only touched on a miss, so the caller can pass a lazy loader and
    avoid reading the database at all in the common case. Pass a callable to
    defer `get_db()` itself -- on rel-amazon that call alone is the expensive
    part.
    """
    path = cache_dir(dataset, cutoff, self_rows, denorm_fk, child_aggs,
                     path_expand)
    if path.exists() and not rebuild:
        c = _load(path, mmap=mmap)
        if verbose:
            print(f"corpus: {c.num_events:,} events <= {c.cutoff} "
                  f"(mmap {path})", flush=True)
        return c

    if db is None:
        raise FileNotFoundError(
            f"no cached corpus at {path} and no db given. Build it with "
            f"`python -m ledger.data.cache --dataset {dataset}`.")
    if callable(db):
        db = db()
    t0 = time.time()
    corpus = EventCorpus.build(db, pd.Timestamp(cutoff), self_rows=self_rows,
                               denorm_fk=denorm_fk, child_aggs=child_aggs,
                               path_expand=path_expand)
    if verbose:
        print(f"corpus: built {corpus.num_events:,} events <= {cutoff} "
              f"in {time.time() - t0:.0f}s", flush=True)
    _save(path, corpus)
    if verbose:
        print(f"corpus: cached to {path}", flush=True)
    # Return the mmapped view, not the in-memory one, so a build process and
    # a cache-hit process behave identically (same dtypes, same writeability).
    return _load(path, mmap=mmap)


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test", "both"])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--self-rows", action="store_true",
                    help="file a fact table's own row in its own entity's "
                         "history (see EventCorpus.build)")
    ap.add_argument("--denorm-fk", action="store_true",
                    help="copy an FK target's low-cardinality categorical and "
                         "numeric columns onto the event (see data/denorm.py)")
    ap.add_argument("--child-aggs", action="store_true",
                    help="add count / sum / per-category counts of a fact "
                         "table's children to its own columns")
    ap.add_argument("--path-expand", default="",
                    help="`<entity>:<bridge>:<events>` (comma-separated) -- "
                         "file events reachable only through a bridge table "
                         "into the entity's history, e.g. "
                         "`facilities:facilities_studies:outcome_analyses` "
                         "for rel-trial site-success (see "
                         "EventCorpus._expand_path)")
    args = ap.parse_args()

    from relbench.datasets import get_dataset
    ds = get_dataset(args.dataset, download=True)
    splits = (["val", "test"] if args.split == "both" else [args.split])

    db_holder = {}

    def get_db():
        if "db" not in db_holder:
            t0 = time.time()
            db_holder["db"] = ds.get_db()
            print(f"get_db: {time.time() - t0:.0f}s", flush=True)
        return db_holder["db"]

    for split in splits:
        cutoff = pd.Timestamp(getattr(ds, f"{split}_timestamp"))
        c = load_corpus(args.dataset, cutoff, db=get_db,
                        rebuild=args.rebuild,
                        self_rows=args.self_rows,
                        denorm_fk=args.denorm_fk,
                        child_aggs=args.child_aggs,
                        path_expand=args.path_expand)
        n_ent = sum(c.schema.entity_counts.values())
        print(f"  {split} @ {cutoff}: {c.num_events:,} events, "
              f"{n_ent:,} entities, "
              f"fact tables {list(c.schema.fact_tables)}", flush=True)
        for t, spec in c.schema.fact_tables.items():
            derived = [x for x in spec.columns if x.source is not None]
            if derived:
                print(f"    {t}: {len(derived)} derived of "
                      f"{len(spec.columns)} columns", flush=True)
                for x in derived:
                    print(f"      {x.kind[:3]} {x.name}", flush=True)


if __name__ == "__main__":
    _main()
