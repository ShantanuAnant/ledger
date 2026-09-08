"""One place that rebuilds a model from a checkpoint.

Every eval script needs the same thing -- take `ck["args"]` and construct the
LEDGER the run actually trained -- and until now each of them did it with its own
hand-copied kwargs list. Four copies drift, and the drift is the project's
worst failure class because most of it is SILENT: a constructor argument that
does not change any parameter SHAPE (`state_normalize`, `state_momentum`) is
simply not restored, `load_state_dict` succeeds, and the script reports a
number for a model that is not the checkpoint. The loud half is no better --
`scripts/eval_rec.py` never passed `window_head`/`query_feats`/`feat_path`, so
a checkpoint trained with the window objective could not be scored on a
recommendation task at all, which is precisely the cross-category comparison
the architecture exists to make.

So: one function, and the eval scripts call it. Adding a constructor argument
to LEDGER means adding it HERE, once, and every readout picks it up.
"""

from __future__ import annotations

import torch

from ledger.model.ledger import LEDGER


def query_feat_width(ck, corpus) -> int:
    """Width of the query-token feature vector THIS checkpoint was trained
    with, AND pin the corpus schema's layout to reproduce it.

    Read from the saved weight, not recomputed from the schema. The per-
    category counter block (2026-08-25) and the dynamics block (2026-08-26)
    both widened what `PackedBatcher.n_query_feats` returns, so recomputing it
    makes every checkpoint trained before those dates fail its state_dict load
    -- which is how rel-f1, rel-hm and rel-stack silently dropped out of a
    board sweep. The weight is the only record of what the run actually used.

    Loading is not enough: the BATCHER must build a vector of the same width,
    so the per-category layout is emptied when the checkpoint predates it (or
    was trained with --no_query_cat).

    Whether the dynamics block was on is a training-CLI fact and comes from
    `ck["args"]`, never guessed from the width -- two different layouts can
    add up to the same number.
    """
    from ledger.data.batching import PackedBatcher, query_cat_layout
    w = ck["model"].get("query_feat_proj.weight")
    if w is None:
        return 0
    want = int(w.shape[1])
    dyn = bool(ck.get("args", {}).get("query_dyn"))
    nbr = bool(ck.get("args", {}).get("query_nbr"))
    if PackedBatcher.n_query_feats(corpus.schema, dyn, nbr) != want:
        corpus.schema._qcat_layout = []          # pre-per-category checkpoint
        query_cat_layout(corpus.schema)
    got = PackedBatcher.n_query_feats(corpus.schema, dyn, nbr)
    if got != want:
        raise SystemExit(
            f"query-feature width mismatch: checkpoint {want}, this corpus "
            f"{got}. The schema changed under the checkpoint (denorm flags? "
            f"a different cutoff?); rebuild the corpus the run used.")
    return want


def corpus_kwargs(ck) -> dict:
    """The three corpus-construction flags the checkpoint was built with.

    They change which events an entity's history holds and which columns every
    event carries, so a mismatch is a shape error at best and a silently
    different model at worst.
    """
    t = ck["args"]
    return dict(self_rows=bool(t.get("self_rows")),
                denorm_fk=bool(t.get("denorm_fk")),
                child_aggs=bool(t.get("child_aggs")))


def build_model(ck, corpus, device="cpu", strict: bool = True) -> LEDGER:
    """Reconstruct and load the checkpoint's model against `corpus`.

    Mutates `corpus.schema`'s query-feature layout where the checkpoint
    requires it (see `query_feat_width`), so call this BEFORE building any
    batch from the corpus.
    """
    t = ck["args"]
    if t.get("no_query_cat"):
        corpus.schema._qcat_layout = []          # match the training layout
    model = LEDGER(
        corpus.schema,
        dim=t["dim"], layers=t["layers"], heads=t.get("heads", 8),
        states=t["states"],
        state_momentum=t.get("state_momentum", 0.1),
        state_normalize=bool(t.get("state_normalize", False)),
        state_center=bool(t.get("state_center", False)),
        state_reduce=t.get("state_reduce", "mean"),
        learned_max=t.get("learned_max", 200_000),
        # `--learned_tables` overrides the size gate at training time, so
        # rebuilding from `learned_max` alone silently constructs a DIFFERENT
        # model: on rel-stack `posts` (334K) exceeds the 200K gate, so the
        # destination catalogue came back as an EMA table while the checkpoint
        # holds a trained embedding.
        learned_tables=([x.strip() for x in
                         str(t.get("learned_tables") or "").split(",")
                         if x.strip()] or None),
        who_feats=(8 if t.get("who_feats") else 0),
        window=bool(t.get("window")),
        rerank=bool(t.get("rerank")),
        no_ema_write=bool(t.get("no_ema_write")),
        # A window-head checkpoint carries `win.*` and `query_emb`, and the
        # wide-and-deep path adds more; constructing without them fails the
        # strict load, which is what used to stop a window-trained checkpoint
        # being scored on a recommendation task.
        window_head=bool(t.get("window_head")),
        win_buckets=t.get("win_buckets", 8),
        win_balance=t.get("win_balance", "legacy"),
        feat_path=bool(t.get("feat_path")),
        query_feats=query_feat_width(ck, corpus),
        next_event=(t.get("objective", "next_event") != "window"),
        # `log_var.*` are parameters, so a learned-weight checkpoint does not
        # load into a fixed-weight model.
        learn_loss_weights=bool(t.get("learn_loss_weights")),
        lw_clamp=t.get("lw_clamp", 3.0),
        branch_layers=t.get("branch_layers", 0),
        branch_kind=t.get("branch_kind", "transformer"),
        # A --query_head checkpoint carries the extra head, and the rate
        # balance changes the window head's own parameters; `load_state_dict`
        # is strict, so omitting either raises rather than silently
        # evaluating a different model.
        query_head=bool(t.get("query_head")),
        rate_balance=t.get("rate_balance", "none"),
        rate_zero_scale=t.get("rate_zero_scale", 1.0),
    ).to(device)
    model.load_state_dict(ck["model"], strict=strict)
    model.eval()
    return model


def load_checkpoint(path, map_location="cpu"):
    """mmap the checkpoint rather than reading it into anonymous memory.

    The rel-avito checkpoint is 39 GB (a 5.96M-entity AdsInfo embedding), and
    a plain load on top of a couple of live training jobs exceeds the 64 GiB
    cgroup limit -- a SIGKILL with no traceback that can take an unrelated job
    with it. Memory-mapped, the weights live in reclaimable page cache and are
    copied straight to the device by `.to(dev)`.
    """
    return torch.load(path, map_location=map_location, weights_only=False,
                      mmap=True)
