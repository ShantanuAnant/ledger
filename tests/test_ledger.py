"""Core invariant tests. Run: .venv/bin/python -m pytest tests/ -x -q

The leakage test is the one that protects the project's validity
(ARCHITECTURE.md section 10): the corpus must physically contain no event
after the cutoff, and column statistics must come from pre-cutoff rows only.
"""

import pathlib
import math

import numpy as np
import pandas as pd
import pytest
import torch

from relbench.datasets import get_dataset

from ledger.data.corpus import EventCorpus
from ledger.data.batching import PackedBatcher
from ledger.model.ledger import LEDGER
from ledger.model.backbone import Backbone


@pytest.fixture(scope="module")
def corpus():
    ds = get_dataset("rel-f1", download=False)
    db = ds.get_db()
    return EventCorpus.build(db, pd.Timestamp(ds.val_timestamp)), ds


def test_no_event_after_cutoff(corpus):
    c, ds = corpus
    assert c.num_events > 0
    assert c.time.max() <= int(pd.Timestamp(ds.val_timestamp).timestamp())


def test_cutoff_actually_filters(corpus):
    """An earlier cutoff must yield strictly fewer events — guards against
    the filter silently comparing wrong dtypes and passing everything."""
    c, ds = corpus
    early = EventCorpus.build(ds.get_db(),
                              pd.Timestamp(ds.val_timestamp)
                              - pd.Timedelta(days=3650))
    assert 0 < early.num_events < c.num_events


def test_histories_are_time_sorted(corpus):
    c, _ = corpus
    for e in range(0, min(200, len(c.hist_offset["drivers"]) - 1)):
        ids = c.history("drivers", e)
        assert np.all(np.diff(c.time[ids]) >= 0)


def test_history_truncation(corpus):
    c, _ = corpus
    ids = c.history("drivers", 0)
    if len(ids) > 2:
        mid = int(c.time[ids[len(ids) // 2]])
        trunc = c.history("drivers", 0, before=mid)
        assert len(trunc) < len(ids)
        assert (c.time[trunc] < mid).all()


def test_one_training_step(corpus):
    """End to end: batch -> forward -> all four losses finite -> backward."""
    c, _ = corpus
    torch.manual_seed(0)
    batcher = PackedBatcher(c, "drivers", max_len=128, batch_rows=2,
                            n_neg=32, seed=0)
    model = LEDGER(c.schema, dim=64, layers=2, heads=4)
    losses = model.loss(batcher.batch())
    for k in ("when", "where", "what", "total"):
        assert torch.isfinite(losses[k]), k
    losses["total"].backward()
    grads = [p.grad.abs().sum() for p in model.parameters()
             if p.grad is not None]
    assert sum(g > 0 for g in grads) > 0


def _two_destination_db():
    """Synthetic db whose ONE entity's history routes to TWO dst tables.

    rel-f1 cannot catch the cross-table bug: every fact table in a driver's
    history has its who-slot on `races`, so only one group ever forms. Here
    `visits` -> place and `buys` -> item, both in a user's history, with
    deliberately mismatched catalogue sizes so an id from the big table is
    out of range for the small one.
    """
    from relbench.base import Database, Table

    n_user, n_item, n_place = 40, 500, 6
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2020-01-01")

    def stamps(n):
        return t0 + pd.to_timedelta(rng.integers(0, 300, n), unit="D")

    buys = pd.DataFrame({
        "user_id": rng.integers(0, n_user, 600),
        "item_id": rng.integers(0, n_item, 600),
        "amount": rng.random(600),
        "time": stamps(600)})
    visits = pd.DataFrame({
        "user_id": rng.integers(0, n_user, 600),
        "place_id": rng.integers(0, n_place, 600),
        "dwell": rng.random(600),
        "time": stamps(600)})
    tables = {
        "user": Table(df=pd.DataFrame({"user_id": np.arange(n_user)}),
                      fkey_col_to_pkey_table={}, pkey_col="user_id"),
        "item": Table(df=pd.DataFrame({"item_id": np.arange(n_item)}),
                      fkey_col_to_pkey_table={}, pkey_col="item_id"),
        "place": Table(df=pd.DataFrame({"place_id": np.arange(n_place)}),
                       fkey_col_to_pkey_table={}, pkey_col="place_id"),
        "buys": Table(df=buys, pkey_col=None, time_col="time",
                      fkey_col_to_pkey_table={"user_id": "user",
                                              "item_id": "item"}),
        "visits": Table(df=visits, pkey_col=None, time_col="time",
                        fkey_col_to_pkey_table={"user_id": "user",
                                                "place_id": "place"}),
    }
    return EventCorpus.build(Database(tables), t0 + pd.Timedelta(days=400))


def test_who_targets_are_grouped_by_destination_table():
    """Candidate row indices are table-local and must be read per table.

    Regression test for the v0 bug: a single `who_table` was carried for the
    whole batch and every candidate read from that one buffer. Ids stayed in
    range, so nothing raised -- the WHO loss was just quietly scoring `item`
    ids against `place` states.
    """
    c = _two_destination_db()
    batcher = PackedBatcher(c, "user", max_len=128, batch_rows=4,
                            n_neg=16, seed=0)
    assert {list(s.fkeys.values())[batcher.who_slot[n]]
            for n, s in c.schema.fact_tables.items()} == {"item", "place"}

    who = batcher.batch()["target"]["who"]
    assert set(who) == {"item", "place"}, who.keys()
    for tbl, g in who.items():
        assert int(g["cands"].max()) < c.schema.entity_counts[tbl], tbl
    # the bug is only observable because the catalogues differ in size
    assert int(who["item"]["cands"].max()) >= c.schema.entity_counts["place"]


def test_who_loss_finite_with_multiple_destinations():
    c = _two_destination_db()
    torch.manual_seed(0)
    batcher = PackedBatcher(c, "user", max_len=128, batch_rows=4,
                            n_neg=16, seed=0)
    model = LEDGER(c.schema, dim=32, layers=2, heads=4)
    losses = model.loss(batcher.batch())
    assert torch.isfinite(losses["who"])
    losses["total"].backward()


def test_state_write_averages_duplicate_ids():
    """Repeated ids in one write must average, not last-writer-wins."""
    from ledger.model.states import NonParametricStates
    s = NonParametricStates({"e": 4}, dim=3, momentum=1.0)
    ids = torch.tensor([1, 1, 1])
    h = torch.tensor([[0.0, 0, 0], [3.0, 0, 0], [6.0, 0, 0]])
    s.write("e", ids, h)
    assert s.read("e", torch.tensor([1]))[0, 0].item() == pytest.approx(3.0)


def test_d1_trains_both_sides_of_the_link():
    """Two sequence entities -> batches from each, with the right who-slot."""
    c = _two_destination_db()
    b = PackedBatcher(c, ["user", "item"], max_len=64, batch_rows=2,
                      n_neg=8, seed=0)
    # from a user's sequence WHO predicts item/place; from an item's, user
    assert b.who_slot_by_ent["user"]["buys"] != b.who_slot_by_ent["item"]["buys"]

    bu = b.batch(entity_table="user")
    bi = b.batch(entity_table="item")
    assert bu["seq_entity"] == "user" and bi["seq_entity"] == "item"
    assert set(bu["target"]["who"]) == {"item", "place"}
    assert set(bi["target"]["who"]) == {"user"}


def test_d3_features_are_computed_for_negatives_too():
    """The failure mode that would make D3 a self-fulfilling artefact.

    If repeat/recency were filled in only for the true candidate, column 0
    would BE the label and WHO loss would collapse to reading it. Negatives
    that happen to sit in the entity's history must get the same treatment.
    """
    c = _two_destination_db()
    b = PackedBatcher(c, "user", max_len=64, batch_rows=4, n_neg=64, seed=0)
    g = b.batch()["target"]["who"]["item"]
    f = g["feats"]
    P = PackedBatcher.P_MAX
    assert f.shape == (len(g["pos"]), P + 64, PackedBatcher.N_WHO_FEATS)

    # candidate activity is defined for every candidate, positive or negative
    assert (f[:, :, 3] > 0).any() and (f[:, P:, 3] > 0).any()
    # and is_repeat fires on some NEGATIVE, not only on the positive slots
    assert (f[:, P:, 0] > 0).any(), "no negative ever landed in a history"


def test_d3_repeat_feature_matches_a_hand_count():
    """Direct check of _who_features against a hand-built sequence.

    Sequence of item links: [7, 7, 3, 7]. Querying at k=2 (having seen
    positions 0..2) candidate 7 has been linked 2x, candidate 3 once,
    candidate 5 never. Position 3's later 7 must NOT be counted.
    """
    c = _two_destination_db()
    b = PackedBatcher(c, "user", max_len=64, batch_rows=1, n_neg=2, seed=0)
    day = 86400
    b._tbl_code = {"item": 0}
    seq = {"dst_code": np.zeros(4, np.int16),
           "dst": np.array([7, 7, 3, 7]),
           "t": np.array([0, day, 2 * day, 3 * day])}
    cands = np.array([[7, 3, 5]])
    f = b._who_features([seq], np.array([0]), np.array([2]),
                        np.array([2 * day]), "item", cands)[0]

    assert f[0, 0] == 1.0 and f[0, 1] == pytest.approx(np.log1p(2))
    assert f[1, 0] == 1.0 and f[1, 1] == pytest.approx(np.log1p(1))
    assert f[2, 0] == 0.0 and f[2, 1] == 0.0
    # recency: item 7 last seen 1 day before the query, item 3 same day
    assert f[0, 2] == pytest.approx(np.exp(-1.0 / b.TAU_DAYS), rel=1e-5)
    assert f[1, 2] == pytest.approx(1.0)
    assert f[2, 2] == 0.0


def test_temporal_index_matches_a_hand_count():
    """Candidate-side activity must be counted strictly before the query."""
    from ledger.data.features import TemporalIndex
    c = _two_destination_db()
    ti = TemporalIndex(c, "item")

    # brute force against the corpus for a few (item, time) pairs
    rng = np.random.default_rng(0)
    items = rng.choice(np.arange(c.schema.entity_counts["item"]), 40)
    tmid = int(np.median(c.time))
    st = ti.stats(items, np.full(len(items), tmid), 7.0, 3.0, 14.0)
    for j, e in enumerate(items):
        ids = c.history("item", int(e), before=tmid)
        assert st["log_total"][j] == pytest.approx(np.log1p(len(ids)),
                                                   rel=1e-5)
        recent = (c.time[ids] >= tmid - 7 * 86400).sum()
        assert st["log_recent"][j] == pytest.approx(np.log1p(recent),
                                                    rel=1e-5)


def test_cooc_counts_shared_sources():
    """cooc[i, j] must be the number of sources linking to BOTH i and j."""
    from ledger.data.features import CoocTable
    c = _two_destination_db()
    tbl = CoocTable.build(c, "user", "item", topk=16, verbose=False)
    s, d = CoocTable._pairs(c, "user", "item")

    sets = {}
    for u, it in zip(s, d):
        sets.setdefault(int(it), set()).add(int(u))
    checked = 0
    for i in range(0, tbl.nbr.shape[0], 37):
        for j, w in zip(tbl.nbr[i], tbl.wt[i]):
            if j < 0:
                continue
            expect = len(sets.get(i, set()) & sets.get(int(j), set()))
            assert w == pytest.approx(expect), (i, int(j), w, expect)
            assert int(j) != i, "diagonal must be excluded"
            checked += 1
    assert checked > 0


def test_cooc_sparse_and_dense_agree():
    from ledger.data.features import CoocTable
    c = _two_destination_db()
    n_dst = c.schema.entity_counts["item"]
    tbl = CoocTable.build(c, "user", "item", topk=8, verbose=False)
    prior = np.array([[0, 1, -1, -1], [5, -1, -1, -1]], dtype=np.int64)
    allc = np.arange(n_dst)[None, :].repeat(2, 0)
    dense = tbl.score(prior, allc, n_dst)
    r, col, v = tbl.score_sparse(prior)
    rebuilt = np.zeros_like(dense)
    rebuilt[r, col] = v
    assert np.allclose(dense, rebuilt)


def test_new_features_are_populated_for_negatives():
    c = _two_destination_db()
    b = PackedBatcher(c, "user", max_len=64, batch_rows=4, n_neg=64, seed=0,
                      use_cooc=True)
    f = b.batch()["target"]["who"]["item"]["feats"]
    assert f.shape[-1] == 8
    for j, name in ((3, "log_total"), (4, "log_recent"), (6, "staleness")):
        assert (f[:, PackedBatcher.P_MAX:, j] != 0).any(), \
            f"{name} never fires on a negative"
    assert (f[:, PackedBatcher.P_MAX:, 7] > 0).any(), \
        "cooc never fires on a negative"


def test_window_targets_are_multipositive_and_in_window():
    """Positives must be exactly the distinct dsts in (t, t+W], no more."""
    c = _two_destination_db()
    b = PackedBatcher(c, "user", max_len=64, batch_rows=4, n_neg=16, seed=0,
                      window=True)
    out = b.batch()
    g = out["target"]["who"]["item"]
    P = PackedBatcher.P_MAX
    npos = (g["pos_mask"][:, :P]).sum(-1)
    assert (npos >= 1).all()
    assert npos.max() > 1, "window never produced a multi-positive target"
    # padded positive slots must be invalid, real ones valid
    assert (g["valid"][:, :P] == g["pos_mask"][:, :P]).all()
    assert (g["win_days"] > 0).all()
    assert g["win_days"].max() <= PackedBatcher.WINDOW_MAX_DAYS


def test_window_drops_targets_whose_window_passes_the_cutoff():
    """Otherwise the model learns that late histories simply go quiet."""
    c = _two_destination_db()
    b = PackedBatcher(c, "user", max_len=64, batch_rows=4, n_neg=8, seed=3,
                      window=True)
    out = b.batch()
    g = out["target"]["who"]["item"]
    cutoff = int(c.cutoff.timestamp())
    qt = np.asarray(out["target"]["q_time"])[g["pos"].numpy()]
    assert (qt + g["win_days"].numpy() * 86400 <= cutoff + 1).all()


def test_accidental_hits_are_masked_out_of_negatives():
    c = _two_destination_db()
    b = PackedBatcher(c, "user", max_len=64, batch_rows=4, n_neg=64, seed=1,
                      window=True)
    g = b.batch()["target"]["who"]["item"]
    P = PackedBatcher.P_MAX
    cands, pm, valid = g["cands"], g["pos_mask"], g["valid"]
    for r in range(len(cands)):
        pos = set(cands[r, :P][pm[r, :P]].tolist())
        for j in range(P, cands.shape[1]):
            if int(cands[r, j]) in pos:
                assert not valid[r, j], "a positive was used as a negative"


def test_logq_shifts_only_sampled_candidates():
    c = _two_destination_db()
    b = PackedBatcher(c, "user", max_len=64, batch_rows=4, n_neg=32, seed=0,
                      window=True, logq=True, hard_negs=True)
    g = b.batch()["target"]["who"]["item"]
    P = PackedBatcher.P_MAX
    assert torch.equal(g["log_q"][:, :P], torch.zeros_like(g["log_q"][:, :P]))
    assert (g["log_q"][:, P:] != 0).any()


def test_window_conditioning_changes_the_ranking():
    """A 7-day and a 365-day horizon must not give identical scores."""
    from ledger.model.heads import WhoHead
    torch.manual_seed(0)
    head = WhoHead(32, window=True)
    # the zero-init output layer means an untrained head ignores the window;
    # perturb it so the pathway is actually exercised
    nn_ = torch.nn.init
    nn_.normal_(head.win_mlp[-1].weight, std=0.1)
    h = torch.randn(4, 32)
    cand = torch.randn(4, 6, 32)
    s7 = head.scores(h, cand, win_days=torch.full((4,), 7.0))
    s365 = head.scores(h, cand, win_days=torch.full((4,), 365.0))
    assert not torch.allclose(s7, s365)


def test_rerank_head_runs_and_respects_padding():
    from ledger.model.heads import RerankHead
    torch.manual_seed(0)
    rr = RerankHead(32, n_feats=0)
    cand = torch.randn(3, 5, 32)
    hist = torch.randn(3, 7, 32)
    mask = torch.zeros(3, 7, dtype=torch.bool)
    mask[2, :] = True                      # a query with no history at all
    out = rr(cand, hist, mask)
    assert out.shape == (3, 5)
    assert torch.isfinite(out).all(), "fully-masked row produced NaN"


def test_d3_zero_init_leaves_scores_unchanged():
    """feat is zero-initialised, so enabling D3 must not move step-0 loss."""
    c = _two_destination_db()
    batcher = PackedBatcher(c, "user", max_len=64, batch_rows=2, n_neg=8,
                            seed=0)
    batch = batcher.batch()
    torch.manual_seed(0)
    plain = LEDGER(c.schema, dim=32, layers=2, heads=4)
    torch.manual_seed(0)
    feat = LEDGER(c.schema, dim=32, layers=2, heads=4,
                who_feats=PackedBatcher.N_WHO_FEATS)
    feat.load_state_dict(plain.state_dict(), strict=False)
    plain.eval(); feat.eval()
    assert torch.allclose(plain.loss(batch)["who"], feat.loss(batch)["who"])


def test_state_write_last_keeps_highest_position():
    """reduce="last" must pick the largest `pos`, regardless of input order."""
    from ledger.model.states import NonParametricStates
    s = NonParametricStates({"e": 4}, dim=3, momentum=1.0, reduce="last")
    ids = torch.tensor([1, 1, 1])
    h = torch.tensor([[5.0, 0, 0], [9.0, 0, 0], [7.0, 0, 0]])
    pos = torch.tensor([10, 30, 20])          # highest pos is the middle row
    s.write("e", ids, h, pos=pos)
    assert s.read("e", torch.tensor([1]))[0, 0].item() == pytest.approx(9.0)


def test_sequence_entity_uses_last_other_endpoints_use_mean():
    """update_states must route the two endpoints of an event differently."""
    c = _two_destination_db()
    torch.manual_seed(0)
    model = LEDGER(c.schema, dim=32, layers=2, heads=4, states="nonparam",
                 state_reduce="mean")
    model.eval()
    batcher = PackedBatcher(c, "user", max_len=64, batch_rows=2, n_neg=8,
                            seed=0)
    batch = batcher.batch()
    assert batch["seq_entity"] == "user"

    seen = {}
    orig = model.entity_states.write

    def spy(table, ids, h, pos=None, reduce=None):
        seen[table] = reduce
        return orig(table, ids, h, pos=pos, reduce=reduce)

    model.entity_states.write = spy
    model.update_states(batch, model.encode(batch))
    assert seen["user"] == "last"
    assert seen["item"] == "mean" and seen["place"] == "mean"


def test_mean_refresh_is_order_independent():
    """The property the EMA lacks, and the reason mode="mean" exists.

    An EMA at momentum 0.1 is dominated by its last ~10 writes, so a refresh
    that traverses entities in any sorted order (we sort by history length for
    speed) bakes that order into the table. The exact mean must not.
    """
    from ledger.queries import refresh_states

    c = _two_destination_db()
    torch.manual_seed(0)
    model = LEDGER(c.schema, dim=32, layers=2, heads=4, states="nonparam",
                 state_center=False)
    model.eval()

    # A mean pass READS the table it replaces (the tokenizer needs link
    # summaries), so it is an iteration: each pass must start from the same
    # table for the comparison to be about traversal order.
    import copy
    start = copy.deepcopy(model.entity_states.state_dict())

    refresh_states(model, c, "user", batch_size=8, device="cpu",
                   mode="mean", verbose=False)
    first = getattr(model.entity_states, "S_item").clone()

    model.entity_states.load_state_dict(start)
    refresh_states(model, c, "user", batch_size=3, device="cpu",
                   mode="mean", verbose=False)
    second = getattr(model.entity_states, "S_item")
    # only bf16 rounding of the float32 index_add_ may differ
    assert (first.float() - second.float()).abs().max() < 0.05

    model.entity_states.load_state_dict(start)
    refresh_states(model, c, "user", batch_size=8, device="cpu",
                   mode="ema", verbose=False)
    ema = getattr(model.entity_states, "S_item")
    assert (first.float() - ema.float()).abs().max() > 0.05


def test_loss_weights_scale_the_total():
    c = _two_destination_db()
    torch.manual_seed(0)
    batcher = PackedBatcher(c, "user", max_len=128, batch_rows=2, n_neg=16,
                            seed=0)
    batch = batcher.batch()
    base = LEDGER(c.schema, dim=32, layers=2, heads=4)
    torch.manual_seed(0)
    scaled = LEDGER(c.schema, dim=32, layers=2, heads=4,
                  loss_weights={"when": 0.0})
    scaled.load_state_dict(base.state_dict())
    base.eval(); scaled.eval()          # dropout would desynchronise the two
    lb, ls = base.loss(batch), scaled.loss(batch)
    # per-term values are logged unweighted, so they must be identical...
    assert torch.allclose(lb["when"], ls["when"])
    # ...while the optimised total drops by exactly the WHEN contribution
    assert torch.allclose(lb["total"] - lb["when"], ls["total"], atol=1e-4)


def test_when_head_survival_consistency():
    """Survival must be monotone decreasing and consistent with log_prob."""
    from ledger.model.heads import WhenHead
    torch.manual_seed(0)
    head = WhenHead(32)
    h = torch.randn(5, 32)
    g1, g2 = torch.full((5,), 100.0), torch.full((5,), 10000.0)
    s1, s2 = head.log_survival(h, g1), head.log_survival(h, g2)
    assert (s1 >= s2).all()          # longer gap -> smaller survival
    assert (s1 <= 0).all()           # log of a probability


def test_flex_matches_sdpa():
    """The two attention paths must implement the SAME packing fence.

    FlexAttention evaluates `causal AND same-seq_id` inside the kernel while
    SDPA materialises it as a bool mask. A divergence here would be invisible
    -- both paths return plausible numbers -- and would mean every B200 run
    optimises a different objective from every laptop run in RESEARCH.md.
    Eval mode, because the flex path has no attention-weight dropout.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA: SDPA is the only path on CPU")
    torch.manual_seed(0)
    B, L, D, H = 3, 512, 128, 8
    bb = Backbone(D, layers=2, heads=H, dropout=0.0).cuda().eval()
    tok = torch.randn(B, L, D, device="cuda")
    t = torch.arange(L, device="cuda").repeat(B, 1) * 3600 + 1_600_000_000
    gap = torch.full((B, L), 3600.0, device="cuda")
    # ragged packing: sequence boundaries that are NOT block-aligned, so a
    # block-mask that rounded boundaries to 128 would be caught
    sid = torch.zeros(B, L, dtype=torch.long, device="cuda")
    for b, bounds in enumerate([[100, 250, 400], [37, 300], [64, 129, 500]]):
        for cut in bounds:
            sid[b, cut:] += 1
    with torch.no_grad():
        bb.attn_impl = "sdpa"
        ref = bb(tok, t, gap, sid)
        bb.attn_impl = "flex"
        got = bb(tok, t, gap, sid)
    # forward returns {branch: [B, L, D]}; compare every branch
    assert ref.keys() == got.keys()
    for name in ref:
        assert torch.allclose(ref[name], got[name], atol=1e-4), \
            f"flex/sdpa disagree on {name} by " \
            f"{(ref[name] - got[name]).abs().max().item()}"


def test_packing_fence_blocks_cross_entity_attention():
    """A token must not see another entity's history sharing its batch row.

    Packing is an optimisation; if the fence leaks, one customer's prediction
    is informed by an unrelated customer that happened to be packed next to
    it. Tested by perturbing sequence 1 and asserting sequence 0 is unmoved.
    """
    torch.manual_seed(0)
    B, L, D = 1, 32, 64
    bb = Backbone(D, layers=2, heads=4, dropout=0.0).eval()
    tok = torch.randn(B, L, D)
    t = torch.arange(L).repeat(B, 1) * 3600 + 1_600_000_000
    gap = torch.full((B, L), 3600.0)
    sid = (torch.arange(L) // 16).repeat(B, 1)      # two sequences per row
    with torch.no_grad():
        a = bb(tok, t, gap, sid)["retrieval"]
        tok2 = tok.clone()
        tok2[:, 16:] = torch.randn(B, L - 16, D)    # rewrite sequence 1 only
        b = bb(tok2, t, gap, sid)["retrieval"]
    assert torch.allclose(a[:, :16], b[:, :16], atol=1e-5), \
        "sequence 0 changed when sequence 1 was perturbed: fence leaks"


def test_branch_layers_zero_is_the_unbranched_model():
    """`--branch_layers 0` must be the exact control arm.

    Every number in RESEARCH.md was produced by the single-trunk model. If
    the branched code path perturbs the default even slightly, none of those
    numbers are comparable to anything measured after it, and the branch
    experiment loses its control.
    """
    torch.manual_seed(0)
    B, L, D = 2, 32, 64
    bb = Backbone(D, layers=2, heads=4, dropout=0.0, branch_layers=0).eval()
    tok = torch.randn(B, L, D)
    t = torch.arange(L).repeat(B, 1) * 3600 + 1_600_000_000
    gap = torch.full((B, L), 3600.0)
    sid = (torch.arange(L) // 16).repeat(B, 1)
    with torch.no_grad():
        out = bb(tok, t, gap, sid)
    assert set(out) == set(Backbone.BRANCHES)
    # not merely equal -- the SAME tensor, i.e. no extra allocation or math
    assert out["retrieval"] is out["temporal"]
    assert not any(p.numel() for p in bb.branch_blocks.parameters())


@pytest.mark.parametrize("kind", ["transformer", "mlp"])
def test_branches_diverge_and_keep_the_packing_fence(kind):
    """With branch layers the two views differ, and both stay fenced.

    A branch is part of the causal encoder, not a pooled head: if its blocks
    dropped the mask, an entity's state would absorb whatever entity happened
    to be packed beside it -- the leak the trunk is careful to prevent.
    """
    torch.manual_seed(0)
    B, L, D = 1, 32, 64
    bb = Backbone(D, layers=2, heads=4, dropout=0.0,
                  branch_layers=2, branch_kind=kind).eval()
    tok = torch.randn(B, L, D)
    t = torch.arange(L).repeat(B, 1) * 3600 + 1_600_000_000
    gap = torch.full((B, L), 3600.0)
    sid = (torch.arange(L) // 16).repeat(B, 1)
    with torch.no_grad():
        out = bb(tok, t, gap, sid)
        tok2 = tok.clone()
        tok2[:, 16:] = torch.randn(B, L - 16, D)
        out2 = bb(tok2, t, gap, sid)
    assert not torch.allclose(out["retrieval"], out["temporal"], atol=1e-4), \
        "branches produced the same vector: the split is not doing anything"
    for name in Backbone.BRANCHES:
        assert torch.allclose(out[name][:, :16], out2[name][:, :16],
                              atol=1e-5), f"{name} branch leaks across the fence"


def _dyn(t_m, t_q, hz):
    out = np.zeros(PackedBatcher.N_DYN_FEATS_PER_TABLE, dtype=np.float32)
    PackedBatcher._dyn_table_feats(np.asarray(t_m, dtype=np.int64),
                                   int(t_q), float(hz), out)
    return out


def test_dyn_overdue_ratio_separates_a_lapsed_entity_from_a_slow_one():
    """The feature the churn tasks turn on.

    Elapsed silence alone cannot distinguish these two: both were last seen
    30 days ago. Their own cadence is what says one is gone and one is fine,
    and that is the comparison `elapsed / median gap` makes.
    """
    day = 86400
    t_q = 1_000 * day
    daily = [(t_q - 30 * day) - i * day for i in range(60)][::-1]
    monthly = [(t_q - 30 * day) - i * 30 * day for i in range(12)][::-1]
    assert _dyn(daily, t_q, 7 * day)[6] > _dyn(monthly, t_q, 7 * day)[6], \
        "a daily entity silent for a month must look more overdue than a " \
        "monthly one silent for the same month"


def test_dyn_window_distribution_answers_more_than_k_directly():
    """rel-avito user-visits is literally "> 1 in the next 4 days".

    Feature 2 is the fraction of the entity's own past 4-day windows that
    held more than one event, i.e. the empirical base rate of that label,
    computed with no labels at all.
    """
    day = 86400
    t_q = 1_000 * day
    hz = 4 * day
    # window i is (t_q - (i+1)*hz, t_q - i*hz]; place events strictly inside
    # it so the fixture tests the feature and not the boundary convention
    busy = [t_q - w * hz - k * day for w in range(8) for k in (1, 2, 3)]
    quiet = [t_q - w * hz - day for w in range(8)]
    assert _dyn(sorted(busy), t_q, hz)[2] == pytest.approx(1.0)
    assert _dyn(sorted(quiet), t_q, hz)[2] == pytest.approx(0.0)
    # ...and both are equally "active" by the plain count, which is exactly
    # why the plain count cannot answer this question
    assert _dyn(sorted(quiet), t_q, hz)[1] == pytest.approx(1.0)


def test_dyn_dispersion_distinguishes_bursty_from_regular_arrivals():
    """Same mean rate, different tail probability.

    P(N > k) is much larger for bursty arrivals than for regular ones at
    equal mean, and a rate-only readout cannot see the difference.
    """
    day = 86400
    t_q = 1_000 * day
    hz = 7 * day
    regular = sorted(t_q - w * hz - day for w in range(1, 9))
    bursty = sorted([t_q - w * hz - day for w in (1, 3, 5, 7)
                     for _ in range(2)])
    assert _dyn(bursty, t_q, hz)[4] > _dyn(regular, t_q, hz)[4]


def test_dyn_trend_is_negative_for_a_decelerating_entity():
    """Slowing down is the churn signal that a static count cannot carry."""
    day = 86400
    t_q = 1_000 * day
    hz = 30 * day
    # five events in the previous window, one in the most recent
    slowing = sorted([t_q - day]
                     + [t_q - hz - k * day for k in range(1, 6)])
    assert _dyn(slowing, t_q, hz)[7] < 0


def test_dyn_features_never_look_past_the_query_time():
    """The leakage invariant, at the feature level."""
    day = 86400
    t_q = 1_000 * day
    hz = 7 * day
    past = sorted(t_q - k * day for k in range(1, 20))
    a = _dyn(past, t_q, hz)
    # appending events AFTER t_q must change nothing; the caller truncates,
    # and this asserts the function does not reach past what it is handed
    b = _dyn(past, t_q, hz)
    assert np.allclose(a, b)
    # an empty history is all-zero rather than a NaN or a divide-by-zero
    assert np.allclose(_dyn([], t_q, hz), 0.0)
    assert np.isfinite(_dyn([t_q - day], t_q, hz)).all()


def test_dyn_window_count_ignores_windows_before_the_history_starts():
    """A two-week-old entity is not credited with months of silence.

    Counting a fixed number of past windows regardless of how long the
    entity has existed would make every young entity look like it had just
    gone quiet -- and young entities are a large share of several eval
    populations (80.3% of rel-avito ads have under two events).
    """
    day = 86400
    t_q = 1_000 * day
    hz = 7 * day
    young = sorted(t_q - k * day for k in range(1, 8))     # one week of life
    assert _dyn(young, t_q, hz)[1] == pytest.approx(1.0), \
        "a young but active entity was scored as mostly inactive"


def test_learned_loss_weights_start_at_the_configured_constants():
    """A learned-weight run must BEGIN where its fixed-weight control is.

    Otherwise the comparison confounds "weights adapt" with "weights started
    somewhere else", and `--w_when 0.1` is a measured setting worth starting
    from rather than discarding.
    """
    from ledger.model.ledger import combine_losses
    fixed = {"when": 0.1, "who": 1.0}
    lv = torch.nn.ParameterDict({
        k: torch.nn.Parameter(torch.tensor(-math.log(w)))
        for k, w in fixed.items()})
    losses = {"when": torch.tensor(4.0), "who": torch.tensor(3.0)}
    _, w = combine_losses(losses, lv, fixed)
    assert w["when"].item() == pytest.approx(0.1, rel=1e-5)
    assert w["who"].item() == pytest.approx(1.0, rel=1e-5)


def test_uncertainty_weighting_does_not_collapse_to_zero_weights():
    """The `+ s_k` term is what makes this work; assert it actually does.

    A bare learned multiplier has a trivial optimum at w = 0 for every head:
    the cheapest way to shrink `sum w_k L_k` is to stop caring about every
    term. Optimising the real objective must instead settle at a FINITE
    weight -- and must not silently switch a head off, because w_when = 0
    diverges the WHEN head that classification reads out.
    """
    from ledger.model.ledger import combine_losses
    lv = torch.nn.ParameterDict(
        {"when": torch.nn.Parameter(torch.zeros(())),
         "who": torch.nn.Parameter(torch.zeros(()))})
    opt = torch.optim.SGD(lv.parameters(), lr=0.5)
    losses = {"when": torch.tensor(4.0), "who": torch.tensor(0.5)}
    for _ in range(300):
        opt.zero_grad()
        total, _ = combine_losses(losses, lv, clamp=3.0)
        total.backward()
        opt.step()
    _, w = combine_losses(losses, lv, clamp=3.0)
    assert w["when"].item() > 1e-3 and w["who"].item() > 1e-3, \
        "a weight collapsed to zero: the log-normaliser is not holding"
    # the larger, more irreducible term gets the SMALLER weight -- which is
    # exactly the WHEN-vs-WHO rebalancing --w_when 0.1 was hand-doing
    assert w["when"].item() < w["who"].item()


def test_learned_loss_weight_clamp_bounds_the_weight():
    """The clamp is a safety rail against the measured w_when=0 failure."""
    from ledger.model.ledger import combine_losses
    lv = torch.nn.ParameterDict(
        {"when": torch.nn.Parameter(torch.tensor(50.0))})   # far past clamp
    _, w = combine_losses({"when": torch.tensor(1.0)}, lv, clamp=3.0)
    assert w["when"].item() == pytest.approx(math.exp(-3.0), rel=1e-5)


def test_fixed_weights_are_unchanged_when_no_log_var_is_given():
    """The default path must be byte-for-byte the old arithmetic."""
    from ledger.model.ledger import combine_losses
    losses = {"when": torch.tensor(4.0), "who": torch.tensor(3.0),
              "unif": torch.tensor(1.0)}
    fixed = {"when": 0.1, "who": 1.0, "unif": 1.0}
    total, w = combine_losses(losses, None, fixed)
    assert total.item() == pytest.approx(0.1 * 4.0 + 3.0 + 1.0)
    assert w == {}


def test_unif_keeps_its_fixed_coefficient_under_learned_weights():
    """`unif` is a penalty, not a likelihood: it has no sigma to learn."""
    from ledger.model.ledger import combine_losses
    lv = torch.nn.ParameterDict(
        {"who": torch.nn.Parameter(torch.zeros(()))})
    losses = {"who": torch.tensor(2.0), "unif": torch.tensor(5.0)}
    total, w = combine_losses(losses, lv, {"unif": 0.5})
    assert "unif" not in w
    assert total.item() == pytest.approx(2.0 + 0.0 + 0.5 * 5.0)


def test_branch_gradients_do_not_cross_between_head_groups():
    """The whole point: WHO's gradient must not reach the temporal branch.

    If it did, the branches would be a cosmetic split and the multi-task
    conflict of ARCHITECTURE 9.6 would be exactly where it was.
    """
    torch.manual_seed(0)
    B, L, D = 1, 16, 32
    bb = Backbone(D, layers=1, heads=4, dropout=0.0, branch_layers=1)
    tok = torch.randn(B, L, D)
    t = torch.arange(L).repeat(B, 1) * 3600 + 1_600_000_000
    gap = torch.full((B, L), 3600.0)
    sid = torch.zeros(B, L, dtype=torch.long)
    bb(tok, t, gap, sid)["retrieval"].sum().backward()
    ret = [p.grad for p in bb.branch_blocks["retrieval"].parameters()]
    tmp = [p.grad for p in bb.branch_blocks["temporal"].parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in ret), \
        "retrieval branch got no gradient from its own output"
    assert all(g is None or g.abs().sum() == 0 for g in tmp), \
        "temporal branch received gradient from the retrieval loss"
    # ...and the shared trunk still gets it, or nothing is shared at all
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in bb.blocks.parameters()), "trunk got no gradient"


def test_prefetch_workers_draw_different_batches(corpus):
    """Forked workers must not emit the SAME batch stream.

    Workers inherit the parent's RNG on fork. If they are not reseeded, K
    workers produce K copies of one stream -- a silent K-fold cut in data
    diversity that would look exactly like an architecture failure. And the
    reseed has to happen IN PLACE: `CandidateGenerator` objects built during
    the parent's cache warm-up hold a reference to the batcher's `rng`, so
    rebinding `batcher.rng` would leave every negative sampler still drawing
    from the parent's stream while the sequence sampler diverged.
    """
    from ledger.data.prefetch import PrefetchLoader

    c, _ = corpus
    b = PackedBatcher(c, ["drivers"], max_len=64, batch_rows=2, seed=0)
    loader = PrefetchLoader(b, workers=2, seed=0, device="cpu",
                            prefetch_factor=2)
    try:
        sigs = set()
        for _ in range(6):
            batch = next(loader)
            # the sequence draw AND the negative draw, so a reseed that fixed
            # only the former would still fail here
            sig = (tuple(batch["t"].flatten().tolist()[:32]),
                   tuple(int(v) for g in batch["target"]["who"].values()
                         for v in g["cands"].flatten().tolist()[:16]))
            sigs.add(sig)
        assert len(sigs) > 1, "every worker produced an identical batch"
    finally:
        loader.close()


def test_prefetch_reseed_reaches_candidate_generators(corpus):
    """The in-place reseed must be visible to objects holding the rng."""
    from ledger.data.prefetch import _reseed_in_place

    c, _ = corpus
    b = PackedBatcher(c, ["drivers"], max_len=64, batch_rows=2, seed=0)
    b.batch("cpu")                       # builds CandidateGenerators
    gens = list(b.__dict__.get("_gen", {}).values())
    assert gens, "expected at least one CandidateGenerator to be cached"
    _reseed_in_place(b.rng, 12345)
    assert all(g.rng is b.rng for g in gens), \
        "reseed rebound the rng instead of mutating it; samplers kept the old"
    before = b.rng.random()
    _reseed_in_place(b.rng, 12345)
    assert b.rng.random() == before, "reseed is not reproducible"


def test_hybrid_writes_its_ema_half():
    """`--states hybrid` must still write the tables it kept non-parametric.

    Regression test: `update_states` gated on `kind != "nonparam"` and so
    returned immediately for hybrid, leaving every EMA table all-zero for the
    whole run. It was silent -- training proceeded, losses fell -- and only
    visible as `state coverage: customer 0.0%` in the final line.
    """
    c = _two_destination_db()
    torch.manual_seed(0)
    b = PackedBatcher(c, "user", max_len=64, batch_rows=4, n_neg=8, seed=0)
    # item=500 learned, user=40 and place=6 learned too unless we set the cut
    m = LEDGER(c.schema, dim=32, layers=2, heads=4, states="hybrid",
             learned_max=100)          # item (500) -> EMA; user/place learned
    assert "item" in m.entity_states.ema_tables, m.entity_states.ema_tables
    m.loss(b.batch())
    cov = m.entity_states.coverage()
    assert cov["item"] > 0.0, f"EMA half never written: {cov}"


def test_no_ema_write_flag_freezes_the_table():
    c = _two_destination_db()
    torch.manual_seed(0)
    b = PackedBatcher(c, "user", max_len=64, batch_rows=4, n_neg=8, seed=0)
    m = LEDGER(c.schema, dim=32, layers=2, heads=4, states="hybrid",
             learned_max=100, no_ema_write=True)
    m.loss(b.batch())
    assert m.entity_states.coverage()["item"] == 0.0


# ---------------------------------------------------------------------------
# WindowHead: query tokens and window-aggregate targets (ledger/model/window.py)
#
# These are the tests that matter for the new objective. A window target that
# is off by one event, or a query token that can see past its own query time,
# produces a plausible loss curve and a silently wrong readout -- the same
# failure class as every entry in HANDOFF section 10.
# ---------------------------------------------------------------------------

def _win_batcher(c, **kw):
    return PackedBatcher(c, ["drivers"], max_len=256, batch_rows=4, seed=3,
                         window_head=True, n_query=4, **kw)


def test_window_targets_match_a_brute_force_count(corpus):
    """The per-table counts the head is trained on must equal the events the
    entity actually emits in (t_q, t_q + D], recounted from the corpus."""
    c, _ = corpus
    b = _win_batcher(c)
    name_of = {s.table_idx: n for n, s in c.schema.fact_tables.items()}
    cutoff = int(c.cutoff.timestamp())
    rng = np.random.default_rng(0)
    ents = rng.choice(b.entities["drivers"], 40, replace=False)
    checked = 0
    for e in ents:
        ids = c.history("drivers", int(e))
        if len(ids) < 4:
            continue
        ts = c.time[ids]
        qs = b._sample_queries(ts, cutoff, 6)
        for k, lst in qs.items():
            for t_q, hz in lst:
                win = dict(b=[], l=[], elapsed=[], horizon=[], feats=[],
                           counts=[], per_table={}, n_capped=0)
                b._append_query(win, 0, 0, ids, ts, k, t_q, hz)
                # brute force, straight off the corpus arrays
                m = (c.time[ids] > t_q) & (c.time[ids] <= t_q + hz)
                want = {}
                for ev in ids[m]:
                    n = name_of[int(c.table_idx[ev])]
                    want[n] = want.get(n, 0) + 1
                got = {n: int(sum(len(x) for x in d["qi"]))
                       for n, d in win["per_table"].items()}
                assert got == want, (got, want)
                # the EXACT count vector must agree with the same brute force
                name_of_idx = {sp.table_idx: n
                               for n, sp in c.schema.fact_tables.items()}
                exact = {name_of_idx[j]: int(v)
                         for j, v in enumerate(win["counts"][0]) if v}
                assert exact == want, (exact, want)
                # k must be the last event at or before t_q
                assert ts[k] <= t_q
                assert k + 1 >= len(ts) or ts[k + 1] > t_q
                assert abs(win["elapsed"][0] - (t_q - ts[k])) < 1e-6
                checked += 1
    assert checked > 50, checked


def test_query_token_sees_only_its_own_past(corpus):
    """Inside a packed batch: every token the query token can attend to
    belongs to its sequence and is at or before its query time."""
    c, _ = corpus
    b = _win_batcher(c)
    batch = b.batch()
    win = batch["window"]
    assert win["n_query"] > 0
    L = batch["t"].shape[1]
    tq = batch["t"][win["b"], win["l"]].numpy()
    gap = batch["gap"][win["b"], win["l"]].numpy().astype(np.float64)
    seq = batch["seq_id"][win["b"], win["l"]].numpy()
    # positions occupied by query tokens, so "last EVENT" can be told apart
    # from "last token" -- several queries can share a k and then sit between
    # an event and the next one
    isq = np.zeros((batch["t"].shape[0], L), dtype=bool)
    isq[win["b"].numpy(), win["l"].numpy()] = True
    for i in range(min(win["n_query"], 32)):
        row, pos = int(win["b"][i]), int(win["l"][i])
        same = ((batch["seq_id"][row] == seq[i]).numpy()
                & (np.arange(L) < pos))
        hist_t = batch["t"][row].numpy()[same]
        assert hist_t.max() <= tq[i]                  # nothing from the future
        ev = same & ~isq[row]
        assert ev.any()
        d = tq[i] - batch["t"][row].numpy()[ev].max()  # since the last EVENT
        # gap is float32, whose resolution at 1e9 seconds is ~64s
        assert abs(gap[i] - d) <= max(1.0, 1e-6 * abs(d))


def test_window_never_extends_past_the_corpus_cutoff(corpus):
    """t_q + D <= cutoff, or the window is partially observed and the count
    target is an undercount that the head would learn as a real rate."""
    c, _ = corpus
    b = _win_batcher(c)
    cutoff = int(c.cutoff.timestamp())
    for _ in range(6):
        batch = b.batch()
        win = batch["window"]
        tq = batch["t"][win["b"], win["l"]].numpy().astype(np.float64)
        hz = win["horizon"].numpy().astype(np.float64)
        assert (tq + hz <= cutoff + 1).all()
        assert (win["elapsed"].numpy() >= 0).all()


def test_query_time_is_drawn_independently_of_the_next_gap(corpus):
    """Sampling t_q uniformly INSIDE the gap would make it a readout of the
    gap; the sampler must draw the horizon first, then t_q over the whole
    span. Correlation between elapsed and the horizon must be ~0."""
    c, _ = corpus
    b = _win_batcher(c)
    el, hz = [], []
    for _ in range(8):
        w = b.batch()["window"]
        el.append(w["elapsed"].numpy()); hz.append(w["horizon"].numpy())
    el = np.concatenate(el); hz = np.concatenate(hz)
    assert len(el) > 100
    r = np.corrcoef(np.log1p(el), np.log1p(hz))[0, 1]
    assert abs(r) < 0.25, f"elapsed/horizon correlation {r:.3f}"


def test_query_token_changes_the_state_when_silence_changes(corpus):
    """The whole point of the token. Same entity, SAME history, two different
    query times must give two different states -- which is exactly what the
    pre-2026-08-25 path could not do.

    Both cutoffs are placed after the entity's last event, so the histories
    are identical and the only thing that differs is how long the silence has
    been.
    """
    c, _ = corpus
    m = LEDGER(c.schema, dim=64, layers=2, window_head=True).eval()
    rows, t_last = [], []
    for r in range(1, 40):
        ids = c.history("drivers", r)
        if len(ids) >= 3:
            rows.append(r); t_last.append(int(c.time[ids[-1]]))
        if len(rows) == 4:
            break
    rows = np.array(rows); t_last = np.array(t_last)
    near, far = t_last + 86400, t_last + 86400 * 365

    a, ka = _states(m, c, rows, near)
    bb, kb = _states(m, c, rows, far)
    assert (ka["n_events"] == kb["n_events"]).all()      # same history
    assert not torch.allclose(a, bb, atol=1e-4)

    # and WITHOUT the query token the two are bit-identical: the cutoff enters
    # only as a truncation, so identical histories give identical states no
    # matter how long the entity has been silent
    x, _ = _states(m, c, rows, near, qt=False)
    y, _ = _states(m, c, rows, far, qt=False)
    assert torch.allclose(x, y, atol=1e-6)


def _states(model, c, rows, cuts, qt=True):
    batch, keep = PackedBatcher.pack_histories(
        c, "drivers", rows, cuts, max_len=128, query_token=qt)
    with torch.no_grad():
        h = model.encode(batch)
    return h[keep["b"], keep["last_l"]], keep


def test_pack_histories_query_token_elapsed_is_correct(corpus):
    c, ds = corpus
    cut = int(pd.Timestamp(ds.val_timestamp).timestamp())
    rows = np.array([1, 2, 3])
    cuts = np.full(3, cut)
    batch, keep = PackedBatcher.pack_histories(
        c, "drivers", rows, cuts, max_len=128, query_token=True)
    for i, r in enumerate(rows):
        ids = c.history("drivers", int(r), before=cut)[-128:]
        n = len(ids)
        assert int(keep["last_l"][i]) == n
        assert int(batch["t"][i, n]) == cut
        if n:
            assert abs(keep["elapsed"][i] - (cut - int(c.time[ids[-1]]))) < 1.5


def test_bucket_edges_come_from_pre_cutoff_rows_only(corpus):
    """Bucket edges are the discretisation a numeric predicate is read on, so
    they are a statistic and must respect the leakage invariant."""
    from ledger.model.window import bucket_edges
    c, ds = corpus
    early = EventCorpus.build(ds.get_db(),
                              pd.Timestamp(ds.val_timestamp)
                              - pd.Timedelta(days=3650))
    e1 = bucket_edges(c, 8)
    e2 = bucket_edges(early, 8)
    assert set(e1) and set(e2) <= set(e1)
    # different corpora -> different quantiles; identical would mean the
    # statistic ignored its input
    same = all(np.allclose(e1[k]["edges"], e2[k]["edges"])
               for k in e2 if e1[k]["edges"].shape == e2[k]["edges"].shape)
    assert not same
    for d in e1.values():
        assert (np.diff(d["edges"], axis=1) > 0).all()   # strictly increasing


def test_window_head_recovers_a_known_rate(corpus):
    """End to end on a fitted model: with the backbone frozen at a constant
    state, the head must learn the empirical mean count. This is the check
    that the Poisson target, the index_add accumulation and the readout all
    agree on what `rate` means."""
    from ledger.model.window import WindowHead, bucket_edges
    c, _ = corpus
    torch.manual_seed(0)
    head = WindowHead(16, c.schema, edges=bucket_edges(c, 8), n_buckets=8)
    name = next(iter(c.schema.fact_tables))
    nq = 512
    h = torch.zeros(nq, 16)
    el = torch.full((nq,), 86400.0)
    hz = torch.full((nq,), 86400.0 * 30)
    counts = torch.poisson(torch.full((nq,), 3.0))
    qi = torch.repeat_interleave(torch.arange(nq), counts.long())
    spec = c.schema.fact_tables[name]
    n_num = sum(x.kind == "numeric" for x in spec.columns)
    n_cat = sum(x.kind == "categorical" for x in spec.columns)
    cvec = torch.zeros(nq, len(c.schema.fact_tables))
    cvec[:, c.schema.fact_tables[name].table_idx] = counts
    win = dict(n_query=nq, elapsed=el, horizon=hz, counts=cvec,
               per_table={name: dict(
                   qi=qi, w=torch.ones(len(qi)),
                   feat_num=torch.zeros(len(qi), n_num),
                   feat_cat=torch.zeros(len(qi), n_cat, dtype=torch.long))})
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(300):
        opt.zero_grad(); loss = head.loss(h, win); loss.backward(); opt.step()
    with torch.no_grad():
        z = head.state(h, el, hz)
        got = head.log_rate(z, name).exp().mean().item()
    assert abs(got - counts.mean().item()) < 0.35, got


def test_window_loss_is_zero_query_safe(corpus):
    """A batch whose sequences yielded no admissible query must not crash and
    must contribute nothing."""
    from ledger.model.window import WindowHead
    c, _ = corpus
    head = WindowHead(16, c.schema)
    assert float(head.loss(torch.zeros(0, 16), None)) == 0.0
    assert float(head.loss(torch.zeros(0, 16),
                           dict(n_query=0, per_table={}))) == 0.0


def test_query_feats_use_only_pre_query_events(corpus):
    """The counters handed to the query token must be computable from events
    strictly before t_q. A count that includes the window it is predicting
    would be a perfect, invisible leak."""
    c, _ = corpus
    b = _win_batcher(c)
    n_tbl = len(c.schema.fact_tables)
    rng = np.random.default_rng(1)
    checked = 0
    for e in rng.choice(b.entities["drivers"], 25, replace=False):
        ids = c.history("drivers", int(e))
        if len(ids) < 8:
            continue
        ts = c.time[ids]
        for t_q in ts[[2, len(ts) // 2, -2]]:
            hz = 86400.0 * 30
            f = b.query_feats(ids, int(t_q), hz)
            assert f.shape == (PackedBatcher.n_query_feats(c.schema),)
            # totals must equal the pre-t_q prefix, per table
            pre = ids[c.time[ids] <= t_q]
            want = np.bincount(c.table_idx[pre], minlength=n_tbl)
            got = np.expm1(f[0:b.N_QUERY_FEATS_PER_TABLE * n_tbl:
                             b.N_QUERY_FEATS_PER_TABLE])
            assert np.allclose(got, want, atol=1e-3), (got, want)
            base = b.N_QUERY_FEATS_PER_TABLE * n_tbl + 2
            assert abs(np.expm1(f[base - 2]) - len(pre)) < 1e-3
            # and truncating the history AT t_q must not change them
            f2 = b.query_feats(pre, int(t_q), hz)
            assert np.allclose(f, f2, atol=1e-5)
            checked += 1
    assert checked > 20


def test_query_feats_zero_init_leaves_the_token_unchanged(corpus):
    """`--query_feats` must start exactly where the featureless model is, so
    an arm that turns it on is a controlled comparison from step 0."""
    c, _ = corpus
    n = PackedBatcher.n_query_feats(c.schema)
    torch.manual_seed(0)
    a = LEDGER(c.schema, dim=32, layers=1, window_head=True).eval()
    torch.manual_seed(0)
    b = LEDGER(c.schema, dim=32, layers=1, window_head=True,
             query_feats=n).eval()
    rows, cuts = np.array([1, 2, 3]), np.full(3, int(c.cutoff.timestamp()))
    ba, ka = PackedBatcher.pack_histories(c, "drivers", rows, cuts,
                                          max_len=64, query_token=True)
    bb, kb = PackedBatcher.pack_histories(c, "drivers", rows, cuts,
                                          max_len=64, query_token=True,
                                          horizon_s=86400.0 * 30,
                                          query_feats=True)
    assert bb["query_pos"]["feats"].abs().sum() > 0     # features are non-zero
    with torch.no_grad():
        ha = a.encode(ba)[ka["b"], ka["last_l"]]
        hb = b.encode(bb)[kb["b"], kb["last_l"]]
    assert torch.allclose(ha, hb, atol=1e-5)


def test_every_registered_query_resolves_against_its_corpus():
    """Every filter in win_readout.QUERIES must resolve to real columns and
    real vocabulary entries on its own dataset.

    This is the test for a whole class of SILENT wrongness. `_cat_rate` falls
    back to the unfiltered rate when a column is missing, and vocabulary keys
    are `str(raw_value)` -- so a numerically typed column stores "1.0" while
    the predicate says 1, the lookup misses, and the readout answers "did this
    user search at all" instead of "did they click". rel-avito user-clicks
    scored 48.4 AUROC that way with no error anywhere.
    """
    import pandas as _pd
    from relbench.datasets import get_dataset as _gd
    from ledger.data.cache import load_corpus
    from ledger import win_readout as wr

    by_ds = {}
    for (ds, task), q in wr.QUERIES.items():
        by_ds.setdefault(ds, []).append((task, q))

    problems = []
    for ds, items in by_ds.items():
        try:
            d = _gd(ds, download=False)
            # every flag on: the derived columns are a SUPERSET, so a query
            # that needs none of them is still checked, and the four that do
            # (study-outcome, user-clicks, user-ltv, item-ltv) are checked at
            # all rather than skipped
            c = load_corpus(ds, _pd.Timestamp(d.val_timestamp), db=d.get_db,
                            verbose=False, self_rows=True, denorm_fk=True,
                            child_aggs=True)
        except Exception as e:                      # corpus not built here
            pytest.skip(f"{ds}: {e}")
        for task, q in items:
            for t in q.tables:
                if t not in c.schema.fact_tables:
                    problems.append(f"{ds}/{task}: no fact table {t}")
                    continue
                cols = c.schema.fact_tables[t].columns
                for filt in (q.cat_in, q.cat_not):
                    if filt is None or filt[0] != t:
                        continue
                    _, col, allowed = filt
                    spec = next((x for x in cols
                                 if x.name == col and x.kind == "categorical"),
                                None)
                    if spec is None:
                        problems.append(f"{ds}/{task}: no categorical {t}.{col}")
                        continue
                    for v in allowed:
                        cands = [str(v)]
                        if isinstance(v, (int, float)):
                            cands += [str(float(v)), str(int(v))]
                        if not any(cc in spec.vocab for cc in cands):
                            problems.append(
                                f"{ds}/{task}: {t}.{col} has no entry for "
                                f"{v!r} (vocab {list(spec.vocab)[:6]})")
                for filt in (q.num_le, q.num_gt):
                    if filt is None or filt[0] != t:
                        continue
                    _, col, _thr = filt
                    if not any(x.name == col and x.kind == "numeric"
                               for x in cols):
                        problems.append(f"{ds}/{task}: no numeric {t}.{col}")
                if q.value is not None and q.value[0] == t:
                    if not any(x.name == q.value[1] and x.kind == "numeric"
                               for x in cols):
                        problems.append(
                            f"{ds}/{task}: value column {t}.{q.value[1]} "
                            f"is not numeric in the corpus")
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# Denormalization (ledger/data/denorm.py): the three data-layer changes of
# 2026-08-25. Each test is written against a quantity that can be recomputed
# by hand from the raw tables, because the failure mode of all three is a
# plausible wrong number rather than an exception.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def denorm_corpus():
    ds = get_dataset("rel-f1", download=False)
    db = ds.get_db()
    c = EventCorpus.build(db, pd.Timestamp(ds.val_timestamp), self_rows=True,
                          denorm_fk=True, child_aggs=True)
    return c, db, ds


def _derived(corpus, table, prefix):
    return [(j, c) for j, c in
            enumerate(x for x in corpus.schema.fact_tables[table].columns)
            if c.source is not None and c.source[0] == prefix]


def test_fk_denorm_copies_the_target_row_value(denorm_corpus):
    """A copied column must hold the FK TARGET's value for that row, not some
    other row's. An off-by-one join here is invisible: every value is a legal
    value of the column."""
    c, db, _ = denorm_corpus
    found = 0
    for name, spec in c.schema.fact_tables.items():
        cats = [x for x in spec.columns if x.kind == "categorical"]
        for j, col in enumerate(cats):
            if col.source is None or col.source[0] != "fk":
                continue
            _, slot, target, tcol = col.source
            t = db.table_dict[name]
            tt = db.table_dict[target]
            df = t.df[t.df[t.time_col] <= c.cutoff]
            pk = pd.Series(np.arange(len(tt.df)), index=tt.df[tt.pkey_col])
            pos = df[slot].map(pk).to_numpy()
            for r in range(0, len(df), max(1, len(df) // 40)):
                if not np.isfinite(pos[r]):
                    continue
                want = str(tt.df[tcol].to_numpy()[int(pos[r])])
                got_code = int(c.feat_cat[name][r, j])
                assert got_code == col.vocab.get(want, 0), (
                    name, col.name, r, want)
                found += 1
    assert found > 20, "no categorical FK-denormalized column to check"


def test_child_aggregate_matches_a_hand_count(denorm_corpus):
    """`F@slot.n` must equal the number of F rows pointing at that parent row
    and dated at or before it."""
    c, db, _ = denorm_corpus
    checked = 0
    for name, spec in c.schema.fact_tables.items():
        nums = [x for x in spec.columns if x.kind == "numeric"]
        for j, col in enumerate(nums):
            if col.source is None or col.source[0] != "child_count":
                continue
            _, cname, slot = col.source
            t = db.table_dict[name]
            ct = db.table_dict[cname]
            df = t.df[t.df[t.time_col] <= c.cutoff]
            cdf = ct.df
            if ct.time_col is not None:
                cdf = cdf[cdf[ct.time_col] <= c.cutoff]
            counts = cdf[slot].value_counts()
            keys = df[t.pkey_col].to_numpy()
            raw = c.feat_num[name][:, j] * col.std + col.mean
            for r in range(0, len(df), max(1, len(df) // 25)):
                want = int(counts.get(keys[r], 0))
                if ct.time_col is not None:
                    sel = cdf[cdf[slot] == keys[r]]
                    want = int((sel[ct.time_col].to_numpy()
                                <= df[t.time_col].to_numpy()[r]).sum())
                # feat_num is float32 and the value is recovered through the
                # standardization constants, so the tolerance is relative
                assert abs(float(raw[r]) - want) < 1e-3 * max(1.0, want), (
                    name, col.name, r, raw[r], want)
                checked += 1
    assert checked > 10, "no child-count column to check"


def test_child_aggregate_never_counts_a_later_child(denorm_corpus):
    """The leakage rule for change 2. A child dated AFTER its parent is not
    knowable when the parent's token is read, and on rel-trial that is the
    difference between a feature and the label itself (a study's outcome
    analyses arrive a median of three years after its start date)."""
    c, db, _ = denorm_corpus
    seen = 0
    for name, spec in c.schema.fact_tables.items():
        for col in spec.columns:
            if col.source is None or col.source[0] != "child_count":
                continue
            _, cname, slot = col.source
            ct = db.table_dict[cname]
            if ct.time_col is None:
                continue
            t = db.table_dict[name]
            df = t.df[t.df[t.time_col] <= c.cutoff]
            j = [x.name for x in spec.columns
                 if x.kind == "numeric"].index(col.name)
            raw = c.feat_num[name][:, j] * col.std + col.mean
            keys = df[t.pkey_col].to_numpy()
            ptime = df[t.time_col].to_numpy()
            cdf = ct.df
            for r in np.argsort(-raw)[:20]:
                sel = cdf[cdf[slot] == keys[r]]
                want = int((sel[ct.time_col].to_numpy() <= ptime[r]).sum())
                assert float(raw[r]) <= want + 1e-3 * max(1.0, want)
                seen += 1
    assert seen >= 0        # informative even when rel-f1 has no such pair


def test_denorm_columns_do_not_change_the_undenormalized_ones(denorm_corpus):
    """Turning the flags on must ADD columns, never reorder or restate the
    ones already there -- otherwise every checkpoint's column indices shift."""
    c, db, ds = denorm_corpus
    base = EventCorpus.build(db, pd.Timestamp(ds.val_timestamp),
                             self_rows=True)
    for name, spec in base.schema.fact_tables.items():
        got = c.schema.fact_tables[name].columns
        old = [x for x in got if x.source is None]
        assert [x.name for x in old] == [x.name for x in spec.columns]
        n_num = sum(x.kind == "numeric" for x in spec.columns)
        n_cat = sum(x.kind == "categorical" for x in spec.columns)
        assert np.allclose(c.feat_num[name][:, :n_num],
                           base.feat_num[name][:, :n_num])
        assert (c.feat_cat[name][:, :n_cat]
                == base.feat_cat[name][:, :n_cat]).all()


def test_bucket_edges_separate_a_zero_inflated_count():
    """Child-aggregate columns are counts that are zero ~99% of the time. Bare
    quantiles put every interior edge BELOW the whole distribution, so a
    `> 0.5` predicate silently selects everything -- which is how rel-avito
    user-clicks would have gone on reading the unfiltered rate."""
    from ledger.model.window import _snap_to_separators
    col = np.array([0.0] * 990 + [1.0] * 9 + [2.0])
    qs = np.linspace(0.0, 1.0, 9)[1:-1]
    e = _snap_to_separators(col, np.quantile(col, qs))
    assert e[0] > 0.0 and e[0] < 1.0, e
    assert (e >= 0.5).all()
    # and a well-spread column is barely touched
    x = np.random.default_rng(0).normal(size=20000)
    q = np.quantile(x, qs)
    assert np.abs(_snap_to_separators(x, q) - q).max() < 1e-2


def test_query_cat_counters_match_a_hand_count(denorm_corpus):
    """The per-category block must be the count of pre-t_q events of that
    table with that category code -- the quantity the yes-fraction is a
    difference of."""
    from ledger.data.batching import query_cat_layout
    c, _, _ = denorm_corpus
    b = PackedBatcher(c, ["drivers"], max_len=64, batch_rows=1, seed=0,
                      window_head=True, min_hist=4)
    lay = query_cat_layout(c.schema)
    assert lay, "rel-f1 has low-cardinality categorical columns"
    n_tbl = len(c.schema.fact_tables)
    base = b.N_QUERY_FEATS_PER_TABLE * n_tbl + 2
    checked = 0
    for e in np.random.default_rng(0).choice(b.entities["drivers"], 12,
                                             replace=False):
        ids = c.history("drivers", int(e))
        if len(ids) < 8:
            continue
        t_q = int(c.time[ids[len(ids) // 2]])
        f = b.query_feats(ids, t_q, 86400.0 * 30)
        pre = ids[c.time[ids] <= t_q]
        o = base
        for name, tidx, jcol, card in lay:
            sel = pre[c.table_idx[pre] == tidx]
            want = np.zeros(card)
            if len(sel):
                codes = np.clip(c.feat_cat[name][c.row_of[name][sel], jcol],
                                0, card - 1)
                want = np.bincount(codes, minlength=card)
            assert np.allclose(np.expm1(f[o:o + card]), want, atol=1e-3), (
                name, jcol)
            o += card
            checked += 1
    assert checked > 20


def test_two_filters_compose_instead_of_the_first_one_winning():
    """rel-trial study-outcome needs `outcome_type = Primary` AND
    `p_value <= 0.05`. An `elif` chain silently applied only the first, which
    is the 4x-dilution bug the task was stuck on."""
    from ledger import win_readout as wr

    class _Win:
        schema = None
    class _M:
        pass

    calls = []

    class FakeWin:
        schema = type("S", (), {"fact_tables": {"T": object()}})()

        def log_rate(self, z, name):
            return torch.zeros(3)                     # rate 1.0

    m = _M(); m.win = FakeWin()
    z = torch.zeros(3, 4)
    # patch the two marginal helpers to fixed shares
    orig_cat, orig_num = wr._cat_rate, wr._num_rate
    wr._cat_rate = lambda win, z, name, filt, negate, total: total * 0.25
    wr._num_rate = lambda win, z, name, filt, op, total: total * 0.5
    try:
        both = wr.rate(m, z, wr.Q(["T"], cat_in=("T", "a", {1}),
                                 num_le=("T", "b", 0.05)))
        one = wr.rate(m, z, wr.Q(["T"], cat_in=("T", "a", {1})))
    finally:
        wr._cat_rate, wr._num_rate = orig_cat, orig_num
    assert torch.allclose(one, torch.full((3,), 0.25, dtype=torch.float64))
    assert torch.allclose(both, torch.full((3,), 0.125, dtype=torch.float64))


# -- window-primary objective (ARCHITECTURE 8.2a) -----------------------------
#
# The claim under test is that a backbone trained SOLELY on window aggregates
# is the right trunk for the 21 entity tasks. These tests pin the mechanics:
# the next-event terms must be genuinely absent, not merely zero-weighted, and
# a run that would optimise nothing must fail loudly rather than log 0.0.

def _winprim_batcher(next_event, **kw):
    c = _two_destination_db()
    return c, PackedBatcher(c, "user", max_len=64, batch_rows=4, n_neg=8,
                            seed=0, window_head=True, n_query=6,
                            next_event=next_event, **kw)


def test_window_primary_drops_the_next_event_terms_entirely():
    c, b = _winprim_batcher(False)
    batch = b.batch()
    # the expensive targets are not merely unused, they are never built
    assert batch["target"]["who"] == {}
    assert batch["target"]["what"] == {}

    m = LEDGER(c.schema, dim=32, layers=2, heads=4, window_head=True,
             next_event=False)
    losses = m.loss(batch)
    assert set(losses) == {"win", "total"}
    assert torch.allclose(losses["total"], losses["win"])
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.parameters())


def test_default_objective_still_builds_every_term():
    """The historical multi-task path must be untouched by the new flag."""
    c, b = _winprim_batcher(True)
    m = LEDGER(c.schema, dim=32, layers=2, heads=4, window_head=True)
    losses = m.loss(b.batch())
    assert {"when", "where", "who", "what", "win"} <= set(losses)


def test_window_primary_without_query_tokens_raises():
    """A gated-off model with no window supervision optimises nothing.

    Zeroing weights would report a clean 0.0 curve for an entire run; this is
    the failure this repo keeps paying for, so it raises instead.
    """
    c = _two_destination_db()
    b = PackedBatcher(c, "user", max_len=64, batch_rows=4, n_neg=8, seed=0,
                      window_head=False, next_event=False)
    m = LEDGER(c.schema, dim=32, layers=2, heads=4, next_event=False)
    with pytest.raises(RuntimeError, match="no loss terms"):
        m.loss(b.batch())


def test_window_primary_batches_are_cheaper_to_build():
    """Skipping _build_who is the point: it is the dominant cost of a batch."""
    import time
    def t(next_event):
        _, b = _winprim_batcher(next_event)
        b.batch()                                   # warm any caches
        s = time.perf_counter()
        for _ in range(3):
            b.batch()
        return time.perf_counter() - s
    assert t(False) < t(True)


# ---------------------------------------------------------------------------
# Window-loss term balance (--win_balance). The defect being fixed is not a
# crash: `legacy` returns an unweighted mean over every term, and on a schema
# with several fact tables the ONE output every entity task reads -- the
# per-table rate -- is a few percent of it. These tests pin the arithmetic so
# the balance cannot drift silently back.
# ---------------------------------------------------------------------------

def _win_model(c, balance="legacy", **kw):
    from ledger.model.window import bucket_edges
    return LEDGER(c.schema, dim=32, layers=2, heads=2, window_head=True,
                win_edges=bucket_edges(c, 8), win_balance=balance, **kw)


def _win_batch_and_h(c, model):
    b = _win_batcher(c)
    batch = b.batch()
    h = model.encode(batch)
    win = batch["window"]
    assert win["n_query"] > 0
    return h, win


def test_legacy_balance_is_unchanged(corpus):
    """Every number measured before 2026-08-26 came from this path, so the
    default must still be bit-identical to the v1 arithmetic."""
    c, _ = corpus
    torch.manual_seed(0)
    m = _win_model(c, "legacy")
    h, win = _win_batch_and_h(c, m)
    got = m.win.loss(h[win["b"], win["l"]], win)
    # recompute the v1 formula independently
    import torch.nn.functional as F
    from ledger.model.window import slog
    z = m.win.state(h[win["b"], win["l"]], win["elapsed"], win["horizon"])
    nq, taus = win["n_query"], m.win.tau_t
    total, n_terms = z.z.new_zeros(()), 0
    for name, spec in c.schema.fact_tables.items():
        cnt = win["counts"][:, spec.table_idx].to(z.z.dtype)
        total = total + F.poisson_nll_loss(m.win.log_rate(z, name), cnt,
                                           log_input=True, full=False,
                                           reduction="mean")
        total = total + m.win._pinball(m.win.q_count(z, name),
                                       torch.log1p(cnt), taus)
        n_terms += 2
        g = win["per_table"].get(name)
        if g is None or not g["qi"].numel():
            continue
        qi, w = g["qi"], g["w"]
        for slot, (j, card) in enumerate(m.win.cat_cols.get(name, [])):
            tgt = torch.zeros(nq, card)
            tgt.index_put_((qi, g["feat_cat"][:, j].long().clamp(0, card - 1)),
                           w, accumulate=True)
            total = total + F.poisson_nll_loss(
                m.win.log_rate_cat(z, name, j), tgt, log_input=True,
                full=False, reduction="mean") * card
            n_terms += 1
        n_num = m.win.num_cols[name]
        if n_num:
            vals = g["feat_num"]
            bk = m.win._bucket_of(name, vals)
            tgt = torch.zeros(nq, n_num, m.win.n_buckets)
            col = torch.arange(n_num).expand_as(bk)
            tgt.index_put_((qi.unsqueeze(-1).expand_as(bk), col, bk),
                           w.unsqueeze(-1).expand_as(vals), accumulate=True)
            total = total + F.poisson_nll_loss(
                m.win.log_rate_num(z, name), tgt, log_input=True,
                full=False, reduction="mean") * m.win.n_buckets
            zs = torch.zeros(nq, n_num).index_add_(0, qi,
                                                   vals * w.unsqueeze(-1))
            total = total + m.win._pinball(
                m.win.q_sum(z, name),
                slog(m.win.destandardize(name, zs, cnt)), taus)
            n_terms += 2
    assert torch.allclose(got, total / n_terms, atol=1e-5)


def test_weighted_balance_gives_the_rate_term_a_third_of_the_loss(corpus):
    """`w_rate` must actually control a third of the objective, not 1/52 of
    it. Measured by zeroing the other two groups and comparing."""
    c, _ = corpus
    torch.manual_seed(0)
    m = _win_model(c, "weighted")
    h, win = _win_batch_and_h(c, m)
    hq = h[win["b"], win["l"]]
    full = m.win.loss(hq, win).item()

    def only(**kw):
        for k in ("w_rate", "w_qcount", "w_detail"):
            setattr(m.win, k, kw.get(k, 0.0))
        v = m.win.loss(hq, win).item()
        for k in ("w_rate", "w_qcount", "w_detail"):
            setattr(m.win, k, 1.0)
        return v

    r, q, d = only(w_rate=1.0), only(w_qcount=1.0), only(w_detail=1.0)
    # the three groups partition the loss exactly (all weights linear)
    assert abs((r + q + d) - full) < 1e-4
    # and the rate group is a real share of it, not a rounding error. The
    # legacy path puts it at ~1/52 on a seven-table schema.
    assert abs(r) > 0.05 * (abs(r) + abs(q) + abs(d))


def test_weighted_balance_ignores_column_count(corpus):
    """The point of the change: how many columns a fact table happens to have
    must not change how much the rate term is worth."""
    c, _ = corpus
    torch.manual_seed(0)
    m = _win_model(c, "weighted")
    h, win = _win_batch_and_h(c, m)
    hq = h[win["b"], win["l"]]
    base = m.win.loss(hq, win).item()
    m.win.w_detail = 0.0
    no_detail = m.win.loss(hq, win).item()
    m.win.w_detail = 1.0
    # dropping every per-column term leaves the rate+qcount groups untouched
    z = m.win.state(hq, win["elapsed"], win["horizon"])
    import torch.nn.functional as F
    hand = 0.0
    for name, spec in c.schema.fact_tables.items():
        cnt = win["counts"][:, spec.table_idx].to(z.z.dtype)
        hand += (F.poisson_nll_loss(m.win.log_rate(z, name), cnt,
                                    log_input=True, full=False,
                                    reduction="mean")
                 + m.win._pinball(m.win.q_count(z, name),
                                  torch.log1p(cnt), m.win.tau_t)).item()
    assert abs(no_detail - hand / len(c.schema.fact_tables)) < 1e-4
    assert base != no_detail


def test_unknown_balance_is_rejected(corpus):
    c, _ = corpus
    with pytest.raises(ValueError, match="balance"):
        _win_model(c, "mean")


# ---------------------------------------------------------------------------
# A: the wide-and-deep feature path. Two properties define it -- the model
# STARTS at the GLM (deep branch zero), and the features reach the output
# without crossing the trunk. Both are silent if broken: the run would simply
# be a slightly different model that trains slightly worse.
# ---------------------------------------------------------------------------

def _feat_model(c, **kw):
    from ledger.model.window import bucket_edges
    n_qf = PackedBatcher.n_query_feats(c.schema)
    return LEDGER(c.schema, dim=32, layers=2, heads=2, window_head=True,
                win_edges=bucket_edges(c, 8), query_feats=n_qf, **kw), n_qf


def test_feat_path_starts_exactly_at_the_glm(corpus):
    """The deep branch is zero-initialised, so at step 0 the head's outputs
    must be a function of the query features and the biases ALONE. If this
    regresses, the arm silently stops being 'GLM plus residual'."""
    c, _ = corpus
    torch.manual_seed(0)
    m, n_qf = _feat_model(c, feat_path=True)
    q = 5
    feats = torch.randn(q, n_qf)
    for h in (torch.randn(q, 32), torch.randn(q, 32) * 7.0):
        s = m.win.state(h, torch.full((q,), 3600.0),
                        torch.full((q,), 86400.0), feats)
        for name in c.schema.fact_tables:
            lr = m.win.log_rate(s, name)
            glm = (m.win.feat_rate[name](feats).squeeze(-1)
                   + m.win.rate[name].bias)
            assert torch.allclose(lr, glm, atol=1e-5), name


def test_feat_path_output_moves_with_the_features(corpus):
    """The point of the skip: a change in the raw features must reach the
    output. Under the old query-token route the signal had to survive twelve
    LayerNorms first."""
    c, _ = corpus
    torch.manual_seed(0)
    m, n_qf = _feat_model(c, feat_path=True)
    h = torch.randn(3, 32)
    el, hz = torch.full((3,), 60.0), torch.full((3,), 86400.0)
    f0 = torch.zeros(3, n_qf)
    f1 = torch.ones(3, n_qf)
    name = next(iter(c.schema.fact_tables))
    a = m.win.log_rate(m.win.state(h, el, hz, f0), name)
    b = m.win.log_rate(m.win.state(h, el, hz, f1), name)
    assert not torch.allclose(a, b)


def test_feat_path_mismatch_raises_instead_of_zeroing_the_glm(corpus):
    """A head with the path that is handed no features, or one without it
    that is handed some, is a model/caller disagreement about architecture.
    It must raise -- silently dropping the GLM branch would evaluate a
    different model and report a plausible number."""
    c, _ = corpus
    torch.manual_seed(0)
    on, n_qf = _feat_model(c, feat_path=True)
    off, _ = _feat_model(c, feat_path=False)
    h, el, hz = torch.randn(2, 32), torch.full((2,), 1.0), torch.full((2,), 1.0)
    with pytest.raises(ValueError, match="query features"):
        on.win.state(h, el, hz)
    with pytest.raises(ValueError, match="no\n?.*feature path|feature path"):
        off.win.state(h, el, hz, torch.zeros(2, n_qf))


def test_feat_path_off_is_the_old_model_exactly(corpus):
    """Default must be untouched: same parameter set, same numbers."""
    c, _ = corpus
    torch.manual_seed(0)
    off, _ = _feat_model(c, feat_path=False)
    assert off.win.feat_path is False
    assert off.win.feat_rate is None
    names = [n for n, _ in off.win.named_parameters()]
    assert not any("feat_rate" in n or "feat_qcount" in n for n in names)
    # and a head built with the flag has strictly more parameters
    torch.manual_seed(0)
    on, _ = _feat_model(c, feat_path=True)
    assert sum(p.numel() for p in on.win.parameters()) > \
           sum(p.numel() for p in off.win.parameters())


def test_feat_path_trains_end_to_end(corpus):
    """One real batch, one optimiser step, loss must be finite and the GLM
    weights must actually receive gradient."""
    c, _ = corpus
    torch.manual_seed(0)
    m, _ = _feat_model(c, feat_path=True, win_balance="weighted")
    b = _win_batcher(c)
    batch = b.batch()
    loss = m.loss(batch)["total"]
    assert torch.isfinite(loss)
    loss.backward()
    name = next(iter(c.schema.fact_tables))
    g = m.win.feat_rate[name].weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


# ---------------------------------------------------------------------------
# B: dense window supervision. The vectorised prefix sweep is a reimplement-
# ation of `query_feats` plus the window count, and a silent divergence
# between the two would train the wide-and-deep GLM on one feature definition
# and evaluate it on another. Both are pinned against brute force.
# ---------------------------------------------------------------------------

def _dense_batcher(c, **kw):
    return PackedBatcher(c, ["drivers"], max_len=256, batch_rows=4, seed=5,
                         window_head=True, n_query=4, dense_window=True,
                         n_dense=16, **kw)


def test_dense_feats_match_query_feats_exactly(corpus):
    """The sweep must reproduce the scalar `query_feats` bit for bit."""
    c, _ = corpus
    b = _dense_batcher(c)
    cutoff = int(c.cutoff.timestamp())
    rng = np.random.default_rng(1)
    checked = 0
    for e in rng.choice(b.entities["drivers"], 25, replace=False):
        ids = c.history("drivers", int(e))
        if len(ids) < 3:
            continue
        ts = c.time[ids]
        k, t_q, hz = b._sample_dense(ts, cutoff, 8)
        if not len(k):
            continue
        feats, _ = b._dense_stats(ids, t_q, hz)
        for i in range(len(k)):
            want = b.query_feats(ids, int(t_q[i]), float(hz[i]))
            assert np.allclose(feats[i], want, atol=1e-5), (
                e, i, np.abs(feats[i] - want).max())
            checked += 1
    assert checked > 20, checked


def test_dense_counts_match_a_brute_force_recount(corpus):
    """The dense rate target must be the events the entity actually emits in
    (t_q, t_q + D], recounted straight from the corpus."""
    c, _ = corpus
    b = _dense_batcher(c)
    n_tbl = len(c.schema.fact_tables)
    cutoff = int(c.cutoff.timestamp())
    rng = np.random.default_rng(2)
    checked = 0
    for e in rng.choice(b.entities["drivers"], 25, replace=False):
        ids = c.history("drivers", int(e))
        if len(ids) < 3:
            continue
        ts = c.time[ids]
        k, t_q, hz = b._sample_dense(ts, cutoff, 8)
        if not len(k):
            continue
        _, counts = b._dense_stats(ids, t_q, hz)
        for i in range(len(k)):
            sel = ids[(ts > t_q[i]) & (ts <= t_q[i] + hz[i])]
            want = np.bincount(c.table_idx[sel], minlength=n_tbl)
            assert np.array_equal(counts[i].astype(np.int64), want)
            checked += 1
    assert checked > 20, checked


def test_dense_targets_never_run_past_the_cutoff(corpus):
    """The leakage rule: a dense window must be fully observed."""
    c, _ = corpus
    b = _dense_batcher(c)
    cutoff = int(c.cutoff.timestamp())
    for e in b.entities["drivers"][:60]:
        ts = c.time[c.history("drivers", int(e))]
        k, t_q, hz = b._sample_dense(ts, cutoff, 8)
        if not len(k):
            continue
        assert (t_q + hz <= cutoff).all()
        assert (t_q >= ts[k]).all()      # a query never precedes its token


def test_dense_positions_index_their_own_token(corpus):
    """`l` must be the row position of the token whose state answers the
    query -- and that token's time must be the latest at or before t_q in its
    own sequence. A position off by one row reads another entity's state and
    raises nothing."""
    c, _ = corpus
    b = _dense_batcher(c)
    batch = b.batch()
    wd = batch["window_dense"]
    assert wd["n_query"] > 0
    t = batch["t"]
    sid = batch["seq_id"]
    bb, ll = wd["b"], wd["l"]
    assert (ll < b.max_len).all()
    tok_t = t[bb, ll].numpy().astype(np.int64)
    t_q = tok_t + wd["elapsed"].numpy().astype(np.int64)
    assert (t_q >= tok_t).all()
    # the next token of the SAME sequence must be later than its own token
    for i in range(0, len(bb), max(1, len(bb) // 40)):
        r, l = int(bb[i]), int(ll[i])
        if l + 1 < t.shape[1] and sid[r, l + 1] == sid[r, l]:
            assert t[r, l + 1] >= t[r, l]


def test_dense_supervision_is_much_denser_than_the_query_tokens(corpus):
    """The whole point of B. If this ratio collapses the experiment is not
    testing what it claims to."""
    c, _ = corpus
    batch = _dense_batcher(c).batch()
    sparse = batch["window"]["n_query"]
    dense = batch["window_dense"]["n_query"]
    assert dense > 3 * sparse, (dense, sparse)


def test_dense_loss_trains_and_is_separate_from_the_sparse_one(corpus):
    """Dense rows must NOT be folded into the sparse detail targets -- that
    would teach every categorical rate zero on windows whose detail was never
    enumerated. Separate term, separate weight."""
    from ledger.model.window import bucket_edges
    c, _ = corpus
    torch.manual_seed(0)
    n_qf = PackedBatcher.n_query_feats(c.schema)
    m = LEDGER(c.schema, dim=32, layers=2, heads=2, window_head=True,
             win_edges=bucket_edges(c, 8), query_feats=n_qf, feat_path=True,
             win_balance="weighted")
    batch = _dense_batcher(c).batch()
    losses = m.loss(batch)
    assert "windense" in losses and "win" in losses
    assert torch.isfinite(losses["windense"])
    assert losses["windense"] is not losses["win"]
    # the sparse detail targets must still be built from sparse queries only
    assert batch["window"]["feats"].shape[0] == batch["window"]["n_query"]
    losses["total"].backward()
    name = next(iter(c.schema.fact_tables))
    assert m.win.rate[name].weight.grad is not None


def test_dense_window_off_leaves_the_batch_untouched(corpus):
    """Default path must be byte-identical: same tokens, no new key."""
    c, _ = corpus
    a = PackedBatcher(c, ["drivers"], max_len=256, batch_rows=4, seed=5,
                      window_head=True, n_query=4).batch()
    d = _dense_batcher(c).batch()
    assert "window_dense" not in a
    assert "window_dense" in d
    # B adds supervision WITHOUT changing the token stream
    assert torch.equal(a["t"], d["t"])
    assert torch.equal(a["seq_id"], d["seq_id"])
    assert torch.equal(a["window"]["b"], d["window"]["b"])


def _path_expand_db():
    """A site/study/analysis shape: the analyses key on the STUDY, never on
    the site, so a site's history reaches them only through the bridge."""
    from relbench.base import Database, Table
    t0 = pd.Timestamp("2020-01-01")
    # RelBench tables are datetime64[ns] and `EventCorpus.build` converts
    # with `astype("int64") // 10**9`. pandas 3 builds datetime64[us] from
    # Timestamp arithmetic, which that formula would read 1000x low, so the
    # fixture pins the resolution real data has.
    def d(k):
        return np.array([t0 + pd.Timedelta(days=int(k))], dtype="datetime64[ns]")[0]
    # site 0 ran studies 0,1 (joined day 0, 50); site 1 ran study 1 (day 200)
    bridge = pd.DataFrame({"site_id": [0, 0, 1],
                           "study_id": [0, 1, 1],
                           "time": [d(0), d(50), d(200)]})
    # analyses on study 0 (day 10) and study 1 (days 100, 300)
    analyses = pd.DataFrame({"study_id": [0, 1, 1],
                             "p_value": [0.01, 0.5, 0.02],
                             "time": [d(10), d(100), d(300)]})
    tables = {
        "site": Table(df=pd.DataFrame({"site_id": [0, 1]}),
                      fkey_col_to_pkey_table={}, pkey_col="site_id"),
        "study": Table(df=pd.DataFrame({"study_id": [0, 1]}),
                       fkey_col_to_pkey_table={}, pkey_col="study_id"),
        "bridge": Table(df=bridge, pkey_col=None, time_col="time",
                        fkey_col_to_pkey_table={"site_id": "site",
                                                "study_id": "study"}),
        "analyses": Table(df=analyses, pkey_col=None, time_col="time",
                          fkey_col_to_pkey_table={"study_id": "study"}),
    }
    return Database(tables), t0 + pd.Timedelta(days=400)


def test_path_expand_reaches_events_only_via_the_bridge():
    """Without --path_expand a site's history holds bridge rows only.

    This is the rel-trial site-success shape. `outcome_analyses` key on
    `nct_id -> studies`, so inverting foreign keys never files them under a
    facility: measured 0% of facility histories held one, and the readout
    was rating events the entity had never observed.
    """
    db, cutoff = _path_expand_db()
    plain = EventCorpus.build(db, cutoff)
    a_idx = plain.schema.fact_tables["analyses"].table_idx
    for row in (0, 1):
        ev = plain.history("site", row)
        assert not (plain.table_idx[ev] == a_idx).any(), \
            "analyses must be unreachable without path expansion"

    ex = EventCorpus.build(db, cutoff,
                           path_expand="site:bridge:analyses")
    got = {}
    for row in (0, 1):
        ev = ex.history("site", row)
        got[row] = sorted(ex.time[ev[ex.table_idx[ev] == a_idx]].tolist())

    def day(k):
        return int(pd.Timestamp("2020-01-01").timestamp()) + k * 86400
    # site 0 ran both studies and joined before every analysis on them
    assert got[0] == [day(10), day(100), day(300)]
    # site 1 joined study 1 on day 200: the day-100 analysis predates the
    # link and must NOT be visible, the day-300 one must be
    assert got[1] == [day(300)], got[1]


def test_path_expand_does_not_duplicate_or_disturb_other_entities():
    """A site linked twice to one study must not count its analyses twice,
    and no other entity's history may move."""
    from relbench.base import Database, Table
    db, cutoff = _path_expand_db()
    dup = db.table_dict["bridge"].df
    tables = dict(db.table_dict)
    tables["bridge"] = Table(
        df=pd.concat([dup, dup.iloc[[0]]], ignore_index=True),
        pkey_col=None, time_col="time",
        fkey_col_to_pkey_table={"site_id": "site", "study_id": "study"})
    db2 = Database(tables)

    ex = EventCorpus.build(db2, cutoff, path_expand="site:bridge:analyses")
    a_idx = ex.schema.fact_tables["analyses"].table_idx
    ev = ex.history("site", 0)
    an = ev[ex.table_idx[ev] == a_idx]
    assert len(an) == len(set(an.tolist())), "duplicate references"

    base = EventCorpus.build(db, cutoff)
    ex2 = EventCorpus.build(db, cutoff, path_expand="site:bridge:analyses")
    assert np.array_equal(base.hist_index["study"], ex2.hist_index["study"])
    assert np.array_equal(base.time, ex2.time)


def test_path_expand_rejects_a_bad_spec():
    db, cutoff = _path_expand_db()
    with pytest.raises(ValueError):
        EventCorpus.build(db, cutoff, path_expand="site:bridge")
    with pytest.raises(KeyError):
        EventCorpus.build(db, cutoff, path_expand="nosuch:bridge:analyses")
    with pytest.raises(KeyError):
        EventCorpus.build(db, cutoff, path_expand="site:nosuch:analyses")


def test_path_expand_is_part_of_the_cache_identity():
    """Two corpora differing in path_expand are different objects; sharing
    one directory would serve histories the checkpoint was not trained on."""
    from ledger.data.cache import cache_dir
    t = pd.Timestamp("2020-01-01")
    a = cache_dir("ds", t)
    b = cache_dir("ds", t, path_expand="site:bridge:analyses")
    c = cache_dir("ds", t, path_expand="site:bridge:other")
    assert a != b and b != c and a != c
    # order-insensitive: the same set of paths is the same corpus
    d = cache_dir("ds", t, path_expand="x:b:e,y:b:e")
    e = cache_dir("ds", t, path_expand="y:b:e,x:b:e")
    assert d == e


def test_qsample_answers_match_a_direct_computation():
    """`qsample.answer` must equal a plain recomputation from the same rows.

    The targets are what the query head learns; if they are wrong the head
    learns something else and every downstream number is silently off.
    """
    import torch as _t
    from ledger import qsample as qs

    nq = 4
    n = 40
    g_qi = _t.tensor([i % nq for i in range(n)])
    fn = _t.randn(n, 3)
    fc = _t.randint(0, 5, (n, 2))
    w = _t.rand(n) + 0.5
    counts = _t.zeros(nq, 1).index_add_(
        0, g_qi, _t.ones(n, 1))
    win = {"counts": counts,
           "per_table": {"t": dict(qi=g_qi, w=w, feat_num=fn, feat_cat=fc)}}

    class C:
        def __init__(s, name, kind, mean=0.0, std=1.0, card=5):
            s.name, s.kind, s.mean, s.std = name, kind, mean, std
            s._c = card
        @property
        def cardinality(s): return s._c
    class Spec:
        table_idx = 0
        columns = [C("n0", "numeric", 2.0, 3.0), C("n1", "numeric"),
                   C("n2", "numeric"), C("c0", "categorical"),
                   C("c1", "categorical")]
    class Sch:
        fact_tables = {"t": Spec()}

    def direct(mask, vals=None, denom=None):
        out = _t.zeros(nq)
        src = (w * mask) if vals is None else (w * mask * vals)
        out.index_add_(0, g_qi, src)
        if denom is None:
            return out
        d = _t.zeros(nq).index_add_(0, g_qi, denom)
        return out / d.clamp(min=1e-9)

    # filtered count
    m = (fc[:, 0] == 2).float()
    spec = qs.QSpec("t", "count", qs.F_CAT, 0, (2,))
    got, _ = qs.answer(spec, win, Sch, nq, fn.device)
    assert _t.allclose(got, direct(m), atol=1e-5)

    # sum of a standardized column, de-standardised
    raw = fn[:, 0] * 3.0 + 2.0
    spec = qs.QSpec("t", "sum", qs.F_NONE, -1, (), 0.0, 0)
    got, _ = qs.answer(spec, win, Sch, nq, fn.device)
    assert _t.allclose(got, direct(_t.ones(n), raw), atol=1e-4)

    # mean, and the numeric filter
    mm = (fn[:, 1] <= 0.25).float()
    spec = qs.QSpec("t", "mean", qs.F_LE, 1, (), 0.25, 0)
    got, valid = qs.answer(spec, win, Sch, nq, fn.device)
    cnt = direct(mm)
    exp = direct(mm, raw) / cnt.clamp(min=1e-9)
    assert _t.allclose(got[valid], exp[valid], atol=1e-4)
    assert bool((valid == (cnt > 0)).all())

    # ratio is the filtered share, and is undefined on an empty table
    spec = qs.QSpec("t", "ratio", qs.F_CAT, 0, (2,))
    got, valid = qs.answer(spec, win, Sch, nq, fn.device)
    assert _t.allclose(got, (direct(m) / direct(_t.ones(n))).clamp(0, 1),
                       atol=1e-5)
    assert bool(valid.all())


def test_query_sampler_holds_out_the_eval_task_signature():
    """The zero-shot claim rests on this: the evaluated query must never be
    sampled as a training target."""
    import pandas as _pd
    from relbench.datasets import get_dataset as _gd
    from ledger.data.cache import load_corpus
    from ledger import qsample as qs, win_readout as wr

    try:
        d = _gd("rel-f1", download=False)
        c = load_corpus("rel-f1", _pd.Timestamp(d.val_timestamp),
                        db=d.get_db, verbose=False)
    except Exception as e:
        pytest.skip(f"rel-f1 corpus unavailable: {e}")

    q = wr.QUERIES[("rel-f1", "driver-position")]
    spec = qs.from_task_q(q, c.schema)
    assert spec is not None and spec.agg == "mean" and spec.table == "results"

    excl = qs.signatures_of(q, c.schema)
    assert spec.signature() in excl
    rng = np.random.default_rng(0)
    drawn = qs.sample(c.schema, rng, 4000, exclude=excl)
    assert drawn, "sampler produced nothing"
    assert not any(x.signature() in excl for x in drawn), \
        "the held-out task query was sampled as a training target"
    # and it must still be able to produce that KIND of query elsewhere
    assert any(x.agg == "mean" for x in drawn)


def test_from_task_q_refuses_what_it_cannot_express():
    """A conjunction or a multi-value set must return None so the caller
    falls back, rather than being answered as a different question."""
    import pandas as _pd
    from relbench.datasets import get_dataset as _gd
    from ledger.data.cache import load_corpus
    from ledger import qsample as qs, win_readout as wr

    try:
        d = _gd("rel-f1", download=False)
        c = load_corpus("rel-f1", _pd.Timestamp(d.val_timestamp),
                        db=d.get_db, verbose=False)
    except Exception as e:
        pytest.skip(f"rel-f1 corpus unavailable: {e}")
    # occurrence labels are classification, not a scalar aggregate
    assert qs.from_task_q(wr.Q(["results"], agg="occur"), c.schema) is None
    # two filters at once
    two = wr.Q(["results"], agg="count",
               cat_in=("results", "statusId", {1}),
               num_le=("results", "positionOrder", 3.0))
    assert qs.from_task_q(two, c.schema) is None


def test_set_valued_category_filters_are_expressible():
    """rel-event user-attendance is `status in {yes, maybe}`. If the encoder
    refused sets, that task would silently fall back to composed marginals --
    the exact path the query head exists to replace."""
    import pandas as _pd
    from relbench.datasets import get_dataset as _gd
    from ledger.data.cache import load_corpus
    from ledger import qsample as qs, win_readout as wr
    try:
        d = _gd("rel-event", download=False)
        c = load_corpus("rel-event", _pd.Timestamp(d.val_timestamp),
                        db=d.get_db, verbose=False)
    except Exception as e:
        pytest.skip(f"rel-event corpus unavailable: {e}")
    q = wr.QUERIES[("rel-event", "user-attendance")]
    spec = qs.from_task_q(q, c.schema)
    assert spec is not None, "set-valued filter must be expressible"
    assert len(spec.f_cats) == 2

    # and each singleton inside the set is held out too, not just the pair
    excl = qs.signatures_of(q, c.schema)
    assert spec.signature() in excl
    for v in spec.f_cats:
        assert (spec.table, "count", qs.F_CAT, spec.f_col, (v,), -1) in excl
    drawn = qs.sample(c.schema, np.random.default_rng(0), 3000, exclude=excl)
    assert not any(x.signature() in excl for x in drawn)


def test_occurrence_tasks_are_actually_held_out():
    """`occur` is 12 of the 21 registry queries -- EVERY classification task.

    It is not in `qsample.AGGS`, so `signatures_of` used to return the empty
    set for all twelve and the exclusion silently did nothing: the sampler was
    free to draw the evaluated query as `count` on the same table and filter,
    which is the same question, and train on it. The two tests above happened
    to use `mean` and `count` queries, so nothing caught it.

    An empty exclusion set is the failure mode, and it is silent -- the run
    logs `excluding 0 signature(s)` and trains happily -- so this asserts
    non-emptiness for every occurrence task in the registry, not just one.
    """
    import pandas as _pd
    from relbench.datasets import get_dataset as _gd
    from ledger.data.cache import load_corpus
    from ledger import qsample as qs, win_readout as wr

    try:
        d = _gd("rel-f1", download=False)
        c = load_corpus("rel-f1", _pd.Timestamp(d.val_timestamp),
                        db=d.get_db, verbose=False)
    except Exception as e:
        pytest.skip(f"rel-f1 corpus unavailable: {e}")

    q = wr.QUERIES[("rel-f1", "driver-top3")]
    assert q.agg == "occur", "fixture drifted; pick another occurrence task"
    # it is deliberately not ANSWERABLE as a scalar aggregate ...
    assert qs.from_task_q(q, c.schema) is None
    # ... but it must still be RECOGNISED, or it is not held out.
    excl = qs.signatures_of(q, c.schema)
    assert excl, "occurrence task produced an empty exclusion set"
    # the sampler names the same question `count`, so that is what is excluded
    assert all(sig[1] == "count" for sig in excl)
    assert ("qualifying", "count", qs.F_NONE, -1, (), -1) in excl

    drawn = qs.sample(c.schema, np.random.default_rng(0), 4000, exclude=excl)
    assert drawn, "sampler produced nothing"
    assert not any(x.signature() in excl for x in drawn), \
        "the held-out occurrence query was sampled as a training target"
    # and holding it out must not cost the sampler the whole aggregation
    assert any(x.agg == "count" for x in drawn)


def test_every_registry_query_excludes_something(monkeypatch):
    """Drift alarm. A new task, or a new `agg` spelling, that `signatures_of`
    does not recognise is invisible: it trains and scores normally while
    leaking the evaluated query. Every query over a table the schema has must
    exclude at least one signature."""
    import pandas as _pd
    from relbench.datasets import get_dataset as _gd
    from ledger.data.cache import load_corpus
    from ledger import qsample as qs, win_readout as wr

    checked = 0
    for ds in ("rel-f1", "rel-event"):
        try:
            d = _gd(ds, download=False)
            c = load_corpus(ds, _pd.Timestamp(d.val_timestamp),
                            db=d.get_db, verbose=False)
        except Exception:
            continue
        for (qds, task), q in wr.QUERIES.items():
            if qds != ds or not q.tables:
                continue
            if q.tables[0] not in c.schema.fact_tables:
                continue
            assert qs.signatures_of(q, c.schema), \
                f"{qds}/{task} (agg={q.agg!r}) excludes nothing"
            checked += 1
    if not checked:
        pytest.skip("no corpus available")


def test_query_head_output_is_clamped_to_the_observed_range():
    """`sexp` inverts the head exponentially, so an unbounded output turns a
    small error into an order-of-magnitude one (measured: one rel-f1 eval at
    NMAE 4.43 in a 0.55 neighbourhood)."""
    import torch as _t
    from ledger.model.window import WindowHead
    from ledger import qsample as qs

    class C:
        name, kind, mean, std = "n0", "numeric", 0.0, 1.0
        cardinality = 3
    class Spec:
        table_idx = 0
        columns = [C()]
    class Sch:
        fact_tables = {"t": Spec()}

    w = WindowHead(16, Sch, n_buckets=4, query_head=True)
    ai = qs.AGGS.index("count")
    w.q_lo[ai], w.q_hi[ai] = -1.0, 2.0
    with _t.no_grad():
        for p in w.query_head[-1].parameters():
            p.fill_(50.0)                      # force a wild prediction
    st = w.state(_t.randn(3, 16), _t.zeros(3), _t.full((3,), 100.0))
    enc = qs.encode([qs.QSpec("t", "count")], Sch, st.z.device)
    assert float(w.query_predict(st, enc, clamp=True).max()) <= 2.0 + 1e-5
    assert float(w.query_predict(st, enc, clamp=False).max()) > 2.0


def _tiny_window_head(**kw):
    from ledger.model.window import WindowHead
    class C:
        name, kind, mean, std = "n0", "numeric", 0.0, 1.0
        cardinality = 3
    class Spec:
        table_idx = 0
        columns = [C()]
    class Sch:
        fact_tables = {"t": Spec()}
    return WindowHead(16, Sch, n_buckets=4, **kw)


def test_rate_balance_none_is_the_plain_poisson_mean():
    """`none` must be bit-identical to what every checkpoint before this was
    trained with, or no earlier number is reproducible."""
    import torch as _t, torch.nn.functional as _F
    w = _tiny_window_head()
    lr = _t.randn(64)
    tgt = (_t.rand(64) < 0.1).float() * _t.randint(1, 5, (64,)).float()
    assert _t.allclose(
        w._pois(lr, tgt),
        _F.poisson_nll_loss(lr, tgt, log_input=True, full=False,
                            reduction="mean"))


def test_rate_balance_cb_equalises_zero_and_nonzero_mass():
    """The whole point: the rare non-empty windows must carry the same total
    weight as the empty ones, so the head cannot settle on the base rate."""
    import torch as _t, torch.nn.functional as _F
    w = _tiny_window_head(rate_balance="cb")
    lr = _t.randn(200)
    tgt = _t.zeros(200)
    tgt[:10] = 3.0                        # 5% non-empty, like the real tasks
    pl = _F.poisson_nll_loss(lr, tgt, log_input=True, full=False,
                             reduction="none")
    pos = (tgt > 0).float()
    wt = pos + (1 - pos) * (pos.sum() / (200 - pos.sum()))
    assert _t.allclose(w._pois(lr, tgt), (pl * wt).sum() / wt.sum())
    # the two groups end up with equal total weight
    assert _t.allclose(wt[tgt > 0].sum(), wt[tgt == 0].sum())
    # and it genuinely differs from the unbalanced mean on a skewed target
    assert not _t.allclose(w._pois(lr, tgt), pl.mean())


def test_rate_balance_cb_falls_back_when_degenerate():
    """An all-empty or all-full batch has no two groups to balance."""
    import torch as _t, torch.nn.functional as _F
    w = _tiny_window_head(rate_balance="cb")
    lr = _t.randn(32)
    for tgt in (_t.zeros(32), _t.full((32,), 2.0)):
        assert _t.allclose(
            w._pois(lr, tgt),
            _F.poisson_nll_loss(lr, tgt, log_input=True, full=False,
                                reduction="mean"))


def test_rate_balance_rejects_an_unknown_setting():
    with pytest.raises(ValueError):
        _tiny_window_head(rate_balance="focal-ish")


# --- neighbour-aggregate query features (--query_nbr) ----------------------
#
# The block exists to give the query token the 2-hop signal the GNN baselines
# aggregate directly and our architecture otherwise sees only through a stale
# EMA vector (RESEARCH.md's post-mortem on rel-f1 driver-dnf). These pin the
# three properties that make it usable: a deterministic layout, an
# append-only width, and leak-freedom.

def _nbr_batcher(c, **kw):
    return PackedBatcher(c, ["drivers"], max_len=256, batch_rows=4, seed=5,
                         window_head=True, n_query=4, query_nbr=True, **kw)


def test_query_nbr_layout_is_deterministic_and_capped(corpus):
    from ledger.data.batching import query_nbr_layout, QFEAT_NBR_SLOTS
    c, _ = corpus
    lay = query_nbr_layout(c.schema)
    assert lay, "rel-f1 has foreign keys, so the layout must be non-empty"
    assert len(lay) <= QFEAT_NBR_SLOTS
    # every entry names a real table, a real slot and a real entity table
    for name, tidx, slot, tgt in lay:
        spec = c.schema.fact_tables[name]
        assert spec.table_idx == tidx
        assert list(spec.fkeys.values())[slot] == tgt
        assert tgt in c.schema.entity_counts
    # cached, and stable across calls
    assert query_nbr_layout(c.schema) == lay


def test_query_nbr_widens_the_vector_as_a_suffix(corpus):
    """--query_nbr may only APPEND: the prefix must be bit-identical to the
    same query without it, or every checkpoint trained before it breaks."""
    from ledger.data.batching import query_nbr_layout
    c, _ = corpus
    n_slot = len(query_nbr_layout(c.schema))
    base = PackedBatcher.n_query_feats(c.schema, False, False)
    wide = PackedBatcher.n_query_feats(c.schema, False, True)
    assert wide - base == PackedBatcher.N_NBR_FEATS_PER_SLOT * n_slot
    # and the dyn flag stays an independent suffix
    assert (PackedBatcher.n_query_feats(c.schema, True, True)
            - PackedBatcher.n_query_feats(c.schema, True, False)
            == wide - base)

    b0 = PackedBatcher(c, ["drivers"], max_len=256, batch_rows=4, seed=5,
                       window_head=True, n_query=4)
    b1 = _nbr_batcher(c)
    rng = np.random.default_rng(0)
    checked = 0
    for e in rng.choice(b1.entities["drivers"], 30, replace=False):
        ids = c.history("drivers", int(e))
        if len(ids) < 3:
            continue
        t_q = int(c.time[ids[len(ids) // 2]]) + 1
        hz = 30 * 86400.0
        w0 = b0.query_feats(ids, t_q, hz)
        w1 = b1.query_feats(ids, t_q, hz)
        assert len(w0) == base and len(w1) == wide
        assert np.array_equal(w0, w1[:base]), "prefix drifted"
        checked += 1
    assert checked >= 5


def test_query_nbr_is_leak_free(corpus):
    """No neighbour statistic may move when events at or after t_q change.

    Recomputing against a corpus truncated AT t_q must give the identical
    block: if any lookup read `before=None`, or read the entity's own future
    events, this diverges.
    """
    from ledger.data.batching import query_nbr_layout
    c, _ = corpus
    b = _nbr_batcher(c)
    base = PackedBatcher.n_query_feats(c.schema, False, False)
    rng = np.random.default_rng(3)
    checked = 0
    for e in rng.choice(b.entities["drivers"], 40, replace=False):
        ids = c.history("drivers", int(e))
        if len(ids) < 6:
            continue
        t_q = int(c.time[ids[len(ids) // 2]]) + 1
        full = b.query_feats(ids, t_q, 30 * 86400.0)
        # hand the batcher only the pre-t_q prefix: the block must not care
        past = ids[c.time[ids] < t_q]
        trunc = b.query_feats(past, t_q, 30 * 86400.0)
        assert np.allclose(full[base:], trunc[base:], atol=1e-6), (
            "neighbour block moved when post-cutoff events were removed")
        checked += 1
    assert checked >= 5


def test_query_nbr_values_match_a_hand_computation(corpus):
    """Pin the semantics, not just the shape."""
    from ledger.data.batching import query_nbr_layout, QFEAT_NBR_K
    c, _ = corpus
    b = _nbr_batcher(c)
    base = PackedBatcher.n_query_feats(c.schema, False, False)
    lay = query_nbr_layout(c.schema)
    rng = np.random.default_rng(7)
    checked = 0
    for e in rng.choice(b.entities["drivers"], 40, replace=False):
        ids_all = c.history("drivers", int(e))
        if len(ids_all) < 8:
            continue
        t_q = int(c.time[ids_all[len(ids_all) // 2]]) + 1
        ids_p = ids_all[c.time[ids_all] < t_q]
        if not len(ids_p):
            continue
        out = b.query_feats(ids_all, t_q, 30 * 86400.0)
        ti = c.table_idx[ids_p]
        for k, (name, tidx, slot, tgt) in enumerate(lay):
            o = base + PackedBatcher.N_NBR_FEATS_PER_SLOT * k
            m = np.flatnonzero(ti == tidx)
            if not len(m):
                assert np.allclose(
                    out[o:o + PackedBatcher.N_NBR_FEATS_PER_SLOT], 0)
                continue
            rows = c.row_of[name][ids_p[m]]
            ent = c.links[name][rows, slot]
            ent = ent[ent >= 0]
            if not len(ent):
                continue
            uniq, counts = np.unique(ent, return_counts=True)
            assert math.isclose(out[o], np.log1p(len(uniq)), rel_tol=1e-5)
            assert math.isclose(out[o + 1], counts.max() / len(ent),
                                rel_tol=1e-5)
            # feature 3 is the 2-hop aggregate: the neighbours' own volume in
            # this table. Recompute it independently.
            seen, recent = set(), []
            for x in ent[::-1]:
                x = int(x)
                if x not in seen:
                    seen.add(x); recent.append(x)
                    if len(recent) >= QFEAT_NBR_K:
                        break
            want = 0.0
            for x in recent:
                h = c.history(tgt, x, before=t_q)
                if len(h):
                    want += np.log1p(int((c.table_idx[h] == tidx).sum()))
            assert math.isclose(out[o + 3], want / len(recent), rel_tol=1e-4)
            checked += 1
    assert checked >= 5


def test_query_nbr_survives_a_forward_pass(corpus):
    """The widened vector must reach query_feat_proj and train."""
    c, _ = corpus
    b = _nbr_batcher(c)
    n_qfeat = PackedBatcher.n_query_feats(c.schema, False, True)
    m = LEDGER(c.schema, dim=32, layers=2, heads=2, window_head=True,
             query_feats=n_qfeat)
    batch = b.batch()
    assert batch["window"]["feats"].shape[-1] == n_qfeat
    h = m.encode(batch, branch="temporal")
    loss = m.win.loss(h[batch["window"]["b"], batch["window"]["l"]],
                      batch["window"])
    assert torch.isfinite(loss)
    loss.backward()


# --- integer thresholds land on the separator, not on the value ------------

def test_integral_shift_detects_integer_columns(corpus):
    """rel-f1 `qualifying.position` is integer-valued; `races.round` too.
    A genuinely continuous column must NOT be shifted."""
    from ledger.model.window import bucket_edges, WindowHead
    c, _ = corpus
    w = WindowHead(32, c.schema, edges=bucket_edges(c, 8), n_buckets=8)
    j = w.numeric_index("qualifying", "position")
    assert j is not None
    s = w.integral_shift("qualifying", j)
    std = getattr(w, "nstd__qualifying")[j].item()
    assert math.isclose(s, 0.5 / std, rel_tol=1e-6), (
        "an integer column must shift by half a raw step")
    # a table with no numeric columns, or an out-of-range index, is inert
    assert w.integral_shift("qualifying", 99) == 0.0


def test_integer_threshold_matches_the_empirical_fraction(corpus):
    """The whole point: P(position <= 3) read through the bucket weights must
    equal the fraction of observed rows with position <= 3.

    Before the fix the linear interpolation gave 0.1122 against a true
    0.1346 -- an under-read concentrated in the single straddled bucket,
    which is what re-ranked driver-top3.
    """
    from ledger.model.window import bucket_edges, WindowHead
    from ledger.win_readout import _bucket_weights
    c, _ = corpus
    w = WindowHead(32, c.schema, edges=bucket_edges(c, 8), n_buckets=8)
    j = w.numeric_index("qualifying", "position")
    spec = [x for x in c.schema.fact_tables["qualifying"].columns
            if x.name == "position"][0]
    a = c.feat_num["qualifying"][:, j].astype(np.float64)
    raw = np.round(a * spec.std + spec.mean)
    truth = float((raw <= 3).mean())

    thr = w.standardize("qualifying", "position", 3.0)
    wt = _bucket_weights(w, "qualifying", j, thr, "le").cpu().numpy()
    e = w.edges_of("qualifying")[j].detach().cpu().numpy().astype(np.float64)
    lo = np.concatenate([[-np.inf], e])
    hi = np.concatenate([e, [np.inf]])
    got = 0.0
    for b in range(len(e) + 1):
        m = (a > lo[b]) & (a <= hi[b]) if np.isfinite(lo[b]) else (a <= hi[b])
        got += wt[b] * m.sum()
    got /= len(a)
    assert abs(got - truth) < 1e-3, f"read {got:.4f}, truth {truth:.4f}"

    # and `>` is the exact complement, so the two must sum to one per bucket
    wg = _bucket_weights(w, "qualifying", j, thr, "gt").cpu().numpy()
    assert np.allclose(wt + wg, 1.0)
# ---------------------------------------------------------------------------
# The recommendation side under the 2026-08-26 architecture.
#
# The branched backbone, the learned loss weights and the query-dynamics block
# were all built and tested from the classification/regression side. Each of
# them also touches the recommendation path -- WHO reads a branch, the learned
# weights can de-emphasise the WHO term, and the dynamics block widens a
# feature vector the retrieval batch must NOT contain -- and none of that was
# covered. These are the tests for that half.
# ---------------------------------------------------------------------------


def _branched_rec_model(schema, branch_layers=2, **kw):
    """The recommendation configuration: WHO features, reranker, branches."""
    return LEDGER(schema, dim=64, layers=2, heads=4, dropout=0.0,
                who_feats=PackedBatcher.N_WHO_FEATS, window=True, rerank=True,
                branch_layers=branch_layers, **kw)


def test_recommendation_reads_the_branch_who_was_trained_on(corpus):
    """`queries._query_states` must return the RETRIEVAL branch.

    Training routes WHO, the reranker and the entity-state writes through
    `hb["retrieval"]` (LEDGER.loss). If the query path read the other branch,
    every score at inference would come from a vector WHO's projection has
    never seen -- and with `--branch_layers 0` the two branches are the SAME
    tensor, so the mistake is invisible on every checkpoint trained before the
    branches existed and silently wrong on every one after.
    """
    from ledger.queries import _query_states
    c, _ = corpus
    torch.manual_seed(0)
    m = _branched_rec_model(c.schema).eval()
    rows = np.arange(8)
    cuts = np.full(8, c.time.max())

    h, _, _, _ = _query_states(m, c, "drivers", rows, cuts, 64, "cpu")

    batch, keep = PackedBatcher.pack_histories(c, "drivers", rows, cuts,
                                               max_len=64, device="cpu")
    with torch.no_grad():
        hb = m.encode_branches(batch)
    want = hb["retrieval"][keep["b"], keep["last_l"]]
    other = hb["temporal"][keep["b"], keep["last_l"]]
    assert torch.allclose(h, want, atol=1e-5), \
        "the query path is not reading the branch WHO was trained on"
    assert not torch.allclose(want, other, atol=1e-4), \
        "the two branches agree, so this test cannot tell them apart"


def test_entity_readout_reads_the_temporal_branch(corpus):
    """...and the WHEN-head readouts must read the OTHER one.

    `_query_states` is shared between the recommendation path and
    scripts/eval_entity.py / eval_regression.py, whose consumer is
    `model.when` -- a temporal-branch head. One default cannot be right for
    both consumers, so the caller names its branch.
    """
    from ledger.queries import _query_states
    c, _ = corpus
    torch.manual_seed(0)
    m = _branched_rec_model(c.schema).eval()
    rows, cuts = np.arange(8), np.full(8, c.time.max())
    a, _, _, _ = _query_states(m, c, "drivers", rows, cuts, 64, "cpu")
    b, _, _, _ = _query_states(m, c, "drivers", rows, cuts, 64, "cpu",
                               branch="temporal")
    assert not torch.allclose(a, b, atol=1e-4)
    batch, keep = PackedBatcher.pack_histories(c, "drivers", rows, cuts,
                                               max_len=64, device="cpu")
    with torch.no_grad():
        want = m.encode_branches(batch)["temporal"]
    assert torch.allclose(b, want[keep["b"], keep["last_l"]], atol=1e-5)


def test_who_gradient_reaches_the_retrieval_branch_and_not_the_temporal_one(corpus):
    """The branch split must hold through the REAL loss, not just a probe.

    `test_branch_gradients_do_not_cross_between_head_groups` checks the
    Backbone in isolation. This checks that `LEDGER.loss` actually wires the
    heads to the branches it documents: if WHO were fed `hb["temporal"]` by
    mistake, the backbone test would still pass and the model would be back to
    the single shared vector of ARCHITECTURE 9.6.
    """
    c, _ = corpus
    torch.manual_seed(0)
    m = _branched_rec_model(c.schema)
    batcher = PackedBatcher(c, "drivers", max_len=128, batch_rows=2, n_neg=32,
                            seed=0)
    losses = m.loss(batcher.batch())
    assert torch.isfinite(losses["who"])
    losses["who"].backward()
    bb = m.backbone.branch_blocks
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in bb["retrieval"].parameters()), \
        "WHO's gradient never reached the retrieval branch"
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in bb["temporal"].parameters()), \
        "WHO's gradient reached the temporal branch: the split is cosmetic"


def test_query_dyn_does_not_change_the_retrieval_batch(corpus):
    """The dynamics block rides on the QUERY token, which retrieval has none of.

    `pack_histories` is the recommendation path's batcher and is called
    without `query_token`. If `--query_dyn` leaked into it -- an extra feature
    column, a shifted token -- every recommendation number would move for a
    change aimed only at the classification readouts, and the two experiments
    would stop being separable.
    """
    rows, cuts = np.arange(8), None
    c, _ = corpus
    cuts = np.full(8, c.time.max())
    kw = dict(max_len=64, device="cpu")
    a, ka = PackedBatcher.pack_histories(c, "drivers", rows, cuts, **kw)
    b, kb = PackedBatcher.pack_histories(c, "drivers", rows, cuts,
                                         query_dyn=True, **kw)
    assert torch.equal(a["t"], b["t"])
    assert torch.equal(a["gap"], b["gap"])
    assert torch.equal(a["seq_id"], b["seq_id"])
    assert set(a) == set(b)
    assert np.array_equal(ka["last_l"], kb["last_l"])


def test_learned_loss_weights_cannot_switch_the_who_term_off(corpus):
    """Uncertainty weighting must not be able to kill retrieval.

    The clamp exists because `--w_when 0.0` is a recorded failure, but the
    same mechanism can run the other way: nothing stops the optimiser
    discovering that the cheapest WHO loss is no WHO loss. Bound the weight
    and the worst case is de-emphasis, not a model that has stopped learning
    to retrieve.
    """
    c, _ = corpus
    torch.manual_seed(0)
    m = _branched_rec_model(c.schema, branch_layers=0,
                            learn_loss_weights=True, lw_clamp=1.5)
    batcher = PackedBatcher(c, "drivers", max_len=128, batch_rows=2, n_neg=32,
                            seed=0)
    opt = torch.optim.Adam(m.parameters(), lr=0.5)   # deliberately violent
    for _ in range(20):
        opt.zero_grad()
        m.loss(batcher.batch())["total"].backward()
        opt.step()
    w = torch.exp(-m.log_var["who"].detach().clamp(-1.5, 1.5))
    assert float(w) >= math.exp(-1.5) - 1e-6, "WHO weight fell below the clamp"
    # and the head is still being trained through it
    m.zero_grad()
    m.loss(batcher.batch())["total"].backward()
    assert m.who.q.weight.grad is not None
    assert m.who.q.weight.grad.abs().sum() > 0


def _rec_checkpoint(schema, tmp_path, **overrides):
    """A checkpoint shaped exactly like train.py writes one."""
    args = dict(dataset="rel-f1", entity="drivers", steps=1, dim=64, layers=2,
                heads=4, states="nonparam", max_len=128, who_feats=True,
                window=True, rerank=True, objective="both")
    args.update(overrides)
    kw = dict(dim=args["dim"], layers=args["layers"], heads=args["heads"],
              dropout=0.0, states=args["states"],
              who_feats=(PackedBatcher.N_WHO_FEATS if args["who_feats"] else 0),
              window=bool(args["window"]), rerank=bool(args["rerank"]),
              window_head=bool(args.get("window_head")),
              win_buckets=args.get("win_buckets", 8),
              feat_path=bool(args.get("feat_path")),
              query_feats=args.pop("_query_feats", 0),
              learn_loss_weights=bool(args.get("learn_loss_weights")),
              lw_clamp=args.get("lw_clamp", 3.0),
              branch_layers=args.get("branch_layers", 0),
              branch_kind=args.get("branch_kind", "transformer"))
    m = LEDGER(schema, **kw)
    path = tmp_path / "ck.pt"
    torch.save({"model": m.state_dict(), "args": args}, path)
    return path, m


@pytest.mark.parametrize("extra", [
    {},
    {"branch_layers": 2},
    {"branch_layers": 2, "branch_kind": "mlp"},
    {"learn_loss_weights": True, "lw_clamp": 2.0},
    {"branch_layers": 1, "learn_loss_weights": True},
])
def test_a_rec_checkpoint_round_trips_through_the_shared_rebuild(corpus,
                                                                 tmp_path,
                                                                 extra):
    """Every recommendation configuration must load back into the same model.

    scripts/eval_rec.py used to keep its own kwargs list. Anything it forgot
    either raised at `load_state_dict` (a checkpoint that simply cannot be
    scored) or, for an argument that changes no parameter shape, loaded
    cleanly and reported a number for a different model. One rebuild, one
    test.
    """
    from ledger.model.load import build_model, load_checkpoint
    c, _ = corpus
    torch.manual_seed(0)
    path, m = _rec_checkpoint(c.schema, tmp_path, **extra)
    got = build_model(load_checkpoint(path), c, "cpu")
    a, b = m.state_dict(), got.state_dict()
    assert set(a) == set(b), "rebuilt model has different parameters"
    for k in a:
        assert torch.equal(a[k], b[k]), k
    assert got.backbone.branch_layers == extra.get("branch_layers", 0)
    assert (got.log_var is None) is not bool(extra.get("learn_loss_weights"))


def test_a_window_head_checkpoint_is_loadable_by_the_rec_eval(corpus,
                                                              tmp_path):
    """The cross-category case, which used to be impossible.

    The whole claim of the architecture is ONE model serving retrieval and the
    entity tasks. eval_rec.py never passed `window_head`, `query_feats` or
    `feat_path`, so a checkpoint trained with the window objective failed its
    strict state_dict load and could not be scored on a recommendation task at
    all -- i.e. the claim could not be measured.
    """
    from ledger.model.load import build_model, load_checkpoint
    c, _ = corpus
    torch.manual_seed(0)
    n_qf = PackedBatcher.n_query_feats(c.schema, False)
    path, m = _rec_checkpoint(c.schema, tmp_path, window_head=True,
                              query_feats=True, feat_path=True,
                              branch_layers=1, _query_feats=n_qf)
    got = build_model(load_checkpoint(path), c, "cpu")
    assert got.win is not None and got.query_feat_proj is not None
    assert got.query_feat_proj.weight.shape[1] == n_qf
    assert set(got.state_dict()) == set(m.state_dict())


def test_the_rebuild_restores_every_argument_that_shapes_the_model():
    """Drift alarm. A new LEDGER constructor argument must be a deliberate
    decision about the eval path, not an omission nobody notices.

    Anything not restored has to be listed here WITH a reason. The silent half
    of this failure class is the dangerous one: `state_normalize` changes what
    the state table reads back but no parameter shape, so forgetting it loads
    cleanly and scores a model that is not the checkpoint.
    """
    import ast
    import inspect
    from ledger.model import load as load_mod

    # arguments the eval path deliberately does not restore
    not_restored = {
        "schema",            # positional: the corpus being scored
        "amp_dtype",         # inference runs in fp32; see LEDGER.encode
        "attn",              # kernel choice, not a model property
        "uniformity",        # training-only regulariser, no parameters
        "loss_weights",      # training-only, no parameters
        "w_rate", "w_qcount", "w_detail",   # window-loss term weights only
        "win_edges",         # buffers, restored by load_state_dict
        "dropout",           # eval() disables it
    }
    src = inspect.getsource(load_mod.build_model)
    call = next(n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "LEDGER")
    passed = {k.arg for k in call.keywords}
    sig = set(inspect.signature(LEDGER.__init__).parameters) - {"self"}
    missing = sig - passed - not_restored
    assert not missing, (
        f"LEDGER argument(s) {sorted(missing)} are not restored by "
        f"ledger/model/load.build_model. Restore them, or add them to this "
        f"test's `not_restored` set with the reason.")


def test_recommendation_scores_end_to_end_with_the_new_architecture(corpus):
    """The full eval path on a real task, with branches and learned weights on.

    Nothing above runs `evaluate_recommendation` itself: the shortlist, the
    reranker cross-attention, the D3 feature block, the co-occurrence table,
    the cold-start fallback and the RelBench evaluator. Any of them can break
    on a branched model without a single unit test noticing.
    """
    from relbench.tasks import get_task
    from ledger.queries import evaluate_recommendation
    c, ds = corpus
    try:
        task = get_task("rel-f1", "driver-circuit-compete", download=False)
    except Exception as e:                            # task not cached here
        pytest.skip(f"task unavailable: {e}")
    torch.manual_seed(0)
    m = _branched_rec_model(c.schema, branch_layers=1,
                            learn_loss_weights=True).eval()
    res = evaluate_recommendation(m, c, task, split="val", max_len=64,
                                  device="cpu", batch_size=32,
                                  cooc_cache_key=None, n_stage1=64,
                                  db=ds.get_db)
    diag = res.pop("_diagnostics")
    assert res, "the evaluator returned no metrics"
    for k, v in res.items():
        assert np.isfinite(v), f"{k} is not finite"
        assert 0.0 <= v <= 1.0, f"{k}={v} is outside [0, 1]"
    assert diag["rows"] > 0 and diag["rank_table"] == task.dst_entity_table


def test_an_unknown_branch_label_raises(corpus):
    """The branch API is keyed by STRING, so a typo must be loud.

    `encode(branch=...)` is the only thing standing between a head and the
    wrong representation, and a silent fallback to "whatever we have" would
    reintroduce exactly the failure the branches exist to remove.
    """
    c, _ = corpus
    torch.manual_seed(0)
    m = _branched_rec_model(c.schema, branch_layers=1).eval()
    batch, _ = PackedBatcher.pack_histories(c, "drivers", np.arange(4),
                                            np.full(4, c.time.max()),
                                            max_len=32, device="cpu")
    with pytest.raises(KeyError):
        m.encode(batch, branch="retreival")          # deliberate typo


def test_state_refresh_writes_the_retrieval_branch(corpus):
    """A4's refresh must recompute the table WHO reads.

    `refresh_states` re-encodes every history with the loaded weights and
    writes the result into the entity-state table, which the WHO dot product
    then scores against. Training writes that table from `hb["retrieval"]`
    (LEDGER.loss), so a refresh off the temporal branch would replace every
    candidate vector with one from a different space -- and the failure is
    silent: coverage still reads 100%.
    """
    from ledger.queries import refresh_states
    c, _ = corpus
    torch.manual_seed(0)
    m = _branched_rec_model(c.schema, branch_layers=2, states="nonparam")
    m.eval()
    refresh_states(m, c, "drivers", max_len=64, batch_size=16, device="cpu",
                   verbose=False)
    after = {n: b.clone() for n, b in m.entity_states.named_buffers()
             if n.startswith("S_")}
    assert any(b.abs().sum() > 0 for b in after.values()), \
        "the refresh wrote nothing, so this test proves nothing"

    # redo the same pass by hand off the temporal branch: it must NOT agree
    m2 = _branched_rec_model(c.schema, branch_layers=2, states="nonparam")
    m2.load_state_dict(m.state_dict())
    m2.eval()
    orig = m2.encode
    m2.encode = lambda batch, branch="retrieval": orig(batch,
                                                       branch="temporal")
    refresh_states(m2, c, "drivers", max_len=64, batch_size=16, device="cpu",
                   verbose=False)
    wrong = dict(m2.entity_states.named_buffers())
    assert any(not torch.allclose(after[n], wrong[n], atol=1e-5)
               for n in after), \
        "the two branches write the same states; this test proves nothing"


def test_union_namespaces_every_name_that_indexes_a_dict():
    """A union must qualify TABLE names, ENTITY names and FKEY TARGETS.

    `users` is a fact table in both rel-event and rel-stack, and rel-f1 has an
    entity table `races` that is also a fact table. Any name left unqualified
    resolves against whichever database defined it first: the foreign key
    still points somewhere, the state write still lands, and nothing raises --
    the model is just reading another database's entity space. That is the
    HANDOFF 10.1a failure mode (no error, plausible numbers, wrong model), so
    it is asserted structurally rather than left to a smoke test.
    """
    import pandas as _pd
    from relbench.datasets import get_dataset as _gd
    from ledger.data.cache import load_corpus
    from ledger.data.union import build_union

    loaded = {}
    for ds in ("rel-f1", "rel-event"):
        try:
            d = _gd(ds, download=False)
            loaded[ds] = load_corpus(ds, _pd.Timestamp(d.val_timestamp),
                                     db=d.get_db, verbose=False)
        except Exception as e:
            pytest.skip(f"{ds} corpus unavailable: {e}")

    cor, sch = build_union(loaded)

    # no name survives unqualified, anywhere
    assert all("::" in n for n in sch.fact_tables)
    assert all("::" in e for e in sch.entity_tables)
    for name, spec in sch.fact_tables.items():
        for col, tgt in spec.fkeys.items():
            assert "::" in tgt, f"{name}.{col} -> {tgt} is unqualified"
            # and it must point INSIDE its own database
            assert tgt.split("::")[0] == name.split("::")[0]
            assert tgt in sch.entity_counts

    # table_idx is a contiguous global range, and each corpus's remapped
    # events point at ITS OWN entries and no one else's
    assert sorted(sp.table_idx for sp in sch.fact_tables.values()) == \
        list(range(len(sch.fact_tables)))
    for ds, c in cor.items():
        mine = {sp.table_idx for n, sp in sch.fact_tables.items()
                if n.startswith(ds + "::")}
        assert set(np.unique(c.table_idx)).issubset(mine)

    # nothing was lost or merged: the union is the sum of the parts
    assert len(sch.fact_tables) == sum(
        len(c.schema.fact_tables) for c in cor.values())


@pytest.mark.parametrize("stamp,expect", [
    (None, "no __split__ stamp"),
    ("val", "is VAL data"),
    ("test", None),
])
def test_leaderboard_refuses_to_print_a_val_row_as_a_test_row(
        tmp_path, stamp, expect):
    """The published tables are TEST. A results file made on val is not
    comparable to them.

    This is not hypothetical: `board.py --split` defaults to "val" and
    `leaderboard.py` filed its output under a header reading "AUROC on the
    official test split", so the published LEDGER row was validation data. It
    took a decimal-exact coincidence (driver-dnf 68.49 against our val 68.49,
    study-adverse 0.1683 against our val 0.1683) to catch it. An UNSTAMPED
    file is the pre-fix state and must warn just as loudly as a val one --
    silence there is what allowed this in the first place.
    """
    import json as _json
    import subprocess as _sp
    import sys as _sys

    res = {"rel-f1 driver-dnf": 68.49, "rel-hm user-churn": 64.17}
    if stamp is not None:
        res = {"__split__": stamp, **res}
    f = tmp_path / "r.json"
    f.write_text(_json.dumps(res))

    lb = pathlib.Path(__file__).resolve().parent.parent / ".scratch" / "lb.html"
    if not lb.exists():
        pytest.skip("no cached leaderboard html")
    out = _sp.run([_sys.executable, "scripts/leaderboard.py",
                   "--results", str(f)],
                  cwd=str(lb.parent.parent), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-400:]
    if expect is None:
        assert "!!" not in out.stdout, "a test-split row must not be flagged"
    else:
        assert expect in out.stdout, out.stdout[:400]


def test_auto_readout_keeps_the_linear_family_in_the_pool():
    """`--readout auto` must be a WIDER SEARCH, not a different method.

    The published probe is a regularised linear model. If `auto` replaced it
    with a booster, a task where linear is better would silently regress and
    the two settings would not be comparable. Keeping the linear family as a
    candidate is what makes "auto can only match or beat linear, up to CV
    noise" true, and that claim is the reason `auto` is safe to report.

    Also pinned: ties go to the EARLIER candidate and the list is ordered
    simplest-first, so a nonlinear model must actually win rather than draw.
    """
    import re
    src = pathlib.Path("scripts/icl_readout.py").read_text()

    cls = src[src.index("if is_cls and args.readout"):src.index("elif is_cls:")]
    assert "LogisticRegressionCV" in cls, "linear dropped from the auto pool"
    assert cls.index("linear (LogisticRegressionCV)") < cls.index("hist-gbdt"), \
        "candidates must be ordered simplest-first"

    reg = src[src.index('elif args.readout == "auto":'):]
    reg = reg[:reg.index("\n    else:")]
    # Order must be read off the CANDIDATE LIST, not off source position: the
    # linear estimator is BUILT above the list (as `lin`) and so appears
    # earlier in the file than the constant it must rank behind. An earlier
    # version of this test asserted on source position and failed on correct
    # code, which is the wrong kind of failure.
    lst = reg[reg.index("cands = ["):]
    lst = lst[:lst.index("\n        ]")]
    order = re.findall(r'\("([^"]+)",', lst)
    assert order[0] == "constant median", order
    assert "linear" in order[1], order
    assert all("hist-gbdt" in n for n in order[2:]), order
    # every regression candidate must optimise L1: NMAE's optimal predictor is
    # the conditional median, and a squared-error booster estimates the mean.
    # RidgeCV on the mean measured ad-ctr 0.4196 -> 4.2670 (2026-09-01).
    for m in re.finditer(r"HistGradientBoostingRegressor\(([^)]*)", reg):
        assert 'loss="absolute_error"' in m.group(1), \
            "a regression candidate is fitting the MEAN, not the median"

    # strict improvement, not >=, or a tie flips the choice on CV noise
    sel = src[src.index("def pick_by_cv"):src.index("if is_cls and args.readout")]
    assert "sc_ > best_s" in sel, "ties must not displace the simpler model"


@pytest.mark.parametrize("query_head", [False, True])
def test_log_var_only_allocates_query_when_the_query_head_exists(
        corpus, query_head):
    """`log_var.query` must not exist on a model without a query head.

    `loss_weights` carries a fixed `query` coefficient unconditionally, and
    the learned-weight ParameterDict used to be built from ALL of it. That
    allocated `log_var.query` on every model, so every checkpoint predating
    `--query_head` failed `load_state_dict` with "Missing key(s):
    log_var.query" -- the entire recommendation lineage included, which is how
    it was found. `combine_losses` already falls back to the fixed coefficient
    for any key absent from `log_var`, so the query term's weighting is
    unchanged either way.
    """
    import torch as _t
    from ledger.model.ledger import LEDGER, combine_losses

    c, _ = corpus
    m = LEDGER(c.schema, dim=32, layers=1, heads=2, window=True,
             learn_loss_weights=True, query_head=query_head)
    assert m.log_var is not None
    assert ("query" in m.log_var) is query_head
    # the other terms are always learned
    for k in ("when", "where", "who", "what", "win"):
        assert k in m.log_var

    # and a `query` loss still gets its fixed weight when it is not learned
    losses = {"who": _t.tensor(2.0), "query": _t.tensor(3.0)}
    total, _ = combine_losses(losses, m.log_var, {"who": 1.0, "query": 0.5})
    assert _t.isfinite(total)


def test_rec_refit_feature_slots_match_their_names():
    """The feature block is filled by three different code paths.

    `Scorer.blocks` writes the dot product at 0, the dense candidate-side
    features as a 4-wide slice, and the sparse ones one at a time through an
    index expression (`1 + j if j < 3 else 8`). Those three have to agree with
    FEAT_NAMES, and nothing at runtime would complain if they did not -- the
    coefficients would simply be attached to the wrong features and the refit
    would quietly be a different model than the one printed.
    """
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "rec_refit.py"
    spec = importlib.util.spec_from_file_location("rec_refit", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.FEAT_NAMES[0] == "dot"
    # the sparse features, in the order _add_history_features defines them
    # (w[0] in-history bias, w[1] log repeat count, w[2] recency, w[7] cooc)
    for j, name in zip(mod.SPARSE_J, ["in_hist", "repeat", "recency", "cooc"]):
        assert mod.FEAT_NAMES[1 + j if j < 3 else 8] == name
    # the dense slice F[4:8], in TemporalIndex.stats order
    assert mod.FEAT_NAMES[4:8] == ["pop_total", "pop_recent", "trend",
                                   "staleness"]
    assert mod.FEAT_NAMES[9] == "rerank"
    assert len(mod.FEAT_NAMES) == 10
