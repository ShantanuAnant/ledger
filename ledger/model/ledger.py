"""LEDGER: tokenizer + backbone + four heads, and the joint loss.

Scope decisions:
- The entity-state table is pluggable: `states="learned"` reproduces v0
  (nn.Embedding + AdamW), `states="nonparam"` is the detached bfloat16 EMA of
  ARCHITECTURE.md section 6 / HANDOFF section 4.1. Both live behind one
  interface in states.py so the choice can be measured.
- Text columns are excluded by the schema layer for now (rel-f1 has none
  that matter); the sentence-embedding route plugs into ColumnSpec later.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..data.corpus import EventCorpus
from .backbone import Backbone
from .heads import WhenHead, WhereHead, WhoHead, WhatHead, RerankHead
from .states import build_states
from .window import WindowHead


def combine_losses(losses: dict, log_var=None, fixed: dict | None = None,
                   clamp: float = 3.0):
    """Per-head losses -> the scalar that is optimised, plus the weights used.

    Two regimes, and the difference is the subject of an experiment:

    * `log_var is None` -- the historical behaviour, `sum_k w_k L_k` with
      `w_k` the hand-set `--w_*` constants.
    * otherwise -- uncertainty weighting, `sum_k [exp(-s_k) L_k + s_k]` with
      `s_k = log sigma_k^2` learned. See LEDGER.__init__ for why the `+ s_k`
      term is not optional and why `s` is clamped.

    A key with no entry in `log_var` (the `unif` regulariser) keeps its fixed
    coefficient in both regimes: it is a penalty, not a likelihood, so it has
    no observation noise to infer.

    Returned as a free function so the arithmetic is testable without
    building a schema, a corpus and a full model.
    """
    fixed = fixed or {}
    total, weights = None, {}
    for k, v in losses.items():
        if log_var is not None and k in log_var:
            s = log_var[k].clamp(-clamp, clamp)
            term = torch.exp(-s) * v + s
            weights[k] = torch.exp(-s.detach())
        else:
            w = fixed.get(k, 1.0)
            term = w * v
        total = term if total is None else total + term
    return total, weights


class EventTokenizer(nn.Module):
    """event -> R^dim  (table embedding + column features + link summaries)."""

    def __init__(self, dim: int, schema, entity_states):
        super().__init__()
        self.table_emb = nn.Embedding(schema.num_fact_tables, dim)
        self.entity_states = entity_states  # dict: table -> nn.Embedding
        self.num_proj = nn.ModuleDict()
        self.cat_emb = nn.ModuleDict()
        self.link_proj = nn.ModuleDict()
        for name, spec in schema.fact_tables.items():
            n_num = sum(c.kind == "numeric" for c in spec.columns)
            cats = [c for c in spec.columns if c.kind == "categorical"]
            if n_num:
                self.num_proj[name] = nn.Linear(n_num, dim)
            if cats:
                self.cat_emb[name] = nn.ModuleList(
                    nn.Embedding(c.cardinality, dim) for c in cats)
            for slot, target in spec.fkeys.items():
                self.link_proj[f"{name}__{slot}"] = nn.Linear(dim, dim,
                                                             bias=False)

    def forward(self, table_name, table_idx, feat_num, feat_cat, links,
                fkey_targets):
        # All tensors are for events of ONE fact table: [N, ...]
        x = self.table_emb(table_idx)
        if table_name in self.num_proj and feat_num.numel():
            x = x + self.num_proj[table_name](feat_num)
        if table_name in self.cat_emb:
            for j, emb in enumerate(self.cat_emb[table_name]):
                x = x + emb(feat_cat[:, j].long())
        for j, (slot, target) in enumerate(fkey_targets.items()):
            e = links[:, j]
            ok = e >= 0
            if ok.any():
                s = self.entity_states.read(target, e.clamp(min=0))
                s = s * ok.unsqueeze(-1)
                x = x + self.link_proj[f"{table_name}__{slot}"](s)
        return x


class LEDGER(nn.Module):
    def __init__(self, schema, dim: int = 256, layers: int = 8, heads: int = 8,
                 dropout: float = 0.1, states: str = "learned",
                 state_momentum: float = 0.1, state_normalize: bool = False,
                 state_center: bool = False, state_reduce: str = "mean",
                 learned_max: int = 200_000,
                 learned_tables: list | None = None,
                 uniformity: float = 0.0, loss_weights: dict | None = None,
                 who_feats: int = 0, window: bool = False,
                 rerank: bool = False, amp_dtype=None,
                 attn: str = "auto", no_ema_write: bool = False,
                 window_head: bool = False, win_edges: dict | None = None,
                 win_buckets: int = 8, query_feats: int = 0,
                 next_event: bool = True, win_balance: str = "legacy",
                 w_rate: float = 1.0, w_qcount: float = 1.0,
                 w_detail: float = 1.0, feat_path: bool = False,
                 branch_layers: int = 0, branch_kind: str = "transformer",
                 learn_loss_weights: bool = False,
                 lw_clamp: float = 3.0, query_head: bool = False,
                 rate_balance: str = "none", rate_zero_scale: float = 1.0):
        super().__init__()
        self.schema = schema
        # WINDOW-PRIMARY MODE. With `next_event=False` the backbone is trained
        # SOLELY on the window-aggregate objective (ARCHITECTURE 8.2a). This
        # is not the same as setting the four weights to zero: the heads are
        # never run, so they cost nothing and -- more importantly -- cannot
        # shape the trunk through a stray gradient path.
        #
        # The motivation is a measurement, not a preference. A frozen probe of
        # the next-event hidden state on rel-stack user-badge reaches 74.2
        # AUROC; two scalars (n_events, elapsed) reach 81.5; 33 count features
        # reach 84.8; and the hidden state ADDED to those 33 makes them worse.
        # Nothing in a next-event objective pressures the state to retain
        # "how many badges so far", so it does not -- and every classification
        # and regression readout is a functional of exactly that.
        self.next_event = next_event
        # Mixed precision is applied to the TOKENIZER + BACKBONE only, and the
        # hidden states are cast back to float32 before any head sees them.
        # That is where essentially all the FLOPs are (attention + MLP), while
        # the heads contain every numerically delicate operation in the model:
        # the WHEN survival term is `log(erfc(z))` in the far tail, and
        # `erfc` is not on autocast's promotion list, so a bf16 z would
        # silently floor the log-survival for censored events -- which is
        # exactly the term churn and regression read out (ARCHITECTURE 8.1).
        # The sampled softmax is likewise a logsumexp over ~264 candidates
        # whose logQ correction spans ~11 nats.
        self.amp_dtype = amp_dtype
        self.no_ema_write = no_ema_write
        # Per-term loss weights (ARCHITECTURE.md 7: "equal weights to start").
        #
        # Equal weights are NOT equal gradients. WHEN is a negative
        # log-likelihood over a continuous gap in seconds and sits around
        # 9-10 nats; WHO sits at 3-5 and WHAT near 1. So WHEN supplies most of
        # the gradient reaching the shared backbone, which is a candidate
        # mechanical explanation for the ~20k-step WHO warm-up observed on
        # rel-hm (RESEARCH.md 2026-08-23) -- an explanation that was attributed
        # to scale without being tested against this one. Configurable so the
        # two can be separated.
        self.loss_weights = {"when": 1.0, "where": 1.0, "who": 1.0,
                             "what": 1.0, "rerank": 1.0, "win": 1.0,
                             "query": 1.0,
                             "windense": 1.0,
                             **(loss_weights or {})}
        # -- LEARNED loss weights (Kendall et al. uncertainty weighting) ----
        #
        # `--w_when 0.1` is a hand-picked constant that ARCHITECTURE 9.6
        # measures as a direct trade of one category against another (86.0 ->
        # 63.7 AUROC to buy 3.10 -> 5.78 MAP). A constant cannot be right at
        # every step either: the two categories peak at opposite ends of a
        # run. So make it a parameter and let the data set it.
        #
        # The naive version of this -- a bare `nn.Parameter` multiplying each
        # term -- has a trivial optimum at w = 0 for every head, because the
        # cheapest way to reduce `sum w_k L_k` is to send every w_k to zero.
        # Uncertainty weighting is the form that does not collapse:
        #
        #     total = sum_k [ exp(-s_k) * L_k + s_k ],   s_k = log sigma_k^2
        #
        # The `+ s_k` is the log-normalizer of the likelihood the weighting is
        # derived from, and it charges for down-weighting: driving s_k up to
        # kill a term costs s_k directly. The stationary point is
        # exp(-s_k) = 1 / (2 L_k)-ish, i.e. weight inversely proportional to
        # how large and how irreducible a term is -- which is the balance
        # ARCHITECTURE 9.6 asks for and could not hand-tune.
        #
        # `s` is CLAMPED because w_when = 0.0 is a known failure mode, not a
        # hypothetical one: RESEARCH.md 2026-08-23 records the WHEN head
        # diverging to 21.9 without gradient, and WHEN is what the
        # classification readout consumes. The clamp bounds the learned weight
        # to exp(+/- lw_clamp), so a term can be de-emphasised but never
        # switched off.
        self.learn_loss_weights = learn_loss_weights
        self.lw_clamp = lw_clamp
        self.log_var = None
        if learn_loss_weights:
            # Initialise AT the configured constants, so a learned-weight run
            # starts exactly where its fixed-weight control is and the
            # comparison is about the schedule, not the starting point.
            # `query` is only a LOSS when the query head exists, and a
            # Parameter allocated for a term that is never computed is a key
            # every earlier checkpoint lacks: `load_state_dict` is strict, so
            # allocating it unconditionally made every pre-`--query_head`
            # checkpoint fail to load with "Missing key(s): log_var.query"
            # -- including the whole recommendation lineage. The fixed
            # coefficient stays in `loss_weights` either way, and
            # `combine_losses` falls back to it for any key absent from
            # `log_var`, so nothing about the query term's weighting changes.
            _lw = {k: w for k, w in self.loss_weights.items()
                   if k != "query" or query_head}
            self.log_var = nn.ParameterDict({
                k: nn.Parameter(torch.tensor(
                    -math.log(max(float(w), 1e-4)), dtype=torch.float32))
                for k, w in _lw.items()})
        kw = {}
        if states in ("nonparam", "hybrid"):
            kw = dict(momentum=state_momentum, normalize=state_normalize,
                      center=state_center, reduce=state_reduce)
            if states == "hybrid":
                kw["learned_max"] = learned_max
                kw["learned_tables"] = learned_tables
        self.entity_states = build_states(states, schema.entity_counts, dim, **kw)
        # variant 3: explicit pressure to spread candidate representations
        # apart. Only the WHO q/c projections receive gradient otherwise, so
        # nothing opposes the collapse measured on 2026-08-22.
        self.uniformity = uniformity
        self.tokenizer = EventTokenizer(dim, schema, self.entity_states)
        self.backbone = Backbone(dim, layers, heads, dropout, attn=attn,
                                 branch_layers=branch_layers,
                                 branch_kind=branch_kind)
        self.when = WhenHead(dim)
        self.where = WhereHead(dim, schema.num_fact_tables)
        self.who = WhoHead(dim, n_feats=who_feats, window=window)
        # n_feats=0 deliberately: the interaction features are already an
        # additive term inside `base_scores`, so re-feeding them here would
        # only duplicate them -- and it would force the eval path to rebuild
        # the whole feature block for the shortlist. The reranker's job is the
        # part a dot product cannot express: which SPECIFIC past events this
        # candidate should attend to.
        self.rerank = RerankHead(dim, n_feats=0) if rerank else None
        self.what = WhatHead(dim, schema)
        # The window head owns the query token: the token is what carries the
        # query TIME into the backbone, and the head is what turns the state
        # at that token into the aggregates every entity task asks for.
        self.win = (WindowHead(dim, schema, edges=win_edges,
                               n_buckets=win_buckets, balance=win_balance,
                               w_rate=w_rate, w_qcount=w_qcount,
                               w_detail=w_detail, n_qfeat=query_feats,
                               feat_path=feat_path, query_head=query_head,
                               rate_balance=rate_balance,
                               rate_zero_scale=rate_zero_scale)
                    if window_head else None)
        self.query_emb = (nn.Parameter(torch.randn(dim) * 0.02)
                          if window_head else None)
        # Per-table history counters fed straight into the query token. Zero
        # initialised, so a model with the flag on starts numerically where
        # the featureless one is and can only be moved away from it by data.
        self.query_feat_proj = None
        if window_head and query_feats:
            self.query_feat_proj = nn.Linear(query_feats, dim, bias=False)
            nn.init.zeros_(self.query_feat_proj.weight)

    def num_out_tables(self):
        """Fact tables whose numeric columns the WHAT head can decode."""
        return set(self.what.num_out.keys())

    def encode(self, batch, branch: str = "retrieval") -> torch.Tensor:
        """batch (see batching.py) -> hidden states [B, L, dim], float32.

        `branch` selects which head group's view to return (see
        Backbone.BRANCHES). The default is "retrieval" because that is what
        `queries.py` and every WHO diagnostic want; the entity-task readouts
        must ask for "temporal" explicitly. With `--branch_layers 0` the two
        are the same tensor and the argument is inert.
        """
        return self.encode_branches(batch)[branch]

    def encode_branches(self, batch) -> dict:
        """As `encode`, but returns every branch from ONE forward pass.

        The trunk is the expensive part and it is shared, so computing both
        views together costs one trunk pass plus the branch layers -- not two
        forward passes. `loss` needs both, so it must call this.
        """
        dev = batch["t"].device
        use_amp = self.amp_dtype is not None and dev.type == "cuda"
        with torch.autocast("cuda", dtype=self.amp_dtype or torch.bfloat16,
                            enabled=use_amp):
            B, L = batch["t"].shape
            dt = self.amp_dtype if use_amp else torch.float32
            tokens = torch.zeros(B, L, self.backbone.dim, device=dev,
                                 dtype=dt)
            for name, ev in batch["per_table"].items():
                spec = self.schema.fact_tables[name]
                tok = self.tokenizer(name, ev["table_idx"], ev["feat_num"],
                                     ev["feat_cat"], ev["links"], spec.fkeys)
                tokens[ev["b"], ev["l"]] = tok.to(dt)
            qp = batch.get("query_pos")
            if qp is not None and self.query_emb is not None:
                # The query token carries no columns and no links -- only its
                # identity plus the (time, gap) the TimeEncoder adds to every
                # token. `gap` for it is the elapsed silence since the
                # entity's last event, which is precisely the quantity the
                # backbone previously never saw.
                qtok = self.query_emb.unsqueeze(0).expand(len(qp["b"]), -1)
                if (self.query_feat_proj is not None
                        and qp.get("feats") is not None):
                    qtok = qtok + self.query_feat_proj(qp["feats"])
                tokens[qp["b"], qp["l"]] = qtok.to(dt)
            hb = self.backbone(tokens, batch["t"], batch["gap"],
                               batch["seq_id"])
        return {k: v.float() for k, v in hb.items()}

    @torch.no_grad()
    def update_states(self, batch, h) -> None:
        """Write each event's hidden state into every entity it links to.

        This is what makes the non-parametric table a *summary of history
        produced by the model* rather than a free parameter. An entity that
        appears in many events accumulates an EMA over the hidden states of
        those events. No-op for LearnedStates.

        Note both endpoints are written: for a transaction token in customer
        c's sequence we update S[customer=c] AND S[article=a]. Without the
        second write, destination entities would never receive a state and the
        WHO head could not rank them -- see queries.py.
        """
        # BUG FIX 2026-08-24: this gate read `!= "nonparam"`, so with
        # `--states hybrid` it returned before writing ANYTHING and the EMA
        # half of the table stayed all-zero for the whole run
        # (`state coverage: customer 0.0%`). HybridStates.write already
        # no-ops on its learned tables, so hybrid belongs here.
        #
        # The rel-amazon run that reached MAP 2.2882 ran under the buggy
        # behaviour -- learned product embeddings plus permanently zero
        # customer states. That is a legitimate configuration and it is the
        # best result the project has produced, so it stays reachable via
        # `--no_ema_write` rather than being lost to the fix.
        if self.no_ema_write:
            return
        if self.entity_states.kind not in ("nonparam", "hybrid"):
            return
        # The sequence entity gets "last" (its state IS its most recent hidden
        # state -- that is what queries.py reads); every other endpoint gets
        # "mean". See NonParametricStates.write.
        seq_ent = batch.get("seq_entity")
        for name, ev in batch["per_table"].items():
            spec = self.schema.fact_tables[name]
            hs = h[ev["b"], ev["l"]]
            for j, target in enumerate(spec.fkeys.values()):
                e = ev["links"][:, j]
                ok = e >= 0
                if ok.any():
                    # global order key: rows are packed independently, so a
                    # bare `l` would tie across batch rows
                    key = ev["b"][ok] * batch["t"].shape[1] + ev["l"][ok]
                    self.entity_states.write(
                        target, e[ok], hs[ok], pos=key,
                        reduce=("last" if target == seq_ent else "mean"))

    def _gather_history(self, h, hist):
        """hist [2, T, R] of (batch_row, token_pos); -1 marks padding.

        -> (states [T, R, d], pad_mask [T, R]) for the reranker's
        cross-attention over each query's own recent tokens.
        """
        b, l = hist[0], hist[1]
        pad = l < 0
        states = h[b.clamp(min=0), l.clamp(min=0)]
        return states * (~pad).unsqueeze(-1), pad

    def loss(self, batch) -> dict:
        hb = self.encode_branches(batch)
        # The branch assignment IS the fix of ARCHITECTURE 9.6: WHO pulls the
        # shared trunk toward "which entity comes next", WHEN/WINDOW pull it
        # toward "how much happens and when". Below the branch point they
        # still share; above it they no longer have to agree.
        h = hb["retrieval"]            # WHO, reranker, entity-state writes
        hg = hb["temporal"]            # WHEN, WHERE, WHAT, WINDOW
        tgt = batch["target"]          # aligned targets for each position
        hs = h[tgt["b"], tgt["l"]]     # states that must predict a next event
        gs = hg[tgt["b"], tgt["l"]]    # temporal branch: WHEN/WHERE/WHAT
        losses = {}
        if self.next_event:
            losses = {
                "when": self.when.loss(gs, tgt["gap"], tgt["censored"]),
                "where": self.where.loss(gs[~tgt["censored"]],
                                         tgt["table_idx"][~tgt["censored"]]),
            }
        # WHO: one sampled-softmax per destination entity table. Candidate row
        # indices are table-local, so each group MUST be read from its own
        # state buffer. Groups are combined as a row-count-weighted mean, so
        # the term keeps the same scale as the single-group v0 loss.
        if self.next_event and tgt["who"]:
            who_tot, unif_tot, n_tot, rr_tot = None, None, 0, None
            for tbl, g in tgt["who"].items():
                cand = self.entity_states.read(tbl, g["cands"])
                n = len(g["pos"])
                hq = hs[g["pos"]]
                valid = g.get("valid")
                pm = g.get("pos_mask")
                if pm is not None and valid is not None:
                    pm = pm & valid
                term = self.who.loss(hq, cand, g.get("feats"),
                                     g.get("win_days"), g.get("log_q"),
                                     pos_mask=pm, valid_mask=valid) * n
                who_tot = term if who_tot is None else who_tot + term
                if self.rerank is not None and pm is not None:
                    hist, hmask = self._gather_history(h, g["hist"])
                    base = self.who.scores(
                        hq, cand, g.get("feats"), g.get("win_days"),
                        g.get("log_q"))
                    if valid is not None:
                        base = base.masked_fill(~valid, float("-inf"))
                    r = self.rerank.loss(cand, hist, hmask, base, pm) * n
                    rr_tot = r if rr_tot is None else rr_tot + r
                if self.uniformity > 0:
                    # Push projected candidates apart (Wang & Isola
                    # uniformity): log E exp(-2||u-v||^2).
                    c = torch.nn.functional.normalize(
                        self.who.c(cand.reshape(-1, cand.shape[-1])), dim=-1)
                    c = c[torch.randperm(len(c), device=c.device)[:512]]
                    d = torch.cdist(c, c).pow(2)
                    iu = torch.triu(torch.ones_like(d, dtype=torch.bool), 1)
                    u = d[iu].mul(-2).exp().mean().clamp_min(1e-12).log() * n
                    unif_tot = u if unif_tot is None else unif_tot + u
                n_tot += n
            losses["who"] = who_tot / n_tot
            if rr_tot is not None:
                losses["rerank"] = rr_tot / n_tot
            if unif_tot is not None:
                losses["unif"] = self.uniformity * unif_tot / n_tot
        # WHAT: grouped per fact table by the batcher
        if self.next_event:
            what = h.new_zeros(())
            for name, w in tgt["what"].items():
                what = what + self.what.loss(gs[w["pos"]], name,
                                             w["feat_num"], w["feat_cat"])
            losses["what"] = what
        win = batch.get("window")
        if self.win is not None and win is not None and win["n_query"]:
            losses["win"] = self.win.loss(hg[win["b"], win["l"]], win)
            # RANDOM-QUERY supervision (ledger/qsample.py). Queries are drawn
            # by the trainer and handed in on the batch, so the model stays
            # free of the sampler's RNG and a run is reproducible from its
            # seed alone.
            specs = batch.get("qspecs")
            if self.win.query_head is not None and specs:
                st = self.win.state(
                    hg[win["b"], win["l"]], win["elapsed"], win["horizon"],
                    win["feats"] if self.win.feat_path else None)
                qloss, nq_used = self.win.query_loss(
                    st, win, specs, self.win.schema)
                if nq_used:
                    losses["query"] = qloss
        # Dense window supervision on ordinary event tokens (batching.py).
        # Logged as its own term so the supervision-density experiment is
        # readable off the curve rather than inferred from the total.
        wd = batch.get("window_dense")
        if self.win is not None and wd is not None and wd["n_query"]:
            losses["windense"] = self.win.loss_dense(hg[wd["b"], wd["l"]], wd)
        if not losses:
            # every term was gated off: an all-zero graph would train nothing
            # and report a clean-looking 0.0 loss curve for the whole run
            raise RuntimeError(
                "no loss terms: next_event=False requires the window head, "
                "and this batch carried neither query tokens nor dense "
                "window targets")
        # `total` is what we optimise; the per-term entries stay UNWEIGHTED so
        # logged curves remain comparable across weightings.
        total, weights = combine_losses(losses, self.log_var,
                                        self.loss_weights, self.lw_clamp)
        losses["total"] = total
        # The realised weights, logged so a run's schedule is inspectable
        # rather than inferred. Detached: diagnostics, not objective.
        for k, w in weights.items():
            losses[f"w_{k}"] = w
        self.update_states(batch, h)
        return losses
