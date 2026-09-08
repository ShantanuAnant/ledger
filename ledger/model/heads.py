"""The four prediction heads and their losses (ARCHITECTURE.md section 7).

Given the entity state h_k (backbone output at token k), predict event k+1:
  WHEN  - log-normal mixture over the time gap, with right-censoring support
  WHERE - softmax over fact tables
  WHO   - sampled-softmax retrieval against the entity-state table
  WHAT  - per-column decoders (numeric MSE, categorical cross-entropy)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2 * math.pi)


class WhenHead(nn.Module):
    """Mixture of log-normals over Δt (seconds). Intensity-free; stable.

    loss(gap)          = -log p(gap)                  observed next event
    censored_loss(gap) = -log P(Δt > gap) = -log S(gap)  survival; this term
    is what churn queries read out, so it is exercised in training, not only
    at inference (ARCHITECTURE.md 7.1).
    """

    def __init__(self, dim: int, n_mix: int = 8):
        super().__init__()
        self.n_mix = n_mix
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                 nn.Linear(dim, 3 * n_mix))

    def _params(self, h):
        raw = self.net(h)
        logit, mu, log_sig = raw.chunk(3, dim=-1)
        return (F.log_softmax(logit, -1), mu,
                log_sig.clamp(-5, 5))

    def log_prob(self, h, gap):
        log_w, mu, log_sig = self._params(h)
        x = torch.log(gap.clamp(min=1.0)).unsqueeze(-1)
        sig = log_sig.exp()
        comp = (-0.5 * ((x - mu) / sig) ** 2 - log_sig
                - 0.5 * LOG_2PI - x)  # -x: Jacobian of log transform
        return torch.logsumexp(log_w + comp, dim=-1)

    def log_survival(self, h, gap):
        """log P(Δt > gap); used for censored events and churn queries."""
        log_w, mu, log_sig = self._params(h)
        x = torch.log(gap.clamp(min=1.0)).unsqueeze(-1)
        z = (x - mu) / log_sig.exp()
        # log(1 - Phi(z)) via the complementary error function, stable tail
        log_sf = torch.log(torch.special.erfc(z / math.sqrt(2)).clamp_min(1e-30)) - math.log(2)
        return torch.logsumexp(log_w + log_sf, dim=-1)

    def loss(self, h, gap, censored):
        lp = torch.where(censored, self.log_survival(h, gap),
                         self.log_prob(h, gap))
        return -lp.mean()


class WhereHead(nn.Module):
    def __init__(self, dim: int, num_tables: int):
        super().__init__()
        self.proj = nn.Linear(dim, num_tables)

    def loss(self, h, target):
        return F.cross_entropy(self.proj(h), target)


class WhoHead(nn.Module):
    """Retrieval over the entity-state table via sampled softmax.

    Training: score the true linked entity against `n_neg` negatives drawn
    by the caller (mixture of uniform and popularity sampling happens in the
    batcher, where event frequencies live). Evaluation: full-catalog scores
    are one matmul against W_c(states); see queries.py.

    Interaction features (D3). A pure inner product cannot express "this
    entity already linked to that candidate", which is the entire content of
    the recency heuristic that outscores the model 2.6x on rel-hm. `feat` adds
    a learned linear term over the features built in batching.py
    (is_repeat, log1p count, recency, log1p popularity).

    It is initialised to ZERO, so a model with features enabled starts exactly
    where the featureless model is and can only be pushed away from it by the
    data. That also keeps old checkpoints numerically reproducible.

    Window conditioning. RelBench link tasks ask "which destinations in the
    next TIMEDELTA", and that timedelta ranges over 52x across the benchmark:
    7 days on rel-hm, 91 on rel-stack, 365 on rel-trial. A next-event
    objective has an implicit horizon of "until whenever the next event
    happens", which matches none of them. So the head takes the horizon as an
    INPUT: during training a window is drawn per target and supervised
    multi-positively over everything in it, and at inference the window is set
    to the task's timedelta. One pretrained model then serves every horizon
    instead of being implicitly tuned to one.
    """

    def __init__(self, dim: int, n_feats: int = 0, window: bool = False,
                 n_win_freq: int = 8):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.c = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5
        self.n_feats = n_feats
        if n_feats:
            self.feat = nn.Linear(n_feats, 1, bias=False)
            nn.init.zeros_(self.feat.weight)
        self.window = window
        if window:
            # Fourier features of log-days, like the TimeEncoder: horizons
            # span 1 day to a year and only their ratio is meaningful.
            self.win_freq = nn.Parameter(torch.logspace(-1, 1, n_win_freq))
            self.win_mlp = nn.Sequential(
                nn.Linear(2 * n_win_freq, dim), nn.GELU(),
                nn.Linear(dim, dim))
            nn.init.zeros_(self.win_mlp[-1].weight)
            nn.init.zeros_(self.win_mlp[-1].bias)

    def condition(self, h, win_days=None):
        """Fold the prediction horizon into the query state."""
        if not self.window or win_days is None:
            return h
        a = torch.log1p(win_days.clamp(min=1e-3)).unsqueeze(-1) * self.win_freq
        return h + self.win_mlp(torch.cat([a.sin(), a.cos()], -1))

    def scores(self, h, cand_states, feats=None, win_days=None, log_q=None):
        # h [N, d]; cand_states [N, C, d]
        s = torch.einsum("nd,nkd->nk", self.q(self.condition(h, win_days)),
                         self.c(cand_states)) * self.scale
        if self.n_feats and feats is not None:
            s = s + self.feat(feats).squeeze(-1)
        if log_q is not None:
            # logQ correction, applied to sampled candidates only. Makes the
            # sampled denominator an unbiased estimate of the full softmax.
            s = s - log_q
        return s

    def shared_logits(self, h, sh_states, win_days=None, sh_log_q=None,
                      sh_feat_score=None):
        """Scores against a pool shared by every query in the batch. [T, S].

        One matmul instead of a [T, S, d] gather -- that is the whole point:
        it is what makes S=8,192 negatives cost 17 MB of candidate states
        instead of 15.8 GB.

        `sh_feat_score` is the D3 feature contribution ALREADY reduced to a
        scalar per (query, candidate). It has to be supplied: giving shared
        negatives no features while positives get theirs would hand the head a
        free margin and let it read the label off the feature block -- the
        same degeneracy documented for the repeat feature in RESEARCH.md
        2026-08-23.
        """
        q = self.q(self.condition(h, win_days))               # [T, d]
        s = (q @ self.c(sh_states).transpose(0, 1)) * self.scale
        if sh_feat_score is not None:
            s = s + sh_feat_score
        if sh_log_q is not None:
            s = s - sh_log_q
        return s

    def loss(self, h, cand_states, feats=None, win_days=None, log_q=None,
             pos_mask=None, valid_mask=None, shared_logits=None,
             shared_valid=None):
        """Sampled softmax; multi-positive when `pos_mask` is given.

        `pos_mask` [N, C] bool marks which candidates are true positives for
        that query (the window may contain several). The loss is then the mean
        over positives of -log p(positive), i.e. each positive competes against
        the shared negative sample. Reduces exactly to plain cross-entropy on
        column 0 when the mask is a single leading True.
        """
        s = self.scores(h, cand_states, feats, win_days, log_q)
        if valid_mask is not None:
            # padded positive slots, and negatives that turned out to BE
            # positives (accidental hits), are removed from both the
            # numerator and the denominator
            s = s.masked_fill(~valid_mask, float("-inf"))
        if pos_mask is None:
            if shared_logits is not None:
                sl = shared_logits
                if shared_valid is not None:
                    sl = sl.masked_fill(~shared_valid, float("-inf"))
                s = torch.cat([s, sl], dim=-1)
            target = torch.zeros(len(s), dtype=torch.long, device=s.device)
            return F.cross_entropy(s, target)

        neg = s.masked_fill(pos_mask, float("-inf"))
        # denominator for positive j = exp(s_j) + sum over all negatives
        lse_neg = torch.logsumexp(neg, dim=-1, keepdim=True)
        if shared_logits is not None:
            # The shared pool joins the DENOMINATOR only; positives are still
            # supervised per query. Accidental hits (a pool member that is in
            # fact one of this query's positives) are masked exactly as in the
            # local set -- and they matter more here, because a pool of 8,192
            # drawn by popularity is far more likely to contain a true item
            # than 256 uniform draws.
            sl = shared_logits
            if shared_valid is not None:
                sl = sl.masked_fill(~shared_valid, float("-inf"))
            lse_neg = torch.logaddexp(
                lse_neg, torch.logsumexp(sl, dim=-1, keepdim=True))
        denom = torch.logaddexp(s, lse_neg)
        nll = (denom - s).masked_fill(~pos_mask, 0.0)
        n_pos = pos_mask.sum(-1).clamp(min=1)
        per_query = nll.sum(-1) / n_pos
        return per_query[pos_mask.any(-1)].mean()


class RerankHead(nn.Module):
    """Cross-interaction reranker over a shortlist.

    The retrieval score is a single dot product: the candidate is compared
    against ONE summary vector of the whole history. That is what makes
    two-tower retrieval cheap and also what caps it -- the candidate cannot
    ask "which of this customer's past purchases is the one that matters for
    me". Here each candidate attends over the last R history token states and
    forms its own view of the history, then is scored from that.

    Deliberately NOT ID-GNN: no message passing, no restriction of the
    candidate set. This is a second stage over a shortlist that always
    includes the dense-retrieval top-K, so structural priors can add
    precision without ever costing recall.
    """

    def __init__(self, dim: int, n_feats: int = 0, heads: int = 4):
        super().__init__()
        self.n_feats = n_feats
        self.q = nn.Linear(dim, dim, bias=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Sequential(
            nn.Linear(2 * dim + n_feats, dim), nn.GELU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, cand_states, hist, hist_mask, feats=None):
        """cand_states [N, C, d]; hist [N, R, d]; hist_mask [N, R] (True=pad).

        -> [N, C] additive rerank logits.
        """
        cq = self.q(cand_states)
        allpad = hist_mask.all(-1, keepdim=True)
        # a row with no history must not produce NaNs from a fully masked
        # softmax; unmask slot 0 and rely on the zero state there
        hm = hist_mask & ~allpad
        z, _ = self.attn(cq, hist, hist, key_padding_mask=hm,
                         need_weights=False)
        z = self.norm(z) * (~allpad).unsqueeze(-1)
        parts = [z, cand_states]
        if self.n_feats and feats is not None:
            parts.append(feats)
        return self.out(torch.cat(parts, -1)).squeeze(-1)

    def loss(self, cand_states, hist, hist_mask, base_scores, pos_mask,
             feats=None):
        """Trained on the SAME candidate distribution it will rerank."""
        s = base_scores.detach() + self(cand_states, hist, hist_mask, feats)
        neg = s.masked_fill(pos_mask, float("-inf"))
        denom = torch.logaddexp(s, torch.logsumexp(neg, -1, keepdim=True))
        nll = (denom - s).masked_fill(~pos_mask, 0.0)
        per_query = nll.sum(-1) / pos_mask.sum(-1).clamp(min=1)
        keep = pos_mask.any(-1)
        if not keep.any():
            return cand_states.new_zeros(())
        return per_query[keep].mean()


class WhatHead(nn.Module):
    """Per-fact-table column decoders. One shared trunk, per-table outputs."""

    def __init__(self, dim: int, schema):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.num_out = nn.ModuleDict()
        self.cat_out = nn.ModuleDict()
        for name, spec in schema.fact_tables.items():
            n_num = sum(c.kind == "numeric" for c in spec.columns)
            cats = [c for c in spec.columns if c.kind == "categorical"]
            if n_num:
                self.num_out[name] = nn.Linear(dim, n_num)
            if cats:
                self.cat_out[name] = nn.ModuleList(
                    nn.Linear(dim, c.cardinality) for c in cats)

    def loss(self, h, table_name, feat_num, feat_cat):
        z = self.trunk(h)
        total = h.new_zeros(())
        if table_name in self.num_out and feat_num.numel():
            total = total + F.mse_loss(self.num_out[table_name](z), feat_num)
        if table_name in self.cat_out:
            for j, lin in enumerate(self.cat_out[table_name]):
                total = total + F.cross_entropy(lin(z), feat_cat[:, j].long())
        return total
