"""Causal entity-history transformer (ARCHITECTURE.md sections 4-5).

Position information comes from event *time*, not token index: the TimeEncoder
injects learned Fourier features of the log time-gap plus calendar features.

ATTENTION (RESEARCH.md "B1"). Sequences are PACKED: one batch row holds several
entities' histories end to end, and `seq_id` fences attention between them. v0
expressed that fence as a dense additive float mask of shape [B*H, L, L], which
`nn.MultiheadAttention` then had to run through its unfused math path. Two
costs, both fatal at B200 batch sizes:

  * memory is O(B * H * L^2). At the ARCHITECTURE 9.1 target of 262,144 tokens
    (e.g. B=32, L=8192, H=8) the mask alone is 8 * 32 * 8192^2 * 2 bytes = 34
    TERABYTES. The mask, not the activations, is what caps the batch.
  * an explicit `attn_mask` disables every fused kernel, so v0 never touched
    flash attention at all.

The fence is not arbitrary, though -- it is block-diagonal, because packing
lays sequences down contiguously. FlexAttention expresses exactly that as a
`mask_mod` predicate and compiles it into the kernel, storing only a coarse
block-sparsity map (O(B * (L/128)^2), ~1 MB at the target above) and skipping
whole blocks it knows are masked. Memory stops depending on L^2 and the packing
becomes a speedup rather than a tax: with many short sequences per row, most
blocks are empty and never get computed.

The SDPA path is kept as the fallback (CPU, tiny shapes, and as the reference
the flex path is tested against -- `test_flex_matches_sdpa`). It still beats v0
by using a bool mask through the fused mem-efficient backend instead of a float
mask through the math path.

Parameter layout is deliberately unchanged: `nn.MultiheadAttention` is retained
purely as the container for `in_proj_weight` / `out_proj`, and we do the
projections by hand. That keeps `state_dict` byte-identical to v0, so every
checkpoint in `runs/` still loads and the rel-hm numbers in RESEARCH.md remain
reproducible against this file.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import (
        flex_attention, create_block_mask)
    _HAS_FLEX = True
except ImportError:                                   # torch < 2.5
    _HAS_FLEX = False

# flex_attention is only fast once compiled, and create_block_mask likewise.
# Compile lazily and once per process: these are module-level so that the four
# concurrent SWEEP-lane jobs of ARCHITECTURE 9.4 each pay it once, not once
# per model instance.
_FLEX_COMPILED = {}


def _flex_fns():
    if not _FLEX_COMPILED:
        _FLEX_COMPILED["attn"] = torch.compile(
            flex_attention, dynamic=False)
        _FLEX_COMPILED["mask"] = torch.compile(
            create_block_mask, dynamic=False)
    return _FLEX_COMPILED["attn"], _FLEX_COMPILED["mask"]


class TimeEncoder(nn.Module):
    """[sin/cos of learned frequencies on log(gap)] + calendar embeddings."""

    def __init__(self, dim: int, n_freq: int = 16):
        super().__init__()
        self.freq = nn.Parameter(torch.logspace(-1, 1, n_freq))
        self.dow = nn.Embedding(7, 16)
        self.month = nn.Embedding(12, 16)
        self.proj = nn.Linear(2 * n_freq + 32, dim)

    def forward(self, t: torch.Tensor, gap: torch.Tensor) -> torch.Tensor:
        # t: unix seconds [B, L]; gap: seconds since entity's previous event
        lg = torch.log1p(gap.clamp(min=0).float())
        ang = lg.unsqueeze(-1) * self.freq
        dow = ((t // 86400) + 4) % 7          # 1970-01-01 was a Thursday
        month = (t // (86400 * 30)) % 12       # coarse month bucket is enough
        cal = torch.cat([self.dow(dow.long()), self.month(month.long())], -1)
        return self.proj(torch.cat([ang.sin(), ang.cos(), cal], -1))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        # Container only -- see the module docstring. Its forward is bypassed.
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(),
            nn.Linear(4 * dim, dim), nn.Dropout(dropout),
        )
        self.heads = heads
        self.head_dim = dim // heads
        self.attn_dropout = dropout

    def _project(self, h):
        B, L, D = h.shape
        qkv = F.linear(h, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        return [z.view(B, L, self.heads, self.head_dim).transpose(1, 2)
                for z in (q, k, v)]           # each [B, H, L, head_dim]

    def forward(self, x, attend):
        B, L, D = x.shape
        h = self.n1(x)
        q, k, v = self._project(h)
        a = attend(q, k, v, self.attn_dropout if self.training else 0.0)
        a = a.transpose(1, 2).reshape(B, L, D)
        x = x + self.attn.out_proj(a)
        return x + self.mlp(self.n2(x))


class MLPBranch(nn.Module):
    """A position-wise residual block: the cheap alternative to a Block.

    A branch built from these has no attention, so it cannot mix information
    across tokens -- it can only re-read the trunk's summary of a position
    through its own parameters. That is strictly less expressive than a
    transformer branch and strictly cheaper, and which of the two is enough is
    an empirical question, so both are selectable.
    """

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.n = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(),
            nn.Linear(4 * dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x, attend):            # `attend` unused; same signature
        return x + self.mlp(self.n(x))


class Backbone(nn.Module):
    """Pre-norm causal transformer over packed event-token sequences.

    Packing: multiple entity sequences are concatenated into one row up to a
    token budget; `seq_id` marks which entity each token belongs to, and the
    attention rule forbids attention across entities as well as into the
    future.

    `attn` selects how that rule is applied: "flex" compiles it into the
    kernel, "sdpa" materialises a bool mask, "auto" takes flex whenever it is
    available on CUDA and falls back otherwise.

    BRANCHES (2026-08-26). Every head used to read the same final hidden
    state, and ARCHITECTURE 9.6 measured what that costs: on rel-stack,
    `w_when 1.0` scores 86.0 classification AUROC and 3.10 MAP while
    `w_when 0.1` scores 63.7 and 5.78. The two categories want the shared
    representation to be different things -- retrieval wants a state that
    discriminates *which entity comes next*, the window/survival readouts want
    one that retains *how much has happened and how long ago* -- and a single
    vector optimised for both lands between them.

    So the trunk stays shared (that is the thesis: one world model) and each
    head group gets `branch_layers` of its own on top. Gradient from WHO and
    gradient from WHEN/WINDOW now reach the trunk through different
    parameters, which is what lets the trunk carry BOTH facts instead of
    trading them off inside one vector.

    Deliberately ADDITIVE rather than a split of `layers`: the trunk keeps its
    exact parameter layout, so `branch_layers=0` is bit-identical to every
    checkpoint in `runs/` and to every number in RESEARCH.md. It is the
    control arm, and it must stay reproducible.
    """

    #: head groups that get their own branch. "retrieval" feeds WHO, the
    #: reranker and the entity-state writes; "temporal" feeds WHEN, WHERE,
    #: WHAT and WINDOW -- i.e. the recommendation side and the
    #: classification/regression side of ARCHITECTURE 9.6's trade-off.
    BRANCHES = ("retrieval", "temporal")

    def __init__(self, dim: int = 512, layers: int = 12, heads: int = 8,
                 dropout: float = 0.1, attn: str = "auto",
                 branch_layers: int = 0, branch_kind: str = "transformer"):
        super().__init__()
        self.dim = dim
        self.attn_impl = attn
        self.time_enc = TimeEncoder(dim)
        self.blocks = nn.ModuleList(
            Block(dim, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.branch_layers = branch_layers
        self.branch_kind = branch_kind
        self.branch_blocks = nn.ModuleDict()
        self.branch_norm = nn.ModuleDict()
        if branch_layers:
            mk = (Block if branch_kind == "transformer"
                  else (lambda d, h, p: MLPBranch(d, p)))
            for name in self.BRANCHES:
                self.branch_blocks[name] = nn.ModuleList(
                    mk(dim, heads, dropout) for _ in range(branch_layers))
                self.branch_norm[name] = nn.LayerNorm(dim)

    def _use_flex(self, x) -> bool:
        if self.attn_impl == "flex":
            return True
        if self.attn_impl == "sdpa":
            return False
        # auto: flex needs CUDA and a sequence long enough to be worth the
        # block-mask construction. Below ~256 tokens SDPA's dense mask is
        # small and compiling is pure overhead.
        return _HAS_FLEX and x.is_cuda and x.shape[1] >= 256

    def _flex_attend(self, seq_id, L):
        """Build the block-sparse mask once and reuse it for every layer."""
        flex, make_mask = _flex_fns()

        def mask_mod(b, h, q_idx, kv_idx):
            # causal AND same-entity: the packing fence, evaluated inside the
            # kernel instead of stored as an [B*H, L, L] tensor
            return (q_idx >= kv_idx) & (seq_id[b, q_idx] == seq_id[b, kv_idx])

        block_mask = make_mask(mask_mod, seq_id.shape[0], None, L, L,
                               device=seq_id.device)

        def attend(q, k, v, dropout_p):
            # FlexAttention has no attention-weight dropout. The MLP dropout
            # in every block is retained; every measurement in RESEARCH.md is
            # of an UNDER-fitted model (loss still falling at the step budget)
            # so removing this particular regulariser is not a live risk. It
            # is a real difference from the SDPA path and the reason
            # `test_flex_matches_sdpa` compares in eval mode.
            return flex(q, k, v, block_mask=block_mask)

        return attend

    def _sdpa_attend(self, seq_id, L):
        causal = torch.ones(L, L, dtype=torch.bool,
                            device=seq_id.device).triu(1)
        same = seq_id.unsqueeze(-1) != seq_id.unsqueeze(-2)   # [B, L, L]
        # True = keep. SDPA takes a bool mask directly, so we never build the
        # [B*H, L, L] float tensor v0 built.
        keep = ~(causal.unsqueeze(0) | same)
        keep = keep.unsqueeze(1)                              # [B, 1, L, L]

        def attend(q, k, v, dropout_p):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=keep, dropout_p=dropout_p)

        return attend

    def forward(self, tokens: torch.Tensor, t: torch.Tensor,
                gap: torch.Tensor, seq_id: torch.Tensor) -> dict:
        """-> {branch name: [B, L, dim]}.

        With `branch_layers=0` every branch is the SAME tensor object, not a
        copy: the trunk output is returned directly, so memory and arithmetic
        are exactly what they were before branches existed.
        """
        # tokens [B, L, dim], t/gap/seq_id [B, L]
        x = tokens + self.time_enc(t, gap)
        L = x.shape[1]
        attend = (self._flex_attend(seq_id, L) if self._use_flex(x)
                  else self._sdpa_attend(seq_id, L))
        for blk in self.blocks:
            x = blk(x, attend)
        x = self.norm(x)
        if not self.branch_layers:
            return {name: x for name in self.BRANCHES}
        out = {}
        for name in self.BRANCHES:
            # Branch blocks see the same packing fence: a branch is part of
            # the causal encoder, not a pooled head, so a token must still not
            # attend across an entity boundary or into the future.
            z = x
            for blk in self.branch_blocks[name]:
                z = blk(z, attend)
            out[name] = self.branch_norm[name](z)
        return out
