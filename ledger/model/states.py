"""Entity-state tables (ARCHITECTURE.md section 6, HANDOFF section 4.1).

Two interchangeable implementations behind one interface, so the choice is a
measured comparison rather than an assumption:

`LearnedStates`     one nn.Embedding per entity table, trained by AdamW.
                    v0 behaviour. Costs params+grad+m+v = 16 bytes/element at
                    fp32, i.e. 328 GB at rel-amazon scale (40M entities,
                    dim 128) -- impossible.

`NonParametricStates`  a bfloat16 buffer holding, for each entity, an EMA of
                    the backbone hidden states of events that entity took part
                    in. No gradients, no optimizer state: 2 bytes/element,
                    i.e. 20 GB at the same scale. This is the standard stale-
                    memory trick from the temporal-graph-network literature.

Scientifically the non-parametric table is also the better object: an entity's
representation should be a summary of its history produced by the model, not a
free parameter that memorises an ID.

IMPORTANT (see RESEARCH.md 2026-08-22): the non-parametric table is required
for MEMORY reasons. The earlier claim that the WHO head cannot learn without it
was based on a 100-step run and is false -- WHO reaches 5.9x better than chance
by step 2000 with learned embeddings. Whether the non-parametric table also
improves convergence is an open question to be measured, not asserted.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedStates(nn.Module):
    """v0: one trainable nn.Embedding per entity table."""

    kind = "learned"

    def __init__(self, entity_counts: dict, dim: int):
        super().__init__()
        self.dim = dim
        self.emb = nn.ModuleDict({
            name: nn.Embedding(n, dim) for name, n in entity_counts.items()
        })
        for e in self.emb.values():
            nn.init.normal_(e.weight, std=0.02)

    def read(self, table: str, ids: torch.Tensor) -> torch.Tensor:
        return self.emb[table](ids)

    @torch.no_grad()
    def write(self, table: str, ids: torch.Tensor, h: torch.Tensor,
              pos: torch.Tensor | None = None,
              reduce: str | None = None) -> None:
        pass  # learned states are updated by the optimizer, not by writes

    @torch.no_grad()
    def reset(self) -> None:
        pass  # nothing to recompute; these ARE the parameters

    def bytes_per_entity(self) -> int:
        return 16 * self.dim  # param + grad + adam m + adam v, fp32


class NonParametricStates(nn.Module):
    """Detached EMA of backbone hidden states, stored in bfloat16.

    Not parameters -- registered as buffers so they move with `.to(device)`
    and are saved in the checkpoint, but carry no gradient and no optimizer
    state.

    `write` is called once per training step with the hidden states of the
    tokens each entity participated in. Entities seen for the first time take
    the hidden state outright; afterwards they EMA toward it.

    Variants (measured 2026-08-22: the plain EMA collapses to mean pairwise
    cosine +0.83 and effective rank 45/128):
      normalize : write h/||h|| so the EMA is directional. Magnitude carries
                  little identity and dominates the average otherwise.
      center    : subtract a running global mean at read time. The shared
                  component is identical across entities and therefore
                  carries no discriminative signal, but it is most of the
                  vector -- removing it is nearly free.
    """

    kind = "nonparam"

    def __init__(self, entity_counts: dict, dim: int, momentum: float = 0.1,
                 dtype: torch.dtype = torch.bfloat16,
                 normalize: bool = False, center: bool = False,
                 reduce: str = "mean"):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.dtype = dtype
        self.normalize = normalize
        self.center = center
        self.reduce = reduce
        self.tables = list(entity_counts)
        self._acc = None          # set only inside a refresh pass
        for name, n in entity_counts.items():
            self.register_buffer(f"S_{name}",
                                 torch.zeros(n, dim, dtype=dtype),
                                 persistent=True)
            self.register_buffer(f"seen_{name}",
                                 torch.zeros(n, dtype=torch.bool),
                                 persistent=True)
            # running global mean, for `center`
            self.register_buffer(f"mu_{name}",
                                 torch.zeros(dim, dtype=torch.float32),
                                 persistent=True)

    def read(self, table: str, ids: torch.Tensor) -> torch.Tensor:
        # float32 for numerics; detached by construction (buffer, no grad)
        s = getattr(self, f"S_{table}")[ids].float()
        if self.center:
            s = s - getattr(self, f"mu_{table}")
        return s

    @torch.no_grad()
    def reset(self) -> None:
        """Zero every state and clear `seen`.

        Used before a full refresh pass so the recomputed table is a function
        of the CURRENT weights alone, rather than an EMA whose oldest terms
        were produced by a model tens of thousands of steps ago.
        """
        for name in self.tables:
            getattr(self, f"S_{name}").zero_()
            getattr(self, f"seen_{name}").zero_()
            getattr(self, f"mu_{name}").zero_()

    # -- exact-mean accumulation (refresh only) ---------------------------
    #
    # An EMA is order-dependent: with momentum 0.1 the final value is
    # essentially the last ~10 writes. That is tolerable during training,
    # where batch order is random, but it makes a single refresh pass a
    # function of the traversal order -- and if that traversal is sorted by
    # anything (history length, say) the resulting table is systematically
    # biased toward whatever sorts last. Measured on rel-hm: a length-sorted
    # EMA refresh moved MAP 0.745 -> 0.179.
    #
    # For a one-shot recompute the order-free definition is what the docstring
    # always claimed: the MEAN of the hidden states of the events the entity
    # took part in. Accumulate sum and count, divide at the end.

    @torch.no_grad()
    def begin_accumulate(self) -> None:
        self._acc = {}
        for name in self.tables:
            n = getattr(self, f"S_{name}").shape[0]
            self._acc[name] = (
                torch.zeros(n, self.dim, dtype=torch.float32,
                            device=getattr(self, f"S_{name}").device),
                torch.zeros(n, dtype=torch.float32,
                            device=getattr(self, f"S_{name}").device))

    @torch.no_grad()
    def end_accumulate(self) -> None:
        for name in self.tables:
            acc, cnt = self._acc[name]
            hit = cnt > 0
            mean = acc[hit] / cnt[hit].unsqueeze(-1)
            if self.normalize:
                mean = torch.nn.functional.normalize(mean, dim=-1)
            getattr(self, f"S_{name}")[hit] = mean.to(self.dtype)
            getattr(self, f"seen_{name}")[hit] = True
            if self.center and hit.any():
                getattr(self, f"mu_{name}").copy_(mean.mean(0))
        self._acc = None

    @torch.no_grad()
    def write(self, table: str, ids: torch.Tensor, h: torch.Tensor,
              pos: torch.Tensor | None = None,
              reduce: str | None = None) -> None:
        """Write hidden states into the table, collapsing duplicate ids.

        `reduce` picks how duplicates within one call collapse:
          "mean" -- average them. Correct for DESTINATION entities: an
                    article's state should summarise all the customers who
                    bought it.
          "last" -- keep the highest token position (`pos`). Correct for the
                    entity whose OWN sequence this is: every token of a
                    customer's history maps to that customer, and at inference
                    `queries.py` builds the query from the LAST hidden state.
                    Averaging there makes training and inference disagree
                    about what a customer state denotes.
        """
        if ids.numel() == 0:
            return
        reduce = reduce or self.reduce
        S = getattr(self, f"S_{table}")
        seen = getattr(self, f"seen_{table}")
        h = h.detach().float()
        if self.normalize:
            h = torch.nn.functional.normalize(h, dim=-1)

        if reduce == "last" and pos is not None and ids.numel() > 1:
            # Sort by position so that "largest row index" == "largest pos",
            # then take the max row index per distinct entity. index_reduce is
            # deterministic; a duplicate index_put_ is not.
            order = torch.argsort(pos)
            ids_s, h_s = ids[order], h[order]
            uniq, inv = torch.unique(ids_s, return_inverse=True)
            pick = torch.full((len(uniq),), -1, dtype=torch.long,
                              device=ids.device)
            pick = pick.index_reduce_(
                0, inv, torch.arange(len(inv), device=ids.device), "amax",
                include_self=False)
            ids, h = uniq, h_s[pick]

        if getattr(self, "_acc", None) is not None:
            acc, cnt = self._acc[table]
            acc.index_add_(0, ids, h)
            cnt.index_add_(0, ids, torch.ones(len(ids), device=h.device))
            return

        # Deduplicate ids within the call. An entity generally appears many
        # times in one batch (every token of a customer's own sequence maps to
        # that customer; a popular article is touched by hundreds of rows).
        # `S[ids] = new` is a non-accumulating index_put_, so with repeated
        # ids exactly one arbitrary row survives -- the EMA this class
        # documents was silently a "last writer wins" per batch. Average the
        # duplicates first, then EMA once per distinct entity.
        if ids.numel() > 1:
            ids, inv = torch.unique(ids, return_inverse=True)
            acc = torch.zeros(len(ids), h.shape[-1], device=h.device)
            acc.index_add_(0, inv, h)
            cnt = torch.zeros(len(ids), device=h.device).index_add_(
                0, inv, torch.ones(len(inv), device=h.device))
            h = acc / cnt.unsqueeze(-1)

        cur = S[ids].float()
        fresh = ~seen[ids]
        new = (1.0 - self.momentum) * cur + self.momentum * h
        new = torch.where(fresh.unsqueeze(-1), h, new)
        if self.normalize:
            new = torch.nn.functional.normalize(new, dim=-1)
        S[ids] = new.to(self.dtype)
        seen[ids] = True
        if self.center:
            mu = getattr(self, f"mu_{table}")
            mu.mul_(1 - self.momentum).add_(self.momentum * new.mean(0))

    def coverage(self) -> dict:
        """Fraction of each entity table that has ever been written.

        Diagnostic: a WHO head cannot rank entities whose state is still zero,
        so low coverage on the destination table is a direct explanation for a
        weak recommendation score.
        """
        out = {}
        for name in self.tables:
            seen = getattr(self, f"seen_{name}")
            out[name] = float(seen.float().mean()) if seen.numel() else 0.0
        return out

    def bytes_per_entity(self) -> int:
        return 2 * self.dim  # bf16, no grad, no optimizer state


class HybridStates(nn.Module):
    """Learned embeddings for small tables, non-parametric EMA for large ones.

    Motivation (RESEARCH.md 2026-08-22): the EMA collapses -- destination
    entities end up ~0.83 mean pairwise cosine, so the WHO head cannot
    discriminate them. But section 4.1's memory argument only bites on the
    genuinely huge tables. rel-hm's article catalogue is 105,542 entities:
    0.05 GB as a learned fp32 embedding at dim 128, 0.22 GB with AdamW state.
    Nothing forces that table to be non-parametric.

    So: any table with <= `learned_max` entities gets a trainable embedding
    (gradients push them apart, which is exactly what retrieval needs);
    anything larger keeps the bf16 EMA. At rel-amazon scale the 40M-entity
    tables stay non-parametric and the memory argument is preserved.
    """

    kind = "hybrid"

    def __init__(self, entity_counts: dict, dim: int, learned_max: int = 200_000,
                 momentum: float = 0.1, normalize: bool = False,
                 center: bool = False, reduce: str = "mean",
                 learned_tables: list | None = None):
        super().__init__()
        self.dim = dim
        # `learned_max` selects by SIZE, which turned out to be the wrong
        # abstraction. What actually needs a trained embedding is the
        # DESTINATION catalogue the WHO head ranks, and destination is not a
        # function of size: it is smaller than the source on rel-amazon
        # (product 506K < customer 1.85M) and rel-hm (article 105K < customer
        # 1.37M), but LARGER on rel-stack (posts 334K > users 255K) and by far
        # the largest on rel-avito (AdsInfo 5.96M). On those two the size gate
        # cannot express the configuration we want at any threshold, so an
        # explicit list overrides it.
        if learned_tables is not None:
            want = set(learned_tables)
            unknown = want - set(entity_counts)
            if unknown:
                raise ValueError(
                    f"learned_tables names non-entity tables: {sorted(unknown)}"
                    f" (entities: {sorted(entity_counts)})")
            self.learned_tables = sorted(want)
            self.ema_tables = sorted(set(entity_counts) - want)
        else:
            self.learned_tables = sorted(
                n for n, c in entity_counts.items() if c <= learned_max)
            self.ema_tables = sorted(
                n for n, c in entity_counts.items() if c > learned_max)
        self.learned = LearnedStates(
            {n: entity_counts[n] for n in self.learned_tables}, dim) \
            if self.learned_tables else None
        self.ema = NonParametricStates(
            {n: entity_counts[n] for n in self.ema_tables}, dim,
            momentum=momentum, normalize=normalize, center=center,
            reduce=reduce) if self.ema_tables else None

    def _which(self, table):
        return self.learned if table in self.learned_tables else self.ema

    @torch.no_grad()
    def reset(self) -> None:
        if self.ema is not None:      # learned tables are not recomputable
            self.ema.reset()

    @torch.no_grad()
    def begin_accumulate(self) -> None:
        if self.ema is not None:
            self.ema.begin_accumulate()

    @torch.no_grad()
    def end_accumulate(self) -> None:
        if self.ema is not None:
            self.ema.end_accumulate()

    def read(self, table: str, ids: torch.Tensor) -> torch.Tensor:
        return self._which(table).read(table, ids)

    @torch.no_grad()
    def write(self, table: str, ids: torch.Tensor, h: torch.Tensor,
              pos: torch.Tensor | None = None,
              reduce: str | None = None) -> None:
        if table in self.ema_tables:
            self.ema.write(table, ids, h, pos=pos, reduce=reduce)

    def coverage(self) -> dict:
        out = {t: 1.0 for t in self.learned_tables}   # always readable
        if self.ema is not None:
            out.update(self.ema.coverage())
        return out

    def bytes_per_entity(self) -> int:
        return 16 * self.dim  # upper bound; learned tables dominate


def build_states(kind: str, entity_counts: dict, dim: int, **kw):
    if kind == "learned":
        return LearnedStates(entity_counts, dim)
    if kind == "nonparam":
        return NonParametricStates(entity_counts, dim, **kw)
    if kind == "hybrid":
        return HybridStates(entity_counts, dim, **kw)
    raise ValueError(f"unknown state kind: {kind}")
