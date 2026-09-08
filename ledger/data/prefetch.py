"""Overlap batch construction with GPU compute across many CPU workers.

Measured on rel-amazon (11.8M events, 2.36M entities, 506K products), one
training step at max_len 512 x batch_rows 8:

    PackedBatcher.batch()   962 ms   (CPU, single-threaded)
    forward+backward+step    28 ms   (B200)

so 97% of wall time is the GPU waiting for numpy. No amount of GPU tuning
moves a number like that, and it is why RESEARCH.md's B200 estimate was
"~1.4x with current code". The work is embarrassingly parallel -- each batch
draws independent entities -- and this machine has 224 cores, so the fix is to
run K batchers concurrently and let the main process consume a queue.

The corpus is memory-mapped (see cache.py) and the batcher's expensive derived
tables (2-hop co-occurrence, temporal index) are built ONCE in the parent
before forking. Workers inherit them copy-on-write, so K workers cost one
build, not K -- on rel-amazon that is 13.5s paid once instead of 13.5s x K.

SEEDING. Forked workers inherit the parent's random state, so without
intervention K workers would emit K copies of the SAME batch stream -- a
silent K-fold reduction in data diversity that would look exactly like an
architecture problem. Each worker reseeds from (base_seed, worker_id), and
critically it reseeds the bit generator IN PLACE: `CandidateGenerator` objects
built in the parent hold a reference to the batcher's `rng` object, so
rebinding `batcher.rng` would leave every negative sampler still drawing from
the parent's stream. `test_prefetch_workers_draw_different_batches` covers it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def raise_fd_limit() -> None:
    """Worker batches are shared as file descriptors; many workers exhaust them.

    Each in-flight batch is ~20 tensors, and with W workers x prefetch_factor
    outstanding batches a single job holds W*P*20 descriptors. At W=48, P=4
    that is ~3,800 -- over the usual 1024 soft limit, and the failure mode is
    an opaque "Too many open files" from a worker mid-run.
    """
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < hard:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


def _reseed_in_place(rng: np.random.Generator, seed: int) -> None:
    """Reseed `rng` without replacing the object other components hold."""
    rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state


class _BatchStream(IterableDataset):
    """Infinite stream of training batches from one PackedBatcher."""

    def __init__(self, batcher, base_seed: int):
        self.batcher = batcher
        self.base_seed = base_seed

    def __iter__(self):
        info = get_worker_info()
        wid = 0 if info is None else info.id
        _reseed_in_place(self.batcher.rng, self.base_seed * 100_003 + wid)
        while True:
            # Built on CPU; the main process pins and moves to GPU. Doing the
            # transfer here would serialise every worker on the CUDA context.
            yield self.batcher.batch("cpu")


def _to_device(obj, device, non_blocking=True):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: _to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device, non_blocking) for v in obj)
    return obj                     # strings (seq_entity), numpy scalars


class PrefetchLoader:
    """Iterator of GPU-resident batches, produced by `workers` processes.

    `workers=0` runs the batcher inline in this process -- the old behaviour,
    kept because it is the reference the parallel path is tested against and
    because a single smoke-test step should not fork anything.
    """

    def __init__(self, batcher, workers: int = 8, seed: int = 0,
                 device: str = "cuda", prefetch_factor: int = 4,
                 warm: bool = True):
        self.batcher = batcher
        self.device = device
        self.workers = workers
        if workers <= 0:
            self.loader = None
            return
        raise_fd_limit()
        if warm:
            # Build the derived tables (cooc, temporal index, candidate
            # generators) in the PARENT so the fork shares them. Without this
            # every worker rebuilds them independently and the first K batches
            # cost K x 13.5s.
            batcher.batch("cpu")
        self.loader = DataLoader(
            _BatchStream(batcher, seed),
            batch_size=None,               # yield batches through unchanged
            num_workers=workers,
            pin_memory=(device != "cpu"),
            prefetch_factor=prefetch_factor,
            persistent_workers=True,
        )
        self._it = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        if self.loader is None:
            return self.batcher.batch(self.device)
        return _to_device(next(self._it), self.device)

    def close(self):
        if self.loader is not None:
            del self._it
            del self.loader
            self.loader = None
