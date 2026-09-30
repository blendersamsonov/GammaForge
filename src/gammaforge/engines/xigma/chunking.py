"""One auto-chunk + OOM-retry utility, used by every chunked stage.

One shared implementation serves every chunked stage; callers do not duplicate sizing or
retry policy:

* **proactive sizing, retry as backup.** A chunk is sized from currently free memory
  before the work starts. The halve-and-retry loop exists because that estimate cannot be
  right — other processes, driver drift — not because it is the plan.
* **numpy is chunked too.** "Plain numpy is bounded by system RAM, so it needs no
  chunking" is wrong: a
  5,000,000-particle by 1,024-point broadcast is over 100 GB in a handful of live
  temporaries. Worse than being wrong, it fails differently — Linux overcommit means an
  oversized host allocation can meet the OOM-killer instead of raising `MemoryError`, so
  on the host the *estimate* is the defence and the retry is barely a safety net. Hence a
  more conservative budget fraction there than on a GPU, where an out-of-memory error is
  a contained, catchable event.
* **a hard ceiling, independent of free memory.** Sizing a chunk purely as a fraction of
  what happens to be free grows it without limit on a generous machine. Measured on the
  the spectrum path: chunk=1 took 9.99 s, chunk=4 7.29 s, chunk=8 7.25 s,
  chunk=16 7.20 s — every doubling past ~8-16 bought nothing while a 128 GB machine
  happily sized it into the hundreds and used ~100 GB for a computation that runs just as
  fast in a few. A budget may shrink a chunk below a caller's ceiling; it may never grow
  it past one.

Chunking must never change an answer, which is why every caller here partitions along an
axis whose entries are independent — the chunk-invariance property of §7 is a test, not
an aspiration.
"""

from __future__ import annotations

import os
from typing import Callable, TypeVar

__all__ = [
    "available_ram_bytes",
    "available_vram_bytes",
    "estimate_chunk",
    "run_in_chunks",
    "MAX_OOM_HALVINGS",
    "SAFETY_FRACTION",
]

T = TypeVar("T")

#: How many times a chunk may be halved before an out-of-memory failure is re-raised.
#: Six halvings is a factor of 64 — past that the estimate was not slightly wrong, and
#: retrying is hiding a problem rather than absorbing one.
MAX_OOM_HALVINGS = 6

#: Fraction of free memory a chunk may budget for, per backend. The GPU number is higher
#: because an allocation failure there is catchable; the host number is lower because
#: getting it wrong risks the OOM-killer rather than an exception.
SAFETY_FRACTION = {"cupy": 0.7, "numpy": 0.5}


def available_vram_bytes() -> int | None:
    """Free GPU memory, or ``None`` if there is no usable cupy/GPU to ask."""
    try:
        import cupy
    except ImportError:
        return None
    try:
        free, _total = cupy.cuda.Device().mem_info
        return int(free)
    except Exception:  # no device, driver mismatch, cupy misconfigured
        return None


def available_ram_bytes() -> int | None:
    """Available system memory, or ``None`` where the platform will not say."""
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGESIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def estimate_chunk(
    n_items: int,
    bytes_per_item: int,
    backend: str = "numpy",
    *,
    ceiling: int | None = None,
    safety_frac: float | None = None,
) -> int:
    """How many items to process at once, given what each one costs in live temporaries.

    ``bytes_per_item`` is the caller's own calibration — it knows how many arrays of what
    shape its inner loop keeps alive, and this module does not. Returns ``n_items`` (do
    not chunk) when the free-memory query is unavailable, since guessing a small chunk
    would be a large, silent slowdown on a machine that simply declines to be measured.
    """
    if n_items <= 0:
        return 1
    limit = min(n_items, ceiling) if ceiling is not None else n_items

    free_bytes = available_vram_bytes() if backend == "cupy" else available_ram_bytes()
    if backend not in SAFETY_FRACTION:
        raise ValueError(f"backend must be one of {sorted(SAFETY_FRACTION)}, got {backend!r}")
    if free_bytes is None:
        return max(1, limit)

    fraction = SAFETY_FRACTION[backend] if safety_frac is None else safety_frac
    affordable = int(free_bytes * fraction / max(bytes_per_item, 1))
    return max(1, min(affordable, limit))


def run_in_chunks(
    n_items: int,
    work: Callable[[int, int], T],
    *,
    chunk: int | None = None,
    bytes_per_item: int = 1,
    backend: str = "numpy",
    ceiling: int | None = None,
) -> list[T]:
    """Call ``work(start, stop)`` over consecutive slices of ``n_items``; collect results.

    ``chunk`` overrides the estimate — which is what the chunk-invariance test varies, and
    what a caller who has measured its own workload should pass. On an out-of-memory
    failure the current chunk is halved and the *same* slice retried, so no work is
    skipped and no item is processed twice.

    The results come back in order, one per slice; combining them (concatenate,
    sum, ...) is the caller's business, because only the caller knows which of its outputs
    are per-item and which are shared accumulators.

    ``work`` must not depend on the partitioning — that is the contract that makes
    chunking invisible, and the reason the OOM path can safely re-run a slice it has
    already half-executed.
    """
    if n_items <= 0:
        return []
    size = int(chunk) if chunk is not None else estimate_chunk(
        n_items, bytes_per_item, backend, ceiling=ceiling
    )
    size = max(1, min(size, n_items))

    results: list[T] = []
    start, halvings = 0, 0
    while start < n_items:
        stop = min(start + size, n_items)
        try:
            results.append(work(start, stop))
        except _out_of_memory_errors(backend):
            if halvings >= MAX_OOM_HALVINGS or size <= 1:
                raise
            size = max(1, size // 2)
            halvings += 1
            _release_pools(backend)
            continue
        _release_pools(backend)
        start = stop
    return results


def _out_of_memory_errors(backend: str) -> tuple[type[BaseException], ...]:
    """What "out of memory" is called on this backend.

    `MemoryError` is included for cupy as well: a GPU run still allocates host-side, and
    catching only the device error would let the host failure escape a loop written to
    survive exactly this.
    """
    if backend != "cupy":
        return (MemoryError,)
    try:
        import cupy
    except ImportError:  # pragma: no cover — unreachable from a cupy run
        return (MemoryError,)
    return (cupy.cuda.memory.OutOfMemoryError, MemoryError)


def _release_pools(backend: str) -> None:
    """Hand freed device blocks back between slices.

    A memory-constrained GPU has no headroom to hold several stale slices' temporaries
    alongside the running accumulators, and Python's collector makes no promise about
    when it will notice. On the host this is a no-op — the allocator already reuses.
    """
    if backend != "cupy":
        return
    try:
        import cupy

        cupy.get_default_memory_pool().free_all_blocks()
    except ImportError:  # pragma: no cover — unreachable from a cupy run
        pass
