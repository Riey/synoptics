"""Data parallelism for ``train.py`` under ``torchrun``: one process per GPU, the whole model on every GPU.

No DDP wrapper (it fights gradient checkpointing and PEFT): every rank runs the single-GPU code on its share of
the rows, and right before each optimizer step ``all_reduce_grads`` sums the trainable parameters' gradients
across ranks. ``train.py`` scales each row's loss by the global accumulation (``1 / grad_accum``), so the sum is
exactly the single-process gradient of the same rows (up to float summation order).

Without ``torchrun`` (``WORLD_SIZE`` unset or 1) ``Dist()`` is a single process and every helper is a no-op,
so a plain ``python train.py`` runs the same code path as before.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any, TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")
BUCKET_BYTES = 256 << 20
TIMEOUT_MIN = 60  # rank 0 writes checkpoints and reports while the others wait in the next collective


class Dist:
    def __init__(self, rank: int = 0, world: int = 1, device: torch.device | None = None) -> None:
        self.rank = rank
        self.world = world
        self.device = device or torch.device("cpu")

    @classmethod
    def from_env(cls, device: str) -> Dist:
        """Join the ``torchrun`` group (NCCL on ``cuda``, gloo on ``cpu``), or a single process without one.
        Under ``torchrun`` a ``cuda`` device becomes ``cuda:LOCAL_RANK``."""
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if world == 1:
            return cls()
        rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
        kind = torch.device(device).type
        if kind == "cuda":
            torch.cuda.set_device(local)
            target = torch.device("cuda", local)
        else:
            target = torch.device(kind)
        minutes = int(os.environ.get("CLEF_FT_DIST_TIMEOUT_MIN", TIMEOUT_MIN))
        dist.init_process_group("nccl" if kind == "cuda" else "gloo", timeout=timedelta(minutes=minutes))
        return cls(rank, world, target)

    @property
    def enabled(self) -> bool:
        return self.world > 1

    @property
    def main(self) -> bool:
        return self.rank == 0

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()

    def barrier(self) -> None:
        if not self.enabled:
            return
        if self.device.type == "cuda":
            dist.barrier(device_ids=[self.device.index])
        else:
            dist.barrier()

    def main_first(self, fn: Callable[[], T]) -> T:
        """``fn()`` on rank 0, then on the others (a download or cache that rank 0 fills for everyone)."""
        if self.main:
            value = fn()
            self.barrier()
            return value
        self.barrier()
        return fn()

    # ---- rows

    def shard(self, items: Sequence[T]) -> list[T]:
        """This rank's items: every ``world``-th from ``rank`` (``gather`` puts them back in order)."""
        return list(items[self.rank::self.world])

    def gather(self, part: list[Any]) -> list[Any]:
        """Every rank's ``shard`` result back in the original order, on every rank."""
        if not self.enabled:
            return list(part)
        parts: list[Any] = [None] * self.world
        dist.all_gather_object(parts, part)
        out: list[Any] = [None] * sum(len(p) for p in parts)
        for rank, items in enumerate(parts):
            out[rank::self.world] = items
        return out

    # ---- numbers

    def _reduce(self, values: Sequence[float], op: Any) -> list[float]:
        if not self.enabled:
            return [float(v) for v in values]
        tensor = torch.tensor([float(v) for v in values], dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=op)
        return tensor.tolist()

    def sum(self, *values: float) -> list[float]:
        return self._reduce(values, dist.ReduceOp.SUM)

    def max(self, *values: float) -> list[float]:
        return self._reduce(values, dist.ReduceOp.MAX)

    # ---- parameters

    def broadcast_params(self, params: Sequence[torch.Tensor]) -> None:
        """Rank 0's values everywhere (the LoRA init is seeded alike on every rank; this makes sure)."""
        if self.enabled:
            for param in params:
                dist.broadcast(param.data, 0)

    def all_reduce_grads(self, params: Sequence[torch.nn.Parameter], bucket_bytes: int = BUCKET_BYTES) -> None:
        """Sum ``param.grad`` over ranks, in flat buckets. A rank that saw no rows for a parameter contributes
        zeros; a parameter no rank has a gradient for keeps ``grad = None`` (AdamW skips it, as in one process)."""
        if not self.enabled:
            return
        has = torch.tensor([p.grad is not None for p in params], dtype=torch.uint8, device=self.device)
        dist.all_reduce(has, op=dist.ReduceOp.MAX)
        grads = []
        for param, flag in zip(params, has.tolist()):
            if flag:
                if param.grad is None:
                    param.grad = torch.zeros_like(param)
                grads.append(param.grad)
        for bucket in _buckets(grads, bucket_bytes):
            flat = torch.cat([g.reshape(-1) for g in bucket])
            dist.all_reduce(flat)
            offset = 0
            for grad in bucket:
                grad.copy_(flat[offset:offset + grad.numel()].view_as(grad))
                offset += grad.numel()


def _buckets(tensors: list[torch.Tensor], limit: int) -> list[list[torch.Tensor]]:
    """Consecutive runs of same-dtype, same-device tensors of at most ``limit`` bytes (a larger tensor alone)."""
    buckets: list[list[torch.Tensor]] = []
    size = 0
    for tensor in tensors:
        nbytes = tensor.numel() * tensor.element_size()
        last = buckets[-1] if buckets else None
        if (last is None or last[0].dtype != tensor.dtype or last[0].device != tensor.device
                or size + nbytes > limit):
            buckets.append([tensor])
            size = nbytes
        else:
            last.append(tensor)
            size += nbytes
    return buckets
