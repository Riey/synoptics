"""NCCL check for ``run_on_instance.sh setup`` (run under ``torchrun --nproc_per_node N``).

Every rank all-reduces a one (the sum must be the world size), then rank 0 times a 256 MB fp32 all-reduce: the
27B LoRA r32 + head gradient sum before each optimizer step is about 0.93 GB (``ftdist.all_reduce_grads``).
Also prints the most GPU memory any rank holds outside PyTorch tensors after that (CUDA context + NCCL buffers):
training needs it on top of the single-GPU peak.
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

ELEMENTS = 64 << 20  # 256 MB of fp32


def main() -> None:
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    one = torch.ones(1, device="cuda")
    dist.all_reduce(one)
    if one.item() != world:
        raise SystemExit(f"rank {rank}: all_reduce of ones gave {one.item()}, expected {world}")
    buf = torch.ones(ELEMENTS, device="cuda")
    dist.all_reduce(buf)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(3):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    seconds = (time.time() - t0) / 3
    del buf
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    outside = torch.tensor([(total - free - torch.cuda.memory_reserved()) / 1e9], device="cuda")
    dist.all_reduce(outside, op=dist.ReduceOp.MAX)
    if rank == 0:
        bus = ELEMENTS * 4 * 2 * (world - 1) / world / seconds / 1e9
        print(f"nccl ok: {world} ranks ({torch.cuda.get_device_name(local)}), 256 MB all_reduce "
              f"{seconds * 1000:.1f} ms, bus {bus:.1f} GB/s; 27B grad sum ~{seconds * 0.93 / 0.268:.2f} s per step; "
              f"context + NCCL {outside.item():.2f} GB per GPU (max), of {total / 1e9:.1f} GB", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
