"""Real HCCL all-reduce/all-gather probe, launched with torchrun."""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from shensi.utils.accelerator import device


def main():
    dev = device("npu", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        value = torch.tensor([float(rank + 1)], device=dev)
        dist.all_reduce(value)
        expected = world * (world + 1) / 2
        if value.item() != expected:
            raise AssertionError(f"all_reduce: {value.item()} != {expected}")
        own = torch.tensor([rank], device=dev, dtype=torch.int64)
        gathered = [torch.empty_like(own) for _ in range(world)]
        dist.all_gather(gathered, own)
        if [x.item() for x in gathered] != list(range(world)):
            raise AssertionError("all_gather mismatch")
        torch.npu.synchronize()
        if rank == 0:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "backend": "hccl",
                        "world_size": world,
                        "all_reduce": value.item(),
                        "scope": "collectives only",
                    }
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
