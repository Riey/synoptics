"""``ftdist`` helpers on two gloo ranks under torchrun (CPU), and as a single process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

PKG = Path(__file__).resolve().parents[1]

WORKER = r"""
import json, sys
sys.path.insert(0, {pkg!r})
import torch
from ftdist import Dist

d = Dist.from_env("cpu")
out = {{"rank": d.rank, "world": d.world, "main": d.main}}
rows = list(range(7))
mine = d.shard(rows)
out["shard"] = mine
out["gathered"] = d.gather([r * 10 for r in mine])
out["sum"] = d.sum(d.rank + 1, 0.5)
out["max"] = d.max(d.rank * 3)

# p1: both ranks have a gradient; p2: only rank 1; p3: nobody; p4 (fp64, own bucket) only rank 0.
params = [torch.nn.Parameter(torch.zeros(5)), torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2)),
          torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))]
params[0].grad = torch.full((5,), float(d.rank + 1))
if d.rank == 1:
    params[1].grad = torch.arange(3.0)
if d.rank == 0:
    params[3].grad = torch.ones(4, dtype=torch.float64)
d.all_reduce_grads(params, bucket_bytes=24)
out["grads"] = [None if p.grad is None else p.grad.tolist() for p in params]

value = torch.nn.Parameter(torch.full((2,), float(d.rank)))
d.broadcast_params([value])
out["broadcast"] = value.tolist()
out["first"] = d.main_first(lambda: d.rank)
d.barrier()
d.close()
open(sys.argv[1] + f"/rank{{out['rank']}}.json", "w").write(json.dumps(out))
"""


def test_two_gloo_ranks(tmp_path: Path) -> None:
    script = tmp_path / "worker.py"
    script.write_text(WORKER.format(pkg=str(PKG)))
    done = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
                           str(script), str(tmp_path)], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "OMP_NUM_THREADS": "1"})
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in (0, 1)]
    assert [r["rank"] for r in results] == [0, 1] and results[0]["main"] and not results[1]["main"]
    assert results[0]["shard"] == [0, 2, 4, 6] and results[1]["shard"] == [1, 3, 5]
    for r in results:
        assert r["world"] == 2
        assert r["gathered"] == [x * 10 for x in range(7)]
        assert r["sum"] == [3.0, 1.0] and r["max"] == [3.0]
        assert r["grads"] == [[3.0] * 5, [0.0, 1.0, 2.0], None, [1.0] * 4]
        assert r["broadcast"] == [0.0, 0.0]
    assert [r["first"] for r in results] == [0, 1]


def test_single_process_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    import ftdist

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    d = ftdist.Dist.from_env("cuda")
    assert not d.enabled and d.main and d.world == 1
    assert d.shard([1, 2, 3]) == [1, 2, 3] and d.gather([1, 2]) == [1, 2]
    assert d.sum(2, 3) == [2.0, 3.0] and d.max(4) == [4.0]
    param = torch.nn.Parameter(torch.zeros(2))
    d.all_reduce_grads([param])
    assert param.grad is None
    assert d.main_first(lambda: 7) == 7
    d.barrier()
    d.close()


def test_buckets_split_by_size_and_dtype() -> None:
    from ftdist import _buckets

    a, b, c = torch.zeros(4), torch.zeros(4), torch.zeros(4, dtype=torch.float64)
    big = torch.zeros(100)
    assert [len(x) for x in _buckets([a, b, c, big], 32)] == [2, 1, 1]
    assert [len(x) for x in _buckets([a, b], 16)] == [1, 1]
