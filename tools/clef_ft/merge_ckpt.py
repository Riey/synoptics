"""Write a Clef backbone checkpoint with a clef-ft LoRA already merged, shard by shard (no full model in memory).

Same arithmetic as ``serve_clef.merge_lora``: W_bf16 <- bf16(float32(W) + alpha/r * float32(B) @ float32(A)), run on
``--device`` (default cuda:0, like the in-server merge). Every other tensor and every non-weight file is copied
unchanged, so the output directory is a drop-in ``CLEF_WEIGHTS`` for serve_clef.py (which can then quantize while
loading instead of holding the whole BF16 model on the GPUs first).

Usage (on the node, in a container with /v mounted):
  python merge_ckpt.py --base /v/models/clef --adapter /v/models/clef-ft-20261004c --out /v/models/clef-merged-20261004c
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    config = json.loads((args.adapter / "adapter" / "adapter_config.json").read_text())
    if config.get("use_dora") or config.get("use_rslora") or config.get("fan_in_fan_out"):
        raise SystemExit(f"unsupported LoRA config in {args.adapter}")
    scale = float(config["lora_alpha"]) / float(config["r"])
    lora = load_file(str(args.adapter / "adapter" / "adapter_model.safetensors"))
    targets = {key[len("base_model.model."):-len(".lora_A.weight")] + ".weight": key
               for key in lora if key.endswith(".lora_A.weight")}
    weight_map = json.loads((args.base / "model.safetensors.index.json").read_text())["weight_map"]
    missing = sorted(t for t in targets if t not in weight_map)
    if missing:
        raise SystemExit(f"{len(missing)} LoRA targets not in the checkpoint, e.g. {missing[:3]}")

    tmp = args.out.with_name(args.out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    shards = sorted(set(weight_map.values()))
    for path in args.base.iterdir():
        if path.name not in shards and path.is_file():
            shutil.copy2(path, tmp / path.name)

    device = torch.device(args.device)
    merged = 0
    started = time.time()
    for shard in shards:
        with safe_open(str(args.base / shard), framework="pt") as f:
            metadata = f.metadata()
        tensors = load_file(str(args.base / shard))
        for name, tensor in tensors.items():
            key = targets.get(name)
            if key is None:
                continue
            a = lora[key].to(device, torch.float32)
            b = lora[key.replace(".lora_A.", ".lora_B.")].to(device, torch.float32)
            with torch.no_grad():
                tensors[name] = (tensor.to(device).float() + scale * (b @ a)).to(tensor.dtype).cpu()
            merged += 1
        save_file(tensors, str(tmp / shard), metadata=metadata)
        print(f"{shard}: done ({merged} merged so far, {time.time() - started:.0f}s)", flush=True)
    if merged != len(targets):
        raise SystemExit(f"merged {merged} tensors, expected {len(targets)}")
    digest = hashlib.sha256((args.adapter / "adapter" / "adapter_model.safetensors").read_bytes()).hexdigest()
    (tmp / "clef_merge.json").write_text(json.dumps(
        {"base": str(args.base), "adapter": str(args.adapter), "adapter_sha256": digest, "scale": scale,
         "merged_tensors": merged, "device": args.device, "torch": torch.__version__}, indent=1))
    shutil.rmtree(args.out, ignore_errors=True)
    tmp.rename(args.out)
    print(f"wrote {args.out}: {merged} LoRA tensors merged in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
