"""Clef 27B + a clef-ft adapter, served precision vs FP8: the same test rows through ``systemone`` on one GPU.

Modes: ``bf16`` (what the node serves: LoRA merged into the BF16 backbone), ``fp8w`` (torchao float8 weight-only on
the language model's Linear layers after the merge), ``fp8dyn`` (float8 dynamic activation + float8 weight, per-row
scales). The vision tower, lm_head and the Clef head stay BF16. Writes one jsonl row per sample (probabilities per
key, ms) and a summary (load s, weights / peak GPU memory, ms p50/p95).

Usage (on the training instance, one GPU per mode):
  python fp8_eval.py --mode fp8w --device cuda:1 --model /workspace/models/clef \
      --adapter /workspace/adapters/clef-ft-20261004b --data /workspace/data/clef_ft_data --out /workspace/out/fp8
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE / "vendor"):
    sys.path.insert(0, str(path))

import torch  # noqa: E402
from PIL import Image  # noqa: E402


def merge_lora(backbone, adapter_dir: Path) -> int:
    """W += alpha/r * B @ A for every LoRA pair (same as serve_clef.py)."""
    from safetensors.torch import load_file

    config = json.loads((adapter_dir / "adapter" / "adapter_config.json").read_text())
    scale = float(config["lora_alpha"]) / float(config["r"])
    tensors = load_file(str(adapter_dir / "adapter" / "adapter_model.safetensors"))
    modules = dict(backbone.named_modules())
    merged = 0
    for key in tensors:
        if not key.endswith(".lora_A.weight"):
            continue
        weight = modules[key[len("base_model.model."):-len(".lora_A.weight")]].weight
        a = tensors[key].to(weight.device, torch.float32)
        b = tensors[key.replace(".lora_A.", ".lora_B.")].to(weight.device, torch.float32)
        with torch.no_grad():
            weight.copy_((weight.float() + scale * (b @ a)).to(weight.dtype))
        del a, b
        merged += 1
    del tensors
    torch.cuda.empty_cache()
    return merged


def quantize(backbone, mode: str) -> int:
    from torchao.quantization import quantize_

    if mode == "fp8w":
        from torchao.quantization import Float8WeightOnlyConfig
        config = Float8WeightOnlyConfig()
    else:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
    count = 0

    def keep(module: torch.nn.Module, fqn: str) -> bool:
        nonlocal count
        ok = isinstance(module, torch.nn.Linear) and "language_model" in fqn and "lm_head" not in fqn
        count += ok
        return ok

    quantize_(backbone, config, filter_fn=keep)
    torch.cuda.empty_cache()
    return count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("bf16", "fp8w", "fp8dyn"), required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from joint_schema_model import ClefModel, JointSchemaHead, systemone
    from patch import patch_conv3d
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    t0 = time.time()
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                                                               device_map={"": str(device)})
    backbone.config.use_cache = False
    merged = merge_lora(backbone, args.adapter)
    quantized = quantize(backbone, args.mode) if args.mode != "bf16" else 0
    head = JointSchemaHead(**json.loads((args.adapter / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(str(args.adapter / "head.safetensors")), strict=True)
    model = ClefModel(backbone, head.to(device=device, dtype=torch.bfloat16)).eval()
    patch_conv3d(model)
    processor = AutoProcessor.from_pretrained(args.model)
    torch.cuda.synchronize(device)
    load_s = time.time() - t0
    weights_gb = torch.cuda.memory_allocated(device) / 2**30
    torch.cuda.reset_peak_memory_stats(device)
    print(f"[{args.mode}] loaded in {load_s:.1f}s, merged {merged}, quantized {quantized}, "
          f"weights {weights_gb:.2f} GiB", flush=True)

    rows = [json.loads(l) for l in (args.data / "dataset.jsonl").open()]
    rows = [r for r in rows if r["split"] == args.split]
    if args.limit:
        rows = rows[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    ms = []
    with (args.out / f"{args.mode}_{args.split}.jsonl").open("w") as f, torch.inference_mode():
        for i, r in enumerate(rows):
            image = Image.open(args.data / r["image"]).convert("RGB")
            body = {"model": "clef", "state": r["record"]["state"], "images": [image],
                    "questions": r["record"]["questions"]}
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            answer = systemone(model, processor, body)
            torch.cuda.synchronize(device)
            ms.append((time.perf_counter() - started) * 1000)
            f.write(json.dumps({"sample_id": r["sample_id"], "ms": round(ms[-1], 1),
                                "probs": {k: a["probabilities"] for k, a in answer["answers"].items()}}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"[{args.mode}] {i + 1}/{len(rows)} p50 {statistics.median(ms):.0f} ms", flush=True)
    ordered = sorted(ms)
    summary = {"mode": args.mode, "rows": len(rows), "load_s": round(load_s, 1), "merged": merged,
               "quantized_linears": quantized, "weights_gib": round(weights_gb, 2),
               "peak_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
               "reserved_gib": round(torch.cuda.max_memory_reserved(device) / 2**30, 2),
               "ms_p50": round(statistics.median(ms), 1),
               "ms_p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 1),
               "ms_first": round(ms[0], 1)}
    (args.out / f"{args.mode}_{args.split}_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
