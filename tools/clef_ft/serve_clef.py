"""Clef decision server: POST /v1/systemone (SystemOne request body), GET /health. Loopback only.

Env: CLEF_MODEL (default /v/models/clef-flash), CLEF_DEVICE (default cuda), CLEF_PORT (default 8085).
CLEF_ADAPTER (optional): a clef-ft checkpoint directory (``adapter/`` PEFT LoRA + ``head.safetensors`` +
``joint_head_config.json``, tools/clef_ft/train.py). The LoRA deltas are merged into the BF16 backbone at load
(W += alpha/r * B @ A; no peft dependency, no per-call cost) and the fine-tuned head replaces the release head.
CLEF_QUANT (optional): ``fp8dyn`` quantizes the language model's Linear layers to float8 after the merge (torchao
0.14.1 in /v/clef/pydeps; see ``quantize_fp8``).
Request images are base64 strings or data URLs; they are decoded to PIL before ``systemone``.

Kept in git as tools/clef_ft/serve_clef.py; deployed as /v/clef/serve.py on the node.
"""

import base64
import io
import os
import sys
import threading
import time

MODEL_DIR = os.environ.get("CLEF_MODEL", "/v/models/clef-flash")
sys.path.insert(0, MODEL_DIR)

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from PIL import Image  # noqa: E402
from joint_schema_model import load_release_model, systemone  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from patch import patch_conv3d  # noqa: E402

DEVICE = os.environ.get("CLEF_DEVICE", "cuda")
ADAPTER_DIR = os.environ.get("CLEF_ADAPTER", "").strip()
HEAD_DIR = ADAPTER_DIR or MODEL_DIR
HEAD_FILE = "head.safetensors" if ADAPTER_DIR else "joint_head.safetensors"
# CLEF_QUANT=fp8dyn: after the merge, the language model's Linear layers become float8 (dynamic activation + weight,
# per-row scales; torchao). Measured 2026-10-04 on test 557 (RTX PRO 6000, same Blackwell sm_120 as the 5090):
# 98.0% verdict agreement with BF16 at yes_min 0.7, same false-yes counts, p50 311 vs 356 ms, weights 28.5 vs 51.2 GiB.
QUANT = os.environ.get("CLEF_QUANT", "").strip()
# CLEF_WEIGHTS (optional): a backbone checkpoint with CLEF_ADAPTER's LoRA already merged (tools/clef_ft/merge_ckpt.py).
# The server then skips the in-memory merge (the adapter still supplies the head) and, with CLEF_QUANT=fp8dyn,
# quantizes while loading. Config, tokenizer and processor files are read from the same directory.
WEIGHTS_DIR = os.environ.get("CLEF_WEIGHTS", "").strip()


def quantize_fp8(backbone) -> int:
    """Float8 dynamic activation + float8 weight (PerRow) on the language model's nn.Linear layers only (the vision
    tower, embeddings and lm_head stay BF16; the Clef head indexes lm_head rows)."""
    import torch
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, quantize_

    count = 0

    def keep(module, fqn: str) -> bool:
        nonlocal count
        ok = isinstance(module, torch.nn.Linear) and "language_model" in fqn and "lm_head" not in fqn
        count += ok
        return ok

    quantize_(backbone, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()), filter_fn=keep)
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
    return count


def merge_lora(backbone, adapter_dir: str) -> int:
    """Merge a PEFT LoRA (saved from ``get_peft_model(backbone)``) into ``backbone`` in place; returns #modules."""
    import json
    from pathlib import Path

    import torch
    from safetensors.torch import load_file

    config = json.loads((Path(adapter_dir) / "adapter" / "adapter_config.json").read_text())
    if config.get("use_dora") or config.get("use_rslora") or config.get("fan_in_fan_out"):
        raise SystemExit(f"unsupported LoRA config in {adapter_dir}")
    scale = float(config["lora_alpha"]) / float(config["r"])
    tensors = load_file(str(Path(adapter_dir) / "adapter" / "adapter_model.safetensors"))
    modules = dict(backbone.named_modules())
    merged = 0
    for key in tensors:
        if not key.endswith(".lora_A.weight"):
            continue
        name = key[len("base_model.model."):-len(".lora_A.weight")]
        module = modules.get(name)
        if module is None or not hasattr(module, "weight"):
            raise SystemExit(f"LoRA target {name} not in the backbone")
        weight = module.weight
        a = tensors[key].to(weight.device, torch.float32)
        b = tensors[key.replace(".lora_A.", ".lora_B.")].to(weight.device, torch.float32)
        with torch.no_grad():
            weight.copy_((weight.float() + scale * (b @ a)).to(weight.dtype))
        del a, b
        merged += 1
    # Each merge makes FP32 temporaries the size of the target weight; the caching allocator keeps those blocks
    # reserved, which showed up as +1.6 GiB/+0.8 GiB over the base model (2026-10-04) and pushed the tracker that
    # shares GPU1 into CUDA OOM. Hand them back before the server starts answering.
    del tensors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return merged


def load_split(path: str) -> tuple:
    """``load_release_model`` with the backbone split over every visible GPU (Clef 27B BF16 = 52 GB)."""
    import json
    from pathlib import Path

    import torch
    from joint_schema_model import ClefModel, JointSchemaHead
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    # "28GiB" for every GPU, or one value per GPU: "24GiB,29GiB"
    caps = os.environ.get("CLEF_MAX_MEMORY", "28GiB").split(",")
    count = torch.cuda.device_count()
    max_memory = {i: caps[i] if len(caps) > 1 else caps[0] for i in range(count)}
    if QUANT and QUANT != "fp8dyn":
        raise SystemExit(f"unknown CLEF_QUANT {QUANT!r} (fp8dyn or empty)")
    if QUANT:
        # transformers pre-allocates one model-sized block per GPU before loading (caching_allocator_warmup); the
        # BF16 weights then share those segments with what stays, so after float8 conversion ~11 GiB per GPU stayed
        # reserved (2026-10-04: 14.2 GiB allocated, 25 GiB reserved). Without the warmup each weight has its own
        # segment and empty_cache() can return the freed BF16 memory.
        import transformers.modeling_utils as modeling_utils
        modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    kwargs = {}
    if WEIGHTS_DIR and QUANT:
        # Pre-merged checkpoint: quantize each Linear as it is loaded, so the GPUs never hold the BF16 model
        # (load peak ~= the float8 model instead of ~27 GB per GPU). Same config and the same layers as
        # quantize_fp8: nn.Linear weights outside the vision tower and lm_head.
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
        from transformers import TorchAoConfig

        kwargs["quantization_config"] = TorchAoConfig(
            quant_type=Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
            modules_to_not_convert=["model.visual", "lm_head"])
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(
        WEIGHTS_DIR or path, dtype=torch.bfloat16, device_map=os.environ.get("CLEF_DEVICE_MAP", "sequential"),
        max_memory=max_memory, **kwargs)
    backbone.config.use_cache = False
    if ADAPTER_DIR and not WEIGHTS_DIR:
        print(f"LoRA merged into {merge_lora(backbone, ADAPTER_DIR)} modules from {ADAPTER_DIR}", flush=True)
    if QUANT and not WEIGHTS_DIR:
        print(f"float8 (dynamic act + weight, per-row) on {quantize_fp8(backbone)} Linear layers", flush=True)
    elif QUANT:
        from torchao.quantization import Float8Tensor

        names = [n for n, m in backbone.named_modules()
                 if isinstance(m, torch.nn.Linear) and isinstance(m.weight, Float8Tensor)]
        stray = [n for n in names if "language_model" not in n]
        if stray:
            raise SystemExit(f"float8 outside the language model: {stray[:3]}")
        print(f"float8 at load (dynamic act + weight, per-row) on {len(names)} Linear layers", flush=True)
    head = JointSchemaHead(**json.loads((Path(HEAD_DIR) / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(Path(HEAD_DIR) / HEAD_FILE), strict=True)
    head_device = backbone.get_output_embeddings().weight.device
    head = head.to(device=head_device, dtype=torch.bfloat16)
    forward = head.forward

    def forward_on_head_device(hidden, input_ids, attention_mask, records, embeddings):
        return forward(hidden.to(head_device), input_ids.to(head_device), attention_mask.to(head_device),
                       records, embeddings)

    head.forward = forward_on_head_device
    print("device map:", sorted({str(v) for v in backbone.hf_device_map.values()}), "head on", head_device, flush=True)
    return ClefModel(backbone, head).eval(), AutoProcessor.from_pretrained(path)


_t0 = time.time()
if DEVICE == "split":
    model, processor = load_split(MODEL_DIR)
elif ADAPTER_DIR:
    raise SystemExit("CLEF_ADAPTER is only wired for CLEF_DEVICE=split")
else:
    model, processor = load_release_model(MODEL_DIR, device=DEVICE)
SWAPPED = 0 if os.environ.get("CLEF_NO_PATCH") else patch_conv3d(model)
LOAD_S = round(time.time() - _t0, 1)
print(f"loaded {MODEL_DIR} on {DEVICE} in {LOAD_S}s, conv3d->linear swapped {SWAPPED}", flush=True)

_lock = threading.Lock()  # sync endpoints run in a thread pool; one forward pass at a time
app = FastAPI()


def _image(value: object) -> Image.Image:
    if not isinstance(value, str):
        raise ValueError("image must be a base64 string or data URL")
    data = value.split(",", 1)[1] if value.startswith("data:") else value
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def _gpu_memory() -> dict:
    import torch

    return {str(i): {"allocated": round(torch.cuda.memory_allocated(i) / 2**30, 2),
                     "reserved": round(torch.cuda.memory_reserved(i) / 2**30, 2)}
            for i in range(torch.cuda.device_count())}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "model": os.path.basename(MODEL_DIR), "adapter": os.path.basename(ADAPTER_DIR) or None,
            "quant": QUANT or None, "weights": os.path.basename(WEIGHTS_DIR) or None, "gpu_gib": _gpu_memory(),
            "device": DEVICE, "load_s": LOAD_S}


@app.post("/v1/systemone")
def decide(body: dict) -> dict:
    try:
        if "images" in body:
            body = {**body, "images": [_image(v) for v in body["images"]]}
        with _lock:
            started = time.perf_counter()
            try:
                out = systemone(model, processor, body)
            finally:
                # The tracker shares GPU1. Keep weights resident, but return unused forward buffers
                # before accepting another call so GroundingDINO can acquire its FP32 workspace.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            out["server_ms"] = round((time.perf_counter() - started) * 1000, 1)
    except (ValueError, KeyError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return out


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("CLEF_PORT", "8085")))
