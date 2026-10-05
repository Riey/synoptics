"""Clef follower fine-tuning: zero-shot, head-only and LoRA+head arms on the multidomain set.

Runs on the rented GPU instance (``run_on_instance.sh`` drives it); every arm also runs on CPU with a tiny model
(``tests/test_train_cpu.py``), which checks the whole path except CUDA, bf16 numerics and memory.

Arms (``--arm``):

* ``zero``: no training. Backbone hidden states for val/test are computed (and cached under ``--cache``, which
  the head arm reuses) and the release head predicts them.
* ``head``: backbone frozen; hidden states of every row cached once (disk), the head trained in fp32 on them:
  AdamW lr 2e-5 wd 0.01, 10 epochs, one row per step, key-weighted CE(label smoothing 0.1) + Brier.
* ``lora``: PEFT LoRA r=32 alpha=64 dropout 0.05 on every ``nn.Linear`` of the language model's decoder layers
  (full and linear attention projections, MLP; found at run time, see ``ftlib.select_lora_targets``), vision
  tower / lm_head / head excluded. BF16 backbone, gradient checkpointing, fp32 head copy; LoRA lr 1e-4, head
  lr 2e-5, wd 0.01, 3 epochs, one row per forward, gradient accumulation 16, 5% warmup + cosine, seed 0.

Every trained epoch: val predictions + metrics with the head cast back to the backbone dtype (what the server
runs), and a checkpoint (``adapter/`` for lora + ``head.safetensors`` in the backbone dtype +
``joint_head_config.json``). The selected epoch (``ftutil.select_checkpoint``: best val current-step key
accuracy with val false-yes rate <= the zero-shot val rate) is reloaded from disk and predicts test once.
Non-selected checkpoints are pruned unless ``--keep-all``.

Outputs in ``--out``: ``run.json`` (args, versions, LoRA targets, kernels), ``throughput.json`` (first 20 train
rows/s, eval rows/s, peak GPU memory), ``zero_val_*``, ``epoch_N/``, ``selected.json``, ``test_predictions.jsonl``,
``test_metrics.{json,md}``, ``result.json`` (summary read by ``ftutil.py done``).

Several GPUs on one machine (``torchrun --nproc_per_node N train.py ...``, see ``ftdist.py``): every rank loads the
whole model on its own GPU; each gradient-accumulation group of ``--grad-accum`` rows (the global batch) is split
across ranks and the LoRA / head gradients are summed before the optimizer step, so the effective batch, the
shuffle (one seeded order for all ranks) and the LR schedule are those of one process. Evaluation and the
hidden-state cache are split across ranks too; rank 0 alone writes files and logs (rows/s are global). The zero /
head arms use the other ranks only for the cache. One process (plain ``python``) is the code path as before.

Checkpoint selection (``--select-on``, default ``orig``) scores val without its near / mined hard-negative rows
(``ftutil.is_near``), which the zero-shot false-yes cap was never met on; the full val metrics are written beside.

Serving a checkpoint: load the release model, ``PeftModel.from_pretrained(backbone, epoch_N/adapter)``, load
``epoch_N/head.safetensors`` into the head (``ClefModel.forward`` already unwraps a PEFT backbone).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import metrics as M  # noqa: E402
from ftlib import OPTIONS, CastingEmbedding, key_loss, select_lora_targets, warmup_cosine  # noqa: E402
from ftdist import Dist  # noqa: E402
from ftutil import is_near, select_checkpoint  # noqa: E402
from joint_schema_model import JointSchemaHead, collate_records, encode_record, load_release_model  # noqa: E402
from patch import patch_conv3d  # noqa: E402

HF_REPOS = {"clef-flash": "Cloudflare/clef-flash", "clef": "Cloudflare/clef"}
WORKSPACE = Path(os.environ.get("CLEF_FT_WORKSPACE", "/workspace"))
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
THROUGHPUT_STEPS = 20
DIST = Dist()  # the torchrun group (main() joins it); a single process otherwise


def log(message: str) -> None:
    if DIST.main:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, value: Any) -> None:
    if not DIST.main:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not DIST.main:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------------------------------ setup


def resolve_model(name: str, models_dir: Path) -> Path:
    """A local model directory: ``name`` itself, ``models_dir/name``, or a fresh HF download into the latter."""
    if Path(name).is_dir():
        return Path(name)
    local = models_dir / name
    if (local / "joint_head.safetensors").exists():
        return local
    if name not in HF_REPOS:
        raise SystemExit(f"model {name!r}: not a directory and not one of {sorted(HF_REPOS)}")
    from huggingface_hub import snapshot_download

    log(f"downloading {HF_REPOS[name]} -> {local}")
    snapshot_download(HF_REPOS[name], local_dir=str(local))
    return local


def kernel_status() -> dict[str, bool]:
    status = {name: importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d")}
    if not status["fla"]:
        log("WARNING: flash-linear-attention (fla) not importable; gated delta layers use the slow torch path")
    return status


def versions() -> dict[str, str]:
    out = {"torch": torch.__version__, "python": sys.version.split()[0]}
    for name in ("transformers", "peft", "accelerate", "safetensors", "tokenizers", "huggingface_hub"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = "missing"
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
        out["cuda"] = str(torch.version.cuda)
    return out


def load_rows(data_dir: Path, splits: tuple[str, ...], limit: int | None) -> dict[str, list[dict[str, Any]]]:
    with (data_dir / "dataset.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    out = {split: [row for row in rows if row["split"] == split] for split in splits}
    if limit:
        out = {split: sel[:limit] for split, sel in out.items()}
    return out


# ------------------------------------------------------------------------------------------------ model


class Runner:
    """The loaded model, processor and the per-row encode / forward / predict steps."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = DIST.device if DIST.enabled else torch.device(args.device)
        self.dtype = DTYPES[args.dtype]
        self.model_dir = DIST.main_first(lambda: resolve_model(args.model, args.models))
        t0 = time.time()
        self.model, self.processor = load_release_model(self.model_dir, device=self.device, dtype=self.dtype)
        self.swapped = patch_conv3d(self.model)
        self.load_s = round(time.time() - t0, 1)
        log(f"loaded {self.model_dir} in {self.load_s}s on {self.device} ({args.dtype}); conv3d->linear {self.swapped}")
        self.model.requires_grad_(False)
        self.backbone = self.model.language_model
        self.release_head = self.model.head
        self.head_config = json.loads((self.model_dir / "joint_head_config.json").read_text())
        self.pad = self.processor.tokenizer.pad_token_id
        self.tag = f"{self.model_dir.name}-{args.dtype}"

    # ---- encoding

    def encode(self, row: dict[str, Any]):
        with Image.open(self.args.data / row["image"]) as image:
            record = {"id": row["sample_id"], "state": row["record"]["state"], "questions": row["record"]["questions"],
                      "images": [image.convert("RGB")]}
        encoded = encode_record(self.processor.tokenizer, record, processor=self.processor)
        for question in encoded.questions:
            if question.option_ids != OPTIONS:
                raise ValueError(f"{row['sample_id']}/{question.question_id}: options {question.option_ids}")
        if [q.question_id for q in encoded.questions] != list(row["labels"]):
            raise ValueError(f"{row['sample_id']}: question order differs from the labels")
        return encoded

    def batch(self, encoded) -> dict[str, Any]:
        return collate_records([encoded], self.pad, self.device)

    def embeddings(self) -> torch.Tensor:
        base = self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone
        return base.get_output_embeddings().weight

    def hidden(self, batch: dict[str, Any]) -> torch.Tensor:
        base = self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone
        outputs = base.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False,
                             return_dict=True, **batch["media"])
        return outputs.last_hidden_state

    # ---- hidden-state cache (zero / head arms)

    def cache_path(self, row: dict[str, Any]) -> Path:
        key = hashlib.sha1(json.dumps([row["record"], row["image_sha256"]], ensure_ascii=False,
                                      sort_keys=True).encode()).hexdigest()[:12]
        return self.args.cache / self.tag / f"{row['sample_id']}_{key}.pt"

    def cached(self, row: dict[str, Any]) -> dict[str, Any]:
        path = self.cache_path(row)
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=False)
        encoded = self.encode(row)
        batch = self.batch(encoded)
        with torch.no_grad():
            hidden = self.hidden(batch)
        item = {"hidden": hidden[0].to("cpu", self.dtype), "input_ids": batch["input_ids"][0].cpu(),
                "encoded": dataclasses.replace(encoded, media=None)}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(item, tmp)
        tmp.replace(path)
        return item

    def cache_rows(self, rows: list[dict[str, Any]]) -> float:
        """Cache every row (each rank its shard); returns rows/s over all ranks for the rows that needed a
        forward (0 when all were cached)."""
        todo = [row for row in DIST.shard(rows) if not self.cache_path(row).exists()]
        total = int(DIST.sum(len(todo))[0])
        log(f"hidden cache {self.tag}: {len(rows) - total} cached, {total} to compute")
        t0 = time.time()
        for index, row in enumerate(todo, 1):
            self.cached(row)
            if index % 50 == 0 or index == len(todo):
                log(f"  cached {index}/{len(todo)} ({index / (time.time() - t0):.2f} rows/s"
                    + (f" on rank 0 of {DIST.world})" if DIST.enabled else ")"))
        sync(self.device)
        DIST.barrier()
        elapsed = time.time() - t0
        return total / elapsed if total else 0.0

    def head_logits_cached(self, head: torch.nn.Module, item: dict[str, Any], dtype: torch.dtype,
                           emb: Any) -> list[torch.Tensor]:
        ids = item["input_ids"].unsqueeze(0).to(self.device)
        mask = torch.ones_like(ids)
        hidden = item["hidden"].unsqueeze(0).to(self.device, dtype)
        return head(hidden, ids, mask, [item["encoded"]], emb)[0]

    # ---- predictions

    def predict_cached(self, head: torch.nn.Module, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Predict with ``head`` (already in the backbone dtype) from cached hidden states."""
        emb = self.embeddings()
        head.eval()
        out = []
        with torch.no_grad():
            for row in rows:
                item = self.cached(row)
                out.append(prediction(row, item["encoded"], self.head_logits_cached(head, item, self.dtype, emb)))
        return out

    def predict_live(self, head: torch.nn.Module, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
        """Full forward per row (the lora arm; each rank its shard, gathered on every rank in ``rows`` order);
        returns predictions and rows/s over all ranks."""
        emb = self.embeddings()
        was_training = self.backbone.training
        self.backbone.eval()
        head.eval()
        out = []
        t0 = time.time()
        with torch.no_grad():
            for row in DIST.shard(rows):
                encoded = self.encode(row)
                batch = self.batch(encoded)
                hidden = self.hidden(batch)
                logits = head(hidden.to(self.dtype), batch["input_ids"], batch["attention_mask"], [encoded], emb)[0]
                out.append(prediction(row, encoded, logits))
        sync(self.device)
        out = DIST.gather(out)
        rate = len(rows) / (time.time() - t0) if rows else 0.0
        self.backbone.train(was_training)
        return out, rate

    def head_copy(self, head: torch.nn.Module) -> torch.nn.Module:
        """``head`` cast to the backbone dtype (what the server runs), detached from training."""
        return copy.deepcopy(head).to(dtype=self.dtype).eval().requires_grad_(False)

    def save_head(self, head: torch.nn.Module, directory: Path) -> None:
        from safetensors.torch import save_file

        if not DIST.main:
            return
        directory.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().to("cpu", self.dtype).contiguous() for k, v in head.state_dict().items()},
                  str(directory / "head.safetensors"))
        write_json(directory / "joint_head_config.json", self.head_config)

    def load_head(self, directory: Path) -> torch.nn.Module:
        from safetensors.torch import load_file

        head = JointSchemaHead(**self.head_config)
        head.load_state_dict(load_file(str(directory / "head.safetensors")), strict=True)
        return head.to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_gb(device: torch.device, reserved: bool = False) -> float | None:
    """Peak tensor memory (or the caching allocator's peak reservation, closer to what the GPU must hold)."""
    if device.type != "cuda":
        return None
    peak = torch.cuda.max_memory_reserved(device) if reserved else torch.cuda.max_memory_allocated(device)
    return round(peak / 1e9, 2)


def prediction(row: dict[str, Any], encoded, logits: list[torch.Tensor]) -> dict[str, Any]:
    probs = {q.question_id: dict(zip(q.option_ids, [round(p, 5) for p in lg.float().softmax(-1).tolist()]))
             for q, lg in zip(encoded.questions, logits)}
    return {"sample_id": row["sample_id"], "split": row["split"], "domain": row["domain"],
            **({"sample_kind": row["sample_kind"]} if "sample_kind" in row else {}),
            "current_step": row["current_step"], "labels": row["labels"],
            "pred": {key: max(p, key=p.get) for key, p in probs.items()}, "probs": probs}


def targets_and_weights(row: dict[str, Any], encoded) -> tuple[list[int], list[float]]:
    keys = [q.question_id for q in encoded.questions]
    return [OPTIONS.index(row["labels"][k]) for k in keys], [float(row["weights"][k]) for k in keys]


def report(out: Path, name: str, preds: list[dict[str, Any]]) -> dict[str, Any]:
    write_jsonl(out / f"{name}_predictions.jsonl", preds)
    metrics = M.write_report(preds, out / f"{name}_metrics", name) if DIST.main else M.compute(preds)
    log(f"{name}: key_acc {M._fmt(metrics['key_acc'])} current_acc {M._fmt(metrics['current_acc'])} "
        f"false_yes {M._fmt(metrics['false_yes'])} current_false_yes {M._fmt(metrics['current_false_yes'])}")
    return metrics


def report_val(out: Path, name: str, preds: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Metrics on all of ``preds``, and on the rows that are not near rows (``{name}_orig_*`` files) when val has
    near rows (None otherwise: the two would be the same)."""
    metrics = report(out, name, preds)
    orig = [p for p in preds if not is_near(p)]
    if len(orig) == len(preds) or not orig:
        return metrics, None
    log(f"{name}: {len(preds) - len(orig)} near rows; scoring the {len(orig)} others apart")
    return metrics, report(out, f"{name}_orig", orig)


def headline(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {key: metrics[key] for key in ("rows", "key_acc", "row_exact", "false_yes", "current_acc",
                                          "current_false_yes", "goal_false_yes", "current_unsure_precision",
                                          "current_unsure_recall")}


# ------------------------------------------------------------------------------------------------ arms


def run_zero(runner: Runner, rows: dict[str, list[dict[str, Any]]], out: Path) -> dict[str, Any]:
    eval_rows = rows["val"] + rows["test"]
    rate = runner.cache_rows(eval_rows)
    if not DIST.main:  # the other ranks only share the cache
        return {"throughput": {"eval_rows_per_s": round(rate, 3)}}
    head = runner.release_head
    val, val_orig = report_val(out, "zero_val", runner.predict_cached(head, rows["val"]))
    test = report(out, "zero_test", runner.predict_cached(head, rows["test"]))
    result = {"throughput": {"eval_rows_per_s": round(rate, 3)}, "val_metrics": headline(val),
              "test_metrics": headline(test)}
    if val_orig is not None:
        result["val_orig_metrics"] = headline(val_orig)
    return result


def finish(runner: Runner, out: Path, epochs: list[dict[str, Any]], zero_val: dict[str, Any],
           zero_val_orig: dict[str, Any] | None, test_predict, keep_all: bool) -> dict[str, Any]:
    """Select an epoch on val (without near rows under ``--select-on orig`` when val has any), predict test with
    it, prune the others. Every rank runs it in the lora arm (``test_predict`` is split across ranks)."""
    use_orig = runner.args.select_on == "orig" and zero_val_orig is not None
    scored = [{"epoch": e["epoch"], "val": e["val_orig"] if use_orig else e["val"]} for e in epochs]
    zero = zero_val_orig if use_orig else zero_val
    selected = select_checkpoint(scored, zero["false_yes"]["rate"])
    dirs = {e["epoch"]: e.get("dir", f"epoch_{e['epoch']}") for e in epochs}
    selected["dir"] = dirs[selected["epoch"]]
    if "label" in next(e for e in epochs if e["epoch"] == selected["epoch"]):
        selected["label"] = next(e for e in epochs if e["epoch"] == selected["epoch"])["label"]
    selected["select_on"] = "orig" if use_orig else "all"
    selected["val_rows"] = zero["rows"]
    selected["val_near_rows"] = zero_val["rows"] - zero_val_orig["rows"] if zero_val_orig is not None else 0
    if use_orig:
        selected["rule"] += " (val without near rows)"
    write_json(out / "selected.json", selected)
    log(f"selected {selected['dir']} on {selected['select_on']} val ({zero['rows']} rows; "
        f"constraint met: {selected['constraint_met']})")
    test = report(out, "test", test_predict(out / selected["dir"]))
    if not keep_all and DIST.main:
        for entry in epochs:
            if entry["epoch"] != selected["epoch"]:
                directory = out / dirs[entry["epoch"]]
                shutil.rmtree(directory / "adapter", ignore_errors=True)
                (directory / "head.safetensors").unlink(missing_ok=True)
    chosen = next(e for e in epochs if e["epoch"] == selected["epoch"])
    result = {"selected": selected, "val_metrics": headline(chosen["val"]), "test_metrics": headline(test),
              "zero_val_metrics": headline(zero_val),
              "epochs": [{"epoch": e["epoch"], **({"label": e["label"]} if "label" in e else {}),
                          "train_loss": e["train_loss"], "val": headline(e["val"]),
                          **({"val_orig": headline(e["val_orig"])} if e["val_orig"] is not None else {})}
                         for e in epochs]}
    if zero_val_orig is not None:
        result["val_orig_metrics"] = headline(chosen["val_orig"])
        result["zero_val_orig_metrics"] = headline(zero_val_orig)
    return result


def run_head(runner: Runner, rows: dict[str, list[dict[str, Any]]], out: Path) -> dict[str, Any]:
    args = runner.args
    rate = runner.cache_rows(rows["val"] + rows["test"] + rows["train"])
    if not DIST.main:  # head training on cached states is cheap: rank 0 alone
        return {"throughput": {"eval_rows_per_s": round(rate, 3)}}
    zero_val, zero_val_orig = report_val(out, "zero_val", runner.predict_cached(runner.release_head, rows["val"]))

    head = copy.deepcopy(runner.release_head).float().train().requires_grad_(True)
    emb32 = CastingEmbedding(runner.embeddings(), torch.float32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)
    epochs: list[dict[str, Any]] = []
    steps = 0
    for epoch in range(1, args.epochs + 1):
        order = rows["train"][:]
        rng.shuffle(order)
        total, count = 0.0, 0
        for row in order:
            item = runner.cached(row)
            logits = runner.head_logits_cached(head, item, torch.float32, emb32)
            loss = key_loss(logits, *targets_and_weights(row, item["encoded"]),
                            false_yes_penalty=args.false_yes_penalty)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
            count += 1
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        val, val_orig = report_val(out / f"epoch_{epoch}", "val",
                                   runner.predict_cached(runner.head_copy(head), rows["val"]))
        runner.save_head(head, out / f"epoch_{epoch}")
        head.train()
        epochs.append({"epoch": epoch, "train_loss": round(total / max(count, 1), 5), "val": val, "val_orig": val_orig})
        log(f"epoch {epoch}: train loss {total / max(count, 1):.4f} ({count} rows)")
        if args.max_steps and steps >= args.max_steps:
            break

    result = finish(runner, out, epochs, zero_val, zero_val_orig,
                    lambda d: runner.predict_cached(runner.load_head(d), rows["test"]), args.keep_all)
    result["throughput"] = {"eval_rows_per_s": round(rate, 3)}
    return result


def run_lora(runner: Runner, rows: dict[str, list[dict[str, Any]]], out: Path) -> dict[str, Any]:
    from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict

    args = runner.args
    zero_preds, eval_rate = runner.predict_live(runner.release_head, rows["val"])
    zero_val, zero_val_orig = report_val(out, "zero_val", zero_preds)

    targets, summary = select_lora_targets(runner.backbone)
    log(f"LoRA targets: {json.dumps(summary)}")
    write_json(out / "lora_targets.json", {"summary": summary, "targets": targets})
    if not targets:
        raise SystemExit("no LoRA target modules found")
    runner.backbone.config.use_cache = False
    runner.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    log(f"gradient checkpointing: {runner.backbone.is_gradient_checkpointing}")
    peft_model = get_peft_model(runner.backbone, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=targets,
        bias="none"))
    runner.backbone = peft_model
    lora_params = [p for p in peft_model.parameters() if p.requires_grad]
    log(f"trainable LoRA params {sum(p.numel() for p in lora_params) / 1e6:.2f} M "
        f"(dtype {lora_params[0].dtype})")
    head = copy.deepcopy(runner.release_head).float().train().requires_grad_(True)
    emb32 = CastingEmbedding(runner.embeddings(), torch.float32)
    trainable = lora_params + list(head.parameters())
    DIST.broadcast_params(trainable)
    if DIST.enabled:  # one seed made the init above alike on every rank; the LoRA dropout masks should differ
        torch.manual_seed(args.seed + DIST.rank)

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lora_lr},
        {"params": list(head.parameters()), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    per_epoch = math.ceil(len(rows["train"]) / args.grad_accum)
    total_updates = per_epoch * args.epochs
    planned = len(rows["train"]) * args.epochs
    if args.max_steps:
        total_updates = min(total_updates, math.ceil(args.max_steps / args.grad_accum))
        planned = min(planned, args.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: warmup_cosine(s, total_updates))

    rng = random.Random(args.seed)  # the same order on every rank
    epochs: list[dict[str, Any]] = []
    micro, updates = 0, 0  # global rows (all ranks), optimizer steps
    train_s = 0.0
    timing: list[float] = []  # this rank's first rows
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runner.device)

    def update() -> None:
        nonlocal updates
        DIST.all_reduce_grads(trainable)
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1

    def checkpoint(epoch: int, part: int, groups_done: int, groups: int, total: float, count: int) -> None:
        """Evaluate val and save adapter + head as checkpoint ``len(epochs) + 1`` (every rank calls it)."""
        nonlocal eval_rate
        total, count = DIST.sum(total, count)
        ordinal = len(epochs) + 1
        name = f"epoch_{epoch}" if args.evals_per_epoch == 1 else f"ckpt_{ordinal}"
        label = f"{epoch - 1 + groups_done / groups:.2f}"
        val_preds, rate = runner.predict_live(runner.head_copy(head), rows["val"])
        eval_rate = rate or eval_rate
        val, val_orig = report_val(out / name, "val", val_preds)
        if DIST.main:
            peft_model.save_pretrained(str(out / name / "adapter"))
        runner.save_head(head, out / name)
        peft_model.train()
        head.train()
        entry = {"epoch": ordinal, "dir": name, "train_loss": round(total / max(count, 1), 5), "val": val,
                 "val_orig": val_orig}
        if args.evals_per_epoch > 1:
            entry["label"] = f"epoch {label}"
        epochs.append(entry)
        where = f"epoch {epoch}" if args.evals_per_epoch == 1 else f"{name} (epoch {label}, part {part})"
        log(f"{where}: train loss {total / max(count, 1):.4f} ({int(count)} rows, {updates} updates)")

    peft_model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        order = rows["train"][:]
        rng.shuffle(order)
        total, count = 0.0, 0
        groups = math.ceil(len(order) / args.grad_accum)
        # checkpoints after these group counts (the last one is the epoch end)
        marks = sorted({max(1, round(groups * j / args.evals_per_epoch)) for j in range(1, args.evals_per_epoch + 1)})
        done_marks = 0
        groups_done = 0
        # One accumulation group = one optimizer step; each rank takes its share of the group's rows.
        for start in range(0, len(order), args.grad_accum):
            group = order[start:start + args.grad_accum]
            if args.max_steps:
                group = group[:args.max_steps - micro]
            group_t0 = time.time()
            mine = DIST.shard(group)
            for index, row in enumerate(mine, 1):
                started = time.time()
                encoded = runner.encode(row)
                batch = runner.batch(encoded)
                hidden = runner.hidden(batch)
                logits = head(hidden.float(), batch["input_ids"], batch["attention_mask"], [encoded], emb32)[0]
                loss = key_loss(logits, *targets_and_weights(row, encoded), false_yes_penalty=args.false_yes_penalty)
                (loss / args.grad_accum).backward()  # the global group size: the rank sum is the group gradient
                total += loss.item()
                count += 1
                if index == len(mine):
                    update()
                if len(timing) < THROUGHPUT_STEPS:
                    sync(runner.device)
                    timing.append(time.time() - started)
                    if len(timing) == THROUGHPUT_STEPS:
                        rank_note = f" on rank 0 (x{DIST.world} ranks)" if DIST.enabled else ""
                        log(f"first {THROUGHPUT_STEPS} rows{rank_note}: {THROUGHPUT_STEPS / sum(timing):.3f} rows/s, "
                            f"peak GPU {peak_memory_gb(runner.device)} GB "
                            f"(reserved {peak_memory_gb(runner.device, reserved=True)} GB)")
            if not mine:  # a group smaller than the world: this rank still joins the gradient sum
                update()
            micro += len(group)
            train_s += time.time() - group_t0
            if micro // 50 > (micro - len(group)) // 50:
                rate = micro / train_s
                log(f"epoch {epoch} row {start + len(group)}/{len(order)} loss {total / max(count, 1):.4f} "
                    f"lr {scheduler.get_last_lr()[0]:.2e} updates {updates} | {rate:.3f} rows/s, "
                    f"train eta {(planned - micro) / rate / 60:.1f} min")
            groups_done += 1
            stop = bool(args.max_steps and micro >= args.max_steps)
            if done_marks < len(marks) and (groups_done == marks[done_marks] or stop):
                done_marks += 1
                checkpoint(epoch, done_marks, groups_done, groups, total, count)
            if stop:
                break
        if args.max_steps and micro >= args.max_steps:
            break

    def test_predict(directory: Path) -> list[dict[str, Any]]:
        DIST.barrier()  # rank 0 wrote the checkpoint every rank loads
        set_peft_model_state_dict(peft_model, load_peft_weights(str(directory / "adapter"), device=str(runner.device)))
        preds, _ = runner.predict_live(runner.load_head(directory), rows["test"])
        return preds

    # Rows/s over the timed rows, skipping the first (allocator / kernel warm-up) when there are more; the ranks
    # run in lockstep (one gradient sum per group), so their rates add up to the global rate.
    timed = timing[1:] if len(timing) > 1 else timing
    train_rps, timed_rows = DIST.sum(len(timed) / sum(timed) if timed else 0.0, len(timed))
    peak, reserved = peak_memory_gb(runner.device), peak_memory_gb(runner.device, reserved=True)
    if DIST.enabled and peak is not None:
        peak, reserved = (round(v, 2) for v in DIST.max(peak, reserved))
    throughput = {
        "train_rows_per_s": round(train_rps, 4) if timed_rows else None,
        "train_rows_timed": int(timed_rows),
        "eval_rows_per_s": round(eval_rate, 4),
        "peak_gpu_memory_gb": peak,
        "peak_gpu_reserved_gb": reserved,
        "grad_accum": args.grad_accum,
        "world_size": DIST.world,
    }
    write_json(out / "throughput.json", throughput)
    result = finish(runner, out, epochs, zero_val, zero_val_orig, test_predict, args.keep_all)
    result["throughput"] = throughput
    result["lora"] = {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                      "targets": summary, "updates": updates, "rows": micro}
    return result


# ------------------------------------------------------------------------------------------------ main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clef follower fine-tuning (zero / head / lora)")
    parser.add_argument("--model", required=True, help="clef-flash | clef | a local model directory")
    parser.add_argument("--arm", required=True, choices=("zero", "head", "lora"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=WORKSPACE / "data" / "clef_ft_data")
    parser.add_argument("--models", type=Path, default=WORKSPACE / "models")
    parser.add_argument("--cache", type=Path, default=WORKSPACE / "cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many training rows (smoke)")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N rows of each split (smoke)")
    parser.add_argument("--epochs", type=int, default=None, help="default: head 10, lora 3")
    parser.add_argument("--head-lr", type=float, default=2e-5)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--false-yes-penalty", type=float, default=1.0,
                        help="extra -log(1 - p_yes) on non-yes labels (0 = the 2026-10-04 run's loss)")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-all", action="store_true", help="keep every epoch's checkpoint")
    parser.add_argument("--evals-per-epoch", type=int, default=1,
                        help="lora: evaluate val and save a checkpoint this many times per epoch (evenly spaced "
                             "optimizer steps, the last at the epoch end); selection is over all of them")
    parser.add_argument("--select-on", choices=("orig", "all"), default="orig",
                        help="val rows the checkpoint is selected on: orig = without near rows (when val has any)")
    args = parser.parse_args(argv)
    if args.epochs is None:
        args.epochs = {"zero": 0, "head": 10, "lora": 3}[args.arm]
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    global DIST
    args = parse_args(argv)
    DIST = Dist.from_env(args.device)
    try:
        return run(args)
    finally:
        DIST.close()
        DIST = Dist()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    kernels = kernel_status()
    write_json(args.out / "run.json", {"argv": sys.argv, "args": {k: str(v) for k, v in vars(args).items()},
                                       "versions": versions(), "kernels": kernels, "world_size": DIST.world})
    rows = load_rows(args.data, ("train", "val", "test"), args.limit or None)
    log(f"rows: " + ", ".join(f"{split} {len(sel)}" for split, sel in rows.items())
        + (f"; {DIST.world} ranks" if DIST.enabled else ""))
    runner = Runner(args)
    if runner.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runner.device)
    result = {"zero": run_zero, "head": run_head, "lora": run_lora}[args.arm](runner, rows, args.out)
    peak = peak_memory_gb(runner.device)
    if args.arm == "lora" and DIST.enabled and peak is not None:  # every rank is still here only in lora
        peak = round(DIST.max(peak)[0], 2)
    result.update({"model": str(args.model), "model_dir": str(runner.model_dir), "arm": args.arm,
                   "seconds": round(time.time() - started, 1), "load_s": runner.load_s,
                   "max_gpu_memory_gb": peak, "kernels": kernels, "conv3d_swapped": runner.swapped,
                   "world_size": DIST.world})
    write_json(args.out / "result.json", result)
    log(f"done in {result['seconds']}s, peak GPU {result['max_gpu_memory_gb']} GB -> {args.out}")
    return result


if __name__ == "__main__":
    main()
