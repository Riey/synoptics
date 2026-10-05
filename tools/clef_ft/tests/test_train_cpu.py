"""End-to-end ``train.py`` on CPU with a tiny random Qwen3.5 Clef (real tokenizer/processor, real module names).

Needs torch, transformers, peft and ``CLEF_FT_TOKENIZER_DIR``: a directory with the clef-flash ``config.json``,
``tokenizer.json``, ``tokenizer_config.json``, ``processor_config.json``, ``chat_template.jinja``
(copy them from ``/v/models/clef-flash``). Skipped otherwise. Checks everything but CUDA, bf16 numerics and memory.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

TOKENIZER_DIR = Path(os.environ.get("CLEF_FT_TOKENIZER_DIR", "/nonexistent"))
pytestmark = pytest.mark.skipif(not (TOKENIZER_DIR / "tokenizer.json").exists(),
                                reason="CLEF_FT_TOKENIZER_DIR with the clef-flash tokenizer files not set")

TRAIN = Path(__file__).resolve().parents[1] / "train.py"
HEAD_CONFIG = {"hidden_size": 64, "width": 32, "routing_layers": 1, "layers": 1, "heads": 2, "feedforward": 64}


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from safetensors.torch import save_file
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    from joint_schema_model import JointSchemaHead

    out = tmp_path_factory.mktemp("tiny-clef")
    config = json.loads((TOKENIZER_DIR / "config.json").read_text())
    text = config["text_config"]
    text.update({"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
                 "layer_types": ["linear_attention"] * 3 + ["full_attention"], "num_attention_heads": 2,
                 "num_key_value_heads": 1, "head_dim": 128, "linear_key_head_dim": 16, "linear_num_key_heads": 2,
                 "linear_num_value_heads": 4, "linear_value_head_dim": 16})
    text["rope_parameters"]["mrope_section"] = [6, 5, 5]  # head_dim 128 * 0.25 / 2
    config["vision_config"].update({"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                                    "out_hidden_size": 64})
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**config))
    model.save_pretrained(out)
    for name in ("tokenizer.json", "tokenizer_config.json", "processor_config.json", "chat_template.jinja"):
        shutil.copy(TOKENIZER_DIR / name, out / name)
    (out / "joint_head_config.json").write_text(json.dumps(HEAD_CONFIG))
    head = JointSchemaHead(**HEAD_CONFIG)
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(out / "joint_head.safetensors"))
    return out


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import build_data
    from conftest import make_source

    root = tmp_path_factory.mktemp("data")
    make_source(root / "src")
    build_data.build(root / "src", root, expected={"train": 4, "val": 2, "test": 2})
    return root / build_data.DATA_DIR


def run(tiny_model: Path, data_dir: Path, tmp_path: Path, *extra: str, dtype: str = "fp32") -> tuple[dict, Path]:
    import train

    out = tmp_path / "out"
    result = train.main(["--model", str(tiny_model), "--data", str(data_dir), "--cache", str(tmp_path / "cache"),
                         "--out", str(out), "--device", "cpu", "--dtype", dtype, *extra])
    return result, out


def test_zero_arm(tiny_model: Path, data_dir: Path, tmp_path: Path) -> None:
    result, out = run(tiny_model, data_dir, tmp_path, "--arm", "zero")
    assert result["val_metrics"]["rows"] == 2 and result["test_metrics"]["rows"] == 2
    preds = [json.loads(line) for line in (out / "zero_val_predictions.jsonl").read_text().splitlines()]
    assert all(set(p["pred"]) == set(p["labels"]) for p in preds)
    assert all(abs(sum(p["probs"]["goal"].values()) - 1) < 1e-3 for p in preds)
    assert len(list((tmp_path / "cache").rglob("*.pt"))) == 4  # val + test cached for the head arm


def test_head_arm_trains_selects_and_prunes(tiny_model: Path, data_dir: Path, tmp_path: Path) -> None:
    result, out = run(tiny_model, data_dir, tmp_path, "--arm", "head", "--epochs", "3", "--head-lr", "1e-3")
    assert [e["epoch"] for e in result["epochs"]] == [1, 2, 3]
    chosen = result["selected"]["epoch"]
    assert (out / f"epoch_{chosen}" / "head.safetensors").exists()
    assert [p.parent.name for p in out.glob("epoch_*/head.safetensors")] == [f"epoch_{chosen}"]
    assert (out / "test_predictions.jsonl").exists() and (out / "test_metrics.md").exists()
    assert result["epochs"][0]["train_loss"] > 0
    # Training moved the head away from the release weights.
    from safetensors.torch import load_file
    trained = load_file(str(out / f"epoch_{chosen}" / "head.safetensors"))
    release = load_file(str(tiny_model / "joint_head.safetensors"))
    assert any(not torch.equal(trained[k], release[k]) for k in release)


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_lora_arm_smoke_path(tiny_model: Path, data_dir: Path, tmp_path: Path, dtype: str) -> None:
    """The instance smoke shape (``--max-steps``), in fp32 and in the bf16 backbone the instance runs."""
    result, out = run(tiny_model, data_dir, tmp_path, "--arm", "lora", "--max-steps", "3", "--grad-accum", "2",
                      "--lora-r", "4", "--lora-alpha", "8", dtype=dtype)
    targets = json.loads((out / "lora_targets.json").read_text())
    summary = targets["summary"]
    assert summary["by_block"] == {"linear_attn": 15, "mlp": 12, "self_attn": 4}
    assert summary["full_attention_layers"] == [3] and summary["linear_attention_layers"] == [0, 1, 2]
    assert not any("visual" in t or "lm_head" in t for t in targets["targets"])
    assert result["lora"]["rows"] == 3 and result["lora"]["updates"] == 2  # one full accumulation + the flush
    assert result["throughput"]["train_rows_per_s"] > 0 and result["throughput"]["eval_rows_per_s"] > 0
    ckpt = out / f"epoch_{result['selected']['epoch']}"
    assert (ckpt / "adapter" / "adapter_config.json").exists() and (ckpt / "head.safetensors").exists()
    config = json.loads((ckpt / "adapter" / "adapter_config.json").read_text())
    assert config["r"] == 4 and config["lora_alpha"] == 8
    from safetensors.torch import load_file
    adapter = load_file(str(ckpt / "adapter" / "adapter_model.safetensors"))
    # lora_B starts at zero; after updates it is not.
    assert any(k.endswith("lora_B.weight") and v.abs().sum() > 0 for k, v in adapter.items())
    assert not any("visual" in k for k in adapter)
    assert result["test_metrics"]["rows"] == 2
    if dtype == "bf16":
        from safetensors.torch import load_file as load
        assert {v.dtype for v in load(str(ckpt / "head.safetensors")).values()} == {torch.bfloat16}


def test_lora_mid_epoch_checkpoints(tiny_model: Path, data_dir: Path, tmp_path: Path) -> None:
    """``--evals-per-epoch 2``: two checkpoints per epoch (ckpt_N dirs), selection over all, the rest pruned."""
    result, out = run(tiny_model, data_dir, tmp_path, "--arm", "lora", "--epochs", "1", "--grad-accum", "1",
                      "--evals-per-epoch", "2", "--lora-r", "4", "--lora-alpha", "8", "--lora-lr", "1e-3")
    entries = result["epochs"]
    assert [e["epoch"] for e in entries] == [1, 2]
    assert [e["label"] for e in entries] == ["epoch 0.50", "epoch 1.00"]
    chosen = out / result["selected"]["dir"]
    assert result["selected"]["dir"] in ("ckpt_1", "ckpt_2") and (chosen / "adapter").is_dir()
    other = {"ckpt_1", "ckpt_2"} - {result["selected"]["dir"]}
    assert not (out / other.pop() / "adapter").exists()
    assert (out / "ckpt_1" / "val_metrics.json").exists() and (out / "ckpt_2" / "val_metrics.json").exists()
    assert result["lora"]["updates"] == result["lora"]["rows"]  # grad-accum 1: one step per row


@pytest.fixture(scope="module")
def near_data_dir(data_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same rows with the first val row marked as a near (mined hard-negative) row."""
    out = tmp_path_factory.mktemp("near") / data_dir.name
    shutil.copytree(data_dir, out)
    rows = [json.loads(line) for line in (out / "dataset.jsonl").read_text().splitlines()]
    next(row for row in rows if row["split"] == "val")["sample_kind"] = "near_s1"
    (out / "dataset.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    return out


def test_selection_scores_val_without_near_rows(tiny_model: Path, near_data_dir: Path, tmp_path: Path) -> None:
    result, out = run(tiny_model, near_data_dir, tmp_path, "--arm", "head", "--epochs", "2", "--head-lr", "1e-3")
    selected = json.loads((out / "selected.json").read_text())
    assert selected["select_on"] == "orig" and selected["val_rows"] == 1 and selected["val_near_rows"] == 1
    assert result["val_metrics"]["rows"] == 2 and result["val_orig_metrics"]["rows"] == 1
    assert result["zero_val_orig_metrics"]["rows"] == 1 and all("val_orig" in e for e in result["epochs"])
    assert (out / "zero_val_orig_metrics.json").exists() and (out / "epoch_1" / "val_orig_metrics.md").exists()
    preds = [json.loads(line) for line in (out / "zero_val_predictions.jsonl").read_text().splitlines()]
    assert [p.get("sample_kind") for p in preds].count("near_s1") == 1

    everything, _ = run(tiny_model, near_data_dir, tmp_path / "all", "--arm", "head", "--epochs", "2", "--head-lr",
                        "1e-3", "--select-on", "all")
    assert everything["selected"]["select_on"] == "all" and everything["selected"]["val_rows"] == 2


def run_cli(tiny_model: Path, data_dir: Path, out: Path, *extra: str, nproc: int = 1) -> dict:
    """``train.py`` as its own process, or ``nproc`` gloo ranks under torchrun; one CPU thread per process."""
    launcher = [sys.executable, str(TRAIN)] if nproc == 1 else [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", str(nproc), str(TRAIN)]
    done = subprocess.run([*launcher, "--model", str(tiny_model), "--data", str(data_dir), "--cache",
                           str(out / "cache"), "--out", str(out), "--device", "cpu", "--dtype", "fp32", *extra],
                          env={**os.environ, "OMP_NUM_THREADS": "1"}, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return json.loads((out / "result.json").read_text())


def gradient_free_masked(name: str, diff: "torch.Tensor") -> "torch.Tensor":
    """``diff`` without the head entries whose true gradient is zero: attention key biases (softmax is shift
    invariant per query) and the scorer's last bias (one constant on every option logit). AdamW turns their
    float-noise gradients into full +-lr steps, so they differ run to run without changing any output."""
    if name.endswith("in_proj_bias"):
        third = diff.numel() // 3
        return torch.cat([diff[:third], diff[2 * third:]])
    if name == "residual_scorer.3.bias":
        return diff * 0
    return diff


def test_lora_two_ranks_match_one_process(tiny_model: Path, data_dir: Path, tmp_path: Path) -> None:
    """Data parallel = one process: same updates, losses, LoRA / head weights and predictions, up to float
    summation order. Group size 3 over 4 train rows splits 2+1 and then 1+0 (an idle rank still joins the sum)."""
    from safetensors.torch import load_file

    args = ("--arm", "lora", "--epochs", "2", "--grad-accum", "3", "--lora-r", "4", "--lora-alpha", "8",
            "--lora-dropout", "0", "--lora-lr", "1e-3", "--head-lr", "1e-3", "--keep-all")
    one = run_cli(tiny_model, data_dir, tmp_path / "one", *args)
    two = run_cli(tiny_model, data_dir, tmp_path / "two", *args, nproc=2)
    assert one["world_size"] == 1 and two["world_size"] == 2 and two["throughput"]["world_size"] == 2
    assert one["lora"]["updates"] == two["lora"]["updates"] == 4 and one["lora"]["rows"] == two["lora"]["rows"] == 8
    assert one["selected"]["epoch"] == two["selected"]["epoch"]
    for a, b in zip(one["epochs"], two["epochs"]):
        assert b["train_loss"] == pytest.approx(a["train_loss"], rel=1e-5)
    worst = {"adapter": 0.0, "head": 0.0, "probs": 0.0}
    moved = 0.0
    release = load_file(str(tiny_model / "joint_head.safetensors"))
    for epoch in (1, 2):
        for kind, name in (("adapter", "adapter/adapter_model.safetensors"), ("head", "head.safetensors")):
            x = load_file(str(tmp_path / "one" / f"epoch_{epoch}" / name))
            y = load_file(str(tmp_path / "two" / f"epoch_{epoch}" / name))
            assert x.keys() == y.keys()
            worst[kind] = max(worst[kind], max(gradient_free_masked(k, x[k] - y[k]).abs().max().item() for k in x))
            if kind == "adapter":
                moved = max(moved, max(x[k].abs().max().item() for k in x if "lora_B" in k))
            else:
                assert any(not torch.equal(x[k], release[k]) for k in x)
    for name in ("zero_val", "test", "epoch_1/val", "epoch_2/val"):
        a = [json.loads(line) for line in (tmp_path / "one" / f"{name}_predictions.jsonl").read_text().splitlines()]
        b = [json.loads(line) for line in (tmp_path / "two" / f"{name}_predictions.jsonl").read_text().splitlines()]
        assert [p["sample_id"] for p in a] == [p["sample_id"] for p in b]
        for p, q in zip(a, b):
            assert p["pred"] == q["pred"]
            worst["probs"] = max(worst["probs"], max(abs(p["probs"][k][o] - q["probs"][k][o])
                                                     for k in p["probs"] for o in p["probs"][k]))
    print(f"1 vs 2 ranks: max |diff| {worst}, max |lora_B| {moved:.3e}")
    assert moved > 1e-3  # the LoRA weights moved, so the comparison means something
    assert worst["adapter"] < 1e-5 and worst["head"] < 1e-5 and worst["probs"] < 1e-4
