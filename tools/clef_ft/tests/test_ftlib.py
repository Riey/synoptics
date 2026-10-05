from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from ftlib import CastingEmbedding, key_loss, select_lora_targets, warmup_cosine  # noqa: E402


def manual(logits: list[float], target: int, smoothing: float = 0.1) -> float:
    z = max(logits)
    exps = [math.exp(v - z) for v in logits]
    total = sum(exps)
    probs = [e / total for e in exps]
    logp = [math.log(p) for p in probs]
    k = len(logits)
    ce = -sum(((1 - smoothing) * (i == target) + smoothing / k) * logp[i] for i in range(k))
    brier = sum((probs[i] - (i == target)) ** 2 for i in range(k))
    return ce + brier


def test_key_loss_matches_hand_computation_and_weights() -> None:
    a, b = [2.0, -1.0, 0.5], [0.0, 0.0, 3.0]
    logits = [torch.tensor(a), torch.tensor(b)]
    got = key_loss(logits, [0, 1], [2.0, 1.0]).item()
    assert got == pytest.approx((2 * manual(a, 0) + manual(b, 1)) / 3, rel=1e-5)
    # Equal weights reduce to the plain mean; weights only re-balance keys.
    assert key_loss(logits, [0, 1], [1.0, 1.0]).item() == pytest.approx((manual(a, 0) + manual(b, 1)) / 2, rel=1e-5)


def test_key_loss_is_differentiable_and_lowest_at_the_target() -> None:
    lg = torch.zeros(3, requires_grad=True)
    loss = key_loss([lg], [2], [1.0])
    loss.backward()
    assert lg.grad[2] < 0 < lg.grad[0]
    assert key_loss([torch.tensor([0.0, 0.0, 9.0])], [2], [1.0]) < key_loss([torch.tensor([9.0, 0.0, 0.0])], [2], [1.0])


def test_false_yes_penalty_only_on_non_yes_labels() -> None:
    a = [0.5, -1.0, 2.0]  # p_yes high
    base_no, base_yes = key_loss([torch.tensor(a)], [0], [1.0]), key_loss([torch.tensor(a)], [2], [1.0])
    p_yes = torch.tensor(a).softmax(-1)[2].item()
    assert key_loss([torch.tensor(a)], [0], [1.0], false_yes_penalty=2.0).item() == pytest.approx(
        base_no.item() - 2.0 * math.log(1 - p_yes), rel=1e-5)
    assert key_loss([torch.tensor(a)], [2], [1.0], false_yes_penalty=2.0).item() == pytest.approx(base_yes.item())
    # stays finite when p_yes saturates
    assert torch.isfinite(key_loss([torch.tensor([0.0, 0.0, 80.0])], [1], [1.0], false_yes_penalty=1.0))
    # the gradient pushes the yes logit down harder for a non-yes label
    plain, pen = torch.tensor(a, requires_grad=True), torch.tensor(a, requires_grad=True)
    key_loss([plain], [1], [1.0]).backward()
    key_loss([pen], [1], [1.0], false_yes_penalty=1.0).backward()
    assert pen.grad[2] > plain.grad[2] > 0


def test_key_loss_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        key_loss([torch.zeros(3)], [0, 1], [1.0])


def fake_clef_backbone() -> torch.nn.Module:
    """Module tree named like Qwen3_5ForConditionalGeneration (+ a sibling Clef head)."""
    nn = torch.nn

    def linear_attn() -> nn.Module:
        m = nn.Module()
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
            setattr(m, name, nn.Linear(4, 4))
        m.conv1d = nn.Conv1d(4, 4, 3)
        return m

    def self_attn() -> nn.Module:
        m = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(m, name, nn.Linear(4, 4))
        return m

    def layer(kind: str) -> nn.Module:
        m = nn.Module()
        setattr(m, kind, linear_attn() if kind == "linear_attn" else self_attn())
        m.mlp = nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(m.mlp, name, nn.Linear(4, 4))
        return m

    root = nn.Module()
    root.model = nn.Module()
    root.model.language_model = nn.Module()
    root.model.language_model.embed_tokens = nn.Embedding(10, 4)
    root.model.language_model.layers = nn.ModuleList(
        [layer("linear_attn"), layer("linear_attn"), layer("linear_attn"), layer("self_attn")])
    root.model.visual = nn.Module()
    root.model.visual.blocks = nn.ModuleList([nn.Module()])
    root.model.visual.blocks[0].attn = nn.Module()
    root.model.visual.blocks[0].attn.qkv = nn.Linear(4, 4)
    # a vision module that also happens to sit under "layers" must still be excluded
    root.model.visual.language_model = nn.Module()
    root.model.visual.language_model.layers = nn.ModuleList([self_attn()])
    root.model.visual.merger = nn.Module()
    root.model.visual.merger.linear_fc1 = nn.Linear(4, 4)
    root.lm_head = nn.Linear(4, 10)
    root.head = nn.Module()
    root.head.memory_projection = nn.Linear(4, 4)
    return root


def test_lora_targets_are_only_language_layer_linears() -> None:
    targets, summary = select_lora_targets(fake_clef_backbone())
    assert all(t.startswith("model.language_model.layers.") for t in targets)
    assert not any("visual" in t or "lm_head" in t or t.startswith("head.") or "conv1d" in t for t in targets)
    assert summary["count"] == len(targets) == 3 * (5 + 3) + 1 * (4 + 3)
    assert summary["by_block"] == {"linear_attn": 15, "mlp": 12, "self_attn": 4}
    assert summary["full_attention_layers"] == [3] and summary["linear_attention_layers"] == [0, 1, 2]
    assert summary["by_leaf"]["in_proj_qkv"] == 3 and summary["by_leaf"]["q_proj"] == 1


def test_lora_targets_inside_a_peft_wrapper_path() -> None:
    wrapper = torch.nn.Module()
    wrapper.base_model = torch.nn.Module()
    wrapper.base_model.model = fake_clef_backbone()
    targets, _ = select_lora_targets(wrapper)
    assert targets and all(t.startswith("base_model.model.model.language_model.layers.") for t in targets)


def test_casting_embedding_indexes_in_the_requested_dtype() -> None:
    weight = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    view = CastingEmbedding(weight, torch.float32)
    rows = view[torch.tensor([1, 3])]
    assert rows.dtype == torch.float32 and rows.tolist() == [[3, 4, 5], [9, 10, 11]]


def test_warmup_cosine() -> None:
    total = 100
    assert warmup_cosine(0, total) == pytest.approx(1 / 5)
    assert warmup_cosine(4, total) == pytest.approx(1.0)
    assert warmup_cosine(5, total) == pytest.approx(1.0)
    assert warmup_cosine(52, total) == pytest.approx(0.5, abs=0.02)
    assert warmup_cosine(99, total) < 0.01
