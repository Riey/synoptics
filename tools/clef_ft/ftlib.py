"""Torch pieces of the Clef fine-tuning recipe, kept small so they can be tested on CPU.

* ``key_loss``: label-smoothed cross entropy + Brier per question, weighted by key (current step 2.0), plus a
  false-``yes`` penalty ``-log(1 - p_yes)`` on questions whose label is not ``yes`` (a false yes moves the app on).
* ``select_lora_targets``: the ``nn.Linear`` modules inside the language model's decoder layers -- full
  attention (``self_attn.{q,k,v,o}_proj``), linear attention / gated delta (``linear_attn.in_proj_*``,
  ``out_proj``) and MLP (``mlp.{gate,up,down}_proj``) -- never the vision tower, ``lm_head`` or the Clef head.
  Names are found at run time, not hard coded, and summarised for the log.
* ``CastingEmbedding``: the head only indexes the output embedding (``weight[token_ids]``); this view returns those
  rows in the head's dtype so a fp32 head never needs a fp32 copy of the whole vocab matrix.
* ``warmup_cosine``: LR multiplier, linear warmup then cosine to 0.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import torch
import torch.nn.functional as F

OPTIONS = ("no", "unsure", "yes")  # encode_record sorts choice options by key
LABEL_SMOOTHING = 0.1
YES = OPTIONS.index("yes")


def key_loss(logits: Sequence[torch.Tensor], targets: Sequence[int], weights: Sequence[float],
             smoothing: float = LABEL_SMOOTHING, false_yes_penalty: float = 0.0) -> torch.Tensor:
    """Weighted mean over a record's questions of CE(label smoothing) + Brier (sum of squared prob errors)
    + ``false_yes_penalty * -log(1 - p_yes)`` when the label is not ``yes``.

    ``logits[i]`` holds one logit per option of question ``i``; ``targets[i]`` is the option index.
    """
    if not (len(logits) == len(targets) == len(weights)) or not logits:
        raise ValueError("logits, targets and weights must be non-empty and of the same length")
    total = logits[0].new_zeros((), dtype=torch.float32)
    for lg, target, weight in zip(logits, targets, weights):
        lg = lg.float().unsqueeze(0)
        index = torch.tensor([target], device=lg.device)
        onehot = F.one_hot(index, lg.shape[-1]).float()
        ce = F.cross_entropy(lg, index, label_smoothing=smoothing)
        brier = ((lg.softmax(-1) - onehot) ** 2).sum()
        term = ce + brier
        if false_yes_penalty and target != YES:
            # log(1 - p_yes) = logsumexp of the other options' logits - logsumexp of all, stable for p_yes -> 1
            others = torch.cat([lg[0, :YES], lg[0, YES + 1:]])
            term = term - false_yes_penalty * (others.logsumexp(-1) - lg[0].logsumexp(-1))
        total = total + float(weight) * term
    return total / float(sum(weights))


_LAYER = re.compile(r"(^|\.)language_model\.layers\.(\d+)\.")
_EXCLUDED = {"visual", "lm_head", "head"}  # path components never wrapped


def select_lora_targets(model: torch.nn.Module) -> tuple[list[str], dict[str, object]]:
    """Full names of the ``nn.Linear`` modules in the language model's decoder layers, and a summary.

    The summary counts targets per block kind (``self_attn`` / ``linear_attn`` / ``mlp``) and per leaf name, and
    lists which layer indexes use full vs linear attention, so the log shows what LoRA wraps.
    """
    targets: list[str] = []
    kinds: Counter[str] = Counter()
    leaves: Counter[str] = Counter()
    attention: dict[str, set[int]] = {"self_attn": set(), "linear_attn": set()}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        match = _LAYER.search(name)
        if match is None or _EXCLUDED & set(name.split(".")):
            continue
        rest = name[match.end():].split(".")
        targets.append(name)
        kinds[rest[0]] += 1
        leaves[rest[-1]] += 1
        if rest[0] in attention:
            attention[rest[0]].add(int(match.group(2)))
    summary = {
        "count": len(targets),
        "by_block": dict(sorted(kinds.items())),
        "by_leaf": dict(sorted(leaves.items())),
        "full_attention_layers": sorted(attention["self_attn"]),
        "linear_attention_layers": sorted(attention["linear_attn"]),
    }
    return targets, summary


class CastingEmbedding:
    """``weight[ids]`` in ``dtype``, without materialising the whole matrix in that dtype."""

    def __init__(self, weight: torch.Tensor, dtype: torch.dtype) -> None:
        self.weight = weight
        self.dtype = dtype

    def __getitem__(self, index: torch.Tensor) -> torch.Tensor:
        return self.weight[index].to(self.dtype)


def warmup_cosine(step: int, total: int, warmup_frac: float = 0.05) -> float:
    """LR multiplier at optimizer step ``step`` (0-based) of ``total``."""
    warmup = max(1, math.ceil(total * warmup_frac))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
