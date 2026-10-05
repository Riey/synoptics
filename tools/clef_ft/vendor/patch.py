"""Swap the vision patch-embed Conv3d for an exact matmul.

On this stack (torch 2.9.1+cu128, RTX 5090, bf16) the patch embed's Conv3d dispatches to
``aten::slow_conv_dilated3d``: 99% of an image forward (1.1 s for one 320x240 image, 9 s for 960x720).
With kernel == stride and no padding, each input row is exactly one patch, so the conv is
``x.flatten(1) @ W.flatten(1).T + b`` — the same arithmetic, one GEMM.
"""

import torch
import torch.nn.functional as F


class PatchConv3dAsLinear(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv3d) -> None:
        super().__init__()
        self.conv = conv
        self.kernel = tuple(conv.kernel_size)

    # the patch embed reads ``self.proj.weight.dtype`` before calling it
    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[2:]) != self.kernel:
            return self.conv(x)
        w = self.conv.weight
        y = F.linear(x.reshape(x.shape[0], -1).to(w.dtype), w.reshape(w.shape[0], -1), self.conv.bias)
        return y.view(x.shape[0], -1, 1, 1, 1)


def patch_conv3d(model: torch.nn.Module) -> int:
    swapped = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if (isinstance(child, torch.nn.Conv3d) and tuple(child.kernel_size) == tuple(child.stride)
                    and all(p == 0 for p in child.padding) and child.groups == 1):
                setattr(parent, name, PatchConv3dAsLinear(child))
                swapped += 1
    return swapped
