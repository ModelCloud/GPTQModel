"""Install selected EXL3 tensors using the fork's packed-weight replay contract."""

import torch
from torch import nn

from ..exllamav3.modules.quant.exl3_lib.quantize import reconstruct_exl3_tensors


@torch.no_grad()
def install_projection(block, expert_index, projection, packed, *, device):
    """Replace only one routed linear, reconstructing from serialized payload.

    Deliberately never accepts the search function's higher-precision weight_q.
    This follows GPTQModel's BF16 dense propagation replay of packed EXL3 weights;
    it does not claim bitwise equality with every fused serving GEMM backend.
    The caller retains durable packed payloads separately for final export.
    """
    names = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
    if projection not in names:
        raise ValueError("only routed gate/up/down projections can be replaced")
    experts = block.mlp.experts
    if type(expert_index) is not int or not 0 <= expert_index < len(experts):
        raise ValueError("invalid routed expert index")
    required = {"trellis", "suh", "svh", "mcg"}
    if set(packed) != required:
        raise ValueError("V4.1 recipe requires exactly trellis/suh/svh/MCG tensors")
    old = getattr(experts[expert_index], names[projection])
    trellis, suh, svh, mcg = (packed[key] for key in ("trellis", "suh", "svh", "mcg"))
    if (trellis.dtype != torch.int16 or trellis.ndim != 3 or trellis.shape[-1] not in (48, 64)
            or tuple(trellis.shape[:2]) != (old.in_features // 16, old.out_features // 16)
            or suh.dtype != torch.float16 or tuple(suh.shape) != (old.in_features,)
            or svh.dtype != torch.float16 or tuple(svh.shape) != (old.out_features,)
            or mcg.dtype != torch.int32 or mcg.numel() != 1):
        raise ValueError("packed V4.1 projection geometry/dtype differs from target")
    if not torch.isfinite(suh).all() or not torch.isfinite(svh).all():
        raise ValueError("nonfinite packed scales")
    weight = reconstruct_exl3_tensors(packed, device=device, dtype=torch.bfloat16)
    if not torch.isfinite(weight).all():
        raise ValueError("nonfinite packed reconstruction")
    replacement = nn.Linear(old.in_features, old.out_features, bias=False, device="meta", dtype=torch.bfloat16)
    replacement.weight = nn.Parameter(weight.T.contiguous(), requires_grad=False)
    setattr(experts[expert_index], names[projection], replacement)
    return replacement
