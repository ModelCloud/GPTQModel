#!/usr/bin/env python
"""Compare GPTQ.quantize output with and without the fused Triton block kernel."""

import copy
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

import torch
import torch.nn as nn

# Set env before importing gptq to control the Triton block path.
# We will import inside function with env set.

def run_one(group_size, method, use_triton):
    if use_triton:
        os.environ["GPTQMODEL_TRITON_BLOCK"] = "1"
    else:
        os.environ["GPTQMODEL_TRITON_BLOCK"] = "0"
    # Re-import to pick up env-var flag.
    import gptqmodel.quantization.gptq as gptq_mod
    importlib.reload(gptq_mod) if "importlib" in globals() else None

    from gptqmodel.quantization import QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    torch.manual_seed(42)
    device = "cuda:0"
    layer = nn.Linear(4096, 4096, bias=False, dtype=torch.float16, device=device)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=group_size,
        sym=False,
        desc_act=False,
        offload_to_disk=False,
        mse=2.0,
        scale_search=method,
    )
    g = GPTQ(layer, qcfg=copy.deepcopy(qcfg))
    g.quantizer.configure(perchannel=True)
    inp = torch.randn(8, 4096, dtype=torch.float16, device=device)
    g.add_batch(inp, None)
    Q, scale, zero, g_idx, duration, avg_loss, damp, nsamples = g.quantize(blocksize=128)
    return Q, scale, zero, avg_loss


if __name__ == "__main__":
    import importlib

    method_map = {
        "activation": "ScaleSearchConfig.ACTIVATION",
    }
    # Use activation method for speed; hessian/hybrid similar.
    from gptqmodel.quantization import ScaleSearchConfig

    for group_size in [128, 64, 32]:
        print(f"\n=== group_size={group_size} ===")
        Q_ref, s_ref, z_ref, loss_ref = run_one(group_size, ScaleSearchConfig.ACTIVATION, use_triton=False)
        Q_triton, s_triton, z_triton, loss_triton = run_one(group_size, ScaleSearchConfig.ACTIVATION, use_triton=True)
        print("Q max diff:", (Q_ref - Q_triton).abs().max().item())
        print("scale max diff:", (s_ref - s_triton).abs().max().item())
        print("zero max diff:", (z_ref - z_triton).abs().max().item())
        print("loss ref:", loss_ref, "loss triton:", loss_triton, "diff:", abs(loss_ref - loss_triton) if isinstance(loss_ref, float) else None)
