#!/usr/bin/env python
"""Correctness test for the fused GPTQ block Triton kernel."""

import copy
import os

import torch
import torch.nn as nn

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

from gptqmodel.quantization import QuantizeConfig, ScaleSearchConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization._gptq_block_triton import gptq_block_triton


def python_block(W1, Q1, Err1, Hinv1, scale, zero, maxq, group_size, debug_i=None):
    """Reproduce the eager per-column loop for one block."""
    rows, count = W1.shape
    debug = {}
    for i in range(count):
        w = W1[:, i]
        d = Hinv1[i, i]
        s = scale[:, i // group_size]
        z = zero[:, i // group_size]
        if i == debug_i:
            debug['w'] = w.clone()
            debug['s'] = s.clone()
            debug['z'] = z.clone()
            debug['ratio'] = (w / s).clone()
        q = torch.clamp(torch.round(w / s) + z, 0, maxq)
        q = s * (q - z)
        Q1[:, i] = q
        err1 = (w - q) / d
        Err1[:, i] = err1
        W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
    return W1, Q1, Err1, debug


def main():
    torch.manual_seed(42)
    device = "cuda:0"
    rows = 2
    count = 128
    group_size = 128
    maxq = 15

    W1 = torch.randn(rows, count, dtype=torch.float32, device=device)
    Q1 = torch.empty_like(W1)
    Err1 = torch.empty_like(W1)
    Hinv1 = torch.randn(count, count, dtype=torch.float32, device=device)
    Hinv1 = Hinv1 @ Hinv1.T + torch.eye(count, device=device) * 1e-3
    Hinv1 = torch.linalg.inv(Hinv1)

    # Fake scales and zeros, one per row per group (Quantizer outputs float32).
    scale = torch.rand(rows, count // group_size, dtype=torch.float32, device=device) * 0.05
    zero = torch.randint(0, 16, (rows, count // group_size), dtype=torch.float32, device=device)

    W1_ref = W1.clone()
    Q1_ref = torch.empty_like(Q1)
    Err1_ref = torch.empty_like(Err1)
    _, _, _, debug = python_block(W1_ref, Q1_ref, Err1_ref, Hinv1, scale, zero, maxq, group_size, debug_i=74)

    W1_triton = W1.clone()
    Q1_triton = torch.empty_like(Q1)
    Err1_triton = torch.empty_like(Err1)
    gptq_block_triton(W1_triton, Q1_triton, Err1_triton, Hinv1, scale, zero, maxq, group_size)

    diff_q = (Q1_triton - Q1_ref).abs()
    print("Q diff:", diff_q.max().item())
    # locate first mismatch column
    mismatch = (diff_q > 1e-5).nonzero()
    if len(mismatch):
        first = mismatch[0].tolist()
        print('first mismatch', first, 'triton', Q1_triton[first[0], first[1]].item(), 'ref', Q1_ref[first[0], first[1]].item())
    idx = (diff_q == diff_q.max()).nonzero()[0].tolist()
    print('max mismatch at', idx, 'triton', Q1_triton[idx[0], idx[1]].item(), 'ref', Q1_ref[idx[0], idx[1]].item())
    print('scale', scale[idx[0], idx[1]//group_size].item(), 'zero', zero[idx[0], idx[1]//group_size].item())
    if len(mismatch):
        first_row = first[0]
        if 'w' in debug:
            print('debug col 74 for row', first_row, 'ratio', debug['ratio'][first_row].item(), 'w', debug['w'][first_row].item(), 's', debug['s'][first_row].item())
    print("Err diff:", (Err1_triton - Err1_ref).abs().max().item())
    print("W1 diff:", (W1_triton - W1_ref).abs().max().item())
    diff_w1 = (W1_triton - W1_ref).abs()
    print('W1 diff before col74', diff_w1[:, :74].max().item())

    torch.testing.assert_close(Q1_triton, Q1_ref, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(Err1_triton, Err1_ref, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(W1_triton, W1_ref, atol=1e-4, rtol=1e-5)
    print("single-block correctness OK")

    # Compare inside a real GPTQ quantize() for one layer (group_size=128).
    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        sym=False,
        desc_act=False,
        offload_to_disk=False,
        mse=2.0,
        scale_search=ScaleSearchConfig.ACTIVATION,
    )
    layer = nn.Linear(4096, 4096, bias=False, dtype=torch.float16, device=device)
    inp = torch.randn(4, 4096, dtype=torch.float16, device=device)

    g_ref = GPTQ(layer, qcfg=copy.deepcopy(qcfg))
    g_ref.quantizer.configure(perchannel=True)
    g_ref.add_batch(inp, None)
    Q_ref, _, _, _, _, _, _, _ = g_ref.quantize(blocksize=128)

    print("Max Q ref abs:", Q_ref.abs().max().item())


if __name__ == "__main__":
    main()
