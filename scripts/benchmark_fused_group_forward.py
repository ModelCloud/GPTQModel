# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Micro-benchmark for the FusedGroupForward same-input linear optimization.

This script isolates the fused vs. separate GEMM cost for a Llama-like QKV
projection so speedups are visible without the rest of the GPTQ solve pipeline.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from gptqmodel.nn_modules.fused_group_forward import FusedGroupForward
from gptqmodel.nn_modules.hooked_linear import HookedLinear


def _make(in_f, out_f, device, dtype):
    m = HookedLinear(in_f, out_f)
    m.weight = nn.Parameter(torch.randn(out_f, in_f, dtype=dtype, device=device))
    m.bias = nn.Parameter(torch.randn(out_f, dtype=dtype, device=device))
    return m


def _run(modules, x, warmup=20, iters=200):
    for _ in range(warmup):
        for m in modules:
            _ = m(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for m in modules:
            _ = m(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    hidden = 2048
    q_out = 2048
    kv_out = 512
    seq_lens = [128, 256, 512, 1024]
    batch = 1

    print("device:", device, "dtype:", dtype)
    print("batch:", batch, "hidden:", hidden, "q_out:", q_out, "kv_out:", kv_out)
    print(f"{'seq_len':>8} {'fused_ms':>10} {'separate_ms':>12} {'speedup':>9}")

    for seq_len in seq_lens:
        q = _make(hidden, q_out, device, dtype)
        k = _make(hidden, kv_out, device, dtype)
        v = _make(hidden, kv_out, device, dtype)
        fg = FusedGroupForward(None, [q, k, v])
        q._fused_group_forward = fg
        k._fused_group_forward = fg
        v._fused_group_forward = fg

        x = torch.randn(batch, seq_len, hidden, dtype=dtype, device=device)

        fused_ms = _run([q, k, v], x)

        q2 = _make(hidden, q_out, device, dtype)
        k2 = _make(hidden, kv_out, device, dtype)
        v2 = _make(hidden, kv_out, device, dtype)
        sep_ms = _run([q2, k2, v2], x)

        print(f"{seq_len:>8} {fused_ms:>10.4f} {sep_ms:>12.4f} {sep_ms / fused_ms:>9.2f}x")


if __name__ == "__main__":
    main()
