# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Measure LeapQuant decode and compensator-fit cost against a dense FP32 step.

The fixture uses model-shaped inputs: unit-norm keys, queries scaled by
``1/sqrt(K)``, and small values, so the recurrent state stays in the range a
real layer produces. The dense reference applies the same delta-rule
recurrence one token at a time without any state quantization.

    python scripts/benchmark_leapquant_state.py --device cuda --tokens 64
"""

import argparse
import statistics
import time

import torch

from gptqmodel.utils.leapquant import LeapQuantConfig, LeapQuantState


def _unit(shape, device):
    tensor = torch.randn(*shape, device=device)
    return tensor / tensor.norm(dim=-1, keepdim=True)


def _measure(call, repeats, warmup=5):
    for _ in range(warmup):
        call()
    if torch.cuda.is_available():
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(repeats):
            call()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats * 1e3
    begin = time.perf_counter()
    for _ in range(repeats):
        call()
    return (time.perf_counter() - begin) / repeats * 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--bits", type=int, choices=(4, 6, 8), default=8)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    batch, heads = args.batch, args.heads
    key_dim, value_dim, window = args.key_dim, args.value_dim, args.window
    tokens = args.tokens

    query = _unit((batch, tokens, heads, key_dim), device) * (key_dim ** -0.5)
    key = _unit((batch, tokens, heads, key_dim), device)
    value = torch.randn(batch, tokens, heads, value_dim, device=device) * 0.1
    alpha = torch.rand(batch, tokens, heads, 1, device=device) * 0.3 + 0.7
    beta = torch.rand(batch, tokens, heads, device=device) * 0.8 + 0.1

    dense_state = torch.zeros(batch, heads, key_dim, value_dim, device=device, dtype=torch.float32)
    position = 0

    def dense_step():
        nonlocal dense_state, position
        index = position % tokens
        position += 1
        decayed = dense_state * alpha[:, index, :, 0][..., None, None]
        prediction = (decayed * key[:, index][..., :, None]).sum(dim=-2)
        update = beta[:, index][..., None] * (value[:, index] - prediction)
        dense_state = decayed + key[:, index][..., :, None] * update[..., None, :]
        return (dense_state * query[:, index][..., :, None]).sum(dim=-2)

    def build():
        state = LeapQuantState(
            batch, heads, key_dim, value_dim,
            LeapQuantConfig(bits=args.bits, window=window, rank=args.rank),
            gate_dim=1, device=device,
        )
        state.quantize_(torch.randn(batch, heads, key_dim, value_dim, device=device) * 0.05)
        return state

    def cycle(state):
        for index in range(window):
            state.step(
                query[:, index], key[:, index], value[:, index], alpha[:, index], beta[:, index]
            )

    for _ in range(tokens):
        dense_step()

    state = build()
    fit_us = statistics.median(
        _measure(lambda: state._fit_compensator(dense_state), args.repeats) for _ in range(5)
    )
    decode_us = statistics.median(
        _measure(lambda: cycle(state), max(2, args.repeats // 4)) / window for _ in range(5)
    )
    dense_us = statistics.median(_measure(dense_step, args.repeats * 4) for _ in range(5))

    print(f"dense_fp32_step_us   {dense_us:9.1f}")
    print(f"leapquant_decode_us  {decode_us:9.1f}")
    print(f"compensator_fit_us   {fit_us:9.1f}")
    print(f"decode_overhead_x    {decode_us / dense_us:9.2f}")
    print(f"checkpoint_bytes     {state.checkpoint_bytes():9d}")


if __name__ == "__main__":
    main()
