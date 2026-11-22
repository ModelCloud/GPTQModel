# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GGUFTritonKernel must not retain a second copy of the packed GGUF weights (#4748).

The AUTO CUDA/ROCm K-quant kernel decoded the packed codes into a re-laid-out
``qs``/``ql``/``qh`` tensor on the first forward.  Those tensors are the same
bytes as ``qweight`` in a different order, so the model ended up holding about
2.1x its packed weight VRAM.  Only the derived scale/min metadata may persist;
the packed code bytes must be read straight out of ``qweight``.

The check is on real device memory: the retained allocator bytes after one
forward are compared with the packed weight bytes.  The same forward is also
checked against an independent FP64 dequantized reference so a memory fix can
never silently change numerics.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import (
    GGUFTorchLinear,
    _dequantize_gguf_tensor_numpy,
)
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


def _build_modules(bits: str, in_features: int, out_features: int, seed: int = 4748):
    """Pack a normal linear into GGUF and mirror it into the Triton kernel."""
    torch.manual_seed(seed)
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.normal_(mean=0.0, std=0.03)

    torch_module = GGUFTorchLinear(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=False,
    )
    torch_module.pack_original(linear, scales=None, zeros=None)
    torch_module.post_init()

    triton_module = GGUFTritonKernel(
        bits=bits,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    )
    triton_module.load_state_dict(torch_module.state_dict(), strict=True)
    return torch_module, triton_module.to("cuda").eval()


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize(
    ("bits", "qtype"),
    [("q4_k", "Q4_K"), ("q5_k", "Q5_K"), ("q6_k", "Q6_K")],
)
def test_triton_cache_keeps_only_one_packed_weight_copy(bits, qtype):
    in_features, out_features = 4096, 1024
    torch_module, module = _build_modules(bits, in_features, out_features)

    packed_bytes = module.qweight.numel() * module.qweight.element_size()
    assert packed_bytes > 0

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()

    inputs = torch.randn(1, in_features, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        decoded = module(inputs)
    torch.cuda.synchronize()

    retained = torch.cuda.memory_allocated() - baseline
    # The cache may keep derived per-sub-block scale/min metadata, but the
    # packed code bytes themselves must stay in ``qweight``.  A full second
    # copy is ~1.0-1.1x the packed bytes; the metadata-only cache is well
    # below 0.5x even when it widens to FP32.
    assert retained <= 0.6 * packed_bytes, (
        f"{qtype}: one forward retained {retained / 2**20:.2f} MiB of device "
        f"memory for {packed_bytes / 2**20:.2f} MiB of packed weights "
        f"({retained / packed_bytes:.2f}x); the Triton cache duplicated the "
        "packed codes instead of reading them from qweight"
    )

    # No cached tensor may be a full-size re-layout of the packed codes.
    for entry in module._gguf_triton_cache.values():
        for value in entry.values():
            if torch.is_tensor(value):
                assert value.numel() * value.element_size() < packed_bytes, (
                    f"{qtype}: cache holds a full-size weight copy of "
                    f"{value.numel() * value.element_size() / 2**20:.2f} MiB"
                )

    # The strided/contiguous forward must match the independent FP64 oracle
    # for both decode (M=1) and a small prefill (M=16).
    dense = torch.from_numpy(
        _dequantize_gguf_tensor_numpy(torch_module.qweight.detach().cpu().numpy(), qtype)
    ).to(device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        assert torch.equal(module(inputs), decoded), f"{qtype}: decode is not deterministic"
    for rows in (1, 16):
        x = torch.randn(rows, in_features, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            got = module(x)
            again = module(x)
        assert torch.equal(got, again), f"{qtype} rows={rows}: forward is not deterministic"
        truth = x.to(torch.float64) @ dense.t().to(torch.float64)
        rel = (got.to(torch.float64) - truth).abs().max().item() / truth.abs().max().item()
        assert rel <= 2e-3, f"{qtype} rows={rows}: rel error {rel:.3e}"
