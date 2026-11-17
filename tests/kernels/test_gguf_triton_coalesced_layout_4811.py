# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""GGUFTritonKernel must read the packed codes coalesced with one resident copy (#4811).

#4748 stopped materializing the permuted ``qs``/``ql``/``qh`` code arrays and
read them as strided views of the canonical ``qweight``.  That keeps one copy,
but the kernels load ``[byte, offs_n]`` tiles with ``offs_n`` as the fast axis,
so the output-feature stride became a whole packed row and every warp load was
uncoalesced (Q6_K prefill +66%).

The fix transposes the packed storage once per module and exposes ``qweight`` as
a transposed view of it.  The code views then have a unit stride on the
output-feature axis again while only one copy of the bytes stays resident.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)

QTYPES = [("q4_k", "Q4_K"), ("q5_k", "Q5_K"), ("q6_k", "Q6_K")]


def _build_modules(bits: str, in_features: int, out_features: int, seed: int = 4811):
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


def _code_views(module):
    """Every packed-code entry the fused kernels load, per cache entry."""
    for entry in module._gguf_triton_cache.values():
        for name in ("qs", "ql", "qh"):
            value = entry.get(name)
            if torch.is_tensor(value):
                yield name, value


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize(("bits", "qtype"), QTYPES)
def test_first_forward_makes_the_code_views_row_contiguous(bits, qtype):
    torch_module, module = _build_modules(bits, 4096, 1024)

    # Before the first forward the packed bytes are still in the canonical
    # checkpoint order, where the code view is uncoalesced: exactly the #4811
    # regression this module must remove.  The parent build keeps that stride.
    canonical_blocks = GGUFTorchLinear._reshape_blocks(module)[0]
    canonical_codes = canonical_blocks[..., 16:].permute(1, 2, 0)
    assert canonical_codes.stride(2) != 1, "canonical layout is unexpectedly coalesced"
    assert not list(_code_views(module))  # no cache yet

    inputs = torch.randn(16, 4096, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        expected = module(inputs)
    torch.cuda.synchronize()

    views = list(_code_views(module))
    assert views, f"{qtype}: no packed-code cache entries"
    for name, value in views:
        assert value.stride(-1) == 1, (
            f"{qtype}: cache code view {name} has stride {value.stride(-1)} on "
            "the output-feature axis; the fused kernels would load it uncoalesced"
        )

    # The permuted storage is still the canonical logical tensor: the block view
    # of the live qweight matches the checkpoint bytes exactly.
    live_blocks, _, _ = GGUFTritonKernel._reshape_blocks(module)
    canonical = torch_module.qweight.to("cuda").contiguous().view(
        torch_module.qweight.shape[0], -1, module.gguf_type_size
    )
    assert torch.equal(live_blocks.contiguous(), canonical)
    assert live_blocks[..., 16:].permute(1, 2, 0).stride(2) == 1

    # ... and the forward is unchanged by the relayout.
    with torch.inference_mode():
        assert torch.equal(module(inputs), expected)


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize(("bits", "qtype"), QTYPES)
def test_relayout_keeps_exactly_one_resident_copy(bits, qtype):
    torch_module, module = _build_modules(bits, 4096, 1024)

    packed_bytes = module.qweight.numel() * module.qweight.element_size()
    assert packed_bytes > 0

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    inputs = torch.randn(1, 4096, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        module(inputs)
    torch.cuda.synchronize()

    retained = torch.cuda.memory_allocated() - baseline
    # Only the derived scale/min metadata may persist; the packed bytes stay in
    # the single transposed ``qweight`` storage.
    assert retained <= 0.6 * packed_bytes, (
        f"{qtype}: one forward retained {retained / 2**20:.2f} MiB for "
        f"{packed_bytes / 2**20:.2f} MiB of packed weights "
        f"({retained / packed_bytes:.2f}x)"
    )
    assert module.qweight.numel() * module.qweight.element_size() == packed_bytes
    for entry in module._gguf_triton_cache.values():
        for value in entry.values():
            if torch.is_tensor(value):
                assert value.numel() * value.element_size() < packed_bytes, (
                    f"{qtype}: cache holds a full-size weight copy"
                )


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize(("bits", "qtype"), QTYPES)
def test_state_dict_round_trip_restores_canonical_bytes(bits, qtype):
    torch_module, module = _build_modules(bits, 4096, 1024)
    canonical = torch_module.qweight.clone()
    inputs = torch.randn(8, 4096, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        expected = module(inputs)

    state = module.state_dict()
    # Saved checkpoints must stay in the canonical GGUF order.
    assert state["qweight"].is_contiguous()
    assert torch.equal(state["qweight"].cpu(), canonical)

    _, reloaded = _build_modules(bits, 4096, 1024)
    reloaded.load_state_dict(state, strict=True)
    # Loading must not leave canonical values inside transposed storage.
    assert torch.equal(reloaded.qweight.contiguous().cpu(), canonical)
    with torch.inference_mode():
        assert torch.equal(reloaded(inputs), expected)
    # A forward re-permutes the storage and still agrees.
    assert reloaded.qweight.stride(0) == 1
    assert torch.equal(reloaded.qweight.contiguous().cpu(), canonical)
    with torch.inference_mode():
        assert torch.equal(reloaded(inputs), expected)


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize(("bits", "qtype"), QTYPES)
def test_device_move_keeps_the_permuted_layout_consistent(bits, qtype):
    torch_module, module = _build_modules(bits, 4096, 1024)
    canonical = torch_module.qweight.contiguous()
    inputs = torch.randn(4, 4096, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        expected = module(inputs)

    module = module.to("cpu")
    blocks, _, _ = GGUFTritonKernel._reshape_blocks(module)
    assert torch.equal(
        blocks.contiguous(), canonical.view(canonical.shape[0], -1, module.gguf_type_size)
    )

    module = module.to("cuda").eval()
    with torch.inference_mode():
        assert torch.equal(module(inputs), expected)
