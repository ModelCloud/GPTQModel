# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import gptqmodel.utils.torch as torch_utils
from gptqmodel.nn_modules.qlinear import PackableQuantLinear
from gptqmodel.nn_modules.qlinear.lookahead import configure_default_lookahead
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear


def _mock_gptq_linear(bits: int, group_size: int, in_features: int, out_features: int) -> tuple[nn.Linear, torch.Tensor, torch.Tensor, torch.Tensor]:
    maxq = (1 << (bits - 1)) - 1
    weight = torch.randn((in_features, out_features), dtype=torch.float32)

    if group_size != -1:
        reshaped = weight.view(in_features // group_size, group_size, out_features)
        w_g = reshaped.permute(1, 0, 2).reshape(group_size, -1)
    else:
        w_g = weight

    scales = torch.maximum(
        w_g.abs().max(dim=0, keepdim=True).values,
        torch.full((1, w_g.shape[1]), 1e-6, device=w_g.device),
    )
    scales = scales / maxq

    q = torch.round(w_g / scales).clamp_(-maxq, maxq)
    ref = (q * scales).to(dtype=torch.float16)

    if group_size != -1:
        ref = ref.reshape(group_size, -1, out_features)
        ref = ref.permute(1, 0, 2).reshape(in_features, out_features)

        q = q.reshape(group_size, -1, out_features)
        q = q.permute(1, 0, 2).reshape(in_features, out_features)

    linear = nn.Linear(in_features, out_features, bias=False)
    linear.weight.data = ref.t().contiguous()

    scales = scales.reshape(-1, out_features).contiguous()
    zeros = torch.zeros_like(scales, dtype=torch.int32)
    g_idx = torch.arange(in_features, dtype=torch.int32) // (
        group_size if group_size != -1 else in_features
    )

    return linear, scales, zeros, g_idx


@pytest.mark.cuda
@pytest.mark.parametrize("group_size", [256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_torch_triton_large_group_sizes(group_size: int, dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bfloat16 not supported on this device")

    torch.cuda.set_device(0)

    bits = 4
    in_features = 4096
    out_features = 4096

    torch.manual_seed(0)

    linear, scales, zeros, g_idx = _mock_gptq_linear(bits, group_size, in_features, out_features)

    torch_module = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
    )
    torch_module.pack_block(linear, scales.T, zeros.T, g_idx=g_idx)
    torch_module.post_init()

    try:
        triton_module = TritonV2Linear(
            bits=bits,
            group_size=group_size,
            desc_act=False,
            sym=True,
            in_features=in_features,
            out_features=out_features,
            pack_dtype=torch.int32,
            bias=False,
        )
    except ValueError as err:
        pytest.skip(f"Triton backend unavailable: {err}")

    triton_module.pack_block(linear, scales.T, zeros.T, g_idx=g_idx)
    triton_module.post_init()

    device = torch.device("cuda:0")
    torch_module = torch_module.to(device=device, dtype=dtype).eval()
    triton_module = triton_module.to(device=device, dtype=dtype).eval()

    batch = 8
    x = torch.randn((batch, in_features), device=device, dtype=dtype)

    with torch.inference_mode():
        torch_out = torch_module(x)
        triton_out = triton_module(x)

    torch_out = torch_out.to(torch.float32)
    triton_out = triton_out.to(torch.float32)

    assert torch.allclose(triton_out, torch_out, rtol=1e-2, atol=1e-2)
    assert torch_out.abs().max() > 0

######### test_torch_weight_cache.py #########


def _make_module(device: torch.device):
    module = TorchLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=True,
        in_features=64,
        out_features=64,
        bias=True,
        pack_dtype=torch.int32,
        adapter=None,
        register_buffers=True,
    ).to(device)

    with torch.no_grad():
        module.qweight.zero_()
        module.qzeros.zero_()
        module.scales.fill_(1.0)
        module.bias.uniform_(-0.1, 0.1)

    module.qzero_format(format=2)
    module.post_init()
    module.eval()
    return module


def test_gptq_post_init_creates_wf_unpack_buffers():
    module = TorchLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=64,
        bias=False,
        pack_dtype=torch.int32,
        adapter=None,
        register_buffers=True,
    )
    module.optimize = lambda *args, **kwargs: None
    module.post_init()

    assert module.enable_wf_unsqueeze is True
    assert module.wf_unsqueeze_zero is not None
    assert module.wf_unsqueeze_neg_one is not None


def test_torch_quant_linear_exposes_weight_metadata():
    module = TorchLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=64,
        out_features=96,
        bias=False,
        pack_dtype=torch.int32,
        adapter=None,
        register_buffers=True,
    )

    weight = module.weight

    assert weight.device == module.qweight.device
    assert weight.dtype == module.scales.dtype
    assert weight.shape == torch.Size((module.out_features, module.in_features))
    assert weight.size(0) == module.out_features
    assert weight.size(1) == module.in_features
    assert weight.T.shape == torch.Size((module.in_features, module.out_features))
    assert ("cuda" in weight.device.type) == weight.is_cuda


def test_cached_forward_matches_baseline():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    module = _make_module(device)

    x = torch.randn(8, module.in_features, device=device, dtype=torch.float16)

    module.enable_weight_cache(False)
    ref = module(x)

    module.enable_weight_cache(True)
    module.clear_weight_cache()
    cached = module(x)

    torch.testing.assert_close(ref, cached)
    assert x.dtype in module._cached_weights
    assert module._cached_weights[x.dtype].device.type == device.type


def test_torch_empty_cache_syncs_before_releasing_allocator(monkeypatch):
    calls = []
    device = torch.device("cpu")

    monkeypatch.setattr(torch_utils, "timed_gc_collect", lambda: calls.append("gc") or 0)
    monkeypatch.setattr(torch_utils, "torch_sync", lambda device=None: calls.append(("sync", device)))
    monkeypatch.setattr(torch_utils, "empty_cache_for_device", lambda device: calls.append(("empty", device)) or True)

    assert torch_utils.torch_empty_cache(device=device, gc=True, sync=True) is True
    assert calls == ["gc", ("sync", device), ("empty", device)]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least 2 CUDA devices")
def test_cross_device_forward_moves_weights_to_input_device():
    module = _make_module(torch.device("cuda:1"))
    module.enable_weight_cache(True)
    module.clear_weight_cache()

    x = torch.randn(8, module.in_features, device=torch.device("cuda:0"), dtype=torch.float16)
    out = module(x)

    assert out.device == x.device
    assert x.dtype in module._cached_weights
    assert module._cached_weights[x.dtype].device == x.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required for lookahead prefetch test")
def test_lookahead_prefetch_single_step():
    device = torch.device("cuda")
    producer = _make_module(device)
    consumer = _make_module(device)

    producer.enable_lookahead(True).set_lookahead_next(consumer)
    consumer.enable_lookahead(True)

    x = torch.randn(4, producer.in_features, device=device, dtype=torch.float16)

    producer(x)
    assert torch.float16 in consumer._prefetched_weights

    consumer(x)
    assert torch.float16 not in consumer._prefetched_weights


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda required for g_idx offload test")
def test_cached_dequant_offloads_g_idx_to_cpu_on_cuda():
    module = _make_module(torch.device("cuda"))
    module._triton_dequant_enabled = False
    module._stream_reset_cache()

    assert module.g_idx.device.type == "cuda"
    assert module._g_idx_long_cache is None

    with torch.inference_mode():
        weights = module.dequantize_weight(num_itr=1)

    assert weights.device.type == "cuda"
    assert module._g_idx_long_cache is not None
    assert module._g_idx_long_cache.device.type == "cuda"
    assert module.g_idx.device.type == "cpu"

    # Cached path should remain usable after offloading original g_idx.
    with torch.inference_mode():
        weights_after = module.dequantize_weight(num_itr=1)
    assert weights_after.device.type == "cuda"


def test_configure_default_lookahead_chain():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    class DummyAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _make_module(device)
            self.k_proj = _make_module(device)
            self.v_proj = _make_module(device)
            self.o_proj = _make_module(device)

    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _make_module(device)
            self.up_proj = _make_module(device)
            self.down_proj = _make_module(device)

    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DummyAttn()
            self.mlp = DummyMLP()

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DummyLayer()])

    model = DummyModel()
    for module in model.modules():
        if isinstance(module, TorchLinear):
            module.enable_lookahead(True)

    configure_default_lookahead(model)

    layer = model.layers[0]
    q_proj = layer.self_attn.q_proj
    k_proj = layer.self_attn.k_proj
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    gate_proj = layer.mlp.gate_proj
    up_proj = layer.mlp.up_proj
    down_proj = layer.mlp.down_proj

    assert q_proj._lookahead_next == (gate_proj, up_proj, down_proj)
    assert q_proj._lookahead_enabled

    for module in (k_proj, v_proj, o_proj):
        assert module._lookahead_next is None
        assert not module._lookahead_enabled

    for module in (gate_proj, up_proj, down_proj):
        assert module._lookahead_next is None
        assert module._lookahead_enabled


def test_cpu_dequant_parity_and_g_idx_cache_allocation():
    bits = 4
    group_size = 128
    in_features = 1024
    out_features = 1024

    torch.manual_seed(0)
    linear, scales, zeros, g_idx = _mock_gptq_linear(bits, group_size, in_features, out_features)

    module = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
    )
    # Keep this unit deterministic by bypassing torch.compile wrappers.
    module.optimize = lambda *args, **kwargs: None
    module.pack_block(linear, scales.T, zeros.T, g_idx=g_idx)
    module.post_init()
    module.eval()
    module = module.to(device=torch.device("cpu"))

    # Cache should be lazy and absent before first fast-path dequant call.
    assert module._g_idx_long_cache is None
    assert module._g_idx_long_cache_state is None

    with torch.inference_mode():
        baseline = PackableQuantLinear.dequantize_weight(module, num_itr=1)
        current = module.dequantize_weight(num_itr=1)

    torch.testing.assert_close(current, baseline, rtol=0, atol=0)

    # First call materializes persistent int64 g_idx cache.
    assert module._g_idx_long_cache is not None
    assert module._g_idx_long_cache.dtype == torch.int64
    assert module._g_idx_long_cache.device.type == "cpu"

    expected_cache_bytes = module.g_idx.numel() * torch.tensor(0, dtype=torch.int64).element_size()
    actual_cache_bytes = module._g_idx_long_cache.numel() * module._g_idx_long_cache.element_size()
    assert actual_cache_bytes == expected_cache_bytes

    # Cache should be reused across subsequent dequant calls.
    cache_ptr = module._g_idx_long_cache.data_ptr()
    with torch.inference_mode():
        _ = module.dequantize_weight(num_itr=1)
    assert module._g_idx_long_cache.data_ptr() == cache_ptr

    # Explicit reset should drop cache and allow re-allocation on next call.
    module._stream_reset_cache()
    assert module._g_idx_long_cache is None
    assert module._g_idx_long_cache_state is None
    with torch.inference_mode():
        _ = module.dequantize_weight(num_itr=1)
    assert module._g_idx_long_cache is not None


def test_cpu_cached_dequant_num_itr_matches_packable():
    bits = 4
    group_size = 128
    in_features = 1024
    out_features = 1024
    num_itr = 4

    torch.manual_seed(0)
    linear, scales, zeros, g_idx = _mock_gptq_linear(bits, group_size, in_features, out_features)

    module = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
    )
    module.optimize = lambda *args, **kwargs: None
    module.pack_block(linear, scales.T, zeros.T, g_idx=g_idx)
    module.post_init()
    module.eval()
    module = module.to(device=torch.device("cpu"))

    with torch.inference_mode():
        baseline = PackableQuantLinear.dequantize_weight(module, num_itr=num_itr)
        current = module.dequantize_weight(num_itr=num_itr)

    assert baseline.shape == (in_features // num_itr, out_features)
    assert current.shape == baseline.shape
    torch.testing.assert_close(current, baseline, rtol=0, atol=0)
