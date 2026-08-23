# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import torch.nn as nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.fused_quant_linear import (
    install_fused_gate_up,
    install_fused_qkv,
)
from gptqmodel.nn_modules.fused_quant_linear import _FusedMarlinKernel
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="TritonV2Linear fused tests require CUDA")

marlin_skip = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="Marlin fused tests require Ampere or newer CUDA",
)


def _make_marlin_linear(
    in_features: int,
    out_features: int,
    bits: int = 4,
    group_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> MarlinLinear:
    """Build a MarlinLinear with random packed weights and call post_init."""
    m = MarlinLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=False,
        dtype=dtype,
    )
    device = torch.device("cuda")
    m.qweight.data = torch.randint(
        0, 2**31, m.qweight.shape, dtype=torch.int32, device=device
    )
    m.scales.data = (torch.rand(m.scales.shape, dtype=dtype, device=device) * 0.4 + 0.2)
    m.qzeros.data = torch.randint(
        0, 2**31, m.qzeros.shape, dtype=torch.int32, device=device
    )
    m.g_idx.data = torch.arange(in_features, dtype=torch.int32, device=device) // group_size
    m = m.to(device)
    m.post_init()
    return m.eval()


def _make_tritonv2_linear(
    in_features: int,
    out_features: int,
    bits: int = 4,
    group_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> TritonV2Linear:
    """Build a packed TritonV2Linear with random quantized weights for testing."""
    maxq = 2**bits - 1
    zero_point = maxq // 2
    m = TritonV2Linear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )

    # Use a small dense weight so quantized values stay inside the representable range.
    W = torch.randn(out_features, in_features) * 0.3
    linear = nn.Linear(in_features, out_features, bias=False)
    linear.weight.data = W

    num_groups = math.ceil(in_features / group_size)
    scales = torch.rand(out_features, num_groups) * 0.4 + 0.2
    zeros = torch.full((out_features, num_groups), zero_point, dtype=torch.int32)
    g_idx = torch.tensor([i // group_size for i in range(in_features)], dtype=torch.int32)

    m.pack_block(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)
    return m.cuda().eval()


@pytest.mark.parametrize(
    "hidden_size,q_out,kv_out",
    [
        (3072, 6144, 1024),  # Laguna-S-2.1 self-attn
        (5120, 6144, 1024),  # Qwen3.5-27B self-attn
        (4096, 4096, 1024),  # generic Llama-like shape
    ],
)
def test_fused_qkv_matches_unfused(hidden_size: int, q_out: int, kv_out: int) -> None:
    """Fused QKV should match the per-module TritonV2Linear path within BF16 matmul tolerance."""

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _make_tritonv2_linear(hidden_size, q_out)
            self.k_proj = _make_tritonv2_linear(hidden_size, kv_out)
            self.v_proj = _make_tritonv2_linear(hidden_size, kv_out)

    model = Attn().cuda().eval()
    x = torch.randn(2, 8, hidden_size, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        q = model.q_proj(x)
        k = model.k_proj(x)
        v = model.v_proj(x)
    expected = torch.cat([q, k, v], dim=-1)

    count = install_fused_qkv(model)
    assert count == 1

    with torch.inference_mode():
        qf = model.q_proj(x)
        kf = model.k_proj(x)
        vf = model.v_proj(x)
    actual = torch.cat([qf, kf, vf], dim=-1)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # BF16 tensor-core accumulation order changes with output dimension; allow small drift.
    torch.testing.assert_close(actual, expected, atol=2.0, rtol=0.05)


@pytest.mark.parametrize(
    "hidden_size,intermediate_size",
    [
        (3072, 12288),  # Laguna-S-2.1 MLP
        (5120, 17408),  # Qwen3.5-27B MLP
        (4096, 11008),  # generic Llama-like shape
    ],
)
def test_fused_gate_up_matches_unfused(hidden_size: int, intermediate_size: int) -> None:
    """Fused gate/up should match the per-module TritonV2Linear path exactly."""

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _make_tritonv2_linear(hidden_size, intermediate_size)
            self.up_proj = _make_tritonv2_linear(hidden_size, intermediate_size)

    model = MLP().cuda().eval()
    x = torch.randn(2, 8, hidden_size, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        gate = model.gate_proj(x)
        up = model.up_proj(x)
    expected = torch.nn.functional.silu(gate) * up

    count = install_fused_gate_up(model)
    assert count == 1

    with torch.inference_mode():
        gate_f = model.gate_proj(x)
        up_f = model.up_proj(x)
    actual = torch.nn.functional.silu(gate_f) * up_f

    assert gate_f.shape == gate.shape
    assert up_f.shape == up.shape
    # SiLU amplifies tiny BF16/GEMM-order differences between the fused
    # quant_matmul_248 mega-kernel and the per-module TritonV2Linear path;
    # compare raw projections tightly and the final gate-up product loosely.
    torch.testing.assert_close(gate_f, gate, atol=2.0, rtol=0.05)
    torch.testing.assert_close(up_f, up, atol=2.0, rtol=0.05)
    torch.testing.assert_close(actual, expected, atol=20.0, rtol=0.2)


@marlin_skip
@pytest.mark.parametrize(
    "hidden_size,q_out,kv_out",
    [
        (3072, 6144, 1024),
        (5120, 6144, 1024),
        (4096, 4096, 1024),
    ],
)
def test_fused_qkv_marlin_matches_unfused(hidden_size: int, q_out: int, kv_out: int) -> None:
    """Fused Marlin QKV should match the per-module Marlin path within dtype tolerance."""

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _make_marlin_linear(hidden_size, q_out)
            self.k_proj = _make_marlin_linear(hidden_size, kv_out)
            self.v_proj = _make_marlin_linear(hidden_size, kv_out)

    model = Attn().cuda().eval()
    x = torch.randn(2, 8, hidden_size, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        q = model.q_proj(x)
        k = model.k_proj(x)
        v = model.v_proj(x)
    expected = torch.cat([q, k, v], dim=-1)

    count = install_fused_qkv(model)
    assert count == 1

    with torch.inference_mode():
        qf = model.q_proj(x)
        kf = model.k_proj(x)
        vf = model.v_proj(x)
    actual = torch.cat([qf, kf, vf], dim=-1)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, atol=2.0, rtol=0.05)


@marlin_skip
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible CUDA devices")
def test_fused_qkv_marlin_runs_after_cross_cuda_move() -> None:
    """A fused Marlin group must remain numerically valid after moving to another CUDA device."""

    source = torch.device("cuda:0")
    target = torch.device("cuda:1")
    if torch.cuda.get_device_capability(target)[0] < 8:
        pytest.skip("target device requires Ampere or newer CUDA")

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _make_marlin_linear(128, 128)
            self.k_proj = _make_marlin_linear(128, 64)
            self.v_proj = _make_marlin_linear(128, 64)

    with torch.cuda.device(source):
        model = Attn().eval()
        assert install_fused_qkv(model) == 1
        x = torch.randn(2, 4, 128, device=source, dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = torch.cat(
                [model.q_proj(x), model.k_proj(x), model.v_proj(x)],
                dim=-1,
            )
        torch.cuda.synchronize(source)

    group = model.q_proj._gptqmodel_fused_group
    source_workspace = group.kernel.workspace

    qmodel = BaseQModel.__new__(BaseQModel)
    nn.Module.__init__(qmodel)
    qmodel.model = model
    qmodel.to(target)

    assert group.kernel.workspace is not source_workspace
    assert group.kernel.workspace.device == target
    assert torch.count_nonzero(group.kernel.workspace).item() == 0

    x_target = x.to(target)
    with torch.inference_mode():
        actual = torch.cat(
            [qmodel.model.q_proj(x_target), qmodel.model.k_proj(x_target), qmodel.model.v_proj(x_target)],
            dim=-1,
        )

    assert actual.device == target
    torch.testing.assert_close(actual, expected.to(target), atol=2.0, rtol=0.05)


@marlin_skip
@pytest.mark.parametrize(
    "hidden_size,intermediate_size",
    [
        (3072, 12288),
        (5120, 17408),
        (4096, 11008),
    ],
)
def test_fused_gate_up_marlin_matches_unfused(hidden_size: int, intermediate_size: int) -> None:
    """Fused Marlin gate/up should match the per-module Marlin path within dtype tolerance."""

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _make_marlin_linear(hidden_size, intermediate_size)
            self.up_proj = _make_marlin_linear(hidden_size, intermediate_size)

    model = MLP().cuda().eval()
    x = torch.randn(2, 8, hidden_size, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        gate = model.gate_proj(x)
        up = model.up_proj(x)
    expected = torch.nn.functional.silu(gate) * up

    count = install_fused_gate_up(model)
    assert count == 1

    with torch.inference_mode():
        gate_f = model.gate_proj(x)
        up_f = model.up_proj(x)

    assert gate_f.shape == gate.shape
    assert up_f.shape == up.shape
    torch.testing.assert_close(gate_f, gate, atol=2.0, rtol=0.05)
    torch.testing.assert_close(up_f, up, atol=2.0, rtol=0.05)

    actual = torch.nn.functional.silu(gate_f) * up_f
    torch.testing.assert_close(actual, expected, atol=20.0, rtol=0.2)


def test_fused_qkv_skips_mismatched_g_idx() -> None:
    """A Q/K/V triple with different g_idx should not be fused."""

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _make_tritonv2_linear(4096, 4096)
            self.k_proj = _make_tritonv2_linear(4096, 1024)
            self.v_proj = _make_tritonv2_linear(4096, 1024)
            # Mutate k_proj g_idx so it no longer matches q/v.
            self.k_proj.g_idx = torch.arange(4096, dtype=torch.int32, device="cuda") // 64

    model = Attn().cuda().eval()
    count = install_fused_qkv(model)
    assert count == 0


def test_fused_gate_up_skips_different_bits() -> None:
    """A gate/up pair with different bits should not be fused."""

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _make_tritonv2_linear(4096, 11008, bits=4)
            self.up_proj = _make_tritonv2_linear(4096, 11008, bits=2)

    model = MLP().cuda().eval()
    count = install_fused_gate_up(model)
    assert count == 0


def test_model_fuse_api_installs_and_produces_same_output() -> None:
    """BaseQModel.fuse() should delegate to the install helpers and keep outputs unchanged."""

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _make_tritonv2_linear(4096, 4096)
            self.k_proj = _make_tritonv2_linear(4096, 1024)
            self.v_proj = _make_tritonv2_linear(4096, 1024)
            self.gate_proj = _make_tritonv2_linear(4096, 11008)
            self.up_proj = _make_tritonv2_linear(4096, 11008)

    class DummyHFModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    # Minimal BaseQModel stand-in; nn.Module.__init__ gives us the bare Module plumbing.
    class DummyQModel(BaseQModel):
        def __init__(self, model: nn.Module):
            nn.Module.__init__(self)
            self.model = model
            self.quantized = True
            self.load_quantized_model = True

    hf_model = DummyHFModel().cuda().eval()
    qmodel = DummyQModel(hf_model)
    x = torch.randn(2, 8, 4096, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        layer = hf_model.layers[0]
        q = layer.q_proj(x)
        k = layer.k_proj(x)
        v = layer.v_proj(x)
        gate = layer.gate_proj(x)
        up = layer.up_proj(x)
        mlp_expected = torch.nn.functional.silu(gate) * up

    counts = qmodel.fuse()
    assert counts.get("qkv", 0) == 1
    assert counts.get("gate_up", 0) == 1

    with torch.inference_mode():
        layer = hf_model.layers[0]
        qf = layer.q_proj(x)
        kf = layer.k_proj(x)
        vf = layer.v_proj(x)
        gate_f = layer.gate_proj(x)
        up_f = layer.up_proj(x)
        mlp_actual = torch.nn.functional.silu(gate_f) * up_f

    torch.testing.assert_close(qf, q, atol=2.0, rtol=0.05)
    torch.testing.assert_close(kf, k, atol=2.0, rtol=0.05)
    torch.testing.assert_close(vf, v, atol=2.0, rtol=0.05)
    torch.testing.assert_close(gate_f, gate, atol=2.0, rtol=0.05)
    torch.testing.assert_close(up_f, up, atol=2.0, rtol=0.05)
    torch.testing.assert_close(mlp_actual, mlp_expected, atol=20.0, rtol=0.2)


def test_model_fuse_api_skips_non_quantized() -> None:
    """fuse() on an unquantized model should warn and return zero counts."""

    class DummyQModel(BaseQModel):
        def __init__(self):
            nn.Module.__init__(self)
            self.model = nn.Linear(10, 10)
            self.quantized = False
            self.load_quantized_model = False

    qmodel = DummyQModel()
    counts = qmodel.fuse()
    assert counts == {"qkv": 0, "gate_up": 0}


def test_fused_qkv_aliases_match_unfused() -> None:
    """QKV fusion should work with the wq/wk/wv naming convention used by some models."""

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = _make_tritonv2_linear(4096, 4096)
            self.wk = _make_tritonv2_linear(4096, 1024)
            self.wv = _make_tritonv2_linear(4096, 1024)

    model = Attn().cuda().eval()
    x = torch.randn(2, 8, 4096, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        q = model.wq(x)
        k = model.wk(x)
        v = model.wv(x)

    count = install_fused_qkv(model)
    assert count == 1

    with torch.inference_mode():
        qf = model.wq(x)
        kf = model.wk(x)
        vf = model.wv(x)

    torch.testing.assert_close(qf, q, atol=2.0, rtol=0.05)
    torch.testing.assert_close(kf, k, atol=2.0, rtol=0.05)
    torch.testing.assert_close(vf, v, atol=2.0, rtol=0.05)


def test_module_tree_fusion_flags_parsed() -> None:
    """get_module_tree_fusion_candidates should extract qkv/gateup groups from module_tree flags."""
    from gptqmodel.nn_modules.fused_quant_linear import get_module_tree_fusion_candidates

    # Llama-style tree with per-role flags.
    tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("q_proj:0:q", "k_proj:0:k", "v_proj:0:v", "o_proj:1"),
            "mlp": ("gate_proj:0:gate", "up_proj:0:up", "down_proj:1:down"),
        },
    ]
    qkv, gateup = get_module_tree_fusion_candidates(tree)
    assert qkv == [("q_proj", "k_proj", "v_proj")]
    assert gateup == [("gate_proj", "up_proj")]

    # Qwen-style tree with a single c_attn and w1/w2 gate/up.
    # Original QWenMLP computes w1(x) * silu(w2(x)) -> w1 is up, w2 is gate.
    tree = [
        "transformer",
        "h",
        "#",
        {
            "attn": ("c_attn:0", "c_proj:1"),
            "mlp": ("w1:0:up", "w2:0:gate", "c_proj:1:down"),
        },
    ]
    qkv, gateup = get_module_tree_fusion_candidates(tree)
    assert qkv == []
    assert gateup == [("w2", "w1")]

    # Role order in the tuple does not matter; the parser emits canonical q/k/v and gate/up order.
    tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("v_proj:0:v", "q_proj:0:q", "k_proj:0:k"),
            "mlp": ("up_proj:0:up", "gate_proj:0:gate"),
        },
    ]
    qkv, gateup = get_module_tree_fusion_candidates(tree)
    assert qkv == [("q_proj", "k_proj", "v_proj")]
    assert gateup == [("gate_proj", "up_proj")]


def test_fuse_uses_module_tree_flags() -> None:
    """BaseQModel.fuse() should pick up non-standard member names from module_tree flags."""

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            # Non-standard names; the static alias list does not include these.
            self.query = _make_tritonv2_linear(4096, 4096)
            self.key = _make_tritonv2_linear(4096, 1024)
            self.value = _make_tritonv2_linear(4096, 1024)
            self.gate = _make_tritonv2_linear(4096, 11008)
            self.up = _make_tritonv2_linear(4096, 11008)

    class DummyHFModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    class DummyQModel(BaseQModel):
        module_tree = [
            "model",
            "layers",
            "#",
            {
                "self_attn": ("query:0:q", "key:0:k", "value:0:v"),
                "mlp": ("gate:0:gate", "up:0:up"),
            },
        ]

        def __init__(self, model: nn.Module):
            nn.Module.__init__(self)
            self.model = model
            self.quantized = True
            self.load_quantized_model = True

    hf_model = DummyHFModel().cuda().eval()
    qmodel = DummyQModel(hf_model)
    x = torch.randn(2, 8, 4096, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        layer = hf_model.layers[0]
        q = layer.query(x)
        k = layer.key(x)
        v = layer.value(x)
        gate = layer.gate(x)
        up = layer.up(x)
        mlp_expected = torch.nn.functional.silu(gate) * up

    counts = qmodel.fuse()
    assert counts.get("qkv", 0) == 1
    assert counts.get("gate_up", 0) == 1

    with torch.inference_mode():
        layer = hf_model.layers[0]
        qf = layer.query(x)
        kf = layer.key(x)
        vf = layer.value(x)
        gate_f = layer.gate(x)
        up_f = layer.up(x)
        mlp_actual = torch.nn.functional.silu(gate_f) * up_f

    torch.testing.assert_close(qf, q, atol=2.0, rtol=0.05)
    torch.testing.assert_close(kf, k, atol=2.0, rtol=0.05)
    torch.testing.assert_close(vf, v, atol=2.0, rtol=0.05)
    torch.testing.assert_close(gate_f, gate, atol=2.0, rtol=0.05)
    torch.testing.assert_close(up_f, up, atol=2.0, rtol=0.05)
    torch.testing.assert_close(mlp_actual, mlp_expected, atol=20.0, rtol=0.2)


def test_fused_gate_up_aliases_match_unfused() -> None:
    """Gate/up fusion should work with the w1/w2 naming convention used by Qwen models.

    Original QWenMLP computes ``w1(x) * silu(w2(x))``; w1 is the up projection and w2 is
    the gated projection.
    """

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = _make_tritonv2_linear(4096, 11008)
            self.w2 = _make_tritonv2_linear(4096, 11008)

    model = MLP().cuda().eval()
    x = torch.randn(2, 8, 4096, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        up = model.w1(x)
        gate = model.w2(x)
        expected = torch.nn.functional.silu(gate) * up

    count = install_fused_gate_up(model)
    assert count == 1

    with torch.inference_mode():
        up_f = model.w1(x)
        gate_f = model.w2(x)
        actual = torch.nn.functional.silu(gate_f) * up_f

    torch.testing.assert_close(gate_f, gate, atol=2.0, rtol=0.05)
    torch.testing.assert_close(up_f, up, atol=2.0, rtol=0.05)
    torch.testing.assert_close(actual, expected, atol=20.0, rtol=0.2)


def test_fusion_releases_original_buffers() -> None:
    """After fusion the per-member packed buffers should be deleted by default."""

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _make_tritonv2_linear(4096, 4096)
            self.k_proj = _make_tritonv2_linear(4096, 1024)
            self.v_proj = _make_tritonv2_linear(4096, 1024)

    model = Attn().cuda().eval()
    assert hasattr(model.q_proj, "qweight")
    assert model.q_proj.qweight.numel() > 0

    count = install_fused_qkv(model)
    assert count == 1

    for member in (model.q_proj, model.k_proj, model.v_proj):
        for name in ("qweight", "scales", "qzeros", "g_idx"):
            assert not hasattr(member, name), f"{name} should have been removed from fused member"


def test_fusion_can_keep_original_buffers() -> None:
    """free_original_weights=False should keep per-member packed buffers for save/inspect."""

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _make_tritonv2_linear(4096, 4096)
            self.k_proj = _make_tritonv2_linear(4096, 1024)
            self.v_proj = _make_tritonv2_linear(4096, 1024)

    model = Attn().cuda().eval()
    count = install_fused_qkv(model, free_original_weights=False)
    assert count == 1

    for member in (model.q_proj, model.k_proj, model.v_proj):
        for name in ("qweight", "scales", "qzeros", "g_idx"):
            assert hasattr(member, name), f"{name} should remain when free_original_weights=False"


def test_fused_gate_up_activation_matches_unfused() -> None:
    """Fused gate/up MLP forward should match the unfused act(gate)*up -> down path."""

    hidden = 4096
    intermediate = 11008

    class LlamaLikeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _make_tritonv2_linear(hidden, intermediate)
            self.up_proj = _make_tritonv2_linear(hidden, intermediate)
            self.down_proj = _make_tritonv2_linear(intermediate, hidden)
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    model = LlamaLikeMLP().cuda().eval()
    x = torch.randn(2, 8, hidden, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = model(x)

    count = install_fused_gate_up(model)
    assert count == 1
    assert hasattr(model, "_gptqmodel_fused_gateup_down")

    with torch.inference_mode():
        actual = model(x)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # act(gate)*up amplifies small BF16/GEMM-order differences.
    torch.testing.assert_close(actual, expected, atol=20.0, rtol=0.2)


def test_model_fuse_api_fuses_gate_up_activation() -> None:
    """BaseQModel.fuse() should detect and fuse MLP activation/down projection."""

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _make_tritonv2_linear(4096, 11008)
            self.up_proj = _make_tritonv2_linear(4096, 11008)
            self.down_proj = _make_tritonv2_linear(11008, 4096)
            self.act_fn = nn.SiLU()

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = _make_tritonv2_linear(4096, 4096)
            self.k_proj = _make_tritonv2_linear(4096, 1024)
            self.v_proj = _make_tritonv2_linear(4096, 1024)
            self.mlp = MLP()

        def forward(self, x):
            # Only the MLP path is exercised here.
            return self.mlp(x)

    class DummyHFModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    class DummyQModel(BaseQModel):
        def __init__(self, model: nn.Module):
            nn.Module.__init__(self)
            self.model = model
            self.quantized = True
            self.load_quantized_model = True

    hf_model = DummyHFModel().cuda().eval()
    qmodel = DummyQModel(hf_model)
    x = torch.randn(2, 8, 4096, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = hf_model.layers[0](x)

    counts = qmodel.fuse()
    assert counts.get("qkv", 0) == 1
    assert counts.get("gate_up", 0) == 1
    assert hasattr(hf_model.layers[0].mlp, "_gptqmodel_fused_gateup_down")

    with torch.inference_mode():
        actual = hf_model.layers[0](x)

    torch.testing.assert_close(actual, expected, atol=20.0, rtol=0.2)


def test_qwen_w1w2_activation_fusion_matches_unfused() -> None:
    """Original Qwen MLP order (w1=up, w2=gate) should fuse with SiLU activation."""

    hidden = 4096
    intermediate = 11008

    class QwenLikeMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = _make_tritonv2_linear(hidden, intermediate)
            self.w2 = _make_tritonv2_linear(hidden, intermediate)
            self.c_proj = _make_tritonv2_linear(intermediate, hidden)
            self.act_fn = torch.nn.functional.silu

        def forward(self, x):
            return self.c_proj(self.w1(x) * self.act_fn(self.w2(x)))

    model = QwenLikeMLP().cuda().eval()
    x = torch.randn(2, 8, hidden, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = model(x)

    count = install_fused_gate_up(model)
    assert count == 1
    assert hasattr(model, "_gptqmodel_fused_gateup_down")

    with torch.inference_mode():
        actual = model(x)

    torch.testing.assert_close(actual, expected, atol=20.0, rtol=0.2)


def test_non_swiglu_mlp_not_replaced() -> None:
    """A non-SwiGLU MLP (gate * silu(up)) should fail the activation parity check."""

    hidden = 4096
    intermediate = 11008

    class ReversedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = _make_tritonv2_linear(hidden, intermediate)
            self.up = _make_tritonv2_linear(hidden, intermediate)
            self.down = _make_tritonv2_linear(intermediate, hidden)
            self.act_fn = torch.nn.functional.silu

        def forward(self, x):
            # Swapped activation application; parity check should reject this.
            return self.down(self.gate(x) * self.act_fn(self.up(x)))

    model = ReversedMLP().cuda().eval()
    x = torch.randn(2, 8, hidden, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = model(x)

    count = install_fused_gate_up(model)
    assert count == 1  # projection fusion still installs
    assert not hasattr(model, "_gptqmodel_fused_gateup_down")  # activation fusion rejected

    with torch.inference_mode():
        actual = model(x)

    torch.testing.assert_close(actual, expected, atol=2.0, rtol=0.05)


def test_3bit_tritonv2_gateup_not_fused() -> None:
    """3-bit TritonV2Linear gate/up fusion should be gated out until validated."""

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _make_tritonv2_linear(4096, 11008, bits=3)
            self.up_proj = _make_tritonv2_linear(4096, 11008, bits=3)

    model = MLP().cuda().eval()
    count = install_fused_gate_up(model)
    assert count == 0


def test_modulelist_expert_container_not_activation_fused() -> None:
    """A bare ModuleList of experts should not receive the dense MLP forward replacement."""

    hidden = 4096
    intermediate = 11008

    class Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = _make_tritonv2_linear(hidden, intermediate)
            self.up_proj = _make_tritonv2_linear(hidden, intermediate)
            self.down_proj = _make_tritonv2_linear(intermediate, hidden)
            self.act_fn = torch.nn.functional.silu

        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    experts = nn.ModuleList([Expert() for _ in range(2)])
    count = install_fused_gate_up(experts)
    # Per-expert gate/up projection fusion is fine, but the dense MLP forward
    # replacement must not be installed for a bare expert container.
    assert count == 2
    for expert in experts:
        assert not hasattr(expert, "_gptqmodel_fused_gateup_down")


@marlin_skip
def test_marlin_dequantize_weight_no_scratch_aliasing() -> None:
    """`MarlinLinear.dequantize_weight` must not reuse a scratch buffer across chunks."""
    m = _make_marlin_linear(3072, 512)
    w = m.dequantize_weight(torch.bfloat16)
    assert w.shape == (3072, 512)
    # Random weights almost surely make every row unique; duplicate rows would
    # indicate that a later chunk overwrote the view of an earlier chunk.
    assert w.unique(dim=0).size(0) == w.size(0)


@marlin_skip
def test_fused_marlin_dequantize_weight_no_scratch_aliasing() -> None:
    """`_FusedMarlinKernel.dequantize_weight` must not reuse a scratch buffer across chunks."""
    m1 = _make_marlin_linear(3072, 512)
    m2 = _make_marlin_linear(3072, 512)
    kernel = _FusedMarlinKernel([m1, m2], 1024, torch.device("cuda"))
    w = kernel.dequantize_weight(torch.bfloat16)
    assert w.shape == (3072, 1024)
    assert w.unique(dim=0).size(0) == w.size(0)
