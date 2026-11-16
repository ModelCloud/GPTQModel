# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1 for the mocked dispatch case; the oracle cases carry
# ``pytest.mark.cuda`` and skip unless the device is an Ada (SM 8.9) GPU.
"""``TritonV2Linear`` must reuse the fused GEMV/small-M GEMM dispatch.

The pre-#3982 forward always called ``QuantLinearFunction``, which dequantizes
the full K x N weight before the matmul.  The reroute goes through
``TorchLinear._forward_eager`` when its gates allow, so decode and small-batch
forward take ``gptq_gemv`` / ``gptq_gemm`` like ``BACKEND.TORCH`` already does.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.utils import arch


TOL = 2e-3


@pytest.fixture(autouse=True)
def _arch_cache():
    arch.clear_arch_cache()
    yield
    arch.clear_arch_cache()


def _is_ada_host() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and arch.is_ada(0)
    )


requires_ada = pytest.mark.skipif(not _is_ada_host(), reason="Ada (SM 8.9) CUDA GPU required")


def _make_tritonv2(in_features, out_features, dtype, group_size=128,
                   desc_act=False, bias=False, bits=4):
    module = TritonV2Linear(
        bits=bits, group_size=group_size, sym=True, desc_act=desc_act,
        in_features=in_features, out_features=out_features,
        pack_dtype=torch.int32, bias=bias, register_buffers=True,
    ).to(device="cuda", dtype=dtype).eval()
    module.optimize = lambda *args, **kwargs: None
    module.qzero_format(format=2)
    with torch.no_grad():
        module.qweight.copy_(
            torch.randint(-2**31, 2**31, module.qweight.shape, device="cuda", dtype=torch.int32)
        )
        module.qzeros.copy_(
            torch.randint(-2**31, 2**31, module.qzeros.shape, device="cuda", dtype=torch.int32)
        )
        module.scales.uniform_(0.001, 0.02)
        if desc_act:
            module.g_idx.copy_(module.g_idx[torch.randperm(in_features, device="cuda")])
            module.g_idx[0] = -1
        if bias:
            module.bias.uniform_(-0.05, 0.05)
    module.post_init()
    return module


def _decoded_weights(module):
    pack_factor = 32 // module.bits
    maxq = (1 << module.bits) - 1
    group_index = module.g_idx.to(torch.int64)
    shifts = torch.arange(pack_factor, device=module.qweight.device, dtype=torch.int64) * module.bits
    codes = ((module.qweight.to(torch.int64).unsqueeze(1) >> shifts[None, :, None]) & maxq)
    codes = codes.reshape(module.in_features, module.out_features)
    zeros = ((module.qzeros.to(torch.int64).unsqueeze(-1) >> shifts[None, None, :]) & maxq)
    zeros = zeros.reshape(module.scales.shape[0], module.out_features)
    return (codes - zeros[group_index]).float() * module.scales[group_index].float()


def _oracle_fp64(x, module):
    return x.double() @ _decoded_weights(module).to(x.dtype).double()


def _dense_weight_candidates(module):
    shape = (module.in_features, module.out_features)
    found = []
    for name, value in vars(module).items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        if value.dim() == 2 and tuple(value.shape) in (shape, (shape[1], shape[0])):
            found.append(f"{name}{tuple(value.shape)}")
    for name, value in module.named_buffers():
        if value.is_floating_point() and value.dim() == 2 and tuple(value.shape) in (shape, (shape[1], shape[0])):
            found.append(f"buffer:{name}{tuple(value.shape)}")
    return found


@requires_ada
@pytest.mark.cuda
def test_tritonv2_forward_uses_fused_eager_path(monkeypatch):
    module = _make_tritonv2(2048, 1024, torch.float16)
    calls = []
    original = TorchLinear._forward_eager

    def spy(self, x, out_shape):
        calls.append((tuple(x.shape), tuple(out_shape)))
        return original(self, x, out_shape)

    monkeypatch.setattr(TorchLinear, "_forward_eager", spy)

    for m in (1, 8, 64):
        calls.clear()
        x = torch.randn(m, 2048, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            out = module(x)
        assert calls, f"M={m} bypassed the fused eager path"
        assert out.shape == (m, 1024)
        assert out.dtype == torch.float16


@requires_ada
@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("desc_act", [False, True])
@pytest.mark.parametrize("m", [1, 2, 8, 16, 64])
def test_tritonv2_forward_matches_independent_oracle(dtype, desc_act, m):
    torch.manual_seed(3982)
    module = _make_tritonv2(4096, 1024, dtype, desc_act=desc_act)
    x = torch.randn(m, 4096, device="cuda", dtype=dtype)

    with torch.inference_mode():
        reference = _oracle_fp64(x, module)
        out = module(x)
        again = module(x)
    torch.cuda.synchronize()

    assert out.dtype == dtype
    assert out.shape == (m, 1024)
    # The visible output is rounded to the activation dtype, so compare in
    # FP64 with a tolerance that covers one output ulp plus the FP32
    # accumulation bound.
    torch.testing.assert_close(
        out.double(),
        reference,
        rtol=2 * torch.finfo(dtype).eps,
        atol=TOL,
    )
    assert torch.equal(out, again)
    assert _dense_weight_candidates(module) == [], "forward retained a dense weight"


@requires_ada
@pytest.mark.cuda
def test_tritonv2_m1_still_uses_gemv():
    module = _make_tritonv2(4096, 1024, torch.float16)
    x1 = torch.randn(1, 4096, device="cuda", dtype=torch.float16)
    x8 = torch.randn(8, 4096, device="cuda", dtype=torch.float16)
    assert module._can_use_rocm_gemv(x1) is True
    assert module._can_use_rocm_gemm(x8) is True
