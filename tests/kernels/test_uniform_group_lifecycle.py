# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The fused small-M GPTQ gate must follow the live ``g_idx`` tensor."""

import copy

import pytest
import torch

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear
from gptqmodel.quantization import FORMAT


K = 128
N = 64
GROUP = 32
IDENTITY = torch.arange(K, dtype=torch.int32) // GROUP
# Interleaved groups keep the same row count per group but are not uniform.
ACT_ORDER = torch.tensor([0, 1, 2, 3] * 32, dtype=torch.int32)


def _module(cls, g_idx, scale_values):
    source = torch.nn.Linear(K, N, bias=False)
    with torch.no_grad():
        source.weight.fill_(42.0)
    module = cls(
        bits=4,
        group_size=GROUP,
        sym=False,
        desc_act=True,
        in_features=K,
        out_features=N,
        bias=False,
        format=FORMAT.GPTQ_V2,
        register_buffers=True,
    )
    scales = torch.tensor(scale_values, dtype=torch.float16).reshape(1, 4).expand(N, 4)
    zeros = torch.zeros((N, 4), dtype=torch.int32)
    module.pack_block(source, scales, zeros, g_idx.clone(), workers=1)
    if hasattr(module, "_init_wf_unsqueeze_buffers"):
        module._init_wf_unsqueeze_buffers()
    module.post_init()
    # The packed zero words are literal zeros, which are also valid v2 zero
    # points for this fixture and open the fused-kernel gate.
    module.qzero_format(2)
    return module


def _flag(module):
    return module._gptq_gemm_uniform_group_for_current_g_idx()


def _fp64_oracle(x, module):
    pack_factor = module.pack_factor
    shifts = torch.arange(
        pack_factor, device=module.qweight.device, dtype=torch.int64
    ) * module.bits
    codes = (
        module.qweight.to(torch.int64).unsqueeze(1) >> shifts[None, :, None]
    ) & module.maxq
    codes = codes.reshape(module.in_features, module.out_features)
    zeros = (
        module.qzeros.to(torch.int64).unsqueeze(-1) >> shifts[None, None, :]
    ) & module.maxq
    zeros = zeros.reshape(module.scales.shape[0], module.out_features)
    g_idx = module.g_idx.to(torch.int64)
    weight = (
        codes.double() - zeros[g_idx].double()
    ) * module.scales[g_idx].double()
    return x.double() @ weight


def test_uniform_group_flag_tracks_load_move_and_in_place_writes():
    module = _module(TorchLinear, IDENTITY, [1.0, 2.0, 3.0, 4.0])
    donor = _module(TorchLinear, ACT_ORDER, [4.0, 3.0, 2.0, 1.0])
    assert _flag(module) is True
    assert _flag(donor) is False

    with torch.inference_mode():
        module.load_state_dict(donor.state_dict())
    assert module._gptq_gemm_uniform_group is False
    assert _flag(module) is False

    module.g_idx.copy_(IDENTITY)
    assert _flag(module) is True
    module.g_idx[0] = 3
    assert _flag(module) is False

    module.g_idx.copy_(IDENTITY)
    assert _flag(module) is True
    module = module.half()
    assert _flag(module) is True
    module = module.to(device="cpu", dtype=torch.float16)
    assert _flag(module) is True

    donor.g_idx.copy_(ACT_ORDER)
    donor._gptq_gemm_uniform_group = True
    donor._gptq_gemm_uniform_id = id(donor.g_idx)
    donor._gptq_gemm_uniform_version = donor.g_idx._version
    cloned = copy.deepcopy(donor)
    assert _flag(cloned) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused GPTQ kernel requires CUDA or ROCm")
@pytest.mark.parametrize("cls", [TorchLinear, TritonV2Linear])
@pytest.mark.parametrize("rows", [2, 8, 16, 32, 64])
def test_loaded_act_order_state_matches_fresh_module(cls, rows):
    fresh = _module(cls, ACT_ORDER, [0.25, 0.5, 0.75, 1.0])
    loaded = _module(cls, IDENTITY, [1.0, 1.0, 1.0, 1.0])
    with torch.inference_mode():
        loaded.load_state_dict(fresh.state_dict())

    fresh = fresh.eval().to(device="cuda", dtype=torch.float16)
    loaded = loaded.eval().to(device="cuda", dtype=torch.float16)
    x = torch.randn(rows, K, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        expected = fresh(x)
        actual = loaded(x)

    oracle = _fp64_oracle(x, loaded)
    assert not loaded._gptq_gemm_uniform_group_for_current_g_idx()
    torch.testing.assert_close(actual.double(), oracle, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(expected.double(), oracle, rtol=2e-3, atol=2e-3)
