# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#5793: a BF16 forward must not rewrite the FP16 scales of a Marlin layer.

Both Marlin wrappers used to move the registered scales to the activation
dtype inside ``forward``: GPTQ re-registered the cast parameter and AWQ
assigned it to ``scales.data``. On an FP16 checkpoint one BF16 request
therefore rounded the stored scales permanently, and every later FP16 request
cast the rounded values back and ran on different quantized weights. The
metadata probe below reproduces the old write with the native launch and the
stream workspace stubbed, so it runs without CUDA and fails on the parent
revision. The native test reuses the repository's GPTQ and AWQ fixtures and
checks the real kernel path on a Marlin-capable GPU.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from gptqmodel.nn_modules.qlinear import marlin, marlin_awq

from .test_marlin_tile_determinism_4735 import _build_awq, _build_gptq, _skip_reason


# 1.00390625 is exact in FP16 and rounds to 1.0 in BF16, so one BF16 cast is
# visible in the stored value.
_SCALE_VALUE = 1.00390625


def _bare_module(cls):
    """Construct the wrapper state ``post_init`` leaves behind, without a GPU."""

    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    module.out_features = module.in_features = 128
    module.is_lm_head = False
    module._marlin_tile_padding = None
    module.qweight = torch.empty(0, dtype=torch.int32)
    module.qzeros = torch.empty(0, dtype=torch.int32)
    module.g_idx = torch.empty(0, dtype=torch.int32)
    module.g_idx_sort_indices = torch.empty(0, dtype=torch.int32)
    module.workspace = torch.zeros(128, dtype=torch.int32)
    module._marlin_stream_workspaces = {}
    module.bias = module.weight_type = module.adapter = None
    module.fp32 = module.is_k_full = True
    module.scales = torch.nn.Parameter(
        torch.full((1, 128), _SCALE_VALUE, dtype=torch.float16), requires_grad=False
    )
    return module


@pytest.mark.parametrize(
    ("cls", "owner", "launch_names"),
    [
        (marlin.MarlinLinear, marlin, ("apply_gptq_marlin_linear", "apply_gptq_marlin_linear_padded")),
        (
            marlin_awq.AwqMarlinLinear,
            marlin_awq,
            ("apply_awq_marlin_linear", "apply_awq_marlin_linear_padded"),
        ),
    ],
    ids=["gptq", "awq"],
)
@pytest.mark.parametrize("padded", [False, True], ids=["aligned", "padded"])
def test_forward_keeps_the_registered_fp16_scales(cls, owner, launch_names, padded):
    module = _bare_module(cls)
    if padded:
        module._marlin_tile_padding = (256, 128)
    launch_name = launch_names[1] if padded else launch_names[0]

    original = module.scales.detach().clone()
    launches = []

    def fake_launch(**kwargs):
        launches.append(
            (
                kwargs["input"].dtype,
                kwargs["weight_scale"].dtype,
                kwargs["weight_scale"].detach().clone(),
            )
        )
        return torch.zeros(
            (kwargs["input"].shape[0], kwargs["output_size_per_partition"]),
            dtype=kwargs["input"].dtype,
        )

    x = torch.ones(1, 128, dtype=torch.float16)
    with patch.object(owner, "marlin_stream_workspace", return_value=module.workspace):
        with patch.object(owner, launch_name, fake_launch):
            module(x)
            module(x.to(torch.bfloat16))
            module(x)

    assert module.scales.dtype == torch.float16
    torch.testing.assert_close(module.scales, original, rtol=0, atol=0)

    # The kernel still receives a BF16 operand for the BF16 call, derived from
    # the FP16 stored value; the call-local cast must not disappear.
    assert [entry[1] for entry in launches] == [
        torch.float16,
        torch.bfloat16,
        torch.float16,
    ]
    assert torch.equal(launches[1][2], original.to(torch.bfloat16))
    assert torch.equal(launches[0][2], launches[2][2])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("build", [_build_gptq, _build_awq], ids=["gptq", "awq"])
def test_native_forward_preserves_fp16_scales_and_output(build):
    reason = _skip_reason(torch.float16)
    if reason is not None:
        pytest.skip(reason)

    module, weight = build(256, 256, torch.float16)
    original = module.scales.detach().clone()
    x = torch.randn(4, 256, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        first = module(x)
        module(x.to(torch.bfloat16))
        after = module(x)

    assert module.scales.dtype == torch.float16
    torch.testing.assert_close(module.scales, original, rtol=0, atol=0)
    torch.testing.assert_close(after, first, rtol=0, atol=0)

    reference = x.double() @ weight.double()
    error = (first.double() - reference).abs().max().item()
    assert error < 0.05, f"FP16 Marlin disagrees with the FP64 oracle: max|err|={error:.4f}"
