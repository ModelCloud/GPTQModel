# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""FP16 inference must not round before a cancelling bias or route BF16 through FP16.

Each case uses an independent FP64 oracle. The weight is 42 on every column and
the bias is -2000, so K=128 and x=12.5 give a pre-bias product of 67200 (outside
the FP16 range) and a true result of 65200. A kernel that rounds the product
before adding the bias returns inf; a correct one returns the FP16-rounded
65216.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear._numerics import matmul_fp32_reduce
from gptqmodel.quantization import FORMAT


K = 128
OUT = 32
WEIGHT = 42.0
BIAS = -2000.0
X_VALUE = 12.5
EXPECTED = X_VALUE * K * WEIGHT + BIAS  # 65200.0
ROUNDED = float(torch.tensor(EXPECTED, dtype=torch.float16))
NVIDIA = torch.cuda.is_available() and not torch.version.hip


def _assert_cancelled(out: torch.Tensor):
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all(), f"expected finite output, got {out.flatten()[:4]}"
    torch.testing.assert_close(
        out.float(),
        torch.full_like(out.float(), ROUNDED),
        rtol=0,
        atol=0,
    )


def _x(rows: int = 1, device="cpu") -> torch.Tensor:
    return torch.full((rows, K), X_VALUE, dtype=torch.float16, device=device)


def _gptq_fixture(cls, **kwargs):
    source = nn.Linear(K, OUT, bias=True, dtype=torch.float16)
    with torch.no_grad():
        source.weight.fill_(WEIGHT)
        source.bias.fill_(BIAS)
    module = cls(
        bits=4,
        group_size=128,
        sym=False,
        desc_act=False,
        in_features=K,
        out_features=OUT,
        bias=True,
        format=FORMAT.GPTQ_V2,
        **kwargs,
    )
    module.pack_block(
        source,
        torch.full((OUT, 1), WEIGHT / 15.0),
        torch.zeros((OUT, 1)),
        torch.arange(K, dtype=torch.int32) // 128,
        workers=1,
    )
    if hasattr(module, "_init_wf_unsqueeze_buffers"):
        module._init_wf_unsqueeze_buffers()
    return module.eval()


def _awq_fixture(cls, **kwargs):
    module = cls(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=K,
        out_features=OUT,
        bias=True,
        register_buffers=True,
        **kwargs,
    )
    module.qweight.fill_(-1)  # every nibble is 15
    module.qzeros.zero_()     # zero point 0
    module.scales.fill_(WEIGHT / 15.0)
    module.bias.fill_(BIAS)
    module.post_init()
    return module.eval()


def test_matmul_fp32_reduce_adds_bias_before_the_cpu_cast():
    weight = torch.full((K, OUT), WEIGHT, dtype=torch.float16)
    bias = torch.full((OUT,), BIAS, dtype=torch.float16)
    _assert_cancelled(matmul_fp32_reduce(_x(), weight, bias))


def test_cpu_torch_aten_adds_bias_before_the_cast():
    from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear

    _assert_cancelled(_gptq_fixture(TorchAtenLinear)(_x()))


def test_cpu_torch_aten_awq_adds_bias_before_the_cast():
    from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear

    _assert_cancelled(_awq_fixture(TorchAtenAwqLinear, checkpoint_format=FORMAT.GEMM)(_x()))


def test_cpu_torch_fused_awq_adds_bias_before_the_cast():
    from gptqmodel.nn_modules.qlinear.torch_fused_awq import TorchFusedAwqLinear

    _assert_cancelled(_awq_fixture(TorchFusedAwqLinear, checkpoint_format=FORMAT.GEMM)(_x()))


def test_cpu_awq_torch_adds_bias_before_the_cast():
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    _assert_cancelled(_awq_fixture(AwqTorchLinear)(_x()))


@pytest.mark.skipif(not NVIDIA, reason="Native FP16 kernels require NVIDIA CUDA")
def test_native_qqq_rejects_bf16_activations():
    from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear

    source = nn.Linear(K, 512, bias=False, dtype=torch.float16)
    with torch.no_grad():
        source.weight.fill_(WEIGHT)
        source.weight[:, 0] = 0.0
    scales = torch.full((512, 1), WEIGHT, dtype=torch.float16)
    x = torch.zeros((1, K), dtype=torch.bfloat16, device="cuda")
    x[0, 0] = 70000.0

    with pytest.raises(NotImplementedError):
        QQQLinear(
            bits=4,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=K,
            out_features=512,
            bias=False,
            dtype=torch.bfloat16,
        )

    # AUTO selects the portable FP32 kernel for BF16; it must stay finite.
    portable = QQQTorchLinear(
        bits=4,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=K,
        out_features=512,
        bias=False,
        register_buffers=False,
    )
    portable.pack(source, scales)
    with torch.inference_mode():
        out = portable.cuda().eval()(x)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="bitsandbytes INT8 requires an accelerator",
)
def test_bitsandbytes_int8_bf16_keeps_row_scale_in_fp32():
    pytest.importorskip("bitsandbytes")
    from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear, bitsandbytes_available

    if not bitsandbytes_available():
        pytest.skip("bitsandbytes backend unavailable")

    source = nn.Linear(K, 512, bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.fill_(WEIGHT)
        source.weight[:, 0] = 0.0
        source.bias.zero_()
    module = BitsAndBytesLinear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=K,
        out_features=512,
        bias=True,
        dtype=torch.bfloat16,
    )
    module.pack_original(linear=source, scales=None, zeros=None)
    module = module.cuda().eval()
    module.post_init()

    outlier = torch.zeros((1, K), dtype=torch.bfloat16, device="cuda")
    outlier[0, 0] = 70000.0
    with torch.inference_mode():
        out = module(outlier)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()

    # A finite FP16-range input whose true BF16 output is far above 65504 must
    # not overflow the FP16 dequant epilogue.
    large = torch.full((1, K), 604.0, dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        out = module(large)
    assert torch.isfinite(out).all()
    # One weight column is zero (set above), so 127 of the 128 terms contribute.
    # BF16 has a 7-bit mantissa; at 3.2e6 its output ulp is 16384, so the single
    # final cast may differ from the exact product by up to half an ulp.
    expected = 604.0 * (K - 1) * WEIGHT
    torch.testing.assert_close(
        out.float(),
        torch.full_like(out.float(), expected),
        rtol=float(torch.finfo(torch.bfloat16).eps),
        atol=0,
    )


def _marlin_available() -> bool:
    return NVIDIA and torch.cuda.get_device_capability()[0] >= 8


@pytest.mark.skipif(not NVIDIA, reason="ExLlamaV2 requires NVIDIA CUDA")
@pytest.mark.parametrize("backend", ["gptq", "awq"])
def test_exllamav2_bias_cancels_before_the_fp16_cast(backend):
    from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear
    from gptqmodel.nn_modules.qlinear.exllamav2_awq import AwqExllamaV2Linear
    from gptqmodel.utils.exllamav2 import ScratchSpace

    cls = ExllamaV2Linear if backend == "gptq" else AwqExllamaV2Linear
    module = cls(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=K,
        out_features=128,
        bias=True,
        register_buffers=True,
        checkpoint_format=FORMAT.GPTQ_V2 if backend == "gptq" else FORMAT.GEMM,
    ).cuda().eval()
    module.qweight.fill_(-1)
    module.qzeros.fill_(-2004318072)  # every zero code is 8
    module.scales.fill_(6.0)          # (15 - 8) * 6 == 42
    module.bias.fill_(BIAS)
    module.post_init(ScratchSpace(module.scratch_space_fixed(), 0))

    _assert_cancelled(module(_x(device="cuda")))


@pytest.mark.skipif(not _marlin_available(), reason="Marlin requires NVIDIA Ampere or newer")
@pytest.mark.parametrize("backend", ["gptq", "awq"])
def test_marlin_bias_cancels_before_the_fp16_cast(backend):
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear

    cls = MarlinLinear if backend == "gptq" else AwqMarlinLinear
    module = cls(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=K,
        out_features=128,
        bias=True,
        register_buffers=True,
        checkpoint_format=FORMAT.GPTQ_V2 if backend == "gptq" else FORMAT.GEMM,
    ).cuda().eval()
    module.qweight.fill_(-1)
    module.qzeros.fill_(-2004318072)
    module.scales.fill_(6.0)
    module.bias.fill_(BIAS)
    module.post_init()

    _assert_cancelled(module(_x(device="cuda")))


@pytest.mark.skipif(not NVIDIA, reason="AwqGEMM requires NVIDIA CUDA")
@pytest.mark.parametrize("rows", [1, 1025])
def test_awq_gemm_bias_cancels_before_the_fp16_cast(rows):
    from gptqmodel.nn_modules.qlinear.gemm_awq import AwqGEMMLinear

    module = AwqGEMMLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=K,
        out_features=128,
        bias=True,
        register_buffers=True,
    ).cuda().eval()
    module.qweight.fill_(-1)
    module.qzeros.zero_()
    module.scales.fill_(WEIGHT / 15.0)
    module.bias.fill_(BIAS)
    module.post_init()

    _assert_cancelled(module(_x(rows, device="cuda")))
