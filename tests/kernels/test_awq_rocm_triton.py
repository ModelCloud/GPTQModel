import pytest
import torch

from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.gemm_awq_triton import AwqGEMMTritonLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import select_quant_linear


pytestmark = pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="ROCm GPU required",
)


def _unpack_awq_codes(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, device=packed.device, dtype=torch.int64)
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=packed.device)
    codes = (packed.to(torch.int64).unsqueeze(-1) >> shifts) & 15
    return codes.index_select(-1, order).flatten(-2)


def test_rocm_awq_gemm_auto_selects_triton_without_native_jit(monkeypatch):
    def unexpected_native_jit():
        raise AssertionError("native AWQ CUDA assembly must not compile on ROCm")

    monkeypatch.setattr(
        "gptqmodel.nn_modules.qlinear.gemm_awq.awq_runtime_available",
        unexpected_native_jit,
    )
    selected = select_quant_linear(
        bits=4,
        group_size=128,
        desc_act=False,
        sym=True,
        device=DEVICE.ROCM,
        backend=BACKEND.AUTO,
        format=FORMAT.GEMM,
        quant_method=METHOD.AWQ,
        pack_dtype=torch.int32,
        dtype=torch.float16,
    )

    assert selected is AwqGEMMTritonLinear


@pytest.mark.parametrize("rows", [1, 16, 128, 256])
@pytest.mark.parametrize(
    "in_features,out_features,group_size", [(512, 512, 32), (4096, 4096, 128)]
)
def test_rocm_awq_triton_matches_independent_torch_oracle(
    rows, in_features, out_features, group_size
):
    torch.manual_seed(19 + rows + group_size)
    device = torch.device("cuda:0")
    groups = in_features // group_size
    qweight = torch.randint(
        -(2**31), 2**31 - 1,
        (in_features, out_features // 8), device=device, dtype=torch.int32,
    )
    qzeros = torch.randint(
        -(2**31), 2**31 - 1,
        (groups, out_features // 8), device=device, dtype=torch.int32,
    )
    qweight[0].zero_()
    qweight[1].fill_(-1)
    qzeros[0].zero_()
    qzeros[1].fill_(-1)
    scales = torch.rand(
        groups, out_features, device=device, dtype=torch.float16
    ).mul_(0.1).add_(0.01).contiguous()
    inputs = torch.randn(rows, in_features, device=device, dtype=torch.float16)

    module = AwqGEMMTritonLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        register_buffers=True,
    ).to(device).eval()
    module.qweight.copy_(qweight)
    module.qzeros.copy_(qzeros)
    module.scales.copy_(scales)
    module.post_init()

    group_indices = torch.arange(in_features, device=device) // group_size
    codes = _unpack_awq_codes(qweight).float()
    zeros = _unpack_awq_codes(qzeros)[group_indices].float()
    dense_weight = ((codes - zeros) * scales[group_indices].float()).half()
    reference_fp32 = inputs.float() @ dense_weight.float()
    reference_visible = reference_fp32.half()

    with torch.inference_mode():
        actual = module(inputs)
    torch.cuda.synchronize(device)

    assert actual.dtype == inputs.dtype
    assert actual.shape == reference_visible.shape
    torch.testing.assert_close(actual, reference_visible, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual.float(), reference_fp32, rtol=2e-3, atol=2e-3)
