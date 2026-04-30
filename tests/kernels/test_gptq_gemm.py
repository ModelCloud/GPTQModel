import pytest
import torch

from gptqmodel import extension
from gptqmodel.utils.gptq_gemm import gptq_gemm_qweight_to_b_packed, gptq_gemm_runtime_available
from gptqmodel.utils.torch import HAS_CUDA


def _signed_i32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def test_gptq_gemm_qweight_to_b_packed_layout():
    qweight = torch.tensor(
        [
            [
                _signed_i32(sum((i + 1) << (4 * i) for i in range(8))),
                _signed_i32(sum((8 - i) << (4 * i) for i in range(8))),
            ]
        ],
        dtype=torch.int32,
    )

    packed = gptq_gemm_qweight_to_b_packed(qweight)

    assert packed.dtype == torch.uint8
    assert packed.tolist() == [
        [0x21, 0x78],
        [0x43, 0x56],
        [0x65, 0x34],
        [0x87, 0x12],
    ]


def test_gptq_gemm_extension_is_registered():
    assert "gptq_gemm" in extension.available_extensions()


def test_gptq_gemm_runtime_unavailable_without_cuda():
    if HAS_CUDA:
        pytest.skip("CUDA host should exercise the real GPTQ-GEMM runtime tests.")
    assert not gptq_gemm_runtime_available()
