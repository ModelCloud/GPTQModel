from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "source_path",
    [
        "gptqmodel_ext/awq/quantization/gemv_cuda.cu",
        "gptqmodel_ext/awq/quantization/gemm_cuda_gen.cu",
        "gptqmodel_ext/paroquant/rotation.cu",
    ],
)
def test_cuda_entrypoints_guard_device_and_launch(source_path):
    source = (ROOT / source_path).read_text()
    assert "OptionalCUDAGuard" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in source


def test_awq_gemv_native_contract_checks_tensor_layout_and_device():
    source = (ROOT / "gptqmodel_ext/awq/quantization/gemv_cuda.cu").read_text()
    for contract in (
        "is_contiguous",
        "scalar_type() == at::kHalf",
        "must be on the same CUDA device",
        "does not support zero-row launches",
        "group_size == 64 || group_size == 128",
    ):
        assert contract in source


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_awq_gemv_uses_tensor_device_instead_of_current_device():
    from gptqmodel.utils.awq import (
        awq_gemmv2_forward,
        awq_gemv_forward,
        awq_runtime_error,
        prewarm_awq_extension,
    )

    if not prewarm_awq_extension():
        pytest.skip(awq_runtime_error())

    original_device = torch.cuda.current_device()
    input_device = torch.device("cuda:1" if original_device == 0 else "cuda:0")
    torch.cuda.set_device(original_device)

    qweight = torch.zeros((64, 16), device=input_device, dtype=torch.int32)
    qzeros = torch.zeros((64, 2), device=input_device, dtype=torch.int32)
    scales = torch.ones((64, 16), device=input_device, dtype=torch.float16)

    decode = awq_gemv_forward(
        torch.ones((1, 128), device=input_device, dtype=torch.float16),
        qweight,
        scales,
        qzeros,
        64,
    )
    prefill = awq_gemmv2_forward(
        torch.ones((9, 128), device=input_device, dtype=torch.float16),
        qweight,
        scales,
        qzeros,
        64,
        8,
    )

    assert torch.cuda.current_device() == original_device
    assert decode.device == input_device and decode.shape == (1, 64)
    assert prefill.device == input_device and prefill.shape == (9, 64)
    assert torch.count_nonzero(decode) == 0
    assert torch.count_nonzero(prefill) == 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_paroquant_rotation_uses_tensor_device_instead_of_current_device():
    from gptqmodel import extension
    from gptqmodel.utils.paroquant import (
        _load_rotation_extension,
        build_identity_rotation_buffers,
    )

    if not _load_rotation_extension():
        pytest.skip("ParoQuant CUDA extension is unavailable")

    original_device = torch.cuda.current_device()
    input_device = torch.device("cuda:1" if original_device == 0 else "cuda:0")
    torch.cuda.set_device(original_device)
    x = torch.ones((1, 128), device=input_device, dtype=torch.float16)
    pairs, theta, scales = build_identity_rotation_buffers(
        in_features=128,
        group_size=128,
        krot=1,
        device=input_device,
        dtype=torch.float16,
    )

    out = extension.op("paroquant", "rotate")(
        x,
        pairs,
        theta,
        scales,
        128,
        -1,
        -1,
    )

    assert torch.cuda.current_device() == original_device
    assert out.device == input_device
    torch.testing.assert_close(out, x)
