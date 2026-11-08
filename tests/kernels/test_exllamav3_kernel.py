# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear


def _get_quantize_exl3():
    if not torch.cuda.is_available():
        pytest.skip("EXL3 kernel verification requires CUDA/HIP.")

    try:
        from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"EXL3 quantizer unavailable: {exc}")

    return quantize_exl3


def _require_exl3_quantize_shmem(bits: int) -> None:
    """Skip when the device cannot satisfy the kernel's dynamic shared-memory request.

    The compact 2-bit kernel fits its cost buffers into exactly 64 KiB by
    reusing dead storage for reduction scratch. Other layouts need 704 B extra.
    """
    if not torch.cuda.is_available():
        return
    required = 2 * (65536 >> bits) * 2 + 512 + 64 + 128
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    limit = getattr(props, "shared_memory_per_block_optin", None)
    if limit is None:
        limit = props.shared_memory_per_block
    if bits == 2 and 65536 <= limit < required:
        required = 65536
    if required > limit:
        pytest.skip(
            f"{props.name} allows {limit} B of dynamic shared memory per block; "
            f"EXL3 {bits}-bit codebooks need {required} B."
        )


def _get_block_trace():
    if not torch.cuda.is_available():
        pytest.skip("EXL3 kernel verification requires CUDA/HIP.")

    try:
        from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import block_trace
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"EXL3 quantizer unavailable: {exc}")

    return block_trace


def _get_finalize_capture_h():
    if not torch.cuda.is_available():
        pytest.skip("EXL3 kernel verification requires CUDA/HIP.")

    try:
        from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import finalize_capture_H
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"EXL3 quantizer unavailable: {exc}")

    return finalize_capture_H


class _ScalarReadRecorder(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self) -> None:
        self.scalar_reads = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if str(func).startswith("aten._local_scalar_dense"):
            self.scalar_reads += 1
        return func(*args, **(kwargs or {}))


def _host_block_trace(A: torch.Tensor, B: torch.Tensor, block_size: int = 1024) -> float:
    """Reference accumulation: one host read per block, Python float additions."""

    total = 0.0
    for j_start in range(0, B.shape[1], block_size):
        j_end = min(j_start + block_size, B.shape[1])
        partial = torch.einsum(
            "ik,ij,jk->",
            A,
            B[:, j_start:j_end],
            A[j_start:j_end, :],
        )
        total += partial.item()
    return total


@pytest.mark.parametrize("columns", [0, 1, 1024, 1025, 4096])
def test_exllamav3_block_trace_matches_host_accumulation(columns: int):
    """Device-side accumulation must be bit-identical and read the scalar once."""

    block_trace = _get_block_trace()
    torch.manual_seed(41)
    # ``A`` is the [in_features, out_features] error/weight matrix and ``B`` the
    # [in_features, in_features] Hessian; the loop streams Hessian columns.
    A = torch.randn(columns, 96, device="cuda", dtype=torch.float32)
    B = torch.randn(columns, columns, device="cuda", dtype=torch.float32)

    with _ScalarReadRecorder() as recorder:
        actual = block_trace(A, B)

    assert recorder.scalar_reads == 0
    assert actual.dtype == torch.float64
    assert actual.device.type == "cuda"
    assert actual.item() == _host_block_trace(A, B)


def test_exllamav3_finalize_capture_h_regularizes_diagonal_in_place():
    """Diagonal regularization and LDL zeroing must match the indexed reference."""

    finalize_capture_H = _get_finalize_capture_h()
    torch.manual_seed(31)
    size = 128
    basis = torch.randn(size, size, device="cuda", dtype=torch.float32)
    hessian = basis @ basis.T + torch.eye(size, device="cuda", dtype=torch.float32) * size
    count = 4
    sigma_reg = 0.025

    scaled = hessian / count
    expected_diag = scaled.diagonal() + sigma_reg * torch.diag(scaled).mean()

    q_fallback, _, L, _, diag = finalize_capture_H(
        {"H": hessian, "count": count, "finalized": False},
        {"sigma_reg": sigma_reg, "seed": 787},
        False,
    )

    assert q_fallback is False
    assert torch.equal(diag, expected_diag)
    assert L is not None
    assert torch.count_nonzero(L.diagonal()) == 0


def _clone_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor.clone() if isinstance(tensor, torch.Tensor) else tensor
        for name, tensor in tensors.items()
    }


def test_exllamav3_fallback_reports_unweighted_mse_without_fake_hessian_error():
    quantize_exl3 = _get_quantize_exl3()
    torch.manual_seed(23)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.float32)
    quant_args = {
        "K": 2,
        "devices": [weight.device],
        "apply_out_scales": True,
        "sigma_reg": 0.025,
        "seed": 787,
        "mcg": True,
    }

    _, proxy_error, _ = quantize_exl3(
        weight=weight,
        H_data={
            "H": torch.zeros(256, 256, device="cuda", dtype=torch.float32),
            "count": 1,
            "finalized": False,
        },
        quant_args=quant_args,
        return_weight_q=True,
    )

    metrics = quant_args["error_metrics"]
    assert quant_args["q_fallback"] is True
    assert metrics["quantizer_path"] == "fallback"
    assert metrics["reported_metric_kind"] == "unweighted_reconstruction_mse"
    assert metrics["reported_metric_value"] == pytest.approx(proxy_error)
    assert proxy_error == pytest.approx(metrics["reconstruction"]["mse"])
    assert metrics["hessian_weighted_error_numerator"] is None
    assert metrics["hessian_weighted_reference_denominator"] is None
    assert metrics["hessian_weighted_relative_error"] is None
    assert metrics["hessian_metric_status"] == "not_applicable_fallback"


def test_exllamav3_projection_seed_does_not_mutate_global_cuda_rng():
    quantize_exl3 = _get_quantize_exl3()
    device = torch.device("cuda:0")
    global_generator = torch.Generator(device=device).manual_seed(9917)
    torch.cuda.set_rng_state(global_generator.get_state(), device=device)
    state_before = torch.cuda.get_rng_state(device=device)
    input_generator = torch.Generator(device=device).manual_seed(23)
    weight = torch.randn(
        256,
        128,
        device=device,
        dtype=torch.float32,
        generator=input_generator,
    )
    hessian = torch.eye(256, device=device, dtype=torch.float32)
    quant_args = {
        "K": 2,
        "devices": [device],
        "apply_out_scales": True,
        "sigma_reg": 0.025,
        "seed": 787,
        "mcg": True,
    }

    quantize_exl3(
        weight=weight,
        H_data={"H": hessian, "count": 256, "finalized": False},
        quant_args=quant_args,
        return_weight_q=False,
    )

    assert torch.equal(torch.cuda.get_rng_state(device=device), state_before)


def _build_kernels(
    bits: int, codebook: str
) -> tuple[torch.Tensor, ExllamaV3TorchLinear, ExllamaV3Linear]:
    quantize_exl3 = _get_quantize_exl3()
    _require_exl3_quantize_shmem(bits)

    torch.manual_seed(17)
    in_features = {
        "3inst": 128,
        "mcg": 256,
        "mul1": 384,
    }[codebook]
    out_features = 128
    weight = torch.randn(in_features, out_features, device="cuda", dtype=torch.float32)
    quant_device = weight.device
    hessian = torch.eye(in_features, device="cuda", dtype=torch.float32)
    bias = torch.randn(out_features, device="cuda", dtype=torch.float16)

    quant_args: dict[str, object] = {
        "K": bits,
        "devices": [quant_device],
        "apply_out_scales": True,
        "sigma_reg": 0.025,
        "seed": 787,
    }
    if codebook == "mcg":
        quant_args["mcg"] = True
    elif codebook == "mul1":
        quant_args["mul1"] = True

    weight_q, _, out_tensors = quantize_exl3(
        weight=weight,
        H_data={"H": hessian, "count": in_features, "finalized": False},
        quant_args=quant_args,
        return_weight_q=True,
    )
    metrics = quant_args["error_metrics"]
    assert metrics["schema"] == "gptqmodel.exl3-trellis-error"
    assert metrics["reported_metric_kind"] == "hessian_weighted_relative_error"
    assert metrics["hessian_metric_status"] == "ok"
    assert metrics["hessian_weighted_error_numerator"] >= 0.0
    assert metrics["hessian_weighted_reference_denominator"] > 0.0
    assert metrics["reconstruction"]["domain"] == "regularized_exl3_search_space"
    assert metrics["reconstruction"]["tile_count"] == (in_features // 16) * (
        out_features // 16
    )

    base_tensors = dict(out_tensors)
    base_tensors["bias"] = bias

    torch_kernel = ExllamaV3TorchLinear.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name=f"kernel_{codebook}_{bits}",
        tensors=_clone_tensors(base_tensors),
    ).eval()

    cuda_kernel = ExllamaV3Linear.from_tensors(
        in_features=in_features,
        out_features=out_features,
        name=f"kernel_{codebook}_{bits}",
        tensors=_clone_tensors(base_tensors),
    ).eval()

    try:
        cuda_kernel.post_init()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"EXL3 CUDA runtime unavailable: {exc}")

    return weight_q, torch_kernel, cuda_kernel


@pytest.mark.parametrize(
    ("bits", "codebook"),
    [
        (2, "3inst"),
        (2, "mcg"),
        (4, "mul1"),
    ],
)
def test_exllamav3_torch_weight_matches_quantized_reference(bits: int, codebook: str):
    weight_q, torch_kernel, _ = _build_kernels(bits, codebook)

    with torch.inference_mode():
        dense_weight = torch_kernel.get_weight_tensor(dtype=torch.float32)

    assert dense_weight.dtype == torch.float32
    assert dense_weight.shape == weight_q.shape
    assert torch.allclose(dense_weight, weight_q, rtol=3e-3, atol=3e-3)


def _quantize_single_tensor(*, single_use: bool):
    """Quantize one deterministic tensor and report the result plus the cached Hessian state."""

    quantize_exl3 = _get_quantize_exl3()

    torch.manual_seed(23)
    in_features, out_features = 128, 128
    weight = torch.randn(in_features, out_features, device="cuda", dtype=torch.float32)
    hessian = torch.eye(in_features, device="cuda", dtype=torch.float32)
    quant_args = {
        "K": 4,
        "devices": [weight.device],
        "apply_out_scales": True,
        "sigma_reg": 0.025,
        "seed": 787,
    }
    h_data: dict[str, object] = {"H": hessian, "count": in_features, "finalized": False}
    if single_use:
        h_data["single_use"] = True

    weight_q, _, _ = quantize_exl3(
        weight=weight,
        H_data=h_data,
        quant_args=quant_args,
        return_weight_q=True,
    )

    return weight_q, h_data


def test_exllamav3_single_use_h_data_skips_host_cache():
    """A single-use ``H_data`` must not spend a device->host transfer caching ``L``.

    ``finalize_capture_H`` hands back the very tensor it caches in ``H_data``, so the
    cached entry is only re-read by a caller that re-quantizes against the same Hessian.
    The layer processor builds one dict per module and drops it as soon as the call
    returns, so it opts into ``single_use`` and expects the entry to be released instead
    of copied to the host -- without changing a single quantized value.
    """

    cached_weight_q, cached_h_data = _quantize_single_tensor(single_use=False)
    cached_l = cached_h_data["L"]
    assert isinstance(cached_l, torch.Tensor)
    assert cached_l.device.type == "cpu"

    released_weight_q, released_h_data = _quantize_single_tensor(single_use=True)
    assert released_h_data["L"] is None
    assert torch.equal(released_weight_q, cached_weight_q)


@pytest.mark.parametrize(
    ("bits", "codebook"),
    [
        (2, "3inst"),
        (2, "mcg"),
        (4, "mul1"),
    ],
)
def test_exllamav3_cuda_small_batch_matches_torch_reference(bits: int, codebook: str):
    _, torch_kernel, cuda_kernel = _build_kernels(bits, codebook)
    x = torch.randn(9, torch_kernel.in_features, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        torch_out = torch_kernel(x)
        cuda_out = cuda_kernel(x)

    assert torch_out.dtype == x.dtype
    assert cuda_out.dtype == x.dtype
    assert cuda_out.shape == torch_out.shape
    assert torch.allclose(cuda_out, torch_out, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    ("bits", "codebook"),
    [
        (2, "3inst"),
        (2, "mcg"),
        (4, "mul1"),
    ],
)
def test_exllamav3_cuda_large_batch_matches_torch_reference(bits: int, codebook: str):
    _, torch_kernel, cuda_kernel = _build_kernels(bits, codebook)
    x = torch.randn(40, torch_kernel.in_features, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        torch_out = torch_kernel(x)
        cuda_out = cuda_kernel(x)

    assert torch_out.dtype == x.dtype
    assert cuda_out.dtype == x.dtype
    assert cuda_out.shape == torch_out.shape
    assert torch.allclose(cuda_out, torch_out, rtol=8e-2, atol=8e-2)
