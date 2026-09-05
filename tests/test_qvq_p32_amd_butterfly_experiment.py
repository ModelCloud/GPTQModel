"""Algebra/tail checks for experimental kernels, not model-quality tests."""

import pytest
import torch

triton = pytest.importorskip("triton")


@pytest.mark.parametrize("size_k", [5120, 6144])
@pytest.mark.parametrize("block_n", [2, 4, 8])
@pytest.mark.parametrize("residual", [False, True])
def test_full_k_gemv_reduction_and_padding(size_k, block_n, residual):
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires AMD GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("experiment targets gfx950")
    from scripts.qvq_p32_amd_butterfly_experiment import folded_gemv_full_k_kernel

    g = torch.Generator(device="cuda").manual_seed(20260905 + size_k)
    x = torch.randn(size_k, generator=g, device="cuda", dtype=torch.float16) * 0.01
    w = torch.randn((32, size_k), generator=g, device="cuda", dtype=torch.float16)
    low = torch.randn(w.shape, generator=g, device="cuda", dtype=torch.float16) * 0.001
    effective = w.float() + low.float() if residual else w.float()
    reference = effective @ x.float()
    output = torch.full((36,), 999.0, device="cuda", dtype=torch.float16)
    folded_gemv_full_k_kernel[(32 // block_n,)](
        x, w, low, output, size_k, block_n, triton.next_power_of_2(size_k), residual,
        num_warps=4, num_stages=1, waves_per_eu=0,
    )
    torch.testing.assert_close(output[:32].float(), reference, atol=2e-3, rtol=0)
    assert (output[32:] == 999.0).all()


@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("block_p", [32, 64])
def test_composite_trim_algebra_and_canary(warps, block_p):
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires AMD GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("experiment targets gfx950")
    from gptqmodel.utils.qvq_amd import _qvq_p32_composite_hadamard_constants
    from scripts.qvq_p32_amd_butterfly_experiment import composite_trim_kernel

    base, power, base_size, width = _qvq_p32_composite_hadamard_constants(torch.device("cuda", 0))
    generator = torch.Generator(device="cuda").manual_seed(20260905)
    x = torch.randn((3, 40, 128), generator=generator, device="cuda") * 0.01
    sv = torch.randn(5120, generator=generator, device="cuda", dtype=torch.float16)
    reference = ((base[:40, :40] @ (x @ power)).reshape(3, 5120) / 5120**0.5) * sv
    output = torch.full((4, 5120), 999.0, device="cuda", dtype=torch.float16)
    composite_trim_kernel[(3, width // block_p)](
        x, power, base, sv, output, 5120, base_size, 64, width, block_p, 5120**-0.5,
        num_warps=warps, num_stages=1, waves_per_eu=0, matrix_instr_nonkdim=16, kpack=1,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:3].float(), reference, atol=2e-3, rtol=0)
    assert (output[3] == 999.0).all()


@pytest.mark.parametrize("variant", ["gather", "split"])
@pytest.mark.parametrize("rows", [1, 3, 128, 513])
def test_butterfly128_algebra_and_tail(variant, rows):
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires AMD GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("experiment targets gfx950")
    from scripts.qvq_p32_amd_butterfly_experiment import (
        fht128_kernel,
        fht128_split_kernel,
    )

    kernel = fht128_kernel if variant == "gather" else fht128_split_kernel
    generator = torch.Generator(device="cuda").manual_seed(20260905 + rows)
    x = torch.randn((rows, 128), device="cuda", generator=generator)
    # The identity case verifies every sign/permutation exactly.
    if rows == 128:
        x = torch.eye(128, device="cuda")
    h = torch.tensor(
        [[1.0 if (i & j).bit_count() % 2 == 0 else -1.0 for j in range(128)] for i in range(128)],
        device="cuda",
    )
    reference = x @ h
    # Canary rows detect writes past the tail.
    output = torch.full((rows + 4, 128), 999.0, device="cuda")
    kernel[(triton.cdiv(rows, 4),)](x, output, rows, 4, num_warps=4)
    torch.testing.assert_close(output[:rows], reference, atol=1e-4, rtol=0)
    assert (output[rows:] == 999.0).all()
    if rows == 128:
        assert torch.equal(output[:rows], h)
