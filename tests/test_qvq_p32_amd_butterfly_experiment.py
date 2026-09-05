"""Algebra/tail checks for experimental kernels, not model-quality tests."""

import pytest
import torch

triton = pytest.importorskip("triton")


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
