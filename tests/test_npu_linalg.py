# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from gptqmodel.quantization.npu_linalg import npu_inverse_cholesky_factor
from gptqmodel.utils.torch import HAS_NPU


pytestmark = pytest.mark.skipif(not HAS_NPU, reason="Ascend NPU is required")


def _spd_matrix(size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(size, size, generator=generator, dtype=torch.float32)
    return values.matmul(values.T) + torch.eye(size, dtype=torch.float32) * 0.25


def test_npu_inverse_cholesky_factor_matches_cpu_reference():
    device = torch.device("npu:0")

    for size in (8, 64, 128):
        matrix_cpu = _spd_matrix(size, seed=1000 + size)
        matrix_npu = matrix_cpu.to(device=device)

        factor_npu = npu_inverse_cholesky_factor(matrix_npu)
        torch.npu.synchronize()

        reference = torch.linalg.cholesky(
            torch.cholesky_inverse(torch.linalg.cholesky(matrix_cpu)),
            upper=True,
        )

        assert factor_npu.device.type == "npu"
        assert torch.allclose(factor_npu.cpu(), reference, atol=4e-5, rtol=5e-4)

        reconstructed = factor_npu.T.matmul(factor_npu).cpu()
        expected_inverse = torch.linalg.inv(matrix_cpu)
        assert torch.allclose(reconstructed, expected_inverse, atol=5e-5, rtol=5e-4)


def test_npu_inverse_cholesky_factor_rejects_non_positive_definite_matrix():
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32, device="npu:0")

    with pytest.raises(torch._C._LinAlgError):
        npu_inverse_cholesky_factor(matrix)


def test_gptq_npu_hessian_inverse_avoids_torch_npu_cpu_fallback_warnings():
    script = textwrap.dedent(
        """
        import torch
        import torch.nn as nn
        from gptqmodel.quantization.config import QuantizeConfig
        from gptqmodel.quantization.gptq import GPTQ
        from gptqmodel.utils.torch import HAS_NPU

        if not HAS_NPU:
            raise RuntimeError("Ascend NPU is not available")

        torch.npu.set_device(0)
        torch.manual_seed(0)

        module = nn.Linear(16, 16, bias=False, device="npu:0", dtype=torch.float16)
        gptq = GPTQ(module, qcfg=QuantizeConfig(damp_percent=0.05, damp_auto_increment=0.05))

        base = torch.randn(16, 16, dtype=torch.float32)
        hessian_cpu = base.matmul(base.T) + torch.eye(16, dtype=torch.float32) * 0.25
        hessian = hessian_cpu.to(device="npu:0")

        factor, damp = gptq.hessian_inverse(hessian)
        torch.npu.synchronize()

        damped = hessian_cpu.clone()
        diag = damped.diagonal()
        diag.add_(damp * torch.mean(diag))
        expected_inverse = torch.linalg.inv(damped)
        reconstructed = factor.T.matmul(factor).cpu()

        if not torch.allclose(reconstructed, expected_inverse, atol=5e-5, rtol=5e-4):
            raise AssertionError("NPU Hessian inverse does not match CPU reference")

        print("ok")
        """
    )

    env = os.environ.copy()
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )

    combined_output = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0, combined_output
    assert "ok" in proc.stdout
    assert "npu_cpu_fallback" not in combined_output
    assert "linalg_cholesky_ex" not in combined_output
    assert "cholesky_inverse" not in combined_output
