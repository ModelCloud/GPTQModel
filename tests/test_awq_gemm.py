# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
from awq_test_utils import run_quantized_awq_generation_test

from gptqmodel import BACKEND
from gptqmodel.quantization import FORMAT
from tests.accelerator_gates import requires_nvidia_cuda


pytestmark = [
    pytest.mark.model,
    pytest.mark.slow,
    pytest.mark.skipif(
        not requires_nvidia_cuda(),
        reason="the native AWQ GEMM/GEMV pack kernels are CUDA-only",
    ),
]


def test_awq_gemm_quantized_model_loads_and_generates():
    run_quantized_awq_generation_test(FORMAT.GEMM, BACKEND.GEMM)
