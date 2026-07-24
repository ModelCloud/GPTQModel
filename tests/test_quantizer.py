# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the core Quantizer helper."""

import pytest
import torch

from gptqmodel.quantization import QuantizeConfig, Quantizer
from gptqmodel.quantization.quantizer import quantize


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def test_quantize_accepts_scalar_and_tensor_maxq():
    """The standalone quantize() helper must produce identical output when maxq
    is supplied as a Python scalar or as a 0-d GPU tensor.
    """
    device = "cuda:0"
    rows = 256
    x = torch.randn(rows, 1, dtype=torch.float16, device=device)
    scale = torch.rand(rows, 1, dtype=torch.float16, device=device) * 0.1
    zero = torch.randint(0, 16, (rows, 1), dtype=torch.float16, device=device)
    maxq = 15

    q_scalar = quantize(x, scale, zero, maxq, requires_groupwise_processing=False)
    q_tensor = quantize(x, scale, zero, torch.tensor(maxq, device=device), requires_groupwise_processing=False)
    torch.testing.assert_close(q_scalar, q_tensor)


def test_quantizer_quantize_caches_maxq_value():
    """Quantizer.quantize should compute its integer maxq_value once and reuse
    it, while still matching the reference standalone quantize() output.
    """
    device = "cuda:0"
    qcfg = QuantizeConfig(bits=4, group_size=128, sym=False)
    q = Quantizer(qcfg).to(device)
    q.configure(bits=4, sym=False)

    q.scale = torch.rand(256, 1, dtype=torch.float16, device=device) * 0.1
    q.zero = torch.randint(0, 16, (256, 1), dtype=torch.float16, device=device)
    x = torch.randn(256, 1, dtype=torch.float16, device=device)

    out = q.quantize(x)
    ref = quantize(x, q.scale, q.zero, int(q.maxq.item()), q.requires_groupwise_processing())
    torch.testing.assert_close(out, ref)
    assert getattr(q, "_maxq_value", None) == int(q.maxq.item())
