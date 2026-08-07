# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS")


def test_auto_import_does_not_force_mps_cpu_fallback():
    env = os.environ.copy()
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "import gptqmodel.models.auto; "
                "assert 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ"
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_gptq_quantizes_on_mps_without_cpu_fallback(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    device = torch.device("mps")
    layer = torch.nn.Linear(32, 24, bias=False, device=device)
    gptq = GPTQ(layer, QuantizeConfig(bits=4, group_size=8, damp_percent=0.01))
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(torch.randn(2, 16, 32, device=device), None)

    qweight, *_ = gptq.quantize(blocksize=16)
    torch.mps.synchronize()

    assert qweight.device.type == "mps"
    assert torch.isfinite(qweight).all()
