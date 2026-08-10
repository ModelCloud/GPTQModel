# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest
import torch

import gptqmodel.quantization.gptq as gptq_module
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
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
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


def _quantize_mps_layer(
    use_mps_block, *, bits=4, group_size=32, sym=False, desc_act=False
):
    gptq_module._USE_GPTQ_MPS_BLOCK = use_mps_block
    torch.manual_seed(617)
    device = torch.device("mps")
    layer = torch.nn.Linear(128, 128, bias=False, device=device)
    gptq = GPTQ(
        layer,
        QuantizeConfig(
            bits=bits,
            group_size=group_size,
            sym=sym,
            damp_percent=0.01,
            desc_act=desc_act,
            act_group_aware=False,
            offload_to_disk=False,
        ),
    )
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(torch.randn(4, 128, device=device), None)
    output = gptq.quantize(blocksize=128)
    torch.mps.synchronize()
    return output


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
@pytest.mark.skipif(
    not hasattr(torch.mps, "compile_shader"),
    reason="Metal shader runtime is not available",
)
@pytest.mark.parametrize(
    ("bits", "group_size", "sym", "desc_act"),
    [(2, 32, False, False), (4, 64, True, True), (8, 128, False, True)],
)
def test_native_mps_block_matches_eager_gptq_end_to_end(
    monkeypatch, bits, group_size, sym, desc_act
):
    config = {
        "bits": bits,
        "group_size": group_size,
        "sym": sym,
        "desc_act": desc_act,
    }
    eager_output = _quantize_mps_layer(use_mps_block=False, **config)
    launches = 0
    native_block = gptq_module.gptq_block_mps

    def counted_block(*args, **kwargs):
        nonlocal launches
        launches += 1
        return native_block(*args, **kwargs)

    monkeypatch.setattr(gptq_module, "gptq_block_mps", counted_block)
    native_output = _quantize_mps_layer(use_mps_block=True, **config)

    assert launches == 1
    for eager, native in zip(eager_output[:4], native_output[:4]):
        torch.testing.assert_close(native, eager, atol=0, rtol=0)
    assert native_output[5] == pytest.approx(eager_output[5], abs=1e-7)
    assert native_output[6:] == eager_output[6:]

    torch.manual_seed(991)
    heldout = torch.randn(16, 128, device="mps")
    eager_projection = heldout @ eager_output[0].T
    native_projection = heldout @ native_output[0].T
    torch.testing.assert_close(native_projection, eager_projection, atol=0, rtol=0)


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)
def test_mps_block_failure_falls_back_to_exact_eager_result(monkeypatch):
    eager_output = _quantize_mps_layer(use_mps_block=False)

    def deliberate_failure(*_args, **_kwargs):
        raise RuntimeError("deliberate Metal failure")

    monkeypatch.setattr(gptq_module, "gptq_block_mps", deliberate_failure)
    fallback_output = _quantize_mps_layer(use_mps_block=True)

    for eager, fallback in zip(eager_output[:4], fallback_output[:4]):
        torch.testing.assert_close(fallback, eager, atol=0, rtol=0)
    assert fallback_output[5:] == eager_output[5:]
