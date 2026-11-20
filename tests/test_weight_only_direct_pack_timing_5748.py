# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5748: direct-pack rows must log the measured pack time.

GGUF, BitsAndBytes and NF12 pack from the original dense weight during
finalization. ``quantize_module`` logged a placeholder ``0.000`` before that
work ran, and finalization then reconstructed the whole dense weight only to
compute a logging-only MAE. The row now gets the measured ``pack_original``
duration, and the MAE diagnostic is opt-in.
"""

import time
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.looper import weight_only_processor
from gptqmodel.looper.weight_only_processor import _mean_abs_error
from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.quantization.config import GGUFConfig


pytestmark = [pytest.mark.cpu]

_SIZE = 128
_PACK_SLEEP = 0.1


class _OneProjection(BaseQModel):
    module_tree = ["model", "layers", "#", {"": ("proj:0",)}]


def _make_model(size: int = _SIZE) -> _OneProjection:
    torch.manual_seed(123)
    native = nn.Module()
    native.model = nn.Module()
    native.model.layers = nn.ModuleList([nn.Module()])
    native.model.layers[0].proj = nn.Linear(size, size, bias=False, dtype=torch.bfloat16)
    native.config = SimpleNamespace(
        use_cache=False,
        tie_word_embeddings=False,
        model_type="direct_pack_timing_fixture",
    )
    config = GGUFConfig(bits="q4_0", smoother=None, offload_to_disk=False, device="cpu")
    return _OneProjection(
        model=native.eval(),
        quantized=False,
        quantize_config=config,
        tokenizer=None,
    )


def _slow_pack(original, calls):
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        result = original(*args, **kwargs)
        time.sleep(_PACK_SLEEP)
        calls.append(time.perf_counter() - started)
        return result

    return wrapper


def _counting(original, calls):
    def wrapper(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    return wrapper


def test_direct_pack_logs_pack_time_without_dense_diagnostic(monkeypatch):
    model = _make_model()
    pack_calls = []
    decode_calls = []
    mae_calls = []

    original_pack = GGUFTorchLinear.pack_original

    def fail_dense_diagnostic(*args, **kwargs):
        raise AssertionError("the weight-error diagnostic must stay off by default")

    monkeypatch.setattr(GGUFTorchLinear, "pack_original", _slow_pack(original_pack, pack_calls))
    monkeypatch.setattr(GGUFTorchLinear, "dequantize_weight", fail_dense_diagnostic)
    monkeypatch.setattr(weight_only_processor, "_mean_abs_error", fail_dense_diagnostic)

    result = model.quantize()

    [entry] = result["weight_only_gguf"]
    assert len(pack_calls) == 1
    assert float(entry["time"]) == pytest.approx(pack_calls[0], abs=0.05)
    assert float(entry["time"]) >= _PACK_SLEEP
    assert entry["time"] != "0.000"
    assert entry["loss"] == "n/a"
    # ``quant_log.csv`` is written from ``model.quant_log``.
    assert model.quant_log[0]["time"] == entry["time"]
    assert model.quant_log[0]["loss"] == "n/a"
    assert decode_calls == []
    assert mae_calls == []
    assert model.quant_region_timer.snapshot().get("submodule_finalize_weight_error", {}).get("count", 0) == 0


def test_requested_weight_error_diagnostic_preserves_loss_and_packed_bytes(monkeypatch):
    reference = _make_model()
    reference_weight = reference.model.model.layers[0].proj.weight.detach().clone()
    reference_pack_calls = []

    original_pack = GGUFTorchLinear.pack_original
    original_decode = GGUFTorchLinear.dequantize_weight

    def forbidden(*args, **kwargs):
        raise AssertionError("default direct pack must not reconstruct the dense weight")

    monkeypatch.setattr(GGUFTorchLinear, "pack_original", _slow_pack(original_pack, reference_pack_calls))
    monkeypatch.setattr(GGUFTorchLinear, "dequantize_weight", forbidden)
    monkeypatch.setattr(weight_only_processor, "_mean_abs_error", forbidden)
    reference_result = reference.quantize()
    [reference_entry] = reference_result["weight_only_gguf"]
    reference_packed = reference.model.model.layers[0].proj.qweight.clone()
    assert reference_entry["loss"] == "n/a"
    assert reference_packed.numel() > 0

    measured_pack_calls = []
    decode_calls = []
    mae_calls = []
    monkeypatch.setattr(GGUFTorchLinear, "pack_original", _slow_pack(original_pack, measured_pack_calls))
    monkeypatch.setattr(
        GGUFTorchLinear,
        "dequantize_weight",
        _counting(original_decode, decode_calls),
    )
    monkeypatch.setattr(
        weight_only_processor,
        "_mean_abs_error",
        _counting(_mean_abs_error, mae_calls),
    )

    model = _make_model()
    result = model.quantize(measure_weight_error=True)

    [entry] = result["weight_only_gguf"]
    assert len(measured_pack_calls) == 1
    assert float(entry["time"]) == pytest.approx(measured_pack_calls[0], abs=0.05)
    assert float(entry["time"]) >= _PACK_SLEEP
    assert model.quant_log[0]["time"] == entry["time"]
    assert len(decode_calls) == 1
    assert len(mae_calls) == 1

    qmodule = model.model.model.layers[0].proj
    expected = _mean_abs_error(reference_weight, qmodule.dequantize_weight().T)
    assert entry["loss"] == f"gguf: {expected:.7f}"
    assert torch.equal(qmodule.qweight, reference_packed)

    diagnostic = model.quant_region_timer.snapshot()["submodule_finalize_weight_error"]
    assert diagnostic["count"] == 1
    assert diagnostic["total"] > 0

    torch.manual_seed(7)
    x = torch.randn(4, _SIZE, dtype=torch.bfloat16)
    assert torch.equal(qmodule(x), reference.model.model.layers[0].proj(x))
