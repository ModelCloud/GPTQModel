import io
import os

import pytest
import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization import GSQConfig
from gptqmodel.quantization.qvq import QVQQuantizationTelemetry, quantize_qvq_linear


pytestmark = pytest.mark.skipif(not os.environ.get("GPU_ALLOCATOR_LEASE_ID"), reason="requires a GPU lease")


@pytest.mark.parametrize("bits", [1, 1.5, 2, 2.5, 3, 3.5])
def test_gsq_quantize_save_reload_forward_and_disabled_parity(bits, monkeypatch):
    import gptqmodel.quantization.qvq_gsq as fitter

    generator = torch.Generator(device="cuda").manual_seed(7)
    weight = torch.randn(16, 32, device="cuda", generator=generator) * 0.1
    x = torch.randn(48, 32, device="cuda", generator=generator)
    gradients = torch.randn(48, 16, device="cuda", generator=generator)
    h, g = x.T @ x, gradients.T @ gradients
    kwargs = dict(bits=bits, output_hessian=g, seed=7, rounding="yaqa", v2b2_p32=True, bank_count=2)
    baseline = quantize_qvq_linear(weight, h, **kwargs)
    original = fitter.refine_p32_fisher

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled GSQ must not enter the fitter")

    monkeypatch.setattr(fitter, "refine_p32_fisher", forbidden)
    disabled = quantize_qvq_linear(weight, h, gsq=GSQConfig(), **kwargs)
    for name, tensor in baseline.serialized_tensors().items():
        assert torch.equal(tensor, disabled.serialized_tensors()[name])
    assert disabled.gsq_diagnostics is None
    monkeypatch.setattr(fitter, "refine_p32_fisher", original)
    telemetry = QVQQuantizationTelemetry()
    enabled = quantize_qvq_linear(weight, h, gsq=GSQConfig(enabled=True, steps=4, candidates=3),
                                  telemetry=telemetry, **kwargs)
    diag = enabled.gsq_diagnostics
    assert diag["after"] <= diag["before"]
    assert diag["objective"] == "normalized_prepared_yaqa_fisher"
    for name in ("SU", "SV", "bank_ids", "bank_alt_id"):
        assert torch.equal(getattr(enabled, name), getattr(baseline, name))
    tensors = enabled.serialized_tensors()
    assert {k: v.numel() for k, v in tensors.items()} == {
        k: v.numel() for k, v in baseline.serialized_tensors().items()}
    buffer = io.BytesIO()
    torch.save(tensors, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    for name, tensor in tensors.items():
        assert torch.equal(tensor, restored[name])
    layer = QVQLinear(bits=bits, in_features=32, out_features=16, tensors=restored,
                      dtype=torch.float32, out_dtype=torch.float32, v2b2_p32=True, bank_count=2).eval()
    expected = x @ enabled.weight.T
    actual = layer(x)
    assert torch.isfinite(actual).all()
    # The CUDA backend uses FP16 compute even for FP32 inputs. Apply the
    # repository's established localized kernel gates, not FP32 tolerances.
    delta = (actual.float() - expected.float()).abs()
    assert delta.mean() <= 2e-3
    assert delta.max() <= 0.046875
    original_layer = QVQLinear(bits=bits, in_features=32, out_features=16, tensors=tensors,
                               dtype=torch.float32, out_dtype=torch.float32, v2b2_p32=True, bank_count=2).eval()
    torch.testing.assert_close(actual, original_layer(x), atol=0, rtol=0)


@pytest.mark.parametrize("overrides,match", [
    ({"rounding": "block_ldlq"}, "P32 YAQA"),
    ({"input_hadamard": False}, "plain YAQA"),
    ({"gsq": GSQConfig(enabled=True, max_candidate_bytes=1)}, "max_candidate_bytes"),
])
def test_gsq_low_level_rejects_unsupported(overrides, match):
    kwargs = dict(bits=2.5, rounding="yaqa", v2b2_p32=True, bank_count=2,
                  output_hessian=torch.eye(16, device="cuda"), gsq=GSQConfig(enabled=True)) | overrides
    with pytest.raises(ValueError, match=match):
        quantize_qvq_linear(torch.eye(16, device="cuda"), torch.eye(16, device="cuda"), **kwargs)
