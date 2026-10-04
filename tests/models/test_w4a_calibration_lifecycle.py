# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CPU metadata transaction checks; numeric carriers are covered on GB10."""

from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear import w4a_boundary
from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation
from gptqmodel.quantization import activation_calibration as calibration
from gptqmodel.quantization.config import QuantizeConfig


def _fixture(monkeypatch):
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn, self.mlp = torch.nn.Module(), torch.nn.Module()
            self.self_attn.q_proj = torch.nn.Module()
            self.self_attn.q_proj.register_buffer("qweight", torch.zeros(1, dtype=torch.int32))

        def forward(self, hidden, **_kwargs):
            value = hidden.codes if isinstance(hidden, W4AActivation) else hidden
            for owner, name, _key in specs:
                value = getattr(owner, f"_w4a_{name}_quantizer")(
                    value, "w4a_nvfp4", recipe="least_squares").codes
            return W4AActivation("w4a_nvfp4", value, torch.ones(1), tuple(value.shape), torch.bfloat16)

    layer = Layer()
    specs = list(w4a_boundary.llama_nvfp4_boundaries([layer], [True]))
    for owner, name, key in specs:
        owner.add_module(f"_w4a_{name}_quantizer", w4a_boundary.NVFP4BoundaryQuantizer(key, "cpu"))
    core = SimpleNamespace(model=SimpleNamespace(layers=[layer]), config=SimpleNamespace(), training=False,
                           _w4a_stream_mode="w4a_nvfp4", _w4a_stream_fused_norms=True,
                           _w4a_stream_recipe="least_squares", _w4a_stream_global_scales=None)
    # The placeholder carrier exercises the metadata transaction only. It
    # deliberately does not claim to represent hardware numeric encoding.
    monkeypatch.setattr(w4a_boundary, "pack_activation", lambda x, mode, **kwargs:
                        W4AActivation(mode, x, torch.ones(1), tuple(x.shape), torch.bfloat16,
                                       global_scale=kwargs.get("global_scale")))
    monkeypatch.setattr(calibration, "_capture_inputs", lambda _core, _samples:
                        [calibration._LayerSample(torch.ones(2, 128), {})])
    qcfg = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False, rotation="hadamard",
                          activation={"mode": "w4a_nvfp4", "recipe": "least_squares"})
    return core, qcfg, specs


def test_public_api_updates_serializable_config_and_hf_metadata(monkeypatch, tmp_path):
    core, qcfg, _specs = _fixture(monkeypatch)
    wrapper = SimpleNamespace(model=core, quantize_config=qcfg, quantized=True)
    report = BaseQModel.calibrate_activations(wrapper, [torch.arange(2)])
    assert len(report["global_scales"]) == 5
    assert qcfg.activation_global_scales == core._w4a_stream_global_scales == report["global_scales"]
    assert core.config.quantization_config["activation"] == qcfg.activation
    qcfg.save_pretrained(str(tmp_path))
    loaded = QuantizeConfig.from_pretrained(str(tmp_path))
    assert loaded.activation_global_scales == report["global_scales"]


@pytest.mark.parametrize("existing_hf_policy", [False, True])
def test_failed_config_commit_restores_runtime_and_metadata(monkeypatch, existing_hf_policy):
    core, qcfg, specs = _fixture(monkeypatch)
    previous = qcfg.activation
    hf_previous = {"previous": "policy"}
    if existing_hf_policy:
        core.config.quantization_config = hf_previous
    def reject(_payload):
        raise ValueError("deliberate config rejection")
    monkeypatch.setattr(type(qcfg), "from_quant_config", staticmethod(reject))
    with pytest.raises(ValueError, match="deliberate"):
        calibration.calibrate_nvfp4_producers(core, [torch.arange(2)], quantize_config=qcfg)
    assert qcfg.activation is previous
    assert core._w4a_stream_global_scales is None
    if existing_hf_policy:
        assert core.config.quantization_config is hf_previous
    else:
        assert not hasattr(core.config, "quantization_config")
    for owner, name, _key in specs:
        module = getattr(owner, f"_w4a_{name}_quantizer")
        assert not module.calibrated and module.observer is None
        assert module.global_scale.item() == 1


def test_mismatched_save_policy_fails_before_observing(monkeypatch):
    core, qcfg, specs = _fixture(monkeypatch)
    qcfg.activation["recipe"] = "nvidia"
    with pytest.raises(ValueError, match="save configuration"):
        calibration.calibrate_nvfp4_producers(core, [torch.arange(2)], quantize_config=qcfg)
    assert all(not getattr(owner, f"_w4a_{name}_quantizer").calibrated for owner, name, _key in specs)


def test_public_api_requires_quantized_weights():
    with pytest.raises(ValueError, match="already quantized"):
        BaseQModel.calibrate_activations(SimpleNamespace(quantized=False), [torch.arange(2)])


def test_calibration_export_preserves_mixed_sub_policies(tmp_path):
    """The export CLI must keep attention/MLP sub-policies while fitting scales.

    Replacing the whole activation dict with an all-NVFP4 policy made reload
    expect producer scales that calibration never fitted.
    """
    import json

    from tests.models.w4a_nvfp4_calibrate import export_view

    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"native-weight-fixture")
    config = {
        "bits": 4, "group_size": 128, "sym": True, "desc_act": False,
        "pack_dtype": "int32", "quant_method": "gptq", "rotation": "hadamard",
        "activation": {
            "mode": "w4a_nvfp4", "recipe": "least_squares",
            "attention": {"mode": "w4afp8"},
            "mlp": {"mode": "w4afp8", "layers": [0]},
        },
    }
    (source / "quantize_config.json").write_text(json.dumps(config))
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    scales = {"model.layers.0.output": .0123, "model.layers.1.output": .0456}
    output = tmp_path / "output"
    export_view(source, output, None, scales, "least_squares")
    exported = json.loads((output / "quantize_config.json").read_text())
    activation = exported["activation"]
    assert activation["attention"] == {"mode": "w4afp8"}
    assert activation["mlp"] == {"mode": "w4afp8", "layers": [0]}
    assert activation["global_scales"] == scales
    # The exported metadata must round-trip into the same mixed policy the
    # runtime installer resolves, without inventing extra NVFP4 producers.
    reloaded = QuantizeConfig.from_quant_config(exported)
    assert reloaded.activation_attention_mode == "w4afp8"
    assert reloaded.activation_attention_recipe is None
    assert reloaded.activation_mlp_fp8_layers == (0,)
