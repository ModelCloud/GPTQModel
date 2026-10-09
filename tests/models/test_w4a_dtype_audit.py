# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tests.models.w4a_dtype_audit import (
    _assert_carrier_transport,
    _audit_handoffs,
    _boundary_policy,
    _checkpoint_fingerprint,
    _nvfp4_decode_reference,
    _nvfp4_linear_reference,
    _scale_layout_probe_rows,
    _verify_checkpoint_fingerprint,
)
from tests.w4a_hardware_marks import NVFP4_HARDWARE


@pytest.mark.parametrize("name,mode", [
    ("model.layers.0", "w4afp8"),
    ("model.layers.0.input_layernorm", "w4afp8"),
    ("model.layers.0.self_attn.o_proj", "w4afp8"),
    ("model.layers.0.post_attention_layernorm", "w4afp8"),
    ("model.layers.0.mlp.down_proj", "w4afp8"),
    ("model.layers.1.post_attention_layernorm", "w4a_nvfp4"),
    ("model.layers.1.mlp.gate_proj", "w4a_nvfp4"),
])
def test_audit_policy_follows_attention_and_per_layer_mlp_overrides(name, mode):
    from gptqmodel.quantization import QuantizeConfig

    config = QuantizeConfig(bits=4, group_size=128, desc_act=False, activation={
        "mode": "w4a_nvfp4", "recipe": "least_squares_grid",
        "attention": {"mode": "w4afp8"}, "mlp": {"mode": "w4afp8", "layers": [0]},
    })
    assert _boundary_policy(name, config) == (mode, "least_squares_grid" if mode == "w4a_nvfp4" else None)


@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
@pytest.mark.parametrize("name", ["model.layers.0", "model.layers.0.input_layernorm",
                                 "model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj"])
def test_audit_policy_inherits_unmodified_stream_defaults(mode, name):
    from gptqmodel.quantization import QuantizeConfig

    config = QuantizeConfig(bits=4, group_size=128, desc_act=False, activation=mode)
    assert _boundary_policy(name, config) == (mode, "least_squares" if mode == "w4a_nvfp4" else None)


# The audited stream keeps NVFP4 as its base carrier even when the attention
# lane is promoted to FP8, so this CUDA case stays GB10 / SM 12.1 only.
@NVFP4_HARDWARE
@pytest.mark.parametrize("attention,promoted", [("w4afp8", ()), ("w4a_nvfp4", (0,)),
                                              ("w4afp8", (0,)), ("w4afp8", (0, 1)), (None, ())])
def test_audit_handles_actual_mixed_carriers(fingerprint_checkpoint, monkeypatch, attention, promoted):
    from transformers import AutoTokenizer

    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.qlinear.w4a_llama_stream import install_w4a_llama_stream
    from gptqmodel.quantization import QuantizeConfig
    from tests.models import w4a_dtype_audit
    from tests.models.test_w4a_producer_calibration import _tiny_packed_nvfp4

    core = _tiny_packed_nvfp4()
    core.generation_config.eos_token_id = None
    activation = {"mode": "w4a_nvfp4"}
    if attention is not None:
        activation["attention"] = {"mode": attention}
    if promoted:
        activation["mlp"] = {"mode": "w4afp8", "layers": list(promoted)}
    config = QuantizeConfig(bits=4, group_size=128, desc_act=False, activation=activation)
    install_w4a_llama_stream(core, "w4a_nvfp4", "least_squares", attention_mode=attention,
                             mlp_fp8_layers=promoted)

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = core
            self.quantize_config = config

        def forward(self, **kwargs):
            return core(**kwargs)

    class Tokenizer:
        pad_token_id = 1

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.arange(3, 9)[None]}

    monkeypatch.setattr(GPTQModel, "load", lambda *_args, **_kwargs: Wrapper())
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: Tokenizer())
    monkeypatch.setattr(w4a_dtype_audit, "require_w4a_test_headroom", lambda **_kwargs: None)
    report = w4a_dtype_audit.audit(fingerprint_checkpoint, "w4a_nvfp4")
    fp4_modules = (0 if attention == "w4afp8" else 8) + 3 * (2 - len(promoted))
    assert report["full_coverage"] and report["counts"]["w4a_linear"] == 14
    assert report["independent_gemm_checks"] == fp4_modules * 3
    assert report["handoffs_checked"] == 45


@pytest.fixture
def fingerprint_checkpoint(tmp_path):
    for name in ("model.safetensors", "config.json", "quantize_config.json",
                 "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_bytes(b"abc")
    return tmp_path


def test_checkpoint_fingerprint_records_exact_bytes_and_absent_options(fingerprint_checkpoint):
    result = _checkpoint_fingerprint(fingerprint_checkpoint)
    assert result["model.safetensors"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert result["chat_template.jinja"] is None
    _verify_checkpoint_fingerprint(fingerprint_checkpoint, result)


@pytest.mark.parametrize("name", ["model.safetensors", "config.json", "quantize_config.json",
                                  "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                                  "generation_config.json", "special_tokens_map.json",
                                  "added_tokens.json", "tokenizer.model"])
def test_checkpoint_fingerprint_rejects_changed_or_added_files(fingerprint_checkpoint, name):
    before = _checkpoint_fingerprint(fingerprint_checkpoint)
    (fingerprint_checkpoint / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint files changed"):
        _verify_checkpoint_fingerprint(fingerprint_checkpoint, before)


def test_checkpoint_fingerprint_tracks_symlink_target(fingerprint_checkpoint, tmp_path):
    path = fingerprint_checkpoint / "model.safetensors"
    target = tmp_path / "actual_weights"
    path.rename(target)
    path.symlink_to(target)
    before = _checkpoint_fingerprint(fingerprint_checkpoint)
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checkpoint files changed"):
        _verify_checkpoint_fingerprint(fingerprint_checkpoint, before)


def test_checkpoint_fingerprint_rejects_missing_required_and_broken_optional(fingerprint_checkpoint):
    (fingerprint_checkpoint / "chat_template.jinja").symlink_to(fingerprint_checkpoint / "missing")
    with pytest.raises(FileNotFoundError):
        _checkpoint_fingerprint(fingerprint_checkpoint)
    (fingerprint_checkpoint / "chat_template.jinja").unlink()
    (fingerprint_checkpoint / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        _checkpoint_fingerprint(fingerprint_checkpoint)


def _carrier():
    return {
        "kind": "encoded", "mode": "w4a_nvfp4", "recipe": "least_squares",
        "shape": [1, 8, 128], "model_dtype": "torch.bfloat16", "rotation_applied": False,
        "codes": "torch.float4_e2m1fn_x2", "scales": "torch.float8_e4m3fn",
        "global_scale": "torch.float32", "token_scale": "torch.float32",
        "codes_ptr": 101, "scales_ptr": 102, "global_scale_ptr": 103, "token_scale_ptr": 104,
        "codes_version": 0, "scales_version": 0, "global_scale_version": 0, "token_scale_version": 0,
    }


@pytest.mark.parametrize("field", list(_carrier()))
def test_transport_rejects_changed_carrier_field(field):
    producer = _carrier()
    consumer = deepcopy(producer)
    consumer[field] = "corrupted"
    with pytest.raises(AssertionError, match="changed|lost"):
        _assert_carrier_transport(producer, consumer, "norm -> q_proj")


def test_transport_requires_outer_scale_evidence():
    producer = _carrier()
    consumer = deepcopy(producer)
    del consumer["global_scale_ptr"]
    with pytest.raises(AssertionError, match="Missing carrier field"):
        _assert_carrier_transport(producer, consumer, "norm -> q_proj")


def _calls():
    return [{"input": _carrier(), "output": _carrier()} for _ in range(3)]


def _samples():
    layer = "model.layers.0"
    return {f"{layer}.{name}": {"calls": _calls()} for name in (
        "input_layernorm", "post_attention_layernorm", "self_attn", "self_attn.q_proj",
        "self_attn.k_proj", "self_attn.v_proj", "mlp", "mlp.gate_proj", "mlp.up_proj",
    )}


def test_all_prefill_and_decode_handoffs_checked():
    assert _audit_handoffs(_samples(), ["model.layers.0"]) == 21


def test_correct_decode_does_not_hide_prefill_scale_corruption():
    samples = _samples()
    samples["model.layers.0.self_attn.q_proj"]["calls"][0]["input"]["token_scale_ptr"] += 1
    with pytest.raises(AssertionError, match="token_scale_ptr.*invocation 0"):
        _audit_handoffs(samples, ["model.layers.0"])


def test_missing_consumer_invocation_rejected():
    samples = _samples()
    samples["model.layers.0.mlp.up_proj"]["calls"].pop()
    with pytest.raises(AssertionError, match="Different invocation counts"):
        _audit_handoffs(samples, ["model.layers.0"])


@pytest.mark.parametrize("rows,width", [(1, 128), (33, 256), (129, 256)])
@pytest.mark.parametrize("with_token_scale", [False, True])
def test_independent_decode_covers_all_codes_scale_tiles_and_outer_scales(rows, width, with_token_scale):
    codes = torch.arange(rows * width).remainder(16).reshape(rows, width).to(torch.uint8)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).view(torch.float4_e2m1fn_x2)
    local = (torch.arange(rows * (width // 16)).remainder(7) + 1).reshape(rows, width // 16)
    padded = (rows + 127) // 128 * 128
    swizzled = torch.zeros((width // 128, padded * 8), dtype=torch.float32)
    # Explicit scalar layout construction is independent of the reference's
    # reshape/permutation inverse. Padding rows are deliberately zero.
    for row in range(rows):
        for block in range(width // 16):
            offset = ((row // 128) * 1024 + ((block % 8) // 4) * 512
                      + (row % 32) * 16 + ((row % 128) // 32) * 4 + block % 4)
            swizzled[block // 8, offset] = local[row, block]
    token_scale = torch.linspace(.5, 2., rows) if with_token_scale else None
    carrier = SimpleNamespace(codes=packed, scales=swizzled.to(torch.float8_e4m3fn),
                              global_scale=torch.tensor(.1875), token_scale=token_scale,
                              shape=(1, rows, width))
    magnitudes = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)
    expected = magnitudes[(codes & 7).long()] * torch.where(codes < 8, 1., -1.)
    expected *= local.repeat_interleave(16, dim=-1) * .1875
    if token_scale is not None:
        expected *= token_scale.double()[:, None]
    torch.testing.assert_close(_nvfp4_decode_reference(carrier), expected.reshape(carrier.shape), rtol=0, atol=0)


@pytest.mark.parametrize("zero_format", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("accumulation_dtype", [torch.float32, torch.float64])
def test_native_gptq_consumer_oracle_applies_all_scales_and_bias(zero_format, dtype, accumulation_dtype):
    rows, width, outputs = 3, 256, 128
    generator = torch.Generator().manual_seed(8931)
    logical_weights = torch.randint(0, 16, (width, outputs), generator=generator, dtype=torch.int32)
    shifts = torch.arange(8) * 4
    packed_weights = (logical_weights.reshape(width // 8, 8, outputs).long()
                      << shifts[None, :, None]).sum(1).to(torch.int32)
    zero = 7 if zero_format == 1 else 8
    packed_zero = sum(zero << (4 * offset) for offset in range(8))
    zeros = torch.full((2, outputs // 8), packed_zero, dtype=torch.int64).to(torch.int32)
    weight_scales = torch.linspace(.03125, .125, outputs).repeat(2, 1).to(dtype)
    weight_scales[1] *= 2
    bias = torch.linspace(-.25, .25, outputs).to(dtype)
    module = SimpleNamespace(qweight=packed_weights, qzeros=zeros, scales=weight_scales,
                             g_idx=torch.arange(width, dtype=torch.int32) // 128,
                             qzero_format=lambda: zero_format, bias=bias, out_features=outputs,
                             online_full_had=False, online_partial_had=False)
    codes = torch.arange(rows * width).remainder(16).reshape(rows, width).to(torch.uint8)
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).view(torch.float4_e2m1fn_x2)
    # Constant within each K-group, with different scales between groups.
    local = torch.ones((2, 1024), dtype=torch.float32)
    local[1] *= 3
    token = torch.tensor([.5, 1., 2.])
    carrier = SimpleNamespace(codes=packed, scales=local.to(torch.float8_e4m3fn),
                              global_scale=torch.tensor(.1875), token_scale=token,
                              shape=(1, rows, width), rotation_applied=False)
    values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float64)[(codes & 7).long()]
    values *= torch.where(codes < 8, 1., -1.)
    values[:, 128:] *= 3
    values *= .1875 * token.double()[:, None]
    weights = (logical_weights.double() - 8) * weight_scales.double().repeat_interleave(128, dim=0)
    row_indices = torch.tensor([0, 2])
    expected = values[row_indices] @ weights + bias.double()
    actual = _nvfp4_linear_reference(module, carrier, row_indices, accumulation_dtype=accumulation_dtype)
    tolerance = 1e-6 if accumulation_dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual.double(), expected,
                               rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("rows", [1, 7, 33, 128, 129, 2117])
def test_scale_layout_probes_cover_tile_transitions_and_tail(rows):
    indices = _scale_layout_probe_rows(rows)
    assert indices == sorted(set(indices))
    assert indices[0] == 0 and indices[-1] == rows - 1
    assert all(0 <= index < rows for index in indices)
    for transition in (16, 32, 64, 96, 128):
        for index in (transition - 1, transition):
            if index < rows:
                assert index in indices
