# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import gptqmodel.models.base as base_module
import gptqmodel.quantization.nf12 as nf12_quantization
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.nn_modules.hooked_linear import HookedLinear
from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear
from gptqmodel.quantization import AutoDtypeCast, QuantizeConfig
from gptqmodel.utils import auto_dtype_cast


def _bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return torch.equal(
        actual.contiguous().view(torch.int16),
        expected.contiguous().view(torch.int16),
    )


def test_auto_dtype_cast_defaults_to_auto_nf12_and_roundtrips():
    config = QuantizeConfig(offload_to_disk=False)

    assert config.auto_dtype_cast is AutoDtypeCast.AUTO_NF12
    payload = config.to_dict()
    assert payload["auto_dtype_cast"] == "auto_nf12"
    assert QuantizeConfig.from_quant_config(payload).auto_dtype_cast is AutoDtypeCast.AUTO_NF12

    disabled = QuantizeConfig(auto_dtype_cast="off", offload_to_disk=False)
    assert disabled.auto_dtype_cast is AutoDtypeCast.OFF


def test_auto_dtype_cast_rejects_unknown_policy():
    with pytest.raises(ValueError, match="auto_dtype_cast"):
        QuantizeConfig(auto_dtype_cast="fp8", offload_to_disk=False)


def test_fp16_eligibility_requires_an_exact_bf16_roundtrip(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)

    exact = torch.full((16, 64), 0.5, dtype=torch.float16)
    exact.view(torch.int16)[0, 0] = -32768  # -0.0
    assert auto_dtype_cast.fp16_is_bf16_exact(exact)
    exact_module = nn.Linear(64, 16, bias=False, dtype=torch.float16)
    with torch.no_grad():
        exact_module.weight.copy_(exact)
    assert auto_dtype_cast.eligible_source_module(exact_module, "cpu") is not None

    inexact = torch.full((16, 64), 1.001, dtype=torch.float16)
    assert not auto_dtype_cast.fp16_is_bf16_exact(inexact)
    inexact_module = nn.Linear(64, 16, bias=False, dtype=torch.float16)
    with torch.no_grad():
        inexact_module.weight.copy_(inexact)
    assert auto_dtype_cast.eligible_source_module(inexact_module, "cpu") is None


def test_only_supported_matrix_modules_are_eligible(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)

    assert (
        auto_dtype_cast.eligible_source_module(nn.Linear(64, 16, bias=True, dtype=torch.bfloat16), "cpu").kind
        == "linear"
    )
    assert (
        auto_dtype_cast.eligible_source_module(nn.Embedding(16, 64, dtype=torch.bfloat16), "cpu").kind == "embedding"
    )
    assert auto_dtype_cast.eligible_source_module(nn.LayerNorm(64, dtype=torch.bfloat16), "cpu") is None
    assert auto_dtype_cast.eligible_source_module(nn.Linear(65, 16, bias=False, dtype=torch.bfloat16), "cpu") is None


def test_hooked_linear_cast_keeps_calibration_hook(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = HookedLinear(64, 16)
    source.weight = nn.Parameter(torch.randn(16, 64, dtype=torch.bfloat16) * 0.02)
    source.bias = nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
    calls = []
    source.forward_hook = lambda module, args, output: calls.append((module, output))

    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    actual = packed(torch.randn(2, 64, dtype=torch.bfloat16))

    assert spec.kind == "hooked_linear"
    assert not auto_dtype_cast.AutoNF12HookedLinear.SUPPORTS_BACKEND_SELECTION
    assert len(calls) == 1
    assert calls[0][0] is packed
    assert actual.dtype is torch.bfloat16
    assert actual.shape == (2, 16)


def test_linear_cast_and_restore_preserve_bf16_bits(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = nn.Linear(64, 16, bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
        source.bias.copy_(torch.randn_like(source.bias))

    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cpu")

    assert _bits_equal(restored.weight, source.weight)
    assert _bits_equal(restored.bias, source.bias)
    assert "weight" not in packed.state_dict()


def test_fp16_exact_cast_and_restore_preserves_source_dtype_and_bits(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = nn.Linear(64, 16, bias=False, dtype=torch.float16)
    with torch.no_grad():
        source.weight.copy_(torch.full_like(source.weight, 0.5))
        source.weight.view(torch.int16)[0, 0] = -32768

    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cpu")

    assert restored.weight.dtype is torch.float16
    assert _bits_equal(restored.weight, source.weight)


def test_embedding_cast_and_restore_preserve_bf16_bits(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = nn.Embedding(16, 64, padding_idx=0, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
        source.weight[0].zero_()

    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    ids = torch.tensor([[0, 1, 15], [2, 3, 4]])
    actual = packed(ids)
    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cpu")

    assert actual.dtype is torch.bfloat16
    assert _bits_equal(actual, source(ids))
    assert restored.padding_idx == 0
    assert _bits_equal(restored.weight, source.weight)


def test_exact_fp16_embedding_keeps_source_dtype(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = nn.Embedding(16, 64, dtype=torch.float16)
    with torch.no_grad():
        source.weight.copy_(torch.full_like(source.weight, 0.5))
        source.weight.view(torch.int16)[0, 0] = -32768

    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    ids = torch.tensor([0, 1, 15])
    actual = packed(ids)

    assert actual.dtype is torch.float16
    assert _bits_equal(actual, source(ids))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None or torch.cuda.get_device_capability(0)[0] < 8,
    reason="requires NVIDIA SM80 or newer",
)
def test_cuda_embedding_uses_only_row_major_packed_storage(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)

    source = nn.Embedding(32, 128, dtype=torch.bfloat16, device="cuda:0")
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
        source.weight[7, 0] = -0.0
        source.weight[15, 127] = 2.0**-20
    ids = torch.tensor([[-1, 0, 7], [15, 31, 32]], dtype=torch.int64, device="cuda:0")

    spec = auto_dtype_cast.eligible_source_module(source, "cuda:0")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    actual = packed(ids)
    oracle = torch.zeros_like(actual)
    valid = (ids >= 0) & (ids < source.num_embeddings)
    oracle[valid] = source(ids[valid])

    assert packed.row_major
    assert {name for name, _ in packed.named_buffers()} == {
        "nf12_row_packed",
        "nf12_row_escapes",
        "nf12_exp_base",
    }
    assert packed.nf12_row_packed.nbytes + packed.nf12_row_escapes.nbytes < source.weight.nbytes
    assert _bits_equal(actual, oracle)

    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cuda:0")
    assert restored.weight.dtype is torch.bfloat16
    assert _bits_equal(restored.weight, source.weight)


def test_forward_role_packs_and_quant_source_role_restores_dense(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(64, 16, bias=True, dtype=torch.bfloat16)

    model = Wrapper()
    with torch.no_grad():
        model.linear.weight.copy_(torch.randn_like(model.linear.weight) * 0.02)
        model.linear.bias.copy_(torch.randn_like(model.linear.bias))
    expected_weight = model.linear.weight.detach().clone()
    expected_bias = model.linear.bias.detach().clone()

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = model
    harness.quantize_config = QuantizeConfig(offload_to_disk=False)
    harness._turtle_lock = threading.RLock()
    harness.turtle_model = None
    harness.auto_dtype_cast_events = []
    named = NamedModule(model.linear, name="linear", full_name="linear", layer_index=0)

    packed = harness._maybe_auto_dtype_cast_forward_module(
        model.linear, device=torch.device("cpu"), named_module=named
    )
    assert model.linear is packed
    assert "weight" not in packed.state_dict()

    restored = harness.shell_module_materialize(
        target_submodule=packed,
        device=torch.device("cpu"),
        role="quant_source",
        named_module=named,
    )
    assert model.linear is restored
    assert type(restored) is nn.Linear
    assert _bits_equal(restored.weight, expected_weight)
    assert _bits_equal(restored.bias, expected_bias)
    assert "auto_dtype_cast_source_spec" not in named.state
    assert harness.auto_dtype_cast_events == [
        {
            "module": "linear",
            "device": "cpu",
            "kind": "linear",
            "source_dtype": "bfloat16",
            "source_bytes": expected_weight.numel() * 2 + expected_bias.numel() * 2,
            "packed_bytes": sum(t.numel() * t.element_size() for t in packed.buffers()),
        }
    ]


def test_native_floatx_forward_plan_is_left_untouched(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)

    model = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = model
    harness.quantize_config = QuantizeConfig(offload_to_disk=False)
    harness.quantize_config._native_floatx_forward_plan = {
        "source_format": "fp8",
        "mode": "native",
    }
    named = NamedModule(model, name="", full_name="linear", layer_index=0)

    result = harness._maybe_auto_dtype_cast_forward_module(model, device=torch.device("cpu"), named_module=named)

    assert result is model
    assert "auto_dtype_cast_source_spec" not in named.state


def test_decoded_floatx_forward_plan_uses_nf12(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)

    model = Wrapper()
    source = model.linear
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = model
    harness.quantize_config = QuantizeConfig(offload_to_disk=False)
    harness.quantize_config._native_floatx_forward_plan = {
        "source_format": "fp8",
        "mode": "decode",
    }
    named = NamedModule(source, name="linear", full_name="linear", layer_index=0)

    result = harness._maybe_auto_dtype_cast_forward_module(source, device=torch.device("cpu"), named_module=named)

    assert result is not source
    assert model.linear is result
    assert named.state["auto_dtype_cast_source_spec"].kind == "linear"


def test_processor_task_releases_dense_storage_then_rebinds_exact_source(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    source = nn.Linear(64, 16, bias=True, dtype=torch.bfloat16)
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
        source.bias.copy_(torch.randn_like(source.bias))
    expected_weight = source.weight.detach().clone()
    expected_bias = source.bias.detach().clone()
    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    named = NamedModule(source, name="linear", full_name="linear", layer_index=0)
    named.module = packed
    task = SimpleNamespace(module=source)
    processor = SimpleNamespace(tasks={"linear": task})

    ModuleLooper._release_processor_task_source_storage(processor, named, source)

    assert task.module.weight.is_meta
    assert task.module.bias.is_meta
    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cpu")
    ModuleLooper._rebind_processor_task_module(processor, named, None, restored)
    assert task.module is restored
    assert _bits_equal(task.module.weight, expected_weight)
    assert _bits_equal(task.module.bias, expected_bias)


def test_hooked_linear_cast_preserves_transformed_hook_operand(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))
    monkeypatch.setattr(
        "gptqmodel.nn_modules.hooked_linear.apply_online_hadamard",
        lambda value, **_kwargs: value.flip(-1),
    )
    monkeypatch.setattr(
        NF12Linear,
        "_apply_rotation_to_input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("rotation applied twice")),
    )

    source = HookedLinear.from_linear(
        nn.Linear(64, 16, bias=False, dtype=torch.bfloat16)
    )
    source.online_full_had = True
    seen = []
    source.forward_hook = lambda _module, args, _output: seen.append(args[0].clone())
    packed = auto_dtype_cast.AutoNF12HookedLinear.from_linear(source)
    x = torch.zeros(1, 64, dtype=torch.bfloat16)
    x[0, 0] = 1

    source(x)
    packed(x)

    assert _bits_equal(seen[0], x.flip(-1))
    assert _bits_equal(seen[1], seen[0])


@pytest.mark.parametrize("mode", ["w4afp8", "w4a_nvfp4"])
def test_hooked_linear_cast_preserves_w4a_replay_semantics(monkeypatch, mode):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))
    monkeypatch.setattr(
        "gptqmodel.nn_modules.hooked_linear.apply_online_hadamard",
        lambda value, **_kwargs: value.flip(-1),
    )

    source = HookedLinear.from_linear(
        nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    )
    with torch.no_grad():
        source.weight.copy_(torch.eye(64, dtype=torch.bfloat16))
    source.online_full_had = True
    source._w4a_stream_replay_mode = mode
    source._w4a_stream_replay_recipe = "least_squares" if mode == "w4a_nvfp4" else None
    source._w4a_replay_model_dtype = torch.bfloat16
    seen = []
    source.forward_hook = lambda _module, args, _output: seen.append(args[0].clone())
    packed = auto_dtype_cast.AutoNF12HookedLinear.from_linear(source)
    x = torch.linspace(-3, 3, 128, dtype=torch.bfloat16).reshape(2, 64)

    expected = source(x)
    actual = packed(x)

    assert len(seen) == 2
    assert _bits_equal(seen[0], seen[1])
    assert _bits_equal(actual, expected)


def test_restore_hooked_linear_preserves_runtime_metadata_and_forward(monkeypatch):
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))
    monkeypatch.setattr(
        "gptqmodel.nn_modules.hooked_linear.apply_online_hadamard",
        lambda value, **_kwargs: value.flip(-1),
    )

    source = HookedLinear.from_linear(
        nn.Linear(64, 16, bias=True, dtype=torch.bfloat16)
    )
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
        source.bias.copy_(torch.randn_like(source.bias))
    source.online_full_had = True
    source.online_partial_had = False
    source.had_dim = -1
    source.had_K = None
    source.K = 1
    source.groups = 2
    source.module_name = "proj"
    source.full_name = "layer.proj"
    source.name = "proj"
    source.forward_hook_last = False
    source._w4a_replay_disabled = True
    source._w4a_norm_preapplied = False
    source._w4a_stream_replay_mode = "w4afp8"
    source._w4a_stream_replay_recipe = None
    source._w4a_replay_model_dtype = torch.bfloat16
    source._w4a_stream_replay_pre_hook = True
    source._w4a_replay_input_hook_active = True
    source._w4a_rotation_preapplied = False
    source._w4a_activation_global_scale = 0.25
    source._w4a_headroom_probe = True
    custom_hook_seen = []
    source.forward_hook = lambda _module, args, _output: custom_hook_seen.append(args[0].clone())
    registered_hook_seen = []
    source.register_forward_hook(
        lambda _module, _args, _output: registered_hook_seen.append(True)
    )

    spec = auto_dtype_cast.eligible_source_module(source, "cpu")
    packed = auto_dtype_cast.cast_source_module(source, spec)
    restored = auto_dtype_cast.restore_source_module(packed, spec, device="cpu")
    x = torch.randn(3, 64, dtype=torch.bfloat16)

    expected = source(x)
    actual = restored(x)

    assert type(restored) is HookedLinear
    assert _bits_equal(restored.weight, source.weight)
    assert _bits_equal(restored.bias, source.bias)
    for name in (
        "online_full_had",
        "online_partial_had",
        "had_dim",
        "had_K",
        "K",
        "groups",
        "module_name",
        "forward_hook",
        "forward_hook_last",
        "full_name",
        "name",
        "_w4a_replay_disabled",
        "_w4a_norm_preapplied",
        "_w4a_stream_replay_mode",
        "_w4a_stream_replay_recipe",
        "_w4a_replay_model_dtype",
        "_w4a_stream_replay_pre_hook",
        "_w4a_replay_input_hook_active",
        "_w4a_rotation_preapplied",
        "_w4a_activation_global_scale",
        "_w4a_headroom_probe",
    ):
        assert getattr(restored, name) == getattr(source, name)
    assert _bits_equal(actual, expected)
    assert len(custom_hook_seen) == 2
    assert len(registered_hook_seen) == 2


def test_quantization_pipeline_installs_nf12_modules_for_every_eligible_kind(monkeypatch):
    """Exercise the live module-installation boundary used by the quantizer."""

    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    monkeypatch.setattr(nf12_quantization, "_native_encode_runtime", lambda *_args: (None, None))

    class PipelineModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(16, 64, dtype=torch.bfloat16)
            self.proj = nn.Linear(64, 16, bias=True, dtype=torch.bfloat16)
            self.hooked = HookedLinear(64, 16)
            self.hooked.weight = nn.Parameter(torch.randn(16, 64, dtype=torch.bfloat16) * 0.02)
            self.hooked.bias = nn.Parameter(torch.randn(16, dtype=torch.bfloat16))
            self.norm = nn.LayerNorm(64, dtype=torch.bfloat16)
            self.unsupported = nn.Linear(63, 16, bias=False, dtype=torch.bfloat16)
            self.max_norm_embedding = nn.Embedding(16, 64, max_norm=1.0, dtype=torch.bfloat16)

    class PipelineProcessor:
        def __init__(self) -> None:
            self.tasks = {}

        def preprocess(self, named_module: NamedModule) -> None:
            self.tasks[named_module.name] = SimpleNamespace(module=named_module.module)

        def is_skipped(self, _named_module: NamedModule) -> bool:
            return False

    model = PipelineModel()
    ids = torch.tensor([0, 1, 15])
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    expected_embedding = model.embed_tokens(ids).clone()
    expected_proj = model.proj(x).clone()
    expected_hooked = model.hooked(x).clone()
    names = [
        "embed_tokens",
        "proj",
        "hooked",
        "norm",
        "unsupported",
        "max_norm_embedding",
    ]
    full = {name: getattr(model, name) for name in names}
    processor = PipelineProcessor()

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = model
    harness.quantize_config = QuantizeConfig(offload_to_disk=False)
    harness._turtle_lock = threading.RLock()
    harness.turtle_model = None
    harness.auto_dtype_cast_events = []

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = harness
    looper.input_embeddings_name = "embed_tokens"
    looper.output_embeddings_name = None
    looper.layer_module_coverage = SimpleNamespace(
        record_hit=lambda *_args: None,
        record_miss=lambda *_args: None,
    )

    subset = looper.create_named_modules(
        module=model,
        full=full,
        is_lm_head_module=False,
        layer_index=0,
        layers_prefix="layer",
        names=names,
        processor=processor,
        fallback=torch.device("cpu"),
    )

    assert isinstance(model.embed_tokens, auto_dtype_cast.AutoNF12Embedding)
    assert isinstance(model.proj, NF12Linear)
    assert isinstance(model.hooked, auto_dtype_cast.AutoNF12HookedLinear)
    assert isinstance(model.norm, nn.LayerNorm)
    assert isinstance(model.unsupported, nn.Linear)
    assert isinstance(model.max_norm_embedding, nn.Embedding)

    assert subset["embed_tokens"].module is model.embed_tokens
    assert subset["proj"].module is model.proj
    assert subset["hooked"].module is model.hooked
    assert subset["norm"].module is model.norm
    assert subset["unsupported"].module is model.unsupported
    assert subset["max_norm_embedding"].module is model.max_norm_embedding

    for name in ("embed_tokens", "proj", "hooked"):
        assert subset[name].state["auto_dtype_cast_was_applied"]
        assert subset[name].state["auto_dtype_cast_source_spec"] is not None
        assert processor.tasks[name].module.weight.is_meta
    assert not processor.tasks["norm"].module.weight.is_meta
    assert not processor.tasks["unsupported"].module.weight.is_meta
    assert not processor.tasks["max_norm_embedding"].module.weight.is_meta

    assert _bits_equal(model.embed_tokens(ids), expected_embedding)
    torch.testing.assert_close(model.proj(x), expected_proj, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(model.hooked(x), expected_hooked, rtol=2e-3, atol=2e-3)
