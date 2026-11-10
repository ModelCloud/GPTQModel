# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for the ERNIE 4.5 VL MoE generate-hook cleanup.

`pre_quantize_generate_hook_end()` restores the captured modules after input
capture. With `offload_to_disk=False` it used to call
`setattr(core_model, vision_module, ...)`, passing the module *object* as the
attribute name:

    TypeError: attribute name must be string, not 'Linear'

The cleanup must therefore address `vision_tower` and `resampler_model` by
their string attribute names, and both disk-offload configurations must work.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models import definitions
from gptqmodel.models.definitions.ernie4_5_vl_moe import Ernie4_5_VLMoeQModel


_HIDDEN = 64  # 64x64 float32 == 16 KiB, above the 4 KiB disk-offload floor


class _CoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.embed_tokens = nn.Embedding(_HIDDEN, _HIDDEN)
        self.language_model.rotary_emb = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.vision_tower = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
        self.resampler_model = nn.Linear(_HIDDEN, _HIDDEN, bias=False)


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _CoreModel()


def _definition(model=None, *, offload_to_disk: bool):
    instance = object.__new__(Ernie4_5_VLMoeQModel)
    nn.Module.__init__(instance)
    instance.model = model if model is not None else _Wrapper()
    instance.quantized = True
    instance.load_quantized_model = False
    instance.quantize_config = SimpleNamespace(
        device=torch.device("cpu"),
        offload_to_disk=offload_to_disk,
        offload_to_disk_path="",
    )
    return instance


def test_hook_start_materializes_named_modules():
    definition = _definition(offload_to_disk=False)

    definition.pre_quantize_generate_hook_start()

    core_model = definition.model.model
    # The names must still resolve after the materialize step; a module object
    # used as an attribute name would have raised before this point.
    assert isinstance(core_model.vision_tower, nn.Linear)
    assert isinstance(core_model.resampler_model, nn.Linear)
    assert isinstance(core_model.language_model.embed_tokens, nn.Embedding)
    assert isinstance(core_model.language_model.rotary_emb, nn.Linear)


def test_hook_end_without_disk_offload_uses_string_attribute_names(monkeypatch, tmp_path):
    definition = _definition(offload_to_disk=False)
    definition.quantize_config.offload_to_disk_path = str(tmp_path)

    moved = []

    def _recording_move_to(obj, device, dtype=None):
        moved.append(obj)
        return obj

    monkeypatch.setattr(definitions.ernie4_5_vl_moe, "move_to", _recording_move_to)
    core_model = definition.model.model
    vision_tower = core_model.vision_tower
    resampler_model = core_model.resampler_model

    definition.pre_quantize_generate_hook_end()

    assert moved == [
        core_model.language_model.embed_tokens,
        core_model.language_model.rotary_emb,
        vision_tower,
        resampler_model,
    ]
    # Both modules must remain registered under their real names.
    assert core_model.vision_tower is vision_tower
    assert core_model.resampler_model is resampler_model
    assert hasattr(core_model, "vision_tower")
    assert hasattr(core_model, "resampler_model")


def test_hook_end_without_disk_offload_releases_real_modules(tmp_path):
    definition = _definition(offload_to_disk=False)
    definition.quantize_config.offload_to_disk_path = str(tmp_path)
    core_model = definition.model.model
    vision_tower = core_model.vision_tower
    resampler_model = core_model.resampler_model

    definition.pre_quantize_generate_hook_end()

    assert core_model.vision_tower is vision_tower
    assert core_model.resampler_model is resampler_model
    # The non-offload path moves real (materialized) tensors to CPU.
    assert not vision_tower.weight.is_meta
    assert not resampler_model.weight.is_meta
    assert vision_tower.weight.device.type == "cpu"
    assert resampler_model.weight.device.type == "cpu"


def test_hook_end_with_disk_offload_keeps_modules_registered(tmp_path):
    definition = _definition(offload_to_disk=True)
    definition.quantize_config.offload_to_disk_path = str(tmp_path)
    core_model = definition.model.model
    vision_tower = core_model.vision_tower
    resampler_model = core_model.resampler_model

    definition.pre_quantize_generate_hook_end()

    assert core_model.vision_tower is vision_tower
    assert core_model.resampler_model is resampler_model
    # Disk offload leaves meta placeholders plus restore hooks in place.
    assert vision_tower.weight.is_meta
    assert resampler_model.weight.is_meta
    assert core_model.language_model.embed_tokens.weight.is_meta
    assert core_model.language_model.rotary_emb.weight.is_meta
    assert hasattr(vision_tower, "_hf_hook")
    assert hasattr(resampler_model, "_hf_hook")


def test_hook_end_releases_other_modules_after_one_offload_fails(monkeypatch, tmp_path):
    definition = _definition(offload_to_disk=True)
    definition.quantize_config.offload_to_disk_path = str(tmp_path)
    core_model = definition.model.model
    failing_module = core_model.language_model.embed_tokens
    real_offload = definitions.ernie4_5_vl_moe.offload_to_disk

    def failing_first_offload(*, model, module, disk_path):
        if module is failing_module:
            raise OSError("embedding offload failed")
        return real_offload(model=model, module=module, disk_path=disk_path)

    monkeypatch.setattr(definitions.ernie4_5_vl_moe, "offload_to_disk", failing_first_offload)

    with pytest.raises(OSError, match="embedding offload failed"):
        definition.pre_quantize_generate_hook_end()

    assert not failing_module.weight.is_meta
    assert core_model.language_model.rotary_emb.weight.is_meta
    assert core_model.vision_tower.weight.is_meta
    assert core_model.resampler_model.weight.is_meta


@pytest.mark.parametrize("offload_to_disk", [True, False])
def test_hook_end_skips_absent_optional_modules(tmp_path, offload_to_disk):
    definition = _definition(offload_to_disk=offload_to_disk)
    definition.quantize_config.offload_to_disk_path = str(tmp_path)
    core_model = definition.model.model
    del core_model.resampler_model

    definition.pre_quantize_generate_hook_end()

    assert not hasattr(core_model, "resampler_model")


def _tiny_ernie4_5_vl_moe():
    from transformers.models.ernie4_5_vl_moe.configuration_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeConfig,
    )
    from transformers.models.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe import (
        Ernie4_5_VLMoeForConditionalGeneration,
    )

    config = Ernie4_5_VLMoeConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 64,
            "max_position_embeddings": 64,
            "moe_intermediate_size": [40, 32],
            "moe_k": 2,
            "moe_num_experts": 2,
            "moe_num_shared_experts": 1,
            "mlp_layer_types": ["dense", "sparse"],
            "layer_types": ["full_attention", "full_attention"],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [4, 4, 8],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "patch_size": 16,
            "image_size": 32,
            "spatial_merge_size": 2,
        },
    )
    config._attn_implementation = "eager"
    return Ernie4_5_VLMoeForConditionalGeneration(config)


@pytest.mark.parametrize("offload_to_disk", [True, False])
def test_hook_round_trip_on_real_ernie4_5_vl_moe(tmp_path, offload_to_disk):
    model = _tiny_ernie4_5_vl_moe()
    core_model = model.model
    vision_tower = core_model.vision_tower
    resampler_model = core_model.resampler_model
    owned_keys_before = set(model.state_dict().keys())

    definition = _definition(model=model, offload_to_disk=offload_to_disk)
    definition.quantize_config.offload_to_disk_path = str(tmp_path)

    definition.pre_quantize_generate_hook_start()
    definition.pre_quantize_generate_hook_end()

    # The layout resolver and the register names must agree after cleanup; this
    # is the exact contract the input-capture cleanup relies on.
    assert Ernie4_5_VLMoeQModel._resolve_multimodal_layout(model) == ("model", core_model)
    assert core_model.vision_tower is vision_tower
    assert core_model.resampler_model is resampler_model
    # Cleanup only changes storage placement, never checkpoint ownership.
    assert set(model.state_dict().keys()) == owned_keys_before


def test_hook_start_uses_shell_materialize_when_lazy_loading():
    """The lazy/turtle path must materialize by name through the shell helper.

    `_materialize_module` only takes the plain `move_to` shortcut when the
    instance has no `_turtle_lock`. A loaded wrapper always has it, so the
    hook start has to route through `shell_module_materialize` and register the
    returned module under the same attribute name.
    """

    definition = _definition(offload_to_disk=False)
    # `_turtle_lock` is what selects the shell-materialize branch.
    definition._turtle_lock = object()

    calls = []

    def _fake_shell_materialize(target_submodule, device, **kwargs):
        calls.append((target_submodule, device))
        return target_submodule

    definition.shell_module_materialize = _fake_shell_materialize
    core_model = definition.model.model

    definition.pre_quantize_generate_hook_start()

    assert len(calls) == 4, "hook start must route all four modules through shell_module_materialize"
    assert all(device == torch.device("cpu") for _, device in calls)
    # Modules, not attribute names, are handed to the shell helper.
    assert all(isinstance(module, nn.Module) for module, _ in calls)
    # Registration survived the shell round trip, under the original names.
    assert isinstance(core_model.vision_tower, nn.Linear)
    assert isinstance(core_model.resampler_model, nn.Linear)
    assert isinstance(core_model.language_model.embed_tokens, nn.Embedding)
    assert isinstance(core_model.language_model.rotary_emb, nn.Linear)


def test_hook_end_releases_by_name_when_lazy_loading():
    """Cleanup stays a CPU release on the lazy path; it must not re-register."""

    definition = _definition(offload_to_disk=False)
    definition._turtle_lock = object()

    def _fail_shell_materialize(*args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("hook end must not materialize; it releases")

    definition.shell_module_materialize = _fail_shell_materialize
    core_model = definition.model.model
    vision_tower = core_model.vision_tower
    resampler_model = core_model.resampler_model

    definition.pre_quantize_generate_hook_end()

    assert core_model.vision_tower is vision_tower
    assert core_model.resampler_model is resampler_model
    assert core_model.vision_tower.weight.device.type == "cpu"
    assert core_model.resampler_model.weight.device.type == "cpu"


def test_capture_cleanup_runs_when_forward_raises(monkeypatch):
    """`cache_inputs` must release capture state even when the forward fails.

    The generate hooks materialize modules onto the forward device and the
    first-layer pre-hook is registered before the calibration loop. If the
    forward raises, a cleanup that sits *after* the try/finally (rather than
    inside it) is skipped, leaving the model mutated and the pre-hook attached
    for the rest of the run.
    """

    import types

    from gptqmodel.looper import stage_inputs_capture as capture_mod
    from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
    from gptqmodel.models.base import BaseQModel

    events = []

    class _ExplodingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hooked_layer = nn.Linear(4, 4)
            self.hooked_layer.full_name = "test_layer"
            self.config = types.SimpleNamespace(model_type="test")

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            raise RuntimeError("calibration forward failed")

    model = _ExplodingModel()

    class _FakeGPTQModel:
        capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
        capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
        finalize_input_capture_example = BaseQModel.finalize_input_capture_example
        move_input_capture_example = BaseQModel.move_input_capture_example
        prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs
        run_input_capture = BaseQModel.run_input_capture
        _sanitize_input_ids_for_embeddings = BaseQModel._sanitize_input_ids_for_embeddings

        def __init__(self):
            self.quantize_config = types.SimpleNamespace(
                device=torch.device("cpu"),
                calibration_data_device=torch.device("cpu"),
                compute_device_filter=None,
            )
            self.model = model
            self.ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
            self.ATTENTION_MASKS_DTYPE = torch.long
            self.INPUT_EMBEDDING_EXTRA_ARGS = None
            self.quant_region_timer = None

        def shell_module_materialize(self, target_submodule, device, **kwargs):
            del device, kwargs
            return target_submodule

        def get_base_modules(self, model):
            del model
            return []

        @classmethod
        def get_modules_with_direct_meta_tensors(cls, model):
            del model
            return []

        def pre_quantize_generate_hook_start(self):
            events.append("hook_start")

        def pre_quantize_generate_hook_end(self):
            events.append("hook_end")

    class _Looper:
        def __init__(self, gptq_model):
            self.gptq_model = gptq_model

        def _batch_row_count(self, batch_inputs):
            del batch_inputs
            return 1

    class _NullCM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(capture_mod, "DEVICE_THREAD_POOL", types.SimpleNamespace(read_lock=lambda d: _NullCM()))
    monkeypatch.setattr(capture_mod, "ctx", lambda *cms: _NullCM())
    monkeypatch.setattr(capture_mod, "device_ctx", lambda device: _NullCM())
    monkeypatch.setattr(capture_mod, "await_torch_linear_prewarm", lambda *a, **k: None)

    class _Logger:
        def info(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

        def warn(self, *a, **k):
            pass

        def pb(self, *a, **k):
            class _Bar:
                def manual(self):
                    return self

                def set(self, **kw):
                    return self

                def title(self, *a):
                    return self

                def subtitle(self, *a):
                    return self

                def draw(self):
                    return self

                def close(self):
                    pass

            return _Bar()

    capture = StageInputsCapture(_Looper(_FakeGPTQModel()), logger=_Logger())
    calibration_data = [{"input_ids": torch.randint(0, 10, (1, 4))}]

    handle_removed = {"count": 0}
    real_register = nn.Module.register_forward_pre_hook

    class _TrackedHandle:
        def __init__(self, inner):
            self._inner = inner

        def remove(self):
            handle_removed["count"] += 1
            return self._inner.remove()

    def _tracking_register(self, hook, with_kwargs=False):
        return _TrackedHandle(real_register(self, hook, with_kwargs=with_kwargs))

    monkeypatch.setattr(nn.Module, "register_forward_pre_hook", _tracking_register, raising=False)

    with pytest.raises(RuntimeError, match="calibration forward failed"):
        capture.cache_inputs(
            layers=[model.hooked_layer],
            calibration_data=calibration_data,
            use_cache=False,
        )

    # Cleanup ran on the exception path, in order, and released the pre-hook.
    assert events == ["hook_start", "hook_end"]
    assert handle_removed["count"] == 1


def test_capture_removes_hook_even_when_release_raises(monkeypatch):
    """A failing generate-hook cleanup must still detach the capture pre-hook."""

    import types

    from gptqmodel.looper import stage_inputs_capture as capture_mod
    from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
    from gptqmodel.models.base import BaseQModel

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.hooked_layer = nn.Linear(4, 4)
            self.hooked_layer.full_name = "test_layer"
            self.config = types.SimpleNamespace(model_type="test")

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            return None

    model = _Model()
    removed = {"count": 0}

    class _FakeGPTQModel:
        capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
        capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
        finalize_input_capture_example = BaseQModel.finalize_input_capture_example
        move_input_capture_example = BaseQModel.move_input_capture_example
        prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs
        run_input_capture = BaseQModel.run_input_capture
        _sanitize_input_ids_for_embeddings = BaseQModel._sanitize_input_ids_for_embeddings

        def __init__(self):
            self.quantize_config = types.SimpleNamespace(
                device=torch.device("cpu"),
                calibration_data_device=torch.device("cpu"),
                compute_device_filter=None,
            )
            self.model = model
            self.ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
            self.ATTENTION_MASKS_DTYPE = torch.long
            self.INPUT_EMBEDDING_EXTRA_ARGS = None
            self.quant_region_timer = None

        def shell_module_materialize(self, target_submodule, device, **kwargs):
            del device, kwargs
            return target_submodule

        def get_base_modules(self, model):
            del model
            return []

        @classmethod
        def get_modules_with_direct_meta_tensors(cls, model):
            del model
            return []

        def pre_quantize_generate_hook_start(self):
            pass

        def pre_quantize_generate_hook_end(self):
            raise RuntimeError("release failed")

    class _Looper:
        def __init__(self, gptq_model):
            self.gptq_model = gptq_model

        def _batch_row_count(self, batch_inputs):
            del batch_inputs
            return 1

    class _NullCM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(capture_mod, "DEVICE_THREAD_POOL", types.SimpleNamespace(read_lock=lambda d: _NullCM()))
    monkeypatch.setattr(capture_mod, "ctx", lambda *cms: _NullCM())
    monkeypatch.setattr(capture_mod, "device_ctx", lambda device: _NullCM())
    monkeypatch.setattr(capture_mod, "await_torch_linear_prewarm", lambda *a, **k: None)

    class _Logger:
        def info(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

        def warn(self, *a, **k):
            pass

        def pb(self, *a, **k):
            class _Bar:
                def manual(self):
                    return self

                def set(self, **kw):
                    return self

                def title(self, *a):
                    return self

                def subtitle(self, *a):
                    return self

                def draw(self):
                    return self

                def close(self):
                    pass

            return _Bar()

    capture = StageInputsCapture(_Looper(_FakeGPTQModel()), logger=_Logger())

    real_register = nn.Module.register_forward_pre_hook

    class _TrackedHandle:
        def __init__(self, inner):
            self._inner = inner

        def remove(self):
            removed["count"] += 1
            return self._inner.remove()

    def _tracking_register(self, hook, with_kwargs=False):
        return _TrackedHandle(real_register(self, hook, with_kwargs=with_kwargs))

    monkeypatch.setattr(nn.Module, "register_forward_pre_hook", _tracking_register, raising=False)

    with pytest.raises(RuntimeError, match="release failed"):
        capture.cache_inputs(
            layers=[model.hooked_layer],
            calibration_data=[{"input_ids": torch.randint(0, 10, (1, 4))}],
            use_cache=False,
        )

    assert removed["count"] == 1


def test_capture_removes_hook_when_hook_start_raises(monkeypatch):
    """A failing `pre_quantize_generate_hook_start` must not leak the pre-hook.

    The start hook materializes side modules onto the forward device. When it
    raises partway through (for example an OOM on the vision tower), the
    capture has already registered its first-layer pre-hook and must still
    detach it before the exception escapes the stage.
    """

    import types

    from gptqmodel.looper import stage_inputs_capture as capture_mod
    from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
    from gptqmodel.models.base import BaseQModel

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.hooked_layer = nn.Linear(4, 4)
            self.hooked_layer.full_name = "test_layer"
            self.config = types.SimpleNamespace(model_type="test")

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            return None

    model = _Model()
    removed = {"count": 0}

    class _FakeGPTQModel:
        capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
        capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
        finalize_input_capture_example = BaseQModel.finalize_input_capture_example
        move_input_capture_example = BaseQModel.move_input_capture_example
        prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs
        run_input_capture = BaseQModel.run_input_capture
        _sanitize_input_ids_for_embeddings = BaseQModel._sanitize_input_ids_for_embeddings

        def __init__(self):
            self.quantize_config = types.SimpleNamespace(
                device=torch.device("cpu"),
                calibration_data_device=torch.device("cpu"),
                compute_device_filter=None,
            )
            self.model = model
            self.ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
            self.ATTENTION_MASKS_DTYPE = torch.long
            self.INPUT_EMBEDDING_EXTRA_ARGS = None
            self.quant_region_timer = None

        def shell_module_materialize(self, target_submodule, device, **kwargs):
            del device, kwargs
            return target_submodule

        def get_base_modules(self, model):
            del model
            return []

        @classmethod
        def get_modules_with_direct_meta_tensors(cls, model):
            del model
            return []

        def pre_quantize_generate_hook_start(self):
            raise RuntimeError("hook start failed while materializing")

        def pre_quantize_generate_hook_end(self):
            pass

    class _Looper:
        def __init__(self, gptq_model):
            self.gptq_model = gptq_model

        def _batch_row_count(self, batch_inputs):
            del batch_inputs
            return 1

    class _NullCM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(capture_mod, "DEVICE_THREAD_POOL", types.SimpleNamespace(read_lock=lambda d: _NullCM()))
    monkeypatch.setattr(capture_mod, "ctx", lambda *cms: _NullCM())
    monkeypatch.setattr(capture_mod, "device_ctx", lambda device: _NullCM())
    monkeypatch.setattr(capture_mod, "await_torch_linear_prewarm", lambda *a, **k: None)

    class _Logger:
        def info(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

        def warn(self, *a, **k):
            pass

        def pb(self, *a, **k):
            class _Bar:
                def manual(self):
                    return self

                def set(self, **kw):
                    return self

                def title(self, *a):
                    return self

                def subtitle(self, *a):
                    return self

                def draw(self):
                    return self

                def close(self):
                    pass

            return _Bar()

    capture = StageInputsCapture(_Looper(_FakeGPTQModel()), logger=_Logger())

    real_register = nn.Module.register_forward_pre_hook

    class _TrackedHandle:
        def __init__(self, inner):
            self._inner = inner

        def remove(self):
            removed["count"] += 1
            return self._inner.remove()

    def _tracking_register(self, hook, with_kwargs=False):
        return _TrackedHandle(real_register(self, hook, with_kwargs=with_kwargs))

    monkeypatch.setattr(nn.Module, "register_forward_pre_hook", _tracking_register, raising=False)

    with pytest.raises(RuntimeError, match="hook start failed while materializing"):
        capture.cache_inputs(
            layers=[model.hooked_layer],
            calibration_data=[{"input_ids": torch.randint(0, 10, (1, 4))}],
            use_cache=False,
        )

    assert removed["count"] == 1
    assert len(model.hooked_layer._forward_pre_hooks) == 0


def test_capture_runs_cleanup_when_progress_bar_close_raises(monkeypatch):
    """A failing progress-bar close must not skip the capture cleanup.

    Closing the bar is cosmetic; releasing the generate-hook state and detaching
    the pre-hook are not. When the two were sequential statements in the same
    `finally`, a raising `close()` escaped before either obligation ran, leaving
    the model mutated and the pre-hook live for the rest of the quantization run.
    """

    import types

    from gptqmodel.looper import stage_inputs_capture as capture_mod
    from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
    from gptqmodel.models.base import BaseQModel

    events = []

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.hooked_layer = nn.Linear(4, 4)
            self.hooked_layer.full_name = "test_layer"
            self.config = types.SimpleNamespace(model_type="test")

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            return None

    model = _Model()

    class _FakeGPTQModel:
        capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
        capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
        finalize_input_capture_example = BaseQModel.finalize_input_capture_example
        move_input_capture_example = BaseQModel.move_input_capture_example
        prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs
        run_input_capture = BaseQModel.run_input_capture
        _sanitize_input_ids_for_embeddings = BaseQModel._sanitize_input_ids_for_embeddings

        def __init__(self):
            self.quantize_config = types.SimpleNamespace(
                device=torch.device("cpu"),
                calibration_data_device=torch.device("cpu"),
                compute_device_filter=None,
            )
            self.model = model
            self.ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
            self.ATTENTION_MASKS_DTYPE = torch.long
            self.INPUT_EMBEDDING_EXTRA_ARGS = None
            self.quant_region_timer = None

        def shell_module_materialize(self, target_submodule, device, **kwargs):
            del device, kwargs
            return target_submodule

        def get_base_modules(self, model):
            del model
            return []

        @classmethod
        def get_modules_with_direct_meta_tensors(cls, model):
            del model
            return []

        def pre_quantize_generate_hook_start(self):
            events.append("hook_start")

        def pre_quantize_generate_hook_end(self):
            events.append("hook_end")

    class _Looper:
        def __init__(self, gptq_model):
            self.gptq_model = gptq_model

        def _batch_row_count(self, batch_inputs):
            del batch_inputs
            return 1

    class _NullCM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(capture_mod, "DEVICE_THREAD_POOL", types.SimpleNamespace(read_lock=lambda d: _NullCM()))
    monkeypatch.setattr(capture_mod, "ctx", lambda *cms: _NullCM())
    monkeypatch.setattr(capture_mod, "device_ctx", lambda device: _NullCM())
    monkeypatch.setattr(capture_mod, "await_torch_linear_prewarm", lambda *a, **k: None)

    class _Logger:
        def info(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

        def warn(self, *a, **k):
            pass

        def pb(self, *a, **k):
            class _Bar:
                def manual(self):
                    return self

                def set(self, **kw):
                    return self

                def title(self, *a):
                    return self

                def subtitle(self, *a):
                    return self

                def draw(self):
                    return self

                def close(self):
                    raise RuntimeError("progress bar close failed")

            return _Bar()

    capture = StageInputsCapture(_Looper(_FakeGPTQModel()), logger=_Logger())
    removed = {"count": 0}
    real_register = nn.Module.register_forward_pre_hook

    class _TrackedHandle:
        def __init__(self, inner):
            self._inner = inner

        def remove(self):
            removed["count"] += 1
            return self._inner.remove()

    def _tracking_register(self, hook, with_kwargs=False):
        return _TrackedHandle(real_register(self, hook, with_kwargs=with_kwargs))

    monkeypatch.setattr(nn.Module, "register_forward_pre_hook", _tracking_register, raising=False)

    with pytest.raises(RuntimeError, match="progress bar close failed"):
        capture.cache_inputs(
            layers=[model.hooked_layer],
            calibration_data=[{"input_ids": torch.randint(0, 10, (1, 4))}],
            use_cache=False,
        )

    assert events == ["hook_start", "hook_end"]
    assert removed["count"] == 1
    assert len(model.hooked_layer._forward_pre_hooks) == 0


@pytest.mark.parametrize("forward_raises", [True, False])
def test_capture_preserves_original_error_when_progress_bar_close_raises(monkeypatch, forward_raises):
    """A renderer failure follows the same precedence as a release failure.

    Closing the capture progress bar is cosmetic, so its error must not replace
    the calibration error that is already unwinding. It must still propagate
    when nothing else failed, so a broken renderer cannot pass silently. In both
    cases the model must be released and the pre-hook detached, because the
    renderer failure must not be able to skip the cleanup.
    """

    import types

    from gptqmodel.looper import stage_inputs_capture as capture_mod
    from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
    from gptqmodel.models.base import BaseQModel

    events = []

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.hooked_layer = nn.Linear(4, 4)
            self.hooked_layer.full_name = "test_layer"
            self.config = types.SimpleNamespace(model_type="test")

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            if forward_raises:
                raise RuntimeError("calibration forward failed")
            return None

    model = _Model()

    class _FakeGPTQModel:
        capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
        capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
        finalize_input_capture_example = BaseQModel.finalize_input_capture_example
        move_input_capture_example = BaseQModel.move_input_capture_example
        prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs
        run_input_capture = BaseQModel.run_input_capture
        _sanitize_input_ids_for_embeddings = BaseQModel._sanitize_input_ids_for_embeddings

        def __init__(self):
            self.quantize_config = types.SimpleNamespace(
                device=torch.device("cpu"),
                calibration_data_device=torch.device("cpu"),
                compute_device_filter=None,
            )
            self.model = model
            self.ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
            self.ATTENTION_MASKS_DTYPE = torch.long
            self.INPUT_EMBEDDING_EXTRA_ARGS = None
            self.quant_region_timer = None

        def shell_module_materialize(self, target_submodule, device, **kwargs):
            del device, kwargs
            return target_submodule

        def get_base_modules(self, model):
            del model
            return []

        @classmethod
        def get_modules_with_direct_meta_tensors(cls, model):
            del model
            return []

        def pre_quantize_generate_hook_start(self):
            events.append("hook_start")

        def pre_quantize_generate_hook_end(self):
            events.append("hook_end")

    class _Looper:
        def __init__(self, gptq_model):
            self.gptq_model = gptq_model

        def _batch_row_count(self, batch_inputs):
            del batch_inputs
            return 1

    class _NullCM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(capture_mod, "DEVICE_THREAD_POOL", types.SimpleNamespace(read_lock=lambda d: _NullCM()))
    monkeypatch.setattr(capture_mod, "ctx", lambda *cms: _NullCM())
    monkeypatch.setattr(capture_mod, "device_ctx", lambda device: _NullCM())
    monkeypatch.setattr(capture_mod, "await_torch_linear_prewarm", lambda *a, **k: None)

    warnings = []

    class _Logger:
        def info(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

        def warn(self, *a, **k):
            warnings.append((a, k))

        def pb(self, *a, **k):
            class _Bar:
                def manual(self):
                    return self

                def set(self, **kw):
                    return self

                def title(self, *a):
                    return self

                def subtitle(self, *a):
                    return self

                def draw(self):
                    return self

                def close(self):
                    raise RuntimeError("progress bar close failed")

            return _Bar()

    capture = StageInputsCapture(_Looper(_FakeGPTQModel()), logger=_Logger())

    if forward_raises:
        with pytest.raises(RuntimeError, match="calibration forward failed"):
            capture.cache_inputs(
                layers=[model.hooked_layer],
                calibration_data=[{"input_ids": torch.randint(0, 10, (1, 4))}],
                use_cache=False,
            )
        assert len(warnings) == 1
        assert "progress-bar close failed" in warnings[0][0][0]
    else:
        with pytest.raises(RuntimeError, match="progress bar close failed"):
            capture.cache_inputs(
                layers=[model.hooked_layer],
                calibration_data=[{"input_ids": torch.randint(0, 10, (1, 4))}],
                use_cache=False,
            )
        assert warnings == []

    assert events == ["hook_start", "hook_end"]
    assert len(model.hooked_layer._forward_pre_hooks) == 0


def test_release_module_skips_never_materialized_meta_module():
    """A checkpoint-only `meta` module has nothing to release.

    `move_to()` intentionally raises for a meta module with no Accelerate
    offload hook, so a release that touched it would turn a configuration that
    never materialized the module into a cleanup error. It must be skipped and
    stay registered under its original name.
    """

    definition = _definition(offload_to_disk=False)
    core_model = definition.model.model
    core_model.vision_tower.to_empty(device=torch.device("meta"))
    vision_tower = core_model.vision_tower
    assert vision_tower.weight.is_meta

    definition.pre_quantize_generate_hook_end()

    assert core_model.vision_tower is vision_tower
    assert core_model.vision_tower.weight.is_meta
    assert not hasattr(core_model.vision_tower, "_hf_hook")


@pytest.mark.parametrize("offload_to_disk", [True, False])
def test_release_module_handles_partially_materialized_module(tmp_path, offload_to_disk):
    """A partially materialized module still has real tensors to release.

    A start hook that raises partway through a device move (for example an OOM
    while materializing the vision tower) leaves already-moved leaves on the
    forward device next to untouched `meta` ones. `is_meta_module()` sees the
    `meta` leaves and the module looks exactly like one that never left the
    checkpoint, but skipping it would leak the materialized tensors. The release
    must move those tensors and leave the `meta` leaves alone.

    Both offload configurations must release the mixed module: the disk branch
    cannot do it, because `offload_to_disk()` early-returns on any module that
    still has a `meta` leaf.

    `_apply()` is recorded rather than inferred from the device, so this also
    fails on a CPU-only host where the materialized leaf is already on CPU and a
    plain device assertion would pass for an early-return implementation.
    """

    definition = _definition(offload_to_disk=offload_to_disk)
    definition.quantize_config.offload_to_disk_path = str(tmp_path)
    core_model = definition.model.model
    loaded_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    applied = []

    class _PartiallyMaterialized(nn.Module):
        def __init__(self):
            super().__init__()
            self.loaded = nn.Linear(_HIDDEN, _HIDDEN, bias=False, device=loaded_device)
            self.pending = nn.Linear(_HIDDEN, _HIDDEN, bias=False, device="meta")

        def _apply(self, fn, recurse=True):
            applied.append(True)
            return super()._apply(fn, recurse=recurse)

    mixed = _PartiallyMaterialized()
    core_model.vision_tower = mixed
    assert mixed.loaded.weight.device == loaded_device
    assert mixed.pending.weight.is_meta

    definition.pre_quantize_generate_hook_end()

    # The mixed-leaf release ran instead of being skipped as "never materialized".
    assert applied == [True], "partially materialized module must be released, not skipped"
    assert core_model.vision_tower is mixed
    assert mixed.pending.weight.is_meta
    assert not mixed.loaded.weight.is_meta
    assert mixed.loaded.weight.device == torch.device("cpu")


@pytest.mark.parametrize("offload_to_disk", [True, False])
def test_capture_releases_partial_module_after_hook_start_raises(monkeypatch, tmp_path, offload_to_disk):
    """A mid-move allocator failure must not leave materialized tensors behind.

    This is the reachable version of the partial-materialization state through
    the real cleanup path: `pre_quantize_generate_hook_start` copies the vision
    tower to the forward device, the allocator fails after the first leaf moved,
    and the capture unwinds through its `finally`. Both offload configurations
    must release the already-moved leaf, and neither may touch the `meta` one.

    The check is `_apply()`-based, not device-based, so it fails on a CPU-only
    host too, where the "moved" leaf is already on CPU.
    """

    import types

    from gptqmodel.looper import stage_inputs_capture as capture_mod
    from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
    from gptqmodel.models.base import BaseQModel
    from gptqmodel.models.definitions.ernie4_5_vl_moe import Ernie4_5_VLMoeQModel

    class _Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = nn.Module()
            self.language_model.embed_tokens = nn.Embedding(_HIDDEN, _HIDDEN)
            self.language_model.rotary_emb = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
            self.vision_tower = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
            self.resampler_model = nn.Linear(_HIDDEN, _HIDDEN, bias=False)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Core()
            self.hooked_layer = nn.Linear(4, 4)
            self.hooked_layer.full_name = "model.language_model.layers.0"
            self.config = types.SimpleNamespace(model_type="ernie4_5_vl_moe")

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            return None

    applied = []

    class _PartiallyMaterialized(nn.Module):
        def __init__(self):
            super().__init__()
            self.loaded = nn.Linear(_HIDDEN, _HIDDEN, bias=False)
            self.pending = nn.Linear(_HIDDEN, _HIDDEN, bias=False, device="meta")

        def _apply(self, fn, recurse=True):
            applied.append(True)
            return super()._apply(fn, recurse=recurse)

    model = _Model()

    class _FakeGPTQModel(Ernie4_5_VLMoeQModel):
        capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
        capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
        finalize_input_capture_example = BaseQModel.finalize_input_capture_example
        move_input_capture_example = BaseQModel.move_input_capture_example
        prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs
        run_input_capture = BaseQModel.run_input_capture
        _sanitize_input_ids_for_embeddings = BaseQModel._sanitize_input_ids_for_embeddings

        def __init__(self):
            nn.Module.__init__(self)
            self.model = model
            self.quantized = True
            self.load_quantized_model = False
            self.quantize_config = types.SimpleNamespace(
                device=torch.device("cpu"),
                calibration_data_device=torch.device("cpu"),
                compute_device_filter=None,
                offload_to_disk=offload_to_disk,
                offload_to_disk_path=str(tmp_path),
            )
            self.ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
            self.ATTENTION_MASKS_DTYPE = torch.long
            self.INPUT_EMBEDDING_EXTRA_ARGS = None
            self.quant_region_timer = None

        def shell_module_materialize(self, target_submodule, device, **kwargs):
            del device, kwargs
            return target_submodule

        def get_base_modules(self, model):
            del model
            return []

        @classmethod
        def get_modules_with_direct_meta_tensors(cls, model):
            del model
            return []

        def pre_quantize_generate_hook_start(self):
            # Emulate the mid-move allocator failure: the vision tower is
            # already in its half-moved state when the error escapes.
            model.model.vision_tower = _PartiallyMaterialized()
            raise torch.OutOfMemoryError("simulated allocator failure moving the vision tower")

    class _Looper:
        def __init__(self, gptq_model):
            self.gptq_model = gptq_model

        def _batch_row_count(self, batch_inputs):
            del batch_inputs
            return 1

    class _NullCM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(capture_mod, "DEVICE_THREAD_POOL", types.SimpleNamespace(read_lock=lambda d: _NullCM()))
    monkeypatch.setattr(capture_mod, "ctx", lambda *cms: _NullCM())
    monkeypatch.setattr(capture_mod, "device_ctx", lambda device: _NullCM())
    monkeypatch.setattr(capture_mod, "await_torch_linear_prewarm", lambda *a, **k: None)

    class _Logger:
        def info(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

        def warn(self, *a, **k):
            pass

        def pb(self, *a, **k):
            class _Bar:
                def manual(self):
                    return self

                def set(self, **kw):
                    return self

                def title(self, *a):
                    return self

                def subtitle(self, *a):
                    return self

                def draw(self):
                    return self

                def close(self):
                    pass

            return _Bar()

    capture = StageInputsCapture(_Looper(_FakeGPTQModel()), logger=_Logger())

    with pytest.raises(torch.OutOfMemoryError, match="simulated allocator failure"):
        capture.cache_inputs(
            layers=[model.hooked_layer],
            calibration_data=[{"input_ids": torch.randint(0, 10, (1, 4))}],
            use_cache=False,
        )

    vision_tower = model.model.vision_tower
    assert applied == [True], "the already-moved leaf must be released, not skipped"
    assert vision_tower.pending.weight.is_meta
    assert not vision_tower.loaded.weight.is_meta
    assert vision_tower.loaded.weight.device == torch.device("cpu")


@pytest.mark.parametrize("forward_raises", [True, False])
def test_capture_preserves_original_error_when_release_also_raises(monkeypatch, forward_raises):
    """Cleanup errors must not replace the error the capture was already raising.

    When the calibration forward fails *and* the model-specific release fails,
    the capture error is the actionable one; the release failure is a symptom
    of the same broken window. With no capture error in flight, a failing
    release must still propagate so it cannot pass silently.
    """

    import types

    from gptqmodel.looper import stage_inputs_capture as capture_mod
    from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
    from gptqmodel.models.base import BaseQModel

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.hooked_layer = nn.Linear(4, 4)
            self.hooked_layer.full_name = "test_layer"
            self.config = types.SimpleNamespace(model_type="test")

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            if forward_raises:
                raise RuntimeError("calibration forward failed")
            return None

    model = _Model()

    class _FakeGPTQModel:
        capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
        capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
        finalize_input_capture_example = BaseQModel.finalize_input_capture_example
        move_input_capture_example = BaseQModel.move_input_capture_example
        prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs
        run_input_capture = BaseQModel.run_input_capture
        _sanitize_input_ids_for_embeddings = BaseQModel._sanitize_input_ids_for_embeddings

        def __init__(self):
            self.quantize_config = types.SimpleNamespace(
                device=torch.device("cpu"),
                calibration_data_device=torch.device("cpu"),
                compute_device_filter=None,
            )
            self.model = model
            self.ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
            self.ATTENTION_MASKS_DTYPE = torch.long
            self.INPUT_EMBEDDING_EXTRA_ARGS = None
            self.quant_region_timer = None

        def shell_module_materialize(self, target_submodule, device, **kwargs):
            del device, kwargs
            return target_submodule

        def get_base_modules(self, model):
            del model
            return []

        @classmethod
        def get_modules_with_direct_meta_tensors(cls, model):
            del model
            return []

        def pre_quantize_generate_hook_start(self):
            pass

        def pre_quantize_generate_hook_end(self):
            raise RuntimeError("release failed")

    class _Looper:
        def __init__(self, gptq_model):
            self.gptq_model = gptq_model

        def _batch_row_count(self, batch_inputs):
            del batch_inputs
            return 1

    class _NullCM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(capture_mod, "DEVICE_THREAD_POOL", types.SimpleNamespace(read_lock=lambda d: _NullCM()))
    monkeypatch.setattr(capture_mod, "ctx", lambda *cms: _NullCM())
    monkeypatch.setattr(capture_mod, "device_ctx", lambda device: _NullCM())
    monkeypatch.setattr(capture_mod, "await_torch_linear_prewarm", lambda *a, **k: None)

    warnings = []

    class _Logger:
        def info(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

        def warn(self, *a, **k):
            warnings.append((a, k))

        def pb(self, *a, **k):
            class _Bar:
                def manual(self):
                    return self

                def set(self, **kw):
                    return self

                def title(self, *a):
                    return self

                def subtitle(self, *a):
                    return self

                def draw(self):
                    return self

                def close(self):
                    pass

            return _Bar()

    capture = StageInputsCapture(_Looper(_FakeGPTQModel()), logger=_Logger())

    if forward_raises:
        with pytest.raises(RuntimeError, match="calibration forward failed"):
            capture.cache_inputs(
                layers=[model.hooked_layer],
                calibration_data=[{"input_ids": torch.randint(0, 10, (1, 4))}],
                use_cache=False,
            )
        assert len(warnings) == 1
        assert "release failed" in warnings[0][0][0]
    else:
        with pytest.raises(RuntimeError, match="release failed"):
            capture.cache_inputs(
                layers=[model.hooked_layer],
                calibration_data=[{"input_ids": torch.randint(0, 10, (1, 4))}],
                use_cache=False,
            )
        assert warnings == []

    assert len(model.hooked_layer._forward_pre_hooks) == 0


@pytest.mark.parametrize("caller_in_except", [True, False])
def test_capture_cleanup_scope_is_this_call_not_the_thread(monkeypatch, caller_in_except):
    """The capture/release precedence must come from *this* call's exit state.

    `sys.exc_info()` is thread state. Read inside the `finally`, it can report
    an exception that an enclosing caller handler is processing -- a retry
    loop, `requantize()`, or an application `except` -- while this capture
    completed successfully. Sampling it there would mark a successful capture
    as failed and swallow the release error, so the precedence flag has to be
    driven by this `try` statement's own `else`/`finally` path.
    """

    import types

    from gptqmodel.looper import stage_inputs_capture as capture_mod
    from gptqmodel.looper.stage_inputs_capture import StageInputsCapture
    from gptqmodel.models.base import BaseQModel

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.hooked_layer = nn.Linear(4, 4)
            self.hooked_layer.full_name = "test_layer"
            self.config = types.SimpleNamespace(model_type="test")

        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            return None

    model = _Model()

    class _FakeGPTQModel:
        capture_first_layer_positional_inputs = BaseQModel.capture_first_layer_positional_inputs
        capture_first_layer_input_kwargs = BaseQModel.capture_first_layer_input_kwargs
        finalize_input_capture_example = BaseQModel.finalize_input_capture_example
        move_input_capture_example = BaseQModel.move_input_capture_example
        prepare_layer_replay_kwargs = BaseQModel.prepare_layer_replay_kwargs
        run_input_capture = BaseQModel.run_input_capture
        _sanitize_input_ids_for_embeddings = BaseQModel._sanitize_input_ids_for_embeddings

        def __init__(self):
            self.quantize_config = types.SimpleNamespace(
                device=torch.device("cpu"),
                calibration_data_device=torch.device("cpu"),
                compute_device_filter=None,
            )
            self.model = model
            self.ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
            self.ATTENTION_MASKS_DTYPE = torch.long
            self.INPUT_EMBEDDING_EXTRA_ARGS = None
            self.quant_region_timer = None

        def shell_module_materialize(self, target_submodule, device, **kwargs):
            del device, kwargs
            return target_submodule

        def get_base_modules(self, model):
            del model
            return []

        @classmethod
        def get_modules_with_direct_meta_tensors(cls, model):
            del model
            return []

        def pre_quantize_generate_hook_start(self):
            pass

        def pre_quantize_generate_hook_end(self):
            raise RuntimeError("release failed")

    class _Looper:
        def __init__(self, gptq_model):
            self.gptq_model = gptq_model

        def _batch_row_count(self, batch_inputs):
            del batch_inputs
            return 1

    class _NullCM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(capture_mod, "DEVICE_THREAD_POOL", types.SimpleNamespace(read_lock=lambda d: _NullCM()))
    monkeypatch.setattr(capture_mod, "ctx", lambda *cms: _NullCM())
    monkeypatch.setattr(capture_mod, "device_ctx", lambda device: _NullCM())
    monkeypatch.setattr(capture_mod, "await_torch_linear_prewarm", lambda *a, **k: None)

    warnings = []

    class _Logger:
        def info(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

        def warn(self, *a, **k):
            warnings.append((a, k))

        def pb(self, *a, **k):
            class _Bar:
                def manual(self):
                    return self

                def set(self, **kw):
                    return self

                def title(self, *a):
                    return self

                def subtitle(self, *a):
                    return self

                def draw(self):
                    return self

                def close(self):
                    pass

            return _Bar()

    capture = StageInputsCapture(_Looper(_FakeGPTQModel()), logger=_Logger())
    call = {
        "layers": [model.hooked_layer],
        "calibration_data": [{"input_ids": torch.randint(0, 10, (1, 4))}],
        "use_cache": False,
    }

    if caller_in_except:
        # The capture itself succeeds; the only live exception belongs to an
        # enclosing caller handler and must not change this call's contract.
        try:
            raise ValueError("caller handler, unrelated to this capture")
        except ValueError:
            with pytest.raises(RuntimeError, match="release failed"):
                capture.cache_inputs(**call)
        assert warnings == []
    else:
        with pytest.raises(RuntimeError, match="release failed"):
            capture.cache_inputs(**call)
        assert warnings == []

    assert len(model.hooked_layer._forward_pre_hooks) == 0
