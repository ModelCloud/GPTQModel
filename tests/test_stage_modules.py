import types

import torch

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.stage_inputs_capture import StageInputsCapture


class _DummyQModel:
    def __init__(self):
        self.support_batch_quantize = False
        self.quantize_config = types.SimpleNamespace(device=None, vram_strategy="exclusive")
        self.layer_callback = None


def _make_looper():
    processors = [types.SimpleNamespace(layer_count=0, pb=None)]
    return ModuleLooper(model=_DummyQModel(), processors=processors)


def test_cache_inputs_delegates_to_stage_capture(monkeypatch):
    looper = _make_looper()
    sentinel = object()
    captured = {}

    class FakeStage:
        def __init__(self, looper_arg, logger):
            captured["looper"] = looper_arg
            captured["logger"] = logger

        def cache_inputs(self, **kwargs):
            captured["kwargs"] = kwargs
            return sentinel

    monkeypatch.setattr(
        "gptqmodel.looper.module_looper.StageInputsCapture",
        FakeStage,
    )

    layers = [object()]
    data = [{"hidden_states": torch.zeros(1, 2, 2)}]
    result = looper.cache_inputs(layers=layers, calibration_data=data, use_cache=False)

    assert result is sentinel
    assert captured["looper"] is looper
    assert captured["kwargs"]["layers"] == layers
    assert captured["kwargs"]["calibration_data"] is data


class _TinyLayer(torch.nn.Module):
    def forward(self, hidden_states, attention_mask=None, position_ids=None, **kwargs):
        return hidden_states


class _TinyModel(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.visual_tokenizer = types.SimpleNamespace(dtype=torch.float32)

    def forward(self, *, hidden_states, attention_mask=None, position_ids=None, use_cache=False, **kwargs):
        return self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


class _TinyGptqModel:
    ATTENTION_MASKS_REQUIRED_FOR_INPUT = False
    ATTENTION_MASKS_DTYPE = torch.long
    INPUT_EMBEDDING_EXTRA_ARGS = {}

    def __init__(self):
        self.layer = _TinyLayer()
        self.model = _TinyModel(self.layer)
        self.quantize_config = types.SimpleNamespace(device=torch.device("cpu"))
        self._hook_started = False
        self._hook_finished = False

    def shell_module_materialize(self, target_submodule, device):
        target_submodule.to(device)
        return target_submodule

    def get_base_modules(self, model):
        return []

    def pre_quantize_generate_hook_start(self):
        self._hook_started = True

    def pre_quantize_generate_hook_end(self):
        self._hook_finished = True


class _TinyLooper:
    def __init__(self, gptq_model):
        self.gptq_model = gptq_model

    def _batch_row_count(self, batch_inputs):
        if not batch_inputs:
            return 0
        tensor = batch_inputs[0]
        return int(tensor.shape[0]) if tensor.ndim > 0 else int(tensor.numel())


def test_stage_inputs_capture_collects_real_inputs():
    gptq_model = _TinyGptqModel()
    looper = _TinyLooper(gptq_model)
    stage = StageInputsCapture(looper, logger=None)

    hidden = torch.ones(1, 2, 3)
    attention = torch.ones(1, 2)
    position_ids = torch.arange(2).unsqueeze(0)
    extra = torch.tensor([5.0])

    dataset = [
        {
            "hidden_states": hidden.clone(),
            "attention_mask": attention.clone(),
            "position_ids": position_ids.clone(),
            "extra": extra.clone(),
        }
    ]

    cache = stage.cache_inputs(layers=[gptq_model.layer], calibration_data=dataset, use_cache=False)

    assert len(cache.layer_inputs) == 1
    assert torch.equal(cache.layer_inputs[0][0], hidden)
    assert torch.equal(cache.attention_masks[0], attention.long())
    assert torch.equal(cache.position_ids[0], position_ids)
    assert torch.equal(cache.layer_input_kwargs[0]["extra"], extra.unsqueeze(0))
    assert gptq_model._hook_started is True
    assert gptq_model._hook_finished is True
