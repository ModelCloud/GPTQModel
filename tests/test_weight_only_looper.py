# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from torch import nn

import gptqmodel.looper.weight_only_looper as weight_only_looper_module
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.quantization.config import RTNConfig


class _FakeProgress:
    def __init__(self):
        self.current_iter_step = 0
        self.titles = []
        self.subtitles = []
        self.draw_calls = []
        self.closed = False

    def manual(self):
        return self

    def set(self, **_kwargs):
        return self

    def title(self, value):
        self.titles.append(value)
        return self

    def subtitle(self, value):
        self.subtitles.append(value)
        return self

    def draw(self, force: bool = False):
        self.draw_calls.append((self.current_iter_step, force))
        return self

    def close(self):
        self.closed = True


class _FakeLogger:
    def __init__(self):
        self.progress = _FakeProgress()
        self.iterable = None

    def pb(self, iterable, *, output_interval=None):
        del output_interval
        self.iterable = list(iterable)
        return self.progress

    def info(self, *_args, **_kwargs):
        return None


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            use_cache=True,
            model_type="tiny_weight_only_progress",
            tie_word_embeddings=False,
        )
        self.layers = nn.ModuleList([_TinyLayer(), _TinyLayer()])


class _FakeQModel:
    def __init__(self, qcfg):
        self.model = _TinyModel()
        self.quantize_config = qcfg
        self.layer_modules_strict = True
        self.lm_head = "lm_head"
        self.tokenizer = None
        self.quant_log = None

    def extract_layers_node(self):
        return ["layers"]

    def get_modules_with_direct_meta_tensors(self, _model):
        return []

    def simple_layer_modules(self, **_kwargs):
        return [["linear"]]

    def pre_quantize(self, module):
        return module

    def post_quantize(self, module):
        return module


class _FakeProcessor:
    def __init__(self, qcfg):
        self.qcfg = qcfg
        self.log = []
        self.layer_count = None
        self.pb = None
        self.memory_calls = []
        self.quantized = []
        self.finalized = []
        self.finalize_called = False

    def name(self):
        return "fake_weight_only"

    def collect_memory_info(self, layer_index):
        self.memory_calls.append(layer_index)

    def quantize_module(self, module):
        self.quantized.append(module.full_name)
        return self.qcfg

    def submodule_finalize(self, module, _model, *, qcfg=None):
        self.finalized.append((module.full_name, qcfg))

    def finalize(self, *, model):
        del model
        self.finalize_called = True


def test_weight_only_looper_reports_logbar_progress(monkeypatch):
    qcfg = RTNConfig(bits=4, group_size=4, offload_to_disk=False, device="cpu")
    qcfg.lm_head = False
    fake_logger = _FakeLogger()
    processor = _FakeProcessor(qcfg)
    model = _FakeQModel(qcfg)

    monkeypatch.setattr(weight_only_looper_module, "log", fake_logger)
    monkeypatch.setattr(
        weight_only_looper_module,
        "get_layers_with_prefixes",
        lambda _model, _nodes: (list(model.model.layers), ["layers.0", "layers.1"]),
    )

    looper = WeightOnlyLooper(model=model, processor=processor)
    total_log = looper.loop()

    assert total_log == {"fake_weight_only": []}
    assert model.model.config.use_cache is True
    assert processor.layer_count == 2
    assert processor.pb is fake_logger.progress
    assert processor.memory_calls == [0, 1]
    assert processor.quantized == ["layers.0.linear", "layers.1.linear"]
    assert processor.finalize_called is True

    assert fake_logger.iterable == [0, 1]
    assert fake_logger.progress.titles == [
        "Weight-only quantization (2 layers)",
        "Weight-only quantizing layer 0 of 1",
        "Weight-only quantizing layer 1 of 1",
    ]
    assert fake_logger.progress.subtitles == ["", ""]
    assert fake_logger.progress.draw_calls[0] == (0, False)
    assert fake_logger.progress.draw_calls[-1] == (2, False)
    assert fake_logger.progress.closed is True
