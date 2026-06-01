# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch import nn

import gptqmodel.looper.weight_only_looper as weight_only_looper_module
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper
from gptqmodel.quantization.config import RTNConfig, VramStrategy


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

    def next(self):
        self.current_iter_step += 1
        return self

    def close(self):
        self.closed = True


class _FakeLogger:
    def __init__(self):
        self.progress = _FakeProgress()
        self.progresses = []
        self.iterable = None
        self.iterables = []

    def pb(self, iterable, *, output_interval=None):
        del output_interval
        iterable_list = list(iterable)
        self.iterables.append(iterable_list)
        if self.iterable is None:
            self.iterable = iterable_list
        if not self.progresses:
            progress = self.progress
        else:
            progress = _FakeProgress()
        self.progresses.append(progress)
        return progress

    def info(self, *_args, **_kwargs):
        return None


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)


class _WideLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_a = nn.Linear(4, 4, bias=False)
        self.linear_b = nn.Linear(4, 4, bias=False)
        self.linear_c = nn.Linear(4, 4, bias=False)


class _ExpertMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 4, bias=False)
        self.up_proj = nn.Linear(4, 4, bias=False)


class _ExpertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)


class _SharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(4, 4, bias=False)


class _ExpertBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([_ExpertMlp(), _ExpertMlp()])
        self.shared_expert = _SharedExpert()


class _ExpertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _ExpertAttention()
        self.mlp = _ExpertBlock()

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
        self.quant_devices = []

    def name(self):
        return "fake_weight_only"

    def collect_memory_info(self, layer_index):
        self.memory_calls.append(layer_index)

    def quantize_module(self, module, *, device=None):
        self.quantized.append(module.full_name)
        self.quant_devices.append(device)
        return SimpleNamespace(device=device if device is not None else self.qcfg.device)

    def submodule_finalize(self, module, _model, *, qcfg=None):
        self.finalized.append((module.full_name, getattr(qcfg, "device", None)))

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
    assert fake_logger.iterables == [[0, 1], [0], [0]]
    assert fake_logger.progresses[1].titles == [
        "Layer 0 Submodule finalize 0/1",
        "Layer 0 Finalize 1/1",
    ]
    assert fake_logger.progresses[2].titles == [
        "Layer 1 Submodule finalize 0/1",
        "Layer 1 Finalize 1/1",
    ]
    assert fake_logger.progresses[1].closed is True
    assert fake_logger.progresses[2].closed is True


def test_weight_only_looper_quantizes_subset_across_multiple_devices(monkeypatch):
    qcfg = RTNConfig(bits=4, group_size=4, offload_to_disk=False, device="cuda:0")
    qcfg.lm_head = False
    fake_logger = _FakeLogger()
    processor = _FakeProcessor(qcfg)
    model = _FakeQModel(qcfg)
    model.model.layers = nn.ModuleList([_WideLayer()])
    model.simple_layer_modules = lambda **_kwargs: [["linear_a", "linear_b", "linear_c"]]

    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    submitted_devices = []

    def fake_submit(device, fn, *args, **kwargs):
        submitted_devices.append(device)
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future

    monkeypatch.setattr(weight_only_looper_module, "log", fake_logger)
    monkeypatch.setattr(weight_only_looper_module, "select_forward_devices", lambda _device: devices)
    monkeypatch.setattr(weight_only_looper_module, "device_ctx", lambda _device: nullcontext())
    monkeypatch.setattr(weight_only_looper_module, "move_to", lambda obj, *, device, dtype=None: obj)
    monkeypatch.setattr(weight_only_looper_module, "rehome_module_to_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(weight_only_looper_module.DEVICE_THREAD_POOL, "submit", fake_submit)
    monkeypatch.setattr(
        weight_only_looper_module,
        "get_layers_with_prefixes",
        lambda _model, _nodes: (list(model.model.layers), ["layers.0"]),
    )

    looper = WeightOnlyLooper(model=model, processor=processor)
    looper.loop()

    expected_devices = [torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:0")]
    assert submitted_devices[:3] == expected_devices
    assert submitted_devices[3:] == expected_devices
    assert processor.quantized == [
        "layers.0.linear_a",
        "layers.0.linear_b",
        "layers.0.linear_c",
    ]
    assert processor.quant_devices == expected_devices
    assert processor.finalized == [
        ("layers.0.linear_a", torch.device("cuda:0")),
        ("layers.0.linear_b", torch.device("cuda:1")),
        ("layers.0.linear_c", torch.device("cuda:0")),
    ]


def test_weight_only_looper_applies_compute_device_filter(monkeypatch):
    qcfg = RTNConfig(bits=4, group_size=4, offload_to_disk=False, device="cuda:0")
    qcfg.lm_head = False
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    captured = {}

    def compute_device_filter(candidates):
        captured["candidates"] = list(candidates)
        return [candidates[1]]

    qcfg.compute_device_filter = compute_device_filter
    monkeypatch.setattr(weight_only_looper_module, "select_forward_devices", lambda _device: devices)

    looper = WeightOnlyLooper(model=_FakeQModel(qcfg), processor=_FakeProcessor(qcfg))

    assert captured["candidates"] == devices
    assert looper._quant_devices == [torch.device("cuda:1")]


def test_weight_only_looper_respects_dense_and_moe_vram_strategy_devices(monkeypatch):
    qcfg = RTNConfig(bits=4, group_size=4, offload_to_disk=False, device="cuda:0")
    qcfg.lm_head = False
    qcfg.true_sequential = False
    qcfg.dense_vram_strategy = VramStrategy.EXCLUSIVE
    qcfg.dense_vram_strategy_devices = ["cuda:0"]
    qcfg.moe_vram_strategy = VramStrategy.BALANCED
    qcfg.moe_vram_strategy_devices = ["cuda:1", "cuda:2"]
    fake_logger = _FakeLogger()
    processor = _FakeProcessor(qcfg)
    model = _FakeQModel(qcfg)
    model.model.layers = nn.ModuleList([_ExpertLayer()])
    model.simple_layer_modules = lambda **_kwargs: [
        ["self_attn.q_proj"],
        ["mlp.experts.0.gate_proj", "mlp.experts.0.up_proj"],
        ["mlp.experts.1.gate_proj", "mlp.experts.1.up_proj"],
        ["mlp.shared_expert.down_proj"],
    ]

    devices = [torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")]

    def fake_submit(_device, fn, *args, **kwargs):
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future

    monkeypatch.setattr(weight_only_looper_module, "log", fake_logger)
    monkeypatch.setattr(weight_only_looper_module, "select_forward_devices", lambda _device: devices)
    monkeypatch.setattr(weight_only_looper_module, "device_ctx", lambda _device: nullcontext())
    monkeypatch.setattr(weight_only_looper_module, "move_to", lambda obj, *, device, dtype=None: obj)
    monkeypatch.setattr(weight_only_looper_module, "rehome_module_to_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(weight_only_looper_module.DEVICE_THREAD_POOL, "submit", fake_submit)
    monkeypatch.setattr(
        weight_only_looper_module,
        "get_layers_with_prefixes",
        lambda _model, _nodes: (list(model.model.layers), ["layers.0"]),
    )

    looper = WeightOnlyLooper(model=model, processor=processor)
    looper.loop()

    assert processor.finalized == [
        ("layers.0.self_attn.q_proj", torch.device("cuda:0")),
        ("layers.0.mlp.experts.0.gate_proj", torch.device("cuda:1")),
        ("layers.0.mlp.experts.0.up_proj", torch.device("cuda:1")),
        ("layers.0.mlp.experts.1.gate_proj", torch.device("cuda:2")),
        ("layers.0.mlp.experts.1.up_proj", torch.device("cuda:2")),
        ("layers.0.mlp.shared_expert.down_proj", torch.device("cuda:0")),
    ]
