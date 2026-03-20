# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import inference_speed as inference_speed_module
import torch
from inference_speed import InferenceSpeed


class _FakeProgress:
    def __init__(self, iterable):
        self._iterable = iterable

    def title(self, _text):
        return self

    def __iter__(self):
        return iter(self._iterable)


class _FakeLogger:
    def pb(self, iterable):
        return _FakeProgress(iterable)


class _FakeBatch(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    @classmethod
    def from_pretrained(cls, _model_path):
        return cls()

    def __call__(self, prompts, **_kwargs):
        batch = len(prompts)
        return _FakeBatch({"input_ids": torch.zeros((batch, 3), dtype=torch.long)})


class _FakeModel:
    device = "cuda:0"

    @classmethod
    def from_quantized(cls, *_args, **_kwargs):
        return cls()

    def generate(self, input_ids, max_new_tokens, **_kwargs):
        batch, prompt_len = input_ids.shape
        return torch.zeros((batch, prompt_len + max_new_tokens), dtype=torch.long)


def test_inference_speed_excludes_warmup_from_asserted_throughput(monkeypatch):
    class _Harness(InferenceSpeed):
        NUM_RUNS = 2
        MAX_NEW_TOKENS = 10
        PROMPTS = ["a", "b"]

    timestamps = iter([
        0.0, 10.0,   # warmup: intentionally slow
        100.0, 101.0,
        200.0, 201.0,
    ])

    monkeypatch.setattr(inference_speed_module, "logger", _FakeLogger())
    monkeypatch.setattr(inference_speed_module, "GPTQModel", _FakeModel)
    monkeypatch.setattr(inference_speed_module, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setattr(inference_speed_module, "torch_empty_cache", lambda: None)
    monkeypatch.setattr(inference_speed_module.time, "time", lambda: next(timestamps))

    measured_tps = _Harness().inference(
        model_path="unused",
        backend="fake-backend",
        tokens_per_second=20.0,
        warmup_runs=1,
        device="cuda",
    )

    assert measured_tps == 20.0


def test_inference_speed_pins_bare_cuda_to_current_device(monkeypatch):
    class _Harness(InferenceSpeed):
        NUM_RUNS = 1
        MAX_NEW_TOKENS = 1
        PROMPTS = ["a"]

    captured = {}

    class _CapturingModel(_FakeModel):
        @classmethod
        def from_quantized(cls, model_path, backend, device):
            captured["device"] = device
            return cls()

    timestamps = iter([0.0, 1.0])

    monkeypatch.setattr(inference_speed_module, "logger", _FakeLogger())
    monkeypatch.setattr(inference_speed_module, "GPTQModel", _CapturingModel)
    monkeypatch.setattr(inference_speed_module, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setattr(inference_speed_module, "torch_empty_cache", lambda: None)
    monkeypatch.setattr(inference_speed_module.time, "time", lambda: next(timestamps))
    monkeypatch.setattr(inference_speed_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(inference_speed_module.torch.cuda, "current_device", lambda: 3)

    _Harness().inference(
        model_path="unused",
        backend="fake-backend",
        tokens_per_second=1.0,
        warmup_runs=0,
        device="cuda",
    )

    assert captured["device"] == "cuda:3"
