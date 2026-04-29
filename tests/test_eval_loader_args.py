# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

import pytest

from gptqmodel import BACKEND
from tests import eval as eval_module


def test_eval_string_model_load_filters_eval_only_keys(monkeypatch):
    captured = {}

    class FakeGPTQModelEngine:
        def __init__(self, **kwargs):
            captured["engine_kwargs"] = kwargs

        def build(self, model_config):
            captured["model_kwargs"] = dict(model_config.kwargs["model_kwargs"])
            raise RuntimeError("sentinel-load-stop")

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to_dict(self):
            return dict(self.kwargs)

    fake_evalution = SimpleNamespace(
        GPTQModel=FakeGPTQModelEngine,
        Model=FakeModel,
    )

    model_args = {
        "backend": BACKEND.EXLLAMA_V3,
        "device": "cuda:0",
        "gptqmodel": True,
        "model_id_or_path": "/tmp/stale-model",
        "pretrained": "/tmp/stale-pretrained",
        "tokenizer": object(),
        "trust_remote_code": False,
    }

    with pytest.raises(RuntimeError, match="sentinel-load-stop"):
        eval_module._build_evalution_runtime(
            evalution=fake_evalution,
            model_or_id_or_path="/tmp/current-model",
            llm_backend="gptqmodel",
            backend=BACKEND.EXLLAMA_V3,
            batch_size=1,
            trust_remote_code=True,
            model_args=model_args,
            tokenizer=None,
        )

    assert captured["engine_kwargs"]["backend"] == BACKEND.EXLLAMA_V3.value
    assert captured["engine_kwargs"]["device"] == "cuda:0"
    assert captured["engine_kwargs"]["trust_remote_code"] is True
    assert captured["model_kwargs"] == {}
    for key in ("gptqmodel", "pretrained", "tokenizer"):
        assert key not in captured["model_kwargs"]


def test_build_evalution_runtime_supports_vllm_engine_options():
    captured = {}

    class FakeVLLM:
        def __init__(self, **kwargs):
            captured["engine_kwargs"] = kwargs

        def build(self, model_config):
            captured["model_config"] = model_config
            return "session"

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to_dict(self):
            return dict(self.kwargs)

    fake_evalution = SimpleNamespace(
        VLLM=FakeVLLM,
        Model=FakeModel,
    )

    engine, model_config, session = eval_module._build_evalution_runtime(
        evalution=fake_evalution,
        model_or_id_or_path="/tmp/model",
        llm_backend="vllm",
        backend=BACKEND.AUTO,
        batch_size=1,
        trust_remote_code=True,
        model_args={
            "dtype": "float16",
            "gpu_memory_utilization": 0.8,
            "tensor_parallel_size": "2",
            "quantization": "gptq",
            "tokenizer_mode": "auto",
            "max_model_len": "4096",
            "foo": "bar",
        },
        tokenizer=None,
    )

    assert session == "session"
    assert engine is not None
    assert model_config.kwargs["path"] == "/tmp/model"
    assert model_config.kwargs["model_kwargs"] == {"foo": "bar"}
    assert captured["engine_kwargs"]["dtype"] == "float16"
    assert captured["engine_kwargs"]["gpu_memory_utilization"] == 0.8
    assert captured["engine_kwargs"]["tensor_parallel_size"] == 2
    assert captured["engine_kwargs"]["quantization"] == "gptq"
    assert captured["engine_kwargs"]["max_model_len"] == 4096


def test_build_evalution_runtime_supports_sglang_engine_options():
    captured = {}

    class FakeSGLang:
        def __init__(self, **kwargs):
            captured["engine_kwargs"] = kwargs

        def build(self, model_config):
            captured["model_config"] = model_config
            return "session"

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to_dict(self):
            return dict(self.kwargs)

    fake_evalution = SimpleNamespace(
        SGLang=FakeSGLang,
        Model=FakeModel,
    )

    engine, model_config, session = eval_module._build_evalution_runtime(
        evalution=fake_evalution,
        model_or_id_or_path="/tmp/model",
        llm_backend="sglang",
        backend=BACKEND.AUTO,
        batch_size=2,
        trust_remote_code=True,
        model_args={
            "dtype": "float16",
            "device": "cuda",
            "gpu_memory_utilization": "0.75",
            "tensor_parallel_size": "2",
            "quantization": "gptq",
            "tokenizer_mode": "auto",
            "max_model_len": "8192",
            "attention_backend": "flashinfer",
            "sampling_backend": "pytorch",
            "max_running_requests": "16",
            "max_total_tokens": "32768",
            "random_seed": "123",
            "sampling_params": {"top_p": 0.9},
            "foo": "bar",
        },
        tokenizer=None,
    )

    assert session == "session"
    assert engine is not None
    assert model_config.kwargs["path"] == "/tmp/model"
    assert model_config.kwargs["model_kwargs"] == {"foo": "bar", "random_seed": '123'}
    assert captured["engine_kwargs"]["dtype"] == "float16"
    assert captured["engine_kwargs"]["device"] == "cuda"
    assert captured["engine_kwargs"]["batch_size"] == 2
    assert captured["engine_kwargs"]["trust_remote_code"] is True
    assert captured["engine_kwargs"]["quantization"] == "gptq"
    assert captured["engine_kwargs"]["context_length"] == 8192
    assert captured["engine_kwargs"]["tp_size"] == 2
    assert captured["engine_kwargs"]["mem_fraction_static"] == 0.75
    assert captured["engine_kwargs"]["attention_backend"] == "flashinfer"
    assert captured["engine_kwargs"]["sampling_backend"] == "pytorch"
    assert captured["engine_kwargs"]["max_running_requests"] == 16
    assert captured["engine_kwargs"]["max_total_tokens"] == 32768
    assert captured["engine_kwargs"]["sampling_params"] == {"top_p": 0.9}


def test_build_evalution_runtime_supports_gptqmodel_seed():
    captured = {}

    class FakeGPTQModel:
        def __init__(self, **kwargs):
            captured["engine_kwargs"] = kwargs

        def build(self, model_config):
            captured["model_config"] = model_config
            return "session"

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to_dict(self):
            return dict(self.kwargs)

    fake_evalution = SimpleNamespace(
        GPTQModel=FakeGPTQModel,
        Model=FakeModel,
    )

    engine, model_config, session = eval_module._build_evalution_runtime(
        evalution=fake_evalution,
        model_or_id_or_path="/tmp/model",
        llm_backend="gptqmodel",
        backend=BACKEND.AUTO,
        batch_size=4,
        trust_remote_code=True,
        model_args={
            "dtype": "float16",
            "seed": 898,
            "device": "cuda:0",
            "foo": "bar",
        },
        tokenizer=None,
    )

    assert session == "session"
    assert engine is not None
    assert model_config.kwargs["path"] == "/tmp/model"
    assert model_config.kwargs["model_kwargs"] == {"foo": "bar"}
    assert captured["engine_kwargs"]["dtype"] == "float16"
    assert captured["engine_kwargs"]["device"] == "cuda:0"
    assert captured["engine_kwargs"]["batch_size"] == 4
    assert captured["engine_kwargs"]["seed"] == 898


def test_build_evalution_runtime_drops_removed_gptqmodel_path_for_strict_engine_signature():
    captured = {}

    class FakeGPTQModel:
        def __init__(
            self,
            *,
            dtype=None,
            attn_implementation=None,
            device=None,
            device_map=None,
            seed=None,
            batch_size=None,
            trust_remote_code=None,
            padding_side=None,
            backend=None,
        ):
            captured["engine_kwargs"] = {
                "dtype": dtype,
                "attn_implementation": attn_implementation,
                "device": device,
                "device_map": device_map,
                "seed": seed,
                "batch_size": batch_size,
                "trust_remote_code": trust_remote_code,
                "padding_side": padding_side,
                "backend": backend,
            }

        def build(self, model_config):
            captured["model_config"] = model_config
            return "session"

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to_dict(self):
            return dict(self.kwargs)

    fake_evalution = SimpleNamespace(
        GPTQModel=FakeGPTQModel,
        Model=FakeModel,
    )

    engine, model_config, session = eval_module._build_evalution_runtime(
        evalution=fake_evalution,
        model_or_id_or_path="/tmp/model",
        llm_backend="gptqmodel",
        backend=BACKEND.AUTO,
        batch_size=2,
        trust_remote_code=False,
        model_args={
            "dtype": "float16",
            "seed": 7,
            "device": "cuda:0",
            "foo": "bar",
        },
        tokenizer=None,
    )

    assert session == "session"
    assert engine is not None
    assert model_config.kwargs["path"] == "/tmp/model"
    assert model_config.kwargs["model_kwargs"] == {"foo": "bar"}
    assert captured["engine_kwargs"] == {
        "dtype": "float16",
        "attn_implementation": None,
        "device": "cuda:0",
        "device_map": None,
        "seed": 7,
        "batch_size": 2,
        "trust_remote_code": False,
        "padding_side": "left",
        "backend": "auto",
    }
