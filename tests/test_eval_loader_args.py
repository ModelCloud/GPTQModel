# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from tests import eval as eval_module
from tests.eval import evaluate


def test_eval_string_model_load_filters_eval_only_keys(monkeypatch):
    captured = {}

    def _fake_load(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("sentinel-load-stop")

    monkeypatch.setattr(GPTQModel, "load", _fake_load)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            padding_side="left",
            pad_token_id=0,
            pad_token="</s>",
            eos_token="</s>",
            unk_token="<unk>",
        ),
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
        evaluate(
            model_or_id_or_path="/tmp/current-model",
            tasks=["arc_challenge"],
            batch_size=1,
            backend=BACKEND.EXLLAMA_V3,
            llm_backend="gptqmodel",
            model_args=model_args,
            trust_remote_code=True,
        )

    assert captured["model_id_or_path"] == "/tmp/current-model"
    assert captured["backend"] == BACKEND.EXLLAMA_V3
    assert captured["device"] == "cuda:0"
    assert captured["trust_remote_code"] is True
    for key in ("gptqmodel", "pretrained", "tokenizer"):
        assert key not in captured


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
