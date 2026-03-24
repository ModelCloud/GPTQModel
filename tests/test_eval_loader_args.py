# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.utils.eval import evaluate


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
