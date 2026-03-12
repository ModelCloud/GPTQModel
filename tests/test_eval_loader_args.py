# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.utils.eval import EVAL


def test_eval_string_model_load_filters_eval_only_keys(monkeypatch):
    captured = {}

    def _fake_load(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("sentinel-load-stop")

    monkeypatch.setattr(GPTQModel, "load", _fake_load)

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
        GPTQModel.eval(
            model_or_id_or_path="/tmp/current-model",
            framework=EVAL.LM_EVAL,
            tasks=[EVAL.LM_EVAL.ARC_CHALLENGE],
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
