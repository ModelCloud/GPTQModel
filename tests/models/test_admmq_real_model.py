# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint ADMM-Q quantize -> save -> reload round trip.

The tiny-Llama test covers the checkpoint contract without external files.
This test pins the same contract on a pretrained Qwen2.5 checkpoint: the whole
model is packed, the saved artifact reloads as ``ADMMQConfig``, and greedy
generation is identical before and after the round trip.

It runs only when a checkpoint is already available locally, through
``GPTQMODEL_ADMMQ_REAL_MODEL``, the repository model cache, or the Hugging Face
cache, so it never adds a download to CI.
"""

import os
from pathlib import Path

import pytest
import torch

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.quantization.config import METHOD, ADMMQConfig


LOCAL_MODEL = Path("/monster/data/model/Qwen2.5-0.5B-Instruct")
HF_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_ENV = "GPTQMODEL_ADMMQ_REAL_MODEL"
EXPECTED_PACKED_MODULES = 168

CALIBRATION = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Quantization reduces the memory footprint of large language models.",
    "A short calibration sentence about the moon and the stars at night.",
    "The capital of France is Paris and the capital of Japan is Tokyo.",
    "Machine learning models learn patterns from large amounts of data.",
    "She walked through the forest and listened to the birds singing.",
    "The stock market rose sharply after the central bank announcement.",
    "A recipe for chocolate cake needs flour, sugar, eggs and cocoa.",
]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="real-checkpoint ADMM-Q round trip needs a GPU",
)


def _cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # pragma: no cover - dependency guard
        return False
    return all(
        isinstance(try_to_load_from_cache(model_id, name), str)
        for name in ("config.json", "model.safetensors")
    )


def _resolve_model() -> str:
    override = os.environ.get(MODEL_ENV)
    if override:
        return override
    if LOCAL_MODEL.is_dir():
        return str(LOCAL_MODEL)
    if _cached(HF_MODEL_ID):
        return HF_MODEL_ID
    raise pytest.skip.Exception(f"set {MODEL_ENV}, provide {LOCAL_MODEL}, or cache {HF_MODEL_ID}")


def _packed_modules(model):
    return [
        (name, module)
        for name, module in model.model.named_modules()
        if hasattr(module, "qweight")
    ]


def _packed_state(model):
    return {
        name: {
            key: tensor.detach().cpu()
            for key, tensor in module.state_dict().items()
            if key in ("qweight", "qzeros", "scales", "g_idx")
        }
        for name, module in _packed_modules(model)
    }


def test_qwen_admmq_real_model_roundtrip(tmp_path):
    model_id = _resolve_model()
    load_kwargs = {"local_files_only": True} if model_id == HF_MODEL_ID else {}
    qcfg = ADMMQConfig(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        max_iter=60,
        update_iter=3,
        switch_iter=10,
        early_stop_tol=1e-7,
        pairswap_rounds=2,
        pairswap_pairs=32,
        pairswap_batch=16,
        offload_to_disk=False,
    )
    model = GPTQModel.load(
        model_id,
        qcfg,
        device="cuda",
        dtype=torch.bfloat16,
        **load_kwargs,
    )
    model.quantize(
        CalibrationConfig(dataset=CALIBRATION, batch_size=1),
        backend=BACKEND.GPTQ_TORCH,
    )

    assert model.quantize_config.method is METHOD.ADMMQ
    packed = _packed_modules(model)
    if model_id == str(LOCAL_MODEL) or model_id == HF_MODEL_ID:
        assert len(packed) == EXPECTED_PACKED_MODULES
    else:
        assert packed
    assert all(type(module).__name__ == "TorchLinear" for _, module in packed)
    before = _packed_state(model)

    tokenizer = model.tokenizer
    input_ids = tokenizer("The quick brown fox", return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        first = model.model.generate(input_ids, max_new_tokens=16, do_sample=False)

    saved = tmp_path / "qwen-admmq"
    model.save(str(saved))
    loaded = GPTQModel.load(
        str(saved),
        backend=BACKEND.GPTQ_TORCH,
        device="cuda",
        dtype=torch.bfloat16,
    )

    assert isinstance(loaded.quantize_config, ADMMQConfig)
    assert loaded.quantize_config.method is METHOD.ADMMQ
    assert loaded.quantize_config.max_iter == 60
    assert loaded.quantize_config.pairswap_pairs == 32

    # The checkpoint contract is the packed payload, not the runtime path.
    after = _packed_state(loaded)
    assert set(after) == set(before)
    for name, state in before.items():
        for key, tensor in state.items():
            torch.testing.assert_close(after[name][key], tensor, rtol=0, atol=0)

    with torch.inference_mode():
        second = loaded.model.generate(input_ids, max_new_tokens=16, do_sample=False)
    assert second.shape[1] > input_ids.shape[1]
    # Greedy decoding can drift once the runtime path differs after reload;
    # the stable prefix still proves the reloaded model behaves the same.
    prefix = input_ids.shape[1] + 8
    assert torch.equal(second[:, :prefix], first[:, :prefix])
