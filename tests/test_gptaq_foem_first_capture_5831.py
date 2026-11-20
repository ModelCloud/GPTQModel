# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #5831: GPTAQ/FOEM reuse the native first-layer capture.

The native and quantized processors replay the same prepared calibration
batches (#4869). The quantized pass used to run the model-level capture forward
a second time even though the native pass had just recorded identical
deterministic inputs. It now replays a private copy of that completed capture,
so the initial forward runs once per batch while the packed tensors stay
bitwise identical to the old double-capture path.
"""

from __future__ import annotations

import random

import pytest
import torch
import transformers

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.looper import gptq_processor, module_looper, native_processor
from gptqmodel.models.definitions.qwen2 import Qwen2QModel
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, GPTQConfig
from tests.tiny_remote_models import build_wordlevel_tokenizer


ROW_COUNT = 6
ROW_LENGTH = 96
CONFIGS = {
    "gptaq": GPTAQConfig(device="cpu"),
    "foem": FOEMConfig(alpha=0.25, device="cpu"),
}


def _tiny_model(model_dir):
    """Build a 2-layer Qwen2 checkpoint with a local word-level tokenizer."""

    torch.manual_seed(0)
    transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
        )
    ).to(torch.float16).save_pretrained(model_dir)
    build_wordlevel_tokenizer().save_pretrained(model_dir)
    return model_dir


def _rows():
    generator = torch.Generator().manual_seed(5831)
    return [
        {
            "input_ids": torch.randint(4, 128, (ROW_LENGTH,), generator=generator),
            "attention_mask": torch.ones(ROW_LENGTH, dtype=torch.long),
        }
        for _ in range(ROW_COUNT)
    ]


def _quantize(model_dir, rows, extra):
    """Quantize the tiny model and count model-level capture forwards."""

    random.seed(5831)
    model = GPTQModel.load(
        str(model_dir),
        quantize_config=GPTQConfig(
            bits=4,
            group_size=32,
            device="cpu",
            offload_to_disk=False,
            **extra,
        ),
    )
    capture = model.run_input_capture
    forwards = [0]

    def count(*args, **kwargs):
        forwards[0] += 1
        return capture(*args, **kwargs)

    model.run_input_capture = count
    model.quantize(CalibrationConfig(dataset=rows, batch_size=1), backend=None)
    return model, forwards[0]


def _state_dict_bytes(model):
    return {
        name: value.contiguous().reshape(-1).view(torch.uint8).clone()
        for name, value in model.model.state_dict().items()
    }


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_initial_capture_runs_once_per_batch(tmp_path, name):
    model_dir = _tiny_model(tmp_path / "tiny")
    _model, forwards = _quantize(model_dir, _rows(), {name: CONFIGS[name]})

    assert forwards == ROW_COUNT, (
        f"{name} ran {forwards} model-level capture forwards for {ROW_COUNT} "
        "batches; the native capture must be reused instead of repeated"
    )


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_reused_capture_matches_double_capture(tmp_path, name, monkeypatch):
    model_dir = _tiny_model(tmp_path / "tiny")
    rows = _rows()

    reused_model, reused_forwards = _quantize(model_dir, rows, {name: CONFIGS[name]})
    reused = _state_dict_bytes(reused_model)

    monkeypatch.setattr(
        module_looper.ModuleLooper,
        "_reusable_initial_capture",
        lambda self, processor, previous_processor: None,
    )
    double_model, double_forwards = _quantize(model_dir, rows, {name: CONFIGS[name]})
    doubled = _state_dict_bytes(double_model)

    assert reused_forwards == ROW_COUNT
    assert double_forwards == 2 * ROW_COUNT
    assert reused.keys() == doubled.keys()
    for tensor_name in reused:
        assert torch.equal(reused[tensor_name], doubled[tensor_name]), (
            f"{name} reuse changed packed tensor {tensor_name}"
        )


def test_reused_cache_is_a_private_copy(tmp_path, monkeypatch):
    model_dir = _tiny_model(tmp_path / "tiny")
    rows = _rows()
    caches = {}

    def record(label, original):
        def receive(processor, cache):
            original(processor, cache)
            inner = processor.inputs_cache.unwrap()
            # Snapshot the handoff state: replay replaces ``layer_inputs`` as
            # the pass advances.
            caches[label] = (inner, [list(batch) for batch in inner.layer_inputs])

        return receive

    monkeypatch.setattr(
        native_processor.NativeProcessor,
        "receive_input_cache",
        record("native", native_processor.NativeProcessor.receive_input_cache),
    )
    monkeypatch.setattr(
        gptq_processor.GPTQProcessor,
        "receive_input_cache",
        record("gptq", gptq_processor.GPTQProcessor.receive_input_cache),
    )

    _quantize(model_dir, rows, {"gptaq": CONFIGS["gptaq"]})

    native_cache, native_inputs = caches["native"]
    gptq_cache, gptq_inputs = caches["gptq"]
    assert gptq_cache is not native_cache
    assert gptq_inputs is not native_inputs
    assert len(gptq_inputs) == len(native_inputs) == ROW_COUNT
    for native_batch, gptq_batch in zip(native_inputs, gptq_inputs):
        assert len(gptq_batch) == len(native_batch)
        for native_tensor, gptq_tensor in zip(native_batch, gptq_batch):
            assert gptq_tensor is not native_tensor, (
                "the GPTQ processor must not replay the native processor's tensors"
            )
            assert torch.equal(gptq_tensor, native_tensor)


def test_model_capture_override_disables_reuse(tmp_path, monkeypatch):
    model_dir = _tiny_model(tmp_path / "tiny")
    rows = _rows()
    original = Qwen2QModel.pre_quantize_generate_hook_start

    def hooked(self):
        return original(self)

    monkeypatch.setattr(Qwen2QModel, "pre_quantize_generate_hook_start", hooked)

    _model, forwards = _quantize(model_dir, rows, {"gptaq": CONFIGS["gptaq"]})

    assert forwards == 2 * ROW_COUNT, (
        "a model that overrides a capture hook must keep its own second capture"
    )
