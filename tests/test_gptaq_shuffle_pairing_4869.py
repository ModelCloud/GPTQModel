# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #4869: GPTAQ must pair native and quantized batches.

With ``CalibrationConfig(sort="shuffle")`` every processor used to prepare its
own copy of the calibration data, so ``NativeProcessor`` and ``GPTQProcessor``
captured two different batch orders. GPTAQ pairs those captures by batch index,
which mixed unrelated sequences: variable-length rows crashed with a size
mismatch, and equal-length rows silently produced a non-zero ``dXXT`` at layer
0, where the native and quantized inputs are identical and the correct value is
exactly zero.
"""

from __future__ import annotations

import torch
import transformers

from gptqmodel import CalibrationConfig, GPTQModel
from gptqmodel.looper import gptq_processor, native_processor
from gptqmodel.quantization.config import GPTAQConfig, GPTQConfig
from tests.tiny_remote_models import build_wordlevel_tokenizer


ROW_COUNT = 13
EQUAL_LENGTH = 128
LAYER0_Q_PROJ = "model.layers.0.self_attn.q_proj"


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
            max_position_embeddings=256,
            pad_token_id=0,
        )
    ).to(torch.float16).save_pretrained(model_dir)
    build_wordlevel_tokenizer().save_pretrained(model_dir)
    return model_dir


def _rows(lengths):
    generator = torch.Generator().manual_seed(4869)
    return [
        {"input_ids": torch.randint(4, 128, (length,), generator=generator).tolist()}
        for length in lengths
    ]


def _batch_order(dataset):
    """Return the ``(first token, row length)`` of every prepared batch."""

    order = []
    for row in dataset:
        input_ids = row["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            order.append((int(input_ids.reshape(-1)[0]), int(input_ids.shape[-1])))
        else:
            order.append((int(input_ids[0]), len(input_ids)))
    return order


def _quantize_with_recorders(model_dir, rows, monkeypatch):
    captured = {}
    tasks = []

    class RecordingNative(native_processor.NativeProcessor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["native"] = _batch_order(self.calibration_dataset)

    class RecordingGPTQ(gptq_processor.GPTQProcessor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["gptq"] = _batch_order(self.calibration_dataset)

    class RecordingGPTAQ(gptq_processor.GPTAQ):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._record_name = getattr(self._named_module, "full_name", self.name)
            self._record_dxxT_abs_max = []
            tasks.append(self)

        def add_batch(self, inp, out, batch_index=None):
            super().add_batch(inp, out, batch_index=batch_index)
            if self.dXXT is not None:
                self._record_dxxT_abs_max.append(float(self.dXXT.abs().max()))

    monkeypatch.setattr(native_processor, "NativeProcessor", RecordingNative)
    monkeypatch.setattr(gptq_processor, "GPTQProcessor", RecordingGPTQ)
    monkeypatch.setattr(gptq_processor, "GPTAQ", RecordingGPTAQ)

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=GPTQConfig(bits=4, group_size=32, gptaq=GPTAQConfig()),
        dtype=torch.float16,
    )
    model.quantize(CalibrationConfig(dataset=rows, sort="shuffle", batch_size=1))
    return model, captured, tasks


def test_shuffle_prepares_one_batch_order_for_every_processor(tmp_path, monkeypatch):
    model_dir = _tiny_model(tmp_path / "tiny")
    rows = _rows([EQUAL_LENGTH] * ROW_COUNT)

    _model, captured, tasks = _quantize_with_recorders(model_dir, rows, monkeypatch)

    assert captured["native"], "native processor prepared no batches"
    assert captured["gptq"], "gptq processor prepared no batches"
    assert captured["gptq"] == captured["native"], (
        "GPTAQ pairs native and quantized inputs by batch index, so both "
        "processors must replay the same batch order"
    )


def test_shuffle_keeps_layer0_gptaq_dxxt_zero(tmp_path, monkeypatch):
    """At layer 0 both captures are identical, so ``dXXT`` must be exactly 0."""

    model_dir = _tiny_model(tmp_path / "tiny")
    rows = _rows([EQUAL_LENGTH] * ROW_COUNT)

    _model, _captured, tasks = _quantize_with_recorders(model_dir, rows, monkeypatch)

    layer0 = next(task for task in tasks if task._record_name == LAYER0_Q_PROJ)
    assert layer0._record_dxxT_abs_max, "layer-0 GPTAQ consumed no batches"
    assert layer0._record_dxxT_abs_max[-1] == 0.0, (
        "layer-0 dXXT must be zero: the native and quantized captures are "
        f"identical, got {layer0._record_dxxT_abs_max[-1]!r}"
    )


def test_shuffle_does_not_crash_on_variable_length_rows(tmp_path, monkeypatch):
    model_dir = _tiny_model(tmp_path / "tiny")
    lengths = [64 + 8 * index for index in range(ROW_COUNT)]
    rows = _rows(lengths)

    _model, captured, _tasks = _quantize_with_recorders(model_dir, rows, monkeypatch)

    assert captured["gptq"] == captured["native"]
