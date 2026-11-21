# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""#5504: ``batch_size="auto"`` must be resolved by every calibration entry point.

``quantize()`` was the only caller that turned the sentinel into an int, so
``requantize(norm=...)`` and ``adapter.generate()`` (post-quantization EoRA)
handed the raw ``"auto"`` string to ``prepare_calibration_dataset`` and crashed
in ``range(0, len(sorted_dataset), batch_size)`` with a ``TypeError`` that never
named ``batch_size``. The resolution now lives in one helper that all three
entry points call.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, NormConfig
from gptqmodel.models.base import BaseQModel
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.calibration import AUTO_BATCH_SIZE


VOCAB, HIDDEN, INTER, LAYERS = 128, 128, 256, 3
SAMPLES = [{"input_ids": [(i * 7 + j * 3) % VOCAB for j in range(24)]} for i in range(8)]


def _stub_model(**overrides):
    """A ``BaseQModel`` whose device sizing is stubbed out."""

    model = BaseQModel.__new__(BaseQModel)
    torch.nn.Module.__init__(model)
    model.quantize_config = SimpleNamespace(device=overrides.get("device", "cpu"))
    model.model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=HIDDEN, num_hidden_layers=LAYERS)
    )
    model._auto_calibration_batch_size = lambda calibration: overrides.get("picked", 4)
    return model


def test_sentinel_resolves_to_the_auto_choice():
    model = _stub_model(picked=8)
    calibration = CalibrationConfig(dataset=SAMPLES, batch_size=AUTO_BATCH_SIZE, min_length=1)

    assert model._calibration_batch_size(calibration) == 8


@pytest.mark.parametrize("batch_size", [1, 2, 16])
def test_concrete_batches_pass_through(batch_size):
    model = _stub_model(picked=8)
    calibration = CalibrationConfig(dataset=SAMPLES, batch_size=batch_size, min_length=1)

    assert model._calibration_batch_size(calibration) == batch_size


def test_resolution_tolerates_a_model_without_a_quantize_config():
    """``adapter.generate()`` binds before the run has a quantize config."""

    model = _stub_model()
    model.quantize_config = None
    model._auto_calibration_batch_size = BaseQModel._auto_calibration_batch_size.__get__(model)
    calibration = CalibrationConfig(dataset=SAMPLES, batch_size=AUTO_BATCH_SIZE, min_length=1)

    assert model._calibration_batch_size(calibration) == 1


def _build_quantized(tmp_path: Path, calibration: CalibrationConfig | None = None) -> Path:
    source = tmp_path / "src"
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    Qwen2ForCausalLM(config).to(torch.float16).save_pretrained(source)

    model = GPTQModel.load(
        str(source),
        QuantizeConfig(bits=4, group_size=128),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
    )
    if calibration is None:
        calibration = CalibrationConfig(dataset=SAMPLES, batch_size=1, min_length=1)
    model.quantize(calibration, backend=BACKEND.TORCH)
    checkpoint = tmp_path / "quantized"
    model.save(str(checkpoint))
    return checkpoint


def test_quantize_auto_batch_preserves_one_shot_calibration(tmp_path):
    calibration = CalibrationConfig(
        dataset=(row for row in SAMPLES[:4]), batch_size=AUTO_BATCH_SIZE, min_length=1
    )

    checkpoint = _build_quantized(tmp_path, calibration)

    reloaded = GPTQModel.load(
        str(checkpoint), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16
    )
    logits = reloaded.model(input_ids=torch.tensor([SAMPLES[0]["input_ids"]])).logits
    assert torch.isfinite(logits).all()


def test_requantize_norm_accepts_the_auto_sentinel(tmp_path):
    """Without the fix this raises ``TypeError: 'str' object cannot be ...``."""

    checkpoint = _build_quantized(tmp_path)
    model = GPTQModel.load(
        str(checkpoint), backend=BACKEND.TORCH, device="cpu", dtype=torch.float16
    )
    calibration = CalibrationConfig(dataset=SAMPLES, batch_size=AUTO_BATCH_SIZE, min_length=1)

    model.requantize(
        calibration,
        norm=NormConfig(teacher=str(tmp_path / "src"), steps=1),
    )
