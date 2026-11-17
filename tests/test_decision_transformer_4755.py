# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Regression coverage for ModelCloud/GPTQModel#4755.

Decision Transformer checkpoints ship no tokenizer and their forward consumes
named tensors (``states``/``actions``/``returns_to_go``/``timesteps``) instead of
``input_ids``. This test builds a tiny random-weight ``DecisionTransformerModel``
on disk, quantizes it with tensor-only calibration samples, and checks that the
quantized forward still runs and stays finite. It is CPU-only so it does not
compete for the accelerator pool.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import DecisionTransformerConfig, DecisionTransformerModel

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.models.auto import MODEL_MAP, SUPPORTED_MODELS
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.calibration import prepare_calibration_dataset


STATE_DIM = 8
ACT_DIM = 3
SEQ_LEN = 8


def _tiny_config() -> DecisionTransformerConfig:
    return DecisionTransformerConfig(
        n_layer=2,
        n_head=2,
        n_embd=32,
        hidden_size=32,
        n_inner=64,
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        max_length=16,
        max_ep_len=16,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
    )


def _build_checkpoint(path):
    """Save a tiny random-weight DT checkpoint; DT never writes a tokenizer."""

    torch.manual_seed(0)
    model = DecisionTransformerModel(_tiny_config()).eval()
    model.save_pretrained(path)
    assert (path / "config.json").is_file()
    assert (path / "model.safetensors").is_file()
    assert not list(path.glob("tokenizer*"))
    return path


def _sample(seed: int, batch: int = 1, seq: int = SEQ_LEN):
    """One calibration batch of the named tensors ``DecisionTransformerModel`` expects."""

    generator = torch.Generator().manual_seed(seed)
    return {
        "states": torch.randn(batch, seq, STATE_DIM, generator=generator),
        "actions": torch.randn(batch, seq, ACT_DIM, generator=generator),
        "rewards": torch.randn(batch, seq, 1, generator=generator),
        "returns_to_go": torch.randn(batch, seq, 1, generator=generator),
        "timesteps": torch.arange(seq).reshape(1, seq).repeat(batch, 1),
        "attention_mask": torch.ones(batch, seq, dtype=torch.long),
    }


def _stub_qmodel(calibration_inputs: str):
    return SimpleNamespace(
        tokenizer=None,
        support_batch_quantize=True,
        calibration_inputs=calibration_inputs,
        model=SimpleNamespace(config=SimpleNamespace(n_positions=16)),
    )


def test_decision_transformer_definition_is_registered():
    assert "decision_transformer" in SUPPORTED_MODELS
    assert "decision_transformer" in MODEL_MAP

    from gptqmodel.models.definitions.decision_transformer import DecisionTransformerQModel

    assert MODEL_MAP["decision_transformer"] is DecisionTransformerQModel
    assert DecisionTransformerQModel.extract_layers_node() == ["encoder.h"]
    # No tokenizer ships with these checkpoints and calibration is tensor kwargs.
    assert DecisionTransformerQModel.require_load_tokenizer is False
    assert DecisionTransformerQModel.calibration_inputs == "tensor_kwargs"


def test_tensor_kwargs_calibration_passes_rows_through():
    sample = _sample(seed=1)
    batches = prepare_calibration_dataset(
        _stub_qmodel("tensor_kwargs"),
        calibration_dataset=[sample],
        calibration_data_min_length=1,
    )

    assert len(batches) == 1
    assert set(batches[0]) == set(sample)
    for key, value in batches[0].items():
        assert torch.is_tensor(value), key
        assert value.shape == sample[key].shape, key


def test_tensor_kwargs_calibration_is_gated_by_the_definition():
    """A plain text model must keep rejecting tensor-only rows (no generic dict path)."""

    with pytest.raises(ValueError, match="unsupported calibration example structure"):
        prepare_calibration_dataset(
            _stub_qmodel("token_ids"),
            calibration_dataset=[_sample(seed=1)],
            calibration_data_min_length=1,
        )


def test_tensor_kwargs_calibration_rejects_non_tensor_values():
    sample = _sample(seed=1)
    sample["timesteps"] = [[0, 1, 2]]

    with pytest.raises(ValueError, match="tensor"):
        prepare_calibration_dataset(
            _stub_qmodel("tensor_kwargs"),
            calibration_dataset=[sample],
            calibration_data_min_length=1,
        )


def test_tiny_decision_transformer_quantizes_with_tensor_calibration(tmp_path):
    checkpoint = _build_checkpoint(tmp_path / "checkpoint")

    reference = DecisionTransformerModel.from_pretrained(checkpoint).eval()
    samples = [_sample(seed) for seed in range(3)]
    with torch.inference_mode():
        expected = reference(**samples[0], return_dict=True)

    model = GPTQModel.load(
        str(checkpoint),
        quantize_config=QuantizeConfig(bits=4, group_size=32, device="cpu"),
        backend=BACKEND.TORCH,
        dtype=torch.float32,
    )
    # A Decision Transformer checkpoint has no tokenizer to load.
    assert model.tokenizer is None

    model.quantize(
        CalibrationConfig(dataset=samples, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )

    with torch.inference_mode():
        actual = model.model(**samples[0], return_dict=True)

    for name in ("action_preds", "state_preds", "return_preds"):
        tensor = getattr(actual, name)
        assert torch.isfinite(tensor).all(), f"`{name}` is not finite after quantization"
        assert tensor.shape == getattr(expected, name).shape

    # Quantization must actually replace the Conv1D projections, not silently
    # fall back to the dense model.
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear

    packed = [
        name
        for name, module in model.model.named_modules()
        if isinstance(module, BaseQuantLinear)
    ]
    assert packed, "no module was quantized"
    assert all(".attn." in name or ".mlp." in name for name in packed)

    # The second tokenizer call site is the quantized-load path: a saved DT must
    # reload without a tokenizer and still run. TorchLinear is fp16/bf16 only, so
    # reload in bfloat16 and cast the float calibration sample to match.
    quantized_dir = tmp_path / "quantized"
    model.save_quantized(str(quantized_dir))
    reloaded = GPTQModel.load(
        str(quantized_dir),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.bfloat16,
    )
    assert reloaded.tokenizer is None

    bf16_sample = {
        key: value.to(torch.bfloat16) if value.is_floating_point() else value
        for key, value in samples[0].items()
    }
    with torch.inference_mode():
        reloaded_output = reloaded.model(**bf16_sample, return_dict=True)
    assert torch.isfinite(reloaded_output.action_preds).all()
