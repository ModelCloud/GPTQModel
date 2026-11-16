# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""PanGu-alpha's last decoder layer must replay with ``ln_f`` + ``custom_query``.

Issue #4093 item 5: ``GPTPanguModel.forward`` applies ``ln_f`` before the last
block and passes it ``custom_query = wqe(position_ids)``, so that block's
``q_proj`` reads query embeddings. Without a replay hook GPTQModel replayed
layer 0's kwargs (``custom_query=None``) and calibrated the last layer on the
wrong input (arg0 relative difference 0.953; GPTQ worse than RTN).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from gptqmodel.models.definitions.pangu_alpha import PanguAlphaQModel


PANGU_PATH = Path("/monster/data/model/pangu_alpha_2_6B")


class _PanguTransformer(nn.Module):
    def __init__(self, layers: int = 3, dim: int = 16, vocab: int = 64):
        super().__init__()
        self.h = nn.ModuleList(
            [nn.Module() for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.wqe = nn.Embedding(vocab, dim)


class _PanguModel(nn.Module):
    def __init__(self, layers: int = 3, dim: int = 16, vocab: int = 64):
        super().__init__()
        self.transformer = _PanguTransformer(layers=layers, dim=dim, vocab=vocab)


def _instance(model: _PanguModel) -> PanguAlphaQModel:
    instance = PanguAlphaQModel.__new__(PanguAlphaQModel)
    nn.Module.__init__(instance)
    instance.model = model
    return instance


def test_last_layer_replay_applies_final_norm_and_custom_query():
    torch.manual_seed(0)
    model = _PanguModel()
    last_layer = model.transformer.h[-1]
    hidden = torch.randn(2, 7, 16)
    position_ids = torch.arange(7).unsqueeze(0).expand(2, -1)

    layer_input = [hidden.clone()]
    additional_inputs = {"position_ids": position_ids}

    returned = _instance(model).prepare_layer_replay_kwargs(
        layer=last_layer,
        layer_input=layer_input,
        additional_inputs=additional_inputs,
        target_device=torch.device("cpu"),
    )

    assert returned is additional_inputs
    # HF normalizes before the top-query block; replay must match.
    torch.testing.assert_close(
        layer_input[0], model.transformer.ln_f(hidden), rtol=0.0, atol=0.0
    )
    assert torch.is_tensor(returned["custom_query"])
    torch.testing.assert_close(
        returned["custom_query"],
        model.transformer.wqe(position_ids),
        rtol=0.0,
        atol=0.0,
    )


def test_non_last_layer_replay_is_untouched():
    torch.manual_seed(0)
    model = _PanguModel()
    hidden = torch.randn(1, 5, 16)
    layer_input = [hidden.clone()]
    additional_inputs = {"position_ids": torch.arange(5).unsqueeze(0)}

    _instance(model).prepare_layer_replay_kwargs(
        layer=model.transformer.h[0],
        layer_input=layer_input,
        additional_inputs=additional_inputs,
        target_device=torch.device("cpu"),
    )

    assert torch.equal(layer_input[0], hidden)
    assert "custom_query" not in additional_inputs


def test_last_layer_replay_rebuilds_missing_position_ids():
    torch.manual_seed(0)
    model = _PanguModel()
    hidden = torch.randn(3, 4, 16)

    additional_inputs = _instance(model).prepare_layer_replay_kwargs(
        layer=model.transformer.h[-1],
        layer_input=[hidden.clone()],
        additional_inputs={},
        target_device=torch.device("cpu"),
    )

    assert additional_inputs["custom_query"].shape == (3, 4, 16)
    assert additional_inputs["custom_query"].dtype == model.transformer.wqe.weight.dtype


@pytest.mark.skipif(
    not PANGU_PATH.is_dir(),
    reason=f"local PanGu-alpha fixture is unavailable: {PANGU_PATH}",
)
def test_pangu_last_layer_gptq_beats_rtn(tmp_path):
    """Acceptance criterion: GPTQ must be closer than RTN on the top-query layer."""

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import GPTQConfig, RTNConfig
    from tests.tiny_remote_models import build_tiny_remote_model

    path = build_tiny_remote_model(
        PANGU_PATH,
        tmp_path / "tiny",
        {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_layers": 4,
            "num_heads": 4,
            "max_position_embeddings": 64,
            "layer_norm_epsilon": 1e-5,
            "activation_function": "gelu",
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
    )
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        str(path), trust_remote_code=True
    ).eval()
    generator = torch.Generator().manual_seed(0)

    def sample():
        input_ids = torch.randint(4, 128, (1, 32), generator=generator)
        mask = torch.ones(1, 32, dtype=torch.long)
        mask[:, -int(torch.randint(1, 12, (1,), generator=generator).item()):] = 0
        return {"input_ids": input_ids, "attention_mask": mask}

    calibration = [sample() for _ in range(16)]
    position_ids = torch.arange(32).unsqueeze(0).expand(4, -1)
    custom_query = hf.transformer.wqe(position_ids)
    with torch.no_grad():
        reference = hf.transformer.h[-1].attn.q_proj(custom_query).clone()

    errors = {}
    for label, config in (
        ("RTN", RTNConfig(bits=4, group_size=32, device="cpu")),
        ("GPTQ", GPTQConfig(bits=4, group_size=32, device="cpu")),
    ):
        model = GPTQModel.load(
            str(path),
            quantize_config=config,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        model.quantize(calibration, batch_size=1)
        save_path = tmp_path / label
        model.save(str(save_path))
        del model

        reloaded = GPTQModel.load(
            str(save_path), device="cpu", trust_remote_code=True, dtype=torch.float32
        )
        with torch.no_grad():
            quantized = reloaded.model.transformer.h[-1].attn.q_proj(custom_query)
        errors[label] = float((quantized - reference).norm() / reference.norm())
        del reloaded

    assert errors["GPTQ"] < errors["RTN"], (
        f"GPTQ last-layer q_proj error {errors['GPTQ']:.4f} must beat RTN "
        f"{errors['RTN']:.4f}; the top-query layer is calibrated on the wrong input"
    )
