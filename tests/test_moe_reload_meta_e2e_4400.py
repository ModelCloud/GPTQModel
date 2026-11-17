# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4400 item 2: the meta expert shell reloads bitwise-identically."""

import pytest
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from gptqmodel import GPTQModel
from gptqmodel.models import loader as loader_module
from gptqmodel.quantization.config import RTNConfig


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the GPTQ reload kernel needs an accelerator",
)


def _quantized_checkpoint(tmp_path):
    dense = tmp_path / "dense"
    config = OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts_per_tok=2,
        num_local_experts=4,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    OlmoeForCausalLM(config).save_pretrained(dense, safe_serialization=True)

    model = GPTQModel.load(
        str(dense), quantize_config=RTNConfig(bits=4, group_size=32, device="cpu")
    )
    model.quantize(calibration=None, backend="auto")
    checkpoint = tmp_path / "quantized"
    model.save(str(checkpoint))
    return checkpoint


def _load(checkpoint):
    return GPTQModel.load(str(checkpoint), device="cuda:0", backend="gptq_torch")


def _capture(model):
    ids = torch.tensor([[1, 5, 9, 17, 33]], device=next(model.model.parameters()).device)
    with torch.no_grad():
        logits = model.model(input_ids=ids).logits.float().cpu()
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.model.state_dict().items()
    }
    return logits, state


def test_meta_expert_shell_reloads_bitwise_identically(tmp_path, monkeypatch):
    checkpoint = _quantized_checkpoint(tmp_path)

    fixed_logits, fixed_state = _capture(_load(checkpoint))

    # The pre-fix path let Defuser materialize one dense bf16 Linear per expert.
    monkeypatch.setattr(loader_module, "_meta_fused_expert_params", lambda model: 0)
    old_logits, old_state = _capture(_load(checkpoint))

    assert torch.equal(fixed_logits, old_logits)
    assert fixed_state.keys() == old_state.keys()
    for name in fixed_state:
        assert torch.equal(fixed_state[name], old_state[name]), name
