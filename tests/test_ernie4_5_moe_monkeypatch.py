# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""The ERNIE 4.5 MoE forward patch must only replace the old remote forward.

`Ernie4_5_MoeQModel.monkey_patch` installs a copy of the official remote-code
forward, which builds its attention mask with `self._update_causal_mask`.
transformers 5 replaced that helper with `create_causal_mask`, and the official
`baidu/ERNIE-4.5-*` checkpoints ship no `auto_map`, so the native class is
loaded. Patching it raised `AttributeError: 'Ernie4_5_MoeModel' object has no
attribute '_update_causal_mask'` at the first calibration forward.
"""

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.ernie4_5_moe import Ernie4_5_MoeQModel


class _NativeErnie(nn.Module):
    """transformers 5 layout: no `_update_causal_mask` helper."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden_states):
        return self.linear(hidden_states)


class _RemoteErnie(nn.Module):
    """Old remote-code layout: owns the helper the patch calls."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def _update_causal_mask(self, *args, **kwargs):
        del args, kwargs
        return None

    def forward(self, hidden_states):
        return self.linear(hidden_states)


def _bare_qmodel(inner_model):
    qmodel = Ernie4_5_MoeQModel.__new__(Ernie4_5_MoeQModel)
    nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = nn.Module()
    qmodel.model.model = inner_model
    qmodel.__dict__["load_quantized_model"] = False
    return qmodel


def test_native_transformers_class_is_not_patched():
    inner = _NativeErnie()
    original_forward = type(inner).forward
    qmodel = _bare_qmodel(inner)

    qmodel.monkey_patch()

    assert type(inner).forward is original_forward
    # The native forward still runs; the removed helper is never touched.
    assert inner(torch.zeros(1, 4)).shape == (1, 4)


def test_remote_code_class_still_gets_the_compatibility_forward():
    inner = _RemoteErnie()
    original_forward = type(inner).forward
    qmodel = _bare_qmodel(inner)

    qmodel.monkey_patch()

    assert type(inner).forward is not original_forward


def test_native_ernie4_5_moe_model_has_no_legacy_mask_helper():
    """Pin the reason for the guard against the installed transformers build."""

    modeling = pytest.importorskip("transformers.models.ernie4_5_moe.modeling_ernie4_5_moe")
    assert hasattr(modeling, "Ernie4_5_MoeModel")
    assert not hasattr(modeling.Ernie4_5_MoeModel, "_update_causal_mask")


def test_native_ernie4_5_moe_calibration_forward_does_not_raise():
    """The regression: the patched forward used to raise on the native class."""

    modeling = pytest.importorskip("transformers.models.ernie4_5_moe.modeling_ernie4_5_moe")
    config = modeling.Ernie4_5_MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        moe_num_experts=4,
        moe_k=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = modeling.Ernie4_5_MoeForCausalLM(config).eval()
    qmodel = _bare_qmodel(model.model)

    qmodel.monkey_patch()

    with torch.no_grad():
        output = model(input_ids=torch.randint(0, config.vocab_size, (1, 8)))
    assert output.logits.shape == (1, 8, config.vocab_size)
