# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import torch

from gptqmodel.models.base import BaseQModel


class _RecorderModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.last_kwargs = None

    def generate(self, *args, **kwargs):
        self.last_kwargs = kwargs
        return kwargs["attention_mask"]


def test_base_qmodel_generate_normalizes_causal_attention_mask():
    qmodel = BaseQModel.__new__(BaseQModel)
    qmodel.model = _RecorderModel()
    qmodel.tokenizer = None

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tril(torch.ones((1, 1, 3, 3), dtype=torch.long))

    normalized = qmodel.generate(input_ids=input_ids, attention_mask=attention_mask)

    assert normalized.shape == (1, 3)
    assert normalized.dtype is torch.bool
    assert normalized.tolist() == [[True, True, True]]
    assert qmodel.model.last_kwargs["attention_mask"].shape == (1, 3)
