# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from gptqmodel.models.definitions.ovis import OvisQModel


class _DummyInnerModel:
    def __init__(self):
        self.calls = []

    def generate(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        return "ok"


def test_ovis_qmodel_generate_accepts_input_ids_keyword():
    qmodel = OvisQModel.__new__(OvisQModel)
    qmodel.model = _DummyInnerModel()
    qmodel.device = torch.device("cpu")

    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1]])
    pixel_values = [torch.zeros(1, 3, 4, 4)]

    output = qmodel.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        max_new_tokens=8,
    )

    assert output == "ok"
    assert len(qmodel.model.calls) == 1

    forwarded_inputs, forwarded_kwargs = qmodel.model.calls[0]
    assert forwarded_inputs is input_ids
    assert "input_ids" not in forwarded_kwargs
    assert forwarded_kwargs["attention_mask"] is attention_mask
    assert forwarded_kwargs["pixel_values"] is pixel_values
    assert forwarded_kwargs["max_new_tokens"] == 8
