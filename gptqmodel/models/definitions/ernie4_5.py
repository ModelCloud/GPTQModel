# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ...utils.logger import setup_logger
from ..fusable import LlamaFusableConfig
from . import LlamaQModel


log = setup_logger()


class Ernie4_5QModel(LlamaQModel):
    fusable = LlamaFusableConfig().derive("ernie4_5", parent_types={
        "qkv": (
            "transformers.models.ernie4_5.modeling_ernie4_5.Ernie4_5Attention",
        ),
        "gate_up": (
            "transformers.models.ernie4_5.modeling_ernie4_5.Ernie4_5MLP",
        ),
    })

    pass
