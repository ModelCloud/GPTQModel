# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from ...utils.logger import setup_logger
from ..fusable import FusableConfig, LlamaFusableConfig
from . import LlamaQModel


log = setup_logger()


class Ernie4_5QModel(LlamaQModel):
    fusable = FusableConfig("ernie4_5").merge(LlamaFusableConfig())

    pass
