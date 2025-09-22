# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from transformers import AutoModel

from . import LlamaQModel


class DreamQModel(LlamaQModel):
    loader = AutoModel
    # TODO: fix dream attention mask tensor size/dtype issues due to batching/padding
    support_batch_quantize = False
