# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
from ..local import LocalModelLoader
from . import LlamaQModel


class DreamQModel(LlamaQModel):
    fusable = None  # No qualified executor for this definition.

    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.dream.configuration_dream.DreamConfig",
        model_class="gptqmodel.models.ports.dream.modeling_dream.DreamModel",
        # Dream-v0 ships DreamTokenizer only as checkpoint Python.
        remote_tokenizer_model_types=("dream",),
    )

    # TODO: fix dream attention mask tensor size/dtype issues due to batching/padding
    support_batch_quantize = False
