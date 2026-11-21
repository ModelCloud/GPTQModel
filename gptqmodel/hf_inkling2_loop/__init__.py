# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Vendored from huggingface/transformers at `src/transformers/models/inkling2_loop`
# (huggingface/transformers#51050), Apache-2.0, Copyright 2026 the HuggingFace
# Team. GPT-QModel ships its own copy so Inkling 2 Mini quantizes on every
# supported Transformers release instead of waiting for a modeling-code release.
# Only import paths and 5.24-only helpers were adapted; the module tree, tensor
# names, and forward math are unchanged.

"""Inkling 2 Mini (`inkling2_loop`) modeling code, vendored for quantization."""

from .configuration_inkling2_loop import Inkling2LoopConfig, Inkling2LoopTextConfig
from .modeling_inkling2_loop import (
    Inkling2LoopForCausalLM,
    Inkling2LoopForConditionalGeneration,
    Inkling2LoopModel,
    Inkling2LoopPreTrainedModel,
    Inkling2LoopTextModel,
)


__all__ = [
    "Inkling2LoopConfig",
    "Inkling2LoopTextConfig",
    "Inkling2LoopPreTrainedModel",
    "Inkling2LoopTextModel",
    "Inkling2LoopModel",
    "Inkling2LoopForCausalLM",
    "Inkling2LoopForConditionalGeneration",
]
