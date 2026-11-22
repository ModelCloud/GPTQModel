# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""Inkling 2 Mini: looped hybrid-attention MoE with a vision tower.

The implementation is derived from the released checkpoint's tensor layout and
from the architecture description, not from any upstream modeling file. It
targets the Transformers release GPT-QModel is built against; there is no
version-conditional compatibility layer.
"""

from .configuration import (
    Inkling2LoopConfig,
    Inkling2LoopTextConfig,
    Inkling2LoopVisionConfig,
)
from .experts import (
    Inkling2LoopExpertProjections,
    Inkling2LoopExperts,
    Inkling2LoopPackedExpertLinear,
    Inkling2LoopPackedExperts,
)
from .modeling import (
    Inkling2LoopForCausalLM,
    Inkling2LoopForConditionalGeneration,
    Inkling2LoopModel,
    Inkling2LoopPreTrainedModel,
    Inkling2LoopTextModel,
)
from .tenstorrent_nf12 import install_inkling2_nf12
from .vision import Inkling2LoopVisionModel


__all__ = [
    "Inkling2LoopConfig",
    "Inkling2LoopExpertProjections",
    "Inkling2LoopExperts",
    "Inkling2LoopPackedExpertLinear",
    "Inkling2LoopForCausalLM",
    "Inkling2LoopForConditionalGeneration",
    "Inkling2LoopModel",
    "Inkling2LoopPackedExperts",
    "Inkling2LoopPreTrainedModel",
    "Inkling2LoopTextConfig",
    "Inkling2LoopTextModel",
    "Inkling2LoopVisionConfig",
    "Inkling2LoopVisionModel",
    "install_inkling2_nf12",
]
