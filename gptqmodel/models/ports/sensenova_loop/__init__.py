# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Package-owned SenseNova-Loop over the vendored Transformers 5.23 Qwen3.5 classes.

The released processor and tokenizer are the native Qwen3-VL processor and
Qwen2 tokenizer; they are re-exported so the definition's loader names them
without consulting checkpoint metadata.
"""

from ..native.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from ..native.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from .configuration_sensenova_loop import SenseNovaLoopConfig, execution_plan
from .modeling_sensenova_loop import (
    MHCRouter,
    SenseNovaLoopForConditionalGeneration,
    SenseNovaLoopMHCDecoderLayer,
    SenseNovaLoopModel,
    SenseNovaLoopTextModel,
)


__all__ = [
    "MHCRouter",
    "Qwen2Tokenizer",
    "Qwen3VLProcessor",
    "SenseNovaLoopConfig",
    "SenseNovaLoopForConditionalGeneration",
    "SenseNovaLoopMHCDecoderLayer",
    "SenseNovaLoopModel",
    "SenseNovaLoopTextModel",
    "execution_plan",
]
