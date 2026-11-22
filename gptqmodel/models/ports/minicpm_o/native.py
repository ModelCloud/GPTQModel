# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native decoders used by the MiniCPM-o LLM and TTS heads.

Streaming code passes Transformers 5 ``Cache`` objects directly, so no cache adaptation is needed.
"""

from ..native.llama.modeling_llama import LlamaModel
from ..native.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from ..native.qwen3.modeling_qwen3 import Qwen3ForCausalLM


__all__ = ["LlamaModel", "Qwen2ForCausalLM", "Qwen3ForCausalLM"]
