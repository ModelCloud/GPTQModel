# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native decoders used by the MiniCPM-o LLM and TTS heads.

Streaming code passes Transformers 5 ``Cache`` objects directly, so no cache adaptation is needed.
"""

from transformers import LlamaModel, Qwen2ForCausalLM, Qwen3ForCausalLM


__all__ = ["LlamaModel", "Qwen2ForCausalLM", "Qwen3ForCausalLM"]
