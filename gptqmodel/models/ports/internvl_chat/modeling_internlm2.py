# SPDX-License-Identifier: MIT
"""Read-only output alias for the checkpoint-native InternLM2 decoder."""

from ..internlm2.modeling_internlm2 import InternLM2ForCausalLM


class InternVLInternLM2ForCausalLM(InternLM2ForCausalLM):
    @property
    def lm_head(self):
        # A property adds no module registration and preserves output.* keys.
        return self.output
