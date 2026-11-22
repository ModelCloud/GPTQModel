# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Package-owned InternLM2/2.5 implementation; imported lazily by its definition."""

from .configuration_internlm2 import InternLM2Config


def __getattr__(name):
    if name in ("InternLM2Model", "InternLM2ForCausalLM"):
        from . import modeling_internlm2

        return getattr(modeling_internlm2, name)
    raise AttributeError(name)
