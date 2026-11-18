# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Direct vendored Transformers 5.23 architecture, with no AutoModel or checkpoint dispatch."""

from ..native.ernie4_5_vl_moe.modeling_ernie4_5_vl_moe import (
    Ernie4_5_VLMoeForConditionalGeneration,
)


__all__ = ["Ernie4_5_VLMoeForConditionalGeneration"]
