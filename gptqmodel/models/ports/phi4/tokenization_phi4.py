# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: MIT
"""The released serialized BPE, not TF5's differently constructed GPT2 regex."""

from transformers import TokenizersBackend as Phi4MMTokenizer


__all__ = ["Phi4MMTokenizer"]
