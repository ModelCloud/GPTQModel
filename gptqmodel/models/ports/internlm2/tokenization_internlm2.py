# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""InternLM2/2.5 tokenizer; its inspected upstream algorithm matches InternLM1."""

from ..internlm.tokenization_internlm import InternLMTokenizer


class InternLM2Tokenizer(InternLMTokenizer):
    """Use the checkpoint's distinct InternLM2 SentencePiece vocabulary and specials."""
