# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Exact native serialized-BPE backend; normalization is owned by Tokenicer."""

from transformers import TokenizersBackend as ZDTaichu5Tokenizer


__all__ = ["ZDTaichu5Tokenizer"]
