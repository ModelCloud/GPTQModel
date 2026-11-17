# Copyright 2020 The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
"""GPT-2 reversible byte alphabet, retained for the pinned Kimi-K2 tokenizer."""


def bytes_to_unicode():
    values = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(values, map(chr, chars)))
