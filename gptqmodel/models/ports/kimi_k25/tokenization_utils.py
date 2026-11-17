# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
"""Pinned GPT-2 reversible byte alphabet; see NOTICE and LICENSE-APACHE-2.0."""


def bytes_to_unicode():
    values = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    alphabet = values.copy()
    offset = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            alphabet.append(256 + offset)
            offset += 1
    return dict(zip(values, map(chr, alphabet)))
