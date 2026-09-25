# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.8-27B text projection shapes for MLX kernel validation.

Derived from Qwen/Qwen3.8-27B config at revision
1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0. Shapes use the PyTorch
weight convention (out_features, in_features). Full attention has an output
gate on Q, while linear attention uses a fused QKV projection.
"""

MODEL_ID = "Qwen/Qwen3.8-27B"
CONFIG_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CONFIG_URL = f"https://huggingface.co/{MODEL_ID}/blob/{CONFIG_REVISION}/config.json"

HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 17408
HEAD_DIM = 256
ATTENTION_HEADS = 24
KEY_VALUE_HEADS = 4
LINEAR_KEY_HEADS = 16
LINEAR_VALUE_HEADS = 48
LINEAR_HEAD_DIM = 128
CHECKPOINT_DTYPE = "bfloat16"

QWEN38_27B_PROJECTIONS = (
    ("full_attn.q_proj", ATTENTION_HEADS * HEAD_DIM * 2, HIDDEN_SIZE),
    ("full_attn.k_proj", KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE),
    ("full_attn.v_proj", KEY_VALUE_HEADS * HEAD_DIM, HIDDEN_SIZE),
    ("full_attn.o_proj", HIDDEN_SIZE, ATTENTION_HEADS * HEAD_DIM),
    (
        "linear_attn.in_proj_qkv",
        (2 * LINEAR_KEY_HEADS + LINEAR_VALUE_HEADS) * LINEAR_HEAD_DIM,
        HIDDEN_SIZE,
    ),
    ("linear_attn.out_proj", HIDDEN_SIZE, LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM),
    ("mlp.gate_proj", INTERMEDIATE_SIZE, HIDDEN_SIZE),
    ("mlp.up_proj", INTERMEDIATE_SIZE, HIDDEN_SIZE),
    ("mlp.down_proj", HIDDEN_SIZE, INTERMEDIATE_SIZE),
)
