# Copyright 2022 Salesforce authors, The EleutherAI, and HuggingFace Teams. All rights reserved.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CodeGen2 TPU-v3 attention with current Transformers cache/generation APIs.

Adapted from Salesforce/codegen2-1B_P, revision
47f05cd3c1b357745fbead74202fc1efe87b6d25 (see PROVENANCE.md).
Four-partition checkpoints retain the native Transformers implementation.
"""

import torch
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.codegen.modeling_codegen import (
    CodeGenAttention as NativeCodeGenAttention,
)
from transformers.models.codegen.modeling_codegen import (
    CodeGenBlock as NativeCodeGenBlock,
)
from transformers.models.codegen.modeling_codegen import (
    CodeGenForCausalLM as NativeCodeGenForCausalLM,
)
from transformers.models.codegen.modeling_codegen import (
    CodeGenMLP,
    apply_rotary_pos_emb,
)
from transformers.models.codegen.modeling_codegen import (
    CodeGenModel as NativeCodeGenModel,
)
from transformers.models.codegen.modeling_codegen import (
    CodeGenPreTrainedModel as NativeCodeGenPreTrainedModel,
)

from .configuration_codegen import CodeGenConfig


class CodeGen2Attention(NativeCodeGenAttention):
    def forward(self, hidden_states, layer_past=None, attention_mask=None, position_ids=None,
                use_cache=False, output_attentions=False):
        qkv = self.qkv_proj(hidden_states)
        # Original TPU-v3 checkpoint rows are [Q_i, V_i, K_i] for i in 0..7.
        # Shapes alone cannot distinguish them from native TPU-v4's four groups.
        qkv = qkv.reshape(*qkv.shape[:-1], 8, 3, self.num_attention_heads // 8, self.head_dim)
        query, value, key = qkv.unbind(dim=-3)
        shape = (*hidden_states.shape[:-1], self.num_attention_heads, self.head_dim)
        query, key, value = (x.reshape(shape) for x in (query, key, value))

        sincos = self.embed_positions.to(position_ids.device)[position_ids]
        sin, cos = sincos.chunk(2, dim=-1)
        rotary_dim = self.rotary_dim
        query = torch.cat((apply_rotary_pos_emb(query[..., :rotary_dim], sin, cos), query[..., rotary_dim:]), -1)
        key = torch.cat((apply_rotary_pos_emb(key[..., :rotary_dim], sin, cos), key[..., rotary_dim:]), -1)
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        if layer_past is not None:
            key, value = layer_past.update(key.to(hidden_states.dtype), value, self.layer_idx)

        # Salesforce scales before the additive mask. Preserve that ordering for
        # non-binary attention biases as well as ordinary causal/padding masks.
        weights = torch.matmul(query.float(), key.float().transpose(-1, -2)) / self.scale_attn
        if attention_mask is not None:
            weights = weights + attention_mask
        weights = self.attn_dropout(weights.softmax(-1).to(value.dtype))
        output = torch.matmul(weights, value).transpose(1, 2).reshape(hidden_states.shape)
        return self.resid_dropout(self.out_proj(output)), weights


class CodeGen2Block(NativeCodeGenBlock):
    def __init__(self, config, layer_idx):
        # Inherit native parallel-residual forward and checkpointing machinery.
        GradientCheckpointingLayer.__init__(self)
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CodeGen2Attention(config, layer_idx)
        self.mlp = CodeGenMLP(config.n_inner if config.n_inner is not None else 4 * config.n_embd, config)


class CodeGen2Model(NativeCodeGenModel):
    _no_split_modules = ["CodeGen2Block"]

    def __init__(self, config):
        NativeCodeGenPreTrainedModel.__init__(self, config)
        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(CodeGen2Block(config, i) for i in range(config.n_layer))
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.rotary_dim = config.rotary_dim
        self.gradient_checkpointing = False
        self.post_init()


class CodeGenForCausalLM(NativeCodeGenForCausalLM):
    config_class = CodeGenConfig
    _no_split_modules = ["CodeGenBlock", "CodeGen2Block"]
    # Original checkpoints persist this deterministic triangular buffer. Current
    # Transformers regenerates the causal mask from positions and cache state.
    _keys_to_ignore_on_load_unexpected = [r"transformer\.h\.\d+\.attn\.causal_mask"]

    def __init__(self, config):
        config.validate_layout()
        NativeCodeGenPreTrainedModel.__init__(self, config)
        model_class = CodeGen2Model if config.qkv_partitions == 8 else NativeCodeGenModel
        self.transformer = model_class(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.post_init()
