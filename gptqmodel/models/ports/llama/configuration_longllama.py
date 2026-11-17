# coding=utf-8
# Copyright 2023 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""LongLLaMA model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)
LONGLLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "syzymon/long_llama_3b": "https://huggingface.co/syzymon/long_llama_3b/resolve/main/config.json"
}


class LongLlamaConfig(PretrainedConfig):
    model_type = "longllama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        last_context_length=1024,
        mem_layers=None,
        mem_positionals=True,
        mem_dtype="bfloat16",
        mem_attention_grouping=None,
        torch_attention=False,
        gradient_checkpoint_every_ith=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.last_context_length = last_context_length
        self.mem_layers = list(mem_layers or [])
        self.mem_positionals = mem_positionals
        self.mem_dtype = mem_dtype
        self.mem_attention_grouping = mem_attention_grouping
        self.torch_attention = torch_attention
        self.gradient_checkpoint_every_ith = gradient_checkpoint_every_ith
        if mem_dtype not in {"float16", "bfloat16", "float32", "float64"}:
            raise ValueError("mem_dtype must name a floating point Torch dtype")
        if max_position_embeddings <= 0 or last_context_length <= 0:
            raise ValueError("Context lengths must be positive")
        if mem_attention_grouping is not None and (
            len(mem_attention_grouping) != 2 or min(mem_attention_grouping) <= 0
        ):
            raise ValueError("mem_attention_grouping must contain two positive integers")
        kv_heads = kwargs.pop("num_key_value_heads", num_attention_heads)
        if kv_heads != num_attention_heads:
            raise ValueError("LongLLaMA checkpoints use MHA, not GQA/MQA")
        self.num_key_value_heads = num_attention_heads
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
