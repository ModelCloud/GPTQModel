# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A quantized reload must keep the router tensors pinned to fp32 (#5151).

transformers 5.x declares ``_keep_in_fp32_modules_strict`` on the DeepSeek-V3 /
GLM-4.x / ERNIE-4.5-MoE families. Those tensors (``e_score_correction_bias`` is
a buffer, ERNIE's ``gate.weight`` is a parameter) must stay fp32 for every half
dtype. The quantized-load path cast them to the requested bf16, which erased
most expert-to-expert differences and re-routed 11-36% of tokens per layer.
"""

import torch
from tokenizers import Tokenizer, models
from transformers import (
    Glm4MoeLiteConfig,
    Glm4MoeLiteForCausalLM,
    PreTrainedTokenizerFast,
)

from gptqmodel import GPTQModel
from gptqmodel.quantization.config import RTNConfig


VOCAB = 128
DEVICE = "cpu"


def _write_source(tmp_path):
    config = Glm4MoeLiteConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=96,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        mlp_layer_types=["dense", "sparse"],
    )
    config._attn_implementation = "eager"
    model = Glm4MoeLiteForCausalLM(config).eval()

    # Values in the issue's range: bf16 spacing at ~8 is 0.03-0.06, so the cast
    # erases the per-expert differences.
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.copy_(torch.linspace(7.4, 9.0, buffer.numel()))

    source = tmp_path / "source"
    model.save_pretrained(source)
    tokenizer = Tokenizer(
        models.WordLevel(vocab={str(i): i for i in range(VOCAB)}, unk_token="0")
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="0",
    ).save_pretrained(source)
    return source, dict(model.named_buffers())


def test_reload_keeps_strict_fp32_router_bias(tmp_path):
    source, source_buffers = _write_source(tmp_path)
    bias_name = "model.layers.1.mlp.gate.e_score_correction_bias"
    assert source_buffers[bias_name].dtype == torch.float32

    quantized = GPTQModel.load(
        str(source),
        RTNConfig(bits=4, group_size=32, device=DEVICE, offload_to_disk=False),
    )
    quantized.quantize()
    saved = tmp_path / "quantized"
    quantized.save(str(saved))

    reloaded = GPTQModel.load(str(saved), dtype=torch.bfloat16, device=DEVICE)
    reloaded_buffers = dict(reloaded.model.named_buffers())

    assert reloaded_buffers[bias_name].dtype == torch.float32
    assert torch.equal(
        reloaded_buffers[bias_name].cpu(), source_buffers[bias_name].cpu()
    ), "the reload must not round the fp32 router bias"
