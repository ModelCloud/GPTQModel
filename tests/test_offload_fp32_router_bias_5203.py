# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Disk-offload quantization must keep keep-in-fp32 router tensors at fp32 (#5203).

``offload_to_disk`` builds the shell through ``from_config(dtype=bf16)`` and
``LazyTurtle`` materializes every leaf from that shell slot's dtype. A family
whose router registers ``e_score_correction_bias`` without an explicit dtype
therefore got a bf16 slot: the fp32 checkpoint value was rounded on the way in,
the saved checkpoint stored BF16, and the #5157 reload restore (which only
widens a slot when the checkpoint stored fp32) never fired.

The shell must apply the transformers keep-in-fp32 contract, so the quantize-time
model, the saved checkpoint and the reload all keep the bias fp32.
"""

import glob

import torch
from safetensors import safe_open
from tokenizers import Tokenizer, models
from transformers import (
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    PreTrainedTokenizerFast,
)

from gptqmodel import GPTQModel
from gptqmodel.quantization.config import RTNConfig
from gptqmodel.utils.hf import pin_keep_in_fp32_shell_tensors


VOCAB = 128
DEVICE = "cpu"
BIAS_NAME = "model.layers.1.mlp.gate.e_score_correction_bias"


def _write_source(tmp_path):
    config = DeepseekV3Config(
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
        first_k_dense_replace=1,
        n_group=2,
        topk_group=1,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    model = DeepseekV3ForCausalLM(config).eval()

    # bf16 spacing at ~8 is 0.03-0.06, so a bf16 slot visibly rounds these.
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
    return source, dict(model.named_buffers())[BIAS_NAME].clone()


def _saved_dtype(directory, name):
    dtypes = []
    for path in glob.glob(f"{directory}/*.safetensors"):
        with safe_open(path, framework="pt") as handler:
            if name in handler.keys():
                dtypes.append(handler.get_tensor(name).dtype)
    assert dtypes, f"{name} missing from the saved safetensors shards"
    return dtypes[0]


def test_offload_quantize_keeps_strict_fp32_router_bias(tmp_path):
    source, source_bias = _write_source(tmp_path)
    assert source_bias.dtype == torch.float32

    quantized = GPTQModel.load(
        str(source),
        RTNConfig(bits=4, group_size=32, device=DEVICE, offload_to_disk=True),
    )
    quantized.quantize()

    live_bias = dict(quantized.model.named_buffers())[BIAS_NAME]
    assert live_bias.dtype == torch.float32
    assert torch.equal(live_bias.cpu(), source_bias), (
        "disk-offload materialization must not round the fp32 router bias"
    )

    saved = tmp_path / "quantized"
    quantized.save(str(saved))
    assert _saved_dtype(saved, BIAS_NAME) == torch.float32

    reloaded = GPTQModel.load(str(saved), dtype=torch.bfloat16, device=DEVICE)
    reloaded_bias = dict(reloaded.model.named_buffers())[BIAS_NAME]
    assert reloaded_bias.dtype == torch.float32
    assert torch.equal(reloaded_bias.cpu(), source_bias)


class _PinnedShell(torch.nn.Module):
    _keep_in_fp32_modules_strict = ["keep_buf", "keep_param"]
    _keep_in_fp32_modules = ["nonstrict_buf"]

    def __init__(self):
        super().__init__()
        self.keep_param = torch.nn.Parameter(torch.zeros(4))
        self.other_param = torch.nn.Parameter(torch.zeros(4))
        self.register_buffer("keep_buf", torch.zeros(4))
        self.register_buffer("nonstrict_buf", torch.zeros(4))
        self.register_buffer("other_buf", torch.zeros(4))


def _bf16_meta_shell():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            shell = _PinnedShell()
    finally:
        torch.set_default_dtype(previous)
    assert shell.keep_buf.dtype is torch.bfloat16
    return shell


def test_shell_pin_honors_the_strict_declaration_for_half_builds():
    # Both half dtypes pin `_keep_in_fp32_modules_strict`; the fp16-only
    # non-strict list is not a shell dtype contract (see the helper docstring).
    for dtype in (torch.bfloat16, torch.float16):
        shell = _bf16_meta_shell()
        assert pin_keep_in_fp32_shell_tensors(shell, dtype=dtype) == 2
        assert shell.keep_buf.dtype is torch.float32
        assert shell.keep_param.dtype is torch.float32
        assert shell.nonstrict_buf.dtype is torch.bfloat16
        assert shell.other_buf.dtype is torch.bfloat16
        assert shell.other_param.dtype is torch.bfloat16
        assert shell.keep_buf.is_meta and shell.keep_param.is_meta

    # An fp32 build already satisfies the contract and is left untouched.
    shell = _bf16_meta_shell()
    assert pin_keep_in_fp32_shell_tensors(shell, dtype=torch.float32) == 0
    assert shell.keep_buf.dtype is torch.bfloat16
