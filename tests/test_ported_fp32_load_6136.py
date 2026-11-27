# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Construction-time fp32 parameters in the Hunyuan-MoE and Zamba ports (#6136).

Both ports allocate one parameter in fp32 exactly as upstream Transformers
does: Hunyuan-MoE's router ``mlp.gate.wg`` and Zamba's ``mamba.A_log``. A bf16
checkpoint therefore loads with those parameters still fp32. Since #4620 the
post-load guard only exempts ``_keep_in_fp32_modules(_strict)`` declarations, so
both dense loads (and Hunyuan's quantize load with ``offload_to_disk=False``)
raised "floating parameters have mixed dtypes" even though the values were not
corrupted.

The ports now declare those modules in ``_keep_in_fp32_modules_strict``. The
guard must still reject a genuinely mixed load: the negative control casts a
bf16 parameter to fp32 and expects the same error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.models.ports.native.hunyuan_v1_moe.configuration_hunyuan_v1_moe import HunYuanMoEV1Config
from gptqmodel.models.ports.native.hunyuan_v1_moe.modeling_hunyuan_v1_moe import HunYuanMoEV1ForCausalLM
from gptqmodel.models.ports.native.zamba.configuration_zamba import ZambaConfig
from gptqmodel.models.ports.native.zamba.modeling_zamba import ZambaForCausalLM
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils.hf import _compile_keep_in_fp32_matcher, assert_loaded_model_state


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ports"


def _tiny_tokenizer(path: Path) -> None:
    tokens = ["<pad>", "<s>", "</s>", "<unk>"] + [f"t{i}" for i in range(95)]
    backend = Tokenizer(WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
    ).save_pretrained(path)


def _tiny_hunyuan_config() -> HunYuanMoEV1Config:
    """The released Hunyuan-A13B config, shrunk to two layers and eight experts."""

    payload = json.loads((FIXTURES / "hunyuan_v1_moe" / "released_config.json").read_text())
    payload.update(
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        attention_head_dim=16,
        num_experts=8,
        moe_topk=2,
        moe_intermediate_size=[64, 64],
        num_shared_expert=[1, 1],
        vocab_size=99,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        eod_token_id=3,
        sep_token_id=4,
        dtype="bfloat16",
        torch_dtype="bfloat16",
    )
    payload.pop("auto_map", None)
    return HunYuanMoEV1Config(**payload)


def _tiny_zamba_config() -> ZambaConfig:
    """The released Zamba-7B-v1 config, shrunk to four layers."""

    payload = json.loads((FIXTURES / "zamba" / "released_config.json").read_text())
    payload.update(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        attention_head_dim=32,
        attention_hidden_size=128,
        n_mamba_heads=2,
        mamba_d_state=16,
        mamba_dt_rank=32,
        mamba_expand=2,
        layers_block_type=["mamba", "hybrid", "mamba", "hybrid"],
        vocab_size=99,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=128,
        use_mamba_kernels=False,
        dtype="bfloat16",
        torch_dtype="bfloat16",
    )
    return ZambaConfig(**payload)


def _write_hunyuan_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    model = HunYuanMoEV1ForCausalLM(_tiny_hunyuan_config()).eval()
    model.to(torch.bfloat16)
    # The hub checkpoint stores the router weight as F32.
    for layer in model.model.layers:
        layer.mlp.gate.wg.to(torch.float32)
    model.save_pretrained(path)
    _tiny_tokenizer(path)


def _write_zamba_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    ZambaForCausalLM(_tiny_zamba_config()).eval().to(torch.bfloat16).save_pretrained(path)
    _tiny_tokenizer(path)


def test_hunyuan_and_zamba_declare_their_construction_fp32_modules():
    """The declarations must cover the exact runtime parameter names."""

    hunyuan = HunYuanMoEV1ForCausalLM(_tiny_hunyuan_config())
    for name, parameter in hunyuan.named_parameters():
        if not name.endswith("mlp.gate.wg.weight"):
            parameter.data = parameter.data.to(torch.bfloat16)
    matcher = _compile_keep_in_fp32_matcher(hunyuan)
    assert matcher is not None
    hunyuan_fp32 = {
        name for name, parameter in hunyuan.named_parameters() if parameter.dtype is torch.float32
    }
    assert hunyuan_fp32 == {
        "model.layers.0.mlp.gate.wg.weight",
        "model.layers.1.mlp.gate.wg.weight",
    }
    assert all(matcher(name) for name in hunyuan_fp32)

    zamba = ZambaForCausalLM(_tiny_zamba_config())
    for name, parameter in zamba.named_parameters():
        if not name.endswith("A_log"):
            parameter.data = parameter.data.to(torch.bfloat16)
    matcher = _compile_keep_in_fp32_matcher(zamba)
    assert matcher is not None
    zamba_fp32 = {
        name for name, parameter in zamba.named_parameters() if parameter.dtype is torch.float32
    }
    assert zamba_fp32 == {
        "model.layers.0.mamba.A_log",
        "model.layers.1.mamba_decoder.mamba.A_log",
        "model.layers.2.mamba.A_log",
        "model.layers.3.mamba_decoder.mamba.A_log",
    }
    assert all(matcher(name) for name in zamba_fp32)


def test_hunyuan_a13b_bf16_checkpoint_loads_with_the_router_in_fp32(tmp_path):
    _write_hunyuan_checkpoint(tmp_path)

    loaded = GPTQModel.load(str(tmp_path), device="cpu")

    assert isinstance(loaded.model, HunYuanMoEV1ForCausalLM)
    router = loaded.model.model.layers[0].mlp.gate.wg.weight
    assert router.dtype is torch.float32
    assert loaded.model.model.layers[0].self_attn.q_proj.weight.dtype is torch.bfloat16


def test_hunyuan_a13b_quantize_load_without_disk_offload(tmp_path):
    _write_hunyuan_checkpoint(tmp_path)

    loaded = GPTQModel.load(
        str(tmp_path),
        quantize_config=QuantizeConfig(
            bits=4,
            group_size=128,
            offload_to_disk=False,
            device="cpu",
        ),
    )

    assert isinstance(loaded.model, HunYuanMoEV1ForCausalLM)
    assert loaded.model.model.layers[0].mlp.gate.wg.weight.dtype is torch.float32


def test_zamba_7b_v1_bf16_checkpoint_loads_with_a_log_in_fp32(tmp_path):
    _write_zamba_checkpoint(tmp_path)

    loaded = GPTQModel.load(str(tmp_path), device="cpu")

    assert isinstance(loaded.model, ZambaForCausalLM)
    assert loaded.model.model.layers[0].mamba.A_log.dtype is torch.float32
    assert loaded.model.model.layers[1].mamba_decoder.mamba.A_log.dtype is torch.float32
    assert loaded.model.model.embed_tokens.weight.dtype is torch.bfloat16


def test_mixed_load_control_still_rejects_a_corrupted_bf16_parameter(tmp_path):
    """The declaration exempts only the declared fp32 pin, never other corruption."""

    _write_zamba_checkpoint(tmp_path)
    loaded = GPTQModel.load(str(tmp_path), device="cpu")
    loaded.model.model.embed_tokens.weight.data = loaded.model.model.embed_tokens.weight.data.float()

    with pytest.raises(RuntimeError, match="mixed dtypes"):
        assert_loaded_model_state(
            loaded.model,
            api_name="test.load",
            require_uniform_dtype=True,
        )
