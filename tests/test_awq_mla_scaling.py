# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ scale folding must survive DeepSeek-V2/V3 MLA attention.

The MLA block holds layers with different inputs in one `module_tree` block
(`q_b_proj` reads `q_a`, `kv_b_proj` reads `kv_a`). AWQ used to fold the whole
block into `input_layernorm`:

* `scale_ln_fcs` multiplied a 32-wide scale into a 64-wide weight, and
* `_search_best_scale` re-ran `self_attn` on the `q_b_proj` input, raising
  `RuntimeError: mm: mat1 and mat2 shapes cannot be multiplied (20x32 and 128x32)`.

The fix keeps only consumers whose input width matches the fold source in the
group (the rest stay quantized but unscaled) and refuses to fold into a norm of
the wrong width.
"""

from pathlib import Path

import defuser
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch import nn
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.models.base import NON_QUANTIZE_FLAGS
from gptqmodel.models.definitions.deepseek_v2 import DeepSeekV2QModel
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import METHOD


_TEXTS = [
    "tiny deepseek moe calibration sample number one with enough tokens",
    "another synthetic calibration example that is intentionally verbose",
]


def _config() -> DeepseekV2Config:
    # Every quantized `out_features` is a multiple of 64 so the AWQ clip search's
    # `oc_batch_size` assertion holds; the values stay tiny enough for CPU.
    return DeepseekV2Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        kv_lora_rank=64,
        q_lora_rank=64,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=64,
        n_routed_experts=4,
        n_shared_experts=1,
        moe_intermediate_size=64,
        first_k_dense_replace=0,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _write_fixture(model_dir: Path) -> DeepseekV2Config:
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        _TEXTS,
        trainer=WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(model_dir)
    config = _config()
    torch.manual_seed(0)
    DeepseekV2ForCausalLM(config).save_pretrained(model_dir)
    return config


def _bare_qmodel(hf_model):
    qmodel = DeepSeekV2QModel.__new__(DeepSeekV2QModel)
    nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = hf_model
    return qmodel


def _features(definition, layer, config, batch: int = 1, seq: int = 6):
    modules = dict(layer.named_modules())
    features = {}
    for block in definition.full_layer_modules(config, is_awq_quantize=True, include_capture_only=True):
        for raw_name in block:
            if any(flag in raw_name for flag in NON_QUANTIZE_FLAGS):
                continue
            name = raw_name
            for flag in NON_QUANTIZE_FLAGS:
                name = name.replace(flag, "")
            module = modules.get(name)
            if isinstance(module, nn.Linear):
                features[name] = torch.randn(batch, seq, module.in_features)
    features["mlp"] = torch.randn(batch, seq, config.hidden_size)
    return features


def test_mla_plain_quantization_does_not_emit_scale_groups():
    torch.manual_seed(0)
    hf_model = DeepseekV2ForCausalLM(_config()).eval()
    defuser.convert_model(hf_model, cleanup_original=True)
    layer = hf_model.model.layers[0]
    qmodel = _bare_qmodel(hf_model)

    nodes = qmodel.awq_get_modules_for_scaling(layer, _features(DeepSeekV2QModel, layer, hf_model.config), {})
    # MLA's normalized latent stages and expert branches are not qualified for
    # source folding. The public quantization test below still packs linears.
    assert nodes == []


@pytest.mark.cpu
@pytest.mark.slow
def test_mla_awq_quantization_completes(tmp_path: Path):
    """End-to-end regression for the `_search_best_scale` shape crash."""

    model_dir = tmp_path / "native"
    config = _write_fixture(model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    calibration = [dict(tokenizer(text, return_tensors="pt")) for text in _TEXTS]

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=16,
        quant_method=METHOD.AWQ,
        device="cpu",
    )
    model = GPTQModel.load(str(model_dir), quantize_config=quantize_config, backend=BACKEND.TORCH)
    model.quantize(calibration, batch_size=1, backend=BACKEND.TORCH, calibration_data_min_length=1)

    modules = dict(model.model.named_modules())
    for expert_index in range(config.n_routed_experts):
        name = f"model.layers.0.mlp.experts.{expert_index}.gate_proj"
        assert isinstance(modules[name], AwqTorchLinear), name
    assert isinstance(modules["model.layers.0.self_attn.q_a_proj"], AwqTorchLinear)
    assert isinstance(modules["model.layers.0.self_attn.q_b_proj"], AwqTorchLinear)
