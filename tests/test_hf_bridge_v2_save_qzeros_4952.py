# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The HF bridge must save GPTQ v2 qzeros unchanged (issue #4952).

``AutoModelForCausalLM.from_pretrained`` reaches the GPTQ bridge's
``hf_gptqmodel_post_init_for_load``.  For a v2 checkpoint that hook recorded
``_checkpoint_format_on_disk = GPTQ_V2`` but never called ``qzero_format(2)``,
so the module kept the constructor's v1 placeholder while its live words were
v2.  ``state_dict()``/``save_pretrained()`` then applied the v1->v2 correction
(+1 per zero point) to already-v2 words, and a reload decoded every zero point
one high (wikitext nll 3.10 -> 15.23 in the issue).

The same bridge converts a v1 checkpoint to the v2 runtime layout, so its save
must keep emitting the v1 ``zero - 1`` words.  These cases assert the raw
``qzeros`` bytes and the logits, not just the config label.
"""

import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)

from gptqmodel.nn_modules.qlinear import GPTQQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT, QuantizeConfig
from gptqmodel.utils.model import convert_gptq_v2_to_v1_format


HIDDEN = 32
INTERMEDIATE = 64
VOCAB = 32
GROUP = 32
BITS = 4

QPROJ = "model.layers.0.self_attn.q_proj"
QZEROS_KEY = f"{QPROJ}.qzeros"


def _save_tokenizer(directory):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        ["the quick brown fox jumps over the lazy dog", "a tiny llama sample"],
        WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(directory)


def _tiny_llama(seed=0):
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(seed)
    return config, LlamaForCausalLM(config).half()


def _build_checkpoint(path, fmt):
    """Write a tiny GPTQ v1 or v2 checkpoint and return its saved state."""

    os.makedirs(path, exist_ok=True)
    config, model = _tiny_llama()
    qcfg = QuantizeConfig(
        bits=BITS,
        group_size=GROUP,
        desc_act=False,
        sym=True,
        format=fmt,
        offload_to_disk=False,
    )
    for name, linear in list(model.named_modules()):
        if not isinstance(linear, torch.nn.Linear) or name == "lm_head":
            continue
        quant = TorchLinear(
            bits=BITS,
            group_size=GROUP,
            desc_act=False,
            sym=True,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=False,
            format=FORMAT.GPTQ_V2,
        )
        scales = torch.full((linear.out_features, linear.in_features // GROUP), 0.01)
        zeros = torch.full_like(scales, 8)
        quant.pack_original(linear, scales, zeros)
        model.set_submodule(name, quant)

    if fmt == FORMAT.GPTQ:
        convert_gptq_v2_to_v1_format(model, qcfg, TorchLinear)
    for module in model.modules():
        if isinstance(module, TorchLinear):
            module._checkpoint_format_on_disk = fmt

    # A real checkpoint never carries the runtime layout marker.
    state = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.endswith(".qzeros_format")
    }
    save_file(state, os.path.join(path, "model.safetensors"), metadata={"format": "pt"})
    config.architectures = ["LlamaForCausalLM"]
    config.save_pretrained(path)
    qcfg.save_pretrained(path)
    _save_tokenizer(path)

    # ``AutoModelForCausalLM`` discovers a quantized checkpoint through
    # ``config.json``; ``GPTQModel.load`` reads ``quantize_config.json``.
    config_path = os.path.join(path, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        model_config = json.load(handle)
    model_config["quantization_config"] = qcfg.to_dict()
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(model_config, handle, indent=2)
    return state


def _hf_load(path):
    """Load through the plain Transformers bridge, not ``GPTQModel.load``."""

    return AutoModelForCausalLM.from_pretrained(
        str(path),
        device_map="cpu",
        dtype=torch.float16,
        local_files_only=True,
    )


def _logits(model):
    input_ids = torch.tensor([[1, 3, 4, 5]])
    with torch.inference_mode():
        return model(input_ids=input_ids, use_cache=False).logits.float()


def _assert_hf_bridge_module(model):
    module = model.model.layers[0].self_attn.q_proj
    assert isinstance(module, GPTQQuantLinear), (
        "the checkpoint did not load through the GPTQ HF bridge: "
        f"{type(module).__name__}"
    )
    return module


@pytest.mark.parametrize("fmt", [FORMAT.GPTQ, FORMAT.GPTQ_V2], ids=["gptq_v1", "gptq_v2"])
def test_hf_bridge_load_save_reload_preserves_qzeros(tmp_path, fmt):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source_state = _build_checkpoint(str(source), fmt)
    disk = source_state[QZEROS_KEY]

    model = _hf_load(source)
    module = _assert_hf_bridge_module(model)
    # Both checkpoint conventions decode v2 in memory on the Torch kernel.
    assert module.qzero_format() == 2
    # The persistent marker must be initialized, not left as uninitialized
    # memory (the issue observed 7 and 409534578 on the HF path).
    assert int(module.qzeros_format) == 2

    # `state_dict()` emits the checkpoint convention, so it must match the file.
    assert torch.equal(model.state_dict()[QZEROS_KEY].cpu(), disk)

    reference = _logits(model)
    model.save_pretrained(str(target))

    saved = load_file(os.path.join(str(target), "model.safetensors"))[QZEROS_KEY]
    assert torch.equal(saved, disk), (
        f"{fmt.value}: save_pretrained shifted qzeros by "
        f"{int((saved.to(torch.int64) - disk.to(torch.int64)).abs().max())}"
    )

    reloaded = _hf_load(target)
    assert torch.equal(reloaded.state_dict()[QZEROS_KEY].cpu(), disk)
    assert torch.equal(_logits(reloaded), reference)


@pytest.mark.parametrize("fmt", [FORMAT.GPTQ, FORMAT.GPTQ_V2], ids=["gptq_v1", "gptq_v2"])
def test_hf_bridge_state_dict_round_trip_is_identity(tmp_path, fmt):
    source = tmp_path / "source"
    source_state = _build_checkpoint(str(source), fmt)
    disk = source_state[QZEROS_KEY]

    model = _hf_load(source)
    _assert_hf_bridge_module(model)
    reference = _logits(model)

    state_dict = model.state_dict()
    assert torch.equal(state_dict[QZEROS_KEY].cpu(), disk)

    result = model.load_state_dict(state_dict)
    assert not result.missing_keys and not result.unexpected_keys
    assert torch.equal(model.state_dict()[QZEROS_KEY].cpu(), disk)
    assert torch.equal(_logits(model), reference)
