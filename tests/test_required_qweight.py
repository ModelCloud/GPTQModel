# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import accelerate
import pytest
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.model import (
    _checkpoint_tensor_keys,
    convert_gptq_v2_to_v1_format,
    validate_checkpoint_qweights,
)


QWEIGHT_KEY = "model.layers.0.self_attn.q_proj.qweight"


def _save_checkpoint(path, format, sharded, missing=None, tied=False, stale_index=False):
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        tie_word_embeddings=tied,
        bos_token_id=1,
        eos_token_id=2,
    )
    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        format=format,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = LlamaForCausalLM(config).half()
    for name, linear in list(model.named_modules()):
        if not isinstance(linear, torch.nn.Linear) or name == "lm_head":
            continue
        quant = TorchLinear(
            bits=4,
            group_size=32,
            desc_act=False,
            sym=True,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=False,
            format=FORMAT.GPTQ_V2,
        )
        scales = torch.full((linear.out_features, linear.in_features // 32), 0.01)
        quant.pack_original(linear, scales, torch.full_like(scales, 8))
        model.set_submodule(name, quant)
    if format == FORMAT.GPTQ:
        convert_gptq_v2_to_v1_format(model, qcfg, TorchLinear)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    keys = list(state)
    if missing is not None:
        del state[missing]
    if sharded:
        weight_map = {
            name: f"model-{1 + index % 2:05d}-of-00002.safetensors"
            for index, name in enumerate(keys if stale_index else state)
        }
        for shard in set(weight_map.values()):
            save_file(
                {name: value for name, value in state.items() if weight_map[name] == shard},
                str(path / shard),
                metadata={"format": "pt"},
            )
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map}),
            encoding="utf-8",
        )
    else:
        save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    config.architectures = ["LlamaForCausalLM"]
    config.save_pretrained(path)
    qcfg.save_pretrained(str(path))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}, unk_token="[UNK]")),
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    tokenizer.save_pretrained(path)
    return state


def _load_checkpoint(path):
    return GPTQModel.load(
        str(path),
        backend=BACKEND.GPTQ_TORCH,
        device="cpu",
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).model


@pytest.mark.parametrize("format", [FORMAT.GPTQ, FORMAT.GPTQ_V2])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("missing", [None, "lm_head.weight", QWEIGHT_KEY.replace("qweight", "g_idx")])
def test_complete_or_derived_checkpoint_weights_load(tmp_path, format, sharded, missing):
    state = _save_checkpoint(tmp_path, format, sharded, missing, tied=missing == "lm_head.weight")

    model = _load_checkpoint(tmp_path)

    torch.testing.assert_close(model.state_dict()[QWEIGHT_KEY], state[QWEIGHT_KEY])
    if missing == "lm_head.weight":
        assert model.lm_head.weight is model.model.embed_tokens.weight
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([[1, 3, 4]]), use_cache=False).logits
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("format", [FORMAT.GPTQ, FORMAT.GPTQ_V2])
@pytest.mark.parametrize("sharded,stale_index", [(False, False), (True, False), (True, True)])
def test_missing_qweight_is_rejected(tmp_path, format, sharded, stale_index):
    _save_checkpoint(tmp_path, format, sharded, QWEIGHT_KEY, stale_index=stale_index)

    with pytest.raises(ValueError, match="Missing required quantized weights") as exc:
        _load_checkpoint(tmp_path)

    assert QWEIGHT_KEY in str(exc.value)


def test_shared_quantized_module_loads_through_saved_alias(tmp_path):
    model = torch.nn.Module()
    model.first = TorchLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=False,
        format=FORMAT.GPTQ_V2,
    )
    model.second = model.first
    state = {name: torch.ones_like(value) for name, value in model.state_dict().items() if name.startswith("second.")}
    checkpoint = tmp_path / "model.safetensors"
    save_file(state, str(checkpoint), metadata={"format": "pt"})

    validate_checkpoint_qweights(model, checkpoint, FORMAT.GPTQ_V2)
    accelerate.load_checkpoint_in_model(model, str(checkpoint), device_map={"": "cpu"})

    assert model.first is model.second
    torch.testing.assert_close(model.first.qweight, state["second.qweight"])


@pytest.mark.parametrize("format", [value for value in FORMAT if value not in (FORMAT.GPTQ, FORMAT.GPTQ_V2)])
def test_other_storage_formats_do_not_require_gptq_qweights(tmp_path, format):
    model = TorchLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=32,
        out_features=32,
        bias=False,
        format=FORMAT.GPTQ_V2,
    )

    validate_checkpoint_qweights(model, tmp_path / "not-a-gptq-checkpoint", format)


@pytest.mark.parametrize("directory", [False, True])
def test_shard_verification_preserves_default_index_key_semantics(tmp_path, directory):
    shard = "model-00001-of-00001.safetensors"
    save_file({"present": torch.ones(1)}, str(tmp_path / shard))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"missing": shard}}), encoding="utf-8")
    checkpoint = tmp_path if directory else index

    assert _checkpoint_tensor_keys(checkpoint) == {"missing"}
    assert _checkpoint_tensor_keys(checkpoint, verify_shards=True) == {"present"}


def test_checkpoint_metadata_does_not_deserialize_pickle_weights(tmp_path):
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"not a pickle checkpoint")

    assert _checkpoint_tensor_keys(checkpoint, verify_shards=True) is None
