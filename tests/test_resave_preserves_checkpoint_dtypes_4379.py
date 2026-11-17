# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Re-saving a loaded checkpoint must publish the dtype it read (#4379).

The export shell is built in the model dtype. Accelerate's ``dtype=None`` load
copies each checkpoint tensor into that shell slot, so a legacy checkpoint that
stores a non-quantized buffer (a bias, or EXL3's ``suh``/``svh`` sign vectors)
at another dtype was narrowed to the model dtype on re-save. The loader and the
export shell now restore every such buffer from the checkpoint at its stored
dtype.
"""

import json

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.models.writer import _quantization_config_for_model_config
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import EXL3Config, QuantizeConfig


CALIBRATION = [
    {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "attention_mask": torch.ones((1, 8), dtype=torch.long),
    }
]


def _save_tokenizer(directory):
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(
        ["tiny calibration text", "another dtype round trip"],
        trainer=trainers.WordLevelTrainer(
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
            vocab_size=64,
        ),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    ).save_pretrained(directory)


def _tiny_source(directory):
    directory.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        dtype=torch.bfloat16,
        attention_bias=True,
        mlp_bias=True,
    )
    LlamaForCausalLM(config).to(dtype=torch.bfloat16).save_pretrained(directory)
    _save_tokenizer(directory)
    return directory


def _shard_tensors(directory):
    """Map every stored tensor name to (dtype, raw bytes)."""

    tensors = {}
    for filename in sorted(directory.iterdir()):
        if not filename.name.endswith(".safetensors"):
            continue
        with safe_open(str(filename), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                tensors[name] = (
                    tensor.dtype,
                    tensor.contiguous().view(torch.uint8).clone(),
                )
    return tensors


def _rewrite_biases_as_fp16(directory):
    """Turn a bf16 checkpoint into a legacy fp16-bias checkpoint."""

    changed = 0
    for filename in directory.iterdir():
        if not filename.name.endswith(".safetensors"):
            continue
        tensors = load_file(str(filename))
        updated = False
        for name, tensor in list(tensors.items()):
            if name.endswith(".bias") and tensor.dtype != torch.float16:
                tensors[name] = tensor.to(torch.float16)
                changed += 1
                updated = True
        if updated:
            save_file(tensors, str(filename), metadata={"format": "pt"})
    return changed


def test_gptq_resave_preserves_fp16_biases_under_bf16_config(tmp_path):
    source = _tiny_source(tmp_path / "source")
    quantized = tmp_path / "quantized"
    resaved = tmp_path / "resaved"

    model = GPTQModel.load(
        str(source),
        quantize_config=QuantizeConfig(
            bits=4, group_size=32, desc_act=False, device="cpu"
        ),
        backend=BACKEND.TORCH,
    )
    model.quantize(
        CalibrationConfig(
            dataset=CALIBRATION,
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.GPTQ_TORCH,
    )
    model.save(str(quantized))
    del model

    # The legacy checkpoint stores fp16 biases next to a bf16 model config.
    assert _rewrite_biases_as_fp16(quantized) == 7
    before = _shard_tensors(quantized)
    biases = {
        name: payload for name, payload in before.items() if name.endswith(".bias")
    }
    assert len(biases) == 7
    assert all(dtype is torch.float16 for dtype, _ in biases.values())

    loaded = GPTQModel.load(str(quantized), backend=BACKEND.TORCH, device="cpu")
    loaded.save(str(resaved))
    del loaded

    after = _shard_tensors(resaved)
    assert set(after) == set(before)
    for name, (dtype, payload) in biases.items():
        assert after[name][0] is torch.float16, (name, after[name][0])
        assert torch.equal(after[name][1], payload), name


def _write_synthetic_exl3_checkpoint(directory):
    """Write a dense bf16 Llama whose q_proj is an fp16 EXL3 payload."""

    directory.mkdir(parents=True, exist_ok=True)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        dtype=torch.bfloat16,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).to(dtype=torch.bfloat16).save_pretrained(directory)
    _save_tokenizer(directory)

    module_name = "model.layers.0.self_attn.q_proj"
    generator = torch.Generator().manual_seed(4379)
    tensors = {
        # 48 columns / 16 == 3 bits per weight.
        "trellis": torch.randint(
            -32768, 32767, (8, 8, 64), dtype=torch.int16, generator=generator
        ),
        "suh": (torch.rand(128, generator=generator) * 0.02 + 0.001).to(
            torch.float16
        ),
        "svh": (torch.rand(128, generator=generator) * 0.02 + 0.001).to(
            torch.float16
        ),
    }
    module = ExllamaV3TorchLinear.from_tensors(
        in_features=128,
        out_features=128,
        name=module_name,
        tensors=tensors,
    )
    qcfg = EXL3Config(bits=4.0, device="cpu")
    qcfg.tensor_storage = {
        module_name: {
            "stored_tensors": {
                f"{module_name}.{name}": {
                    "shape": list(tensor.shape),
                    "torch_dtype": str(tensor.dtype).split(".")[-1],
                }
                for name, tensor in tensors.items()
            },
            "quant_format": "exl3",
            "bits_per_weight": 4,
        }
    }

    shard = directory / "model.safetensors"
    state = load_file(str(shard))
    state.pop(f"{module_name}.weight")
    state.pop(f"{module_name}.bias", None)
    for name, tensor in module.state_dict().items():
        state[f"{module_name}.{name}"] = tensor
    save_file(state, str(shard), metadata={"format": "pt"})

    qcfg.save_pretrained(directory)
    config_path = directory / "config.json"
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_config["quantization_config"] = _quantization_config_for_model_config(
        qcfg, FORMAT.EXL3
    )
    config_path.write_text(json.dumps(saved_config, indent=2), encoding="utf-8")
    return directory


def test_exl3_resave_preserves_fp16_sign_vectors(tmp_path):
    checkpoint = _write_synthetic_exl3_checkpoint(tmp_path / "exl3")
    resaved = tmp_path / "resaved"

    before = _shard_tensors(checkpoint)
    sign_vectors = {
        name: payload
        for name, payload in before.items()
        if name.endswith((".suh", ".svh"))
    }
    assert len(sign_vectors) == 2
    assert all(dtype is torch.float16 for dtype, _ in sign_vectors.values())

    loaded = GPTQModel.load(
        str(checkpoint), backend=BACKEND.EXL3_TORCH, device="cpu"
    )
    loaded_buffers = dict(loaded.model.named_buffers())
    for name, (_, payload) in sign_vectors.items():
        assert loaded_buffers[name].dtype is torch.float16, name
        assert torch.equal(
            loaded_buffers[name].contiguous().view(torch.uint8).cpu(), payload
        ), name

    loaded.save(str(resaved))
    del loaded

    after = _shard_tensors(resaved)
    for name, (dtype, payload) in sign_vectors.items():
        assert after[name][0] is torch.float16, (name, after[name][0])
        assert torch.equal(after[name][1], payload), name

    # The re-saved artifact must remain loadable.
    GPTQModel.load(str(resaved), backend=BACKEND.EXL3_TORCH, device="cpu")
