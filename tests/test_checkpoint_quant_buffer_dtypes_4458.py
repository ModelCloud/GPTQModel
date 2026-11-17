# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Quantized-module buffers load at their checkpoint dtype (#4458, #4452).

Accelerate casts every floating buffer to the requested model dtype. A
quantized module's buffers are quantization state, not activation operands, so
that cast rounded QQQ's FP16 ``s_group`` and FP32 ``s_channel`` (and the same
class of buffer in other formats) to BF16 and made a reloaded model worse than
the in-session one. Only the buffers a kernel declares in
``MODEL_DTYPE_BUFFERS`` follow the model dtype.

The re-save shell restores checkpoint dtypes for every model buffer. It must
skip checkpoint tensors whose module no longer exists (#4452).
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.utils.model import load_checkpoint_in_model_then_tie_weights


LOAD_DTYPE = torch.bfloat16


def _gptq():
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    return TorchLinear(
        bits=4, group_size=32, desc_act=False, sym=True, in_features=128,
        out_features=64, bias=True, register_buffers=True,
    )


def _awq():
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    return AwqTorchLinear(
        bits=4, group_size=32, desc_act=False, sym=True, in_features=128,
        out_features=64, bias=True, register_buffers=True,
    )


def _qqq():
    from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear

    return QQQTorchLinear(
        bits=4, group_size=128, desc_act=False, sym=True, in_features=256,
        out_features=128, bias=True,
    )


def _fp8():
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    from gptqmodel.quantization.config import FP8Config

    return TorchFP8Linear(
        bits=8, group_size=-1, desc_act=False, sym=True, in_features=128,
        out_features=64, bias=True, register_buffers=True,
        **FP8Config(weight_scale_method="row").quant_linear_init_kwargs(),
    )


def _gguf():
    from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
    from gptqmodel.quantization.config import GGUFConfig

    return GGUFTorchLinear(
        bits=GGUFConfig(bits="q4_k_m").runtime_bits, group_size=-1, desc_act=False,
        sym=True, in_features=256, out_features=64, bias=True, register_buffers=True,
    )


def _exl3():
    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear

    return ExllamaV3TorchLinear.from_tensors(
        in_features=128,
        out_features=128,
        name="proj",
        tensors={
            "trellis": torch.zeros((8, 8, 64), dtype=torch.int16),
            "suh": torch.ones(128, dtype=torch.float16),
            "svh": torch.ones(128, dtype=torch.float16),
            "bias": torch.zeros(128, dtype=torch.float16),
        },
    )


def _paroquant():
    from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear

    return ParoLinear(
        bits=4, group_size=32, desc_act=False, sym=True, in_features=128,
        out_features=64, bias=True, krot=4, register_buffers=True,
    )


def _macaq():
    from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear

    return MacaqLinear(
        bits=4, group_size=32, in_features=128, out_features=64, bias=True,
        activation_precision="w4a16", dtype=LOAD_DTYPE,
    )


def _vptq():
    from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear

    module = VPTQLinear(
        bits=2, in_features=64, out_features=64, bias=True,
        vector_lens=[-1, 4], num_centroids=[-1, 16], num_res_centroids=[-1, -1],
        group_size=64, group_num=-1, enable_norm=True, norm_dim=0,
        enable_perm=False, dtype=torch.float16,
    )
    module._allocate_buffers(bias=True)
    return module


def _bitsandbytes(bits):
    from gptqmodel.nn_modules.qlinear.bitsandbytes import (
        BITSANDBYTES_AVAILABLE,
        BitsAndBytesLinear,
    )

    if not BITSANDBYTES_AVAILABLE:
        pytest.skip("optional bitsandbytes dependency unavailable")
    return BitsAndBytesLinear(
        bits=bits, group_size=-1, desc_act=False, sym=True, in_features=128,
        out_features=64, bias=True, register_buffers=True, dtype=torch.float16,
    )


FACTORIES = {
    "gptq": _gptq,
    "awq": _awq,
    "qqq": _qqq,
    "fp8": _fp8,
    "gguf": _gguf,
    "exl3": _exl3,
    "paroquant": _paroquant,
    "macaq": _macaq,
    "vptq": _vptq,
    "bitsandbytes_4bit": lambda: _bitsandbytes(4),
    "bitsandbytes_8bit": lambda: _bitsandbytes(8),
}


class _Model(nn.Module):
    def __init__(self, factory):
        super().__init__()
        self.proj = factory()

    def tie_weights(self):
        pass


def _fill_floating_buffers(module: nn.Module, seed: int) -> None:
    """Give every floating buffer values that BF16 cannot represent exactly."""

    generator = torch.Generator().manual_seed(seed)
    for name, buffer in module.named_buffers():
        if not buffer.is_floating_point() or buffer.numel() == 0:
            continue
        values = torch.rand(buffer.shape, generator=generator) * 0.5 + 0.25
        buffer.copy_((values + 1e-3 / 3).to(buffer.dtype))


@pytest.mark.parametrize("kind", sorted(FACTORIES))
def test_quant_buffers_load_at_checkpoint_dtype(tmp_path, kind):
    factory = FACTORIES[kind]
    source = _Model(factory)
    _fill_floating_buffers(source, seed=4458)
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {name: tensor.contiguous() for name, tensor in source.state_dict().items()},
        str(checkpoint),
        metadata={"format": "pt"},
    )

    target = _Model(factory)
    load_checkpoint_in_model_then_tie_weights(
        target,
        checkpoint=str(checkpoint),
        dtype=LOAD_DTYPE,
        device_map={"": "cpu"},
    )

    model_dtype_buffers = type(target.proj).MODEL_DTYPE_BUFFERS
    loaded = dict(target.named_buffers())
    exact = []
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            stored = handle.get_tensor(name)
            # VPTQ codebooks are `nn.Embedding` parameters, not module buffers.
            if not stored.is_floating_point() or name not in loaded:
                continue
            value = loaded[name]
            if name.rpartition(".")[2] in model_dtype_buffers:
                # Activation-side operands keep following the model dtype.
                assert value.dtype == LOAD_DTYPE, (name, value.dtype)
                assert torch.equal(value, stored.to(LOAD_DTYPE)), name
                continue
            assert value.dtype == stored.dtype, (name, value.dtype, stored.dtype)
            assert torch.equal(
                value.contiguous().view(torch.uint8), stored.contiguous().view(torch.uint8)
            ), name
            if stored.dtype != LOAD_DTYPE:
                exact.append(name)
    if kind not in {"gguf", "macaq"}:
        # Every other format has quantization state stored at a dtype the BF16
        # load would round; GGUF and MACaQ keep theirs in integer containers.
        assert exact, kind


def _save_tokenizer(directory):
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.train_from_iterator(
        ["tiny calibration text", "another dtype round trip"],
        trainer=trainers.WordLevelTrainer(
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"], vocab_size=64
        ),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    ).save_pretrained(directory)


def _tiny_llama(directory):
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
    )
    LlamaForCausalLM(config).to(dtype=torch.bfloat16).save_pretrained(directory)
    _save_tokenizer(directory)
    return directory


CALIBRATION = [
    {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]),
        "attention_mask": torch.ones((1, 8), dtype=torch.long),
    }
]


@pytest.fixture(scope="module")
def qqq_checkpoint(tmp_path_factory):
    from gptqmodel.quantization.config import QQQConfig

    root = tmp_path_factory.mktemp("qqq_4458")
    source = _tiny_llama(root / "source")
    model = GPTQModel.load(
        str(source),
        quantize_config=QQQConfig(
            bits=4, group_size=128, desc_act=False, device="cpu", offload_to_disk=False
        ),
        backend=BACKEND.TORCH,
    )
    model.quantize(CalibrationConfig(dataset=CALIBRATION, batch_size=1, min_length=1))
    quantized = root / "quantized"
    model.save(str(quantized))
    return quantized


def test_qqq_reload_keeps_exact_scales_under_bf16(qqq_checkpoint):
    loaded = GPTQModel.load(str(qqq_checkpoint), backend=BACKEND.TORCH, device="cpu")
    buffers = dict(loaded.model.named_buffers())
    checked = 0
    with safe_open(str(qqq_checkpoint / "model.safetensors"), framework="pt") as handle:
        for name in handle.keys():
            if not name.endswith((".s_group", ".s_channel")):
                continue
            stored = handle.get_tensor(name)
            if stored.numel() == 0:
                continue
            assert buffers[name].dtype == stored.dtype, name
            assert torch.equal(buffers[name].cpu(), stored), name
            checked += 1
    # Seven s_channel tensors plus down_proj's grouped s_group.
    assert checked == 8


def test_resave_skips_checkpoint_tensors_without_a_module(tmp_path, qqq_checkpoint):
    """A legacy per-layer ``rotary_emb.inv_freq`` key has no module today (#4452)."""

    checkpoint = tmp_path / "legacy"
    checkpoint.mkdir()
    for path in qqq_checkpoint.iterdir():
        (checkpoint / path.name).write_bytes(path.read_bytes())
    shard = checkpoint / "model.safetensors"
    state = load_file(str(shard))
    orphan = "model.layers.0.self_attn.rotary_emb.inv_freq"
    state[orphan] = torch.arange(16, dtype=torch.float32)
    save_file(state, str(shard), metadata={"format": "pt"})

    loaded = GPTQModel.load(str(checkpoint), backend=BACKEND.TORCH, device="cpu")
    resaved = tmp_path / "resaved"
    loaded.save(str(resaved))

    with safe_open(str(resaved / "model.safetensors"), framework="pt") as handle:
        keys = set(handle.keys())
    assert orphan not in keys
    assert {name for name in state if name != orphan} <= keys
    json.loads((resaved / "config.json").read_text(encoding="utf-8"))
