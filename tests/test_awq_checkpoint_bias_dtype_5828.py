# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""BF16 GPT-QModel load and save preserve a checkpoint-owned FP32 AWQ bias."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from gptqmodel import GPTQModel
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.ports.native.llama.configuration_llama import LlamaConfig
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, AWQConfig
from gptqmodel.utils.importer import build_kernel_support_maps


QPROJ = "model.layers.0.self_attn.q_proj"
BIAS = -257


def _build_awq_checkpoint(path: Path) -> None:
    """Build the issue's tiny Llama AWQ checkpoint without a network download."""

    path.mkdir(parents=True)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=64,
        attention_bias=True,
    )
    torch.manual_seed(1)
    model = LlamaQModel.loader.from_config(config, dtype=torch.float16)
    qcfg = AWQConfig(bits=4, group_size=32, sym=False, offload_to_disk=False)
    qcfg.meta["quantizer"] = ["gptqmodel:7.11.6"]
    for name, linear in list(model.named_modules()):
        if not isinstance(linear, torch.nn.Linear) or name == "lm_head":
            continue
        quant = AwqTorchLinear(
            bits=4,
            group_size=32,
            desc_act=False,
            sym=False,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            register_buffers=True,
            format=FORMAT.GEMM,
        )
        scales = torch.full(
            (linear.out_features, linear.in_features // 32), 1 / 32, dtype=torch.float16
        )
        zeros = torch.full_like(scales, 8)
        if name == QPROJ:
            scales.fill_(0.125)
            linear.weight = torch.nn.Parameter(torch.full_like(linear.weight, 0.125))
            linear.bias = torch.nn.Parameter(
                torch.full_like(linear.bias, BIAS, dtype=torch.float32)
            )
        quant.pack(linear, scales, zeros)
        model.set_submodule(name, quant)
    state = {
        name: value.clone()
        for name, value in model.state_dict().items()
        if not name.endswith(".qzeros_format")
    }
    config.quantization_config = qcfg.to_dict()
    config.save_pretrained(path)
    qcfg.save_pretrained(path)
    save_file(state, str(path / "model.safetensors"), metadata={"format": "pt"})
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "x": 1}, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(path)


def _load_bf16(path: Path):
    return GPTQModel.load(
        str(path),
        device="cpu",
        backend="awq_torch",
        dtype=torch.bfloat16,
        local_files_only=True,
    )


def _assert_fp32_bias_and_bf16_output(model):
    projection = model.model.get_submodule(QPROJ)
    x = torch.full((1, 64), 32, dtype=torch.bfloat16)
    with torch.inference_mode():
        output = projection(x)

    # The exact dot is 32 * 64 * 0.125 - 257 = -1. BF16 rounds -257 to -256,
    # which cancels the product and returns 0.
    torch.testing.assert_close(projection.bias, torch.full((64,), BIAS, dtype=torch.float32))
    torch.testing.assert_close(output, torch.full((1, 64), -1, dtype=torch.bfloat16))


def test_bf16_load_and_resave_keep_fp32_awq_bias(tmp_path):
    # Isolate the load from the cold kernel-registry import (#5776).
    build_kernel_support_maps()
    checkpoint = tmp_path / "checkpoint"
    _build_awq_checkpoint(checkpoint)
    loaded = _load_bf16(checkpoint)
    _assert_fp32_bias_and_bf16_output(loaded)

    saved_dir = tmp_path / "saved"
    loaded.save(str(saved_dir))
    saved = load_file(str(saved_dir / "model.safetensors"))[QPROJ + ".bias"]
    torch.testing.assert_close(saved, torch.full((64,), BIAS, dtype=torch.float32))
    _assert_fp32_bias_and_bf16_output(_load_bf16(saved_dir))
