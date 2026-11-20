# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""BF16 Transformers load must keep a checkpoint-owned FP32 AWQ bias (#5828).

The HF bridge installs a quantized shell before Transformers assigns checkpoint
tensors. Transformers materializes each floating tensor at the shell slot's
dtype, so an FP32 bias in an AWQ checkpoint was rounded to BF16 (-257 -> -256)
and ``save_pretrained`` published the rounded value. The native loader keeps
checkpoint-owned buffers at their stored dtype; the bridge must publish that
dtype on the shell too, while the activation dtype still follows the caller.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)
from transformers import AwqConfig as HfAwqConfig

from gptqmodel import GPTQModel
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, AWQConfig
from gptqmodel.utils.importer import build_kernel_support_maps


QPROJ = "model.layers.0.self_attn.q_proj"
INPUT = torch.tensor([[1, 9, 2, 5]])
BIAS = -257


def _build_awq_checkpoint(path: Path) -> dict:
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
    model = LlamaForCausalLM(config).half()
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
    return state


def _hf_load(path: Path):
    return AutoModelForCausalLM.from_pretrained(
        str(path),
        device_map="cpu",
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=HfAwqConfig(backend="torch_awq"),
    )


@pytest.fixture(scope="module")
def awq_checkpoint(tmp_path_factory):
    # Isolate the load from the cold kernel-registry import (#5776).
    build_kernel_support_maps()
    path = tmp_path_factory.mktemp("awq_bias_5828") / "checkpoint"
    _build_awq_checkpoint(path)
    return path


def test_bf16_hf_load_keeps_fp32_awq_bias(awq_checkpoint, tmp_path, monkeypatch):
    # Transformers' AWQ quantizer rewrites bf16 to fp16 when CUDA/XPU is
    # visible. The rule under test is the CPU/bf16 one, so pin the visibility
    # probe instead of depending on the test host's accelerators.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    xpu = getattr(torch, "xpu", None)
    if xpu is not None:
        monkeypatch.setattr(xpu, "is_available", lambda: False)

    x = torch.full((1, 64), 32, dtype=torch.bfloat16)
    native = GPTQModel.load(
        str(awq_checkpoint), device="cpu", backend="awq_torch", dtype=torch.bfloat16
    )
    native_q = native.model.get_submodule(QPROJ)
    with torch.inference_mode():
        native_out = native_q(x)

    hf = _hf_load(awq_checkpoint)
    hf_q = hf.get_submodule(QPROJ)
    with torch.inference_mode():
        hf_out = hf_q(x)

    # The exact dot is 32 * 64 * 0.125 - 257 = -1. BF16 rounds -257 to -256,
    # which cancels the product and returns 0.
    assert native_q.bias.dtype == torch.float32
    assert float(native_q.bias[0]) == BIAS
    assert float(native_out[0, 0]) == -1.0
    assert hf_q.bias.dtype == torch.float32
    assert float(hf_q.bias[0]) == BIAS
    assert float(hf_out[0, 0]) == -1.0
    assert hf_out.dtype == torch.bfloat16

    saved_dir = tmp_path / "saved"
    hf.save_pretrained(saved_dir)
    saved = load_file(str(saved_dir / "model.safetensors"))[QPROJ + ".bias"]
    assert saved.dtype == torch.float32
    assert float(saved[0]) == BIAS

    reloaded = _hf_load(saved_dir)
    reloaded_q = reloaded.get_submodule(QPROJ)
    assert reloaded_q.bias.dtype == torch.float32
    assert float(reloaded_q.bias[0]) == BIAS
