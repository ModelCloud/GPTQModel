# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4554.

AWQ ``FORMAT.GEMV`` had no in-session kernel for the CPU the quantize loop
leaves the packed layers on: the native ``AwqGEMVLinear`` is CUDA-only and the
Triton reader needs an accelerator. ``quantize()`` therefore raised after the
whole calibration and the documented quantize -> save flow lost the result.
These tests pin the CPU reader and the end-to-end flow.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer, models, pre_tokenizers
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.models._const import DEVICE
from gptqmodel.models.base import rehome_quantized_kernel, select_in_session_quant_linear
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear, AwqGEMVTorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import AWQConfig


HIDDEN_SIZE = 64
VOCAB_SIZE = 64


def _write_tiny_llama(directory: Path) -> None:
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    LlamaForCausalLM(config).half().eval().save_pretrained(directory)
    tokenizer = Tokenizer(models.WordLevel(
        vocab={"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "hello": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    ).save_pretrained(directory)


def _calibration():
    return [
        {
            "input_ids": torch.randint(4, VOCAB_SIZE, (1, 32)),
            "attention_mask": torch.ones(1, 32, dtype=torch.long),
        }
        for _ in range(4)
    ]


def test_awq_gemv_cpu_insession_kernel_is_the_documented_fallback():
    qcfg = AWQConfig(bits=4, format="gemv", group_size=128, offload_to_disk=False)

    selected = select_in_session_quant_linear(qcfg, torch.device("cpu"))

    assert selected is AwqGEMVTorchLinear
    assert DEVICE.CPU in selected.SUPPORTS_DEVICES
    # Explicit-only: accelerator AUTO order must not change (native CUDA GEMV
    # stays first there, and ROCm keeps its actionable refusal).
    assert selected.SUPPORTS_FORMATS[FORMAT.GEMV] == 0


def test_rehome_switches_native_gemv_module_to_cpu_reader():
    qcfg = AWQConfig(bits=4, format="gemv", group_size=128, offload_to_disk=False)
    module = AwqGEMVLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=128,
        out_features=128,
        bias=False,
        pack_dtype=torch.int32,
        register_buffers=True,
    )

    class _Tree(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = module

    fake_model = SimpleNamespace(
        model=_Tree(),
        quantize_config=qcfg,
        qlinear_kernel=AwqGEMVLinear,
    )

    rehome_quantized_kernel(fake_model)

    assert type(module) is AwqGEMVTorchLinear
    assert fake_model.qlinear_kernel is AwqGEMVTorchLinear
    assert module.qweight.device.type == "cpu"


def test_awq_gemv_cpu_reader_matches_the_gemm_reader_numerics():
    # Both layouts encode the same codes/scales, so the portable GEMV reader
    # must land on the portable GEMM reader's result (the GEMV path keeps the
    # reduction wide, so the residual is one FP16 rounding).
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear

    torch.manual_seed(4554)
    in_features = out_features = 256
    group_size = 128
    linear = nn.Linear(in_features, out_features, bias=True).half()
    scales = torch.full((out_features, in_features // group_size), 0.01, dtype=torch.float16)
    zeros = torch.full((out_features, in_features // group_size), 8.0, dtype=torch.float16)
    kwargs = {
        "bits": 4,
        "group_size": group_size,
        "sym": False,
        "desc_act": False,
        "in_features": in_features,
        "out_features": out_features,
        "bias": True,
        "pack_dtype": torch.int32,
        "register_buffers": True,
    }
    gemv = AwqGEMVTorchLinear(**kwargs)
    gemv.pack(linear, scales, zeros)
    gemm = AwqTorchLinear(**kwargs)
    gemm.pack(linear, scales, zeros)

    inputs = torch.randn(3, in_features).half()
    with torch.inference_mode():
        gemv_out = gemv(inputs)
        gemm_out = gemm(inputs)

    assert gemv_out.dtype == inputs.dtype
    torch.testing.assert_close(gemv_out, gemm_out, rtol=2e-2, atol=2e-2)


def test_awq_gemv_quantize_save_and_reload_on_cpu(tmp_path):
    source = tmp_path / "source"
    _write_tiny_llama(source)
    out = tmp_path / "out"
    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        format=FORMAT.GEMV,
        device="cpu",
        offload_to_disk=False,
    )
    model = GPTQModel.load(str(source), quantize_config=qcfg, dtype=torch.float16)

    # The CPU quantize path used to raise before calibration because no packing
    # kernel advertised CPU support.
    model.quantize(_calibration(), batch_size=1, calibration_data_min_length=1)

    quant_linear = [m for m in model.model.modules() if isinstance(m, AwqGEMVLinear)]
    assert quant_linear
    assert all(isinstance(m, AwqGEMVTorchLinear) for m in quant_linear)
    assert model.qlinear_kernel is AwqGEMVTorchLinear

    ids = torch.tensor([[1, 4, 4, 4]])
    with torch.inference_mode():
        logits = model(ids).logits
    assert logits.shape == (1, 4, VOCAB_SIZE)
    assert torch.isfinite(logits).all()

    model.save(str(out))
    saved = json.loads((out / "config.json").read_text())
    assert saved["quantization_config"]["format"] == "gemv"

    # The checkpoint keeps the requested GEMV packing: qweight is
    # [out_features, in_features / 8], not the GEMM [in_features, out/8] layout.
    tensors = load_file(str(out / "model.safetensors"))
    assert tensors["model.layers.0.self_attn.q_proj.qweight"].shape == (
        HIDDEN_SIZE,
        HIDDEN_SIZE // 8,
    )

    reloaded = GPTQModel.load(str(out), dtype=torch.float16, device="cpu")
    assert reloaded.qlinear_kernel is AwqGEMVTorchLinear
    with torch.inference_mode():
        relogits = reloaded(ids).logits
    assert torch.equal(relogits, logits)
