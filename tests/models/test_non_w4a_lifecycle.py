# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Ordinary quantization never executes the opt-in W4A lifecycle."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.looper import gptq_processor
from gptqmodel.models import writer
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear, dequantize_model
from gptqmodel.quantization import FORMAT, METHOD, QuantizeConfig, RTNConfig
from gptqmodel.utils.importer import AUTO_BACKEND_KERNEL_MAPPING


def _forbid_w4a(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("Ordinary quantization entered an opt-in W4A path")

    # These guards also work against the PR base, which has no W4A helpers.
    for name in ("_record_activation_amax", "begin_activation_scale_probe", "end_activation_scale_probe"):
        if hasattr(gptq_processor.GPTQProcessor, name):
            monkeypatch.setattr(gptq_processor.GPTQProcessor, name, unexpected)
    for owner, names in (
        (gptq_processor, ("enable_w4afp8_replay", "enable_w4a_nvfp4_replay")),
        (writer, ("_load_native_w4a_for_save",)),
    ):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, unexpected)
    for name, module in tuple(sys.modules.items()):
        if name.startswith("gptqmodel.nn_modules.qlinear.w4a"):
            for attr in ("install_w4a_llama_replay", "install_w4a_llama_stream", "pack_activation",
                         "round_w4a_replay_operand", "set_w4a_replay_enabled"):
                if hasattr(module, attr):
                    monkeypatch.setattr(module, attr, unexpected)
            for attr in ("W4AFP8Linear", "W4ANVFP4Linear"):
                cls = getattr(module, attr, None)
                if cls is not None:
                    monkeypatch.setattr(cls, "post_init", unexpected)
                    monkeypatch.setattr(cls, "forward", unexpected)


CASES = [
    ("gptq_lazy", torch.float16, "cpu"),
    ("gptq", torch.float16, "cuda"),
    ("gptq", torch.bfloat16, "cuda"),
    ("gptq_act_order", torch.bfloat16, "cuda"),
    ("gptq_asym", torch.bfloat16, "cuda"),
    ("gptq_rotation", torch.bfloat16, "cuda"),
    ("awq", torch.float16, "cuda"),
    ("awq", torch.bfloat16, "cuda"),
    ("awq_lazy", torch.float16, "cuda"),
    ("rtn", torch.float16, "cuda"),
    ("rtn", torch.bfloat16, "cuda"),
    ("rtn_lazy", torch.float16, "cuda"),
]


@pytest.mark.parametrize("lane,dtype,device", CASES,
                         ids=[f"{lane}-{str(dtype).split('.')[-1]}-{device}" for lane, dtype, device in CASES])
def test_quantize_save_reload_without_activation_policy(tmp_path, monkeypatch, lane, dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for this lifecycle case")
    _forbid_w4a(monkeypatch)
    torch.manual_seed(413)
    source, saved, exported = (tmp_path / name for name in ("source", "quantized", "exported"))
    config = LlamaConfig(vocab_size=32, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64,
                         bos_token_id=2, eos_token_id=3, pad_token_id=1)
    LlamaForCausalLM(config).save_pretrained(source)
    words = ["[UNK]", "[PAD]", "<s>", "</s>", "a", "fox", "runs", "under", "the", "moon"]
    inner = Tokenizer(models.WordLevel(vocab={word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    inner.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner, unk_token="[UNK]", pad_token="[PAD]",
                                        bos_token="<s>", eos_token="</s>")
    tokenizer.save_pretrained(source)
    lazy = lane.endswith("_lazy")
    method = lane.removesuffix("_lazy")
    kwargs = {"bits": 4, "group_size": 32, "desc_act": False, "sym": True,
              "offload_to_disk": lazy, "device": device}
    backend = BACKEND.GPTQ_TORCH
    if method == "awq":
        kwargs.update(method=METHOD.AWQ, format=FORMAT.GEMM, sym=False)
        backend = BACKEND.AWQ_TORCH
    elif lane == "gptq_act_order":
        kwargs.update(desc_act=True)
    elif lane == "gptq_asym":
        kwargs.update(sym=False)
    elif lane == "gptq_rotation":
        kwargs.update(rotation="hadamard")
    qcfg = (RTNConfig if method == "rtn" else QuantizeConfig)(**kwargs)
    assert "activation" not in qcfg.to_dict()
    for formats in AUTO_BACKEND_KERNEL_MAPPING.values():
        for candidates in formats.values():
            assert not any("w4a" in candidate.value for candidate in candidates)

    wrapper = GPTQModel.load(str(source), qcfg, dtype=dtype)
    assert (wrapper.turtle_model is not None) == lazy
    materialized = []
    materialize = wrapper.shell_module_materialize

    def record_materialize(*args, **kwargs):
        materialized.append(kwargs.get("module_path") or getattr(kwargs.get("named_module"), "full_name", None))
        return materialize(*args, **kwargs)

    monkeypatch.setattr(wrapper, "shell_module_materialize", record_materialize)
    calibration = ["a fox runs under the moon " * 3, "the moon a fox " * 5]
    wrapper.quantize(None if method == "rtn" else calibration, batch_size=1, backend=backend)
    if lazy:
        assert materialized
        if method != "rtn":  # RTN materializes weights without StageInputsCapture.
            assert "model.layers.0" in materialized
    quantized = [module for module in wrapper.model.modules() if isinstance(module, BaseQuantLinear)]
    assert len(quantized) == 14
    assert not any(type(module).__name__.startswith("W4A") for module in quantized)
    assert not hasattr(wrapper.model, "_w4a_stream_mode")
    wrapper.save(str(saved))
    assert "activation" not in json.loads((saved / "quantize_config.json").read_text())

    loaded = GPTQModel.load(str(saved), backend=backend, device=device, dtype=dtype)
    assert not any(t.is_meta for t in loaded.model.model.layers[0].parameters())
    ids = tokenizer("a fox runs", return_tensors="pt").input_ids.to(device)
    seen = []

    def capture(_module, args, output):
        assert isinstance(args[0], torch.Tensor) and isinstance(output, torch.Tensor)
        assert args[0].dtype == output.dtype == dtype
        seen.append(_module)

    handles = [module.register_forward_hook(capture) for module in loaded.model.modules()
               if isinstance(module, BaseQuantLinear)]
    with torch.inference_mode():
        logits = loaded.model(input_ids=ids, use_cache=False).logits
        generated = loaded.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                    max_new_tokens=2, do_sample=False)
    for handle in handles:
        handle.remove()
    assert len(set(seen)) == 14
    assert torch.isfinite(logits).all()
    assert not hasattr(loaded.model, "_w4a_stream_mode")
    loaded.save(str(exported))
    original_state = load_file(str(saved / "model.safetensors"))
    exported_state = load_file(str(exported / "model.safetensors"))
    assert original_state.keys() == exported_state.keys()
    for key in original_state:
        if not original_state[key].is_floating_point():
            torch.testing.assert_close(exported_state[key], original_state[key], rtol=0, atol=0)
    assert not any("activation_global_scale" in key or "_weight_e4m3" in key for key in exported_state)
    # Keep deterministic artifacts so this same test can be run on the PR base
    # and compared tensor-for-tensor, including the unchanged legacy exporter.
    save_file({"logits": logits.cpu().contiguous(), "tokens": generated.cpu().contiguous()},
              str(tmp_path / "inference.safetensors"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_ordinary_gptq_dequantization_preserves_bias_contract(dtype, bias):
    model = torch.nn.Module()
    model.config = SimpleNamespace(quantization_config={"bits": 4})
    model.proj = TorchLinear(bits=4, group_size=32, sym=False, desc_act=False,
                             in_features=128, out_features=128, bias=bias, register_buffers=True).to(dtype)
    model.proj.qweight.fill_(0x76543210)
    model.proj.qzeros.zero_()
    model.proj.qzero_format(2)
    model.proj.scales.copy_(torch.arange(1, 5)[:, None] / 8.)
    if bias:
        model.proj.bias.copy_(torch.linspace(-1, 1, 128))
    expected_bias = model.proj.bias.half() if bias else None
    # Independently unpack the repeated hexadecimal fixture and apply the
    # four group scales; do not use the implementation's dequantizer as oracle.
    rows = torch.arange(128)
    weight = (rows.remainder(8) * ((rows // 32 + 1) / 8.)).half()[:, None].expand(128, 128)
    x = torch.linspace(-1, 1, 128).half()[None]
    expected = torch.nn.functional.linear(x, weight.T, expected_bias)
    dequantize_model(model)
    assert type(model.proj) is torch.nn.Linear
    assert (model.proj.bias is not None) == bias
    assert model.proj.weight.dtype == torch.float16
    assert not hasattr(model.config, "quantization_config")
    torch.testing.assert_close(model.proj(x), expected, rtol=0, atol=0)
