# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Local real-checkpoint fusion acceptance tests; no checkpoint downloads."""

import gc
import os
import time
from pathlib import Path

import pytest
import torch

from gptqmodel import GPTQModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils.backend import BACKEND


@pytest.mark.parametrize("method,path,backend", [
    ("gptq", "/monster/data/model/Llama-3.2-1B-Instruct-HF-GPTQV2", BACKEND.AUTO),
    ("awq", "/monster/data/model/Llama-3.2-1B-Instruct-AWQ-bit4-g128-symTrue-descFalse", BACKEND.AUTO),
])
def test_llama32_1b_fusion_save_reload(method, path, backend, tmp_path):
    path = os.environ.get(f"GPTQMODEL_LLAMA32_FUSION_{method.upper()}", path)
    if not Path(path).is_dir():
        pytest.skip(f"Local Llama 3.2 1B {method} checkpoint unavailable: {path}")
    if not torch.cuda.is_available():
        pytest.skip("Real 1B acceptance test requires a CUDA accelerator")
    model = GPTQModel.load(path, backend=backend, device="cuda:0", fuse=False,
                          dtype=torch.float16, attn_implementation="eager")
    assert model.fusion_plan() is None
    assert model.quantize_config.bits == 4 and model.quantize_config.group_size == 128
    assert model.quantize_config.sym and not model.quantize_config.desc_act
    kernel = "MarlinLinear" if method == "gptq" else "AwqMarlinLinear"
    assert {type(m).__name__ for m in model.model.modules() if isinstance(m, BaseQuantLinear)} == {kernel}
    inputs = model.tokenizer("The capital of France is", return_tensors="pt").to("cuda:0")
    keys = set(model.model.state_dict())

    def measure(wrapper):
        timings = []
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = wrapper.model(**inputs).logits
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        return output, sorted(timings)[1]

    with torch.inference_mode():
        baseline, separate_seconds = measure(model)
        separate_tokens = model.model.generate(**inputs, max_new_tokens=12, do_sample=False)
    model.fuse()
    group_count = 32
    assert len(model.fusion_plan().groups) == group_count
    with torch.inference_mode():
        model.model(**inputs)  # Allocate packed fusion operands before timing.
        fused, fused_seconds = measure(model)
        fused_tokens = model.model.generate(**inputs, max_new_tokens=12, do_sample=False)
    # Composed model drift accumulates per-projection rounding. Kernel-level
    # elementwise bounds remain in the independent native projection tests.
    assert torch.equal(fused.argmax(-1), baseline.argmax(-1))
    assert torch.equal(fused_tokens, separate_tokens)
    assert model.fusion_plan().to_dict()["fused_calls"] > 0
    assert set(model.model.state_dict()) == keys
    packed_bytes = sum(t.numel() * t.element_size() for g in model.fusion_plan().groups
                       for name, t in list(g.packed.named_parameters()) + list(g.packed.named_buffers())
                       if name not in ("g_idx", "g_idx_sort_indices"))
    print({"method": method, "max_logit_error": (fused - baseline).abs().max().item(),
           "separate_ms": separate_seconds * 1000, "fused_ms": fused_seconds * 1000,
           "extra_packed_bytes": packed_bytes, "selected_kernel": kernel, "generated_ids": fused_tokens.tolist()})
    destination = tmp_path / method
    model.save(str(destination))
    expected = fused.cpu()
    del baseline, fused, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = GPTQModel.load(str(destination), backend=backend, device="cuda:0",
                             dtype=torch.float16, attn_implementation="eager")
    assert len(reloaded.fusion_plan().groups) == group_count  # Automatic after load.
    with torch.inference_mode():
        actual = reloaded.model(**inputs).logits.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reloaded.fuse(False)
    assert reloaded.fusion_plan() is None


@pytest.mark.parametrize("automatic", [True, False])
def test_tiny_llama_fusion_after_fresh_quantization(tmp_path, automatic):
    if not torch.cuda.is_available():
        pytest.skip("Fresh quantization lifecycle requires a CUDA accelerator")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    source = tmp_path / "source"
    config = LlamaConfig(vocab_size=8, hidden_size=128, intermediate_size=256, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                         bos_token_id=2, eos_token_id=3, pad_token_id=1)
    config._attn_implementation = "eager"
    LlamaForCausalLM(config).save_pretrained(source)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "<s>": 2, "</s>": 3,
                                            "a": 4, "b": 5, "c": 6, "d": 7}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
                                       bos_token="<s>", eos_token="</s>")
    tokenizer.save_pretrained(source)
    qcfg = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False, offload_to_disk=False)
    model = GPTQModel.load(str(source), qcfg, device="cuda:0", dtype=torch.float16, fuse=automatic,
                          attn_implementation="eager")
    assert not model.quantized
    assert model.fusion_plan() is None
    model.quantize(["a b c d " * 16, "d c b a " * 16], backend=BACKEND.GPTQ_TORCH)
    assert model.quantized
    if automatic:
        assert len(model.fusion_plan().groups) == 2
    else:
        assert model.fusion_plan() is None
        model.fuse()
        assert len(model.fusion_plan().groups) == 2
