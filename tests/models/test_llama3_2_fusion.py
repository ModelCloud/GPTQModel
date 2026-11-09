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
from gptqmodel.quantization import FORMAT, QuantizeConfig
from gptqmodel.utils.backend import BACKEND
from tests.benchmark_llama_fusion import (
    assert_fusion_drift_within_ulp,
    assert_greedy_divergence_is_tie_break,
    logit_ulp_envelope,
    storage_footprint,
)


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
    model = GPTQModel.load(path, backend=backend, device="cuda:0",
                          dtype=torch.float16, attn_implementation="eager")
    assert model._fusion_plan is None
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
        # Teacher-force both modes on the same token sequence so that any
        # disagreement can be judged against the margin the unfused model
        # itself reported at that position.
        teacher_unfused = model.model(separate_tokens, use_cache=False).logits
    original_storage = storage_footprint(model)
    plan = model.fuse()
    # One qkv and one gate/up group per decoder layer: derive it, never pin a
    # literal, so the test also serves checkpoints other than Llama 3.2 1B.
    layers = model.model.config.num_hidden_layers
    group_count = 2 * layers
    assert len(plan.groups) == group_count
    with torch.inference_mode():
        model.model(**inputs)  # Allocate packed fusion operands before timing.
        fused, fused_seconds = measure(model)
        fused_tokens = model.model.generate(**inputs, max_new_tokens=12, do_sample=False)
        teacher_fused = model.model(separate_tokens, use_cache=False).logits
    # Composed model drift accumulates per-projection rounding. Kernel-level
    # elementwise bounds remain in the independent native projection tests.
    # Fusion widens N, so Marlin may pick a different reduction tree: require the
    # drift to stay inside a dtype-derived ULP envelope and require greedy output
    # to diverge only at near-ties, instead of demanding an identical reduction.
    logit_drift = assert_fusion_drift_within_ulp(fused, baseline)
    greedy = assert_greedy_divergence_is_tie_break(teacher_fused, teacher_unfused,
                                                   label=f"{method} greedy decode")
    assert logit_drift["top1_agreement"] == 1.0
    assert plan.to_dict()["fused_calls"] > 0
    assert set(model.model.state_dict()) == keys
    fused_storage = storage_footprint(model, plan)
    extra_packed_bytes = fused_storage["packed_operand_storage_bytes"] - original_storage["packed_operand_storage_bytes"]
    assert extra_packed_bytes == 0
    print({"method": method, "max_logit_error": (fused - baseline).abs().max().item(),
           "ulp_envelope": logit_ulp_envelope(baseline), "greedy_tie_breaks": greedy,
           "separate_ms": separate_seconds * 1000, "fused_ms": fused_seconds * 1000,
           "extra_packed_bytes": extra_packed_bytes, "storage": fused_storage,
           "selected_kernel": kernel, "generated_ids": fused_tokens.tolist()})
    destination = tmp_path / method
    model.save(str(destination))
    expected = fused.cpu()
    del baseline, fused, model, plan
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = GPTQModel.load(str(destination), backend=backend, device="cuda:0", fuse=True,
                             dtype=torch.float16, attn_implementation="eager")
    reloaded_plan = reloaded.fuse()
    assert len(reloaded_plan.groups) == group_count  # Explicit opt-in after load.
    with torch.inference_mode():
        actual = reloaded.model(**inputs).logits.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    reloaded.fuse(False)
    assert reloaded._fusion_plan is None


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
    qcfg = QuantizeConfig(bits=4, group_size=128, sym=True, desc_act=False, format=FORMAT.GPTQ_V2,
                         offload_to_disk=False)
    options = {"fuse": True} if automatic else {}
    model = GPTQModel.load(str(source), qcfg, device="cuda:0", dtype=torch.float16, **options,
                          attn_implementation="eager")
    assert not model.quantized
    assert model._fusion_plan is None
    model.quantize(["a b c d " * 16, "d c b a " * 16], backend=BACKEND.GPTQ_TORCH)
    assert model.quantized
    if automatic:
        plan = model.fuse()
        assert len(plan.groups) == 2
    else:
        assert model._fusion_plan is None
        plan = model.fuse()
        assert len(plan.groups) == 2
    model.to("cuda:0").eval()
    ids = torch.tensor([[2, 4, 5, 6]], device="cuda:0")
    with torch.inference_mode():
        expected = model.model(ids, use_cache=False).logits.cpu()
    before = {name: tensor.cpu().clone() for name, tensor in model.model.state_dict().items()}
    destination = tmp_path / "quantized"
    model.save(str(destination))
    reloaded = GPTQModel.load(str(destination), backend=BACKEND.GPTQ_TORCH, device="cuda:0",
                             dtype=torch.float16, attn_implementation="eager")
    assert reloaded._fusion_plan is None
    reloaded_plan = reloaded.fuse()
    assert len(reloaded_plan.groups) == 2
    with torch.inference_mode():
        actual = reloaded.model(ids, use_cache=False).logits.cpu()
    from safetensors.torch import load_file
    raw = load_file(str(destination / "model.safetensors"))
    assert {key for key in before if key.endswith(("qweight", "qzeros", "scales"))} <= set(raw)
    for key, tensor in raw.items():
        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
