# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-model resolution evidence for the protocol ``smoothquant`` action.

The CPU toy models cannot show the defect this file pins: a 1-layer model never
reuses a freed activation address. These tests load a real checkpoint and assert
that every matched projection resolves to its *own* layer's norm, which the
previous ``data_ptr()`` resolver failed to do on every model the issue listed.

They skip unless the checkpoint is already in the local Hugging Face cache, so
they do not add a download to CI.
"""

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="real-model resolution needs CUDA")

CASES = (
    ("Qwen/Qwen2.5-0.5B-Instruct", "float16", r".*self_attn\.(q_proj|k_proj|v_proj)$"),
    ("unsloth/Llama-3.2-1B-Instruct", "bfloat16", r".*self_attn\.(q_proj|k_proj|v_proj)$"),
)


def _cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # pragma: no cover - dependency guard
        return False
    return all(
        isinstance(try_to_load_from_cache(model_id, name), str)
        for name in ("config.json", "model.safetensors")
    )


@pytest.mark.parametrize("model_id,dtype,pattern", CASES)
def test_resolution_picks_the_preceding_norm_of_each_layer(model_id, dtype, pattern):
    if not _cached(model_id):
        pytest.skip(f"{model_id} is not in the local Hugging Face cache")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gptqmodel.quantization import protocol as protocol_module
    from gptqmodel.quantization.protocol import (
        ExecutionPlan,
        Rule,
        Stage,
        apply_plan_actions,
        smoothquant,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype), device_map="cuda:0", local_files_only=True
    ).eval()
    input_ids = tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids.to(
        model.device
    )

    rule = protocol_module._normalize_rule(Rule(match=pattern))
    capture = protocol_module._capture_smoothquant_statistics(model, lambda: model(input_ids))

    layers = sorted(
        {
            name.split(".self_attn.")[0]
            for name in capture.predecessors
            if name.endswith(".self_attn.q_proj") and rule.matches(name)
        }
    )
    assert layers, "no projection matched"
    for layer in layers:
        for projection in ("q_proj", "k_proj", "v_proj"):
            name = f"{layer}.self_attn.{projection}"
            assert capture.predecessors[name] == {f"{layer}.input_layernorm"}, name

    with torch.no_grad():
        before = model(input_ids).logits.clone()
    plan = ExecutionPlan(
        version=2,
        stages=(Stage(name="balance", rules=(Rule(match=pattern, actions=(smoothquant(0.5),)),)),),
    )
    report = apply_plan_actions(model, plan, calibration=lambda: model(input_ids))
    assert report.applied
    with torch.no_grad():
        after = model(input_ids).logits

    diff = (after.to(torch.float64) - before.to(torch.float64)).abs().max().item()
    eps = float(torch.finfo(getattr(torch, dtype)).eps)
    limit = 1e-3 + 8.0 * eps * before.to(torch.float64).abs().max().item()
    assert diff <= limit, f"{model_id} fold drift {diff:.6g} exceeded {limit:.6g}"


@pytest.mark.parametrize("model_id,dtype", [(case[0], case[1]) for case in CASES])
def test_tied_output_head_is_refused_on_a_real_model(model_id, dtype):
    if not _cached(model_id):
        pytest.skip(f"{model_id} is not in the local Hugging Face cache")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gptqmodel.quantization.protocol import (
        ExecutionPlan,
        ProtocolActionError,
        Rule,
        Stage,
        apply_plan_actions,
        smoothquant,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=getattr(torch, dtype), device_map="cuda:0", local_files_only=True
    ).eval()
    if getattr(model.config, "tie_word_embeddings", False) is not True:
        pytest.skip(f"{model_id} does not tie its output head")
    input_ids = tokenizer("hello", return_tensors="pt").input_ids.to(model.device)
    snapshot = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    plan = ExecutionPlan(
        version=2,
        stages=(Stage(name="balance", rules=(Rule(match="^lm_head$", actions=(smoothquant(0.5),)),)),),
    )
    with pytest.raises(ProtocolActionError):
        apply_plan_actions(model, plan, calibration=lambda: model(input_ids))

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, snapshot[name]), f"{name} changed"

    # The structural check is independent of how the head is called: scaling
    # `lm_head.weight` would also change `embed_tokens.weight`, which does not
    # consume the folded norm output.
    from gptqmodel.quantization import protocol as protocol_module

    modules = dict(model.named_modules())
    rule = protocol_module._normalize_rule(Rule(match="^lm_head$"))
    with pytest.raises(ProtocolActionError, match="its weight is shared with"):
        protocol_module._assert_consumer_weights_exclusive(
            modules, [("lm_head", model.lm_head)], "model.norm", rule
        )
