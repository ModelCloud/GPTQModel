# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""PEFT LoRA must be able to wrap the FP8 kernel (#4948).

``TorchFP8Linear`` advertises ``SUPPORTS_TRAINING = True`` and its backward is
correct, but PEFT routes every ``BaseQuantLinear`` through
``peft.tuners.lora.gptq.dispatch_gptq``, which runs

    target.qweight = target_base_layer.qweight

FP8 keeps its codes in ``weight``, so that line raised
``AttributeError: 'TorchFP8Linear' object has no attribute 'qweight'`` before
any training started. The kernel now exposes ``qweight`` as a read-only alias
of ``weight``, so the dispatcher finds what it expects and the LoRA forward
still runs through the FP8 module.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.quantization import FP8Config


HIDDEN = 64
LAYERS = 2
VOCAB = 128


def _force_gptq_dispatch(monkeypatch) -> None:
    """Make PEFT take its GPT-QModel branch.

    ``peft.tuners.lora.gptq.dispatch_gptq`` gates on
    ``peft.import_utils.is_gptqmodel_available()``, which asks
    ``importlib.metadata`` for an installed ``gptqmodel`` distribution. A source
    checkout on ``PYTHONPATH`` has no such metadata, so the branch would be
    skipped and the test would silently stop covering the bug. Pin it on; the
    function only reports whether the package is installed.
    """

    import peft.import_utils as peft_import_utils
    from peft.tuners.lora import awq as peft_awq
    from peft.tuners.lora import gptq as peft_gptq
    from peft.utils import other as peft_other
    from peft.utils import quantization_utils as peft_quant

    # PEFT imports the predicate by value into several modules, and the AWQ
    # dispatcher calls it before the GPTQ one, so patch every holder.
    for module in (peft_import_utils, peft_awq, peft_gptq, peft_other, peft_quant):
        monkeypatch.setattr(module, "is_gptqmodel_available", lambda: True, raising=False)


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=HIDDEN * 2,
        vocab_size=VOCAB,
        max_position_embeddings=64,
    )
    torch.manual_seed(4948)
    return LlamaForCausalLM(config).to(torch.bfloat16).eval()


def _fp8_module(source: torch.nn.Linear, fp8_config: FP8Config) -> TorchFP8Linear:
    module = TorchFP8Linear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=source.in_features,
        out_features=source.out_features,
        bias=source.bias is not None,
        **fp8_config.quant_linear_init_kwargs(),
    )
    module.pack(source, scales=None, zeros=None)
    return module.eval()


def _fp8_attention_model():
    """A tiny Llama whose ``q_proj``/``v_proj`` are FP8 modules."""

    model = _tiny_llama()
    fp8_config = FP8Config()
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.q_proj = _fp8_module(attn.q_proj, fp8_config)
        attn.v_proj = _fp8_module(attn.v_proj, fp8_config)
    return model


def test_fp8_module_exposes_the_packed_weight_as_qweight():
    model = _tiny_llama()
    module = _fp8_module(model.model.layers[0].self_attn.q_proj, FP8Config())

    assert module.qweight is module.weight
    assert module.qweight.dtype == module.weight.dtype

    # ``dispatch_gptq`` writes the tensor back to the module it read it from,
    # so the alias must accept a rebind and keep the two names in sync.
    replacement = torch.zeros_like(module.weight)
    module.qweight = replacement
    assert module.qweight is replacement
    assert module.weight is replacement


def test_peft_lora_wraps_the_fp8_kernel_and_trains(monkeypatch):
    pytest.importorskip("peft", reason="peft is required for the LoRA dispatch test")
    from peft import LoraConfig, get_peft_model

    _force_gptq_dispatch(monkeypatch)
    model = _fp8_attention_model()
    fp8_weights = {
        id(module): module.weight.detach().clone()
        for module in model.modules()
        if isinstance(module, TorchFP8Linear)
    }
    assert len(fp8_weights) == 2 * LAYERS

    # The dispatcher used to raise here: AttributeError: no attribute 'qweight'.
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )

    wrapped = [
        module
        for name, module in peft_model.named_modules()
        if name.endswith("q_proj") and hasattr(module, "quant_linear_module")
    ]
    assert len(wrapped) == LAYERS
    assert all(isinstance(module.quant_linear_module, TorchFP8Linear) for module in wrapped)

    input_ids = torch.randint(0, VOCAB, (2, 8))
    peft_model.train()
    logits = peft_model(input_ids=input_ids).logits
    assert torch.isfinite(logits).all()
    logits.float().square().mean().backward()

    trained = 0
    for module in wrapped:
        for adapter_name in module.lora_A:
            grad_a = module.lora_A[adapter_name].weight.grad
            grad_b = module.lora_B[adapter_name].weight.grad
            assert grad_a is not None and torch.isfinite(grad_a).all()
            assert grad_b is not None and torch.isfinite(grad_b).all()
            trained += 1
    assert trained == LAYERS

    # LoRA must not have touched the frozen FP8 codes.
    for module in peft_model.modules():
        if isinstance(module, TorchFP8Linear):
            assert torch.equal(module.weight.detach(), fp8_weights[id(module)]), module.extra_repr()


def test_peft_lora_forward_matches_the_unadapted_fp8_forward(monkeypatch):
    pytest.importorskip("peft", reason="peft is required for the LoRA dispatch test")
    """Attaching an adapter with zeroed ``lora_B`` must not change the output."""

    from peft import LoraConfig, get_peft_model

    _force_gptq_dispatch(monkeypatch)
    model = _fp8_attention_model()
    input_ids = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        before = model(input_ids=input_ids).logits.clone()

    peft_model = get_peft_model(
        model,
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
    )
    # ``lora_B`` starts at zero, so the adapter is a no-op at step 0.
    with torch.no_grad():
        after = peft_model(input_ids=input_ids).logits

    torch.testing.assert_close(after, before, rtol=1e-2, atol=1e-2)
