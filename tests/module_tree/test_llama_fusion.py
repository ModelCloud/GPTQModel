# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real Llama shells, independently constructed integer weights, and consumers."""

import copy
import weakref

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedModel

from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, AWQConfig, QuantizeConfig
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq


def _wrapper(method="gptq", dtype=torch.float16, device="cpu", *, quantized=True, automatic=False, dynamic=None,
             kv_heads=2, bits=4):
    torch.manual_seed(912)
    config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=kv_heads, vocab_size=64,
                         attention_bias=True, mlp_bias=True)
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(dtype=dtype).eval()
    weights = {}
    for layer_index, layer in enumerate(model.model.layers):
        for path in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            linear = layer.get_submodule(path)
            size_in, size_out = linear.in_features, linear.out_features
            group_size = 16
            codes = torch.randint(0, 1 << bits, (size_in, size_out))
            zeros = torch.full((size_in // group_size, size_out), 1 << (bits - 1), dtype=torch.int32)
            scales = torch.full(zeros.shape, 1 / (32 * (1 << bits)), dtype=dtype)
            indices = torch.arange(size_in, dtype=torch.int32) // group_size
            weight = ((codes - zeros[indices]) * scales[indices]).to(dtype)
            linear.weight.data.copy_(weight.t())
            linear.bias.data.uniform_(-0.01, 0.01)
            cls = TorchLinear if method == "gptq" else AwqTorchLinear
            kwargs = {"format": FORMAT.GPTQ_V2} if method == "gptq" else {}
            module = cls(bits=bits, group_size=group_size, sym=False, desc_act=False,
                         in_features=size_in, out_features=size_out, bias=True,
                         register_buffers=True, dtype=dtype, **kwargs)
            if method == "gptq":
                module.pack_block(linear, scales.t(), zeros.t().float(), indices)
                module._init_wf_unsqueeze_buffers()
            else:
                qweight, qzeros = pack_awq(codes, zeros, bits)
                module.qweight, module.qzeros, module.scales = qweight, qzeros, scales
                module.bias = linear.bias.detach().clone()
            module.eval().to(device=device, dtype=dtype)
            for name in ("qweight", "qzeros", "scales", "bias", "g_idx"):
                value = getattr(module, name, None)
                if isinstance(value, torch.Tensor):
                    setattr(module, name, value.clone())
            parent, leaf = path.rsplit(".", 1)
            setattr(layer.get_submodule(parent), leaf, module)
            weights[f"model.layers.{layer_index}.{path}"] = (weight.to(device), module.bias.clone())
    model.to(device)
    qcfg = (QuantizeConfig if method == "gptq" else AWQConfig)(bits=bits, group_size=16, sym=False, dynamic=dynamic)
    wrapper = LlamaQModel(model, quantized=quantized, quantize_config=qcfg, fusion_enabled=automatic)
    return wrapper, weights


@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_real_llama_fusion_oracle_logits_and_generation(method, dtype, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    model, weights = _wrapper(method, dtype, device)
    ids = torch.tensor([[1, 5, 8, 3]], device=device)
    with torch.inference_mode():
        baseline = model.model(ids).logits
        tokens = model.model.generate(ids, max_new_tokens=3, do_sample=False)
    state_before = {name: tensor.clone() for name, tensor in model.model.state_dict().items()}
    assert model.fuse() is model
    planner = model.fusion_plan()
    assert len(planner.groups) == (4 if method == "gptq" else 2)
    if method == "awq":
        assert all("AWQ GQA" in d.reason for d in planner.decisions if not d.enabled)
    else:
        assert all(d.enabled for d in planner.decisions)
    for group in planner.groups:
        path = f"{group.decision.parent}.{group.decision.paths[0].rsplit('.', 1)[-1]}"
        x = torch.randn(2, 3, 64, device=device, dtype=dtype)
        expected = torch.cat([((x.float() @ weights[f"{group.decision.parent}.{p.rsplit('.', 1)[-1]}"][0].float()).to(dtype)
                               + weights[f"{group.decision.parent}.{p.rsplit('.', 1)[-1]}"][1])
                              for p in group.decision.paths], dim=-1)
        with torch.inference_mode():
            result = group._operand(group._key())(x)
        torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-3)
        assert result.dtype == dtype, path
    with torch.inference_mode():
        fused = model.model(ids).logits
        generated = model.model.generate(ids, max_new_tokens=3, do_sample=False)
    torch.testing.assert_close(fused, baseline, rtol=2e-3, atol=2e-3)
    assert torch.equal(tokens, generated)
    assert planner.to_dict()["fused_calls"] > 0
    for group in planner.groups:
        assert group.context.get() is None  # No activation survives its parent.
    assert state_before.keys() == model.model.state_dict().keys()
    for name, tensor in model.model.state_dict().items():
        assert torch.equal(tensor, state_before[name]), name
    model.fuse(False)
    assert model.fusion_plan() is None
    assert all(group.packed is None for group in planner.groups)
    with torch.inference_mode():
        torch.testing.assert_close(model.model(ids).logits, baseline, rtol=0, atol=0)


def test_automatic_opt_out_and_definition_overrides(monkeypatch):
    model, _ = _wrapper(automatic=True)
    assert len(model.fusion_plan().groups) == 4
    monkeypatch.setattr(LlamaQModel, "fusion_overrides", {"qkv": False})
    model.fuse(overrides={"model.layers.0.mlp": False})
    assert len(model.fusion_plan().groups) == 1
    assert model.fusion_plan().groups[0].decision.parent == "model.layers.1.mlp"
    model.fuse(False)
    assert model.fusion_plan() is None
    with pytest.raises(TypeError):
        model.fuse(overrides={"qkv": "yes"})
    assert model.fusion_plan() is None


@pytest.mark.parametrize("override", [{"bits": 8}, {"group_size": 32}, {"sym": True},
                                      {"desc_act": True}, {"mse": 0.2}, {"format": "gptq_v2"}])
def test_dynamic_effective_config_must_match(override):
    model, _ = _wrapper(dynamic={r"model\.layers\.0\.self_attn\.q_proj$": override})
    model.fuse()
    decisions = {d.parent: d for d in model.fusion_plan().decisions}
    assert not decisions["model.layers.0.self_attn"].enabled
    assert "configurations differ" in decisions["model.layers.0.self_attn"].reason
    assert decisions["model.layers.1.self_attn"].enabled


@pytest.mark.parametrize("field,value", [("bits", 8), ("group_size", 32), ("sym", True),
                                         ("desc_act", True), ("adapter", object())])
def test_actual_kernel_mismatch_rejects_fusion(field, value):
    model, _ = _wrapper()
    setattr(model.model.model.layers[0].self_attn.k_proj, field, value)
    model.fuse()
    assert not next(d for d in model.fusion_plan().decisions if d.parent == "model.layers.0.self_attn").enabled


def test_group_indices_hooks_and_excluded_roles_fail_closed():
    model, _ = _wrapper()
    layer = model.model.model.layers[0]
    layer.self_attn.k_proj.g_idx[0] = 1
    handle = layer.mlp.up_proj.register_forward_pre_hook(lambda *args: None)
    model.fuse()
    assert len(model.fusion_plan().groups) == 2
    handle.remove()
    model.fuse(False)


def test_prequantized_models_rejected_and_transformers_api_has_no_conflict():
    model, _ = _wrapper(quantized=False)
    with pytest.raises(RuntimeError, match="post-quantized"):
        model.fuse()
    assert not hasattr(PreTrainedModel, "fuse")


@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_state_load_mutation_dtype_and_no_retained_dense_weights(monkeypatch, method):
    model, _ = _wrapper(method, automatic=True)
    ids = torch.tensor([[1, 5, 8]])
    with torch.no_grad():
        model.model(ids)
    group = model.fusion_plan().groups[0]
    packed = group.packed
    references = []
    helper = "dequantize_weight" if method == "gptq" else "_dequantize_weight_for_inference"
    original = getattr(packed, helper)

    def tracked(*args, **kwargs):
        weight = original(*args, **kwargs)
        references.append(weakref.ref(weight))
        return weight

    monkeypatch.setattr(packed, helper, tracked)
    with torch.no_grad():
        model.model(ids)
        model.model(ids)
    assert references and all(ref() is None for ref in references)
    saved = copy.deepcopy(model.model.state_dict())
    group.modules[0].scales.mul_(0.5)
    with torch.no_grad():
        model.model(ids)
    assert group.packed is not packed
    changed = group.packed
    model.model.load_state_dict(saved)
    with torch.no_grad():
        model.model(ids)
    assert group.packed is not changed
    changed = group.packed
    model.to(torch.bfloat16)
    with torch.no_grad():
        model.model(ids)
    assert group.packed is not changed
    assert group.packed.scales.dtype == torch.bfloat16
    assert group.context.get() is None


def test_awq_mha_qkv_remains_eligible():
    model, _ = _wrapper("awq", kv_heads=4, automatic=True)
    assert len(model.fusion_plan().groups) == 4
    with torch.no_grad():
        fused = model.model(torch.tensor([[1, 2]])).logits
        model.fuse(False)
        separate = model.model(torch.tensor([[1, 2]])).logits
    torch.testing.assert_close(fused, separate, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("bits", [2, 8])
def test_other_word_aligned_widths(method, bits):
    model, weights = _wrapper(method, bits=bits)
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = model.model(ids).logits
        model.fuse()
        actual = model.model(ids).logits
    assert model.fusion_plan().groups
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    x = torch.randn(2, 3, 64, dtype=torch.float16)
    for group in model.fusion_plan().groups:
        oracle = []
        for path in group.decision.paths:
            weight, bias = weights[f"{group.decision.parent}.{path.rsplit('.', 1)[-1]}"]
            oracle.append((x.float() @ weight.float()).half() + bias)
        with torch.no_grad():
            output = group.packed(x)
        torch.testing.assert_close(output, torch.cat(oracle, dim=-1), rtol=2e-3, atol=2e-3)


def test_inference_tensor_explicit_clear_invalidates_packed_operand():
    model, _ = _wrapper(automatic=True)
    group = model.fusion_plan().groups[0]
    with torch.inference_mode():
        group.modules[0].scales = group.modules[0].scales.clone()
        ids = torch.tensor([[1, 2]])
        model.model(ids)
        old = group.packed
        group.modules[0].scales.mul_(0.5)
        group.modules[0].clear_weight_cache()
        assert group.packed is None
        actual = model.model(ids).logits
        assert group.packed is not old
        model.fuse(False)
        expected = model.model(ids).logits
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_parent_exception_clears_activations_and_other_inputs_fall_back(monkeypatch):
    model, _ = _wrapper(automatic=True)
    group = model.fusion_plan().groups[0]
    x = torch.randn(1, 2, 64, dtype=torch.float16)

    def different_inputs():
        return group.modules[0](x), group.modules[1](x + 1), group.modules[2](x)

    monkeypatch.setattr(group, "parent_forward", different_inputs)
    with torch.no_grad():
        actual = group.parent()
        expected = (group.originals[0](x), group.originals[1](x + 1), group.originals[2](x))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)

    def failure():
        group.modules[0](x)
        raise RuntimeError("parent failure")

    monkeypatch.setattr(group, "parent_forward", failure)
    with pytest.raises(RuntimeError, match="parent failure"), torch.no_grad():
        group.parent()
    assert group.context.get() is None


def test_training_and_direct_projection_calls_use_original_execution():
    model, _ = _wrapper(automatic=True)
    group = model.fusion_plan().groups[0]
    module = group.modules[0]
    x = torch.randn(1, 2, 64, dtype=torch.float16, requires_grad=True)
    module(x).sum().backward()
    assert x.grad is not None
    assert group.calls == 0
    model.train()
    model.model(torch.tensor([[1, 2]]))
    assert all(g.calls == 0 for g in model.fusion_plan().groups)
