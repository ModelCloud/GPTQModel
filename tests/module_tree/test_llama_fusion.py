# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real Llama shells, independently constructed integer weights, and consumers."""

import copy
import gc
import inspect
import io
import weakref
from dataclasses import replace

import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedModel

import gptqmodel.models.fusion as fusion
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.models.fusable import LlamaFusableConfig
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, AWQConfig, QuantizeConfig
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq
from gptqmodel.utils.marlin import marlin_runtime_available, marlin_runtime_error


def _wrapper(method="gptq", dtype=torch.float16, device="cpu", *, quantized=True, automatic=None, dynamic=None,
             kv_heads=2, bits=4, native=False, canonical=False, definition=LlamaQModel):
    if native and CUDA_HOME is None and not marlin_runtime_available(dtype):
        reason = marlin_runtime_error(dtype).partition(" | ")[0]
        pytest.skip(f"Native Marlin extension unavailable for {dtype}: {reason}")
    torch.manual_seed(912)
    aligned = native or canonical
    config = LlamaConfig(hidden_size=128 if aligned else 64, intermediate_size=256 if aligned else 128, num_hidden_layers=2,
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
            group_size = 128 if aligned else 16
            codes = torch.randint(0, 1 << bits, (size_in, size_out))
            zeros = torch.full((size_in // group_size, size_out), 1 << (bits - 1), dtype=torch.int32)
            scales = torch.full(zeros.shape, 1 / (32 * (1 << bits)), dtype=dtype)
            indices = torch.arange(size_in, dtype=torch.int32) // group_size
            weight = ((codes - zeros[indices]) * scales[indices]).to(dtype)
            linear.weight.data.copy_(weight.t())
            linear.bias.data.uniform_(-0.01, 0.01)
            cls = TorchLinear if method == "gptq" else AwqTorchLinear
            kwargs = {"format": FORMAT.GPTQ_V2} if method == "gptq" else {}
            module = cls(bits=bits, group_size=group_size, sym=aligned, desc_act=False,
                         in_features=size_in, out_features=size_out, bias=True,
                         register_buffers=True, dtype=dtype, **kwargs)
            if method == "gptq":
                module.pack_block(linear, scales.t(), zeros.t().float(), indices)
                if canonical:
                    module.qzero_format(format=2)
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
            if native:
                kernel = MarlinLinear if method == "gptq" else AwqMarlinLinear
                converted = kernel(bits=bits, group_size=group_size, sym=True, desc_act=False,
                                   in_features=size_in, out_features=size_out, bias=True,
                                   register_buffers=True, dtype=dtype).to(device)
                for name in ("qweight", "qzeros", "scales", "bias", "g_idx"):
                    src, dst = getattr(module, name, None), getattr(converted, name, None)
                    if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor) and src.shape == dst.shape:
                        dst.data.copy_(src)
                if method == "gptq":
                    converted.qzero_format(format=2)
                converted.post_init()
                module = converted.eval()
            parent, leaf = path.rsplit(".", 1)
            setattr(layer.get_submodule(parent), leaf, module)
            weights[f"model.layers.{layer_index}.{path}"] = (weight.to(device), linear.bias.detach().to(device=device, dtype=dtype).clone())
    model.to(device)
    qcfg = (QuantizeConfig if method == "gptq" else AWQConfig)(bits=bits, group_size=128 if aligned else 16,
                                                           sym=aligned, dynamic=dynamic)
    kwargs = {} if automatic is None else {"fusion_enabled": automatic}
    wrapper = definition(model, quantized=quantized, quantize_config=qcfg, **kwargs)
    return wrapper, weights


@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_symmetric_marlin_fusion_real_llama_oracle(method, dtype):
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("Native Marlin fusion requires NVIDIA CUDA hardware")
    model, weights = _wrapper(method, dtype, "cuda:0", native=True)
    ids = torch.tensor([[1, 2, 3]], device="cuda:0")
    with torch.inference_mode():
        before = model.model(ids).logits
        plan = model.fuse()
        after = model.model(ids).logits
        assert len(plan.groups) == 4
        for group in plan.groups:
            x = torch.randn(2, 3, 128, dtype=dtype, device="cuda:0")
            oracle = []
            for path in group.decision.paths:
                weight, bias = weights[f"{group.decision.parent}.{path.rsplit('.', 1)[-1]}"]
                oracle.append((x.float() @ weight.float()).to(dtype) + bias)
            torch.testing.assert_close(group.packed(x), torch.cat(oracle, dim=-1), rtol=2e-3, atol=2e-3)
            assert torch.count_nonzero(group.packed.workspace) == 0
    torch.testing.assert_close(after, before, rtol=2e-3, atol=2e-3)
    group = plan.groups[0]
    cached = group.packed
    group.modules[0].clear_weight_cache()
    assert group.packed is None
    with torch.inference_mode():
        model.model(ids)
    assert group.packed is not cached
    leaves = list(group.modules)
    model.fuse(False)
    assert all(not hasattr(leaf, "clear_weight_cache") for leaf in leaves)


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
    planner = model.fuse()
    assert planner is model._fusion_plan
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
    assert state_before.keys() == model.model.state_dict().keys()
    for name, tensor in model.model.state_dict().items():
        assert torch.equal(tensor, state_before[name]), name
    model.fuse(False)
    assert model._fusion_plan is None
    assert all(group.packed is None for group in planner.groups)
    with torch.inference_mode():
        torch.testing.assert_close(model.model(ids).logits, baseline, rtol=0, atol=0)


def test_automatic_opt_out_and_definition_overrides(monkeypatch):
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    assert len(plan.groups) == 4
    monkeypatch.setattr(LlamaQModel, "fusable", replace(LlamaFusableConfig(), overrides={"qkv": False}))
    plan = model.fuse(overrides={"model.layers.0.mlp": False})
    assert len(plan.groups) == 1
    assert plan.groups[0].decision.parent == "model.layers.1.mlp"
    model.fuse(False)
    assert model._fusion_plan is None
    with pytest.raises(TypeError):
        model.fuse(overrides={"qkv": "yes"})
    assert model._fusion_plan is None


def test_fusion_is_opt_in_at_every_entry_point():
    from gptqmodel import GPTQModel

    for entry, option in ((GPTQModel.load, "fuse"), (LlamaQModel.from_pretrained, "fuse"),
                          (LlamaQModel.from_quantized, "fuse"), (LlamaQModel.__init__, "fusion_enabled")):
        assert inspect.signature(entry).parameters[option].default is False
    model, _ = _wrapper()
    assert not model._auto_fusion
    assert model._fusion_plan is None
    plan = model.fuse()
    assert len(plan.groups) == 4


@pytest.mark.parametrize("overrides", [None, {"qkv": False, "gate_up": False}])
def test_fuse_returns_inspectable_plan_and_disable_returns_none(overrides):
    model, _ = _wrapper()
    assert not hasattr(model, "fusion_plan")
    plan = model.fuse(overrides=overrides)
    assert isinstance(plan, fusion.FusionPlanner)
    assert len(plan.groups) == (4 if overrides is None else 0)
    output = io.StringIO()
    plan.print(file=output)
    assert output.getvalue() == plan.render() + "\n"
    assert plan.to_dict()["groups"] and plan.to_dict()["fused_calls"] == 0
    assert model.fuse(False) is None
    assert all(not group.active and group.packed is None for group in plan.groups)
    assert "FusionPlanner" in plan.render()


@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("native", [False, True])
def test_shared_version_guard_covers_writes_through_every_sibling(method, native):
    if native and (not torch.cuda.is_available() or torch.version.hip):
        pytest.skip("Native Marlin fusion requires NVIDIA CUDA hardware")
    model, _ = _wrapper(method, device="cuda:0" if native else "cpu", native=native, automatic=True)
    plan = model.fuse()
    for group in plan.groups:
        for index in range(len(group.modules)):
            for name in ("qweight", "qzeros", "scales", "bias"):
                tensor = getattr(group.modules[index], name)
                if tensor is None or not tensor.numel():
                    continue
                assert group.guard()
                before = getattr(group.packed, name)._version
                with torch.no_grad():
                    tensor.add_(1)
                assert getattr(group.packed, name)._version > before
                assert not group.guard()
                assert group._refresh()


@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("native", [False, True])
def test_source_operands_share_only_fused_storage_and_unfuse_restores_it(method, native):
    if native and (not torch.cuda.is_available() or torch.version.hip):
        pytest.skip("Native Marlin fusion requires NVIDIA CUDA hardware")
    from gptqmodel.utils.model import get_state_dict_for_save

    model, _ = _wrapper(method, device="cuda:0" if native else "cpu", native=native)
    original = {name: tensor.clone() for name, tensor in model.model.state_dict().items()}
    # Weak references catch old allocations retained in guards or closures.
    references = [weakref.ref(m.qweight) for layer in model.model.model.layers
                  for m in ((layer.self_attn.q_proj, layer.mlp.gate_proj)
                            if method == "gptq" or native else (layer.mlp.gate_proj,))]
    plan = model.fuse()
    assert all(ref() is None for ref in references)
    groups = plan.groups
    for group in groups:
        for name in ("qweight", "qzeros", "scales", "bias"):
            operand = getattr(group.packed, name)
            if operand is None or not operand.numel():
                continue
            assert operand.is_contiguous()
            dim = 0 if name == "bias" else 1
            torch.testing.assert_close(operand, torch.cat([getattr(m, name) for m in group.modules], dim=dim),
                                       rtol=0, atol=0)
            for module in group.modules:
                tensor = getattr(module, name)
                assert tensor.untyped_storage().data_ptr() == operand.untyped_storage().data_ptr()
        assert group.guard()
    state = get_state_dict_for_save(model.model)
    assert {name for name, tensor in original.items() if tensor.numel()} <= set(state)
    for name, entry in state.items():
        torch.testing.assert_close(entry.source, original[name], rtol=0, atol=0)
    del state
    # Split kernel calls are valid without breaking sharing, including Marlin
    # which needs ephemeral contiguous copies of column views.
    for group in groups:
        x = torch.randn(1, 1, group.modules[0].in_features,
                        device=group.modules[0].qweight.device, dtype=torch.float16)
        with torch.no_grad():
            separate = torch.cat([module(x) for module in group.modules], dim=-1)
            combined = group.packed(x)
        torch.testing.assert_close(separate, combined, rtol=2e-3, atol=2e-3)
        assert group.guard()
    packed_refs = [weakref.ref(group.packed.qweight) for group in groups]
    model.fuse(False)
    assert all(ref() is None for ref in packed_refs)
    pointers = []
    for group in groups:
        for module in group.modules:
            assert module.qweight.is_contiguous()
            pointers.append(module.qweight.untyped_storage().data_ptr())
    assert len(set(pointers)) == len(pointers)
    for name, tensor in model.model.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


@pytest.mark.parametrize("override", [{"bits": 8}, {"group_size": 32}, {"sym": True},
                                      {"desc_act": True}, {"mse": 0.2}, {"format": "gptq_v2"}])
def test_dynamic_effective_config_must_match(override):
    model, _ = _wrapper(dynamic={r"model\.layers\.0\.self_attn\.q_proj$": override})
    plan = model.fuse()
    decisions = {d.parent: d for d in plan.decisions}
    assert not decisions["model.layers.0.self_attn"].enabled
    assert "configurations differ" in decisions["model.layers.0.self_attn"].reason
    assert decisions["model.layers.1.self_attn"].enabled


@pytest.mark.parametrize("field,value", [("bits", 8), ("group_size", 32), ("sym", True),
                                         ("desc_act", True), ("adapter", object())])
def test_actual_kernel_mismatch_rejects_fusion(field, value):
    model, _ = _wrapper()
    setattr(model.model.model.layers[0].self_attn.k_proj, field, value)
    plan = model.fuse()
    assert not next(d for d in plan.decisions if d.parent == "model.layers.0.self_attn").enabled


def test_group_indices_hooks_and_excluded_roles_fail_closed():
    model, _ = _wrapper()
    layer = model.model.model.layers[0]
    layer.self_attn.k_proj.g_idx[0] = 1
    handle = layer.mlp.up_proj.register_forward_pre_hook(lambda *args: None)
    plan = model.fuse()
    assert len(plan.groups) == 2
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
    plan = model.fuse()
    ids = torch.tensor([[1, 5, 8]])
    with torch.no_grad():
        model.model(ids)
    group = plan.groups[0]
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


def test_awq_mha_qkv_remains_eligible():
    model, _ = _wrapper("awq", kv_heads=4, automatic=True)
    plan = model.fuse()
    assert len(plan.groups) == 4
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
        plan = model.fuse()
        actual = model.model(ids).logits
    assert plan.groups
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    x = torch.randn(2, 3, 64, dtype=torch.float16)
    for group in plan.groups:
        oracle = []
        for path in group.decision.paths:
            weight, bias = weights[f"{group.decision.parent}.{path.rsplit('.', 1)[-1]}"]
            oracle.append((x.float() @ weight.float()).half() + bias)
        with torch.no_grad():
            output = group.packed(x)
        torch.testing.assert_close(output, torch.cat(oracle, dim=-1), rtol=2e-3, atol=2e-3)


def test_inference_tensor_explicit_clear_invalidates_packed_operand():
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    group = plan.groups[0]
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
    plan = model.fuse()
    group = plan.groups[0]
    x = torch.randn(1, 2, 64, dtype=torch.float16)

    def different_inputs(_x, *args, **kwargs):
        return group.modules[0](x), group.modules[1](x + 1), group.modules[2](x)

    monkeypatch.setattr(group, "parent_forward", different_inputs)
    with torch.no_grad():
        actual = group.parent(x)
        expected = (group.originals[0](x), group.originals[1](x + 1), group.originals[2](x))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)

    def failure(_x, *args, **kwargs):
        group.modules[0](x)
        raise RuntimeError("parent failure")

    monkeypatch.setattr(group, "parent_forward", failure)
    with pytest.raises(RuntimeError, match="parent failure"), torch.no_grad():
        group.parent(x)


def test_training_and_direct_projection_calls_use_original_execution():
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    group = plan.groups[0]
    module = group.modules[0]
    x = torch.randn(1, 2, 64, dtype=torch.float16, requires_grad=True)
    module(x).sum().backward()
    assert x.grad is not None
    assert group.calls == 0
    model.train()
    model.model(torch.tensor([[1, 2]]))
    assert all(g.calls == 0 for g in plan.groups)


def test_eval_fused_cpu_then_train_restores_checkpoint_layout():
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    modules = [module for group in plan.groups for module in group.modules]
    original_shapes = {
        id(module): tuple(module.qweight.shape)
        for module in modules
    }

    with torch.inference_mode():
        model.model(torch.tensor([[1, 2]]))
    assert any(group.calls for group in plan.groups)

    model.train()
    assert all(
        tuple(module.qweight.shape) == original_shapes[id(module)]
        for module in modules
    )

    output = model.model(torch.tensor([[1, 2]])).logits
    output.float().sum().backward()


def test_steady_state_uses_compiled_descriptors_and_unchanged_leaf_forwards(monkeypatch):
    model, _ = _wrapper()
    leaves = list(model.model.model.layers[0].self_attn.children())
    original_forwards = [leaf.forward for leaf in leaves]
    signature = inspect.signature(model.model.model.layers[0].self_attn.forward)
    plan = model.fuse()
    assert [leaf.forward for leaf in leaves] == original_forwards
    assert inspect.signature(model.model.model.layers[0].self_attn.forward) == signature

    def forbidden(*args, **kwargs):
        raise AssertionError("full metadata inspection reached the token loop")

    for group in plan.groups:
        assert group.packed is not None
        monkeypatch.setattr(group, "_key", forbidden)
    monkeypatch.setattr(fusion, "_incompatible", forbidden)
    with torch.inference_mode():
        for _ in range(3):
            model.model(torch.tensor([[1, 2, 3]]))
    assert all(group.calls == 3 for group in plan.groups)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("mutation", ["inplace", "replacement", "data", "load", "dtype"])
def test_compiled_guard_invalidates_changed_weights(native, mutation):
    if native and (not torch.cuda.is_available() or torch.version.hip):
        pytest.skip("Native mutation guard test requires NVIDIA CUDA")
    device = "cuda:0" if native else "cpu"
    model, _ = _wrapper(native=native, device=device, automatic=True)
    plan = model.fuse()
    group = plan.groups[0]
    module = group.modules[0]
    old = group.packed
    if mutation == "inplace":
        module.scales.mul_(0.5)
    elif mutation == "replacement":
        replacement = module.scales.clone() * 0.5
        module.scales = torch.nn.Parameter(replacement, requires_grad=False) if native else replacement
    elif mutation == "data":
        module.scales.data = module.scales.data * 0.5
    elif mutation == "load":
        model.model.load_state_dict(copy.deepcopy(model.model.state_dict()))
    else:
        model.to(torch.bfloat16)
    ids = torch.tensor([[1, 2]], device=device)
    with torch.inference_mode():
        actual = model.model(ids).logits
        assert group.packed is not old
        model.fuse(False)
        expected = model.model(ids).logits
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_runtime_hooks_and_replaced_modules_use_original_parent():
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    group = plan.groups[0]
    ids = torch.tensor([[1, 2]])
    handle = group.modules[1].register_forward_pre_hook(lambda _m, args: (args[0] + 0.25,))
    with torch.inference_mode():
        actual = model.model(ids).logits
        assert group.packed is None
        model.fuse(False)
        expected = model.model(ids).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    handle.remove()
    plan = model.fuse()
    group = plan.groups[0]
    group.parent.k_proj = torch.nn.Linear(64, 32, bias=True, dtype=torch.float16)
    with torch.inference_mode():
        actual = model.model(ids).logits
        assert group.packed is None
        model.fuse(False)
        expected = model.model(ids).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_direct_parent_releases_activations_on_success_and_exception(monkeypatch):
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    group = next(g for g in plan.groups if g.decision.family == "gate_up")
    references = []
    original = group.packed_forward

    def tracked(x):
        result = original(x)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(group, "packed_forward", tracked)
    x = torch.randn(1, 2, 64, dtype=torch.float16)
    with torch.no_grad():
        for _ in range(3):
            group.parent(x)
    assert references and all(ref() is None for ref in references)

    def failure(*args, **kwargs):
        raise RuntimeError("output projection failed")

    monkeypatch.setattr(group.parent.down_proj, "forward", failure)
    with pytest.raises(RuntimeError, match="output projection failed"), torch.no_grad():
        group.parent(x)
    gc.collect()
    assert all(ref() is None for ref in references)


def test_weight_gradient_and_global_hook_fallbacks():
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    group = next(g for g in plan.groups if g.decision.family == "gate_up")
    group.modules[0].scales.requires_grad_(True)
    x = torch.randn(1, 2, 64, dtype=torch.float16)
    group.parent(x).sum().backward()
    assert group.modules[0].scales.grad is not None
    assert group.calls == 0
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    handle = torch.nn.modules.module.register_module_forward_hook(lambda *_args: None)
    try:
        with torch.no_grad():
            model.model(torch.tensor([[1, 2]]))
        assert all(g.calls == 0 for g in plan.groups)
    finally:
        handle.remove()


def test_stale_parent_forward_cannot_reenable_removed_fusion():
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    group = next(g for g in plan.groups if g.decision.family == "gate_up")
    wrapped = group.parent.forward
    # A third-party wrapper installed later may still reference our old method.
    group.parent.forward = lambda *args, **kwargs: wrapped(*args, **kwargs)
    model.fuse(False)
    with torch.no_grad():
        group.parent(torch.randn(1, 2, 64, dtype=torch.float16))
    assert group.calls == 0
    assert group.packed is None


def _different_inputs(self, x):
    return self.gate_proj(x) * self.up_proj(x + 1)


def _repeated_projection(self, x):
    return self.gate_proj(x) + self.gate_proj(x) + self.up_proj(x)


def _projection_weight_access(self, x):
    return self.gate_proj(x) * self.up_proj(x) + self.gate_proj.weight.sum()


def _preprocess_input(self, x):
    x = x * 2
    return self.gate_proj(x) * self.up_proj(x)


def _mutate_input(self, x):
    x.mul_(2)
    return self.gate_proj(x) * self.up_proj(x)


def _indirectly_mutate_input(self, x):
    gate = self.gate_proj(x)
    self.preprocess(x)
    return gate * self.up_proj(x)


def _escape_input_alias(self, x):
    alias = x
    self.preprocess(alias)
    return self.gate_proj(x) * self.up_proj(x)


def _conditional_projection(self, x):
    if self.training:
        return self.gate_proj(x) * self.up_proj(x)
    return x


def _dynamic_metadata_axis(self, x, axis=0):
    x.size(axis)
    return self.gate_proj(x) * self.up_proj(x)


@pytest.mark.parametrize("forward", [_different_inputs, _repeated_projection, _projection_weight_access,
                                     _preprocess_input, _mutate_input, _indirectly_mutate_input,
                                     _escape_input_alias, _conditional_projection, _dynamic_metadata_axis])
def test_unfamiliar_forward_fails_closed(forward):
    assert fusion._parent_template(forward, ("gate_proj", "up_proj")) is None


def test_same_storage_dtype_and_shape_changes_are_detected():
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    group = plan.groups[0]
    scales = group.modules[0].scales
    pointer = scales.data_ptr()
    scales.data = scales.data.view(torch.bfloat16)
    assert scales.data_ptr() == pointer
    assert not group.guard()
    model.fuse(False)
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    group = plan.groups[0]
    scales = group.modules[0].scales
    pointer = scales.data_ptr()
    scales.data = scales.data.as_strided((scales.numel(),), (1,))
    assert scales.data_ptr() == pointer
    assert not group.guard()


def test_runtime_projection_forward_override_uses_original_parent():
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    group = plan.groups[0]
    module = group.modules[0]
    original = module.forward
    module.forward = lambda x: original(x) * 0.75
    ids = torch.tensor([[1, 2]])
    with torch.no_grad():
        actual = model.model(ids).logits
        assert group.packed is None
        model.fuse(False)
        expected = model.model(ids).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_compiled_parent_uses_original_path():
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    group = next(g for g in plan.groups if g.decision.family == "gate_up")
    x = torch.randn(1, 2, 64, dtype=torch.float16)
    compiled = torch.compile(group.parent, backend="eager")
    try:
        with torch.inference_mode():
            actual = compiled(x)
            expected = group.parent_forward(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert group.calls == 0
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("previously_fused", [False, True])
def test_native_construction_failure_rolls_back_installed_groups(monkeypatch, previously_fused):
    model, _ = _wrapper(automatic=previously_fused)
    previous = model._fusion_plan
    original = fusion._FusionGroup._operand
    calls = 0

    def fail_second(group, key):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("native descriptor construction failed")
        return original(group, key)

    monkeypatch.setattr(fusion._FusionGroup, "_operand", fail_second)
    with pytest.raises(RuntimeError, match="native descriptor construction failed"):
        model.fuse()
    assert model._fusion_plan is previous
    if previously_fused:
        assert all(group.active and group.packed is not None for group in previous.groups)
    else:
        for layer in model.model.model.layers:
            for parent in (layer.self_attn, layer.mlp):
                assert parent.forward.__func__ is type(parent).forward
            assert layer.mlp.gate_proj.qweight.is_contiguous()
    model.fuse(False)


def test_supported_dtype_move_rebuilds_and_keeps_fused_execution():
    model, _ = _wrapper(automatic=True)
    plan = model.fuse()
    groups = plan.groups
    model.model.to(dtype=torch.bfloat16)
    with torch.inference_mode():
        output = model.model(torch.tensor([[1, 2]])).logits
    assert output.dtype == torch.bfloat16 and torch.isfinite(output).all()
    assert all(group.calls > 0 and group.input_dtype == torch.bfloat16 for group in groups)
    model.fuse(False)
