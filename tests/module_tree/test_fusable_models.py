# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Explicit registry admission plus actual native topology/forward qualification.

The inventory records independently inspected model parents, not synthetic trees.
CPU/meta tests always run; native numerical tests additionally require NVIDIA.
"""

import copy
import importlib
import inspect
import json
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.fusable import FusableConfig, require_fusable_config
from gptqmodel.models.fusion import FusionDecision, FusionPlanner, _FusionGroup, _parent_template
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT, QuantizeConfig
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq
from gptqmodel.utils.model import get_layers_with_prefixes


INVENTORY = json.loads((Path(__file__).parent / "fixtures/fusable_models.json").read_text())["models"]
QUALIFIED = {key: row for key, row in INVENTORY.items() if "profiles" in row}


def _type_name(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve(path):
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


def _config(key):
    if key not in CONFIG_MAPPING:
        pytest.skip(f"Installed Transformers has no {key!r} implementation for this native qualification fixture")
    config = transformers.AutoConfig.for_model(key)

    def restore(current, delta):
        def merge(base, changes):
            result = copy.deepcopy(base)
            for name, value in changes.items():
                result[name] = merge(result[name], value) if isinstance(result.get(name), dict) and isinstance(value, dict) else copy.deepcopy(value)
            return result

        for name, value in delta.items():
            child = vars(current).get(name)
            if isinstance(child, transformers.PretrainedConfig) and isinstance(value, dict):
                restore(child, value)
            elif isinstance(child, dict) and isinstance(value, dict):
                setattr(current, name, merge(child, value))
            else:
                setattr(current, name, copy.deepcopy(value))
        # Default heterogeneous overrides refer to layers beyond the tiny fixture.
        if hasattr(current, "_per_layer_config"):
            current._per_layer_config = None
        if current.model_type == "gptj":
            current.rotary_dim = 16
        if current.model_type == "gemma3n_text":
            current.intermediate_size = [256] * current.num_hidden_layers
        sections = {
            "qwen2_vl_text": [4, 6, 6], "qwen2_5_vl_text": [4, 6, 6], "qwen2_5_omni_text": [4, 6, 6],
            "glm4v_text": [4, 6, 6], "glm4v_moe_text": [2, 3, 3], "glm_ocr_text": [4, 6, 6],
            "hunyuan_vl_text": [4, 6, 6],
        }
        if current.model_type in sections:
            current.rope_parameters["mrope_section"] = sections[current.model_type]
        current._attn_implementation = "eager"
        # Config deltas need not repeat unchanged wrapper sub-configurations.
        for child in vars(current).values():
            if isinstance(child, transformers.PretrainedConfig) and child._attn_implementation != "eager":
                restore(child, {})

    restore(config, QUALIFIED[key]["config"])
    return config


def _build(key):
    config = _config(key)
    loader = getattr(transformers, QUALIFIED[key]["loader"])
    with torch.device("meta"):
        model = loader.from_config(config) if hasattr(loader, "from_config") else loader(config)
    layers, names = get_layers_with_prefixes(model, MODEL_MAP[key].extract_layers_node())
    assert layers, f"No declared decoder layers for {key}"
    return model, layers, names


def test_registry_has_explicit_fusion_qualification_for_every_alias():
    assert INVENTORY.keys() == MODEL_MAP.keys(), "New model support needs a fusion eligibility review"
    for key, definition in MODEL_MAP.items():
        row = INVENTORY[key]
        assert row["definition"] == definition.__name__
        config = definition.fusable
        if "profiles" in row:
            assert isinstance(config, FusableConfig) and key in config.model_types
            profiles = config.profiles_for(key)
            actual = {p.name: list(p.roles) for p in profiles}
            assert actual == row["profiles"], key
        else:
            assert row["reason"], key
            assert config is None or key not in config.model_types, key
            with pytest.raises(ValueError):
                require_fusable_config(definition, key)


@pytest.mark.parametrize("key", sorted(QUALIFIED))
def test_actual_model_definition_reaches_fusion_planner(key, monkeypatch):
    model, layers, names = _build(key)
    # The fixture installs FP16 quantized projections; admission validates the
    # real model activation dtype along with the combined native output shape.
    model.to(dtype=torch.float16)
    definition = MODEL_MAP[key]
    # Processor downloads and unrelated model-wide patches are loader concerns.
    monkeypatch.setattr(definition, "require_load_processor", False)
    monkeypatch.setattr(definition, "require_monkeypatch", False)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: SimpleNamespace())
    wrapper = definition(model, quantized=True, quantize_config=QuantizeConfig(bits=4, group_size=128, sym=True))
    plan = wrapper.module_plan()
    expected = set()
    for group in QUALIFIED[key]["groups"]:
        found = False
        for layer, layer_name in zip(layers, names):
            try:
                parent = layer.get_submodule(group["parent"])
            except AttributeError:
                continue
            if _type_name(type(parent)) != group["parent_type"]:
                continue
            if not all(isinstance(layer.get_submodule(path), torch.nn.Linear) for path in group["paths"]):
                continue
            assert not plan.is_moe(group["parent"])
            profile = next(p for p in definition.fusable.profiles_for(key) if p.name == group["family"])
            for role, path in zip(profile.roles, group["paths"]):
                assert plan.role_paths_in_parent(group["parent"], role) == (path,)
                assert plan.find(path).quantize
                linear = layer.get_submodule(path)
                module = TorchLinear(bits=4, group_size=128, sym=True, desc_act=False,
                                     in_features=linear.in_features, out_features=linear.out_features,
                                     bias=linear.bias is not None, dtype=torch.float16,
                                     register_buffers=True, format=FORMAT.GPTQ_V2)
                module._init_wf_unsqueeze_buffers()
                setattr(parent, path.rsplit(".", 1)[-1], module)
            expected.add((layer_name + "." + group["parent"], group["family"]))
            found = True
        assert found, group
    assert wrapper._fusion_plan is None and not wrapper._auto_fusion
    planner = FusionPlanner(wrapper)
    assert {(g.decision.parent, g.decision.family) for g in planner.groups} == expected
    # All alternatives, multiple stacks and aliases use the same public plan API.
    assert wrapper.module_plan() is plan
    assert expected


@contextmanager
def _capture_constructors(classes):
    originals = {cls: cls.__init__ for cls in classes}
    try:
        for cls, original in originals.items():
            def captured(self, *args, _original=original, **kwargs):
                _original(self, *args, **kwargs)
                self._fusion_test_ctor = (args, kwargs)
            cls.__init__ = captured
        yield
    finally:
        for cls, original in originals.items():
            cls.__init__ = original


# Common text parents (for example Qwen2 in audio/vision wrappers) execute once
# per method/dtype. Every wrapper still has its own topology test above.
NUMERICAL = {}
for _key, _row in QUALIFIED.items():
    for _group in _row["groups"]:
        NUMERICAL.setdefault((_group["parent_type"], _group["family"]), (_key, _group))


@lru_cache(maxsize=None)
def _parent_fixture(parent_type, family):
    key, group = NUMERICAL[parent_type, family]
    cls = _resolve(parent_type)
    with _capture_constructors([cls]):
        model, layers, names = _build(key)
    for layer, name in zip(layers, names):
        try:
            parent = layer.get_submodule(group["parent"])
        except AttributeError:
            continue
        if type(parent) is cls:
            root = model.get_submodule(name.rsplit(".", 2)[0])
            return model, parent, root, group
    raise AssertionError(f"Missing actual parent: {parent_type}")


def _quantized_projection(linear, method, dtype):
    k, n = linear.in_features, linear.out_features
    assert k % 128 == 0 and n % 64 == 0, (k, n)
    codes = torch.randint(0, 16, (k, n))
    zeros = torch.full((k // 128, n), 8, dtype=torch.int32)
    scales = torch.full(zeros.shape, 1 / 512, dtype=dtype)
    indices = torch.arange(k, dtype=torch.int32) // 128
    weight = ((codes - zeros[indices]) * scales[indices]).to(dtype)
    linear = linear.cpu().to(dtype)
    linear.weight.data.copy_(weight.t())
    kwargs = {"bits": 4, "group_size": 128, "sym": True, "desc_act": False, "in_features": k, "out_features": n,
              "bias": linear.bias is not None, "register_buffers": True, "dtype": dtype}
    if method == "gptq":
        source = TorchLinear(**kwargs, format=FORMAT.GPTQ_V2)
        source.pack_block(linear, scales.t(), zeros.t().float(), indices)
        module = MarlinLinear(**kwargs).cuda()
        for name in ("qweight", "qzeros", "scales", "bias", "g_idx"):
            src, dst = getattr(source, name, None), getattr(module, name, None)
            if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor) and src.shape == dst.shape:
                dst.data.copy_(src.cuda())
        module.qzero_format(format=2)
    else:
        module = AwqMarlinLinear(**kwargs).cuda()
        qweight, qzeros = pack_awq(codes, zeros, 4)
        module.qweight.data.copy_(qweight.cuda())
        module.qzeros.data.copy_(qzeros.cuda())
        module.scales.data.copy_(scales.cuda())
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.detach().cuda())
    module.post_init()
    bias = linear.bias.detach().cuda().clone() if linear.bias is not None else None
    return module.eval(), weight.cuda(), bias


def _position_embeddings(root, parent, x):
    rotary = getattr(root, "rotary_emb", None)
    if rotary is not None and "config" in inspect.signature(type(rotary).__init__).parameters:
        with torch.device("cpu"):
            rotary = type(rotary)(config=getattr(root, "config", parent.config))
        kwargs = {}
        if "layer_type" in inspect.signature(rotary.forward).parameters:
            kwargs["layer_type"] = (getattr(parent, "layer_type", None)
                                    or parent.config.layer_types[parent.layer_idx])
        return rotary.cuda()(x, torch.arange(x.shape[1], device=x.device).unsqueeze(0), **kwargs)
    dim = getattr(parent, "head_dim", 32)
    return (torch.ones(1, x.shape[1], dim, device=x.device, dtype=x.dtype),
            torch.zeros(1, x.shape[1], dim, device=x.device, dtype=x.dtype))


@pytest.mark.parametrize("parent_type,family", sorted(NUMERICAL))
@pytest.mark.parametrize("method", ["gptq", "awq"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_native_parent_fusion_matches_forward_and_integer_weight_oracle(parent_type, family, method, dtype):
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("Native Marlin qualification requires NVIDIA CUDA hardware")
    torch.manual_seed(901)
    model, meta, root, declaration = _parent_fixture(parent_type, family)
    names = tuple(path.rsplit(".", 1)[-1] for path in declaration["paths"])
    assert _parent_template(type(meta).forward, names) is not None
    args, kwargs = meta._fusion_test_ctor
    with torch.device("cpu"):
        parent = type(meta)(*copy.deepcopy(args), **copy.deepcopy(kwargs))
    parent.apply(model._init_weights)
    parent = parent.to(dtype).eval()
    modules, weights, biases = [], [], []
    for name in names:
        module, weight, bias = _quantized_projection(getattr(parent, name), method, dtype)
        setattr(parent, name, module)
        modules.append(module)
        weights.append(weight)
        biases.append(bias)
    parent = parent.cuda()
    x = torch.randn(1, 3, modules[0].in_features, device="cuda:0", dtype=dtype)
    kwargs = {}
    for name, param in list(inspect.signature(parent.forward).parameters.items())[1:]:
        if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            continue
        if name == "attention_mask" and parent_type == "transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextAttention":
            kwargs[name] = torch.full((1, 1, 3, 3), -torch.inf, device=x.device, dtype=dtype).triu(diagonal=1)
        elif name == "position_embeddings":
            kwargs[name] = _position_embeddings(root, parent, x)
        elif name == "position_ids":
            kwargs[name] = torch.arange(3, device=x.device).unsqueeze(0)
        elif name == "cache_position":
            kwargs[name] = torch.arange(3, device=x.device)
        elif name in ("use_cache", "output_attentions"):
            kwargs[name] = False
        elif name in ("shared_kv_states", "kv_shared_layer_cache"):
            kwargs[name] = {}
        elif param.default is inspect.Parameter.empty:
            kwargs[name] = None
    group = _FusionGroup(FusionDecision(family, declaration["parent"], tuple(declaration["paths"])), parent, modules)
    try:
        with torch.inference_mode():
            before = parent(x, **copy.deepcopy(kwargs))
            group.install()
            after = parent(x, **copy.deepcopy(kwargs))
            assert group.calls == 1
            torch.testing.assert_close(after, before, rtol=2e-3, atol=2e-3)
            expected = [(x.float() @ weight.float()).to(dtype) for weight in weights]
            expected = [value + bias if bias is not None else value for value, bias in zip(expected, biases)]
            actual = group.packed(x)
            assert actual.dtype == dtype
            torch.testing.assert_close(actual, torch.cat(expected, dim=-1), rtol=2e-3, atol=2e-3)
            assert group.calls == 1  # Oracle invokes the packed operator, not parent.forward.
            for module in modules:
                assert module.qweight.untyped_storage().data_ptr() == group.packed.qweight.untyped_storage().data_ptr()
    finally:
        group.remove()
    assert group.packed is None
    assert all(module.qweight.is_contiguous() for module in modules)
