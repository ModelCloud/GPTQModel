# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Fast CPU tests: generated pure Torch, independent math, state and fullgraph."""

import ast
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from gptqmodel_compiler import Program, compile_program, llama32_1b, llama_program
from gptqmodel_compiler.ir import VerificationError, verify


REFERENCE = Path(__file__).with_name("reference.py")
spec = importlib.util.spec_from_file_location("dsl_reference", REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def tiny(dtype="float32"):
    return llama_program(
        hidden_size=32,
        intermediate_size=80,
        layers=2,
        vocabulary=96,
        query_heads=4,
        kv_heads=2,
        head_dim=8,
        capacity=24,
        dtype=dtype,
    )


def weights(model):
    return {
        name: model.parameter_view(name).detach().clone()
        for name in model._execute.__globals__["MANIFEST"]["parameters"]
    }


def oracle_config(p):
    a = next(op.attributes for b in p.blocks for op in b.body if op.kind == "rope")
    return {
        "query_heads": a["query_heads"],
        "kv_heads": a["kv_heads"],
        "head_dim": a["head_dim"],
        "layers": len(p.resources),
        "eps": 1e-5,
        **a,
    }


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_both_targets_match_independent_oracle_and_cached_chunks(dtype):
    p = tiny(dtype)
    generator = torch.Generator().manual_seed(13)
    source = {}
    for parameter in p.parameters:
        if not parameter.alias_of:
            value = (
                torch.ones(parameter.shape)
                if len(parameter.shape) == 1
                else torch.randn(parameter.shape, generator=generator) * 0.02
            )
            source[parameter.name] = value.to(getattr(torch, dtype))
        else:
            source[parameter.name] = source[parameter.alias_of]
    eager = compile_program(p).instantiate().load_weights(source)
    models = [eager]
    if dtype == "float32":
        optimized = compile_program(p, target="optimized").instantiate(compile_graph=False).load_weights(source)
        models.append(optimized)
    else:
        with pytest.raises(NotImplementedError, match="float32 only"):
            compile_program(p, target="optimized")
    ids = torch.tensor([[1, 2, 8, 4, 5], [7, 3, 2, 9, 6]])
    expected, expected_cache = reference.llama_reference(
        ids, source, oracle_config(p), high_precision=dtype == "float32"
    )
    for model in models:
        actual, state = model(ids)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        for actual_cache, wanted in zip(state.cache, expected_cache):
            torch.testing.assert_close(actual_cache, wanted, atol=2e-3, rtol=2e-3)
        first, state = model(ids[:, :2])
        second, state = model(ids[:, 2:], state)
        torch.testing.assert_close(torch.cat((first, second), dim=1), actual, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(actual.argmax(-1), expected.argmax(-1), rtol=0, atol=0)
        assert state.length == 5
    assert len(eager.projections) == 15
    if dtype == "float32":
        assert len(optimized.projections) == 9
        assert sum(x.numel() for x in eager.parameters()) == sum(x.numel() for x in optimized.parameters())


def test_eager_observation_identity_and_explicit_empty_resource_interfaces():
    a = compile_program(tiny())
    model = a.instantiate().initialize_for_test(4)
    group = next(x for x in a.manifest["observations"] if x["operation"] == "decoder.0.qkv")
    observed, handles = [], []
    for key in group["module_keys"]:
        handles.append(model.projections[key].register_forward_pre_hook(lambda _, args: observed.append(args[0])))
    model(torch.tensor([[1, 2]]))
    assert len(observed) == 3 and observed[0] is observed[1] is observed[2]
    assert next(b for b in a.manifest["blocks"] if b["name"] == "decoder.0.mlp")["resources"] == []
    for handle in handles:
        handle.remove()


def test_inductor_fullgraph_prefill_decode_and_mutation():
    p = tiny()
    a = compile_program(p, target="optimized")
    model = a.instantiate().initialize_for_test(3)
    baseline = a.instantiate(compile_graph=False).load_weights(weights(model))
    ids = torch.tensor([[1, 2, 3]])
    expected, s0 = baseline(ids)
    actual, s1 = model(ids)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    for token in [4, 5]:
        x = torch.tensor([[token]])
        s0.phase = s1.phase = "decode"
        expected, s0 = baseline(x, s0)
        actual, s1 = model(x, s1)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    from torch._dynamo import explain

    state = model.new_request()
    report = explain(model._execute)(ids, torch.arange(3), state.cache)
    assert report.graph_count == 1 and report.graph_break_count == 0
    before = model(ids)[0].clone()
    with torch.no_grad():
        next(model.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="stale"):
        model(torch.tensor([[6]]), s1)
    assert not torch.equal(model(ids)[0], before)


def test_resource_preflight_fork_isolation_and_failure_rollback():
    model = compile_program(tiny()).instantiate().initialize_for_test(1)
    _, state = model(torch.tensor([[1, 2]]))
    original = tuple(x.clone() for x in state.cache)
    branch = state.fork()
    model(torch.tensor([[3]]), branch)
    assert state.length == 2 and branch.length == 3
    for x, y in zip(state.cache, original):
        assert torch.equal(x, y)
    bad = state.fork()
    bad.cache = (bad.cache[0].double(), *bad.cache[1:])
    with pytest.raises(ValueError, match="dtype/shape"):
        model(torch.tensor([[4]]), bad)
    bad = state.fork()
    bad.cache = (bad.cache[0], bad.cache[0], *bad.cache[2:])
    with pytest.raises(ValueError, match="aliases"):
        model(torch.tensor([[4]]), bad)
    with pytest.raises(ValueError, match="capacity"):
        model(torch.ones(1, 24, dtype=torch.int64), state)

    def fail(*_):
        raise RuntimeError("candidate failed")

    model._compiled = fail
    with pytest.raises(RuntimeError, match="candidate failed"):
        model(torch.tensor([[4]]), state)
    assert state.length == 2
    for x, y in zip(state.cache, original):
        assert torch.equal(x, y)


def test_serialized_ir_and_generated_module_have_no_transformers_dependency(tmp_path):
    p = tiny()
    assert Program.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    artifact = compile_program(p)
    path = artifact.write(tmp_path)
    imported = set()
    for node in ast.walk(ast.parse(artifact.source)):
        if isinstance(node, ast.Import):
            imported.update(x.name for x in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= {"math", "re", "threading", "torch", "torch.nn"}
    script = f"""import importlib.util, sys
class Deny:
    def find_spec(self,name,*args):
        if name.startswith(("transformers","gptqmodel.")):
            raise AssertionError(name)
sys.meta_path.insert(0,Deny())
spec=importlib.util.spec_from_file_location("generated",{str(path)!r})
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import torch
model=m.GeneratedModel().initialize_for_test(2)
assert model(torch.tensor([[1,2]]))[0].shape == (1,2,96)
assert not any(x.startswith("transformers") for x in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_verifier_rejects_unknown_ops_wrong_shapes_and_missing_resource_ports():
    p = tiny()
    block = p.blocks[1]
    for invalid in [
        replace(block, resources=()),
        replace(block, body=(replace(block.body[0], kind="python_callback"), *block.body[1:])),
        replace(
            block,
            body=(
                replace(block.body[0], outputs=(replace(block.body[0].outputs[0], shape=("B", "T", 31)),)),
                *block.body[1:],
            ),
        ),
    ]:
        with pytest.raises(VerificationError):
            compile_program(replace(p, blocks=(p.blocks[0], invalid, *p.blocks[2:])))
    with pytest.raises(VerificationError, match="packed"):
        compile_program(replace(p, parameters=(replace(p.parameters[0], storage="int4"), *p.parameters[1:])))
    with pytest.raises(VerificationError):
        compile_program(replace(p, torch_target="2.16"))
    verify(llama32_1b())


def test_checkpoint_aliases_and_storage_survive_repeated_forwards(tmp_path):
    from safetensors.torch import save_file

    from gptqmodel_compiler.checkpoint import load_safetensors

    source = compile_program(tiny()).instantiate().initialize_for_test(6)
    state_dict = weights(source)
    state_dict.pop("lm_head.weight")
    save_file(state_dict, str(tmp_path / "model.safetensors"))
    for target in ["eager", "optimized"]:
        model = load_safetensors(compile_program(tiny(), target=target).instantiate(compile_graph=False), tmp_path)
        before = [(p.data_ptr(), p.numel(), p.dtype) for p in model.parameters()]
        for _ in range(2):
            model(torch.tensor([[1, 2]]))
        assert before == [(p.data_ptr(), p.numel(), p.dtype) for p in model.parameters()]
        assert (
            model.parameter_view("model.embed_tokens.weight").data_ptr()
            == model.parameter_view("lm_head.weight").data_ptr()
        )
        torch.testing.assert_close(
            model(torch.tensor([[1, 2]]))[0], source(torch.tensor([[1, 2]]))[0], atol=2e-3, rtol=2e-3
        )


def test_preflight_rejects_broken_parameter_ties_and_stale_storage():
    artifact = compile_program(tiny())
    model = artifact.instantiate().initialize_for_test(2)
    alias = artifact.manifest["linear_aliases"][0]
    model.projections[alias["key"]].weight = torch.nn.Parameter(
        model.parameter_view(alias["parameter"]).detach().clone(), requires_grad=False
    )
    with pytest.raises(ValueError, match="alias was broken"):
        model(torch.tensor([[1, 2]]))
    model = artifact.instantiate().initialize_for_test(2)
    key = next(iter(model.projections))
    model.projections[key].weight = torch.nn.Parameter(model.projections[key].weight[:-1].clone(), requires_grad=False)
    with pytest.raises(ValueError, match="storage changed"):
        model(torch.tensor([[1, 2]]))


def test_checkpoint_rejects_quantized_float_payload_and_conflicting_tie():
    model = compile_program(tiny()).instantiate().initialize_for_test(2)
    source = weights(model)
    source["lm_head.weight"].add_(1)
    with pytest.raises(ValueError, match="tied checkpoint"):
        model.load_weights(source)
    source = weights(model)
    source["model.embed_tokens.weight"] = source["model.embed_tokens.weight"].to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="tensor mismatch"):
        model.load_weights(source)


def test_rejects_unknown_attributes_and_unsupported_framework(monkeypatch):
    p = tiny()
    block = p.blocks[0]
    invalid = replace(block, body=(replace(block.body[0], attributes={"implicit_activation": "relu"}),))
    with pytest.raises(VerificationError, match="attributes"):
        compile_program(replace(p, blocks=(invalid, *p.blocks[1:])))
    artifact = compile_program(p)
    monkeypatch.setattr(torch, "__version__", "2.16.0")
    with pytest.raises(RuntimeError, match="requires stable PyTorch 2.15"):
        artifact.instantiate()
