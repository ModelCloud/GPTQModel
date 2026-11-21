# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Verify typed blocks and lower their bodies to readable standalone Torch source."""

import json
import linecache
import sys
import types
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .ir import Program, verify


@dataclass(frozen=True)
class Artifact:
    source: str
    manifest: dict
    program: Program

    def write(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.py"
        path.write_text(self.source, encoding="utf-8")
        (directory / "program.json").write_text(json.dumps(self.program.to_dict(), indent=2) + "\n", encoding="utf-8")
        (directory / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
        return path

    def instantiate(self, **kwargs):
        # Source is generated only from the verified closed operation registry.
        name = f"_gptqmodel_dsl_{self.manifest['program_id']}_{self.manifest['target']}"
        path = f"<{name}>"
        module = types.ModuleType(name)
        module.__file__ = path
        sys.modules[name] = module
        linecache.cache[path] = (len(self.source), None, self.source.splitlines(keepends=True), path)
        exec(compile(self.source, path, "exec"), module.__dict__)
        return module.GeneratedModel(**kwargs)


def compile_program(program: Program, *, target: str = "eager") -> Artifact:
    """Compile graph bodies, not a model-class wrapper. No Torch import is needed."""
    verify(program)
    if target not in ("eager", "optimized"):
        raise ValueError("target must be eager or optimized")
    optimized = target == "optimized"
    if optimized and program.dtype != "float32":
        raise NotImplementedError(
            "optimized Torch 2.15 lowering is qualified for float32 only; "
            "low-precision fusion failed the full-model oracle gate; use target='eager'"
        )
    params = {x.name: x for x in program.parameters}
    operations = [op for b in program.blocks for op in b.body]
    usage = Counter(p for op in operations for p in op.parameters)
    if set(usage) != set(params) or any(count != 1 for count in usage.values()):
        raise ValueError("v1 requires each declared parameter role exactly once; use explicit tied aliases")
    storage, bindings, linear_aliases, projections, rope = [], {}, [], {}, {}

    def canonical(name):
        return params[name].alias_of or name

    # Plan ownership before emitting calls. A fused projection owns one native
    # storage allocation; logical checkpoint weights are row views, not copies.
    linear_index = 0
    for op in operations:
        if op.kind not in ("linear", "qkv", "gate_up"):
            continue
        groups = [op.parameters] if optimized else [(p,) for p in op.parameters]
        calls = []
        for group in groups:
            key = f"l{linear_index}"
            linear_index += 1
            if any(params[p].alias_of for p in group):
                if len(group) != 1:
                    raise ValueError("fusing tied parameter views is not implemented")
                linear_aliases.append({"key": key, "parameter": canonical(group[0])})
            else:
                rows = sum(params[p].shape[0] for p in group)
                storage.append({"kind": "linear", "key": key, "shape": (rows, params[group[0]].shape[1])})
                offset = 0
                for p in group:
                    stop = offset + params[p].shape[0]
                    bindings[p] = {"kind": "linear", "key": key, "start": offset, "stop": stop}
                    offset = stop
            calls.append(key)
        projections[op.name] = calls
    for p in program.parameters:
        if p.alias_of or p.name in bindings:
            continue
        key = f"p{len(storage)}"
        storage.append({"kind": "parameter", "key": key, "shape": p.shape})
        bindings[p.name] = {"kind": "parameter", "key": key}
    for p in program.parameters:
        if p.alias_of:
            bindings[p.name] = dict(bindings[p.alias_of])

    values, lines, next_value = {}, [], 0
    for index, t in enumerate(program.inputs):
        values[t.name] = ["tokens", "positions"][index]
    counts = Counter(x for op in operations for x in op.inputs)
    counts.update(t.name for t in program.outputs)
    cache_results = {}
    for op in operations:
        if op.kind == "cache_append":
            counts.update(x.name for x in op.outputs)

    def weight(name):
        b = bindings[name]
        if b["kind"] != "parameter":
            raise ValueError(f"{name}: non-projection parameter expected")
        return f"self.weights[{b['key']!r}]"

    resource_indices = {r.name: i for i, r in enumerate(program.resources)}
    for block in program.blocks:
        lines.append(f"        # Typed block {block.name!r}; resources={tuple(u.resource for u in block.resources)!r}")
        for op in block.body:
            xs = [values[x] for x in op.inputs]
            outs = []
            for t in op.outputs:
                variable = f"v{next_value}"
                next_value += 1
                values[t.name] = variable
                outs.append(variable)
            result = ",".join(outs)
            a = op.attributes
            if op.kind == "embedding":
                expression = f"F.embedding({xs[0]},{weight(op.parameters[0])})"
            elif op.kind == "rms_norm":
                expression = f"_rms({xs[0]},{weight(op.parameters[0])},{a['eps']!r})"
            elif op.kind in ("linear", "qkv", "gate_up"):
                calls = projections[op.name]
                if optimized and len(op.parameters) > 1:
                    widths = tuple(params[p].shape[0] for p in op.parameters)
                    expression = f"self.projections[{calls[0]!r}]({xs[0]}).split({widths!r},dim=-1)"
                else:
                    expression = ",".join(f"self.projections[{key!r}]({xs[0]})" for key in calls)
            elif op.kind == "rope":
                key = f"rope_{len(rope)}"
                rope[key] = a
                expression = (
                    f"_rope({','.join(xs)},self.{key},{a['query_heads']!r},{a['kv_heads']!r},{a['head_dim']!r})"
                )
            elif op.kind == "cache_append":
                index = resource_indices[op.resources[0].resource]
                r = program.resources[index]
                expression = (
                    f"torch.cat((past[{2 * index}],{xs[0]}),dim=2),"
                    f"torch.cat((past[{2 * index + 1}],{xs[1]}.reshape({xs[1]}.shape[0],{xs[1]}.shape[1],"
                    f"{r.heads},{r.head_dim}).transpose(1,2)),dim=2)"
                )
                cache_results[index] = outs
            elif op.kind == "attention":
                expression = f"_attention({','.join(xs)},{optimized!r})"
            elif op.kind == "add":
                expression = f"{xs[0]}+{xs[1]}"
            elif op.kind == "silu_mul":
                expression = f"F.silu({xs[0]})*{xs[1]}"
            else:
                raise AssertionError("verifier admitted an operation without lowering")
            lines.append(f"        {result} = {expression}")
            dead = []
            for name in op.inputs:
                counts[name] -= 1
                if counts[name] == 0 and values[name] not in dead:
                    dead.append(values[name])
            for t in op.outputs:
                if counts[t.name] == 0:
                    dead.append(values[t.name])
            if dead:
                lines.append(f"        del {','.join(dead)}")
    cache = [value for index in range(len(program.resources)) for value in cache_results[index]]
    lines.append(f"        return {values[program.outputs[0].name]},({','.join(cache)},)")
    manifest = {
        "version": 1,
        "torch_target": "2.15",
        "program_id": program.fingerprint,
        "target": target,
        "dtype": program.dtype,
        "storage": storage,
        "bindings": bindings,
        "linear_aliases": linear_aliases,
        "rope": rope,
        "parameters": {
            p.name: {"shape": p.shape, "alias_of": p.alias_of, "storage": p.storage} for p in program.parameters
        },
        "resources": [vars(r) for r in program.resources],
        "vocabulary": program.outputs[0].shape[-1],
        "blocks": [
            {
                "name": b.name,
                "inputs": [vars(t) for t in b.inputs],
                "outputs": [vars(t) for t in b.outputs],
                "resources": [vars(u) for u in b.resources],
            }
            for b in program.blocks
        ],
        "observations": [
            {
                "operation": op.name,
                "value": op.inputs[0],
                "parameters": op.parameters,
                "module_keys": projections[op.name],
            }
            for op in operations
            if op.name in projections
        ],
    }
    prelude = Path(__file__).with_name("_runtime.py").read_text(encoding="utf-8")
    source = (
        prelude
        + f"\n\nMANIFEST = {manifest!r}\n\n\nclass GeneratedModel(_ModelBase):\n"
        + "    def _execute(self,tokens,positions,past):\n"
        + "\n".join(lines)
        + "\n"
    )
    # Catch generator syntax defects without importing/executing Torch or model code.
    compile(source, "<generated-model>", "exec")
    return Artifact(source, manifest, program)
