# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Version-one, serializable Python DSL; executable operations have closed semantics.

No callbacks, Python expressions or imported model classes are accepted as bodies.
The text in the design RFC is pseudocode; these records are its first executable
authoring syntax. Shapes use B/T for the current chunk and S for visible history.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int | str, ...]


@dataclass(frozen=True)
class Parameter:
    name: str
    shape: tuple[int, ...]
    alias_of: str | None = None
    storage: str = "native_float"


@dataclass(frozen=True)
class Resource:
    name: str
    dtype: str
    heads: int
    head_dim: int
    capacity: int
    owner: str = "request"
    lifetime: str = "request"
    layout: str = "B,H,S,D"


@dataclass(frozen=True)
class ResourceUse:
    resource: str
    access: str
    at: str
    phases: tuple[str, ...] = ("prefill", "decode", "verify", "calibrate")


@dataclass(frozen=True)
class Operation:
    name: str
    kind: str
    inputs: tuple[str, ...]
    outputs: tuple[Tensor, ...]
    parameters: tuple[str, ...]
    attributes: dict[str, Any]
    resources: tuple[ResourceUse, ...]


@dataclass(frozen=True)
class Block:
    name: str
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    resources: tuple[ResourceUse, ...]
    body: tuple[Operation, ...]


@dataclass(frozen=True)
class Program:
    name: str
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    parameters: tuple[Parameter, ...]
    resources: tuple[Resource, ...]
    blocks: tuple[Block, ...]
    dtype: str
    version: int = 1
    torch_target: str = "2.15"

    def to_dict(self) -> dict:
        # Normalize tuples for portable JSON and deterministic fingerprints.
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> "Program":
        def tensor(x):
            return Tensor(x["name"], x["dtype"], tuple(x["shape"]))

        def use(x):
            return ResourceUse(x["resource"], x["access"], x["at"], tuple(x["phases"]))

        def operation(x):
            return Operation(
                x["name"],
                x["kind"],
                tuple(x["inputs"]),
                tuple(tensor(v) for v in x["outputs"]),
                tuple(x["parameters"]),
                x["attributes"],
                tuple(use(v) for v in x["resources"]),
            )

        return cls(
            name=data["name"],
            inputs=tuple(tensor(x) for x in data["inputs"]),
            outputs=tuple(tensor(x) for x in data["outputs"]),
            parameters=tuple(
                Parameter(x["name"], tuple(x["shape"]), x.get("alias_of"), x.get("storage", "native_float"))
                for x in data["parameters"]
            ),
            resources=tuple(Resource(**x) for x in data["resources"]),
            blocks=tuple(
                Block(
                    x["name"],
                    tuple(tensor(v) for v in x["inputs"]),
                    tuple(tensor(v) for v in x["outputs"]),
                    tuple(use(v) for v in x["resources"]),
                    tuple(operation(v) for v in x["body"]),
                )
                for x in data["blocks"]
            ),
            dtype=data["dtype"],
            version=data["version"],
            torch_target=data["torch_target"],
        )


class VerificationError(ValueError):
    """An invalid program never reaches code generation or resource writes."""


def verify(program: Program) -> None:
    """Infer every supported operation's type; declarations are checked, not trusted."""

    def require(condition, message):
        if not condition:
            raise VerificationError(message)

    def unique(items, label):
        result = {x.name: x for x in items}
        require(len(result) == len(items), f"duplicate {label}")
        return result

    require(program.version == 1 and program.torch_target == "2.15", "only DSL v1 / Torch 2.15 supported")
    require(program.dtype in ("float32", "float16", "bfloat16"), "unsupported activation dtype")
    params = unique(program.parameters, "parameter")
    resources = unique(program.resources, "resource")
    values = unique(program.inputs, "input")
    require(len(program.inputs) == 2, "v1 entry requires token IDs and positions")
    require(
        [(x.dtype, x.shape) for x in program.inputs] == [("int64", ("B", "T")), ("int64", ("T",))],
        "entry must be int64[B,T] token IDs and int64[T] contiguous positions",
    )
    require(program.outputs and program.resources, "v1 requires outputs and declared KV resources")
    for p in params.values():
        require(p.storage == "native_float", f"{p.name}: packed quantized lowering is not implemented")
        require(p.shape and all(type(n) is int and n > 0 for n in p.shape), f"{p.name}: invalid shape")
        if p.alias_of:
            require(p.alias_of in params and not params[p.alias_of].alias_of, f"{p.name}: invalid alias")
            require(p.shape == params[p.alias_of].shape, f"{p.name}: alias shape mismatch")
    for r in resources.values():
        require(r.dtype == program.dtype and r.layout == "B,H,S,D", f"{r.name}: incompatible KV schema")
        require(r.owner == "request" and r.lifetime == "request", f"{r.name}: invalid ownership")
        require(
            all(type(n) is int and n > 0 for n in (r.heads, r.head_dim, r.capacity)),
            f"{r.name}: invalid capacity/shape",
        )
    operation_names, writers, block_names = set(), set(), set()
    for block in program.blocks:
        require(block.name not in block_names, f"duplicate block {block.name}")
        block_names.add(block.name)
        local = unique(block.inputs, f"{block.name} input")
        for t in local.values():
            require(values.get(t.name) == t, f"{block.name}: missing/mistyped input {t.name}")
        uses = []
        for op in block.body:
            require(op.name not in operation_names, f"duplicate operation {op.name}")
            operation_names.add(op.name)
            require(all(x in local for x in op.inputs), f"{op.name}: undeclared value dependency")
            require(all(x in params for x in op.parameters), f"{op.name}: missing parameter")
            xs, ps, a = [local[x] for x in op.inputs], [params[x] for x in op.parameters], op.attributes
            allowed_attributes = (
                {"eps"}
                if op.kind == "rms_norm"
                else {
                    "type",
                    "query_heads",
                    "kv_heads",
                    "head_dim",
                    "theta",
                    "factor",
                    "original_context",
                    "low_frequency",
                    "high_frequency",
                }
                if op.kind == "rope"
                else set()
            )
            require(set(a) == allowed_attributes, f"{op.name}: unknown or missing operation attributes")
            dt = program.dtype
            expected = []

            def output(shape, dtype=dt):
                expected.append((dtype, tuple(shape)))

            def arity(inputs, parameters, outputs):
                require(
                    (len(xs), len(ps), len(op.outputs)) == (inputs, parameters, outputs), f"{op.name}: invalid arity"
                )

            def linear(x, p):
                require(
                    x.dtype == dt and len(p.shape) == 2 and x.shape[-1] == p.shape[1],
                    f"{op.name}: incompatible linear",
                )
                output((*x.shape[:-1], p.shape[0]))

            if op.kind == "embedding":
                arity(1, 1, 1)
                require(xs[0].dtype == "int64" and len(ps[0].shape) == 2, f"{op.name}: embedding type")
                output((*xs[0].shape, ps[0].shape[1]))
            elif op.kind == "rms_norm":
                arity(1, 1, 1)
                require(ps[0].shape == (xs[0].shape[-1],) and xs[0].dtype == dt, f"{op.name}: norm gain")
                require(type(a.get("eps")) in (int, float) and 0 < a["eps"] < 1, f"{op.name}: norm eps")
                output(xs[0].shape)
            elif op.kind in ("linear", "qkv", "gate_up"):
                count = {"linear": 1, "qkv": 3, "gate_up": 2}[op.kind]
                arity(1, count, count)
                for p in ps:
                    linear(xs[0], p)
            elif op.kind == "rope":
                arity(3, 0, 2)
                hq, hk, d = a.get("query_heads"), a.get("kv_heads"), a.get("head_dim")
                require(
                    all(type(n) is int and n > 0 for n in (hq, hk, d)) and d % 2 == 0 and hq % hk == 0,
                    f"{op.name}: invalid GQA dimensions",
                )
                require(
                    xs[0].shape == ("B", "T", hq * d)
                    and xs[1].shape == ("B", "T", hk * d)
                    and xs[2].shape == ("T",)
                    and xs[2].dtype == "int64",
                    f"{op.name}: RoPE shape",
                )
                require(a.get("type") == "llama3", f"{op.name}: unknown RoPE semantics")
                require(
                    a.get("theta", 0) > 0
                    and a.get("factor", 0) > 0
                    and a.get("original_context", 0) > 0
                    and a.get("high_frequency", 0) > a.get("low_frequency", 0) > 0,
                    f"{op.name}: RoPE constants",
                )
                output(("B", hq, "T", d))
                output(("B", hk, "T", d))
            elif op.kind == "cache_append":
                arity(2, 0, 2)
                require(len(op.resources) == 1, f"{op.name}: missing cache resource port")
                use = op.resources[0]
                require(
                    use.resource in resources and use.access == "read_append" and use.at == op.name,
                    f"{op.name}: illegal resource mapping",
                )
                require(
                    set(use.phases) == {"prefill", "decode", "verify", "calibrate"},
                    f"{op.name}: incompatible phase interface",
                )
                require(use.resource not in writers, f"{op.name}: multiple cache writers")
                writers.add(use.resource)
                r = resources[use.resource]
                require(
                    xs[0].shape == ("B", r.heads, "T", r.head_dim) and xs[1].shape == ("B", "T", r.heads * r.head_dim),
                    f"{op.name}: cache shape",
                )
                output(("B", r.heads, "S", r.head_dim))
                output(("B", r.heads, "S", r.head_dim))
            elif op.kind == "attention":
                arity(4, 0, 1)
                q, k, v, pos = xs
                require(
                    len(q.shape) == 4
                    and len(k.shape) == 4
                    and k.shape == v.shape
                    and q.shape[0] == k.shape[0] == "B"
                    and q.shape[2] == "T"
                    and k.shape[2] == "S"
                    and q.shape[-1] == k.shape[-1]
                    and q.shape[1] % k.shape[1] == 0
                    and pos.dtype == "int64"
                    and pos.shape == ("T",),
                    f"{op.name}: attention types",
                )
                output(("B", "T", q.shape[1] * q.shape[-1]))
            elif op.kind in ("add", "silu_mul"):
                arity(2, 0, 1)
                require(
                    xs[0].shape == xs[1].shape and xs[0].dtype == xs[1].dtype == dt, f"{op.name}: elementwise types"
                )
                output(xs[0].shape)
            else:
                raise VerificationError(f"{op.name}: unsupported operation {op.kind!r}")
            require(op.kind == "cache_append" or not op.resources, f"{op.name}: undeclared state semantics")
            require([(x.dtype, x.shape) for x in op.outputs] == expected, f"{op.name}: output type mismatch")
            for t in op.outputs:
                require(t.name not in values and t.name not in local, f"{op.name}: duplicate SSA value {t.name}")
                local[t.name] = t
            uses.extend(op.resources)
        require(tuple(uses) == block.resources, f"{block.name}: resource interface/body mismatch")
        for t in block.outputs:
            require(local.get(t.name) == t, f"{block.name}: missing output {t.name}")
            require(t.name not in values, f"{block.name}: duplicate output {t.name}")
            values[t.name] = t
    require(writers == set(resources), "resource manifest has unbound/unused cache slots")
    require(all(values.get(t.name) == t for t in program.outputs), "program result mismatch")
    require(
        len(program.outputs) == 1 and program.outputs[0].shape[:2] == ("B", "T"),
        "v1 requires one [B,T,vocabulary] logits result",
    )
