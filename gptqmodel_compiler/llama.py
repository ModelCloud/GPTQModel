# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Llama computation authored in the DSL, not a compiler architecture switch."""

from .ir import Block, Operation, Parameter, Program, Resource, ResourceUse, Tensor


def llama_program(
    *,
    hidden_size: int,
    intermediate_size: int,
    layers: int,
    vocabulary: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    capacity: int,
    dtype: str = "float32",
    rms_epsilon: float = 1e-5,
    rope_theta: float = 500000.0,
    rope_factor: float = 32.0,
    original_context: int = 8192,
    low_frequency: float = 1.0,
    high_frequency: float = 4.0,
    name: str = "llama",
) -> Program:
    """Instantiate every layer, value, parameter and resource before compilation."""
    for dimension in (hidden_size, intermediate_size, layers, vocabulary, query_heads, kv_heads, head_dim, capacity):
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("model dimensions must be positive integers")
    if query_heads * head_dim != hidden_size or query_heads % kv_heads or head_dim % 2:
        raise ValueError("invalid Llama GQA/head dimensions")
    h, i, nq, nkv, d = hidden_size, intermediate_size, query_heads, kv_heads, head_dim
    tokens, positions = Tensor("tokens", "int64", ("B", "T")), Tensor("positions", "int64", ("T",))
    parameters, resources, blocks = [], [], []

    def weight(key, shape, alias=None):
        parameters.append(Parameter(key, shape, alias))
        return key

    def tensor(key, shape):
        return Tensor(key, dtype, shape)

    def op(key, kind, inputs, outputs, params=(), attributes=None, uses=()):
        return Operation(
            key, kind, tuple(x.name for x in inputs), tuple(outputs), tuple(params), attributes or {}, tuple(uses)
        )

    embedding = weight("model.embed_tokens.weight", (vocabulary, h))
    x = tensor("embedded", ("B", "T", h))
    blocks.append(Block("embedding", (tokens,), (x,), (), (op("embed", "embedding", (tokens,), (x,), (embedding,)),)))
    for layer in range(layers):
        prefix, scope = f"model.layers.{layer}", f"decoder.{layer}"
        norm = weight(f"{prefix}.input_layernorm.weight", (h,))
        n, q, k, v = (
            tensor(f"{scope}.{key}", ("B", "T", width))
            for key, width in [("normalized", h), ("q", nq * d), ("k", nkv * d), ("v", nkv * d)]
        )
        qr, kr = tensor(f"{scope}.q_rot", ("B", nq, "T", d)), tensor(f"{scope}.k_rot", ("B", nkv, "T", d))
        kh, vh = tensor(f"{scope}.k_history", ("B", nkv, "S", d)), tensor(f"{scope}.v_history", ("B", nkv, "S", d))
        a, projected, residual = (
            tensor(f"{scope}.{key}", ("B", "T", h)) for key in ["context", "attention", "residual"]
        )
        qkv = tuple(
            weight(f"{prefix}.self_attn.{key}_proj.weight", (width, h))
            for key, width in [("q", nq * d), ("k", nkv * d), ("v", nkv * d)]
        )
        out = weight(f"{prefix}.self_attn.o_proj.weight", (h, nq * d))
        cache = Resource(f"{scope}.kv", dtype, nkv, d, capacity)
        resources.append(cache)
        use = ResourceUse(cache.name, "read_append", f"{scope}.append")
        body = (
            op(f"{scope}.input_norm", "rms_norm", (x,), (n,), (norm,), {"eps": rms_epsilon}),
            op(f"{scope}.qkv", "qkv", (n,), (q, k, v), qkv),
            op(
                f"{scope}.rope",
                "rope",
                (q, k, positions),
                (qr, kr),
                attributes={
                    "type": "llama3",
                    "query_heads": nq,
                    "kv_heads": nkv,
                    "head_dim": d,
                    "theta": rope_theta,
                    "factor": rope_factor,
                    "original_context": original_context,
                    "low_frequency": low_frequency,
                    "high_frequency": high_frequency,
                },
            ),
            op(use.at, "cache_append", (kr, v), (kh, vh), uses=(use,)),
            op(f"{scope}.attend", "attention", (qr, kh, vh, positions), (a,)),
            op(f"{scope}.out", "linear", (a,), (projected,), (out,)),
            op(f"{scope}.attention_residual", "add", (x, projected), (residual,)),
        )
        blocks.append(Block(f"{scope}.attention", (x, positions), (residual,), (use,), body))
        post = weight(f"{prefix}.post_attention_layernorm.weight", (h,))
        normalized, gate, up, product, down, y = (
            tensor(f"{scope}.{key}", ("B", "T", width))
            for key, width in [
                ("mlp_input", h),
                ("gate", i),
                ("up", i),
                ("product", i),
                ("mlp_output", h),
                ("output", h),
            ]
        )
        gate_w, up_w = tuple(weight(f"{prefix}.mlp.{key}_proj.weight", (i, h)) for key in ["gate", "up"])
        down_w = weight(f"{prefix}.mlp.down_proj.weight", (h, i))
        blocks.append(
            Block(
                f"{scope}.mlp",
                (residual,),
                (y,),
                (),
                (
                    op(f"{scope}.post_norm", "rms_norm", (residual,), (normalized,), (post,), {"eps": rms_epsilon}),
                    op(f"{scope}.gate_up", "gate_up", (normalized,), (gate, up), (gate_w, up_w)),
                    op(f"{scope}.activation", "silu_mul", (gate, up), (product,)),
                    op(f"{scope}.down", "linear", (product,), (down,), (down_w,)),
                    op(f"{scope}.mlp_residual", "add", (residual, down), (y,)),
                ),
            )
        )
        x = y
    normalized, logits = tensor("final_normalized", ("B", "T", h)), tensor("logits", ("B", "T", vocabulary))
    norm, head = weight("model.norm.weight", (h,)), weight("lm_head.weight", (vocabulary, h), embedding)
    blocks.append(
        Block(
            "output",
            (x,),
            (logits,),
            (),
            (
                op("final_norm", "rms_norm", (x,), (normalized,), (norm,), {"eps": rms_epsilon}),
                op("lm_head", "linear", (normalized,), (logits,), (head,)),
            ),
        )
    )
    return Program(name, (tokens, positions), (logits,), tuple(parameters), tuple(resources), tuple(blocks), dtype)


def llama32_1b(*, dtype: str = "float32", capacity: int = 131072) -> Program:
    if not 0 < capacity <= 131072:
        raise ValueError("Llama 3.2 1B capacity must be in [1,131072]")
    return llama_program(
        hidden_size=2048,
        intermediate_size=8192,
        layers=16,
        vocabulary=128256,
        query_heads=32,
        kv_heads=8,
        head_dim=64,
        capacity=capacity,
        dtype=dtype,
        name="llama-3.2-1b-instruct",
    )
