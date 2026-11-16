# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Independent equation oracle. No compiler/runtime or Transformers helper imports.

The tiny fixtures can evaluate linear algebra in FP64, rounding at the model's
declared value boundaries. Checkpoint runs use native linear dtype to bound RAM.
"""

import math

import torch


@torch.no_grad()
def llama_reference(tokens, weights, config, *, past=None, offset=0, high_precision=False, trace=None):
    activation = weights["model.embed_tokens.weight"].dtype
    b, t = tokens.shape
    nq, nkv, d = config["query_heads"], config["kv_heads"], config["head_dim"]

    def linear(x, key):
        w = weights[key]
        if trace is not None:
            trace[key] = x.detach().clone()
        if high_precision:
            return (x.double() @ w.double().T).to(activation)
        return torch.nn.functional.linear(x, w)

    def norm(x, key):
        # Independent variance expression and explicit low-precision boundary.
        z = x.double() if high_precision else x.float()
        variance = torch.sum(z * z, dim=-1, keepdim=True) / x.shape[-1]
        normalized = (z * (variance + config["eps"]).rsqrt()).to(activation)
        return normalized * weights[key]

    # Python scalar piecewise equation, independent of the vectorized generator.
    frequencies = []
    for j in range(d // 2):
        f = config["theta"] ** (-2 * j / d)
        wavelength = 2 * math.pi / f
        if wavelength > config["original_context"] / config["low_frequency"]:
            f /= config["factor"]
        elif wavelength >= config["original_context"] / config["high_frequency"]:
            blend = (config["original_context"] / wavelength - config["low_frequency"]) / (
                config["high_frequency"] - config["low_frequency"]
            )
            f = f * (blend + (1 - blend) / config["factor"])
        frequencies.append(f)
    frequencies = torch.tensor(frequencies, dtype=torch.float32, device=tokens.device)
    angles = torch.arange(offset, offset + t, device=tokens.device, dtype=torch.float32)[:, None] * frequencies
    cosine, sine = angles.cos().to(activation), angles.sin().to(activation)

    def rotate(x, heads):
        x = x.view(b, t, heads, d).permute(0, 2, 1, 3)
        left, right = x[..., : d // 2], x[..., d // 2 :]
        return torch.cat((left * cosine - right * sine, right * cosine + left * sine), dim=-1)

    x = weights["model.embed_tokens.weight"][tokens]
    next_cache = []
    for layer in range(config["layers"]):
        prefix = f"model.layers.{layer}"
        saved = x
        z = norm(x, f"{prefix}.input_layernorm.weight")
        q = rotate(linear(z, f"{prefix}.self_attn.q_proj.weight"), nq)
        k = rotate(linear(z, f"{prefix}.self_attn.k_proj.weight"), nkv)
        v = linear(z, f"{prefix}.self_attn.v_proj.weight").view(b, t, nkv, d).permute(0, 2, 1, 3)
        if past is not None:
            k = torch.cat((past[2 * layer], k), dim=2)
            v = torch.cat((past[2 * layer + 1], v), dim=2)
        next_cache.extend((k.contiguous(), v.contiguous()))
        # One independent head at a time: no production GQA repeat/SDPA helper.
        result = []
        for head in range(nq):
            compute = torch.float64 if high_precision else torch.float32
            query, key, value = (
                q[:, head].to(compute),
                k[:, head // (nq // nkv)].to(compute),
                v[:, head // (nq // nkv)].to(compute),
            )
            scores = torch.matmul(query, key.transpose(1, 2)) * d**-0.5
            for row in range(t):
                scores[:, row, offset + row + 1 :] = -torch.inf
            # Use the specified Torch primitive. Replacing softmax by an
            # algebraically equivalent exp/sum changes BF16 rounding downstream.
            probability = torch.softmax(scores, dim=-1)
            result.append((probability @ value).to(activation))
        attended = torch.stack(result, dim=2).reshape(b, t, nq * d)
        x = saved + linear(attended, f"{prefix}.self_attn.o_proj.weight")
        saved = x
        z = norm(x, f"{prefix}.post_attention_layernorm.weight")
        gate = linear(z, f"{prefix}.mlp.gate_proj.weight")
        up = linear(z, f"{prefix}.mlp.up_proj.weight")
        # Match the explicit activation rounding boundary of Torch SiLU.
        gate = torch.nn.functional.silu(gate)
        x = saved + linear(gate * up, f"{prefix}.mlp.down_proj.weight")
    x = norm(x, "model.norm.weight")
    return linear(x, "model.embed_tokens.weight"), tuple(next_cache)
