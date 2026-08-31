# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Llama-family graph hooks for the generic QVQ transform planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .qvq_transform_planner import (
    ProjectionRole,
    ProjectionSemantic,
    QVQTransformPlan,
    TransformKind,
    TransformPlacement,
)
from .rotation.hadamard_utils import matmul_hadU


@dataclass(frozen=True)
class _RandomHadamardBasis:
    signs: torch.Tensor

    @property
    def size(self) -> int:
        return self.signs.numel()

    def apply_right(self, values: torch.Tensor) -> torch.Tensor:
        signs = self.signs.to(device=values.device, dtype=values.dtype)
        return matmul_hadU(values * signs)

    def apply_right_transpose(self, values: torch.Tensor) -> torch.Tensor:
        signs = self.signs.to(device=values.device, dtype=values.dtype)
        return matmul_hadU(values, transpose=True) * signs

    def apply_left_transpose(self, values: torch.Tensor) -> torch.Tensor:
        return self.apply_right(values.transpose(0, 1)).transpose(0, 1)


def _random_hadamard_basis(size: int, *, seed: int) -> _RandomHadamardBasis:
    if size < 1 or size & (size - 1):
        raise ValueError(f"QVQ randomized Hadamard basis requires a power-of-two width, got {size}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int8).mul_(2).sub_(1)
    return _RandomHadamardBasis(signs.to(torch.float32))


def _copy_weight(module: torch.nn.Module, value: torch.Tensor) -> None:
    module.weight.data.copy_(value.to(device=module.weight.device, dtype=module.weight.dtype))


def _copy_bias(module: torch.nn.Module, value: torch.Tensor) -> None:
    if module.bias is None:
        module.bias = torch.nn.Parameter(value.to(device=module.weight.device, dtype=module.weight.dtype))
    else:
        module.bias.data.copy_(value.to(device=module.bias.device, dtype=module.bias.dtype))


def _fuse_norm(norm: torch.nn.Module, linears: tuple[torch.nn.Module, ...]) -> None:
    norm_weight = norm.weight.detach().to(torch.float32)
    norm_bias = getattr(norm, "bias", None)
    for linear in linears:
        original = linear.weight.detach().to(torch.float32)
        _copy_weight(linear, original * norm_weight.unsqueeze(0))
        if norm_bias is not None:
            correction = original @ norm_bias.detach().to(torch.float32)
            current = torch.zeros_like(correction) if linear.bias is None else linear.bias.detach().to(torch.float32)
            _copy_bias(linear, current + correction)
    norm.weight.data.fill_(1)
    if norm_bias is not None:
        norm.bias.data.zero_()


def _transform_output_pairs(
    linear: torch.nn.Module,
    pair_maps: torch.Tensor,
    *,
    head_count: int,
    head_dim: int,
    q_per_kv: int,
) -> None:
    """Apply split-half rotary-pair maps to projection output coordinates."""

    weight = linear.weight.detach().to(torch.float32).reshape(head_count, head_dim, linear.in_features)
    half = head_dim // 2
    first, second = weight[:, :half], weight[:, half:]
    pairs = torch.stack((first, second), dim=2)
    map_indices = torch.arange(head_count, device=pair_maps.device) // q_per_kv
    maps = pair_maps.index_select(0, map_indices)
    transformed = torch.einsum("hpab,hpbi->hpai", maps.transpose(-1, -2), pairs.to(maps.device))
    merged = torch.cat((transformed[:, :, 0], transformed[:, :, 1]), dim=1)
    _copy_weight(linear, merged.reshape(linear.out_features, linear.in_features))
    if linear.bias is not None:
        bias = linear.bias.detach().to(torch.float32).reshape(head_count, head_dim)
        bias_pairs = torch.stack((bias[:, :half], bias[:, half:]), dim=2)
        transformed_bias = torch.einsum(
            "hpab,hpb->hpa",
            maps.transpose(-1, -2),
            bias_pairs.to(maps.device),
        )
        _copy_bias(linear, torch.cat((transformed_bias[:, :, 0], transformed_bias[:, :, 1]), dim=1).reshape(-1))


class LlamaQVQTransformImplementor:
    """Architecture-specific Llama graph rewrite; runtime modules see only descriptors."""

    _ROLE_PATHS = (
        (ProjectionRole.ATTENTION_Q, "self_attn.q_proj"),
        (ProjectionRole.ATTENTION_K, "self_attn.k_proj"),
        (ProjectionRole.ATTENTION_V, "self_attn.v_proj"),
        (ProjectionRole.ATTENTION_O, "self_attn.o_proj"),
        (ProjectionRole.MLP_GATE, "mlp.gate_proj"),
        (ProjectionRole.MLP_UP, "mlp.up_proj"),
        (ProjectionRole.MLP_DOWN, "mlp.down_proj"),
    )

    @staticmethod
    def _layers(model: torch.nn.Module):
        root = getattr(model, "model", None)
        layers = None if root is None else getattr(root, "layers", None)
        if layers is None:
            raise ValueError("Llama QVQ transform implementor requires model.model.layers")
        return root, layers

    @staticmethod
    def _resolve(parent: torch.nn.Module, path: str) -> torch.nn.Module:
        value = parent
        for component in path.split("."):
            value = getattr(value, component)
        return value

    def analyze_model_graph(self, model: torch.nn.Module) -> tuple[ProjectionSemantic, ...]:
        _, layers = self._layers(model)
        semantics = []
        for layer_index, layer in enumerate(layers):
            for role, path in self._ROLE_PATHS:
                module = self._resolve(layer, path)
                if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
                    raise TypeError(f"Llama QVQ projection `{path}` is not linear-like")
                semantics.append(
                    ProjectionSemantic(
                        name=f"model.layers.{layer_index}.{path}",
                        layer_index=layer_index,
                        role=role,
                        in_features=int(module.in_features),
                        out_features=int(module.out_features),
                        module=module,
                    )
                )
        return tuple(semantics)

    @torch.inference_mode()
    def rewrite_dense_weights(
        self,
        model: torch.nn.Module,
        plan: QVQTransformPlan,
        *,
        seed: int,
    ) -> dict[str, Any]:
        root, layers = self._layers(model)
        if plan.arm == "A0":
            return {"arm": "A0", "rewritten": False, "blockers": []}
        if plan.arm in {"A2", "A8", "A9"}:
            raise NotImplementedError(
                f"{plan.arm} requires a held-out-fitted structured orthogonal basis; "
                "the Llama rewriter refuses to substitute a fixed Hadamard and call it learned"
            )
        config = model.config
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        num_kv_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(config, "head_dim", hidden_size // num_heads))
        if head_dim % 2:
            raise ValueError("RoPE-compatible QVQ pair folding requires an even head dimension")
        q_per_kv = num_heads // num_kv_heads
        descriptor_by_role = {
            (descriptor.module_name, descriptor.role): descriptor for descriptor in plan.modules
        }
        del descriptor_by_role

        residual_folded = plan.arm not in {"A24", "A25", "A26", "A27", "A28", "A29", "A30"}
        if residual_folded:
            residual_basis = _random_hadamard_basis(hidden_size, seed=seed)
            self._rewrite_residual_basis(model, root, layers, residual_basis)
        metadata: dict[str, Any] = {
            "arm": plan.arm,
            "rewritten": True,
            "residual_seed": seed,
            "residual_kind": next(
                descriptor.input_transform.kind.value
                for descriptor in plan.modules
                if descriptor.role == ProjectionRole.ATTENTION_Q
            ),
            "blockers": [],
            "residual_folded": residual_folded,
        }

        vo_enabled = any(
            descriptor.role == ProjectionRole.ATTENTION_V
            and descriptor.output_transform.placement == TransformPlacement.FOLDED
            for descriptor in plan.modules
        )
        if vo_enabled:
            for layer_index, layer in enumerate(layers):
                self._rewrite_vo(layer, layer_index, num_heads, num_kv_heads, head_dim, q_per_kv, seed)
            metadata["vo_folded"] = True

        qk_enabled = any(
            descriptor.role in {ProjectionRole.ATTENTION_Q, ProjectionRole.ATTENTION_K}
            and descriptor.output_transform.kind == TransformKind.ROPE_PAIR
            for descriptor in plan.modules
        )
        if qk_enabled:
            for layer_index, layer in enumerate(layers):
                attention = layer.self_attn
                if getattr(attention, "q_norm", None) is not None or getattr(attention, "k_norm", None) is not None:
                    blocker = f"layer {layer_index} has q_norm/k_norm between projection and RoPE"
                    metadata["blockers"].append(blocker)
                    raise ValueError(blocker)
                if plan.arm != "A21":
                    self._rewrite_qk(
                        attention,
                        layer_index,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        q_per_kv,
                        seed,
                        use_scaling=plan.arm != "A20",
                    )
            metadata["qk_rope_pair_folded"] = True
            metadata["qk_rope_pair_variant"] = {
                "A20": "rotation_only",
                "A21": "identity",
            }.get(plan.arm, "rotation_and_reciprocal_scaling")

        swiglu_enabled = any(
            descriptor.role == ProjectionRole.MLP_GATE
            and descriptor.output_transform.kind == TransformKind.PERMUTATION
            for descriptor in plan.modules
        )
        if swiglu_enabled:
            for layer_index, layer in enumerate(layers):
                self._rewrite_swiglu(
                    layer.mlp,
                    layer_index,
                    seed,
                    use_scaling=plan.arm not in {"A27", "A28"},
                )
            metadata["swiglu_permutation_scaling_folded"] = True
        return metadata

    @staticmethod
    def _rewrite_residual_basis(model, root, layers, basis: _RandomHadamardBasis) -> None:
        lm_head = model.lm_head
        embedding = root.embed_tokens
        if embedding.weight is lm_head.weight:
            lm_head.weight = torch.nn.Parameter(lm_head.weight.detach().clone())
        for layer in layers:
            _fuse_norm(
                layer.input_layernorm,
                (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj),
            )
            _fuse_norm(
                layer.post_attention_layernorm,
                (layer.mlp.gate_proj, layer.mlp.up_proj),
            )
        _fuse_norm(root.norm, (lm_head,))

        _copy_weight(embedding, basis.apply_right(embedding.weight.detach().to(torch.float32)))
        _copy_weight(lm_head, basis.apply_right(lm_head.weight.detach().to(torch.float32)))
        for layer in layers:
            for module in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
            ):
                _copy_weight(module, basis.apply_right(module.weight.detach().to(torch.float32)))
            for module in (layer.self_attn.o_proj, layer.mlp.down_proj):
                _copy_weight(module, basis.apply_left_transpose(module.weight.detach().to(torch.float32)))
                if module.bias is not None:
                    _copy_bias(
                        module,
                        basis.apply_right(module.bias.detach().to(torch.float32).unsqueeze(0)).squeeze(0),
                    )

    @staticmethod
    def _rewrite_vo(layer, layer_index, num_heads, num_kv_heads, head_dim, q_per_kv, seed):
        bases = [
            _random_hadamard_basis(head_dim, seed=seed + 1009 * (layer_index + 1) + head)
            for head in range(num_kv_heads)
        ]
        v_proj = layer.self_attn.v_proj
        v_weight = v_proj.weight.detach().to(torch.float32).reshape(num_kv_heads, head_dim, v_proj.in_features)
        transformed_v = torch.stack(
            [basis.apply_left_transpose(v_weight[index]) for index, basis in enumerate(bases)]
        )
        _copy_weight(v_proj, transformed_v.reshape(v_proj.out_features, v_proj.in_features))
        if v_proj.bias is not None:
            v_bias = v_proj.bias.detach().to(torch.float32).reshape(num_kv_heads, head_dim)
            _copy_bias(
                v_proj,
                torch.stack(
                    [basis.apply_right(row.unsqueeze(0)).squeeze(0) for basis, row in zip(bases, v_bias)]
                ).reshape(-1),
            )

        o_proj = layer.self_attn.o_proj
        o_weight = o_proj.weight.detach().to(torch.float32).reshape(o_proj.out_features, num_heads, head_dim)
        transformed_o = torch.stack(
            [bases[head // q_per_kv].apply_right(o_weight[:, head]) for head in range(num_heads)],
            dim=1,
        )
        _copy_weight(o_proj, transformed_o.reshape(o_proj.out_features, o_proj.in_features))

    @staticmethod
    def _rewrite_qk(
        attention,
        layer_index,
        num_heads,
        num_kv_heads,
        head_dim,
        q_per_kv,
        seed,
        *,
        use_scaling: bool,
    ):
        generator = torch.Generator(device="cpu").manual_seed(seed + 65537 * (layer_index + 1))
        angles = (torch.rand((num_kv_heads, head_dim // 2), generator=generator) - 0.5) * 1.5
        log2_scales = torch.randint(-1, 2, angles.shape, generator=generator, dtype=torch.int64)
        scales = torch.exp2(log2_scales.to(torch.float32)) if use_scaling else torch.ones_like(angles)
        cosine, sine = torch.cos(angles), torch.sin(angles)
        rotations = torch.stack(
            (
                torch.stack((cosine, sine), dim=-1),
                torch.stack((-sine, cosine), dim=-1),
            ),
            dim=-2,
        )
        k_maps = scales[..., None, None] * rotations
        q_maps = scales.reciprocal()[..., None, None] * rotations
        _transform_output_pairs(
            attention.q_proj,
            q_maps,
            head_count=num_heads,
            head_dim=head_dim,
            q_per_kv=q_per_kv,
        )
        _transform_output_pairs(
            attention.k_proj,
            k_maps,
            head_count=num_kv_heads,
            head_dim=head_dim,
            q_per_kv=1,
        )

    @staticmethod
    def _rewrite_swiglu(mlp, layer_index: int, seed: int, *, use_scaling: bool = True):
        intermediate = mlp.gate_proj.out_features
        generator = torch.Generator(device="cpu").manual_seed(seed + 104729 * (layer_index + 1))
        permutation = torch.randperm(intermediate, generator=generator)
        exponents = torch.randint(-1, 2, (intermediate,), generator=generator, dtype=torch.int64)
        scales = (
            torch.exp2(exponents.to(torch.float32))
            if use_scaling
            else torch.ones_like(exponents, dtype=torch.float32)
        )
        gate_weight = mlp.gate_proj.weight.detach().to(torch.float32)
        up_weight = mlp.up_proj.weight.detach().to(torch.float32)
        down_weight = mlp.down_proj.weight.detach().to(torch.float32)
        _copy_weight(mlp.gate_proj, gate_weight[permutation])
        _copy_weight(mlp.up_proj, up_weight[permutation] * scales.unsqueeze(1))
        _copy_weight(mlp.down_proj, down_weight[:, permutation] / scales.unsqueeze(0))
        if mlp.gate_proj.bias is not None:
            _copy_bias(mlp.gate_proj, mlp.gate_proj.bias.detach().to(torch.float32)[permutation])
        if mlp.up_proj.bias is not None:
            _copy_bias(
                mlp.up_proj,
                mlp.up_proj.bias.detach().to(torch.float32)[permutation] * scales,
            )


__all__ = ["LlamaQVQTransformImplementor"]
