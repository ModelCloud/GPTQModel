# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import inspect

import torch

from ..base import BaseQModel


class BloomQModel(BaseQModel):
    pre_lm_head_norm_module = "transformer.ln_f"

    module_tree = [
        "transformer",
        "h",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attention": ("query_key_value:0", "dense:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            # BloomGelu between the two projections is not positively homogeneous.
            "mlp": ("dense_h_to_4h:0:nosrc", "dense_4h_to_h:1"),
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        if getattr(module, "apply_residual_connection_post_layernorm", False):
            return []
        return super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)

    def awq_module_replay_inputs(self, module, x, module_kwargs):
        """Supply a fixed additive residual, but never guess attention context."""
        x, module_kwargs = super().awq_module_replay_inputs(module, x, module_kwargs)
        try:
            parameters = inspect.signature(module.forward).parameters
        except (TypeError, ValueError):
            return x, module_kwargs
        if "residual" not in parameters and "alibi" not in parameters:
            return x, module_kwargs

        kwargs = dict(module_kwargs)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise NotImplementedError("Bloom AWQ replay requires [batch, sequence, hidden] activations")
        if "residual" in parameters and not torch.is_tensor(kwargs.get("residual")):
            kwargs["residual"] = x
        if "alibi" not in parameters:
            return x, kwargs

        batch_size, seq_len = x.shape[:2]
        num_heads = getattr(module, "num_heads", None)
        if not isinstance(num_heads, int) or num_heads < 1:
            raise NotImplementedError("Bloom AWQ replay requires an explicit attention head count")
        mask = kwargs.get("attention_mask")
        if (
            not torch.is_tensor(mask)
            or mask.dim() != 4
            or mask.shape[0] not in (1, batch_size)
            or mask.shape[1:] != (1, seq_len, seq_len)
        ):
            raise NotImplementedError("Bloom AWQ replay requires a captured, sequence-aligned attention_mask")
        alibi = kwargs.get("alibi")
        position_ids = kwargs.get("position_ids")
        if not torch.is_tensor(position_ids):
            position_ids = kwargs.get("_awq_rotary_position_ids")
        expected_shape = (batch_size * num_heads, 1, seq_len)
        if torch.is_tensor(position_ids):
            positions = position_ids
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)
            if (
                positions.dim() == 2
                and positions.shape[-1] == seq_len
                and positions.shape[0] in (1, batch_size)
            ):
                if positions.shape[0] == 1 and batch_size > 1:
                    positions = positions.expand(batch_size, -1)
                kwargs["alibi"] = self._awq_rebuild_alibi(
                    alibi, positions, num_heads, x.dtype, x.device,
                )
                return x, kwargs
        if not torch.is_tensor(alibi) or alibi.shape != expected_shape:
            got = tuple(alibi.shape) if torch.is_tensor(alibi) else None
            raise NotImplementedError(
                "Bloom AWQ replay requires a captured, batch/sequence-aligned "
                f"alibi (expected {expected_shape}, got {got}) or replay "
                "position_ids to rebuild it"
            )
        return x, kwargs

    @staticmethod
    def _awq_rebuild_alibi(alibi, position_ids, num_heads, dtype, device):
        """Rebuild ALiBi for packed rows from their real positions.

        ``build_alibi_tensor`` is ``slope * position`` per key and is
        query-independent, so a packed replay uses the retained keys'
        positions instead of the captured sample's ``0..S-1`` range.
        """

        try:
            from transformers.models.bloom.modeling_bloom import _get_alibi_slopes

            slopes = _get_alibi_slopes(num_heads, device)
        except Exception:
            # Derive from the captured tensor only when the helper is absent.
            # Index 1 is the first key after BOS, so a padded sample would not
            # give the slope; the helper above is the authoritative source.
            if not (
                torch.is_tensor(alibi)
                and alibi.dim() == 3
                and alibi.shape[-1] > 1
                and alibi.shape[0] >= num_heads
            ):
                raise NotImplementedError(
                    "Bloom AWQ replay cannot resolve ALiBi slopes to rebuild "
                    "the packed alibi"
                )
            slopes = alibi.detach().reshape(-1, num_heads, 1, alibi.shape[-1])[0, :, 0, 1]
        # `build_alibi_tensor` multiplies fp32 slopes by the integer position
        # and only then casts to the activation dtype; match that order so the
        # rebuilt tensor is bit-identical to the captured one.
        slopes = slopes.to(device=device, dtype=torch.float32)
        positions = position_ids.to(device=device, dtype=torch.float32)
        batch = int(positions.shape[0])
        packed = slopes.view(1, num_heads, 1, 1) * positions.view(batch, 1, 1, -1)
        return packed.reshape(batch * num_heads, 1, int(positions.shape[-1])).to(dtype)

    def awq_module_replay_chunk_kwargs(self, module, module_kwargs, *, start, length, full_batch_size):
        alibi = module_kwargs.get("alibi")
        if not torch.is_tensor(alibi) or "alibi" not in inspect.signature(module.forward).parameters:
            return {}
        num_heads = alibi.shape[0] // full_batch_size
        return {"alibi": alibi[start * num_heads : (start + length) * num_heads]}


__all__ = ["BloomQModel"]
