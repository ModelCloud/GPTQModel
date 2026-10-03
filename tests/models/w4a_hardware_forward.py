# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Encoded NVFP4 forward values with a differentiable weight-QAD surrogate.

The auxiliary runtime uses the current rounded GPTQ codes. Dense boundary
snapshots exist only during a training forward/backward frame; inference keeps
its ordinary encoded carrier contract and native checkpoint representation.
"""

from contextlib import contextmanager

import torch


class _ForwardValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, surrogate, actual):
        if surrogate.shape != actual.shape or surrogate.device != actual.device:
            raise ValueError("Hardware and surrogate values must have matching shapes/devices")
        ctx.input_dtype = surrogate.dtype
        return actual.detach()

    @staticmethod
    def backward(ctx, gradient):
        return gradient.to(ctx.input_dtype), None


def hardware_value(surrogate, actual):
    """Exact hardware forward; identity straight-through derivative."""
    return _ForwardValue.apply(surrogate, actual)


class HardwareForward:
    def __init__(self, student, runtime, weight_modules):
        from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation
        from gptqmodel.nn_modules.qlinear.w4a_boundary import NVFP4BoundaryQuantizer
        from gptqmodel.nn_modules.qlinear.w4a_llama_stream import _bind_forward
        from gptqmodel.nn_modules.qlinear.w4a_nvfp4 import W4ANVFP4Linear

        if (getattr(student, "_w4a_replay_version", None) != 4
                or getattr(runtime, "_w4a_stream_version", None) != 4
                or student._w4a_replay_recipe != runtime._w4a_stream_recipe
                or student._w4a_replay_global_scales != runtime._w4a_stream_global_scales):
            raise ValueError("Hardware forward requires matching version-4 NVFP4 policies")
        if len(student.model.layers) != len(runtime.model.layers):
            raise ValueError("Hardware and surrogate decoder layouts differ")
        if any(layer.self_attn.attention_dropout != 0 for layer in student.model.layers):
            raise ValueError("Hardware-forward adaptation currently requires zero attention dropout")
        self.student, self.runtime = student, runtime.eval()
        self.weight_modules = weight_modules
        self.runtime_linears = {name: module for name, module in runtime.named_modules()
                                if isinstance(module, W4ANVFP4Linear)}
        if len(self.runtime_linears) != 7 * len(runtime.model.layers):
            raise ValueError("Hardware forward requires all seven projections in every decoder")
        if not set(weight_modules) <= set(self.runtime_linears):
            raise ValueError("Trainable weights are missing from the hardware model")
        for name, module in weight_modules.items():
            if not torch.equal(module._w4a_qad_scales, self.runtime_linears[name].scales.float()):
                raise ValueError(f"Hardware GPTQ scales differ from training: {name}")
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            if not torch.equal(student.get_parameter(name), runtime.get_parameter(name)):
                raise ValueError(f"Hardware nonquantized parameter differs: {name}")
        if not torch.equal(student.model.rotary_emb.inv_freq, runtime.model.rotary_emb.inv_freq):
            raise ValueError("Hardware and surrogate RoPE frequencies differ")
        self.values, self.handles, self.originals = {}, [], []
        self.active = False
        self.capture_active = False
        self.logits = None

        def decoded(value):
            return value.decode(torch.float32) if isinstance(value, W4AActivation) else value.detach()

        try:
            for name, module in self.runtime_linears.items():
                def capture_input(_m, args, *, name=name):
                    if self.capture_active:
                        self.values[("input", name)] = decoded(args[0])
                def capture_output(_m, _args, value, *, name=name):
                    if self.capture_active:
                        self.values[("output", name)] = decoded(value)
                self.handles.extend([module.register_forward_pre_hook(capture_input),
                                     module.register_forward_hook(capture_output)])
                proxy = student.get_submodule(name)
                def use_input(module, args, *, name=name):
                    if not getattr(module, "_w4a_replay_disabled", False):
                        return (self.replace(args[0], ("input", name)), *args[1:])
                    return None
                def use_output(module, _args, value, *, name=name):
                    if not getattr(module, "_w4a_replay_disabled", False):
                        return self.replace(value, ("output", name))
                    return value
                self.handles.extend([proxy.register_forward_pre_hook(use_input),
                                     proxy.register_forward_hook(use_output)])
            self.producer_keys = []
            for module in runtime.modules():
                if isinstance(module, NVFP4BoundaryQuantizer):
                    self.producer_keys.append(module.key)
                    def capture_boundary(module, _args, value):
                        if self.capture_active:
                            self.values[("producer", module.key)] = decoded(value)
                    self.handles.append(module.register_forward_hook(capture_boundary))
            for index, layer in enumerate(student.model.layers):
                prefix = f"model.layers.{index}"
                original = getattr(layer, "_old_forward", layer.forward)
                self.originals.append((layer, original))
                def forward(layer, hidden_states, attention_mask=None, position_ids=None,
                            past_key_values=None, use_cache=False, position_embeddings=None,
                            *, prefix=prefix, original=original, **kwargs):
                    if getattr(layer, "_w4a_replay_disabled", False):
                        return original(hidden_states, attention_mask=attention_mask,
                                        position_ids=position_ids, past_key_values=past_key_values,
                                        use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
                    if use_cache or past_key_values is not None:
                        raise ValueError("Hardware training frames do not support KV caching")
                    def boundary(value, suffix):
                        return self.replace(value.float(), ("producer", f"{prefix}.{suffix}"))
                    x = boundary(hidden_states, "input") if layer._w4a_replay_round_input else hidden_states
                    residual = x
                    x = layer.input_layernorm(x)
                    x, _ = layer.self_attn(hidden_states=x, attention_mask=attention_mask,
                                          position_ids=position_ids, past_key_values=None,
                                          use_cache=False, position_embeddings=position_embeddings, **kwargs)
                    x = boundary(residual.float() + x.float(), "post_attention_residual")
                    residual = x
                    x = layer.post_attention_layernorm(x)
                    x = layer.mlp(x)
                    return boundary(residual.float() + x.float(), "output")
                _bind_forward(layer, forward)
        except BaseException:
            self.close()
            raise

    def replace(self, value, key):
        if not self.active or key not in self.values:
            raise RuntimeError("Hardware values require a live forward/backward frame")
        return hardware_value(value, self.values[key])

    @torch.no_grad()
    def sync_weights(self):
        """Refresh hardware planes only for modules whose rounded codes changed."""
        from .w4a_nvfp4_weight_qad import _latent_codes, _pack_centered_int4

        if self.active:
            raise RuntimeError("Cannot change hardware weights during a backward frame")
        changed = 0
        for name, module in self.weight_modules.items():
            centered = _latent_codes(module).round().clamp(-8, 7)
            centered = centered.reshape(module.out_features, module.in_features).to(torch.int8).T.contiguous()
            packed = _pack_centered_int4(centered)
            target = self.runtime_linears[name]
            if not torch.equal(target.qweight, packed):
                target.qweight.copy_(packed)
                target.post_init()
                changed += 1
        return changed

    @contextmanager
    def frame(self, input_ids, *, logits_to_keep=0):
        from gptqmodel.nn_modules.qlinear import w4a_llama_replay as replay

        from .w4a_nvfp4_weight_qad import straight_through_activation_round

        if self.active:
            raise RuntimeError("Hardware training frames cannot overlap")
        self.active = True
        original_round = replay._round
        try:
            replay._round = straight_through_activation_round
            self.capture_active = True
            with torch.no_grad():
                result = self.runtime(input_ids=input_ids, use_cache=False, logits_to_keep=logits_to_keep)
            self.capture_active = False
            self.logits = result.logits.detach()
            expected = 2 * len(self.runtime_linears) + len(self.producer_keys)
            if len(self.values) != expected:
                raise AssertionError("Incomplete encoded hardware boundary capture")
            yield self
        finally:
            replay._round = original_round
            self.capture_active = False
            self.active = False
            self.values.clear()
            self.logits = None

    def close(self):
        from gptqmodel.nn_modules.qlinear.w4a_llama_stream import _bind_forward

        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for layer, original in self.originals:
            _bind_forward(layer, original.__func__)
        self.originals.clear()
