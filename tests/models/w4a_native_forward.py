# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native W4A16 values for the differentiable preservation objective."""

from contextlib import contextmanager

import torch

from .w4a_hardware_forward import hardware_value


class NativeForward:
    """Match TorchLinear inference, including BF16 weight rounding and rotation.

    The student supplies derivatives; the auxiliary native model supplies
    forward values. Only the disabled A4 replay lane uses these substitutions.
    """

    def __init__(self, student, runtime, weight_modules):
        from gptqmodel.nn_modules.qlinear.torch import TorchLinear

        self.student, self.runtime = student, runtime.eval()
        self.weight_modules = weight_modules
        self.linears = {name: module for name, module in runtime.named_modules()
                        if isinstance(module, TorchLinear)}
        if (len(student.model.layers) != len(runtime.model.layers)
                or len(self.linears) != 7 * len(runtime.model.layers)
                or not set(weight_modules) <= set(self.linears)):
            raise ValueError("Native preservation requires matching complete GPTQ decoders")
        for name, module in weight_modules.items():
            if not torch.equal(module._w4a_qad_scales, self.linears[name].scales.float()):
                raise ValueError(f"Native preservation GPTQ scales differ: {name}")
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            if not torch.equal(student.get_parameter(name), runtime.get_parameter(name)):
                raise ValueError(f"Native preservation nonquantized parameter differs: {name}")
        if not torch.equal(student.model.rotary_emb.inv_freq, runtime.model.rotary_emb.inv_freq):
            raise ValueError("Native preservation RoPE frequencies differ")
        self.values, self.handles = {}, []
        self.active = self.capture_active = False
        self.logits = None
        try:
            for name, module in self.linears.items():
                def capture_input(module, args, *, name=name):
                    if self.capture_active:
                        x = args[0]
                        # TorchLinear rotates inside forward; the student's
                        # MLP replay rotates before calling its down Linear.
                        x = module._apply_rotation_to_input(x.reshape(-1, module.in_features)).reshape_as(x)
                        self.values[("input", name)] = x.detach()

                def capture_output(_module, _args, output, *, name=name):
                    if self.capture_active:
                        self.values[("output", name)] = output.detach()

                def use_input(module, args, *, name=name):
                    if getattr(module, "_w4a_replay_disabled", False):
                        return (self.replace(args[0], ("input", name)), *args[1:])

                def use_output(module, _args, output, *, name=name):
                    return (self.replace(output, ("output", name))
                            if getattr(module, "_w4a_replay_disabled", False) else output)

                proxy = student.get_submodule(name)
                self.handles.extend([module.register_forward_pre_hook(capture_input),
                                     module.register_forward_hook(capture_output),
                                     proxy.register_forward_pre_hook(use_input),
                                     proxy.register_forward_hook(use_output)])
        except BaseException:
            self.close()
            raise

    def replace(self, value, key):
        if not self.active or key not in self.values:
            raise RuntimeError("Native values require a live forward/backward frame")
        return hardware_value(value, self.values[key])

    @torch.no_grad()
    def sync_weights(self):
        from .w4a_nvfp4_weight_qad import _latent_codes, _pack_centered_int4

        if self.active:
            raise RuntimeError("Cannot change native weights during a backward frame")
        changed = 0
        for name, module in self.weight_modules.items():
            codes = _latent_codes(module).round().clamp(-8, 7)
            codes = codes.reshape(module.out_features, module.in_features).to(torch.int8).T.contiguous()
            packed = _pack_centered_int4(codes)
            target = self.linears[name]
            if not torch.equal(target.qweight, packed):
                target.qweight.copy_(packed)
                target.clear_weight_cache()
                target._stream_reset_cache()
                target._reset_prefetch_state()
                changed += 1
        return changed

    @contextmanager
    def frame(self, input_ids, *, logits_to_keep=0):
        if self.active:
            raise RuntimeError("Native preservation frames cannot overlap")
        if not getattr(self.student, "_w4a_replay_disabled", False):
            raise RuntimeError("Native preservation requires disabled activation replay")
        self.active = True
        try:
            self.capture_active = True
            with torch.no_grad():
                self.logits = self.runtime(input_ids=input_ids, use_cache=False,
                                           logits_to_keep=logits_to_keep).logits.detach()
            self.capture_active = False
            if len(self.values) != 2 * len(self.linears):
                raise AssertionError("Incomplete native preservation capture")
            yield self
        finally:
            self.capture_active = self.active = False
            self.values.clear()
            self.logits = None

    def close(self):
        if self.active:
            raise RuntimeError("Cannot close an active native preservation frame")
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
