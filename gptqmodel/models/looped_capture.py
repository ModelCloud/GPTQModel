# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Full-forward calibration capture for looped / recurrent decoders.

HRM-Text, Ouro and Nanbeige 4.2 run the same physical decoder layers several
times per forward. The default capture aborts the forward at the first layer,
so a layer's processor only ever saw loop 0's input; the Hessian was therefore
accumulated from one of N distinct input distributions (issue #4093 item 6,
same class as #4055 for HRM-Text).

This mixin materializes the complete model, runs one full forward per
calibration sample with a pre-hook on every physical layer, and replays every
invocation to the layer currently being quantized.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from ..looper.input_cache import InputCache
from ..utils.looper_helpers import normalize_device_like
from ..utils.model import get_module, move_to, nested_move_to
from ..utils.torch import CPU


class LoopedLayerCaptureMixin:
    """Serve every invocation of a layer that a looped decoder reuses."""

    _looped_per_layer: Optional[Dict[int, Dict[str, List[Any]]]] = None
    _looped_input_caches: Optional[List[InputCache]] = None

    def _materialize_looped_capture_model(self, layers, layer_names, device):
        """Materialize the complete recurrent model for one calibration pass.

        The real forward runs every layer on every loop, so all layers must be
        executable at once. The standard capture stage only materializes the
        first layer because it aborts the forward there; that is exactly why it
        cannot see the later invocations.
        """

        for module_name in self.get_base_modules(self.model):
            module = get_module(self.model, module_name)
            if module is not None:
                self.shell_module_materialize(
                    target_submodule=module,
                    device=device,
                    module_path=module_name,
                )

        for module_name in self.get_modules_with_direct_meta_tensors(self.model):
            module = get_module(self.model, module_name)
            if module is not None:
                self.shell_direct_meta_materialize(
                    target_submodule=module,
                    device=device,
                )

        for layer, layer_name in zip(layers, layer_names):
            self.shell_module_materialize(
                target_submodule=layer,
                device=device,
                module_path=layer_name,
            )

        # The capture forward runs to the causal head, not just layer zero.
        # Without this the head stays on CPU and the logits matmul fails on a
        # GPU capture with "mat2 is on cpu".
        head = get_module(self.model, self.lm_head)
        if head is not None:
            self.shell_module_materialize(
                target_submodule=head,
                device=device,
                module_path=self.lm_head,
            )

    def _capture_looped_layer_invocations(self, layers, layer_names, calibration_data, device):
        """Record the input of every layer invocation for every calibration sample.

        A recurrent layer is executed several times with different states; the
        flattened layer chain only sees the first invocation. Feeding every
        invocation back to the layer's processor accumulates the Hessian over
        the true input distribution instead of one loop.
        """

        del layer_names
        per_layer = {
            id(layer): {
                "layer_inputs": [],
                "layer_input_kwargs": [],
                "position_ids": [],
                "attention_masks": [],
            }
            for layer in layers
        }
        handles = []

        def make_hook(layer):
            def hook(module, args, kwargs):
                del module
                hidden_states = args[0] if args else kwargs.get("hidden_states")
                if not torch.is_tensor(hidden_states):
                    raise TypeError(
                        "Looped-decoder calibration hook expected a hidden-states tensor"
                    )

                bucket = per_layer[id(layer)]
                bucket["layer_inputs"].append(
                    [move_to(hidden_states.detach(), device=CPU)]
                )

                attention_mask = kwargs.get("attention_mask")
                bucket["attention_masks"].append(
                    None
                    if attention_mask is None
                    else move_to(attention_mask.detach(), device=CPU)
                )
                position_ids = kwargs.get("position_ids")
                bucket["position_ids"].append(
                    None
                    if position_ids is None
                    else move_to(position_ids.detach(), device=CPU)
                )

                replay_kwargs = {}
                for key, value in kwargs.items():
                    if key in ("hidden_states", "attention_mask", "position_ids"):
                        continue
                    if key in ("past_key_values", "past_key_value"):
                        continue
                    replay_kwargs[key] = nested_move_to(value, device=CPU)
                bucket["layer_input_kwargs"].append(replay_kwargs)

            return hook

        for layer in layers:
            handles.append(
                layer.register_forward_pre_hook(make_hook(layer), with_kwargs=True)
            )
        try:
            for example in calibration_data:
                example = self.move_input_capture_example(example, device)
                self.model(**example, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        return per_layer

    def build_quantization_input_cache(
        self,
        *,
        layers: Sequence[torch.nn.Module],
        calibration_data: Iterable[Dict[str, Any]],
        use_cache: bool,
        layer_names: Optional[List[str]] = None,
    ):
        """Serve every layer's true looped invocations instead of one chain pass."""

        if use_cache:
            raise ValueError("Looped-decoder calibration input capture does not use a cache")
        if not layers:
            return None
        if layer_names is None:
            layer_names = [None] * len(layers)

        device = normalize_device_like(self.quantize_config.device) or torch.device("cpu")
        self._materialize_looped_capture_model(layers, layer_names, device)
        per_layer = self._capture_looped_layer_invocations(
            layers, layer_names, calibration_data, device
        )
        for layer in layers:
            bucket = per_layer[id(layer)]
            if not bucket["layer_inputs"]:
                raise RuntimeError(
                    "Looped-decoder calibration captured no invocation for a decoder layer"
                )

        self._looped_per_layer = per_layer
        cache = InputCache([], [], [], [], src_inputs=[])
        # Seed the cache with the first layer's invocations so a caller that
        # reads it before `pre_quantize` still sees real inputs.
        first = next(iter(per_layer.values()))
        cache.layer_inputs = first["layer_inputs"]
        cache.layer_input_kwargs = first["layer_input_kwargs"]
        cache.position_ids = first["position_ids"]
        cache.attention_masks = first["attention_masks"]
        cache.src_inputs = []
        # The replay loop normally bounds itself by the calibration batch count.
        # A looped decoder has more invocations than batches, so publish the
        # largest bucket and let the looper size the replay from it.
        cache.captured_batch_count = max(
            len(bucket["layer_inputs"]) for bucket in per_layer.values()
        )
        self._looped_input_caches = getattr(self, "_looped_input_caches", []) or []
        self._looped_input_caches.append(cache)
        return cache

    def pre_quantize(self, module: torch.nn.Module) -> torch.nn.Module:
        """Point the shared cache at the invocations of the layer being quantized."""

        per_layer = getattr(self, "_looped_per_layer", None)
        bucket = per_layer.get(id(module)) if per_layer is not None else None
        if bucket is not None:
            for cache in getattr(self, "_looped_input_caches", ()) or ():
                cache.layer_inputs = bucket["layer_inputs"]
                cache.layer_input_kwargs = bucket["layer_input_kwargs"]
                cache.position_ids = bucket["position_ids"]
                cache.attention_masks = bucket["attention_masks"]
                cache.src_inputs = []
        return super().pre_quantize(module)

    def after_quantize(self) -> None:
        """Release the looped calibration captures after the loop completes."""

        self._looped_per_layer = None
        self._looped_input_caches = []
        return super().after_quantize()
