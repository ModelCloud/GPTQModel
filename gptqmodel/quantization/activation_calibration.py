# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Activation-only calibration of an already loaded native INT4 Llama stream."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from ..nn_modules.qlinear.w4a_activation import W4AActivation
from ..nn_modules.qlinear.w4a_boundary import NVFP4BoundaryQuantizer, nvfp4_producer_specs
from .activation_floatx import nvfp4_global_scale


class _CaptureComplete(Exception):
    pass


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device)
    if isinstance(value, W4AActivation):
        return replace(value, codes=_move(value.codes, device), scales=_move(value.scales, device),
                       global_scale=_move(value.global_scale, device), token_scale=_move(value.token_scale, device),
                       reference=_move(value.reference, device))
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    raise TypeError(f"Unsupported calibration cache value: {type(value).__name__}")


@dataclass
class _LayerSample:
    hidden: torch.Tensor | W4AActivation
    kwargs: dict


class BoundaryMaximum:
    def __init__(self):
        self.amax = 0.0
        self.tokens = 0
        self.calls = 0

    def observe(self, value):
        if value.ndim < 2 or not value.numel() or not bool(torch.isfinite(value).all()):
            raise ValueError("Activation calibration requires nonempty finite token tensors")
        self.amax = max(self.amax, float(value.detach().abs().amax()))
        self.tokens += value.numel() // value.shape[-1]
        self.calls += 1

    def scale(self, recipe):
        if not self.calls:
            raise ValueError("Cannot calibrate an unobserved producer")
        # Persistent FP32 global scale; no rounding onto a BF16 input grid.
        return float(nvfp4_global_scale(self.amax, recipe=recipe))


def _capture_inputs(core, samples):
    cached = []
    def capture(_module, args, kwargs):
        kwargs = dict(kwargs)
        hidden = args[0] if args else kwargs.pop("hidden_states")
        if len(args) > 1:
            raise ValueError("Unexpected positional decoder arguments during calibration")
        if kwargs.get("past_key_values") is not None:
            raise ValueError("Calibration must run without KV cache")
        cached.append(_LayerSample(_move(hidden, "cpu"), _move(kwargs, "cpu")))
        raise _CaptureComplete
    handle = core.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        device = core.model.embed_tokens.weight.device
        for ids in samples:
            if ids.ndim != 1 or not ids.numel():
                raise ValueError("Calibration samples must be unpadded nonempty token vectors")
            try:
                core.model(input_ids=ids[None].to(device), use_cache=False)
            except _CaptureComplete:
                pass
            else:
                raise RuntimeError("The first decoder input was not captured")
    finally:
        handle.remove()
    return cached


@torch.inference_mode()
def calibrate_nvfp4_producers(core, samples, *, quantize_config=None, progress=None) -> dict:
    """Freeze FP32 global scales in topological order, keeping every weight fixed.

    Uses maximum calibration with the selected runtime block-scale recipe.
    Only one layer's input/output samples are cached in CPU memory at a time.
    The caller must provide data disjoint from its downstream evaluation sets.
    When supplied, quantize_config is updated for ordinary checkpoint saving.
    On failure all producer scales and policy metadata are restored. Callers
    without a config must export the returned scale map themselves.
    """
    if getattr(core, "_w4a_stream_mode", None) != "w4a_nvfp4":
        raise ValueError("Producer calibration requires an installed NVFP4 stream")
    if core.training:
        raise ValueError("Producer calibration requires evaluation mode")
    layers = core.model.layers
    if not samples:
        raise ValueError("Calibration requires samples")
    recipe = core._w4a_stream_recipe
    if recipe not in {"nvidia", "four_six", "least_squares", "least_squares_grid"}:
        raise ValueError("Producer calibration cannot use legacy headroom recipes")
    if quantize_config is not None and (
            quantize_config.activation_mode != "w4a_nvfp4"
            or quantize_config.activation_recipe != recipe):
        raise ValueError("The save configuration must match the installed producer policy")
    # Enumerate only the boundaries the installed policy actually stages as
    # NVFP4. A mixed stream keeps FP8 on attention or on promoted MLP layers;
    # those boundaries own no calibrated scale and no quantizer module.
    stream_mode = getattr(core, "_w4a_stream_mode", None)
    specs = nvfp4_producer_specs(
        layers, [True] * len(layers),
        attention_mode=getattr(core, "_w4a_stream_attention_mode", None) or stream_mode,
        mode=stream_mode,
        recipe=getattr(core, "_w4a_stream_recipe", None),
        mlp_fp8_layers=tuple(getattr(core, "_w4a_stream_mlp_fp8_layers", None) or ()),
    )
    producers = [(key, getattr(owner, f"_w4a_{name}_quantizer")) for owner, name, key in specs]
    if any(not isinstance(module, NVFP4BoundaryQuantizer) or module.observer is not None
           for _, module in producers):
        raise ValueError("Every producer must have an idle NVFP4 boundary quantizer")
    initial = [(module, module.scale_bits.clone(), module.calibrated) for _, module in producers]
    old_policy = core._w4a_stream_global_scales
    old_activation = quantize_config.activation if quantize_config is not None else None
    had_hf_policy = hasattr(core.config, "quantization_config")
    old_hf_policy = getattr(core.config, "quantization_config", None)
    report, scales = {}, {}
    try:
        cached = _capture_inputs(core, samples)
        for index, layer in enumerate(layers):
            device = layer.self_attn.q_proj.qweight.device
            prefix = f"model.layers.{index}."
            for key, producer in producers:
                if not key.startswith(prefix):
                    continue
                maximum = BoundaryMaximum()
                def observe(value):
                    maximum.observe(value)
                    raise _CaptureComplete
                producer.observer = observe
                try:
                    for sample in cached:
                        try:
                            layer(_move(sample.hidden, device), **_move(sample.kwargs, device))
                        except _CaptureComplete:
                            pass
                        else:
                            raise RuntimeError(f"Producer not reached: {key}")
                finally:
                    producer.observer = None
                value = maximum.scale(recipe)
                producer.set_scale(value)
                scales[key] = value
                report[key] = {"amax": maximum.amax, "global_scale": value,
                               "tokens": maximum.tokens, "samples": maximum.calls}
                if progress is not None:
                    progress(key, report[key])
            # Actual calibrated carriers become the next layer's inputs.
            for sample in cached:
                result = layer(_move(sample.hidden, device), **_move(sample.kwargs, device))
                if not isinstance(result, W4AActivation):
                    raise TypeError("Calibrated decoder did not return an encoded activation")
                sample.hidden = _move(result, "cpu")
        core._w4a_stream_global_scales = dict(scales)
        if quantize_config is not None:
            payload = quantize_config.to_dict()
            payload["activation"] = {**quantize_config.activation, "global_scales": dict(scales)}
            validated = type(quantize_config).from_quant_config(payload)
            quantize_config.activation = validated.activation
            core.config.quantization_config = quantize_config.to_dict()
    except BaseException:
        for module, bits, calibrated in initial:
            module.observer = None
            module.scale_bits.copy_(bits)
            module.calibrated = calibrated
        core._w4a_stream_global_scales = old_policy
        if quantize_config is not None:
            quantize_config.activation = old_activation
            if had_hf_policy:
                core.config.quantization_config = old_hf_policy
            elif hasattr(core.config, "quantization_config"):
                del core.config.quantization_config
        raise
    return {"algorithm": "producer_maximum", "recipe": recipe,
            "global_scales": scales, "boundaries": report,
            "upstream_activations": "calibrated_encoded_carriers", "weight_updates": 0}


@torch.inference_mode()
def measure_nvfp4_producers(core, samples) -> dict:
    """Measure held-out local reconstruction error without modifying scales."""
    statistics = {}
    handles = []
    def measure(module, args, encoded):
        reference = args[0].float()
        decoded = encoded.decode(torch.float32)
        if not bool(torch.isfinite(decoded).all()):
            raise ValueError(f"Nonfinite calibrated carrier: {module.key}")
        row = statistics.setdefault(module.key, {"sse": 0.0, "energy": 0.0, "elements": 0,
                                                 "amax": 0.0, "calls": 0})
        row["sse"] += float((reference - decoded).square().sum())
        row["energy"] += float(reference.square().sum())
        row["elements"] += reference.numel()
        row["amax"] = max(row["amax"], float(reference.abs().amax()))
        row["calls"] += 1
    try:
        for module in core.modules():
            if isinstance(module, NVFP4BoundaryQuantizer):
                handles.append(module.register_forward_hook(measure))
        device = core.model.embed_tokens.weight.device
        for ids in samples:
            core.model(input_ids=ids[None].to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    for row in statistics.values():
        row["relative_rmse"] = (row["sse"] / max(row["energy"], 1e-30)) ** 0.5
    return statistics
