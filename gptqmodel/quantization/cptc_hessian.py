# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Two-sided partial-trace Hessian sketches for CPTC calibration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.functional import cross_entropy

from ..utils.logger import setup_logger
from ..utils.torch import as_torch_device, tf32_high_precision_guard


log = setup_logger()


@dataclass
class HessianSketch:
    input: torch.Tensor
    output: torch.Tensor

    @classmethod
    def empty(cls, k, n, device):
        return cls(torch.zeros((k, k), device=device, dtype=torch.float32),
                   torch.zeros((n, n), device=device, dtype=torch.float32))

    @torch.no_grad()
    def add(self, inputs, gradients):
        # A sequence's weight gradient is G = gradients.T @ inputs.
        # Sketch B accumulates G.T @ G and G @ G.T. Contract the shorter
        # token axis first so the full per-example G need not be materialized.
        for x, dy in zip(inputs, gradients, strict=True):
            x, dy = x.float(), dy.float()
            if x.shape[0] < min(x.shape[1], dy.shape[1]):
                self.input.addmm_(x.T, (dy @ dy.T) @ x)
                self.output.addmm_(dy.T, (x @ x.T) @ dy)
            else:
                grad_weight = dy.T @ x
                self.input.addmm_(grad_weight.T, grad_weight)
                self.output.addmm_(grad_weight, grad_weight.T)


def _capture_projection(sketch, statistics_mask):
    def capture(module, args, output):
        inputs = args[0].detach()
        mask = statistics_mask()

        def accumulate(gradient):
            x = inputs.reshape(1, -1, inputs.shape[-1])
            dy = gradient.reshape(1, -1, gradient.shape[-1])
            if mask is not None:
                x, dy = x[:, mask], dy[:, mask]
            sketch.add(x, dy)

        output.register_hook(accumulate)

    return capture


@torch.inference_mode(False)
def collect_hessian_sketches(model, projections, batches, *, device, seed, exclude_token_id=None):
    """Estimate partial traces of the model Fisher on a dense teacher."""
    parameters = list(model.parameters())
    requires_grad = [p.requires_grad for p in parameters]
    handles = []
    sketches = {}
    rng = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    samples = 0
    keep_statistics = None
    try:
        model.eval()
        model.requires_grad_(False)
        for name, module in projections.items():
            sketch = HessianSketch.empty(module.in_features, module.out_features, device)
            sketches[name] = sketch
            handles.append(module.register_forward_hook(_capture_projection(sketch, lambda: keep_statistics)))
        handles.append(model.get_input_embeddings().register_forward_hook(
            lambda _module, _args, output: output.requires_grad_(True)
        ))
        with torch.inference_mode(False), torch.enable_grad(), tf32_high_precision_guard():
            for batch in batches:
                ids = batch["input_ids"].to(device)
                mask = batch.get("attention_mask")
                mask = mask.to(device).bool() if mask is not None else torch.ones_like(ids, dtype=torch.bool)
                for tokens, keep in zip(ids.split(1), mask.split(1), strict=True):
                    # Preparer batches can include padding. Strip it per
                    # sequence so padding neither enters the model nor G.
                    tokens = tokens[keep][None, :]
                    if tokens.numel() == 0:
                        continue
                    if exclude_token_id is not None:
                        keep_statistics = tokens.flatten() != exclude_token_id
                    logits = model(input_ids=tokens, use_cache=False).logits.flatten(0, 1)
                    with torch.no_grad():
                        target = torch.cat([
                            torch.multinomial(part.float().softmax(-1), 1, generator=rng).flatten()
                            for part in logits.split(128)
                        ])
                    cross_entropy(logits.float(), target, reduction="sum").backward()
                    del logits
                    samples += 1
                    if samples % 16 == 0:
                        log.info(f"CPTC: collected two-sided Hessian sketches for {samples} sequences")
        if samples == 0:
            raise ValueError("CPTC Hessian calibration requires at least one nonempty sequence")
        return sketches
    finally:
        for handle in handles:
            handle.remove()
        for parameter, old in zip(parameters, requires_grad, strict=True):
            parameter.requires_grad_(old)
        model.train(was_training)


@torch.inference_mode(False)
def collect_model_hessian_sketches(qmodel, batches, *, exclude_attention_sinks=False):
    from ..utils.model import get_layers_with_prefixes
    from ..utils.offload import materialize_remaining_meta_modules
    from .cptc_yaqa import validate_tile_shape

    model, config = qmodel.model, qmodel.quantize_config
    subsets = qmodel.simple_layer_modules(model.config, config)
    layers, paths = get_layers_with_prefixes(model, qmodel.extract_layers_node())
    projections = {}
    for path, layer in zip(paths, layers, strict=True):
        for subset in subsets:
            for relative in subset:
                name = f"{path}.{relative}"
                if config.dynamic_get(layer_name=name) is False:
                    continue
                projection = layer.get_submodule(relative)
                if not qmodel.should_quantize_module(model, name, projection, config):
                    continue
                projections[name] = projection
    if config.lm_head and config.dynamic_get(layer_name=qmodel.lm_head) is not False:
        projections[qmodel.lm_head] = model.get_submodule(qmodel.lm_head)
    for name, projection in projections.items():
        if not isinstance(projection, torch.nn.Linear):
            raise ValueError(f"CPTC YAQA requires a linear projection at {name}")
        validate_tile_shape(projection.in_features, projection.out_features, config.block_size)

    device = as_torch_device(config.device)
    materialize_remaining_meta_modules(qmodel)
    original_device = model.device
    try:
        model.to(device)
        return collect_hessian_sketches(
            model, projections, batches, device=device, seed=config.yaqa_seed,
            exclude_token_id=qmodel.tokenizer.bos_token_id if exclude_attention_sinks else None,
        )
    finally:
        model.to(original_device)
