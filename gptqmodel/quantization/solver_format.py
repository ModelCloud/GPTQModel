# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU solver/format lifecycle with stage-wise Hessians and packed replay."""

import time
from collections import OrderedDict
from collections.abc import Mapping

import torch

from ..nn_modules.qlinear.mxfp4 import MXFP4Linear
from .format_reference import decode_elements, encode_elements, fit_grid


@torch.no_grad()
def solve_gptq_hessian(weight, hessian, solver):
    if weight.device.type != "cpu" or hessian.device.type != "cpu":
        raise ValueError("The format solver requires CPU tensors")
    if weight.ndim != 2 or hessian.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("Hessian geometry does not match the weight")
    if not torch.isfinite(hessian).all():
        raise ValueError("Calibration Hessian is nonfinite")
    grid = fit_grid(weight, "mxfp4")
    hessian = hessian.double().clone()
    diagonal = hessian.diagonal()
    if not diagonal.any():
        return grid.pack(grid.encode(weight))
    order = diagonal.argsort(descending=True, stable=True) if solver.desc_act else torch.arange(weight.shape[1])
    ridge = solver.damp_percent * max(float(diagonal.mean()), torch.finfo(torch.float64).tiny)
    diagonal.add_(ridge)
    hessian = hessian[order][:, order]
    inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
    upper = torch.linalg.cholesky(inverse, upper=True)
    del inverse, hessian
    working = weight.detach().double()[:, order].clone()
    scales = grid.expanded()[:, order]
    codes = torch.empty_like(working, dtype=torch.uint8)
    for start in range(0, weight.shape[1], solver.block_size):
        end = min(start + solver.block_size, weight.shape[1])
        block = working[:, start:end].clone()
        errors = torch.empty_like(block)
        for column in range(end - start):
            index = start + column
            code = encode_elements(block[:, column] / scales[:, index], "e2m1")
            quantized = decode_elements(code, "e2m1") * scales[:, index]
            codes[:, order[index]] = code
            error = (block[:, column] - quantized) / upper[index, index]
            errors[:, column] = error
            block[:, column:] -= error[:, None] * upper[index, index:end]
        working[:, end:] -= errors @ upper[start:end, end:]
    return grid.pack(codes)


class Hessian:
    def __init__(self, columns):
        self.sum = torch.zeros(columns, columns, dtype=torch.float64)
        self.tokens = 0
        self.mask = None

    def capture(self, module, args):
        value = args[0].detach().reshape(-1, self.sum.shape[0]).double()
        if self.mask is not None:
            mask = self.mask.reshape(-1).bool()
            if mask.numel() != value.shape[0]:
                raise ValueError("Calibration token mask does not match module inputs")
            value = value[mask]
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite calibration activations")
        self.sum.addmm_(value.T, value)
        self.tokens += value.shape[0]


def _calibration_batches(calibration, tokenizer, batch_size):
    if type(batch_size) is not int or batch_size != 1:
        raise ValueError("CPU solver/format calibration currently requires batch_size=1")
    if calibration is None:
        raise ValueError("GPTQ requires calibration")
    batches = []
    for example in calibration:
        if isinstance(example, str):
            if tokenizer is None:
                raise ValueError("Text calibration requires a tokenizer")
            example = tokenizer(example, return_tensors="pt")
        elif isinstance(example, (list, tuple, torch.Tensor)):
            example = {"input_ids": example}
        if not isinstance(example, Mapping) or "input_ids" not in example:
            raise ValueError("Calibration needs text or input_ids mappings/rows")
        if set(example) - {"input_ids", "attention_mask"}:
            raise ValueError("CPU calibration supports input_ids and attention_mask only")
        ids = torch.as_tensor(example["input_ids"])
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.dtype not in (torch.int32, torch.int64) or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2:
            raise ValueError("Calibration must contain integer token rows of length >= 2")
        mask = torch.as_tensor(example.get("attention_mask", torch.ones_like(ids)))
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.shape != ids.shape or not ((mask == 0) | (mask == 1)).all() or not mask.any():
            raise ValueError("Calibration mask must contain valid nonempty token selections")
        batches.append({"input_ids": ids.cpu().long(), "attention_mask": mask.cpu().long()})
    if not batches:
        raise ValueError("Calibration cannot be empty")
    return batches


class _CapturedInputs(Exception):
    pass


def _clone_inputs(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_inputs(item) for item in value)
    if isinstance(value, list):
        return [_clone_inputs(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_inputs(item) for key, item in value.items()}
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    raise ValueError(f"Unsupported stateful replay input: {type(value).__name__}")


def _next_inputs(args, kwargs, output):
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor):
        raise ValueError("Decoder replay must return hidden states")
    if args:
        return (hidden, *args[1:]), kwargs
    return args, {**kwargs, "hidden_states": hidden}


@torch.no_grad()
def quantize_model(wrapper, calibration, plan, *, tokenizer=None, batch_size=1):
    batches = _calibration_batches(calibration, tokenizer or wrapper.tokenizer, batch_size)
    layers = wrapper.model.get_submodule(plan.layer_prefix)
    captured, solutions, records = [], {}, []
    training = wrapper.model.training
    wrapper.model.eval()
    started = time.perf_counter()

    def capture(module, args, kwargs):
        captured.append((_clone_inputs(args), _clone_inputs(kwargs)))
        raise _CapturedInputs()

    hook = layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for batch in batches:
            try:
                wrapper.model(**batch, use_cache=False)
            except _CapturedInputs:
                pass
        if len(captured) != len(batches):
            raise ValueError("Calibration did not reach the declared first decoder layer")
    finally:
        hook.remove()
        wrapper.model.train(training)
    wrapper.model.eval()
    try:
        for index, layer in enumerate(layers):
            groups = OrderedDict()
            for entry in plan.assignments:
                if entry.layer == index:
                    groups.setdefault(entry.group, []).append(entry)
            for entries in groups.values():
                accumulators, hooks = {}, []
                try:
                    for entry in entries:
                        module = wrapper.model.get_submodule(entry.name)
                        accumulator = Hessian(module.in_features)
                        accumulators[entry.name] = accumulator
                        hooks.append(module.register_forward_pre_hook(accumulator.capture))
                    for (args, kwargs), batch in zip(captured, batches):
                        for accumulator in accumulators.values():
                            accumulator.mask = batch["attention_mask"]
                        layer(*args, **kwargs)
                finally:
                    for handle in hooks:
                        handle.remove()
                for entry in entries:
                    accumulator = accumulators.pop(entry.name)
                    if not accumulator.tokens:
                        raise ValueError(f"No calibration activations for {entry.name}")
                    module = wrapper.model.get_submodule(entry.name)
                    solution = solve_gptq_hessian(module.weight, accumulator.sum / accumulator.tokens, entry.config.solver)
                    replacement = MXFP4Linear.from_solution(solution, module.bias)
                    wrapper.model.set_submodule(entry.name, replacement)
                    wrapper._solver_format_partial = True
                    solutions[entry.name] = {**entry.config.to_dict(), "in_features": module.in_features,
                                            "out_features": module.out_features, "bias": module.bias is not None}
                    records.append({"module": entry.name, "tokens": accumulator.tokens,
                                    "storage_bits": solution.storage_bits})
                    del module, solution, accumulator
            captured = [_next_inputs(args, kwargs, layer(*args, **kwargs)) for args, kwargs in captured]
            print(f"GPTQ -> MXFP4: layer {index + 1}/{len(layers)}; elapsed {time.perf_counter() - started:.1f}s", flush=True)
    finally:
        wrapper.model.train(training)
    wrapper._format_solutions = solutions
    wrapper._solver_format_partial = False
    wrapper.quantized = True
    wrapper.quantize_config = None
    return records
