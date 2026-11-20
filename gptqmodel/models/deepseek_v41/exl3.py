# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Packed Q4NX projections for DeepSeek V4.1 Flash."""

import re
import weakref

import torch

from ...exllamav3.ext import _runtime_op
from ...exllamav3.modules.quant.exl3_lib.quantize import validate_exl3_quantization_geometry
from ...exllamav3.trellis import validate_trellis_k
from ...nn_modules.exllamav3 import ExllamaV3Linear
from ...nn_modules.exllamav3_bf16_kernels import FP16_SAFE_AMAX, try_prescale
from ...nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from ...utils.backend import BACKEND


def is_routed_projection(path):
    return re.fullmatch(r"layers\.\d+\.ffn\.experts\.\d+\.w[123]", path) is not None


def make_q4nx_projection(source, path, spec, backend, kernel=None):
    if not isinstance(spec, dict) or set(spec) != {"bits", "codebook"} or spec["codebook"] != "q4nx":
        raise ValueError(f"Invalid DeepSeek V4.1 Q4NX projection specification for {path}")
    if not is_routed_projection(path) or not getattr(source, "fp4", False):
        raise ValueError(f"DeepSeek V4.1 Q4NX checkpoint cannot replace {path}")
    k, n = source.in_features, source.out_features
    stored_n = validate_exl3_quantization_geometry(k, n, label=path)
    bits = spec["bits"]
    validate_trellis_k(bits, False, False, q4nx=True)
    tensors = {
        "trellis": torch.empty((k // 16, stored_n // 16, int(bits * 16)), dtype=torch.int16),
        "suh": torch.empty(k, dtype=torch.float16),
        "svh": torch.empty(stored_n, dtype=torch.float16),
        "q4nx": torch.empty((), dtype=torch.int32),
    }
    if backend in (BACKEND.TORCH, BACKEND.EXL3_TORCH):
        return ExllamaV3TorchLinear.from_tensors(in_features=k, out_features=n, name=path, tensors=tensors)
    return V41Q4NXLinear.from_tensors(in_features=k, out_features=n, name=path, tensors=tensors, kernel=kernel)


class V41Q4NXLinear(ExllamaV3Linear):
    _route_owner = None

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ("trellis", "suh", "svh", "q4nx", "bias"):
            self._clear_routing_tables()

    def _clear_routing_tables(self):
        owner = self._route_owner() if self._route_owner is not None else None
        if owner is not None:
            owner._q4nx_tables = None

    def clear_weight_cache(self):
        super().clear_weight_cache()
        self._clear_routing_tables()

    def _apply(self, fn, recurse=True):
        self._clear_routing_tables()
        return super()._apply(fn, recurse=recurse)

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_route_owner", None)
        return state

    def post_init(self):
        # A fused layer can consume all packed buffers directly. The ordinary
        # per-projection handle is only built if the projection is called.
        self.clear_weight_cache()
        self._coerce_buffer_dtypes()

    def _needs_torch_fallback(self):
        return (super()._needs_torch_fallback() or torch.version.hip is not None
                or (self.trellis.is_cuda and torch.cuda.get_device_capability(self.trellis.device) < (8, 0)))


def bind_q4nx_routing(model, paths):
    for name in {path.partition(".experts.")[0] for path in paths}:
        layer = model.get_submodule(name)
        layer._q4nx_routing = True
        layer._q4nx_tables = None


def _routing_tables(layer, device):
    banks = [tuple(getattr(expert, name) for expert in layer.experts) for name in ("w1", "w3", "w2")]
    widths = []
    for bank in banks:
        first = bank[0]
        if not isinstance(first, V41Q4NXLinear):
            return False
        shape = tuple(first.trellis.shape)
        if len(shape) != 3 or shape[-1] not in (40, 48, 56):
            return False
        k, n = shape[0] * 16, shape[1] * 16
        if any(
            not isinstance(linear, V41Q4NXLinear)
            or linear.q4nx is None or linear.bias is not None
            or linear.in_features != k or linear.out_features != n
            or linear.trellis.dtype != torch.int16
            or tuple(linear.trellis.shape) != shape
            or any(t is None or t.device != device or not t.is_contiguous()
                   for t in (linear.trellis, linear.suh, linear.svh))
            or any(t.dtype != torch.float16 or t.shape != (width,)
                   for t, width in ((linear.suh, k), (linear.svh, n)))
            or any(t.data_ptr() % 16 for t in (linear.trellis, linear.suh, linear.svh))
            for linear in bank
        ):
            return False
        widths.append(shape[-1] / 16)
    if widths[0] != widths[1]:
        return False
    owner = weakref.ref(layer)
    for expert in layer.experts:
        expert._route_owner = owner
    for bank in banks:
        for linear in bank:
            linear._route_owner = owner
    # Allocate the immutable pointer tables once. The projections remain the
    # owners of packed weights and sign vectors. Device moves and reassignment
    # invalidate the tables before the next launch.
    pointers = torch.tensor(
        [[getattr(linear, leaf).data_ptr() for linear in bank]
         for bank in banks for leaf in ("trellis", "suh", "svh")],
        dtype=torch.int64, device=device,
    )
    return tuple(pointers.unbind(0)), widths, banks[0][0].out_features, tuple(layer.experts)


def routed_q4nx(layer, x, indices, weights):
    if (x.dtype != torch.bfloat16 or x.numel() == 0 or not x.is_cuda
            or torch.version.hip is not None or x.requires_grad or weights.requires_grad):
        return None
    limit = layer.experts[0].limit
    topk = indices.size(1)
    # A normalized route weight is at most abs(scale). With the Flash clamp
    # this bounds the down input and its 128-wide rotation inside FP16.
    if (not layer.gate.normalize or topk <= 1 or topk > 16 or limit <= 0
            or not (limit * limit * abs(layer.gate.scale) <= FP16_SAFE_AMAX / 2)):
        return None
    tables = layer._q4nx_tables
    if tables and tables[-1] != tuple(layer.experts):
        tables = None
    if tables is None:
        if torch.cuda.get_device_capability(x.device) < (8, 0) or torch.cuda.is_current_stream_capturing():
            return None
        tables = _routing_tables(layer, x.device)
        layer._q4nx_tables = tables
    if tables is False:
        return None

    pointers, widths, intermediate, _ = tables
    if pointers[0].device != x.device:
        return None
    op = _runtime_op("exl3_moe_coop")
    prepared = try_prescale(x.contiguous())
    if prepared is None:
        return None
    native_input, scales = prepared
    # Keep the expert-ID accumulation order of V41MoE's reference loop.
    ordered, order = indices.sort(dim=1)
    weights = weights.gather(1, order)
    width = x.size(1)
    outputs = []
    for rows, ids, rw, scale in zip(
        native_input.split(256 // topk), ordered.split(256 // topk),
        weights.split(256 // topk), scales.split(256 // topk), strict=True,
    ):
        capacity = min(256, max(32, 2 * ids.numel()))
        had_g, had_u = rows.new_empty((2, capacity, width)).unbind(0)
        gu_g, gu_u = rows.new_empty((2, capacity, intermediate), dtype=torch.float32).unbind(0)
        act = rows.new_empty((capacity, intermediate))
        down = rows.new_empty((capacity, width), dtype=torch.float32)
        out = rows.new_empty(rows.shape, dtype=torch.float32)
        count = capacity * (intermediate // 128 + 2) + rows.size(0) * (width // 128) + 3
        counters = rows.new_zeros((count,), dtype=torch.int32)
        op(
            rows, ids, rw, -1, -1, width, *pointers, None, None, None, *widths,
            False, False, 0, limit, True, had_g, had_u, gu_g, gu_u, act,
            down, counters, out, q4nx=True, flash_input_scale=scale,
        )
        outputs.append(out)
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)
