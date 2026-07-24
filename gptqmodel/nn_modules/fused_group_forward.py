# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fuse same-input linear forward passes during GPTQ calibration.

Transformer layers often feed the same hidden-states tensor into several
projections (q/k/v, gate/up).  Running one larger GEMM and splitting the output
is usually faster than launching multiple smaller GEMMs, and the per-module
outputs are still fed into the existing GPTQ forward hooks so each module
accumulates its own Hessian.
"""

import threading
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import transformers


_NOT_QUANTIZE_FLAG = ":!"
_CAPTURE_ONLY_FLAG = ":?"


def _clean_module_name(name: str) -> str:
    """Strip quant/capture markers from module_tree names."""

    name = name.split(_NOT_QUANTIZE_FLAG, 1)[0]
    name = name.split(_CAPTURE_ONLY_FLAG, 1)[0]
    return name


def _get_linear_dims(module: nn.Module) -> Tuple[int, int]:
    """Return (in_features, out_features) for common linear types."""

    in_features = getattr(module, "in_features", None)
    out_features = getattr(module, "out_features", None)
    if in_features is not None and out_features is not None:
        return int(in_features), int(out_features)
    nx = getattr(module, "nx", None)
    nf = getattr(module, "nf", None)
    if nx is not None and nf is not None:
        return int(nx), int(nf)
    return 0, 0


def _is_supported_fused_module(module: nn.Module) -> bool:
    """We can only fuse plain linear/Conv1D projections."""

    return isinstance(module, (nn.Linear, transformers.Conv1D))


def _is_conv1d(module: nn.Module) -> bool:
    return isinstance(module, transformers.Conv1D)


def _weight_concat_dim(module: nn.Module) -> int:
    """Output-dim concat index: 0 for nn.Linear, 1 for transformers.Conv1D."""

    return 1 if _is_conv1d(module) else 0


def _make_input_fingerprint(input: torch.Tensor) -> Tuple[object, ...]:
    """Key by original tensor storage so q/k/v with the same input share one cache."""

    try:
        ptr = input.untyped_storage().data_ptr()
    except Exception:
        ptr = input.data_ptr()
    return (ptr, tuple(input.shape), str(input.dtype), str(input.device))


def _find_common_parent(root: nn.Module, members: List[nn.Module]) -> nn.Module:
    """Return the nearest module that contains all group members."""

    def _path(target: nn.Module) -> List[str]:
        for name, m in root.named_modules():
            if m is target:
                return name.split(".") if name else []
        return []

    paths = [_path(m) for m in members]
    if not paths or not all(paths):
        return root
    common: List[str] = []
    for parts in zip(*paths):
        if all(p == parts[0] for p in parts):
            common.append(parts[0])
        else:
            break
    if not common:
        return root
    return root.get_submodule(".".join(common))


class FusedGroupForward:
    """Coordinate a single fused GEMM for a group of same-input linear modules."""

    def __init__(self, parent_module: nn.Module, members: List[nn.Module], splice: str = "view"):
        self.parent = parent_module
        self.members = members
        self.splice = splice if splice in {"view", "contiguous_copy"} else "view"
        self.member_to_slice: Dict[int, Tuple[int, int]] = {}
        start = 0
        for m in members:
            _, out_dim = _get_linear_dims(m)
            self.member_to_slice[id(m)] = (start, start + out_dim)
            start += out_dim
        self.total_out_dim = start
        self.lock = threading.Lock()
        self._cache: Dict[
            Tuple[object, ...],
            Tuple[torch.Tensor, Dict[int, torch.Tensor]],
        ] = {}
        self._fused_weight_cache: Dict[
            Tuple[Tuple[int, ...], str, str],
            Tuple[torch.Tensor, Optional[torch.Tensor]],
        ] = {}

    def _weight_cache_key(self, target_device: torch.device, dtype: torch.dtype) -> Optional[Tuple[Tuple[int, ...], str, str]]:
        """Fingerprint member weights for cache invalidation."""

        ptrs: List[int] = []
        for m in self.members:
            w = getattr(m, "weight", None)
            if not isinstance(w, torch.Tensor):
                return None
            ptrs.append(w.untyped_storage().data_ptr())
        return (tuple(ptrs), str(target_device), str(dtype))

    def _build_fused_weight(
        self,
        target_device: torch.device,
        dtype: torch.dtype,
        weight_key: Tuple[Tuple[int, ...], str, str],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Concatenate member weights/biases along the output dimension (cached)."""

        cached = self._fused_weight_cache.get(weight_key)
        if cached is not None:
            return cached

        weights: List[torch.Tensor] = []
        biases: List[Optional[torch.Tensor]] = []
        for m in self.members:
            w = getattr(m, "weight", None)
            if not isinstance(w, torch.Tensor):
                return None, None
            if w.device != target_device or w.dtype != dtype:
                w = w.to(device=target_device, dtype=dtype)
            weights.append(w)
            b = getattr(m, "bias", None)
            if isinstance(b, torch.Tensor):
                if b.device != target_device or b.dtype != dtype:
                    b = b.to(device=target_device, dtype=dtype)
                biases.append(b)
            else:
                biases.append(None)

        if not weights:
            return None, None

        fused_weight = torch.cat(weights, dim=_weight_concat_dim(self.members[0]))

        fused_bias = None
        if any(b is not None for b in biases):
            bias_parts = []
            for b, m in zip(biases, self.members):
                s, e = self.member_to_slice[id(m)]
                if b is None:
                    bias_parts.append(
                        torch.zeros(e - s, device=target_device, dtype=dtype)
                    )
                else:
                    bias_parts.append(b)
            fused_bias = torch.cat(bias_parts, dim=0)

        self._fused_weight_cache = {weight_key: (fused_weight, fused_bias)}
        return fused_weight, fused_bias

    def __call__(
        self,
        requester: nn.Module,
        input: torch.Tensor,
        target_device: torch.device,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return the requester's output slice and the moved input used for the hook."""

        requester_slice = self.member_to_slice.get(id(requester))
        if requester_slice is None:
            return None

        requester_weight = getattr(requester, "weight", None)
        if not isinstance(requester_weight, torch.Tensor):
            return None
        dtype = requester_weight.dtype
        weight_key = self._weight_cache_key(target_device, dtype)
        if weight_key is None:
            return None

        input_key = _make_input_fingerprint(input)
        cache_key = (input_key, weight_key)

        with self.lock:
            entry = self._cache.get(cache_key)
            if entry is not None:
                moved_input, outputs = entry
                output = outputs.get(id(requester))
                if output is not None:
                    return output, moved_input
                # Entry exists but missing slice; recompute below.

            if input.device != target_device:
                moved_input = input.to(device=target_device)
            else:
                moved_input = input

            fused_weight, fused_bias = self._build_fused_weight(
                target_device, dtype, weight_key
            )
            if fused_weight is None:
                return None

            if _is_conv1d(self.members[0]):
                orig_shape = moved_input.shape
                flat = moved_input.reshape(-1, orig_shape[-1])
                fused_out = torch.matmul(flat, fused_weight)
                if fused_bias is not None:
                    fused_out = fused_out + fused_bias
                fused_out = fused_out.reshape(*orig_shape[:-1], fused_weight.shape[1])
            else:
                fused_out = torch.nn.functional.linear(
                    moved_input, fused_weight, fused_bias
                )

            outputs: Dict[int, torch.Tensor] = {}
            for m in self.members:
                s, e = self.member_to_slice[id(m)]
                output_slice = fused_out[..., s:e]
                if self.splice == "contiguous_copy":
                    output_slice = output_slice.contiguous()
                outputs[id(m)] = output_slice

            # One entry is enough; input tensors change per calibration batch.
            if self._cache:
                self._cache.clear()
            self._cache[cache_key] = (moved_input, outputs)

            return outputs[id(requester)], moved_input

    def clear(self) -> None:
        """Drop cached tensors so the module can be moved/offloaded without GPU leakage."""

        with self.lock:
            self._cache.clear()
            self._fused_weight_cache.clear()


def _all_same_device_dtype(members: List[nn.Module]) -> bool:
    """Fusion is only safe when all weights live on the same device/dtype."""

    first = members[0]
    first_w = getattr(first, "weight", None)
    if not isinstance(first_w, torch.Tensor):
        return False
    for m in members[1:]:
        w = getattr(m, "weight", None)
        if not isinstance(w, torch.Tensor):
            return False
        if w.device != first_w.device or w.dtype != first_w.dtype:
            return False
    return True


def install_fused_group_forward(
    layer_module: nn.Module,
    layer_modules_blocks: List[List[str]],
    enabled: bool = True,
    splice: str = "view",
    logger=None,
) -> int:
    """Attach FusedGroupForward helpers to all same-input linear groups in a layer.

    Args:
        layer_module: the concrete decoder layer (e.g. LlamaDecoderLayer).
        layer_modules_blocks: output of BaseQModel.build_layer_modules for the layer.
        enabled: master switch; False is a no-op.
        splice: how to return per-member outputs — 'contiguous_copy' or 'view'.
        logger: optional logger for diagnostics.

    Returns:
        Number of groups that were successfully wired.
    """

    if not enabled:
        return 0

    def _try_install_group(members: List[nn.Module]) -> bool:
        if len(members) < 2:
            return False
        first_in_dim = _get_linear_dims(members[0])[0]
        if first_in_dim == 0:
            return False
        if not all(_get_linear_dims(m)[0] == first_in_dim for m in members[1:]):
            return False
        if not _all_same_device_dtype(members):
            return False
        parent = _find_common_parent(layer_module, members)
        fused = FusedGroupForward(parent, members, splice=splice)
        for m in members:
            m._fused_group_forward = fused
        return True

    installed = 0
    for block in layer_modules_blocks:
        members: List[nn.Module] = []
        for raw_name in block:
            name = _clean_module_name(raw_name)
            try:
                m = layer_module.get_submodule(name)
            except AttributeError:
                if _try_install_group(members):
                    installed += 1
                members = []
                continue
            if not _is_supported_fused_module(m):
                if _try_install_group(members):
                    installed += 1
                members = []
                continue
            in_dim, out_dim = _get_linear_dims(m)
            if in_dim == 0 or out_dim == 0:
                if _try_install_group(members):
                    installed += 1
                members = []
                continue
            if members and _get_linear_dims(members[0])[0] != in_dim:
                if _try_install_group(members):
                    installed += 1
                members = []
            members.append(m)
        if _try_install_group(members):
            installed += 1

    return installed


def clear_fused_group_forward_caches(module: nn.Module) -> bool:
    """Release GPU tensors held by a module's FusedGroupForward helper, if any."""

    fg = getattr(module, "_fused_group_forward", None)
    if isinstance(fg, FusedGroupForward):
        fg.clear()
        delattr(module, "_fused_group_forward")
        return True
    return False
