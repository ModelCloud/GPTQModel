# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Fuse same-input linear forward passes during GPTQ calibration.

Transformer layers often feed the same hidden-states tensor into several
projections (q/k/v, gate/up).  Running one larger GEMM and splitting the output
is usually faster than launching multiple smaller GEMMs, and the per-module
outputs are still fed into the existing GPTQ forward hooks so each module
accumulates its own Hessian.

To avoid doubling weight memory, FusedGroupForward lazily builds one contiguous
shared storage buffer and makes each member's `weight` Parameter a view into
that buffer.  The quantizer writes each reconstructed `wq` back into the
shared buffer, so the fused forward always uses the current weights without
keeping a second concatenated copy.
"""

import threading
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pcre
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
    """Coordinate a single fused GEMM for a group of same-input linear modules.

    Weight data is stored once in a shared contiguous buffer.  Each member's
    ``nn.Parameter`` is a view into that buffer, so the fused forward never
    holds an extra concatenated copy of the weights.
    """

    def __init__(self, parent_module: nn.Module, members: List[nn.Module], splice: str = "view"):
        self.parent = parent_module
        self.members = members
        self.splice = splice if splice in {"view", "contiguous_copy"} else "view"
        self.concat_dim = _weight_concat_dim(members[0])
        self.in_dim, _ = _get_linear_dims(members[0])
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

        # Shared storage; allocated lazily on the target device at first use.
        self.fused_weight_storage: Optional[torch.Tensor] = None
        self.fused_bias_storage: Optional[torch.Tensor] = None
        self._version = 0

    def _weight_view(self, s: int, e: int) -> torch.Tensor:
        """Return a view into the shared weight storage for a member's slice."""

        if self.concat_dim == 0:
            return self.fused_weight_storage[s:e, :]
        return self.fused_weight_storage[:, s:e]

    def _initialize_storage(self, target_device: torch.device, dtype: torch.dtype) -> None:
        """Allocate or rebuild shared storage on target_device and reparameterize members as views.

        This is called lazily at the first forward so the storage lands on the
        device the model forward is actually executing on.
        """

        if (
            self.fused_weight_storage is not None
            and self.fused_weight_storage.device == target_device
            and self.fused_weight_storage.dtype == dtype
        ):
            return

        with self.lock:
            # Double-check after acquiring the lock.
            if (
                self.fused_weight_storage is not None
                and self.fused_weight_storage.device == target_device
                and self.fused_weight_storage.dtype == dtype
            ):
                return

            # _initialize_storage may be called while the parent forward is under
            # inference mode, but the shared buffer must be a normal tensor so
            # later quantizer updates (copy_ of reconstructed wq) are allowed.
            with torch.inference_mode(False):
                if self.concat_dim == 0:
                    weight_shape = (self.total_out_dim, self.in_dim)
                else:
                    weight_shape = (self.in_dim, self.total_out_dim)

                fused_weight = torch.empty(
                    weight_shape,
                    dtype=dtype,
                    device=target_device,
                )

                has_bias = False
                for m in self.members:
                    b = getattr(m, "bias", None)
                    if isinstance(b, torch.nn.Parameter) or isinstance(b, torch.Tensor):
                        has_bias = True
                        break

                fused_bias = None
                if has_bias:
                    fused_bias = torch.zeros(self.total_out_dim, dtype=dtype, device=target_device)

                for m in self.members:
                    s, e = self.member_to_slice[id(m)]
                    w = getattr(m, "weight", None)
                    if not isinstance(w, torch.Tensor):
                        continue
                    # Copy member weight into the shared buffer; the original
                    # Parameter will be replaced by a view and can be freed.
                    fused_weight.narrow(self.concat_dim, s, e - s).copy_(w.detach().to(device=target_device, dtype=dtype))

                    b = getattr(m, "bias", None)
                    if fused_bias is not None and isinstance(b, torch.Tensor):
                        fused_bias[s:e].copy_(b.detach().to(device=target_device, dtype=dtype))

                    # Replace the member's Parameter with a view into shared storage.
                    requires_grad = getattr(w, "requires_grad", True)
                    if self.concat_dim == 0:
                        weight_view = fused_weight[s:e, :]
                    else:
                        weight_view = fused_weight[:, s:e]
                    m.weight = nn.Parameter(weight_view, requires_grad=requires_grad)
                    if fused_bias is not None:
                        # If this member had a bias, replace it with a view too; otherwise leave None.
                        if isinstance(b, torch.nn.Parameter) or (isinstance(b, torch.Tensor) and b is not None):
                            bias_requires_grad = getattr(b, "requires_grad", requires_grad)
                            m.bias = nn.Parameter(fused_bias[s:e], requires_grad=bias_requires_grad)

                self.fused_weight_storage = fused_weight
                self.fused_bias_storage = fused_bias
                self._version = 0

    def has_storage(self) -> bool:
        return self.fused_weight_storage is not None

    def _cache_key(
        self,
        input: torch.Tensor,
        target_device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tuple[object, ...]]:
        if self.fused_weight_storage is None:
            return None
        input_key = _make_input_fingerprint(input)
        weight_key = (
            self._version,
            self.fused_weight_storage.untyped_storage().data_ptr(),
            str(dtype),
            str(target_device),
        )
        return (input_key, weight_key)

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

        # Ensure shared storage exists on the device the forward is running on.
        self._initialize_storage(target_device, dtype)
        if self.fused_weight_storage is None:
            return None

        cache_key = self._cache_key(input, target_device, dtype)
        if cache_key is None:
            return None

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

            fused_weight = self.fused_weight_storage
            fused_bias = self.fused_bias_storage

            if _is_conv1d(self.members[0]):
                orig_shape = moved_input.shape
                flat = moved_input.reshape(-1, orig_shape[-1])
                fused_out = torch.matmul(flat, fused_weight)
                if fused_bias is not None:
                    fused_out = fused_out + fused_bias
                fused_out = fused_out.reshape(*orig_shape[:-1], fused_weight.shape[1])
            else:
                fused_out = torch.nn.functional.linear(moved_input, fused_weight, fused_bias)

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

    def update_member_weight(self, module: nn.Module, wq: torch.Tensor) -> bool:
        """Write a reconstructed weight back into the shared storage and refresh the view.

        The quantizer calls this instead of ``module.weight.data = wq`` so the
        fused forward buffer stays consistent with the per-member weight.
        """

        slice_info = self.member_to_slice.get(id(module))
        if slice_info is None or self.fused_weight_storage is None:
            return False
        s, e = slice_info

        target_device = self.fused_weight_storage.device
        dtype = self.fused_weight_storage.dtype
        wq_moved = wq.to(device=target_device, dtype=dtype)

        self.fused_weight_storage.narrow(self.concat_dim, s, e - s).copy_(wq_moved)
        self._version += 1

        # Make sure the module's Parameter is a view into the updated storage.
        requires_grad = getattr(module.weight, "requires_grad", True)
        module.weight = nn.Parameter(self._weight_view(s, e), requires_grad=requires_grad)
        return True

    def clear(self) -> None:
        """Drop cached tensors so the module can be moved/offloaded without GPU leakage.

        The shared weight storage is retained because the member Parameter views
        still point to it; it will be freed naturally when those views are gone.
        """

        with self.lock:
            self._cache.clear()


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


def _expert_index_from_name(name: str) -> Optional[int]:
    """Extract the expert index from a module name, or None for non-expert paths."""

    pattern = r"[./]experts[./](\d+)[./]"
    m = pcre.search(pattern, name)
    if not m:
        return None
    return int(m.group(1))


def _is_moe_down_proj_group(names: List[str]) -> bool:
    """Return True when members are per-expert down-projections (different inputs).

    In MoE MLPs the gate/up projections all consume the same hidden state, so
    fusing them across experts is valid. The down projection (or Mixtral w2)
    consumes the expert-specific activated intermediate, so cross-expert fusion
    would waste work without sharing inputs.
    """

    if len(names) < 2:
        return False
    suffixes = {"down_proj", "w2"}
    last_parts = [n.rsplit(".", 1)[-1] for n in names]
    if not all(p in suffixes for p in last_parts):
        return False
    expert_indices = set()
    pattern = r"[./]experts[./](\d+)[./]"
    for n in names:
        m = pcre.search(pattern, n)
        if not m:
            return False
        expert_indices.add(int(m.group(1)))
    return len(expert_indices) > 1


def _partition_into_subgroups(
    members: List[nn.Module],
    names: List[str],
) -> List[Tuple[List[nn.Module], List[str]]]:
    """Partition a candidate group into safe fusion subgroups.

    Modules belonging to different MoE experts should not be fused into one giant
    group.  This keeps per-expert gate/up pairs together and avoids cross-device
    copies during multi-GPU quantization.
    """

    if not members or len(members) < 2:
        return []

    # If none of the names contain an expert index, the whole list is one group.
    has_experts = any(_expert_index_from_name(n) is not None for n in names)
    if not has_experts:
        return [(members, names)]

    # Group by expert index; shared/non-expert modules go into their own bucket.
    groups: Dict[Optional[int], List[Tuple[nn.Module, str]]] = defaultdict(list)
    for m, n in zip(members, names):
        idx = _expert_index_from_name(n)
        groups[idx].append((m, n))

    result: List[Tuple[List[nn.Module], List[str]]] = []
    for idx in sorted(groups, key=lambda k: (k is not None, k or -1)):
        sub_members = [m for m, _ in groups[idx]]
        sub_names = [n for _, n in groups[idx]]
        if len(sub_members) >= 2:
            result.append((sub_members, sub_names))
    return result


def _try_install_group(
    layer_module: nn.Module,
    members: List[nn.Module],
    names: List[str],
    splice: str,
    logger=None,
) -> bool:
    if len(members) < 2:
        return False
    if _is_moe_down_proj_group(names):
        return False
    first_in_dim = _get_linear_dims(members[0])[0]
    if first_in_dim == 0:
        return False
    if not all(_get_linear_dims(m)[0] == first_in_dim for m in members[1:]):
        return False
    if not _all_same_device_dtype(members):
        return False

    # Partition into per-expert/shared subgroups and install a FusedGroupForward
    # for each one.  This bounds the fused weight size and respects multi-GPU
    # device placement for MoE models.
    subgroups = _partition_into_subgroups(members, names)
    if not subgroups:
        return False

    installed_any = False
    for sub_members, _ in subgroups:
        if len(sub_members) < 2:
            continue
        if not _all_same_device_dtype(sub_members):
            continue
        parent = _find_common_parent(layer_module, sub_members)
        fused = FusedGroupForward(parent, sub_members, splice=splice)
        for m in sub_members:
            m._fused_group_forward = fused
        installed_any = True
    return installed_any


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

    installed = 0
    for block in layer_modules_blocks:
        members: List[nn.Module] = []
        member_names: List[str] = []
        for raw_name in block:
            name = _clean_module_name(raw_name)
            try:
                m = layer_module.get_submodule(name)
            except AttributeError:
                if _try_install_group(layer_module, members, member_names, splice=splice, logger=logger):
                    installed += 1
                members = []
                member_names = []
                continue
            if not _is_supported_fused_module(m):
                if _try_install_group(layer_module, members, member_names, splice=splice, logger=logger):
                    installed += 1
                members = []
                member_names = []
                continue
            in_dim, out_dim = _get_linear_dims(m)
            if in_dim == 0 or out_dim == 0:
                if _try_install_group(layer_module, members, member_names, splice=splice, logger=logger):
                    installed += 1
                members = []
                member_names = []
                continue
            if members and _get_linear_dims(members[0])[0] != in_dim:
                if _try_install_group(layer_module, members, member_names, splice=splice, logger=logger):
                    installed += 1
                members = []
                member_names = []
            members.append(m)
            member_names.append(name)
        if _try_install_group(layer_module, members, member_names, splice=splice, logger=logger):
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
