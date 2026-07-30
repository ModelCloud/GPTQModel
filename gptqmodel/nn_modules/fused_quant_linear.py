# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import re
import uuid
from types import MethodType
from typing import Any, Callable, List, Optional, Tuple
from weakref import ref

import torch
import torch.nn as nn

from ..utils.logger import setup_logger

log = setup_logger()

try:
    from transformers.activations import ACT2FN
except Exception:
    ACT2FN = {}


_FUSED_QKV_CACHE_ATTR = "_gptqmodel_fused_qkv_cache"
_FUSED_GATEUP_CACHE_ATTR = "_gptqmodel_fused_gateup_cache"


def _has_active_rotation(module: nn.Module) -> bool:
    """Return True if any online Hadamard state is configured for this module."""
    return (
        getattr(module, "online_full_had", False)
        or getattr(module, "online_partial_had", False)
        or getattr(module, "had_K", None) is not None
    )


def _projection_device(module: nn.Module) -> Optional[torch.device]:
    for name in ("qweight", "scales", "qzeros", "g_idx"):
        tensor = getattr(module, name, None)
        if isinstance(tensor, torch.Tensor):
            return tensor.device
    return None


# Per-role fusion flags. Each projection in a group is tagged with its semantic
# role so grouping does not depend on tuple ordering or subgroup naming.
_ROLE_Q = "q"
_ROLE_K = "k"
_ROLE_V = "v"
_ROLE_GATE = "gate"
_ROLE_UP = "up"
_ROLE_DOWN = "down"
_FUSION_ROLES = frozenset([_ROLE_Q, _ROLE_K, _ROLE_V, _ROLE_GATE, _ROLE_UP, _ROLE_DOWN])

# Canonical output order inside a fused group.
_ROLE_SORT_KEY = {
    _ROLE_Q: 0,
    _ROLE_K: 1,
    _ROLE_V: 2,
    _ROLE_GATE: 0,
    _ROLE_UP: 1,
    _ROLE_DOWN: 99,
}
_ROLE_CATEGORY = {
    _ROLE_Q: "qkv",
    _ROLE_K: "qkv",
    _ROLE_V: "qkv",
    _ROLE_GATE: "gateup",
    _ROLE_UP: "gateup",
    _ROLE_DOWN: "down",
}


def _parse_module_spec(spec: Any) -> Tuple[str, List[str]]:
    """Return the module name and the list of flags from a module-tree spec."""
    if not isinstance(spec, str):
        return str(spec), []
    parts = spec.split(":")
    return parts[0], parts[1:]


def _numeric_group_from_flags(flags: List[str]) -> Optional[int]:
    for flag in flags:
        if flag.isdigit():
            return int(flag)
    return None


def _role_and_category_from_flags(flags: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return the first recognized fusion role in `flags` and its category."""
    for flag in flags:
        if flag in _FUSION_ROLES:
            return flag, _ROLE_CATEGORY[flag]
    return None, None


def _get_activation_fn(name: str) -> Optional[Callable]:
    """Return a callable activation given a string name, if known."""
    if not name:
        return None
    name = str(name).lower()
    fn = ACT2FN.get(name)
    if fn is not None:
        return fn
    mapping = {
        "silu": torch.nn.functional.silu,
        "swish": torch.nn.functional.silu,
        "gelu": torch.nn.functional.gelu,
        "relu": torch.nn.functional.relu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "identity": lambda t: t,
        "linear": lambda t: t,
    }
    return mapping.get(name)


def _detect_mlp_activation(parent: nn.Module) -> Optional[Callable]:
    """Find the gate activation callable used by an MLP module, or None."""
    act_fn = getattr(parent, "act_fn", None)
    if callable(act_fn) and not isinstance(act_fn, nn.Module):
        return act_fn
    if isinstance(act_fn, nn.Module):
        return act_fn

    hidden_act = getattr(parent, "hidden_act", None)
    if hidden_act is None:
        config = getattr(parent, "config", None)
        if config is not None:
            hidden_act = getattr(config, "hidden_act", None)
    if hidden_act is not None:
        if isinstance(hidden_act, str):
            return _get_activation_fn(hidden_act)
        if callable(hidden_act):
            return hidden_act

    for attr in ("activation_function", "activation", "act"):
        act_fn = getattr(parent, attr, None)
        if callable(act_fn):
            return act_fn
    return None


def _is_safe_mlp_parent(parent: nn.Module, parent_name: str) -> bool:
    """Return True when `parent` looks like a dense MLP (not an expert container)."""
    if isinstance(parent, (nn.ModuleList, nn.ModuleDict)):
        return False
    last = parent_name.split(".")[-1] if parent_name else ""
    if last.isdigit():
        return False
    try:
        sig = inspect.signature(parent.forward)
    except Exception:
        return False
    params = list(sig.parameters.values())
    # `parent.forward` is a bound method, so `self` is not in the signature.
    if len(params) not in (1, 2):
        return False
    for p in params:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            return False
    return True


def _find_gate_up_down_module(
    parent: nn.Module,
    members: List[nn.Module],
    gate_out_features: int,
    hidden_size: int,
) -> Optional[nn.Module]:
    """Find the MLP down projection that consumes the gate/up intermediate size."""
    for child in parent.children():
        if child in members:
            continue
        if not (hasattr(child, "in_features") and hasattr(child, "out_features")):
            continue
        if child.in_features == gate_out_features and child.out_features == hidden_size:
            return child
    return None


def _iter_module_child_tuples(tree: Any) -> Any:
    """Yield every tuple of module specs found in a module_tree."""
    if isinstance(tree, (list, tuple)):
        if tree and all(isinstance(item, str) for item in tree):
            yield tree
            return
        for item in tree:
            yield from _iter_module_child_tuples(item)
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _iter_module_child_tuples(value)


def get_module_tree_fusion_candidates(tree: Any) -> Tuple[List[Tuple[str, ...]], List[Tuple[str, ...]]]:
    """Parse per-role fusion flags from a model definition's module_tree.

    Returns (qkv_candidates, gateup_candidates). Candidates are built from
    modules that share the same parent and numeric group and carry compatible
    roles (q/k/v for QKV, gate/up for gate/up). `down` is recognized but not
    fused. The output tuple is ordered q -> k -> v and gate -> up.
    """
    qkv_candidates: List[Tuple[str, ...]] = []
    gateup_candidates: List[Tuple[str, ...]] = []
    seen_qkv: set[Tuple[str, ...]] = set()
    seen_gateup: set[Tuple[str, ...]] = set()

    for child_tuple in _iter_module_child_tuples(tree):
        # key: (numeric_group, category) -> list of (sort_key, name)
        groups: dict[Tuple[int, str], List[Tuple[int, str]]] = {}
        for spec in child_tuple:
            name, flags = _parse_module_spec(spec)
            if not name:
                continue
            role, category = _role_and_category_from_flags(flags)
            if role is None or category == "down":
                continue
            group = _numeric_group_from_flags(flags)
            if group is None:
                group = 0
            key = (group, category)
            groups.setdefault(key, []).append((_ROLE_SORT_KEY[role], name))

        for (group, category), members in groups.items():
            if len(members) < 2:
                continue
            # Sort by canonical role order; ties keep source order.
            members.sort(key=lambda item: item[0])
            member_tuple = tuple(name for _, name in members)
            if category == "qkv" and member_tuple not in seen_qkv:
                seen_qkv.add(member_tuple)
                qkv_candidates.append(member_tuple)
            elif category == "gateup" and member_tuple not in seen_gateup:
                seen_gateup.add(member_tuple)
                gateup_candidates.append(member_tuple)

    return qkv_candidates, gateup_candidates


class _FusedTritonKernel:
    """Backend-specific fused GEMM provider for TritonV2-style GPTQ weights.

    Concatenates packed `qweight`/`scales`/`qzeros` along the output dimension and
    runs one `QuantLinearFunction` call (dequantize + `torch.matmul`).
    """

    def __init__(
        self,
        modules: List[nn.Module],
        total_out_features: int,
        device: torch.device,
    ):
        first = modules[0]
        self.in_features = first.in_features
        self.total_out_features = total_out_features
        self.bits = first.bits
        self.pack_dtype_bits = first.pack_dtype_bits
        self.maxq = first.maxq

        self.qweight = torch.cat([m.qweight for m in modules], dim=1).contiguous().to(device)
        self.scales = torch.cat([m.scales for m in modules], dim=1).contiguous().to(device)
        self.qzeros = torch.cat([m.qzeros for m in modules], dim=1).contiguous().to(device)
        self.g_idx = first.g_idx.contiguous().to(device)
        self.bias = None

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        from .triton_utils.dequant import QuantLinearFunction

        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        if not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()

        out = QuantLinearFunction.apply(
            x_flat,
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            self.bits,
            self.pack_dtype_bits,
            self.maxq,
        )
        out = out.reshape(*orig_shape[:-1], self.total_out_features)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


class _FusedTritonMegaKernel:
    """Triton mega-kernel provider for fused GPTQ QKV/gate-up inference.

    Like `_FusedTritonKernel`, it concatenates packed buffers along the output
    dimension, but it calls the fused `quant_matmul_248` Triton kernel (dequant
    + GEMM in one launch) instead of dequantizing a full FP16/BF16 weight tensor
    and then running `torch.matmul`. This removes the large temporary weight
    allocation, reduces launch count, and is bitwise-exact with separate
    per-member `quant_matmul_248` calls.

    The mega-kernel is most beneficial for small-M (decode) QKV workloads.
    For large-M prefill or for gate/up shapes where it underperforms, it falls
    back to the dense `QuantLinearFunction` path used by `TritonV2Linear`.
    """

    # Prefill M threshold: above this, `dequant + torch.matmul` is faster.
    _PREFILL_M_THRESHOLD = 256

    def __init__(
        self,
        modules: List[nn.Module],
        total_out_features: int,
        device: torch.device,
    ):
        first = modules[0]
        self.in_features = first.in_features
        self.total_out_features = total_out_features
        self.bits = first.bits
        self.maxq = first.maxq
        self.pack_dtype_bits = first.pack_dtype_bits

        self.qweight = torch.cat([m.qweight for m in modules], dim=1).contiguous().to(device)
        self.scales = torch.cat([m.scales for m in modules], dim=1).contiguous().to(device)
        self.qzeros = torch.cat([m.qzeros for m in modules], dim=1).contiguous().to(device)
        self.g_idx = first.g_idx.contiguous().to(device)
        self.bias = None

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        if not x_flat.is_contiguous():
            x_flat = x_flat.contiguous()

        # Large-M prefill is faster through the dequant+cuBLAS path.
        if x_flat.shape[0] > self._PREFILL_M_THRESHOLD:
            from .triton_utils.dequant import QuantLinearFunction

            out = QuantLinearFunction.apply(
                x_flat,
                self.qweight,
                self.scales,
                self.qzeros,
                self.g_idx,
                self.bits,
                self.pack_dtype_bits,
                self.maxq,
            )
        else:
            from .triton_utils.kernels import quant_matmul_248

            out = quant_matmul_248(
                x_flat,
                self.qweight,
                self.scales,
                self.qzeros,
                self.g_idx,
                self.bits,
                self.maxq,
            )
        out = out.reshape(*orig_shape[:-1], self.total_out_features)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out


class _FusedMarlinKernel:
    """Backend-specific fused GEMM provider for Marlin-packed GPTQ weights.

    Marlin repacks `qweight` to `(padded_in_features // 16, padded_out_features * 16 // bits)`
    (for 4-bit this is `(in//16, out*2)`). We concatenate along the second (output) dimension,
    and concatenate `scales`/`bias` along the output dimension as well.
    """

    def __init__(
        self,
        modules: List[nn.Module],
        total_out_features: int,
        device: torch.device,
    ):
        from ..utils.marlin import marlin_make_workspace_new

        first = modules[0]
        self.in_features = first.in_features
        self.padded_in_features = first.padded_in_features
        self.total_out_features = total_out_features
        # All member out_features are multiples of 64, so no per-member padding remains.
        self.padded_out_features = total_out_features

        self.weight_type = first.weight_type
        self.is_k_full = first.is_k_full
        self.fp32 = first.fp32
        self.packed_prefill = first.packed_prefill
        self.packed_prefill_min_rows = first.packed_prefill_min_rows
        self.packed_prefill_config = first.packed_prefill_config

        # qweight is Marlin-packed with output dimension in columns.
        self.qweight = torch.cat([m.qweight for m in modules], dim=1).contiguous().to(device)
        self.scales = torch.cat([m.scales for m in modules], dim=1).contiguous().to(device)
        # GPTQ Marlin absorbs zero points into qweight; qzeros is intentionally empty
        # but we still own a contiguous device copy so original buffers can be released.
        self.qzeros = first.qzeros.contiguous().to(device)
        self.g_idx = first.g_idx.contiguous().to(device)
        self.g_idx_sort_indices = first.g_idx_sort_indices.contiguous().to(device)
        self.bias = None
        self.workspace = marlin_make_workspace_new(device, min_workspace_blocks=128)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        from ..utils.marlin import apply_gptq_marlin_linear

        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.shape[-1] != self.padded_in_features:
            pad = self.padded_in_features - self.in_features
            x_2d = torch.nn.functional.pad(x_2d, (0, pad))

        # Marlin requires weight scales to match activation dtype at runtime.
        if x_2d.dtype != self.scales.dtype:
            self.scales = self.scales.to(x_2d.dtype)

        marlin_input = x_2d.contiguous()
        rows = marlin_input.shape[0]
        use_packed_prefill = (
            self.packed_prefill
            and not (marlin_input.ndim >= 3 and marlin_input.shape[-2] == 1)
            and rows >= self.packed_prefill_min_rows
        )

        out = apply_gptq_marlin_linear(
            input=marlin_input,
            weight=self.qweight,
            weight_scale=self.scales,
            weight_zp=self.qzeros,
            g_idx=self.g_idx,
            g_idx_sort_indices=self.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=self.weight_type,
            output_size_per_partition=self.padded_out_features,
            input_size_per_partition=self.padded_in_features,
            is_k_full=self.is_k_full,
            bias=self.bias,
            use_fp32_reduce=self.fp32,
            use_atomics=False,
            use_packed_prefill=use_packed_prefill,
            packed_prefill_config=self.packed_prefill_config,
        )
        out = out.reshape(*orig_shape[:-1], self.padded_out_features)
        if self.padded_out_features != self.total_out_features:
            out = out[..., : self.total_out_features]
        return out

    @torch.inference_mode()
    def dequantize_weight(
        self,
        dtype: Optional[torch.dtype] = None,
        max_chunk_rows: int = 1024,
    ) -> torch.Tensor:
        """Return a dense (in_features, total_out_features) weight tensor.

        Like `MarlinLinear.dequantize_weight`, this feeds an identity matrix through
        the Marlin GEMM.  The result is exact with respect to `self.compute` and
        can be sliced into per-member dense weights for grouped GEMM dispatch.
        """
        from ..utils.marlin import gptq_marlin_gemm, marlin_make_workspace_new

        target_dtype = dtype or self.scales.dtype
        device = self.qweight.device

        k_padded = self.padded_in_features
        n_padded = self.padded_out_features
        rows = self.in_features
        out_cols = self.total_out_features

        if k_padded == 0 or n_padded == 0 or rows == 0:
            return torch.empty(rows, out_cols, dtype=target_dtype, device=device)

        workspace = getattr(self, "workspace", None)
        if workspace is None:
            workspace = marlin_make_workspace_new(device, min_workspace_blocks=128)

        chunks: list[torch.Tensor] = []
        c = torch.empty((max_chunk_rows, n_padded), dtype=target_dtype, device=device)

        for start in range(0, rows, max_chunk_rows):
            end = min(start + max_chunk_rows, rows)
            cur_m = end - start

            a = torch.zeros((cur_m, k_padded), dtype=target_dtype, device=device)
            local_idx = torch.arange(cur_m, device=device)
            a[local_idx, local_idx + start] = 1.0

            c_chunk = c if cur_m == max_chunk_rows else torch.empty((cur_m, n_padded), dtype=target_dtype, device=device)

            gptq_marlin_gemm(
                a,
                c_chunk,
                self.qweight,
                None,
                self.scales,
                None,
                self.qzeros,
                self.g_idx,
                self.g_idx_sort_indices,
                workspace,
                self.weight_type,
                cur_m,
                n_padded,
                k_padded,
                self.is_k_full,
                False,
                self.fp32,
                False,
                False,
                0,
            )
            chunks.append(c_chunk[:, :out_cols].clone())

        return torch.cat(chunks, dim=0)


def _create_fused_kernel(
    modules: List[nn.Module],
    total_out_features: int,
    device: torch.device,
    category: str = "",
):
    """Return the correct backend-specific fused kernel for `modules`."""
    first = modules[0]
    from .qlinear.tritonv2 import TritonV2Linear

    if isinstance(first, TritonV2Linear):
        # Use the fused dequant+GEMM Triton mega-kernel for small-M QKV decode
        # workloads. It requires the standard 32-bit packed TritonV2 layout that
        # `quant_matmul_248` assumes. Gate/up and large-M prefill fall back to
        # the dense `QuantLinearFunction` path used by `TritonV2Linear`.
        if (
            category == "qkv"
            and first.pack_dtype_bits == 32
            and first.bits in (2, 4, 8)
        ):
            return _FusedTritonMegaKernel(modules, total_out_features, device)
        return _FusedTritonKernel(modules, total_out_features, device)

    from .qlinear.marlin import MarlinLinear

    if isinstance(first, MarlinLinear):
        return _FusedMarlinKernel(modules, total_out_features, device)

    raise TypeError(f"Unsupported module type for fused inference: {type(first)}")


# Buffers that become redundant once a fused kernel owns a contiguous copy of the
# packed weights. Deleting them from the member modules recovers the VRAM that would
# otherwise be duplicated (the fused kernel already contains the concatenated data).
_REDUNDANT_FUSION_BUFFERS = frozenset([
    "qweight",
    "scales",
    "qzeros",
    "g_idx",
    "g_idx_sort_indices",
])


class _FusedQuantGroup:
    """Container for a set of same-input BaseQuantLinear modules fused along out_features.

    The group is backend-agnostic; it delegates the actual fused GEMM to a backend-specific
    provider (TritonV2 or Marlin) and only handles output slicing and cross-member caching.
    Only the first member actually launches the GEMM; the other members return cached
    slices from that single launch.
    """

    def __init__(
        self,
        modules: List[nn.Module],
        attr_name: str,
        category: str = "",
        free_original_weights: bool = True,
    ):
        self._refs = tuple(ref(m) for m in modules)
        self.attr_name = attr_name
        self.cache_key = uuid.uuid4().hex

        # Static attributes from the first module; all members were already validated
        # to share these values.
        first = modules[0]
        self.in_features = first.in_features
        self.total_out_features = sum(m.out_features for m in modules)

        # Build output slices for each member.
        cursor = 0
        self.slices: List[Tuple[int, int]] = []
        for m in modules:
            out = m.out_features
            self.slices.append((cursor, cursor + out))
            cursor += out

        device = _projection_device(first) or first.qweight.device
        self.kernel = _create_fused_kernel(modules, self.total_out_features, device, category)

        if free_original_weights:
            self._release_member_buffers(modules)

    def _release_member_buffers(self, modules: List[nn.Module]) -> None:
        """Remove the now-redundant packed buffers from member modules.

        The fused kernel owns a contiguous concatenated copy, so keeping the per-member
        `qweight`/`scales`/`qzeros`/`g_idx` buffers doubles memory use. Deleting them
        keeps the member modules lightweight references to output slices. Any small
        non-redundant buffers (e.g. `bias`, `wf_unsqueeze_neg_one`) are preserved.
        """
        for m in modules:
            for name in _REDUNDANT_FUSION_BUFFERS:
                if hasattr(m, name):
                    try:
                        delattr(m, name)
                    except (AttributeError, RuntimeError):
                        pass

    def _compute(self, x: torch.Tensor) -> torch.Tensor:
        return self.kernel.compute(x)

    def forward(self, member_idx: int, x: torch.Tensor) -> torch.Tensor:
        cache_attr = self.attr_name
        cache = getattr(x, cache_attr, None)
        if cache is not None and cache[0] == self.cache_key:
            if member_idx != 0:
                outputs = cache[1]
                consumed = cache[2]
                consumed.add(member_idx)
                # Free the retained fused output once every non-primary slice has been
                # consumed, so a reused input tensor cannot return stale results on the
                # next forward and memory is released promptly.
                if len(consumed) == len(outputs) - 1:
                    try:
                        setattr(x, cache_attr, None)
                    except (AttributeError, RuntimeError) as exc:
                        log.warn.once(
                            f"Unable to clear fused projection cache on input tensor: {exc}. "
                            "This is harmless but may retain memory briefly."
                        )
                return outputs[member_idx]
            # member_idx == 0 always recomputes and refreshes the cache.

        fused = self._compute(x)
        outputs = []
        for start, end in self.slices:
            outputs.append(fused[..., start:end])

        # Cache the sibling slices on the input tensor. The primary member (0)
        # recomputes; later siblings return their cached slice and clear the cache
        # after the final sibling has consumed it.
        consumed: set[int] = set()
        if member_idx != 0:
            consumed.add(member_idx)
        try:
            setattr(x, cache_attr, (self.cache_key, outputs, consumed))
        except (AttributeError, RuntimeError) as exc:
            log.warn.once(
                f"Unable to cache fused projection slices on input tensor: {exc}. "
                "Every member of the fused group will recompute the full GEMM, which is slower."
            )

        return outputs[member_idx]

    def __call__(self, member_idx: int, x: torch.Tensor) -> torch.Tensor:
        return self.forward(member_idx, x)

    def dequantize_weight(self, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
        """Return a dense (in_features, total_out_features) weight tensor.

        TritonV2-backed groups use the fast unpack/dequant path. Marlin-backed
        groups delegate to the kernel's own identity-GEMM dequantization, which is
        exact with respect to the fused Marlin forward and therefore safe for
        grouped GEMM dispatch.
        """
        kernel = self.kernel
        if isinstance(kernel, _FusedMarlinKernel):
            return kernel.dequantize_weight(dtype)

        from .triton_utils.dequant import dequant

        target_dtype = dtype if dtype is not None else torch.float16
        return dequant(
            target_dtype,
            kernel.qweight,
            kernel.scales,
            kernel.qzeros,
            kernel.g_idx,
            kernel.bits,
            kernel.pack_dtype_bits,
            kernel.maxq,
        )


@torch._dynamo.disable
def _fused_projection_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    group = self._gptqmodel_fused_group
    return group.forward(self._gptqmodel_fused_idx, x)


@torch._dynamo.disable
def _fused_gateup_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Fused gate/up MLP forward: one fused GEMM, then act(gate) * up and down."""
    group = self._gptqmodel_fused_gateup_group
    down = self._gptqmodel_fused_gateup_down
    act_fn = self._gptqmodel_fused_gateup_act
    gate_size = self._gptqmodel_fused_gateup_gate_size

    gate_up = group._compute(x)
    orig_shape = gate_up.shape
    gate_up_2d = gate_up.reshape(-1, gate_up.shape[-1])
    gate = gate_up_2d[:, :gate_size]
    up = gate_up_2d[:, gate_size:]

    # act_fn returns a new tensor; mul_(up) reuses it for the product, saving one
    # full-size allocation compared to a separate act(gate) * up expression.
    act = act_fn(gate)
    act = act.mul_(up)
    del gate, up, gate_up_2d, gate_up

    act = act.reshape(*orig_shape[:-1], gate_size)
    return down(act)


def _can_fuse_modules(modules: List[nn.Module]) -> bool:
    if not modules or any(m is None for m in modules):
        return False
    if any(getattr(m, "_gptqmodel_fused_group", None) is not None for m in modules):
        return False

    first = modules[0]
    if not hasattr(first, "qweight") or not hasattr(first, "scales"):
        return False

    required_attrs = (
        "in_features", "out_features", "bits", "group_size", "desc_act", "sym",
        "pack_dtype", "pack_dtype_bits", "maxq", "g_idx",
    )
    for attr in required_attrs:
        if not all(hasattr(m, attr) for m in modules):
            return False

    # Supported backends: TritonV2 (and TrilinLinear, which inherits from it) and Marlin.
    from .qlinear.tritonv2 import TritonV2Linear
    from .qlinear.marlin import MarlinLinear
    backend_type = None
    if isinstance(first, TritonV2Linear):
        backend_type = "triton"
        # TritonV2Linear.forward dispatches to a dedicated matmul_3bit on sm80+ for 3-bit
        # weights, while the fused helper currently routes through QuantLinearFunction.
        # Gate 3-bit fusion out until a matched 3-bit fused path is validated.
        if first.bits == 3:
            return False
    elif isinstance(first, MarlinLinear):
        backend_type = "marlin"
    else:
        return False
    if not all(isinstance(m, type(first)) for m in modules):
        return False

    if any(m.training for m in modules):
        return False
    if any(getattr(m, "adapter", None) is not None for m in modules):
        return False
    if any(_has_active_rotation(m) for m in modules):
        return False
    if any(getattr(m, "bias", None) is not None for m in modules):
        # Bias concatenation is supported in principle; this gate is conservative for the MVP.
        return False

    # Quantization metadata must match exactly.
    for attr in ("in_features", "bits", "group_size", "desc_act", "sym", "pack_dtype", "pack_dtype_bits", "maxq"):
        if any(getattr(m, attr) != getattr(first, attr) for m in modules):
            return False

    # g_idx must be identical (same input grouping / desc_act permutation).
    if any(not torch.equal(m.g_idx, first.g_idx) for m in modules):
        return False

    # All buffers must live on the same device and be contiguous.
    devices = {_projection_device(m) for m in modules}
    if len(devices) != 1 or None in devices:
        return False
    device = devices.pop()
    for name in ("qweight", "scales", "qzeros", "g_idx"):
        for m in modules:
            tensor = getattr(m, name)
            if tensor.device != device or not tensor.is_contiguous():
                return False

    if backend_type == "marlin":
        # Marlin repacks weights into 64-column tiles; per-member output padding would leave
        # gaps in the concatenated output, so every member must be a clean multiple of 64.
        if any(m.out_features % 64 != 0 for m in modules):
            return False
        if any(m.padded_out_features != m.out_features for m in modules):
            return False
        if any(m.padded_in_features != m.in_features for m in modules):
            return False
        if any(not torch.equal(m.g_idx_sort_indices, first.g_idx_sort_indices) for m in modules):
            return False
        if any(m.weight_type != first.weight_type for m in modules):
            return False
        if any(m.is_k_full != first.is_k_full for m in modules):
            return False
        # Scales must be the same dtype so the fused kernel can use one scale buffer.
        if any(m.scales.dtype != first.scales.dtype for m in modules):
            return False

    # Output sizes must be divisible by the packing factor so that qzeros packs cleanly.
    pack_factor = first.pack_dtype_bits // first.bits
    if any(m.out_features % pack_factor != 0 for m in modules):
        return False

    return True


def _check_fused_gateup_mlp_parity(
    parent: nn.Module,
    gate: nn.Module,
    up: nn.Module,
    down: nn.Module,
    act_fn: Callable,
    group: _FusedQuantGroup,
    tol: float = 20.0,
) -> bool:
    """Run one forward with the original and fused paths to verify the structure matches."""
    kernel = group.kernel
    qweight = getattr(kernel, "qweight", None)
    if not isinstance(qweight, torch.Tensor):
        return False
    device = qweight.device

    dtype = getattr(kernel, "scales", None)
    if isinstance(dtype, torch.Tensor):
        dtype = dtype.dtype
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    hidden_size = getattr(gate, "in_features", None)
    if hidden_size is None:
        return False

    try:
        x = torch.randn(1, hidden_size, device=device, dtype=dtype)
        parent._gptqmodel_fused_gateup_group = group
        parent._gptqmodel_fused_gateup_down = down
        parent._gptqmodel_fused_gateup_act = act_fn
        parent._gptqmodel_fused_gateup_gate_size = gate.out_features
        with torch.inference_mode():
            original = parent.forward(x)
            fused = _fused_gateup_mlp_forward(parent, x)
    except Exception:
        return False

    if not (torch.isfinite(original).all() and torch.isfinite(fused).all()):
        return False
    return (original - fused).abs().max().item() <= tol


def _maybe_install_fused_gateup_activation(
    parent: nn.Module,
    parent_name: str,
    members: List[nn.Module],
    group: _FusedQuantGroup,
) -> bool:
    """If this parent is a dense MLP, replace its forward with a fused gate/up/down pass.

    This is only installed when we can unambiguously identify the down projection and
    the activation function, the parent is not an expert container, and a sample forward
    through the original and fused paths produces identical outputs.
    """
    gate = members[0]
    up = members[1]
    if not (hasattr(gate, "out_features") and hasattr(up, "out_features")):
        return False
    if gate.out_features != up.out_features:
        return False

    down = _find_gate_up_down_module(parent, members, gate.out_features, gate.in_features)
    if down is None:
        return False

    act_fn = _detect_mlp_activation(parent)
    if act_fn is None:
        return False

    if not _is_safe_mlp_parent(parent, parent_name):
        return False

    if not _check_fused_gateup_mlp_parity(parent, gate, up, down, act_fn, group):
        for attr in (
            "_gptqmodel_fused_gateup_group",
            "_gptqmodel_fused_gateup_down",
            "_gptqmodel_fused_gateup_act",
            "_gptqmodel_fused_gateup_gate_size",
        ):
            if hasattr(parent, attr):
                delattr(parent, attr)
        return False

    parent._gptqmodel_fused_gateup_original_forward = parent.forward
    parent.forward = MethodType(_fused_gateup_mlp_forward, parent)
    return True


def _is_moe_expert_parent(parent_name: str) -> bool:
    """Return True if `parent_name` points inside an MoE expert container."""
    if not parent_name:
        return False
    markers = ("experts.", "shared_experts.", "shared_expert")
    return any(marker in parent_name for marker in markers)


_MOE_INDIVIDUAL_EXPERT_RE = re.compile(r"\.(experts|shared_experts)\.\d+(?:\.|$)")


def _is_moe_individual_expert(parent_name: str) -> bool:
    """Return True if `parent_name` is one expert inside an MoE expert list."""
    if not parent_name:
        return False
    return _MOE_INDIVIDUAL_EXPERT_RE.search(parent_name) is not None


def _install_fused_group(
    model: nn.Module,
    member_names: Tuple[str, ...],
    attr_name: str,
    free_original_weights: bool = True,
    fuse_activation: bool = True,
) -> int:
    installed = 0
    for parent_name, parent in model.named_modules():
        members = [getattr(parent, name, None) for name in member_names]
        if not _can_fuse_modules(members):
            continue

        category = "qkv" if attr_name == _FUSED_QKV_CACHE_ATTR else "gateup"
        # Fusing gate/up inside each individual MoE expert duplicates packed
        # weight memory (torch.cat per expert).  The batched/offset Marlin MoE
        # mega-kernel handles gate and up more efficiently from their original
        # packed buffers; skip per-expert gate/up fusion when this parent is an
        # individual expert inside an MoE list.
        if category == "gateup" and _is_moe_individual_expert(parent_name):
            continue
        group = _FusedQuantGroup(
            members,
            attr_name=attr_name,
            category=category,
            free_original_weights=free_original_weights,
        )
        for idx, member in enumerate(members):
            member._gptqmodel_fused_group = group
            member._gptqmodel_fused_idx = idx
            member._gptqmodel_fused_original_forward = member.forward
            member.forward = MethodType(_fused_projection_forward, member)

        if category == "gateup" and fuse_activation and not _is_moe_expert_parent(parent_name):
            _maybe_install_fused_gateup_activation(parent, parent_name, members, group)

        installed += 1
    return installed


_QKV_CANDIDATES: List[Tuple[str, ...]] = [
    ("q_proj", "k_proj", "v_proj"),
    ("wq", "wk", "wv"),
    ("query", "key", "value"),
    ("q", "k", "v"),
]

_GATE_UP_CANDIDATES: List[Tuple[str, ...]] = [
    ("gate_proj", "up_proj"),
    ("w2", "w1"),  # Original Qwen MLP: w1(x) * silu(w2(x)) -> gate=w2, up=w1
    ("w1", "w3"),  # Llama-1/2 MLP gate/up
    ("gate", "up"),
]


def _install_fused_groups(
    model: torch.nn.Module,
    candidates: List[Tuple[str, ...]],
    attr_name: str,
    free_original_weights: bool = True,
    fuse_activation: bool = True,
) -> int:
    """Try multiple naming conventions for the same fused projection group."""
    installed = 0
    for member_names in candidates:
        installed += _install_fused_group(model, member_names, attr_name, free_original_weights=free_original_weights, fuse_activation=fuse_activation)
    return installed


def install_fused_qkv(
    model: torch.nn.Module,
    candidates: Optional[List[Tuple[str, ...]]] = None,
    free_original_weights: bool = True,
) -> int:
    """Fuse compatible attention QKV projection triples into one quantized GEMM."""
    if candidates is None:
        candidates = _QKV_CANDIDATES
    return _install_fused_groups(model, candidates, _FUSED_QKV_CACHE_ATTR, free_original_weights=free_original_weights)


def install_fused_gate_up(
    model: torch.nn.Module,
    candidates: Optional[List[Tuple[str, ...]]] = None,
    free_original_weights: bool = True,
    fuse_activation: bool = True,
) -> int:
    """Fuse compatible MLP gate/up projection pairs into one quantized GEMM."""
    if candidates is None:
        candidates = _GATE_UP_CANDIDATES
    return _install_fused_groups(
        model,
        candidates,
        _FUSED_GATEUP_CACHE_ATTR,
        free_original_weights=free_original_weights,
        fuse_activation=fuse_activation,
    )


def install_fused_quant_modules(
    model: torch.nn.Module,
    free_original_weights: bool = True,
    fuse_activation: bool = True,
) -> dict[str, int]:
    """Install all supported fused quantized modules. Returns counts per fusion type."""
    return {
        "qkv": install_fused_qkv(model, free_original_weights=free_original_weights),
        "gate_up": install_fused_gate_up(
            model,
            free_original_weights=free_original_weights,
            fuse_activation=fuse_activation,
        ),
    }


__all__ = [
    "install_fused_qkv",
    "install_fused_gate_up",
    "install_fused_quant_modules",
    "get_module_tree_fusion_candidates",
]
