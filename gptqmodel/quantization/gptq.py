# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# Based on original gptq algorithm and code from https://github.com/IST-DASLab/gptq

import bisect
import contextlib
import math
import os
import threading
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from torch.nn.modules.conv import _ConvNd

from ..looper.named_module import NamedModule
from ..quantization import QuantizeConfig
from ..quantization.config import (
    AdaptiveClippingConfig,
    AdaptiveDampingConfig,
    DampConfig,
    FallbackStrategy,
    LengthAwareConfig,
    LengthAwareMode,
    ScaleSearchConfig,
    SmoothMSE,
)
from ..utils import gte_python_3_14, has_gil_disabled
from ..utils.device import get_device
from ..utils.env import env_flag
from ..utils.gptq_block import gptq_block_cuda
from ..utils.gptq_block_mps import gptq_block_mps, gptq_block_mps_supported
from ..utils.logger import setup_logger
from ..utils.torch import (
    TORCH_GTE_28,
    TORCH_GTE_214,
    cholesky_inverse,
    linalg_cholesky,
    linalg_cholesky_ex,
    torch_compile,
    torch_sync,
)
from .fallback_smooth import mse_optimal_quant, smooth_block
from .gar import (
    compose_final_perm,
    compute_global_perm,
    compute_local_perms,
    extend_perm_with_tail,
    invert_perm,
)
from .gsq_scalar import gsq_enabled_for, refine_affine_scalar
from .npu_linalg import npu_inverse_cholesky_factor
from .quantizer import HF_OPTIMUM, Quantizer


try:
    from ..nn_modules.qlinear.pack_block_ext import gptq_block_cpu
except Exception:
    gptq_block_cpu = None

_USE_GPTQ_CUDA_BLOCK = env_flag("GPTQMODEL_CUDA_BLOCK", default=True)
_USE_GPTQ_MPS_BLOCK = env_flag("GPTQMODEL_MPS_BLOCK", default=True)
_USE_GPTQ_MPS_FUSED_PARAMS = env_flag("GPTQMODEL_MPS_FUSED_PARAMS", default=True)
_USE_GPTQ_MPS_FAST_HESSIAN = env_flag("GPTQMODEL_MPS_FAST_HESSIAN", default=True)
_USE_GPTQ_MPS_ASYNC_HESSIAN = env_flag("GPTQMODEL_MPS_ASYNC_HESSIAN", default=True)


log = setup_logger()

def _log_hessian_verbose() -> bool:
    """Verbose per-module hessian-inverse markers are opt-in.

    The QuantizationRegionTimer measurement is always recorded and flushed
    periodically, which provides aggregate stall isolation without per-module
    line noise.
    """
    return env_flag("GPTQMODEL_LOG_HESSIAN")

_WORKSPACE_LOCKS_GUARD = threading.Lock()

# Shared workspaces are cached globally per device so that concurrent GPTQ
# instances reuse temporary buffers instead of repeatedly allocating large
# tensors during Hessian accumulation. Each device retains at most a single
# workspace; when size or dtype requirements change, the prior buffer is
# discarded to avoid unbounded cache growth.
_WORKSPACE_CACHE: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}
_WORKSPACE_LOCKS: Dict[Tuple[str, Optional[int]], threading.Lock] = {}
_BF16_SUPPORT_CACHE: Dict[Tuple[str, Optional[int]], bool] = {}
# The defer depth is worker-local, not shared state. ThreadX CUDA workers can
# enter this scope independently without a lock or cross-device serialization.
_HESSIAN_SYNC_STATE = threading.local()


@contextlib.contextmanager
def defer_hessian_sync():
    """Defer per-update accelerator synchronization within one worker thread.

    The caller must synchronize the worker's device before exposing accumulated
    Hessians to another stream. Nesting is supported so lifecycle scopes can be
    composed without accidentally restoring eager synchronization too early.
    """

    depth = int(getattr(_HESSIAN_SYNC_STATE, "depth", 0))
    _HESSIAN_SYNC_STATE.depth = depth + 1
    try:
        yield
    finally:
        _HESSIAN_SYNC_STATE.depth = depth


def _hessian_sync_deferred() -> bool:
    return bool(getattr(_HESSIAN_SYNC_STATE, "depth", 0))


def _device_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    dev = torch.device(device)
    return dev.type, dev.index


def _device_reduction_key(device: torch.device) -> Tuple[str, int]:
    """Order reductions by device identity, never worker arrival order."""
    dev = torch.device(device)
    return dev.type, -1 if dev.index is None else dev.index


def _workspace_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    return _device_cache_key(device)


def _workspace_lock(key: Tuple[str, Optional[int]]) -> threading.Lock:
    lock = _WORKSPACE_LOCKS.get(key)
    if lock is not None:
        return lock

    with _WORKSPACE_LOCKS_GUARD:
        lock = _WORKSPACE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _WORKSPACE_LOCKS[key] = lock
    return lock


def _needs_workspace_resize(
    workspace: Optional[torch.Tensor],
    dtype: torch.dtype,
    required_rows: int,
    cols: int,
) -> bool:
    if workspace is None:
        return True
    if workspace.ndim != 2:
        return True
    if workspace.dtype != dtype:
        return True
    if workspace.shape[1] != cols:
        return True
    if workspace.shape[0] < required_rows:
        return True
    return False


@contextlib.contextmanager
def _lease_workspace(
    device: torch.device,
    dtype: torch.dtype,
    cols: int,
    required_rows: int,
) -> Tuple[torch.Tensor, bool]:
    key = _workspace_cache_key(device)
    lock = _workspace_lock(key)
    with lock:
        workspace = _WORKSPACE_CACHE.pop(key, None)
        reused = workspace is not None and not _needs_workspace_resize(
            workspace,
            dtype,
            required_rows,
            cols,
        )
        if not reused:
            rows = max(required_rows, 1)
            workspace = torch.empty((rows, cols), dtype=dtype, device=device)
    try:
        yield workspace, reused
    finally:
        with lock:
            _WORKSPACE_CACHE[key] = workspace


def _device_supports_bfloat16(device: torch.device) -> bool:
    cache_key = _device_cache_key(device)
    cached = _BF16_SUPPORT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    dev = torch.device(device)
    if dev.type == "meta":
        _BF16_SUPPORT_CACHE[cache_key] = False
        return False

    try:
        a = torch.zeros((1, 1), dtype=torch.bfloat16, device=dev)
        b = torch.zeros((1, 1), dtype=torch.bfloat16, device=dev)
        _ = torch.matmul(a, b)
        support = True
    except Exception:
        support = False

    _BF16_SUPPORT_CACHE[cache_key] = support
    return support


def _is_base_quant_linear_like(layer: nn.Module) -> bool:
    # Avoid importing BaseQuantLinear here; qlinear imports quantization code.
    return any(
        cls.__module__ == "gptqmodel.nn_modules.qlinear" and cls.__name__ == "BaseQuantLinear"
        for cls in type(layer).__mro__
    )


def get_number_of_rows_and_cols(layer: nn.Module):
    # return layer.weight.shape[0], math.prod(layer.weight.shape[1:])
    if isinstance(layer, NamedModule):
        layer = layer.module

    if isinstance(layer, transformers.Conv1D):
        # transformers.Conv1D: weight shape is (n_in, n_out)
        return layer.weight.shape[1], layer.weight.shape[0]
    elif isinstance(layer, nn.Embedding):
        V, D = layer.weight.shape
        return D, V  # rows = embedding_dim, cols = vocab_size (token axis)
    elif _is_base_quant_linear_like(layer):
        # BaseQuantLinear exposes dimensions without a normal dense weight.
        return layer.in_features, layer.out_features
    else:
        # weight shape is (n_out, n_in)
        return layer.weight.shape[0], math.prod(layer.weight.shape[1:])


@torch.inference_mode()
def _hessian_inverse_try_cholesky(H: torch.Tensor, diag_delta: torch.Tensor):
    """Apply a diagonal delta to ``H`` in place and attempt its Cholesky decomposition.

    Returns the lower Cholesky factor ``L`` and a 0-dim boolean ``success``
    tensor (``info == 0``). The caller is responsible for restoring ``H``'s
    diagonal between damping attempts; this helper intentionally does not clone
    so that non-shared Hessians can avoid a full matrix copy.
    """
    H.diagonal().add_(diag_delta)
    L, info = linalg_cholesky_ex(H, upper=False)
    success = (info == 0).view(())
    return L, success


@torch.inference_mode()
def _hessian_inverse_factor(L: torch.Tensor):
    """Return the upper Cholesky factor of ``H^{-1}`` from the lower Cholesky
    factor ``L`` of ``H``, matching the original ``cholesky_inverse`` + ``cholesky``
    sequence: ``U^T U = H^{-1}`` with ``U`` upper triangular.
    """
    return linalg_cholesky(cholesky_inverse(L), upper=True)


class GPTQ:
    @staticmethod
    def _resolve_effective_blocksize(
        blocksize: int,
        group_size: int,
        *,
        hessian_inverse_available: bool,
        use_online_group_damping: bool,
        use_adaptive_clipping: bool,
    ) -> int:
        """Preserve legacy GPTQ update order unless group-local work requires otherwise."""

        effective_block = blocksize
        if not hessian_inverse_available and group_size > 0:
            effective_block = group_size
        if (use_online_group_damping or use_adaptive_clipping) and group_size > 0:
            effective_block = min(blocksize, group_size)
        return effective_block

    @staticmethod
    def resolve_module_source(module: nn.Module) -> nn.Module:
        """Resolve the dense module view GPTQ should quantize for one wrapper."""

        if isinstance(module, NamedModule):
            quant_source = module.state.get("quant_source_module")
            if isinstance(quant_source, nn.Module):
                return quant_source
            return module.module
        return module

    @staticmethod
    def _sequence_count_for_input(inp: torch.Tensor) -> int:
        """Batch dimension of a 3-D+ activation tensor, or 1 for already-flattened inputs."""

        if inp.dim() >= 3:
            return max(1, int(inp.shape[0]))
        return 1

    def _lookup_length_bucket(self, length: float) -> Optional[int]:
        """Map a token length to the index of its configured bucket."""

        cfg = self.length_aware_config
        if cfg is None or cfg.bucket_boundaries is None or len(cfg.bucket_boundaries) == 0:
            return None
        bucket_idx = bisect.bisect_right(cfg.bucket_boundaries, length) - 1
        max_idx = len(cfg.bucket_boundaries) - 2
        return max(0, min(bucket_idx, max_idx))

    def _disable_length_aware_for_flat_input(self, inp: torch.Tensor) -> None:
        """Disable length-aware normalization when an activation has lost sequence membership."""

        if (
            getattr(self, "length_aware", False)
            and not isinstance(self.module, nn.Embedding)
            and torch.is_tensor(inp)
            and inp.dim() < 3
        ):
            log.warn(
                "GPTQ module '%s' received a %d-D flattened activation while length-aware "
                "normalization is active. Per-sequence membership is lost; disabling length-aware "
                "normalization for this module.",
                self.name,
                inp.dim(),
            )
            self.length_aware = False
            self.length_aware_config = LengthAwareConfig(mode=LengthAwareMode.DISABLED)

    def __init__(self, module: nn.Module, qcfg: Optional[QuantizeConfig] = None, region_timer=None):
        # Hessian capture/materialization may run concurrently on multiple device
        # workers when Python's GIL is disabled. Re-entrancy is required because
        # target-device selection is also used while materialization holds the lock.
        self.lock = threading.RLock()
        self.region_timer = region_timer

        # self.num_tied_handles = 0
        # if qcfg.tied_gptq_handle is not None:
        #     qcfg.tied_gptq_handle.num_tied_handles += 1

        # Flags indicating issues
        # self.issue_zero_samples = False
        # self.issue_nan_hessian = False
        # self.issue_non_invertible = False

        # self.W = module.weight
        resolved_module = self.resolve_module_source(module)
        self.rows, self.columns = get_number_of_rows_and_cols(resolved_module)
        if isinstance(module, NamedModule):
            self.module = resolved_module
            self.name = module.name
            self._named_module = module
        else:
            self.name = HF_OPTIMUM
            self.module = resolved_module
            self._named_module = None

        self._original_rows = self.rows
        self._original_columns = self.columns
        if self._named_module is not None:
            pad_info = self._named_module.state.get("tp_pad_info")
        else:
            pad_info = getattr(self.module, "_tp_pad_info", None)
        if isinstance(pad_info, dict):
            pad_cols = int(pad_info.get("pad_cols", 0) or 0)
            pad_cols = max(pad_cols, 0)
        else:
            pad_info = None
            pad_cols = 0

        self._tp_pad_info = pad_info
        self._tp_pad_cols = pad_cols
        if self._tp_pad_cols:
            self.columns += self._tp_pad_cols

        module_device = get_device(self.module)
        setattr(self.module, "target_device", module_device)

        if module_device.type == "meta":
            self._final_hessian_device_hint = torch.device("cpu")
        else:
            self._final_hessian_device_hint = torch.device(module_device)

        self.validate_module(self.module)

        self.qcfg = qcfg if qcfg else QuantizeConfig()  # HF compat will not pass qcfg
        if pad_cols and (self.qcfg.gptaq is not None or self.qcfg.foem is not None):
            full_name = self._named_module.full_name if self._named_module is not None else self.name
            if gsq_enabled_for(getattr(self.qcfg, "gsq", None), full_name):
                raise ValueError("GSQ with tensor-parallel padded GPTAQ/FOEM is not supported")
        hessian_cfg = getattr(self.qcfg, "hessian", None)
        self.length_aware_config = getattr(hessian_cfg, "length_aware", None)
        if not isinstance(self.length_aware_config, LengthAwareConfig):
            self.length_aware_config = LengthAwareConfig(mode=LengthAwareMode.DISABLED)
        self.length_aware = (
            self.length_aware_config.mode is not LengthAwareMode.DISABLED
            and not isinstance(self.module, nn.Embedding)
        )
        if self.length_aware and type(self) is not GPTQ:
            log.warn(
                "HessianConfig.length_aware is only implemented for the standard GPTQ algorithm; "
                "disabling it for `%s`.",
                type(self).__name__,
            )
            self.length_aware = False
            self.length_aware_config = LengthAwareConfig(mode=LengthAwareMode.DISABLED)

        # Bucketed EQUAL_PER_BUCKET_WEIGHT requires pre-materialized boundaries
        # (and at least one of bucket_weights/bucket_scales). Without them the
        # estimator would silently fall back to per-sequence SINGLE scaling.
        if self.length_aware and self.length_aware_config.mode is LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT:
            if self.length_aware_config.bucket_boundaries is None or (
                self.length_aware_config.bucket_weights is None
                and self.length_aware_config.bucket_scales is None
            ):
                log.warn(
                    "GPTQ module '%s' uses LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT without "
                    "materialized bucket_boundaries/bucket_weights; disabling length-aware normalization.",
                    self.name,
                )
                self.length_aware = False
                self.length_aware_config = LengthAwareConfig(mode=LengthAwareMode.DISABLED)
        self._validate_act_group_aware_shape()

        self.module_copy = None

        self.nsamples = 0
        # Tracks the largest sample count ever observed for this module. Any
        # later code path (especially Hessian materialization) that lowers
        # ``self.nsamples`` below this value is a regression and must fail.
        self._max_observed_nsamples = 0

        self.quantizer = self.create_quantizer(name=self.name)

        # fwd counter
        self.fwd_counter = 0

        self.fallback = self.qcfg.fallback
        self.expected_nsamples: Optional[float] = None

        # For non-Embedding modules: dense Hessian (columns x columns)
        self.H: Optional[torch.Tensor] = None
        # For Embedding modules: diagonal Hessian stored as 1D vector (length = vocab_size)
        self._H_diag: Optional[torch.Tensor] = None

        # Store per-device Hessian contributions so multi-GPU calibration can
        # keep local accumulators and merge only once when quantization begins.
        # For non-Embedding modules: map device -> 2D partial (columns x columns)
        self._device_hessian_partials: Dict[torch.device, torch.Tensor] = {}
        # For Embedding modules: map device -> 1D token frequency vector (length = vocab)
        self._device_embedding_counts: Dict[torch.device, torch.Tensor] = {}
        self._device_sample_counts: Dict[torch.device, int] = {}
        self._device_sequence_counts: Dict[torch.device, int] = {}
        self._hessian_dirty: bool = False
        self._hessian_rebuild_invalid: bool = False

        # GPTQ same-input Hessian sharing:
        # These fields are attached by GPTQProcessor.prepare_subset() when
        # several compatible modules in the same subset consume the same input
        # activation. The task still owns quantization and loss computation, but
        # Hessian accumulation/materialization and inverse/Cholesky can be
        # shared across q/k/v or gate/up groups. They stay None for GPTAQ, FOEM,
        # embeddings, and modules without a same-input peer.
        # ``_shared_hessian_source`` records the exact borrowed tensor. Identity,
        # rather than processor-wide mutable flags, decides whether in-place
        # damping is safe when quantization workers run concurrently.
        self._hessian_is_shared = False
        self._shared_hessian_source = None
        self._shared_hessian_inverse_cache = None
        self._shared_hessian_inverse_lock = None
        self._shared_hessian_inverse_key = None
        self._shared_hessian_inverse_ref_counts = None
        self._shared_hessian_state = None
        self._shared_hessian_stats = None

        self._borrow_workspace_stats = {
            "requests": 0,
            "staging_requests": 0,
            "staging_hits": 0,
            "staging_misses": 0,
            "materialized_requests": 0,
            "materialized_hits": 0,
            "materialized_misses": 0,
        }
        self._borrow_workspace_totals = {
            "requests": 0,
            "materialized_hits": 0,
            "materialized_misses": 0,
            "staging_hits": 0,
            "staging_misses": 0,
        }
        self._borrow_workspace_last_summary: Optional[Dict[str, object]] = None
        self._borrow_workspace_stage_dtype: Optional[torch.dtype] = None
        self._borrow_workspace_last_chunk_rows: Optional[int] = None

    def _set_nsamples(self, nsamples: int) -> None:
        """Set ``self.nsamples`` and assert it never drops below the observed maximum."""

        if not isinstance(nsamples, int):
            nsamples = int(nsamples)
        self.nsamples = nsamples
        assert self.nsamples >= self._max_observed_nsamples, (
            f"Module `{getattr(self, 'name', '<unknown>')}` sample count regressed from "
            f"{self._max_observed_nsamples} to {self.nsamples}."
        )
        if self.nsamples > self._max_observed_nsamples:
            self._max_observed_nsamples = self.nsamples

    def _validate_act_group_aware_shape(self) -> None:
        if not getattr(self.qcfg, "act_group_aware", False):
            return

        # Small group sizes interact badly with activation-aware reordering:
        # the calibration Hessian is overfit when each group contains only a
        # few columns, so we disable GAR automatically unless the user explicitly
        # requested it.
        if self.qcfg._normalize_act_group_aware_for_small_groups():
            log.warn(
                f"QuantizeConfig: group_size={self.qcfg.group_size} <= 32; auto-disabling "
                f"`act_group_aware` because activation-aware reordering overfits the "
                f"calibration Hessian for small groups. Set `act_group_aware=False` "
                f"explicitly to silence this warning."
            )

        # Re-check after the safeguard in case it disabled GAR.
        if not getattr(self.qcfg, "act_group_aware", False):
            return

        group_size = int(getattr(self.qcfg, "group_size", -1) or -1)
        if group_size <= 0:
            raise ValueError(
                f"Quantization: Module `{self.name}` -> `act_group_aware=True` requires `group_size > 0`, "
                f"got `{group_size}`."
            )

    @staticmethod
    def validate_module(module):
        pass
        # assert isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d,
        #                            transformers.Conv1D)), f"We supports only linear and convolutional layers. actual = `{module}`"

    # def has_hessian_issues(self) -> bool:
    #     return any([self.issue_zero_samples, self.issue_nan_hessian, self.issue_non_invertible])

    def create_quantizer(self, name: str) -> Quantizer:
        return Quantizer(qcfg=self.qcfg, name=name, region_timer=self.region_timer)

    def shape(self):
        if hasattr(self, "module"):
            return self.module.weight.shape
        else:
            return (0, 0)

    def mock_hessian_inverse(self, H: torch.Tensor, release_input: bool = False):
        """Mock hessian inverse for fast testing."""
        if H is None:
            H = self.H
        # Use the anchor/base damping, not the clamp floor, so adaptive and
        # static configs behave consistently.
        damp = getattr(self.qcfg.damp, "base_percdamp", self.qcfg.damp.min)
        # Return identity matrix instead of complex inversion
        identity = torch.eye(H.shape[0], dtype=torch.float32, device=H.device)
        if release_input and getattr(self, "H", None) is H:
            self.H = None
            del H
        return identity, damp

    def log_cpu_fallback(self, stage: str, source_device: torch.device) -> None:
        """Explain when a memory-heavy GPTQ step moves from CUDA to CPU."""

        log.warn(
            "Quantization: Module `%s` -> CUDA OOM during %s on %s; falling back to CPU. "
            "Due to this fallback, the calculation may take much longer than normal.",
            self.name,
            stage,
            source_device,
        )

    def clone_module(self, copy=True, device: torch.device = None):
        if not device:
            device = self.module.weight.data.device

        clone = self.module.weight.data.to(copy=copy, device=device)

        if isinstance(self.module, nn.Embedding):
            # Embedding weight is [V, D] -> we operate on [D, V]
            clone = clone.t()

        if isinstance(self.module, _ConvNd):
            clone = clone.flatten(1)

        if isinstance(self.module, transformers.pytorch_utils.Conv1D):
            clone = clone.t()

        if self._tp_pad_cols:
            pad = torch.zeros(
                (clone.shape[0], self._tp_pad_cols),
                dtype=clone.dtype,
                device=clone.device,
            )
            clone = torch.cat((clone, pad), dim=1)

        return clone.float()

    @staticmethod
    def truncate_last_dim(tensor: torch.Tensor, length: int) -> torch.Tensor:
        if tensor.dim() == 0:
            return tensor

        trim = min(length, tensor.shape[-1])
        if trim == tensor.shape[-1]:
            return tensor

        return tensor.narrow(tensor.dim() - 1, 0, trim).contiguous()

    @staticmethod
    def build_group_index(
        columns: int,
        group_size: int,
        device: torch.device,
        *,
        source_perm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build GPTQ group ids directly as a tensor, avoiding large Python lists."""

        if source_perm is not None:
            g_idx = source_perm.to(device=device, dtype=torch.int32)
            g_idx.div_(group_size, rounding_mode="floor")
            return g_idx

        g_idx = torch.arange(columns, device=device, dtype=torch.int32)
        g_idx.div_(group_size, rounding_mode="floor")
        return g_idx

    def _reshape_input(self, inp: torch.Tensor) -> Tuple[int, torch.Tensor, torch.device]:
        inp_device = get_device(inp)

        if isinstance(self.module, (nn.Linear, transformers.Conv1D)):
            reshaped_inp = inp.reshape(-1, inp.shape[-1])
        else:
            if isinstance(self.module, nn.Conv1d):
                reshaped_inp = inp.reshape(
                    inp.size(0) * self.module.groups,
                    inp.size(1) // self.module.groups,
                    inp.shape[2],
                    1,
                )
                unfold = nn.Unfold(
                    self.module.kernel_size + (1,),
                    dilation=self.module.dilation + (1,),
                    padding=self.module.padding + (0,),
                    stride=self.module.stride + (1,),
                )
                reshaped_inp = unfold(reshaped_inp)
            else:
                reshaped_inp = inp.reshape(
                    inp.size(0) * self.module.groups,
                    inp.size(1) // self.module.groups,
                    inp.shape[2],
                    inp.shape[3],
                )
                unfold = nn.Unfold(
                    self.module.kernel_size,
                    dilation=self.module.dilation,
                    padding=self.module.padding,
                    stride=self.module.stride,
                )
                reshaped_inp = unfold(reshaped_inp)
            reshaped_inp = reshaped_inp.transpose(1, 2).flatten(0, 1)

        reshaped_inp = reshaped_inp.contiguous()
        if self._tp_pad_cols:
            pad = reshaped_inp.new_zeros((reshaped_inp.shape[0], self._tp_pad_cols))
            reshaped_inp = torch.cat((reshaped_inp, pad), dim=1)
            del pad

        canonical_device = torch.device(inp_device)
        batch_token_size = reshaped_inp.shape[0]
        return batch_token_size, reshaped_inp, canonical_device

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor, batch_index: Optional[int] = None):
        # Embedding: accumulate token counts only (1D)
        if isinstance(self.module, nn.Embedding):
            ids = inp.reshape(-1).to(torch.long)
            dev = torch.device(get_device(ids))
            counts = torch.bincount(ids, minlength=self.columns).to(torch.float32).to(dev)

            with self.lock:
                self.fwd_counter += 1
                existing = self._device_embedding_counts.get(dev)
                if existing is None:
                    self._device_embedding_counts[dev] = counts
                else:
                    existing.add_(counts)
                    del counts
                tok_n = ids.numel()
                self._device_sample_counts[dev] = self._device_sample_counts.get(dev, 0) + tok_n
                self._set_nsamples(self.nsamples + tok_n)
                self._hessian_dirty = True
            return

        # Non-Embedding: accumulate directly into a per-device partial to avoid
        # allocating a columns x columns temporary for every batch.
        sequence_count = self._sequence_count_for_input(inp)
        batch_token_size, reshaped_inp, canonical_device = self._reshape_input(inp)
        if batch_token_size == 0:
            del reshaped_inp
            return

        dev = torch.device(canonical_device)

        with self.lock:
            self.fwd_counter += 1

            existing = self._device_hessian_partials.get(dev)
            if existing is None:
                existing = torch.zeros(
                    (self.columns, self.columns),
                    dtype=torch.float32,
                    device=dev,
                )
                self._device_hessian_partials[dev] = existing

            try:
                self.compute_hessian_xtx(reshaped_inp, out=existing, sequence_count=sequence_count)
            except RuntimeError as exc:
                if (
                    dev.type == "cuda"
                    and "out of memory" in str(exc).lower()
                ):
                    log.warn(
                        "GPTQ module '%s' fell back to CPU Hessian accumulation due to GPU OOM during batch processing.",
                        getattr(self, "name", "<unknown>"),
                    )
                    reshaped_inp_cpu = reshaped_inp.to(device=torch.device("cpu"))
                    del reshaped_inp
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    cpu_dev = torch.device("cpu")
                    existing_cpu = self._device_hessian_partials.get(cpu_dev)
                    if existing_cpu is None:
                        existing_cpu = torch.zeros(
                            (self.columns, self.columns),
                            dtype=torch.float32,
                            device=cpu_dev,
                        )
                        self._device_hessian_partials[cpu_dev] = existing_cpu
                    self.compute_hessian_xtx(reshaped_inp_cpu, out=existing_cpu, sequence_count=sequence_count)
                    del reshaped_inp_cpu
                    dev = cpu_dev
                else:
                    del reshaped_inp
                    raise
            else:
                del reshaped_inp

            self._device_sample_counts[dev] = self._device_sample_counts.get(dev, 0) + batch_token_size
            if self.length_aware:
                self._device_sequence_counts[dev] = self._device_sequence_counts.get(dev, 0) + sequence_count
                self._last_batch_sequence_count = sequence_count
            self._set_nsamples(self.nsamples + batch_token_size)
            self._hessian_dirty = True

    def add_batch_from_hessian(
        self,
        batch_token_size: int,
        xtx: Optional[torch.Tensor],
        device: torch.device,
        sequence_count: int = 1,
    ) -> None:
        """Accumulate a precomputed dense Hessian contribution.

        The source `xtx` may be shared by multiple GPTQ tasks, so a new task
        accumulator never stores it by reference. Existing accumulators are
        updated in-place, preserving each module's independent lifecycle.
        """

        if batch_token_size == 0 or xtx is None:
            return

        dev = torch.device(device)

        with self.lock:
            self.fwd_counter += 1

            existing = self._device_hessian_partials.get(dev)
            if existing is None:
                self._device_hessian_partials[dev] = xtx.to(
                    device=dev,
                    dtype=torch.float32,
                    copy=True,
                ).detach()
            else:
                if xtx.device != existing.device or xtx.dtype != torch.float32:
                    existing.add_(xtx.to(device=existing.device, dtype=torch.float32))
                else:
                    existing.add_(xtx)

            self._device_sample_counts[dev] = self._device_sample_counts.get(dev, 0) + batch_token_size
            if self.length_aware:
                sequence_count = max(1, sequence_count)
                self._device_sequence_counts[dev] = self._device_sequence_counts.get(dev, 0) + sequence_count
            self._set_nsamples(self.nsamples + batch_token_size)
            self._hessian_dirty = True

    def record_shared_hessian_batch(
        self,
        batch_token_size: int,
        shared_state,
        *,
        sequence_count: int = 1,
        observation_count: Optional[int] = None,
    ) -> None:
        """Record that this task observed a batch accumulated by a shared Hessian state."""

        if batch_token_size == 0:
            return

        obs = observation_count if observation_count is not None else 1
        with self.lock:
            self.fwd_counter += obs
            self._set_nsamples(self.nsamples + batch_token_size)
            if self.length_aware:
                dev = next(iter(self._device_sample_counts), None) or torch.device("cpu")
                self._device_sequence_counts[dev] = self._device_sequence_counts.get(dev, 0) + sequence_count
            self._hessian_dirty = True
            self._shared_hessian_state = shared_state

    def preferred_staging_dtype(self, input_dtype: torch.dtype, device: torch.device) -> torch.dtype:
        device = torch.device(device)

        staging_dtype = self.qcfg.hessian.staging_dtype
        if staging_dtype == torch.float32:
            return torch.float32

        if input_dtype not in (torch.float16, torch.bfloat16):
            return torch.float32

        if staging_dtype == torch.bfloat16:
            if not _device_supports_bfloat16(device):
                return torch.float32
            return torch.bfloat16

        if staging_dtype == torch.float16:
            return torch.float16

        return torch.float32

    def resolve_hessian_chunk_size(self, rows: int, stage_dtype: torch.dtype) -> Optional[int]:
        if rows == 0:
            return None

        cfg_chunk = self.qcfg.hessian.chunk_size
        if cfg_chunk is not None:
            return max(1, min(cfg_chunk, rows))

        bytes_budget = self.qcfg.hessian.chunk_bytes
        if bytes_budget is not None:
            bytes_per_row = self.columns * stage_dtype.itemsize
            if bytes_per_row > 0:
                chunk_rows = bytes_budget // bytes_per_row
                if chunk_rows > 0:
                    return max(1, min(int(chunk_rows), rows))
            return 1

        return None

    @contextlib.contextmanager
    def borrow_materialized_chunk_fp32(
        self,
        chunk: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        if rows == 0:
            yield chunk.new_zeros((0, self.columns), dtype=torch.float32)
            return

        device = chunk.device
        stage_dtype = self.preferred_staging_dtype(chunk.dtype, device)

        stats = self._borrow_workspace_stats
        stats["requests"] += 1

        with _lease_workspace(device, stage_dtype, self.columns, rows) as (
            staging_workspace,
            staging_reused,
        ):
            stats["staging_requests"] += 1
            if staging_reused:
                stats["staging_hits"] += 1
            else:
                stats["staging_misses"] += 1

            staging_view = staging_workspace[:rows, :]
            # copy_ performs dtype conversion directly into the leased
            # workspace, avoiding a full chunk-sized converted temporary.
            staging_view.copy_(chunk)

            if stage_dtype == torch.float32:
                stats["materialized_requests"] += 1
                if staging_reused:
                    stats["materialized_hits"] += 1
                else:
                    stats["materialized_misses"] += 1

                try:
                    yield staging_view
                finally:
                    if device.type == "cuda":
                        torch.cuda.current_stream(device).synchronize()
            else:
                with _lease_workspace(
                    device,
                    torch.float32,
                    self.columns,
                    rows,
                ) as (
                    fp32_workspace,
                    fp32_reused,
                ):
                    stats["materialized_requests"] += 1
                    if fp32_reused:
                        stats["materialized_hits"] += 1
                    else:
                        stats["materialized_misses"] += 1

                    try:
                        fp32_view = fp32_workspace[:rows, :]
                        # Convert directly into the fp32 materialization
                        # workspace instead of allocating staging_view.float().
                        fp32_view.copy_(staging_view)
                        yield fp32_view
                    finally:
                        if device.type == "cuda":
                            torch.cuda.current_stream(device).synchronize()

    def compute_hessian_xtx(
        self,
        matrix: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        sequence_count: int = 1,
    ) -> torch.Tensor:
        rows = matrix.shape[0]
        if rows == 0:
            if out is not None:
                return out
            return torch.zeros((self.columns, self.columns), dtype=torch.float32, device=matrix.device)

        sequence_count = max(1, sequence_count)
        per_sequence_length = rows / sequence_count
        scale_length = per_sequence_length
        sequence_count_normalization = False
        if self.length_aware and self.length_aware_config is not None:
            cfg = self.length_aware_config
            if cfg.mode is LengthAwareMode.SEQUENCE_COUNT:
                # The GSQ author implementation accumulates raw token Gram
                # matrices and applies 2 / number_of_sequences once at
                # materialization.  It must not divide each sequence by its
                # token length first.
                sequence_count_normalization = True
            elif cfg.mode is LengthAwareMode.EQUAL_PER_BUCKET_WEIGHT and cfg.bucket_weights is not None:
                bucket_idx = self._lookup_length_bucket(per_sequence_length)
                if bucket_idx is not None:
                    scale_length = per_sequence_length * cfg.bucket_weights[bucket_idx]
            elif cfg.bucket_scales is not None:
                bucket_idx = self._lookup_length_bucket(per_sequence_length)
                if bucket_idx is not None:
                    scale_length = cfg.bucket_scales[bucket_idx]
            if not sequence_count_normalization and cfg.min_length is not None and cfg.min_length > 0:
                scale_length = max(scale_length, float(cfg.min_length))

        # CPU fallback: route to the compiled extension which calls ATen's
        # AVX-512/MKL-optimized addmm path with lower Python overhead than
        # torch.addmm_ in the GPU-OOM fallback loop.
        if matrix.device.type == "cpu":
            from ..nn_modules.qlinear.pack_block_ext import hessian_xtx_cpu

            stage_dtype = self.preferred_staging_dtype(matrix.dtype, matrix.device)
            chunk_size = self.resolve_hessian_chunk_size(rows, stage_dtype)
            self._borrow_workspace_stage_dtype = stage_dtype
            self._borrow_workspace_last_chunk_rows = chunk_size if chunk_size is not None else rows

            if sequence_count_normalization:
                length_aware_scale = 1.0
            elif self.length_aware:
                length_aware_scale = 1.0 / scale_length
            else:
                length_aware_scale = 1.0
            if chunk_size is None:
                return hessian_xtx_cpu(matrix, out, beta=1.0 if out is not None else 0.0, alpha=length_aware_scale)

            if out is None:
                xtx = torch.zeros(
                    (self.columns, self.columns),
                    dtype=torch.float32,
                    device=matrix.device,
                )
            else:
                xtx = out

            for start in range(0, rows, chunk_size):
                rows_this = min(chunk_size, rows - start)
                source = matrix[start:start + rows_this]
                with self.borrow_materialized_chunk_fp32(source, rows_this) as materialized:
                    if out is None:
                        xtx.add_(hessian_xtx_cpu(materialized, None, beta=0.0, alpha=length_aware_scale))
                    else:
                        hessian_xtx_cpu(materialized, xtx, beta=1.0, alpha=length_aware_scale)

            if not _hessian_sync_deferred():
                torch_sync(device=xtx.device)
            return xtx

        stage_dtype = self.preferred_staging_dtype(matrix.dtype, matrix.device)
        chunk_size = self.resolve_hessian_chunk_size(rows, stage_dtype)
        self._borrow_workspace_stage_dtype = stage_dtype
        self._borrow_workspace_last_chunk_rows = chunk_size if chunk_size is not None else rows

        if sequence_count_normalization:
            length_aware_scale = 1.0
        elif self.length_aware:
            length_aware_scale = 1.0 / scale_length
        else:
            length_aware_scale = 1.0
        if chunk_size is None:
            mat32 = matrix.to(dtype=torch.float32)
            if out is None:
                xtx = torch.matmul(mat32.T, mat32)
                if self.length_aware and not sequence_count_normalization:
                    xtx.div_(scale_length)
            else:
                out.addmm_(mat32.T, mat32, beta=1.0, alpha=length_aware_scale)
                xtx = out
            del mat32
            # MPS exposes a single ordered command stream.  The unchunked path
            # owns no reusable staging workspace, so returning the on-device
            # tensor asynchronously is safe and lets consecutive calibration
            # batches queue without a host barrier after every X^T X. PyTorch's
            # command buffer retains operand storage through completion, even if
            # the corresponding Python temporaries are released. Keep the existing
            # synchronization contract for multi-stream accelerators; the chunked
            # path below also retains it because its leased staging buffers may be
            # reused by the next caller.
            requires_sync = matrix.device.type != "mps" or not _USE_GPTQ_MPS_ASYNC_HESSIAN
            if requires_sync and not _hessian_sync_deferred():
                torch_sync(device=xtx.device)
            return xtx

        if out is None:
            xtx = torch.zeros((self.columns, self.columns), dtype=torch.float32, device=matrix.device)
        else:
            xtx = out

        for start in range(0, rows, chunk_size):
            rows_this = min(chunk_size, rows - start)
            source = matrix[start:start + rows_this]
            with self.borrow_materialized_chunk_fp32(source, rows_this) as materialized:
                if out is None:
                    xtx.add_(torch.matmul(materialized.T, materialized), alpha=length_aware_scale)
                else:
                    xtx.addmm_(materialized.T, materialized, beta=1.0, alpha=length_aware_scale)

        if not _hessian_sync_deferred():
            torch_sync(device=xtx.device)
        return xtx

    def process_batch(self, inp: torch.Tensor) -> Tuple[int, Optional[torch.Tensor], torch.device]:
        if isinstance(self.module, nn.Embedding):
            inp_device = get_device(inp)
            ids = inp.reshape(-1).to(torch.long)
            counts = torch.bincount(ids, minlength=self.columns).to(torch.float32)
            return ids.numel(), counts, torch.device(inp_device)

        sequence_count = self._sequence_count_for_input(inp)
        batch_token_size, reshaped_inp, canonical_device = self._reshape_input(inp)
        if batch_token_size == 0:
            del reshaped_inp
            return 0, None, canonical_device

        try:
            xtx = self.compute_hessian_xtx(reshaped_inp, sequence_count=sequence_count)
        except RuntimeError as exc:
            if (
                canonical_device.type == "cuda"
                and "out of memory" in str(exc).lower()
            ):
                log.warn(
                    "GPTQ module '%s' fell back to CPU Hessian accumulation due to GPU OOM during batch processing.",
                    getattr(self, "name", "<unknown>"),
                )
                reshaped_inp_cpu = reshaped_inp.to(device=torch.device("cpu"))
                del reshaped_inp
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                canonical_device = torch.device("cpu")
                xtx = self.compute_hessian_xtx(reshaped_inp_cpu, sequence_count=sequence_count)
                del reshaped_inp_cpu
            else:
                del reshaped_inp
                raise
        else:
            del reshaped_inp

        self._last_batch_sequence_count = sequence_count
        self._snapshot_borrow_workspace_stats(context="process_batch")
        return batch_token_size, xtx, canonical_device

    def _select_hessian_target_device(self, requested: Optional[torch.device]) -> torch.device:
        with self.lock:
            if requested is not None:
                return torch.device(requested)

            hint = getattr(self, "_final_hessian_device_hint", None)
            if hint is not None:
                return torch.device(hint)

            # Prefer a device that already has partials
            if self._device_hessian_partials:
                partial_device = next(iter(self._device_hessian_partials.keys()))
                return torch.device(partial_device)
            if self._device_embedding_counts:
                partial_device = next(iter(self._device_embedding_counts.keys()))
                return torch.device(partial_device)

            return torch.device("cpu")

    def materialize_global_hessian(self, target_device: Optional[torch.device] = None) -> None:
        with self.lock:
            # Select the destination under the same lock as partial-state reads;
            # this closes the GIL=0 window between selection and merge.
            device = self._select_hessian_target_device(target_device)

            shared_state = getattr(self, "_shared_hessian_state", None)
            if shared_state is not None and not isinstance(self.module, nn.Embedding):
                self.materialize_shared_hessian(shared_state, device)
                return

            # Embedding path: merge 1D counts
            if isinstance(self.module, nn.Embedding):
                # The diagonal Hessian is materialized once and consumed by the
                # quantize() path; do not rebuild it from already-cleared partials
                # on later calls such as a mock-quantization recursion.
                if not self._hessian_dirty and self._H_diag is not None:
                    if self._H_diag.device != device:
                        self._H_diag = self._H_diag.to(device=device)
                    self.H = None
                    self._set_nsamples(getattr(self, "_hessian_total_samples", 0))
                    self._hessian_dirty = False
                    self._final_hessian_device_hint = device
                    return

                total_old = getattr(self, "_hessian_total_samples", 0)
                total_new = sum(
                    int(counts.sum().item())
                    for counts in self._device_embedding_counts.values()
                )
                total = total_old + total_new

                # Reuse buffer if possible; a reused buffer already contains the
                # previously merged and scaled diagonal, so do not zero it unless
                # there is no prior data.
                if (
                    self._H_diag is not None
                    and self._H_diag.shape == (self.columns,)
                ):
                    if self._H_diag.device != device or self._H_diag.dtype != torch.float32:
                        self._H_diag = self._H_diag.to(device=device, dtype=torch.float32)
                    diag = self._H_diag
                else:
                    torch_sync(device)
                    diag = torch.zeros(self.columns, dtype=torch.float32, device=device)
                    total_old = 0

                if total_new > 0:
                    if total > 0:
                        scale_new = 2.0 / float(total)
                        if total_old > 0:
                            diag.mul_(float(total_old) / float(total))
                    else:
                        scale_new = 1.0

                    for partial_device in sorted(self._device_embedding_counts, key=_device_reduction_key):
                        counts = self._device_embedding_counts[partial_device]
                        counts = counts.to(device=device, dtype=torch.float32)
                        diag.add_(counts, alpha=scale_new)

                # Apply a tiny floor to avoid zeros for unseen tokens (stabilizes inverse)
                abs_max = max(diag.max().item(), 1.0)
                floor = abs_max * 1e-6
                torch.maximum(diag, torch.tensor(floor, dtype=diag.dtype, device=diag.device), out=diag)

                self._H_diag = diag
                self.H = None  # No dense matrix for Embedding
                self._set_nsamples(total)
                self._hessian_total_samples = total
                self._hessian_dirty = False
                self._final_hessian_device_hint = device
                # Keep per-device sample counts so a mock-quantization recursion (after
                # ``self.H`` is released for peak memory) can still report the correct
                # number of observed calibration tokens. The merged counts are freed.
                self._device_embedding_counts.clear()
                return

            # Non-Embedding path: original dense merge
            if not self._hessian_dirty and self.H is not None:
                if self.H.device != device:
                    self.H = self.H.to(device=device)
                return

            total_tokens = sum(self._device_sample_counts.values())

            # If the Hessian partials have already been merged and freed (e.g. the
            # mock-quantization recursion path after ``self.H`` was released for peak
            # memory), rebuilding a dense Hessian here would create an all-zero matrix.
            # That makes every column look dead and zeroes the whole weight. The
            # per-device sample counts are intentionally preserved so the correct
            # number of calibration tokens is still known; route straight to fallback.
            if total_tokens > 0 and not self._device_hessian_partials and self.H is None:
                self._hessian_rebuild_invalid = True
                self._set_nsamples(total_tokens)
                return

            # Track how many samples the existing ``self.H`` already represents so
            # a second materialization (new batches after ``self.H`` was already
            # built) re-weights the existing Hessian instead of zeroing it.
            # ``total_samples`` is cumulative; ``_hessian_total_samples`` is the
            # count already merged into ``self.H``.
            if self.length_aware:
                total_old = getattr(self, "_hessian_total_sequences", 0) if self.H is not None else 0
                total = sum(self._device_sequence_counts.values())
            else:
                total_old = getattr(self, "_hessian_total_samples", 0) if self.H is not None else 0
                total = total_tokens

            # Reuse the existing tensor when possible to avoid an extra allocation.
            if self.H is not None and self.H.shape == (self.columns, self.columns):
                if self.H.device != device or self.H.dtype != torch.float32:
                    self.H = self.H.to(device=device, dtype=torch.float32)
                result_accum = self.H
            else:
                torch_sync(device)  # try to avoid torch.AcceleratorError: CUDA error: unspecified launch failure
                result_accum = torch.zeros(
                    (self.columns, self.columns),
                    dtype=torch.float32,
                    device=device,
                )

            if total_tokens == 0:
                if self._max_observed_nsamples > 0:
                    self._hessian_rebuild_invalid = True
                self.H = result_accum
                self._set_nsamples(self._max_observed_nsamples)
                self._hessian_dirty = False
                self._final_hessian_device_hint = device
                self._device_hessian_partials.clear()
                return

            # Merge per-device partials into the final Hessian and delete each
            # partial as it is added so the peak stays at result + one partial.
            if total > 0:
                if total_old > 0:
                    result_accum.mul_(float(total_old) / float(total))
                scale_new = 2.0 / float(total)
            else:
                scale_new = 1.0

            # GPU workers can finish in a different order after restart. A
            # stable reduction order avoids changing floating-point rounding.
            for partial_device in sorted(list(self._device_hessian_partials), key=_device_reduction_key):
                partial = self._device_hessian_partials.pop(partial_device)
                if partial.device != result_accum.device or partial.dtype != torch.float32:
                    try:
                        partial = partial.to(device=result_accum.device, dtype=torch.float32)
                    except Exception:
                        log.warn(f"Quantization: Module `{self.name}` -> Retry partial.to 1/2 in 0.25s")
                        time.sleep(0.25)
                        try:
                            partial = partial.to(device=result_accum.device, dtype=torch.float32)
                        except Exception:
                            log.warn(f"Quantization: Module `{self.name}` -> Retry partial.to 2/2 in 0.75s")
                            time.sleep(0.75)
                            partial = partial.to(device=result_accum.device, dtype=torch.float32)
                result_accum.add_(partial, alpha=scale_new)
                del partial

            self.H = result_accum
            self._hessian_total_samples = total_tokens
            if self.length_aware:
                self._hessian_total_sequences = total
            self._set_nsamples(total_tokens)
            self._hessian_dirty = False
            self._final_hessian_device_hint = result_accum.device
            # Keep per-device sample counts so a mock-quantization recursion (after
            # ``self.H`` is released for peak memory) can still report the correct
            # number of observed calibration tokens. The Hessian partials have
            # already been merged into ``self.H`` and are freed above.
            del result_accum

    def adopt_hessian_from(self, leader: "GPTQ") -> None:
        """Replace this task's Hessian statistics with a private copy of ``leader``'s."""
        if leader is self:
            return
        if leader.columns != self.columns:
            raise ValueError(
                f"GPTQ: cannot share Hessian from `{leader.name}` ({leader.columns} columns) "
                f"with `{self.name}` ({self.columns} columns)."
            )

        leader.materialize_global_hessian()
        with leader.lock:
            source = leader.H
            nsamples = leader.nsamples
            fwd_counter = leader.fwd_counter
            if source is None:
                source = leader.create_H(None)
            target_device = self._select_hessian_target_device(
                getattr(self.module, "target_device", None)
            )
            copied = source.detach().to(
                device=target_device, dtype=torch.float32, copy=True
            )

        with self.lock:
            self.H = copied
            self.nsamples = nsamples
            self._max_observed_nsamples = max(self._max_observed_nsamples, nsamples)
            self.fwd_counter = fwd_counter
            self._device_hessian_partials.clear()
            self._device_sample_counts.clear()
            self._device_sequence_counts.clear()
            self._hessian_dirty = False
            self._hessian_rebuild_invalid = False
            self._final_hessian_device_hint = copied.device
            # Adoption is an independent task-local result. Do not let the
            # existing QVQ shared-state cache repin or overwrite it later.
            self._shared_hessian_state = None
            self._shared_hessian_source = None

    def materialize_shared_hessian(self, shared_state, device: torch.device) -> None:
        """Materialize this task's Hessian from a same-input shared state.

        Multiple GPTQ tasks may point at the same final `H` tensor after this
        method. The tensor is read-only from the task's perspective once the
        shared state is clean, so sharing it avoids duplicate materialization.
        """

        with shared_state["lock"]:
            hessian = shared_state.get("H")
            hessian_event = shared_state.get("H_event")
            if (
                hessian is not None
                and not shared_state.get("dirty", False)
                and shared_state.get("total_samples", 0) > 0
            ):
                if hessian_event is not None:
                    if device.type == "cuda":
                        torch.cuda.current_stream(device).wait_event(hessian_event)
                    else:
                        hessian_event.synchronize()
                # The shared Hessian is read-only once clean.  Give this task a
                # private per-device copy rather than moving the canonical tensor,
                # so other workers that already reference the canonical storage
                # are not affected by device repinning under free-threading.
                if hessian.device != device:
                    task_hessian = hessian.to(device=device)
                    # Private per-device copy: in-place damping is safe, so do not
                    # mark it as shared-canonical to avoid redundant clones.
                    shared_source = None
                else:
                    task_hessian = hessian
                    shared_source = hessian
                self.H = task_hessian
                self._shared_hessian_source = shared_source
                # ``record_shared_hessian_batch`` is the sole authority for this
                # module's observed sample count. In particular, zero means the
                # module was inactive and must remain eligible for RTN fallback.
                self._hessian_dirty = False
                self._final_hessian_device_hint = task_hessian.device
                return

            partials = shared_state["partials"]
            partial_events = shared_state.setdefault("partial_events", {})
            sample_counts = shared_state["sample_counts"]
            sequence_counts = shared_state.get("sequence_counts", {})
            total_old_tokens = int(shared_state.get("total_samples", 0) or 0)
            total_new_tokens = sum(sample_counts.values())
            total_tokens = total_old_tokens + total_new_tokens if total_old_tokens > 0 else total_new_tokens

            # If this is a re-accumulation after a previous materialization, the
            # existing Hessian was scaled by 2/total_old and must be re-weighted
            # so the new partials can be merged without losing earlier batches.
            if self.length_aware:
                total_old = int(shared_state.get("total_sequences", 0) or 0)
                total_new = sum(sequence_counts.values())
            else:
                total_old = total_old_tokens
                total_new = total_new_tokens
            total = total_old + total_new if total_old > 0 else total_new

            # The shared Hessian was freed but the sample count is still recorded;
            # rebuilding from an empty partial set would produce an all-zero H.
            if hessian is None and total_old_tokens > 0 and not partials:
                self._hessian_rebuild_invalid = True
                return

            if (
                hessian is not None
                and hessian.shape == (self.columns, self.columns)
            ):
                if hessian_event is not None:
                    if device.type == "cuda":
                        torch.cuda.current_stream(device).wait_event(hessian_event)
                    else:
                        hessian_event.synchronize()
                # Reuse the existing buffer only when it already matches the
                # target device/dtype.  Otherwise materialize into a private
                # copy and let the final assignment replace the canonical tensor.
                if hessian.device != device or hessian.dtype != torch.float32:
                    result_accum = hessian.to(device=device, dtype=torch.float32)
                else:
                    result_accum = hessian
            else:
                torch_sync(device)
                result_accum = torch.zeros(
                    (self.columns, self.columns),
                    dtype=torch.float32,
                    device=device,
                )

            # If we are reusing a previously allocated Hessian buffer but no prior
            # scaled contributions are recorded, zero it to avoid double-counting.
            if total_old == 0:
                result_accum.zero_()

            if total_new:
                # Merge shared partials directly into result_accum. The existing
                # Hessian (if any) was scaled by 2/total_old, so re-weight it by
                # total_old/total and add each new partial scaled by 2/total.
                # This avoids allocating a second full columns x columns buffer.
                if total > 0:
                    scale_new = 2.0 / float(total)
                    if total_old > 0:
                        result_accum.mul_(float(total_old) / float(total))
                else:
                    scale_new = 1.0

                while partials:
                    partial_device, partial = partials.popitem()
                    partial_event = partial_events.pop(partial_device, None)
                    if partial_event is not None:
                        if result_accum.device.type == "cuda":
                            torch.cuda.current_stream(result_accum.device).wait_event(partial_event)
                        else:
                            partial_event.synchronize()
                    if (
                        partial.device != result_accum.device
                        or partial.dtype != torch.float32
                    ):
                        partial = partial.to(
                            device=result_accum.device, dtype=torch.float32
                        )
                    result_accum.add_(partial, alpha=scale_new)
                    del partial

            shared_state["H"] = result_accum
            shared_state["dirty"] = False
            shared_state["total_samples"] = total_tokens
            if self.length_aware:
                shared_state["total_sequences"] = total
            sample_counts.clear()
            partial_events.clear()
            if self.length_aware:
                sequence_counts.clear()

            self.H = result_accum
            self._shared_hessian_source = result_accum
            # Keep the module-local sample count, including an authoritative zero.
            self._hessian_dirty = False
            self._final_hessian_device_hint = result_accum.device
            if result_accum.device.type == "cuda":
                completion = torch.cuda.Event(enable_timing=False, blocking=False)
                completion.record(torch.cuda.current_stream(result_accum.device))
                shared_state["H_event"] = completion
            else:
                shared_state["H_event"] = None

    def finalize_hessian(self, target_device: Optional[torch.device] = None) -> torch.Tensor:
        self.materialize_global_hessian(target_device=target_device)
        if isinstance(self.module, nn.Embedding):
            # For Embedding, the Hessian is diagonal-only (stored in self._H_diag).
            # Keep self.H as None to avoid accidental dense use.
            return torch.tensor([])  # unused by Embedding quantize path
        if self.H is None:
            self.H = self.create_H(target_device)
        return self.H

    def create_H(self, target_device):
        return torch.zeros((self.columns, self.columns), dtype=torch.float32,
                           device=self._select_hessian_target_device(target_device))

    def _fallback_quantize(
        self,
        strategy: FallbackStrategy,
        blocksize: int,
        target_device: Optional[torch.device] = None,
    ):
        """Apply a lightweight quantization fallback using the requested strategy."""
        maxq = 2 ** self.qcfg.bits - 1
        sigma = 3.0
        effective_group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns
        start_time = time.time()
        smooth_method = getattr(self.fallback, "smooth", None)
        mse_steps = 32
        mse_maxshrink = 0.8
        if isinstance(smooth_method, SmoothMSE):
            mse_steps = smooth_method.steps
            mse_maxshrink = smooth_method.maxshrink

        if target_device is None:
            target_device = self.H.device if self.H is not None else self.module.weight.device
        W = self.clone_module(device=target_device)
        Q = torch.empty_like(W)
        scale_chunks = []
        zero_chunks = []

        for start in range(0, self.columns, effective_group_size):
            end = min(start + effective_group_size, self.columns)
            block = W[:, start:end]

            if isinstance(smooth_method, SmoothMSE):
                dequant, scale, zero = mse_optimal_quant(
                    block,
                    self.qcfg,
                    maxq,
                    steps=mse_steps,
                    maxshrink=mse_maxshrink,
                )
            else:
                block_mod, scale_factor = smooth_block(
                    block,
                    self.fallback,
                    group_size=effective_group_size,
                )
                if strategy == FallbackStrategy.MIDPOINT:
                    w_min = block_mod.min(dim=1, keepdim=True).values
                    w_max = block_mod.max(dim=1, keepdim=True).values
                    mid = (w_max + w_min) / 2.0
                    scale = torch.clamp((w_max - w_min) / maxq, min=1e-8)
                    zero_mid = torch.full_like(scale, maxq / 2.0)
                    q = torch.round((block_mod - mid) / scale + zero_mid)
                    q = torch.clamp(q, 0, maxq)
                    zero = torch.round(zero_mid - (mid / scale))
                    zero = torch.clamp(zero, 0, maxq)
                    dequant = (q - zero) * scale
                elif strategy == FallbackStrategy.MEAN:
                    mean = block_mod.mean(dim=1, keepdim=True)
                    max_dev = torch.max((block_mod - mean).abs(), dim=1, keepdim=True).values
                    max_dev = torch.clamp(max_dev, min=1e-8)
                    scale = (2 * max_dev) / maxq
                    zero_mid = torch.full_like(scale, maxq / 2.0)
                    q = torch.round((block_mod - mean) / scale + zero_mid)
                    q = torch.clamp(q, 0, maxq)
                    zero = torch.round(zero_mid - (mean / scale))
                    zero = torch.clamp(zero, 0, maxq)
                    dequant = (q - zero) * scale
                elif strategy == FallbackStrategy.MEDIAN:
                    median = block_mod.median(dim=1, keepdim=True).values
                    max_dev = torch.max((block_mod - median).abs(), dim=1, keepdim=True).values
                    max_dev = torch.clamp(max_dev, min=1e-8)
                    scale = (2 * max_dev) / maxq
                    zero_mid = torch.full_like(scale, maxq / 2.0)
                    q = torch.round((block_mod - median) / scale + zero_mid)
                    q = torch.clamp(q, 0, maxq)
                    zero = torch.round(zero_mid - (median / scale))
                    zero = torch.clamp(zero, 0, maxq)
                    dequant = (q - zero) * scale
                elif strategy == FallbackStrategy.STDCLIP:
                    mean = block_mod.mean(dim=1, keepdim=True)
                    std = block_mod.std(dim=1, keepdim=True, unbiased=False)
                    std = torch.clamp(std, min=1e-8)
                    lo = mean - sigma * std
                    hi = mean + sigma * std
                    scale = torch.clamp((hi - lo) / maxq, min=1e-8)
                    zero = torch.round(-lo / scale)
                    zero = torch.clamp(zero, 0, maxq)
                    q = torch.round(block_mod / scale + zero)
                    q = torch.clamp(q, 0, maxq)
                    dequant = (q - zero) * scale
                elif strategy == FallbackStrategy.RTN:
                    self.quantizer.find_params(block_mod, weight=True)
                    dequant = self.quantizer.quantize(block_mod)
                    scale = self.quantizer.scale
                    zero = self.quantizer.zero
                else:
                    raise ValueError(f"Unsupported fallback strategy: {strategy}")

                if scale_factor is not None:
                    scale = scale * scale_factor
                    dequant = dequant * scale_factor

            Q[:, start:end] = dequant

            scale_block = scale if scale.dim() > 1 else scale.unsqueeze(1)
            zero_block = zero if zero.dim() > 1 else zero.unsqueeze(1)
            if scale_block.shape[1] > 1:
                scale_block = scale_block.mean(dim=1, keepdim=True)
            if zero_block.shape[1] > 1:
                zero_block = zero_block.mean(dim=1, keepdim=True)
            scale_chunks.append(scale_block)
            zero_chunks.append(zero_block)

        scale = torch.cat(scale_chunks, dim=1)
        zero = torch.cat(zero_chunks, dim=1)

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            Q = Q[:, :valid_cols]
            scale = self.truncate_last_dim(scale, valid_cols)
            zero = self.truncate_last_dim(zero, valid_cols)
        else:
            valid_cols = self.columns

        group_size = effective_group_size if effective_group_size != -1 else self.columns
        g_idx = self.build_group_index(valid_cols, group_size, Q.device)

        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()

        if Q.shape != self.module.weight.shape:
            Q = Q.reshape(self.module.weight.shape).to(self.module.weight.dtype)
        else:
            Q = Q.to(self.module.weight.dtype)

        Q = Q.to(device=self.module.weight.data.device, non_blocking=False)
        mean_abs_err = (Q - self.module.weight.data).abs().mean().item()
        duration = time.time() - start_time
        avg_loss = f"fallback({strategy.value}): {mean_abs_err:.7f}"
        damp = 0.0

        self.H = None
        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    # FIXME, optimum needs fasterquant, we need to remove it
    def fasterquant(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
    ):
        return self.hf_quantize(blocksize, percdamp, damp_auto_increment, group_size, actorder, static_groups)

    # public api exposed to hf
    def hf_quantize(
            self,
            blocksize=128,
            percdamp=0.01,
            damp_auto_increment=0.0015,
            group_size=-1,
            actorder=False,
            static_groups=False,
            act_group_aware: Optional[bool] = None,
    ):
        self.qcfg.group_size = group_size
        self.qcfg.damp_percent = percdamp
        self.qcfg.damp_auto_increment = damp_auto_increment
        self.qcfg.adaptive_damping = DampConfig(min=percdamp, max=percdamp, step=damp_auto_increment)
        self.qcfg.desc_act = actorder
        if act_group_aware is not None:
            self.qcfg.act_group_aware = act_group_aware
        self.qcfg._resolve_activation_ordering(actorder, act_group_aware)
        self.qcfg.static_groups = static_groups
        (Q, scale, zero, g_idx, duration, avg_loss, damp_percent, nsamples) = self.quantize(blocksize=blocksize)
        self.module.weight.data = Q
        return scale, zero, g_idx, duration, avg_loss, damp_percent

    def _shared_hessian_inverse_cache_key(
        self,
        H: torch.Tensor,
        damp: Optional[float] = None,
    ):
        """Return an inverse-cache key when this task uses a shared Hessian tensor.

        The resolved damping value is included in the key so modules that share
        the same Hessian but resolve to different adaptive damping factors do not
        reuse each other's inverses.
        """

        key = self._shared_hessian_inverse_key
        if key is None:
            return None
        cache_key: Tuple[object, ...] = (
            key,
            str(torch.device(H.device)),
            str(H.dtype),
            tuple(H.shape),
        )
        if damp is not None:
            cache_key = cache_key + (float(damp),)
        return cache_key

    def _release_shared_hessian_inverse_cache_entry(self, cache, cache_key) -> None:
        """Drop a shared inverse cache entry after its last group consumer.

        Reference counts are tracked per shared-Hessian group (``base_key``), while
        cache entries are keyed by the full cache key including resolved damping.
        This lets different consumers of the same Hessian cache distinct damped
        inverses while still releasing the group refcount correctly.
        """

        ref_counts = self._shared_hessian_inverse_ref_counts
        if cache is None or cache_key is None or ref_counts is None:
            return

        base_key = self._shared_hessian_inverse_key
        if base_key is None:
            return

        remaining = ref_counts.get(base_key)
        if remaining is None:
            return

        remaining -= 1
        if remaining <= 0:
            ref_counts.pop(base_key, None)
            # Drop all cache entries keyed from this shared-Hessian group, not
            # just the one released by the final consumer, in case distinct
            # damping keys were populated by different group members.
            for k in list(cache.keys()):
                if k[0] == base_key:
                    cache.pop(k, None)
        else:
            ref_counts[base_key] = remaining

    @staticmethod
    def _stable_probe_seed(dim: int) -> int:
        """Return a deterministic seed derived only from the probe dimension.

        The probe is part of the Hessian spectral approximation, so module names
        and cache ownership must not change its value for identical mathematical
        inputs. The dimension is sufficient to create a reproducible local RNG
        stream without consuming global RNG state.
        """

        return (0x9E3779B9 ^ int(dim)) & 0xFFFFFFFF

    @classmethod
    def _stable_probe_tensor(
        cls,
        shape: Tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Draw a deterministic local probe on CPU, then move it to the Hessian device.

        MPS does not support constructing a device-local ``torch.Generator``.
        A CPU generator also gives every backend the same probe values without
        consuming the process-global RNG stream.
        """

        seed = cls._stable_probe_seed(shape[0])
        gen = torch.Generator(device="cpu").manual_seed(seed)
        return torch.randn(shape, generator=gen, dtype=dtype, device="cpu").to(device=device)

    @staticmethod
    def _lambda_max_kernel(H: torch.Tensor, v0: torch.Tensor, n_iter: int):
        """Pure-tensor power iteration for the largest eigenvalue only."""

        v = v0 / torch.linalg.norm(v0)
        for _ in range(n_iter):
            v = H @ v
            v = v / torch.linalg.norm(v)
        return v @ (H @ v)

    @classmethod
    def _compiled_lambda_max_kernel(cls):
        if getattr(cls, "_lambda_max_kernel_compiled", None) is None:
            # Python 3.14 free-threading + torch < 2.14 crashes inside dynamo
            # for the power-iteration loop; keep this kernel eager until a
            # stable no-GIL compile is verified.
            if has_gil_disabled() and gte_python_3_14() and not TORCH_GTE_214:
                cls._lambda_max_kernel_compiled = cls._lambda_max_kernel
            else:
                cls._lambda_max_kernel_compiled = torch_compile(
                    cls._lambda_max_kernel,
                    backend="inductor",
                    fullgraph=False,
                )
        return cls._lambda_max_kernel_compiled

    @classmethod
    def _lanczos_eigen_max(cls, H: torch.Tensor, n_iter: int):
        """Fast Lanczos estimate of the largest eigenvalue of ``H`` via ``torch.lobpcg``."""

        A = H.contiguous()
        if A.shape[0] == 1:
            return A[0, 0]
        if A.shape[0] == 2:
            # torch.lobpcg requires rows >= 3 * k. Use the closed-form
            # eigenvalue of a symmetric 2x2 matrix for tiny GPTQ layers.
            a, c = A[0, 0], A[1, 1]
            b = (A[0, 1] + A[1, 0]) * 0.5
            return a * 0.5 + c * 0.5 + torch.hypot(a - c, 2 * b) * 0.5

        X = cls._stable_probe_tensor((A.shape[0], 1), dtype=A.dtype, device=A.device)
        largest_vals, _ = torch.lobpcg(A, k=1, X=X, largest=True, niter=n_iter)
        return largest_vals[0]

    @torch.inference_mode()
    def _estimate_hessian_eigen_spectrum(
        self,
        H: torch.Tensor,
        method: str,
        n_iter: int,
        effective_diag: Optional[torch.Tensor] = None,
    ):
        """Estimate the largest eigenvalue of ``H``.

        Only ``lambda_max`` is returned; ``lambda_min`` is treated as 0 in the
        damping formula, giving the safe upper bound ``lambda >= lambda_max / (K-1)``.
        Probe vectors are drawn from a dimension-seeded local CPU generator and
        moved to the Hessian device, so estimates depend only on mathematical inputs,
        work on backends without device-local generators, and do not perturb the
        global RNG stream. The power iteration loop is compiled when ``torch.compile``
        is available to reduce kernel-launch overhead.

        For ``method="diagonal"``, ``effective_diag`` (the diagonal of the matrix
        actually being factorized, including any applied floor) is used as a fast
        lower-bound proxy for ``lambda_max``. This is cheap but can under-damp
        highly ill-conditioned layers because ``max(diag(H))`` is only a lower bound
        on the true largest eigenvalue.
        """

        d = H.shape[0]
        if d <= 1:
            return None, None

        if method == "diagonal":
            diag = effective_diag if effective_diag is not None else H.diagonal()
            return diag.max().item(), 0.0

        if method == "lanczos":
            if n_iter <= 0:
                return None, None
            lambda_max_t = self._lanczos_eigen_max(H, n_iter)
            return lambda_max_t.item(), 0.0

        # method == "power_iteration" (default)
        if n_iter <= 0:
            return None, None

        v0 = self._stable_probe_tensor((d,), dtype=H.dtype, device=H.device)
        lambda_max_t = self._compiled_lambda_max_kernel()(H, v0, n_iter)
        return lambda_max_t.item(), 0.0

    def _module_damp_factor(self, damp_cfg: AdaptiveDampingConfig) -> float:
        """Return the per-module scaling factor from the model's module-tree flags.

        The first matching role flag in ``module_factors`` is used (e.g. ``q``,
        ``k``, ``v``, ``o``, ``gate``, ``up``, ``down``). If no module-tree
        flags are available, fall back to the raw module-name suffix.
        """

        if not damp_cfg.module_prior_enabled:
            return 1.0

        known = set(damp_cfg.module_factors.keys())
        flags = frozenset()
        if self._named_module is not None:
            flags = self._named_module.state.get("module_tree_flags", frozenset())
        matched = known & flags
        if matched:
            return damp_cfg.module_factors[sorted(matched)[0]]

        short_name = self.name.split(".")[-1] if self.name else ""
        return damp_cfg.module_factors.get(short_name, 1.0)

    def _resolve_initial_damp(
        self,
        H: torch.Tensor,
        current_diag: torch.Tensor,
        mean: torch.Tensor,
        lambda_spectral: Optional[float] = None,
    ):
        """Choose a calibration-Hessian-aware GPTQ damping fraction.

        ``current_diag`` is the diagonal of the effective Hessian (including any
        applied positive-definiteness floor) and ``mean`` is its mean. For the
        default configuration the damping is determined entirely by the
        activation Hessian ``H = (2 / N) X.T @ X``:

            r = lambda_spectral / mean(current_diag)
            damp = base_percdamp * r ** spectral_alpha

        An explicitly enabled module prior additionally multiplies this result
        by ``module_factor``. It is not part of the calibration-derived default.

        For ``method="diagonal"`` the eigenvalue estimate is also taken from
        ``current_diag`` so the proxy is consistent with the denominator. The
        result is clamped to ``damp_cfg.min`` / ``damp_cfg.max``.  When adaptive
        damping is disabled or unavailable, the static ``damp_cfg.min`` is
        returned.

        If ``lambda_spectral`` is provided, the eigenvalue estimation is skipped.
        Callers can hoist the (potentially expensive) power-iteration/Lanczos
        estimate outside of retry loops and reuse it across attempts.
        """

        damp_cfg = self.qcfg.damp
        if not isinstance(damp_cfg, AdaptiveDampingConfig) or not damp_cfg.enabled:
            return damp_cfg.min
        if H is None or H.ndim != 2:
            return damp_cfg.base_percdamp

        # Skip adaptive logic on devices where matmul timing is untested.
        if H.device.type == "npu":
            return damp_cfg.base_percdamp

        mean_val = mean.item()
        if not math.isfinite(mean_val) or mean_val <= 0:
            return damp_cfg.base_percdamp

        if lambda_spectral is None:
            lambda_spectral, _ = self._estimate_hessian_eigen_spectrum(
                H, damp_cfg.method, damp_cfg.eigen_iterations, effective_diag=current_diag
            )
        if lambda_spectral is None or not math.isfinite(lambda_spectral):
            return damp_cfg.base_percdamp

        module_factor = self._module_damp_factor(damp_cfg)
        spectral_ratio = lambda_spectral / mean_val
        spectral_factor = spectral_ratio ** damp_cfg.spectral_alpha

        damp = damp_cfg.base_percdamp * module_factor * spectral_factor
        # The configured `min`/`max` bound the final value, and the user-supplied
        # baseline (`base_percdamp`) is always respected as a ceiling so explicit
        # high `damp_percent` values are never silently lowered by `max_percdamp`.
        clamp_min = damp_cfg.min
        clamp_max = max(damp_cfg.max, damp_cfg.base_percdamp)
        damp = max(clamp_min, min(damp, clamp_max))

        if not (0 < damp < 1):
            return damp_cfg.base_percdamp

        log.info(
            f"Quantization: Module `{self.name}` -> Adaptive damping selected "
            f"`damp_percent={damp:.6f}` (base={damp_cfg.base_percdamp:.4f}, "
            f"module_factor={module_factor:.2f}, spectral_ratio={spectral_ratio:.4e}, "
            f"spectral_factor={spectral_factor:.4f})."
        )
        return damp

    @staticmethod
    def _update_group_feedback(
        L_g: float,
        feedback_target: Optional[float],
        ema_decay: float,
        gamma: float,
        factor_min: float,
        factor_max: float,
    ) -> Tuple[float, float]:
        """Update the EMA target and error-update scale from a group loss.

        Returns ``(new_feedback_target, error_update_scale)``.  A group loss
        larger than the EMA target yields ``error_update_scale < 1.0`` (smaller
        error update, i.e. more damping), while a smaller loss yields a value
        above ``1.0``.
        """
        if feedback_target is None:
            return L_g, 1.0
        if L_g == 0 or feedback_target == 0:
            return feedback_target, 1.0
        if not math.isfinite(L_g):
            return feedback_target, factor_min
        feedback_target = ema_decay * feedback_target + (1 - ema_decay) * L_g
        error_update_scale = (feedback_target / L_g) ** gamma
        error_update_scale = max(factor_min, min(factor_max, error_update_scale))
        return feedback_target, error_update_scale

    @staticmethod
    def _compute_group_loss(
        err_slice: torch.Tensor,
        h_diag_slice: Optional[torch.Tensor],
        use_hessian_weighting: bool,
    ) -> float:
        """Return the GPTQ group loss used by the experimental feedback controller.

        For the upper inverse-Cholesky factor ``U`` used by GPTQ, the canonical
        correction coefficient is ``E = (W - Q) / diag(U)`` and its block loss
        contribution is ``0.5 * ||E||_F^2``. Optional multiplication by the raw
        Hessian diagonal is an experimental heuristic, not the canonical GPTQ
        objective, because ``E`` already incorporates inverse-Hessian geometry.
        """
        if h_diag_slice is not None and use_hessian_weighting:
            return float((0.5 * torch.sum(h_diag_slice * (err_slice ** 2))).item())
        return float((0.5 * torch.sum(err_slice ** 2)).item())

    @staticmethod
    def _group_error_scale(
        error_update_scale: float,
        size_factor: float,
        scale_min: float = 0.8,
        scale_max: float = 1.2,
    ) -> float:
        """Return the per-column error scale for a quantization group.

        The size prior is applied as a divisor so that larger groups receive
        a smaller error update (more damping) and smaller groups receive a
        larger error update, matching the v4.2 design. The final scale is
        clamped to [scale_min, scale_max] to avoid accidentally large updates.
        """
        scale = error_update_scale / size_factor
        return max(scale_min, min(scale, scale_max))

    @torch.inference_mode()
    def hessian_inverse(
        self,
        H: torch.Tensor,
        release_input: bool = False,
    ):
        """Return the GPTQ inverse/Cholesky result.

        When ``release_input`` is true, ``self.H`` is released as soon as the
        Cholesky factor ``L`` is available, before the dense inverse factorization
        that produces ``Hinv``. Callers must clone any Hessian-derived objective
        data (diagonal, scale-search Hessians) before calling.
        """

        # Resolve the initial damping before the cache lookup so the cache key can
        # distinguish consumers of the same shared Hessian that resolve to different
        # adaptive damping values. The expensive eigen estimate is hoisted out of
        # the floor-attempt loop and reused by ``_compute_hessian_inverse_uncached``.
        # Do not re-attach an external Hessian to ``self.H``; callers that pass an
        # explicit tensor keep ownership.  Only resolve a missing argument from the
        # instance buffer.
        if H is None:
            H = self.H

        initial_damp = None
        lambda_spectral = None
        cache_damp = self.qcfg.damp.min
        if H is not None and H.ndim == 2 and torch.isfinite(H).all():
            damp_cfg = self.qcfg.damp
            orig_diag = H.diagonal().clone()
            mean = orig_diag.mean()
            mean_val = mean.item()
            if (
                math.isfinite(mean_val)
                and mean_val > 0
                and isinstance(damp_cfg, AdaptiveDampingConfig)
                and damp_cfg.enabled
                and H.device.type != "npu"
            ):
                # Power iteration and Lanczos see the same unmodified H across
                # floor attempts, so the eigen estimate can be computed once.
                # The diagonal method is cheap and reads the floored diagonal, so
                # it is left to be recomputed per attempt.
                if damp_cfg.method in ("power_iteration", "lanczos"):
                    lambda_spectral, _ = self._estimate_hessian_eigen_spectrum(
                        H, damp_cfg.method, damp_cfg.eigen_iterations, effective_diag=orig_diag
                    )
                initial_damp = self._resolve_initial_damp(H, orig_diag, mean, lambda_spectral=lambda_spectral)
                cache_damp = initial_damp

        timer = getattr(self, "region_timer", None)
        timer_cm = (
            timer.measure("hessian_inverse")
            if timer is not None
            else contextlib.nullcontext()
        )
        with timer_cm:
            cache = self._shared_hessian_inverse_cache
            cache_lock = self._shared_hessian_inverse_lock
            cache_key = self._shared_hessian_inverse_cache_key(H, damp=cache_damp)
            if cache is None or cache_lock is None or cache_key is None:
                if _log_hessian_verbose():
                    log.info(f"GPTQ: hessian_inverse begin {self.name} shape={tuple(H.shape)}")
                result = self._compute_hessian_inverse_uncached(
                    H,
                    initial_damp=initial_damp,
                    lambda_spectral=lambda_spectral,
                    release_input=release_input,
                )
                if _log_hessian_verbose():
                    log.info(f"GPTQ: hessian_inverse end {self.name}")
                return result

            with cache_lock:
                cached = cache.get(cache_key)
                stats = self._shared_hessian_stats
                if cached is not None:
                    if stats is not None:
                        stats["inverse_hits"] = int(stats.get("inverse_hits", 0)) + 1
                    self._release_shared_hessian_inverse_cache_entry(cache, cache_key)
                    if release_input and getattr(self, "H", None) is H:
                        self.H = None
                    return cached

                if stats is not None:
                    stats["inverse_misses"] = int(stats.get("inverse_misses", 0)) + 1

                if _log_hessian_verbose():
                    log.info(f"GPTQ: hessian_inverse begin {self.name} shape={tuple(H.shape)}")
                result = self._compute_hessian_inverse_uncached(
                    H,
                    initial_damp=initial_damp,
                    lambda_spectral=lambda_spectral,
                    release_input=release_input,
                )
                if _log_hessian_verbose():
                    log.info(f"GPTQ: hessian_inverse end {self.name}")
                if result[0] is not None:
                    cache[cache_key] = result
                self._release_shared_hessian_inverse_cache_entry(cache, cache_key)
                return result

    @torch.inference_mode()
    def _compute_hessian_inverse_uncached(
        self,
        H: torch.Tensor,
        initial_damp: Optional[float] = None,
        lambda_spectral: Optional[float] = None,
        release_input: bool = False,
    ):
        # A Hessian with non-finite entries can only produce a non-finite inverse.
        # Bail out early so the caller can fall back instead of propagating NaN/Inf.
        if not torch.isfinite(H).all():
            log.warn(
                f"Quantization: Module `{self.name}` -> Hessian contains non-finite values; "
                "skipping Cholesky inversion and using fallback."
            )
            return None, 1.0

        # Keep the original Hessian untouched; only a clone's diagonal is modified.
        orig_diag = H.diagonal().clone()
        d = orig_diag.numel()

        # When a block is numerically singular, pure damping can stall at 1.0.
        # Prepare a tiny diagonal floor (relative to the largest entry) that we
        # only inject if the normal damping loop fails. Keeping the scale near 1e-6
        # of the dominant entry keeps the bias negligible for healthy layers while
        # still rescuing pathological Hessian blocks.
        base_abs_max = torch.max(orig_diag.abs())
        finite_nonzero = torch.isfinite(base_abs_max) & (base_abs_max != 0)
        base_abs_max = torch.where(finite_nonzero, base_abs_max, base_abs_max.new_ones(()) * 1.0)
        floor_base = base_abs_max * 1e-6
        damp_cfg = self.qcfg.damp
        max_floor_attempts = 6
        used_damp = damp_cfg.min
        last_error = None

        # The expensive eigen-spectrum estimate only depends on the unmodified H,
        # so compute it once and reuse it across floor attempts. For the cheap
        # diagonal method the estimate is derived from the floored diagonal and
        # will be recomputed per attempt.
        if (
            lambda_spectral is None
            and isinstance(damp_cfg, AdaptiveDampingConfig)
            and damp_cfg.enabled
            and H.device.type != "npu"
            and damp_cfg.method in ("power_iteration", "lanczos")
        ):
            lambda_spectral, _ = self._estimate_hessian_eigen_spectrum(
                H, damp_cfg.method, damp_cfg.eigen_iterations, effective_diag=orig_diag
            )

        attempt = 0
        while attempt <= max_floor_attempts:
            if attempt == 0:
                current_diag = orig_diag
            else:
                floor_increment = floor_base * (10.0 ** (attempt - 1))
                current_diag = torch.clamp(orig_diag + floor_increment, min=floor_increment)
                if attempt == 1:
                    log.warn(
                        f"Quantization: Module `{self.name}` -> Applying Hessian diagonal floor "
                        f"(+{floor_increment:.2e}) to recover positive definiteness."
                    )
                else:
                    log.warn(
                        f"Quantization: Module `{self.name}` -> Increasing Hessian diagonal "
                        f"floor to +{floor_increment:.2e}."
                    )

            mean = current_diag.mean()
            if attempt == 0 and initial_damp is not None:
                damp_scalar = initial_damp
            else:
                damp_scalar = self._resolve_initial_damp(H, current_diag, mean, lambda_spectral=lambda_spectral)
            damp_per_col = torch.full((d,), damp_scalar, device=H.device, dtype=torch.float32)

            damp_recovery_started = False
            recovery_initial_damp = None
            recovery_last_damp = None
            # Shared Hessians are referenced by multiple GPTQ tasks; never mutate
            # the exact borrowed tensor in place. A reordered Hessian is private,
            # even if the task remains attached to a shared inverse-cache group.
            is_shared_hessian = getattr(self, "_shared_hessian_source", None) is H

            while (damp_per_col > 0).all().item() and (damp_per_col < 1).all().item():
                # Build the diagonal adjustment for this damping attempt.
                diag_delta = (current_diag - orig_diag) + damp_per_col * mean

                if H.device.type == "npu":
                    # NPU uses a dedicated eager path until a compiled equivalent is validated.
                    H_eff = H.clone()
                    H_eff.diagonal().add_(diag_delta)
                    try:
                        Hinv_result = npu_inverse_cholesky_factor(H_eff)
                        is_valid = (
                            Hinv_result is not None
                            and torch.isfinite(Hinv_result).all().item()
                            and (Hinv_result.diagonal() > 0).all().item()
                        )
                        if is_valid:
                            used_damp = float(damp_per_col.mean().item())
                            if damp_recovery_started:
                                log.warn(
                                    f"Quantization: Module `{self.name}` -> Damp recovery succeeded at "
                                    f"`damp_percent={used_damp:.5f}` "
                                    f"(started at {recovery_initial_damp:.5f})."
                                )
                            if release_input:
                                if getattr(self, "H", None) is H:
                                    self.H = None
                                del H, H_eff
                            return Hinv_result, used_damp
                        # Treat a numerically bad inverse the same as a Cholesky failure.
                        raise torch.linalg.LinAlgError(
                            "Hessian inverse contains non-finite or non-positive-definite entries."
                        )
                    except torch._C._LinAlgError as e:
                        last_error = e
                        if damp_cfg.step != 0:
                            if not damp_recovery_started:
                                damp_recovery_started = True
                                recovery_initial_damp = float(damp_per_col.mean().item())
                                log.warn(
                                    f"Quantization: Module `{self.name}` -> Starting damp recovery at "
                                    f"`damp_percent={recovery_initial_damp:.5f}`, increment step `{damp_cfg.step:.5f}`."
                                )
                            next_damp = damp_per_col + damp_cfg.step
                            if torch.equal(next_damp, damp_per_col):
                                log.warn(
                                    f"Quantization: Module `{self.name}` -> Damp recovery increment "
                                    f"`{damp_cfg.step}` is below {damp_per_col.dtype} resolution; "
                                    "stopping this recovery attempt."
                                )
                                break
                            damp_per_col = next_damp
                            recovery_last_damp = float(damp_per_col.mean().item())
                        else:
                            log.warn(
                                f"Quantization: Module `{self.name}` -> Hessian Cholesky failed with "
                                f"`damp_percent={float(damp_per_col.mean().item()):.5f}` and no auto increment configured."
                            )
                            break
                    continue

                if H.device.type == "cpu":
                    # Compiled CPU path: fuses diagonal damp + Cholesky + inverse
                    # into one extension call using ATen's MKL/LAPACK AVX-512 path.
                    from ..nn_modules.qlinear.pack_block_ext import hessian_inverse_cholesky_cpu

                    Hinv_result, success = hessian_inverse_cholesky_cpu(H, diag_delta)
                else:
                    # Use the eager Cholesky path for damp recovery; torch.compile
                    # on these ops showed no speedup in micro-benchmarks and can hang
                    # when the compiled graph raises a RuntimeError and is re-entered.
                    # Avoid a full H clone for private Hessians; restore the diagonal
                    # before each damping attempt because _hessian_inverse_try_cholesky
                    # applies diag_delta in place.
                    if is_shared_hessian:
                        H_eff = H.clone()
                    else:
                        H_eff = H
                    # _hessian_inverse_try_cholesky adds diag_delta in place.  Restore
                    # the original diagonal in a finally block so an exception (e.g.
                    # CUDA OOM) does not leave the caller's Hessian carrying leftover
                    # damping when quantize() falls back to CPU and retries.
                    if H.device.type == "mps" and _USE_GPTQ_MPS_FAST_HESSIAN:
                        # The throwing Cholesky API performs its own status check.
                        # On the healthy path this avoids the repeated host scalar
                        # synchronizations required by cholesky_ex/info and the
                        # intermediate dense-inverse checks below. The operations
                        # and their order are otherwise identical to the canonical
                        # factorization. Any failure enters the existing recovery.
                        mps_result = None
                        try:
                            H_eff.diagonal().copy_(orig_diag)
                            H_eff.diagonal().add_(diag_delta)
                            mps_lower = linalg_cholesky(H_eff, upper=False)
                            mps_dense_inverse = torch.empty_like(mps_lower)
                            cholesky_inverse(mps_lower, upper=False, out=mps_dense_inverse)
                            del mps_lower
                            mps_result = linalg_cholesky(
                                mps_dense_inverse,
                                upper=True,
                                out=mps_dense_inverse,
                            )
                            mps_valid = (
                                torch.isfinite(mps_result).all()
                                & (mps_result.diagonal() > 0).all()
                            ).item()
                        except RuntimeError as e:
                            last_error = e
                            mps_valid = False
                        finally:
                            H_eff.diagonal().copy_(orig_diag)
                        if mps_valid:
                            used_damp = float(damp_per_col.mean().item())
                            if damp_recovery_started:
                                log.warn(
                                    f"Quantization: Module `{self.name}` -> Damp recovery succeeded at "
                                    f"`damp_percent={used_damp:.5f}` "
                                    f"(started at {recovery_initial_damp:.5f})."
                                )
                            if release_input:
                                if getattr(self, "H", None) is H:
                                    self.H = None
                                del H, H_eff
                            return mps_result, used_damp
                        mps_result = None
                        try:
                            del mps_dense_inverse
                        except NameError:
                            pass
                    try:
                        H_eff.diagonal().copy_(orig_diag)
                        L, success = _hessian_inverse_try_cholesky(H_eff, diag_delta)
                    finally:
                        H_eff.diagonal().copy_(orig_diag)
                    if success.item():
                        # CPU/CUDA Cholesky inverse supports an aliased output.
                        # Reuse the factor buffer; the original Hessian remains
                        # intact for a damping retry if either step fails.
                        reuse_factor = L.device.type in ("cpu", "cuda")
                        Hinv_dense = L if reuse_factor else torch.empty_like(L)
                        try:
                            cholesky_inverse(L, upper=False, out=Hinv_dense)
                        except RuntimeError as e:
                            Hinv_result = None
                            success = damp_per_col.new_tensor(False, dtype=torch.bool)
                            last_error = e
                        else:
                            is_dense_inverse_valid = (
                                torch.isfinite(Hinv_dense).all().item()
                                and (Hinv_dense.diagonal() > 0).all().item()
                            )
                            if not is_dense_inverse_valid:
                                Hinv_result = None
                                success = damp_per_col.new_tensor(False, dtype=torch.bool)
                                last_error = torch.linalg.LinAlgError(
                                    "Dense Hessian inverse is non-finite or not positive-definite."
                                )
                            else:
                                # Factor the dense inverse in place. The input buffer
                                # is overwritten with the upper Cholesky factor ``U``,
                                # while the original Hessian ``H`` stays intact so
                                # damping recovery can continue if this step fails.
                                del L
                                info = Hinv_dense.new_empty((), dtype=torch.int32)
                                try:
                                    Hinv_result, info = linalg_cholesky_ex(
                                        Hinv_dense, upper=True, out=(Hinv_dense, info)
                                    )
                                except RuntimeError as e:
                                    Hinv_result = None
                                    success = damp_per_col.new_tensor(False, dtype=torch.bool)
                                    last_error = e
                                else:
                                    if info.item() != 0:
                                        Hinv_result = None
                                        success = damp_per_col.new_tensor(False, dtype=torch.bool)
                                        last_error = torch.linalg.LinAlgError(
                                            f"Cholesky factorization of H^-1 failed with info={info.item()}."
                                        )

                if success.item() and Hinv_result is not None:
                    is_valid = (
                        torch.isfinite(Hinv_result).all().item()
                        and (Hinv_result.diagonal() > 0).all().item()
                    )
                    if is_valid:
                        used_damp = float(damp_per_col.mean().item())
                        if damp_recovery_started:
                            log.warn(
                                f"Quantization: Module `{self.name}` -> Damp recovery succeeded at "
                                f"`damp_percent={used_damp:.5f}` "
                                f"(started at {recovery_initial_damp:.5f})."
                            )
                        if release_input:
                            if getattr(self, "H", None) is H:
                                self.H = None
                            del H
                            try:
                                del H_eff
                            except NameError:
                                pass
                        try:
                            del L
                        except NameError:
                            pass
                        return Hinv_result, used_damp
                    # The factorization produced a non-finite or non-positive-definite
                    # inverse. Treat it as a Cholesky failure and continue damping.
                    last_error = torch.linalg.LinAlgError(
                        "Final upper Cholesky factor is non-finite or not positive-definite."
                    )
                    success = damp_per_col.new_tensor(False, dtype=torch.bool)
                    try:
                        del L
                    except NameError:
                        pass

                if damp_cfg.step != 0:
                    if not damp_recovery_started:
                        damp_recovery_started = True
                        recovery_initial_damp = float(damp_per_col.mean().item())
                        log.warn(
                            f"Quantization: Module `{self.name}` -> Starting damp recovery at "
                            f"`damp_percent={recovery_initial_damp:.5f}`, increment step `{damp_cfg.step:.5f}`."
                        )
                    next_damp = damp_per_col + damp_cfg.step
                    if torch.equal(next_damp, damp_per_col):
                        log.warn(
                            f"Quantization: Module `{self.name}` -> Damp recovery increment "
                            f"`{damp_cfg.step}` is below {damp_per_col.dtype} resolution; "
                            "stopping this recovery attempt."
                        )
                        break
                    damp_per_col = next_damp
                    recovery_last_damp = float(damp_per_col.mean().item())
                else:
                    log.warn(
                        f"Quantization: Module `{self.name}` -> Hessian Cholesky failed with "
                        f"`damp_percent={float(damp_per_col.mean().item()):.5f}` and no auto increment configured."
                    )
                    break

            if damp_recovery_started:
                final_damp = recovery_last_damp if recovery_last_damp is not None else float(damp_per_col.mean().item())
                log.warn(
                    f"Quantization: Module `{self.name}` -> Damp recovery failed after reaching "
                    f"`damp_percent={final_damp:.5f}`."
                )

            attempt += 1

        log.error(
            f"Quantization: Module `{self.name}` -> Hessian remained non positive-definite "
            f"after diagonal floor attempts. Last `damp_percent` tried = {float(damp_per_col.mean().item()):.5f}."
        )
        if last_error is not None:
            log.debug(f"Hessian failure detail: {last_error}")
        return None, 1.0

    @torch.inference_mode()
    def hessian_inverse_diag(self, h_diag: torch.Tensor):
        """
        Embedding-only: returns the inverse of a damped diagonal Hessian.

        We compute:
            h_eff = clamp(h_diag, min=floor) + damp * mean(h_diag)
            Hinv_diag = 1.0 / h_eff

        This keeps the behavior aligned with dense-path damping, without building VxV.
        """
        assert h_diag.dim() == 1, "Embedding Hessian diagonal must be a 1D vector."
        # Use the anchor/base damping, not the clamp floor, so adaptive and
        # static configs behave consistently.
        damp = getattr(self.qcfg.damp, "base_percdamp", self.qcfg.damp.min)
        mean = torch.mean(h_diag)
        # Apply a small floor to stabilize inverse for rare/unseen tokens
        abs_max = max(h_diag.max().item(), 1.0)
        floor = abs_max * 1e-6
        h_eff = torch.clamp(h_diag, min=floor) + damp * mean
        Hinv_diag = 1.0 / h_eff
        return Hinv_diag, damp

    @torch.inference_mode()
    def quantize(
            self,
            blocksize=128,
    ):
        config = getattr(self.qcfg, "gsq", None)
        module_name = self._named_module.full_name if self._named_module is not None else self.name
        if getattr(self, "_gsq_active", False) or not gsq_enabled_for(config, module_name):
            return self._quantize_impl(blocksize=blocksize)

        from ..utils.fallback import should_use_fallback
        if should_use_fallback(self.fallback, float(self.nsamples), self.expected_nsamples):
            self.gsq_diagnostics = {"status": "skipped", "reason": "data_independent_fallback"}
            return self._quantize_impl(blocksize=blocksize)
        start = time.time()
        target = self.clone_module()
        # Preserve the original-column calibration metric before GPTQ consumes,
        # permutes, damps or releases it. Embeddings keep a diagonal metric.
        # FOEM accumulates H directly, unlike GPTQ/GPTAQ's partial buffers.
        if self.qcfg.foem is not None:
            hessian = getattr(self, "H", None)
            if hessian is None:
                raise ValueError("FOEM GSQ requires unconsumed calibration statistics")
        else:
            hessian = self.finalize_hessian(target_device=target.device)
        if isinstance(self.module, nn.Embedding):
            hessian = self._H_diag
        hessian = None if hessian is None else hessian.detach().clone()
        cross_moment = None
        cross_alpha = 1.0
        asymmetric_config = self.qcfg.foem if self.qcfg.foem is not None else self.qcfg.gptaq
        if asymmetric_config is not None and asymmetric_config.alpha != 0:
            cross = getattr(self, "dXXT", None)
            if hessian is None or cross is None or getattr(self, "_hessian_rebuild_invalid", False):
                raise ValueError("asymmetric GSQ requires unconsumed paired calibration statistics")
            cross_moment = cross.detach().clone()
            cross_alpha = asymmetric_config.alpha
        self._gsq_active = True
        try:
            result = self._quantize_impl(blocksize=blocksize)
        finally:
            self._gsq_active = False
        weight, scales, zeros, groups, duration, avg_loss, damp, samples = result
        if isinstance(avg_loss, str) and avg_loss.startswith("fallback("):
            self.gsq_diagnostics = {"status": "skipped", "reason": "data_independent_fallback"}
            return result
        canonical = weight
        if isinstance(self.module, (nn.Embedding, transformers.Conv1D)):
            canonical = canonical.T
        elif isinstance(self.module, _ConvNd):
            canonical = canonical.flatten(1)
        width = canonical.shape[1]
        if hessian is not None:
            hessian = hessian[:width] if hessian.ndim == 1 else hessian[:width, :width]
        fitted = refine_affine_scalar(
            canonical, scales.to(weight.device), zeros.to(weight.device), groups.to(weight.device),
            target=target[:, :width].to(weight.device), bits=self.qcfg.bits, config=config,
            hessian=hessian, cross_moment=cross_moment, cross_alpha=cross_alpha,
        )
        refined = fitted.weight
        if isinstance(self.module, (nn.Embedding, transformers.Conv1D)):
            refined = refined.T
        self.gsq_diagnostics = {
            "objective": "calibration_hessian" if hessian is not None else "weight_mse",
            "before": fitted.before, "after": fitted.after, "learn_scales": config.learn_scales,
        }
        if cross_moment is not None:
            self.gsq_diagnostics["objective"] = "asymmetric_quadratic_without_constant"
            self.gsq_diagnostics["alpha"] = cross_alpha
        if self.qcfg.foem is not None:
            self.gsq_diagnostics["initializer"] = "foem"
            self.gsq_diagnostics["initializer_beta"] = self.qcfg.foem.beta
        return (refined.reshape(weight.shape).contiguous(), fitted.scales.to(scales.device),
                fitted.zeros.to(zeros.device), fitted.g_idx.to(groups.device),
                time.time()-start, avg_loss, damp, samples)

    @torch.inference_mode()
    def _quantize_impl(
            self,
            blocksize=128,
    ):
        # self.H = self.H.to(device=CUDA_0)
        # log.info(f"Quantization `{self.name}` using samples: `{self.nsamples}`")
        start = time.time()

        target_device = getattr(self.module, "target_device", None)
        result_device = torch.device(self.module.weight.data.device)
        cpu_fallback_used = False
        from ..utils.fallback import resolve_fallback_strategy, resolve_threshold, should_use_fallback

        resolved_strategy = resolve_fallback_strategy(self.fallback)
        fallback_requested = should_use_fallback(
            self.fallback,
            float(self.nsamples),
            self.expected_nsamples,
        )
        threshold_raw, is_percent = resolve_threshold(self.fallback, self.expected_nsamples)
        fallback_configured = threshold_raw is not None

        if fallback_requested:
            use_hessian = False
            threshold_text = str(getattr(self.fallback, "threshold", None))
            threshold_info = f", threshold_raw={threshold_raw}" if threshold_raw is not None and is_percent else ""
            log.warn(
                f"Quantization: Module `{self.name}` -> "
                f"Using `{resolved_strategy.value}` fallback quantization (observed {self.nsamples} samples, threshold={threshold_text}{threshold_info}, max_total={self.expected_nsamples})."
            )
            # RTN/MIDPOINT fallback does not read Hessian values. Resolve the
            # compute device before releasing the potentially large partials.
            with self.lock:
                fallback_device = self._select_hessian_target_device(target_device)
                self._device_hessian_partials.clear()
                self._device_sample_counts.clear()
                self._device_sequence_counts.clear()
                self._hessian_dirty = False
            return self._fallback_quantize(
                resolved_strategy, blocksize, target_device=fallback_device
            )
        else:
            use_hessian = True
            self.finalize_hessian(target_device=target_device)

        # A mock-quantization recursion (or any path that released ``self.H`` after
        # the partials were already merged) has no calibration statistics left to
        # rebuild a dense Hessian. Continuing would treat an all-zero matrix as valid
        # and zero the entire weight. Route straight to a data-independent fallback.
        if getattr(self, "_hessian_rebuild_invalid", False):
            log.warn(
                f"Quantization: Module `{self.name}` -> Hessian data already consumed, "
                f"using `{resolved_strategy.value}` fallback."
            )
            return self._fallback_quantize(resolved_strategy, blocksize)

        # The top-level `hessian_inverse` method (cache/locks and the data-dependent
        # damp-recovery loop) is not compiled: torch.compile cannot represent the
        # recovery loop without graph breaks. The heavy Cholesky/inverse steps
        # inside `_compute_hessian_inverse_uncached` are compiled via module-level
        # helpers, so the dense Hessian path is still graph-break-free on GPU.
        # The guard is kept for older PyTorch builds where compiling the whole
        # method was previously enabled.
        if not TORCH_GTE_28 and not self.qcfg.mock_quantization:
            self.hessian_inverse = torch_compile(self.hessian_inverse)

        if self.qcfg.mock_quantization:
            # Use simplified hessian inverse (identity matrix)
            self.hessian_inverse = self.mock_hessian_inverse

        # -----------------------------
        # Embedding-specialized path
        # -----------------------------
        if isinstance(self.module, nn.Embedding):
            # Clone weight as [D, V] (columns == tokens)
            if self.module_copy is None:
                W = self.clone_module(device=self._final_hessian_device_hint)
            else:
                W = self.module_copy.to(device=self._final_hessian_device_hint)
                del self.module_copy

            # Prepare diagonal Hessian inverse
            if self._H_diag is None:
                # No samples? fall back to uniform diag
                self._H_diag = torch.ones(self.columns, dtype=torch.float32, device=W.device)
            Hinv_diag, damp = self.hessian_inverse_diag(self._H_diag)

            # Optional activation ordering (desc_act) or group-aware ordering using diagonal values
            # Note: we reuse the same logic/perm utilities as dense path, but with diag only.
            if self.qcfg.desc_act:
                perm = torch.argsort(self._H_diag, descending=True)
                W = W[:, perm]
                invperm = torch.argsort(perm)

            elif self.qcfg.act_group_aware:
                diag_h = self._H_diag
                local_perms, local_values = compute_local_perms(
                    diag_h, self.qcfg.group_size, return_values=True
                )
                global_perm = compute_global_perm(
                    diag_h,
                    self.qcfg.group_size,
                    precomputed_values=local_values,
                )
                del local_values
                final_perm = compose_final_perm(local_perms, global_perm, self.qcfg.group_size)
                W = W[:, final_perm]

            # Vectorized weight-only quantization over columns (tokens)
            scale = []
            zero = []
            now_idx = 1

            # Embedding loss is only reported after quantization; keep one
            # scalar instead of a full [embedding_dim, vocab] loss tensor.
            loss_sum = W.new_zeros(())
            # The full output buffer is only reordered/sliced before return.
            # Keep it in the final module dtype instead of fp32 to reduce peak
            # memory; per-block Q1 remains fp32 for quantization math/loss.
            Q = torch.empty_like(W, dtype=self.module.weight.dtype)

            # Fast vectorized path (no cross-column error feedback for Embedding)
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2]
                # Every column in Q1 is written before use; skip the zero-fill
                # on this block scratch buffer.
                Q1 = torch.empty_like(W1)

                if self.qcfg.group_size != -1:
                    # Group-wise parameter finding across columns
                    group_start_cols = list(range(i1, i2, self.qcfg.group_size))
                    for group_start in group_start_cols:
                        group_end = min(group_start + self.qcfg.group_size, self.columns)
                        if group_start < group_end:
                            self.quantizer.find_params(W[:, group_start:group_end], weight=True)
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1

                    # Use the latest computed scale/zero to quantize this block vectorized
                    if len(scale) > 0 and len(zero) > 0:
                        latest_scale = scale[-1]
                        latest_zero = zero[-1]

                        if latest_scale.dim() == 1:
                            latest_scale = latest_scale.view(-1, 1)
                        if latest_zero.dim() == 1:
                            latest_zero = latest_zero.view(-1, 1)

                        maxq_val = 2 ** self.qcfg.bits - 1
                        if self.qcfg.sym:
                            Q1 = latest_scale * torch.clamp(
                                torch.round(W1 / latest_scale),
                                -(maxq_val // 2),
                                maxq_val // 2
                            )
                        else:
                            quantized = torch.clamp(
                                torch.round(W1 / latest_scale) + latest_zero,
                                0,
                                maxq_val
                            )
                            Q1 = latest_scale * (quantized - latest_zero)
                    else:
                        # Fallback per-column
                        for i in range(count):
                            w = W1[:, i]
                            q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                            Q1[:, i] = q
                else:
                    # No grouping -> parameterize once and quantize vectorized
                    self.quantizer.find_params(W, weight=True)
                    latest_scale = self.quantizer.scale
                    latest_zero = self.quantizer.zero
                    if latest_scale.dim() == 1:
                        latest_scale = latest_scale.view(-1, 1)
                    if latest_zero.dim() == 1:
                        latest_zero = latest_zero.view(-1, 1)

                    maxq_val = 2 ** self.qcfg.bits - 1
                    if self.qcfg.sym:
                        Q1 = latest_scale * torch.clamp(
                            torch.round(W1 / latest_scale),
                            -(maxq_val // 2),
                            maxq_val // 2
                        )
                    else:
                        quantized = torch.clamp(
                            torch.round(W1 / latest_scale) + latest_zero,
                            0,
                            maxq_val
                        )
                        Q1 = latest_scale * (quantized - latest_zero)

                # Fill losses using diagonal inverse (no cross-column update)
                # d = 1 / h_eff (see hessian_inverse_diag); we use it to scale the loss similarly to dense path.
                for i in range(count):
                    col_idx = i1 + i
                    d = Hinv_diag[col_idx]
                    # hessian_inverse_diag clamps h_eff to a positive floor, so
                    # d is always positive. Avoid a per-column CUDA scalar bool.
                    loss_sum.add_(torch.sum((W1[:, i] - Q1[:, i]) ** 2 / (d ** 2)) / 2)

                Q[:, i1:i2] = Q1

            # Undo permutations if applied
            if self.qcfg.desc_act:
                Q = Q[:, invperm]
                # g_idx and scales/zeros will be re-ordered below once concatenated
            elif self.qcfg.act_group_aware:
                inv_final = invert_perm(final_perm)
                Q = Q[:, inv_final]
                # Note: if you need to keep per-group scale/zero in act_group_aware mode,
                # reorder them following the dense path approach (shown below after concatenation).

            # Prepare g_idx (group indices per column)
            group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns
            g_idx = self.build_group_index(self.columns, group_size, Q.device)

            # Finalize scale/zero concatenation
            if scale == []:
                # Ensure we have valid scale/zero if group_size == -1
                self.quantizer.find_params(Q, weight=True)
                scale.append(self.quantizer.scale)
                zero.append(self.quantizer.zero)

            scale = torch.cat(scale, dim=1)
            zero = torch.cat(zero, dim=1)

            # Reorder scale/zero/g_idx if we used permutations
            if self.qcfg.desc_act:
                scale = scale[:, invperm]
                zero = zero[:, invperm]
                g_idx = g_idx[invperm]
            elif self.qcfg.act_group_aware:
                # Reorder scale/zero with inverse global perm (same as dense path)
                inv_global_perm = invert_perm(global_perm).to(device=scale.device)
                # `scale` and `zero` are already concatenated tensors here, so
                # reorder columns directly on-device instead of syncing a
                # permutation list to Python and rebuilding through slices.
                scale = scale.index_select(1, inv_global_perm)
                zero = zero.index_select(1, inv_global_perm.to(device=zero.device))

            # Cropping if TP padding existed
            if self._tp_pad_cols:
                valid_cols = self._original_columns
                Q = Q[:, :valid_cols]
                g_idx = g_idx[:valid_cols]
                scale = self.truncate_last_dim(scale, valid_cols)
                zero = self.truncate_last_dim(zero, valid_cols)

            # Convert back to original embedding weight shape [V, D]
            Q_out = Q.t()
            if Q_out.shape != self.module.weight.shape:
                Q_out = Q_out.reshape(self.module.weight.shape).to(self.module.weight.dtype)
            else:
                Q_out = Q_out.to(self.module.weight.dtype)

            duration = time.time() - start
            # Compute avg_loss
            if self.nsamples != 0:
                avg_loss = loss_sum.item() / self.nsamples
                if math.isnan(avg_loss):
                    if self.qcfg.mock_quantization:
                        # Mock retry already failed; do not recurse again.
                        if fallback_configured:
                            log.warn(
                                f"Quantization: mock retry also produced `NaN` loss for `{self.name}`; "
                                "returning current result under fallback."
                            )
                            avg_loss = 999999999
                        else:
                            raise ValueError(
                                f"Quantization: mock retry also produced `NaN` loss for `{self.name}`; "
                                "increase calibration or enable fallback."
                            )
                    elif fallback_configured:
                        log.info(f"Quantization: Failed due to NaN loss for `{self.name}`, retry with mock quantization.")
                        self.qcfg.mock_quantization = True
                        return self.quantize(blocksize=blocksize)
                    else:
                        raise ValueError(
                            f"Quantization: NaN loss for `{self.name}`; increase calibration or enable fallback."
                        )
            else:
                if fallback_configured:
                    log.warn(f"Quantization: Module `{self.name}` -> using fail safe mode. Please check calibration sufficiency.")
                else:
                    log.warn(f"Quantization: `{self.name}` may be inactive due to model inference logic.")
                avg_loss = 999999999

            return Q_out.to(device=self.module.weight.data.device, non_blocking=False), scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

        # -----------------------------
        # Original dense path (non-Embedding)
        # -----------------------------

        if self.module_copy is None:
            W = self.clone_module(device=self.H.device)
        else:
            W = self.module_copy.to(device=self.H.device)
            del self.module_copy

        # Grouped quantization refreshes the quantizer from the corresponding
        # weight/Hessian slice before quantizing the first column in every
        # group. A full-tensor range search would therefore be overwritten
        # before use while allocating candidates for the largest projection.
        uses_group_params = int(getattr(self.qcfg, "group_size", -1) or -1) > 0
        if not uses_group_params:
            self.quantizer.find_params(W, weight=True, hessian=self.H)

        # H = self.H.to(device=self.H.device)

        if use_hessian:
            # Read the Hessian diagonal as a view, compute the dead-column mask,
            # then drop the view so the dense Hessian can be released for peak
            # memory during activation-order permutation and inversion.
            h_diag = self.H.diagonal()
            dead = h_diag == 0
            del h_diag
            if dead.any().item():
                # Shared Hessians are immutable borrowed state. A dead-column
                # repair is an in-place write, so isolate only the task that
                # actually needs the repair instead of corrupting peers that may
                # be quantizing concurrently.
                if getattr(self, "_shared_hessian_source", None) is self.H:
                    self.H = self.H.clone()
                    self._shared_hessian_source = None
                self.H[dead, dead] = 1
                W[:, dead] = 0

        for legacy_name in ("_adjacent_model_config", "adjacent_config"):
            if getattr(self.qcfg, legacy_name, None) is not None:
                raise ValueError(f"{legacy_name} was renamed to adjacent_model.")
        adjacent_model = getattr(self.qcfg, "adjacent_model", None)
        adjacent_reference_weight = None
        adjacent_reference_hessian = None
        if adjacent_model is not None:
            from .adjacent_model import AdjacentModelConfig

            if not isinstance(adjacent_model, AdjacentModelConfig):
                raise TypeError("adjacent_model must be an AdjacentModelConfig.")
            if not use_hessian:
                raise ValueError("Whole-model AdjacentExact requires Hessian-driven GPTQ.")
            if self.qcfg.desc_act:
                raise ValueError("Whole-model AdjacentExact does not support desc_act=True.")
            if not 1 <= int(self.qcfg.group_size) <= 128:
                raise ValueError("Whole-model AdjacentExact requires group_size in [1, 128].")
            if W.device.type != "cuda" or self.H.device.type != "cuda":
                raise ValueError("Whole-model AdjacentExact requires CUDA-resident weights and Hessians.")
            # Keep the original column order and undamped objective. GPTQ may
            # permute and then release its working Hessian before postprocessing.
            adjacent_reference_weight = W.clone()
            adjacent_reference_hessian = self.H.clone()

        # g_idx = []
        scale = []
        zero = []
        now_idx = 1

        if self.qcfg.static_groups:
            import copy

            groups = []
            # The region_timer contains a threading.Lock and cannot be deepcopied.
            # Stash it, deepcopy the quantizer without it, then restore it on each clone.
            original_timer = self.quantizer.region_timer
            self.quantizer.region_timer = None
            try:
                for i in range(0, self.columns, self.qcfg.group_size):
                    quantizer = copy.deepcopy(self.quantizer)
                    # Share the same region timer across group clones so all
                    # scale-search timing aggregates into one region.
                    quantizer.region_timer = original_timer
                    group_end = min(i + self.qcfg.group_size, self.columns)
                    quantizer.find_params(
                        W[:, i:group_end],
                        weight=True,
                        hessian=self.H[i:group_end, i:group_end],
                    )

                    scale.append(quantizer.scale)
                    zero.append(quantizer.zero)
                    groups.append(quantizer)
            finally:
                self.quantizer.region_timer = original_timer

        if self.qcfg.desc_act and use_hessian:
            perm = torch.argsort(self.H.diagonal(), descending=True)
            old_H = self.H
            H_perm = None
            weight_permuted = False
            try:
                # First permute; then drop the original Hessian before the second
                # copy is allocated so only two [columns, columns] tensors coexist.
                H_perm = old_H[perm]
                self.H = None
                del old_H
                W = W[:, perm]
                weight_permuted = True
                self.H = H_perm[:, perm]
                del H_perm
            except RuntimeError as exc:
                source_device = H_perm.device if H_perm is not None else old_H.device
                if source_device.type != "cuda" or "out of memory" not in str(exc).lower():
                    raise

                self.log_cpu_fallback("Hessian permutation", source_device)
                cpu_fallback_used = True
                cpu_device = torch.device("cpu")
                perm = perm.to(device=cpu_device)
                W = W.to(device=cpu_device)
                if not weight_permuted:
                    W = W[:, perm]
                if H_perm is not None:
                    H_cpu = H_perm.to(device=cpu_device)
                    del H_perm
                    self.H = H_cpu[:, perm]
                    del H_cpu
                else:
                    H_cpu = old_H.to(device=cpu_device)
                    del old_H
                    self.H = H_cpu[perm][:, perm]
                    del H_cpu
                if not uses_group_params:
                    self.quantizer.find_params(W, weight=True, hessian=self.H)
            invperm = torch.argsort(perm)
            self._hessian_is_shared = False
            self._shared_hessian_source = None

        elif self.qcfg.act_group_aware and use_hessian:
            diag_h = self.H.diagonal()
            local_perms, local_values = compute_local_perms(
                diag_h, self.qcfg.group_size, return_values=True
            )
            global_perm = compute_global_perm(
                diag_h,
                self.qcfg.group_size,
                precomputed_values=local_values,
            )
            del local_values
            final_perm = compose_final_perm(local_perms, global_perm, self.qcfg.group_size)
            final_perm = extend_perm_with_tail(final_perm, self.columns)
            if self.qcfg.static_groups:
                # Static quantizers are created in original group order above,
                # while GAR processes full groups in ``global_perm`` order.
                # Reorder only the working quantizer list; the serialized
                # scale/zero lists must remain in original group order after Q
                # is restored to its original column layout.
                reordered_group_count = int(global_perm.numel())
                quantizer_group_order = global_perm.tolist()
                groups = [groups[i] for i in quantizer_group_order] + groups[reordered_group_count:]
                del quantizer_group_order
            # ``diag_h`` is a view of the original Hessian; keeping it alive would
            # prevent the unpermuted Hessian from being freed after ``self.H`` is
            # reassigned.  ``local_perms`` is also no longer needed.
            del diag_h, local_perms
            old_H = self.H
            H_perm = None
            weight_permuted = False
            try:
                W = W[:, final_perm]
                weight_permuted = True
                # Free the unpermuted Hessian before allocating the fully permuted
                # copy; only two [columns, columns] tensors are live at once.
                H_perm = old_H[final_perm]
                self.H = None
                del old_H
                self.H = H_perm[:, final_perm]
                del H_perm
            except RuntimeError as exc:
                source_device = H_perm.device if H_perm is not None else old_H.device
                if source_device.type != "cuda" or "out of memory" not in str(exc).lower():
                    raise

                self.log_cpu_fallback("act-group Hessian permutation", source_device)
                cpu_fallback_used = True
                cpu_device = torch.device("cpu")
                final_perm = final_perm.to(device=cpu_device)
                W = W.to(device=cpu_device)
                if not weight_permuted:
                    W = W[:, final_perm]
                if H_perm is not None:
                    H_cpu = H_perm.to(device=cpu_device)
                    del H_perm
                    self.H = H_cpu[:, final_perm]
                    del H_cpu
                else:
                    H_cpu = old_H.to(device=cpu_device)
                    del old_H
                    self.H = H_cpu[final_perm][:, final_perm]
                    del H_cpu
                if not uses_group_params:
                    self.quantizer.find_params(W, weight=True, hessian=self.H)
            self._hessian_is_shared = False
            self._shared_hessian_source = None

        damp_cfg = self.qcfg.damp
        use_online_group_damping = (
            use_hessian
            and not self.qcfg.mock_quantization
            and isinstance(damp_cfg, AdaptiveDampingConfig)
            and damp_cfg.enabled
            and damp_cfg.online_feedback_enabled
            and self.qcfg.group_size > 0
        )
        clip_cfg = self.qcfg.adaptive_clipping
        use_adaptive_clipping = (
            use_hessian
            and not self.qcfg.mock_quantization
            and self.qcfg.group_size > 0
            and isinstance(clip_cfg, AdaptiveClippingConfig)
            and clip_cfg.enabled
            and clip_cfg.per_group
        )
        # Capture all Hessian-derived objective data before the inverse step.
        # hessian_inverse() will release self.H as soon as the lower Cholesky
        # factor is known, so these clones must be taken here.
        group_scale_search_diagonal = None
        group_scale_search_diagonal_prepared = None
        group_scale_search_hessians = None
        scale_search = getattr(self.qcfg, "scale_search", None)
        if (
            use_hessian
            and uses_group_params
            and not self.qcfg.static_groups
            and float(getattr(self.qcfg, "mse", 0.0) or 0.0) > 0.0
            and not use_adaptive_clipping
        ):
            if scale_search in {ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.MARLIN_ACTIVATION}:
                # clone() is required: retaining a diagonal view would keep the
                # complete dense Hessian storage alive.
                group_scale_search_diagonal = self.H.diagonal().clone()
                full_group_columns = (self.columns // self.qcfg.group_size) * self.qcfg.group_size
                if full_group_columns:
                    # Every 128-column GPTQ block revisits the same per-group
                    # activation importance.  Normalize all complete groups
                    # once with the identical row-wise FP32 operations used by
                    # find_params_batched; tail groups continue through the
                    # original per-call path below.
                    group_scale_search_diagonal_prepared = (
                        self.quantizer._prepare_scale_search_hessian_batched(
                            group_scale_search_diagonal[:full_group_columns].reshape(
                                -1,
                                self.qcfg.group_size,
                            ),
                            method=scale_search,
                        )
                    )
            elif scale_search in {
                ScaleSearchConfig.HESSIAN,
                ScaleSearchConfig.HYBRID,
                ScaleSearchConfig.MARLIN,
            }:
                group_size = self.qcfg.group_size
                group_scale_search_hessians = tuple(
                    self.H[start:min(start + group_size, self.columns), start:min(start + group_size, self.columns)].clone()
                    for start in range(0, self.columns, group_size)
                )

        clip_hessian_diag = None
        group_loss_hessian_diag = None
        clip_need_diag = (
            use_adaptive_clipping and getattr(clip_cfg, "metric", None) == "hessian_diag"
        )
        loss_need_diag = (
            use_online_group_damping
            and isinstance(damp_cfg, AdaptiveDampingConfig)
            and getattr(damp_cfg, "group_error_use_hessian_weighting", False)
        )
        if clip_need_diag or loss_need_diag:
            _shared_hessian_diag = self.H.diagonal().float().clone()
            if clip_need_diag:
                clip_hessian_diag = _shared_hessian_diag
            if loss_need_diag:
                group_loss_hessian_diag = _shared_hessian_diag

        if use_hessian:
            try:
                Hinv, damp = self.hessian_inverse(self.H, release_input=True)
            except RuntimeError as exc:
                if self.H.device.type != "cuda" or "out of memory" not in str(exc).lower():
                    raise

                # Full-attention blocks on very large models can exceed GPU memory during the
                # dense Hessian inverse; finish that module on CPU instead of aborting the run.
                self.log_cpu_fallback("Hessian inverse", self.H.device)
                cpu_fallback_used = True
                cpu_device = torch.device("cpu")
                self.H = self.H.to(device=cpu_device)
                W = W.to(device=cpu_device)
                # Objective snapshots were captured before inversion so they
                # still live on the original CUDA device. Keep every enabled
                # scale/clipping objective colocated with the CPU fallback.
                if group_scale_search_diagonal is not None:
                    group_scale_search_diagonal = group_scale_search_diagonal.to(device=cpu_device)
                if group_scale_search_diagonal_prepared is not None:
                    group_scale_search_diagonal_prepared = group_scale_search_diagonal_prepared.to(
                        device=cpu_device
                    )
                if group_scale_search_hessians is not None:
                    group_scale_search_hessians = tuple(
                        hessian.to(device=cpu_device) for hessian in group_scale_search_hessians
                    )
                if clip_hessian_diag is not None:
                    clip_hessian_diag = clip_hessian_diag.to(device=cpu_device)
                if group_loss_hessian_diag is not None:
                    group_loss_hessian_diag = group_loss_hessian_diag.to(device=cpu_device)
                if not uses_group_params:
                    self.quantizer.find_params(W, weight=True, hessian=self.H)
                Hinv, damp = self.hessian_inverse(self.H, release_input=True)
        else:
            Hinv, damp = None, 0.0

        # If the Hessian could not be inverted, the old (pre-VRAM) path did not
        # pass Hessian-derived objective data to find_params or the loss
        # feedback. Drop the captured clones here so the failure path behaves
        # the same and does not retain a full-Hessian worth of memory for a
        # module that will not use error feedback.
        if Hinv is None:
            group_scale_search_diagonal = None
            group_scale_search_diagonal_prepared = None
            group_scale_search_hessians = None
            group_loss_hessian_diag = None

        def group_scale_search_hessian(group_start: int, group_end: int) -> torch.Tensor | None:
            if group_scale_search_diagonal is not None:
                return group_scale_search_diagonal[group_start:group_end]
            if group_scale_search_hessians is not None:
                return group_scale_search_hessians[group_start // self.qcfg.group_size]
            if clip_hessian_diag is not None:
                return clip_hessian_diag[group_start:group_end]
            return None

        def group_gptq_inverse_cholesky(group_start: int, group_end: int) -> torch.Tensor | None:
            """Return a view of the exact correction geometry for one group.

            The view is intentionally not cloned: ``Hinv`` remains live for the
            GPTQ update itself, and copying every group would add avoidable VRAM.
            Earlier-group corrections are already reflected in the working
            weights supplied to the clipping search.
            """

            if (
                Hinv is not None
                and use_adaptive_clipping
                and getattr(clip_cfg, "metric", None) == "gptq_error"
            ):
                return Hinv[group_start:group_end, group_start:group_end]
            return None

        # Loss is only reported after quantization; keep a scalar accumulator
        # instead of a second full weight-sized tensor during GPTQ. Accumulate
        # in FP32 so the diagnostic is stable even when W1 is FP16.
        loss_sum = torch.zeros((), device=W.device, dtype=torch.float32) if Hinv is not None else None
        # The retained full output buffer is not used for further arithmetic,
        # only permutation/slicing/final return. Store it in the final module
        # dtype while keeping block scratch tensors in fp32.
        Q = torch.empty_like(W, dtype=self.module.weight.dtype)

        # Use simplified loop when mock_quantization is active
        if self.qcfg.mock_quantization:
            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2]
                # Mock quantization either replaces Q1 with a vectorized result
                # or fills every column in the fallback path.
                Q1 = torch.empty_like(W1)

                # Handle group quantization parameters efficiently (similar to original)
                if self.qcfg.group_size != -1:
                    if not self.qcfg.static_groups:
                        # Find parameters for entire groups at once (optimized)
                        group_start_cols = list(range(i1, i2, self.qcfg.group_size))
                        for group_start in group_start_cols:
                            group_end = min(group_start + self.qcfg.group_size, self.columns)
                            if group_start < group_end:
                                self.quantizer.find_params(
                                    W[:, group_start:group_end],
                                    weight=True,
                                    hessian=group_scale_search_hessian(group_start, group_end),
                                )
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                    else:
                        # Static groups - use pre-computed groups
                        for i in range(count):
                            idx = i1 + i
                            if self.qcfg.desc_act:
                                idx = perm[idx]
                            self.quantizer = groups[idx // self.qcfg.group_size]

                    # Vectorized quantization for the entire block (major optimization)
                    if len(scale) > 0 and len(zero) > 0:
                        # Use latest scale and zero for the entire block
                        latest_scale = scale[-1]
                        latest_zero = zero[-1]

                        # Vectorized quantization using broadcasting
                        # Reshape scales and zeros to match block dimensions
                        if latest_scale.dim() == 1:
                            latest_scale = latest_scale.view(-1, 1)
                        if latest_zero.dim() == 1:
                            latest_zero = latest_zero.view(-1, 1)

                        # Apply quantization formula using the cloned weights W1
                        maxq_val = 2 ** self.qcfg.bits - 1
                        if self.qcfg.sym:
                            # Symmetric quantization: Q = scale * clamp(round(x/scale), -maxq/2, maxq/2)
                            Q1 = latest_scale * torch.clamp(
                                torch.round(W1 / latest_scale),
                                -(maxq_val // 2),
                                maxq_val // 2
                            )
                        else:
                            # Asymmetric quantization: Q = scale * (clamp(round(x/scale) + zero, 0, maxq) - zero)
                            quantized = torch.clamp(
                                torch.round(W1 / latest_scale) + latest_zero,
                                0,
                                maxq_val
                            )
                            Q1 = latest_scale * (quantized - latest_zero)
                    else:
                        # Fallback to individual quantization if no scale/zero available
                        for i in range(count):
                            w = W1[:, i]
                            q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                            Q1[:, i] = q
                else:
                    # No grouping - vectorized quantization for entire block
                    maxq_val = 2 ** self.qcfg.bits - 1
                    if hasattr(self.quantizer, 'scale') and hasattr(self.quantizer, 'zero'):
                        latest_scale = self.quantizer.scale
                        latest_zero = self.quantizer.zero

                        if latest_scale.dim() == 1:
                            latest_scale = latest_scale.view(-1, 1)
                        if latest_zero.dim() == 1:
                            latest_zero = latest_zero.view(-1, 1)

                        if self.qcfg.sym:
                            Q1 = latest_scale * torch.clamp(
                                torch.round(W1 / latest_scale),
                                -(maxq_val // 2),
                                maxq_val // 2
                            )
                        else:
                            quantized = torch.clamp(
                                torch.round(W1 / latest_scale) + latest_zero,
                                0,
                                maxq_val
                            )
                            Q1 = latest_scale * (quantized - latest_zero)
                    else:
                        # Fallback to individual quantization
                        for i in range(count):
                            w = W1[:, i]
                            q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                            Q1[:, i] = q

                Q[:, i1:i2] = Q1
        else:
            # Original heavy loop for normal quantization
            # RTN fallback has no cross-column Hessian propagation, while online
            # feedback is explicitly group-local. The legacy/static GPTQ path
            # intentionally keeps `blocksize`: changing its partition alters
            # floating-point update order and can change scales and codes.
            effective_block = self._resolve_effective_blocksize(
                blocksize,
                int(self.qcfg.group_size or -1),
                hessian_inverse_available=Hinv is not None,
                use_online_group_damping=use_online_group_damping,
                use_adaptive_clipping=use_adaptive_clipping,
            )

            if use_online_group_damping:
                group_error_ema_decay = damp_cfg.group_error_ema_decay
                group_error_gamma = damp_cfg.group_error_gamma
                group_error_factor_min = damp_cfg.group_error_factor_min
                group_error_factor_max = damp_cfg.group_error_factor_max
                error_update_scale = 1.0
                feedback_target = None
                current_group_start = -1
                current_group_end = -1
                current_group_error_scale = 1.0
            for i1 in range(0, self.columns, effective_block):
                i2 = min(i1 + effective_block, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone().float()
                # Q1 and Err1 are column-complete scratch buffers. Avoid
                # zero-filling them because no element is read before it is
                # assigned by the quantization loop below. Use FP32 so
                # per-column updates keep the same precision as the native
                # CUDA block kernel.
                Q1 = torch.empty_like(W1)
                Err1 = torch.empty_like(W1) if Hinv is not None else None

                if Hinv is not None:
                    Hinv1 = Hinv[i1:i2, i1:i2].clone().contiguous()
                    if use_online_group_damping:
                        group_start = i1
                        group_end = i2
                        actual = count
                        if damp_cfg.group_size_prior_enabled:
                            size_factor = (
                                actual / damp_cfg.group_size_prior_reference
                            ) ** damp_cfg.group_size_prior_beta
                        else:
                            size_factor = 1.0
                        current_group_start = group_start
                        current_group_end = group_end
                        # Experimental online relaxation scales the GPTQ
                        # correction coefficient, not the Hessian inverse.
                        # Larger groups are damped more by dividing the error
                        # scale by the size prior; high group loss reduces the
                        # scale via the inverted feedback exponent.
                        current_group_error_scale = self._group_error_scale(
                            error_update_scale,
                            size_factor,
                            damp_cfg.group_error_scale_min,
                            damp_cfg.group_error_scale_max,
                        )
                        # Online feedback assumes the block is a single group
                        # so the group start/end boundaries are well-defined.
                        assert count <= self.qcfg.group_size, (
                            "Online group feedback requires block size <= group size"
                        )

                group_size = self.qcfg.group_size
                block_maxq_value = getattr(self.quantizer, "_maxq_value", None)
                if block_maxq_value is None:
                    block_maxq_value = int(self.quantizer.maxq.item())
                # Native block kernels implement bounded integer-code GPTQ.
                # Ternary quantization uses the negative maxq sentinel and must
                # retain the distinct eager formula in the serial path below.
                integer_block_quantization = 0 < block_maxq_value <= 255
                batched_scale = batched_zero = None
                batched_group_count = 0
                batched_first_global_idx = 0
                mps_block_eligible = False
                mps_find_params = False
                mps_scale_search = "none"
                mps_scale_search_importance = None
                mps_scale_search_candidates = 0
                mps_scale_search_mse = float(getattr(self.qcfg, "mse", 0.0) or 0.0)
                if (
                    group_size != -1
                    and not self.qcfg.static_groups
                    and group_size <= count
                ):
                    # Batch full groups in this block to amortize Python/kernel launch overhead.
                    full_groups_end = i2 - ((i2 - i1) % group_size)
                    if full_groups_end > i1:
                        batched_first_global_idx = i1 // group_size
                        batched_group_count = (full_groups_end - i1) // group_size
                        batched_last = i1 + batched_group_count * group_size
                        x_3d = W[:, i1:batched_last].reshape(W.shape[0], batched_group_count, group_size)
                        batched_hessian = None
                        batched_hessian_prepared = None
                        if (
                            scale_search in {ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.MARLIN_ACTIVATION}
                            and group_scale_search_diagonal is not None
                        ):
                            # Keep the raw diagonal for fused backends such as
                            # MPS that perform their own normalization.
                            batched_hessian = group_scale_search_diagonal[i1:batched_last].reshape(
                                batched_group_count, group_size
                            )
                            if group_scale_search_diagonal_prepared is not None:
                                batched_hessian_prepared = group_scale_search_diagonal_prepared[
                                    batched_first_global_idx : batched_first_global_idx + batched_group_count
                                ].reshape(batched_group_count, group_size)
                        elif (
                            scale_search in {
                                ScaleSearchConfig.HESSIAN,
                                ScaleSearchConfig.HYBRID,
                                ScaleSearchConfig.MARLIN,
                            }
                            and group_scale_search_hessians is not None
                        ):
                            batched_hessian = torch.stack(
                                group_scale_search_hessians[
                                    batched_first_global_idx : batched_first_global_idx + batched_group_count
                                ],
                                dim=0,
                            )
                        elif clip_hessian_diag is not None:
                            batched_hessian = clip_hessian_diag[i1:batched_last].reshape(
                                batched_group_count, group_size
                            )
                        mps_block_eligible = (
                            _USE_GPTQ_MPS_BLOCK
                            and integer_block_quantization
                            and Hinv is not None
                            and W1.device.type == "mps"
                            and gptq_block_mps_supported()
                            and count <= 128
                            and count % group_size == 0
                            and batched_group_count == count // group_size
                            and not use_online_group_damping
                        )
                        if (
                            mps_block_eligible
                            and _USE_GPTQ_MPS_FUSED_PARAMS
                            and not use_adaptive_clipping
                            # The fused Metal parameter search currently scores
                            # only one rounded affine zero point.  Keep using the
                            # Metal GPTQ block with externally searched params for
                            # asymmetric 2-bit weights so it cannot disagree with
                            # the canonical dual-orientation objective.
                            and not (self.qcfg.bits == 2 and not self.qcfg.sym)
                        ):
                            mps_method = scale_search
                            if mps_method is None and mps_scale_search_mse > 0.0:
                                mps_method = ScaleSearchConfig.MSE
                            elif isinstance(mps_method, str):
                                mps_method = ScaleSearchConfig(mps_method)
                            mps_scale_search_candidates = int(self.quantizer.maxshrink * self.quantizer.grid)
                            if mps_scale_search_mse <= 0.0 or mps_method is None:
                                mps_find_params = True
                            elif mps_scale_search_candidates > 0 and mps_method == ScaleSearchConfig.MSE:
                                mps_find_params = True
                                mps_scale_search = "mse"
                            elif (
                                mps_scale_search_candidates > 0
                                and mps_method == ScaleSearchConfig.ACTIVATION
                                and (batched_hessian is None or batched_hessian.is_contiguous())
                            ):
                                mps_find_params = True
                                if batched_hessian is None:
                                    # The established activation objective falls back
                                    # to ordinary MSE without calibration importance.
                                    mps_scale_search = "mse"
                                else:
                                    mps_scale_search = "activation"
                                    mps_scale_search_importance = batched_hessian
                        if mps_find_params:
                            param_shape = (W1.shape[0], batched_group_count)
                            batched_scale = torch.empty(param_shape, device=W1.device, dtype=torch.float32)
                            batched_zero = torch.empty_like(batched_scale)
                        else:
                            quantizer_hessian = (
                                batched_hessian_prepared
                                if batched_hessian_prepared is not None
                                else batched_hessian
                            )
                            batched_scale, batched_zero = self.quantizer.find_params_batched(
                                x_3d,
                                weight=True,
                                hessian=quantizer_hessian,
                                hessian_prepared=batched_hessian_prepared is not None,
                                gptq_inverse_cholesky=group_gptq_inverse_cholesky(i1, batched_last),
                            )

                # Fast native CUDA path: one kernel launch per block for the
                # common grouped-GPTQ case. Falls back to the serial loop below
                # for unsupported configurations or extension build failures.
                cuda_block_done = False
                if (
                    _USE_GPTQ_CUDA_BLOCK
                    and integer_block_quantization
                    and Hinv is not None
                    and W1.is_cuda
                    and count <= 128
                    and group_size > 0
                    and not self.qcfg.static_groups
                    and count % group_size == 0
                    and batched_group_count == count // group_size
                    and not use_online_group_damping
                ):
                    try:
                        Q1, Err1 = gptq_block_cuda(
                            W1,
                            Hinv1,
                            batched_scale,
                            batched_zero,
                            block_maxq_value,
                            group_size,
                            groupwise=self.quantizer.requires_groupwise_processing(),
                            out=(Q1, Err1),
                        )
                    except Exception as exc:
                        log.warn(
                            f"Quantization: Module `{self.name}` -> CUDA block kernel failed, "
                            f"falling back to serial loop: {exc}"
                        )
                    else:
                        # Append per-group scale/zero for downstream packing/return.
                        if batched_group_count > 0:
                            scale.extend(batched_scale.chunk(batched_group_count, dim=1))
                            zero.extend(batched_zero.chunk(batched_group_count, dim=1))
                            now_idx = batched_first_global_idx + batched_group_count + 1
                        cuda_block_done = True

                # Native Metal path for the same common grouped-GPTQ case.
                # One thread owns each row, preserving the serial dependency
                # between columns without paying one Python dispatch per column.
                mps_block_done = False
                if mps_block_eligible:
                    try:
                        Q1, Err1 = gptq_block_mps(
                            W1,
                            Hinv1,
                            batched_scale,
                            batched_zero,
                            block_maxq_value,
                            group_size,
                            groupwise=self.quantizer.requires_groupwise_processing(),
                            find_params=mps_find_params,
                            symmetric=self.qcfg.sym,
                            scale_search=mps_scale_search,
                            importance=mps_scale_search_importance,
                            candidate_count=mps_scale_search_candidates,
                            grid=self.quantizer.grid,
                            mse=mps_scale_search_mse,
                            out=(Q1, Err1),
                        )
                    except Exception as exc:
                        W1.copy_(W[:, i1:i2])
                        if mps_find_params:
                            batched_scale, batched_zero = self.quantizer.find_params_batched(
                                x_3d,
                                weight=True,
                                hessian=batched_hessian,
                                gptq_inverse_cholesky=group_gptq_inverse_cholesky(i1, batched_last),
                            )
                        log.warning(
                            f"Quantization: Module `{self.name}` -> MPS block kernel failed, "
                            f"falling back to serial loop: {exc}"
                        )
                    else:
                        if batched_group_count > 0:
                            scale.extend(batched_scale.chunk(batched_group_count, dim=1))
                            zero.extend(batched_zero.chunk(batched_group_count, dim=1))
                            now_idx = batched_first_global_idx + batched_group_count + 1
                        mps_block_done = True

                # Compiled CPU block path for the same grouped case.  This is
                # especially important for large MLP shapes like mlp.down where
                # the eager per-column torch.addr loop is slow on CPU.
                cpu_block_done = False
                if (
                    not use_online_group_damping
                    and os.environ.get("GPTQMODEL_BLOCK_CPU", "1") != "0"
                    and integer_block_quantization
                    and not cuda_block_done
                    and not mps_block_done
                    and gptq_block_cpu is not None
                    and Hinv is not None
                    and count <= 128
                    and not self.qcfg.static_groups
                    and W1.device.type == "cpu"
                    and (
                        group_size == -1
                        or (
                            group_size > 0
                            and (
                                (group_size <= count and i1 % group_size == 0)
                                or (
                                    group_size > count
                                    and i2 <= min(((i1 // group_size) + 1) * group_size, self.columns)
                                )
                            )
                        )
                    )
                ):
                    try:
                        groupwise = self.quantizer.requires_groupwise_processing()

                        if group_size == -1:
                            # Per-row/channel scale was computed once for the whole layer.
                            cpu_scale = self.quantizer.scale.view(W1.size(0), -1).contiguous()
                            cpu_zero = self.quantizer.zero.view(W1.size(0), -1).contiguous()
                            cpu_group_size = count
                            cpu_num_groups = 1
                            first_global_group = 0
                        elif group_size > count:
                            # This block is a slice of a larger group. Re-use the
                            # group scale computed at the group start, or compute it
                            # now on the full group columns.
                            group_start = (i1 // group_size) * group_size
                            group_end = min(group_start + group_size, self.columns)
                            if i1 % group_size == 0 or self.quantizer.scale is None:
                                self.quantizer.find_params(
                                    W[:, group_start:group_end],
                                    weight=True,
                                    hessian=group_scale_search_hessian(group_start, group_end),
                                    gptq_inverse_cholesky=group_gptq_inverse_cholesky(group_start, group_end),
                                )
                            cpu_scale = self.quantizer.scale.view(W1.size(0), -1).contiguous()
                            cpu_zero = self.quantizer.zero.view(W1.size(0), -1).contiguous()
                            cpu_group_size = count
                            cpu_num_groups = 1
                            first_global_group = i1 // group_size
                        else:
                            # group_size <= count.  Full groups are already batched;
                            # add a tail scale if the block ends mid-group.
                            num_full_groups = count // group_size
                            tail = count - num_full_groups * group_size
                            first_global_group = i1 // group_size

                            segments_scale = [batched_scale]
                            segments_zero = [batched_zero]
                            if tail > 0:
                                tail_start = i1 + num_full_groups * group_size
                                tail_group_end = min(tail_start + group_size, self.columns)
                                self.quantizer.find_params(
                                    W[:, tail_start:tail_group_end],
                                    weight=True,
                                    hessian=group_scale_search_hessian(tail_start, tail_group_end),
                                    gptq_inverse_cholesky=group_gptq_inverse_cholesky(
                                        tail_start, tail_group_end
                                    ),
                                )
                                tail_scale = self.quantizer.scale.view(W1.size(0), -1)
                                tail_zero = self.quantizer.zero.view(W1.size(0), -1)
                                segments_scale.append(tail_scale)
                                segments_zero.append(tail_zero)

                            if len(segments_scale) == 1:
                                cpu_scale = batched_scale
                                cpu_zero = batched_zero
                            else:
                                cpu_scale = torch.cat(segments_scale, dim=1)
                                cpu_zero = torch.cat(segments_zero, dim=1)
                            cpu_group_size = group_size
                            cpu_num_groups = num_full_groups + (1 if tail > 0 else 0)

                        Q1, Err1 = gptq_block_cpu(
                            W1,
                            Hinv1,
                            cpu_scale,
                            cpu_zero,
                            block_maxq_value,
                            cpu_group_size,
                            groupwise=groupwise,
                        )

                        # Append only the groups that are completed by this block and
                        # have not already been recorded by an earlier fallback path.
                        if first_global_group == now_idx - 1:
                            if group_size == -1:
                                scale.extend([cpu_scale[:, k : k + 1] for k in range(cpu_num_groups)])
                                zero.extend([cpu_zero[:, k : k + 1] for k in range(cpu_num_groups)])
                                now_idx = first_global_group + cpu_num_groups + 1
                            elif group_size > count:
                                group_end = min(first_global_group * group_size + group_size, self.columns)
                                if i2 == group_end or i2 == self.columns:
                                    scale.extend([cpu_scale[:, k : k + 1] for k in range(cpu_num_groups)])
                                    zero.extend([cpu_zero[:, k : k + 1] for k in range(cpu_num_groups)])
                                    now_idx = first_global_group + cpu_num_groups + 1
                            else:
                                completed_groups = num_full_groups
                                if tail > 0:
                                    tail_start = i1 + num_full_groups * group_size
                                    tail_group_end = min(tail_start + group_size, self.columns)
                                    if i2 == tail_group_end or i2 == self.columns:
                                        completed_groups = cpu_num_groups
                                if completed_groups > 0:
                                    scale.extend([cpu_scale[:, k : k + 1] for k in range(completed_groups)])
                                    zero.extend([cpu_zero[:, k : k + 1] for k in range(completed_groups)])
                                    now_idx = first_global_group + completed_groups + 1
                        cpu_block_done = True
                    except Exception as exc:
                        log.warn(
                            f"Quantization: Module `{self.name}` -> CPU block kernel failed, "
                            f"falling back to serial loop: {exc}"
                        )

                if not cuda_block_done and not mps_block_done and not cpu_block_done:
                    for i in range(count):
                        w = W1[:, i]
                        if Hinv is not None:
                            d = Hinv1[i, i]

                        if self.qcfg.group_size != -1:
                            if not self.qcfg.static_groups:
                                if (i1 + i) % self.qcfg.group_size == 0:
                                    group_start = i1 + i
                                    group_end = min(group_start + self.qcfg.group_size, self.columns)
                                    local_group = (group_start - i1) // group_size
                                    if local_group < batched_group_count:
                                        self.quantizer.scale = batched_scale[:, local_group : local_group + 1]
                                        self.quantizer.zero = batched_zero[:, local_group : local_group + 1]
                                    else:
                                        self.quantizer.find_params(
                                            W[:, group_start:group_end],
                                            weight=True,
                                            hessian=group_scale_search_hessian(group_start, group_end),
                                            gptq_inverse_cholesky=group_gptq_inverse_cholesky(
                                                group_start, group_end
                                            ),
                                        )

                                if ((i1 + i) // self.qcfg.group_size) - now_idx == -1:
                                    scale.append(self.quantizer.scale)
                                    zero.append(self.quantizer.zero)
                                    now_idx += 1
                            else:
                                idx = i1 + i
                                if self.qcfg.desc_act:
                                    idx = perm[idx]

                                self.quantizer = groups[idx // self.qcfg.group_size]

                        # Inline the quantizer.quantize() formula to avoid the
                        # per-column Python method-call overhead and keep the
                        # column loop in pure eager tensor dispatch.
                        q_scale = self.quantizer.scale
                        q_zero = self.quantizer.zero
                        q_maxq = getattr(self.quantizer, "_maxq_value", None)
                        if q_maxq is None:
                            q_maxq = int(self.quantizer.maxq.item())
                        q_requires_groupwise = self.quantizer.requires_groupwise_processing()
                        w_col = w.unsqueeze(1)
                        if q_maxq < 0:
                            q = (
                                (w_col > q_scale / 2).to(w_col.dtype) * q_scale
                                + (w_col < q_zero / 2).to(w_col.dtype) * q_zero
                            )
                        elif q_requires_groupwise:
                            q = q_scale * torch.clamp(
                                torch.round(w_col / q_scale), -q_maxq, q_maxq
                            )
                        else:
                            q = q_scale * (
                                torch.clamp(
                                    torch.round(w_col / q_scale) + q_zero, 0, q_maxq
                                )
                                - q_zero
                            )
                        q = q.flatten()
                        Q1[:, i] = q
                        if Hinv is not None:
                            diff = w - q
                            err1 = diff / d
                            if use_online_group_damping:
                                err1.mul_(current_group_error_scale)
                            W1[:, i:] = torch.addr(W1[:, i:], err1, Hinv1[i, i:], alpha=-1.0)
                            Err1[:, i] = err1

                        if (
                            use_online_group_damping
                            and damp_cfg.group_error_enabled
                            and (i1 + i + 1) == current_group_end
                        ):
                            # Measure the unscaled canonical GPTQ correction for
                            # this group and update the experimental EMA target
                            # for the next group. Using the raw coefficient
                            # (before the size prior and feedback scale) decouples
                            # the controller from its own output.
                            local_group_start = max(current_group_start - i1, 0)
                            local_group_end = current_group_end - i1
                            err_slice = Err1[:, local_group_start:local_group_end].float()
                            if damp_cfg.group_error_measure_raw_residual:
                                err_slice = err_slice / current_group_error_scale
                            h_diag_slice = None
                            if group_loss_hessian_diag is not None:
                                h_diag_slice = group_loss_hessian_diag[
                                    current_group_start:current_group_end
                                ].to(err_slice.device)
                            L_g = self._compute_group_loss(
                                err_slice,
                                h_diag_slice,
                                damp_cfg.group_error_use_hessian_weighting,
                            )
                            logged_size_factor = size_factor
                            logged_error_scale = current_group_error_scale
                            group_id = i1 // group_size
                            num_groups = (self.columns + group_size - 1) // group_size
                            feedback_target, error_update_scale = self._update_group_feedback(
                                L_g,
                                feedback_target,
                                group_error_ema_decay,
                                group_error_gamma,
                                group_error_factor_min,
                                group_error_factor_max,
                            )
                            log.info(
                                "Quantization: Module `%s` -> group %d/%d "
                                "loss=%.6f ema_target=%.6f next_error_update_scale=%.4f "
                                "size_factor=%.4f error_scale=%.4f",
                                self.name,
                                group_id,
                                num_groups,
                                L_g,
                                feedback_target,
                                error_update_scale,
                                logged_size_factor,
                                logged_error_scale,
                            )

                Q[:, i1:i2] = Q1.to(W.dtype)
                if Hinv is not None:
                    # Recompute the block loss from Err1 instead of per-column
                    # scalar add_ calls; this avoids thousands of tiny syncs.
                    loss_sum.add_((Err1.float() ** 2).sum() / 2)
                    # Update the remaining weights in-place with a single fused
                    # addmm instead of matmul+sub, avoiding a temporary tensor.
                    torch.addmm(W[:, i2:], Err1, Hinv[i1:i2, i2:], alpha=-1, out=W[:, i2:])

                del W1, Q1, Err1
                if Hinv is not None:
                    del Hinv1

        # TODO: why is there a torch_sync here? There are no streaming ops here?
        # torch_sync(device=self.module.target_device)

        if Hinv is not None:
            del Hinv
            if self.nsamples != 0:
                loss_sum_item = loss_sum.item()
                avg_loss = loss_sum_item / self.nsamples

                if math.isnan(avg_loss):
                    print("Losses sum item:", loss_sum_item)
                    if self.qcfg.mock_quantization:
                        # Mock retry already failed; fall back to a data-independent strategy
                        # rather than recursing forever.
                        if fallback_configured:
                            log.warn(
                                f"Quantization: mock retry also produced `NaN` loss for `{self.name}`; "
                                f"using `{resolved_strategy.value}` fallback."
                            )
                            return self._fallback_quantize(resolved_strategy, blocksize)
                        else:
                            raise ValueError(
                                f"Quantization: Failed due to `NaN` loss for `{self.name}`; "
                                "please try increasing calibration data samples or enable fallback=True"
                            )
                    if fallback_configured:
                        log.info(
                            f"Quantization: Failed due to `NaN` loss for `{self.name}`, "
                            f"use mock quantization retry for `{self.name}`."
                        )
                        self.qcfg.mock_quantization = True
                        return self.quantize(blocksize=blocksize)
                    else:
                        raise ValueError(
                            f"Quantization: Failed due to `NaN` loss for `{self.name}`; "
                            "please try increasing calibration data samples or enable fallback=True."
                        )
            else:
                if fallback_configured:
                    log.warn(
                        f"Quantization: Module `{self.name}` -> using fail safe mode. "
                        "Please check if calibration data is sufficient."
                    )
                else:
                    log.warn(f"Quantization: `{self.name}` is not activated due to model inference logic (MoE)")
                avg_loss = f"{resolved_strategy.value} fallback" if fallback_configured else 999999999
        else:
            avg_loss = f"{resolved_strategy.value} fallback" if fallback_configured else 999999999

        if loss_sum is not None:
            del loss_sum
        del self.H
        del W

        group_size = self.qcfg.group_size if self.qcfg.group_size != -1 else self.columns

        if self.qcfg.static_groups and self.qcfg.desc_act:
            g_idx = self.build_group_index(
                self.columns,
                group_size,
                Q.device,
                source_perm=perm,
            )
        else:
            g_idx = self.build_group_index(self.columns, group_size, Q.device)

        if self.qcfg.desc_act and use_hessian:
            invperm = invperm.to(device=Q.device)
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]
            del perm, invperm

        elif self.qcfg.act_group_aware and use_hessian:
            inv_final = invert_perm(final_perm).to(device=Q.device)
            Q = Q[:, inv_final]
            if not self.qcfg.static_groups:
                inv_global_perm = invert_perm(global_perm)
                inv_global_perm_list = inv_global_perm.tolist()
                reordered_group_count = len(inv_global_perm_list)
                temp_scale = [scale[i] for i in inv_global_perm_list]
                temp_scale.extend(scale[reordered_group_count:])
                scale = temp_scale
                temp_zero = [zero[i] for i in inv_global_perm_list]
                temp_zero.extend(zero[reordered_group_count:])
                zero = temp_zero
                del inv_global_perm, inv_global_perm_list
            del final_perm, inv_final, global_perm

        if adjacent_model is not None:
            if adjacent_reference_weight is None or adjacent_reference_hessian is None:
                raise AssertionError("Adjacent whole-model references were not captured.")
            from .adjacent_model import apply_adjacent_model_hybrid

            Q, adjacent_stats = apply_adjacent_model_hybrid(
                module_name=self.name,
                weight=adjacent_reference_weight,
                hessian=adjacent_reference_hessian,
                classic_quantized=Q,
                scale_parts=scale,
                zero_parts=zero,
                bits=self.qcfg.bits,
                group_size=self.qcfg.group_size,
                config=adjacent_model,
            )
            adjacent_model.record(adjacent_stats)
            del adjacent_reference_weight, adjacent_reference_hessian

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            Q = Q[:, :valid_cols]
            g_idx = g_idx[:valid_cols]

        if isinstance(self.module, transformers.Conv1D):
            Q = Q.t()

        if Q.shape != self.module.weight.shape:
            Q = Q.reshape(self.module.weight.shape).to(self.module.weight.dtype)
        else:
            Q = Q.to(self.module.weight.dtype)

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)

        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        if self._tp_pad_cols:
            valid_cols = self._original_columns
            scale = self.truncate_last_dim(scale, valid_cols)
            zero = self.truncate_last_dim(zero, valid_cols)

        if cpu_fallback_used and Q.device != result_device:
            log.info(
                "Quantization: Module `%s` -> CPU fallback complete; moving final quantized weights back to %s.",
                self.name,
                result_device,
            )

        Q = Q.to(device=result_device, non_blocking=False)

        duration = time.time() - start

        # Quantization is complete; retain the authoritative ``nsamples`` field
        # but release per-device counters that were only needed to materialize
        # (or detect an invalid mock-quantization rebuild of) the Hessian.
        with self.lock:
            self._device_sample_counts.clear()
            self._device_sequence_counts.clear()

        return Q, scale, zero, g_idx, duration, avg_loss, damp, self.nsamples

    def borrow_materialized_chunk_stats(self, reset: bool = False) -> Dict[str, int]:
        stats = dict(self._borrow_workspace_stats)
        if reset:
            for key in self._borrow_workspace_stats:
                self._borrow_workspace_stats[key] = 0
        return stats

    def _snapshot_borrow_workspace_stats(self, *, context: str) -> None:
        stats = self.borrow_materialized_chunk_stats(reset=True)
        total_requests = int(stats.get("requests", 0) or 0)
        if total_requests == 0:
            return

        materialized_hits = int(stats.get("materialized_hits", 0) or 0)
        materialized_misses = int(stats.get("materialized_misses", 0) or 0)
        staging_hits = int(stats.get("staging_hits", 0) or 0)
        staging_misses = int(stats.get("staging_misses", 0) or 0)
        chunk_rows = self._borrow_workspace_last_chunk_rows
        stage_dtype = self._borrow_workspace_stage_dtype
        stage_dtype_str = str(stage_dtype) if stage_dtype is not None else "n/a"
        hit_rate = materialized_hits / total_requests if total_requests else 0.0

        summary = {
            "context": context,
            "requests": total_requests,
            "materialized_hits": materialized_hits,
            "materialized_misses": materialized_misses,
            "staging_hits": staging_hits,
            "staging_misses": staging_misses,
            "chunk_rows": chunk_rows,
            "staging_dtype": stage_dtype_str,
            "hit_rate": hit_rate,
        }
        self._borrow_workspace_last_summary = summary

        totals = self._borrow_workspace_totals
        totals["requests"] += total_requests
        totals["materialized_hits"] += materialized_hits
        totals["materialized_misses"] += materialized_misses
        totals["staging_hits"] += staging_hits
        totals["staging_misses"] += staging_misses

    def log_workspace_stats(self, *, context: str, reset: bool = True) -> None:
        totals = self._borrow_workspace_totals
        total_requests = int(totals.get("requests", 0) or 0)
        if total_requests == 0:
            if reset:
                self.reset_workspace_stats()
            return

        total_hits = int(totals.get("materialized_hits", 0) or 0)
        total_misses = int(totals.get("materialized_misses", 0) or 0)
        total_hit_rate = total_hits / total_requests if total_requests else 0.0

        last = self._borrow_workspace_last_summary or {}
        last_requests = int(last.get("requests", 0) or 0)
        last_hits = int(last.get("materialized_hits", 0) or 0)
        last_misses = int(last.get("materialized_misses", 0) or 0)
        last_hit_rate = float(last.get("hit_rate", 0.0) or 0.0)
        rows_label = last.get("chunk_rows", "n/a")
        stage_dtype = last.get("staging_dtype", "n/a")

        log.info(
            "GPTQ workspace cache [%s]: module=%s rows=%s staging_dtype=%s "
            "requests=%d hits=%d misses=%d hit_rate=%.2f total_requests=%d "
            "total_hits=%d total_misses=%d total_hit_rate=%.2f",
            context,
            getattr(self, "name", "<unknown>"),
            rows_label,
            stage_dtype,
            last_requests,
            last_hits,
            last_misses,
            last_hit_rate,
            total_requests,
            total_hits,
            total_misses,
            total_hit_rate,
        )

        if reset:
            self.reset_workspace_stats()

    def reset_workspace_stats(self) -> None:
        for key in self._borrow_workspace_stats:
            self._borrow_workspace_stats[key] = 0
        for key in self._borrow_workspace_totals:
            self._borrow_workspace_totals[key] = 0
        self._borrow_workspace_last_summary = None
        self._borrow_workspace_stage_dtype = None
        self._borrow_workspace_last_chunk_rows = None

    def free(self):
        with self.lock:
            self._device_hessian_partials.clear()
            self._device_sample_counts.clear()
            self._device_sequence_counts.clear()
            self._hessian_dirty = False
        if hasattr(self, "H"):
            del self.H
        if hasattr(self, "_H_diag"):
            del self._H_diag
        del self.quantizer
        if hasattr(self, "module_copy"):
            del self.module_copy

        if self._named_module is not None:
            self._named_module.state.pop("tp_pad_info", None)

        target = getattr(self, "module", None)
        if target is not None:
            del self.module

        # torch_empty_cache(self.device)


__all__ = ["GPTQ"]
