# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Transactional decoder-layer output alignment for fixed QVQ trellises."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..models.base import module_tree_flags_are_moe
from ..nn_modules.hooked_linear import HookedLinear
from ..nn_modules.qlinear.qvq import (
    _FP16_STABLE_HADAMARD_MIN_WIDTH,
    QVQLinear,
    _qvq_fp16_emulated_hadamard_fallback,
)
from ..quantization.config import OutputAlignConfig
from ..quantization.qvq import reconstruct_qvq_inner_weight, rht_reconstruct_weight
from ..quantization.rotation.hadamard_utils import matmul_hadU, matmul_hadU_stable
from ..utils.attn_mask import apply_keep_mask_bt, normalize_seq_mask
from ..utils.backend import BACKEND
from ..utils.logger import setup_logger
from ..utils.model import recurse_getattr, recurse_setattr
from ..utils.module_locks import parent_module_lock
from .named_module import NamedModule

log = setup_logger()


@dataclass
class _ReplayBatch:
    inputs: list[torch.Tensor]
    input_kwargs: dict[str, Any]
    target: torch.Tensor
    position_ids: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]


@dataclass
class _LayerAlignmentState:
    modules: dict[str, NamedModule] = field(default_factory=dict)
    layer_module: Optional[nn.Module] = None
    replay_batches: list[_ReplayBatch] = field(default_factory=list)
    alignment_passes: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


class _FixedTrellisAlignmentLinear(nn.Module):
    """Differentiable QVQ linear whose decoded trellis exists only in this stage."""

    def __init__(
        self,
        *,
        inner_weight: torch.Tensor,
        SU: torch.Tensor,
        SV: torch.Tensor,
        bias: Optional[torch.Tensor],
        output_dtype: Optional[torch.dtype],
    ) -> None:
        super().__init__()
        self.register_buffer("inner_weight", inner_weight.detach().to(torch.float32))
        self.SU = nn.Parameter(SU.detach().to(device=inner_weight.device, dtype=torch.float32).clone())
        self.SV = nn.Parameter(SV.detach().to(device=inner_weight.device, dtype=torch.float32).clone())
        self.register_buffer(
            "bias",
            None if bias is None else bias.detach().to(device=inner_weight.device, dtype=torch.float32).clone(),
        )
        self.output_dtype = output_dtype

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_shape = inputs.shape
        work = inputs.reshape(-1, input_shape[-1])
        # Production QVQLinear returns the activation dtype. CUDA alone needs
        # the remembered module dtype to select its range-safe FP16 surrogate
        # while replay may be entered through autocast with FP32 inputs.
        # MPS has no equivalent autocast here; using a stale/materialization
        # weight dtype can turn one Q/K/V projection FP32 and break attention.
        output_dtype = (
            self.output_dtype
            if inputs.device.type == "cuda" and self.output_dtype is not None
            else inputs.dtype
        )
        fp16_cuda = work.device.type == "cuda" and output_dtype == torch.float16
        if fp16_cuda:
            # Match the production FP16 decoder's operation ordering while
            # keeping the large inner reduction in FP32. The previous FP32
            # ordinary-Hadamard surrogate could cross the FP16 range boundary
            # even when the stable production path remained finite, aborting
            # a valid fixed-trellis alignment pass.
            # Keep the pre-scale in FP32 as well.  The production range-safe
            # Hadamard path normalizes before any FP16 narrowing, so a finite
            # value such as 60,000 multiplied by SU=2 must not become inf here.
            transformed = work.to(torch.float32) * self.SU
            transformed = (
                matmul_hadU_stable(transformed)
                if transformed.shape[-1] >= _FP16_STABLE_HADAMARD_MIN_WIDTH
                else matmul_hadU(transformed)
            )
            # The surrounding alignment pass runs under model autocast. Merely
            # converting the operands to FP32 is insufficient: CUDA autocast
            # narrows matmul back to FP16, which can overflow the factorized
            # QVQ inner result before the output Hadamard/SV scale restores its
            # intended range. Disable autocast around this reduction so the
            # differentiable surrogate matches the production kernel's FP32
            # accumulator *and* FP32 output contract.
            with torch.autocast(device_type=work.device.type, enabled=False):
                work = transformed.to(torch.float32) @ self.inner_weight
            work = _qvq_fp16_emulated_hadamard_fallback(
                work,
                post_scale=self.SV,
                bias=self.bias,
                scale_mode=3 if work.shape[-1] >= _FP16_STABLE_HADAMARD_MIN_WIDTH else 4,
            )
        else:
            work = work.to(torch.float32)
            work = matmul_hadU(work * self.SU)
            with torch.autocast(device_type=work.device.type, enabled=False):
                work = work @ self.inner_weight
            work = matmul_hadU(work) * self.SV
            if self.bias is not None:
                work = work + self.bias
        return work.reshape(*input_shape[:-1], self.SV.numel()).to(output_dtype)


class QVQOutputAlignmentAttachment:
    """Attach fixed-trellis output correction to existing looper hook points.

    The attachment never changes the QVQ codec or inference path. It owns all
    replay state, temporary decoded weights, optimization, validation, and
    rollback so the already-complex layer lifecycle only delegates to its
    generic grouped-optimization hooks.
    """

    def __init__(self, config: OutputAlignConfig) -> None:
        self.config = config
        self.gptq_model = None
        self._layers: dict[int, _LayerAlignmentState] = {}
        # QTIP's blockwise phase trains each dense decoder layer on that
        # layer's pristine input and output. The main quantization lifecycle
        # intentionally advances a separate noisy stream for Hessian/error
        # replay, so keep the clean stream entirely inside this attachment.
        # Only the current/next layer can be live; entries are transferred
        # into bounded replay state and removed as their layer is captured.
        self._clean_layer_inputs: dict[int, list[list[torch.Tensor]]] = {}
        self._active_clean_layer_inputs: dict[int, list[list[torch.Tensor]]] = {}
        self._lock = threading.RLock()

    def bind_model(self, gptq_model) -> None:
        self.gptq_model = gptq_model

    def register_module(self, module: NamedModule) -> None:
        state = self._layer_state(module.layer_index)
        with state.lock:
            state.modules[module.name] = module

    def _layer_state(self, layer_index: int) -> _LayerAlignmentState:
        with self._lock:
            return self._layers.setdefault(layer_index, _LayerAlignmentState())

    @staticmethod
    def _cpu_clone(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            with torch.inference_mode(False):
                return value.detach().to(device="cpu", copy=True)
        if isinstance(value, dict):
            return {key: QVQOutputAlignmentAttachment._cpu_clone(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [QVQOutputAlignmentAttachment._cpu_clone(inner) for inner in value]
        if isinstance(value, tuple):
            return tuple(QVQOutputAlignmentAttachment._cpu_clone(inner) for inner in value)
        return value

    @staticmethod
    def _primary(value: Any) -> torch.Tensor:
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("QVQ output alignment received an empty layer output.")
            value = value[0]
        if not isinstance(value, torch.Tensor):
            raise TypeError("QVQ output alignment requires tensor decoder-layer outputs.")
        return value

    def receive_pristine_layer_module(self, *, layer_index: int, layer_module: nn.Module) -> None:
        state = self._layer_state(layer_index)
        with state.lock:
            if state.layer_module is None:
                state.layer_module = layer_module

    def clean_group_layer_inputs(
        self,
        *,
        layer_index: int,
        layer_inputs: list[list[torch.Tensor]],
    ) -> list[list[torch.Tensor]]:
        """Return and retain QTIP's pristine input stream for one layer."""

        with self._lock:
            clean_inputs = self._clean_layer_inputs.pop(layer_index, layer_inputs)
            self._active_clean_layer_inputs[layer_index] = clean_inputs
        return clean_inputs

    def receive_clean_layer_inputs(
        self,
        *,
        layer_index: int,
        layer_inputs: list[list[torch.Tensor]],
    ) -> None:
        """Advance one CPU-owned pristine stream without touching noisy replay."""

        with self._lock:
            self._clean_layer_inputs[layer_index + 1] = self._cpu_clone(layer_inputs)

    def receive_layer_forward_context(
        self,
        *,
        layer_index: int,
        layer_inputs: list[list[torch.Tensor]],
        layer_input_kwargs: list[dict[str, torch.Tensor]],
        layer_outputs: list[list[torch.Tensor]],
        position_ids: list[Optional[torch.Tensor]],
        attention_masks: list[Optional[torch.Tensor]],
    ) -> None:
        if len(layer_inputs) != len(layer_outputs):
            raise RuntimeError("QVQ output alignment requires aligned layer input/output batch counts.")
        if layer_input_kwargs and len(layer_input_kwargs) != len(layer_inputs):
            raise RuntimeError("QVQ output alignment requires aligned layer input kwargs.")

        with self._lock:
            # The lifecycle passes its noisy stream here because other grouped
            # optimizers deliberately train on noisy inputs. QVQ follows the
            # authoritative QTIP clean-input/clean-target contract instead.
            layer_inputs = self._active_clean_layer_inputs.pop(layer_index, layer_inputs)

        maximum = self.config.maximum_train_batches + self.config.maximum_validation_batches
        state = self._layer_state(layer_index)
        with state.lock:
            if state.replay_batches:
                return
            for index, (input_batch, output_batch) in enumerate(zip(layer_inputs, layer_outputs)):
                if len(state.replay_batches) >= maximum:
                    break
                kwargs = layer_input_kwargs[index] if layer_input_kwargs else {}
                state.replay_batches.append(
                    _ReplayBatch(
                        inputs=self._cpu_clone(input_batch),
                        input_kwargs=self._cpu_clone(kwargs),
                        target=self._cpu_clone(self._primary(output_batch)),
                        position_ids=self._cpu_clone(position_ids[index]) if index < len(position_ids) else None,
                        attention_mask=(
                            self._cpu_clone(attention_masks[index]) if index < len(attention_masks) else None
                        ),
                    )
                )

    def _split_batches(self, batches: list[_ReplayBatch]) -> tuple[list[_ReplayBatch], list[_ReplayBatch]]:
        if len(batches) < 2:
            raise RuntimeError("QVQ output alignment requires at least two calibration batches for disjoint validation.")
        validation_count = max(1, math.ceil(len(batches) * self.config.validation_fraction))
        validation_count = min(validation_count, self.config.maximum_validation_batches, len(batches) - 1)
        train_count = min(len(batches) - validation_count, self.config.maximum_train_batches)
        return batches[:train_count], batches[-validation_count:]

    @staticmethod
    def _move_value(value: Any, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            with torch.inference_mode(False):
                return value.to(device=device, copy=True)
        if isinstance(value, dict):
            return {
                key: QVQOutputAlignmentAttachment._move_value(inner, device)
                for key, inner in value.items()
            }
        if isinstance(value, list):
            return [QVQOutputAlignmentAttachment._move_value(inner, device) for inner in value]
        if isinstance(value, tuple):
            return tuple(QVQOutputAlignmentAttachment._move_value(inner, device) for inner in value)
        return value

    def _forward_batch(self, layer: nn.Module, batch: _ReplayBatch, device: torch.device) -> torch.Tensor:
        if self.gptq_model is None:
            raise RuntimeError("QVQ output alignment is not bound to a model.")
        inputs = [self._move_value(value, device) for value in batch.inputs]
        additional_inputs = {
            key: self._move_value(value, device)
            for key, value in batch.input_kwargs.items()
            if key not in {"past_key_values", "past_key_value"}
        }
        if batch.attention_mask is not None:
            additional_inputs["attention_mask"] = self._move_value(batch.attention_mask, device)
        else:
            additional_inputs.setdefault("attention_mask", None)
        if batch.position_ids is not None:
            additional_inputs["position_ids"] = self._move_value(batch.position_ids, device)
        additional_inputs["use_cache"] = False
        additional_inputs = self.gptq_model.prepare_layer_replay_kwargs(
            layer=layer,
            layer_input=inputs,
            additional_inputs=additional_inputs,
            target_device=device,
        )
        primary_dtype = inputs[0].dtype if inputs and isinstance(inputs[0], torch.Tensor) else None
        autocast_enabled = device.type == "cuda" and primary_dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(
            device_type=device.type,
            dtype=primary_dtype if autocast_enabled else None,
            enabled=autocast_enabled,
        ):
            output = layer(*inputs, **additional_inputs)
        return self._primary(output)

    @staticmethod
    def _masked_loss(prediction: torch.Tensor, target: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        target = target.to(device=prediction.device, dtype=prediction.dtype)
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"QVQ output alignment prediction shape {tuple(prediction.shape)} does not match "
                f"target {tuple(target.shape)}."
            )
        keep_mask = None
        if attention_mask is not None and prediction.ndim >= 3:
            keep_mask = normalize_seq_mask(
                attention_mask.to(device=prediction.device),
                seq_len=prediction.shape[1],
            )
        prediction = apply_keep_mask_bt(prediction, keep_mask)
        target = apply_keep_mask_bt(target, keep_mask)
        if prediction.numel() == 0:
            raise RuntimeError("QVQ output alignment validation contains no non-padding values.")
        return F.mse_loss(prediction.to(torch.float32), target.to(torch.float32))

    def _evaluate(self, layer: nn.Module, batches: list[_ReplayBatch], device: torch.device) -> float:
        total = 0.0
        with torch.no_grad():
            for batch in batches:
                prediction = self._forward_batch(layer, batch, device)
                total += float(self._masked_loss(prediction, batch.target, batch.attention_mask).item())
        result = total / len(batches)
        if not math.isfinite(result):
            raise RuntimeError("QVQ output alignment produced a non-finite validation loss.")
        return result

    @staticmethod
    def _rebuild_inference_tensors(layer: nn.Module) -> None:
        """Make frozen layer tensors autograd-saveable without changing their values."""
        with torch.inference_mode(False):
            for child in layer.modules():
                for name, parameter in list(child._parameters.items()):
                    if parameter is not None and parameter.is_inference():
                        child._parameters[name] = nn.Parameter(
                            parameter.detach().clone(),
                            requires_grad=parameter.requires_grad,
                        )
                for name, buffer in list(child._buffers.items()):
                    if buffer is not None and buffer.is_inference():
                        child._buffers[name] = buffer.detach().clone()

    def _build_temporary_module(self, module: NamedModule, device: torch.device) -> _FixedTrellisAlignmentLinear:
        if not isinstance(module.module, nn.Linear):
            raise NotImplementedError(
                "QVQ output alignment currently supports dense nn.Linear decoder leaves only; "
                f"`{module.full_name}` is {module.module.__class__.__name__}."
            )
        module.stream_sync()
        with parent_module_lock(module.full_name):
            runtime_config = module.state["_qvq_runtime_config"]
            bits, codebook = runtime_config[:2]
            vector_size = runtime_config[2] if len(runtime_config) > 2 else 2
            trellis_window = runtime_config[4] if len(runtime_config) > 4 else 16
            dual_v2 = runtime_config[5] if len(runtime_config) > 5 else False
            v2b4_p64 = runtime_config[6] if len(runtime_config) > 6 else False
            v2b2_p32 = runtime_config[7] if len(runtime_config) > 7 else False
            v2b2_p32_lr = runtime_config[8] if len(runtime_config) > 8 else False
            trellis = module.state["trellis"].to(device=device)
            SU = module.state["SU"].to(device=device)
            SV = module.state["SV"].to(device=device)
            bank_ids = module.state.get("bank_ids")
            if bank_ids is not None:
                bank_ids = bank_ids.to(device=device)
            bank_alt_id = module.state.get("bank_alt_id")
            if bank_alt_id is not None:
                bank_alt_id = bank_alt_id.to(device=device)
        inner = reconstruct_qvq_inner_weight(
            trellis,
            bits=bits,
            in_features=module.module.in_features,
            out_features=module.module.out_features,
            codebook_version=codebook,
            vector_size=vector_size,
            trellis_window=trellis_window,
            bank_ids=bank_ids,
            dual_v2=dual_v2,
            v2b4_p64=v2b4_p64,
            v2b2_p32=v2b2_p32,
            v2b2_p32_lr=v2b2_p32_lr,
            bank_alt_id=bank_alt_id,
        )
        return _FixedTrellisAlignmentLinear(
            inner_weight=inner,
            SU=SU,
            SV=SV,
            bias=module.module.bias,
            output_dtype=module.module_dtype,
        )

    def _build_runtime_module(
        self,
        module: NamedModule,
        device: torch.device,
        *,
        SU: torch.Tensor,
        SV: torch.Tensor,
    ) -> QVQLinear:
        """Build the real decoder solely for the layer-scope acceptance gate."""

        with parent_module_lock(module.full_name):
            runtime_config = module.state["_qvq_runtime_config"]
            bits, codebook = runtime_config[:2]
            vector_size = runtime_config[2] if len(runtime_config) > 2 else 2
            bank_count = runtime_config[3] if len(runtime_config) > 3 else 1
            trellis_window = runtime_config[4] if len(runtime_config) > 4 else 16
            dual_v2 = runtime_config[5] if len(runtime_config) > 5 else False
            v2b4_p64 = runtime_config[6] if len(runtime_config) > 6 else False
            v2b2_p32 = runtime_config[7] if len(runtime_config) > 7 else False
            v2b2_p32_lr = runtime_config[8] if len(runtime_config) > 8 else False
            trellis = module.state["trellis"].to(device=device)
            bank_ids = module.state.get("bank_ids")
            if bank_ids is not None:
                bank_ids = bank_ids.to(device=device)
            bank_alt_id = module.state.get("bank_alt_id")
            if bank_alt_id is not None:
                bank_alt_id = bank_alt_id.to(device=device)
        runtime = QVQLinear(
            bits=bits,
            in_features=module.module.in_features,
            out_features=module.module.out_features,
            bias=module.module.bias is not None,
            backend=BACKEND.QVQ,
            name=module.full_name,
            dtype=module.module_dtype,
            tensors={
                "trellis": trellis,
                "SU": SU.detach().to(device=device, dtype=torch.float32),
                "SV": SV.detach().to(device=device, dtype=torch.float32),
                **({} if bank_ids is None else {"bank_ids": bank_ids}),
                **({} if bank_alt_id is None else {"bank_alt_id": bank_alt_id}),
                **(
                    {}
                    if module.module.bias is None
                    else {"bias": module.module.bias.detach().to(device=device).clone()}
                ),
            },
            codebook_version=codebook,
            vector_size=vector_size,
            trellis_window=trellis_window,
            bank_count=bank_count,
            dual_v2=dual_v2,
            v2b4_p64=v2b4_p64,
            v2b2_p32=v2b2_p32,
            v2b2_p32_lr=v2b2_p32_lr,
        )
        runtime.eval()
        runtime.post_init()
        return runtime

    def _candidate_weights(
        self,
        temporary_modules: dict[str, _FixedTrellisAlignmentLinear],
    ) -> dict[str, torch.Tensor]:
        return {
            name: rht_reconstruct_weight(
                temporary.inner_weight,
                temporary.SU,
                temporary.SV,
            )
            for name, temporary in temporary_modules.items()
        }

    @staticmethod
    def _install_differentiable_dense_modules(
        layer: nn.Module,
        *,
        excluded_names: set[str],
    ) -> dict[str, HookedLinear]:
        """Temporarily bypass inference-only capture wrappers during autograd."""

        originals = {}
        for name, child in list(layer.named_modules()):
            if not name or name in excluded_names or not isinstance(child, HookedLinear):
                continue
            replacement = nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            replacement.weight = child.weight
            replacement.bias = child.bias
            replacement.train(child.training)
            recurse_setattr(layer, name, replacement)
            originals[name] = child
        return originals

    def align_layer(self, layer_index: int, *, finalize: bool = True) -> Optional[dict[str, float]]:
        """Run autograd with inference mode explicitly disabled by this attachment."""

        with torch.inference_mode(False):
            return self._align_layer_impl(layer_index, finalize=finalize)

    def layer_is_fully_staged(self, layer_index: int) -> bool:
        """Return whether every registered target owns a complete fixed codec."""

        with self._lock:
            state = self._layers.get(layer_index)
        if state is None:
            return False
        with state.lock:
            return bool(state.modules) and all(
                {"trellis", "SU", "SV", "_qvq_runtime_config"}.issubset(module.state)
                for module in state.modules.values()
            )

    def modules_are_fully_staged(self, layer_index: int, module_names: set[str]) -> bool:
        """Return whether every module in one completed subset owns a codec."""

        with self._lock:
            state = self._layers.get(layer_index)
        if state is None or not module_names:
            return False
        with state.lock:
            return all(
                name in state.modules
                and {"trellis", "SU", "SV", "_qvq_runtime_config"}.issubset(state.modules[name].state)
                for name in module_names
            )

    def discard_layer(self, layer_index: int) -> None:
        """Drop bounded replay state when the enclosing subset did not finish."""

        with self._lock:
            self._layers.pop(layer_index, None)

    def _align_layer_impl(self, layer_index: int, *, finalize: bool) -> Optional[dict[str, float]]:
        started = time.perf_counter()
        with self._lock:
            state = self._layers.get(layer_index)
            if finalize:
                self._layers.pop(layer_index, None)
        if state is None:
            return None
        with state.lock:
            modules = [
                module
                for module in state.modules.values()
                if {"trellis", "SU", "SV", "_qvq_runtime_config"}.issubset(module.state)
            ]
            layer = state.layer_module
            batches = list(state.replay_batches)
            state.alignment_passes += 1
            alignment_pass = state.alignment_passes
        if not modules:
            return None
        if layer is None:
            raise RuntimeError(f"QVQ output alignment did not capture decoder layer {layer_index}.")
        if any(module_tree_flags_are_moe(module.state.get("module_tree_flags", frozenset())) for module in modules):
            raise NotImplementedError(
                "QVQ output alignment does not yet support MoE decoder layers; module-tree tags identified expert/router leaves."
            )

        train_batches, validation_batches = self._split_batches(batches)
        device = modules[0].module.weight.device
        if any(module.module.weight.device != device for module in modules):
            raise RuntimeError("QVQ output alignment requires one decoder layer's modules to share a device.")

        original_training = layer.training
        original_requires_grad: list[tuple[nn.Parameter, bool]] = []
        original_parameters: list[tuple[nn.Parameter, torch.Tensor]] = []
        original_buffers: list[tuple[nn.Module, str, torch.device, torch.Tensor]] = []
        original_modules: dict[str, nn.Module] = {}
        original_weights: dict[str, torch.Tensor] = {}
        original_staged: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        temporary_modules: dict[str, _FixedTrellisAlignmentLinear] = {}
        original_dense_wrappers: dict[str, HookedLinear] = {}
        candidate_installed = False
        try:
            baseline_loss = self._evaluate(layer, validation_batches, device)
            self._rebuild_inference_tensors(layer)
            original_requires_grad = [
                (parameter, parameter.requires_grad)
                for parameter in layer.parameters()
            ]
            original_parameters = [
                (parameter, parameter.detach().to(device="cpu", copy=True))
                for parameter in layer.parameters()
            ]
            original_buffers = [
                (child, name, buffer.device, buffer.detach().to(device="cpu", copy=True))
                for child in layer.modules()
                for name, buffer in child._buffers.items()
                if buffer is not None
            ]
            # QTIP trains the mixed decoder layer in FP32 and autocasts its
            # forwards to the model dtype. Keep that numerical contract while
            # restoring every live parameter/buffer dtype before returning.
            layer.float()
            original_dense_wrappers = self._install_differentiable_dense_modules(
                layer,
                excluded_names={module.name for module in modules},
            )

            for module in modules:
                original = recurse_getattr(layer, module.name)
                if original is not module.module:
                    raise RuntimeError(f"QVQ output alignment lost live module identity for `{module.full_name}`.")
                original_modules[module.name] = original
                original_weights[module.name] = original.weight.detach().to(device="cpu", copy=True)
                module.stream_sync()
                with parent_module_lock(module.full_name):
                    original_staged[module.name] = (
                        module.state["SU"].clone(),
                        module.state["SV"].clone(),
                    )
                temporary = self._build_temporary_module(module, device)
                recurse_setattr(layer, module.name, temporary)
                temporary_modules[module.name] = temporary

            layer.eval()
            for module in modules:
                temporary = temporary_modules[module.name]
                recurse_setattr(
                    layer,
                    module.name,
                    self._build_runtime_module(
                        module,
                        device,
                        SU=temporary.SU,
                        SV=temporary.SV,
                    ),
                )
            runtime_baseline_loss = self._evaluate(layer, validation_batches, device)
            for name, temporary in temporary_modules.items():
                recurse_setattr(layer, name, temporary)

            # Output alignment is an SV/output-channel correction.  SU is the
            # fixed RHT sign transform that defines the serialized codec and
            # must not become a learned dense vector during the temporary
            # replay.  Keep later dense siblings trainable for sequential
            # compensation, but freeze every staged QVQ SU parameter.
            optimization_parameters = list(layer.parameters())
            for parameter in optimization_parameters:
                parameter.requires_grad_(True)
            for temporary in temporary_modules.values():
                temporary.SU.requires_grad_(False)
            optimization_parameters = [parameter for parameter in optimization_parameters if parameter.requires_grad]
            if not optimization_parameters:
                raise RuntimeError("QVQ output alignment found no trainable SV or dense compensation parameters.")
            optimizer_class = torch.optim.AdamW if self.config.optimizer == "adamw" else torch.optim.Adam
            optimizer = optimizer_class(
                optimization_parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            gradient_scaling = device.type == "cuda" and modules[0].module_dtype == torch.float16
            scaler = torch.amp.GradScaler(device.type, enabled=gradient_scaling)
            best_loss = self._evaluate(layer, validation_batches, device)
            best_state = [parameter.detach().to(device="cpu", copy=True) for parameter in optimization_parameters]
            last_train_loss = float("nan")
            with torch.enable_grad():
                for _ in range(self.config.epochs):
                    train_total = 0.0
                    for batch in train_batches:
                        optimizer.zero_grad(set_to_none=True)
                        prediction = self._forward_batch(layer, batch, device)
                        loss = self._masked_loss(prediction, batch.target, batch.attention_mask)
                        if not torch.isfinite(loss):
                            raise RuntimeError("QVQ output alignment produced a non-finite training loss.")
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        train_total += float(loss.detach().item())
                    last_train_loss = train_total / len(train_batches)
                    validation_loss = self._evaluate(layer, validation_batches, device)
                    if validation_loss < best_loss:
                        best_loss = validation_loss
                        best_state = [
                            parameter.detach().to(device="cpu", copy=True)
                            for parameter in optimization_parameters
                        ]

            for parameter, best_value in zip(optimization_parameters, best_state):
                parameter.data.copy_(best_value.to(device=parameter.device, dtype=parameter.dtype))
            candidate_weights = self._candidate_weights(temporary_modules)

            for module in modules:
                temporary = temporary_modules[module.name]
                recurse_setattr(
                    layer,
                    module.name,
                    self._build_runtime_module(
                        module,
                        device,
                        SU=temporary.SU,
                        SV=temporary.SV,
                    ),
                )
            runtime_candidate_loss = self._evaluate(layer, validation_batches, device)

            for name, original in original_modules.items():
                recurse_setattr(layer, name, original)
                original.weight.data.copy_(candidate_weights[name].to(device=device, dtype=original.weight.dtype))
            candidate_installed = True
            candidate_loss = self._evaluate(layer, validation_batches, device)
            improvement_factor = 1.0 - self.config.minimum_relative_improvement
            dense_required = baseline_loss * improvement_factor
            runtime_required = runtime_baseline_loss * improvement_factor
            accepted = candidate_loss < dense_required and runtime_candidate_loss < runtime_required
            if accepted:
                for module in modules:
                    temporary = temporary_modules[module.name]
                    with parent_module_lock(module.full_name):
                        module.state["SU"] = temporary.SU.detach().to(device="cpu", dtype=torch.float32).clone()
                        module.state["SV"] = temporary.SV.detach().to(device="cpu", dtype=torch.float32).clone()
            else:
                for parameter, original_value in original_parameters:
                    parameter.data = original_value.to(device=parameter.device, copy=True)
                candidate_installed = False

            relative_improvement = (baseline_loss - candidate_loss) / max(
                baseline_loss,
                torch.finfo(torch.float32).eps,
            )
            result = {
                "baseline_validation_loss": baseline_loss,
                "temporary_validation_loss": best_loss,
                "candidate_validation_loss": candidate_loss,
                "runtime_baseline_validation_loss": runtime_baseline_loss,
                "runtime_candidate_validation_loss": runtime_candidate_loss,
                "relative_improvement": relative_improvement,
                "runtime_relative_improvement": (
                    (runtime_baseline_loss - runtime_candidate_loss)
                    / max(runtime_baseline_loss, torch.finfo(torch.float32).eps)
                ),
                "last_train_loss": last_train_loss,
                "accepted": float(accepted),
                "alignment_pass": float(alignment_pass),
                "staged_modules": float(len(modules)),
                "finalize": float(finalize),
                "seconds": time.perf_counter() - started,
                "train_batches": float(len(train_batches)),
                "validation_batches": float(len(validation_batches)),
                "optimizer_steps": float(len(train_batches) * self.config.epochs),
                "gradient_scaling": float(gradient_scaling),
            }
            log.info(
                "QVQ output alignment: layer=%d accepted=%s dense=%.10g->%.10g runtime=%.10g->%.10g seconds=%.3f",
                layer_index,
                accepted,
                baseline_loss,
                candidate_loss,
                runtime_baseline_loss,
                runtime_candidate_loss,
                result["seconds"],
            )
            return result
        except BaseException:
            for name, original in original_modules.items():
                if recurse_getattr(layer, name) is not original:
                    recurse_setattr(layer, name, original)
                original.weight.data.copy_(
                    original_weights[name].to(device=original.weight.device, dtype=original.weight.dtype)
                )
            for module in modules:
                staged = original_staged.get(module.name)
                if staged is not None:
                    with parent_module_lock(module.full_name):
                        module.state["SU"], module.state["SV"] = staged
            for parameter, original_value in original_parameters:
                parameter.data = original_value.to(device=parameter.device, copy=True)
            raise
        finally:
            for name, original in original_dense_wrappers.items():
                recurse_setattr(layer, name, original)
            layer.train(original_training)
            for parameter, requires_grad in original_requires_grad:
                parameter.requires_grad_(requires_grad)
            if not candidate_installed and original_modules:
                for parameter, original_value in original_parameters:
                    parameter.data = original_value.to(device=parameter.device, copy=True)
            elif candidate_installed:
                for parameter, original_value in original_parameters:
                    parameter.data = parameter.data.to(
                        device=original_value.device if original_value.device.type != "cpu" else parameter.device,
                        dtype=original_value.dtype,
                        copy=True,
                    )
            for child, name, original_device, original_value in original_buffers:
                # A forward hook or backend may clear a non-persistent cache
                # while alignment runs. Transactionality means restoring the
                # original tensor even when the current buffer became None.
                child._buffers[name] = original_value.to(device=original_device, copy=True)

    def clear(self) -> None:
        with self._lock:
            self._layers.clear()
            self._clean_layer_inputs.clear()
            self._active_clean_layer_inputs.clear()


__all__ = ["QVQOutputAlignmentAttachment"]
