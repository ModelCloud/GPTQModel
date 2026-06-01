# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Weight-only quantization loop for methods that do not capture activations.

This looper intentionally does not share the activation-capture lifecycle used
by GPTQ/AWQ calibration flows. Weight-only methods such as RTN, FP8, NVFP4, or
GGUF can usually process each linear layer directly, so the control flow here
stays narrow: iterate quantizable modules, quantize weights, finalize, and
optionally offload.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import as_completed
from typing import Dict, List, Optional, Tuple

import torch
from defuser.modeling.replace_modules import materialize_model

from .. import DEVICE_THREAD_POOL
from ..looper.module_preprocessor import ModulePreProcessor
from ..looper.weight_only_processor import WeightOnlyProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import CPU, SUPPORTS_MODULE_TYPES
from ..nn_modules.converter import MODULE_CONVERTER_MAP
from ..quantization.config import BitsAndBytesConfig, FP8Config, GGUFConfig, RTNConfig, VramStrategy
from ..utils.device import get_device
from ..utils.device_telemetry import emit_device_telemetry
from ..utils.logger import log_time_block, setup_logger
from ..utils.looper_helpers import device_ctx, normalize_device_like, rehome_module_to_device, select_forward_devices
from ..utils.model import (
    find_modules,
    get_layer_name,
    get_layers_with_prefixes,
    get_module,
    get_module_by_name_prefix,
    move_to,
)
from ..utils.offload import offload_to_disk


log = setup_logger()


class WeightOnlyLooper:
    """Run the simplified per-layer lifecycle for weight-only quantization."""

    def __init__(self, model: BaseQModel, processor: WeightOnlyProcessor):
        """Initializes the looper with the model being quantized and its processor."""

        self.gptq_model = model
        self.processor = processor
        self._quant_devices = self._resolve_quant_devices()
        self._quant_device_rr = 0
        self._dense_device_rr = 0
        self._moe_group_device_rr = 0
        self._module_device_map: Dict[str, torch.device] = {}
        self._moe_group_device_map: Dict[str, torch.device] = {}
        self._quant_device_lock = threading.Lock()
        self._resolve_strategy_device_pools()

    def _resolve_quant_devices(self) -> List[torch.device]:
        """Resolve the device pool used by weight-only module quantization."""

        quant_config = self.gptq_model.quantize_config
        quant_device_hint = getattr(quant_config, "device", None)
        normalized_quant_device = normalize_device_like(quant_device_hint)
        quant_devices = select_forward_devices(normalized_quant_device) if normalized_quant_device else [CPU]
        if not quant_devices:
            quant_devices = [CPU]

        compute_device_filter = getattr(quant_config, "compute_device_filter", None)
        if compute_device_filter is not None:
            quant_devices_filtered = compute_device_filter(quant_devices)
            if len(quant_devices_filtered) >= 1:
                quant_devices = quant_devices_filtered
            else:
                log.warn(
                    "WeightOnlyLooper: compute_device_filter returned empty device list. "
                    "Using all devices for weight-only quantization."
                )

        return [normalize_device_like(device) or CPU for device in quant_devices]

    @staticmethod
    def _normalize_vram_strategy(value) -> VramStrategy:
        """Normalize one user-facing VRAM strategy value."""

        if isinstance(value, VramStrategy):
            return value
        if isinstance(value, str):
            try:
                return VramStrategy(value.lower())
            except ValueError:
                return VramStrategy.EXCLUSIVE
        return VramStrategy.EXCLUSIVE

    def _resolve_strategy_device_pool(
        self,
        configured_devices: Optional[List[str]],
        *,
        label: str,
    ) -> List[torch.device]:
        """Resolve one dense/MoE strategy device list against visible quant devices."""

        if not configured_devices:
            return list(self._quant_devices)

        available_by_name = {
            str(normalize_device_like(device)): normalize_device_like(device)
            for device in self._quant_devices
            if normalize_device_like(device) is not None
        }
        resolved: List[torch.device] = []
        for device_name in configured_devices:
            normalized = normalize_device_like(device_name)
            if normalized is None:
                raise ValueError(f"WeightOnlyLooper: {label} contains unsupported device value: {device_name!r}.")
            matched = available_by_name.get(str(normalized))
            if matched is None:
                raise ValueError(
                    f"WeightOnlyLooper: {label} {configured_devices} must be a subset of visible compute devices "
                    f"{list(available_by_name.keys())}."
                )
            if matched not in resolved:
                resolved.append(matched)

        if not resolved:
            raise ValueError(f"WeightOnlyLooper: {label} is empty after normalization.")

        return resolved

    def _resolve_strategy_device_pools(self) -> None:
        """Resolve dense and MoE pools once so module assignment stays deterministic."""

        quant_config = self.gptq_model.quantize_config
        self._dense_vram_strategy = self._normalize_vram_strategy(
            getattr(quant_config, "dense_vram_strategy", VramStrategy.EXCLUSIVE)
        )
        self._moe_vram_strategy = self._normalize_vram_strategy(
            getattr(quant_config, "moe_vram_strategy", VramStrategy.EXCLUSIVE)
        )
        dense_strategy_devices = getattr(quant_config, "dense_vram_strategy_devices", None)
        moe_strategy_devices = getattr(quant_config, "moe_vram_strategy_devices", None)
        self._dense_quant_devices = self._resolve_strategy_device_pool(
            dense_strategy_devices,
            label="dense_vram_strategy_devices",
        )
        self._moe_quant_devices = self._resolve_strategy_device_pool(
            moe_strategy_devices,
            label="moe_vram_strategy_devices",
        )
        self._dense_vram_strategy_explicit = bool(dense_strategy_devices) or self._dense_vram_strategy != VramStrategy.EXCLUSIVE
        self._moe_vram_strategy_explicit = bool(moe_strategy_devices) or self._moe_vram_strategy != VramStrategy.EXCLUSIVE

    @staticmethod
    def _extract_moe_group_key(module_name: Optional[str]) -> Optional[str]:
        """Return the expert family key used to co-locate gate/up/down modules."""

        if not module_name or ".experts." not in module_name:
            return None
        prefix, remainder = module_name.split(".experts.", 1)
        expert_id = remainder.split(".", 1)[0]
        if not expert_id:
            return None
        return f"{prefix}.experts.{expert_id}"

    def _select_strategy_device_for_module(self, named_module: NamedModule) -> Optional[torch.device]:
        """Apply dense/MoE VRAM strategy before falling back to generic round-robin."""

        module_name = getattr(named_module, "name", None) or getattr(named_module, "full_name", None)
        moe_group_key = self._extract_moe_group_key(module_name) or self._extract_moe_group_key(
            getattr(named_module, "full_name", None)
        )

        if moe_group_key is not None and self._moe_vram_strategy_explicit and self._moe_quant_devices:
            cached = self._moe_group_device_map.get(moe_group_key)
            if cached is not None:
                return cached
            # MoE BALANCED spreads expert families across the MoE pool; each
            # expert family keeps all projections on the same device.
            if self._moe_vram_strategy == VramStrategy.BALANCED and len(self._moe_quant_devices) > 1:
                device = self._moe_quant_devices[self._moe_group_device_rr % len(self._moe_quant_devices)]
                self._moe_group_device_rr += 1
            else:
                device = self._moe_quant_devices[0]
            self._moe_group_device_map[moe_group_key] = device
            return device

        if self._dense_vram_strategy_explicit and self._dense_quant_devices:
            # Dense EXCLUSIVE pins attention/router/shared/dense MLP work to
            # the first dense device; BALANCED round-robins dense modules.
            if self._dense_vram_strategy == VramStrategy.BALANCED and len(self._dense_quant_devices) > 1:
                device = self._dense_quant_devices[self._dense_device_rr % len(self._dense_quant_devices)]
                self._dense_device_rr += 1
                return device
            return self._dense_quant_devices[0]

        return None

    def _assign_quant_device_for_module(self, named_module: NamedModule, fallback_device: torch.device) -> torch.device:
        """Pick and memoize the quantization device for one named module."""

        key = getattr(named_module, "full_name", None) or named_module.name
        with self._quant_device_lock:
            cached = self._module_device_map.get(key)
            if cached is not None:
                emit_device_telemetry(
                    "weight_only_quant_device_cache_hit",
                    module=key,
                    target_device=cached,
                )
                return cached

            preferred_device = normalize_device_like(named_module.state.get("preferred_quant_device"))
            if preferred_device is not None and any(device == preferred_device for device in self._quant_devices):
                device = preferred_device
                emit_device_telemetry(
                    "weight_only_quant_device_preferred_hint",
                    module=key,
                    target_device=device,
                )
            elif (strategy_device := self._select_strategy_device_for_module(named_module)) is not None:
                device = strategy_device
                emit_device_telemetry(
                    "weight_only_quant_device_strategy",
                    module=key,
                    target_device=device,
                    dense_strategy=self._dense_vram_strategy.value,
                    moe_strategy=self._moe_vram_strategy.value,
                )
            elif len(self._quant_devices) <= 1:
                device = self._quant_devices[0]
            else:
                device = self._quant_devices[self._quant_device_rr % len(self._quant_devices)]
                self._quant_device_rr += 1

            if device is None:
                device = fallback_device

            self._module_device_map[key] = device
            emit_device_telemetry(
                "weight_only_quant_device_assign",
                module=key,
                target_device=device,
                fallback_device=fallback_device,
            )
            return device

    def _prepare_named_module_for_quantization(self, named: NamedModule, target_device: torch.device) -> None:
        """Move a weight-only module to the assigned quantization device."""

        try:
            previous_device = get_device(named.module)
        except Exception:
            previous_device = None
        with device_ctx(target_device):
            move_to(named.module, device=target_device)
            rehome_module_to_device(named.module, target_device, move_parameters=True, move_buffers=True)
            named.target_device = target_device
            named.module.target_device = target_device
        emit_device_telemetry(
            "weight_only_quant_prepare",
            module=getattr(named, "full_name", named.name),
            previous_device=previous_device,
            target_device=target_device,
        )

    def _move_named_module_to_cpu(self, named: NamedModule) -> None:
        """Return a processed weight-only module to CPU before finalization."""

        module_label = getattr(named, "full_name", named.name)
        try:
            previous_device = get_device(named.module)
        except Exception:
            previous_device = None

        start = time.perf_counter()
        if previous_device != CPU:
            emit_device_telemetry(
                "weight_only_quant_to_cpu_start",
                module=module_label,
                previous_device=previous_device,
                target_device=CPU,
            )
            move_to(named.module, device=CPU)
            rehome_module_to_device(named.module, CPU, move_parameters=True, move_buffers=True)

        named.target_device = CPU
        named.module.target_device = CPU

        duration = time.perf_counter() - start
        timer = getattr(self.gptq_model, "quant_region_timer", None)
        if timer is not None:
            timer.record("weight_only_move_to_cpu", duration, source=module_label)
        emit_device_telemetry(
            "weight_only_quant_to_cpu_complete",
            module=module_label,
            previous_device=previous_device,
            target_device=CPU,
            duration_ms=round(duration * 1000.0, 3),
        )

    def _quantize_named_module(
        self,
        named: NamedModule,
        target_device: torch.device,
    ) -> Tuple[NamedModule, Optional[RTNConfig | GGUFConfig | FP8Config | BitsAndBytesConfig]]:
        """Run one module's weight-only quantization on its assigned device."""

        self._prepare_named_module_for_quantization(named, target_device)
        with device_ctx(target_device):
            active_qcfg = self.processor.quantize_module(named, device=target_device)
        self._move_named_module_to_cpu(named)
        return named, active_qcfg

    @torch.inference_mode()
    def _finalize_quantized_module(
        self,
        named: NamedModule,
        active_qcfg: RTNConfig | GGUFConfig | FP8Config | BitsAndBytesConfig,
    ) -> str:
        """Move a quantized module back to CPU, pack it, and optionally offload it."""

        # Finalization creates and replaces modules in the model tree. The
        # shared pack helpers require CPU resident source tensors, while the
        # per-module qcfg still carries the GPU target for pack_gpu paths.
        module_label = getattr(named, "full_name", named.name)
        try:
            current_device = get_device(named.module)
        except Exception:
            current_device = None
        if current_device != CPU:
            self._move_named_module_to_cpu(named)
        else:
            named.target_device = CPU
            named.module.target_device = CPU

        start = time.perf_counter()
        emit_device_telemetry(
            "weight_only_submodule_finalize_start",
            module=module_label,
            target_device=getattr(active_qcfg, "device", None),
        )
        try:
            with log_time_block("weight_only_submodule_finalize", logger=log, module_name=module_label):
                self.processor.submodule_finalize(
                    named,
                    self.gptq_model,
                    qcfg=active_qcfg,
                )
            self._offload_quantized_module(named)
        finally:
            duration = time.perf_counter() - start
            timer = getattr(self.gptq_model, "quant_region_timer", None)
            if timer is not None:
                timer.record("weight_only_submodule_finalize", duration, source=module_label)
            emit_device_telemetry(
                "weight_only_submodule_finalize_complete",
                module=module_label,
                target_device=getattr(active_qcfg, "device", None),
                duration_ms=round(duration * 1000.0, 3),
            )

        with self._quant_device_lock:
            self._module_device_map[module_label] = CPU
        return module_label

    def _finalize_target_device(
        self,
        active_qcfg: RTNConfig | GGUFConfig | FP8Config | BitsAndBytesConfig,
    ) -> torch.device:
        """Resolve the worker device for one finalize task."""

        target_device = normalize_device_like(getattr(active_qcfg, "device", None))
        if target_device is not None and any(target_device == device for device in self._quant_devices):
            return target_device
        return CPU

    def _finalize_subset_modules(
        self,
        quantized_modules: List[Tuple[NamedModule, RTNConfig | GGUFConfig | FP8Config | BitsAndBytesConfig]],
    ) -> None:
        """Finalize one subset, using the device pool when multiple finalize targets exist."""

        if not quantized_modules:
            return

        finalize_tasks = [
            (named, active_qcfg, self._finalize_target_device(active_qcfg))
            for named, active_qcfg in quantized_modules
        ]
        unique_targets = {
            (target_device.type, target_device.index)
            for _, _, target_device in finalize_tasks
        }
        use_parallel_finalize = len(finalize_tasks) > 1 and len(unique_targets) > 1

        emit_device_telemetry(
            "weight_only_finalize_subset",
            module_count=len(finalize_tasks),
            target_devices=[target_device for _, _, target_device in finalize_tasks],
            parallel=use_parallel_finalize,
        )

        if not use_parallel_finalize:
            for named, active_qcfg, _target_device in finalize_tasks:
                self._finalize_quantized_module(named, active_qcfg)
            return

        futures = [
            DEVICE_THREAD_POOL.submit(
                target_device,
                self._finalize_quantized_module,
                named,
                active_qcfg,
            )
            for named, active_qcfg, target_device in finalize_tasks
        ]
        for future in as_completed(futures):
            future.result()

    def _quantize_subset_modules(
        self,
        named_modules: List[NamedModule],
    ) -> List[Tuple[NamedModule, RTNConfig | GGUFConfig | FP8Config | BitsAndBytesConfig]]:
        """Quantize one subset, using multiple devices when available."""

        results_by_name: Dict[str, Tuple[NamedModule, RTNConfig | GGUFConfig | FP8Config | BitsAndBytesConfig]] = {}
        task_specs = [
            (named, self._assign_quant_device_for_module(named, CPU))
            for named in named_modules
        ]
        if not task_specs:
            return []

        if len(self._quant_devices) <= 1 or len(task_specs) <= 1:
            for named, target_device in task_specs:
                quantized, active_qcfg = self._quantize_named_module(named, target_device)
                if active_qcfg is not None:
                    results_by_name[quantized.full_name] = (quantized, active_qcfg)
        else:
            future_map = {
                DEVICE_THREAD_POOL.submit(
                    target_device,
                    self._quantize_named_module,
                    named,
                    target_device,
                ): named
                for named, target_device in task_specs
            }
            for future in as_completed(future_map):
                quantized, active_qcfg = future.result()
                if active_qcfg is not None:
                    results_by_name[quantized.full_name] = (quantized, active_qcfg)

        return [
            results_by_name[named.full_name]
            for named in named_modules
            if named.full_name in results_by_name
        ]

    def _resolve_named_module(
        self,
        *,
        layer_module: torch.nn.Module,
        full: Dict[str, torch.nn.Module],
        layer_index: int,
        layer_path: Optional[str],
        module_name: str,
        is_lm_head_module: bool,
    ) -> Optional[NamedModule]:
        """Resolve a quantizable submodule and normalize it into a NamedModule."""
        resolved = full.get(module_name)
        if resolved is None:
            resolved, _ = get_module_by_name_prefix(layer_module, module_name)
            if resolved is None:
                if self.gptq_model.layer_modules_strict:
                    raise ValueError(f"layer module item `{module_name}` not found in model, please check your model config.")
                return None

        if isinstance(resolved, NamedModule):
            return resolved

        layer_name = self.gptq_model.lm_head if is_lm_head_module else f"{layer_path}.{module_name}"
        named = NamedModule(
            resolved,
            name=module_name,
            full_name=layer_name,
            layer_index=layer_index,
        )
        full[module_name] = named
        return named

    def _offload_quantized_module(self, module: NamedModule) -> None:
        """Persist an already-quantized module to disk when offload is enabled."""
        quant_config = getattr(self.gptq_model, "quantize_config", None)
        if not quant_config or not getattr(quant_config, "offload_to_disk", False):
            return
        offload_path = getattr(quant_config, "offload_to_disk_path", None)
        if not offload_path:
            return

        module_full_name = getattr(module, "full_name", None)
        target_module = (
            self.gptq_model.model.get_submodule(module_full_name)
            if module_full_name
            else module
        )
        offload_to_disk(
            model=self.gptq_model.model,
            module=target_module,
            disk_path=offload_path,
        )

    def loop(self, **kwargs):
        """Quantize layers directly from weights without calibration forwards."""
        quant_config = self.gptq_model.quantize_config
        if not isinstance(quant_config, (RTNConfig, GGUFConfig, FP8Config, BitsAndBytesConfig)):
            raise NotImplementedError(
                "Weight-only looper only supports `RTNConfig`, `GGUFConfig`, "
                "`FP8Config`, and `BitsAndBytesConfig` today."
            )

        if quant_config.lm_head:
            if self.gptq_model.model.config.tie_word_embeddings and hasattr(self.gptq_model.model.model, "_tied_weights_keys"):
                tied_keys = self.gptq_model.model._tied_weights_keys
                for item in tied_keys:
                    if self.gptq_model.lm_head in item:
                        raise NotImplementedError(
                            "quantization of `lm_head` layer with `tied_weights=True` model state is not supported. Please check model has `tied_weights=False`."
                        )

            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if lm_head_module is None:
                raise ValueError(f"could not find layer {self.gptq_model.lm_head} in the model, exit...")
            if not isinstance(lm_head_module, tuple(SUPPORTS_MODULE_TYPES)):
                raise NotImplementedError(
                    f"This type({type(lm_head_module)}) of lm_head quantization is currently not supported. SUPPORTS_MODULE_TYPES is {SUPPORTS_MODULE_TYPES}"
                )

        forward_pass_use_cache = (
            self.gptq_model.model.config.use_cache
            if hasattr(self.gptq_model.model.config, "use_cache")
            else False
        )
        # No calibration forwards are executed here, but disabling cache keeps
        # behavior aligned with the standard quantization path and avoids stale
        # decoder-cache state while layers are being replaced.
        self.gptq_model.model.config.use_cache = False

        layers, layer_names = get_layers_with_prefixes(
            self.gptq_model.model,
            self.gptq_model.extract_layers_node(),
        )

        for module_name in self.gptq_model.get_modules_with_direct_meta_tensors(self.gptq_model.model):
            module = get_module(self.gptq_model.model, module_name)
            if module is not None:
                self.gptq_model.shell_direct_meta_materialize(
                    target_submodule=module,
                    device=CPU,
                )

        if quant_config.offload_to_disk:
            log.info("Offloading base modules to disk...")
            offload_to_disk(
                model=self.gptq_model.model,
                module=self.gptq_model.get_base_modules(model=self.gptq_model.model),
                disk_path=quant_config.offload_to_disk_path,
            )

        layer_modules = self.gptq_model.simple_layer_modules(
            model_config=self.gptq_model.model.config,
            quantize_config=quant_config,
            is_awq_quantize=False,
            include_capture_only=False,
        )
        if not quant_config.true_sequential:
            layer_modules = [sum(layer_modules, [])]

        layer_count = len(layers)
        total_layers = layer_count + (1 if quant_config.lm_head else 0)
        pb = log.pb(range(total_layers)).manual().set(show_left_steps=1)
        pb.title(f"Weight-only quantization ({total_layers} layers)")
        self.processor.layer_count = layer_count
        self.processor.pb = pb
        preprocessor = None
        if getattr(quant_config, "preprocessors", None):
            preprocessor = ModulePreProcessor(
                tokenizer=self.gptq_model.tokenizer,
                qcfg=quant_config,
                calibration=None,
                prepare_dataset_func=None,
                calibration_concat_size=None,
                calibration_sort=None,
                calibration_concat_separator=None,
                batch_size=1,
            )

        try:
            for layer_index in range(total_layers):
                is_lm_head_module = layer_index >= layer_count

                # Transformer blocks and lm_head follow the same weight-only
                # lifecycle, but lm_head is resolved from the root model.
                if is_lm_head_module:
                    module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
                    subsets = [[self.gptq_model.lm_head]]
                else:
                    module = layers[layer_index]
                    subsets = layer_modules
                    # Flattened layer names preserve the source stack for split decoders.
                    layer_name = get_layer_name(layer_names, layer_index)
                if is_lm_head_module:
                    layer_name = None

                if pb is not None:
                    layer_title = (
                        "Weight-only quantizing lm_head"
                        if is_lm_head_module
                        else f"Weight-only quantizing layer {layer_index} of {layer_count - 1}"
                    )
                    pb.current_iter_step = layer_index
                    pb.title(layer_title).subtitle("").draw()

                module = self.gptq_model.pre_quantize(module)
                if not is_lm_head_module:
                    # Preserve existing module conversion behavior so the new
                    # lifecycle stays compatible with model-specific wrappers.
                    model_type = self.gptq_model.model.config.model_type
                    if model_type in MODULE_CONVERTER_MAP:
                        converter = MODULE_CONVERTER_MAP[model_type]
                        module = converter(module, self.gptq_model.model.config)
                    layers[layer_index] = module

                # Resolve concrete submodules after any pre-quantization
                # transforms so quantization targets the final layer layout.
                materialize_model(module)
                full = find_modules(module, name=self.gptq_model.lm_head if is_lm_head_module else "")

                self.processor.collect_memory_info(layer_index)
                for subset_names in subsets:
                    subset_named_modules: List[NamedModule] = []
                    for module_name in subset_names:
                        named = self._resolve_named_module(
                            layer_module=module,
                            full=full,
                            layer_index=layer_index,
                            layer_path=layer_name,
                            module_name=module_name,
                            is_lm_head_module=is_lm_head_module,
                        )
                        if named is None:
                            continue

                        if preprocessor is not None:
                            preprocessor.preprocess(named)
                            if isinstance(named.state.get("auto_module_decoder"), dict):
                                prepared = self.gptq_model.shell_module_materialize(
                                    target_submodule=named.module,
                                    device=CPU,
                                    role="quant_source",
                                    named_module=named,
                                )
                                if prepared is not named.module:
                                    named.module = prepared

                        subset_named_modules.append(named)

                    self._finalize_subset_modules(self._quantize_subset_modules(subset_named_modules))

                # Submodule-level offload may swap packed tensors to meta/disk placeholders.
                # Skip the layer-wide CPU move in that case to avoid `.to()` on meta buffers.
                if getattr(self.gptq_model.quantize_config, "offload_to_disk", False):
                    if not is_lm_head_module:
                        layers[layer_index] = module
                elif is_lm_head_module:
                    self.gptq_model.post_quantize(module)
                else:
                    layers[layer_index] = self.gptq_model.post_quantize(module)
                if pb is not None:
                    pb.current_iter_step = layer_index + 1
                    pb.draw()
        finally:
            if pb is not None:
                pb.close()
            self.gptq_model.model.config.use_cache = forward_pass_use_cache

        total_log = {self.processor.name(): self.processor.log}
        self.gptq_model.quant_log = self.processor.log
        self.processor.finalize(model=self.gptq_model)
        return total_log


__all__ = ["WeightOnlyLooper"]
