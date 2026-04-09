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

from typing import Dict, Optional

import torch
from defuser.modeling.replace_modules import materialize_model

from ..looper.module_preprocessor import ModulePreProcessor
from ..looper.weight_only_processor import WeightOnlyProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import CPU, SUPPORTS_MODULE_TYPES
from ..nn_modules.converter import MODULE_CONVERTER_MAP
from ..quantization.config import BitsAndBytesConfig, FP8Config, GGUFConfig, RTNConfig
from ..utils.logger import setup_logger
from ..utils.model import find_modules, get_module, get_module_by_name_prefix, move_to
from ..utils.offload import offload_to_disk


log = setup_logger()


class WeightOnlyLooper:
    """Run the simplified per-layer lifecycle for weight-only quantization."""

    def __init__(self, model: BaseQModel, processor: WeightOnlyProcessor):
        """Initializes the looper with the model being quantized and its processor."""

        self.gptq_model = model
        self.processor = processor

    def _resolve_named_module(
        self,
        *,
        layer_module: torch.nn.Module,
        full: Dict[str, torch.nn.Module],
        layer_index: int,
        layers_prefix: Optional[str],
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

        layer_name = self.gptq_model.lm_head if is_lm_head_module else f"{layers_prefix}.{layer_index}.{module_name}"
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

        layers, layers_prefix = get_module_by_name_prefix(
            self.gptq_model.model,
            self.gptq_model.extract_layers_node(),
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
                    for module_name in subset_names:
                        named = self._resolve_named_module(
                            layer_module=module,
                            full=full,
                            layer_index=layer_index,
                            layers_prefix=layers_prefix,
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

                        # Weight-only quantization happens entirely within the
                        # processor; no captured activations are needed.
                        active_qcfg = self.processor.quantize_module(named)
                        if active_qcfg is None:
                            continue

                        # Finalization and optional disk offload expect the
                        # packed module to be back on CPU memory.
                        move_to(named.module, device=CPU)
                        named.target_device = CPU
                        named.module.target_device = CPU

                        self.processor.submodule_finalize(
                            named,
                            self.gptq_model,
                            qcfg=active_qcfg,
                        )
                        self._offload_quantized_module(named)

                # Submodule-level offload may swap packed tensors to meta/disk placeholders.
                # Skip the layer-wide CPU move in that case to avoid `.to()` on meta buffers.
                if getattr(self.gptq_model.quantize_config, "offload_to_disk", False):
                    if not is_lm_head_module:
                        layers[layer_index] = module
                elif is_lm_head_module:
                    self.gptq_model.post_quantize(module)
                else:
                    layers[layer_index] = self.gptq_model.post_quantize(module)
        finally:
            self.gptq_model.model.config.use_cache = forward_pass_use_cache

        total_log = {self.processor.name(): self.processor.log}
        self.gptq_model.quant_log = self.processor.log
        self.processor.finalize(model=self.gptq_model)
        return total_log


__all__ = ["WeightOnlyLooper"]
