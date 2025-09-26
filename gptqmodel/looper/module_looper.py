# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy
import gc
import contextlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Dict, List, Tuple

import torch

from ..looper.dequantize_processor import DequantizeProcessor
from ..looper.eora_processor import EoraProcessor
from ..looper.gptq_processor import GPTQProcessor
from ..looper.input_cache import InputCache
from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import SUPPORTS_MODULE_TYPES
from ..nn_modules.hooked_linear import (
    STOP_FORWARD_EXCEPTION,
    HookedLinear,
    StopForward,
    replace_module_with_hooked_legacy,
)
from ..utils import ASYNC_WORKER
from ..utils.attn_mask import apply_keep_mask_bt, normalize_seq_mask
from ..utils.logger import setup_logger
from ..utils.model import (
    find_modules,
    get_device,
    get_module,
    get_module_by_name_prefix,
    move_to,
    nested_move_to,
)
from ..utils.offload import offload_to_disk
from ..utils.structure import print_module_tree  # noqa: F401  (kept for dev prints)
from ..utils.torch import (
    ALL_DEVICES,
    CPU,
    DEFAULT_BALANCE_STRATEGY,
    HAS_CUDA,
    META,
    BalanceStrategy,
    device_next,
    device_next_reset,
    torch_empty_cache,  # noqa: F401 (available for callers)
    torch_sync,
)
from .awq_processor import AWQProcessor
from ..utils.threads import  device_ctx, activate_device

log = setup_logger()


class ModuleLooper:
    def __init__(self, model: BaseQModel, processors: List[LoopProcessor]):
        self.processors = processors
        self.gptq_model = model
        self.support_batch_quantize = model.support_batch_quantize
        self.lock = threading.Lock()
        # Thread-local storage so masking is thread-safe during parallel forwards
        self._tls = threading.local()

    # ---------- Helpers for deepcopy safety (streams & transient attrs) ----------

    def _is_stream_obj(self, v):
        try:
            if hasattr(torch, "cuda"):
                StreamT = getattr(torch.cuda, "Stream", None)
                if StreamT is not None and isinstance(v, StreamT):
                    return True
            if hasattr(torch, "xpu"):
                StreamX = getattr(torch.xpu, "Stream", None)
                if StreamX is not None and isinstance(v, StreamX):
                    return True
        except Exception:
            pass
        return False

    def _strip_unpickleable_attrs(self, root_mod: torch.nn.Module):
        """
        Remove attributes that break deepcopy (streams & our transient per-module device fields).
        Returns a list of (obj, attr_name, value) for restoration.
        """
        removed = []
        for m in root_mod.modules():
            d = getattr(m, "__dict__", None)
            if not d:
                continue
            for k in list(d.keys()):
                v = d.get(k, None)
                if k in ("target_device", "target_device_stream") or self._is_stream_obj(v):
                    removed.append((m, k, v))
                    try:
                        delattr(m, k)
                    except Exception:
                        d.pop(k, None)
        return removed

    def _restore_unpickleable_attrs(self, removed):
        for obj, k, v in removed:
            try:
                setattr(obj, k, v)
            except Exception:
                # ignore if object changed or attr is gone
                pass

    # ---------- Masked hook wrapper (reads keep-mask from TLS) ----------

    def _masked_hook_wrapper(self, processor: LoopProcessor, inner_hook):
        def hook(module, inputs, output):
            keep = getattr(self._tls, "current_attention_mask", None)

            # Mask first tensor-like input if it's [B, S, ...]
            new_inputs = inputs
            try:
                if isinstance(inputs, (tuple, list)) and len(inputs) > 0 and torch.is_tensor(inputs[0]):
                    x = inputs[0]
                    if keep is not None and x.dim() >= 3:
                        xk = apply_keep_mask_bt(x, keep)
                        if isinstance(inputs, tuple):
                            new_inputs = (xk,) + tuple(inputs[1:])
                        else:
                            new_inputs = [xk] + list(inputs[1:])
            except Exception:
                new_inputs = inputs  # never break forward because of masking

            # Mask primary tensor output if it's [B, S, ...]
            new_output = output
            try:
                if isinstance(output, (tuple, list)) and len(output) > 0:
                    y0 = output[0]
                    if torch.is_tensor(y0) and keep is not None and y0.dim() >= 3:
                        yk = apply_keep_mask_bt(y0, keep)
                        if isinstance(output, tuple):
                            new_output = (yk,) + tuple(output[1:])
                        else:
                            new_output = [yk] + list(output[1:])
                elif torch.is_tensor(output) and keep is not None and output.dim() >= 3:
                    new_output = apply_keep_mask_bt(output, keep)
            except Exception:
                new_output = output

            return inner_hook(module, new_inputs, new_output)

        return hook

    # ---------- Input caching (unchanged except for minor style) ----------

    def cache_inputs(self, layers, calibration_data, use_cache):
        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []

        cur_layer_device = get_device(layers[0])
        data_device = cur_layer_device

        # TODO: HookLinear add register_forward_pre_hook()
        def store_input_hook(module, args, kwargs):
            # Positional arguments.
            layer_input = []
            if kwargs.get("hidden_states") is not None:
                layer_input.append(move_to(kwargs["hidden_states"], device=data_device))
            else:
                # If hidden_states is not in kwargs, get it from the first positional argument
                # If error occurs here, check the model's modeling code
                layer_input.append(move_to(args[0], device=data_device))

            layer_inputs.append(layer_input)

            # Keyword arguments.
            # TODO FIX ME..why is Qwen2_5OmniDecoderLayer harded here?
            if kwargs.get("attention_mask") is not None and str(type(module)) != "<class 'transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniDecoderLayer'>":
                attention_masks.append(kwargs["attention_mask"].to(device=data_device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to(pos_ids, device=data_device))
            one_kwargs = {}
            for (k, v) in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to(v, device=data_device)
            layer_input_kwargs.append(one_kwargs)

            raise STOP_FORWARD_EXCEPTION

        # move layer to target device
        if cur_layer_device == META:
            layers[0] = self.gptq_model.shell_module_materialize(
                target_submodule=layers[0],
                device=self.gptq_model.quantize_config.device,
            )
            cur_layer_device = self.gptq_model.quantize_config.device
        else:
            layers[0] = layers[0].to(self.gptq_model.quantize_config.device)

        ori_outside_layer_module_devices = {}
        for module_name in self.gptq_model.get_base_modules(self.gptq_model.model):
            module, _ = get_module_by_name_prefix(self.gptq_model.model, [module_name])

            if module is None:
                continue

            m_device = get_device(module)
            ori_outside_layer_module_devices[module_name] = CPU if m_device == META else m_device
            if module is not None:
                self.gptq_model.shell_module_materialize(
                    target_submodule=module,
                    device=cur_layer_device,
                )

        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)

        # TODO FIX ME.. remove hard coded Ovis code
        is_ovis = self.gptq_model.__class__.__name__ == "OvisGPTQ"

        # LifeCycle: start pre-first layer embedding hook
        self.gptq_model.pre_quantize_generate_hook_start()

        for example in calibration_data:
            for k, v in example.items():
                if str(type(layers[0])) == "<class 'transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniDecoderLayer'>":
                    data_device = self.gptq_model.quantize_config.device
                else:
                    data_device = self.gptq_model.quantize_config.device if k == "pixel_values" else cur_layer_device
                if isinstance(v, list):
                    for index in range(len(v)):
                        if len(v[index].shape) == 1:
                            v[index] = v[index].unsqueeze(0)
                        v[index] = move_to(
                            v[index].to(self.gptq_model.model.visual_tokenizer.dtype) if is_ovis else v[index],
                            device=data_device,
                        )
                else:
                    if len(v.shape) == 1:
                        v = v.unsqueeze(0)
                    example[k] = move_to(v, device=data_device)
            try:
                if str(type(layers[0])) == "<class 'transformers.models.qwen2_5_omni.modeling_qwen2_5_omni.Qwen2_5OmniDecoderLayer'>":
                    self.gptq_model.model.generate(**example, return_audio=False)
                else:
                    self.gptq_model.model(**example, use_cache=use_cache)
            except StopForward:
                pass

        # LifeCycle: pre-first layer embedding hook
        self.gptq_model.pre_quantize_generate_hook_end()
        handle.remove()

        return InputCache(
            layer_inputs=layer_inputs,
            layer_input_kwargs=layer_input_kwargs,
            position_ids=position_ids,
            attention_masks=attention_masks,
        )

    # ---------- Main loop ----------

    @torch.inference_mode
    def loop(self, fail_safe: bool = False, **kwargs):
        if self.gptq_model.quantize_config.lm_head:
            if self.gptq_model.model.config.tie_word_embeddings and hasattr(
                self.gptq_model.model.model, "_tied_weights_keys"
            ):
                tied_keys = self.gptq_model.model._tied_weights_keys
                for item in tied_keys:
                    if self.gptq_model.lm_head in item:
                        raise NotImplementedError(
                            "quantization of `lm_head` layer with `tied_weights=True` model state is not supported. "
                            "Please check model has `tied_weights=False`."
                        )

            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if lm_head_module is None:
                raise ValueError(f"could not find layer {self.gptq_model.lm_head} in the model, exit...")

            if not isinstance(lm_head_module, tuple(SUPPORTS_MODULE_TYPES)):
                raise NotImplementedError(
                    f"This type({type(lm_head_module)}) of lm_head quantization is currently not "
                    f"supported. SUPPORTS_MODULE_TYPES is {SUPPORTS_MODULE_TYPES}"
                )

            lm_head_quant_config = {"bits": 8, "group_size": 32, "sym": True, "desc_act": False, "mse": 2.4}
            if self.gptq_model.quantize_config.dynamic is None:
                self.gptq_model.quantize_config.dynamic = {self.gptq_model.lm_head: lm_head_quant_config}
            elif self.gptq_model.quantize_config.dynamic_get(self.gptq_model.lm_head, default=None) is None:
                self.gptq_model.quantize_config.dynamic[self.gptq_model.lm_head] = lm_head_quant_config

        forward_pass_use_cache = (
            self.gptq_model.model.config.use_cache if hasattr(self.gptq_model.model.config, "use_cache") else False
        )
        self.gptq_model.model.config.use_cache = False

        layers, layers_prefix = get_module_by_name_prefix(
            self.gptq_model.model, self.gptq_model.extract_layers_node()
        )

        # Build input caches for processors
        for p_index, processor in enumerate(self.processors):
            if not processor.verify_calibration_dataset(p_index):
                if isinstance(processor, EoraProcessor) or (
                    isinstance(processor, GPTQProcessor) and self.gptq_model.quantize_config.v2
                ):
                    prev_processor = self.processors[p_index - 1]
                    processor.set_calibration_dataset(prev_processor.calibration_dataset)
                    processor.receive_input_cache(copy.copy(prev_processor.inputs_cache))
                elif isinstance(processor, DequantizeProcessor):
                    processor.set_calibration_dataset([])
                    processor.receive_input_cache(InputCache([], [], [], []))
                continue

            input_cache = self.cache_inputs(layers=layers, calibration_data=processor.calibration_dataset, use_cache=False)
            processor.receive_input_cache(input_cache)

        # release calibration_dataset
        for processor in self.processors:
            processor.release_calibration_dataset()

        layer_modules = self.gptq_model.simple_layer_modules(model_config=self.gptq_model.model.config)
        if not self.gptq_model.quantize_config.true_sequential:
            layer_modules = [sum(layer_modules, [])]

        layer_count = len(layers)
        quant_modules_pb = (
            log.pb(layer_count + 1 if self.gptq_model.quantize_config.lm_head else layer_count).manual().set(left_steps_offset=1)
        )

        for processor in self.processors:
            processor.layer_count = layer_count
            processor.pb = quant_modules_pb

        shared_kv_cache_dict = {}

        replace_module_with_hooked_legacy(
            self.gptq_model.model, quant_lm_head=self.gptq_model.quantize_config.lm_head
        )

        if self.gptq_model.quantize_config.lm_head:
            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if lm_head_module and isinstance(lm_head_module, torch.nn.Linear):
                hooked_lm_head = HookedLinear.from_linear(lm_head_module)
                module_path = self.gptq_model.lm_head.split(".")
                parent = self.gptq_model.model
                for part in module_path[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, module_path[-1], hooked_lm_head)

        for layer_index in quant_modules_pb:
            is_lm_head_module = layer_index >= layer_count

            if is_lm_head_module:
                quant_modules_pb.title("Quantizing lm_head").draw()
                module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            else:
                quant_modules_pb.title(f"Quantizing layer {layer_index} of {layer_count - 1}").draw()
                module = layers[layer_index]

            if module.__class__.__name__.lower() == "mllamacrossattentiondecoderlayer":
                # TODO FIXME: currently we do not support quantizing cross attention layer (pixel_values)
                continue

            module = self.gptq_model.pre_quantize(module)

            cur_layer_device = get_device(module)
            full = find_modules(module, name=self.gptq_model.lm_head if is_lm_head_module else "")

            for p_index, processor in enumerate(self.processors):
                processor.log_call_count = 0  # reset
                processor.collect_memory_info(layer_index)

                modules = [[self.gptq_model.lm_head]] if is_lm_head_module else layer_modules

                # for NativeProcessor we process one time forward on all grouped module subsets
                if processor.fwd_all_modules_in_single_pass:
                    modules = [sum(modules, [])]

                # AWQ handles its own per-layer ops
                if isinstance(processor, AWQProcessor):
                    named_childs = dict()
                    for index, names in enumerate(modules):
                        named_modules = self.crate_named_modules(
                            full=full,
                            is_lm_head_module=is_lm_head_module,
                            layer_index=layer_index,
                            layers_prefix=layers_prefix,
                            names=names,
                            processor=processor,
                            fail_safe=fail_safe,
                        )
                        named_childs.update(named_modules)

                    processor.layer_quantize(module, cur_layer_device, named_childs)
                    continue  # skip module_looper path for awq

                layer_inputs = processor.inputs_cache.layer_inputs
                if is_lm_head_module:
                    layer_inputs = self.gptq_model.lm_head_pre_quantize_generate_hook(layer_inputs)
                layer_input_kwargs = processor.inputs_cache.layer_input_kwargs
                position_ids = processor.inputs_cache.position_ids
                attention_masks = processor.inputs_cache.attention_masks

                processed_subset = {}

                for index, names in enumerate(modules):
                    subset = self.crate_named_modules(
                        full=full,
                        is_lm_head_module=is_lm_head_module,
                        layer_index=layer_index,
                        layers_prefix=layers_prefix,
                        names=names,
                        processor=processor,
                        fail_safe=fail_safe,
                    )

                    if len(subset) == 0:
                        continue

                    handle = []
                    device_next_reset()

                    subset_size = len(subset)
                    for idx, (name, m) in enumerate(subset.items()):
                        is_last = idx == subset_size - 1

                        m.module.target_device, m.module.target_device_stream = device_next()

                        # Wrap the processor hook with masking
                        if hasattr(subset[name], "forward_hook"):
                            original_hook = processor.pre_process_fwd_hook(name)
                            subset[name].forward_hook = self._masked_hook_wrapper(processor, original_hook)
                            if is_last:
                                subset[name].forward_hook_last = True
                        else:
                            original_hook = processor.pre_process_fwd_hook(name)
                            handle.append(
                                subset[name].register_forward_hook(
                                    self._masked_hook_wrapper(processor, original_hook)
                                )
                            )

                    # ---------------- Parallel pre-quant forward capture ----------------
                    fwd_start = time.time()

                    # Device list (exclude META)
                    devices = [d for d in ALL_DEVICES if d != META]
                    num_devices = max(1, len(devices))
                    num_batches = processor.num_batches

                    # Pre-allocate outputs to preserve order
                    layer_outputs: List[List[torch.Tensor]] = [[] for _ in range(num_batches)]

                    # Helper: build per-device clone and subset with hooks
                    def _build_clone_and_subset(target_device):
                        if target_device == META:
                            return None, None
                        activate_device(target_device)
                        with device_ctx(target_device):
                            removed = self._strip_unpickleable_attrs(module)
                            try:
                                clone = copy.deepcopy(module)
                            finally:
                                self._restore_unpickleable_attrs(removed)

                            try:
                                clone = clone.to(target_device)
                            except Exception:
                                return None, None

                            full_clone = find_modules(
                                clone, name=self.gptq_model.lm_head if is_lm_head_module else ""
                            )
                            subset_clone: Dict[str, NamedModule] = {}
                            for n in names:
                                if n in full_clone:
                                    nm = full_clone[n]
                                    if not isinstance(nm, NamedModule):
                                        nm = NamedModule(
                                            nm,
                                            name=n,
                                            full_name=(
                                                self.gptq_model.lm_head
                                                if is_lm_head_module
                                                else f"{layers_prefix}.{layer_index}.{n}"
                                            ),
                                            layer_index=layer_index,
                                        )
                                        full_clone[n] = nm
                                    subset_clone[n] = nm

                            # install hooks on clone
                            for idx2, (n, nm) in enumerate(subset_clone.items()):
                                orig_hook = processor.pre_process_fwd_hook(n)
                                nm.forward_hook = self._masked_hook_wrapper(processor, orig_hook)
                                nm.forward_hook_last = idx2 == (len(subset_clone) - 1)
                            return clone, subset_clone

                    # Helper: split batches into contiguous ranges
                    def _split_ranges(n_items, n_parts):
                        base = n_items // n_parts
                        rem = n_items % n_parts
                        ranges = []
                        start = 0
                        for i in range(n_parts):
                            sz = base + (1 if i < rem else 0)
                            ranges.append((start, start + sz))
                            start += sz
                        return ranges

                    # Worker: run a partition on a specific device/clone
                    def _run_partition(dev: torch.device, batch_range: Tuple[int, int], clone, subset_clone):
                        activate_device(dev)
                        with device_ctx(dev):
                            start_idx, end_idx = batch_range
                            for j in range(start_idx, end_idx):
                                # Inputs
                                layer_input = []
                                for k, layer_inp in enumerate(layer_inputs[j]):
                                    layer_input.append(move_to(layer_inp, device=dev, stream=False))

                                raw_mask = attention_masks[j]
                                layer_attention_mask = (
                                    raw_mask if raw_mask is None else move_to(raw_mask, device=dev, stream=False)
                                )

                                # TLS keep-mask for hooks
                                if raw_mask is not None:
                                    seq_len = (
                                        layer_input[0].shape[1]
                                        if (len(layer_input) > 0 and layer_input[0].dim() >= 2)
                                        else None
                                    )
                                    self._tls.current_attention_mask = normalize_seq_mask(
                                        layer_attention_mask, seq_len=seq_len
                                    )
                                else:
                                    self._tls.current_attention_mask = None

                                additional_layer_inputs = (
                                    {"attention_mask": layer_attention_mask} if self.support_batch_quantize else {}
                                )
                                layer_position_ids = (
                                    None if not position_ids else move_to(position_ids[j], device=dev, stream=False)
                                )
                                if layer_position_ids is not None:
                                    additional_layer_inputs["position_ids"] = layer_position_ids
                                for kk, vv in layer_input_kwargs[j].items():
                                    additional_layer_inputs[kk] = nested_move_to(vv, device=dev, stream=False)

                                # Forward on clone
                                try:
                                    if hasattr(clone, "reuse_kv") and clone.reuse_kv:
                                        additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(
                                            layer_index - 1
                                        )
                                        out = clone(*layer_input) if is_lm_head_module else clone(
                                            *layer_input, **additional_layer_inputs
                                        )
                                        if shared_kv_cache_dict.get(layer_index) is None:
                                            shared_kv_cache_dict[layer_index] = out[-1]
                                    else:
                                        out = clone(*layer_input) if is_lm_head_module else clone(
                                            *layer_input, **additional_layer_inputs
                                        )
                                except StopForward:
                                    out = None
                                finally:
                                    self._tls.current_attention_mask = None
                                    del layer_input
                                    del additional_layer_inputs

                                if not processor.fwd_after_process:
                                    out0 = out[0] if isinstance(out, tuple) else out
                                    layer_outputs[j] = [out0]

                            # clear hooks on the clone subset
                            if subset_clone is not None:
                                for nm in subset_clone.values():
                                    nm.forward_hook = None
                                    nm.forward_hook_last = False

                    if num_devices == 1:
                        # Keep the single-thread path for zero-overhead when only one device
                        layer_outputs = []
                        for j in range(num_batches):
                            layer_input = []
                            for k, layer_inp in enumerate(layer_inputs[j]):
                                layer_input.append(move_to(layer_inp, device=cur_layer_device, stream=False))

                            raw_mask = attention_masks[j]
                            layer_attention_mask = (
                                raw_mask if raw_mask is None else move_to(raw_mask, device=cur_layer_device, stream=False)
                            )

                            if raw_mask is not None:
                                seq_len = (
                                    layer_input[0].shape[1]
                                    if (len(layer_input) > 0 and layer_input[0].dim() >= 2)
                                    else None
                                )
                                self._tls.current_attention_mask = normalize_seq_mask(
                                    layer_attention_mask, seq_len=seq_len
                                )
                            else:
                                self._tls.current_attention_mask = None

                            additional_layer_inputs = (
                                {"attention_mask": layer_attention_mask} if self.support_batch_quantize else {}
                            )
                            layer_position_ids = (
                                None if not position_ids else move_to(position_ids[j], device=cur_layer_device, stream=False)
                            )
                            if layer_position_ids is not None:
                                additional_layer_inputs["position_ids"] = layer_position_ids
                            for kk, vv in layer_input_kwargs[j].items():
                                additional_layer_inputs[kk] = nested_move_to(vv, device=cur_layer_device, stream=False)

                            try:
                                if hasattr(module, "reuse_kv") and module.reuse_kv:
                                    additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(layer_index - 1)
                                    layer_output = module(*layer_input) if is_lm_head_module else module(
                                        *layer_input, **additional_layer_inputs
                                    )
                                    if shared_kv_cache_dict.get(layer_index) is None:
                                        shared_kv_cache_dict[layer_index] = layer_output[-1]
                                else:
                                    layer_output = module(*layer_input) if is_lm_head_module else module(
                                        *layer_input, **additional_layer_inputs
                                    )
                            except StopForward:
                                layer_output = None
                            finally:
                                self._tls.current_attention_mask = None
                                del layer_input
                                del additional_layer_inputs

                            if not processor.fwd_after_process:
                                if isinstance(layer_output, tuple):
                                    layer_outputs.append([layer_output[0]])
                                else:
                                    layer_outputs.append([layer_output])
                    else:
                        # Parallel across devices: build clones, split ranges, run
                        clones = []
                        for dev in devices:
                            clone, subset_clone = _build_clone_and_subset(dev)
                            clones.append((dev, clone, subset_clone))

                        # remove devices with failed clone build
                        clones = [(d, c, s) for (d, c, s) in clones if c is not None]

                        if len(clones) == 0:
                            # fallback to single device path on cur_layer_device
                            layer_outputs = []
                            for j in range(num_batches):
                                layer_input = []
                                for k, layer_inp in enumerate(layer_inputs[j]):
                                    layer_input.append(move_to(layer_inp, device=cur_layer_device, stream=False))
                                raw_mask = attention_masks[j]
                                layer_attention_mask = (
                                    raw_mask if raw_mask is None else move_to(raw_mask, device=cur_layer_device, stream=False)
                                )
                                if raw_mask is not None:
                                    seq_len = (
                                        layer_input[0].shape[1]
                                        if (len(layer_input) > 0 and layer_input[0].dim() >= 2)
                                        else None
                                    )
                                    self._tls.current_attention_mask = normalize_seq_mask(
                                        layer_attention_mask, seq_len=seq_len
                                    )
                                else:
                                    self._tls.current_attention_mask = None

                                additional_layer_inputs = (
                                    {"attention_mask": layer_attention_mask} if self.support_batch_quantize else {}
                                )
                                layer_position_ids = (
                                    None
                                    if not position_ids
                                    else move_to(position_ids[j], device=cur_layer_device, stream=False)
                                )
                                if layer_position_ids is not None:
                                    additional_layer_inputs["position_ids"] = layer_position_ids
                                for kk, vv in layer_input_kwargs[j].items():
                                    additional_layer_inputs[kk] = nested_move_to(
                                        vv, device=cur_layer_device, stream=False
                                    )
                                try:
                                    if hasattr(module, "reuse_kv") and module.reuse_kv:
                                        additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(
                                            layer_index - 1
                                        )
                                        layer_output = module(*layer_input) if is_lm_head_module else module(
                                            *layer_input, **additional_layer_inputs
                                        )
                                        if shared_kv_cache_dict.get(layer_index) is None:
                                            shared_kv_cache_dict[layer_index] = layer_output[-1]
                                    else:
                                        layer_output = module(*layer_input) if is_lm_head_module else module(
                                            *layer_input, **additional_layer_inputs
                                        )
                                except StopForward:
                                    layer_output = None
                                finally:
                                    self._tls.current_attention_mask = None
                                    del layer_input
                                    del additional_layer_inputs

                                if not processor.fwd_after_process:
                                    if isinstance(layer_output, tuple):
                                        layer_outputs.append([layer_output[0]])
                                    else:
                                        layer_outputs.append([layer_output])
                        else:
                            ranges = _split_ranges(num_batches, len(clones))
                            with ThreadPoolExecutor(max_workers=len(clones)) as pool:
                                futures = []
                                for (dev, clone, subset_clone), (s, e) in zip(clones, ranges):
                                    if s == e:
                                        continue
                                    futures.append(
                                        pool.submit(_run_partition, dev, (s, e), clone, subset_clone)
                                    )
                                for fut in as_completed(futures):
                                    fut.result()

                            # Move outputs back to cur_layer_device for consistency
                            if not processor.fwd_after_process:
                                for j in range(num_batches):
                                    if layer_outputs[j]:
                                        out0 = layer_outputs[j][0]
                                        if get_device(out0) != cur_layer_device:
                                            layer_outputs[j][0] = move_to(
                                                out0, device=cur_layer_device, stream=False
                                            )

                    # --------------------------------------------------------------------
                    if not processor.fwd_after_process:
                        processor.receive_layer_inputs(layer_outputs)
                        del layer_outputs

                    fwd_end = time.time()
                    processor.set_fwd_time(fwd_end - fwd_start)

                    for h in handle:
                        h.remove()
                    for name in subset:
                        if hasattr(subset[name], "forward_hook"):
                            subset[name].forward_hook = None
                            subset[name].forward_hook_last = False

                    # MoE coverage check for GPTQ
                    moe_skip_modules = []
                    if isinstance(processor, GPTQProcessor):
                        for name in subset:
                            if processor.tasks[name].fwd_counter == 0:
                                log.error(
                                    f"`{name}` was not invoked, if it is a MoE module, it may lack sufficient calibration data routed to it."
                                )
                                moe_skip_modules.append(name)
                        if not fail_safe:
                            for name in moe_skip_modules:
                                subset.pop(name)

                    # ---- Process hook (can use GPU thread pool) ----
                    if len(ALL_DEVICES) <= 1:
                        for name in subset:
                            m = subset[name]
                            processor.process(module=m)
                            processed_subset[name] = m
                    else:
                        max_workers = (
                            len(ALL_DEVICES)
                            if DEFAULT_BALANCE_STRATEGY == BalanceStrategy.GPU
                            else max(1, len(ALL_DEVICES) - 1)
                        )
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = []

                            def process_module(name, m):
                                # prevent cuda sync memory ctx bugs
                                m_device = get_device(m)
                                if HAS_CUDA and m_device is not None and getattr(m_device, "type", "") == "cuda":
                                    torch.cuda.set_device(m_device)
                                processor.process(module=m)
                                return name, m

                            for name in subset:
                                m = subset[name]
                                futures.append(executor.submit(process_module, name, m))

                            for future in futures:
                                name, m = future.result()
                                processed_subset[name] = m

                        torch_sync()
                    # ---- End Process hook ----

                is_last_module = layer_index == len(quant_modules_pb) - 1
                # second forward after process()
                if not is_last_module and processor.fwd_after_process:
                    layer_outputs = []
                    for j in range(processor.num_batches):
                        layer_input = []
                        for k, layer_inp in enumerate(layer_inputs[j]):
                            layer_input.append(move_to(layer_inp, device=cur_layer_device))

                        raw_mask = attention_masks[j]
                        layer_attention_mask = (
                            raw_mask if raw_mask is None else move_to(raw_mask, device=cur_layer_device)
                        )

                        if raw_mask is not None:
                            seq_len = (
                                layer_input[0].shape[1]
                                if (len(layer_input) > 0 and layer_input[0].dim() >= 2)
                                else None
                            )
                            self._tls.current_attention_mask = normalize_seq_mask(
                                layer_attention_mask, seq_len=seq_len
                            )
                        else:
                            self._tls.current_attention_mask = None

                        additional_layer_inputs = (
                            {"attention_mask": layer_attention_mask} if self.support_batch_quantize else {}
                        )
                        layer_position_ids = (
                            None if not position_ids else move_to(position_ids[j], device=cur_layer_device)
                        )
                        if layer_position_ids is not None:
                            additional_layer_inputs["position_ids"] = layer_position_ids
                        for kk, vv in layer_input_kwargs[j].items():
                            additional_layer_inputs[kk] = nested_move_to(vv, device=cur_layer_device)

                        if hasattr(module, "reuse_kv") and module.reuse_kv:
                            additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(layer_index - 1)

                        if is_lm_head_module:
                            module_output = module(*layer_input)
                        else:
                            module_output = module(*layer_input, **additional_layer_inputs)

                        layer_output = module_output[0] if isinstance(module_output, tuple) else module_output
                        layer_output = move_to(layer_output, device=cur_layer_device)

                        layer_outputs.append([layer_output])

                        self._tls.current_attention_mask = None
                        del layer_input
                        del additional_layer_inputs

                # Finalize module after last processor
                if p_index == len(self.processors) - 1:
                    torch_sync()
                    if not is_lm_head_module:
                        layers[layer_index] = self.gptq_model.post_quantize(module)
                    else:
                        self.gptq_model.post_quantize(module)

                if processor.fwd_after_process:
                    processor.clear_cache_data()
                    processor.receive_layer_inputs(layer_outputs)

                if p_index == len(self.processors) - 1:
                    torch_sync()
                    for reverse_p in reversed(self.processors):
                        for name in processed_subset:
                            def finalize_module(module):
                                # prevent cuda sync memory ctx bugs
                                m_device = get_device(module)
                                if HAS_CUDA and m_device is not None and getattr(m_device, "type", "") == "cuda":
                                    torch.cuda.set_device(m_device)

                                reverse_p.submodule_finalize(module, self.gptq_model)

                                # checking for disk offloading
                                offload_to_disk(
                                    model=self.gptq_model.model,
                                    module=self.gptq_model.model.get_submodule(module.full_name),
                                    disk_path=self.gptq_model.quantize_config.offload_to_disk_path,
                                )

                            module = processed_subset[name]
                            if self.gptq_model.quantize_config.offload_to_disk:
                                ASYNC_WORKER.submit(partial(finalize_module, module=module))

        # LifeCycle: All sub-modules have finalized meaning quantization work is complete
        ASYNC_WORKER.join()

        # paranoid safety check
        torch_sync()
        torch_sync(device=CPU)

        total_log = {}

        for reverse_p in reversed(self.processors):
            if isinstance(reverse_p, (GPTQProcessor, EoraProcessor, DequantizeProcessor)):
                pass
            else:
                log.info(f"{reverse_p.name()} summary:\n{reverse_p.log}")

            processor_name = reverse_p.name()
            total_log[processor_name] = reverse_p.log
            if processor_name in ["gptq", "gptq v2"]:
                self.gptq_model.quant_log = reverse_p.log

            for module_log in reverse_p.log:
                log.info(module_log)
            reverse_p.log_plotly()

            reverse_p.finalize(model=self.gptq_model, **kwargs)

        self.gptq_model.model.config.use_cache = forward_pass_use_cache

        return total_log

    # ---------- Named-module creation ----------

    def crate_named_modules(
        self,
        full,
        is_lm_head_module,
        layer_index,
        layers_prefix,
        names,
        processor,
        fail_safe,
    ) -> Dict[str, NamedModule]:
        is_awq_quant = isinstance(processor, AWQProcessor)
        subset = {}
        for n in names:
            if n in full:
                subset[n] = full[n]
            elif self.gptq_model.layer_modules_strict:
                raise ValueError(
                    f"layer module item `{n}` not found in model, please check your model config."
                )

        skipped_modules = []
        for name in subset:
            layer_name = (
                self.gptq_model.lm_head
                if is_lm_head_module
                else f"{layers_prefix}.{layer_index}.{name}"
            )

            # wrap with NamedModule and seed state if needed
            if not isinstance(subset[name], NamedModule):
                named_module = NamedModule(subset[name], name=name, full_name=layer_name, layer_index=layer_index)
                if isinstance(processor, EoraProcessor):
                    named_module.state.update({"wq": processor.quantized_weights[layer_name]})
                subset[name] = named_module
                full[name] = named_module

            if not is_awq_quant:
                if isinstance(processor, GPTQProcessor):
                    processor.preprocess(subset[name], fail_safe=fail_safe)
                else:
                    processor.preprocess(subset[name])
                if processor.is_skipped(subset[name]):
                    skipped_modules.append(name)

        if not is_awq_quant:
            for name in skipped_modules:
                subset.pop(name)
        return subset
