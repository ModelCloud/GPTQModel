# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import copy
import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List

import torch

from ..looper.dequantize_processor import DequantizeProcessor
from ..looper.eora_processor import EoraProcessor
from ..looper.gptq_processor import GPTQProcessor
from ..looper.input_cache import InputCache
from ..looper.loop_processor import LoopProcessor
from ..looper.named_module import NamedModule
from ..models import BaseQModel
from ..models._const import CUDA, SUPPORTS_MODULE_TYPES
from ..nn_modules.hooked_linear import (STOP_FORWARD_EXCEPTION, HookedLinear,
                                        StopForward, replace_module_with_hooked_legacy)
from ..utils import ASYNC_WORKER
from ..utils.attn_mask import apply_keep_mask_bt, normalize_seq_mask
from ..utils.device import get_device
from ..utils.logger import setup_logger
from ..utils.model import find_modules, get_module, get_module_by_name_prefix, move_to, nested_move_to
from ..utils.offload import offload_to_disk
from ..utils.structure import print_module_tree
from ..utils.torch import (ALL_DEVICES, CPU, DEFAULT_BALANCE_STRATEGY, HAS_CUDA, META, BalanceStrategy,
                           device_next, device_next_reset, torch_empty_cache, torch_sync)
from .awq_processor import AWQProcessor

log = setup_logger()


class ModuleLooper():
    def __init__(self, model: BaseQModel, processors: List[LoopProcessor]):
        self.processors = processors
        self.gptq_model = model
        self.support_batch_quantize = model.support_batch_quantize
        self.lock = threading.Lock()

    # NEW: Wrap an existing hook so its inputs/outputs are pre-masked for GPTQ stats.
    # We *do not* alter the module's actual computation; only what the hook
    # passes down to the processor capture path is masked.
    def _masked_hook_wrapper(self, processor: LoopProcessor, inner_hook):
        def hook(module, inputs, output):
            keep = getattr(processor, "current_attention_mask", None)

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
                # Never break the forward due to masking; fall back to original
                new_inputs = inputs

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

    def cache_inputs(self, layers, calibration_data, use_cache):
        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []

        cur_layer_device = get_device(layers[0])
        data_device = cur_layer_device

        # TODO HookLinear add register_forward_pre_hook()
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
                        v[index] = move_to(v[index].to(self.gptq_model.model.visual_tokenizer.dtype) if is_ovis else v[index],
                                                  device=data_device)
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

        return InputCache(layer_inputs=layer_inputs, layer_input_kwargs=layer_input_kwargs, position_ids=position_ids,
                          attention_masks=attention_masks)

    @torch.inference_mode
    def loop(self, fail_safe: bool = False, **kwargs):
        if self.gptq_model.quantize_config.lm_head:
            if self.gptq_model.model.config.tie_word_embeddings and hasattr(self.gptq_model.model.model, "_tied_weights_keys"):
                tied_keys = self.gptq_model.model._tied_weights_keys
                for item in tied_keys:
                    if self.gptq_model.lm_head in item:
                        raise NotImplementedError("quantization of `lm_head` layer with `tied_weights=True` model state is not supported. Please check model has `tied_weights=False`.")

            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if get_module(self.gptq_model.model, key=self.gptq_model.lm_head) is None:
                raise ValueError(f"could not find layer {self.gptq_model.lm_head} in the model, exit...")

            if not isinstance(lm_head_module, tuple(SUPPORTS_MODULE_TYPES)):
                raise NotImplementedError(f"This type({type(lm_head_module)}) of lm_head quantization is currently not "
                                          f"supported. SUPPORTS_MODULE_TYPES is {SUPPORTS_MODULE_TYPES}")

            lm_head_quant_config = {"bits": 8, "group_size": 32, "sym": True, "desc_act": False, "mse": 2.4}
            if self.gptq_model.quantize_config.dynamic is None:
                self.gptq_model.quantize_config.dynamic = {self.gptq_model.lm_head: lm_head_quant_config}
            elif self.gptq_model.quantize_config.dynamic_get(self.gptq_model.lm_head, default=None) is None:
                self.gptq_model.quantize_config.dynamic[self.gptq_model.lm_head] = lm_head_quant_config

        forward_pass_use_cache = self.gptq_model.model.config.use_cache if hasattr(self.gptq_model.model.config, "use_cache") else False
        self.gptq_model.model.config.use_cache = False
        layers, layers_prefix = get_module_by_name_prefix(self.gptq_model.model, self.gptq_model.extract_layers_node())

        for p_index, processor in enumerate(self.processors):
            if not processor.verify_calibration_dataset(p_index):
                if isinstance(processor, EoraProcessor) or\
                        (isinstance(processor, GPTQProcessor) and self.gptq_model.quantize_config.v2):
                    prev_processor = self.processors[p_index - 1]
                    processor.set_calibration_dataset(prev_processor.calibration_dataset)
                    # If calibration_dataset is None or Empty, the input_cache of the previous processor is used.
                    processor.receive_input_cache(copy.copy(prev_processor.inputs_cache))
                elif isinstance(processor, DequantizeProcessor):
                    # DequantizeProcessor does not perform any operations on dataset.
                    processor.set_calibration_dataset([])
                    processor.receive_input_cache(InputCache([], [], [], []))

                continue

            input_cache = self.cache_inputs(layers=layers,
                                            calibration_data=processor.calibration_dataset,
                                            use_cache=False)
            processor.receive_input_cache(input_cache)

        # release calibration_dataset
        for processor in self.processors:
            processor.release_calibration_dataset()

        layer_modules = self.gptq_model.simple_layer_modules(model_config=self.gptq_model.model.config)

        if not self.gptq_model.quantize_config.true_sequential:
            layer_modules = [sum(layer_modules, [])]

        layer_count = len(layers)
        quant_modules_pb = (log.pb(layer_count + 1 if self.gptq_model.quantize_config.lm_head else layer_count)
                            .manual()
                            .set(left_steps_offset=1))

        for processor in self.processors:
            processor.layer_count = layer_count
            processor.pb = quant_modules_pb

        shared_kv_cache_dict = {}

        replace_module_with_hooked_legacy(self.gptq_model.model, quant_lm_head=self.gptq_model.quantize_config.lm_head)

        if self.gptq_model.quantize_config.lm_head:
            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if lm_head_module and isinstance(lm_head_module, torch.nn.Linear):
                hooked_lm_head = HookedLinear.from_linear(lm_head_module)
                module_path = self.gptq_model.lm_head.split('.')
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

            if module.__class__.__name__.lower() == "MllamaCrossAttentionDecoderLayer".lower():
                # TODO FIXME: currently we not support quantizing cross attention layer (pixel_values)
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
                    # merge all subsets into one
                    modules = [sum(modules, [])]

                # TODO: integrated AWQ module forward/hooks within module lopper so everything is unified
                # AWQ does it's own per layer module hooks and calculations. Logic has not been fully integrated into
                # the module_looper so we wil let awq handle per layer operations for now
                if isinstance(processor, AWQProcessor):
                    named_childs = dict()
                    for index, names in enumerate(modules):
                        named_modules = self.crate_named_modules(full=full,
                                                                 is_lm_head_module=is_lm_head_module,
                                                                 layer_index=layer_index, layers_prefix=layers_prefix,
                                                                 names=names,
                                                                 processor=processor,
                                                                 fail_safe=fail_safe)
                        named_childs.update(named_modules)

                    # awq uses model.layers[0] for quantization instead of model.layers.0.self_attn.q_proj
                    processor.layer_quantize(module, cur_layer_device, named_childs)
                    # skip module_looper processing for awq
                    continue

                layer_inputs = processor.inputs_cache.layer_inputs
                if is_lm_head_module:
                    layer_inputs = self.gptq_model.lm_head_pre_quantize_generate_hook(layer_inputs)
                layer_input_kwargs = processor.inputs_cache.layer_input_kwargs
                position_ids = processor.inputs_cache.position_ids
                attention_masks = processor.inputs_cache.attention_masks

                processed_subset = {}

                for index, names in enumerate(modules):
                    subset = self.crate_named_modules(full=full, is_lm_head_module=is_lm_head_module,
                                                      layer_index=layer_index, layers_prefix=layers_prefix,
                                                      names=names,
                                                      processor=processor,
                                                      fail_safe=fail_safe)

                    if len(subset) == 0:
                        continue

                    handle = []
                    device_next_reset()

                    subset_size = len(subset)
                    for idx, (name, m) in enumerate(subset.items()):
                        is_last = (idx == subset_size - 1)

                        m.module.target_device, m.module.target_device_stream = device_next()

                        # Wrap the processor hook with masking
                        if hasattr(subset[name], 'forward_hook'):
                            original_hook = processor.pre_process_fwd_hook(name)
                            subset[name].forward_hook = self._masked_hook_wrapper(processor, original_hook)
                            if is_last:
                                subset[name].forward_hook_last = True
                        else:
                            # Older registration path
                            original_hook = processor.pre_process_fwd_hook(name)
                            handle.append(subset[name].register_forward_hook(
                                self._masked_hook_wrapper(processor, original_hook)
                            ))

                    # ---- Start Pre-Quantized Forward ----
                    fwd_start = time.time()

                    layer_outputs = []
                    for j in range(processor.num_batches):
                        layer_input = []
                        for k, layer_inp in enumerate(layer_inputs[j]):
                            layer_input.append(move_to(layer_inp, device=cur_layer_device, stream=False))

                        raw_mask = attention_masks[j]
                        layer_attention_mask = raw_mask if raw_mask is None else move_to(raw_mask, device=cur_layer_device, stream=False)

                        # Compute and set keep-mask for this batch, for hook wrappers to consume
                        if raw_mask is not None:
                            # Assume hidden_states is first arg with shape [B, S, H]
                            seq_len = layer_input[0].shape[1] if (len(layer_input) > 0 and layer_input[0].dim() >= 2) else None
                            keep_mask_bs = normalize_seq_mask(layer_attention_mask, seq_len=seq_len)
                            # We don't require LoopProcessor to declare this attribute; set dynamically.
                            setattr(processor, "current_attention_mask", keep_mask_bs)
                        else:
                            setattr(processor, "current_attention_mask", None)

                        additional_layer_inputs = {"attention_mask": layer_attention_mask} if self.support_batch_quantize else {}
                        layer_position_ids = (
                            None if not position_ids else move_to(position_ids[j], device=cur_layer_device, stream=False)
                        )
                        if layer_position_ids is not None:
                            additional_layer_inputs["position_ids"] = layer_position_ids
                        for k, v in layer_input_kwargs[j].items():
                            additional_layer_inputs[k] = nested_move_to(v, device=cur_layer_device, stream=False)

                        try:
                            # reuse_kv special-case
                            if hasattr(module, "reuse_kv") and module.reuse_kv:
                                additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(layer_index - 1)
                                layer_output = module(*layer_input) if is_lm_head_module else module(*layer_input, **additional_layer_inputs)
                                if shared_kv_cache_dict.get(layer_index) is None:
                                    shared_kv_cache_dict[layer_index] = layer_output[-1]
                            else:
                                layer_output = module(*layer_input) if is_lm_head_module else module(*layer_input, **additional_layer_inputs)
                        except StopForward:
                            pass
                        finally:
                            # Clear the per-batch mask no matter what
                            setattr(processor, "current_attention_mask", None)
                            del layer_input
                            del additional_layer_inputs

                        if not processor.fwd_after_process:
                            if isinstance(layer_output, tuple):
                                layer_outputs.append([layer_output[0]])
                            else:
                                layer_outputs.append([layer_output])

                    if not processor.fwd_after_process:
                        processor.receive_layer_inputs(layer_outputs)
                        del layer_outputs

                    fwd_end = time.time()
                    fwd_time = fwd_end - fwd_start
                    processor.set_fwd_time(fwd_time)

                    for h in handle:
                        h.remove()

                    for name in subset:
                        if hasattr(subset[name], 'forward_hook'):
                            subset[name].forward_hook = None
                            subset[name].forward_hook_last = False

                    # MoE coverage check for GPTQ
                    moe_skip_modules = []
                    if isinstance(processor, GPTQProcessor):
                        for name in subset:
                            if processor.tasks[name].fwd_counter == 0:
                                log.error(f"`{name}` was not invoked, if it is a MoE module, it may lack sufficient calibration data routed to it.")
                                moe_skip_modules.append(name)

                        if not fail_safe:
                            for name in moe_skip_modules:
                                subset.pop(name)

                    # ---- Start Process Hook ----
                    if len(ALL_DEVICES) <= 1:
                        for name_index, name in enumerate(subset):
                            m = subset[name]
                            processor.process(module=m)
                            processed_subset[name] = m
                    else:
                        max_workers = len(ALL_DEVICES) if DEFAULT_BALANCE_STRATEGY == BalanceStrategy.GPU else len(ALL_DEVICES) - 1
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = []

                            def process_module(name, m):
                                # prevent cuda sync memory ctx bugs
                                m_device = get_device(m)
                                if HAS_CUDA and m_device is not None and m_device.type == "cuda":
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
                    # ---- End Process Hook ----

                is_last_module = layer_index == len(quant_modules_pb) - 1
                # second forward after process()
                if not is_last_module and processor.fwd_after_process:
                    layer_outputs = []
                    for j in range(processor.num_batches):
                        layer_input = []
                        for k, layer_inp in enumerate(layer_inputs[j]):
                            layer_input.append(move_to(layer_inp, device=cur_layer_device))

                        raw_mask = attention_masks[j]
                        layer_attention_mask = raw_mask if raw_mask is None else move_to(raw_mask, device=cur_layer_device)

                        # Keep-mask for this replay, for completeness (in case hooks capture again)
                        if raw_mask is not None:
                            seq_len = layer_input[0].shape[1] if (len(layer_input) > 0 and layer_input[0].dim() >= 2) else None
                            keep_mask_bs = normalize_seq_mask(layer_attention_mask, seq_len=seq_len)
                            setattr(processor, "current_attention_mask", keep_mask_bs)
                        else:
                            setattr(processor, "current_attention_mask", None)

                        additional_layer_inputs = {"attention_mask": layer_attention_mask} if self.support_batch_quantize else {}
                        layer_position_ids = None if not position_ids else move_to(position_ids[j], device=cur_layer_device)
                        if layer_position_ids is not None:
                            additional_layer_inputs["position_ids"] = layer_position_ids
                        for k, v in layer_input_kwargs[j].items():
                            additional_layer_inputs[k] = nested_move_to(v, device=cur_layer_device)

                        if hasattr(module, "reuse_kv") and module.reuse_kv:
                            additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(layer_index - 1)

                        module_output = None
                        if is_lm_head_module:
                            module_output = module(*layer_input)
                        else:
                            module_output = module(*layer_input, **additional_layer_inputs)

                        if isinstance(module_output, tuple):
                            layer_output = module_output[0]
                        else:
                            layer_output = module_output

                        layer_output = move_to(
                            layer_output,
                            device=cur_layer_device,
                        )

                        layer_outputs.append([layer_output])

                        # Clear per-batch mask
                        setattr(processor, "current_attention_mask", None)

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
                                if HAS_CUDA and m_device is not None and m_device.type == "cuda":
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
                                ASYNC_WORKER.submit(partial(
                                    finalize_module,
                                    module=module,
                                ))

        # LifeCycle: All sub-modules have finalized meaning quantization work is complete
        ASYNC_WORKER.join()

        # paranoid safety check
        torch_sync()
        torch_sync(device=CPU)

        total_log = {}

        for reverse_p in reversed(self.processors):
            if isinstance(reverse_p, GPTQProcessor):
                pass
            elif isinstance(reverse_p, EoraProcessor):
                pass
            elif isinstance(reverse_p, DequantizeProcessor):
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

    def crate_named_modules(self, full, is_lm_head_module, layer_index, layers_prefix, names, processor, fail_safe) -> Dict[str, NamedModule]:
        is_awq_quant = isinstance(processor, AWQProcessor)
        subset = {}
        for n in names:
            if n in full:
                subset[n] = full[n]
            # some modules have layer_modules that are dynamic based on config
            # ref: deepseek v2/v3/r1
            elif self.gptq_model.layer_modules_strict:
                raise ValueError(f"layer module item `{n}` not found in model, please check your model config.")
        skipped_modules = []
        for name in subset:
            layer_name = self.gptq_model.lm_head if is_lm_head_module else f"{layers_prefix}.{layer_index}.{name}"

            # gptq task is created and stored inside processor
            if not isinstance(subset[name], NamedModule):
                named_module = NamedModule(subset[name], name=name, full_name=layer_name,
                                           layer_index=layer_index)
                if isinstance(processor, EoraProcessor):
                    named_module.state.update({
                        "wq": processor.quantized_weights[layer_name],
                    })

                subset[name] = named_module
                full[name] = named_module

            if not is_awq_quant:
                if isinstance(processor, GPTQProcessor):
                    processor.preprocess(subset[name], fail_safe=fail_safe)
                else:
                    processor.preprocess(subset[name])
                # some modules are skipped
                if processor.is_skipped(subset[name]):
                    skipped_modules.append(name)

        if not is_awq_quant:
            for name in skipped_modules:
                subset.pop(name)
        return subset
