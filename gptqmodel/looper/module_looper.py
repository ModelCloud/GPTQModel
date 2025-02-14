# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import List

import torch
from gptqmodel.looper.input_cache import InputCache

from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.looper.named_module import STAT_GPTQ_FWD_TIME, NamedModule
from gptqmodel.models import BaseGPTQModel
from gptqmodel.models._const import SUPPORTS_MODULE_TYPES
from gptqmodel.nn_modules.hooked_linear import replace_linear_with_hooked_linear
from gptqmodel.quantization.gptq import CPU
from gptqmodel.utils.device import get_gpu_usage_memory, get_cpu_usage_memory
from gptqmodel.utils.logger import setup_logger
from gptqmodel.utils.model import (find_modules, get_device, get_module, get_module_by_name_prefix,
                                   get_moe_layer_modules, move_to, nested_move_to)
from gptqmodel.utils.progress import ProgressBar
from gptqmodel.utils.torch import torch_empty_cache

logger = setup_logger()


class ModuleLooper():
    def __init__(self, model: BaseGPTQModel, processors: List[LoopProcessor]):
        self.processors = processors
        self.gptq_model = model

    def cache_inputs(self, layers, auto_gc, calibration_data, calibration_enable_gpu_cache):
        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []

        cur_layer_device = get_device(layers[0])
        data_device = cur_layer_device if calibration_enable_gpu_cache else CPU

        # TODO HookLinear add register_forward_pre_hook()
        def store_input_hook(_, args, kwargs):
            # Positional arguments.
            layer_input = []
            for inp in args:
                layer_input.append(move_to(inp, data_device))
            if len(layer_input) == 0:
                # Some models put hidden_states in kwargs instead of args.
                # For example, gptj ...
                if kwargs.get("hidden_states") is not None:
                    layer_input.append(move_to(kwargs["hidden_states"], data_device))

            layer_inputs.append(layer_input)

            # Keyword arguments.
            if kwargs.get("attention_mask") is not None:
                attention_masks.append(kwargs["attention_mask"].to(data_device))
            else:
                attention_masks.append(None)

            pos_ids = kwargs.get("position_ids", None)
            if pos_ids is not None:
                position_ids.append(move_to(pos_ids, data_device))
            one_kwargs = {}
            for (k, v) in kwargs.items():  # make sure other arguments also be captured
                if k not in ["hidden_states", "attention_mask", "position_ids"]:
                    one_kwargs[k] = nested_move_to(v, data_device)
            layer_input_kwargs.append(one_kwargs)

            raise ValueError

        # move layer to target device
        layers[0] = layers[0].to(self.gptq_model.quantize_config.device)
        ori_outside_layer_module_devices = {}
        for module_name in self.gptq_model.base_modules:
            module = get_module_by_name_prefix(self.gptq_model.model, module_name)

            if module is None:
                continue

            ori_outside_layer_module_devices[module_name] = get_device(module)
            if module is not None:
                move_to(module, cur_layer_device)
        # TODO: make this optional, backporting https://github.com/huggingface/optimum/blob/main/optimum/gptq/quantizer.py
        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
        is_ovis = self.gptq_model.__class__.__name__ == "OvisGPTQ"
        self.gptq_model.pre_quantize_generate_hook_start()
        for example in calibration_data:
            for k, v in example.items():
                data_device = self.gptq_model.quantize_config.device if k == "pixel_values" else cur_layer_device
                if isinstance(v, list):
                    for module_index in range(len(v)):
                        if len(v[module_index].shape) == 1:
                            v[module_index] = v[module_index].unsqueeze(0)
                        v[module_index] = move_to(v[module_index].to(torch.bfloat16) if is_ovis else v[module_index],
                                                  data_device)
                else:
                    if len(v.shape) == 1:
                        v = v.unsqueeze(0)
                    example[k] = move_to(v, data_device)
            try:
                if is_ovis:
                    self.gptq_model.generate(inputs=example.pop("input_ids"), max_new_tokens=1024, **example)
                else:
                    self.gptq_model.model(**example)
            except ValueError:
                pass
        self.gptq_model.pre_quantize_generate_hook_end()
        handle.remove()
        move_to(layers[0], CPU)
        for module_name in self.gptq_model.base_modules:
            module = get_module_by_name_prefix(self.gptq_model.model, module_name)
            if module is not None:
                move_to(module, ori_outside_layer_module_devices[module_name])
        if auto_gc:
            torch_empty_cache()
        return InputCache(layer_inputs=layer_inputs, layer_input_kwargs=layer_input_kwargs, position_ids=position_ids,
                          attention_masks=attention_masks)

    def loop(self, auto_gc=True, calibration_enable_gpu_cache=True, buffered_fwd=False, **kwargs):
        if self.gptq_model.quantize_config.lm_head:
            if self.gptq_model.model.config.tie_word_embeddings and hasattr(self.gptq_model.model.model, "_tied_weights_keys"):
                tied_keys = self.gptq_model.model._tied_weights_keys
                for item in tied_keys:
                    if self.gptq_model.lm_head in item:
                        raise NotImplementedError("quantizing lm_head with tied weights has not been supported "
                                                  "currently")

            lm_head_module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
            if get_module(self.gptq_model.model, key=self.gptq_model.lm_head) is None:
                raise ValueError(f"could not find layer {self.gptq_model.lm_head} in the model, exit...")

            if not isinstance(lm_head_module, tuple(SUPPORTS_MODULE_TYPES)):
                raise NotImplementedError(f"This type({type(lm_head_module)}) of lm_head quantization is currently not "
                                          f"supported. SUPPORTS_MODULE_TYPES is {SUPPORTS_MODULE_TYPES}")

            lm_head_quant_config = {"bits": 8, "group_size": 32, "sym": True, "desc_act": False, "mse": 2.4}
            if self.gptq_model.quantize_config.dynamic is None:
                self.gptq_model.quantize_config.dynamic = {self.gptq_model.lm_head: lm_head_quant_config}
            elif self.gptq_model.quantize_config.dynamic_get(self.gptq_model.lm_head, default_value=None) is None:
                self.gptq_model.quantize_config.dynamic[self.gptq_model.lm_head] = lm_head_quant_config

        forward_pass_use_cache = self.gptq_model.model.config.use_cache if hasattr(self.gptq_model.model.config, "use_cache") else False
        self.gptq_model.model.config.use_cache = False

        layers = get_module_by_name_prefix(self.gptq_model.model, self.gptq_model.layers_node)

        for processor in self.processors:
            processor.num_batches = len(processor.calibration_dataset)
            input_cache = self.cache_inputs(layers=layers, auto_gc=auto_gc,
                                            calibration_data=processor.calibration_dataset,
                                            calibration_enable_gpu_cache=calibration_enable_gpu_cache)
            processor.receive_input_cache(input_cache)

        layer_modules = self.gptq_model.layer_modules

        if not self.gptq_model.quantize_config.true_sequential:
            layer_modules = [sum(layer_modules, [])]

        # dynamic expert layer index for model defs
        if self.gptq_model.dynamic_expert_index is not None:
            num_experts = getattr(self.gptq_model.model.config, self.gptq_model.dynamic_expert_index)
            layer_modules = get_moe_layer_modules(layer_modules=self.gptq_model.layer_modules,
                                                  num_experts=num_experts)

        quantizers = {}

        layer_count = len(layers)
        quant_modules_pb = ProgressBar(range(layer_count + 1 if self.gptq_model.quantize_config.lm_head else layer_count))
        gpu_memorys = []
        cpu_memorys = []
        durations = []
        avg_losses = []
        module_names = []
        shared_kv_cache_dict = {}

        # replace linear with hooked linear
        replace_linear_with_hooked_linear(self.gptq_model.model)

        for module_index in quant_modules_pb:
            is_lm_head_module = module_index >= layer_count

            if is_lm_head_module:
                quant_modules_pb.set_description("Quantizing lm_head")
                module = get_module(self.gptq_model.model, key=self.gptq_model.lm_head)
                layer_inputs = self.gptq_model.lm_head_pre_quantize_generate_hook(layer_inputs)
            else:
                quant_modules_pb.set_description(f"Quantizing layer {module_index} of {layer_count - 1}")
                module = layers[module_index]

            if module.__class__.__name__.lower() == "MllamaCrossAttentionDecoderLayer".lower():
                # TODO FIXME: currently we not support quantizing cross attention layer (pixel_values)
                continue

            self.gptq_model.pre_quantize(module)

            cur_layer_device = get_device(module)
            full = find_modules(module, name=self.gptq_model.lm_head if is_lm_head_module else "")
            modules = [[self.gptq_model.lm_head]] if is_lm_head_module else layer_modules

            for p_index, processor in enumerate(self.processors):
                if processor.logger_task is not None:
                    gpu_memory = get_gpu_usage_memory()
                    cpu_memory = get_cpu_usage_memory()
                    processor.logger_task.get_logger().report_scalar(
                        title='GPU Memory',
                        series='GPU Memory',
                        value=gpu_memory,
                        iteration=module_index,
                    )

                    processor.logger_task.get_logger().report_scalar(
                        title='CPU Memory',
                        series='CPU Memory',
                        value=cpu_memory,
                        iteration=module_index,
                    )
                    gpu_memorys.append(gpu_memory)
                    cpu_memorys.append(cpu_memory)

                layer_inputs = processor.inputs_cache.layer_inputs
                layer_input_kwargs = processor.inputs_cache.layer_input_kwargs
                position_ids = processor.inputs_cache.position_ids
                attention_masks = processor.inputs_cache.attention_masks

                subset = {}
                for index, names in enumerate(modules):
                    for n in names:
                        assert n in full, f"module {n} has wrong type, check your config"
                        subset[n] = full[n]

                    skipped_modules = []

                    for name in subset:
                        layer_name = self.gptq_model.lm_head if is_lm_head_module else f"{self.gptq_model.layers_node}.{module_index}.{name}"
                        if self.gptq_model.quantize_config.dynamic is not None:
                            if self.gptq_model.quantize_config.dynamic_get(layer_name=layer_name) == False:  # noqa: E712
                                logger.info(f"skip module: {layer_name}")

                                skipped_modules.append(name)
                                continue

                        # gptq task is created and stored inside processor
                        if not isinstance(subset[name], NamedModule):
                            named_module = NamedModule(subset[name], name=name, full_name=layer_name,
                                                      layer_index=module_index)
                            subset[name] = named_module

                        processor.preprocess(subset[name], buffered_fwd)

                    for name in skipped_modules:
                        subset.pop(name)

                    if len(processor.tasks) == 0:
                        continue

                    handle = []
                    for name in subset:
                        if hasattr(subset[name], 'forward_hook'):
                            subset[name].forward_hook = processor.preprocess_fwd_hook(name)
                        else:
                            # TODO FIXME: do we even need to hook into modules that are not quantizable?
                            assert (f"forward_hook missing for module name: `{name}`, layer name: {layer_name}")
                            handle.append(subset[name].register_forward_hook(processor.preprocess_fwd_hook(name)))

                    # logger.info(f"layer-{i}: Begin Forward() Pass")
                    fwd_start = time.time()
                    for j in range(processor.num_batches):
                        layer_input = []
                        for k, layer_inp in enumerate(layer_inputs[j]):
                            layer_input.append(move_to(layer_inp, cur_layer_device))

                        mask = attention_masks[j]
                        layer_attention_mask = mask if mask is None else move_to(mask, cur_layer_device)

                        additional_layer_inputs = {"attention_mask": layer_attention_mask}
                        layer_position_ids = (
                            None if not position_ids else move_to(position_ids[j], cur_layer_device)
                        )
                        if layer_position_ids is not None:
                            additional_layer_inputs["position_ids"] = layer_position_ids
                        for k, v in layer_input_kwargs[j].items():
                            additional_layer_inputs[k] = nested_move_to(v, cur_layer_device)

                        with torch.no_grad():
                            # reuse_kv is a flag to reuse the kv cache, only for the hamba model
                            if hasattr(module, "reuse_kv"):
                                if module.reuse_kv:
                                    additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(
                                        module_index - 1)

                                layer_output = module(*layer_input) if is_lm_head_module else module(*layer_input,
                                                                                                     **additional_layer_inputs)
                                if shared_kv_cache_dict.get(module_index) is None:
                                    shared_kv_cache_dict[module_index] = layer_output[-1]
                            else:
                                module(*layer_input) if is_lm_head_module else module(*layer_input,
                                                                                      **additional_layer_inputs)

                        del layer_input
                        del additional_layer_inputs

                    fwd_end = time.time()
                    fwd_time = fwd_end - fwd_start

                    # TODO fix me: don't use string
                    # module.state.update({STAT_GPTQ_FWD_TIME: fwd_time})

                    for h in handle:
                        h.remove()

                    for name in subset:
                        if hasattr(subset[name], 'forward_hook'):
                            subset[name].forward_hook = None

                    for name_index, name in enumerate(subset):
                        processor.process(module=subset[name])

                        processor.post_process(module=subset[name])

                    if index == len(layer_modules) - 1:
                        if auto_gc:
                            torch_empty_cache()

                is_last_module = module_index == len(quant_modules_pb) - 1
                layer_outputs = []
                if not is_last_module:
                    for j in range(processor.num_batches):
                        layer_input = []
                        for k, layer_inp in enumerate(layer_inputs[j]):
                            layer_input.append(move_to(layer_inp, cur_layer_device))

                        mask = attention_masks[j]
                        layer_attention_mask = mask if mask is None else move_to(mask, cur_layer_device)

                        additional_layer_inputs = {"attention_mask": layer_attention_mask}
                        layer_position_ids = None if not position_ids else move_to(position_ids[j], cur_layer_device)
                        if layer_position_ids is not None:
                            additional_layer_inputs["position_ids"] = layer_position_ids
                        for k, v in layer_input_kwargs[j].items():
                            additional_layer_inputs[k] = nested_move_to(v, cur_layer_device)

                        if hasattr(module, "reuse_kv"):
                            if module.reuse_kv:
                                additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(module_index - 1)

                        with torch.no_grad():
                            layer_output = move_to(
                                module(*layer_input)[0] if is_lm_head_module else
                                module(*layer_input, **additional_layer_inputs)[0],
                                cur_layer_device if calibration_enable_gpu_cache else CPU,
                            )
                            layer_outputs.append([layer_output])

                        del layer_input
                        del additional_layer_inputs
                        if processor.num_batches > 1 and j == processor.num_batches - 1:
                            if auto_gc:
                                torch_empty_cache()

                # TODO move to processor?
                if not is_lm_head_module:
                    layers[module_index] = self.gptq_model.post_quantize(module)
                else:
                    self.gptq_model.post_quantize(module)

                processor.clear_cache_data()

                processor.receive_layer_inputs(layer_outputs)

                # if last processor, we need to call finalize in reverse
                if p_index == len(self.processors) - 1:
                    for reverse_p in reversed(self.processors):
                        for name in subset:
                            reverse_p.submodule_finalize(subset[name])

                del module

                if auto_gc:
                    torch_empty_cache()

        # logger.info(f"Quantization summary:\n{self.quant_log}")
        # for module_log in self.quant_log:
        #     logger.info(module_log)
        if any(p.logger_task for p in self.processors):
            from gptqmodel.utils.plotly import create_plotly

        for reverse_p in reversed(self.processors):
            reverse_p.model_finalize(model=self.gptq_model, **kwargs)

            if reverse_p.logger_task is not None:
                x = list(range(layer_count))
                gpu_fig = create_plotly(x=x, y=gpu_memorys, xaxis_title="layer", yaxis_title="GPU usage (GB)")
                cpu_fig = create_plotly(x=x, y=cpu_memorys, xaxis_title="layer", yaxis_title="CPU usage (GB)")
                loss_fig = create_plotly(x=module_names, y=avg_losses, xaxis_title="layer", yaxis_title="loss")
                time_fig = create_plotly(x=module_names, y=durations, xaxis_title="layer", yaxis_title="time")

                with reverse_p.logger_task.get_logger() as l:
                    l.report_plotly('GPU Memory', 'GPU Memory', gpu_fig)
                    l.report_plotly('CPU Memory', 'CPU Memory', cpu_fig)
                    l.report_plotly('avg_loss', 'avg_loss', loss_fig)
                    l.report_plotly('quant_time', 'quant_time', time_fig)


        self.gptq_model.model.config.use_cache = forward_pass_use_cache

        self.gptq_model.quantized = True
        if auto_gc:
            torch_empty_cache()

        # TODO return
        # return self.gptq_model.quant_log