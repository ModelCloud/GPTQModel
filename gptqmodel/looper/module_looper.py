import time
from typing import Tuple

import torch
from torch import nn

from gptqmodel.nn_modules.hooked_linear import replace_linear_with_hooked_linear
from gptqmodel.quantization.gptq import CPU
from gptqmodel.utils.logger import setup_logger
from gptqmodel.utils.model import get_module_by_name_prefix, get_device, move_to, nested_move_to, get_moe_layer_modules, \
    get_module, find_modules
from gptqmodel.utils.progress import ProgressBar
from gptqmodel.utils.torch import torch_empty_cache

logger = setup_logger()

class ModuleLooper():
    def __init__(self, ):
        self.processors = []
        self.model = None

        self.state = dict()
        pass

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except Exception:
            return getattr(self.model, item)

    def cache_inputs(self, layers, auto_gc, calibration_dataset, calibration_enable_gpu_cache):
        layer_inputs = []
        attention_masks = []
        position_ids = []
        layer_input_kwargs = []
        layer_outputs = []

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
        layers[0] = layers[0].to(self.quantize_config.device)
        ori_outside_layer_module_devices = {}
        for module_name in self.base_modules:
            module = get_module_by_name_prefix(self.model, module_name)

            if module is None:
                continue

            ori_outside_layer_module_devices[module_name] = get_device(module)
            if module is not None:
                move_to(module, cur_layer_device)
        # TODO: make this optional, backporting https://github.com/huggingface/optimum/blob/main/optimum/gptq/quantizer.py
        handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
        is_ovis = self.__class__.__name__ == "OvisGPTQ"
        self.pre_quantize_generate_hook_start()
        for example in calibration_dataset:
            for k, v in example.items():
                data_device = self.quantize_config.device if k == "pixel_values" else cur_layer_device
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
                    self.generate(inputs=example.pop("input_ids"), max_new_tokens=1024, **example)
                else:
                    self.model(**example)
            except ValueError:
                pass
        self.pre_quantize_generate_hook_end()
        handle.remove()
        move_to(layers[0], CPU)
        for module_name in self.base_modules:
            module = get_module_by_name_prefix(self.model, module_name)
            if module is not None:
                move_to(module, ori_outside_layer_module_devices[module_name])
        if auto_gc:
            torch_empty_cache()
        return attention_masks, layer_input_kwargs, layer_inputs, layer_outputs, position_ids

    def loop(self, auto_gc=True, calibration_enable_gpu_cache=True , buffered_fwd=False,):
        # TODO: lm_head quantize

        layers = get_module_by_name_prefix(self.model, self.layers_node)

        for processor in self.processors:
            processor.num_batches = len(processor.calibration_dataset)
            inputs = self.cache_inputs(layers=layers,auto_gc=auto_gc, calibration_dataset=processor.calibration_dataset,
                                                 calibration_enable_gpu_cache=calibration_enable_gpu_cache)
            processor.receive_inputs(inputs)

        layer_modules = self.layer_modules

        if not self.quantize_config.true_sequential:
            layer_modules = [sum(layer_modules, [])]

        # dynamic expert layer index for model defs
        if self.dynamic_expert_index is not None:
            num_experts = getattr(self.model.config, self.dynamic_expert_index)
            layer_modules = get_moe_layer_modules(layer_modules=self.layer_modules,
                                                  num_experts=num_experts)

        quantizers = {}

        layer_count = len(layers)
        quant_modules_pb = ProgressBar(range(layer_count + 1 if self.quantize_config.lm_head else layer_count))
        gpu_memorys = []
        cpu_memorys = []
        durations = []
        avg_losses = []
        module_names = []
        shared_kv_cache_dict = {}

        # replace linear with hooked linear
        replace_linear_with_hooked_linear(self.model)

        for module_index in quant_modules_pb:
            is_lm_head_module = module_index >= layer_count
            layer_name = self.lm_head if is_lm_head_module else f"{self.layers_node}.{module_index}.{name}"
            if is_lm_head_module:
                quant_modules_pb.set_description("Quantizing lm_head")
                module = get_module(self.model, key=self.lm_head)
                layer_inputs = self.lm_head_pre_quantize_generate_hook(layer_inputs)
            else:
                quant_modules_pb.set_description(f"Quantizing layer {module_index} of {layer_count - 1}")
                module = layers[module_index]

            if module.__class__.__name__.lower() == "MllamaCrossAttentionDecoderLayer".lower():
                # TODO FIXME: currently we not support quantizing cross attention layer (pixel_values)
                continue

            # TODO log clearml

            self.pre_quantize(module)

            cur_layer_device = get_device(module)
            full = find_modules(module, name=self.lm_head if is_lm_head_module else "")
            modules = [[self.lm_head]] if is_lm_head_module else layer_modules

            for processor in self.processors:
                attention_masks, layer_input_kwargs, layer_inputs, layer_outputs, position_ids = processor.inputs_cache

                for index, names in enumerate(modules):
                    subset = {n: full[n] for n in names if n in full}
                    skipped_modules = []

                    for name in subset:
                        if self.quantize_config.dynamic is not None:
                            if self.quantize_config.dynamic_get(layer_name=layer_name) == False:  # noqa: E712
                                logger.info(f"skip module: {layer_name}")

                                skipped_modules.append(name)
                                continue

                        # gptq task is created and stored inside processor
                        processor.preprocess(subset[name], name, layer_name, buffered_fwd)

                    for name in skipped_modules:
                        subset.pop(name)

                    if len(processor.tasks) == 0:
                        continue

                    handle = []
                    for name in subset:
                        if hasattr(subset[name], 'forward_hook'):
                            subset[name].forward_hook =  processor.task_hook(name)
                        else:
                            handle.append(subset[name].register_forward_hook(processor.task_hook(name)))

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
                                    additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(module_index - 1)

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

                    for h in handle:
                        h.remove()

                    for name in subset:
                        if hasattr(subset[name], 'forward_hook'):
                            subset[name].forward_hook = None

                    if index == len(layer_modules) - 1:
                        if auto_gc:
                            torch_empty_cache()

                    for name_index, name in enumerate(subset):
                        # TODO This doesn't update the state correctly.
                        # We want forloop{ state.update(A_processor) -> state.update(B_processor)}
                        self.state.update(processor.process(module, self.state))

