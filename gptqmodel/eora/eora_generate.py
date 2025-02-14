from typing import Dict, List, Optional, Union

import torch
from gptqmodel.models._const import CPU, SUPPORTS_MODULE_TYPES
from gptqmodel.nn_modules.hooked_linear import replace_linear_with_hooked_linear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.logger import setup_logger
from gptqmodel.utils.model import (find_modules, get_device, get_module, get_module_by_name_prefix,
                                   get_moe_layer_modules, move_to, nested_move_to)
from gptqmodel.utils.progress import ProgressBar
from gptqmodel.utils.torch import torch_empty_cache

logger = setup_logger()

def eora_generate(
        model,
        calibration_dataset: Union[List[Dict[str, Union[List[int], torch.LongTensor]]], List[str], List[int]],
        batch_size: int = 1,
        quantized_weights: Dict = None,
        lora_rank: int = 64,
        calibration_enable_gpu_cache: bool = True,
        # Setting a fixed calibration_dataset_concat_size may improve the performance of the quantized model.
        calibration_dataset_concat_size: Optional[int] = None,
        auto_gc: bool = True,
) -> Dict[str, torch.Tensor]:
    print('Starting EoRA...')

    if model.quantized:
        raise EnvironmentError("quantize() is called a model that is already quantized")

    if len(calibration_dataset) == 0:
        raise ValueError("Calibration dataset must not be empty.")

    min_calibration_dataset_size = 256
    min_calibration_dataset_input_ids_avg_length = 256

    if len(calibration_dataset) < min_calibration_dataset_size:
        logger.warning(f"Calibration dataset size should be more than {min_calibration_dataset_size}. "
                       f"Current: {len(calibration_dataset)}.")

    if model.quantize_config.format == FORMAT.BITBLAS:
        from ..nn_modules.qlinear.bitblas import BITBLAS_AVAILABLE, BITBLAS_INSTALL_HINT
        if BITBLAS_AVAILABLE is False:
            raise ValueError(BITBLAS_INSTALL_HINT)

    calibration_dataset = model.prepare_dataset(calibration_dataset=calibration_dataset,
                                               calibration_dataset_concat_size=calibration_dataset_concat_size,
                                               batch_size=batch_size)

    # Calculate the average length of the average input_ids
    total_input_ids_length = 0
    max_input_id_length = 0
    for row in calibration_dataset:
        input_ids = row["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() <= 2:
                input_ids_length = input_ids.shape[-1]
            else:
                raise ValueError(
                    "Expected a 1-dimensional tensor or 2-dimensional tensor for 'input_ids', but got a tensor with {0} dimensions.".format(
                        input_ids.dim()))
        else:
            input_ids_length = len(input_ids)

        if input_ids_length > max_input_id_length:
            max_input_id_length = input_ids_length
        total_input_ids_length += input_ids_length
    avg = total_input_ids_length / len(calibration_dataset)

    if avg < min_calibration_dataset_input_ids_avg_length:
        logger.warning(f"The average length of input_ids of calibration_dataset should be greater than "
                       f"{min_calibration_dataset_input_ids_avg_length}: actual avg: {avg}.")

    if model.quantize_config.lm_head:
        if model.model.config.tie_word_embeddings and hasattr(model.model.model, "_tied_weights_keys"):
            tied_keys = model.model._tied_weights_keys
            for item in tied_keys:
                if model.lm_head in item:
                    raise NotImplementedError("quantizing lm_head with tied weights has not been supported "
                                              "currently")

        lm_head_module = get_module(model.model, key=model.lm_head)
        if get_module(model.model, key=model.lm_head) is None:
            raise ValueError(f"could not find layer {model.lm_head} in the model, exit...")

        if not isinstance(lm_head_module, tuple(SUPPORTS_MODULE_TYPES)):
            raise NotImplementedError(f"This type({type(lm_head_module)}) of lm_head quantization is currently not "
                                      f"supported. SUPPORTS_MODULE_TYPES is {SUPPORTS_MODULE_TYPES}")

        lm_head_quant_config = {"bits": 8, "group_size": 32, "sym": True, "desc_act": False, "mse": 2.4}
        if model.quantize_config.dynamic is None:
            model.quantize_config.dynamic = {model.lm_head: lm_head_quant_config}
        elif model.quantize_config.dynamic_get(model.lm_head, default_value=None) is None:
            model.quantize_config.dynamic[model.lm_head] = lm_head_quant_config

    forward_pass_use_cache = model.model.config.use_cache if hasattr(model.model.config, "use_cache") else False
    model.model.config.use_cache = False

    layer_inputs = []
    attention_masks = []
    position_ids = []
    layer_input_kwargs = []
    layer_outputs = []

    num_batches = len(calibration_dataset)
    layers = get_module_by_name_prefix(model.model, model.layers_node)

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
    layers[0] = layers[0].to(model.quantize_config.device)

    ori_outside_layer_module_devices = {}
    for module_name in model.base_modules:
        module = get_module_by_name_prefix(model.model, module_name)

        if module is None:
            continue

        ori_outside_layer_module_devices[module_name] = get_device(module)
        if module is not None:
            move_to(module, cur_layer_device)

    # TODO: make this optional, backporting https://github.com/huggingface/optimum/blob/main/optimum/gptq/quantizer.py
    handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
    is_ovis = model.__class__.__name__ == "OvisGPTQ"
    model.pre_quantize_generate_hook_start()
    for example in calibration_dataset:
        for k, v in example.items():
            data_device = model.quantize_config.device if k == "pixel_values" else cur_layer_device
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
                model.generate(inputs=example.pop("input_ids"), max_new_tokens=1024, **example)
            else:
                model.model(**example)
        except ValueError:
            pass
    model.pre_quantize_generate_hook_end()
    handle.remove()

    move_to(layers[0], CPU)

    for module_name in model.base_modules:
        module = get_module_by_name_prefix(model.model, module_name)
        if module is not None:
            move_to(module, ori_outside_layer_module_devices[module_name])

    if auto_gc:
        torch_empty_cache()

    layer_modules = model.layer_modules
    layer_modules = [sum(layer_modules, [])]

    # dynamic expert layer index for model defs
    if model.dynamic_expert_index is not None:
        num_experts = getattr(model.model.config, model.dynamic_expert_index)
        layer_modules = get_moe_layer_modules(layer_modules=layer_modules,
                                              num_experts=num_experts)

    layer_count = len(layers)
    quant_modules_pb = ProgressBar(range(layer_count + 1 if model.quantize_config.lm_head else layer_count))
    shared_kv_cache_dict = {}

    # replace linear with hooked linear
    replace_linear_with_hooked_linear(model.model)

    lowrank_dict = {}
    for module_index in quant_modules_pb:
        is_lm_head_module = module_index >= layer_count
        if is_lm_head_module:
            quant_modules_pb.set_description("Quantizing lm_head")
            module = get_module(model.model, key=model.lm_head)
            layer_inputs = model.lm_head_pre_quantize_generate_hook(layer_inputs)
        else:
            quant_modules_pb.set_description(f"Construction EoRA for layer {module_index} of {layer_count - 1}")
            module = layers[module_index]

        model.pre_quantize(module)

        cur_layer_device = get_device(module)
        full = find_modules(module, name=model.lm_head if is_lm_head_module else "")
        modules = [[model.lm_head]] if is_lm_head_module else layer_modules
        for index, names in enumerate(modules):
            # TODO Need to be consistent with quantization and skip some modules according to dynamic.
            subset = {n: full[n] for n in names if n in full}

            subset_eigen_scaling_diag_matrix = {}
            for name in subset:
                subset_eigen_scaling_diag_matrix[name] = 0

            eigen_nsamples = len(calibration_dataset)

            def hook(name):

                def tmpp(_, input, output):
                    inp = input[0].detach().float()
                    if inp.dim() == 2:
                        inp = inp.unsqueeze(0)

                    tmp = inp.shape[0]
                    adds = torch.matmul(inp.transpose(1, 2), inp)
                    adds_sum = torch.sum(adds, dim=0)

                    subset_eigen_scaling_diag_matrix[name] *= eigen_nsamples / (eigen_nsamples + tmp)

                    subset_eigen_scaling_diag_matrix[name] += adds_sum / eigen_nsamples

                    del inp, adds, adds_sum, output
                    torch.cuda.empty_cache()

                return tmpp

            handle = []
            for name in subset:
                if hasattr(subset[name], 'forward_hook'):
                    subset[name].forward_hook = hook(name)
                else:
                    handle.append(subset[name].register_forward_hook(hook(name)))

            for j in range(num_batches):
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

            for h in handle:
                h.remove()

            for name in subset:
                if hasattr(subset[name], 'forward_hook'):
                    subset[name].forward_hook = None

            if index == len(layer_modules) - 1:
                if auto_gc:
                    torch_empty_cache()

            for name_index, name in enumerate(subset):
                layer_name = model.lm_head if is_lm_head_module else f"{model.layers_node}.{module_index}.{name}"
                quant_modules_pb.set_description(
                    f"Generating EoRA of {name} in layer {module_index} of {layer_count - 1}")

                original_weight = subset[name].weight.data

                dev = original_weight.device

                quantized_weight = quantized_weights[layer_name].to(dev)

                delta = original_weight - quantized_weight

                ## save this later for SVD

                raw_scaling_diag_matrix = subset_eigen_scaling_diag_matrix[name].double().to(dev)

                L, Q = torch.linalg.eigh(raw_scaling_diag_matrix)
                if (L < 0).any().item():
                    print(f"found negative eigenvalues in {name}")
                    minimum = torch.min(L[L > 0])
                    L[L < 0] = minimum

                sqrtEigenvalues = torch.sqrt(L)
                scaling_diag_matrix = Q @ torch.diag(sqrtEigenvalues)
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception:
                    print("Warning: scaling_diag_matrix is not full rank!")
                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)

                scaling_diag_matrix = scaling_diag_matrix.float()
                scaling_matrix_inv = scaling_matrix_inv.float()
                ##
                delta_scale = torch.matmul(delta.to(torch.float32), scaling_diag_matrix)

                r = lora_rank

                U, S, V = torch.linalg.svd(delta_scale, full_matrices=False)
                lowrank_r = r
                truc_s = S[:lowrank_r]
                truc_u = U[:, :lowrank_r]
                truc_v = torch.matmul(V[:lowrank_r, :], scaling_matrix_inv)
                truc_sigma = torch.diag(truc_s)

                sqrtS = torch.sqrt(truc_sigma)
                B = torch.matmul(truc_u, sqrtS).to(quantized_weight.dtype)
                A = torch.matmul(sqrtS, truc_v).to(quantized_weight.dtype)

                comp_weight = quantized_weight + B @ A

                subset[name].weight.data = comp_weight.to(subset[name].weight.data.dtype)

                lowrank_dict[f'{layer_name}.lora_A.weight'] = A.cpu().to(torch.float16)
                lowrank_dict[f'{layer_name}.lora_B.weight'] = B.cpu().to(torch.float16)
                del B, A, quantized_weight, U, S, V, L, Q
        is_last_quant = module_index == len(quant_modules_pb) - 1
        if not is_last_quant:
            for j in range(num_batches):
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
                if num_batches > 1 and j == num_batches - 1:
                    if auto_gc:
                        torch_empty_cache()

        if not is_lm_head_module:
            layers[module_index] = model.post_quantize(module)
        else:
            model.post_quantize(module)

        del module
        del layer_inputs

        if not is_last_quant:
            layer_inputs, layer_outputs = (
                layer_outputs,
                [],
            )  # TODO: is it really OK to cache only the first positional argument?

        if auto_gc:
            torch_empty_cache()

    model.model.config.use_cache = forward_pass_use_cache
    if auto_gc:
        torch_empty_cache()

    return lowrank_dict
