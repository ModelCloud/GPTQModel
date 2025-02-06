import torch
import torch.nn as nn
from gptqmodel import GPTQModel
from .modelutils import find_layers
from .eora_calibration_dataloader import get_loaders
from gptqmodel.models.base import * 
from ..utils.logger import setup_logger

from gptqmodel.utils.model import get_module_by_name_prefix, get_device, move_to, nested_move_to, torch_empty_cache, get_moe_layer_modules, find_modules
## import const
from gptqmodel.models._const import CPU, CUDA, CUDA_0
from gptqmodel.utils.progress import ProgressBar
from gptqmodel.nn_modules.hooked_linear import replace_linear_with_hooked_linear
import time
logger = setup_logger()

@torch.no_grad()
def get_eora(model_id, quant_config, data_name, quantized_weights, eora_nsamples, eora_rank, dev):
    print('Starting ...')


    ## get the full-precision model
    model = GPTQModel.load(model_id_or_path=model_id, quantize_config=quant_config)
    layers_node = model.layers_node
    model = model.model
    ## not quite sure if this is needed for other type of model besides LLaMA
    model.seqlen = 2048
    ## prepare eora dataloader
    dataloader = get_loaders(data_name=data_name, nsamples=eora_nsamples, seqlen=model.seqlen, model=model_id)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    try:
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    except:
        print("Current model does not have rotary_emb")


    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (eora_nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    ## this only apply to normal attention (flash attention will require different shape)
    cache = {'i': 0, 'attention_mask': None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            ## need to add this due to version shift of transformers from v4.36 to 4.49 
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_embeddings = cache['position_embeddings']

    print('Ready.')
    lowrank_dict = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        
        sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}
            
            subset_eigen_scaling_diag_matrix = {}
            for name in subset:
                subset_eigen_scaling_diag_matrix[name] = 0

            def hook(name):

                def tmpp(_, input, output):
                    inp = input[0].detach().float()
                    if inp.dim() == 2:
                        inp = inp.unsqueeze(0)
                    
                    tmp = inp.shape[0]
                    adds = torch.matmul(inp.transpose(1,2), inp)
                    adds_sum = torch.sum(adds, dim=0)
                    subset_eigen_scaling_diag_matrix[name] *= eora_nsamples / (eora_nsamples+tmp)
                    
                    subset_eigen_scaling_diag_matrix[name] += adds_sum / eora_nsamples
                    
                    del inp, adds, adds_sum, output
                    torch.cuda.empty_cache()
                return tmpp
            
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(hook(name)))

            for j in range(eora_nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings = position_embeddings)[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_name = f"{layers_node}.{i}.{name}"
                print(layer_name)
                print('Start eigen projection ...')
                original_weight = subset[name].weight.data
                
                quantized_weight = quantized_weights[layer_name].to(dev)

                delta = original_weight - quantized_weight

                ## save this later for SVD

                raw_scaling_diag_matrix = subset_eigen_scaling_diag_matrix[name].double().to("cuda")
                
                L, Q = torch.linalg.eigh(raw_scaling_diag_matrix)
                if (L < 0).any().item():
                    print(f"found negative eigenvalues in {name}")
                    minimum = torch.min(L[L > 0])
                    L[L < 0] = minimum

                sqrtEigenvalues = torch.sqrt(L)
                scaling_diag_matrix = Q @ torch.diag(sqrtEigenvalues)
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception as e:
                    print("Warning: scaling_diag_matrix is not full rank!")
                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)

                scaling_diag_matrix = scaling_diag_matrix.float()
                scaling_matrix_inv = scaling_matrix_inv.float()
                ##
                delta_scale = torch.matmul(delta.to(torch.float32), scaling_diag_matrix)

                r=eora_rank

                U, S, V = torch.linalg.svd(delta_scale, full_matrices=False)
                lowrank_r = r
                truc_s = S[:lowrank_r]
                truc_u = U[:, :lowrank_r]
                truc_v = torch.matmul(V[:lowrank_r, :], scaling_matrix_inv)
                truc_sigma = torch.diag(truc_s)
                
                sqrtS = torch.sqrt(truc_sigma)
                B = torch.matmul(truc_u, sqrtS).to(quantized_weight.dtype)
                A = torch.matmul(sqrtS, truc_v).to(quantized_weight.dtype)

                comp_weight = quantized_weight + B@A

                subset[name].weight.data = comp_weight.to(subset[name].weight.data.dtype)

                lowrank_dict[f'{layer_name}.lora_A.weight'] = A.cpu().to(torch.float16)
                lowrank_dict[f'{layer_name}.lora_B.weight'] = B.cpu().to(torch.float16)
                del B, A, quantized_weight, U, S, V, L, Q

               

        for j in range(eora_nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_embeddings = position_embeddings)[0]


        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    del model
    torch.cuda.empty_cache()

    return lowrank_dict
 


@torch.no_grad()
def get_eora_optimize(model_id, quant_config, quantized_weights, calibration_dataset, batch_size, eora_rank, calibration_enable_gpu_cache = True, auto_gc = True):
    raise NotImplementedError
    # print('Starting ...')

    # ## get the full-precision model
    # model = GPTQModel.load(model_id_or_path=model_id, quantize_config=quant_config, device=torch.device("cuda"))
    # ## 
    # base_modules = model.base_modules
    # layers_node = model.layers_node
    # layer_modules = model.layer_modules
    # dynamic_expert_index = model.dynamic_expert_index
    # ## 
    # min_calibration_dataset_size = 256
    # min_calibration_dataset_input_ids_avg_length = 256

    # if len(calibration_dataset) < min_calibration_dataset_size:
    #     logger.warning(f"Calibration dataset size should be more than {min_calibration_dataset_size}. "
    #                     f"Current: {len(calibration_dataset)}.")
        
    # calibration_dataset = model.prepare_dataset(calibration_dataset, batch_size,)

    # # Calculate the average length of the average input_ids
    # total_input_ids_length = 0
    # max_input_id_length = 0
    # for row in calibration_dataset:
    #     input_ids = row["input_ids"]
    #     if isinstance(input_ids, torch.Tensor):
    #         if input_ids.dim() <= 2:
    #             input_ids_length = input_ids.shape[-1]
    #         else:
    #             raise ValueError(
    #                 "Expected a 1-dimensional tensor or 2-dimensional tensor for 'input_ids', but got a tensor with {0} dimensions.".format(
    #                     input_ids.dim()))
    #     else:
    #         input_ids_length = len(input_ids)

    #     if input_ids_length > max_input_id_length:
    #         max_input_id_length = input_ids_length
    #     total_input_ids_length += input_ids_length
    # avg = total_input_ids_length / len(calibration_dataset)

    # if avg < min_calibration_dataset_input_ids_avg_length:
    #     logger.warning(f"The average length of input_ids of calibration_dataset should be greater than "
    #                     f"{min_calibration_dataset_input_ids_avg_length}: actual avg: {avg}.")

    # ## probably do not need to tackle lm_head (skip)
    # model = model.model
    # forward_pass_use_cache = model.config.use_cache if hasattr(model.config, "use_cache") else False
    # model.config.use_cache = False

    # layer_inputs = []
    # attention_masks = []
    # position_ids = []
    # layer_input_kwargs = []
    # layer_outputs = []
    
    # num_batches = len(calibration_dataset)
    # layers = get_module_by_name_prefix(model, layers_node)

    # cur_layer_device = get_device(layers[0])
    # data_device = cur_layer_device if calibration_enable_gpu_cache else CPU

    # #
    # def store_input_hook(_, args, kwargs):
    #     # Positional arguments.
    #     layer_input = []
    #     for inp in args:
    #         layer_input.append(move_to(inp, data_device))
    #     if len(layer_input) == 0:
    #         # Some models put hidden_states in kwargs instead of args.
    #         # For example, gptj ...
    #         if kwargs.get("hidden_states") is not None:
    #             layer_input.append(move_to(kwargs["hidden_states"], data_device))

    #     layer_inputs.append(layer_input)

    #     # Keyword arguments.
    #     if kwargs.get("attention_mask") is not None:
    #         attention_masks.append(kwargs["attention_mask"].to(data_device))
    #     else:
    #         attention_masks.append(None)

    #     pos_ids = kwargs.get("position_ids", None)
    #     if pos_ids is not None:
    #         position_ids.append(move_to(pos_ids, data_device))
    #     one_kwargs = {}
    #     for (k, v) in kwargs.items():  # make sure other arguments also be captured
    #         if k not in ["hidden_states", "attention_mask", "position_ids"]:
    #             one_kwargs[k] = nested_move_to(v, data_device)
    #     layer_input_kwargs.append(one_kwargs)

    # # move layer to target device
    # print(f"quant_config.device {quant_config.device}")
    # layers[0] = layers[0].to(quant_config.device)
    # # model.model.embed_tokens = model.model.embed_tokens.to("cuda:0")
    # # model.model.norm = model.model.norm.to("cuda:0")

    # ori_outside_layer_module_devices = {}
    # for module_name in base_modules:
    #     module = get_module_by_name_prefix(model, module_name)

    #     if module is None:
    #         continue

    #     ori_outside_layer_module_devices[module_name] = get_device(module)
    #     if module is not None:
    #         move_to(module, cur_layer_device)

    # handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
    
    # # model.model.embed_tokens = model.model.embed_tokens.to("cuda:0")
    # # model.model.norm = model.model.norm.to("cuda:0")

    # for example in calibration_dataset:
    #     for k, v in example.items():
    #         if isinstance(v, list):
    #             for i in range(len(v)):
    #                 if len(v[i].shape) == 1:
    #                     v[i] = v[i].unsqueeze(0)
    #                 v[i] = move_to(v[i], cur_layer_device)
                    
    #         else:
    #             if len(v.shape) == 1:
    #                 v = v.unsqueeze(0)
    #             example[k] = move_to(v, cur_layer_device)
                
    #     try:
    #         ### Here I don't know why there is a device error with model on gpu and example on cpu
    #         # print(example['input_ids'].device)
    #         # print(example['attention_mask'].device)
    #         print("sean 2 debug")
    #         for name, layer in model.named_parameters():    
    #             print(name, layer, layer.device)
    #         example['input_ids'] = example['input_ids'].to("cuda:0")
    #         example['attention_mask'] = example['attention_mask'].to("cuda:0")
    #         model(**example)
    #     except ValueError:
    #         pass
    
    # handle.remove()
    # move_to(layers[0], CPU)
    # model.model.embed_tokens = model.model.embed_tokens.to(CPU)
    # model.model.norm = model.model.norm.to(CPU)

    # for module_name in base_modules:
    #     module = get_module_by_name_prefix(model, module_name)
    #     if module is not None:
    #         move_to(module, ori_outside_layer_module_devices[module_name])

    # if auto_gc:
    #     torch_empty_cache()

    # layer_modules = [sum(layer_modules, [])]

    # # dynamic expert layer index for model defs
    # if dynamic_expert_index is not None:
    #     num_experts = getattr(model.config, dynamic_expert_index)
    #     layer_modules = get_moe_layer_modules(layer_modules=layer_modules,
    #                                             num_experts=num_experts)

    
    # layer_count = len(layers)
    # layer_pb = ProgressBar(range(layer_count))
    # gpu_memorys = []
    # cpu_memorys = []
    # durations = []
    # avg_losses = []
    # module_names = []
    # shared_kv_cache_dict = {}

    # # replace linear with hooked linear
    # replace_linear_with_hooked_linear(model)

    # lowrank_dict = {}
    # for i in layer_pb:
    #     layer_pb.set_description(f"Construction EoRA for layer {i} of {layer_count - 1}")
    #     layer = layers[i]

    #     if get_device(layer) == CPU and quant_config.device != CPU:
    #         move_to(layer, quant_config.device)
        
    #     cur_layer_device = get_device(layer)
            
    #     full = find_modules(layer, name="")
    #     modules = layer_modules
    #     for index, names in enumerate(modules):
    #         subset = {n: full[n] for n in names if n in full}

    #         subset_eigen_scaling_diag_matrix = {}
    #         for name in subset:
    #             subset_eigen_scaling_diag_matrix[name] = 0

    #         eigen_nsamples = len(calibration_dataset)
    #         print(f"eigen_nsamples {eigen_nsamples}")
    #         def hook(name):

    #             def tmpp(_, input, output):
    #                 inp = input[0].detach().float()
    #                 if inp.dim() == 2:
    #                     inp = inp.unsqueeze(0)
                    
    #                 tmp = inp.shape[0]
    #                 adds = torch.matmul(inp.transpose(1,2), inp)
    #                 adds_sum = torch.sum(adds, dim=0)
                    
    #                 subset_eigen_scaling_diag_matrix[name] *= eigen_nsamples / (eigen_nsamples+tmp)
                    
    #                 subset_eigen_scaling_diag_matrix[name] += adds_sum / eigen_nsamples
                    
    #                 del inp, adds, adds_sum, output
    #                 torch.cuda.empty_cache()
    #             return tmpp

    #         handle = []
    #         for name in subset:
    #             if hasattr(subset[name], 'forward_hook'):
    #                 subset[name].forward_hook = hook(name)
    #             else:
    #                 handle.append(subset[name].register_forward_hook(hook(name)))

    #         fwd_start = time.time()
    #         for j in range(num_batches):
    #             layer_input = []
    #             for k, layer_inp in enumerate(layer_inputs[j]):
    #                 layer_input.append(move_to(layer_inp, cur_layer_device))

    #             mask = attention_masks[j]
    #             layer_attention_mask = mask if mask is None else move_to(mask, cur_layer_device)

    #             additional_layer_inputs = {"attention_mask": layer_attention_mask}
    #             layer_position_ids = (
    #                 None if not position_ids else move_to(position_ids[j], cur_layer_device)
    #             )
    #             if layer_position_ids is not None:
    #                 additional_layer_inputs["position_ids"] = layer_position_ids
    #             for k, v in layer_input_kwargs[j].items():
    #                 additional_layer_inputs[k] = nested_move_to(v, cur_layer_device)

    #             with torch.no_grad():
    #                 # reuse_kv is a flag to reuse the kv cache, only for the hamba model
    #                 if hasattr(layer, "reuse_kv"):
    #                     if layer.reuse_kv:
    #                         additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(i - 1)

    #                     layer_output = layer(*layer_input, **additional_layer_inputs)
    #                     if shared_kv_cache_dict.get(i) is None:
    #                         shared_kv_cache_dict[i] = layer_output[-1]
    #                 else:
    #                     layer(*layer_input, **additional_layer_inputs)

    #             del layer_input
    #             del additional_layer_inputs

    #         fwd_end = time.time()
    #         fwd_time = fwd_end - fwd_start

    #         for h in handle:
    #             h.remove()

    #         for name in subset:
    #             if hasattr(subset[name], 'forward_hook'):
    #                 subset[name].forward_hook = None

    #         if index == len(layer_modules) - 1:
    #             if auto_gc:
    #                 torch_empty_cache()

    #         for name_index, name in enumerate(subset):
    #             layer_name = f"{layers_node}.{i}.{name}"
    #             layer_pb.set_description(f"Generating EoRA of {name} in layer {i} of {layer_count - 1}")

    #             original_weight = subset[name].weight.data

    #             dev = original_weight.device

    #             quantized_weight = quantized_weights[layer_name].to(dev)

    #             delta = original_weight - quantized_weight

    #             ## save this later for SVD

    #             raw_scaling_diag_matrix = subset_eigen_scaling_diag_matrix[name].double().to(dev)
                
    #             L, Q = torch.linalg.eigh(raw_scaling_diag_matrix)
    #             if (L < 0).any().item():
    #                 print(f"found negative eigenvalues in {name}")
    #                 minimum = torch.min(L[L > 0])
    #                 L[L < 0] = minimum
                
    #             sqrtEigenvalues = torch.sqrt(L)
    #             scaling_diag_matrix = Q @ torch.diag(sqrtEigenvalues)
    #             try:
    #                 scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
    #             except Exception as e:
    #                 print("Warning: scaling_diag_matrix is not full rank!")
    #                 scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
    #                 scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)

    #             scaling_diag_matrix = scaling_diag_matrix.float()
    #             scaling_matrix_inv = scaling_matrix_inv.float()
    #             ##
    #             delta_scale = torch.matmul(delta.to(torch.float32), scaling_diag_matrix)

    #             r=eora_rank

    #             U, S, V = torch.linalg.svd(delta_scale, full_matrices=False)
    #             lowrank_r = r
    #             truc_s = S[:lowrank_r]
    #             truc_u = U[:, :lowrank_r]
    #             truc_v = torch.matmul(V[:lowrank_r, :], scaling_matrix_inv)
    #             truc_sigma = torch.diag(truc_s)
                
    #             sqrtS = torch.sqrt(truc_sigma)
    #             B = torch.matmul(truc_u, sqrtS).to(quantized_weight.dtype)
    #             A = torch.matmul(sqrtS, truc_v).to(quantized_weight.dtype)

    #             comp_weight = quantized_weight + B@A

    #             subset[name].weight.data = comp_weight.to(subset[name].weight.data.dtype)

    #             lowrank_dict[f'{layer_name}.lora_A.weight'] = A.cpu().to(torch.float16)
    #             lowrank_dict[f'{layer_name}.lora_B.weight'] = B.cpu().to(torch.float16)
    #             del B, A, quantized_weight, U, S, V, L, Q

    #     for j in range(num_batches):
    #         layer_input = []
    #         for k, layer_inp in enumerate(layer_inputs[j]):
    #             layer_input.append(move_to(layer_inp, cur_layer_device))

    #         mask = attention_masks[j]
    #         layer_attention_mask = mask if mask is None else move_to(mask, cur_layer_device)

    #         additional_layer_inputs = {"attention_mask": layer_attention_mask}
    #         layer_position_ids = None if not position_ids else move_to(position_ids[j], cur_layer_device)
    #         if layer_position_ids is not None:
    #             additional_layer_inputs["position_ids"] = layer_position_ids
    #         for k, v in layer_input_kwargs[j].items():
    #             additional_layer_inputs[k] = nested_move_to(v, cur_layer_device)

    #         if hasattr(layer, "reuse_kv"):
    #             if layer.reuse_kv:
    #                 additional_layer_inputs["kv_last_layer"] = shared_kv_cache_dict.get(i - 1)

    #         with torch.no_grad():
    #             layer_output = move_to(
    #                 layer(*layer_input, **additional_layer_inputs)[0],
    #                 cur_layer_device if calibration_enable_gpu_cache else CPU,
    #             )
    #             layer_outputs.append([layer_output])

    #         del layer_input
    #         del additional_layer_inputs
    #         if num_batches > 1 and j == num_batches - 1:
    #             if auto_gc:
    #                 torch_empty_cache()


    #     move_to(layer, CPU)
    #     del layer
    #     del layer_inputs
    #     layer_inputs, layer_outputs = (
    #         layer_outputs,
    #         [],
    #     )
    #     if auto_gc:
    #         torch_empty_cache()
        
    #     model.config.use_cache = forward_pass_use_cache
    #     if auto_gc:
    #         torch_empty_cache()
        
    # return lowrank_dict
