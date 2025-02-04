import torch
import torch.nn as nn
from gptqmodel import GPTQModel
from .modelutils import find_layers
from .eora_calibration_dataloader import get_loaders

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

                lowrank_dict[f'{layer_name}.lora_A.weight'] = A.cpu()
                lowrank_dict[f'{layer_name}.lora_B.weight'] = B.cpu()
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
