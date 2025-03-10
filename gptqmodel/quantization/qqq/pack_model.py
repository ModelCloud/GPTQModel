from typing import Type

import torch
import transformers
from tqdm import tqdm
from torch import nn

from gptqmodel import BACKEND
from gptqmodel.eora import find_layers, recurse_setattr
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQQuantLinear

@torch.no_grad()
def pack_model(
    model,
    quantizers,
    bits,
    group_size,
    force_layer_back_to_cpu: bool = False,
) -> Type[BaseQuantLinear]:
    CPU = torch.device("cpu")
    if force_layer_back_to_cpu:
        model.to(CPU)

    # logger.info("Packing model...")
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(
        model,
        quantizers,
        bits,
        group_size,
    )
    qlayers = find_layers(model, [QQQQuantLinear])

    pbar = tqdm(qlayers.keys(), leave=True)
    for name in pbar:
        pbar.set_description(f"Packing {name}...", refresh=True)

        scale, zero, g_idx, scale_extra = quantizers[name]
        # so far can only pack layer on CPU
        layer_device = qlayers[name].device
        qlayers[name].to(CPU)
        layers[name], scale, zero, g_idx, scale_extra = (
            layers[name].to(CPU),
            scale.to(CPU),
            zero.to(CPU),
            g_idx.to(CPU),
            scale_extra.to(CPU) if scale_extra is not None else None,
        )
        qlayers[name].pack(layers[name], scale, scale_extra)
        qlayers[name].to(layer_device)
        del layers[name]

    print("Model packed.")
    return QQQQuantLinear

def make_quant(
    module,
    names,
    bits,
    group_size,
    trainable: bool = False,
):
    if isinstance(module, QQQQuantLinear):
        return

    for name, submodule in module.named_modules():
        if name in names:
            ori_layer_device = next(submodule.parameters()).device

            if isinstance(submodule, nn.Linear):
                in_features = submodule.in_features
                out_features = submodule.out_features
            elif isinstance(submodule, nn.Conv2d):
                in_features = submodule.in_channels
                out_features = submodule.out_channels
            elif isinstance(submodule, transformers.pytorch_utils.Conv1D):
                in_features = submodule.weight.shape[0]
                out_features = submodule.weight.shape[1]
            bias = submodule.bias is not None
            new_layer = QQQQuantLinear(
                bits=bits,
                group_size=group_size,
                desc_act=False, # TODO use real value
                sym=False, # TODO use real value
                in_features=in_features,
                out_features=out_features,
                pack_dtype=torch.int32, # TODO use real value
                bias=bias,
                #weight_dtype=submodule.qweight.dtype if isinstance(submodule, BaseQuantLinear) else submodule.weight.dtype,
                name=name,
                lm_head_name="lm_head", # TODO use real value
                backend=BACKEND.QQQ,
                adapter=None, # TODO use real value
            )
            new_layer.device = ori_layer_device
            recurse_setattr(module, name, new_layer.to(ori_layer_device))
