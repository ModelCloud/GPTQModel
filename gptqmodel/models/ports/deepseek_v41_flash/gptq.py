# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Projection-level GPTQ for the DeepSeek V4.1 source checkpoint."""

import math
from collections import defaultdict

import torch
from safetensors import safe_open
from torch import nn

from ....looper.gptq_processor import clone_gptq_config_for_module
from ....looper.loop_processor import prepare_calibration_dataset_once
from ....nn_modules.qlinear.torch import TorchLinear
from ....quantization.config import FORMAT, METHOD, CalibrationConfig
from ....quantization.dtype import dequantize_fp8, is_fp4_packed_dtype
from ....quantization.gptq import GPTQ
from ....utils.model_dequant import dequantize_deepseek_v4_fp4_expert
from ....utils.packing import hf_pack_layer
from ....utils.torch import as_torch_device
from .layers import V41MarlinLinear, V41NativeMXLinear, V41PackedLinear, fp8_roundtrip


_SOURCE_LINEAR_TYPES = (nn.Linear, V41MarlinLinear, V41NativeMXLinear, V41PackedLinear)


def _quantizable_path(path):
    return path.startswith(("layers.", "vision.", "aligner.")) and ".engram." not in path


class V41GPTQLinear(nn.Module):
    def __init__(self, in_features, out_features, groups, rounded_input, bias, spec):
        super().__init__()
        self.rounded_input = rounded_input
        self.projections = nn.ModuleList(
            TorchLinear(
                bits=spec["bits"], group_size=spec["group_size"], sym=spec["sym"], desc_act=spec["desc_act"],
                pack_dtype=getattr(torch, spec["pack_dtype"]), in_features=in_features,
                out_features=out_features // groups, bias=bias, format=FORMAT.GPTQ_V2, compile=False,
            )
            for _ in range(groups)
        )

    def post_init(self):
        for module in self.projections:
            module.qzero_format(format=2)
            module.post_init()

    def forward(self, x):
        values = fp8_roundtrip(x) if self.rounded_input else x.to(torch.bfloat16)
        if len(self.projections) == 1:
            return self.projections[0](values).to(x.dtype)
        return torch.stack([
            module(group) for module, group in zip(self.projections, values.unbind(-2), strict=True)
        ], dim=-2).to(x.dtype)


def make_gptq_projection(source, spec):
    if not isinstance(spec, dict) or set(spec) != {"bits", "group_size", "sym", "desc_act", "pack_dtype"}:
        raise ValueError("Invalid DeepSeek V4.1 GPTQ projection specification")
    if spec["pack_dtype"] not in ("int8", "int16", "int32"):
        raise ValueError("Invalid DeepSeek V4.1 GPTQ pack dtype")
    groups = getattr(source, "groups", 1)
    return V41GPTQLinear(
        source.in_features, source.out_features, groups, isinstance(source, V41PackedLinear),
        getattr(source, "bias", None) is not None, spec,
    )


def _source_weight(model, path, device):
    name = f"{path}.weight"
    scales = f"{path}.scale"
    with safe_open(model.source_shards[model.source_weight_map[name]], framework="pt", device="cpu") as handler:
        weight = handler.get_tensor(name).to(device)
    if scales not in model.source_weight_map:
        return weight
    with safe_open(model.source_shards[model.source_weight_map[scales]], framework="pt", device="cpu") as handler:
        scale = handler.get_tensor(scales).to(device)
    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float8_e8m0fnu)
    if weight.dtype == torch.int8 or is_fp4_packed_dtype(weight.dtype):
        return dequantize_deepseek_v4_fp4_expert(weight.view(torch.int8), scale)
    return dequantize_fp8(weight, scale=scale)


def _quantize_projection(model, path, module, inputs, config):
    resident = module.qweight if isinstance(module, V41MarlinLinear) else module.weight
    destination = resident.device
    device = destination if config.device is None else as_torch_device(config.device)
    source = _source_weight(model, path, device)
    groups = getattr(module, "groups", 1)
    rounded_input = not isinstance(module, nn.Linear)
    bias = getattr(module, "bias", None)
    n, k = source.shape
    spec = {
        "bits": int(config.bits), "group_size": config.group_size, "sym": config.sym, "desc_act": config.desc_act,
        "pack_dtype": str(config.pack_dtype).removeprefix("torch."),
    }
    replacement = V41GPTQLinear(k, n, groups, rounded_input, bias is not None, spec)
    reports = []
    for index, (kernel, weight) in enumerate(zip(replacement.projections, source.chunk(groups), strict=True)):
        dense = nn.Linear(k, weight.size(0), bias=bias is not None, device="meta", dtype=weight.dtype)
        dense.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            dense.bias = nn.Parameter(bias.detach().to(device), requires_grad=False)
        task = GPTQ(dense, qcfg=config)
        try:
            task.quantizer.configure(perchannel=True)
            for value in inputs:
                group = (value if groups == 1 else value[..., index, :]).to(device)
                task.add_batch(fp8_roundtrip(group) if rounded_input else group, None)
            quantized, scales, zeros, group_index, seconds, loss, damp, samples = task.quantize()
        finally:
            task.free()
        if not samples or not math.isfinite(loss) or not torch.isfinite(scales).all():
            raise ValueError(f"Invalid GPTQ result for DeepSeek V4.1 projection {path}")
        dense.weight = nn.Parameter(quantized.cpu(), requires_grad=False)
        if dense.bias is not None:
            dense.bias = nn.Parameter(dense.bias.cpu(), requires_grad=False)
        hf_pack_layer(
            kernel, dense, scales.cpu(), zeros.cpu(), group_index.cpu(),
            pack_impl=config.pack_impl, device=device, checkpoint_format=FORMAT.GPTQ_V2,
        )
        reports.append({"module": path, "group": index, "samples": samples, "loss": loss, "damp": damp, "duration": seconds})
    replacement.to(destination)
    replacement.post_init()
    return replacement, spec, reports


@torch.no_grad()
def quantize_projections(wrapper, calibration, projections):
    config = wrapper.quantize_config
    if config is None or config.method != METHOD.GPTQ or config.format not in (FORMAT.GPTQ, FORMAT.GPTQ_V2):
        raise ValueError("DeepSeek V4.1 projection quantization requires a GPTQ QuantizeConfig")
    if wrapper.quantized:
        raise ValueError("DeepSeek V4.1 GPTQ requires a published source checkpoint")
    if not isinstance(calibration, CalibrationConfig) or calibration.dataset is None:
        raise TypeError("DeepSeek V4.1 GPTQ requires CalibrationConfig with a dataset")
    if calibration.exclude_attention_sinks or calibration.adapter_dataset is not None:
        raise ValueError("DeepSeek V4.1 projection calibration supports the supplied token rows directly")
    if (
        not isinstance(projections, (list, tuple)) or not projections
        or any(not isinstance(path, str) or not path for path in projections)
        or len(set(projections)) != len(projections)
    ):
        raise ValueError("Specify unique DeepSeek V4.1 backbone or vision projection paths")

    model = wrapper.model
    selected = {}
    for path in projections:
        if not _quantizable_path(path):
            raise ValueError(f"DeepSeek V4.1 preserves the published weights for {path}")
        module = model.get_submodule(path)
        if not isinstance(module, _SOURCE_LINEAR_TYPES) or path + ".weight" not in model.source_weight_map:
            raise ValueError(f"DeepSeek V4.1 projection GPTQ does not support {path}")
        resolved = clone_gptq_config_for_module(config, path)
        if resolved is None:
            continue
        if resolved.gptaq is not None or resolved.foem is not None or resolved.adapter:
            raise ValueError("DeepSeek V4.1 projection quantization uses GPTQ")
        selected[path] = (module, resolved)
    if not selected:
        raise ValueError("No DeepSeek V4.1 projections matched the GPTQ configuration")

    batches = prepare_calibration_dataset_once(
        calibration.bind(wrapper.prepare_dataset, batch_size=wrapper._calibration_batch_size(calibration))
    )
    activations = defaultdict(list)

    def hook(path):
        def capture(_module, args):
            activations[path].append(args[0].detach().to(device="cpu", copy=True))
        return capture

    handles = []
    try:
        for path, (module, _) in selected.items():
            handles.append(module.register_forward_pre_hook(hook(path)))
        for batch in batches:
            model(**batch, use_cache=False, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()

    replacements, specs, reports = {}, {}, []
    for path, (module, resolved) in selected.items():
        inputs = activations.pop(path, None)
        if not inputs:
            raise ValueError(f"Calibrate DeepSeek V4.1 projection {path} with inputs that exercise it")
        replacement, spec, report = _quantize_projection(model, path, module, inputs, resolved)
        replacements[path] = replacement
        specs[path] = spec
        reports.extend(report)
    for path, replacement in replacements.items():
        model.set_submodule(path, replacement)
    model.config.gptqmodel_v41_projections = specs
    wrapper.quantized = True
    return {"gptq": reports}
