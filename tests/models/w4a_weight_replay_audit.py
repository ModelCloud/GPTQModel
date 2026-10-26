# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Construct the weight-adaptation forward from a saved native INT4 checkpoint."""

import argparse
import gc
import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


@contextmanager
def weight_replay_model(checkpoint: Path, *, hardware_forward: bool = False):
    """Audit the actual training forward, with no optimizer or weight updates."""
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear import w4a_llama_replay as replay
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear

    from .w4a_nvfp4_norm_qat import _ROTATION_FIELDS, _activation_replay_config, _dequantize_for_replay
    from .w4a_nvfp4_weight_qad import _install_trainable_gptq_codes, straight_through_activation_round
    from .w4a_quality_regression import prepare_reference

    qcfg = _activation_replay_config(checkpoint)
    if qcfg.activation_mode != "w4a_nvfp4" or not getattr(qcfg, "rotation", None):
        raise ValueError("Weight replay audit requires the fused-norm NVFP4 stream")
    with TemporaryDirectory(prefix="w4a-weight-replay-") as directory:
        reference = prepare_reference(checkpoint, Path(directory) / "w4a16")
        wrapper = GPTQModel.load(str(reference), backend=BACKEND.GPTQ_TORCH,
                                device="cuda:0", dtype=torch.bfloat16)
    core = wrapper.model
    quantized = {name: module for name, module in core.named_modules() if isinstance(module, TorchLinear)}
    if len(quantized) != 112 or len(core.model.layers) != 16:
        raise ValueError("Weight replay audit currently requires the fully quantized 1B Llama")
    codes = {name: W4AFP8Linear._centered_int4_codes(module).detach().cpu().to(torch.int8)
             for name, module in quantized.items()}
    scales = {name: module.scales.detach().cpu().float().clone() for name, module in quantized.items()}
    rotation = {name: {field: getattr(module, field, None) for field in _ROTATION_FIELDS}
                for name, module in quantized.items()}
    core = _dequantize_for_replay(core, device="cuda:0", dtype=torch.bfloat16)
    modules = dict(core.named_modules())
    for name, fields in rotation.items():
        for field, value in fields.items():
            setattr(modules[name], field, value)
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    _, weight_modules = _install_trainable_gptq_codes(core, codes, scales)
    del codes, scales, quantized, modules, wrapper
    replay.install_w4a_llama_replay(core, qcfg)
    core.config.use_cache = False
    core.eval()
    original_round = replay._round
    replay._round = straight_through_activation_round
    hardware = None
    try:
        if hardware_forward:
            from .w4a_hardware_forward import HardwareForward

            runtime = GPTQModel.load(str(checkpoint), backend=BACKEND.GPTQ_W4A_NVFP4,
                                    device="cuda:0", dtype=torch.bfloat16).model.eval()
            hardware = HardwareForward(core, runtime, weight_modules)
            if hardware.sync_weights() != 0:
                raise AssertionError("Initial hardware codes differ from the saved checkpoint")
            core._w4a_training_hardware = hardware
        yield core
    finally:
        if hardware is not None:
            hardware.close()
            del core._w4a_training_hardware
        replay._round = original_round


def trace_first_layer(checkpoint: Path, calibration: Path, output: Path, sequence_length: int = 256):
    """Locate the first differing operand using the same held-out input IDs."""
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.nn_modules.qlinear.w4a_activation import W4AActivation

    from .w4a_calibration_data import file_digest
    from .w4a_gb10_memory import require_w4a_test_headroom
    from .w4a_nvfp4_norm_qat import _calibration_ids

    require_w4a_test_headroom(require_scope=True)
    if output.exists():
        raise FileExistsError(output)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    ids = _calibration_ids(tokenizer, calibration, 1, sequence_length, partition="selection")[0]
    manifest_hash = file_digest(calibration / "manifest.json")
    names = ["model.embed_tokens", "model.layers.0.input_layernorm",
             "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj",
             "model.layers.0.self_attn.v_proj", "model.layers.0.self_attn.o_proj",
             "model.layers.0.post_attention_layernorm", "model.layers.0.mlp.gate_proj",
             "model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj", "model.layers.0"]

    def capture(core):
        tensors, handles = {}, []
        frequency = core.model.rotary_emb.inv_freq
        tensors["rotary_inv_freq"] = frequency.detach().float().cpu().clone()
        tensors["rotary_frequency_dtype_bits"] = torch.tensor(32 if frequency.dtype == torch.float32 else 16)
        def record(key, value):
            value = value[0] if isinstance(value, tuple) else value
            if isinstance(value, W4AActivation):
                value = value.decode(torch.float32)
            tensors[key] = value.detach().float().cpu().contiguous().clone()
        try:
            attention = core.model.layers[0].self_attn
            quantizer = getattr(attention, "_w4a_output_quantizer", None)
            def raw_attention(_module, args):
                from gptqmodel.quantization.activation_floatx import nvfp4_global_scale
                value = args[0]
                record("attention_before_pack", value)
                tensors["attention_source_dtype_bits"] = torch.tensor(
                    16 if value.dtype == torch.bfloat16 else 32 if value.dtype == torch.float32 else -1)
                tensors["attention_dynamic_global_scale"] = nvfp4_global_scale(
                    value.detach().abs().amax(), grid_dtype=value.dtype, recipe="least_squares").cpu()
            handles.append((quantizer if quantizer is not None else attention.o_proj)
                           .register_forward_pre_hook(raw_attention, prepend=True))
            for name in names:
                module = core.get_submodule(name)
                def before(_module, args, kwargs, *, name=name):
                    record(name + ".input", args[0] if args else kwargs["hidden_states"])
                def after(_module, _args, result, *, name=name):
                    record(name + ".output", result)
                handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
                handles.append(module.register_forward_hook(after))
            with torch.inference_mode():
                core(input_ids=ids[None].cuda(), use_cache=False, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        if len(tensors) != 2 * len(names) + 5:
            raise AssertionError("Incomplete first-layer trace")
        return tensors

    core = GPTQModel.load(str(checkpoint), backend=BACKEND.GPTQ_W4A_NVFP4,
                         device="cuda:0", dtype=torch.bfloat16).model.eval()
    runtime = capture(core)
    del core
    gc.collect()
    torch.cuda.empty_cache()
    with weight_replay_model(checkpoint) as core:
        replay = capture(core)
    if file_digest(calibration / "manifest.json") != manifest_hash:
        raise ValueError("Calibration manifest changed during replay audit")
    output.mkdir(parents=True, exist_ok=False)
    save_file(runtime, str(output / "runtime.safetensors"))
    save_file(replay, str(output / "replay.safetensors"))
    rows = {}
    keys = [name + "." + side for name in names for side in ("input", "output")]
    keys += ["attention_before_pack", "attention_source_dtype_bits", "attention_dynamic_global_scale"]
    keys += ["rotary_inv_freq", "rotary_frequency_dtype_bits"]
    for key in keys:
        a, b = runtime[key].double(), replay[key].double()
        if a.shape != b.shape or not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise AssertionError(f"Invalid boundary tensors: {key}")
        delta = (a - b).abs()
        rows[key] = {"shape": list(a.shape), "max_abs_error": float(delta.max()),
                     "relative_l2": float(delta.norm() / a.norm().clamp_min(1e-30)),
                     "outside_tolerance": int((delta > .002 + .002 * a.abs()).sum())}
    report = {"checkpoint": str(checkpoint), "data_manifest_sha256": manifest_hash,
              "partition": "selection", "tokens": len(ids), "boundaries": rows}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    args = parser.parse_args()
    trace_first_layer(args.checkpoint, args.calibration, args.output, args.sequence_length)
