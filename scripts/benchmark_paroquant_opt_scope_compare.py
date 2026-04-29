#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_REPO = Path("/root/official_paroquant")
if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))

from paroquant.optim.qlinear import PseudoQuantizedLinear
from paroquant.optim.rotation import transform_to_kernel_data
from paroquant.optim.train import get_random_rotation_pairs, optimize_module as official_optimize_module
from paroquant.optim.util import catch_first_layer_input, get_blocks, get_calib_dataset, get_named_linears, move_embed, set_module_by_name

from gptqmodel.looper.input_cache import InputCache
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
from gptqmodel.quantization.paroquant.optimization import optimize_paroquant_linear


DEFAULT_MODEL = "/monster/data/model/Llama-3.2-1B-Instruct"
DEFAULT_MODULE_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


@dataclass
class BenchRow:
    case: str
    total_s: float
    val_smoothl1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ParoQuant module/compute_block/layer modes against official whole-layer optimization."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--official-repo", default=str(OFFICIAL_REPO))
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16"))
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--train-batches", type=int, default=2, help="Calibration batches captured for train.")
    parser.add_argument("--val-batches", type=int, default=1, help="Calibration batches captured for validation.")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--train-rows", type=int, default=2048)
    parser.add_argument("--val-rows", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--krot", type=int, default=8)
    parser.add_argument("--pair-ratio", type=float, default=0.5)
    parser.add_argument("--rotation-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=1)
    parser.add_argument("--rotation-lr", type=float, default=0.05)
    parser.add_argument("--weight-lr", type=float, default=1e-5)
    parser.add_argument("--quantizer-lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        choices=("local_module", "local_compute_block", "local_layer", "official_layer"),
        default=None,
        help="Optional repeated case filter. By default all cases run.",
    )
    parser.add_argument("--skip-official", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _dtype_from_label(label: str) -> torch.dtype:
    return torch.bfloat16 if str(label).strip().lower() == "bf16" else torch.float16


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _module_seed(base_seed: int, layer_index: int, full_name: str) -> int:
    leaf = full_name.rsplit(".", 1)[-1]
    seed_material = f"{base_seed}:{layer_index}:{leaf}".encode("utf-8")
    digest = hashlib.blake2b(seed_material, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _init_rotation_data(weight: torch.Tensor, *, seed: int, group_size: int, num_rotations: int, device: torch.device):
    grouped = weight.view(weight.shape[0], -1, group_size).permute(1, 0, 2)
    all_pairs = get_random_rotation_pairs(
        grouped,
        group_size=group_size,
        num_rotations=num_rotations,
        num_pairs_factor=0.5,
        seed=seed,
    )
    pair_tensors = [torch.tensor(pairs, device="cpu", dtype=torch.int32) for pairs in all_pairs]
    initial_angles = [torch.zeros(pairs.shape[0], device="cpu") for pairs in pair_tensors]
    npairs, angles, mask = transform_to_kernel_data(pair_tensors, initial_angles, group_size=group_size)
    return [npairs.to(device), angles.to(device), mask.to(device)]


def _capture_module_inputs(
    layer: torch.nn.Module,
    *,
    input_batches: list[torch.Tensor],
    kwargs: dict[str, Any],
    module_names: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    captured = {name: [] for name in module_names}
    handles = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu()
            captured[name].append(x)

        return hook

    for module_name in module_names:
        target = dict(layer.named_modules())[module_name]
        handles.append(target.register_forward_pre_hook(make_hook(module_name)))

    layer = layer.to(device)
    try:
        with torch.no_grad():
            for batch in input_batches:
                _ = layer(batch.to(device=device, dtype=dtype), **kwargs)
    finally:
        for handle in handles:
            handle.remove()
        layer.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {name: torch.cat(parts, dim=0) for name, parts in captured.items()}


def _build_processor(
    *,
    opt_scope: str,
    kwargs: dict[str, Any],
    all_input_batches: list[torch.Tensor],
    bits: int,
    group_size: int,
    krot: int,
    pair_ratio: float,
    train_rows: int,
    val_rows: int,
    batch_size: int,
    rotation_epochs: int,
    finetune_epochs: int,
    rotation_lr: float,
    weight_lr: float,
    quantizer_lr: float,
    seed: int,
) -> ParoQuantProcessor:
    processor = object.__new__(ParoQuantProcessor)

    def dynamic_get(_module_name=None, _key=None, default=None, **_kwargs):
        return default

    sanitized_kwargs = {k: v for k, v in kwargs.items() if k not in ("attention_mask", "position_ids", "use_cache")}
    processor.qcfg = SimpleNamespace(
        opt_scope=opt_scope,
        runtime_bits=bits,
        group_size=group_size,
        sym=True,
        krot=krot,
        opt_seed=seed,
        opt_pair_ratio=pair_ratio,
        opt_pair_impl="fast",
        opt_quantizer_impl="reference",
        opt_fused_rotation=True,
        opt_stage_cudagraph=True,
        opt_stage_impl="fast",
        opt_channel_scale_clamp_min=1e-2,
        opt_channel_scale_clamp_max=1e2,
        opt_train_samples=train_rows,
        opt_validation_samples=val_rows,
        opt_batch_size=batch_size,
        opt_rotation_lr=rotation_lr,
        opt_weight_lr=weight_lr,
        opt_quantizer_lr=quantizer_lr,
        opt_rotation_epochs=rotation_epochs,
        opt_finetune_epochs=finetune_epochs,
        dynamic_get=dynamic_get,
    )
    processor.gptq_model = SimpleNamespace(support_batch_quantize=True, rotary_embedding=None)
    processor.model = None
    processor._batch_tls = __import__("threading").local()
    processor.lock = __import__("threading").Lock()
    processor.tasks = {}
    processor.calculate_w_wq_diff = False
    processor.fallback = True
    processor._rotary_cache = {}
    processor._rotary_source_id = None
    processor._rotary_lock = __import__("threading").Lock()
    processor.inputs_cache = InputCache(
        layer_inputs=[[batch] for batch in all_input_batches],
        layer_input_kwargs=[sanitized_kwargs for _ in all_input_batches],
        position_ids=[kwargs.get("position_ids")] * len(all_input_batches),
        attention_masks=[kwargs.get("attention_mask")] * len(all_input_batches),
    )
    return processor


def _evaluate_group_loss(
    processor: ParoQuantProcessor,
    layer: torch.nn.Module,
    *,
    input_batches: list[torch.Tensor],
    output_batches: list[torch.Tensor],
    kwargs: dict[str, Any],
) -> float:
    sanitized_kwargs = [{k: v for k, v in kwargs.items() if k not in ("attention_mask", "position_ids", "use_cache")} for _ in input_batches]
    positions = [kwargs.get("position_ids")] * len(input_batches)
    masks = [kwargs.get("attention_mask")] * len(input_batches)
    return processor._evaluate_group_layer(
        layer,
        input_batches=[[batch] for batch in input_batches],
        input_kwargs_batches=sanitized_kwargs,
        target_batches=[[batch] for batch in output_batches],
        position_ids=positions,
        attention_masks=masks,
        use_amp=torch.cuda.is_available(),
    )


def _benchmark_local_module(
    *,
    base_layer: torch.nn.Module,
    layer_index: int,
    train_input_batches: list[torch.Tensor],
    val_input_batches: list[torch.Tensor],
    train_output_batches: list[torch.Tensor],
    val_output_batches: list[torch.Tensor],
    kwargs: dict[str, Any],
    module_names: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> BenchRow:
    layer = copy.deepcopy(base_layer).to(dtype=dtype)
    all_inputs = train_input_batches + val_input_batches
    all_outputs = train_output_batches + val_output_batches
    processor = _build_processor(
        opt_scope="module",
        kwargs=kwargs,
        all_input_batches=all_inputs,
        bits=args.bits,
        group_size=args.group_size,
        krot=args.krot,
        pair_ratio=args.pair_ratio,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        batch_size=args.batch_size,
        rotation_epochs=args.rotation_epochs,
        finetune_epochs=args.finetune_epochs,
        rotation_lr=args.rotation_lr,
        weight_lr=args.weight_lr,
        quantizer_lr=args.quantizer_lr,
        seed=args.seed,
    )
    inputs = _capture_module_inputs(
        layer,
        input_batches=all_inputs,
        kwargs=kwargs,
        module_names=module_names,
        device=device,
        dtype=dtype,
    )

    _sync(device)
    start = time.perf_counter()
    for module_name in module_names:
        module = dict(layer.named_modules())[module_name]
        result = optimize_paroquant_linear(
            weight=module.weight.data,
            bias=module.bias.data if module.bias is not None else None,
            inputs=inputs[module_name],
            bits=args.bits,
            group_size=args.group_size,
            sym=True,
            krot=args.krot,
            pair_ratio=args.pair_ratio,
            train_rows=args.train_rows,
            val_rows=args.val_rows,
            batch_size=args.batch_size,
            rotation_epochs=args.rotation_epochs,
            finetune_epochs=args.finetune_epochs,
            rotation_lr=args.rotation_lr,
            weight_lr=args.weight_lr,
            quantizer_lr=args.quantizer_lr,
            seed=_module_seed(args.seed, layer_index, f"model.layers.{layer_index}.{module_name}"),
            fused_rotation=True,
            stage_cudagraph=True,
            stage_impl="fast",
            pair_impl="fast",
            quantizer_impl="reference",
            scale_clamp_min=1e-2,
            scale_clamp_max=1e2,
        )
        module.weight.data = result.pseudo_weight.to(device=module.weight.device, dtype=module.weight.dtype)
    _sync(device)
    total_s = time.perf_counter() - start

    val_loss = _evaluate_group_loss(
        processor,
        layer.to(device),
        input_batches=val_input_batches,
        output_batches=val_output_batches,
        kwargs=kwargs,
    )
    return BenchRow(case="local_module", total_s=total_s, val_smoothl1=val_loss)


def _benchmark_local_group(
    *,
    opt_scope: str,
    base_layer: torch.nn.Module,
    layer_index: int,
    train_input_batches: list[torch.Tensor],
    val_input_batches: list[torch.Tensor],
    train_output_batches: list[torch.Tensor],
    val_output_batches: list[torch.Tensor],
    kwargs: dict[str, Any],
    module_names: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> BenchRow:
    del layer_index
    layer = copy.deepcopy(base_layer).to(dtype=dtype)
    all_inputs = train_input_batches + val_input_batches
    all_outputs = train_output_batches + val_output_batches
    processor = _build_processor(
        opt_scope=opt_scope,
        kwargs=kwargs,
        all_input_batches=all_inputs,
        bits=args.bits,
        group_size=args.group_size,
        krot=args.krot,
        pair_ratio=args.pair_ratio,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        batch_size=args.batch_size,
        rotation_epochs=args.rotation_epochs,
        finetune_epochs=args.finetune_epochs,
        rotation_lr=args.rotation_lr,
        weight_lr=args.weight_lr,
        quantizer_lr=args.quantizer_lr,
        seed=args.seed,
    )
    state = SimpleNamespace(
        layer_module=layer,
        pristine_layer_module=copy.deepcopy(layer).cpu(),
        layer_inputs=[[batch] for batch in all_inputs],
        layer_input_kwargs=[{k: v for k, v in kwargs.items() if k not in ("attention_mask", "position_ids", "use_cache")} for _ in all_inputs],
        layer_outputs=[[batch] for batch in all_outputs],
        modules={
            name: NamedModule(dict(layer.named_modules())[name], name, f"model.layers.{args.layer_idx}.{name}", args.layer_idx)
            for name in module_names
        },
    )

    _sync(device)
    start = time.perf_counter()
    groups = processor._optimization_groups_for_layer(state)
    for _label, group_modules in groups:
        results, _ = processor._optimize_group(state, group_modules)
        for named_module in group_modules:
            original_weight = named_module.weight.data.detach().clone()
            processor._apply_optimization_result(named_module, results[named_module.name], original_weight)
    _sync(device)
    total_s = time.perf_counter() - start

    val_loss = _evaluate_group_loss(
        processor,
        layer.to(device),
        input_batches=val_input_batches,
        output_batches=val_output_batches,
        kwargs=kwargs,
    )
    return BenchRow(case=f"local_{opt_scope}", total_s=total_s, val_smoothl1=val_loss)


def _benchmark_official_layer(
    *,
    base_layer: torch.nn.Module,
    train_input_batches: list[torch.Tensor],
    val_input_batches: list[torch.Tensor],
    train_output_batches: list[torch.Tensor],
    val_output_batches: list[torch.Tensor],
    kwargs: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> BenchRow:
    layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.float32)
    named_pseudo_modules = {}
    for name, old_module in get_named_linears(layer).items():
        weight = old_module.weight.float()
        rotation_pairs = _init_rotation_data(
            weight,
            seed=args.seed,
            group_size=args.group_size,
            num_rotations=args.krot,
            device=device,
        )
        channel_scales = torch.ones(1, weight.shape[1], dtype=weight.dtype, device=device)
        new_module = PseudoQuantizedLinear(
            old_module,
            rotation_pairs,
            channel_scales,
            group_size=args.group_size,
            n_bits=args.bits,
            num_rotations=args.krot,
        )
        set_module_by_name(layer, name, new_module)
        named_pseudo_modules[name] = new_module

    for param in layer.parameters():
        param.requires_grad = False

    _sync(device)
    start = time.perf_counter()
    for step_params, epochs in (
        ({"channel_scales": args.rotation_lr, "angles": args.rotation_lr}, args.rotation_epochs),
        ({"weight": args.weight_lr, "quantizer": args.quantizer_lr}, args.finetune_epochs),
    ):
        optim_params = []
        for new_module in named_pseudo_modules.values():
            new_module.set_optim_enabled(**{name: True for name in step_params})
            for param_name, lr in step_params.items():
                optim_params.append(
                    dict(
                        params=new_module.get_optim_params(param_name),
                        lr=lr,
                        weight_decay=0.01,
                        betas=(0.9, 0.95),
                        eps=1e-10,
                    )
                )

        official_optimize_module(
            layer,
            ([batch.to(device=device, dtype=dtype) for batch in train_input_batches], [batch.to(device=device, dtype=dtype) for batch in train_output_batches]),
            ([batch.to(device=device, dtype=dtype) for batch in val_input_batches], [batch.to(device=device, dtype=dtype) for batch in val_output_batches]),
            {k: (v.to(device=device) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()},
            optim_params,
            loss_fn="smooth_l1",
            n_iter=epochs,
            gradient_accumulation_steps=1,
            early_stop=None,
            post_optim_callback=lambda _module: [pseudo_module.reset_angles_by_mask() for pseudo_module in named_pseudo_modules.values()],
        )
    _sync(device)
    total_s = time.perf_counter() - start

    layer = layer.to(device=device, dtype=dtype)
    with torch.no_grad():
        total = 0.0
        for inp, target in zip(val_input_batches, val_output_batches):
            preds = layer(inp.to(device=device, dtype=dtype), **{k: (v.to(device=device) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()})
            if isinstance(preds, tuple):
                preds = preds[0]
            total += float(F.smooth_l1_loss(preds, target.to(device=device, dtype=preds.dtype)).item())
    return BenchRow(case="official_layer", total_s=total_s, val_smoothl1=total / max(1, len(val_input_batches)))


def _load_first_layer_io(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    move_embed(model, device)
    blocks = get_blocks(model)
    blocks[args.layer_idx].to(device)

    train_samples = torch.stack(
        get_calib_dataset("pileval", tokenizer=tokenizer, n_samples=args.train_batches, block_size=args.block_size, seed=args.seed, split="train"),
        dim=0,
    ).to(device)
    val_samples = torch.stack(
        get_calib_dataset("pileval", tokenizer=tokenizer, n_samples=args.val_batches, block_size=args.block_size, seed=args.seed + 1, split="validation"),
        dim=0,
    ).to(device)

    train_input_batches, kwargs = catch_first_layer_input(model, blocks, train_samples, batch_size=1)
    val_input_batches, _ = catch_first_layer_input(model, blocks, val_samples, batch_size=1)
    layer = blocks[args.layer_idx].to(device)

    with torch.no_grad():
        train_output_batches = []
        for batch in train_input_batches:
            out = layer(batch.to(device=device, dtype=dtype), **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            train_output_batches.append(out.detach().cpu())

        val_output_batches = []
        for batch in val_input_batches:
            out = layer(batch.to(device=device, dtype=dtype), **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            val_output_batches.append(out.detach().cpu())

    base_layer = copy.deepcopy(layer).cpu()
    kwargs = {
        key: (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
        for key, value in kwargs.items()
        if key not in ("past_key_value", "past_key_values")
    }
    kwargs["use_cache"] = False

    return {
        "base_layer": base_layer,
        "train_input_batches": [batch.detach().cpu() for batch in train_input_batches],
        "val_input_batches": [batch.detach().cpu() for batch in val_input_batches],
        "train_output_batches": train_output_batches,
        "val_output_batches": val_output_batches,
        "kwargs": kwargs,
    }


def main() -> int:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_label(args.dtype)
    module_names = DEFAULT_MODULE_NAMES

    print(f"[setup] loading layer {args.layer_idx} IO from {args.model}", flush=True)
    captured = _load_first_layer_io(args, device, dtype)

    rows: list[BenchRow] = []
    requested_cases = set(args.cases or [])
    cases = [
        ("local_module", lambda: _benchmark_local_module(module_names=module_names, device=device, dtype=dtype, args=args, layer_index=args.layer_idx, **captured)),
        ("local_compute_block", lambda: _benchmark_local_group(opt_scope="compute_block", module_names=module_names, device=device, dtype=dtype, args=args, layer_index=args.layer_idx, **captured)),
        ("local_layer", lambda: _benchmark_local_group(opt_scope="layer", module_names=module_names, device=device, dtype=dtype, args=args, layer_index=args.layer_idx, **captured)),
    ]
    if not args.skip_official:
        cases.append(("official_layer", lambda: _benchmark_official_layer(device=device, dtype=dtype, args=args, **captured)))

    for label, fn in cases:
        if requested_cases and label not in requested_cases:
            continue
        print(f"[run] {label}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        row = fn()
        rows.append(row)
        print(f"[done] {label} total_s={row.total_s:.3f} val_smoothl1={row.val_smoothl1:.6e}", flush=True)

    table_rows = [[row.case, f"{row.total_s:.3f}", f"{row.val_smoothl1:.6e}"] for row in rows]
    print(
        tabulate(
            table_rows,
            headers=["case", "total_s", "val_smoothl1"],
            tablefmt="grid",
        )
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps([asdict(row) for row in rows], indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
