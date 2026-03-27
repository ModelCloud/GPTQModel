#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from tabulate import tabulate
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel.quantization.paroquant import optimization as local_opt
from gptqmodel.utils.paroquant import apply_paroquant_rotation, build_identity_rotation_buffers
from gptqmodel.utils.paroquant_benchmark import _normalize_model_dtype, load_nm_calibration


DEFAULT_MODEL = "/monster/data/model/Llama-3.2-1B-Instruct"
DEFAULT_OFFICIAL_REPO = "/root/official_paroquant"
DEFAULT_ASSET_DIR = "benchmark_assets"
DEFAULT_CAPTURE_ROWS = 4096
DEFAULT_CALIBRATION_SAMPLES = 512
DEFAULT_MODULES = (
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
)


@dataclass(frozen=True)
class LocalCase:
    label: str
    stage_impl: str
    pair_impl: str
    quantizer_impl: str


@dataclass
class BenchResult:
    module: str
    impl: str
    pair_s: float
    setup_s: float
    stage1_s: float
    stage2_s: float
    export_s: float
    opt_s: float
    total_s: float
    train_loss: float
    val_loss: float
    repeat: int


LOCAL_CASES: tuple[LocalCase, ...] = (
    LocalCase("local_rrr", "reference", "reference", "reference"),
    LocalCase("local_rfr", "reference", "fast", "reference"),
    LocalCase("local_ffr", "fast", "fast", "reference"),
    LocalCase("local_fff", "fast", "fast", "fast"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local ParoQuant implementations against the official PR reference on saved activations."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--official-repo", default=DEFAULT_OFFICIAL_REPO)
    parser.add_argument("--asset-dir", default=DEFAULT_ASSET_DIR)
    parser.add_argument("--asset-name", default="llama32_1b_layer0_mlp_gate_up_down")
    parser.add_argument("--module", dest="modules", action="append", default=None)
    parser.add_argument("--model-dtype", default="fp16")
    parser.add_argument("--calibration-samples", type=int, default=DEFAULT_CALIBRATION_SAMPLES)
    parser.add_argument("--capture-rows", type=int, default=DEFAULT_CAPTURE_ROWS)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--krot", type=int, default=8)
    parser.add_argument("--pair-ratio", type=float, default=0.5)
    parser.add_argument("--train-rows", type=int, default=1024)
    parser.add_argument("--val-rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rotation-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--rotation-lr", type=float, default=0.05)
    parser.add_argument("--weight-lr", type=float, default=1e-5)
    parser.add_argument("--quantizer-lr", type=float, default=1e-6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-recapture", action="store_true")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--module-filter", default=None, help="Substring filter applied after loading the asset.")
    return parser.parse_args()


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_timer(device: torch.device):
    class _Timer:
        def __enter__(self):
            sync_cuda(device)
            self.start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            sync_cuda(device)
            self.elapsed = time.perf_counter() - self.start
            return False

    return _Timer()


def _get_named_module(model, module_name: str):
    module_map = dict(model.named_modules())
    if module_name not in module_map:
        raise KeyError(f"Module `{module_name}` not found.")
    return module_map[module_name]


def _tokenize_calibration_sample(tokenizer, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "input_ids" in sample:
        input_ids = torch.as_tensor(sample["input_ids"], dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        attention_mask = sample.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    if "messages" in sample:
        rendered = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(rendered, add_special_tokens=True, return_tensors="pt")
        return {
            "input_ids": tokenized["input_ids"].to(dtype=torch.long),
            "attention_mask": tokenized.get("attention_mask", torch.ones_like(tokenized["input_ids"])).to(dtype=torch.long),
        }

    if "text" in sample:
        tokenized = tokenizer(sample["text"], add_special_tokens=True, return_tensors="pt")
        return {
            "input_ids": tokenized["input_ids"].to(dtype=torch.long),
            "attention_mask": tokenized.get("attention_mask", torch.ones_like(tokenized["input_ids"])).to(dtype=torch.long),
        }

    raise ValueError(f"Unsupported calibration sample keys: {sorted(sample.keys())}")


def _capture_module_inputs(
    model,
    tokenizer,
    module_names: list[str],
    calibration_dataset: list[dict[str, Any]],
    *,
    max_rows: int,
) -> dict[str, torch.Tensor]:
    captured: dict[str, list[torch.Tensor]] = {name: [] for name in module_names}
    captured_rows = {name: 0 for name in module_names}
    hooks = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if not inputs or captured_rows[name] >= max_rows:
                return
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu()
            remaining = max_rows - captured_rows[name]
            if remaining <= 0:
                return
            piece = x[:remaining].contiguous()
            if piece.numel() == 0:
                return
            captured[name].append(piece)
            captured_rows[name] += piece.shape[0]

        return hook

    for name in module_names:
        module = _get_named_module(model, name)
        hooks.append(module.register_forward_pre_hook(make_hook(name)))

    model_device = next(model.parameters()).device
    try:
        for sample in calibration_dataset:
            if all(count >= max_rows for count in captured_rows.values()):
                break
            tokenized = _tokenize_calibration_sample(tokenizer, sample)
            with torch.inference_mode():
                model(
                    input_ids=tokenized["input_ids"].to(device=model_device),
                    attention_mask=tokenized["attention_mask"].to(device=model_device),
                )
    finally:
        for hook in hooks:
            hook.remove()

    flattened: dict[str, torch.Tensor] = {}
    for name in module_names:
        pieces = captured[name]
        if not pieces:
            raise RuntimeError(f"Failed to capture calibration activations for `{name}`.")
        flattened[name] = torch.cat(pieces, dim=0)[:max_rows].contiguous()
    return flattened


def _asset_paths(asset_dir: Path, asset_name: str) -> tuple[Path, Path]:
    return asset_dir / f"{asset_name}.safetensors", asset_dir / f"{asset_name}.json"


def _module_key(module_name: str) -> str:
    return module_name.replace(".", "__")


def _capture_asset(args: argparse.Namespace, asset_path: Path, metadata_path: Path) -> None:
    modules = args.modules or list(DEFAULT_MODULES)
    asset_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=False,
        torch_dtype=_normalize_model_dtype(args.model_dtype),
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    calibration_dataset = load_nm_calibration(args.calibration_samples)
    captured = _capture_module_inputs(
        model,
        tokenizer,
        modules,
        calibration_dataset,
        max_rows=args.capture_rows,
    )

    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {
        "model": args.model,
        "model_dtype": str(args.model_dtype),
        "calibration_samples": int(args.calibration_samples),
        "capture_rows": int(args.capture_rows),
        "modules": [],
    }
    for module_name in modules:
        module = _get_named_module(model, module_name)
        module_id = _module_key(module_name)
        weight = module.weight.detach().to(dtype=torch.float32, device="cpu").contiguous()
        tensors[f"{module_id}.weight"] = weight
        if getattr(module, "bias", None) is not None:
            tensors[f"{module_id}.bias"] = module.bias.detach().to(dtype=torch.float32, device="cpu").contiguous()
        tensors[f"{module_id}.inputs"] = captured[module_name].to(dtype=torch.float32, device="cpu").contiguous()
        metadata["modules"].append(
            {
                "name": module_name,
                "key": module_id,
                "rows": int(captured[module_name].shape[0]),
                "in_features": int(weight.shape[1]),
                "out_features": int(weight.shape[0]),
                "has_bias": getattr(module, "bias", None) is not None,
            }
        )

    save_file(tensors, str(asset_path))
    metadata_path.write_text(json.dumps(metadata, indent=2))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_asset(asset_path: Path, metadata_path: Path, module_filter: str | None) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    metadata = json.loads(metadata_path.read_text())
    tensors = load_file(str(asset_path), device="cpu")
    loaded: dict[str, dict[str, torch.Tensor]] = {}
    for module_info in metadata["modules"]:
        module_name = module_info["name"]
        if module_filter and module_filter not in module_name:
            continue
        module_id = module_info["key"]
        loaded[module_name] = {
            "weight": tensors[f"{module_id}.weight"].contiguous(),
            "inputs": tensors[f"{module_id}.inputs"].contiguous(),
        }
        bias_key = f"{module_id}.bias"
        if bias_key in tensors:
            loaded[module_name]["bias"] = tensors[bias_key].contiguous()
        else:
            loaded[module_name]["bias"] = None
    if not loaded:
        raise ValueError("No modules matched the requested asset filter.")
    return metadata, loaded


def _prepare_rows(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    inputs: torch.Tensor,
    *,
    train_rows: int,
    val_rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    weight_opt = weight.to(device=device, dtype=torch.float32).contiguous()
    bias_opt = None if bias is None else bias.to(device=device, dtype=torch.float32).contiguous()
    rows = local_opt._sample_activation_rows(inputs, max_rows=max(1, int(train_rows) + int(val_rows)))
    rows = rows.to(device=device, dtype=torch.float32).contiguous()
    targets = F.linear(rows, weight_opt, bias_opt)
    train_count = min(rows.shape[0], max(1, int(train_rows)))
    val_count = min(max(1, int(val_rows)), max(1, rows.shape[0] - train_count))
    inputs_train = rows[:train_count].contiguous()
    targets_train = targets[:train_count].contiguous()
    inputs_val = rows[-val_count:].contiguous()
    targets_val = targets[-val_count:].contiguous()
    return weight_opt, bias_opt, inputs_train, targets_train, inputs_val, targets_val


def _batch_list(rows: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    return [rows[start:start + batch_size].contiguous() for start in range(0, rows.shape[0], batch_size)]


def _common_loss(module_or_weight: nn.Module | torch.Tensor, bias: torch.Tensor | None, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    with torch.no_grad():
        if isinstance(module_or_weight, torch.Tensor):
            preds = F.linear(inputs, module_or_weight, bias)
        else:
            preds = module_or_weight(inputs)
        return float(F.smooth_l1_loss(preds, targets).item())


def _warmup_local_kernel(device: torch.device, group_size: int, krot: int) -> None:
    if device.type != "cuda":
        return
    x = torch.randn(32, group_size, device=device, dtype=torch.float32)
    pairs, theta, scales = build_identity_rotation_buffers(
        in_features=group_size,
        group_size=group_size,
        krot=krot,
        device=device,
        dtype=torch.float32,
    )
    apply_paroquant_rotation(x, pairs, theta, scales, group_size)
    sync_cuda(device)


def _import_official(repo_path: Path):
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    from paroquant.kernels.cuda import scaled_pairwise_rotation  # noqa: F401
    from paroquant.optim.qlinear import PseudoQuantizedLinear
    from paroquant.optim.rotation import transform_to_kernel_data
    from paroquant.optim.train import get_random_rotation_pairs, optimize_module

    return PseudoQuantizedLinear, transform_to_kernel_data, get_random_rotation_pairs, optimize_module


def _run_local_case(
    *,
    module_name: str,
    case: LocalCase,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    inputs: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    repeat: int,
) -> BenchResult:
    seed = int(args.seed) + (repeat * 1000)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    with cuda_timer(device) as total_timer:
        with cuda_timer(device) as setup_timer:
            weight_opt, bias_opt, inputs_train, targets_train, inputs_val, targets_val = _prepare_rows(
                weight,
                bias,
                inputs,
                train_rows=args.train_rows,
                val_rows=args.val_rows,
                device=device,
            )

        normalized_group_size = local_opt._normalize_group_size(args.group_size, weight_opt.shape[1])
        quantizer_sym = local_opt._quantizer_sym_for_impl(True, case.quantizer_impl)
        with cuda_timer(device) as pair_timer:
            if case.pair_impl == "reference":
                pairs, theta_mask = local_opt.build_random_rotation_buffers_reference(
                    in_features=weight_opt.shape[1],
                    group_size=normalized_group_size,
                    krot=args.krot,
                    pair_ratio=args.pair_ratio,
                    seed=seed,
                    device=device,
                )
            else:
                pairs, theta_mask = local_opt.build_random_rotation_buffers(
                    in_features=weight_opt.shape[1],
                    group_size=normalized_group_size,
                    krot=args.krot,
                    pair_ratio=args.pair_ratio,
                    seed=seed,
                    device=device,
                )

        model = local_opt._ParoQuantOptimLinear(
            weight_opt,
            bias_opt,
            bits=args.bits,
            group_size=normalized_group_size,
            quantizer_sym=quantizer_sym,
            pairs=pairs,
            theta_mask=theta_mask,
            fused_rotation=True,
        ).to(device=device, dtype=torch.float32)
        model.reset_masked_angles()

        with cuda_timer(device) as stage1_timer:
            local_opt._run_stage(
                model=model,
                inputs_train=inputs_train,
                targets_train=targets_train,
                inputs_val=inputs_val,
                targets_val=targets_val,
                param_groups=[
                    {"params": [model.channel_scales_opt], "lr": args.rotation_lr},
                    {"params": [model.theta], "lr": args.rotation_lr},
                ],
                epochs=args.rotation_epochs,
                batch_size=args.batch_size,
                stage_impl=case.stage_impl,
            )

        with cuda_timer(device) as stage2_timer:
            model.init_quantizer()
            train_loss, val_loss = local_opt._run_stage(
                model=model,
                inputs_train=inputs_train,
                targets_train=targets_train,
                inputs_val=inputs_val,
                targets_val=targets_val,
                param_groups=[
                    {"params": [model.weight], "lr": args.weight_lr},
                    {"params": model.quantizer.optim_params(), "lr": args.quantizer_lr},
                ],
                epochs=args.finetune_epochs,
                batch_size=args.batch_size,
                stage_impl=case.stage_impl,
            )

        with cuda_timer(device) as export_timer:
            result = local_opt._result_from_model(
                model,
                train_loss=train_loss,
                val_loss=val_loss,
                used_identity=False,
            )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return BenchResult(
        module=module_name,
        impl=case.label,
        pair_s=pair_timer.elapsed,
        setup_s=setup_timer.elapsed,
        stage1_s=stage1_timer.elapsed,
        stage2_s=stage2_timer.elapsed,
        export_s=export_timer.elapsed,
        opt_s=pair_timer.elapsed + setup_timer.elapsed + stage1_timer.elapsed + stage2_timer.elapsed,
        total_s=total_timer.elapsed,
        train_loss=float(result.train_loss),
        val_loss=float(result.val_loss),
        repeat=repeat,
    )


def _run_official_case(
    *,
    module_name: str,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    inputs: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    repeat: int,
    official_repo: Path,
) -> BenchResult:
    seed = int(args.seed) + (repeat * 1000)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    PseudoQuantizedLinear, transform_to_kernel_data, get_random_rotation_pairs, optimize_module = _import_official(
        official_repo
    )

    with cuda_timer(device) as total_timer:
        with cuda_timer(device) as setup_timer:
            weight_opt, bias_opt, inputs_train, targets_train, inputs_val, targets_val = _prepare_rows(
                weight,
                bias,
                inputs,
                train_rows=args.train_rows,
                val_rows=args.val_rows,
                device=device,
            )
            train_input_batches = _batch_list(inputs_train, args.batch_size)
            train_output_batches = _batch_list(targets_train, args.batch_size)
            val_input_batches = _batch_list(inputs_val, args.batch_size)
            val_output_batches = _batch_list(targets_val, args.batch_size)

        normalized_group_size = local_opt._normalize_group_size(args.group_size, weight_opt.shape[1])
        with cuda_timer(device) as pair_timer:
            weight_grouped = weight_opt.view(weight_opt.shape[0], -1, normalized_group_size).permute(1, 0, 2)
            all_pairs = get_random_rotation_pairs(
                weight_grouped,
                group_size=normalized_group_size,
                num_rotations=args.krot,
                num_pairs_factor=args.pair_ratio,
                seed=seed,
            )
            all_pairs = [torch.tensor(pairs, device="cpu", dtype=torch.int32) for pairs in all_pairs]
            initial_angles = [torch.zeros(pair_tensor.shape[0], device="cpu") for pair_tensor in all_pairs]
            npairs, angles, mask = transform_to_kernel_data(
                all_pairs,
                initial_angles,
                group_size=normalized_group_size,
            )
            linear = nn.Linear(
                weight_opt.shape[1],
                weight_opt.shape[0],
                bias=bias_opt is not None,
                device=device,
                dtype=torch.float32,
            )
            linear.weight.data.copy_(weight_opt)
            if bias_opt is not None:
                linear.bias.data.copy_(bias_opt)
            channel_scales = torch.ones((1, weight_opt.shape[1]), dtype=weight_opt.dtype, device=device)
            module = PseudoQuantizedLinear(
                linear,
                [npairs.to(device), angles.to(device), mask.to(device)],
                channel_scales,
                group_size=normalized_group_size,
                n_bits=args.bits,
                num_rotations=args.krot,
            )

        def _param_group(params: list[nn.Parameter], lr: float) -> dict[str, object]:
            return {
                "params": params,
                "lr": lr,
                "weight_decay": 0.01,
                "betas": (0.9, 0.95),
                "eps": 1e-10,
            }

        with cuda_timer(device) as stage1_timer:
            module.set_optim_enabled(channel_scales=True, angles=True)
            optimize_module(
                module,
                (train_input_batches, train_output_batches),
                (val_input_batches, val_output_batches),
                {},
                [
                    _param_group(module.get_optim_params("channel_scales"), args.rotation_lr),
                    _param_group(module.get_optim_params("angles"), args.rotation_lr),
                ],
                loss_fn="smooth_l1",
                n_iter=args.rotation_epochs,
                gradient_accumulation_steps=1,
                early_stop=None,
                post_optim_callback=lambda current: current.reset_angles_by_mask(),
            )

        with cuda_timer(device) as stage2_timer:
            module.set_optim_enabled(weight=True, quantizer=True)
            optimize_module(
                module,
                (train_input_batches, train_output_batches),
                (val_input_batches, val_output_batches),
                {},
                [
                    _param_group(module.get_optim_params("weight"), args.weight_lr),
                    _param_group(module.get_optim_params("quantizer"), args.quantizer_lr),
                ],
                loss_fn="smooth_l1",
                n_iter=args.finetune_epochs,
                gradient_accumulation_steps=1,
                early_stop=None,
                post_optim_callback=lambda current: current.reset_angles_by_mask(),
            )

        with cuda_timer(device) as export_timer:
            train_loss = _common_loss(module, None, inputs_train, targets_train)
            val_loss = _common_loss(module, None, inputs_val, targets_val)
            _ = module.pseudo_weight()

    del module
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return BenchResult(
        module=module_name,
        impl="official_pr18",
        pair_s=pair_timer.elapsed,
        setup_s=setup_timer.elapsed,
        stage1_s=stage1_timer.elapsed,
        stage2_s=stage2_timer.elapsed,
        export_s=export_timer.elapsed,
        opt_s=pair_timer.elapsed + setup_timer.elapsed + stage1_timer.elapsed + stage2_timer.elapsed,
        total_s=total_timer.elapsed,
        train_loss=train_loss,
        val_loss=val_loss,
        repeat=repeat,
    )


def _summarize(results: list[BenchResult]) -> list[list[str]]:
    rows: list[list[str]] = []
    by_module: dict[str, list[BenchResult]] = {}
    for result in results:
        by_module.setdefault(result.module, []).append(result)

    for module_name, module_results in by_module.items():
        official_total = statistics.median(
            [result.total_s for result in module_results if result.impl == "official_pr18"]
        )
        ordered_impls = ["official_pr18", *[case.label for case in LOCAL_CASES]]
        for impl in ordered_impls:
            if not any(result.impl == impl for result in module_results):
                continue
            selected = [result for result in module_results if result.impl == impl]
            median_total = statistics.median(result.total_s for result in selected)
            rows.append(
                [
                    module_name,
                    impl,
                    f"{statistics.median(result.pair_s for result in selected):.3f}",
                    f"{statistics.median(result.setup_s for result in selected):.3f}",
                    f"{statistics.median(result.stage1_s for result in selected):.3f}",
                    f"{statistics.median(result.stage2_s for result in selected):.3f}",
                    f"{statistics.median(result.export_s for result in selected):.3f}",
                    f"{statistics.median(result.opt_s for result in selected):.3f}",
                    f"{median_total:.3f}",
                    f"{official_total / median_total:.3f}x" if median_total > 0 else "",
                    f"{statistics.mean(result.train_loss for result in selected):.6f}",
                    f"{statistics.mean(result.val_loss for result in selected):.6f}",
                ]
            )
    return rows


def _print_table(title: str, rows: list[list[str]]) -> None:
    print(title)
    print(
        tabulate(
            rows,
            headers=[
                "module",
                "impl",
                "pair_s",
                "setup_s",
                "stage1_s",
                "stage2_s",
                "export_s",
                "opt_s",
                "total_s",
                "vs_official",
                "train_l1",
                "val_l1",
            ],
            tablefmt="grid",
        )
    )


def main() -> int:
    args = parse_args()
    asset_dir = Path(args.asset_dir)
    asset_path, metadata_path = _asset_paths(asset_dir, args.asset_name)
    if args.force_recapture or not asset_path.exists() or not metadata_path.exists():
        _capture_asset(args, asset_path, metadata_path)

    if args.capture_only:
        print(f"Saved activation asset to {asset_path}")
        print(f"Saved metadata to {metadata_path}")
        return 0

    _, modules = _load_asset(asset_path, metadata_path, args.module_filter)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _warmup_local_kernel(device, args.group_size, args.krot)
    _import_official(Path(args.official_repo))

    results: list[BenchResult] = []
    for module_name, bundle in modules.items():
        for repeat in range(args.repeats):
            results.append(
                _run_official_case(
                    module_name=module_name,
                    weight=bundle["weight"],
                    bias=bundle["bias"],
                    inputs=bundle["inputs"],
                    args=args,
                    device=device,
                    repeat=repeat,
                    official_repo=Path(args.official_repo),
                )
            )
            for case in LOCAL_CASES:
                results.append(
                    _run_local_case(
                        module_name=module_name,
                        case=case,
                        weight=bundle["weight"],
                        bias=bundle["bias"],
                        inputs=bundle["inputs"],
                        args=args,
                        device=device,
                        repeat=repeat,
                    )
                )

    _print_table("ParoQuant Official vs Local", _summarize(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
