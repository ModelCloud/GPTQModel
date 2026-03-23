# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
from tabulate import tabulate

from gptqmodel import GPTQModel
from gptqmodel.nn_modules.qlinear.paroquant import ParoQuantQuantLinear
from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonQuantLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.eval import EVAL, evaluate, format_eval_result_table, get_eval_task_results


_NM_CALIBRATION_PATH = "/monster/data/model/dataset/nm-calibration"
_NM_CALIBRATION_PARQUET = Path("/monster/data/model/dataset/nm-calibration/llm.parquet")
_DEFAULT_MODEL = "/monster/data/model/Llama-3.2-1B-Instruct"


def _visible_cuda_device_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(torch.device("cuda:0"))


def _single_gpu_device_map() -> dict[str, str] | str:
    return {"": "cuda:0"} if torch.cuda.is_available() else "cpu"


def _load_local_calibration_parquet():
    if not _NM_CALIBRATION_PARQUET.exists():
        raise FileNotFoundError(f"Calibration parquet not found at {_NM_CALIBRATION_PARQUET}")

    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None:
        records = pd.read_parquet(_NM_CALIBRATION_PARQUET).to_dict(orient="records")
    else:
        import pyarrow.parquet as pq

        records = pq.read_table(_NM_CALIBRATION_PARQUET).to_pylist()

    normalized = []
    for record in records:
        item = {}
        for key, value in dict(record).items():
            if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
                value = value.tolist()
            item[key] = value
        normalized.append(item)
    return normalized


def load_nm_calibration(rows: int) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except Exception:
        dataset = _load_local_calibration_parquet()
    else:
        try:
            dataset = load_dataset(path=_NM_CALIBRATION_PATH, name="LLM", split="train")
        except Exception:
            dataset = _load_local_calibration_parquet()

    if rows <= 0:
        return list(dataset)
    if hasattr(dataset, "select"):
        dataset = dataset.select(range(min(rows, len(dataset))))
    else:
        dataset = list(dataset)[:rows]
    return list(dataset)


def build_first_layer_only_dynamic(model) -> dict[str, dict[str, Any]]:
    layers_node = model.extract_layers_node()
    if isinstance(layers_node, (list, tuple)):
        if not layers_node:
            raise ValueError("Model did not expose a layers node for first-layer-only ParoQuant benchmarking.")
        layers_node = layers_node[0]
    layers_node = str(layers_node).strip()
    if not layers_node:
        raise ValueError("Model layers node resolved to an empty string.")
    escaped_layers_node = layers_node.replace(".", r"\.")
    layer_count = len(getattr(model.model, "model", model.model).layers)
    return {
        f"-:^{escaped_layers_node}\\.{layer_idx}\\.": {}
        for layer_idx in range(1, layer_count)
    }


def make_paroquant_config(
    *,
    dynamic: dict[str, dict[str, Any]],
    sym: bool = True,
    bits: int = 4,
    group_size: int = 128,
    krot: int = 8,
    opt_rotation_epochs: int = 10,
    opt_finetune_epochs: int = 10,
    opt_train_samples: int = 2048,
    opt_validation_samples: int = 64,
    opt_batch_size: int = 16,
    offload_to_disk: bool = False,
) -> QuantizeConfig:
    if sym is not True:
        raise ValueError("ParoQuant benchmark: `sym=False` is disabled; use `sym=True`.")
    return QuantizeConfig(
        method=METHOD.PAROQUANT,
        format=FORMAT.PAROQUANT,
        bits=bits,
        group_size=group_size,
        sym=sym,
        desc_act=False,
        krot=krot,
        dynamic=dynamic,
        offload_to_disk=offload_to_disk,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        opt_rotation_epochs=opt_rotation_epochs,
        opt_finetune_epochs=opt_finetune_epochs,
        opt_train_samples=opt_train_samples,
        opt_validation_samples=opt_validation_samples,
        opt_batch_size=opt_batch_size,
    )


def _cleanup_model(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prepare_eval_tokenizer(model) -> None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            tokenizer.pad_token_id = eos_token_id


def _suite_kwargs(max_rows: Optional[int]) -> dict[str, Any] | None:
    if max_rows is None:
        return None
    return {"max_rows": int(max_rows)}


def run_fp16_eval(
    *,
    model_path: str = _DEFAULT_MODEL,
    eval_batch_size: int = 64,
    eval_max_rows: Optional[int] = None,
) -> dict[str, Any]:
    model = GPTQModel.load(
        model_path,
        trust_remote_code=False,
        dtype=torch.float16,
        device_map=_single_gpu_device_map(),
    )
    _prepare_eval_tokenizer(model)

    wall_start = time.perf_counter()
    eval_result = evaluate(
        model_or_id_or_path=model,
        tokenizer=model.tokenizer,
        tasks=[EVAL.LM_EVAL.GSM8K_PLATINUM_COT],
        framework=EVAL.LM_EVAL,
        batch_size=eval_batch_size,
        model_args={"padding_side": "left"},
        apply_chat_template=True,
        suite_kwargs=_suite_kwargs(eval_max_rows),
    )
    wall_s = time.perf_counter() - wall_start
    metrics = get_eval_task_results(eval_result)
    formatted = format_eval_result_table(eval_result)
    _cleanup_model(model)
    return {
        "mode": "fp16",
        "eval_wall_s": wall_s,
        "metrics": metrics,
        "eval_table": formatted,
    }


def _prehook_capture_inputs(module_names: set[str], max_rows: int = 256):
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if name in captured or not inputs:
                return
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1])[:max_rows].contiguous()
            captured[name] = x

        return hook

    return captured, hooks, make_hook


def _tokenize_calibration_sample(model, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
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
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    tokenizer = model.tokenizer
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

    raise ValueError(f"Unsupported calibration sample keys for ParoQuant kernel benchmark: {sorted(sample.keys())}")


def _clone_triton_module(module: ParoQuantQuantLinear) -> ParoQuantTritonQuantLinear:
    cloned = ParoQuantTritonQuantLinear(
        bits=module.bits,
        group_size=module.group_size,
        sym=module.sym,
        desc_act=module.desc_act,
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
        register_buffers=True,
        krot=module.krot,
        fp32_accum=module.fp32_accum,
    ).to(device=module.qweight.device)
    cloned.qweight.copy_(module.qweight)
    cloned.qzeros.copy_(module.qzeros)
    cloned.scales.copy_(module.scales)
    if module.bias is not None:
        cloned.bias.copy_(module.bias)
    cloned.pairs.copy_(module.pairs)
    cloned.theta.copy_(module.theta)
    cloned.channel_scales.copy_(module.channel_scales)
    cloned.post_init()
    cloned.eval()
    return cloned


def _dense_forward(module: ParoQuantQuantLinear, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        x_flat = x.reshape(-1, x.shape[-1])
        rotated = module._rotate_inputs(x_flat)
        return module._forward_dense(rotated)


def _benchmark_module_ms(module, x: torch.Tensor, warmup: int = 5, iters: int = 20) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        start = time.perf_counter()
        for _ in range(iters):
            module(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
    return (time.perf_counter() - start) * 1e3 / iters


def benchmark_quantized_first_layer_kernels(
    model,
    calibration_dataset: list[dict[str, Any]],
    *,
    capture_rows: int = 256,
    warmup: int = 5,
    iters: int = 20,
) -> list[list[str]]:
    qmodules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, ParoQuantQuantLinear)
    }
    if not qmodules:
        return []

    captured, hooks, make_hook = _prehook_capture_inputs(set(qmodules), max_rows=capture_rows)
    for name, module in qmodules.items():
        hooks.append(module.register_forward_pre_hook(make_hook(name)))

    try:
        sample = calibration_dataset[0]
        tokenized = _tokenize_calibration_sample(model, sample)
        model_device = next(model.model.parameters()).device
        with torch.inference_mode():
            model.model(
                input_ids=tokenized["input_ids"].to(device=model_device),
                attention_mask=tokenized["attention_mask"].to(device=model_device),
            )
    finally:
        for hook in hooks:
            hook.remove()

    rows = []
    for name, module in qmodules.items():
        x = captured.get(name)
        if x is None or x.numel() == 0:
            continue
        x = x.to(device=module.qweight.device, dtype=torch.float16)
        triton_module = _clone_triton_module(module)
        with torch.inference_mode():
            dense = _dense_forward(module, x)
            cuda_out = module(x).reshape_as(dense)
            triton_out = triton_module(x).reshape_as(dense)

        cuda_diff = (cuda_out - dense).abs()
        triton_diff = (triton_out - dense).abs()
        cuda_vs_triton = (cuda_out - triton_out).abs()

        dense_ms = _benchmark_module_ms(lambda inp: _dense_forward(module, inp), x, warmup=warmup, iters=iters)
        cuda_ms = _benchmark_module_ms(module, x, warmup=warmup, iters=iters)
        triton_ms = _benchmark_module_ms(triton_module, x, warmup=warmup, iters=iters)

        rows.append(
            [
                name,
                str(tuple(x.shape)),
                f"{cuda_diff.max().item():.6f}",
                f"{cuda_diff.mean().item():.6f}",
                f"{triton_diff.max().item():.6f}",
                f"{triton_diff.mean().item():.6f}",
                f"{cuda_vs_triton.max().item():.6f}",
                f"{dense_ms:.3f}",
                f"{cuda_ms:.3f}",
                f"{triton_ms:.3f}",
            ]
        )
        del triton_module

    return rows


def _region_rows(snapshot: dict[str, dict[str, Any]]) -> list[list[str]]:
    populated = [
        (name, stat)
        for name, stat in snapshot.items()
        if int(stat.get("count", 0))
    ]
    total = sum(float(stat.get("total", 0.0)) for _, stat in populated) or 1.0
    populated.sort(key=lambda item: float(item[1].get("total", 0.0)), reverse=True)
    return [
        [
            name,
            str(int(stat.get("count", 0))),
            f"{float(stat.get('last', 0.0)):.3f}",
            f"{float(stat.get('total', 0.0)) / max(int(stat.get('count', 0)), 1):.3f}",
            f"{float(stat.get('total', 0.0)):.3f}",
            f"{100.0 * float(stat.get('total', 0.0)) / total:.1f}%",
            str(stat.get("source") or ""),
        ]
        for name, stat in populated
    ]


def _module_time_rows(quant_logs: dict[str, list[dict[str, Any]]]) -> list[list[str]]:
    rows = []
    for entry in quant_logs.get("paroquant", []):
        rows.append(
            [
                str(entry.get("layer", "")),
                str(entry.get("module", "")),
                str(entry.get("feat: in, out", "")),
                str(entry.get("samples", "")),
                str(entry.get("loss", "")),
                str(entry.get("time", "")),
            ]
        )
    rows.sort(key=lambda item: float(item[-1] or 0.0), reverse=True)
    return rows


def run_paroquant_first_layer_case(
    *,
    model_path: str = _DEFAULT_MODEL,
    calibration_rows: int = 64,
    calibration_concat_size: int = 2048,
    quant_batch_size: int = 1,
    eval_batch_size: int = 64,
    eval_max_rows: Optional[int] = None,
    sym: bool = True,
    fused_opt_rotation: bool = True,
    opt_rotation_epochs: int = 10,
    opt_finetune_epochs: int = 10,
    opt_train_samples: int = 2048,
    opt_validation_samples: int = 64,
    opt_batch_size: int = 16,
) -> dict[str, Any]:
    if sym is not True:
        raise ValueError("ParoQuant benchmark: `sym=False` is disabled; use `sym=True`.")
    os.environ["GPTQMODEL_PAROQUANT_OPT_FUSED_ROTATION"] = "1" if fused_opt_rotation else "0"

    calibration_dataset = load_nm_calibration(calibration_rows)
    probe_model = GPTQModel.load(
        model_path,
        quantize_config=QuantizeConfig(method=METHOD.PAROQUANT, format=FORMAT.PAROQUANT),
        trust_remote_code=False,
        dtype=torch.float16,
        device_map=_single_gpu_device_map(),
    )
    dynamic = build_first_layer_only_dynamic(probe_model)
    _cleanup_model(probe_model)
    probe_model = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    qcfg = make_paroquant_config(
        dynamic=dynamic,
        sym=sym,
        opt_rotation_epochs=opt_rotation_epochs,
        opt_finetune_epochs=opt_finetune_epochs,
        opt_train_samples=opt_train_samples,
        opt_validation_samples=opt_validation_samples,
        opt_batch_size=opt_batch_size,
    )
    model = GPTQModel.load(
        model_path,
        quantize_config=qcfg,
        trust_remote_code=False,
        dtype=torch.float16,
    )
    _prepare_eval_tokenizer(model)
    try:
        quant_start = time.perf_counter()
        quant_logs = model.quantize(
            calibration_dataset,
            calibration_concat_size=calibration_concat_size,
            calibration_sort="desc",
            batch_size=quant_batch_size,
        )
        quant_wall_s = time.perf_counter() - quant_start

        if torch.cuda.is_available():
            model.model.to("cuda:0")

        eval_start = time.perf_counter()
        eval_result = evaluate(
            model_or_id_or_path=model,
            tokenizer=model.tokenizer,
            tasks=[EVAL.LM_EVAL.GSM8K_PLATINUM_COT],
            framework=EVAL.LM_EVAL,
            batch_size=eval_batch_size,
            model_args={"padding_side": "left"},
            apply_chat_template=True,
            suite_kwargs=_suite_kwargs(eval_max_rows),
        )
        eval_wall_s = time.perf_counter() - eval_start
        kernel_rows = benchmark_quantized_first_layer_kernels(model, calibration_dataset)

        return {
            "mode": "paroquant_first_layer",
            "device": _visible_cuda_device_name(),
            "fused_opt_rotation": fused_opt_rotation,
            "sym": sym,
            "quant_wall_s": quant_wall_s,
            "eval_wall_s": eval_wall_s,
            "quant_logs": quant_logs,
            "quant_region_snapshot": model.quant_region_timer.snapshot(),
            "module_time_rows": _module_time_rows(quant_logs),
            "region_rows": _region_rows(model.quant_region_timer.snapshot()),
            "eval_metrics": get_eval_task_results(eval_result),
            "eval_table": format_eval_result_table(eval_result),
            "kernel_rows": kernel_rows,
        }
    finally:
        _cleanup_model(model)
        if probe_model is not None:
            _cleanup_model(probe_model)


def comparison_rows(*cases: dict[str, Any]) -> list[list[str]]:
    rows = []
    for case in cases:
        metric = case.get("metrics") or case.get("eval_metrics") or {}
        gsm8k = metric.get("gsm8k_platinum_cot", {})
        score = gsm8k.get("exact_match,flexible-extract")
        if score is None and gsm8k:
            for candidate_key in ("acc,num", "acc", "exact_match"):
                if candidate_key in gsm8k:
                    score = gsm8k[candidate_key]
                    break
            if score is None:
                score = next(iter(gsm8k.values()))
        label = case.get("label") or case.get("mode", "")
        rows.append(
            [
                label,
                str(case.get("sym", "")),
                str(case.get("fused_opt_rotation", "")),
                "" if score is None else f"{float(score):.6f}",
                "" if "quant_wall_s" not in case else f"{float(case['quant_wall_s']):.3f}",
                "" if "eval_wall_s" not in case else f"{float(case['eval_wall_s']):.3f}",
            ]
        )
    return rows


def render_case_tables(case: dict[str, Any]) -> dict[str, str]:
    return {
        "comparison": tabulate(
            comparison_rows(case),
            headers=["case", "sym", "fused_opt", "gsm8k_platinum_cot", "quant_wall_s", "eval_wall_s"],
            tablefmt="grid",
        ),
        "module_times": tabulate(
            case.get("module_time_rows", []),
            headers=["layer", "module", "feat", "samples", "loss", "time_s"],
            tablefmt="grid",
        ),
        "regions": tabulate(
            case.get("region_rows", []),
            headers=["region", "count", "last_s", "avg_s", "total_s", "pct", "source"],
            tablefmt="grid",
        ),
        "kernels": tabulate(
            case.get("kernel_rows", []),
            headers=[
                "module",
                "input_shape",
                "cuda_max_abs",
                "cuda_mean_abs",
                "triton_max_abs",
                "triton_mean_abs",
                "cuda_vs_triton_max_abs",
                "dense_ms",
                "cuda_ms",
                "triton_ms",
            ],
            tablefmt="grid",
        ),
        "eval": case.get("eval_table", ""),
    }


def write_case_json(case: dict[str, Any], output_path: str | os.PathLike[str]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(case, handle, indent=2, sort_keys=True)
