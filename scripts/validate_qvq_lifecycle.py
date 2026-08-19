#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Quantize a real-model layer scope with QVQ and gate live/reloaded logits."""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models.base import (
    MODULE_TREE_ATTENTION_FLAGS,
    MODULE_TREE_FLAG_K,
    MODULE_TREE_FLAG_O,
    MODULE_TREE_FLAG_Q,
    MODULE_TREE_FLAG_V,
)
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization import (
    FORMAT,
    ModuleGranularReplayConfig,
    OutputAlignConfig,
    QVQConfig,
    YaqaConfig,
)
from gptqmodel.quantization.qvq_rates import normalize_qvq_rate
from gptqmodel.quantization.qvq_yaqa import (
    YAQA_PAPER_MINIMUM_SEQUENCES,
    YAQA_PAPER_REGULARIZATION,
)
from gptqmodel.utils.model import get_layers_with_prefixes
from gptqmodel.utils.qvq_validation import (
    assert_qvq_dense_accuracy as _assert_dense_accuracy,
)
from gptqmodel.utils.qvq_validation import (
    assert_qvq_reload_parity as _assert_reload_parity,
)
from gptqmodel.utils.qvq_validation import qvq_accuracy_metrics as _accuracy_metrics
from gptqmodel.utils.qvq_validation import validate_qvq_lifecycle_args as _validate_args

DEFAULT_PROMPTS = (
    "The capital of France is",
    "A prime number is an integer greater than one that",
    "Water freezes at zero degrees Celsius because",
    "In a right triangle, the Pythagorean theorem states",
)
QVQ_INFERENCE_DTYPE = torch.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--dataset", default="/monster/data/model/dataset/nm-calibration")
    parser.add_argument("--dataset-config", default="LLM")
    parser.add_argument("--row-start", type=int, default=0, help="Starting row for ordinary calibration.")
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument(
        "--concat-size",
        type=int,
        default=0,
        help="Concatenated calibration length; 0 preserves natural rows.",
    )
    parser.add_argument("--calibration-sort", choices=("none", "asc", "desc"), default="none")
    parser.add_argument(
        "--exclude-module",
        action="append",
        default=[],
        help="Exact dense module path to exclude from QVQ; may be repeated.",
    )
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--bits", type=float, default=2.0)
    parser.add_argument(
        "--format",
        choices=("qvq", "qvq_v4", "qvq_v2b2_p32", "qvq_v2b4_p64"),
        default="qvq",
        help="QVQ codec format, including segmented banked-V2 lifecycle validation formats.",
    )
    parser.add_argument(
        "--bank-count",
        type=int,
        choices=(1, 2, 4),
        default=1,
        help="QVQ bank count: 2 for V2B2-P32, 4 for V4/V2B4-P64, otherwise 1.",
    )
    parser.add_argument(
        "--attention-bits",
        type=float,
        default=None,
        help="Override explicitly tagged Q/K/V/O projections; never inferred from module names.",
    )
    parser.add_argument("--rounding", choices=("block_ldlq", "yaqa"), default="block_ldlq")
    parser.add_argument(
        "--propagated-bank-selection",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable propagation-aware bank selection; omitted uses the QVQ default.",
    )
    parser.add_argument("--yaqa-seed", type=int, default=0)
    parser.add_argument("--yaqa-regularization", type=float, default=YAQA_PAPER_REGULARIZATION)
    parser.add_argument("--yaqa-minimum-sequences", type=int, default=YAQA_PAPER_MINIMUM_SEQUENCES)
    parser.add_argument(
        "--yaqa-rows",
        type=int,
        default=None,
        help="Independent YAQA Fisher rows; omitted reuses the ordinary calibration stream.",
    )
    parser.add_argument(
        "--yaqa-row-start",
        type=int,
        default=0,
        help="Starting row in the independent YAQA dataset.",
    )
    parser.add_argument(
        "--yaqa-dataset",
        default=None,
        help="Independent YAQA dataset path/name; omitted uses --dataset.",
    )
    parser.add_argument(
        "--yaqa-dataset-config",
        default=None,
        help="Independent YAQA dataset config; omitted uses --dataset-config.",
    )
    parser.add_argument("--module-granular-replay", action="store_true")
    parser.add_argument("--module-replay-search-row-start", type=int, default=0)
    parser.add_argument("--module-replay-search-rows", type=int, default=0)
    parser.add_argument("--module-replay-confirmation-row-start", type=int, default=0)
    parser.add_argument("--module-replay-confirmation-rows", type=int, default=0)
    parser.add_argument(
        "--module-replay-subsets",
        nargs="+",
        default=("attention_qkvo",),
        choices=(
            "attention_qk",
            "attention_vo",
            "attention_qkvo",
            "mlp_gate_up",
            "mlp_down",
            "mlp_gate_up_down",
        ),
    )
    parser.add_argument("--output-alignment", action="store_true")
    parser.add_argument("--output-alignment-no-pristine-hessian", action="store_true")
    parser.add_argument("--output-alignment-lr", type=float, default=1e-5)
    parser.add_argument("--output-alignment-epochs", type=int, default=1)
    parser.add_argument("--output-alignment-optimizer", choices=("adam", "adamw"), default="adam")
    parser.add_argument("--output-alignment-weight-decay", type=float, default=0.0)
    parser.add_argument("--output-alignment-train-batches", type=int, default=32)
    parser.add_argument("--output-alignment-validation-batches", type=int, default=16)
    parser.add_argument("--output-alignment-validation-fraction", type=float, default=0.2)
    parser.add_argument("--output-alignment-minimum-improvement", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--yaqa-batch-size", type=int, default=8)
    parser.add_argument("--yaqa-no-activation-checkpointing", action="store_true")
    parser.add_argument("--yaqa-mps-cleanup-interval", type=int, default=8)
    parser.add_argument("--yaqa-sequence-sort", choices=("none", "asc", "desc"), default="desc")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-forward-kld", type=float, required=True)
    parser.add_argument("--min-top1-agreement", type=float, required=True)
    parser.add_argument("--reload-rtol", type=float, default=0.0)
    parser.add_argument("--reload-atol", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _model_device(model) -> torch.device:
    for tensor in list(model.model.parameters()) + list(model.model.buffers()):
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device("cpu")


def _calibration_controls(args: argparse.Namespace) -> tuple[int | None, str | None, dict[str, dict] | None]:
    """Normalize natural-row preparation and exact dense-module exclusions."""

    if args.concat_size < 0:
        raise ValueError("--concat-size must be nonnegative; use 0 to preserve natural rows")
    if len(args.exclude_module) != len(set(args.exclude_module)) or any(not name for name in args.exclude_module):
        raise ValueError("--exclude-module values must be nonempty and unique")
    calibration_concat_size = args.concat_size or None
    calibration_sort = None if args.calibration_sort == "none" else args.calibration_sort
    dynamic = {f"-:^{re.escape(name)}$": {} for name in args.exclude_module} or None
    return calibration_concat_size, calibration_sort, dynamic


def _calibration_row_range(args: argparse.Namespace, *, dataset_length: int) -> range:
    """Resolve an exact ordinary-calibration slice for disjoint replication runs."""

    if args.row_start < 0:
        raise ValueError("--row-start must be nonnegative.")
    row_stop = args.row_start + args.rows
    if row_stop > dataset_length:
        raise ValueError(
            f"Ordinary calibration requires rows [{args.row_start}, {row_stop}), "
            f"but dataset contains only {dataset_length} rows."
        )
    return range(args.row_start, row_stop)


def _yaqa_calibration_controls(args: argparse.Namespace) -> tuple[str, str, int, int] | None:
    """Resolve an exact, independent YAQA stream without changing base calibration."""

    if args.yaqa_rows is None:
        if args.yaqa_row_start != 0 or args.yaqa_dataset is not None or args.yaqa_dataset_config is not None:
            raise ValueError("Independent YAQA dataset controls require --yaqa-rows.")
        return None
    if args.rounding != "yaqa":
        raise ValueError("--yaqa-rows requires --rounding yaqa.")
    if args.yaqa_rows < 1:
        raise ValueError("--yaqa-rows must be positive.")
    if args.yaqa_row_start < 0:
        raise ValueError("--yaqa-row-start must be nonnegative.")
    dataset = args.dataset if args.yaqa_dataset is None else args.yaqa_dataset
    dataset_config = args.dataset_config if args.yaqa_dataset_config is None else args.yaqa_dataset_config
    return dataset, dataset_config, args.yaqa_row_start, args.yaqa_rows


def _install_semantic_attention_bits(model, bits: float | None, *, layers: int) -> tuple[str, ...]:
    """Install exact per-module rates from explicit module-tree attention roles.

    The benchmark must not infer projection semantics from strings such as
    ``q_proj`` or ``o_proj``. Requiring all four roles also prevents a partially
    tagged model definition from silently running a different experiment.
    """

    if bits is None:
        return ()
    rate = normalize_qvq_rate(bits)
    qcfg = model.quantize_config
    groups = model.simple_layer_modules(
        model_config=model.model.config,
        quantize_config=qcfg,
        is_awq_quantize=False,
        include_capture_only=False,
    )
    role_paths: dict[str, list[str]] = {
        MODULE_TREE_FLAG_Q: [],
        MODULE_TREE_FLAG_K: [],
        MODULE_TREE_FLAG_V: [],
        MODULE_TREE_FLAG_O: [],
    }
    for path in (path for group in groups for path in group):
        roles = model.get_module_tree_flags(path) & MODULE_TREE_ATTENTION_FLAGS
        if len(roles) > 1:
            raise ValueError(f"QVQ mixed-rate benchmark requires one attention role per module; `{path}` has {roles}.")
        if roles:
            role_paths[next(iter(roles))].append(path)

    missing_roles = sorted(role for role, paths in role_paths.items() if not paths)
    if missing_roles:
        raise ValueError(f"QVQ mixed-rate benchmark module tree is missing attention roles: {missing_roles}.")

    _, layer_names = get_layers_with_prefixes(model.model, model.extract_layers_node())
    if not layer_names:
        raise ValueError("QVQ mixed-rate benchmark could not resolve any decoder layers.")
    if layers > len(layer_names):
        raise ValueError(f"QVQ mixed-rate benchmark requested {layers} layers, but the model has {len(layer_names)}.")
    layer_names = layer_names[:layers]
    relative_paths = tuple(dict.fromkeys(path for paths in role_paths.values() for path in paths))
    targets = tuple(f"{layer_name}.{path}" for layer_name in layer_names for path in relative_paths)
    existing_dynamic = dict(qcfg.dynamic or {})
    conflicting = [name for name in targets if qcfg.dynamic_get(name, default=None) is not None]
    if conflicting:
        raise ValueError(f"QVQ attention-rate override conflicts with existing dynamic rules: {conflicting[:4]}.")

    exact_overrides = {f"+:^{re.escape(name)}$": {"bits": rate} for name in targets}
    qcfg._invalidate_dynamic_cache()
    qcfg.dynamic = {**existing_dynamic, **exact_overrides}
    qcfg.__post_init__()
    return targets


@torch.inference_mode()
def _masked_logits(model, prompts: tuple[str, ...], tokenizer=None) -> tuple[torch.Tensor, torch.Tensor]:
    tokenizer = model.tokenizer if tokenizer is None else tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True)
    keep_mask = encoded["attention_mask"].to(dtype=torch.bool).cpu()
    runtime_model = model.model if hasattr(model, "model") and hasattr(model.model, "generate") else model
    device = _model_device(model) if runtime_model is not model else next(runtime_model.parameters()).device
    encoded = {name: value.to(device) for name, value in encoded.items()}
    logits = runtime_model(**encoded).logits.detach().to(dtype=torch.float32, device="cpu")
    return logits[keep_mask], keep_mask


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.module_granular_replay:
        if args.format != "qvq_v2b2_p32" or args.rounding != "yaqa":
            raise ValueError("Module-granular replay requires V2B2-P32 with YAQA rounding.")
        if args.module_replay_search_rows < 2 or args.module_replay_confirmation_rows < 1:
            raise ValueError("Module-granular replay requires at least two search rows and one confirmation row.")
        search_range = range(
            args.module_replay_search_row_start,
            args.module_replay_search_row_start + args.module_replay_search_rows,
        )
        confirmation_range = range(
            args.module_replay_confirmation_row_start,
            args.module_replay_confirmation_row_start + args.module_replay_confirmation_rows,
        )
        if set(search_range).intersection(confirmation_range):
            raise ValueError("Module replay search and confirmation row ranges must be disjoint.")
    elif args.module_replay_search_rows or args.module_replay_confirmation_rows:
        raise ValueError("Module replay row controls require --module-granular-replay.")
    calibration_concat_size, calibration_sort, dynamic = _calibration_controls(args)
    yaqa_controls = _yaqa_calibration_controls(args)

    output = Path(args.output).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    config = QVQConfig(
        bits=args.bits,
        format=FORMAT(args.format),
        bank_count=args.bank_count,
        device=args.device,
        dynamic=dynamic,
        rounding=args.rounding,
        propagated_bank_selection=args.propagated_bank_selection,
        yaqa=YaqaConfig(
            seed=args.yaqa_seed,
            regularization=args.yaqa_regularization,
            minimum_sequences=args.yaqa_minimum_sequences,
            batch_size=args.yaqa_batch_size,
            activation_checkpointing=not args.yaqa_no_activation_checkpointing,
            mps_cleanup_interval=args.yaqa_mps_cleanup_interval,
            sequence_sort=args.yaqa_sequence_sort,
        ),
        output_alignment=(
            OutputAlignConfig(
                learning_rate=args.output_alignment_lr,
                epochs=args.output_alignment_epochs,
                optimizer=args.output_alignment_optimizer,
                weight_decay=args.output_alignment_weight_decay,
                maximum_train_batches=args.output_alignment_train_batches,
                maximum_validation_batches=args.output_alignment_validation_batches,
                validation_fraction=args.output_alignment_validation_fraction,
                minimum_relative_improvement=args.output_alignment_minimum_improvement,
                pristine_hessian=not args.output_alignment_no_pristine_hessian,
            )
            if args.output_alignment
            else None
        ),
        module_granular_replay=(
            ModuleGranularReplayConfig(subsets=tuple(args.module_replay_subsets))
            if args.module_granular_replay
            else None
        ),
        # Live inference, save, and exact reload parity all require one fully
        # materialized model. A disk-offloaded shell can retain meta tensors
        # after quantization and cannot safely be moved as a whole with .to().
        offload_to_disk=False,
    )
    reference_load_kwargs = {
        "dtype": QVQ_INFERENCE_DTYPE,
        "attn_implementation": "eager",
        "trust_remote_code": args.trust_remote_code,
    }
    if torch.device(args.device).type == "mps":
        # Transformers/Accelerate direct-to-MPS loading can terminate the
        # process while materializing sharded weights. CPU materialization
        # followed by one explicit transfer is stable and leaves the dense
        # reference numerically unchanged.
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            low_cpu_mem_usage=True,
            **reference_load_kwargs,
        ).to(args.device)
    else:
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map={"": args.device},
            **reference_load_kwargs,
        )
    reference_module_names = {name for name, _ in reference_model.named_modules()}
    missing_exclusions = sorted(set(args.exclude_module) - reference_module_names)
    if missing_exclusions:
        raise ValueError(f"QVQ exclusion paths were not found in the dense model: {missing_exclusions}")
    reference_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    dense_logits, keep_mask = _masked_logits(reference_model, DEFAULT_PROMPTS, tokenizer=reference_tokenizer)
    del reference_model, reference_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    started = time.perf_counter()
    model = GPTQModel.load(
        args.model,
        quantize_config=config,
        dtype=QVQ_INFERENCE_DTYPE,
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
    )
    load_seconds = time.perf_counter() - started
    attention_rate_modules = _install_semantic_attention_bits(model, args.attention_bits, layers=args.layers)

    dataset = load_dataset(args.dataset, name=args.dataset_config, split="train")
    calibration = dataset.select(_calibration_row_range(args, dataset_length=len(dataset)))
    yaqa_calibration = None
    if yaqa_controls is not None:
        yaqa_dataset_name, yaqa_dataset_config, yaqa_row_start, yaqa_rows = yaqa_controls
        if yaqa_dataset_name == args.dataset and yaqa_dataset_config == args.dataset_config:
            yaqa_dataset = dataset
        else:
            yaqa_dataset = load_dataset(yaqa_dataset_name, name=yaqa_dataset_config, split="train")
        yaqa_row_stop = yaqa_row_start + yaqa_rows
        if yaqa_row_stop > len(yaqa_dataset):
            raise ValueError(
                f"Independent YAQA calibration requires rows [{yaqa_row_start}, {yaqa_row_stop}), "
                f"but dataset contains only {len(yaqa_dataset)} rows."
            )
        yaqa_calibration = yaqa_dataset.select(range(yaqa_row_start, yaqa_row_stop))
    module_replay_search_calibration = None
    module_replay_confirmation_calibration = None
    if args.module_granular_replay:
        search_stop = args.module_replay_search_row_start + args.module_replay_search_rows
        confirmation_stop = args.module_replay_confirmation_row_start + args.module_replay_confirmation_rows
        if max(search_stop, confirmation_stop) > len(dataset):
            raise ValueError("Module replay row range exceeds the selected dataset.")
        module_replay_search_calibration = dataset.select(
            range(args.module_replay_search_row_start, search_stop)
        )
        module_replay_confirmation_calibration = dataset.select(
            range(args.module_replay_confirmation_row_start, confirmation_stop)
        )
    quant_started = time.perf_counter()
    quant_log = model.quantize(
        calibration,
        calibration_concat_size=calibration_concat_size,
        calibration_sort=calibration_sort,
        batch_size=args.batch_size,
        backend=BACKEND.QVQ,
        yaqa_calibration=yaqa_calibration,
        module_replay_search_calibration=module_replay_search_calibration,
        module_replay_confirmation_calibration=module_replay_confirmation_calibration,
        layer_scope=slice(0, args.layers),
    )
    quant_seconds = time.perf_counter() - quant_started
    output_alignment_passes = {}
    for rows in quant_log.values():
        for row in rows:
            if "output_alignment_seconds" not in row:
                continue
            layer_index = str(row["layer"])
            stats = {
                key.removeprefix("output_alignment_"): value
                for key, value in row.items()
                if key.startswith("output_alignment_")
            }
            alignment_pass = str(int(stats["alignment_pass"]))
            layer_passes = output_alignment_passes.setdefault(layer_index, {})
            previous = layer_passes.setdefault(alignment_pass, stats)
            if previous != stats:
                raise AssertionError(
                    f"QVQ output-alignment telemetry differs within layer {layer_index} pass {alignment_pass}"
                )
    output_alignment_layers = {
        layer_index: [passes[key] for key in sorted(passes, key=int)]
        for layer_index, passes in output_alignment_passes.items()
    }
    module_granular_replay_results = {
        row["full_name"]: row["module_granular_replay"]
        for rows in quant_log.values()
        for row in rows
        if row.get("module_granular_replay") is not None
    }

    qvq_modules = [name for name, module in model.model.named_modules() if isinstance(module, QVQLinear)]
    if not qvq_modules:
        raise AssertionError("The QVQ lifecycle did not install any QVQLinear modules")
    incorrectly_quantized_exclusions = sorted(set(args.exclude_module) & set(qvq_modules))
    if incorrectly_quantized_exclusions:
        raise AssertionError(f"QVQ quantized explicitly excluded modules: {incorrectly_quantized_exclusions}")
    qvq_module_rates = {
        name: normalize_qvq_rate(module.bits)
        for name, module in model.model.named_modules()
        if isinstance(module, QVQLinear)
    }
    missing_attention_modules = sorted(set(attention_rate_modules) - set(qvq_module_rates))
    if missing_attention_modules:
        raise AssertionError(f"QVQ did not quantize tagged attention modules: {missing_attention_modules}")
    wrong_attention_rates = {
        name: qvq_module_rates[name]
        for name in attention_rate_modules
        if qvq_module_rates[name] != normalize_qvq_rate(args.attention_bits)
    }
    if wrong_attention_rates:
        raise AssertionError(f"QVQ installed incorrect attention rates: {wrong_attention_rates}")
    attention_rate_module_set = set(attention_rate_modules)
    base_rate = normalize_qvq_rate(args.bits)
    wrong_base_rates = {
        name: rate
        for name, rate in qvq_module_rates.items()
        if name not in attention_rate_module_set and rate != base_rate
    }
    if wrong_base_rates:
        raise AssertionError(f"QVQ installed incorrect non-attention rates: {wrong_base_rates}")
    rate_counts: dict[str, int] = {}
    for rate in qvq_module_rates.values():
        key = str(rate)
        rate_counts[key] = rate_counts.get(key, 0) + 1
    qvq_weight_numel = sum(
        module.in_features * module.out_features
        for module in model.model.modules()
        if isinstance(module, QVQLinear)
    )
    qvq_trellis_bytes = sum(
        module.trellis.numel() * module.trellis.element_size()
        for module in model.model.modules()
        if isinstance(module, QVQLinear)
    )
    qvq_selector_bytes = sum(
        0 if module.bank_ids is None else module.bank_ids.numel() * module.bank_ids.element_size()
        for module in model.model.modules()
        if isinstance(module, QVQLinear)
    )
    qvq_payload_bytes = qvq_trellis_bytes + qvq_selector_bytes
    qvq_auxiliary_bytes = sum(
        tensor.numel() * tensor.element_size()
        for module in model.model.modules()
        if isinstance(module, QVQLinear)
        for tensor in (module.SU, module.SV)
    )
    qvq_bank_metadata_bytes = sum(
        0 if module.bank_alt_id is None else module.bank_alt_id.numel() * module.bank_alt_id.element_size()
        for module in model.model.modules()
        if isinstance(module, QVQLinear)
    )
    qvq_auxiliary_bytes += qvq_bank_metadata_bytes
    qvq_storage = {
        "weight_numel": qvq_weight_numel,
        "trellis_bytes": qvq_trellis_bytes,
        "selector_bytes": qvq_selector_bytes,
        "bank_metadata_bytes": qvq_bank_metadata_bytes,
        "payload_bytes": qvq_payload_bytes,
        "auxiliary_bytes": qvq_auxiliary_bytes,
        "payload_bits_per_weight": qvq_payload_bytes * 8 / qvq_weight_numel,
        "effective_bits_per_weight": (qvq_payload_bytes + qvq_auxiliary_bytes) * 8 / qvq_weight_numel,
    }

    # A scoped lifecycle may leave processed and untouched layers on their
    # staging devices. Validation needs one explicit, coherent inference map.
    model.model.to(args.device)
    live_inference_started = time.perf_counter()
    live_logits, live_mask = _masked_logits(model, DEFAULT_PROMPTS)
    live_inference_seconds = time.perf_counter() - live_inference_started
    if not torch.equal(keep_mask, live_mask):
        raise AssertionError("Tokenizer padding mask changed for the live post-quantization model")
    live_accuracy = _accuracy_metrics(dense_logits, live_logits)
    _assert_dense_accuracy(live_accuracy, args)

    # GPTQModel's tokenizer loader may retain the model `config` as a runtime
    # initialization kwarg. It is not tokenizer metadata and Transformers
    # cannot JSON-serialize it into tokenizer_config.json.
    tokenizer_init_kwargs = getattr(model.tokenizer, "init_kwargs", None)
    if isinstance(tokenizer_init_kwargs, dict):
        tokenizer_init_kwargs.pop("config", None)
        tokenizer_init_kwargs.pop("model_config", None)

    save_started = time.perf_counter()
    model.save(str(output))
    save_seconds = time.perf_counter() - save_started
    post_save_logits, post_save_mask = _masked_logits(model, DEFAULT_PROMPTS)
    if not torch.equal(keep_mask, post_save_mask):
        raise AssertionError("Tokenizer padding mask changed while saving the live QVQ model")
    live_post_save_accuracy = _assert_reload_parity(live_logits, post_save_logits, args)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    reload_started = time.perf_counter()
    reloaded = GPTQModel.load(
        str(output),
        backend=BACKEND.QVQ,
        dtype=QVQ_INFERENCE_DTYPE,
        # Reload onto the same backend as the live model. `auto` can place a
        # partially quantized MPS checkpoint on CPU, turning a serialization
        # parity assertion into a cross-backend numerical comparison.
        device_map={"": args.device},
        attn_implementation="eager",
        trust_remote_code=args.trust_remote_code,
    )
    reload_seconds = time.perf_counter() - reload_started
    reload_inference_started = time.perf_counter()
    reloaded_logits, reloaded_mask = _masked_logits(reloaded, DEFAULT_PROMPTS)
    reload_inference_seconds = time.perf_counter() - reload_inference_started
    if not torch.equal(keep_mask, reloaded_mask):
        raise AssertionError("Tokenizer padding mask changed across save/reload")
    reload_accuracy = _accuracy_metrics(dense_logits, reloaded_logits)
    _assert_dense_accuracy(reload_accuracy, args)
    live_reload_accuracy = _assert_reload_parity(live_logits, reloaded_logits, args)

    payload = {
        "model": str(args.model),
        "output": str(output),
        "python_gil_enabled": getattr(__import__("sys"), "_is_gil_enabled", lambda: True)(),
        "torch": torch.__version__,
        "inference_dtype": str(QVQ_INFERENCE_DTYPE),
        "device": args.device,
        "bits": args.bits,
        "attention_bits": args.attention_bits,
        "attention_rate_modules": list(attention_rate_modules),
        "qvq_rate_counts": rate_counts,
        "qvq_storage": qvq_storage,
        "rounding": args.rounding,
        "yaqa_seed": args.yaqa_seed,
        "yaqa_regularization": args.yaqa_regularization,
        "yaqa_minimum_sequences": args.yaqa_minimum_sequences,
        "yaqa_calibration": (
            {
                "dataset": yaqa_controls[0],
                "dataset_config": yaqa_controls[1],
                "row_start": yaqa_controls[2],
                "rows": len(yaqa_calibration),
            }
            if yaqa_controls is not None
            else None
        ),
        "output_alignment": (
            {
                "learning_rate": args.output_alignment_lr,
                "epochs": args.output_alignment_epochs,
                "maximum_train_batches": args.output_alignment_train_batches,
                "maximum_validation_batches": args.output_alignment_validation_batches,
                "validation_fraction": args.output_alignment_validation_fraction,
                "minimum_relative_improvement": args.output_alignment_minimum_improvement,
                "pristine_hessian": not args.output_alignment_no_pristine_hessian,
            }
            if args.output_alignment
            else None
        ),
        "layers": args.layers,
        "calibration_row_start": args.row_start,
        "calibration_rows": len(calibration),
        "calibration_concat_size": calibration_concat_size,
        "calibration_sort": calibration_sort,
        "excluded_modules": args.exclude_module,
        "batch_size": args.batch_size,
        "non_padding_eval_tokens": int(keep_mask.sum().item()),
        "qvq_module_count": len(qvq_modules),
        "qvq_modules": qvq_modules,
        "seconds": {
            "load_dense": load_seconds,
            "quantize": quant_seconds,
            "save": save_seconds,
            "reload": reload_seconds,
            "live_inference": live_inference_seconds,
            "reloaded_inference": reload_inference_seconds,
        },
        "acceptance": {
            "max_forward_kld": args.max_forward_kld,
            "min_top1_agreement": args.min_top1_agreement,
            "reload_rtol": args.reload_rtol,
            "reload_atol": args.reload_atol,
        },
        "accuracy_live_vs_dense": live_accuracy,
        "accuracy_reloaded_vs_dense": reload_accuracy,
        "accuracy_live_after_save_vs_before_save": live_post_save_accuracy,
        "accuracy_reloaded_vs_live": live_reload_accuracy,
        "quant_log_rows": sum(len(rows) for rows in quant_log.values()),
        "output_alignment_layers": output_alignment_layers,
        "module_granular_replay": (
            {
                "subsets": args.module_replay_subsets,
                "search_row_start": args.module_replay_search_row_start,
                "search_rows": args.module_replay_search_rows,
                "confirmation_row_start": args.module_replay_confirmation_row_start,
                "confirmation_rows": args.module_replay_confirmation_rows,
                "results": module_granular_replay_results,
            }
            if args.module_granular_replay
            else None
        ),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
