# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

from model_test import _env_choice, _env_flag, _env_int

from gptqmodel.utils.paroquant_benchmark import run_paroquant_first_layer_case


def resolve_paroquant_first_layer_case_env(
    *,
    env_prefix: str,
    default_num_quant_layers: int,
    default_opt_scope: str,
) -> dict[str, Any]:
    """Resolve a shared ParoQuant first-layer integration env schema."""
    num_quant_layers = _env_int(f"{env_prefix}_NUM_LAYERS", default=default_num_quant_layers)
    opt_scope = _env_choice(f"{env_prefix}_OPT_SCOPE", default=default_opt_scope)
    eval_max_rows = os.environ.get(f"{env_prefix}_MAX_ROWS")
    eval_batch = _env_choice(f"{env_prefix}_EVAL_BATCH", default="auto")
    eval_batch_size: int | str = "auto" if eval_batch == "auto" else int(eval_batch)
    resolved_eval_max_rows = 128 if eval_max_rows in {None, ""} else int(eval_max_rows)
    return {
        "num_quant_layers": num_quant_layers,
        "calibration_rows": _env_int(f"{env_prefix}_CAL_ROWS", default=32),
        "eval_batch_size": eval_batch_size,
        "eval_max_rows": resolved_eval_max_rows,
        "eval_model_args": {
            "dtype": os.environ.get(f"{env_prefix}_EVAL_DTYPE", "bfloat16"),
            "attn_implementation": os.environ.get(
                f"{env_prefix}_ATTN_IMPL",
                "paged|flash_attention_2",
            ),
            "device": os.environ.get(f"{env_prefix}_EVAL_DEVICE", "cuda:0"),
        },
        "eval_suite_kwargs": {
            "batch_size": _env_int(f"{env_prefix}_SUITE_BATCH", default=24),
            "max_new_tokens": _env_int(f"{env_prefix}_MAX_NEW_TOKENS", default=96),
            "stream": _env_flag(f"{env_prefix}_STREAMING", default=True),
            "max_rows": resolved_eval_max_rows,
        },
        "sym": True,
        "fused_opt_rotation": _env_flag(f"{env_prefix}_FUSED", default=True),
        "opt_scope": opt_scope,
        "opt_rotation_epochs": _env_int(f"{env_prefix}_ROT_EPOCHS", default=4),
        "opt_finetune_epochs": _env_int(f"{env_prefix}_FT_EPOCHS", default=4),
        "opt_train_samples": _env_int(f"{env_prefix}_TRAIN_ROWS", default=512),
        "opt_validation_samples": _env_int(f"{env_prefix}_VAL_ROWS", default=64),
        "opt_batch_size": _env_int(f"{env_prefix}_OPT_BATCH", default=16),
    }


def run_paroquant_first_layer_case_from_resolved(resolved: dict[str, Any]) -> dict[str, Any]:
    """Run a ParoQuant first-layer integration case from resolved shared options."""
    return run_paroquant_first_layer_case(
        num_quant_layers=resolved["num_quant_layers"],
        calibration_rows=resolved["calibration_rows"],
        eval_batch_size=resolved["eval_batch_size"],
        eval_max_rows=resolved["eval_max_rows"],
        eval_model_args=resolved["eval_model_args"],
        eval_suite_kwargs=resolved["eval_suite_kwargs"],
        sym=resolved["sym"],
        fused_opt_rotation=resolved["fused_opt_rotation"],
        opt_scope=resolved["opt_scope"],
        opt_rotation_epochs=resolved["opt_rotation_epochs"],
        opt_finetune_epochs=resolved["opt_finetune_epochs"],
        opt_train_samples=resolved["opt_train_samples"],
        opt_validation_samples=resolved["opt_validation_samples"],
        opt_batch_size=resolved["opt_batch_size"],
    )


def run_paroquant_first_layer_case_from_env(
    *,
    env_prefix: str,
    default_num_quant_layers: int,
    default_opt_scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build and run a ParoQuant first-layer integration case from a shared env schema."""
    resolved = resolve_paroquant_first_layer_case_env(
        env_prefix=env_prefix,
        default_num_quant_layers=default_num_quant_layers,
        default_opt_scope=default_opt_scope,
    )

    result = run_paroquant_first_layer_case_from_resolved(resolved)
    return result, resolved


def assert_basic_paroquant_first_layer_result(
    result: dict[str, Any],
    *,
    num_quant_layers: int,
    opt_scope: str,
) -> dict[str, Any]:
    """Assert the common success conditions for ParoQuant first-layer integration runs."""
    assert result["module_time_rows"], "expected per-module quantization timings"
    assert result["num_quant_layers"] == num_quant_layers
    assert result["opt_scope"] == opt_scope
    assert "gsm8k_platinum_cot" in result["eval_metrics"], "expected gsm8k platinum metrics"
    gsm8k_metrics = result["eval_metrics"]["gsm8k_platinum_cot"]
    assert "acc,num" in gsm8k_metrics, "expected gsm8k_platinum_cot acc,num metric"
    return gsm8k_metrics
