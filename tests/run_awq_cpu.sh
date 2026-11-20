#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    printf 'Usage: bash %s {unit|dense|moe} [pytest options]\n' "${0##*/}"
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    unit|dense|moe) suite=$1; shift ;;
    *) usage >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

case "$suite" in
    unit)
        selection=(
            tests/test_docs_contract_refs.py
            tests/test_gptq_processor_hessian_release_5026.py
            tests/test_gptq_processor_hessian_lifetime_5026.py
            tests/test_awq_scale_search_safety.py
            tests/test_awq_moe_router_scaling.py
            tests/test_awq_scaling_groups.py
            tests/test_awq_scale_folding.py
            tests/test_awq_empty_groups.py
            tests/test_awq_mla_scaling.py
            tests/test_awq_weight_mean.py
            tests/test_awq_per_module_bits.py
            tests/test_awq_position_kwargs.py
            tests/test_awq_scale_search_memory.py
            tests/test_awq_hybrid_recurrent_mask_4671.py
            tests/test_rotation_format_gate_4666.py
            tests/test_rotation_validate_first_4935.py
            tests/test_awq_conv1d_gpt2_4887.py
            tests/test_calibration_messages_tools_4858.py
            tests/test_calibration_tools_formats_4934.py
            tests/test_calibration_messages_arrow_5082.py
            tests/test_awq_hybrid_moe_root_4903.py
            tests/test_awq_deepseek_v4_hash_moe_4880.py
            tests/test_weight_only_failed_run_device_5011.py
            tests/test_awq_torch_fused_mode_5018.py
            tests/test_awq_hash_moe_kept_rows_5015.py
            tests/test_awq_cohere2_moe_parallel_norm_4898.py
            tests/test_cpu_int4_snapshot_5013.py
            tests/test_training_kernel_inplace_4974.py
            tests/test_peft_adalora_training_4974.py
            tests/test_hymba_awq_4891.py
            tests/test_smooth_percentile_bounds_5005.py
            tests/test_hf_bridge_cpu_offload_5022.py
            tests/test_failed_run_model_state_5009.py
            tests/module_tree/test_awq_relationships.py
            tests/module_tree/test_granitemoehybrid_model_tree.py
            tests/module_tree/test_registered_module_plans.py
            tests/module_tree/test_definition_subset_ordering.py
        )
        ;;
    dense|moe)
        selection=(tests/test_awq_hf_cpu.py -k "$suite" -s)
        ;;
esac

exec python -m pytest "${selection[@]}" -q -o addopts='' -o log_cli=false \
    --timeout=300 -o junit_family=legacy "$@"
