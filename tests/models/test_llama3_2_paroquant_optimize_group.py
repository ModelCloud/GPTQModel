# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from paroquant_first_layer_case_helper import (
    assert_basic_paroquant_first_layer_result,
    resolve_paroquant_first_layer_case_env,
    run_paroquant_first_layer_case_from_resolved,
)
from tabulate import tabulate


@pytest.mark.cuda
def test_llama3_2_paroquant_optimize_group_first_2_layers(capsys: pytest.CaptureFixture[str]):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for grouped ParoQuant integration test")

    if os.environ.get("GPTQMODEL_RUN_PAROQUANT_OPTIMIZE_GROUP_TEST") != "1":
        pytest.skip("Set GPTQMODEL_RUN_PAROQUANT_OPTIMIZE_GROUP_TEST=1 to run this grouped integration test.")

    resolved = resolve_paroquant_first_layer_case_env(
        env_prefix="GPTQMODEL_PAROQUANT_GROUP_TEST",
        default_num_quant_layers=2,
        default_opt_scope="compute_block",
    )
    if resolved["opt_scope"] == "module":
        pytest.skip("Grouped optimize_group integration test requires opt_scope=compute_block or opt_scope=layer.")

    result = run_paroquant_first_layer_case_from_resolved(resolved)
    gsm8k_metrics = assert_basic_paroquant_first_layer_result(
        result,
        num_quant_layers=resolved["num_quant_layers"],
        opt_scope=resolved["opt_scope"],
    )
    assert result["opt_scope"] in {"compute_block", "layer"}

    summary_rows = [
        ["num_quant_layers", str(result["num_quant_layers"])],
        ["opt_scope", str(result["opt_scope"])],
        ["quant_wall_s", f"{float(result['quant_wall_s']):.3f}"],
        ["eval_wall_s", f"{float(result['eval_wall_s']):.3f}"],
        ["gsm8k_platinum_cot acc,num", f"{float(gsm8k_metrics['acc,num']):.6f}"],
    ]
    summary_table = tabulate(summary_rows, headers=["metric", "value"], tablefmt="grid")
    module_times = tabulate(
        result["module_time_rows"],
        headers=["layer", "module", "feat", "samples", "loss", "time_s"],
        tablefmt="grid",
    )

    with capsys.disabled():
        print("\nParoQuant Optimize Group Summary", flush=True)
        print(summary_table, flush=True)
        print("\nParoQuant Optimize Group Eval", flush=True)
        print(result.get("eval_table", ""), flush=True)
        print("\nParoQuant Optimize Group Module Times", flush=True)
        print(module_times, flush=True)
