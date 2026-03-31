# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
from paroquant_first_layer_case_helper import (
    assert_basic_paroquant_first_layer_result,
    run_paroquant_first_layer_case_from_env,
)


@pytest.mark.cuda
def test_llama3_2_paroquant_first_4_layers_full_model():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for prefix-layer ParoQuant test")

    if os.environ.get("GPTQMODEL_RUN_PAROQUANT_FIRST_LAYER_TEST") != "1":
        pytest.skip("Set GPTQMODEL_RUN_PAROQUANT_FIRST_LAYER_TEST=1 to run this prefix-layer integration test.")

    result, resolved = run_paroquant_first_layer_case_from_env(
        env_prefix="GPTQMODEL_PAROQUANT_TEST",
        default_num_quant_layers=4,
        default_opt_scope="module",
    )
    assert_basic_paroquant_first_layer_result(
        result,
        num_quant_layers=resolved["num_quant_layers"],
        opt_scope=resolved["opt_scope"],
    )
