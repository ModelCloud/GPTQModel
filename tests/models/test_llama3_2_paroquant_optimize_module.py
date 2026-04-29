# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from paroquant_optimize_case import BaseLlama3_2ParoQuantOptimizeTest, _resolve_save_path


class TestLlama3_2_ParoQuant(BaseLlama3_2ParoQuantOptimizeTest):
    __test__ = True
    SAVE_PATH = _resolve_save_path(
        "GPTQMODEL_PAROQUANT_MODULE_SAVE_PATH",
        "/tmp/paroquant_evalution_saved_ckpt_module",
    )
    OPT_SCOPE = "module"
