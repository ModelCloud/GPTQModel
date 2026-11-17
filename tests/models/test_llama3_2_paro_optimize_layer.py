# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from paro_optimize_case import BaseLlama3_2ParoOptimizeTest, _resolve_save_path


class TestLlama3_2_Paro(BaseLlama3_2ParoOptimizeTest):
    __test__ = True
    SAVE_PATH = _resolve_save_path(
        "GPTQMODEL_PARO_LAYER_SAVE_PATH",
        "/tmp/paro_evalution_saved_ckpt_layer",
    )
    OPT_SCOPE = "layer"
    TRAIN_ON_NOISY_INPUTS_DEFAULT = True
