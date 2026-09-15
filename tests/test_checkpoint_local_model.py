# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch


@pytest.mark.parametrize("mode", ["term", "kill-hessian"])
def test_local_llama_checkpoint_recovery(tmp_path, mode):
    from test_checkpoint_quantization import test_subprocess_recovery

    model = os.environ.get("GPTQMODEL_CHECKPOINT_TEST_MODEL")
    if not model or torch.cuda.device_count() != 2:
        pytest.skip("set GPTQMODEL_CHECKPOINT_TEST_MODEL and expose two CUDA devices")
    test_subprocess_recovery(
        tmp_path, "llama", mode, device="cuda:0", required_gpus=2, native_model=model
    )
