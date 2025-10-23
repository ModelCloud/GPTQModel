# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

from inference_speed import InferenceSpeed  # noqa: E402
from parameterized import parameterized  # noqa: E402

from gptqmodel.utils import BACKEND  # noqa: E402


class TestInferenceSpeedTorchFused(InferenceSpeed):
    @parameterized.expand(
        [
            (InferenceSpeed.NATIVE_MODEL_ID, BACKEND.TORCH_FUSED, 2.57),
        ]
    )
    def test_inference_speed_torch_fused(self, model_path, backend, tokens_per_second):
        self.inference(model_path=model_path, backend=backend, tokens_per_second=tokens_per_second, device="cpu")
