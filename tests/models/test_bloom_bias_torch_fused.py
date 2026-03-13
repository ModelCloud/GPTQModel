# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import tempfile

from model_test import ModelTest
from parameterized import parameterized

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.models._const import DEVICE


class TestBloom_With_Bias_TorchFused(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/bloom-560m"

    @parameterized.expand([
        BACKEND.TORCH,
        BACKEND.TORCH_FUSED
    ])
    def test_with_torch_fused_cpu(self, backend):
        origin_model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=QuantizeConfig(),
            backend=backend,
        )
        tokenizer = self.load_tokenizer(self.NATIVE_MODEL_ID)
        calibration_dataset = self.load_dataset(tokenizer, self.DATASET_SIZE)
        origin_model.quantize(calibration_dataset, backend=BACKEND.TORCH)
        with tempfile.TemporaryDirectory() as tmpdir:
            origin_model.save(tmpdir)

            model = GPTQModel.load(
                tmpdir,
                backend=BACKEND.TORCH_FUSED,
                device=DEVICE.CPU,
            )
            generate_str = self.generate_stable_with_limit(
                model,
                tokenizer,
                "The capital city of France is named",
            )

            print(f"generate_str: {generate_str}")

            assert "paris" in generate_str.lower() or "city" in generate_str.lower()
