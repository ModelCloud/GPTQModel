# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
import tempfile

from gptqmodel import GPTQModel, QuantizeConfig, BACKEND
from gptqmodel.models._const import DEVICE
from model_test import ModelTest

class TestBloom_With_TorchFused(ModelTest):
    NATIVE_MODEL_ID = "bigscience/bloom-560m"

    # def test_with_torch_cpu(self):
    #     origin_model = GPTQModel.load(
    #         self.NATIVE_MODEL_ID,
    #         quantize_config=QuantizeConfig(),
    #         backend=BACKEND.TORCH,
    #     )
    #     tokenizer = self.load_tokenizer(self.NATIVE_MODEL_ID)
    #     calibration_dataset = self.load_dataset(tokenizer, self.DATASET_SIZE)
    #     origin_model.quantize(calibration_dataset, backend=BACKEND.TORCH)
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         origin_model.save(tmpdir)
    #
    #         model = GPTQModel.load(
    #             tmpdir,
    #             backend=BACKEND.TORCH,
    #             device=DEVICE.CPU,
    #         )
    #         generate_str = tokenizer.decode(
    #             model.generate(**tokenizer("The capital of France is is", return_tensors="pt").to(model.device),
    #                            max_new_tokens=512)[0])
    #
    #         print(f"generate_str: {generate_str}")
    #
    #         self.assertIn("paris", generate_str.lower())

    def test_with_torch_fused_cpu(self):
        origin_model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=QuantizeConfig(),
            backend=BACKEND.TORCH,
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
            generate_str = tokenizer.decode(
                model.generate(**tokenizer("The capital of France is is", return_tensors="pt").to(model.device),
                               max_new_tokens=512)[0])

            print(f"generate_str: {generate_str}")

            self.assertIn("paris", generate_str.lower())
