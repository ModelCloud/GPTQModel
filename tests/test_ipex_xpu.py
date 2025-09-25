# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# -- end do not touch

import tempfile  # noqa: E402

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.models._const import DEVICE  # noqa: E402
from models.model_test import ModelTest  # noqa: E402


class TestsTorchFused(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"

    def test(self):
        origin_model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=QuantizeConfig(),
            backend=BACKEND.TORCH_FUSED,
            device=DEVICE.XPU,
        )
        tokenizer = self.load_tokenizer(self.NATIVE_MODEL_ID)
        calibration = self.load_dataset(tokenizer)
        origin_model.quantize(calibration, backend=BACKEND.TORCH_FUSED)
        with tempfile.TemporaryDirectory() as tmpdir:
          origin_model.save(tmpdir)

          model = GPTQModel.load(
              tmpdir,
              backend=BACKEND.TORCH_FUSED,
              device=DEVICE.XPU,
          )

        self.assertInference(model=model,tokenizer=tokenizer)

