# Copyright 2025 ModelCloud
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -- do not touch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# -- end do not touch

import tempfile  # noqa: E402

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.models._const import DEVICE  # noqa: E402
from models.model_test import ModelTest  # noqa: E402


class TestsIPEX(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"

    def test(self):
        origin_model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=QuantizeConfig(),
            backend=BACKEND.IPEX,
            device=DEVICE.XPU,
        )
        tokenizer = self.load_tokenizer(self.NATIVE_MODEL_ID)
        calibration_dataset = self.load_dataset(tokenizer)
        origin_model.quantize(calibration_dataset, backend=BACKEND.IPEX)
        with tempfile.TemporaryDirectory() as tmpdir:
          origin_model.save(tmpdir)

          model = GPTQModel.load(
              tmpdir,
              backend=BACKEND.IPEX,
              device=DEVICE.XPU,
          )
          generate_str = tokenizer.decode(model.generate(**tokenizer("The capital of France is is", return_tensors="pt").to(model.device), max_new_tokens=2)[0])

          print(f"generate_str: {generate_str}")

          self.assertIn("paris", generate_str.lower())
