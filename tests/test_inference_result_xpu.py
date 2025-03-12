# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
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

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tempfile

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig
from gptqmodel.models._const import DEVICE
from models.model_test import ModelTest
from parameterized import parameterized


class TestInferenceResultXPU(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"

    @parameterized.expand(
        [
            (BACKEND.TORCH, DEVICE.XPU, True),
            (BACKEND.TORCH, DEVICE.XPU, False),
        ]
    )
    def testTritonXPU(self, backend, device, template):
        origin_model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=QuantizeConfig(),
            backend=backend,
            device=device,
        )
        tokenizer = self.load_tokenizer(self.NATIVE_MODEL_ID)
        calibration_dataset = self.load_dataset(tokenizer, rows=128)
        origin_model.quantize(calibration_dataset, backend=BACKEND.TRITON)

        with tempfile.TemporaryDirectory() as tmpdir:
          origin_model.save(tmpdir)

          messages = [
              [{"role": "user", "content": "The capital of France is"}],
              [{"role": "user", "content": "The capital of the United Kingdom is"}],
              [{"role": "user", "content": "The largest ocean on Earth is"}],
              [{"role": "user", "content": "The world’s longest river is"}],
              [{"role": "user", "content": "The tallest mountain in the world is"}],
              [{"role": "user", "content": "How are you?"}],
              [{"role": "user", "content": "I love reading and ??."}],
              [{"role": "user", "content": "What is the official language of China?"}],
              [{"role": "user", "content": "I am a good ??."}],
              [{"role": "user", "content": "What is the official language of France?"}],
          ]

          model = GPTQModel.load(
              tmpdir,
              backend=backend,
              device=device,
          )

          tokenizer = model.tokenizer

          for message in messages:
              if template:
                  inputs_tensor = tokenizer.apply_chat_template(
                      message,
                      add_generation_prompt=True,
                      return_tensors='pt').to(model.device)
              else:
                  inputs_tensor = tokenizer(message[0]["content"], return_tensors="pt")["input_ids"].to(model.device)

              result = model.generate(inputs_tensor, max_length=128, eos_token_id=tokenizer.eos_token_id)
              generate_str = tokenizer.batch_decode(result[:, inputs_tensor.size(-1):],  skip_special_tokens=True)
              print(f"generate_str: {generate_str}")
