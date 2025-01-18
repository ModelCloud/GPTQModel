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

import os
import unittest

import openai
from gptqmodel import GPTQModel

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class TestOpeniServer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortext-v1"
        self.HOST = "127.0.0.1"
        self.PORT = 23900
        self.model = GPTQModel.load(self.MODEL_ID)


    def test_openai_server(self):
        self.model.serve(host=self.HOST, port=self.PORT, async_mode=True)
        self.model.serve_wait_until_ready()
        client = openai.Client(base_url=f"http://{self.HOST}:{self.PORT}/v1", api_key="None")
        messages = [
            {"role": "user", "content": "1+1=?"},
        ]
        response = client.chat.completions.create(
            model=self.MODEL_ID,
            messages=messages,
            temperature=0,
        )
        result_text = response.choices[0].text
        self.assertEqual(result_text.strip(), "1 + 1 = 2")
        self.model.serve_shutdown()
