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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import tempfile  # noqa: E402

import transformers  # noqa: E402
from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.quantization import FORMAT, QuantizeConfig  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from packaging.version import Version  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestQuantWithTrustRemoteTrue(ModelTest):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/MiniCPM-2B-dpo-bf16"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, use_fast=True, trust_remote_code=True)

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.calibration_dataset = self.load_dataset(self.tokenizer)

    def test_diff_batch(self):
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            format=FORMAT.GPTQ,
        )
        args = {}
        # if flash_attn was installed and _attn_implementation_autoset was None, flash attention would be loaded
        # but device map is cpu, it will trow non-supported device error
        if Version(transformers.__version__) >= Version("4.46.0"):
            args["_attn_implementation_autoset"] = True
        model = GPTQModel.load(
            self.MODEL_ID,
            quantize_config=quantize_config,
            trust_remote_code=True,
            **args,
        )

        model.quantize(self.calibration_dataset, batch_size=64)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)

            del model
            py_files = [f for f in os.listdir(tmp_dir) if f.endswith('.py')]
            expected_files = ["modeling_minicpm.py", "configuration_minicpm.py"]
            for file in expected_files:
                self.assertIn(file, py_files, f"File {file} is missing in the actual files list")



