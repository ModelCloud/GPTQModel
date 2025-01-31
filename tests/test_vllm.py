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

import importlib.util  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import torch  # noqa: E402
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestLoadVLLM(ModelTest):

    @classmethod
    def setUpClass(self):
        if importlib.util.find_spec("flashinfer") is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "flashinfer", "-i",
                                   f"https://flashinfer.ai/whl/cu{torch.version.cuda.replace('.', '')}/torch{'.'.join(torch.__version__.split('.')[:2])}"])

        if importlib.util.find_spec("vllm") is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "vllm>=0.6.2"])

        from vllm import SamplingParams  # noqa: E402
        self.MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
        self.SHARDED_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit-sharded"
        self.prompts = [
            self.INFERENCE_PROMPT,
        ]
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=16, top_k=1)

    def release_vllm_model(self):
        from vllm.distributed.parallel_state import destroy_model_parallel  # noqa: E402

        destroy_model_parallel()
        torch_empty_cache()

    def test_load_vllm(self):
        model = GPTQModel.load(
            self.MODEL_ID,
            device="cuda",
            backend=BACKEND.VLLM,
            gpu_memory_utilization=0.8,
        )

        tokenizer = model.get_tokenizer()

        self.assertInference(model, tokenizer)

        del model
        self.release_vllm_model()

    def test_load_shared_vllm(self):
        model = GPTQModel.load(
            self.SHARDED_MODEL_ID,
            device="cuda",
            backend=BACKEND.VLLM,
            gpu_memory_utilization=0.8,
        )
        tokenizer = model.get_tokenizer()

        self.assertInference(model, tokenizer)

        del model
        self.release_vllm_model()

    def test_dynamic(self):
        NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(NATIVE_MODEL_ID, use_fast=True)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        calibration_dataset = self.load_dataset(tokenizer)

        # support dynamic override of bits, group_size, desc_act, sym for each layer/module match
        #
        dynamic = {
            # `.*\.` matches the layers_node prefix
            # layer index start at 0
            r"-:model\.layers\.0\..*": {},  # skip 0 layers
            r".*\.18\..*(gate|up).*": {"bits": 8, "group_size": 64},  # match layer 18 gate and up module
            r".*\.19\..*(gate|up).*": {"bits": 8, "group_size": 64},  # match layer 19 gate and up module
            r".*\.20\..*(gate|up).*": {"bits": 8, "group_size": 64},  # match layer 20 gate and up module
            r".*\.21\..*(gate|up).*": {"bits": 8, "group_size": 64},  # match layer 21 gate and up module
        }
        quantize_config = QuantizeConfig(
            bits=4,
            dynamic=dynamic,
            group_size=128,
        )
        model = GPTQModel.load(
            NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )
        model.quantize(calibration_dataset, batch_size=4)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tokenizer.save_pretrained(tmp_dir)
            model.save(tmp_dir)

            del model

            model = GPTQModel.load(
                tmp_dir,
                device="cuda",
                backend=BACKEND.VLLM,
                gpu_memory_utilization=0.8,
            )

            tokenizer = model.get_tokenizer()

            for name, submodule in model.named_modules():
                if name == 'model.model.layers.0.self_attn.q_proj' and isinstance(submodule, BaseQuantLinear):  # module 0 was skipped
                    raise ValueError("first layer should be native module")

            self.assertInference(model, tokenizer)

            del model
            self.release_vllm_model()
