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

from datasets import load_dataset  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear  # noqa: E402
from gptqmodel.quantization import QuantizeConfig  # noqa: E402
from gptqmodel.utils import Perplexity  # noqa: E402


class TestDynamic(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"  # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tmp_dir = None

    def calculate_avg_ppl(self, model, tokenizer):
        ppl = Perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_path="wikitext",
            dataset_name="wikitext-2-raw-v1",
            split="test",
            text_column="text",
        )

        all = ppl.calculate(n_ctx=512, n_batch=512)

        # average ppl
        avg = sum(all) / len(all)

        return avg

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.TemporaryDirectory()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.NATIVE_MODEL_ID, use_fast=True)

        if not cls.tokenizer.pad_token_id:
            cls.tokenizer.pad_token_id = cls.tokenizer.eos_token_id

        cls.calibration_dataset = cls.load_dataset(cls.tokenizer)

        # support dynamic override of bits, group_size, desc_act, sym for each layer/module match
        dynamic = {
            # `.*\.` matches the layers_node prefix
            # layer index start at 0
            r".*\.18\..*gate.*": {"bits": 8, "group_size": 64},  # match layer 18 gate module
            r".*\.19\..*gate.*": {"bits": 8, "group_size": 64},  # match layer 19 gate module
            r".*\.20\..*gate.*": {"bits": 8, "group_size": 64},  # match layer 20 gate module
            r".*\.21\..*gate.*": {"bits": 8, "group_size": 64},  # match layer 21 gate module
        }
        quantize_config = QuantizeConfig(
            bits=4,
            dynamic=dynamic,
            group_size=128,
        )
        model = GPTQModel.load(
            cls.NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )
        model.quantize(cls.calibration_dataset, batch_size=4)

        model.save(cls.tmp_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()
        assert not os.path.exists(cls.tmp_dir.name)

    @parameterized.expand(
        [
            (BACKEND.TRITON),
            (BACKEND.MARLIN),
        ]
    )
    def test_dynamic_bits(self, backend):
        model = GPTQModel.load(
            self.tmp_dir.name,
            backend=backend,
        )

        for _, submodule in model.named_modules():
            if isinstance(submodule, TritonV2QuantLinear if backend == BACKEND.TRITON else MarlinQuantLinear):
                break
        else:
            raise ValueError("Did not find a " + "tritonv2 linear layer" if backend == BACKEND.TRITON else "marlin linear layer")

        dynamic_bits_ppl = self.calculate_avg_ppl(model, self.tokenizer)

        del model
        assert dynamic_bits_ppl < 10

    def test_skip_module(self):
        dynamic = {
            r"-:model\.layers\.0\..*": {},  # skip 0 layers
        }
        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=QuantizeConfig(dynamic=dynamic),
        )
        model.quantize(self.calibration_dataset, batch_size=4)

        for name, submodule in model.named_modules():
            if name == 'model.model.layers.0.self_attn.q_proj' and isinstance(submodule, BaseQuantLinear):  # module 0 was skipped
                raise ValueError("first layer should be native module")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            del model

            q_model = GPTQModel.load(tmp_dir)
            self.assertInference(model=q_model,tokenizer=self.tokenizer)
