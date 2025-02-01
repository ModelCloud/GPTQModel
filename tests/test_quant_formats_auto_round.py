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

import json  # noqa: E402
import logging  # noqa: E402
import tempfile  # noqa: E402

from datasets import load_dataset  # noqa: E402
from gptqmodel import BACKEND, GPTQModel, __version__, get_best_device  # noqa: E402
from gptqmodel.quantization import FORMAT, QUANT_CONFIG_FILENAME, QUANT_METHOD  # noqa: E402
from gptqmodel.quantization.config import (META_FIELD_QUANTIZER, META_QUANTIZER_GPTQMODEL,  # noqa: E402
                                           AutoRoundQuantizeConfig, QuantizeConfig)
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class TestQuantization(ModelTest):

    @classmethod
    def setUpClass(self):
        self.pretrained_model_id = "/monster/data/model/TinyLlama-1.1B-intermediate-step-1431k-3T"

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_id, use_fast=True)

        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x['text']) >= 512)
        self.calibration_dataset = [self.tokenizer(example["text"]) for example in traindata.select(range(1024))]


    @parameterized.expand(
        [
            (QUANT_METHOD.AUTO_ROUND, BACKEND.EXLLAMA_V2, True, FORMAT.GPTQ, 4),
        ]
    )
    def test_quantize(self, method: QUANT_METHOD, backend: BACKEND, sym: bool, format: FORMAT, bits: int):
        if method == QUANT_METHOD.GPTQ:
            quantize_config = QuantizeConfig(
                bits=bits,
                group_size=128,
                desc_act=False if format == FORMAT.MARLIN else True,
                sym=sym,
                format=format,
                damp_percent=0.05
            )
        elif method == QUANT_METHOD.AUTO_ROUND:
            quantize_config = AutoRoundQuantizeConfig(
                bits=bits,
                group_size=128,
                sym=sym,
                format=format,
            )
        else:
            raise ValueError(f"Invalid quantization method: {method}")

        model = GPTQModel.load(
            self.pretrained_model_id,
            quantize_config=quantize_config,
        )
        model.quantize(self.calibration_dataset, batch_size=32)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save(tmpdirname)

            logging.info(f"Saved config mem: {model.quantize_config}")

            with open(tmpdirname + "/" + QUANT_CONFIG_FILENAME, "r") as f:
                file_dict = json.loads(f.read())

                # make sure the json dict saved to file matches config in memory
                assert model.quantize_config.to_dict() == file_dict
                logging.info(f"Saved config file: {file_dict}")

            model = GPTQModel.load(
                tmpdirname,
                device=get_best_device(backend),
                backend=backend,
            )

            logging.info(f"Loaded config: {model.quantize_config}")

            versionable = model.quantize_config.meta_get_versionable(META_FIELD_QUANTIZER)
            assert META_QUANTIZER_GPTQMODEL in [v[0] for v in versionable]
            for producer, _version in versionable:
                if producer == META_QUANTIZER_GPTQMODEL:
                    assert _version == __version__

            del model
            torch_empty_cache()

            # skip compat test with sym=False and v1 since we do meta version safety check
            if not sym and format == FORMAT.GPTQ or format == FORMAT.IPEX:
                return

            # test compat: 1) with simple dict type 2) is_marlin_format
            compat_quantize_config = {
                "bits": bits,
                "group_size": 128,
                "sym": sym,
                "desc_act": False if format == FORMAT.MARLIN else True,
                "is_marlin_format": backend == BACKEND.MARLIN,
            }

            model = GPTQModel.load(
                tmpdirname,
                device=get_best_device(backend),
                quantize_config=compat_quantize_config,
            )
            assert isinstance(model.quantize_config, QuantizeConfig)

            del model
            torch_empty_cache()

